from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from models.robocasa_tiny_evaluator import RoboCasaVAEWorldModel
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trace-archive", default="runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", default="runs/robocasa/world_evaluator/vae_trace_calibrated")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=260)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-encoder-decoder", action="store_true")
    parser.add_argument("--detach-rollout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = RoboCasaVAEWorldModel(
        proprio_dim=int(ckpt["proprio_dim"]),
        action_dim=int(ckpt["action_dim"]),
        task_count=int(ckpt["task_count"]),
        latent_dim=int(ckpt["latent_dim"]),
        width=int(ckpt.get("width", 512)),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    if args.freeze_encoder_decoder:
        for name, param in model.named_parameters():
            if name.startswith(("image.", "encoder.", "mu.", "logvar.", "decoder_")):
                param.requires_grad_(False)

    traces = _load_traces(Path(args.trace_archive))
    if not traces:
        raise ValueError("no traces found")
    obs_cache = {ep: _initial_obs(Path(args.dataset_root), ep) for ep in sorted({trace["episode_id"] for trace in traces})}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(args.seed))
    history = []
    best_loss = math.inf
    best_state = None
    best_step = 0
    started = time.time()

    for step in range(1, int(args.steps) + 1):
        idx = rng.integers(0, len(traces), size=min(int(args.batch_size), len(traces)))
        batch_traces = [traces[int(i)] for i in idx]
        loss, metrics = _trace_loss(
            model,
            ckpt,
            batch_traces,
            obs_cache,
            Path(args.dataset_root),
            int(args.rollout_steps),
            device,
            detach_rollout=bool(args.detach_rollout),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        rec = {"step": step, **metrics}
        history.append(rec)
        if metrics["loss"] < best_loss:
            best_loss = float(metrics["loss"])
            best_step = step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            print(f"step={step} trace_loss={metrics['loss']:.6f} success_loss={metrics['success_loss']:.6f} progress_loss={metrics['progress_loss']:.6f}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = dict(ckpt)
    out_ckpt["state_dict"] = model.state_dict()
    out_ckpt["base_checkpoint"] = str(Path(args.checkpoint))
    out_ckpt["trace_calibrated"] = True
    torch.save(out_ckpt, out_dir / "vae_world_model_trace_calibrated.pt")
    best_ckpt = dict(out_ckpt)
    if best_state is not None:
        best_ckpt["state_dict"] = best_state
        best_ckpt["best_step"] = int(best_step)
        best_ckpt["best_trace_loss"] = float(best_loss)
    torch.save(best_ckpt, out_dir / "vae_world_model_trace_calibrated_best.pt")
    metrics = {
        "checkpoint": str(out_dir / "vae_world_model_trace_calibrated.pt"),
        "best_checkpoint": str(out_dir / "vae_world_model_trace_calibrated_best.pt"),
        "best_step": int(best_step),
        "best_trace_loss": float(best_loss),
        "traces": len(traces),
        "train_seconds": time.time() - started,
        "freeze_encoder_decoder": bool(args.freeze_encoder_decoder),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _trace_loss(model, ckpt, traces, obs_cache, dataset_root: Path, rollout_steps: int, device, *, detach_rollout: bool):
    losses = []
    success_losses = []
    progress_losses = []
    for trace in traces:
        ep = int(trace["episode_id"])
        obs = obs_cache[ep]
        task_id = _task_id(dataset_root, ep, ckpt)
        task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
        agent = torch.as_tensor(obs["agent"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        wrist = torch.as_tensor(obs["wrist"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        proprio = (torch.as_tensor(obs["proprio"][None], dtype=torch.float32, device=device) - _tensor(ckpt, "proprio_mean", device)) / _tensor(ckpt, "proprio_std", device)
        latent = model.encode(agent, wrist, proprio, task_t)
        actions = trace["actions"][:rollout_steps]
        success = trace["success"][: len(actions)].astype(np.float32)
        if len(actions) == 0:
            continue
        logits = []
        progress = []
        for step, action in enumerate(actions):
            action_t = (torch.as_tensor(action[None], dtype=torch.float32, device=device) - _tensor(ckpt, "action_mean", device)) / _tensor(ckpt, "action_std", device)
            latent, _ = model.step(latent, action_t, task_t)
            prog, logit = model.heads(latent, task_t)
            logits.append(logit)
            progress.append(prog)
            if detach_rollout:
                latent = latent.detach()
        logits_t = torch.cat(logits, dim=0)
        progress_t = torch.cat(progress, dim=0)
        target_success = torch.as_tensor(success, dtype=torch.float32, device=device)
        final_success = float(success[-1])
        progress_target = torch.linspace(0.0, final_success, len(actions), dtype=torch.float32, device=device)
        success_loss = F.binary_cross_entropy_with_logits(logits_t, target_success)
        progress_loss = F.mse_loss(torch.sigmoid(progress_t), progress_target)
        losses.append(success_loss + 0.25 * progress_loss)
        success_losses.append(success_loss.detach())
        progress_losses.append(progress_loss.detach())
    loss = torch.stack(losses).mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "success_loss": float(torch.stack(success_losses).mean().cpu()),
        "progress_loss": float(torch.stack(progress_losses).mean().cpu()),
    }


def _load_traces(archive: Path) -> list[dict]:
    out = []
    for line in archive.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        eval_path = Path(row["eval_path"])
        if not eval_path.is_absolute():
            eval_path = Path.cwd() / eval_path
        payload = json.loads(eval_path.read_text())
        for detail in payload.get("details", []):
            trace_path = Path(detail["trace_path"])
            if not trace_path.is_absolute():
                trace_path = Path.cwd() / trace_path
            trace = np.load(trace_path)
            out.append(
                {
                    "episode_id": int(trace["episode_id"][0]),
                    "actions": np.asarray(trace["actions"], dtype=np.float32),
                    "success": np.asarray(trace["success"], dtype=np.float32),
                }
            )
    return out


def _initial_obs(dataset_root: Path, episode_idx: int) -> dict:
    frame = pd.read_parquet(dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet", columns=["observation.state"])
    return {
        "agent": _first_frame64(dataset_root, episode_idx, "robot0_agentview_left"),
        "wrist": _first_frame64(dataset_root, episode_idx, "robot0_agentview_right"),
        "proprio": np.asarray(frame["observation.state"].iloc[0], dtype=np.float32),
    }


def _first_frame64(dataset_root: Path, episode_idx: int, view: str) -> np.ndarray:
    path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4"
    try:
        frame = next(iio.imiter(path))
    except Exception:
        cap = cv2.VideoCapture(str(path))
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            raise OSError(f"could not read first frame: {path}")
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = np.asarray(frame, dtype=np.uint8)[..., :3]
    if image.shape[:2] != (64, 64):
        image = np.asarray(Image.fromarray(image).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)
    return image


def _task_id(dataset_root: Path, episode_idx: int, checkpoint: dict) -> int:
    if not bool(checkpoint.get("condition_on_robocasa_task_index", False)):
        return 0
    frame = pd.read_parquet(dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet", columns=["task_index"])
    return int(frame["task_index"].iloc[0])


def _tensor(checkpoint: dict, key: str, device: torch.device) -> torch.Tensor:
    value = checkpoint[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)


if __name__ == "__main__":
    main()
