from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.eval_robocasa_tiny_evaluator_correlation import _evaluator_task_id, _initial_obs, _tensor
from eval.eval_robocasa_tiny_evaluator_correlation import _load_evaluator
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace-calibrate a RoboCasa VAE world evaluator on mixed-task archives.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--trace-archive", required=True)
    parser.add_argument("--out-dir", default="runs/autorobobench/world_model_evaluator/mixed_trace_calibrated")
    parser.add_argument("--train-split", action="append", default=["train", "calibration"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=260)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--success-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=0.25)
    parser.add_argument("--freeze-encoder-decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detach-rollout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    model = _load_evaluator(checkpoint, device)
    model.load_state_dict(checkpoint["state_dict"])
    if bool(args.freeze_encoder_decoder):
        for name, param in model.named_parameters():
            if name.startswith(("image.", "encoder.", "mu.", "logvar.", "decoder_")):
                param.requires_grad_(False)

    dataset_roots = _dataset_roots(Path(args.manifest))
    traces = _load_traces(
        Path(args.trace_archive),
        train_splits={str(split) for split in args.train_split},
    )
    if not traces:
        raise ValueError("no training traces found")
    obs_cache = {
        (trace["task_alias"], int(trace["episode_id"])): _initial_obs(
            dataset_roots[trace["task_alias"]],
            int(trace["episode_id"]),
        )
        for trace in traces
    }

    trainable = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_loss = math.inf
    best_state = None
    best_step = 0
    started = time.time()

    for step in range(1, int(args.steps) + 1):
        idx = rng.integers(0, len(traces), size=min(int(args.batch_size), len(traces)))
        batch_traces = [traces[int(i)] for i in idx]
        loss, metrics = _trace_loss(
            model=model,
            checkpoint=checkpoint,
            traces=batch_traces,
            obs_cache=obs_cache,
            dataset_roots=dataset_roots,
            rollout_steps=int(args.rollout_steps),
            device=device,
            success_weight=float(args.success_weight),
            progress_weight=float(args.progress_weight),
            detach_rollout=bool(args.detach_rollout),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        rec = {"step": step, **metrics}
        history.append(rec)
        if metrics["loss"] < best_loss:
            best_loss = float(metrics["loss"])
            best_step = int(step)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            print(
                f"step={step} trace_loss={metrics['loss']:.6f} "
                f"success_loss={metrics['success_loss']:.6f} progress_loss={metrics['progress_loss']:.6f}",
                flush=True,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = dict(checkpoint)
    out_ckpt["state_dict"] = model.state_dict()
    out_ckpt["base_checkpoint"] = str(Path(args.checkpoint))
    out_ckpt["trace_calibrated"] = True
    out_ckpt["trace_archive"] = str(Path(args.trace_archive))
    out_ckpt["train_splits"] = [str(split) for split in args.train_split]
    torch.save(out_ckpt, out_dir / "vae_world_model_mixed_trace_calibrated.pt")
    best_ckpt = dict(out_ckpt)
    if best_state is not None:
        best_ckpt["state_dict"] = best_state
        best_ckpt["best_step"] = int(best_step)
        best_ckpt["best_trace_loss"] = float(best_loss)
    torch.save(best_ckpt, out_dir / "vae_world_model_mixed_trace_calibrated_best.pt")
    metrics = {
        "checkpoint": str(out_dir / "vae_world_model_mixed_trace_calibrated.pt"),
        "best_checkpoint": str(out_dir / "vae_world_model_mixed_trace_calibrated_best.pt"),
        "base_checkpoint": str(Path(args.checkpoint)),
        "best_step": int(best_step),
        "best_trace_loss": float(best_loss),
        "traces": len(traces),
        "train_splits": [str(split) for split in args.train_split],
        "train_seconds": time.time() - started,
        "freeze_encoder_decoder": bool(args.freeze_encoder_decoder),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _dataset_roots(manifest_path: Path) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text())
    return {str(task["alias"]): Path(task["dataset_path"]) for task in manifest["tasks"]}


def _load_traces(archive: Path, *, train_splits: set[str]) -> list[dict]:
    traces: list[dict] = []
    for line in archive.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("split", "train")) not in train_splits:
            continue
        eval_path = Path(row["eval_path"])
        if not eval_path.is_absolute():
            eval_path = Path.cwd() / eval_path
        payload = json.loads(eval_path.read_text())
        for detail in payload.get("details", []):
            trace_path = Path(detail["trace_path"])
            if not trace_path.is_absolute():
                trace_path = Path.cwd() / trace_path
            trace = np.load(trace_path)
            task_alias = str(detail.get("task_alias") or trace["task_alias"][0])
            traces.append(
                {
                    "task_alias": task_alias,
                    "episode_id": int(trace["episode_id"][0]),
                    "actions": np.asarray(trace["actions"], dtype=np.float32),
                    "success": np.asarray(trace["success"], dtype=np.float32),
                }
            )
    return traces


def _trace_loss(
    *,
    model,
    checkpoint: dict,
    traces: list[dict],
    obs_cache: dict[tuple[str, int], dict],
    dataset_roots: dict[str, Path],
    rollout_steps: int,
    device: torch.device,
    success_weight: float,
    progress_weight: float,
    detach_rollout: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    success_losses = []
    progress_losses = []
    for trace in traces:
        task_alias = str(trace["task_alias"])
        episode_id = int(trace["episode_id"])
        obs = obs_cache[(task_alias, episode_id)]
        dataset_root = dataset_roots[task_alias]
        task_id = _evaluator_task_id(dataset_root, episode_id, checkpoint)
        task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
        agent = torch.as_tensor(obs["agent"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        wrist = torch.as_tensor(obs["wrist"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        proprio = (
            torch.as_tensor(obs["proprio"][None], dtype=torch.float32, device=device)
            - _tensor(checkpoint, "proprio_mean", device)
        ) / _tensor(checkpoint, "proprio_std", device)
        latent = model.encode(agent, wrist, proprio, task_t)
        actions = trace["actions"][:rollout_steps]
        success = trace["success"][: len(actions)].astype(np.float32)
        if len(actions) == 0:
            continue
        logits = []
        progress = []
        for action in actions:
            action_t = (
                torch.as_tensor(action[None], dtype=torch.float32, device=device)
                - _tensor(checkpoint, "action_mean", device)
            ) / _tensor(checkpoint, "action_std", device)
            latent, _ = model.step(latent, action_t, task_t)
            progress_t, logit = model.heads(latent, task_t)
            logits.append(logit)
            progress.append(progress_t)
            if detach_rollout:
                latent = latent.detach()
        logits_t = torch.cat(logits, dim=0)
        progress_t = torch.cat(progress, dim=0)
        target_success = torch.as_tensor(success, dtype=torch.float32, device=device)
        final_success = float(success[-1])
        progress_target = torch.linspace(0.0, final_success, len(actions), dtype=torch.float32, device=device)
        success_loss = F.binary_cross_entropy_with_logits(logits_t, target_success)
        progress_loss = F.mse_loss(torch.sigmoid(progress_t), progress_target)
        losses.append(success_weight * success_loss + progress_weight * progress_loss)
        success_losses.append(success_loss.detach())
        progress_losses.append(progress_loss.detach())
    if not losses:
        raise ValueError("empty trace batch")
    loss = torch.stack(losses).mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "success_loss": float(torch.stack(success_losses).mean().cpu()),
        "progress_loss": float(torch.stack(progress_losses).mean().cpu()),
    }


if __name__ == "__main__":
    main()
