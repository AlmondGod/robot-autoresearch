from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from autorobobench.robocasa_runtime import ensure_robocasa_runtime
from models.robocasa_latent_flow import RoboCasaLatentResidualFlow
from models.robocasa_tiny_evaluator import RoboCasaVAEWorldModel
from train.common import device_from_arg

ensure_robocasa_runtime()

from train.train_robocasa_tiny_evaluator import (
    _batch,
    _concat,
    _episode_task_index,
    _episode_transitions,
    _filtered_manifest,
)


DEFAULT_BASE = "runs/robocasa/world_evaluator/vae_robocasa5_stride4_w768_z512/vae_world_model_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a rectified-flow latent residual model on frozen RoboCasa VAE latents.")
    parser.add_argument("--checkpoint", default=DEFAULT_BASE)
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--out-dir", default="runs/robocasa/world_evaluator/latent_flow_stride4_w768_z512")
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--robocasa-task-index", action="append", type=int, default=[])
    parser.add_argument("--condition-on-robocasa-task-index", action="store_true")
    parser.add_argument("--train-demos-per-task", type=int, default=80)
    parser.add_argument("--val-episode-id", action="append", type=int, default=[])
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--success-window", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    base_ckpt = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    if base_ckpt.get("model_type") != "robocasa_vae_world_model":
        raise ValueError(f"expected robocasa_vae_world_model checkpoint, got {base_ckpt.get('model_type')}")
    base = _load_base(base_ckpt, device)
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False)

    manifest = _filtered_manifest(Path(args.manifest), args.task_alias)
    train, val = _load_flow_data(
        manifest,
        train_demos_per_task=int(args.train_demos_per_task),
        val_episode_ids=set(args.val_episode_id),
        robocasa_task_indices=set(args.robocasa_task_index),
        condition_on_robocasa_task_index=bool(args.condition_on_robocasa_task_index),
        frame_stride=int(args.frame_stride),
        success_window=float(args.success_window),
    )
    if len(train) == 0 or len(val) == 0:
        raise ValueError("need non-empty train and val transition data")
    _apply_base_norm(train, base_ckpt)
    _apply_base_norm(val, base_ckpt)

    flow = RoboCasaLatentResidualFlow(
        latent_dim=int(base_ckpt["latent_dim"]),
        action_dim=int(base_ckpt["action_dim"]),
        task_count=int(base_ckpt["task_count"]),
        hidden=int(args.hidden),
        depth=int(args.depth),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_val = math.inf
    best_state = None
    best_step = 0
    started = time.time()

    for step in range(1, int(args.steps) + 1):
        flow.train()
        idx = rng.integers(0, len(train), size=int(args.batch_size))
        batch = _batch(train, idx, device)
        loss, parts = _flow_loss(flow, base, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        record = {"step": step, **parts}
        history.append(record)
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            val_metrics = _eval(flow, base, val, device, int(args.batch_size), int(args.flow_steps), rng_seed=int(args.seed) + step)
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            if val_metrics["sample_mse"] < best_val:
                best_val = float(val_metrics["sample_mse"])
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in flow.state_dict().items()}
            print(
                f"step={step} loss={parts['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
                f"val_sample_mse={val_metrics['sample_mse']:.6f} val_base_mse={val_metrics['base_mse']:.6f}",
                flush=True,
            )

    val_metrics = _eval(flow, base, val, device, int(args.batch_size), int(args.flow_steps), rng_seed=int(args.seed) + 999)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": flow.state_dict(),
        "model_type": "robocasa_vae_latent_residual_flow",
        "base_checkpoint": str(Path(args.checkpoint)),
        "latent_dim": int(base_ckpt["latent_dim"]),
        "action_dim": int(base_ckpt["action_dim"]),
        "task_count": int(base_ckpt["task_count"]),
        "hidden": int(args.hidden),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "flow_steps": int(args.flow_steps),
        "manifest": str(Path(args.manifest)),
        "condition_on_robocasa_task_index": bool(args.condition_on_robocasa_task_index),
    }
    torch.save(checkpoint, out_dir / "latent_flow.pt")
    best_checkpoint = dict(checkpoint)
    if best_state is not None:
        best_checkpoint["state_dict"] = best_state
        best_checkpoint["best_step"] = int(best_step)
        best_checkpoint["best_val_sample_mse"] = float(best_val)
    torch.save(best_checkpoint, out_dir / "latent_flow_best.pt")
    metrics = {
        "checkpoint": str(out_dir / "latent_flow.pt"),
        "best_checkpoint": str(out_dir / "latent_flow_best.pt"),
        "base_checkpoint": str(Path(args.checkpoint)),
        "best_step": int(best_step),
        "best_val_sample_mse": float(best_val),
        "val": val_metrics,
        "train_samples": len(train),
        "val_samples": len(val),
        "train_demos_per_task": int(args.train_demos_per_task),
        "val_episode_ids": [int(ep) for ep in args.val_episode_id],
        "robocasa_task_indices": [int(idx) for idx in args.robocasa_task_index],
        "frame_stride": int(args.frame_stride),
        "flow_steps": int(args.flow_steps),
        "hidden": int(args.hidden),
        "depth": int(args.depth),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "train_seconds": time.time() - started,
        "tasks": [task["alias"] for task in manifest["tasks"]],
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _load_base(checkpoint: dict, device: torch.device) -> RoboCasaVAEWorldModel:
    model = RoboCasaVAEWorldModel(
        proprio_dim=int(checkpoint["proprio_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        task_count=int(checkpoint["task_count"]),
        latent_dim=int(checkpoint["latent_dim"]),
        width=int(checkpoint.get("width", 512)),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _load_flow_data(
    manifest: dict,
    *,
    train_demos_per_task: int,
    val_episode_ids: set[int],
    robocasa_task_indices: set[int],
    condition_on_robocasa_task_index: bool,
    frame_stride: int,
    success_window: float,
):
    train_parts: list[dict[str, np.ndarray]] = []
    val_parts: list[dict[str, np.ndarray]] = []
    for task in manifest["tasks"]:
        dataset_root = Path(task["dataset_path"])
        episode_paths = sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet"))
        train_loaded = 0
        for episode_path in episode_paths:
            episode_idx = int(episode_path.stem.split("_")[-1])
            is_explicit_val = episode_idx in val_episode_ids
            if not is_explicit_val and train_loaded >= int(train_demos_per_task):
                continue
            robocasa_idx = _episode_task_index(episode_path)
            if robocasa_task_indices and robocasa_idx not in robocasa_task_indices:
                continue
            part = _episode_transitions(
                dataset_root=dataset_root,
                episode_path=episode_path,
                episode_idx=episode_idx,
                task_id=robocasa_idx if condition_on_robocasa_task_index else int(task["task_id"]),
                frame_stride=frame_stride,
                success_window=success_window,
            )
            if is_explicit_val:
                split = "val"
                val_parts.append(part)
            else:
                split = "train"
                train_parts.append(part)
                train_loaded += 1
            print(
                f"loaded {task['alias']} episode={episode_idx} split={split} transitions={len(part['action'])}",
                flush=True,
            )
    return _concat(train_parts), _concat(val_parts)


def _apply_base_norm(data, checkpoint: dict) -> None:
    proprio_mean = _np_stat(checkpoint, "proprio_mean")
    proprio_std = _np_stat(checkpoint, "proprio_std")
    action_mean = _np_stat(checkpoint, "action_mean")
    action_std = _np_stat(checkpoint, "action_std")
    data.proprio = ((data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    data.next_proprio = ((data.next_proprio - proprio_mean) / proprio_std).astype(np.float32)
    data.action = ((data.action - action_mean) / action_std).astype(np.float32)


def _np_stat(checkpoint: dict, key: str) -> np.ndarray:
    value = checkpoint[key]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _flow_loss(
    flow: RoboCasaLatentResidualFlow,
    base: RoboCasaVAEWorldModel,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        z = base.encode(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
        next_z = base.encode(batch["next_agent"], batch["next_wrist"], batch["next_proprio"], batch["task_id"])
        target = next_z - z
    noise = torch.randn_like(target)
    t = torch.rand((target.shape[0],), device=target.device, dtype=target.dtype)
    t_view = t[:, None]
    x_t = (1.0 - t_view) * noise + t_view * target
    pred_v = flow(x_t, t, z, batch["action"], batch["task_id"])
    truth_v = target - noise
    loss = F.mse_loss(pred_v, truth_v)
    zero_mse = F.mse_loss(torch.zeros_like(target), target)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "target_mse_zero": float(zero_mse.detach().cpu()),
    }


def _eval(
    flow: RoboCasaLatentResidualFlow,
    base: RoboCasaVAEWorldModel,
    data,
    device: torch.device,
    batch_size: int,
    flow_steps: int,
    *,
    rng_seed: int,
) -> dict[str, float]:
    flow.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(int(rng_seed))
    totals = {"loss": 0.0, "sample_mse": 0.0, "base_mse": 0.0, "zero_mse": 0.0}
    count = 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            z = base.encode(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            next_z = base.encode(batch["next_agent"], batch["next_wrist"], batch["next_proprio"], batch["task_id"])
            target = next_z - z
            noise = torch.randn(target.shape, generator=rng, device=device, dtype=target.dtype)
            t = torch.rand((target.shape[0],), generator=rng, device=device, dtype=target.dtype)
            x_t = (1.0 - t[:, None]) * noise + t[:, None] * target
            pred_v = flow(x_t, t, z, batch["action"], batch["task_id"])
            loss = F.mse_loss(pred_v, target - noise)
            sample = flow.sample_residual(
                latent=z,
                action=batch["action"],
                task_id=batch["task_id"],
                steps=flow_steps,
                noise=noise,
            )
            base_next, _ = base.step(z, batch["action"], batch["task_id"])
            n = len(idx)
            totals["loss"] += float(loss.detach().cpu()) * n
            totals["sample_mse"] += float(F.mse_loss(sample, target).detach().cpu()) * n
            totals["base_mse"] += float(F.mse_loss(base_next - z, target).detach().cpu()) * n
            totals["zero_mse"] += float(F.mse_loss(torch.zeros_like(target), target).detach().cpu()) * n
            count += n
    flow.train()
    return {key: value / max(1, count) for key, value in totals.items()}


if __name__ == "__main__":
    main()
