from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.robocasa_tiny_evaluator import RoboCasaVAEWorldModel, vae_world_model_loss
from train.common import device_from_arg
from train.train_robocasa_tiny_evaluator import _batch, _filtered_manifest, _load_data, _mean_std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--out-dir", default="runs/robocasa/world_evaluator/vae")
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--robocasa-task-index", action="append", type=int, default=[])
    parser.add_argument("--condition-on-robocasa-task-index", action="store_true")
    parser.add_argument("--train-demos-per-task", type=int, default=80)
    parser.add_argument("--val-episode-id", action="append", type=int, default=[])
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--success-window", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--proprio-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=0.5)
    parser.add_argument("--success-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    manifest = _filtered_manifest(Path(args.manifest), args.task_alias)
    train, val = _load_data(
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

    proprio_mean, proprio_std = _mean_std(np.concatenate([train.proprio, train.next_proprio], axis=0))
    train.proprio = ((train.proprio - proprio_mean) / proprio_std).astype(np.float32)
    train.next_proprio = ((train.next_proprio - proprio_mean) / proprio_std).astype(np.float32)
    val.proprio = ((val.proprio - proprio_mean) / proprio_std).astype(np.float32)
    val.next_proprio = ((val.next_proprio - proprio_mean) / proprio_std).astype(np.float32)
    action_mean, action_std = _mean_std(train.action)
    train.action = ((train.action - action_mean) / action_std).astype(np.float32)
    val.action = ((val.action - action_mean) / action_std).astype(np.float32)

    device = device_from_arg(args.device)
    task_count = int(max(train.task_id.max(initial=0), val.task_id.max(initial=0)) + 1)
    model = RoboCasaVAEWorldModel(
        proprio_dim=int(train.proprio.shape[-1]),
        action_dim=int(train.action.shape[-1]),
        task_count=task_count,
        latent_dim=int(args.latent_dim),
        width=int(args.width),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_val = math.inf
    best_state = None
    best_step = 0
    started = time.time()

    for step in range(1, int(args.steps) + 1):
        idx = rng.integers(0, len(train), size=int(args.batch_size))
        batch = _batch(train, idx, device)
        out = model(batch)
        loss, parts = _loss(out, batch, args)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        rec = {"step": step, **parts}
        history.append(rec)
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            val_metrics = _eval(model, val, device, int(args.batch_size), args)
            rec.update({f"val_{key}": value for key, value in val_metrics.items()})
            if val_metrics["loss"] < best_val:
                best_val = float(val_metrics["loss"])
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(
                f"step={step} loss={parts['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
                f"val_psnr={val_metrics['psnr']:.2f} val_progress_mae={val_metrics['progress_mae']:.4f}",
                flush=True,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_type": "robocasa_vae_world_model",
        "proprio_dim": int(train.proprio.shape[-1]),
        "action_dim": int(train.action.shape[-1]),
        "task_count": task_count,
        "latent_dim": int(args.latent_dim),
        "width": int(args.width),
        "dropout": float(args.dropout),
        "manifest": str(Path(args.manifest)),
        "views": ["robot0_agentview_left", "robot0_agentview_right"],
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "condition_on_robocasa_task_index": bool(args.condition_on_robocasa_task_index),
    }
    torch.save(checkpoint, out_dir / "vae_world_model.pt")
    best_checkpoint = dict(checkpoint)
    if best_state is not None:
        best_checkpoint["state_dict"] = best_state
        best_checkpoint["best_step"] = int(best_step)
        best_checkpoint["best_val_loss"] = float(best_val)
    torch.save(best_checkpoint, out_dir / "vae_world_model_best.pt")
    metrics = {
        "checkpoint": str(out_dir / "vae_world_model.pt"),
        "best_checkpoint": str(out_dir / "vae_world_model_best.pt"),
        "best_step": int(best_step),
        "best_val_loss": float(best_val),
        "val": _eval(model, val, device, int(args.batch_size), args),
        "train_samples": len(train),
        "val_samples": len(val),
        "train_demos_per_task": int(args.train_demos_per_task),
        "val_episode_ids": [int(ep) for ep in args.val_episode_id],
        "robocasa_task_indices": [int(idx) for idx in args.robocasa_task_index],
        "frame_stride": int(args.frame_stride),
        "latent_dim": int(args.latent_dim),
        "width": int(args.width),
        "train_seconds": time.time() - started,
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args) -> tuple[torch.Tensor, dict[str, float]]:
    return vae_world_model_loss(
        outputs,
        batch,
        recon_weight=float(args.recon_weight),
        kl_weight=float(args.kl_weight),
        latent_weight=float(args.latent_weight),
        proprio_weight=float(args.proprio_weight),
        progress_weight=float(args.progress_weight),
        success_weight=float(args.success_weight),
    )


def _eval(model: RoboCasaVAEWorldModel, data, device: torch.device, batch_size: int, args) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "kl": 0.0,
        "latent_loss": 0.0,
        "proprio_loss": 0.0,
        "progress_loss": 0.0,
        "success_loss": 0.0,
    }
    mse_total = 0.0
    progress_abs = 0.0
    success_correct = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            out = model(batch)
            loss, parts = _loss(out, batch, args)
            target = _target(batch)
            mse = F.mse_loss(out["recon"], target)
            n = len(idx)
            totals["loss"] += float(loss.detach().cpu()) * n
            for key in totals:
                if key != "loss":
                    totals[key] += float(parts[key]) * n
            mse_total += float(mse.detach().cpu()) * n
            progress_abs += float((torch.sigmoid(out["progress"]) - batch["progress"]).abs().sum().detach().cpu())
            success_correct += float(((torch.sigmoid(out["success_logit"]) >= 0.5) == (batch["success"] >= 0.5)).sum().detach().cpu())
            count += n
    model.train()
    mse = mse_total / max(1, count)
    return {
        **{key: value / max(1, count) for key, value in totals.items()},
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "progress_mae": progress_abs / max(1, count),
        "success_acc": success_correct / max(1, count),
    }


def _target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    agent = batch["agent"] / 255.0 if batch["agent"].max() > 1.5 else batch["agent"]
    wrist = batch["wrist"] / 255.0 if batch["wrist"].max() > 1.5 else batch["wrist"]
    return torch.cat([agent, wrist], dim=1).clamp(0.0, 1.0)


if __name__ == "__main__":
    main()
