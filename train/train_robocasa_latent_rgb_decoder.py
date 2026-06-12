from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.robocasa_tiny_evaluator import RoboCasaLatentRGBDecoder, RoboCasaTinyEvaluator
from train.common import device_from_arg
from train.train_robocasa_tiny_evaluator import _batch, _filtered_manifest, _load_data, _mean_std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--out-dir", default="runs/robocasa/world_evaluator/latent_rgb_decoder")
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--robocasa-task-index", action="append", type=int, default=[])
    parser.add_argument("--condition-on-robocasa-task-index", action="store_true")
    parser.add_argument("--train-demos-per-task", type=int, default=80)
    parser.add_argument("--val-episode-id", action="append", type=int, default=[])
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    evaluator_ckpt = torch.load(args.evaluator, map_location=device, weights_only=False)
    evaluator = RoboCasaTinyEvaluator(
        proprio_dim=int(evaluator_ckpt["proprio_dim"]),
        action_dim=int(evaluator_ckpt["action_dim"]),
        task_count=int(evaluator_ckpt["task_count"]),
        latent_dim=int(evaluator_ckpt["latent_dim"]),
        width=int(evaluator_ckpt.get("width", 512)),
        dropout=float(evaluator_ckpt.get("dropout", 0.0)),
    ).to(device)
    evaluator.load_state_dict(evaluator_ckpt["state_dict"])
    evaluator.eval()
    for param in evaluator.parameters():
        param.requires_grad_(False)

    manifest = _filtered_manifest(Path(args.manifest), args.task_alias)
    train, val = _load_data(
        manifest,
        train_demos_per_task=int(args.train_demos_per_task),
        val_episode_ids=set(args.val_episode_id),
        robocasa_task_indices=set(args.robocasa_task_index),
        condition_on_robocasa_task_index=bool(args.condition_on_robocasa_task_index),
        frame_stride=int(args.frame_stride),
        success_window=0.9,
    )
    proprio_mean, proprio_std = _mean_std(np.concatenate([train.proprio, train.next_proprio], axis=0))
    train.proprio = ((train.proprio - proprio_mean) / proprio_std).astype(np.float32)
    val.proprio = ((val.proprio - proprio_mean) / proprio_std).astype(np.float32)

    decoder = RoboCasaLatentRGBDecoder(
        latent_dim=int(evaluator_ckpt["latent_dim"]),
        task_count=int(evaluator_ckpt["task_count"]),
        width=int(args.width),
    ).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_val = math.inf
    best_state = None
    best_step = 0
    started = time.time()

    for step in range(1, int(args.steps) + 1):
        idx = rng.integers(0, len(train), size=int(args.batch_size))
        batch = _batch(train, idx, device)
        target = _target_rgb(batch)
        with torch.no_grad():
            latent = evaluator.encode(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
        pred = decoder(latent, batch["task_id"])
        loss = F.mse_loss(pred, target) + 0.1 * F.l1_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        rec = {"step": step, "loss": float(loss.detach().cpu())}
        history.append(rec)
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            val_metrics = _eval(decoder, evaluator, val, device, int(args.batch_size))
            rec.update({f"val_{k}": v for k, v in val_metrics.items()})
            if val_metrics["loss"] < best_val:
                best_val = float(val_metrics["loss"])
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}
            print(f"step={step} loss={rec['loss']:.6f} val_loss={val_metrics['loss']:.6f} val_psnr={val_metrics['psnr']:.2f}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": decoder.state_dict(),
        "model_type": "robocasa_latent_rgb_decoder",
        "latent_dim": int(evaluator_ckpt["latent_dim"]),
        "task_count": int(evaluator_ckpt["task_count"]),
        "width": int(args.width),
        "evaluator": str(Path(args.evaluator)),
        "views": ["robot0_agentview_left", "robot0_agentview_right"],
    }
    torch.save(ckpt, out_dir / "latent_rgb_decoder.pt")
    best_ckpt = dict(ckpt)
    if best_state is not None:
        best_ckpt["state_dict"] = best_state
        best_ckpt["best_step"] = int(best_step)
        best_ckpt["best_val_loss"] = float(best_val)
    torch.save(best_ckpt, out_dir / "latent_rgb_decoder_best.pt")
    metrics = {
        "checkpoint": str(out_dir / "latent_rgb_decoder.pt"),
        "best_checkpoint": str(out_dir / "latent_rgb_decoder_best.pt"),
        "best_step": int(best_step),
        "best_val_loss": float(best_val),
        "val": _eval(decoder, evaluator, val, device, int(args.batch_size)),
        "train_samples": len(train),
        "val_samples": len(val),
        "frame_stride": int(args.frame_stride),
        "train_seconds": time.time() - started,
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _target_rgb(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    agent = batch["agent"] / 255.0 if batch["agent"].max() > 1.5 else batch["agent"]
    wrist = batch["wrist"] / 255.0 if batch["wrist"].max() > 1.5 else batch["wrist"]
    return torch.cat([agent, wrist], dim=1).clamp(0.0, 1.0)


def _eval(
    decoder: RoboCasaLatentRGBDecoder,
    evaluator: RoboCasaTinyEvaluator,
    data,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    decoder.eval()
    total = 0.0
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            target = _target_rgb(batch)
            latent = evaluator.encode(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            pred = decoder(latent, batch["task_id"])
            mse = F.mse_loss(pred, target)
            loss = mse + 0.1 * F.l1_loss(pred, target)
            total += float(loss.detach().cpu()) * len(idx)
            total_mse += float(mse.detach().cpu()) * len(idx)
            count += len(idx)
    decoder.train()
    mse = total_mse / max(1, count)
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return {"loss": total / max(1, count), "mse": mse, "psnr": psnr}


if __name__ == "__main__":
    main()
