from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime
from train.common import device_from_arg
from train.train_autorobobench_robocasa_bc5 import (
    _append_progress_features,
    _augment,
    _batch,
    _chunk_weights,
    _read_video64,
    _masked_mean_std,
    _mean_std,
    load_split_data,
)
from tasks.robocasa_microwave_peak.policy import MicrowaveSingleTaskChunkPolicy


ensure_robocasa_runtime()


DEFAULT_MANIFEST = "data/autorobobench/robocasa_microwave_peak_manifest.json"
DEFAULT_SPLIT = "data/autorobobench/robocasa_microwave_peak_splits.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single-task microwave peak optimizer.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_microwave_peak/single_task_chunk")
    parser.add_argument("--train-episodes-per-task", type=int, default=80)
    parser.add_argument("--val-episodes-per-task", type=int, default=10)
    parser.add_argument("--chunk-horizon", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-noise", type=float, default=0.004)
    parser.add_argument("--proprio-noise", type=float, default=0.004)
    parser.add_argument("--chunk-decay", type=float, default=0.82)
    parser.add_argument("--action-smooth", type=float, default=0.0005)
    parser.add_argument("--progress-conditioning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-scale", type=float, default=750.0)
    parser.add_argument("--variant-conditioning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-commit-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    start_time = time.monotonic()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    train_data, val_data, split_summary = load_split_data(
        manifest,
        split,
        task_aliases=set(),
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        chunk_horizon=int(args.chunk_horizon),
        frame_stride=int(args.frame_stride),
    )
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("need both train and val samples for microwave peak")
    variant_info = _variant_info(manifest, split)
    if args.variant_conditioning:
        _replace_with_variant_ids(train_data, variant_info["episode_to_variant"])
        _replace_with_variant_ids(val_data, variant_info["episode_to_variant"])

    raw_proprio_dim = int(train_data.proprio.shape[-1])
    if args.progress_conditioning:
        _append_progress_features(train_data, float(args.progress_scale))
        _append_progress_features(val_data, float(args.progress_scale))

    proprio_mean, proprio_std = _mean_std(train_data.proprio)
    action_mean, action_std = _masked_mean_std(train_data.actions, train_data.mask)
    train_data.proprio = ((train_data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    val_data.proprio = ((val_data.proprio - proprio_mean) / proprio_std).astype(np.float32)
    train_data.actions = ((train_data.actions - action_mean) / action_std).astype(np.float32)
    val_data.actions = ((val_data.actions - action_mean) / action_std).astype(np.float32)

    device = device_from_arg(str(args.device))
    model = MicrowaveSingleTaskChunkPolicy(
        proprio_dim=int(train_data.proprio.shape[-1]),
        chunk_horizon=int(args.chunk_horizon),
        action_dim=int(train_data.actions.shape[-1]),
        width=int(args.width),
        dropout=float(args.dropout),
        variant_count=int(variant_info["variant_count"]) if args.variant_conditioning else 1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))

    best_val = float("inf")
    best_step = 0
    best_state = _checkpoint_state(model)
    history: list[dict] = []
    for step in range(1, int(args.steps) + 1):
        if args.max_train_seconds > 0 and time.monotonic() - start_time >= float(args.max_train_seconds):
            break
        idx = rng.integers(0, len(train_data), size=int(args.batch_size))
        batch = _augment(_batch(train_data, idx, device), float(args.image_noise), float(args.proprio_noise))
        pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
        loss = _loss(pred, batch["actions"], batch["mask"], float(args.chunk_decay), float(args.action_smooth))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        row = {"step": step, "train_loss": float(loss.detach().cpu()), "elapsed_seconds": time.monotonic() - start_time}
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            val_loss = _eval_loss(model, val_data, device, max(64, int(args.batch_size)), float(args.chunk_decay))
            row["val_loss"] = val_loss
            if val_loss < best_val:
                best_val = val_loss
                best_step = step
                best_state = _checkpoint_state(model)
            print(f"step={step} train_loss={row['train_loss']:.6f} val_loss={val_loss:.6f}", flush=True)
        history.append(row)

    final_val = _eval_loss(model, val_data, device, max(64, int(args.batch_size)), float(args.chunk_decay))
    if final_val < best_val:
        best_val = final_val
        best_step = int(history[-1]["step"] if history else 0)
        best_state = _checkpoint_state(model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "policy_type": "robocasa_microwave_single_task_chunk",
        "state_dict": _checkpoint_state(model),
        "chunk_horizon": int(args.chunk_horizon),
        "action_dim": int(train_data.actions.shape[-1]),
        "proprio_dim": int(train_data.proprio.shape[-1]),
        "raw_proprio_dim": raw_proprio_dim,
        "width": int(args.width),
        "dropout": float(args.dropout),
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "progress_conditioning": bool(args.progress_conditioning),
        "progress_scale": float(args.progress_scale),
        "variant_conditioning": bool(args.variant_conditioning),
        "variant_count": int(variant_info["variant_count"]) if args.variant_conditioning else 1,
        "variant_embeddings": variant_info["variant_embeddings"] if args.variant_conditioning else None,
        "variant_embedding_ids": variant_info["variant_embedding_ids"] if args.variant_conditioning else None,
        "variant_embedding_episode_ids": variant_info["variant_embedding_episode_ids"] if args.variant_conditioning else None,
        "eval_commit_steps": int(args.eval_commit_steps),
        "manifest": str(args.manifest),
        "split": str(args.split),
    }
    best_checkpoint = dict(checkpoint)
    best_checkpoint["state_dict"] = best_state
    torch.save(checkpoint, out_dir / "policy.pt")
    torch.save(best_checkpoint, out_dir / "policy_best.pt")
    metrics = {
        "policy_type": "robocasa_microwave_single_task_chunk",
        "checkpoint": str(out_dir / "policy.pt"),
        "best_checkpoint": str(out_dir / "policy_best.pt"),
        "steps_completed": int(history[-1]["step"] if history else 0),
        "best_step": int(best_step),
        "best_val_action_loss": float(best_val),
        "final_val_action_loss": float(final_val),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_seconds": float(time.monotonic() - start_time),
        "split_summary": split_summary,
        "chunk_horizon": int(args.chunk_horizon),
        "frame_stride": int(args.frame_stride),
        "eval_commit_steps": int(args.eval_commit_steps),
        "progress_conditioning": bool(args.progress_conditioning),
        "progress_scale": float(args.progress_scale),
        "variant_conditioning": bool(args.variant_conditioning),
        "variant_count": int(variant_info["variant_count"]) if args.variant_conditioning else 1,
        "seed": int(args.seed),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, chunk_decay: float, action_smooth: float) -> torch.Tensor:
    per_step = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    weights = _chunk_weights(pred.shape[1], chunk_decay, pred.device, pred.dtype)
    loss = (per_step * mask * weights).sum() / (mask * weights).sum().clamp_min(1.0)
    if action_smooth > 0 and pred.shape[1] > 1:
        smooth = (pred[:, 1:] - pred[:, :-1]).square().mean(dim=-1)
        smooth_mask = mask[:, 1:] * mask[:, :-1]
        loss = loss + float(action_smooth) * (smooth * smooth_mask).sum() / smooth_mask.sum().clamp_min(1.0)
    return loss


def _eval_loss(model: MicrowaveSingleTaskChunkPolicy, data, device: torch.device, batch_size: int, chunk_decay: float) -> float:
    model.eval()
    total = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            per_step = F.smooth_l1_loss(pred, batch["actions"], reduction="none").mean(dim=-1)
            weights = _chunk_weights(pred.shape[1], chunk_decay, pred.device, pred.dtype)
            total = total + (per_step * batch["mask"] * weights).sum()
            denom = denom + (batch["mask"] * weights).sum()
    model.train()
    return float((total / denom.clamp_min(1.0)).detach().cpu())


def _checkpoint_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _replace_with_variant_ids(data, episode_to_variant: dict[int, int]) -> None:
    data.task_id = np.asarray([episode_to_variant[int(ep)] for ep in data.episode_idx], dtype=np.int64)


def _variant_info(manifest: dict, split: dict) -> dict:
    task = manifest["tasks"][0]
    dataset_root = Path(task["dataset_path"])
    if not dataset_root.is_absolute():
        dataset_root = Path.cwd() / dataset_root
    episode_ids = sorted(
        {
            int(ep)
            for split_task in split["tasks"]
            for key in ("train_episode_ids", "val_episode_ids")
            for ep in split_task.get(key, [])
        }
    )
    episode_to_variant: dict[int, int] = {}
    for episode_id in episode_ids:
        parquet = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        frame = pd.read_parquet(parquet, columns=["task_index"])
        episode_to_variant[episode_id] = int(frame["task_index"].iloc[0])

    train_ids = [int(ep) for ep in split["tasks"][0]["train_episode_ids"]]
    embeddings = []
    variant_ids = []
    embedding_episode_ids = []
    for episode_id in train_ids:
        embeddings.append(_initial_embedding(dataset_root, episode_id))
        variant_ids.append(episode_to_variant[episode_id])
        embedding_episode_ids.append(episode_id)
    return {
        "episode_to_variant": episode_to_variant,
        "variant_count": max(episode_to_variant.values()) + 1,
        "variant_embeddings": np.stack(embeddings, axis=0).astype(np.float32),
        "variant_embedding_ids": np.asarray(variant_ids, dtype=np.int64),
        "variant_embedding_episode_ids": np.asarray(embedding_episode_ids, dtype=np.int64),
    }


def _initial_embedding(dataset_root: Path, episode_id: int) -> np.ndarray:
    parts = []
    for view in ("robot0_agentview_left", "robot0_agentview_right"):
        frames = _read_video64(dataset_root, episode_id, view)
        parts.append(_image_embedding(frames[0]))
    return np.concatenate(parts).astype(np.float32)


def _image_embedding(image: np.ndarray) -> np.ndarray:
    small = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((16, 16), Image.Resampling.BILINEAR)
    return (np.asarray(small, dtype=np.float32).reshape(-1) / 255.0).astype(np.float32)


if __name__ == "__main__":
    main()
