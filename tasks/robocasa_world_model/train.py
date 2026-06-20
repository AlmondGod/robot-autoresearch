from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_world_model.data import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    TransitionData,
    load_transition_data,
    make_stats,
    normalize_data,
    save_json,
)
from tasks.robocasa_world_model.model import RoboCasaWorldModel
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RoboCasa learned world model.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_world_model/base")
    parser.add_argument("--train-episodes-per-task", type=int, default=20)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--task-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=0, help="Set >0 to train a VAE latent dynamics model.")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=0.25)
    parser.add_argument("--reward-weight", type=float, default=0.25)
    parser.add_argument("--success-weight", type=float, default=0.25)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = device_from_arg(str(args.device))
    train_raw, val_raw, summary = load_transition_data(
        manifest_path=args.manifest,
        split_path=args.split,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        task_aliases=set(args.task_alias),
        frame_stride=int(args.frame_stride),
    )
    if len(train_raw) == 0 or len(val_raw) == 0:
        raise ValueError("need both train and val transitions for world-model training")
    stats = make_stats(train_raw)
    train = normalize_data(train_raw, stats)
    val = normalize_data(val_raw, stats)
    task_count = int(max(train.task_id.max(initial=0), val.task_id.max(initial=0)) + 1)
    model = RoboCasaWorldModel(
        state_dim=int(train.state.shape[-1]),
        action_dim=int(train.action.shape[-1]),
        task_count=task_count,
        width=int(args.width),
        depth=int(args.depth),
        task_dim=int(args.task_dim),
        latent_dim=int(args.latent_dim),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    start_time = time.monotonic()
    for step in range(1, int(args.steps) + 1):
        if float(args.max_train_seconds) > 0 and time.monotonic() - start_time >= float(args.max_train_seconds):
            break
        model.train()
        idx = rng.integers(0, len(train), size=int(args.batch_size))
        batch = _batch(train, idx, device)
        loss, metrics = model.loss(
            batch,
            state_weight=float(args.state_weight),
            progress_weight=float(args.progress_weight),
            reward_weight=float(args.reward_weight),
            success_weight=float(args.success_weight),
            kl_weight=float(args.kl_weight),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % max(1, int(args.steps) // 20) == 0:
            val_metrics = _eval(model, val, int(args.batch_size), device)
            row = {
                "step": int(step),
                "elapsed_seconds": time.monotonic() - start_time,
                **{key: float(value.detach().cpu()) for key, value in metrics.items()},
                **{f"val_{key}": float(value) for key, value in val_metrics.items()},
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row["val_score_loss"] < best_val:
                best_val = row["val_score_loss"]
                _save_checkpoint(out_dir / "policy_best.pt", model, stats, args, summary, history, step)

    final_metrics = _eval(model, val, int(args.batch_size), device)
    _save_checkpoint(out_dir / "policy_last.pt", model, stats, args, summary, history, len(history))
    payload = {
        "task": "robocasa_world_model",
        "checkpoint": str(out_dir / "policy_best.pt"),
        "last_checkpoint": str(out_dir / "policy_last.pt"),
        "train_transitions": len(train),
        "val_transitions": len(val),
        "summary": summary,
        "final_val": final_metrics,
        "best_val_score_loss": best_val,
        "history": history,
        "seconds": time.monotonic() - start_time,
    }
    save_json(out_dir / "train_metrics.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _batch(data: TransitionData, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state": torch.as_tensor(data.state[idx], dtype=torch.float32, device=device),
        "action": torch.as_tensor(data.action[idx], dtype=torch.float32, device=device),
        "next_state": torch.as_tensor(data.next_state[idx], dtype=torch.float32, device=device),
        "progress": torch.as_tensor(data.progress[idx], dtype=torch.float32, device=device),
        "next_progress": torch.as_tensor(data.next_progress[idx], dtype=torch.float32, device=device),
        "reward": torch.as_tensor(data.reward[idx], dtype=torch.float32, device=device),
        "success": torch.as_tensor(data.success[idx], dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(data.task_id[idx], dtype=torch.long, device=device),
    }


@torch.no_grad()
def _eval(model: RoboCasaWorldModel, data: TransitionData, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {
        "state_mse": 0.0,
        "progress_mse": 0.0,
        "reward_mse": 0.0,
        "success_bce": 0.0,
        "score_loss": 0.0,
    }
    count = 0
    for start in range(0, len(data), batch_size):
        idx = np.arange(start, min(len(data), start + batch_size))
        batch = _batch(data, idx, device)
        total, metrics = model.loss(batch)
        n = len(idx)
        for key in ("state_mse", "progress_mse", "reward_mse", "success_bce"):
            sums[key] += float(metrics[key].detach().cpu()) * n
        sums["score_loss"] += float(total.detach().cpu()) * n
        count += n
    return {key: value / max(1, count) for key, value in sums.items()}


def _save_checkpoint(
    path: Path,
    model: RoboCasaWorldModel,
    stats: dict[str, np.ndarray],
    args: argparse.Namespace,
    summary: list[dict],
    history: list[dict],
    step: int,
) -> None:
    cfg = {
        "state_dim": int(model.state_dim),
        "action_dim": int(model.action_dim),
        "task_count": int(model.task_count),
        "width": int(args.width),
        "depth": int(args.depth),
        "task_dim": int(args.task_dim),
        "latent_dim": int(args.latent_dim),
        "dropout": float(args.dropout),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "stats": stats,
            "summary": summary,
            "history": history,
            "step": int(step),
            "task": "robocasa_world_model",
        },
        path,
    )


if __name__ == "__main__":
    main()
