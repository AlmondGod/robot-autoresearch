from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime
from models.robocasa_sequence_flow import RoboCasaTemporalChunkBC
from train.common import device_from_arg
from train.train_autorobobench_robocasa_bc5 import (
    TemporalChunkData,
    _augment,
    _batch,
    _chunk_weights,
    _mean_std,
    _task_texts_for_split,
    _weighted_masked_mean_std,
    load_split_data,
)


ensure_robocasa_runtime()


@dataclass
class RecapData:
    data: TemporalChunkData
    sample_weight: np.ndarray
    advantage: np.ndarray
    source_id: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RECAP-style offline RoboCasa policy.")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_recap_offline/baseline")
    parser.add_argument("--train-episodes-per-task", type=int, default=4)
    parser.add_argument("--val-episodes-per-task", type=int, default=2)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--chunk-horizon", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-noise", type=float, default=0.01)
    parser.add_argument("--proprio-noise", type=float, default=0.01)
    parser.add_argument("--action-smooth", type=float, default=0.001)
    parser.add_argument("--chunk-decay", type=float, default=1.0)
    parser.add_argument("--experience-multiplier", type=float, default=1.0)
    parser.add_argument("--bad-action-noise", type=float, default=0.35)
    parser.add_argument("--bad-sample-weight", type=float, default=0.35)
    parser.add_argument("--correction-fraction", type=float, default=0.25)
    parser.add_argument("--correction-weight", type=float, default=1.0)
    parser.add_argument("--eval-advantage", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    start_time = time.monotonic()
    rng = np.random.default_rng(int(args.seed))
    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    task_aliases = set(args.task_alias)
    task_texts = _task_texts_for_split(manifest, split, task_aliases)
    train_demo, val_demo, split_summary = load_split_data(
        manifest,
        split,
        task_aliases=task_aliases,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        chunk_horizon=int(args.chunk_horizon),
        frame_stride=int(args.frame_stride),
    )
    if len(train_demo) == 0 or len(val_demo) == 0:
        raise ValueError("training and validation splits must both be non-empty")

    raw_proprio_dim = int(train_demo.proprio.shape[-1])
    train_recap = _build_recap_data(
        train_demo,
        rng=rng,
        experience_multiplier=float(args.experience_multiplier),
        bad_action_noise=float(args.bad_action_noise),
        bad_sample_weight=float(args.bad_sample_weight),
        correction_fraction=float(args.correction_fraction),
        correction_weight=float(args.correction_weight),
    )
    val_recap = RecapData(
        data=_with_advantage(train_like=val_demo, advantage=np.ones((len(val_demo),), dtype=np.float32)),
        sample_weight=np.ones((len(val_demo),), dtype=np.float32),
        advantage=np.ones((len(val_demo),), dtype=np.float32),
        source_id=np.zeros((len(val_demo),), dtype=np.int64),
    )

    action_mean, action_std = _weighted_masked_mean_std(train_recap.data.actions, train_recap.data.mask)
    proprio_mean, proprio_std = _mean_std(train_recap.data.proprio)
    train_data = _normalize_data(train_recap.data, proprio_mean, proprio_std, action_mean, action_std)
    val_data = _normalize_data(val_recap.data, proprio_mean, proprio_std, action_mean, action_std)
    task_count = int(max(train_recap.data.task_id.max(initial=0), val_recap.data.task_id.max(initial=0))) + 1

    device = device_from_arg(str(args.device))
    model = RoboCasaTemporalChunkBC(
        proprio_dim=int(train_data.proprio.shape[-1]),
        chunk_horizon=int(args.chunk_horizon),
        action_dim=int(train_data.actions.shape[-1]),
        task_count=task_count,
        width=int(args.width),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val_loss = math.inf
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for step in range(1, int(args.steps) + 1):
        if float(args.max_train_seconds) > 0 and time.monotonic() - start_time >= float(args.max_train_seconds):
            break
        idx = rng.integers(0, len(train_data), size=int(args.batch_size), endpoint=False)
        batch = _batch(train_data, idx, device)
        batch = _augment_except_advantage(
            batch,
            image_noise=float(args.image_noise),
            proprio_noise=float(args.proprio_noise),
            raw_proprio_dim=raw_proprio_dim,
        )
        sample_weight = torch.as_tensor(train_recap.sample_weight[idx], dtype=torch.float32, device=device)
        pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
        loss = _weighted_chunk_loss(
            pred,
            batch["actions"],
            batch["mask"],
            sample_weight=sample_weight,
            chunk_decay=float(args.chunk_decay),
        )
        if float(args.action_smooth) > 0 and pred.shape[1] > 1:
            loss = loss + float(args.action_smooth) * (pred[:, 1:] - pred[:, :-1]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 25 == 0 or step == int(args.steps):
            val_loss = _eval_loss(model, val_data, device, batch_size=max(64, int(args.batch_size)))
            row = {"step": int(step), "train_loss": float(loss.detach().cpu()), "val_action_mse_normalized": float(val_loss)}
            history.append(row)
            print(json.dumps(row), flush=True)
            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_step = int(step)
                best_state = _state_dict_cpu(model)

    final_val_loss = _eval_loss(model, val_data, device, batch_size=max(64, int(args.batch_size)))
    if final_val_loss < best_val_loss:
        best_val_loss = float(final_val_loss)
        best_step = int(history[-1]["step"] if history else 0)
        best_state = _state_dict_cpu(model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_counts = {
        "demo": int((train_recap.source_id == 0).sum()),
        "bad_rollout": int((train_recap.source_id == 1).sum()),
        "correction": int((train_recap.source_id == 2).sum()),
    }
    checkpoint = {
        "state_dict": _state_dict_cpu(model),
        "policy_type": "autorobobench_robocasa_recap_offline",
        "chunk_horizon": int(args.chunk_horizon),
        "action_dim": int(train_data.actions.shape[-1]),
        "proprio_dim": int(train_data.proprio.shape[-1]),
        "raw_proprio_dim": raw_proprio_dim,
        "task_count": task_count,
        "width": int(args.width),
        "dropout": float(args.dropout),
        "policy_kind": "advantage_conditioned_bc",
        "task_texts": task_texts,
        "views": ["robot0_agentview_left", "robot0_agentview_right"],
        "manifest": str(Path(args.manifest)),
        "split": str(Path(args.split)),
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "recap_eval_advantage": float(args.eval_advantage),
        "recap_source_counts": source_counts,
        "recap_bad_action_noise": float(args.bad_action_noise),
        "recap_bad_sample_weight": float(args.bad_sample_weight),
        "recap_correction_fraction": float(args.correction_fraction),
    }
    torch.save(checkpoint, out_dir / "policy.pt")
    best_checkpoint = dict(checkpoint)
    if best_state is not None:
        best_checkpoint["state_dict"] = best_state
        best_checkpoint["best_step"] = int(best_step)
        best_checkpoint["best_val_action_mse_normalized"] = float(best_val_loss)
    torch.save(best_checkpoint, out_dir / "policy_best.pt")

    metrics = {
        "checkpoint": str(out_dir / "policy.pt"),
        "best_checkpoint": str(out_dir / "policy_best.pt"),
        "best_step": int(best_step),
        "best_val_action_mse_normalized": float(best_val_loss),
        "final_val_action_mse_normalized": float(final_val_loss),
        "train_samples": int(len(train_data)),
        "val_samples": int(len(val_data)),
        "source_counts": source_counts,
        "split_summary": split_summary,
        "chunk_horizon": int(args.chunk_horizon),
        "frame_stride": int(args.frame_stride),
        "train_episodes_per_task": int(args.train_episodes_per_task),
        "val_episodes_per_task": int(args.val_episodes_per_task),
        "steps_completed": int(history[-1]["step"] if history else 0),
        "train_seconds": float(time.monotonic() - start_time),
        "width": int(args.width),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "image_noise": float(args.image_noise),
        "proprio_noise": float(args.proprio_noise),
        "action_smooth": float(args.action_smooth),
        "experience_multiplier": float(args.experience_multiplier),
        "bad_action_noise": float(args.bad_action_noise),
        "bad_sample_weight": float(args.bad_sample_weight),
        "correction_fraction": float(args.correction_fraction),
        "correction_weight": float(args.correction_weight),
        "eval_advantage": float(args.eval_advantage),
        "history": history,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _build_recap_data(
    demo: TemporalChunkData,
    *,
    rng: np.random.Generator,
    experience_multiplier: float,
    bad_action_noise: float,
    bad_sample_weight: float,
    correction_fraction: float,
    correction_weight: float,
) -> RecapData:
    demo_part = _with_advantage(train_like=demo, advantage=np.ones((len(demo),), dtype=np.float32))
    parts = [demo_part]
    weights = [np.ones((len(demo),), dtype=np.float32)]
    advantages = [np.ones((len(demo),), dtype=np.float32)]
    sources = [np.zeros((len(demo),), dtype=np.int64)]

    bad_count = max(0, int(round(len(demo) * max(0.0, experience_multiplier))))
    if bad_count > 0:
        bad_idx = rng.integers(0, len(demo), size=bad_count, endpoint=False)
        bad_base = _take(demo, bad_idx)
        _, action_std = _weighted_masked_mean_std(demo.actions, demo.mask)
        noise = rng.normal(0.0, 1.0, size=bad_base.actions.shape).astype(np.float32)
        noisy_actions = bad_base.actions + noise * action_std.reshape(1, 1, -1) * float(bad_action_noise)
        noisy_actions = np.clip(noisy_actions, -1.0, 1.0).astype(np.float32)
        noisy_actions = np.where(bad_base.mask[..., None] > 0, noisy_actions, bad_base.actions)
        bad_base.actions = noisy_actions
        bad_adv = -np.ones((bad_count,), dtype=np.float32)
        parts.append(_with_advantage(train_like=bad_base, advantage=bad_adv))
        weights.append(np.full((bad_count,), float(bad_sample_weight), dtype=np.float32))
        advantages.append(bad_adv)
        sources.append(np.ones((bad_count,), dtype=np.int64))

        correction_count = max(0, int(round(bad_count * np.clip(correction_fraction, 0.0, 1.0))))
        if correction_count > 0:
            correction_idx = bad_idx[:correction_count]
            correction_base = _take(demo, correction_idx)
            correction_adv = np.ones((correction_count,), dtype=np.float32)
            parts.append(_with_advantage(train_like=correction_base, advantage=correction_adv))
            weights.append(np.full((correction_count,), float(correction_weight), dtype=np.float32))
            advantages.append(correction_adv)
            sources.append(np.full((correction_count,), 2, dtype=np.int64))

    return RecapData(
        data=_concat(parts),
        sample_weight=np.concatenate(weights, axis=0).astype(np.float32),
        advantage=np.concatenate(advantages, axis=0).astype(np.float32),
        source_id=np.concatenate(sources, axis=0).astype(np.int64),
    )


def _with_advantage(*, train_like: TemporalChunkData, advantage: np.ndarray) -> TemporalChunkData:
    advantage = advantage.astype(np.float32).reshape(-1, 1)
    proprio = np.concatenate([train_like.proprio.astype(np.float32), advantage], axis=-1)
    return TemporalChunkData(
        agent=train_like.agent,
        wrist=train_like.wrist,
        proprio=proprio,
        actions=train_like.actions,
        mask=train_like.mask,
        task_id=train_like.task_id,
        episode_idx=train_like.episode_idx,
        frame_idx=train_like.frame_idx,
    )


def _take(data: TemporalChunkData, idx: np.ndarray) -> TemporalChunkData:
    return TemporalChunkData(
        agent=data.agent[idx].copy(),
        wrist=data.wrist[idx].copy(),
        proprio=data.proprio[idx].copy(),
        actions=data.actions[idx].copy(),
        mask=data.mask[idx].copy(),
        task_id=data.task_id[idx].copy(),
        episode_idx=data.episode_idx[idx].copy(),
        frame_idx=data.frame_idx[idx].copy(),
    )


def _concat(parts: list[TemporalChunkData]) -> TemporalChunkData:
    return TemporalChunkData(
        agent=np.concatenate([part.agent for part in parts], axis=0),
        wrist=np.concatenate([part.wrist for part in parts], axis=0),
        proprio=np.concatenate([part.proprio for part in parts], axis=0),
        actions=np.concatenate([part.actions for part in parts], axis=0),
        mask=np.concatenate([part.mask for part in parts], axis=0),
        task_id=np.concatenate([part.task_id for part in parts], axis=0),
        episode_idx=np.concatenate([part.episode_idx for part in parts], axis=0),
        frame_idx=np.concatenate([part.frame_idx for part in parts], axis=0),
    )


def _normalize_data(
    data: TemporalChunkData,
    proprio_mean: np.ndarray,
    proprio_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> TemporalChunkData:
    return TemporalChunkData(
        agent=data.agent,
        wrist=data.wrist,
        proprio=((data.proprio - proprio_mean) / proprio_std).astype(np.float32),
        actions=((data.actions - action_mean.reshape(1, 1, -1)) / action_std.reshape(1, 1, -1)).astype(np.float32),
        mask=data.mask.astype(np.float32),
        task_id=data.task_id,
        episode_idx=data.episode_idx,
        frame_idx=data.frame_idx,
    )


def _augment_except_advantage(
    batch: dict[str, torch.Tensor],
    *,
    image_noise: float,
    proprio_noise: float,
    raw_proprio_dim: int,
) -> dict[str, torch.Tensor]:
    batch = _augment(batch, image_noise=image_noise, proprio_noise=0.0)
    if proprio_noise > 0:
        noise = torch.randn_like(batch["proprio"][:, :raw_proprio_dim]) * float(proprio_noise)
        batch["proprio"] = batch["proprio"].clone()
        batch["proprio"][:, :raw_proprio_dim] = batch["proprio"][:, :raw_proprio_dim] + noise
    return batch


def _weighted_chunk_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    sample_weight: torch.Tensor,
    chunk_decay: float,
) -> torch.Tensor:
    per_step = (pred - target).square().mean(dim=-1)
    chunk_weight = _chunk_weights(pred.shape[1], chunk_decay, pred.device, pred.dtype)
    sample_weight = sample_weight.reshape(-1, 1).to(dtype=pred.dtype)
    weights = mask * chunk_weight * sample_weight
    return (per_step * weights).sum() / weights.sum().clamp_min(1.0)


def _eval_loss(model: RoboCasaTemporalChunkBC, data: TemporalChunkData, device: torch.device, batch_size: int) -> float:
    model.eval()
    total = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            per_step = (pred - batch["actions"]).square().mean(dim=-1)
            total = total + (per_step * batch["mask"]).sum()
            denom = denom + batch["mask"].sum()
    model.train()
    return float((total / denom.clamp_min(1.0)).detach().cpu())


def _state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


if __name__ == "__main__":
    main()
