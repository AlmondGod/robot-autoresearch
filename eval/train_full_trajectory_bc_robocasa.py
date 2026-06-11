from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
import robocasa.utils.lerobot_utils as LU

from train.common import device_from_arg


class RoboCasaFullTrajectoryBC(nn.Module):
    def __init__(
        self,
        *,
        proprio_dim: int,
        horizon: int,
        action_dim: int,
        task_count: int,
        width: int = 256,
        dropout: float = 0.0,
        task_dim: int = 32,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.dropout = nn.Dropout(dropout)
        self.image = nn.Sequential(
            nn.Conv2d(6, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, width),
            nn.ReLU(),
        )
        prop_width = max(64, width // 2)
        self.proprio = nn.Sequential(
            nn.Linear(proprio_dim, prop_width),
            nn.ReLU(),
            nn.Linear(prop_width, prop_width),
            nn.ReLU(),
        )
        self.task = nn.Embedding(task_count, task_dim)
        self.head = nn.Sequential(
            nn.Linear(width + prop_width + task_dim, 2 * width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, horizon * action_dim),
        )

    def forward(
        self,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        image_feat = self.image(torch.cat([agent, wrist], dim=1))
        proprio_feat = self.proprio(proprio)
        task_feat = self.task(task_id)
        h = self.dropout(torch.cat([image_feat, proprio_feat, task_feat], dim=-1))
        return self.head(h).reshape(agent.shape[0], self.horizon, self.action_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--out-dir", default="runs/robocasa5/full_traj_bc")
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-demos-per-task", type=int, default=40)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss", choices=["mse", "huber", "l1"], default="mse")
    parser.add_argument("--temporal-decay", type=float, default=1.0)
    parser.add_argument("--front-weight", type=float, default=1.0)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--image-noise", type=float, default=0.0)
    parser.add_argument("--proprio-noise", type=float, default=0.0)
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--action-smooth", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    if args.task_alias:
        keep = set(args.task_alias)
        manifest["tasks"] = [task for task in manifest["tasks"] if task["alias"] in keep]
        for new_task_id, task in enumerate(manifest["tasks"]):
            task["task_id"] = new_task_id
        manifest["task_count"] = len(manifest["tasks"])
        if not manifest["tasks"]:
            raise ValueError(f"no tasks left after filtering for aliases={sorted(keep)}")
    rows = _load_rows(manifest, args.horizon)
    train_rows, val_rows = _split_rows(rows, args.train_demos_per_task)
    if not train_rows or not val_rows:
        raise ValueError("need both train and val rows for RoboCasa full-trajectory BC")

    device = device_from_arg(args.device)
    action_dim = int(train_rows[0]["actions"].shape[-1])
    proprio_dim = int(train_rows[0]["proprio"].shape[-1])
    task_count = int(manifest["task_count"])
    model = RoboCasaFullTrajectoryBC(
        proprio_dim=proprio_dim,
        horizon=args.horizon,
        action_dim=action_dim,
        task_count=task_count,
        width=args.width,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = _temporal_weights(args.horizon, args.temporal_decay, args.front_weight, args.tail_weight, device)
    rng = np.random.default_rng(args.seed)
    history: list[dict] = []

    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(train_rows), size=args.batch_size)
        batch = _batch(train_rows, idx, device)
        batch = _augment_batch(batch, args.image_noise, args.proprio_noise, args.action_noise)
        pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
        per = _per_step_loss(pred, batch["actions"], args.loss)
        loss = (per * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)
        if args.action_smooth > 0 and pred.shape[1] > 1:
            loss = loss + args.action_smooth * (pred[:, 1:] - pred[:, :-1]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_value = float(loss.detach().cpu())
        history.append({"step": step, "train_loss": loss_value})
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(f"step={step} full_traj_loss={loss_value:.6f}", flush=True)

    val_loss = _eval_loss(model, val_rows, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "horizon": args.horizon,
            "action_dim": action_dim,
            "proprio_dim": proprio_dim,
            "task_count": task_count,
            "width": args.width,
            "views": manifest["views"],
            "manifest": str(Path(args.manifest)),
        },
        out_dir / "full_traj.pt",
    )
    metrics = {
        "checkpoint": str(out_dir / "full_traj.pt"),
        "train_demos": len(train_rows),
        "val_demos": len(val_rows),
        "horizon": args.horizon,
        "val_action_mse": val_loss,
        "width": args.width,
        "dropout": args.dropout,
        "loss": args.loss,
        "temporal_decay": args.temporal_decay,
        "front_weight": args.front_weight,
        "tail_weight": args.tail_weight,
        "image_noise": args.image_noise,
        "proprio_noise": args.proprio_noise,
        "action_noise": args.action_noise,
        "action_smooth": args.action_smooth,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "task_count": task_count,
        "tasks": [task["alias"] for task in manifest["tasks"]],
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _load_rows(manifest: dict, horizon: int) -> list[dict]:
    rows: list[dict] = []
    for task in manifest["tasks"]:
        dataset_root = Path(task["dataset_path"])
        for episode_path in sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet")):
            episode_idx = int(episode_path.stem.split("_")[-1])
            frame = pd.read_parquet(episode_path)
            actions = LU.get_episode_actions(dataset_root, episode_idx).astype(np.float32)
            proprio = np.asarray(frame["observation.state"].iloc[0], dtype=np.float32)
            padded = np.zeros((horizon, actions.shape[-1]), dtype=np.float32)
            mask = np.zeros((horizon,), dtype=np.float32)
            n = min(horizon, len(actions))
            padded[:n] = actions[:n]
            mask[:n] = 1.0
            rows.append(
                {
                    "task_id": int(task["task_id"]),
                    "task_name": task["alias"],
                    "episode_idx": episode_idx,
                    "agent": _first_video_frame(dataset_root, episode_idx, "robot0_agentview_left"),
                    "wrist": _first_video_frame(dataset_root, episode_idx, "robot0_agentview_right"),
                    "proprio": proprio,
                    "actions": padded,
                    "mask": mask,
                }
            )
    return rows


def _split_rows(rows: list[dict], train_demos_per_task: int) -> tuple[list[dict], list[dict]]:
    by_task: dict[int, list[dict]] = {}
    for row in rows:
        by_task.setdefault(int(row["task_id"]), []).append(row)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for task_id in sorted(by_task):
        task_rows = sorted(by_task[task_id], key=lambda row: row["episode_idx"])
        train_count = min(train_demos_per_task, max(1, len(task_rows) - 1))
        train_rows.extend(task_rows[:train_count])
        val_rows.extend(task_rows[train_count:])
    return train_rows, val_rows


def _first_video_frame(dataset_root: Path, episode_idx: int, view: str) -> np.ndarray:
    video_path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4"
    frame = iio.imread(video_path, index=0)
    return _resize64(np.asarray(frame, dtype=np.uint8))


def _batch(rows: list[dict], idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "agent": torch.as_tensor(np.stack([rows[int(i)]["agent"] for i in idx]), dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist": torch.as_tensor(np.stack([rows[int(i)]["wrist"] for i in idx]), dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "proprio": torch.as_tensor(np.stack([rows[int(i)]["proprio"] for i in idx]), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.stack([rows[int(i)]["actions"] for i in idx]), dtype=torch.float32, device=device),
        "mask": torch.as_tensor(np.stack([rows[int(i)]["mask"] for i in idx]), dtype=torch.float32, device=device),
        "task_id": torch.as_tensor(np.stack([rows[int(i)]["task_id"] for i in idx]), dtype=torch.long, device=device),
    }


def _resize64(image: np.ndarray) -> np.ndarray:
    if image.shape[0] == 64 and image.shape[1] == 64:
        return image
    return np.asarray(Image.fromarray(image[..., :3]).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)


def _eval_loss(model: RoboCasaFullTrajectoryBC, rows: list[dict], device: torch.device) -> float:
    with torch.no_grad():
        batch = _batch(rows, np.arange(len(rows)), device)
        pred = model(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
        per = (pred - batch["actions"]).square().mean(dim=-1)
        loss = (per * batch["mask"]).sum() / batch["mask"].sum().clamp_min(1.0)
    return float(loss.cpu())


def _per_step_loss(pred: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "huber":
        return nn.functional.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    if mode == "l1":
        return (pred - target).abs().mean(dim=-1)
    return (pred - target).square().mean(dim=-1)


def _temporal_weights(horizon: int, decay: float, front_weight: float, tail_weight: float, device: torch.device) -> torch.Tensor:
    idx = torch.arange(horizon, dtype=torch.float32, device=device)
    weights = torch.ones(horizon, dtype=torch.float32, device=device)
    if decay != 1.0:
        weights = weights * (float(decay) ** idx)
    if front_weight != 1.0:
        weights[: max(1, horizon // 4)] *= front_weight
    if tail_weight != 1.0:
        weights[-max(1, horizon // 4) :] *= tail_weight
    return weights / weights.mean().clamp_min(1e-6)


def _augment_batch(batch: dict[str, torch.Tensor], image_noise: float, proprio_noise: float, action_noise: float) -> dict[str, torch.Tensor]:
    if image_noise > 0:
        batch["agent"] = (batch["agent"] + torch.randn_like(batch["agent"]) * (255.0 * image_noise)).clamp(0.0, 255.0)
        batch["wrist"] = (batch["wrist"] + torch.randn_like(batch["wrist"]) * (255.0 * image_noise)).clamp(0.0, 255.0)
    if proprio_noise > 0:
        batch["proprio"] = batch["proprio"] + torch.randn_like(batch["proprio"]) * proprio_noise
    if action_noise > 0:
        batch["actions"] = batch["actions"] + torch.randn_like(batch["actions"]) * action_noise
    return batch


if __name__ == "__main__":
    main()
