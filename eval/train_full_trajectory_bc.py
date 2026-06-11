from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from PIL import Image

from eval.eval_libero_success import _proprio, _wrist_image
from train.common import device_from_arg


class FullTrajectoryBC(nn.Module):
    def __init__(self, proprio_dim: int, horizon: int, action_dim: int, width: int = 256, dropout: float = 0.0):
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
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, prop_width), nn.ReLU(), nn.Linear(prop_width, prop_width), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(width + prop_width, 2 * width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, horizon * action_dim),
        )

    def forward(self, agent: torch.Tensor, wrist: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        h = self.dropout(torch.cat([self.image(torch.cat([agent, wrist], dim=1)), self.proprio(proprio)], dim=-1))
        return self.head(h).reshape(agent.shape[0], self.horizon, self.action_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/libero_easy1_task0/manifest.json")
    parser.add_argument("--out-dir", default="runs/libero/easy1_task0_full_traj_bc")
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-demos", type=int, default=40)
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
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config_path = Path(".libero_config").resolve()
    if config_path.exists():
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_path))

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    manifest = json.loads(Path(args.manifest).read_text())
    if len(manifest["tasks"]) != 1:
        raise ValueError("full trajectory BC prototype expects one task")
    suite = manifest.get("suite", "libero_object")
    task_ref = manifest["tasks"][0]
    task_name = task_ref["task_name"].removesuffix("_demo")
    benchmark = get_benchmark(suite)(0)
    task = benchmark.get_task({task.name: idx for idx, task in enumerate(benchmark.tasks)}[task_name])
    demos = _load_demos(Path(task_ref["dataset_path"]), args.horizon)
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(demos))
    rng.shuffle(order)
    demos = [demos[int(i)] for i in order]
    train_rows = demos[: min(args.train_demos, len(demos) - 1)]
    val_rows = demos[len(train_rows) :]

    device = device_from_arg(args.device)
    action_dim = int(train_rows[0]["actions"].shape[-1])
    model = FullTrajectoryBC(
        proprio_dim=len(train_rows[0]["proprio"]),
        horizon=args.horizon,
        action_dim=action_dim,
        width=args.width,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    weights = _temporal_weights(args.horizon, args.temporal_decay, args.front_weight, args.tail_weight, device)
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(train_rows), size=args.batch_size)
        batch = _batch(train_rows, idx, device)
        batch = _augment_batch(batch, args.image_noise, args.proprio_noise, args.action_noise)
        pred = model(batch["agent"], batch["wrist"], batch["proprio"])
        per = _per_step_loss(pred, batch["actions"], args.loss)
        loss = (per * batch["mask"] * weights).sum() / (batch["mask"] * weights).sum().clamp_min(1.0)
        if args.action_smooth > 0 and pred.shape[1] > 1:
            loss = loss + args.action_smooth * (pred[:, 1:] - pred[:, :-1]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 500 == 0:
            print(f"step={step} full_traj_loss={float(loss.detach().cpu()):.6f}", flush=True)

    val_loss = _eval_loss(model, val_rows, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "horizon": args.horizon,
            "action_dim": action_dim,
            "proprio_dim": len(train_rows[0]["proprio"]),
        },
        out_dir / "full_traj.pt",
    )
    success = _eval_rollout(model, task, action_dim, args.horizon, args.episodes_per_task, device, out_dir)
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
        **success,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _load_demos(path: Path, horizon: int) -> list[dict]:
    rows = []
    with h5py.File(path, "r") as handle:
        for key in sorted(handle["data"].keys()):
            group = handle["data"][key]
            actions = np.asarray(group["actions"], dtype=np.float32)
            padded = np.zeros((horizon, actions.shape[-1]), dtype=np.float32)
            mask = np.zeros((horizon,), dtype=np.float32)
            n = min(horizon, len(actions))
            padded[:n] = actions[:n]
            mask[:n] = 1.0
            rows.append(
                {
                    "demo": key,
                    "agent": _resize64(np.asarray(group["obs/agentview_rgb"][0], dtype=np.uint8)),
                    "wrist": _resize64(np.asarray(group["obs/eye_in_hand_rgb"][0], dtype=np.uint8)),
                    "proprio": np.asarray(group["robot_states"][0], dtype=np.float32),
                    "actions": padded,
                    "mask": mask,
                }
            )
    return rows


def _batch(rows: list[dict], idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "agent": torch.as_tensor(np.stack([rows[int(i)]["agent"] for i in idx]), dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "wrist": torch.as_tensor(np.stack([rows[int(i)]["wrist"] for i in idx]), dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        "proprio": torch.as_tensor(np.stack([rows[int(i)]["proprio"] for i in idx]), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.stack([rows[int(i)]["actions"] for i in idx]), dtype=torch.float32, device=device),
        "mask": torch.as_tensor(np.stack([rows[int(i)]["mask"] for i in idx]), dtype=torch.float32, device=device),
    }


def _resize64(image: np.ndarray) -> np.ndarray:
    if image.shape[0] == 64 and image.shape[1] == 64:
        return image
    return np.asarray(Image.fromarray(image[..., :3]).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)


def _eval_loss(model: FullTrajectoryBC, rows: list[dict], device: torch.device) -> float:
    if not rows:
        return float("nan")
    with torch.no_grad():
        batch = _batch(rows, np.arange(len(rows)), device)
        pred = model(batch["agent"], batch["wrist"], batch["proprio"])
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


def _eval_rollout(model, task, action_dim: int, horizon: int, episodes: int, device: torch.device, out_dir: Path) -> dict:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    init_states = _as_numpy(torch.load(os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file), weights_only=False))
    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        camera_heights=64,
        camera_widths=64,
    )
    details = []
    successes = 0
    try:
        for episode_idx in range(episodes):
            env.reset()
            obs = env.set_init_state(init_states[episode_idx % init_states.shape[0]])
            row = {
                "agent": np.asarray(obs["agentview_image"], dtype=np.uint8),
                "wrist": np.asarray(_wrist_image(obs), dtype=np.uint8),
                "proprio": _proprio(obs).astype(np.float32),
                "actions": np.zeros((horizon, action_dim), dtype=np.float32),
                "mask": np.ones((horizon,), dtype=np.float32),
            }
            with torch.no_grad():
                action_seq = model(**{k: v for k, v in _batch([row], np.asarray([0]), device).items() if k in {"agent", "wrist", "proprio"}})[0].cpu().numpy()
            done = False
            reward_sum = 0.0
            for action in action_seq:
                obs, reward, done, _info = env.step(action.astype(np.float32))
                reward_sum += float(reward)
                if done:
                    break
            success = bool(done or env.check_success())
            successes += int(success)
            details.append({"episode": episode_idx, "reward_sum": reward_sum, "success": success})
    finally:
        env.close()
    payload = {"success_rate": successes / max(1, episodes), "episodes": episodes, "per_episode": details}
    (out_dir / "success.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _as_numpy(value) -> np.ndarray:
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


if __name__ == "__main__":
    main()
