from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from eval.eval_libero_success import _proprio, _wrist_image
from eval.eval_obs_retrieval import _image_feature
from train.common import device_from_arg


class RetrievalEmbedder(nn.Module):
    def __init__(self, proprio_dim: int, emb_dim: int = 32):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(6, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
        )
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, emb_dim))

    def forward(self, agent: torch.Tensor, wrist: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        h = self.head(torch.cat([self.image(torch.cat([agent, wrist], dim=1)), self.proprio(proprio)], dim=-1))
        return nn.functional.normalize(h, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/libero_easy1_task0/manifest.json")
    parser.add_argument("--out-dir", default="runs/libero/easy1_task0_retrieval_embedder")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
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
        raise ValueError("retrieval embedder prototype expects one task")
    suite = manifest.get("suite", "libero_object")
    task_ref = manifest["tasks"][0]
    task_name = task_ref["task_name"].removesuffix("_demo")
    benchmark = get_benchmark(suite)(0)
    task = benchmark.get_task({task.name: idx for idx, task in enumerate(benchmark.tasks)}[task_name])
    demos = _load_demo_library(Path(task_ref["dataset_path"]))
    demo_keys = sorted(demos)
    init_states = _as_numpy(torch.load(os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file), weights_only=False))

    rows = _collect_queries(task, init_states)
    demo_rows = [demos[key] for key in demo_keys]
    target = _target_distribution(rows, demo_rows)

    device = device_from_arg(args.device)
    model = RetrievalEmbedder(proprio_dim=len(rows[0]["proprio"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    demo_batch = _batch(demo_rows, np.arange(len(demo_rows)), device)
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(rows), size=args.batch_size)
        query_batch = _batch(rows, idx, device)
        query_emb = model(*query_batch)
        demo_emb = model(*demo_batch)
        logits = query_emb @ demo_emb.t() / 0.05
        labels = torch.as_tensor(target[idx], dtype=torch.float32, device=device)
        loss = -(labels * nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 250 == 0:
            print(f"step={step} embed_loss={float(loss.detach().cpu()):.6f}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "demo_keys": demo_keys, "proprio_dim": len(rows[0]["proprio"])}, out_dir / "embedder.pt")
    result = _eval(model, demos, demo_keys, task, init_states, args.episodes_per_task, out_dir, device)
    result.update({"train_rows": len(rows), "demo_count": len(demo_keys), "checkpoint": str(out_dir / "embedder.pt")})
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


def _collect_queries(task, init_states: np.ndarray) -> list[dict]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        camera_heights=64,
        camera_widths=64,
    )
    rows = []
    try:
        for init in init_states:
            env.reset()
            obs = env.set_init_state(init)
            rows.append(_row_from_obs(obs))
    finally:
        env.close()
    return rows


def _load_demo_library(path: Path) -> dict[str, dict[str, np.ndarray]]:
    demos = {}
    with h5py.File(path, "r") as handle:
        for key in sorted(handle["data"].keys()):
            group = handle["data"][key]
            demos[key] = {
                "actions": np.asarray(group["actions"], dtype=np.float32),
                "agent": np.asarray(group["obs/agentview_rgb"][0], dtype=np.uint8),
                "wrist": np.asarray(group["obs/eye_in_hand_rgb"][0], dtype=np.uint8),
                "proprio": np.asarray(group["robot_states"][0], dtype=np.float32),
            }
    return demos


def _row_from_obs(obs: dict) -> dict:
    return {
        "agent": np.asarray(obs["agentview_image"], dtype=np.uint8),
        "wrist": np.asarray(_wrist_image(obs), dtype=np.uint8),
        "proprio": _proprio(obs).astype(np.float32),
    }


def _target_distribution(rows: list[dict], demos: list[dict]) -> np.ndarray:
    out = []
    demo_feats = [{"proprio": d["proprio"], "image": _image_feature(d["agent"], d["wrist"])} for d in demos]
    for row in rows:
        q = {"proprio": row["proprio"], "image": _image_feature(row["agent"], row["wrist"])}
        dists = np.asarray([np.linalg.norm(q["proprio"] - d["proprio"]) + 0.1 * np.linalg.norm(q["image"] - d["image"]) for d in demo_feats], dtype=np.float32)
        weights = np.exp(-(dists - dists.min()) / 0.03)
        out.append(weights / weights.sum())
    return np.stack(out, axis=0)


def _batch(rows: list[dict], idx: np.ndarray, device: torch.device):
    agent = torch.as_tensor(np.stack([rows[int(i)]["agent"] for i in idx]), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    wrist = torch.as_tensor(np.stack([rows[int(i)]["wrist"] for i in idx]), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    proprio = torch.as_tensor(np.stack([rows[int(i)]["proprio"] for i in idx]), dtype=torch.float32, device=device)
    return agent, wrist, proprio


def _eval(model, demos, demo_keys, task, init_states, episodes: int, out_dir: Path, device: torch.device) -> dict:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    demo_batch = _batch([demos[key] for key in demo_keys], np.arange(len(demo_keys)), device)
    with torch.no_grad():
        demo_emb = model(*demo_batch)
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
            row = _row_from_obs(obs)
            with torch.no_grad():
                query_emb = model(*_batch([row], np.asarray([0]), device))
                demo_idx = int((query_emb @ demo_emb.t()).argmax(dim=-1).item())
            demo_key = demo_keys[demo_idx]
            done = False
            reward_sum = 0.0
            for action in demos[demo_key]["actions"][:220]:
                obs, reward, done, _info = env.step(action)
                reward_sum += float(reward)
                if done:
                    break
            success = bool(done or env.check_success())
            successes += int(success)
            details.append({"episode": episode_idx, "demo": demo_key, "reward_sum": reward_sum, "success": success})
    finally:
        env.close()
    payload = {"success_rate": successes / max(1, episodes), "episodes": episodes, "per_episode": details}
    (out_dir / "success.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _as_numpy(value) -> np.ndarray:
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


if __name__ == "__main__":
    main()
