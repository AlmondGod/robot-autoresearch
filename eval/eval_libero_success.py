from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from data.libero_dataset import tokenize_instruction
from models.policy import TinyBCPolicy
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--libero-root", default="third_party/LIBERO")
    parser.add_argument("--manifest", default="data/libero_object5/manifest.json")
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    config_path = Path(".libero_config").resolve()
    if config_path.exists():
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_path))
    root = Path(args.libero_root)
    if not root.exists():
        raise FileNotFoundError(
            "LIBERO checkout not found. Run: python data/download_libero.py --dataset libero_object --use-huggingface"
        )
    try:
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LIBERO is not installed. From third_party/LIBERO, run `pip install -e .` in this environment."
        ) from exc

    device = _device(args.device)
    checkpoint = torch.load(args.policy, map_location=device)
    action_dim = int(checkpoint["action_dim"])
    proprio_dim = int(checkpoint["proprio_dim"])
    action_horizon = int(checkpoint.get("action_horizon", 1))
    history = int(checkpoint.get("history", 1))
    n_embd = int(checkpoint.get("n_embd", 128))
    policy_kind = str(checkpoint.get("policy_kind", "bc"))
    flow_steps = int(checkpoint.get("flow_steps", 8))
    policy = TinyBCPolicy(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        n_embd=n_embd,
        action_horizon=action_horizon,
        max_history=max(history, 1),
        policy_kind=policy_kind,
        flow_steps=flow_steps,
    ).to(device)
    policy.load_state_dict(checkpoint["state_dict"], strict=False)
    policy.eval()

    manifest = json.loads(Path(args.manifest).read_text())
    task_names = [task["task_name"].removesuffix("_demo") for task in manifest["tasks"]]
    benchmark = get_benchmark("libero_object")(0)
    name_to_task_id = {task.name: idx for idx, task in enumerate(benchmark.tasks)}

    per_task = []
    successes = 0
    total = 0
    for local_task_id, task_name in enumerate(task_names):
        benchmark_task_id = name_to_task_id[task_name]
        task = benchmark.get_task(benchmark_task_id)
        env_args = {
            "bddl_file_name": os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
            "camera_heights": args.image_size,
            "camera_widths": args.image_size,
        }
        init_states = torch.load(
            os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file),
            weights_only=False,
        )
        task_successes = 0
        env = OffScreenRenderEnv(**env_args)
        try:
            for episode_idx in range(args.episodes_per_task):
                env.reset()
                obs = env.set_init_state(init_states[episode_idx % init_states.shape[0]])
                for _ in range(5):
                    obs, _, _, _ = env.step(np.zeros(action_dim, dtype=np.float32))
                done = False
                rollout = _RolloutBuffers(history=max(history, 1))
                rollout.append(obs)
                action_queue: list[np.ndarray] = []
                for _step in range(args.max_steps):
                    if not action_queue:
                        action_queue = _policy_action_chunk(
                            policy,
                            rollout,
                            task_name,
                            local_task_id,
                            checkpoint,
                            device,
                        )
                    action = action_queue.pop(0)
                    obs, _reward, done, _info = env.step(action)
                    rollout.append(obs)
                    if done:
                        break
                task_successes += int(done)
                successes += int(done)
                total += 1
        finally:
            env.close()
        per_task.append(
            {
                "task_id": local_task_id,
                "benchmark_task_id": benchmark_task_id,
                "task_name": task_name,
                "success_rate": task_successes / max(1, args.episodes_per_task),
            }
        )

    success_rate = successes / max(1, total)
    out = Path(args.out or Path(args.policy).with_name("libero_success.json"))
    write_success(out, success_rate, {"episodes": total, "per_task": per_task})
    _merge_run_metrics(Path(args.policy).parent, success_rate)
    print(json.dumps({"success_rate": success_rate, "episodes": total, "out": str(out)}, indent=2, sort_keys=True))


def write_success(out: Path, success_rate: float, details: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"success_rate": success_rate, **details}, indent=2, sort_keys=True))


def _merge_run_metrics(run_dir: Path, success_rate: float) -> None:
    path = run_dir / "metrics.json"
    if not path.exists():
        return
    metrics = json.loads(path.read_text())
    metrics["success_rate"] = success_rate
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


class _RolloutBuffers:
    def __init__(self, history: int):
        self.history = history
        self.agent: list[np.ndarray] = []
        self.wrist: list[np.ndarray] = []
        self.proprio: list[np.ndarray] = []

    def append(self, obs: dict) -> None:
        self.agent.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
        self.wrist.append(np.asarray(_wrist_image(obs), dtype=np.uint8))
        self.proprio.append(_proprio(obs))
        self.agent = self.agent[-self.history :]
        self.wrist = self.wrist[-self.history :]
        self.proprio = self.proprio[-self.history :]

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pad_count = max(0, self.history - len(self.agent))
        agent = [self.agent[0]] * pad_count + self.agent
        wrist = [self.wrist[0]] * pad_count + self.wrist
        proprio = [self.proprio[0]] * pad_count + self.proprio
        return np.stack(agent, axis=0), np.stack(wrist, axis=0), np.stack(proprio, axis=0)


def _policy_action_chunk(
    policy: TinyBCPolicy,
    rollout: _RolloutBuffers,
    task_name: str,
    task_id: int,
    checkpoint: dict,
    device: torch.device,
) -> list[np.ndarray]:
    agent, wrist, proprio = rollout.arrays()
    proprio = (proprio - _ckpt_array(checkpoint, "proprio_mean")) / _ckpt_array(checkpoint, "proprio_std")
    instruction = tokenize_instruction(task_name)
    with torch.no_grad():
        images = torch.as_tensor(agent[None], dtype=torch.float32, device=device)
        proprio_t = torch.as_tensor(proprio[None], dtype=torch.float32, device=device)
        task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
        wrist_t = torch.as_tensor(wrist[None], dtype=torch.float32, device=device)
        instruction_t = torch.as_tensor(instruction[None], dtype=torch.long, device=device)
        if str(checkpoint.get("policy_kind", "bc")) == "flow":
            action_chunk = policy.sample_flow(
                images,
                proprio_t,
                task_t,
                wrist_images=wrist_t,
                instruction_tokens=instruction_t,
                steps=int(checkpoint.get("flow_steps", 8)),
            )
        else:
            action_chunk, _ = policy(
                images,
                proprio_t,
                task_t,
                wrist_images=wrist_t,
                instruction_tokens=instruction_t,
            )
    action_chunk_np = action_chunk[0].cpu().numpy()
    action_chunk_np = action_chunk_np * _ckpt_array(checkpoint, "action_std") + _ckpt_array(checkpoint, "action_mean")
    return [np.asarray(action, dtype=np.float32) for action in action_chunk_np]


def _proprio(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


def _wrist_image(obs: dict) -> np.ndarray:
    for key in ("robot0_eye_in_hand_image", "eye_in_hand_image", "wrist_image", "agentview_image"):
        if key in obs:
            return obs[key]
    return obs["agentview_image"]


def _ckpt_array(checkpoint: dict, key: str) -> np.ndarray:
    value = checkpoint[key]
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _device(name: str) -> torch.device:
    return device_from_arg(name)


if __name__ == "__main__":
    main()
