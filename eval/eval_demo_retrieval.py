from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/libero_easy1_task0/manifest.json")
    parser.add_argument("--libero-root", default="third_party/LIBERO")
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--out", default="runs/libero/demo_retrieval_success.json")
    args = parser.parse_args()

    config_path = Path(".libero_config").resolve()
    if config_path.exists():
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_path))

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    manifest = json.loads(Path(args.manifest).read_text())
    suite = manifest.get("suite", "libero_object")
    benchmark = get_benchmark(suite)(0)
    name_to_task_id = {task.name: idx for idx, task in enumerate(benchmark.tasks)}

    per_task = []
    total = 0
    successes = 0
    for local_task_id, task_ref in enumerate(manifest["tasks"]):
        task_name = task_ref["task_name"].removesuffix("_demo")
        task = benchmark.get_task(name_to_task_id[task_name])
        demo_file = Path(task_ref["dataset_path"])
        starts, actions_by_key = _load_demo_starts_and_actions(demo_file)
        init_states = _as_numpy(
            torch.load(
                os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file),
                weights_only=False,
            )
        )
        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
            camera_heights=64,
            camera_widths=64,
        )
        task_successes = 0
        details = []
        try:
            for episode_idx in range(args.episodes_per_task):
                init = init_states[episode_idx % init_states.shape[0]]
                nearest_idx = int(np.linalg.norm(starts - init, axis=1).argmin())
                nearest_key = sorted(actions_by_key)[nearest_idx]
                actions = actions_by_key[nearest_key]
                env.reset()
                env.set_init_state(init)
                done = False
                reward_sum = 0.0
                for action in actions[: args.max_steps]:
                    _obs, reward, done, _info = env.step(action)
                    reward_sum += float(reward)
                    if done:
                        break
                success = bool(done or env.check_success())
                task_successes += int(success)
                successes += int(success)
                total += 1
                details.append(
                    {
                        "episode": episode_idx,
                        "nearest_demo": nearest_key,
                        "nearest_dist": float(np.linalg.norm(starts[nearest_idx] - init)),
                        "reward_sum": reward_sum,
                        "success": success,
                    }
                )
        finally:
            env.close()
        per_task.append(
            {
                "task_id": local_task_id,
                "task_name": task_name,
                "success_rate": task_successes / max(1, args.episodes_per_task),
                "episodes": details,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"success_rate": successes / max(1, total), "episodes": total, "per_task": per_task}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"out": str(out), "episodes": total, "success_rate": payload["success_rate"]}, indent=2))


def _load_demo_starts_and_actions(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    actions = {}
    starts = []
    with h5py.File(path, "r") as handle:
        for key in sorted(handle["data"].keys()):
            starts.append(np.asarray(handle["data"][f"{key}/states"][0], dtype=np.float64))
            actions[key] = np.asarray(handle["data"][f"{key}/actions"], dtype=np.float32)
    return np.stack(starts, axis=0), actions


def _as_numpy(value) -> np.ndarray:
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


if __name__ == "__main__":
    main()
