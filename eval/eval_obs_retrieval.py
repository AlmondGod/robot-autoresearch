from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from eval.eval_libero_success import _proprio, _wrist_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/libero_easy1_task0/manifest.json")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--metric", choices=["proprio", "image", "proprio_image"], default="proprio")
    parser.add_argument("--out", default="runs/libero/obs_retrieval_success.json")
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

    total = 0
    successes = 0
    per_task = []
    for local_task_id, task_ref in enumerate(manifest["tasks"]):
        task_name = task_ref["task_name"].removesuffix("_demo")
        task = benchmark.get_task(name_to_task_id[task_name])
        demos = _load_demo_library(Path(task_ref["dataset_path"]))
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
        details = []
        task_successes = 0
        try:
            for episode_idx in range(args.episodes_per_task):
                env.reset()
                obs = env.set_init_state(init_states[episode_idx % init_states.shape[0]])
                query = _features_from_obs(obs, args.metric)
                nearest_key, nearest_dist = _nearest_demo(query, demos, args.metric)
                done = False
                reward_sum = 0.0
                for action in demos[nearest_key]["actions"][: args.max_steps]:
                    obs, reward, done, _info = env.step(action)
                    reward_sum += float(reward)
                    if done:
                        break
                success = bool(done or env.check_success())
                total += 1
                successes += int(success)
                task_successes += int(success)
                details.append(
                    {
                        "episode": episode_idx,
                        "nearest_demo": nearest_key,
                        "nearest_dist": nearest_dist,
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

    payload = {"metric": args.metric, "success_rate": successes / max(1, total), "episodes": total, "per_task": per_task}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"out": str(out), "episodes": total, "success_rate": payload["success_rate"], "metric": args.metric}, indent=2))


def _load_demo_library(path: Path) -> dict[str, dict[str, np.ndarray]]:
    demos = {}
    with h5py.File(path, "r") as handle:
        for key in sorted(handle["data"].keys()):
            group = handle["data"][key]
            agent = np.asarray(group["obs/agentview_rgb"][0], dtype=np.uint8)
            wrist = np.asarray(group["obs/eye_in_hand_rgb"][0], dtype=np.uint8)
            demos[key] = {
                "actions": np.asarray(group["actions"], dtype=np.float32),
                "proprio": np.asarray(group["robot_states"][0], dtype=np.float32),
                "image": _image_feature(agent, wrist),
            }
    return demos


def _features_from_obs(obs: dict, metric: str) -> dict[str, np.ndarray]:
    return {
        "proprio": _proprio(obs),
        "image": _image_feature(np.asarray(obs["agentview_image"], dtype=np.uint8), np.asarray(_wrist_image(obs), dtype=np.uint8)),
    }


def _nearest_demo(query: dict[str, np.ndarray], demos: dict[str, dict[str, np.ndarray]], metric: str) -> tuple[str, float]:
    best_key = ""
    best_dist = float("inf")
    for key, demo in demos.items():
        if metric == "proprio":
            dist = float(np.linalg.norm(query["proprio"] - demo["proprio"]))
        elif metric == "image":
            dist = float(np.linalg.norm(query["image"] - demo["image"]))
        else:
            prop_dist = np.linalg.norm(query["proprio"] - demo["proprio"])
            image_dist = np.linalg.norm(query["image"] - demo["image"])
            dist = float(prop_dist + 0.1 * image_dist)
        if dist < best_dist:
            best_key = key
            best_dist = dist
    return best_key, best_dist


def _image_feature(agent: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    feat = np.concatenate([_downsample_mean(agent), _downsample_mean(wrist)], axis=0)
    return feat.astype(np.float32) / 255.0


def _downsample_mean(image: np.ndarray, bins: int = 8) -> np.ndarray:
    h, w = image.shape[:2]
    image = image[: h - h % bins, : w - w % bins, :3]
    return image.reshape(bins, image.shape[0] // bins, bins, image.shape[1] // bins, 3).mean(axis=(1, 3)).reshape(-1)


def _as_numpy(value) -> np.ndarray:
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


if __name__ == "__main__":
    main()
