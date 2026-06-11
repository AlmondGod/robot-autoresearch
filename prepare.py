from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


LIBERO_ROOT = Path("third_party/LIBERO")
BENCHMARK_NAME = "libero_easy5"
LIBERO_SUITE = "libero_goal"
EASY_TASKS = [
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet",
]
DATA_DIR = Path("data") / BENCHMARK_NAME
MANIFEST = DATA_DIR / "manifest.json"
VIDEO_SHARD = DATA_DIR / f"{BENCHMARK_NAME}_video.npz"
PAIRED_SHARD = DATA_DIR / f"{BENCHMARK_NAME}_paired.npz"
TASK_COUNT = 5
VIDEO_TASK_COUNT = 5
PAIRED_DEMOS_PER_TASK = 10
VIDEO_REPEAT_FACTOR = 10
IMAGE_SIZE = 64
HISTORY = 4
ACTION_HORIZON = 4
MAX_TRANSITIONS_PER_DEMO = 120
TRAIN_TIME_SECONDS = 300
EVAL_EPISODES_PER_TASK = 1
EVAL_MAX_STEPS = 150


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--make-shards", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--eval-policy", default="")
    parser.add_argument("--eval-out", default="")
    args = parser.parse_args()

    if args.download:
        _run(["python", "data/download_libero.py", "--dataset", LIBERO_SUITE, "--use-huggingface"])
    if args.make_shards:
        _run(
            [
                "python",
                "data/make_libero5.py",
                "--libero-root",
                str(LIBERO_ROOT),
                "--out-dir",
                str(DATA_DIR),
                "--suite",
                LIBERO_SUITE,
                "--task-count",
                str(TASK_COUNT),
                "--video-task-count",
                str(VIDEO_TASK_COUNT),
                "--paired-demos-per-task",
                str(PAIRED_DEMOS_PER_TASK),
                "--video-repeat-factor",
                str(VIDEO_REPEAT_FACTOR),
                "--tasks",
                *EASY_TASKS,
            ]
        )
        _run(
            [
                "python",
                "data/split_video_and_paired.py",
                "--manifest",
                str(MANIFEST),
                "--out-dir",
                str(DATA_DIR),
                "--image-size",
                str(IMAGE_SIZE),
                "--max-transitions-per-demo",
                str(MAX_TRANSITIONS_PER_DEMO),
                "--history",
                str(HISTORY),
                "--action-horizon",
                str(ACTION_HORIZON),
                "--video-repeat-factor",
                str(VIDEO_REPEAT_FACTOR),
            ]
        )
    if args.eval_policy:
        eval_policy(Path(args.eval_policy), Path(args.eval_out) if args.eval_out else None)
    if args.summary or not (args.download or args.make_shards or args.eval_policy):
        print(json.dumps(dataset_summary(), indent=2, sort_keys=True))


def dataset_summary() -> dict:
    summary: dict = {
        "libero_root": str(LIBERO_ROOT),
        "benchmark_name": BENCHMARK_NAME,
        "libero_suite": LIBERO_SUITE,
        "data_dir": str(DATA_DIR),
        "train_time_seconds": TRAIN_TIME_SECONDS,
        "eval_episodes_per_task": EVAL_EPISODES_PER_TASK,
        "eval_max_steps": EVAL_MAX_STEPS,
    }
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        summary["tasks"] = [task["task_name"] for task in manifest["tasks"]]
        summary["video_tasks"] = [task["task_name"] for task in manifest.get("video_tasks", manifest["tasks"])]
        summary["video_repeat_factor"] = int(manifest.get("video_repeat_factor", 1))
        summary["demo_count"] = len(manifest["demos"])
        summary["paired_demo_count"] = sum(1 for demo in manifest["demos"] if demo["paired"])
    if VIDEO_SHARD.exists():
        video = np.load(VIDEO_SHARD)
        summary["video"] = _shape_summary(video)
    if PAIRED_SHARD.exists():
        paired = np.load(PAIRED_SHARD)
        summary["paired"] = _shape_summary(paired)
    return summary


def eval_policy(policy: Path, out: Path | None = None) -> dict:
    out_path = out or policy.with_name("libero_success.json")
    _run(
        [
            "python",
            "eval/eval_libero_success.py",
            "--policy",
            str(policy),
            "--episodes-per-task",
            str(EVAL_EPISODES_PER_TASK),
            "--max-steps",
            str(EVAL_MAX_STEPS),
            "--device",
            "cpu",
            "--out",
            str(out_path),
        ]
    )
    return json.loads(out_path.read_text())


def _shape_summary(data: np.lib.npyio.NpzFile) -> dict:
    fields = {}
    for key in data.files:
        arr = data[key]
        if key in {"frames", "wrist_frames", "next_frames", "proprio", "actions", "instruction_tokens", "split", "task_id"}:
            fields[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if "split" in data.files:
        values, counts = np.unique(data["split"], return_counts=True)
        fields["split_counts"] = {str(value): int(count) for value, count in zip(values, counts, strict=False)}
    if "sample_index" in data.files:
        values, counts = np.unique(data["virtual_split"], return_counts=True)
        fields["virtual_split_counts"] = {str(value): int(count) for value, count in zip(values, counts, strict=False)}
        fields["effective_samples"] = int(len(data["sample_index"]))
        fields["raw_samples"] = int(len(data["frames"]))
        fields["video_repeat_factor"] = int(data["video_repeat_factor"])
    return fields


def _run(cmd: list[str]) -> None:
    if cmd and cmd[0] == "python":
        cmd = [sys.executable, *cmd[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
