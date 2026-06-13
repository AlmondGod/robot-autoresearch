from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--out-dir", default="data/cosmos_robocasa_action")
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--robocasa-task-index", action="append", type=int, default=[])
    parser.add_argument("--max-demos-per-task", type=int, default=0)
    parser.add_argument("--views", nargs="+", default=["robot0_agentview_left", "robot0_agentview_right"])
    parser.add_argument("--layout", choices=["copy_first", "side_by_side"], default="side_by_side")
    parser.add_argument("--resize", type=int, default=320)
    parser.add_argument("--resize-height", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--repo-root", default=".", help="Used to repair manifest paths when data is moved to a new machine.")
    parser.add_argument(
        "--action-format",
        choices=["native12", "bridge7"],
        default="native12",
        help="Bridge7 writes 6D delta action plus binary gripper for official Cosmos action conditioning.",
    )
    args = parser.parse_args()

    manifest = _filtered_manifest(Path(args.manifest), args.task_alias)
    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    ann_root = out_dir / "annotation"
    meta_dir = out_dir / "metas"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (ann_root / split).mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    target_size = _target_size(args)

    rows = []
    for task in manifest["tasks"]:
        dataset_root = _resolve_dataset_root(Path(task["dataset_path"]), Path(args.repo_root))
        episode_paths = sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet"))
        kept = 0
        for episode_path in episode_paths:
            episode_idx = int(episode_path.stem.split("_")[-1])
            frame = pd.read_parquet(episode_path)
            task_index = int(frame["task_index"].iloc[0])
            if args.robocasa_task_index and task_index not in set(args.robocasa_task_index):
                continue
            if args.max_demos_per_task and kept >= int(args.max_demos_per_task):
                break
            uid = f"{task['alias']}_taskidx{task_index}_ep{episode_idx:06d}"
            split = _split_for_index(kept)
            video_out = videos_dir / f"{uid}.mp4"
            ann_out = ann_root / split / f"{uid}.json"
            meta_out = meta_dir / f"{uid}.txt"
            prompt = _prompt(task, task_index)
            _write_video(
                dataset_root,
                episode_idx,
                args.views,
                video_out,
                layout=str(args.layout),
                size=target_size,
                fps=int(args.fps),
            )
            annotation = _annotation(
                frame,
                prompt=prompt,
                task=task,
                task_index=task_index,
                source_episode=episode_idx,
                action_format=str(args.action_format),
                video_path=str(video_out.relative_to(out_dir)),
                split=split,
            )
            ann_out.write_text(json.dumps(annotation, indent=2, sort_keys=True))
            meta_out.write_text(prompt + "\n")
            rows.append(
                {
                    "id": uid,
                    "task": task["alias"],
                    "task_index": task_index,
                    "episode_id": episode_idx,
                    "video": str(video_out.relative_to(out_dir)),
                    "annotation": str(ann_out.relative_to(out_dir)),
                    "meta": str(meta_out.relative_to(out_dir)),
                    "split": split,
                    "frames": len(frame),
                    "prompt": prompt,
                }
            )
            kept += 1
            print(json.dumps(rows[-1]), flush=True)
    summary = {
        "format": "cosmos_action_conditioned_bridge_like",
        "notes": [
            "videos/*.mp4 and annotations/*.json follow the Cosmos action-conditioned Bridge-style directory convention.",
            "RoboCasa actions are preserved at native 12D. Bridge uses 7D, so the Cosmos dataloader/config must set action_dim=12 or adapt this field.",
            "state_eef6 is observation.state[:6]; state_full is the native RoboCasa observation.state vector.",
        ],
        "views": args.views,
        "layout": args.layout,
        "resize_height": int(target_size[0]),
        "resize_width": int(target_size[1]),
        "fps": int(args.fps),
        "action_format": str(args.action_format),
        "clips": len(rows),
        "rows": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_metadata_csv(out_dir, rows)
    print(json.dumps({"out_dir": str(out_dir), "clips": len(rows)}, indent=2))


def _filtered_manifest(path: Path, aliases: list[str]) -> dict:
    manifest = json.loads(path.read_text())
    if aliases:
        keep = set(aliases)
        manifest["tasks"] = [task for task in manifest["tasks"] if task["alias"] in keep]
    if not manifest["tasks"]:
        raise ValueError("no tasks selected")
    return manifest


def _prompt(task: dict, task_index: int) -> str:
    description = str(task.get("description") or task.get("alias") or task.get("robocasa_task"))
    return f"Robot manipulation in a RoboCasa kitchen. Task: {description} RoboCasa task_index={task_index}."


def _resolve_dataset_root(path: Path, repo_root: Path) -> Path:
    if path.exists():
        return path
    parts = path.parts
    if "third_party" in parts:
        suffix = Path(*parts[parts.index("third_party") :])
        moved = repo_root.resolve() / suffix
        if moved.exists():
            return moved
    raise FileNotFoundError(f"dataset root not found: {path}")


def _write_metadata_csv(out_dir: Path, rows: list[dict]) -> None:
    # Not required by the current Diffusers VideoDataset, but useful for auditing
    # and compatible with examples that expect a manifest-style CSV.
    pd.DataFrame(
        [
            {
                "file_name": row["video"],
                "text": row["prompt"],
                "task": row["task"],
                "task_index": row["task_index"],
                "episode_id": row["episode_id"],
                "split": row["split"],
                "frames": row["frames"],
            }
            for row in rows
        ]
    ).to_csv(out_dir / "metadata.csv", index=False)


def _annotation(
    frame: pd.DataFrame,
    *,
    prompt: str,
    task: dict,
    task_index: int,
    source_episode: int,
    action_format: str,
    video_path: str,
    split: str,
) -> dict:
    state_full = np.stack(frame["observation.state"].to_numpy()).astype(float)
    action_full = np.stack(frame["action"].to_numpy()).astype(float)
    action = _format_action(action_full, action_format)
    gripper = _gripper_state(state_full, action)
    return {
        "prompt": prompt,
        "task": str(task["alias"]),
        "task_index": int(task_index),
        "source_episode": int(source_episode),
        "split": split,
        "episode_id": f"{task['alias']}_taskidx{task_index}_ep{source_episode:06d}",
        "episode_metadata": {
            "episode_id": f"{task['alias']}_taskidx{task_index}_ep{source_episode:06d}",
            "segment_id": f"{task['alias']}_taskidx{task_index}_ep{source_episode:06d}",
            "is_eval": split != "train",
        },
        "videos": [{"video_path": video_path}],
        "state": state_full[:, :6].tolist(),
        "state_full": state_full.tolist(),
        "continuous_gripper_state": gripper.tolist(),
        "action": action.tolist(),
        "action_full": action_full.tolist(),
        "reward": frame["next.reward"].astype(float).tolist() if "next.reward" in frame else [],
        "done": frame["next.done"].astype(bool).tolist() if "next.done" in frame else [],
        "timestamp": frame["timestamp"].astype(float).tolist() if "timestamp" in frame else list(range(len(frame))),
        "action_dim": int(action.shape[-1]),
        "action_full_dim": int(action_full.shape[-1]),
        "state_dim": int(state_full.shape[-1]),
    }


def _gripper_state(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    if state.shape[-1] >= 7:
        return state[:, 6].astype(float)
    if action.shape[-1] >= 1:
        return action[:, -1].astype(float)
    return np.zeros((state.shape[0],), dtype=float)


def _format_action(action: np.ndarray, action_format: str) -> np.ndarray:
    if action_format == "native12":
        return action
    if action_format != "bridge7":
        raise ValueError(f"unsupported action format: {action_format}")
    if action.shape[-1] < 7:
        raise ValueError(f"bridge7 requires at least 7 action dims, got {action.shape[-1]}")
    out = np.zeros((action.shape[0], 7), dtype=float)
    out[:, :6] = action[:, :6]
    # Bridge convention is binary open=1, close=0. RoboCasa actions are usually
    # continuous gripper commands, so threshold the native gripper dimension.
    out[:, 6] = (action[:, -1] > 0).astype(float)
    return out


def _target_size(args: argparse.Namespace) -> tuple[int, int]:
    height = int(args.resize_height) if args.resize_height else int(args.resize)
    width = int(args.resize_width) if args.resize_width else int(args.resize)
    return height, width


def _split_for_index(index: int) -> str:
    mod = index % 10
    if mod == 8:
        return "val"
    if mod == 9:
        return "test"
    return "train"


def _write_video(
    dataset_root: Path,
    episode_idx: int,
    views: list[str],
    out: Path,
    *,
    layout: str,
    size: tuple[int, int],
    fps: int,
) -> None:
    streams = [iio.imiter(dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4") for view in views]
    frames = []
    for frame_tuple in zip(*streams, strict=False):
        if layout == "copy_first":
            frames.append(_resize(np.asarray(frame_tuple[0])[..., :3], size))
        else:
            ims = [_resize(np.asarray(frame)[..., :3], size) for frame in frame_tuple]
            frames.append(np.concatenate(ims, axis=1))
    iio.imwrite(out, frames, fps=fps, codec="libx264")


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    return np.asarray(Image.fromarray(image.astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


if __name__ == "__main__":
    main()
