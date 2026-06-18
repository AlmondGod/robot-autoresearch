from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--task-alias", default="OpenDrawer")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--view", default="robot0_agentview_left")
    parser.add_argument("--out-dir", default="data/oscar_robocasa/opendrawer_task0_ep000000")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    task = next((item for item in manifest["tasks"] if item["alias"] == args.task_alias), None)
    if task is None:
        raise ValueError(f"unknown task alias: {args.task_alias}")

    dataset_root = _resolve_dataset_root(Path(task["dataset_path"]), Path(args.repo_root))
    episode_path = dataset_root / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
    video_path = (
        dataset_root
        / "videos"
        / "chunk-000"
        / f"observation.images.{args.view}"
        / f"episode_{args.episode:06d}.mp4"
    )
    frame = pd.read_parquet(episode_path)
    if int(frame["task_index"].iloc[0]) != int(args.task_index):
        raise ValueError(
            f"episode {args.episode} has task_index={int(frame['task_index'].iloc[0])}, expected {args.task_index}"
        )

    rgb = _read_resize_video(video_path, args.height, args.width)
    states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
    cond = _render_condition_video(states, actions, args.height, args.width)
    if len(cond) != len(rgb):
        n = min(len(cond), len(rgb))
        rgb = rgb[:n]
        cond = cond[:n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_mp4(out_dir / "rgb.mp4", rgb, args.fps)
    _write_mp4(out_dir / "gripper_scenario.mp4", cond, args.fps)

    prompt = (
        f"Robot manipulation in a RoboCasa kitchen. Task: {task['description']} "
        f"RoboCasa task_index={args.task_index}."
    )
    with open(out_dir / "caption.pickle", "wb") as f:
        pickle.dump(prompt, f)

    metadata = {
        "format": "oscar_asset_proxy_conditioning",
        "task_alias": args.task_alias,
        "task_index": int(args.task_index),
        "episode": int(args.episode),
        "view": args.view,
        "frames": int(len(rgb)),
        "height": int(args.height),
        "width": int(args.width),
        "fps": int(args.fps),
        "prompt": prompt,
        "source_episode": str(episode_path),
        "source_video": str(video_path),
        "conditioning_note": (
            "gripper_scenario.mp4 is a proxy kinematic/action render derived from RoboCasa "
            "state/action arrays, not the official OSCAR skeleton renderer."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps({"out_dir": str(out_dir), "frames": len(rgb), "prompt": prompt}, indent=2))


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


def _read_resize_video(path: Path, height: int, width: int) -> np.ndarray:
    frames = []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"could not open video: {path}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8))
        frames.append(np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8))
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames)


def _render_condition_video(states: np.ndarray, actions: np.ndarray, height: int, width: int) -> np.ndarray:
    eef = _eef_xyz(states)
    xy = _normalize_xy(eef[:, :2], width, height)
    gripper = states[:, 6] if states.shape[1] > 6 else actions[:, -1]
    action_scale = np.maximum(np.abs(actions).max(axis=0), 1e-6)
    frames = []
    for idx in range(len(states)):
        image = Image.new("RGB", (width, height), (8, 8, 8))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width - 1, height - 1), outline=(60, 60, 60), width=2)

        # Workspace grid.
        for frac in (0.25, 0.5, 0.75):
            x = int(width * frac)
            y = int(height * frac)
            draw.line((x, 0, x, height), fill=(28, 28, 28), width=1)
            draw.line((0, y, width, y), fill=(28, 28, 28), width=1)

        if idx > 0:
            pts = [tuple(map(int, point)) for point in xy[: idx + 1]]
            if len(pts) > 1:
                draw.line(pts, fill=(0, 210, 255), width=3)

        x, y = map(int, xy[idx])
        radius = int(8 + 8 * _unit_interval(gripper[idx], gripper.min(), gripper.max()))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 210, 60), outline=(255, 255, 255))

        # Compact action bars in the lower-left corner encode the current 12D action.
        bar_w = max(2, width // 80)
        bar_gap = max(1, width // 160)
        base_x = 10
        base_y = height - 18
        max_h = max(18, height // 5)
        for dim, value in enumerate(actions[idx]):
            x0 = base_x + dim * (bar_w + bar_gap)
            h = int((abs(float(value)) / float(action_scale[dim])) * max_h)
            color = (80, 220, 120) if value >= 0 else (245, 90, 90)
            draw.rectangle((x0, base_y - h, x0 + bar_w, base_y), fill=color)
        frames.append(np.asarray(image, dtype=np.uint8))
    return np.stack(frames)


def _write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected NHWC RGB frames, got {frames.shape}")
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise OSError(f"could not open video writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def _eef_xyz(states: np.ndarray) -> np.ndarray:
    if states.shape[1] >= 10:
        return states[:, 7:10]
    if states.shape[1] >= 3:
        return states[:, :3]
    padded = np.zeros((len(states), 3), dtype=np.float32)
    padded[:, : states.shape[1]] = states
    return padded


def _normalize_xy(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    norm = (xy - lo) / span
    out = np.empty_like(norm)
    out[:, 0] = 20 + norm[:, 0] * (width - 40)
    out[:, 1] = height - (20 + norm[:, 1] * (height - 40))
    return out


def _unit_interval(value: float, lo: float, hi: float) -> float:
    span = max(float(hi - lo), 1e-6)
    return max(0.0, min(1.0, (float(value) - float(lo)) / span))


if __name__ == "__main__":
    main()
