from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from data.export_robocasa_oscar_asset import _read_resize_video, _resolve_dataset_root, _write_mp4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--archive", default="runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl")
    parser.add_argument("--out-dir", default="data/oscar_robocasa/trace_eval_frontier_ep87")
    parser.add_argument("--episode", type=int, default=87)
    parser.add_argument("--view", default="robot0_agentview_left")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    task = next(item for item in manifest["tasks"] if item["alias"] == "OpenDrawer")
    dataset_root = _resolve_dataset_root(Path(task["dataset_path"]), Path(args.repo_root))
    source_video = (
        dataset_root
        / "videos"
        / "chunk-000"
        / f"observation.images.{args.view}"
        / f"episode_{args.episode:06d}.mp4"
    )
    rgb = _read_resize_video(source_video, args.height, args.width)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in _load_archive(Path(args.archive)):
        eval_path = Path(row["eval_path"])
        if not eval_path.is_absolute():
            eval_path = Path(args.repo_root).resolve() / eval_path
        eval_data = json.loads(eval_path.read_text())
        detail = next(item for item in eval_data["details"] if int(item["episode_id"]) == int(args.episode))
        trace_path = Path(detail["trace_path"])
        trace = np.load(trace_path, allow_pickle=True)
        actions = trace["actions"].astype(np.float32)
        n = min(len(rgb), len(actions))
        candidate_id = int(row["experiment"])
        asset_dir = out_root / f"candidate_{candidate_id:03d}"
        asset_dir.mkdir(parents=True, exist_ok=True)
        _write_mp4(asset_dir / "rgb.mp4", rgb[:n], args.fps)
        _write_mp4(asset_dir / "gripper_scenario.mp4", _render_action_condition(actions[:n], args.height, args.width), args.fps)
        prompt = (
            f"Robot manipulation in a RoboCasa kitchen. Task: {task['description']} "
            f"Candidate policy {candidate_id}, held-out episode {args.episode}."
        )
        with open(asset_dir / "caption.pickle", "wb") as f:
            pickle.dump(prompt, f)
        metadata = {
            "candidate_id": candidate_id,
            "change": row.get("change"),
            "episode": int(args.episode),
            "episode_success": bool(detail["success"]),
            "success_rate": float(row["success_rate"]),
            "frames": int(n),
            "prompt": prompt,
            "trace_path": str(trace_path),
            "source_video": str(source_video),
        }
        (asset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        rows.append({"asset": str(asset_dir), **metadata})
        print(json.dumps(rows[-1]), flush=True)
    (out_root / "manifest.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True))
    print(json.dumps({"out_dir": str(out_root), "assets": len(rows)}, indent=2))


def _load_archive(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _render_action_condition(actions: np.ndarray, height: int, width: int) -> np.ndarray:
    delta = actions[:, :2] if actions.shape[1] >= 2 else np.zeros((len(actions), 2), dtype=np.float32)
    path = np.cumsum(delta, axis=0)
    path = _normalize_xy(path, width, height)
    scale = np.maximum(np.abs(actions).max(axis=0), 1e-6)
    frames = []
    for idx, action in enumerate(actions):
        image = Image.new("RGB", (width, height), (8, 8, 8))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width - 1, height - 1), outline=(60, 60, 60), width=2)
        for frac in (0.25, 0.5, 0.75):
            x = int(width * frac)
            y = int(height * frac)
            draw.line((x, 0, x, height), fill=(28, 28, 28), width=1)
            draw.line((0, y, width, y), fill=(28, 28, 28), width=1)
        pts = [tuple(map(int, point)) for point in path[: idx + 1]]
        if len(pts) > 1:
            draw.line(pts, fill=(0, 210, 255), width=3)
        x, y = map(int, path[idx])
        grip = float(action[-1]) if len(action) else 0.0
        radius = int(10 + 6 * (grip > 0))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 210, 60), outline=(255, 255, 255))
        bar_w = max(2, width // 80)
        bar_gap = max(1, width // 160)
        base_x = 10
        base_y = height - 18
        max_h = max(18, height // 5)
        for dim, value in enumerate(action):
            x0 = base_x + dim * (bar_w + bar_gap)
            h = int((abs(float(value)) / float(scale[dim])) * max_h)
            color = (80, 220, 120) if value >= 0 else (245, 90, 90)
            draw.rectangle((x0, base_y - h, x0 + bar_w, base_y), fill=color)
        frames.append(np.asarray(image, dtype=np.uint8))
    return np.stack(frames)


def _normalize_xy(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    norm = (xy - lo) / span
    out = np.empty_like(norm)
    out[:, 0] = 20 + norm[:, 0] * (width - 40)
    out[:, 1] = height - (20 + norm[:, 1] * (height - 40))
    return out


if __name__ == "__main__":
    main()
