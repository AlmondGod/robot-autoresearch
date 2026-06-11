from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

import robocasa.utils.lerobot_utils as LU
from eval.train_full_trajectory_bc_robocasa import RoboCasaFullTrajectoryBC, _resize64
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-id", type=int, default=90)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render MP4 video")

    dataset_root = Path(args.dataset_root)
    device = device_from_arg(args.device)
    checkpoint = torch.load(args.policy, map_location=device)
    model = RoboCasaFullTrajectoryBC(
        proprio_dim=int(checkpoint["proprio_dim"]),
        horizon=int(checkpoint["horizon"]),
        action_dim=int(checkpoint["action_dim"]),
        task_count=int(checkpoint["task_count"]),
        width=int(checkpoint.get("width", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    row = _load_episode_row(dataset_root, int(args.episode_id))
    with torch.no_grad():
        pred = model(
            torch.as_tensor(row["agent"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
            torch.as_tensor(row["wrist"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
            torch.as_tensor(row["proprio"][None], dtype=torch.float32, device=device),
            torch.as_tensor([row["task_id"]], dtype=torch.long, device=device),
        )[0].cpu().numpy()
    actions = pred[: min(args.max_steps, pred.shape[0])].astype(np.float32)

    frames, success = _rollout_open_loop(
        dataset_root=dataset_root,
        episode_idx=int(args.episode_id),
        actions=actions,
        camera=args.camera,
        width=int(args.width),
        height=int(args.height),
    )
    if frames:
        frames.extend([frames[-1].copy() for _ in range(args.fps)])
    _write_mp4(frames, Path(args.out), int(args.fps), ffmpeg)
    payload = {
        "out": args.out,
        "frames": len(frames),
        "episode_id": int(args.episode_id),
        "success": bool(success),
        "camera": args.camera,
    }
    print(json.dumps(payload, indent=2))


def _load_episode_row(dataset_root: Path, episode_idx: int) -> dict:
    frame = pd.read_parquet(dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet")
    return {
        "task_id": 0,
        "agent": _first_video_frame(dataset_root, episode_idx, "robot0_agentview_left"),
        "wrist": _first_video_frame(dataset_root, episode_idx, "robot0_agentview_right"),
        "proprio": np.asarray(frame["observation.state"].iloc[0], dtype=np.float32),
    }


def _first_video_frame(dataset_root: Path, episode_idx: int, view: str) -> np.ndarray:
    video_path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4"
    frame = iio.imread(video_path, index=0)
    return _resize64(np.asarray(frame, dtype=np.uint8))


def _rollout_open_loop(
    *,
    dataset_root: Path,
    episode_idx: int,
    actions: np.ndarray,
    camera: str,
    width: int,
    height: int,
) -> tuple[list[np.ndarray], bool]:
    import robosuite
    import robocasa  # noqa: F401
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    env_meta = LU.get_env_metadata(dataset_root)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    env = robosuite.make(**env_kwargs)
    model_xml = LU.get_episode_model_xml(dataset_root, episode_idx)
    states = LU.get_episode_states(dataset_root, episode_idx)
    ep_meta = LU.get_episode_meta(dataset_root, episode_idx)
    reset_to(env, {"model": model_xml, "ep_meta": json.dumps(ep_meta), "states": states[0]})

    frames: list[np.ndarray] = []
    success = False
    try:
        frames.append(_compose_frame(env, camera, width, height, 0, success=False))
        for step_idx, action in enumerate(actions, start=1):
            _, _, _, info = env.step(action)
            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            if not success and hasattr(env, "_check_success"):
                try:
                    success = bool(env._check_success())
                except Exception:
                    pass
            frames.append(_compose_frame(env, camera, width, height, step_idx, success=success))
            if success:
                break
    finally:
        try:
            if getattr(env, "viewer", None) is not None:
                env.viewer.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
    return frames, success


def _compose_frame(env, camera: str, width: int, height: int, step_idx: int, success: bool) -> np.ndarray:
    image = env.sim.render(height=height, width=width, camera_name=camera)[::-1]
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    draw = ImageDraw.Draw(pil)
    bar_color = (38, 150, 78) if success else (190, 55, 45)
    draw.rectangle([0, 0, pil.width, 10], fill=bar_color)
    draw.rectangle([10, 16, 250, 46], fill=(255, 255, 255))
    draw.text((18, 24), f"step {step_idx:03d} | success {int(success)}", fill=(20, 20, 20))
    return np.asarray(pil, dtype=np.uint8)


def _write_mp4(frames: list[np.ndarray], out: Path, fps: int, ffmpeg: str) -> None:
    if not frames:
        raise ValueError("no frames to render")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for idx, frame in enumerate(frames):
            ppm = tmp_path / f"frame_{idx:04d}.ppm"
            with ppm.open("wb") as handle:
                h, w, _ = frame.shape
                handle.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
                handle.write(frame.tobytes())
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_path / "frame_%04d.ppm"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
