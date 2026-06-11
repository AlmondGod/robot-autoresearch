from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

from eval.eval_libero_success import _wrist_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-file", required=True)
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--settle-zeros", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    replay(args)


def replay(args: argparse.Namespace) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render MP4 video")
    config_path = Path(".libero_config").resolve()
    if config_path.exists():
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_path))

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.utils.utils as libero_utils

    with h5py.File(args.demo_file, "r") as handle:
        demo = handle["data"][args.demo_key]
        states = np.asarray(demo["states"])
        actions = np.asarray(demo["actions"], dtype=np.float32)
        model_xml = _rewrite_libero_asset_paths(libero_utils.postprocess_model_xml(demo.attrs["model_file"], {}), get_libero_path("assets"))
        bddl_file = str(handle["data"].attrs["bddl_file_name"])
        if not os.path.exists(bddl_file):
            marker = "bddl_files/"
            if marker in bddl_file:
                bddl_file = os.path.join(get_libero_path("bddl_files"), bddl_file.split(marker, 1)[1])
        task_name = Path(args.demo_file).stem.removesuffix("_demo")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=args.image_size,
        camera_widths=args.image_size,
        ignore_done=True,
    )
    frames = []
    done = False
    rewards = []
    dones = []
    state_errors = []
    try:
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        obs = env.set_init_state(states[0])
        frames.append(_compose_frame(obs, task_name, 0, False, None, "init"))
        for _ in range(args.settle_zeros):
            obs, reward, done, _info = env.step(np.zeros(actions.shape[-1], dtype=np.float32))
        limit = len(actions) if args.max_steps <= 0 else min(len(actions), args.max_steps)
        for step_idx, action in enumerate(actions[:limit], start=1):
            obs, reward, done, info = env.step(action)
            rewards.append(float(reward))
            dones.append(bool(done))
            if step_idx < len(states):
                state_errors.append(float(np.linalg.norm(states[step_idx] - env.sim.get_state().flatten())))
            frames.append(_compose_frame(obs, task_name, step_idx, bool(done), float(reward), f"success {env.check_success()}"))
    finally:
        env.close()

    out = Path(args.out)
    if frames:
        frames.extend([frames[-1].copy() for _ in range(args.fps)])
    _write_mp4(frames, out, args.fps, ffmpeg)
    report = {
        "out": str(out),
        "demo_file": args.demo_file,
        "demo_key": args.demo_key,
        "steps": len(rewards),
        "success": bool(done or any(dones)),
        "reward_sum": float(np.sum(rewards)) if rewards else 0.0,
        "done_count": int(np.sum(dones)) if dones else 0,
        "max_state_error": max(state_errors) if state_errors else None,
        "mean_state_error": float(np.mean(state_errors)) if state_errors else None,
        "settle_zeros": args.settle_zeros,
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


def _compose_frame(obs: dict, task_name: str, step_idx: int, done: bool, reward: float | None, note: str) -> np.ndarray:
    agent = np.asarray(obs["agentview_image"], dtype=np.uint8)
    wrist = np.asarray(_wrist_image(obs), dtype=np.uint8)
    scale = 5
    agent_img = Image.fromarray(agent).resize((agent.shape[1] * scale, agent.shape[0] * scale), Image.Resampling.NEAREST)
    wrist_img = Image.fromarray(wrist).resize((wrist.shape[1] * scale, wrist.shape[0] * scale), Image.Resampling.NEAREST)
    pad = 12
    header_h = 42
    width = agent_img.width + wrist_img.width + 3 * pad
    height = header_h + agent_img.height + 2 * pad
    image = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, 8], fill=(38, 150, 78) if done else (190, 55, 45))
    reward_text = "" if reward is None else f" | reward {reward:.3f}"
    draw.text((pad, 16), f"{task_name} | step {step_idx:03d}{reward_text} | {note}", fill=(20, 20, 20))
    image.paste(agent_img, (pad, header_h + pad))
    image.paste(wrist_img, (2 * pad + agent_img.width, header_h + pad))
    return np.asarray(image, dtype=np.uint8)


def _rewrite_libero_asset_paths(xml: str, assets_root: str) -> str:
    root = ET.fromstring(xml)
    asset = root.find("asset")
    if asset is None:
        return xml
    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        path = elem.get("file")
        if not path or "/assets/" not in path or os.path.exists(path) or "robosuite" in path:
            continue
        suffix = path.split("/assets/", 1)[1]
        elem.set("file", os.path.join(assets_root, suffix))
    return ET.tostring(root, encoding="utf8").decode("utf8")


def _write_mp4(frames: list[np.ndarray], out: Path, fps: int, ffmpeg: str) -> None:
    if not frames:
        raise ValueError("no frames to render")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for idx, frame in enumerate(frames):
            _write_ppm(tmp_path / f"frame_{idx:04d}.ppm", frame)
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


def _write_ppm(path: Path, image: np.ndarray) -> None:
    h, w, _ = image.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        handle.write(image.tobytes())


if __name__ == "__main__":
    main()
