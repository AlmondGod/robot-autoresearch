from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from models.robocasa_tiny_evaluator import RoboCasaVAEWorldModel
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    device = device_from_arg(args.device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    if ckpt.get("model_type") != "robocasa_vae_world_model":
        raise ValueError("expected robocasa_vae_world_model checkpoint")
    model = RoboCasaVAEWorldModel(
        proprio_dim=int(ckpt["proprio_dim"]),
        action_dim=int(ckpt["action_dim"]),
        task_count=int(ckpt["task_count"]),
        latent_dim=int(ckpt["latent_dim"]),
        width=int(ckpt.get("width", 512)),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    trace = np.load(args.trace)
    episode_id = int(trace["episode_id"][0])
    actions = np.asarray(trace["actions"], dtype=np.float32)[: int(args.max_steps)]
    sim_success = np.asarray(trace["success"], dtype=np.bool_)[: len(actions)]
    frames = _rollout_frames(
        model=model,
        checkpoint=ckpt,
        dataset_root=Path(args.dataset_root),
        episode_id=episode_id,
        actions=actions,
        sim_success=sim_success,
        stride=max(1, int(args.stride)),
        device=device,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frames, fps=int(args.fps), codec="libx264")
    print(json.dumps({"out": str(out), "episode_id": episode_id, "frames": len(frames), "final_sim_success": bool(sim_success[-1]) if len(sim_success) else False}, indent=2))


def _rollout_frames(
    *,
    model: RoboCasaVAEWorldModel,
    checkpoint: dict,
    dataset_root: Path,
    episode_id: int,
    actions: np.ndarray,
    sim_success: np.ndarray,
    stride: int,
    device: torch.device,
) -> list[np.ndarray]:
    obs = _initial_obs(dataset_root, episode_id)
    task_id = _task_id(dataset_root, episode_id, checkpoint)
    task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
    agent = torch.as_tensor(obs["agent"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    wrist = torch.as_tensor(obs["wrist"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    proprio = (torch.as_tensor(obs["proprio"][None], dtype=torch.float32, device=device) - _tensor(checkpoint, "proprio_mean", device)) / _tensor(checkpoint, "proprio_std", device)
    frames = []
    with torch.no_grad():
        latent = model.encode(agent, wrist, proprio, task_t)
        for step, action in enumerate(actions):
            action_t = (torch.as_tensor(action[None], dtype=torch.float32, device=device) - _tensor(checkpoint, "action_mean", device)) / _tensor(checkpoint, "action_std", device)
            latent, _ = model.step(latent, action_t, task_t)
            if step % stride != 0:
                continue
            decoded = model.decode(latent, task_t)[0].detach().cpu().numpy()
            progress, success_logit = model.heads(latent, task_t)
            raw_success = torch.sigmoid(success_logit).item()
            frames.append(_compose(decoded, step, torch.sigmoid(progress).item(), 1.0 - raw_success, bool(sim_success[step]) if step < len(sim_success) else False))
    if frames:
        frames.extend([frames[-1].copy() for _ in range(12)])
    return frames


def _compose(decoded: np.ndarray, step: int, progress: float, calibrated_success: float, sim_success: bool) -> np.ndarray:
    decoded = np.clip(decoded, 0.0, 1.0)
    left = (np.transpose(decoded[:3], (1, 2, 0)) * 255.0).astype(np.uint8)
    wrist = (np.transpose(decoded[3:6], (1, 2, 0)) * 255.0).astype(np.uint8)
    scale = 4
    canvas = Image.new("RGB", (64 * scale * 2, 64 * scale + 64), "white")
    canvas.paste(Image.fromarray(left).resize((64 * scale, 64 * scale), Image.Resampling.NEAREST), (0, 0))
    canvas.paste(Image.fromarray(wrist).resize((64 * scale, 64 * scale), Image.Resampling.NEAREST), (64 * scale, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 64 * scale, canvas.width, canvas.height), fill=(255, 255, 255))
    draw.text((12, 64 * scale + 8), f"VAE world rollout step={step}", fill=(0, 0, 0))
    draw.text((12, 64 * scale + 28), f"progress={progress:.3f}  calibrated_success={calibrated_success:.3f}  sim_success={sim_success}", fill=(0, 0, 0))
    draw.text((12, 12), "decoded left view", fill=(255, 255, 255))
    draw.text((64 * scale + 12, 12), "decoded wrist/right view", fill=(255, 255, 255))
    return np.asarray(canvas)


def _initial_obs(dataset_root: Path, episode_idx: int) -> dict:
    frame = pd.read_parquet(dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet", columns=["observation.state"])
    return {
        "agent": _first_frame64(dataset_root, episode_idx, "robot0_agentview_left"),
        "wrist": _first_frame64(dataset_root, episode_idx, "robot0_agentview_right"),
        "proprio": np.asarray(frame["observation.state"].iloc[0], dtype=np.float32),
    }


def _first_frame64(dataset_root: Path, episode_idx: int, view: str) -> np.ndarray:
    path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4"
    frame = next(iio.imiter(path))
    image = np.asarray(frame, dtype=np.uint8)[..., :3]
    if image.shape[:2] != (64, 64):
        image = np.asarray(Image.fromarray(image).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)
    return image


def _task_id(dataset_root: Path, episode_idx: int, checkpoint: dict) -> int:
    if not bool(checkpoint.get("condition_on_robocasa_task_index", False)):
        return 0
    frame = pd.read_parquet(dataset_root / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet", columns=["task_index"])
    return int(frame["task_index"].iloc[0])


def _tensor(checkpoint: dict, key: str, device: torch.device) -> torch.Tensor:
    value = checkpoint[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)


if __name__ == "__main__":
    main()
