from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image

from eval.eval_world_model_ranking import _xml_escape
from models.robocasa_tiny_evaluator import RoboCasaTinyEvaluator, RoboCasaVAEWorldModel
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--invert-success", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    ckpt = torch.load(args.evaluator, map_location=device, weights_only=False)
    cls = RoboCasaVAEWorldModel if ckpt.get("model_type") == "robocasa_vae_world_model" else RoboCasaTinyEvaluator
    model = cls(
        proprio_dim=int(ckpt["proprio_dim"]),
        action_dim=int(ckpt["action_dim"]),
        task_count=int(ckpt["task_count"]),
        latent_dim=int(ckpt["latent_dim"]),
        width=int(ckpt.get("width", 512)),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    trace_path = Path(args.trace)
    trace = np.load(trace_path)
    episode_id = int(trace["episode_id"][0])
    actions = np.asarray(trace["actions"], dtype=np.float32)[: int(args.max_steps)]
    sim_success = np.asarray(trace["success"], dtype=np.bool_)[: len(actions)]
    obs = _initial_obs(Path(args.dataset_root), episode_id)
    task_id = _task_id(Path(args.dataset_root), episode_id, ckpt)
    curves = _rollout_curves(model, ckpt, obs, actions, task_id, device, invert_success=bool(args.invert_success))
    curves["sim_success"] = sim_success.astype(np.float32)
    curves["episode_id"] = episode_id
    curves["trace"] = str(trace_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in curves.items()}, indent=2))
    else:
        _write_svg(curves, out)
    print(json.dumps({"out": str(out), "episode_id": episode_id, "steps": len(actions), "final_sim_success": bool(sim_success[-1]) if len(sim_success) else False}, indent=2))


def _rollout_curves(
    model: RoboCasaTinyEvaluator,
    ckpt: dict,
    obs: dict,
    actions: np.ndarray,
    task_id: int,
    device: torch.device,
    *,
    invert_success: bool,
) -> dict[str, np.ndarray]:
    task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
    agent = torch.as_tensor(obs["agent"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    wrist = torch.as_tensor(obs["wrist"][None], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    proprio = (torch.as_tensor(obs["proprio"][None], dtype=torch.float32, device=device) - _tensor(ckpt, "proprio_mean", device)) / _tensor(ckpt, "proprio_std", device)
    progress_values = []
    raw_success_values = []
    calibrated_values = []
    latent_norms = []
    proprio_norms = []
    with torch.no_grad():
        latent = model.encode(agent, wrist, proprio, task_t)
        for action in actions:
            action_t = (torch.as_tensor(action[None], dtype=torch.float32, device=device) - _tensor(ckpt, "action_mean", device)) / _tensor(ckpt, "action_std", device)
            latent, pred_proprio = model.step(latent, action_t, task_t)
            progress, success_logit = model.heads(latent, task_t)
            raw_success = torch.sigmoid(success_logit).item()
            raw_success_values.append(raw_success)
            calibrated_values.append(1.0 - raw_success if invert_success else raw_success)
            progress_values.append(torch.sigmoid(progress).item())
            latent_norms.append(float(latent.norm(dim=-1).item()))
            proprio_norms.append(float(pred_proprio.norm(dim=-1).item()))
    return {
        "progress": np.asarray(progress_values, dtype=np.float32),
        "raw_success": np.asarray(raw_success_values, dtype=np.float32),
        "calibrated_success": np.asarray(calibrated_values, dtype=np.float32),
        "latent_norm": np.asarray(latent_norms, dtype=np.float32),
        "pred_proprio_norm": np.asarray(proprio_norms, dtype=np.float32),
    }


def _write_svg(curves: dict, out: Path) -> None:
    steps = len(curves["progress"])
    width, height = 1000, 560
    left, right, top, bottom = 76, 32, 66, 70
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(i: int) -> float:
        return left + plot_w * i / max(1, steps - 1)

    def y(v: float) -> float:
        return top + plot_h * (1.0 - min(1.0, max(0.0, v)))

    def poly(values: np.ndarray, color: str, width_px: float = 2.5) -> str:
        pts = " ".join(f"{x(i):.1f},{y(float(v)):.1f}" for i, v in enumerate(values))
        return f'<polyline fill="none" stroke="{color}" stroke-width="{width_px}" points="{pts}"/>'

    sim = curves["sim_success"]
    first_success = int(np.argmax(sim > 0.5)) if np.any(sim > 0.5) else None
    title = f"TinyEvaluator Latent Rollout - episode {curves['episode_id']}"
    trace = _xml_escape(str(curves["trace"]))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="32" font-family="Arial" font-size="22" font-weight="700">{title}</text>',
        f'<text x="24" y="54" font-family="Arial" font-size="12" fill="#555">{trace}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    for tick in range(6):
        v = tick / 5
        yy = y(v)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{v:.1f}</text>')
    if first_success is not None:
        sx = x(first_success)
        parts.append(f'<rect x="{sx:.1f}" y="{top}" width="{width - right - sx:.1f}" height="{plot_h}" fill="#2ecc71" opacity="0.08"/>')
        parts.append(f'<line x1="{sx:.1f}" y1="{top}" x2="{sx:.1f}" y2="{top + plot_h}" stroke="#2ecc71" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{sx + 6:.1f}" y="{top + 16}" font-family="Arial" font-size="12" fill="#1d7f43">sim success</text>')
    parts.append(poly(curves["calibrated_success"], "#2f80ed", 3.0))
    parts.append(poly(curves["progress"], "#f2994a", 2.5))
    raw = curves["raw_success"]
    parts.append(poly(raw, "#9b51e0", 1.5))
    parts.extend(
        [
            f'<text x="{left}" y="{height - 28}" font-family="Arial" font-size="13">step</text>',
            f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="13">model output</text>',
            f'<circle cx="{width - 300}" cy="88" r="5" fill="#2f80ed"/><text x="{width - 286}" y="92" font-family="Arial" font-size="12">calibrated success = 1 - raw risk</text>',
            f'<circle cx="{width - 300}" cy="108" r="5" fill="#f2994a"/><text x="{width - 286}" y="112" font-family="Arial" font-size="12">predicted progress</text>',
            f'<circle cx="{width - 300}" cy="128" r="5" fill="#9b51e0"/><text x="{width - 286}" y="132" font-family="Arial" font-size="12">raw success head / risk</text>',
            "</svg>",
        ]
    )
    out.write_text("\n".join(parts) + "\n")


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
