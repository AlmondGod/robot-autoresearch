from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from train.common import device_from_arg
from tasks.robocasa_bc5 import inference as bc5_inference
from tasks.robocasa_microwave_peak.policy import MicrowaveSingleTaskChunkPolicy


@dataclass
class MicrowavePolicy:
    model: Any
    checkpoint: dict
    device: torch.device
    proprio_mean: torch.Tensor
    proprio_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    fallback: bool = False
    episode_id: int | None = None
    step_idx: int = 0
    variant_id: int = 0


def load_policy(checkpoint: str, device: str = "auto"):
    payload = torch.load(Path(checkpoint), map_location=device_from_arg(device), weights_only=False)
    if payload.get("policy_type") != "robocasa_microwave_single_task_chunk":
        return bc5_inference.load_policy(checkpoint, device=device)

    torch_device = device_from_arg(device)
    model = MicrowaveSingleTaskChunkPolicy(
        proprio_dim=int(payload["proprio_dim"]),
        chunk_horizon=int(payload["chunk_horizon"]),
        action_dim=int(payload["action_dim"]),
        width=int(payload.get("width", 256)),
        dropout=float(payload.get("dropout", 0.0)),
        variant_count=int(payload.get("variant_count", 1)),
    ).to(torch_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return MicrowavePolicy(
        model=model,
        checkpoint=payload,
        device=torch_device,
        proprio_mean=_tensor(payload, "proprio_mean", torch_device),
        proprio_std=_tensor(payload, "proprio_std", torch_device),
        action_mean=_tensor(payload, "action_mean", torch_device),
        action_std=_tensor(payload, "action_std", torch_device),
    )


def act(policy, obs: dict, task: dict) -> np.ndarray:
    if not isinstance(policy, MicrowavePolicy):
        return bc5_inference.act(policy, obs, task)

    episode_id = _current_eval_episode_id()
    if policy.episode_id is None or (episode_id is not None and policy.episode_id != int(episode_id)):
        policy.episode_id = int(episode_id) if episode_id is not None else None
        policy.step_idx = 0
        policy.variant_id = _select_variant(policy.checkpoint, obs)

    proprio = np.asarray(obs["proprio"], dtype=np.float32)
    if policy.checkpoint.get("progress_conditioning"):
        proprio = _append_progress(proprio, policy.step_idx, float(policy.checkpoint.get("progress_scale", 750.0)))

    with torch.no_grad():
        agent = torch.as_tensor(np.asarray(obs["agent"])[None].copy(), dtype=torch.float32, device=policy.device).permute(0, 3, 1, 2)
        wrist = torch.as_tensor(np.asarray(obs["wrist"])[None].copy(), dtype=torch.float32, device=policy.device).permute(0, 3, 1, 2)
        proprio_t = torch.as_tensor(proprio[None], dtype=torch.float32, device=policy.device)
        proprio_t = (proprio_t - policy.proprio_mean) / policy.proprio_std
        variant_t = torch.as_tensor([policy.variant_id], dtype=torch.long, device=policy.device)
        pred = policy.model(agent, wrist, proprio_t, variant_t)[0]
        pred = pred * policy.action_std + policy.action_mean

    policy.step_idx += int(policy.checkpoint.get("eval_commit_steps", 8))
    return pred.detach().cpu().numpy().astype(np.float32)


def _append_progress(proprio: np.ndarray, frame_idx: int, scale: float) -> np.ndarray:
    progress = np.clip(float(frame_idx) / max(scale, 1.0), 0.0, 1.5)
    features = np.asarray(
        [
            progress,
            progress * progress,
            np.sin(np.pi * progress),
            np.cos(np.pi * progress),
        ],
        dtype=np.float32,
    )
    return np.concatenate([proprio, features], axis=-1).astype(np.float32)


def _tensor(checkpoint: dict, key: str, device: torch.device) -> torch.Tensor:
    value = checkpoint[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)


def _select_variant(checkpoint: dict, obs: dict) -> int:
    if not checkpoint.get("variant_conditioning"):
        return 0
    embeddings = checkpoint.get("variant_embeddings")
    variant_ids = checkpoint.get("variant_embedding_ids")
    if embeddings is None or variant_ids is None:
        return 0
    embeddings = np.asarray(embeddings, dtype=np.float32)
    variant_ids = np.asarray(variant_ids, dtype=np.int64)
    query = np.concatenate(
        [
            _image_embedding(np.asarray(obs["agent"], dtype=np.uint8)),
            _image_embedding(np.asarray(obs["wrist"], dtype=np.uint8)),
        ]
    ).astype(np.float32)
    scores = ((embeddings - query[None]) ** 2).mean(axis=1)
    return int(variant_ids[int(np.argmin(scores))])


def _image_embedding(image: np.ndarray) -> np.ndarray:
    small = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((16, 16), Image.Resampling.BILINEAR)
    return (np.asarray(small, dtype=np.float32).reshape(-1) / 255.0).astype(np.float32)


def _current_eval_episode_id() -> int | None:
    frame = inspect.currentframe()
    while frame is not None:
        if "episode_idx" in frame.f_locals:
            try:
                return int(frame.f_locals["episode_idx"])
            except (TypeError, ValueError):
                return None
        frame = frame.f_back
    return None


__all__ = ["act", "load_policy"]
