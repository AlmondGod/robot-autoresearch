from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from data.libero_dataset import INSTRUCTION_LENGTH, INSTRUCTION_VOCAB_SIZE


class TinyBCPolicy(nn.Module):
    def __init__(
        self,
        task_count: int = 5,
        action_dim: int = 7,
        proprio_dim: int = 1,
        n_embd: int = 128,
        action_horizon: int = 1,
        max_history: int = 8,
        instruction_vocab_size: int = INSTRUCTION_VOCAB_SIZE,
        instruction_length: int = INSTRUCTION_LENGTH,
        policy_kind: str = "bc",
        flow_steps: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_history = max_history
        self.instruction_length = instruction_length
        self.policy_kind = policy_kind
        self.flow_steps = flow_steps
        self.image = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, n_embd),
        )
        self.wrist_image = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, n_embd),
        )
        self.task_emb = nn.Embedding(task_count, n_embd)
        self.word_emb = nn.Embedding(instruction_vocab_size, n_embd, padding_idx=0)
        self.proprio = nn.Linear(proprio_dim, n_embd)
        self.time_emb = nn.Parameter(torch.zeros(1, max_history, n_embd))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=4,
            dim_feedforward=4 * n_embd,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fuse = nn.Sequential(
            nn.LayerNorm(5 * n_embd),
            nn.Linear(5 * n_embd, n_embd),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, action_horizon * action_dim),
        )
        self.action_in = nn.Linear(action_horizon * action_dim, n_embd)
        self.flow_time = nn.Sequential(
            nn.Linear(1, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd),
        )
        self.flow_decoder = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, action_horizon * action_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        wrist_images: torch.Tensor | None = None,
        instruction_tokens: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ):
        images = _ensure_image_history(images)
        wrist_images = images if wrist_images is None else _ensure_image_history(wrist_images)
        proprio = _ensure_proprio_history(proprio)
        batch, history = images.shape[:2]
        if history > self.max_history:
            images = images[:, -self.max_history :]
            wrist_images = wrist_images[:, -self.max_history :]
            proprio = proprio[:, -self.max_history :]
            history = self.max_history

        obs_h = self.encode_obs(images, proprio, task_id, wrist_images=wrist_images, instruction_tokens=instruction_tokens)
        pred = self.decoder(obs_h).reshape(batch, self.action_horizon, self.action_dim)
        loss = None
        if actions is not None:
            actions = actions.unsqueeze(1) if actions.ndim == 2 else actions
            horizon = min(actions.shape[1], pred.shape[1])
            loss = F.mse_loss(pred[:, :horizon], actions[:, :horizon])
        return pred, loss

    def encode_obs(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        wrist_images: torch.Tensor | None = None,
        instruction_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        images = _ensure_image_history(images)
        wrist_images = images if wrist_images is None else _ensure_image_history(wrist_images)
        proprio = _ensure_proprio_history(proprio)
        batch, history = images.shape[:2]
        if history > self.max_history:
            images = images[:, -self.max_history :]
            wrist_images = wrist_images[:, -self.max_history :]
            proprio = proprio[:, -self.max_history :]
            history = self.max_history

        agent_h = self.image(_flatten_images(images)).reshape(batch, history, -1)
        wrist_h = self.wrist_image(_flatten_images(wrist_images)).reshape(batch, history, -1)
        proprio_h = self.proprio(proprio)
        task_h = self.task_emb(task_id).unsqueeze(1).expand(-1, history, -1)
        instruction_h = self._instruction_embedding(instruction_tokens, task_id).unsqueeze(1).expand(-1, history, -1)
        fused = self.fuse(torch.cat([agent_h, wrist_h, proprio_h, task_h, instruction_h], dim=-1))
        fused = fused + self.time_emb[:, :history]
        return self.temporal(fused)[:, -1]

    def flow_velocity(self, obs_h: torch.Tensor, action_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch = action_t.shape[0]
        action_flat = action_t.reshape(batch, self.action_horizon * self.action_dim)
        t = t.reshape(batch, 1).to(dtype=obs_h.dtype, device=obs_h.device)
        velocity = self.flow_decoder(obs_h + self.action_in(action_flat) + self.flow_time(t))
        return velocity.reshape(batch, self.action_horizon, self.action_dim)

    def sample_flow(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
        wrist_images: torch.Tensor | None = None,
        instruction_tokens: torch.Tensor | None = None,
        steps: int | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs_h = self.encode_obs(images, proprio, task_id, wrist_images=wrist_images, instruction_tokens=instruction_tokens)
        steps = max(1, int(self.flow_steps if steps is None else steps))
        if initial_noise is None:
            action = torch.zeros((obs_h.shape[0], self.action_horizon, self.action_dim), dtype=obs_h.dtype, device=obs_h.device)
        else:
            action = initial_noise.to(dtype=obs_h.dtype, device=obs_h.device)
        dt = 1.0 / steps
        for idx in range(steps):
            t = torch.full((obs_h.shape[0],), (idx + 0.5) * dt, dtype=obs_h.dtype, device=obs_h.device)
            action = action + dt * self.flow_velocity(obs_h, action, t)
        return action

    def _instruction_embedding(self, instruction_tokens: torch.Tensor | None, task_id: torch.Tensor) -> torch.Tensor:
        if instruction_tokens is None:
            instruction_tokens = torch.zeros(
                (task_id.shape[0], self.instruction_length),
                dtype=torch.long,
                device=task_id.device,
            )
        mask = (instruction_tokens != 0).float().unsqueeze(-1)
        summed = (self.word_emb(instruction_tokens) * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom


def _ensure_image_history(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 4:
        images = images.unsqueeze(1)
    if images.ndim != 5:
        raise ValueError(f"expected images as BTHWC or BHWC, got {tuple(images.shape)}")
    if images.shape[-1] != 3:
        raise ValueError(f"expected RGB images in last dim, got {tuple(images.shape)}")
    if images.dtype != torch.float32:
        images = images.float()
    if images.max() > 1.5:
        images = images / 255.0
    return images


def _ensure_proprio_history(proprio: torch.Tensor) -> torch.Tensor:
    if proprio.ndim == 2:
        proprio = proprio.unsqueeze(1)
    if proprio.ndim != 3:
        raise ValueError(f"expected proprio as BTP or BP, got {tuple(proprio.shape)}")
    return proprio.float()


def _flatten_images(images: torch.Tensor) -> torch.Tensor:
    batch, history = images.shape[:2]
    return images.reshape(batch * history, *images.shape[2:]).permute(0, 3, 1, 2).contiguous()
