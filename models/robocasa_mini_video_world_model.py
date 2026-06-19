from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RoboCasaMiniVideoWorldModel(nn.Module):
    """Small action-conditioned video world model for RoboCasa evaluator baselines.

    This is intentionally much smaller than Cosmos/OSCAR-style systems, but keeps
    the same shape of the problem: compress RGB observations into a visual latent,
    roll that latent forward under actions and task conditioning, decode future
    RGB/proprio, and expose progress/success heads for fast policy ranking.
    """

    def __init__(
        self,
        *,
        proprio_dim: int,
        action_dim: int,
        task_count: int,
        latent_dim: int = 512,
        width: int = 512,
        task_dim: int = 64,
        dropout: float = 0.05,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        residual_scale: float = 0.05,
        dynamics_kind: str = "mlp",
    ) -> None:
        super().__init__()
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.task_count = int(task_count)
        self.latent_dim = int(latent_dim)
        self.width = int(width)
        self.residual_scale = float(residual_scale)
        self.dynamics_kind = str(dynamics_kind)

        self.image = nn.Sequential(
            nn.Conv2d(6, 48, 4, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 192, 4, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            nn.Conv2d(192, 256, 4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, width),
            nn.SiLU(),
        )
        self.task = nn.Embedding(task_count, task_dim)
        self.encoder = nn.Sequential(
            nn.Linear(width + proprio_dim + task_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        if self.dynamics_kind == "transformer":
            self.latent_to_tokens = nn.Linear(latent_dim, 4 * width)
            self.action_token = nn.Linear(action_dim, width)
            self.task_token = nn.Linear(task_dim, width)
            self.dyn_pos = nn.Parameter(torch.zeros(1, 6, width))
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=transformer_heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.dynamics = nn.TransformerEncoder(layer, num_layers=transformer_layers)
            self.delta_head = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, latent_dim + proprio_dim),
            )
        elif self.dynamics_kind == "mlp":
            self.dynamics = nn.Sequential(
                nn.Linear(latent_dim + action_dim + task_dim, width),
                nn.LayerNorm(width),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, latent_dim + proprio_dim),
            )
        else:
            raise ValueError(f"unknown dynamics_kind={dynamics_kind!r}")

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + task_dim, width),
            nn.SiLU(),
            nn.Linear(width, 256 * 4 * 4),
            nn.SiLU(),
        )
        self.decoder_up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 6, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.progress_head = nn.Sequential(
            nn.Linear(latent_dim + task_dim, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )
        self.success_head = nn.Sequential(
            nn.Linear(latent_dim + task_dim, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )

    def encode(
        self,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        image_feat = self.image(torch.cat([agent, wrist], dim=1))
        h = torch.cat([image_feat, proprio, self.task(task_id)], dim=-1)
        return self.encoder(h)

    def decode(self, latent: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        h = torch.cat([latent, self.task(task_id)], dim=-1)
        h = self.decoder_fc(h).reshape(latent.shape[0], 256, 4, 4)
        return self.decoder_up(h)

    def step(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dynamics_kind == "transformer":
            latent_tokens = self.latent_to_tokens(latent).reshape(latent.shape[0], 4, self.width)
            action_token = self.action_token(action).unsqueeze(1)
            task_token = self.task_token(self.task(task_id)).unsqueeze(1)
            tokens = torch.cat([latent_tokens, action_token, task_token], dim=1) + self.dyn_pos
            h = self.dynamics(tokens).mean(dim=1)
            out = self.delta_head(h)
        else:
            out = self.dynamics(torch.cat([latent, action, self.task(task_id)], dim=-1))
        delta_z, next_proprio = out[..., : self.latent_dim], out[..., self.latent_dim :]
        next_latent = F.layer_norm(latent + self.residual_scale * torch.tanh(delta_z), (self.latent_dim,))
        return next_latent, next_proprio

    def heads(self, latent: torch.Tensor, task_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([latent, self.task(task_id)], dim=-1)
        return self.progress_head(h).squeeze(-1), self.success_head(h).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z = self.encode(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
        target_z = self.encode(batch["next_agent"], batch["next_wrist"], batch["next_proprio"], batch["task_id"])
        z_next, proprio_next = self.step(z, batch["action"], batch["task_id"])
        progress, success_logit = self.heads(z, batch["task_id"])
        next_progress, next_success_logit = self.heads(z_next, batch["task_id"])
        return {
            "z": z,
            "z_target": target_z.detach(),
            "z_next": z_next,
            "recon": self.decode(z, batch["task_id"]),
            "next_recon": self.decode(z_next, batch["task_id"]),
            "proprio_next": proprio_next,
            "progress": progress,
            "success_logit": success_logit,
            "next_progress": next_progress,
            "next_success_logit": next_success_logit,
        }


def mini_video_world_model_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    recon_weight: float = 1.0,
    next_recon_weight: float = 1.0,
    latent_weight: float = 1.0,
    proprio_weight: float = 1.0,
    progress_weight: float = 0.5,
    success_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    current_rgb = _rgb_target(batch["agent"], batch["wrist"])
    next_rgb = _rgb_target(batch["next_agent"], batch["next_wrist"])
    recon_loss = F.mse_loss(outputs["recon"], current_rgb) + 0.1 * F.l1_loss(outputs["recon"], current_rgb)
    next_recon_loss = F.mse_loss(outputs["next_recon"], next_rgb) + 0.1 * F.l1_loss(outputs["next_recon"], next_rgb)
    latent_loss = F.mse_loss(outputs["z_next"], outputs["z_target"])
    proprio_loss = F.smooth_l1_loss(
        outputs["proprio_next"].contiguous(),
        batch["next_proprio"].contiguous(),
        beta=1.0,
    )
    progress_loss = F.mse_loss(torch.sigmoid(outputs["progress"]), batch["progress"])
    next_progress_loss = F.mse_loss(torch.sigmoid(outputs["next_progress"]), batch["next_progress"])
    success_loss = F.binary_cross_entropy_with_logits(outputs["success_logit"], batch["success"])
    next_success_loss = F.binary_cross_entropy_with_logits(outputs["next_success_logit"], batch["next_success"])
    loss = (
        recon_weight * recon_loss
        + next_recon_weight * next_recon_loss
        + latent_weight * latent_loss
        + proprio_weight * proprio_loss
        + progress_weight * 0.5 * (progress_loss + next_progress_loss)
        + success_weight * 0.5 * (success_loss + next_success_loss)
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "next_recon_loss": float(next_recon_loss.detach().cpu()),
        "latent_loss": float(latent_loss.detach().cpu()),
        "proprio_loss": float(proprio_loss.detach().cpu()),
        "progress_loss": float(progress_loss.detach().cpu()),
        "success_loss": float(success_loss.detach().cpu()),
    }


def _rgb_target(agent: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
    if agent.max() > 1.5:
        agent = agent / 255.0
    if wrist.max() > 1.5:
        wrist = wrist / 255.0
    return torch.cat([agent, wrist], dim=1).clamp(0.0, 1.0)
