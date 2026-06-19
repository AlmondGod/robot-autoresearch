from __future__ import annotations

import torch
from torch import nn


class RoboCasaLatentResidualFlow(nn.Module):
    """Rectified-flow model for action-conditioned VAE latent residuals."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        task_count: int,
        task_dim: int = 32,
        hidden: int = 1024,
        depth: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.task_count = int(task_count)
        self.task = nn.Embedding(task_count, task_dim)
        in_dim = 2 * latent_dim + action_dim + task_dim + 1
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
        ]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                ]
            )
        layers.append(nn.Linear(hidden, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        residual_t: torch.Tensor,
        t: torch.Tensor,
        latent: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        cond = torch.cat([residual_t, latent, action, self.task(task_id), t], dim=-1)
        return self.net(cond)

    @torch.no_grad()
    def sample_residual(
        self,
        *,
        latent: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor,
        steps: int = 8,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.randn_like(latent) if noise is None else noise.to(device=latent.device, dtype=latent.dtype)
        steps = max(1, int(steps))
        dt = 1.0 / steps
        for idx in range(steps):
            t = torch.full((latent.shape[0],), idx * dt, device=latent.device, dtype=latent.dtype)
            x = x + dt * self(x, t, latent, action, task_id)
        return x
