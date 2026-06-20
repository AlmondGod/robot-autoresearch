from __future__ import annotations

import torch
from torch import nn


class MicrowaveSingleTaskChunkPolicy(nn.Module):
    """Single-task visual BC policy for PickPlaceCounterToMicrowave."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        chunk_horizon: int,
        action_dim: int,
        width: int = 256,
        dropout: float = 0.03,
        variant_count: int = 1,
        variant_dim: int = 32,
    ) -> None:
        super().__init__()
        self.chunk_horizon = int(chunk_horizon)
        self.action_dim = int(action_dim)
        self.variant_count = int(max(1, variant_count))
        self.vision = nn.Sequential(
            nn.Conv2d(6, 48, 5, stride=2, padding=2),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 160, 3, stride=2, padding=1),
            nn.GroupNorm(10, 160),
            nn.SiLU(),
            nn.Conv2d(160, width, 3, stride=2, padding=1),
            nn.GroupNorm(max(1, min(16, width // 16)), width),
            nn.SiLU(),
            nn.Flatten(),
        )
        vision_dim = width * 4 * 4
        prop_width = max(128, width // 2)
        self.proprio = nn.Sequential(
            nn.Linear(proprio_dim, prop_width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(prop_width, prop_width),
            nn.LayerNorm(prop_width),
        )
        self.variant = nn.Embedding(self.variant_count, variant_dim)
        self.head = nn.Sequential(
            nn.Linear(vision_dim + prop_width + variant_dim, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, 2 * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, self.chunk_horizon * self.action_dim),
        )

    def forward(
        self,
        agent: torch.Tensor,
        wrist: torch.Tensor,
        proprio: torch.Tensor,
        variant_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if agent.max() > 1.5:
            agent = agent / 255.0
        if wrist.max() > 1.5:
            wrist = wrist / 255.0
        if variant_id is None:
            variant_id = torch.zeros((agent.shape[0],), dtype=torch.long, device=agent.device)
        variant_id = variant_id.clamp(0, self.variant_count - 1)
        image = self.vision(torch.cat([agent, wrist], dim=1))
        prop = self.proprio(proprio)
        variant = self.variant(variant_id)
        out = self.head(torch.cat([image, prop, variant], dim=-1))
        return out.reshape(agent.shape[0], self.chunk_horizon, self.action_dim)
