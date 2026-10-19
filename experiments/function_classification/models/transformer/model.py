"""Small group-aware transformer encoder for PMU interval sequences."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_positions(length: int, dimension: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, device=device)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class GroupTransformerClassifier(nn.Module):
    def __init__(
        self,
        group_count: int,
        input_channels: int,
        class_count: int,
        model_dimension: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_dimension: int = 128,
        dropout: float = 0.15,
        patch_size: int = 4,
    ):
        super().__init__()
        if model_dimension % heads:
            raise ValueError("model dimension must be divisible by attention heads")
        if patch_size <= 0:
            raise ValueError("patch size must be positive")
        self.group_count = group_count
        self.patch_size = patch_size
        self.input_projections = nn.ModuleList(
            nn.Linear(input_channels * patch_size, model_dimension)
            for _ in range(group_count)
        )
        self.group_embeddings = nn.Parameter(
            torch.empty(group_count, model_dimension)
        )
        nn.init.normal_(self.group_embeddings, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # The temporal encoder is shared, while group-specific projections
        # preserve the identities of the independently measured PMU events.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(model_dimension)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(group_count * model_dimension),
            nn.Dropout(dropout),
            nn.Linear(group_count * model_dimension, class_count),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            raise ValueError("transformer input requires a valid-timestep mask")
        remainder = x.shape[2] % self.patch_size
        if remainder:
            padding = self.patch_size - remainder
            x = torch.nn.functional.pad(x, (0, 0, 0, padding))
            mask = torch.nn.functional.pad(mask, (0, padding), value=False)
        batch, groups, length, channels = x.shape
        patches = x.reshape(
            batch, groups, length // self.patch_size, self.patch_size * channels
        )
        valid = mask.reshape(
            batch, groups, length // self.patch_size, self.patch_size
        ).any(dim=3)
        projected = torch.stack(
            [self.input_projections[group](patches[:, group]) for group in range(groups)],
            dim=1,
        )
        positions = sinusoidal_positions(
            projected.shape[2], self.group_embeddings.shape[1], x.device
        )
        projected = (
            projected
            + positions[None, None, :, :]
            + self.group_embeddings[None, :, None, :]
        )
        tokens = projected.reshape(
            batch * groups, projected.shape[2], projected.shape[3]
        )
        flat_valid = valid.reshape(batch * groups, valid.shape[2])
        encoded = self.encoder(tokens, src_key_padding_mask=~flat_valid)
        weights = flat_valid.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = pooled.reshape(batch, groups * pooled.shape[1])
        return self.classifier(pooled)
