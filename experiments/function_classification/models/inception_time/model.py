"""Compact group-aware InceptionTime-style network."""

from __future__ import annotations

import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(self, input_channels: int, filters: int, kernels: tuple[int, ...]):
        super().__init__()
        branch_channels = filters
        self.branches = nn.ModuleList(
            nn.Conv1d(
                input_channels,
                branch_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=False,
            )
            for kernel in kernels
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_channels, branch_channels, kernel_size=1, bias=False),
        )
        output_channels = branch_channels * (len(kernels) + 1)
        self.norm = nn.BatchNorm1d(output_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [branch(x) for branch in self.branches]
        branches.append(self.pool_branch(x))
        return self.activation(self.norm(torch.cat(branches, dim=1)))


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        filters: int,
        depth: int,
        kernels: tuple[int, ...],
    ):
        super().__init__()
        output_channels = filters * (len(kernels) + 1)
        blocks: list[nn.Module] = []
        current_channels = input_channels
        for _ in range(depth):
            blocks.append(InceptionBlock(current_channels, filters, kernels))
            current_channels = output_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(x)).squeeze(-1)


class GroupInceptionClassifier(nn.Module):
    """Encode each independently measured PMU group before late fusion."""

    def __init__(
        self,
        group_count: int,
        input_channels: int,
        class_count: int,
        filters: int = 16,
        depth: int = 3,
        kernels: tuple[int, ...] = (9, 19, 39),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.group_count = group_count
        self.encoders = nn.ModuleList(
            TemporalEncoder(input_channels, filters, depth, kernels)
            for _ in range(group_count)
        )
        embedding_size = self.encoders[0].output_channels
        self.classifier = nn.Sequential(
            nn.LayerNorm(group_count * embedding_size),
            nn.Dropout(dropout),
            nn.Linear(group_count * embedding_size, class_count),
        )

    def forward(
        self, x: torch.Tensor, _mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Stored tensors are [batch, group, time, channel]. Conv1d consumes
        # [batch, channel, time]. Each group has its own encoder because its
        # channel slots refer to different hardware events.
        embeddings = [
            encoder(x[:, group].transpose(1, 2))
            for group, encoder in enumerate(self.encoders)
        ]
        return self.classifier(torch.cat(embeddings, dim=1))
