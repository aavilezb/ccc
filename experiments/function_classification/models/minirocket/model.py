"""A compact, self-contained channel-wise MiniROCKET-style transform."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import torch
import torch.nn.functional as functional


KERNEL_LENGTH = 9


def minirocket_kernels() -> np.ndarray:
    """Return the canonical 84 length-nine kernels with three positive weights."""
    kernels = []
    for positive_positions in combinations(range(KERNEL_LENGTH), 3):
        kernel = np.full(KERNEL_LENGTH, -1.0, dtype=np.float32)
        kernel[list(positive_positions)] = 2.0
        kernels.append(kernel)
    return np.asarray(kernels)


def dilation_values(length: int) -> np.ndarray:
    maximum = max(1, (length - 1) // (KERNEL_LENGTH - 1))
    powers = [1]
    while powers[-1] * 2 <= maximum:
        powers.append(powers[-1] * 2)
    return np.asarray(powers, dtype=np.int32)


def active_channels(x: np.ndarray, group_members: np.ndarray) -> np.ndarray:
    channels = [
        x[:, group, :, channel]
        for group in range(x.shape[1])
        for channel in range(x.shape[3])
        if group_members[group, channel] >= 0
    ]
    if not channels:
        raise ValueError("dataset does not contain any active member channels")
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


class MiniRocketTransform:
    """Channel-independent MiniROCKET kernels with fitted quantile biases.

    PMU groups are independent executions, so every event channel is convolved
    separately. Feature selection spans event identity, kernel, dilation, and
    padding. The only resulting feature is the proportion of positive values.
    """

    def __init__(
        self,
        num_features: int = 10_000,
        seed: int = 2026,
        device: str = "cpu",
        batch_size: int = 16,
        feature_chunk: int = 1024,
    ):
        if num_features <= 0 or batch_size <= 0 or feature_chunk <= 0:
            raise ValueError("feature and batch sizes must be positive")
        self.num_features = num_features
        self.seed = seed
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.feature_chunk = feature_chunk

    def fit(self, x: np.ndarray, group_members: np.ndarray) -> "MiniRocketTransform":
        values = active_channels(x, group_members)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        rng = np.random.default_rng(self.seed)
        kernels = minirocket_kernels()
        dilations = dilation_values(values.shape[2])
        configs_per_dilation = values.shape[1] * len(kernels)
        total_configs = len(dilations) * configs_per_dilation
        selected = rng.choice(
            total_configs,
            size=self.num_features,
            replace=self.num_features > total_configs,
        )
        dilation_slots = selected // configs_per_dilation
        flat_configs = selected % configs_per_dilation
        kernel_indices = flat_configs % len(kernels)
        self.channel_indices_ = (flat_configs // len(kernels)).astype(np.int32)
        self.kernel_indices_ = kernel_indices.astype(np.int16)
        self.dilations_ = dilations[dilation_slots].astype(np.int32)
        # Alternating same/valid padding prevents every feature from sharing
        # identical edge behavior while remaining deterministic.
        self.paddings_ = ((np.arange(self.num_features) + kernel_indices) % 2 == 0)
        golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
        self.quantiles_ = np.mod(
            (np.arange(self.num_features, dtype=np.float64) + 1) * golden_ratio,
            1.0,
        ).astype(np.float32)
        chosen_examples = rng.integers(0, len(values), size=self.num_features)
        self.biases_ = np.empty(self.num_features, dtype=np.float32)
        self.channel_count_ = values.shape[1]
        self.input_length_ = values.shape[2]
        self._fit_biases(values, kernels, chosen_examples)
        return self

    def _weights(self, kernels: np.ndarray) -> torch.Tensor:
        repeated = np.tile(kernels, (self.channel_count_, 1))
        return torch.from_numpy(repeated[:, None, :]).to(self.device)

    def _convolve(
        self,
        values: np.ndarray,
        weights: torch.Tensor,
        dilation: int,
    ) -> np.ndarray:
        tensor = torch.from_numpy(values).to(self.device)
        padding = dilation * (KERNEL_LENGTH // 2)
        with torch.no_grad():
            response = functional.conv1d(
                tensor,
                weights,
                dilation=dilation,
                padding=padding,
                groups=self.channel_count_,
            )
        return response.cpu().numpy()

    def _fit_biases(
        self,
        values: np.ndarray,
        kernels: np.ndarray,
        chosen_examples: np.ndarray,
    ) -> None:
        weights = self._weights(kernels)
        for dilation in np.unique(self.dilations_):
            dilation_features = np.flatnonzero(self.dilations_ == dilation)
            selected_examples = np.unique(chosen_examples[dilation_features])
            for offset in range(0, len(selected_examples), self.batch_size):
                example_batch = selected_examples[offset:offset + self.batch_size]
                response = self._convolve(
                    values[example_batch], weights, int(dilation)
                )
                local_example = {
                    int(example): index for index, example in enumerate(example_batch)
                }
                for padded in (False, True):
                    feature_indices = dilation_features[
                        self.paddings_[dilation_features] == padded
                    ]
                    relevant = feature_indices[
                        np.isin(chosen_examples[feature_indices], example_batch)
                    ]
                    if not len(relevant):
                        continue
                    rows = np.asarray(
                        [local_example[int(chosen_examples[index])] for index in relevant]
                    )
                    configs = (
                        self.channel_indices_[relevant] * len(kernels)
                        + self.kernel_indices_[relevant]
                    )
                    selected_response = response[rows, configs]
                    if not padded:
                        trim = int(dilation) * (KERNEL_LENGTH // 2)
                        selected_response = selected_response[:, trim:-trim]
                    ordered = np.sort(selected_response, axis=1)
                    positions = np.minimum(
                        (self.quantiles_[relevant] * (ordered.shape[1] - 1)).astype(int),
                        ordered.shape[1] - 1,
                    )
                    self.biases_[relevant] = ordered[np.arange(len(relevant)), positions]

    def transform(self, x: np.ndarray, group_members: np.ndarray) -> np.ndarray:
        for attribute in (
            "channel_indices_", "kernel_indices_", "dilations_", "paddings_", "biases_"
        ):
            if not hasattr(self, attribute):
                raise ValueError("MiniRocketTransform must be fitted before transform")
        values = active_channels(x, group_members)
        if values.shape[1:] != (self.channel_count_, self.input_length_):
            raise ValueError("input channel count or sequence length changed")
        kernels = minirocket_kernels()
        weights = self._weights(kernels)
        features = np.empty((len(values), self.num_features), dtype=np.float32)
        for dilation in np.unique(self.dilations_):
            for example_offset in range(0, len(values), self.batch_size):
                example_batch = values[example_offset:example_offset + self.batch_size]
                response = self._convolve(
                    example_batch, weights, int(dilation)
                )
                for padded in (False, True):
                    feature_indices = np.flatnonzero(
                        (self.dilations_ == dilation) & (self.paddings_ == padded)
                    )
                    selected_base = response
                    if not padded:
                        trim = int(dilation) * (KERNEL_LENGTH // 2)
                        selected_base = response[:, :, trim:-trim]
                    for feature_offset in range(0, len(feature_indices), self.feature_chunk):
                        selected_features = feature_indices[
                            feature_offset:feature_offset + self.feature_chunk
                        ]
                        configs = (
                            self.channel_indices_[selected_features] * len(kernels)
                            + self.kernel_indices_[selected_features]
                        )
                        selected_response = selected_base[:, configs]
                        ppv = (
                            selected_response
                            > self.biases_[selected_features][None, :, None]
                        ).mean(axis=2)
                        features[
                            example_offset:example_offset + len(example_batch),
                            selected_features,
                        ] = ppv
        return features

    def state(self) -> dict[str, np.ndarray]:
        return {
            "channel_indices": self.channel_indices_,
            "kernel_indices": self.kernel_indices_,
            "dilations": self.dilations_,
            "paddings": self.paddings_,
            "biases": self.biases_,
            "quantiles": self.quantiles_,
            "channel_count": np.asarray(self.channel_count_),
            "input_length": np.asarray(self.input_length_),
            "seed": np.asarray(self.seed),
        }
