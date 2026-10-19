"""Helpers for evaluating frozen classifiers on external datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import write_evaluation


def validate_classes(external: np.ndarray, checkpoint: list[str]) -> list[str]:
    names = external.tolist()
    if names != checkpoint:
        raise ValueError("external class order does not match the frozen checkpoint")
    return names


def predict_torch(
    model: torch.nn.Module,
    x: np.ndarray,
    mask: np.ndarray | None,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    tensors = [torch.from_numpy(x)]
    if mask is not None:
        tensors.append(torch.from_numpy(mask))
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)
    predicted = []
    model.to(device).eval()
    with torch.no_grad():
        for batch in loader:
            features = batch[0].to(device)
            batch_mask = batch[1].to(device) if len(batch) == 2 else None
            predicted.append(model(features, batch_mask).argmax(dim=1).cpu().numpy())
    return np.concatenate(predicted)


def write_external_evaluation(
    output_directory: Path,
    data: np.lib.npyio.NpzFile,
    predictions: np.ndarray,
    class_names: list[str],
    details: dict,
) -> dict:
    metadata = json.loads(str(data["metadata"]))
    metrics = write_evaluation(
        output_directory,
        data["y"],
        predictions,
        class_names,
        data["example_ids"].tolist(),
        {"external_dataset": metadata, **details},
    )
    return metrics
