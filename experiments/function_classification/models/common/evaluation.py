"""Reproducibility and classification-report helpers."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import random
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def classification_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
) -> dict:
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(f1_score(true_labels, predictions, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true_labels, predictions, labels=labels).tolist(),
        "per_class": classification_report(
            true_labels,
            predictions,
            labels=labels,
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
    }


def write_evaluation(
    output_directory: Path,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
    example_ids: Sequence[str],
    extra_metrics: dict | None = None,
) -> dict:
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(true_labels, predictions, class_names)
    if extra_metrics:
        metrics.update(extra_metrics)
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (output_directory / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("example_id", "true_id", "true_class", "predicted_id", "predicted_class"))
        for example_id, truth, prediction in zip(example_ids, true_labels, predictions):
            writer.writerow(
                (
                    example_id,
                    int(truth),
                    class_names[int(truth)],
                    int(prediction),
                    class_names[int(prediction)],
                )
            )
    write_confusion_matrix(
        output_directory / "confusion_matrix.png",
        np.asarray(metrics["confusion_matrix"]),
        class_names,
    )
    return metrics


def write_confusion_matrix(
    path: Path, matrix: np.ndarray, class_names: Sequence[str]
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib.pyplot as plt

    size = max(7.0, min(18.0, 0.42 * len(class_names) + 4.0))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axis.tick_params(axis="x", rotation=90, labelsize=7)
    axis.tick_params(axis="y", labelsize=7)
    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    if len(class_names) <= 35:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if matrix[row, column] > threshold else "black",
                )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
