#!/usr/bin/env python3
"""Train and evaluate the masked group-aware PMU transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


MODELS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS.parent))
from models.common.evaluation import set_seed, write_evaluation  # noqa: E402
from models.common.torch_training import resolve_device, train_classifier  # noqa: E402
from models.transformer.model import GroupTransformerClassifier  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feedforward-dimension", type=int, default=128)
    parser.add_argument(
        "--patch-size", type=int, default=4,
        help="adjacent intervals per transformer token (default: 4)",
    )
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    if min(
        arguments.epochs,
        arguments.batch_size,
        arguments.patience,
        arguments.dimension,
        arguments.heads,
        arguments.layers,
        arguments.feedforward_dimension,
        arguments.patch_size,
    ) <= 0:
        parser.error("all size and training-count parameters must be positive")
    if arguments.dimension % arguments.heads:
        parser.error("--dimension must be divisible by --heads")

    set_seed(arguments.seed)
    data = np.load(arguments.data, allow_pickle=False)
    x = data["x"].astype(np.float32, copy=False)
    mask = data["mask"].astype(np.bool_, copy=False)
    y = data["y"].astype(np.int64, copy=False)
    splits = data["split"]
    class_names = data["class_names"].tolist()
    example_ids = data["example_ids"].tolist()
    dataset_metadata = json.loads(str(data["metadata"]))
    if len(class_names) < 2:
        raise ValueError("classification requires at least two classes")
    model = GroupTransformerClassifier(
        group_count=x.shape[1],
        input_channels=x.shape[3],
        class_count=len(class_names),
        model_dimension=arguments.dimension,
        heads=arguments.heads,
        layers=arguments.layers,
        feedforward_dimension=arguments.feedforward_dimension,
        dropout=arguments.dropout,
        patch_size=arguments.patch_size,
    )
    device = resolve_device(arguments.device)
    print(f"Training on {device}; examples={len(x)}, classes={len(class_names)}, shape={x.shape}")
    result = train_classifier(
        model,
        x,
        y,
        splits,
        mask=mask,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        patience=arguments.patience,
        device=device,
        seed=arguments.seed,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_family": "transformer",
        "state_dict": model.cpu().state_dict(),
        "class_names": class_names,
        "model_configuration": {
            "group_count": x.shape[1],
            "input_channels": x.shape[3],
            "class_count": len(class_names),
            "model_dimension": arguments.dimension,
            "heads": arguments.heads,
            "layers": arguments.layers,
            "feedforward_dimension": arguments.feedforward_dimension,
            "dropout": arguments.dropout,
            "patch_size": arguments.patch_size,
        },
    }
    torch.save(checkpoint, arguments.output_dir / "model.pt")
    (arguments.output_dir / "history.json").write_text(
        json.dumps(result.history, indent=2) + "\n", encoding="utf-8"
    )
    (arguments.output_dir / "run.json").write_text(
        json.dumps(
            {
                "model_family": "transformer",
                "dataset": str(arguments.data.resolve()),
                "dataset_metadata": dataset_metadata,
                "training_arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(arguments).items()
                },
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "best_epoch": result.best_epoch,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = write_evaluation(
        arguments.output_dir,
        result.true_labels,
        result.predictions,
        class_names,
        [example_ids[index] for index in result.test_indices],
        {"best_epoch": result.best_epoch, "device": str(device)},
    )
    print(f"Test accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print(f"Wrote model and evaluation to {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
