#!/usr/bin/env python3
"""Evaluate a frozen Transformer checkpoint on external traces."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

MODELS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS.parent))
from models.common.external_inference import (  # noqa: E402
    predict_torch,
    validate_classes,
    write_external_evaluation,
)
from models.common.torch_training import resolve_device  # noqa: E402
from models.transformer.model import GroupTransformerClassifier  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--device", default="auto")
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
if checkpoint.get("model_family") != "transformer":
    raise ValueError("checkpoint is not a Transformer model")
data = np.load(args.data, allow_pickle=False)
class_names = validate_classes(data["class_names"], checkpoint["class_names"])
model = GroupTransformerClassifier(**checkpoint["model_configuration"])
model.load_state_dict(checkpoint["state_dict"])
device = resolve_device(args.device)
predictions = predict_torch(
    model,
    data["x"].astype(np.float32, copy=False),
    data["mask"].astype(np.bool_, copy=False),
    device,
    args.batch_size,
)
metrics = write_external_evaluation(
    args.output_dir, data, predictions, class_names,
    {"model_family": "transformer", "checkpoint": str(args.checkpoint.resolve()), "device": str(device)},
)
print(f"External accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
