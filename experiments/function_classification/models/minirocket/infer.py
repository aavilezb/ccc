#!/usr/bin/env python3
"""Evaluate a frozen MiniROCKET transform and classifier on external traces."""

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np

MODELS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS.parent))
from models.common.external_inference import (  # noqa: E402
    write_external_evaluation,
)
from models.minirocket.model import MiniRocketTransform  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--transform", type=Path, required=True)
parser.add_argument("--classifier", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--device", default="auto")
args = parser.parse_args()

data = np.load(args.data, allow_pickle=False)
classifier = joblib.load(args.classifier)
class_names = data["class_names"].tolist()
expected_ids = list(range(len(class_names)))
if classifier.classes_.tolist() != expected_ids:
    raise ValueError("classifier class IDs do not match the external class order")
with np.load(args.transform, allow_pickle=False) as state:
    transform = MiniRocketTransform(
        num_features=len(state["biases"]),
        seed=int(state["seed"].item()),
        device=args.device,
        batch_size=args.batch_size,
    )
    transform.channel_indices_ = state["channel_indices"].copy()
    transform.kernel_indices_ = state["kernel_indices"].copy()
    transform.dilations_ = state["dilations"].copy()
    transform.paddings_ = state["paddings"].copy()
    transform.biases_ = state["biases"].copy()
    transform.quantiles_ = state["quantiles"].copy()
    transform.channel_count_ = int(state["channel_count"].item())
    transform.input_length_ = int(state["input_length"].item())
features = transform.transform(
    data["x"].astype(np.float32, copy=False), data["group_members"]
)
predictions = classifier.predict(features)
metrics = write_external_evaluation(
    args.output_dir, data, predictions, data["class_names"].tolist(),
    {
        "model_family": "minirocket",
        "transform": str(args.transform.resolve()),
        "classifier": str(args.classifier.resolve()),
        "device": str(transform.device),
    },
)
print(f"External accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
