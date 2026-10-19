#!/usr/bin/env python3
"""Fit MiniROCKET features and a validation-selected logistic classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODELS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS.parent))
from models.common.evaluation import set_seed, write_evaluation  # noqa: E402
from models.common.trace_dataset import SPLIT_IDS  # noqa: E402
from models.minirocket.model import MiniRocketTransform  # noqa: E402


def parse_regularization(text: str) -> list[float]:
    try:
        values = [float(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be comma-separated numbers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all regularization values must be positive")
    return values


def classifier(c_value: float, seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value, max_iter=3000, random_state=seed, class_weight="balanced"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=int, default=10_000)
    parser.add_argument("--regularization", type=parse_regularization, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    if arguments.features <= 0 or arguments.batch_size <= 0:
        parser.error("--features and --batch-size must be positive")

    set_seed(arguments.seed)
    data = np.load(arguments.data, allow_pickle=False)
    x = data["x"].astype(np.float32, copy=False)
    y = data["y"].astype(np.int64, copy=False)
    splits = data["split"]
    group_members = data["group_members"]
    class_names = data["class_names"].tolist()
    example_ids = data["example_ids"].tolist()
    dataset_metadata = json.loads(str(data["metadata"]))
    if len(class_names) < 2:
        raise ValueError("classification requires at least two classes")
    train_indices = np.flatnonzero(splits == SPLIT_IDS["train"])
    validation_indices = np.flatnonzero(splits == SPLIT_IDS["validation"])
    test_indices = np.flatnonzero(splits == SPLIT_IDS["test"])
    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise ValueError("train, validation, and test splits must all be nonempty")

    transform = MiniRocketTransform(
        num_features=arguments.features,
        seed=arguments.seed,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    print(f"Fitting {arguments.features} MiniROCKET features on {len(train_indices)} examples")
    transform.fit(x[train_indices], group_members)
    features = transform.transform(x, group_members)
    trials: list[dict] = []
    best_c = arguments.regularization[0]
    best_score = -1.0
    for c_value in arguments.regularization:
        candidate = classifier(c_value, arguments.seed)
        candidate.fit(features[train_indices], y[train_indices])
        predictions = candidate.predict(features[validation_indices])
        score = f1_score(y[validation_indices], predictions, average="macro", zero_division=0)
        trials.append({"C": c_value, "validation_macro_f1": float(score)})
        print(f"C={c_value:g}: validation macro-F1={score:.4f}")
        if score > best_score:
            best_score = float(score)
            best_c = c_value

    final_model = classifier(best_c, arguments.seed)
    fit_indices = np.concatenate((train_indices, validation_indices))
    final_model.fit(features[fit_indices], y[fit_indices])
    predictions = final_model.predict(features[test_indices])
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output_dir / "transform.npz", **transform.state())
    joblib.dump(final_model, arguments.output_dir / "classifier.joblib")
    (arguments.output_dir / "selection.json").write_text(
        json.dumps({"best_C": best_c, "trials": trials}, indent=2) + "\n",
        encoding="utf-8",
    )
    (arguments.output_dir / "run.json").write_text(
        json.dumps(
            {
                "model_family": "minirocket",
                "dataset": str(arguments.data.resolve()),
                "dataset_metadata": dataset_metadata,
                "training_arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(arguments).items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = write_evaluation(
        arguments.output_dir,
        y[test_indices],
        predictions,
        class_names,
        [example_ids[index] for index in test_indices],
        {
            "features": arguments.features,
            "best_C": best_c,
            "validation_macro_f1": best_score,
            "device": arguments.device,
        },
    )
    print(f"Test accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print(f"Wrote transform, classifier, and evaluation to {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
