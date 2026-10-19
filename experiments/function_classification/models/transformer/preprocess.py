#!/usr/bin/env python3
"""Prepare padded and masked PMU interval sequences for the transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


MODELS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS.parent))
from models.common.trace_dataset import (  # noqa: E402
    add_common_preprocess_arguments,
    assign_splits,
    common_metadata,
    encode_labels,
    load_examples,
    padded_tensor,
    read_event_subset,
    save_npz,
    standardize_padded,
    string_array,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_preprocess_arguments(parser)
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="longer groups are downsampled; shorter groups retain a padding mask",
    )
    arguments = parser.parse_args()
    if arguments.max_length <= 0:
        parser.error("--max-length must be positive")

    selected_members = (
        read_event_subset(arguments.event_subset) if arguments.event_subset else None
    )
    result = load_examples(
        arguments.input_root,
        arguments.include_invalid,
        selected_members,
        arguments.input_ids.split(",") if arguments.input_ids else None,
        arguments.programs.split(",") if arguments.programs else None,
    )
    examples = result.examples
    splits = assign_splits(
        examples,
        arguments.validation_fraction,
        arguments.test_fraction,
        arguments.seed,
        arguments.split_unit,
        arguments.validation_units,
        arguments.test_units,
    )
    labels, class_names = encode_labels(examples)
    x, mask, original_lengths, group_members = padded_tensor(
        examples, arguments.max_length
    )
    x, mean, scale = standardize_padded(x, mask, splits, group_members)
    metadata = common_metadata(result, examples, class_names, splits, arguments)
    metadata.update(
        {
            "model_family": "transformer",
            "preprocessing": {
                "maximum_length": arguments.max_length,
                "short_sequence_transform": "right padding with explicit attention mask",
                "long_sequence_transform": "linear downsampling over normalized execution progress",
                "normalization": "per-event training-split masked z-score",
            },
        }
    )
    save_npz(
        arguments.output,
        x=x,
        mask=mask,
        y=labels,
        split=splits,
        class_names=class_names,
        group_members=group_members,
        original_lengths=original_lengths,
        normalization_mean=mean,
        normalization_scale=scale,
        example_ids=string_array(example.example_id for example in examples),
        source_files=string_array(example.source_file for example in examples),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(
        f"Wrote {len(examples)} examples, {len(class_names)} classes, "
        f"shape={x.shape} to {arguments.output}"
    )
    print(f"Splits: {metadata['split_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
