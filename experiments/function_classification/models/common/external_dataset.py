"""Prepare external traces using a frozen training-data preprocessing state."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .trace_dataset import (
    fixed_length_tensor,
    load_examples,
    padded_tensor,
    read_event_subset,
    save_npz,
    string_array,
)


def _load_source(
    input_root: Path,
    reference_data: Path,
    event_subset: Path | None,
    input_ids: tuple[str, ...] | None,
    programs: tuple[str, ...] | None,
    include_invalid: bool,
) -> tuple[object, tuple, np.ndarray, np.ndarray, dict, tuple[int, ...] | None]:
    selected = read_event_subset(event_subset) if event_subset else None
    result = load_examples(
        input_root,
        include_invalid=include_invalid,
        member_indices=selected,
        input_ids=input_ids,
        programs=programs,
    )
    examples = result.examples
    with np.load(reference_data, allow_pickle=False) as reference:
        class_names = reference["class_names"].copy()
        reference_members = reference["group_members"].copy()
        reference_metadata = json.loads(str(reference["metadata"]))
    lookup = {name: index for index, name in enumerate(class_names.tolist())}
    unknown = sorted({example.class_name for example in examples} - set(lookup))
    if unknown:
        raise ValueError(f"external data contains classes absent from checkpoint: {unknown}")
    labels = np.asarray([lookup[example.class_name] for example in examples], dtype=np.int64)
    return result, examples, labels, class_names, {
        "reference_members": reference_members,
        "reference_metadata": reference_metadata,
    }, selected


def _validate_layout(actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(
            "external event-group layout does not match training data: "
            f"external={actual.tolist()}, training={expected.tolist()}"
        )


def _metadata(
    input_root: Path,
    reference_data: Path,
    event_subset: Path | None,
    result: object,
    examples: tuple,
    labels: np.ndarray,
    class_names: np.ndarray,
    reference_metadata: dict,
    selected: tuple[int, ...] | None,
    transform: dict,
    input_ids: tuple[str, ...] | None,
    programs: tuple[str, ...] | None,
    include_invalid: bool,
) -> dict:
    counts = Counter(example.class_name for example in examples)
    return {
        "schema_version": 1,
        "purpose": "external inference only; no preprocessing state fitted",
        "input_root": str(input_root.resolve()),
        "reference_data": str(reference_data.resolve()),
        "reference_split_seed": reference_metadata.get("split_seed"),
        "trace_files": result.trace_files,
        "examples": len(examples),
        "class_count": len(class_names),
        "examples_per_class": {name: counts.get(name, 0) for name in class_names.tolist()},
        "skipped_measurements": result.skipped_measurements,
        "skipped_incomplete_bundles": result.skipped_incomplete_bundles,
        "event_subset": str(event_subset.resolve()) if event_subset else None,
        "selected_member_indices": list(selected) if selected else None,
        "selected_input_ids": list(input_ids) if input_ids else None,
        "selected_programs": list(programs) if programs else None,
        "include_invalid": include_invalid,
        "signal": "interval member count / interval retired-instruction count",
        "normalization": "frozen mean and scale from training data",
        "transform": transform,
    }


def prepare_fixed_external(
    input_root: Path,
    reference_data: Path,
    output: Path,
    event_subset: Path | None = None,
    input_ids: tuple[str, ...] | None = None,
    programs: tuple[str, ...] | None = None,
    include_invalid: bool = False,
) -> dict:
    result, examples, labels, class_names, reference_info, selected = _load_source(
        input_root, reference_data, event_subset, input_ids, programs, include_invalid
    )
    with np.load(reference_data, allow_pickle=False) as reference:
        length = int(reference["x"].shape[2])
        mean = reference["normalization_mean"].copy()
        scale = reference["normalization_scale"].copy()
    x, original_lengths, group_members = fixed_length_tensor(examples, length)
    _validate_layout(group_members, reference_info["reference_members"])
    x = ((x - mean) / scale).astype(np.float32)
    for group, members in enumerate(group_members):
        x[:, group, :, members < 0] = 0.0
    metadata = _metadata(
        input_root, reference_data, event_subset, result, examples, labels,
        class_names, reference_info["reference_metadata"], selected,
        {"kind": "fixed-length interpolation", "length": length}, input_ids,
        programs, include_invalid,
    )
    save_npz(
        output,
        x=x,
        y=labels,
        class_names=class_names,
        group_members=group_members,
        original_lengths=original_lengths,
        example_ids=string_array(example.example_id for example in examples),
        source_files=string_array(example.source_file for example in examples),
        metadata=np.asarray(json.dumps(metadata)),
    )
    return metadata


def prepare_padded_external(
    input_root: Path,
    reference_data: Path,
    output: Path,
    event_subset: Path | None = None,
    input_ids: tuple[str, ...] | None = None,
    programs: tuple[str, ...] | None = None,
    include_invalid: bool = False,
) -> dict:
    result, examples, labels, class_names, reference_info, selected = _load_source(
        input_root, reference_data, event_subset, input_ids, programs, include_invalid
    )
    with np.load(reference_data, allow_pickle=False) as reference:
        maximum_length = int(reference["x"].shape[2])
        mean = reference["normalization_mean"].copy()
        scale = reference["normalization_scale"].copy()
    x, mask, original_lengths, group_members = padded_tensor(examples, maximum_length)
    _validate_layout(group_members, reference_info["reference_members"])
    x = ((x - mean) / scale).astype(np.float32)
    x[~mask] = 0.0
    for group, members in enumerate(group_members):
        x[:, group, :, members < 0] = 0.0
    metadata = _metadata(
        input_root, reference_data, event_subset, result, examples, labels,
        class_names, reference_info["reference_metadata"], selected,
        {"kind": "training-compatible padding/downsampling", "maximum_length": maximum_length},
        input_ids, programs, include_invalid,
    )
    save_npz(
        output,
        x=x,
        mask=mask,
        y=labels,
        class_names=class_names,
        group_members=group_members,
        original_lengths=original_lengths,
        example_ids=string_array(example.example_id for example in examples),
        source_files=string_array(example.source_file for example in examples),
        metadata=np.asarray(json.dumps(metadata)),
    )
    return metadata
