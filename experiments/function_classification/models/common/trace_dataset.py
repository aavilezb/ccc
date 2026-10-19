"""Load grouped libpmc traces as function/input classification examples."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np


SPLIT_NAMES = ("train", "validation", "test")
SPLIT_IDS = {name: index for index, name in enumerate(SPLIT_NAMES)}
EVENT_COLUMNS = [
    "index", "event_name", "event_number", "umask", "mode",
    "sample_period", "type", "role",
]


@dataclass(frozen=True)
class RawExample:
    class_name: str
    implementation: str
    program: str
    input_id: str
    source_file: str
    file_index: int
    run: int
    group_members: tuple[tuple[int, ...], ...]
    groups: tuple[np.ndarray, ...]

    @property
    def example_id(self) -> str:
        return f"{self.source_file}::run={self.run}"


@dataclass(frozen=True)
class LoadResult:
    examples: tuple[RawExample, ...]
    trace_files: int
    skipped_measurements: int
    skipped_incomplete_bundles: int


def class_identifier(configuration: dict) -> tuple[str, str, str, str]:
    implementation = str(configuration.get("class_name", "unknown"))
    executable = Path(str(configuration.get("target_executable", "unknown")))
    program = executable.stem or "unknown"
    input_id = str(configuration.get("input_id", "no_args"))
    return (
        f"{implementation}::{program}::{input_id}",
        implementation,
        program,
        input_id,
    )


def measurement_is_valid(measurement: dict) -> bool:
    if measurement.get("run_mode") != "sampling":
        return False
    if not measurement.get("samples"):
        return False
    if int(measurement.get("lost_samples", 0)) != 0:
        return False
    enabled = float(measurement.get("time_enabled", 0))
    running = float(measurement.get("time_running", 0))
    if enabled <= 0 or running / enabled < 0.99:
        return False
    validation = measurement.get("adaptive_validation", {})
    coverage = float(validation.get("counter_coverage", 0))
    retention = float(validation.get("record_retention", 0))
    return 0.95 <= coverage <= 1.05 and retention >= 0.95


def interval_ratios(
    measurement: dict,
    leader_index: int,
    members: Sequence[int] | None = None,
) -> np.ndarray:
    """Convert cumulative PERF_SAMPLE_READ values to event/instruction intervals."""
    if members is None:
        members = tuple(
            int(value) for value in measurement["collection"]["member_indices"]
        )
    else:
        members = tuple(members)
    indices = (leader_index, *members)
    previous = {index: 0 for index in indices}
    rows: list[list[float]] = []
    for sample in measurement["samples"]:
        counts = sample["counts"]
        current = {index: int(counts[str(index)]) for index in indices}
        deltas = {index: current[index] - previous[index] for index in indices}
        if any(value < 0 for value in deltas.values()):
            raise ValueError("cumulative counter decreased inside a measurement")
        leader_delta = deltas[leader_index]
        if leader_delta > 0:
            rows.append([deltas[index] / leader_delta for index in members])
        previous = current
    if not rows:
        raise ValueError("measurement has no positive leader intervals")
    return np.asarray(rows, dtype=np.float32)


def discover_trace_files(input_root: Path) -> list[Path]:
    if input_root.is_file():
        return [input_root.resolve()]
    if not input_root.is_dir():
        raise ValueError(f"input root does not exist: {input_root}")
    return sorted(path.resolve() for path in input_root.rglob("traces/*.json"))


def load_examples(
    input_root: Path,
    include_invalid: bool = False,
    member_indices: Sequence[int] | None = None,
    input_ids: Sequence[str] | None = None,
    programs: Sequence[str] | None = None,
) -> LoadResult:
    paths = discover_trace_files(input_root)
    if not paths:
        raise ValueError(f"no trace JSON files found under {input_root}")

    examples: list[RawExample] = []
    skipped_measurements = 0
    skipped_bundles = 0
    requested_members = None if member_indices is None else tuple(member_indices)
    if requested_members is not None:
        if not requested_members or len(requested_members) != len(set(requested_members)):
            raise ValueError("selected member indices must be nonempty and unique")
    requested_set = None if requested_members is None else set(requested_members)
    requested_inputs = None if input_ids is None else {str(value) for value in input_ids}
    if requested_inputs is not None and not requested_inputs:
        raise ValueError("selected input IDs must be nonempty")
    requested_programs = None if programs is None else {str(value) for value in programs}
    if requested_programs is not None and not requested_programs:
        raise ValueError("selected programs must be nonempty")
    found_inputs: set[str] = set()
    found_programs: set[str] = set()
    selected_trace_files = 0
    found_members: set[int] = set()
    canonical_members: tuple[tuple[int, ...], ...] | None = None
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read trace {path}: {error}") from error
        if document.get("schema_version") != 2:
            raise ValueError(f"unsupported trace schema in {path}")
        configuration = document.get("configuration", {})
        class_name, implementation, program, input_id = class_identifier(configuration)
        if requested_inputs is not None and input_id not in requested_inputs:
            continue
        if requested_programs is not None and program not in requested_programs:
            continue
        found_inputs.add(input_id)
        found_programs.add(program)
        selected_trace_files += 1
        leader_index = int(configuration.get("leader", {}).get("index", 0))
        by_run: dict[int, dict[int, tuple[tuple[int, ...], np.ndarray]]] = {}
        batch_members: dict[int, tuple[int, ...]] = {}
        file_index = int(document.get("collection", {}).get("file_index", 0))

        for measurement in document.get("measurements", []):
            collection = measurement.get("collection", {})
            try:
                batch = int(collection["batch"])
                run = int(collection["run"])
                collected_members = tuple(
                    int(value) for value in collection["member_indices"]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid collection metadata in {path}: {error}") from error
            members = (
                collected_members
                if requested_set is None
                else tuple(value for value in collected_members if value in requested_set)
            )
            if not members:
                continue
            found_members.update(members)
            previous_members = batch_members.setdefault(batch, members)
            if previous_members != members:
                raise ValueError(f"batch {batch} changes member indices in {path}")
            if not include_invalid and not measurement_is_valid(measurement):
                skipped_measurements += 1
                continue
            try:
                sequence = interval_ratios(measurement, leader_index, members)
            except (KeyError, TypeError, ValueError) as error:
                if include_invalid:
                    raise ValueError(f"invalid samples in {path}: {error}") from error
                skipped_measurements += 1
                continue
            if batch in by_run.setdefault(run, {}):
                raise ValueError(f"duplicate batch {batch}, run {run} in {path}")
            by_run[run][batch] = (members, sequence)

        ordered_batches = sorted(batch_members)
        if not ordered_batches:
            continue
        expected_members = tuple(batch_members[batch] for batch in ordered_batches)
        if canonical_members is None:
            canonical_members = expected_members
        elif canonical_members != expected_members:
            raise ValueError(
                f"event-group layout differs in {path}; preprocess one event configuration at a time"
            )
        for run, groups_by_batch in sorted(by_run.items()):
            if any(batch not in groups_by_batch for batch in ordered_batches):
                skipped_bundles += 1
                continue
            groups = tuple(groups_by_batch[batch][1] for batch in ordered_batches)
            examples.append(
                RawExample(
                    class_name=class_name,
                    implementation=implementation,
                    program=program,
                    input_id=input_id,
                    source_file=str(path),
                    file_index=file_index,
                    run=run,
                    group_members=expected_members,
                    groups=groups,
                )
            )

    if not examples:
        raise ValueError("no complete valid examples were found")
    if requested_set is not None and found_members != requested_set:
        missing = sorted(requested_set - found_members)
        raise ValueError(f"selected member indices are absent from the traces: {missing}")
    if requested_inputs is not None and found_inputs != requested_inputs:
        missing = sorted(requested_inputs - found_inputs)
        raise ValueError(f"selected input IDs are absent from the traces: {missing}")
    if requested_programs is not None and found_programs != requested_programs:
        missing = sorted(requested_programs - found_programs)
        raise ValueError(f"selected programs are absent from the traces: {missing}")
    return LoadResult(
        examples=tuple(examples),
        trace_files=selected_trace_files,
        skipped_measurements=skipped_measurements,
        skipped_incomplete_bundles=skipped_bundles,
    )


def _stable_seed(seed: int, class_name: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{class_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def assign_splits(
    examples: Sequence[RawExample],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    split_unit: str = "trace",
    validation_units: int | None = None,
    test_units: int | None = None,
) -> np.ndarray:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must both be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    if split_unit not in {"trace", "run"}:
        raise ValueError("split unit must be 'trace' or 'run'")
    if (validation_units is None) != (test_units is None):
        raise ValueError("validation and test units must be specified together")
    if validation_units is not None and (validation_units <= 0 or test_units <= 0):
        raise ValueError("validation and test units must be positive")

    assignments = np.full(len(examples), -1, dtype=np.int8)
    by_class: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        by_class.setdefault(example.class_name, []).append(index)

    for class_name, indices in by_class.items():
        unit_to_indices: dict[str, list[int]] = {}
        for index in indices:
            example = examples[index]
            unit = example.source_file
            if split_unit == "run":
                unit = example.example_id
            unit_to_indices.setdefault(unit, []).append(index)
        units = sorted(unit_to_indices)
        if len(units) < 3:
            raise ValueError(
                f"class {class_name!r} has only {len(units)} {split_unit} units; "
                "at least 3 are needed for train/validation/test"
            )
        random.Random(_stable_seed(seed, class_name)).shuffle(units)
        test_count = (
            test_units
            if test_units is not None
            else max(1, int(round(len(units) * test_fraction)))
        )
        validation_count = (
            validation_units
            if validation_units is not None
            else max(1, int(round(len(units) * validation_fraction)))
        )
        if test_count + validation_count >= len(units):
            raise ValueError(
                f"class {class_name!r} has {len(units)} {split_unit} units, "
                f"not enough for {validation_count} validation, {test_count} test, and training"
            )
        split_units = {
            SPLIT_IDS["test"]: units[:test_count],
            SPLIT_IDS["validation"]: units[test_count:test_count + validation_count],
            SPLIT_IDS["train"]: units[test_count + validation_count:],
        }
        for split_id, selected_units in split_units.items():
            for unit in selected_units:
                assignments[unit_to_indices[unit]] = split_id
    if np.any(assignments < 0):
        raise AssertionError("internal error: unassigned examples")
    return assignments


def encode_labels(examples: Sequence[RawExample]) -> tuple[np.ndarray, np.ndarray]:
    class_names = np.asarray(sorted({example.class_name for example in examples}))
    lookup = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray([lookup[example.class_name] for example in examples], dtype=np.int64)
    return labels, class_names


def resample_sequence(sequence: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("resampled length must be positive")
    if len(sequence) == length:
        return sequence.astype(np.float32, copy=True)
    if len(sequence) == 1:
        return np.repeat(sequence, length, axis=0).astype(np.float32)
    old_positions = np.linspace(0.0, 1.0, len(sequence))
    new_positions = np.linspace(0.0, 1.0, length)
    result = np.empty((length, sequence.shape[1]), dtype=np.float32)
    for channel in range(sequence.shape[1]):
        result[:, channel] = np.interp(
            new_positions, old_positions, sequence[:, channel]
        )
    return result


def fixed_length_tensor(
    examples: Sequence[RawExample], length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_count = len(examples[0].groups)
    channel_count = max(len(members) for members in examples[0].group_members)
    x = np.zeros(
        (len(examples), group_count, length, channel_count), dtype=np.float32
    )
    original_lengths = np.zeros((len(examples), group_count), dtype=np.int32)
    group_members = np.full((group_count, channel_count), -1, dtype=np.int32)
    for group_index, members in enumerate(examples[0].group_members):
        group_members[group_index, :len(members)] = members
    for example_index, example in enumerate(examples):
        for group_index, sequence in enumerate(example.groups):
            original_lengths[example_index, group_index] = len(sequence)
            transformed = resample_sequence(sequence, length)
            x[example_index, group_index, :, :sequence.shape[1]] = transformed
    return x, original_lengths, group_members


def padded_tensor(
    examples: Sequence[RawExample], maximum_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group_count = len(examples[0].groups)
    channel_count = max(len(members) for members in examples[0].group_members)
    x = np.zeros(
        (len(examples), group_count, maximum_length, channel_count), dtype=np.float32
    )
    mask = np.zeros((len(examples), group_count, maximum_length), dtype=np.bool_)
    original_lengths = np.zeros((len(examples), group_count), dtype=np.int32)
    group_members = np.full((group_count, channel_count), -1, dtype=np.int32)
    for group_index, members in enumerate(examples[0].group_members):
        group_members[group_index, :len(members)] = members
    for example_index, example in enumerate(examples):
        for group_index, sequence in enumerate(example.groups):
            original_lengths[example_index, group_index] = len(sequence)
            if len(sequence) > maximum_length:
                sequence = resample_sequence(sequence, maximum_length)
            used = len(sequence)
            x[example_index, group_index, :used, :sequence.shape[1]] = sequence
            mask[example_index, group_index, :used] = True
    return x, mask, original_lengths, group_members


def standardize_fixed(
    x: np.ndarray, splits: np.ndarray, group_members: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = x[splits == SPLIT_IDS["train"]]
    if not len(train):
        raise ValueError("training split is empty")
    mean = train.mean(axis=(0, 2), keepdims=True)
    scale = train.std(axis=(0, 2), keepdims=True)
    scale[scale < 1e-6] = 1.0
    standardized = (x - mean) / scale
    for group, members in enumerate(group_members):
        standardized[:, group, :, members < 0] = 0.0
    return standardized.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def standardize_padded(
    x: np.ndarray,
    mask: np.ndarray,
    splits: np.ndarray,
    group_members: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_count = x.shape[1]
    channel_count = x.shape[3]
    mean = np.zeros((1, group_count, 1, channel_count), dtype=np.float32)
    scale = np.ones_like(mean)
    train_indices = np.flatnonzero(splits == SPLIT_IDS["train"])
    if not len(train_indices):
        raise ValueError("training split is empty")
    for group in range(group_count):
        values = x[train_indices, group][mask[train_indices, group]]
        mean[0, group, 0] = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        scale[0, group, 0] = std
    standardized = (x - mean) / scale
    standardized[~mask] = 0.0
    for group, members in enumerate(group_members):
        standardized[:, group, :, members < 0] = 0.0
    return standardized.astype(np.float32), mean, scale


def common_metadata(
    result: LoadResult,
    examples: Sequence[RawExample],
    class_names: np.ndarray,
    splits: np.ndarray,
    arguments: argparse.Namespace,
) -> dict:
    return {
        "schema_version": 1,
        "input_root": str(Path(arguments.input_root).resolve()),
        "trace_files": result.trace_files,
        "examples": len(examples),
        "classes": class_names.tolist(),
        "class_count": len(class_names),
        "split_unit": arguments.split_unit,
        "split_seed": arguments.seed,
        "validation_fraction": arguments.validation_fraction,
        "test_fraction": arguments.test_fraction,
        "validation_units": arguments.validation_units,
        "test_units": arguments.test_units,
        "split_counts": {
            name: int(np.count_nonzero(splits == split_id))
            for name, split_id in SPLIT_IDS.items()
        },
        "skipped_measurements": result.skipped_measurements,
        "skipped_incomplete_bundles": result.skipped_incomplete_bundles,
        "include_invalid": arguments.include_invalid,
        "selected_input_ids": (
            [value for value in arguments.input_ids.split(",") if value]
            if arguments.input_ids else None
        ),
        "selected_programs": (
            [value for value in arguments.programs.split(",") if value]
            if arguments.programs else None
        ),
        "event_subset": (
            str(arguments.event_subset.resolve()) if arguments.event_subset else None
        ),
        "selected_member_indices": (
            list(read_event_subset(arguments.event_subset))
            if arguments.event_subset else None
        ),
        "signal": "interval member count / interval retired-instruction count",
    }


def read_event_subset(path: Path) -> tuple[int, ...]:
    """Read member indices from an adaptive-format event CSV."""
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(
                line for line in source
                if line.strip() and not line.lstrip().startswith("#")
            )
            if reader.fieldnames != EVENT_COLUMNS:
                raise ValueError(
                    "event subset CSV header must be exactly: " + ",".join(EVENT_COLUMNS)
                )
            leaders: list[int] = []
            members: list[int] = []
            for row in reader:
                index = int(row["index"], 0)
                role = row["role"].strip().lower()
                if role == "leader":
                    leaders.append(index)
                elif role == "member":
                    members.append(index)
                else:
                    raise ValueError(f"invalid event role {role!r}")
    except OSError as error:
        raise ValueError(f"cannot read event subset {path}: {error}") from error
    if len(leaders) != 1:
        raise ValueError("event subset CSV must contain exactly one leader")
    if not members or len(members) != len(set(members)):
        raise ValueError("event subset CSV members must be nonempty and unique")
    return tuple(members)


def add_common_preprocess_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation-units", type=int,
        help="exact validation trace/run units per class; must accompany --test-units",
    )
    parser.add_argument(
        "--test-units", type=int,
        help="exact test trace/run units per class; overrides fractional sizing",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--split-unit", choices=("trace", "run"), default="trace",
        help="keep whole trace files together by default; use run only when each class has too few files",
    )
    parser.add_argument(
        "--include-invalid", action="store_true",
        help="include measurements that fail coverage, retention, loss, or scheduling checks",
    )
    parser.add_argument(
        "--event-subset", type=Path,
        help="adaptive-format event CSV whose member indices select input channels",
    )
    parser.add_argument(
        "--input-ids",
        help="optional comma-separated input IDs to include, for example 1 or 1,100",
    )
    parser.add_argument(
        "--programs",
        help="optional comma-separated executable names to include",
    )


def save_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def string_array(values: Iterable[str]) -> np.ndarray:
    values = list(values)
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")
