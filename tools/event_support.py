#!/usr/bin/env python3
"""Run a target once per leader-plus-three-member perf group."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, Sequence


MAX_MEMBERS = 3
DEFAULT_RING_BUFFER_PAGES = 128
MAX_RING_BUFFER_PAGES = 1 << 16
RING_BUFFER_PAGES_ENV = "PMC_RING_BUFFER_PAGES"
HOISTED_SAMPLE_FIELDS = ("pid", "tid", "period")
EXPECTED_COLUMNS = [
    "index",
    "event_name",
    "event_number",
    "umask",
    "mode",
    "sample_period",
    "type",
    "role",
]


@dataclass(frozen=True)
class Event:
    index: int
    event_name: str
    mode: str
    sample_period: int
    event_type: str
    role: str


@dataclass(frozen=True)
class EventConfiguration:
    leader: Event
    members: tuple[Event, ...]


def validate_ring_buffer_pages(value: int) -> int:
    if value <= 0 or value & (value - 1) or value > MAX_RING_BUFFER_PAGES:
        raise ValueError(
            f"ring buffer pages must be a power of two between 1 and {MAX_RING_BUFFER_PAGES}"
        )
    return value


def parse_ring_buffer_pages(value: str) -> int:
    try:
        return validate_ring_buffer_pages(int(value, 0))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def resolve_ring_buffer_pages(command_line_value: int | None) -> int:
    if command_line_value is not None:
        return validate_ring_buffer_pages(command_line_value)
    environment = os.environ.get(RING_BUFFER_PAGES_ENV)
    if environment is None or not environment.strip():
        return DEFAULT_RING_BUFFER_PAGES
    try:
        return validate_ring_buffer_pages(int(environment, 0))
    except ValueError as error:
        raise ValueError(f"invalid {RING_BUFFER_PAGES_ENV}: {error}") from error


def read_event_configuration(path: Path) -> EventConfiguration:
    """
    Load and validate a PMC event CSV into an EventConfiguration.
    Reads ``path`` as UTF-8 CSV, skips blank lines and ``#`` comments, and
    requires a header that matches ``EXPECTED_COLUMNS`` exactly. Each data row
    becomes an ``Event``. Leaders must use ``sampling`` or ``sampling_freq``
    with a positive sample period; members must use ``counting``. The file must
    define unique indices, exactly one leader, and at least one member.
    Args:
        path: Path to the event CSV (e.g. ``events/pmc_group_events_full.csv``).
    Returns:
        EventConfiguration with ``.leader`` (single Event) and ``.members``
        (tuple of counting Events).
    Raises:
        ValueError: Bad header, invalid row, duplicate indices, wrong number of
            leaders, or no members.
        OSError: If ``path`` cannot be opened.
    """
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(
            (line for line in source if line.strip() and not line.lstrip().startswith("#"))
        )
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                "event CSV header must be exactly: " + ",".join(EXPECTED_COLUMNS)
            )
        events: list[Event] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                event = Event(
                    index=int(row["index"], 0),
                    event_name=row["event_name"].strip(),
                    mode=row["mode"].strip().lower(),
                    sample_period=int(row["sample_period"], 0),
                    event_type=row["type"].strip().lower(),
                    role=row["role"].strip().lower(),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid event CSV line {line_number}: {error}") from error
            if not event.event_name:
                raise ValueError(f"event CSV line {line_number} has an empty event name")
            if event.role not in {"leader", "member"}:
                raise ValueError(f"event CSV line {line_number} has invalid role {event.role!r}")
            if event.event_type not in {"raw", "fixed", "hardware"}:
                raise ValueError(
                    f"event CSV line {line_number} has invalid type {event.event_type!r}"
                )
            if event.role == "leader":
                if event.mode not in {"sampling", "sampling_freq"}:
                    raise ValueError("the leader must use sampling or sampling_freq")
                if event.sample_period <= 0:
                    raise ValueError("the leader sample period/frequency must be positive")
            elif event.mode != "counting":
                raise ValueError("all members must use counting mode")
            events.append(event)

    indices = [event.index for event in events]
    if len(indices) != len(set(indices)):
        raise ValueError("event indices must be unique")
    leaders = [event for event in events if event.role == "leader"]
    members = tuple(event for event in events if event.role == "member")
    if len(leaders) != 1:
        raise ValueError(f"event CSV must define exactly one leader; found {len(leaders)}")
    if not members:
        raise ValueError("event CSV must define at least one member")
    return EventConfiguration(leaders[0], members)


def member_batches(members: Sequence[Event], size: int = MAX_MEMBERS) -> list[tuple[Event, ...]]:
    """
    Split member events into fixed-size batches for perf group runs.
    Each batch is a tuple of at most ``size`` members (default ``MAX_MEMBERS``,
    which is 3). The final batch may be shorter if ``len(members)`` is not a
    multiple of ``size``. Order is preserved from the input sequence.
    Args:
        members: Member events to partition (typically from ``select_members``).
        size: Members per batch; must be between 1 and ``MAX_MEMBERS``.
    Returns:
        List of member batches, each a tuple of ``Event``.
    Raises:
        ValueError: If ``size`` is outside ``1..MAX_MEMBERS``.
    """
    if size < 1 or size > MAX_MEMBERS:
        raise ValueError(f"batch size must be between 1 and {MAX_MEMBERS}")
    return [tuple(members[offset : offset + size]) for offset in range(0, len(members), size)]


def select_members(configuration: EventConfiguration, indices: str | None) -> tuple[Event, ...]:
    """
    Select a subset of member events from an EventConfiguration.
    If ``indices`` is empty or ``None``, returns all members in CSV order.
    Otherwise parses a comma-separated list of member indices (base 0/8/10/16
    via ``int(..., 0)``), validates uniqueness, and returns those members in
    the requested order. Leader indices and unknown indices are rejected.
    Args:
        configuration: Parsed event configuration from ``read_event_configuration``.
        indices: Optional comma-separated member indices (e.g. ``"1,3,5"``),
            or ``None``/empty to keep every member.
    Returns:
        Tuple of selected ``Event`` members.
    Raises:
        ValueError: Malformed index list, duplicates, empty selection after
            parsing, or indices that are not members.
    """
    if not indices:
        return configuration.members
    try:
        requested = [int(value.strip(), 0) for value in indices.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("--member-indices must be a comma-separated integer list") from error
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--member-indices must contain unique member indices")
    by_index = {event.index: event for event in configuration.members}
    missing = [index for index in requested if index not in by_index]
    if missing:
        raise ValueError(f"unknown or non-member indices: {missing}")
    return tuple(by_index[index] for index in requested)


def install_termination_handlers() -> None:
    def interrupt(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)


def load_measurements(path: Path) -> list[dict]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read target output {path}: {error}") from error
    if document.get("schema_version") != 2 or not isinstance(document.get("measurements"), list):
        raise RuntimeError(f"target output {path} is not grouped libpmc schema version 2")
    return document["measurements"]


def compact_measurement(measurement: dict) -> dict:
    """Move sample fields that never vary within a measurement up one level.

    A multithreaded target reports several thread ids, so a field is only
    hoisted when every sample agrees on its value.
    """
    samples = measurement.get("samples")
    if not samples:
        return measurement
    for field in HOISTED_SAMPLE_FIELDS:
        values = {sample[field] for sample in samples if field in sample}
        if len(values) != 1 or any(field not in sample for sample in samples):
            continue
        measurement[field] = values.pop()
        for sample in samples:
            del sample[field]
    return measurement


def write_output(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        json.dump(document, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def collect(
    command: Sequence[str],
    csv_path: Path,
    configuration: EventConfiguration,
    members: Sequence[Event],
    runs: int,
    timeout: float,
    output: Path,
    ring_buffer_pages: int = DEFAULT_RING_BUFFER_PAGES,
) -> int:
    ring_buffer_pages = validate_ring_buffer_pages(ring_buffer_pages)
    batches = member_batches(members)
    aggregate: dict = {
        "schema_version": 2,
        "configuration": {
            "event_file": str(csv_path.resolve()),
            "leader": {
                "index": configuration.leader.index,
                "event_name": configuration.leader.event_name,
                "mode": configuration.leader.mode,
                "sample_period": configuration.leader.sample_period,
            },
            "members_per_batch": MAX_MEMBERS,
            "ring_buffer_pages": ring_buffer_pages,
            "runs": runs,
        },
        "measurements": [],
        "failures": [],
    }

    with tempfile.TemporaryDirectory(prefix="libpmc-new-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for batch_number, batch in enumerate(batches, start=1):
            member_indices = [event.index for event in batch]
            print(
                f"Batch {batch_number}/{len(batches)}: leader {configuration.leader.index} "
                f"+ members {member_indices}"
            )
            for run_number in range(1, runs + 1):
                raw_output = temporary_root / f"batch-{batch_number}-run-{run_number}.json"
                environment = os.environ.copy()
                environment["PMC_EVENTS_CSV"] = str(csv_path.resolve())
                environment["PMC_MEMBER_INDICES"] = ",".join(map(str, member_indices))
                environment["PMC_OUTPUT_FILE"] = str(raw_output)
                environment[RING_BUFFER_PAGES_ENV] = str(ring_buffer_pages)
                try:
                    completed = subprocess.run(
                        command,
                        env=environment,
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired as error:
                    aggregate["failures"].append(
                        {
                            "batch": batch_number,
                            "run": run_number,
                            "member_indices": member_indices,
                            "reason": f"timeout after {error.timeout} seconds",
                        }
                    )
                    print(f"  run {run_number}/{runs}: TIMEOUT", file=sys.stderr)
                    continue

                if completed.returncode != 0:
                    aggregate["failures"].append(
                        {
                            "batch": batch_number,
                            "run": run_number,
                            "member_indices": member_indices,
                            "reason": f"target exit code {completed.returncode}",
                            "stderr": completed.stderr[-2000:],
                        }
                    )
                    print(
                        f"  run {run_number}/{runs}: FAILED ({completed.returncode})\n"
                        f"{completed.stderr[-500:]}",
                        file=sys.stderr,
                    )
                    continue

                try:
                    measurements = load_measurements(raw_output)
                except RuntimeError as error:
                    aggregate["failures"].append(
                        {
                            "batch": batch_number,
                            "run": run_number,
                            "member_indices": member_indices,
                            "reason": str(error),
                        }
                    )
                    print(f"  run {run_number}/{runs}: FAILED ({error})", file=sys.stderr)
                    continue
                for measurement in measurements:
                    measurement["batch"] = batch_number
                    measurement["run"] = run_number
                    measurement["member_indices"] = member_indices
                    aggregate["measurements"].append(compact_measurement(measurement))
                print(f"  run {run_number}/{runs}: {len(measurements)} measurement(s)")

    write_output(output, aggregate)
    failures = len(aggregate["failures"])
    successes = len(aggregate["measurements"])
    print(f"Wrote {successes} measurements and {failures} failures to {output}")
    return 0 if successes and failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect synchronized traces from one leader and batches of three counters"
    )
    parser.add_argument("--target", required=True, help="target command, including arguments")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path(__file__).with_name("pmc_group_events.csv"),
        help="group event CSV",
    )
    parser.add_argument("--member-indices", help="optional comma-separated member subset")
    parser.add_argument(
        "--ring-buffer-pages",
        type=parse_ring_buffer_pages,
        help=f"sampling-ring data pages; defaults to ${RING_BUFFER_PAGES_ENV} or "
        f"{DEFAULT_RING_BUFFER_PAGES}, and must be a power of two",
    )
    parser.add_argument("--runs", type=int, default=1, help="runs per member batch")
    parser.add_argument("--timeout", type=float, default=30.0, help="target timeout in seconds")
    parser.add_argument(
        "--total", type=int, default=1, help="iterations, each writing one output file"
    )
    parser.add_argument(
        "--name", default="pmc_group_traces", help="output file base name"
    )
    parser.add_argument("--start", type=int, default=1, help="first output file suffix")
    return parser


def iteration_output(name: str, iteration: int) -> Path:
    return Path("traces") / f"{name}_{iteration}.json"


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.runs < 1 or arguments.timeout <= 0:
        print("--runs and --timeout must be positive", file=sys.stderr)
        return 2
    if arguments.total < 1 or arguments.start < 1:
        print("--total and --start must be positive", file=sys.stderr)
        return 2
    command = shlex.split(arguments.target)
    if not command:
        print("--target did not contain a command", file=sys.stderr)
        return 2
    try:
        # return a class contains a leader and members
        configuration = read_event_configuration(arguments.events) 
        members = select_members(configuration, arguments.member_indices)
        ring_buffer_pages = resolve_ring_buffer_pages(arguments.ring_buffer_pages)
        install_termination_handlers()
        last = arguments.start + arguments.total - 1
        status = 0
        for position, iteration in enumerate(range(arguments.start, last + 1), start=1):
            output = iteration_output(arguments.name, iteration)
            print(f"Iteration {position}/{arguments.total} (file {output})")
            status |= collect(
                command,
                arguments.events,
                configuration,
                members,
                arguments.runs,
                arguments.timeout,
                output,
                ring_buffer_pages,
            )
        return status
    except (OSError, RuntimeError, ValueError, KeyboardInterrupt) as error:
        print(f"collection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
