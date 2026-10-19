#!/usr/bin/env python3
"""Collect adaptively sampled PMU traces for one function/input class

The adaptive collector has four measurement phases:

  1. Single-invocation scout/pilot counting
  2. Repeated counting scout when the original region is too short
  3. Sampling-period calibration
  4. Actual trace collection

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent # the directory containing this file
BUNDLE = HERE.parent # the repo root

# Reuse the CSV parser and schema-v2 compaction code used by the original
# collector. Keeping these imports local to the repository avoids maintaining
# two subtly different interpretations of the event file.
sys.path.insert(0, str(HERE))
from event_support import (  # noqa: E402
    EventConfiguration,
    compact_measurement,
    member_batches,
    read_event_configuration,
    select_members,
    validate_ring_buffer_pages,
    write_output,
)


DEFAULT_EVENTS = BUNDLE / "events" / "pmc_group_events_full.csv"
DEFAULT_OUTPUT_ROOT = BUNDLE / "results"


def positive_int(text: str) -> int:
    value = int(text, 0)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text, 0)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def power_of_two(text: str) -> int:
    value = positive_int(text)
    try:
        return validate_ring_buffer_pages(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def safe_component(text: str) -> str:
    """Return a filesystem component while retaining readable simple values."""
    stripped = text.strip()
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", stripped).strip("._-")
    readable = readable[:64]
    if readable and readable == stripped and stripped not in {".", ".."}:
        return readable
    digest = hashlib.sha256(stripped.encode("utf-8")).hexdigest()[:10]
    return f"{readable or 'value'}-{digest}"


def input_identifier(arguments: Sequence[str]) -> str:
    """
    Build a filesystem-safe directory name from target argv (excluding argv0).
    With no arguments, returns ``no_args``. Otherwise joins each argument through
    ``safe_component`` with ``__``. If the result exceeds 120 characters, truncates
    and appends a short SHA-256 of the original arguments so distinct inputs stay
    unique under the length cap.
    Args:
        arguments: Target command arguments after the executable (``command[1:]``).
    Returns:
        A path component identifying this input set under the result tree.
    """
    if not arguments:
        return "no_args"
    joined = "__".join(safe_component(argument) for argument in arguments)
    if len(joined) <= 120:
        return joined
    digest = hashlib.sha256("\0".join(arguments).encode("utf-8")).hexdigest()[:10]
    return f"{joined[:109]}-{digest}"

def resolve_executable(command: Sequence[str], working_directory: Path) -> Path:
    
    """
    Resolve the path to the target executable.

    Args:
        command: The command to resolve the executable for.
        working_directory: The working directory to resolve the executable in.

    Returns:
        The path to the target executable.
    """
    
    executable = command[0]
    if "/" in executable:
        candidate = Path(executable)
        if not candidate.is_absolute():
            candidate = working_directory / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise ValueError(f"target executable does not exist: {candidate}")
        return candidate
    found = shutil.which(executable)
    if found is None:
        raise ValueError(f"target executable is not on PATH: {executable}")
    return Path(found).resolve()


def choose_sample_target(
    golden_count: int,
    base_samples: int,
    reference_instructions: int,
    minimum_samples: int,
    maximum_samples: int,
) -> int:
    """
    Sample_count = base_samples * math.sqrt(golden_count / reference_instructions)
    This is a sub-linear function because i don't want the samples to grow too fast
    e.g. if golden_count = reference_instructions, then sample_count = base_samples = 100
    """
    scaled = base_samples * math.sqrt(golden_count / reference_instructions)
    return min(maximum_samples, max(minimum_samples, int(scaled + 0.5))) # rounding to the nearest integer


def choose_period(
    golden_count: int,
    desired_samples: int,
    minimum_period: int,
) -> tuple[int, int]:
    maximum_samples_at_floor = max(1, golden_count // minimum_period)
    effective_samples = min(desired_samples, maximum_samples_at_floor)
    # Rounding upward can put the nominal record count just below the target.
    # Floor division keeps golden_count / period at or above effective_samples.
    period = max(minimum_period, golden_count // effective_samples)
    return effective_samples, period


def choose_measurement_repetitions(
    single_invocation_count: int,
    desired_samples: int,
    minimum_period: int,
    maximum_repetitions: int,
) -> tuple[int, int]:
    """Return (repetitions, required_count) for a safe sampling period."""
    if single_invocation_count <= 0:
        raise ValueError("single-invocation golden count must be positive")
    required_count = desired_samples * minimum_period
    repetitions = max(
        1,
        (required_count + single_invocation_count - 1)
        // single_invocation_count,
    )
    if repetitions > maximum_repetitions:
        raise ValueError(
            f"need {repetitions:,} measurement repetitions to reach "
            f"{desired_samples} samples at period {minimum_period:,}, exceeding "
            f"--max-repetitions={maximum_repetitions:,}"
        )
    return repetitions, required_count


def reduce_period_for_sample_floor(
    current_period: int,
    observed_samples: int,
    required_samples: int,
    minimum_period: int,
) -> int | None:
    """Return a more conservative period, or None when no reduction is possible."""
    if observed_samples >= required_samples:
        return current_period
    if current_period <= minimum_period:
        return None
    observed_for_scale = max(1, observed_samples)
    reduced = int(
        current_period * observed_for_scale / required_samples * 0.95
    )
    return max(minimum_period, min(current_period - 1, reduced))


def read_integer(path: str) -> int | None:
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def install_termination_handlers() -> None:
    """
    Map SIGTERM and SIGHUP to KeyboardInterrupt like SIGINT (Ctrl-C).
    Lets kill/hangup stop collection through normal Python exception handling
    instead of terminating the process immediately mid-run.
    """
    def interrupt(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)


def load_target_measurements(path: Path) -> list[dict]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read target output {path}: {error}") from error
    measurements = document.get("measurements") # code_study/scout/measurement.json
    if document.get("schema_version") != 2 or not isinstance(measurements, list):
        raise RuntimeError("target output is not grouped libpmc schema version 2")
    return measurements


def select_measurement(measurements: Sequence[dict], label: str | None) -> dict:
    if label is None:
        if len(measurements) != 1:
            labels = [measurement.get("label") for measurement in measurements]
            raise RuntimeError(
                "target produced "
                f"{len(measurements)} measurements {labels}; use --measurement-label"
            )
        return measurements[0]
    matches = [measurement for measurement in measurements if measurement.get("label") == label]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one measurement labeled {label!r}; found {len(matches)}"
        )
    return matches[0]


def run_target(
    command: Sequence[str],
    working_directory: Path,
    events: Path,
    member_indices: Sequence[int],
    mode: str,
    raw_output: Path,
    timeout: float,
    measurement_label: str | None,
    sample_period: int | None = None,
    ring_buffer_pages: int = 8192,
    context_mixer: str = "none",
    cpu_affinity: int | None = None,
    measurement_repetitions: int = 1,
) -> tuple[dict, float]:
    environment = os.environ.copy()
    environment["PMC_EVENTS_CSV"] = str(events)
    environment["PMC_MEMBER_INDICES"] = ",".join(map(str, member_indices))
    environment["PMC_OUTPUT_FILE"] = str(raw_output)
    environment["PMC_RUN_MODE"] = mode
    environment.pop("PMC_SAMPLE_PERIOD", None)
    environment.pop("PMC_RING_BUFFER_PAGES", None)
    environment["PMC_MEASUREMENT_REPETITIONS"] = str(measurement_repetitions)
    environment["PMC_CONTEXT_MIXER"] = context_mixer
    if cpu_affinity is None:
        environment.pop("PMC_CPU_AFFINITY", None)
    else:
        environment["PMC_CPU_AFFINITY"] = str(cpu_affinity)
    if sample_period is not None:
        environment["PMC_SAMPLE_PERIOD"] = str(sample_period)
        environment["PMC_RING_BUFFER_PAGES"] = str(ring_buffer_pages)

    raw_output.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=working_directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"target timed out after {error.timeout} seconds") from error
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"target exited with status {completed.returncode}: {detail[-2000:]}"
        )

    measurement = select_measurement(
        load_target_measurements(raw_output), measurement_label
    )
    reported_mode = measurement.get("run_mode")
    if reported_mode != mode:
        if reported_mode is None:
            raise RuntimeError(
                "target does not report PMC_RUN_MODE; link it against the "
                "adaptive_sampling copy of pmc.c"
            )
        raise RuntimeError(f"requested {mode} mode but target reported {reported_mode!r}")
    reported_mixer = measurement.get("context_mixer", "none")
    if reported_mixer != context_mixer:
        raise RuntimeError(
            f"requested context mixer {context_mixer!r} but target reported "
            f"{reported_mixer!r}"
        )
    reported_cpu = measurement.get("cpu_affinity")
    if reported_cpu != cpu_affinity:
        raise RuntimeError(
            f"requested CPU affinity {cpu_affinity!r} but target reported "
            f"{reported_cpu!r}"
        )
    if cpu_affinity is not None:
        sample_cpus = {
            int(sample["cpu"])
            for sample in measurement.get("samples", [])
            if "cpu" in sample
        }
        unexpected_cpus = sample_cpus - {cpu_affinity}
        if unexpected_cpus:
            raise RuntimeError(
                f"target was pinned to CPU {cpu_affinity}, but samples reported "
                f"CPUs {sorted(unexpected_cpus)}"
            )
    if mode == "scout":
        if measurement.get("num_samples") != 0:
            raise RuntimeError("scout mode unexpectedly produced samples")
        if measurement.get("leader", {}).get("sample_mode") != "counting":
            raise RuntimeError("scout leader was not opened in counting mode")
    elif measurement.get("leader", {}).get("sample_value") != sample_period:
        effective = measurement.get("leader", {}).get("sample_value")
        raise RuntimeError(
            f"sample-period override failed: requested {sample_period}, got {effective}"
        )
    return measurement, wall_seconds


def calibrate_sample_period(
    command: Sequence[str],
    working_directory: Path,
    events: Path,
    member_groups: Sequence[Sequence[int]],
    raw_output: Path,
    timeout: float,
    measurement_label: str | None,
    initial_period: int,
    minimum_period: int,
    required_samples: int,
    calibration_runs: int,
    maximum_attempts: int,
    ring_buffer_pages: int,
    context_mixer: str,
    cpu_affinity: int | None,
    measurement_repetitions: int,
) -> tuple[int, list[dict], float]:
    """Find a period for which every calibration run retains the sample floor."""
    period = initial_period
    attempts: list[dict] = []
    total_wall_seconds = 0.0
    for attempt_number in range(1, maximum_attempts + 1):
        print(
            f"Calibration:    attempt {attempt_number}/{maximum_attempts}, "
            f"period={period:,}, {calibration_runs} run(s) x "
            f"{len(member_groups)} group(s)",
            flush=True,
        )
        calibration_measurements = []
        # calibrate for each batch of events
        for batch_number, member_indices in enumerate(member_groups, start=1):
            # calibrate with multiple runs 
            for run_number in range(1, calibration_runs + 1):
                try:
                    measurement, wall_seconds = run_target(
                        command,
                        working_directory,
                        events,
                        member_indices,
                        "sampling",
                        raw_output,
                        timeout,
                        measurement_label,
                        period,
                        ring_buffer_pages,
                        context_mixer,
                        cpu_affinity,
                        measurement_repetitions,
                    )
                except RuntimeError as error:
                    raise RuntimeError(
                        f"calibration group {batch_number} run {run_number} "
                        f"failed: {error}"
                    ) from error
                total_wall_seconds += wall_seconds
                calibration_measurements.append(
                    {
                        "batch": batch_number,
                        "run": run_number,
                        "member_indices": list(member_indices),
                        "num_samples": int(measurement.get("num_samples", 0)),
                        "leader_total_count": int(
                            measurement.get("leader_total_count", 0)
                        ),
                        "lost_samples": int(measurement.get("lost_samples", 0)),
                        "time_enabled": int(measurement.get("time_enabled", 0)),
                        "time_running": int(measurement.get("time_running", 0)),
                        "wall_seconds": wall_seconds,
                    }
                )

        minimum_observed = min(
            item["num_samples"] for item in calibration_measurements
        )
        attempts.append(
            {
                "attempt": attempt_number,
                "sample_period": period,
                "minimum_observed_samples": minimum_observed,
                "measurements": calibration_measurements,
            }
        )
        if minimum_observed >= required_samples:
            print(
                f"Calibration:    accepted period={period:,}; minimum retained="
                f"{minimum_observed}",
                flush=True,
            )
            return period, attempts, total_wall_seconds

        reduced_period = reduce_period_for_sample_floor(
            period,
            minimum_observed,
            required_samples,
            minimum_period,
        )
        if reduced_period is None:
            raise RuntimeError(
                f"only {minimum_observed} samples were retained at the hard "
                f"minimum period {minimum_period}; cannot satisfy the required "
                f"floor of {required_samples}"
            )
        print(
            f"Calibration:    only {minimum_observed} samples retained; "
            f"retrying at period={reduced_period:,}",
            flush=True,
        )
        period = reduced_period

    raise RuntimeError(
        f"sample floor calibration did not reach {required_samples} samples "
        f"within {maximum_attempts} attempts"
    )


def annotate_measurement(
    measurement: dict,
    *,
    file_index: int,
    batch: int,
    run: int,
    member_indices: Sequence[int],
    wall_seconds: float,
    golden_count: int,
    sample_period: int,
    target_samples: int,
    measurement_repetitions: int,
) -> dict:
    sampled_count = int(measurement.get("leader_total_count", 0))
    num_samples = int(measurement.get("num_samples", 0))
    expected_records = sampled_count / sample_period if sample_period else 0.0
    measurement["collection"] = {
        "file_index": file_index,
        "batch": batch,
        "run": run,
        "member_indices": list(member_indices),
        "measurement_repetitions": measurement_repetitions,
        "wall_seconds": wall_seconds,
    }
    measurement["adaptive_validation"] = {
        "golden_count": golden_count,
        "target_samples": target_samples,
        "sample_period": sample_period,
        "counter_coverage": sampled_count / golden_count if golden_count else 0.0,
        "expected_records": expected_records,
        "record_retention": num_samples / expected_records if expected_records else 0.0,
    }
    return compact_measurement(measurement)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remaining = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def numeric_summary(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    return {
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "maximum": max(values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="target command, including input arguments")
    parser.add_argument(
        "--events", type=Path, default=DEFAULT_EVENTS,
        help=f"event CSV (default: {DEFAULT_EVENTS})",
    )
    parser.add_argument("--member-indices", help="optional comma-separated member subset")
    parser.add_argument("--measurement-label", help="measurement label when target emits more than one")
    parser.add_argument(
        "--context-mixer",
        choices=("none", "mixed_v1"),
        default="none",
        help="preconditioner run while counters are disabled (default: none)",
    )
    parser.add_argument(
        "--cpu",
        type=nonnegative_int,
        help="pin every target execution to this logical CPU",
    )
    parser.add_argument("--runs", type=positive_int, default=1, help="runs per event group")
    parser.add_argument("--total", type=positive_int, default=1, help="number of trace files")
    parser.add_argument("--start", type=positive_int, default=1, help="first trace-file index")
    parser.add_argument("--name", required=True, help="name of the file that contains accumulated runs ")
    parser.add_argument("--timeout", type=positive_float, default=30.0)

    parser.add_argument("--working-directory", type=Path, default=Path.cwd(), 
    help="by default, use the path of where you launch the script")

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--ring-buffer-pages",
        type=power_of_two,
        default=8192,
        help="sampling-ring data pages (default: 8192, approximately 32 MiB)",
    )
    parser.add_argument("--base-samples", "--target-samples", dest="base_samples", type=positive_int, default=100)
    parser.add_argument("--reference-instructions", type=positive_int, default=10_000_000)
    parser.add_argument("--min-samples", type=positive_int, default=20)
    parser.add_argument("--max-samples", type=positive_int, default=1000)
    parser.add_argument(
        "--min-period", type=positive_int, default=1,
        help="hard minimum retired-instruction period (default: 1)",
    )
    parser.add_argument(
        "--max-repetitions", type=positive_int, default=1_000_000,
        help=(
            "maximum target-function invocations in one measured region "
            "during automatic repetition (default: 1000000)"
        ),
    )
    parser.add_argument(
        "--calibration-runs", type=positive_int, default=3,
        help="sampling runs per event group used to enforce the sample floor",
    )
    parser.add_argument(
        "--max-calibration-attempts", type=positive_int, default=10,
        help="maximum period-reduction attempts (default: 10)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace trace files whose requested indices already exist",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        # python tools/collect_pmc_groups.py --target './myprog "arg1 arg2"'
        # command = ['./myprog', 'arg1 arg2']
        command = shlex.split(arguments.target)
    except ValueError as error:
        parser.error(f"cannot parse --target: {error}")
    if not command:
        parser.error("--target did not contain a command")
    if arguments.min_samples > arguments.max_samples:
        parser.error("--min-samples cannot exceed --max-samples")

    working_directory = arguments.working_directory.resolve() # find the absolute path of the working directory
    events = arguments.events.resolve() # find the absolute path of the event CSV file
    if not working_directory.is_dir():
        parser.error(f"working directory does not exist: {working_directory}")
    if not events.is_file():
        parser.error(f"event CSV does not exist: {events}")

    try:
        # resolve the path to the target executable
        executable = resolve_executable(command, working_directory)
        # return a class contains a leader and members
        configuration: EventConfiguration = read_event_configuration(events)
        # return a list of members. it is expceted to use all the members
        members = select_members(configuration, arguments.member_indices)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    batches = member_batches(members) # Usually three members per batch
    member_groups = [[event.index for event in batch] for batch in batches]

    collection_name = safe_component(arguments.name) # remove special characters
    if collection_name != arguments.name:
        parser.error("--name must contain only letters, digits, '.', '_' or '-'")
        
    input_id = input_identifier(command[1:])
    # builds the output root for one collection run 
    """
    arguments.output_root
      = .../learning_output/ 

        collection_name
            = learning_demo 

        executable
            = .../rebuilt/bin/example_group

        executable.stem
            = example_group

        command
            = [.../example_group, "100"]

        command[1:]
            = ["100"]

        input_identifier(["100"])
            = 100
    """
    result_directory = (
        arguments.output_root.resolve() / collection_name / safe_component(executable.stem) / input_id
    )
    trace_directory = result_directory / "traces"
    manifest_directory = result_directory / "manifests"
    last_index = arguments.start + arguments.total - 1
    trace_paths = [
        trace_directory / f"{collection_name}_{index}.json"
        for index in range(arguments.start, last_index + 1)
    ]
    existing = [path for path in trace_paths if path.exists()]
    if existing and not arguments.overwrite:
        parser.error(
            f"{len(existing)} requested trace file(s) already exist; change --start "
            "or pass --overwrite"
        )

    install_termination_handlers()
    trace_directory.mkdir(parents=True, exist_ok=True)
    manifest_directory.mkdir(parents=True, exist_ok=True)
    collection_started_at = utc_now()
    collection_started = time.perf_counter()
    max_sample_rate_start = read_integer("/proc/sys/kernel/perf_event_max_sample_rate")
    cpu_time_percent_start = read_integer("/proc/sys/kernel/perf_cpu_time_max_percent")

    print(f"Class name:     {arguments.name}")
    print(f"Target:         {shlex.join(command)}")
    print(f"Input ID:       {input_id}")
    print(f"CPU affinity:   {arguments.cpu if arguments.cpu is not None else 'none'}")
    print(f"Context mixer:  {arguments.context_mixer}")
    print(f"Event groups:   {len(batches)} ({len(members)} member events)")
    print(f"Output:         {result_directory}")
    print("Pilot:          counting retired instructions ...", flush=True)

    all_failures: list[dict] = []
    successful_runs = 0
    sampling_target_seconds = 0.0
    counter_coverages: list[float] = []
    record_retentions: list[float] = []
    running_ratios: list[float] = []
    sample_counts: list[float] = []
    total_lost_samples = 0

    """
    CRITICAL: Currently the calibration can only adjust the sampling period according to
    only one function from a target/program. That means if there are multiple different targets with various
    length, a sampling period that is suitable for one function may not be suitable for the other. 
    need to address in the future.
    """
    with tempfile.TemporaryDirectory(prefix="libpmc-classification-") as scratch_text:
        raw_output = Path(scratch_text) / "measurement.json"
        # scout phase: scout mode is set so libpmc will only do a final read of the counter value
        # only one batch of events is used. 
        try:
            pilot, pilot_wall_seconds = run_target(
                command,
                working_directory,
                events,
                [event.index for event in batches[0]],
                "scout",
                raw_output,
                arguments.timeout,
                arguments.measurement_label,
                context_mixer=arguments.context_mixer,
                cpu_affinity=arguments.cpu,
            ) # pilot is the measurement field from the libpmc's output json file
        except RuntimeError as error:
            print(f"pilot failed: {error}", file=sys.stderr)
            return 1

        single_invocation_golden_count = int(pilot.get("leader_total_count", 0))
        if single_invocation_golden_count <= 0:
            print(
                f"pilot failed: invalid golden count "
                f"{single_invocation_golden_count}",
                file=sys.stderr,
            )
            return 1

        # by default base_samples is 100, reference_instructions is 10,000,000
        # min_samples is 20, max_samples is 1000

        desired_samples = choose_sample_target(
            single_invocation_golden_count,
            arguments.base_samples,
            arguments.reference_instructions,
            arguments.min_samples,
            arguments.max_samples,
        )

        try:
            measurement_repetitions, required_golden_count = (
                choose_measurement_repetitions(
                    single_invocation_golden_count,
                    desired_samples,
                    arguments.min_period,
                    arguments.max_repetitions,
                )
            )
        except ValueError as error:
            print(f"pilot failed: {error}", file=sys.stderr)
            return 1

        repeated_pilot = pilot
        repeated_pilot_wall_seconds = 0.0
        repetition_scouts: list[dict] = []
        golden_count = single_invocation_golden_count
        if measurement_repetitions > 1:
            for repetition_attempt in range(
                1, arguments.max_calibration_attempts + 1
            ):
                print(
                    f"Repetition:     scout {repetition_attempt}/"
                    f"{arguments.max_calibration_attempts}, "
                    f"calls={measurement_repetitions:,} ...",
                    flush=True,
                )
                try:
                    repeated_pilot, scout_wall_seconds = run_target(
                        command,
                        working_directory,
                        events,
                        [event.index for event in batches[0]],
                        "scout",
                        raw_output,
                        arguments.timeout,
                        arguments.measurement_label,
                        context_mixer=arguments.context_mixer,
                        cpu_affinity=arguments.cpu,
                        measurement_repetitions=measurement_repetitions,
                    )
                except RuntimeError as error:
                    print(f"repeated pilot failed: {error}", file=sys.stderr)
                    return 1
                repeated_pilot_wall_seconds += scout_wall_seconds
                golden_count = int(
                    repeated_pilot.get("leader_total_count", 0)
                )
                repetition_scouts.append(
                    {
                        "attempt": repetition_attempt,
                        "measurement_repetitions": measurement_repetitions,
                        "golden_count": golden_count,
                        "wall_seconds": scout_wall_seconds,
                    }
                )
                if golden_count >= required_golden_count:
                    break
                if golden_count <= 0:
                    print(
                        f"repeated pilot failed: invalid golden count "
                        f"{golden_count}",
                        file=sys.stderr,
                    )
                    return 1
                corrected_repetitions = max(
                    measurement_repetitions + 1,
                    (
                        measurement_repetitions * required_golden_count
                        + golden_count
                        - 1
                    )
                    // golden_count,
                )
                if corrected_repetitions > arguments.max_repetitions:
                    print(
                        f"repeated pilot failed: corrected repetition count "
                        f"{corrected_repetitions:,} exceeds --max-repetitions="
                        f"{arguments.max_repetitions:,}",
                        file=sys.stderr,
                    )
                    return 1
                measurement_repetitions = corrected_repetitions
            else:
                print(
                    f"repeated pilot failed: did not reach required golden "
                    f"count {required_golden_count:,} within "
                    f"{arguments.max_calibration_attempts} attempts",
                    file=sys.stderr,
                )
                return 1

        target_samples, initial_sample_period = choose_period(
            golden_count, desired_samples, arguments.min_period
        )

        if target_samples < desired_samples:
            print(
                f"repeated pilot failed: golden count {golden_count:,} and "
                f"minimum period {arguments.min_period:,} permit only "
                f"{target_samples} samples, below the desired "
                f"{desired_samples}",
                file=sys.stderr,
            )
            return 1
        
        print(
            f"Pilot result:   single={single_invocation_golden_count:,}, "
            f"calls={measurement_repetitions:,}, golden={golden_count:,}, "
            f"desired={desired_samples}, "
            f"target={target_samples}, initial period={initial_sample_period:,}, "
            f"wall={format_duration(pilot_wall_seconds + repeated_pilot_wall_seconds)}"
        )

        try:
            # this "sample_period" will be used to collect the actual traces
            # the old sample period from csv file is overridden
            sample_period, calibration_attempts, calibration_wall_seconds = (
                calibrate_sample_period(
                    command,
                    working_directory,
                    events,
                    member_groups,
                    raw_output,
                    arguments.timeout,
                    arguments.measurement_label,
                    initial_sample_period,
                    arguments.min_period,
                    arguments.min_samples,
                    arguments.calibration_runs,
                    arguments.max_calibration_attempts,
                    arguments.ring_buffer_pages,
                    arguments.context_mixer,
                    arguments.cpu,
                    measurement_repetitions,
                )
            )
        except RuntimeError as error:
            print(f"calibration failed: {error}", file=sys.stderr)
            return 1

        total_target_runs = arguments.total * len(batches) * arguments.runs
        completed_target_runs = 0
        sampling_started = time.perf_counter()
        for file_position, (file_index, trace_path) in enumerate(
            zip(range(arguments.start, last_index + 1), trace_paths), start=1
        ):
            # prepare the header section of a trace file
            trace_started = time.perf_counter()
            trace_document: dict = {
                "schema_version": 2,
                "configuration": {
                    "class_name": arguments.name,
                    "target_command": command,
                    "target_executable": str(executable),
                    "target_arguments": command[1:],
                    "input_id": input_id,
                    "event_file": str(events),
                    "leader": {
                        "index": configuration.leader.index,
                        "event_name": configuration.leader.event_name,
                        "mode": "sampling",
                        "sample_period": sample_period,
                    },
                    "members_per_batch": 3,
                    "member_indices": [event.index for event in members],
                    "ring_buffer_pages": arguments.ring_buffer_pages,
                    "cpu_affinity": arguments.cpu,
                    "context_mixer": arguments.context_mixer,
                    "measurement_repetitions": measurement_repetitions,
                    "runs": arguments.runs,
                    "pilot_golden_count": golden_count,
                    "single_invocation_golden_count": (
                        single_invocation_golden_count
                    ),
                    "repeated_golden_count": golden_count,
                    "desired_samples": desired_samples,
                    "target_samples": target_samples,
                    "required_minimum_samples": arguments.min_samples,
                    "period_calibration_runs": arguments.calibration_runs,
                    "period_calibration_attempts": len(calibration_attempts),
                },
                "collection": {
                    "file_index": file_index,
                    "started_at": utc_now(),
                },
                "measurements": [],
                "failures": [],
            }
            print(
                f"Trace {file_position}/{arguments.total}: {trace_path.name}",
                flush=True,
            )
            # for each batch of events, collect multiple traces from multiple runs
            for batch_number, batch in enumerate(batches, start=1):
                member_indices = [event.index for event in batch]
                for run_number in range(1, arguments.runs + 1):
                    try:
                        measurement, target_wall_seconds = run_target(
                            command,
                            working_directory,
                            events,
                            member_indices,
                            "sampling",
                            raw_output,
                            arguments.timeout,
                            arguments.measurement_label,
                            sample_period,
                            arguments.ring_buffer_pages,
                            arguments.context_mixer,
                            arguments.cpu,
                            measurement_repetitions,
                        )
                        observed_samples = int(measurement.get("num_samples", 0))
                        if observed_samples < arguments.min_samples:
                            raise RuntimeError(
                                f"sample floor violation: retained "
                                f"{observed_samples}, required at least "
                                f"{arguments.min_samples}, calibrated period "
                                f"{sample_period}"
                            )
                        sampling_target_seconds += target_wall_seconds
                        successful_runs += 1
                        annotated = annotate_measurement(
                            measurement,
                            file_index=file_index,
                            batch=batch_number,
                            run=run_number,
                            member_indices=member_indices,
                            wall_seconds=target_wall_seconds,
                            golden_count=golden_count,
                            sample_period=sample_period,
                            target_samples=target_samples,
                            measurement_repetitions=measurement_repetitions,
                        )
                        trace_document["measurements"].append(annotated)
                        validation = annotated["adaptive_validation"]
                        counter_coverages.append(validation["counter_coverage"])
                        record_retentions.append(validation["record_retention"])
                        enabled = annotated.get("time_enabled", 0)
                        running = annotated.get("time_running", 0)
                        running_ratios.append(running / enabled if enabled else 0.0)
                        sample_counts.append(float(annotated.get("num_samples", 0)))
                        total_lost_samples += int(annotated.get("lost_samples", 0))
                        status = (
                            f"{measurement.get('num_samples', 0)} samples, "
                            f"{format_duration(target_wall_seconds)}"
                        )
                    except RuntimeError as error:
                        failure = {
                            "file_index": file_index,
                            "batch": batch_number,
                            "run": run_number,
                            "member_indices": member_indices,
                            "reason": str(error),
                        }
                        trace_document["failures"].append(failure)
                        all_failures.append(failure)
                        status = f"FAILED: {error}"

                    completed_target_runs += 1
                    elapsed_sampling = time.perf_counter() - sampling_started
                    average = elapsed_sampling / completed_target_runs
                    remaining = total_target_runs - completed_target_runs
                    print(
                        f"  group {batch_number}/{len(batches)} run "
                        f"{run_number}/{arguments.runs}: {status}; "
                        f"ETA {format_duration(average * remaining)}",
                        flush=True,
                    )

            trace_document["collection"]["finished_at"] = utc_now()
            trace_document["collection"]["wall_seconds"] = (
                time.perf_counter() - trace_started
            )
            write_output(trace_path, trace_document)

    total_wall_seconds = time.perf_counter() - collection_started
    manifest = {
        "schema_version": 1,
        "class": {
            "name": arguments.name,
            "program": executable.stem,
            "input_id": input_id,
            "target_command": command,
            "target_executable": str(executable),
            "target_arguments": command[1:],
        },
        "configuration": {
            "event_file": str(events),
            "member_indices": [event.index for event in members],
            "member_groups": [[event.index for event in batch] for batch in batches],
            "runs_per_group": arguments.runs,
            "total_trace_files": arguments.total,
            "start_index": arguments.start,
            "ring_buffer_pages": arguments.ring_buffer_pages,
            "cpu_affinity": arguments.cpu,
            "context_mixer": arguments.context_mixer,
            "base_samples": arguments.base_samples,
            "reference_instructions": arguments.reference_instructions,
            "minimum_samples": arguments.min_samples,
            "maximum_samples": arguments.max_samples,
            "minimum_period": arguments.min_period,
            "maximum_repetitions": arguments.max_repetitions,
            "measurement_repetitions": measurement_repetitions,
            "calibration_runs_per_group": arguments.calibration_runs,
            "maximum_calibration_attempts": arguments.max_calibration_attempts,
        },
        "pilot": {
            "measurement_label": pilot.get("label"),
            "golden_count": golden_count,
            "single_invocation_golden_count": single_invocation_golden_count,
            "repeated_golden_count": golden_count,
            "required_golden_count": required_golden_count,
            "measurement_repetitions": measurement_repetitions,
            "desired_samples": desired_samples,
            "target_samples": target_samples,
            "initial_period": initial_sample_period,
            "chosen_period": sample_period,
            "wall_seconds": pilot_wall_seconds + repeated_pilot_wall_seconds,
            "single_invocation_wall_seconds": pilot_wall_seconds,
            "repeated_scout_wall_seconds": repeated_pilot_wall_seconds,
            "repetition_scouts": repetition_scouts,
            "single_invocation_measurement": compact_measurement(pilot),
            "measurement": compact_measurement(repeated_pilot),
        },
        "period_calibration": {
            "required_minimum_samples": arguments.min_samples,
            "wall_seconds": calibration_wall_seconds,
            "attempts": calibration_attempts,
        },
        "timing": {
            "started_at": collection_started_at,
            "finished_at": utc_now(),
            "pilot_wall_seconds": pilot_wall_seconds,
            "repeated_scout_wall_seconds": repeated_pilot_wall_seconds,
            "calibration_wall_seconds": calibration_wall_seconds,
            "sampling_target_wall_seconds": sampling_target_seconds,
            "collector_wall_seconds": total_wall_seconds,
            "successful_target_runs": successful_runs,
            "failed_target_runs": len(all_failures),
            "mean_successful_target_wall_seconds": (
                sampling_target_seconds / successful_runs if successful_runs else None
            ),
        },
        "quality": {
            "counter_coverage": numeric_summary(counter_coverages),
            "record_retention": numeric_summary(record_retentions),
            "sample_count": numeric_summary(sample_counts),
            "minimum_time_running_ratio": min(running_ratios) if running_ratios else None,
            "total_lost_samples": total_lost_samples,
        },
        "kernel": {
            "release": platform.release(),
            "perf_event_max_sample_rate_start": max_sample_rate_start,
            "perf_event_max_sample_rate_end": read_integer(
                "/proc/sys/kernel/perf_event_max_sample_rate"
            ),
            "perf_cpu_time_max_percent_start": cpu_time_percent_start,
            "perf_cpu_time_max_percent_end": read_integer(
                "/proc/sys/kernel/perf_cpu_time_max_percent"
            ),
        },
        "trace_files": [str(path) for path in trace_paths],
        "failures": all_failures,
    }
    manifest_path = (
        manifest_directory / f"collection_{arguments.start}_{last_index}.json"
    )
    write_output(manifest_path, manifest)
    print()
    print(
        f"Collected {successful_runs}/{arguments.total * len(batches) * arguments.runs} "
        f"sampling runs in {format_duration(total_wall_seconds)}"
    )
    print(f"Target execution time: {format_duration(sampling_target_seconds)}")
    if counter_coverages and record_retentions:
        print(
            f"Quality: coverage {100 * min(counter_coverages):.2f}-"
            f"{100 * max(counter_coverages):.2f}%, retention "
            f"{100 * min(record_retentions):.2f}-"
            f"{100 * max(record_retentions):.2f}%, lost={total_lost_samples}"
        )
    print(f"Traces: {trace_directory}")
    print(f"Manifest: {manifest_path}")
    return 0 if successful_runs and not all_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
