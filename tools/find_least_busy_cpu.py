#!/usr/bin/env python3
"""Select a best-effort least-busy CPU from the caller's affinity mask.

The program samples cumulative per-CPU accounting values from /proc/stat over
an observation interval. By default, a candidate is scored together with all
of its SMT siblings because work on a sibling logical CPU shares physical-core
resources. Selection is only a snapshot; it cannot reserve the chosen CPU.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time


PROC_STAT = Path("/proc/stat")
CPU_SYSFS = Path("/sys/devices/system/cpu")


def positive_float(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number")
    if value <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def nonnegative_int(text):
    try:
        value = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def parse_cpu_list(text):
    """Parse Linux CPU-list syntax such as '0-3,8,10-11'."""
    cpus = set()
    if not text or not text.strip():
        raise ValueError("CPU list is empty")
    for component in text.split(","):
        component = component.strip()
        if not component:
            raise ValueError("CPU list contains an empty component")
        if "-" in component:
            pieces = component.split("-", 1)
            if not pieces[0].isdigit() or not pieces[1].isdigit():
                raise ValueError("invalid CPU range: {0}".format(component))
            first = int(pieces[0], 10)
            last = int(pieces[1], 10)
            if last < first:
                raise ValueError("descending CPU range: {0}".format(component))
            cpus.update(range(first, last + 1))
        else:
            if not component.isdigit():
                raise ValueError("invalid CPU number: {0}".format(component))
            cpus.add(int(component, 10))
    return cpus


def format_cpu_list(cpus):
    values = sorted(cpus)
    if not values:
        return ""
    ranges = []
    first = values[0]
    previous = values[0]
    for cpu in values[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append((first, previous))
        first = previous = cpu
    ranges.append((first, previous))
    return ",".join(
        str(start) if start == end else "{0}-{1}".format(start, end)
        for start, end in ranges
    )


def read_cpu_times(path=PROC_STAT):
    """Return {cpu: (total_ticks, idle_ticks)} from /proc/stat."""
    result = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError("cannot read {0}: {1}".format(path, error))

    for line in lines:
        fields = line.split()
        if not fields or not fields[0].startswith("cpu"):
            continue
        suffix = fields[0][3:]
        if not suffix.isdigit():
            continue
        if len(fields) < 6:
            raise RuntimeError("short CPU accounting line: {0}".format(line))

        # Linux reports user, nice, system, idle, iowait, irq, softirq,
        # steal, guest, and guest_nice. Guest time is already included in user
        # and nice, so only the first eight fields belong in total time.
        try:
            values = [int(value, 10) for value in fields[1:9]]
        except ValueError:
            raise RuntimeError("invalid CPU accounting line: {0}".format(line))
        while len(values) < 8:
            values.append(0)
        total = sum(values)
        idle = values[3] + values[4]
        result[int(suffix, 10)] = (total, idle)

    if not result:
        raise RuntimeError("no per-CPU accounting values found in {0}".format(path))
    return result


def calculate_activity(before, after):
    """Return busy/total deltas and busy fractions for CPUs in both samples."""
    activity = {}
    for cpu in set(before).intersection(after):
        total_delta = after[cpu][0] - before[cpu][0]
        idle_delta = after[cpu][1] - before[cpu][1]
        if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
            continue
        busy_delta = total_delta - idle_delta
        activity[cpu] = {
            "total_delta": total_delta,
            "busy_delta": busy_delta,
            "busy_fraction": busy_delta / float(total_delta),
        }
    return activity


def read_text_if_present(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def cpu_topology(cpu, sysfs_root=CPU_SYSFS):
    topology = sysfs_root / ("cpu{0}".format(cpu)) / "topology"
    siblings_text = read_text_if_present(topology / "thread_siblings_list")
    try:
        siblings = parse_cpu_list(siblings_text) if siblings_text else {cpu}
    except ValueError:
        siblings = {cpu}
    return {
        "siblings": siblings,
        "package_id": read_text_if_present(topology / "physical_package_id"),
        "core_id": read_text_if_present(topology / "core_id"),
        "core_type": read_text_if_present(topology / "core_type"),
    }


def rank_cpus(activity, candidates, topology_by_cpu, account_smt=True):
    """Rank candidate logical CPUs by core score, logical score, then CPU ID."""
    rankings = []
    for cpu in sorted(candidates):
        if cpu not in activity:
            continue
        topology = topology_by_cpu.get(cpu, {"siblings": {cpu}})
        siblings = set(topology.get("siblings", {cpu}))
        measured_siblings = sorted(siblings.intersection(activity))
        if not account_smt:
            measured_siblings = [cpu]

        total_delta = sum(activity[item]["total_delta"] for item in measured_siblings)
        busy_delta = sum(activity[item]["busy_delta"] for item in measured_siblings)
        score = busy_delta / float(total_delta) if total_delta else 1.0
        rankings.append(
            {
                "cpu": cpu,
                "busy_fraction": activity[cpu]["busy_fraction"],
                "score": score,
                "siblings": sorted(siblings),
                "package_id": topology.get("package_id"),
                "core_id": topology.get("core_id"),
                "core_type": topology.get("core_type"),
            }
        )

    rankings.sort(key=lambda item: (item["score"], item["busy_fraction"], item["cpu"]))
    return rankings


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Observe Linux per-CPU accounting and select a best-effort "
            "least-busy CPU from the caller's allowed affinity mask."
        )
    )
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=1.0,
        help="observation interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--cpu-list",
        help="restrict candidates using Linux CPU-list syntax, for example 2-7,10",
    )
    parser.add_argument(
        "--logical-only",
        action="store_true",
        help="score each logical CPU alone instead of including SMT siblings",
    )
    parser.add_argument(
        "--top",
        type=nonnegative_int,
        default=10,
        help="number of ranked candidates to display; 0 displays all (default: 10)",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--cpu-only",
        action="store_true",
        help="print only the selected CPU number for use by scripts",
    )
    output.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON",
    )
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    if not hasattr(os, "sched_getaffinity"):
        print("error: os.sched_getaffinity is unavailable; this tool requires Linux", file=sys.stderr)
        return 1

    allowed = set(os.sched_getaffinity(0))
    candidates = set(allowed)
    if arguments.cpu_list:
        try:
            requested = parse_cpu_list(arguments.cpu_list)
        except ValueError as error:
            print("error: {0}".format(error), file=sys.stderr)
            return 2
        candidates.intersection_update(requested)
    if not candidates:
        print("error: no requested CPUs are in the allowed affinity mask", file=sys.stderr)
        return 1

    try:
        before = read_cpu_times()
        started = time.monotonic()
        time.sleep(arguments.interval)
        after = read_cpu_times()
        elapsed = time.monotonic() - started
        activity = calculate_activity(before, after)
        topology = {cpu: cpu_topology(cpu) for cpu in candidates}
        rankings = rank_cpus(
            activity,
            candidates,
            topology,
            account_smt=not arguments.logical_only,
        )
    except (OSError, RuntimeError) as error:
        print("error: {0}".format(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    if not rankings:
        print("error: no candidate CPU had valid accounting data", file=sys.stderr)
        return 1

    selected = rankings[0]["cpu"]
    if arguments.cpu_only:
        print(selected)
        return 0

    if arguments.json:
        document = {
            "selected_cpu": selected,
            "observation_seconds": elapsed,
            "allowed_cpus": sorted(allowed),
            "candidate_cpus": sorted(candidates),
            "smt_aware": not arguments.logical_only,
            "rankings": [
                {
                    "cpu": item["cpu"],
                    "busy_percent": 100.0 * item["busy_fraction"],
                    "score_percent": 100.0 * item["score"],
                    "siblings": item["siblings"],
                    "package_id": item["package_id"],
                    "core_id": item["core_id"],
                    "core_type": item["core_type"],
                }
                for item in rankings
            ],
        }
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0

    print("Observation: {0:.3f}s".format(elapsed))
    print("Allowed CPUs: {0}".format(format_cpu_list(allowed)))
    print("SMT-aware:    {0}".format("yes" if not arguments.logical_only else "no"))
    print()
    print(" CPU   busy%   score%   package/core   SMT siblings")
    print("----  -------  -------  -------------  ------------")
    visible = rankings if arguments.top == 0 else rankings[: arguments.top]
    for item in visible:
        package = item["package_id"] if item["package_id"] is not None else "?"
        core = item["core_id"] if item["core_id"] is not None else "?"
        print(
            "{0:4d}  {1:7.2f}  {2:7.2f}  {3:>6}/{4:<6}  {5}".format(
                item["cpu"],
                100.0 * item["busy_fraction"],
                100.0 * item["score"],
                package,
                core,
                format_cpu_list(item["siblings"]),
            )
        )
    print()
    print("Selected CPU: {0}".format(selected))
    print("This is a snapshot, not a CPU reservation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
