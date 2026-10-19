#!/usr/bin/env python3
"""Capture collection-host provenance without external Python packages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError:
        return None
    output = result.stdout.strip()
    return output if result.returncode == 0 and output else None


def cpu_fields() -> dict[str, str]:
    fields: dict[str, str] = {}
    text = read_text(Path("/proc/cpuinfo")) or ""
    wanted = {"vendor_id", "cpu family", "model", "model name", "stepping", "microcode"}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in wanted and key not in fields:
            fields[key] = value
    return fields


def os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (read_text(Path("/etc/os-release")) or "").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument(
        "--binary-mode",
        choices=("native-training-build", "frozen-training-payload"),
        required=True,
    )
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--collection-mode", choices=("train", "test"), required=True)
    parser.add_argument("--collection-arguments", nargs=argparse.REMAINDER, default=[])
    arguments = parser.parse_args()

    binaries = {}
    for path in sorted(arguments.binary_dir.glob("*")):
        if path.is_file() and os.access(path, os.X_OK):
            binaries[path.name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    library = (
        arguments.payload_root / "libpmc_group.so"
        if arguments.collection_mode == "test"
        else arguments.binary_dir.parent / "libpmc_group.so"
    )
    exact_payloads = {}
    for path in sorted((arguments.payload_root / "bin").glob("*")):
        if path.is_file():
            exact_payloads[path.name] = sha256(path)
    runtime = {}
    for path in sorted((arguments.payload_root / "runtime").glob("*")):
        if path.is_file():
            runtime[path.name] = sha256(path)
    document = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "machine_id": arguments.machine_id,
        "class_name": arguments.class_name,
        "collection_mode": arguments.collection_mode,
        "binary_mode": arguments.binary_mode,
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "kernel": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "os_release": os_release(),
        "cpu": cpu_fields(),
        "lscpu": command_output(["lscpu"]),
        "python": platform.python_version(),
        "compiler": command_output(["gcc", "--version"]),
        "perf_limits": {
            "perf_event_paranoid": read_text(Path("/proc/sys/kernel/perf_event_paranoid")),
            "perf_event_max_sample_rate": read_text(Path("/proc/sys/kernel/perf_event_max_sample_rate")),
            "perf_cpu_time_max_percent": read_text(Path("/proc/sys/kernel/perf_cpu_time_max_percent")),
        },
        "events": {
            "path": str(arguments.events.resolve()),
            "sha256": sha256(arguments.events),
        },
        "binaries": binaries,
        "exact_elf_payloads": exact_payloads,
        "bundled_runtime": runtime,
        "training_payload_manifest": (
            json.loads((arguments.payload_root / "payload.json").read_text(encoding="utf-8"))
            if (arguments.payload_root / "payload.json").is_file() else None
        ),
        "library": (
            {"path": str(library.resolve()), "sha256": sha256(library)}
            if library.is_file() else None
        ),
        "collection_arguments": arguments.collection_arguments,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=arguments.output.parent,
        prefix=f".{arguments.output.name}.", delete=False,
    ) as temporary:
        json.dump(document, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(arguments.output)
    print(f"Machine metadata: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
