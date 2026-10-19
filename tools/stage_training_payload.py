#!/usr/bin/env python3
"""Freeze a native build and its runtime dependencies for test collection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile


LAUNCHER = """#!/usr/bin/env bash

set -euo pipefail
BUNDLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PROGRAM=$(basename -- "$0")
PAYLOAD_DIR="$BUNDLE_DIR/training_payload"
exec "$PAYLOAD_DIR/runtime/ld-linux-x86-64.so.2" \\
  --inhibit-cache \\
  --library-path "$PAYLOAD_DIR/runtime:$PAYLOAD_DIR" \\
  "$PAYLOAD_DIR/bin/$PROGRAM" "$@"
"""


def output(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({' '.join(command)}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linked_files(objects: list[Path]) -> list[Path]:
    paths = set()
    object_paths = {path.resolve() for path in objects}
    interpreter_pattern = re.compile(r"Requesting program interpreter:\s*([^\]]+)")
    absolute_pattern = re.compile(r"(?:=>\s*)?(/\S+)")
    for object_path in objects:
        for match in interpreter_pattern.finditer(output(["readelf", "-l", str(object_path)])):
            paths.add(Path(match.group(1)).resolve())
        for line in output(["ldd", str(object_path)]).splitlines():
            match = absolute_pattern.search(line)
            if match:
                paths.add(Path(match.group(1)).resolve())
    return sorted(path for path in paths if path.is_file() and path not in object_paths)


def program_interpreter(program: Path) -> Path:
    match = re.search(
        r"Requesting program interpreter:\s*([^\]]+)",
        output(["readelf", "-l", str(program)]),
    )
    if not match:
        raise RuntimeError(f"cannot find ELF interpreter for {program}")
    interpreter = Path(match.group(1)).resolve()
    if not interpreter.is_file():
        raise RuntimeError(f"ELF interpreter does not exist: {interpreter}")
    return interpreter


def parse_shared_object_name(dynamic_section: str) -> str | None:
    """Extract DT_SONAME without depending on readelf's descriptive wording."""
    match = re.search(r"\(SONAME\)[^\n]*\[([^\]]+)\]", dynamic_section)
    return match.group(1) if match else None


def shared_object_name(path: Path) -> str | None:
    """Return an ELF library's DT_SONAME, when present."""
    return parse_shared_object_name(output(["readelf", "-d", str(path)]))


def stage_launchers(directory: Path, program_names: list[str]) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".portable-bin-", dir=directory.parent))
    try:
        for name in program_names:
            launcher = temporary / name
            launcher.write_text(LAUNCHER, encoding="utf-8")
            launcher.chmod(0o755)
        if directory.exists():
            shutil.rmtree(directory)
        temporary.replace(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--launcher-dir", type=Path,
        help="optional directory in which to generate one portable launcher per binary",
    )
    arguments = parser.parse_args()
    build = arguments.build_dir.resolve()
    library = build / "libpmc_group.so"
    binary_directory = build / "bin"
    binaries = sorted(
        path for path in binary_directory.iterdir()
        if path.is_file() and os.access(path, os.X_OK)
    ) if binary_directory.is_dir() else []
    missing = [str(path) for path in [library] if not path.is_file()]
    if not binaries:
        missing.append(f"no executable workload binaries under {binary_directory}")
    if missing:
        raise RuntimeError(f"native build is incomplete: {missing}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".training-payload-", dir=arguments.output.parent))
    try:
        (temporary / "bin").mkdir()
        (temporary / "runtime").mkdir()
        shutil.copy2(library, temporary / library.name)
        for binary in binaries:
            shutil.copy2(binary, temporary / "bin" / binary.name)

        runtime_sources = linked_files([library, *binaries])
        runtime_by_name = {}
        for source in runtime_sources:
            names = {source.name}
            soname = shared_object_name(source)
            if soname:
                names.add(soname)
            for name in names:
                previous = runtime_by_name.get(name)
                if previous and sha256(previous) != sha256(source):
                    raise RuntimeError(f"runtime basename collision for {name}")
                runtime_by_name[name] = source
        runtime_by_name["ld-linux-x86-64.so.2"] = program_interpreter(binaries[0])
        for name, source in sorted(runtime_by_name.items()):
            shutil.copy2(source, temporary / "runtime" / name)

        payload_files = [
            path for path in temporary.rglob("*")
            if path.is_file() and path.name not in {"checksums.sha256", "payload.json"}
        ]
        hashes = {
            str(path.relative_to(temporary)): sha256(path)
            for path in sorted(payload_files)
        }
        (temporary / "checksums.sha256").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "immutable training build and runtime for test-mode collection",
            "training_host": platform.node(),
            "architecture": platform.machine(),
            "kernel": platform.release(),
            "compiler": output(["gcc", "--version"]).splitlines()[0],
            "build_directory": str(build),
            "runtime_sources": {name: str(path) for name, path in runtime_by_name.items()},
            "hashes": hashes,
        }
        (temporary / "payload.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if arguments.output.exists():
            shutil.rmtree(arguments.output)
        temporary.replace(arguments.output)
        if arguments.launcher_dir:
            stage_launchers(
                arguments.launcher_dir.resolve(), [binary.name for binary in binaries]
            )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Training payload: {arguments.output}")
    print(f"Runtime files: {', '.join(sorted(runtime_by_name))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
