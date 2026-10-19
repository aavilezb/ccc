#!/usr/bin/env bash

set -euo pipefail

BUNDLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MACHINE_ID=""
CLASS_NAME="baseline"
EVENTS="$BUNDLE_DIR/events/pmc_group_events_full.csv"
COLLECTION_MODE=""
BINARY_MODE=""
ARCHIVE=0
ALLOW_PMU_MISMATCH=0
PREFLIGHT_ONLY=0
CPU_AFFINITY=""
CONTEXT_MIXER="none"
COLLECTION_ARGUMENTS=()

usage() {
  cat <<'EOF'
Usage: ./collect.sh [bundle options] [collection options]

Build/check the portable workload bundle and collect every program/input with
one command. Both modes default to 40 trace files x 10 runs per class.

Bundle options:
  --mode train             Rebuild natively and freeze a portable test payload
  --mode test              Never rebuild; use the frozen training payload
  --machine-id ID          Host/domain identifier (default: short hostname)
  --class-name NAME        Model label namespace (default: baseline)
  --events full            Collect all events (default; supports later subsets)
  --events deterministic   Collect only deterministic events
  --events FILE            Use another adaptive-format event CSV
  --rebuild                Deprecated alias for --mode train
  --archive                Archive results; train mode also packages trainer and test collector
  --allow-pmu-mismatch     Allow a non-Intel CPU (raw encodings may be invalid)
  --preflight-only         Check compatibility and PMU access, then stop
  -h, --help               Show this help

Useful collection options passed through to the campaign runner include:
  --runs N                 Runs per event group (bundle default: 10)
  --total N                Trace files per class (bundle default: 40)
  --start N                First trace index (default: 1)
  --lengths N,N,...        Restrict input lengths
  --programs NAME,...      Restrict programs
  --cpu N                  Pin every target execution to logical CPU N
  --context-mixer NAME     none or mixed_v1 (default: none)
  --min-samples N          Required retained-sample floor (default: 20)
  --min-period N           Hard period floor (default: 1)
  --max-repetitions N      Cap automatic target-function repetitions
  --calibration-runs N     Floor-calibration runs per PMU group (default: 3)
  --timeout SECONDS        Per-execution timeout
  --overwrite              Replace requested trace indices
  --dry-run                Print commands without PMU collection
EOF
}

require_value() {
  if (($# < 2)); then
    printf 'missing value for %s\n' "$1" >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --mode)
      require_value "$@"
      COLLECTION_MODE=$2
      shift 2
      ;;
    --machine-id)
      require_value "$@"
      MACHINE_ID=$2
      shift 2
      ;;
    --class-name)
      require_value "$@"
      CLASS_NAME=$2
      shift 2
      ;;
    --events)
      require_value "$@"
      case "$2" in
        full) EVENTS="$BUNDLE_DIR/events/pmc_group_events_full.csv" ;;
        deterministic) EVENTS="$BUNDLE_DIR/events/pmc_group_events_deterministic.csv" ;;
        *) EVENTS=$2 ;;
      esac
      shift 2
      ;;
    --rebuild)
      COLLECTION_MODE="train"
      shift
      ;;
    --archive)
      ARCHIVE=1
      shift
      ;;
    --allow-pmu-mismatch)
      ALLOW_PMU_MISMATCH=1
      shift
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
      shift
      ;;
    --cpu)
      require_value "$@"
      CPU_AFFINITY=$2
      COLLECTION_ARGUMENTS+=("$1" "$2")
      shift 2
      ;;
    --context-mixer)
      require_value "$@"
      CONTEXT_MIXER=$2
      COLLECTION_ARGUMENTS+=("$1" "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      COLLECTION_ARGUMENTS+=("$1")
      shift
      ;;
  esac
done

if [[ $COLLECTION_MODE != train && $COLLECTION_MODE != test ]]; then
  printf 'choose exactly one collection mode with --mode train or --mode test\n' >&2
  exit 2
fi

if [[ $(uname -s) != Linux ]]; then
  printf 'this collector requires Linux perf_event_open; macOS and Windows are unsupported\n' >&2
  exit 2
fi
if [[ $(uname -m) != x86_64 ]]; then
  printf 'this event set and the bundled exact binaries require x86_64\n' >&2
  exit 2
fi
if ! command -v python3 >/dev/null; then
  printf 'python3 is required\n' >&2
  exit 2
fi
python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 8))' || {
  printf 'Python 3.8 or newer is required\n' >&2
  exit 2
}

if [[ -z $MACHINE_ID ]]; then
  MACHINE_ID=$(hostname -s 2>/dev/null || uname -n)
fi
if [[ ! $MACHINE_ID =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'machine ID must contain only letters, digits, period, underscore, or hyphen\n' >&2
  exit 2
fi
if [[ ! $CLASS_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'class name must contain only letters, digits, period, underscore, or hyphen\n' >&2
  exit 2
fi
if [[ -n $CPU_AFFINITY && ! $CPU_AFFINITY =~ ^[0-9]+$ ]]; then
  printf 'CPU affinity must be a nonnegative integer\n' >&2
  exit 2
fi
if [[ $CONTEXT_MIXER != none && $CONTEXT_MIXER != mixed_v1 ]]; then
  printf 'context mixer must be none or mixed_v1\n' >&2
  exit 2
fi
if [[ ! -f $EVENTS ]]; then
  printf 'event CSV does not exist: %s\n' "$EVENTS" >&2
  exit 2
fi

vendor=$(awk '/^vendor_id/{print $3; exit}' /proc/cpuinfo)
if [[ $vendor != GenuineIntel && $ALLOW_PMU_MISMATCH -eq 0 ]]; then
  printf 'CPU vendor is %s; the raw event encodings target Intel PMUs\n' "${vendor:-unknown}" >&2
  printf 'use --allow-pmu-mismatch only with a verified replacement event CSV\n' >&2
  exit 2
fi

PAYLOAD_DIR="$BUNDLE_DIR/training_payload"
if [[ $COLLECTION_MODE == train ]]; then
  BINARY_MODE="native-training-build"
  command -v make >/dev/null && command -v gcc >/dev/null || {
    printf 'make and gcc are required in train mode\n' >&2
    exit 2
  }
  command -v readelf >/dev/null && command -v ldd >/dev/null || {
    printf 'readelf and ldd are required to freeze the training payload\n' >&2
    exit 2
  }
  make -C "$BUNDLE_DIR" clean
  make -C "$BUNDLE_DIR" rebuilt
  BIN_DIR="$BUNDLE_DIR/rebuilt/bin"
  python3 "$BUNDLE_DIR/tools/stage_training_payload.py" \
    --build-dir "$BUNDLE_DIR/rebuilt" --output "$PAYLOAD_DIR" \
    --launcher-dir "$BUNDLE_DIR/portable_bin"
else
  BINARY_MODE="frozen-training-payload"
  if [[ ! -f $PAYLOAD_DIR/checksums.sha256 ]]; then
    printf 'training payload is missing; run --mode train on the training host first\n' >&2
    exit 2
  fi
  (cd "$PAYLOAD_DIR" && sha256sum --check --status checksums.sha256) || {
    printf 'frozen training payload checksum verification failed\n' >&2
    exit 2
  }
  BIN_DIR="$BUNDLE_DIR/portable_bin"
fi

dry_run=0
for argument in "${COLLECTION_ARGUMENTS[@]}"; do
  [[ $argument == --dry-run ]] && dry_run=1
done

printf 'Machine ID:    %s\n' "$MACHINE_ID"
printf 'Class name:    %s\n' "$CLASS_NAME"
printf 'Collector mode:%s\n' " $COLLECTION_MODE"
printf 'Binary mode:   %s\n' "$BINARY_MODE"
printf 'Events:        %s\n' "$EVENTS"
printf 'CPU vendor:    %s\n' "${vendor:-unknown}"
printf 'CPU affinity:  %s\n' "${CPU_AFFINITY:-none}"
printf 'Context mixer: %s\n' "$CONTEXT_MIXER"

if ((dry_run == 0)); then
  scratch=$(mktemp -d)
  cleanup() { rm -rf -- "$scratch"; }
  trap cleanup EXIT
  mapfile -t preflight_groups < <(python3 -c '
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from event_support import member_batches, read_event_configuration
c = read_event_configuration(Path(sys.argv[2]))
for batch in member_batches(c.members):
    print(",".join(str(event.index) for event in batch))
' "$BUNDLE_DIR/tools" "$EVENTS")
  printf 'Preflight:     %s binaries and %d PMU group(s) ...\n' \
    "$COLLECTION_MODE" "${#preflight_groups[@]}"
  for preflight_index in "${!preflight_groups[@]}"; do
    preflight_output="$scratch/preflight-$preflight_index.json"
    PMC_EVENTS_CSV="$EVENTS" \
    PMC_MEMBER_INDICES="${preflight_groups[$preflight_index]}" \
    PMC_RUN_MODE=scout \
    PMC_CPU_AFFINITY="$CPU_AFFINITY" \
    PMC_CONTEXT_MIXER="$CONTEXT_MIXER" \
    PMC_OUTPUT_FILE="$preflight_output" \
      "$BIN_DIR/example_group" 10000
    python3 -c '
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
m=d["measurements"][0]
assert m["run_mode"] == "scout" and m["leader_total_count"] > 0
assert m["context_mixer"] == sys.argv[3]
expected_cpu = None if sys.argv[4] == "" else int(sys.argv[4])
assert m["cpu_affinity"] == expected_cpu
print("  group", sys.argv[2] + ":", "count=" + str(m["leader_total_count"]))
' "$preflight_output" "$((preflight_index + 1))" "$CONTEXT_MIXER" "$CPU_AFFINITY"
  done
fi

OUTPUT_ROOT="$BUNDLE_DIR/results/$MACHINE_ID"
METADATA="$OUTPUT_ROOT/machine.json"
python3 "$BUNDLE_DIR/tools/capture_machine.py" \
  --output "$METADATA" \
  --machine-id "$MACHINE_ID" \
  --class-name "$CLASS_NAME" \
  --events "$EVENTS" \
  --binary-dir "$BIN_DIR" \
  --binary-mode "$BINARY_MODE" \
  --payload-root "$PAYLOAD_DIR" \
  --collection-mode "$COLLECTION_MODE" \
  --collection-arguments "${COLLECTION_ARGUMENTS[@]}"

if ((PREFLIGHT_ONLY)); then
  printf 'Preflight completed; no collection requested.\n'
  exit 0
fi

"$BUNDLE_DIR/tools/collect_all.sh" \
  --name "$CLASS_NAME" \
  --runs 10 \
  --total 40 \
  --start 1 \
  --bin-dir "$BIN_DIR" \
  --events "$EVENTS" \
  --output-root "$OUTPUT_ROOT" \
  "${COLLECTION_ARGUMENTS[@]}"

if ((ARCHIVE)); then
  mkdir -p "$BUNDLE_DIR/exports"
  archive="$BUNDLE_DIR/exports/${MACHINE_ID}.tar.gz"
  tar -czf "$archive" -C "$BUNDLE_DIR/results" "$MACHINE_ID"
  printf 'Archive:       %s\n' "$archive"
  if [[ $COLLECTION_MODE == train ]]; then
    trainer_archive="$BUNDLE_DIR/exports/${MACHINE_ID}_trainer.tar.gz"
    tar \
      --exclude='./results' --exclude='./exports' --exclude='./rebuilt' \
      --exclude='./training_payload' --exclude='./__pycache__' \
      --exclude='*/__pycache__' --exclude='*.pyc' \
      -czf "$trainer_archive" -C "$BUNDLE_DIR" .
    printf 'Trainer:      %s\n' " $trainer_archive"

    collector_archive="$BUNDLE_DIR/exports/${MACHINE_ID}_test_collector.tar.gz"
    tar \
      --exclude='./results' --exclude='./exports' --exclude='./rebuilt' \
      --exclude='./__pycache__' --exclude='*/__pycache__' --exclude='*.pyc' \
      -czf "$collector_archive" -C "$BUNDLE_DIR" .
    printf 'Test collector:%s\n' " $collector_archive"
  fi
fi
printf 'Bring back:    %s\n' "$OUTPUT_ROOT"
