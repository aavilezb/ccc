#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_DIR="$SCRIPT_DIR/.."
COLLECTOR="$SCRIPT_DIR/collect_pmc_groups.py"

BIN_DIR="$BUNDLE_DIR/bin"
EVENTS="$BUNDLE_DIR/events/pmc_group_events_full.csv"
OUTPUT_ROOT="$BUNDLE_DIR/results"
NAME="all_functions"
RUNS=10
TOTAL=1
START=1
TIMEOUT=30
LENGTHS=(100 1000 10000 100000 1000000 5000000 20000000)
PROGRAM_NAMES=()
OVERWRITE=0
DRY_RUN=0
COLLECTOR_OPTIONS=()

usage() {
  cat <<'EOF'
Usage: ./collect_all.sh [options]

Run adaptive PMU collection for every executable in the bundle's bin/ and
every configured input length. A timestamped timing CSV and one log per
program/length combination are written under outputs/<name>/campaigns/.

Options:
  --name NAME                    Collection namespace (default: all_functions)
  --runs N                      Runs per event group (default: 10)
  --total N                     Trace files per combination (default: 1)
  --start N                     First trace-file index (default: 1)
  --lengths N,N,...             Input lengths (default: adaptive sweep lengths)
  --programs NAME,NAME,...      Collect only these binary names (default: all)
  --bin-dir DIR                 Directory containing instrumented binaries
  --events FILE                 PMU event CSV
  --output-root DIR             Root output directory
  --timeout SECONDS             Timeout for one target execution (default: 30)
  --cpu N                       Pin every target execution to logical CPU N
  --context-mixer NAME          none or mixed_v1 (default: none)
  --member-indices I,I,...      Restrict member events
  --ring-buffer-pages N         Pass through to the collector
  --base-samples N              Pass through to the collector
  --reference-instructions N    Pass through to the collector
  --min-samples N               Pass through to the collector
  --max-samples N               Pass through to the collector
  --min-period N                Pass through to the collector
  --max-repetitions N           Cap automatic target-function repetitions
  --calibration-runs N          Sample-floor calibration runs per PMU group
  --max-calibration-attempts N  Maximum period-reduction attempts
  --overwrite                   Replace requested existing trace indices
  --dry-run                     Print every collector command without running it
  -h, --help                    Show this help
EOF
}

require_value() {
  if (($# < 2)); then
    printf 'missing value for %s\n' "$1" >&2
    exit 2
  fi
}

is_positive_integer() {
  [[ $1 =~ ^[1-9][0-9]*$ ]]
}

csv_row() {
  local separator=""
  local value
  for value in "$@"; do
    value=${value//\"/\"\"}
    printf '%s"%s"' "$separator" "$value"
    separator=","
  done
  printf '\n'
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

while (($#)); do
  case "$1" in
    --name)
      require_value "$@"
      NAME=$2
      shift 2
      ;;
    --runs)
      require_value "$@"
      RUNS=$2
      shift 2
      ;;
    --total)
      require_value "$@"
      TOTAL=$2
      shift 2
      ;;
    --start)
      require_value "$@"
      START=$2
      shift 2
      ;;
    --lengths)
      require_value "$@"
      IFS=',' read -r -a LENGTHS <<<"$2"
      shift 2
      ;;
    --programs)
      require_value "$@"
      IFS=',' read -r -a PROGRAM_NAMES <<<"$2"
      shift 2
      ;;
    --bin-dir)
      require_value "$@"
      BIN_DIR=$2
      shift 2
      ;;
    --events)
      require_value "$@"
      EVENTS=$2
      shift 2
      ;;
    --output-root)
      require_value "$@"
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --timeout)
      require_value "$@"
      TIMEOUT=$2
      shift 2
      ;;
    --cpu|--context-mixer|--member-indices|--ring-buffer-pages|--base-samples|--reference-instructions|--min-samples|--max-samples|--min-period|--max-repetitions|--calibration-runs|--max-calibration-attempts)
      require_value "$@"
      COLLECTOR_OPTIONS+=("$1" "$2")
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for value in "$RUNS" "$TOTAL" "$START"; do
  if ! is_positive_integer "$value"; then
    printf 'runs, total, and start must be positive integers\n' >&2
    exit 2
  fi
done
if [[ ! $NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'name must contain only letters, digits, period, underscore, or hyphen\n' >&2
  exit 2
fi
if ((${#LENGTHS[@]} == 0)); then
  printf 'at least one length is required\n' >&2
  exit 2
fi
for length in "${LENGTHS[@]}"; do
  if ! is_positive_integer "$length"; then
    printf 'invalid positive integer in --lengths: %s\n' "$length" >&2
    exit 2
  fi
done
if [[ ! -f $COLLECTOR ]]; then
  printf 'collector does not exist: %s\n' "$COLLECTOR" >&2
  exit 2
fi
if [[ ! -f $EVENTS ]]; then
  printf 'event CSV does not exist: %s\n' "$EVENTS" >&2
  exit 2
fi
if [[ ! -d $BIN_DIR ]]; then
  printf 'binary directory does not exist: %s\n' "$BIN_DIR" >&2
  exit 2
fi

BINARIES=()
while IFS= read -r -d '' candidate; do
  [[ -f $candidate && -x $candidate ]] || continue
  if ((${#PROGRAM_NAMES[@]})); then
    selected=0
    for program_name in "${PROGRAM_NAMES[@]}"; do
      if [[ $(basename -- "$candidate") == "$program_name" ]]; then
        selected=1
        break
      fi
    done
    ((selected)) || continue
  fi
  BINARIES+=("$candidate")
done < <(find "$BIN_DIR" -maxdepth 1 -type f -perm -u+x -print0 | sort -z)

if ((${#BINARIES[@]} == 0)); then
  printf 'no selected executable binaries found under %s\n' "$BIN_DIR" >&2
  exit 2
fi
if ((${#PROGRAM_NAMES[@]})); then
  for program_name in "${PROGRAM_NAMES[@]}"; do
    found=0
    for binary in "${BINARIES[@]}"; do
      if [[ $(basename -- "$binary") == "$program_name" ]]; then
        found=1
        break
      fi
    done
    if ((!found)); then
      printf 'selected program is not an executable in %s: %s\n' "$BIN_DIR" "$program_name" >&2
      exit 2
    fi
  done
fi

COMMON_OPTIONS=(
  --events "$EVENTS"
  --runs "$RUNS"
  --total "$TOTAL"
  --start "$START"
  --name "$NAME"
  --timeout "$TIMEOUT"
  --output-root "$OUTPUT_ROOT"
  "${COLLECTOR_OPTIONS[@]}"
)
if ((OVERWRITE)); then
  COMMON_OPTIONS+=(--overwrite)
fi

COMBINATIONS=$((${#BINARIES[@]} * ${#LENGTHS[@]}))
printf 'Programs:     %s\n' "${#BINARIES[@]}"
printf 'Lengths:      %s\n' "${LENGTHS[*]}"
printf 'Combinations: %s\n' "$COMBINATIONS"
printf 'Runs/combo:   %s trace file(s) x %s run(s) per event group\n' "$TOTAL" "$RUNS"

if ((DRY_RUN)); then
  combination=0
  for binary in "${BINARIES[@]}"; do
    for length in "${LENGTHS[@]}"; do
      ((combination += 1))
      printf '[%d/%d] %s + input %s\n' \
        "$combination" "$COMBINATIONS" "$(basename -- "$binary")" "$length"
      printf -v target_command '%q %q' "$binary" "$length"
      command=(python3 "$COLLECTOR" --target "$target_command" "${COMMON_OPTIONS[@]}")
      print_command "${command[@]}"
    done
  done
  exit 0
fi

CAMPAIGN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
CAMPAIGN_DIR="$OUTPUT_ROOT/$NAME/campaigns/$CAMPAIGN_STAMP"
LOG_DIR="$CAMPAIGN_DIR/logs"
TIMING_CSV="$CAMPAIGN_DIR/combo_times.csv"
mkdir -p -- "$LOG_DIR"
csv_row \
  program input started_at finished_at wall_seconds status exit_code \
  golden_count target_samples sample_period pilot_seconds \
  sampling_target_seconds collector_seconds mean_target_seconds \
  successful_target_runs failed_target_runs output_directory manifest log \
  >"$TIMING_CSV"

campaign_started=$(date +%s.%N)
combination=0
campaign_status=0
last_index=$((START + TOTAL - 1))

for binary in "${BINARIES[@]}"; do
  program=$(basename -- "$binary")
  for length in "${LENGTHS[@]}"; do
    ((combination += 1))
    printf '\n[%d/%d] %s + input %s\n' \
      "$combination" "$COMBINATIONS" "$program" "$length"
    printf -v target_command '%q %q' "$binary" "$length"
    command=(python3 "$COLLECTOR" --target "$target_command" "${COMMON_OPTIONS[@]}")
    log_path="$LOG_DIR/${program}__${length}.log"
    output_directory="$OUTPUT_ROOT/$NAME/$program/$length"
    manifest="$output_directory/manifests/collection_${START}_${last_index}.json"
    started_at=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
    combo_started=$(date +%s.%N)

    "${command[@]}" 2>&1 | tee "$log_path"
    exit_code=${PIPESTATUS[0]}

    combo_finished=$(date +%s.%N)
    finished_at=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
    wall_seconds=$(awk -v start="$combo_started" -v finish="$combo_finished" \
      'BEGIN { printf "%.6f", finish - start }')
    if ((exit_code == 0)); then
      status="success"
    else
      status="failed"
      campaign_status=1
    fi

    golden_count=""
    target_samples=""
    sample_period=""
    pilot_seconds=""
    sampling_target_seconds=""
    collector_seconds=""
    mean_target_seconds=""
    successful_target_runs=""
    failed_target_runs=""
    if [[ -f $manifest ]]; then
      metrics=$(python3 -c '
import json
import sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
values = (
    d["pilot"]["golden_count"],
    d["pilot"]["target_samples"],
    d["pilot"]["chosen_period"],
    d["timing"]["pilot_wall_seconds"],
    d["timing"]["sampling_target_wall_seconds"],
    d["timing"]["collector_wall_seconds"],
    d["timing"]["mean_successful_target_wall_seconds"],
    d["timing"]["successful_target_runs"],
    d["timing"]["failed_target_runs"],
)
print("\t".join("" if value is None else str(value) for value in values))
' "$manifest")
      IFS=$'\t' read -r \
        golden_count target_samples sample_period pilot_seconds \
        sampling_target_seconds collector_seconds mean_target_seconds \
        successful_target_runs failed_target_runs <<<"$metrics"
    fi

    csv_row \
      "$program" "$length" "$started_at" "$finished_at" "$wall_seconds" \
      "$status" "$exit_code" "$golden_count" "$target_samples" \
      "$sample_period" "$pilot_seconds" "$sampling_target_seconds" \
      "$collector_seconds" "$mean_target_seconds" "$successful_target_runs" \
      "$failed_target_runs" "$output_directory" "$manifest" "$log_path" \
      >>"$TIMING_CSV"
    printf 'Combination time: %ss (%s)\n' "$wall_seconds" "$status"
  done
done

campaign_finished=$(date +%s.%N)
campaign_seconds=$(awk -v start="$campaign_started" -v finish="$campaign_finished" \
  'BEGIN { printf "%.3f", finish - start }')
printf '\nCampaign finished in %ss\n' "$campaign_seconds"
printf 'Timing CSV: %s\n' "$TIMING_CSV"
printf 'Logs:       %s\n' "$LOG_DIR"
exit "$campaign_status"
