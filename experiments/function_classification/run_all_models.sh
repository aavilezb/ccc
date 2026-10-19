#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
INPUT_ROOT="$SCRIPT_DIR/outputs/baseline"
RUN_NAME="baseline_test5"
TEST_UNITS=5
VALIDATION_UNITS=5
SEED=2026
EPOCHS=30
PATIENCE=6
DEVICE="auto"
MINIROCKET_FEATURES=10000
EVENT_SUBSET=""
INPUT_IDS=""
PROGRAMS=""
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage: ./run_all_models.sh [options]

Preprocess, train, and evaluate InceptionTime, transformer, and MiniROCKET with
an identical trace-held-out split. Five trace files per class are held out for
testing by default, and five different trace files are used for validation.

Options:
  --input-root DIR          Collected class root (default: outputs/baseline)
  --name NAME               Dataset/result identifier (default: baseline_test5)
  --test-units N            Test trace files per class (default: 5)
  --validation-units N      Validation trace files per class (default: 5)
  --seed N                  Split/model seed (default: 2026)
  --epochs N                Neural-network epoch limit (default: 30)
  --patience N              Neural early-stopping patience (default: 6)
  --device DEVICE           auto, cpu, or cuda (default: auto)
  --minirocket-features N   MiniROCKET feature count (default: 10000)
  --event-subset CSV        Train with only member indices listed in this event CSV
  --input-ids IDS           Comma-separated input IDs to include
  --programs NAMES          Comma-separated executable names to include
  --overwrite               Replace existing artifacts for this name
  -h, --help                Show this help
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

elapsed_seconds() {
  awk -v start="$1" -v finish="$2" 'BEGIN { printf "%.6f", finish - start }'
}

while (($#)); do
  case "$1" in
    --input-root)
      require_value "$@"
      INPUT_ROOT=$2
      shift 2
      ;;
    --name)
      require_value "$@"
      RUN_NAME=$2
      shift 2
      ;;
    --test-units)
      require_value "$@"
      TEST_UNITS=$2
      shift 2
      ;;
    --validation-units)
      require_value "$@"
      VALIDATION_UNITS=$2
      shift 2
      ;;
    --seed)
      require_value "$@"
      SEED=$2
      shift 2
      ;;
    --epochs)
      require_value "$@"
      EPOCHS=$2
      shift 2
      ;;
    --patience)
      require_value "$@"
      PATIENCE=$2
      shift 2
      ;;
    --device)
      require_value "$@"
      DEVICE=$2
      shift 2
      ;;
    --minirocket-features)
      require_value "$@"
      MINIROCKET_FEATURES=$2
      shift 2
      ;;
    --event-subset)
      require_value "$@"
      EVENT_SUBSET=$2
      shift 2
      ;;
    --input-ids)
      require_value "$@"
      INPUT_IDS=$2
      shift 2
      ;;
    --programs)
      require_value "$@"
      PROGRAMS=$2
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
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

if [[ ! $RUN_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'name must contain only letters, digits, period, underscore, or hyphen\n' >&2
  exit 2
fi
for value in "$TEST_UNITS" "$VALIDATION_UNITS" "$EPOCHS" "$PATIENCE" "$MINIROCKET_FEATURES"; do
  if ! is_positive_integer "$value"; then
    printf 'split counts, epochs, patience, and feature count must be positive integers\n' >&2
    exit 2
  fi
done
if [[ ! -d $INPUT_ROOT ]]; then
  printf 'input root does not exist: %s\n' "$INPUT_ROOT" >&2
  exit 2
fi
if [[ -n $EVENT_SUBSET && ! -f $EVENT_SUBSET ]]; then
  printf 'event subset does not exist: %s\n' "$EVENT_SUBSET" >&2
  exit 2
fi

BENCHMARK_DIR="$SCRIPT_DIR/models/benchmarks/$RUN_NAME"
LOG_DIR="$BENCHMARK_DIR/logs"
SUMMARY="$BENCHMARK_DIR/summary.csv"
mkdir -p -- "$BENCHMARK_DIR"
exec 9>"$BENCHMARK_DIR/.run.lock"
if ! flock -n 9; then
  printf 'benchmark is already running for name %s\n' "$RUN_NAME" >&2
  exit 2
fi
if [[ -e $SUMMARY && $OVERWRITE -eq 0 ]]; then
  printf 'benchmark already exists: %s; use --overwrite or another --name\n' "$SUMMARY" >&2
  exit 2
fi
mkdir -p -- "$LOG_DIR" "$BENCHMARK_DIR/matplotlib"
export MPLCONFIGDIR="$BENCHMARK_DIR/matplotlib"
csv_row model status preprocess_seconds train_seconds total_seconds accuracy macro_f1 \
  data result_directory preprocess_log train_log >"$SUMMARY"

COMMON_PREPROCESS=(
  --input-root "$INPUT_ROOT"
  --validation-units "$VALIDATION_UNITS"
  --test-units "$TEST_UNITS"
  --split-unit trace
  --seed "$SEED"
)
if [[ -n $EVENT_SUBSET ]]; then
  COMMON_PREPROCESS+=(--event-subset "$EVENT_SUBSET")
fi
if [[ -n $INPUT_IDS ]]; then
  COMMON_PREPROCESS+=(--input-ids "$INPUT_IDS")
fi
if [[ -n $PROGRAMS ]]; then
  COMMON_PREPROCESS+=(--programs "$PROGRAMS")
fi

campaign_started=$(date +%s.%N)
overall_status=0
for model in inception_time transformer minirocket; do
  data="$SCRIPT_DIR/models/$model/data/$RUN_NAME.npz"
  result_directory="$SCRIPT_DIR/models/$model/results/$RUN_NAME"
  preprocess_log="$LOG_DIR/${model}_preprocess.log"
  train_log="$LOG_DIR/${model}_train.log"
  mkdir -p -- "$(dirname -- "$data")" "$result_directory"

  case "$model" in
    inception_time)
      preprocess_command=(
        python3 -u "$SCRIPT_DIR/models/inception_time/preprocess.py"
        "${COMMON_PREPROCESS[@]}" --output "$data" --length 128
      )
      train_command=(
        python3 -u "$SCRIPT_DIR/models/inception_time/train.py"
        --data "$data" --output-dir "$result_directory"
        --epochs "$EPOCHS" --patience "$PATIENCE" --batch-size 128
        --filters 8 --depth 2 --seed "$SEED" --device "$DEVICE"
      )
      ;;
    transformer)
      preprocess_command=(
        python3 -u "$SCRIPT_DIR/models/transformer/preprocess.py"
        "${COMMON_PREPROCESS[@]}" --output "$data" --max-length 128
      )
      train_command=(
        python3 -u "$SCRIPT_DIR/models/transformer/train.py"
        --data "$data" --output-dir "$result_directory"
        --epochs "$EPOCHS" --patience "$PATIENCE" --batch-size 128
        --dimension 32 --heads 4 --layers 2 --feedforward-dimension 64
        --patch-size 4 --seed "$SEED" --device "$DEVICE"
      )
      ;;
    minirocket)
      preprocess_command=(
        python3 -u "$SCRIPT_DIR/models/minirocket/preprocess.py"
        "${COMMON_PREPROCESS[@]}" --output "$data" --length 128
      )
      train_command=(
        python3 -u "$SCRIPT_DIR/models/minirocket/train.py"
        --data "$data" --output-dir "$result_directory"
        --features "$MINIROCKET_FEATURES" --batch-size 64
        --seed "$SEED" --device "$DEVICE"
      )
      ;;
  esac

  printf '\n[%s] preprocessing\n' "$model"
  model_started=$(date +%s.%N)
  preprocess_started=$model_started
  "${preprocess_command[@]}" 2>&1 | tee "$preprocess_log"
  preprocess_status=${PIPESTATUS[0]}
  preprocess_finished=$(date +%s.%N)
  preprocess_seconds=$(elapsed_seconds "$preprocess_started" "$preprocess_finished")
  train_seconds=""
  accuracy=""
  macro_f1=""

  if ((preprocess_status == 0)); then
    printf '[%s] training\n' "$model"
    train_started=$(date +%s.%N)
    "${train_command[@]}" 2>&1 | tee "$train_log"
    train_status=${PIPESTATUS[0]}
    train_finished=$(date +%s.%N)
    train_seconds=$(elapsed_seconds "$train_started" "$train_finished")
  else
    train_status=1
    printf 'training skipped because preprocessing failed\n' >"$train_log"
  fi

  model_finished=$(date +%s.%N)
  total_seconds=$(elapsed_seconds "$model_started" "$model_finished")
  if ((preprocess_status == 0 && train_status == 0)); then
    status="success"
    metrics=$(python3 -c '
import json
import sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d["accuracy"], d["macro_f1"], sep="\t")
' "$result_directory/metrics.json")
    IFS=$'\t' read -r accuracy macro_f1 <<<"$metrics"
  else
    status="failed"
    overall_status=1
  fi
  csv_row "$model" "$status" "$preprocess_seconds" "$train_seconds" \
    "$total_seconds" "$accuracy" "$macro_f1" "$data" "$result_directory" \
    "$preprocess_log" "$train_log" >>"$SUMMARY"
  printf '[%s] status=%s total=%ss accuracy=%s macro_f1=%s\n' \
    "$model" "$status" "$total_seconds" "${accuracy:-n/a}" "${macro_f1:-n/a}"
done

campaign_finished=$(date +%s.%N)
campaign_seconds=$(elapsed_seconds "$campaign_started" "$campaign_finished")
printf '\nAll-model run finished in %ss\n' "$campaign_seconds"
printf 'Summary: %s\n' "$SUMMARY"
exit "$overall_status"
