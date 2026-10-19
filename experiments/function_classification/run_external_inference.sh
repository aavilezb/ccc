#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
INPUT_ROOT="/mnt/linuxstorage/testing_data_cross-os"
RUN_NAME="latitude_ubuntu"
DEVICE="auto"
FULL_TRAINING_NAME="baseline_test5"
DETERMINISTIC_TRAINING_NAME="baseline_deterministic_test5"
INPUT_IDS=""
PROGRAMS=""
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage: ./run_external_inference.sh [options]

Evaluate all frozen full-event and deterministic-event classifiers on an
external trace tree. No model or preprocessing state is fitted.

  --input-root DIR   External trace root
  --name NAME        Output identifier (default: latitude_ubuntu)
  --device DEVICE    auto, cpu, or cuda (default: auto)
  --full-training-name NAME
                     Full-event training artifact name (default: baseline_test5)
  --deterministic-training-name NAME
                     Deterministic training artifact name
                     (default: baseline_deterministic_test5)
  --input-ids IDS    Comma-separated external input IDs to include
  --programs NAMES   Comma-separated external executable names to include
  --overwrite        Replace existing external-inference artifacts
  -h, --help         Show help
EOF
}

while (($#)); do
  case "$1" in
    --input-root) INPUT_ROOT=$2; shift 2 ;;
    --name) RUN_NAME=$2; shift 2 ;;
    --device) DEVICE=$2; shift 2 ;;
    --full-training-name) FULL_TRAINING_NAME=$2; shift 2 ;;
    --deterministic-training-name) DETERMINISTIC_TRAINING_NAME=$2; shift 2 ;;
    --input-ids) INPUT_IDS=$2; shift 2 ;;
    --programs) PROGRAMS=$2; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -d $INPUT_ROOT ]] || { printf 'input root does not exist: %s\n' "$INPUT_ROOT" >&2; exit 2; }
[[ $RUN_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { printf 'invalid name\n' >&2; exit 2; }
[[ $FULL_TRAINING_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { printf 'invalid full training name\n' >&2; exit 2; }
[[ $DETERMINISTIC_TRAINING_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { printf 'invalid deterministic training name\n' >&2; exit 2; }

BENCHMARK_DIR="$SCRIPT_DIR/models/benchmarks/external/$RUN_NAME"
SUMMARY="$BENCHMARK_DIR/summary.csv"
mkdir -p "$BENCHMARK_DIR"
exec 9>"$BENCHMARK_DIR/.run.lock"
flock -n 9 || { printf 'external benchmark is already running: %s\n' "$RUN_NAME" >&2; exit 2; }
if [[ -e $SUMMARY && $OVERWRITE -eq 0 ]]; then
  printf 'result already exists; use --overwrite or another --name\n' >&2
  exit 2
fi
printf 'variant,model,status,examples,accuracy,macro_f1,preprocess_seconds,inference_seconds,result_directory\n' >"$SUMMARY"

input_options=()
if [[ -n $INPUT_IDS ]]; then
  input_options+=(--input-ids "$INPUT_IDS")
fi
if [[ -n $PROGRAMS ]]; then
  input_options+=(--programs "$PROGRAMS")
fi

for variant in full deterministic; do
  if [[ $variant == full ]]; then
    training_name="$FULL_TRAINING_NAME"
    subset_options=()
  else
    training_name="$DETERMINISTIC_TRAINING_NAME"
    subset_options=(--event-subset "$SCRIPT_DIR/../../events/pmc_group_events_deterministic.csv")
  fi
  for model in inception_time transformer minirocket; do
    external_data="$SCRIPT_DIR/models/$model/external_data/${RUN_NAME}_${variant}.npz"
    result_directory="$SCRIPT_DIR/models/$model/external_results/${RUN_NAME}_${variant}"
    reference_data="$SCRIPT_DIR/models/$model/data/$training_name.npz"
    checkpoint_directory="$SCRIPT_DIR/models/$model/results/$training_name"
    if ((OVERWRITE)); then
      mkdir -p "$result_directory"
    fi
    printf '\n[%s/%s] preprocessing external traces\n' "$variant" "$model"
    preprocess_start=$(date +%s.%N)
    python3 -u "$SCRIPT_DIR/models/$model/preprocess_external.py" \
      --input-root "$INPUT_ROOT" --reference-data "$reference_data" \
      --output "$external_data" "${subset_options[@]}" "${input_options[@]}"
    preprocess_finish=$(date +%s.%N)
    printf '[%s/%s] frozen inference\n' "$variant" "$model"
    inference_start=$(date +%s.%N)
    if [[ $model == inception_time ]]; then
      python3 -u "$SCRIPT_DIR/models/$model/infer.py" \
        --data "$external_data" --checkpoint "$checkpoint_directory/model.pt" \
        --output-dir "$result_directory" --device "$DEVICE"
    elif [[ $model == transformer ]]; then
      python3 -u "$SCRIPT_DIR/models/$model/infer.py" \
        --data "$external_data" --checkpoint "$checkpoint_directory/model.pt" \
        --output-dir "$result_directory" --device "$DEVICE"
    else
      python3 -u "$SCRIPT_DIR/models/$model/infer.py" \
        --data "$external_data" --transform "$checkpoint_directory/transform.npz" \
        --classifier "$checkpoint_directory/classifier.joblib" \
        --output-dir "$result_directory" --device "$DEVICE"
    fi
    inference_finish=$(date +%s.%N)
    values=$(python3 -c '
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["external_dataset"]["examples"], d["accuracy"], d["macro_f1"], sep=",")
' "$result_directory/metrics.json")
    IFS=, read -r examples accuracy macro_f1 <<<"$values"
    preprocess_seconds=$(awk -v a="$preprocess_start" -v b="$preprocess_finish" 'BEGIN {printf "%.6f",b-a}')
    inference_seconds=$(awk -v a="$inference_start" -v b="$inference_finish" 'BEGIN {printf "%.6f",b-a}')
    printf '%s,%s,success,%s,%s,%s,%s,%s,%s\n' \
      "$variant" "$model" "$examples" "$accuracy" "$macro_f1" \
      "$preprocess_seconds" "$inference_seconds" "$result_directory" >>"$SUMMARY"
  done
done
printf '\nSummary: %s\n' "$SUMMARY"
