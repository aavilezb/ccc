#!/usr/bin/env bash

set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_ROOT="$ROOT/experiments/function_classification/models/minirocket"
EVENT_SUBSET="$ROOT/events/pmc_group_events_deterministic.csv"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./automation/run_minirocket_ab.sh LOCAL_NONE LOCAL_MIXED REMOTE_NONE REMOTE_MIXED [EXPERIMENT_ID] [DEVICE]

Each dataset argument is the directory containing the program subdirectories,
normally a .../baseline directory. DEVICE defaults to cuda. This reproduces
the deterministic-event MiniRocket A/B evaluation for length 1. Because these
aggressively sampled traces fail the normal coverage/retention gate, the script
uses --include-invalid explicitly for both training and external inference.
EOF
}

if (($# < 4 || $# > 6)); then
  usage >&2
  exit 2
fi

local_none=$1
local_mixed=$2
remote_none=$3
remote_mixed=$4
experiment_id=${5:-context_mixer_replay}
device=${6:-cuda}

if [[ ! $experiment_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'EXPERIMENT_ID must contain only letters, digits, period, underscore, or hyphen\n' >&2
  exit 2
fi

for directory in "$local_none" "$local_mixed" "$remote_none" "$remote_mixed"; do
  if [[ ! -d $directory ]]; then
    printf 'dataset directory does not exist: %s\n' "$directory" >&2
    exit 2
  fi
done

if [[ $device == cuda* ]]; then
  python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    printf 'CUDA is unavailable to this process. Run on the host, not inside a GPU-hidden sandbox.\n' >&2
    exit 2
  }
fi

data_dir="$MODEL_ROOT/data/$experiment_id"
result_dir="$MODEL_ROOT/results/$experiment_id"
external_dir="$MODEL_ROOT/external_results/$experiment_id"
mkdir -p "$data_dir" "$result_dir" "$external_dir"

preprocess_training() {
  local input_root=$1
  local output=$2
  python3 "$MODEL_ROOT/preprocess.py" \
    --input-root "$input_root" \
    --output "$output" \
    --validation-units 5 \
    --test-units 5 \
    --split-unit trace \
    --seed 2026 \
    --include-invalid \
    --event-subset "$EVENT_SUBSET" \
    --input-ids 1 \
    --length 128
}

train_model() {
  local data=$1
  local output=$2
  python3 "$MODEL_ROOT/train.py" \
    --data "$data" \
    --output-dir "$output" \
    --features 10000 \
    --seed 2026 \
    --device "$device"
}

prepare_external() {
  local input_root=$1
  local reference=$2
  local output=$3
  python3 "$MODEL_ROOT/preprocess_external.py" \
    --input-root "$input_root" \
    --reference-data "$reference" \
    --output "$output" \
    --event-subset "$EVENT_SUBSET" \
    --input-ids 1 \
    --include-invalid
}

infer_external() {
  local data=$1
  local model=$2
  local output=$3
  python3 "$MODEL_ROOT/infer.py" \
    --data "$data" \
    --transform "$model/transform.npz" \
    --classifier "$model/classifier.joblib" \
    --output-dir "$output" \
    --device "$device"
}

preprocess_training "$local_none" "$data_dir/control_train.npz"
preprocess_training "$local_mixed" "$data_dir/mixed_train.npz"
train_model "$data_dir/control_train.npz" "$result_dir/control"
train_model "$data_dir/mixed_train.npz" "$result_dir/mixed"
prepare_external "$remote_none" "$data_dir/control_train.npz" "$data_dir/control_external.npz"
prepare_external "$remote_mixed" "$data_dir/mixed_train.npz" "$data_dir/mixed_external.npz"
infer_external "$data_dir/control_external.npz" "$result_dir/control" "$external_dir/control"
infer_external "$data_dir/mixed_external.npz" "$result_dir/mixed" "$external_dir/mixed"

python3 - "$external_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for condition in ("control", "mixed"):
    metrics = json.loads((root / condition / "metrics.json").read_text(encoding="utf-8"))
    print(
        f"{condition}: accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
PY
