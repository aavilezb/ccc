#!/usr/bin/env bash

set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
Usage: ./automation/collect_training_ubuntu22.sh CONDITION MACHINE_ID [CPU] [extra collect.sh arguments]

CONDITION is either none or mixed_v1. The defaults reproduce the length-1
training collection: 40 trace files per class, 10 runs per event group, full
event collection, and CPU 7 when CPU is omitted.

Examples:
  ./automation/collect_training_ubuntu22.sh none desktop-affinity7-none-length1 7
  ./automation/collect_training_ubuntu22.sh mixed_v1 desktop-affinity7-mixed-v1-length1 7
EOF
}

if (($# < 2)); then
  usage >&2
  exit 2
fi

condition=$1
machine_id=$2
cpu=${3:-7}
if (($# >= 3)); then
  shift 3
else
  shift 2
fi

if [[ $condition != none && $condition != mixed_v1 ]]; then
  printf 'CONDITION must be none or mixed_v1\n' >&2
  exit 2
fi
if [[ ! $cpu =~ ^[0-9]+$ ]]; then
  printf 'CPU must be a nonnegative logical CPU number\n' >&2
  exit 2
fi

exec "$ROOT/collect.sh" \
  --mode train \
  --machine-id "$machine_id" \
  --class-name baseline \
  --events full \
  --lengths 1 \
  --runs 10 \
  --total 40 \
  --cpu "$cpu" \
  --context-mixer "$condition" \
  "$@"
