#!/usr/bin/env bash

set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
Usage: ./automation/collect_inference_ubuntu18.sh CONDITION MACHINE_ID [CPU] [extra collect.sh arguments]

Run this from the unpacked test collector on Latitude. CONDITION is none or
mixed_v1. The defaults reproduce the inference collection: 5 trace files per
class, 10 runs per event group, full events, length 1, and CPU 0.

Examples:
  ./automation/collect_inference_ubuntu18.sh none latitude-affinity0-none-length1 0
  ./automation/collect_inference_ubuntu18.sh mixed_v1 latitude-affinity0-mixed-v1-length1 0
EOF
}

if (($# < 2)); then
  usage >&2
  exit 2
fi

condition=$1
machine_id=$2
cpu=${3:-0}
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
  --mode test \
  --machine-id "$machine_id" \
  --class-name baseline \
  --events full \
  --lengths 1 \
  --runs 10 \
  --total 5 \
  --cpu "$cpu" \
  --context-mixer "$condition" \
  "$@"
