#!/usr/bin/env bash

set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:-/tmp/libpmc_test_collector.tar.gz}
output_dir=$(cd -- "$(dirname -- "$output")" && pwd)
output="$output_dir/$(basename -- "$output")"

case "$output" in
  "$ROOT"|"$ROOT"/*)
    printf 'write the archive outside the source tree (for example, /tmp/libpmc_test_collector.tar.gz)\n' >&2
    exit 2
    ;;
esac

for required in \
  "$ROOT/training_payload/checksums.sha256" \
  "$ROOT/training_payload/runtime/ld-linux-x86-64.so.2" \
  "$ROOT/training_payload/runtime/libc.so.6" \
  "$ROOT/portable_bin/example_group"; do
  if [[ ! -f $required ]]; then
    printf 'missing staged payload file: %s\n' "$required" >&2
    printf 'run a successful --mode train collection first\n' >&2
    exit 2
  fi
done

tar \
  --exclude='./results' \
  --exclude='./exports' \
  --exclude='./rebuilt' \
  --exclude='./experiments/function_classification/models/*/data' \
  --exclude='./experiments/function_classification/models/*/results' \
  --exclude='./experiments/function_classification/models/*/external_results' \
  --exclude='./__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -czf "$output" -C "$ROOT" .

tar -tzf "$output" >/dev/null
printf 'Test collector: %s\n' "$output"
