#!/usr/bin/env python3
"""Prepare external traces for a frozen Transformer model."""

import argparse
from pathlib import Path
import sys

MODELS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS.parent))
from models.common.external_dataset import prepare_padded_external  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-root", type=Path, required=True)
parser.add_argument("--reference-data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--event-subset", type=Path)
parser.add_argument("--input-ids")
parser.add_argument("--programs")
parser.add_argument("--include-invalid", action="store_true")
args = parser.parse_args()
metadata = prepare_padded_external(
    args.input_root, args.reference_data, args.output, args.event_subset,
    tuple(args.input_ids.split(",")) if args.input_ids else None,
    tuple(args.programs.split(",")) if args.programs else None,
    args.include_invalid,
)
print(f"Wrote {metadata['examples']} external examples to {args.output}")
print(f"Skipped measurements={metadata['skipped_measurements']}, incomplete bundles={metadata['skipped_incomplete_bundles']}")
