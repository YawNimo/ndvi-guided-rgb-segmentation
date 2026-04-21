#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from landcover_remap import (
    collect_tif_files,
    remap_and_write_masks_with_mapping,
    scan_label_stats,
    summarize_counts,
    validate_labels,
)

PRED_REMAP: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 2}
PRED_EXPECTED_INPUT_LABELS: set[int] = {0, 1, 2, 3}
PRED_EXPECTED_OUTPUT_LABELS: set[int] = {0, 1, 2}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "results" / "unet" / "pred_masks"
    default_output = repo_root / "landcover_verification" / "datasets" / "remapped_pred_masks"

    parser = argparse.ArgumentParser(
        description="Validate and remap prediction mask labels."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Input mask directory (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Output remapped mask directory (default: {default_output})",
    )
    parser.add_argument(
        "--allow-extra-labels",
        action="store_true",
        help="Allow unexpected labels during preflight validation.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only scan/validate labels without writing remapped files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"ERROR: Input path is not a directory: {input_dir}", file=sys.stderr)
        return 1

    mask_paths = collect_tif_files(input_dir)
    if not mask_paths:
        print(f"ERROR: No TIFF masks found in {input_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(mask_paths)} mask files in {input_dir}")
    input_counts = scan_label_stats(mask_paths)
    input_labels = set(input_counts.keys())
    print(f"Preflight labels: {sorted(input_labels)}")
    print(f"Preflight counts: {summarize_counts(input_counts)}")

    missing_labels, unexpected_labels = validate_labels(
        found_labels=input_labels,
        expected_labels=PRED_EXPECTED_INPUT_LABELS,
        allow_extra_labels=args.allow_extra_labels,
    )
    if missing_labels:
        print(f"Warning: expected labels not observed: {sorted(missing_labels)}")
    if unexpected_labels:
        print(f"Warning: unexpected labels observed: {sorted(unexpected_labels)}")

    if args.preflight_only:
        print("Preflight complete. No files written (--preflight-only).")
        return 0

    processed = remap_and_write_masks_with_mapping(mask_paths, output_dir, PRED_REMAP)
    print(f"Wrote {processed} remapped masks to {output_dir}")

    output_paths = collect_tif_files(output_dir)
    output_counts = scan_label_stats(output_paths)
    output_labels = set(output_counts.keys())
    print(f"Output labels: {sorted(output_labels)}")
    print(f"Output counts: {summarize_counts(output_counts)}")

    invalid_output_labels = output_labels - PRED_EXPECTED_OUTPUT_LABELS
    if invalid_output_labels:
        print(
            "ERROR: Invalid output labels found: "
            f"{sorted(invalid_output_labels)}; expected subset of {sorted(PRED_EXPECTED_OUTPUT_LABELS)}",
            file=sys.stderr,
        )
        return 1

    print("Conversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
