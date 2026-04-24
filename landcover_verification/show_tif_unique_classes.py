#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show unique class values present in one TIFF mask or a directory of TIFF masks."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to one .tif/.tiff mask file or a directory containing TIFF masks",
    )
    return parser.parse_args()


def _read_unique_counts(tif_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not tif_path.exists():
        print(f"ERROR: File does not exist: {tif_path}", file=sys.stderr)
        raise SystemExit(1)
    if not tif_path.is_file():
        print(f"ERROR: Not a file: {tif_path}", file=sys.stderr)
        raise SystemExit(1)
    if tif_path.suffix.lower() not in {".tif", ".tiff"}:
        print(f"ERROR: Expected a .tif or .tiff file, got: {tif_path}", file=sys.stderr)
        raise SystemExit(1)

    arr = tifffile.imread(tif_path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        print(f"ERROR: Expected 2D mask, got shape {arr.shape} for {tif_path}", file=sys.stderr)
        raise SystemExit(1)

    return np.unique(arr, return_counts=True)


def _print_unique_counts(tif_path: Path) -> None:
    values, counts = _read_unique_counts(tif_path)

    print(f"File: {tif_path}")
    print(f"Unique classes ({len(values)}): {values.tolist()}")
    print("Counts:")
    for value, count in zip(values.tolist(), counts.tolist()):
        print(f"  class {int(value)}: {int(count)}")
    print()


def _collect_tifs(directory: Path) -> list[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])


def main() -> int:
    args = parse_args()
    input_path: Path = args.input_path

    if not input_path.exists():
        print(f"ERROR: Path does not exist: {input_path}", file=sys.stderr)
        return 1

    if input_path.is_file():
        _print_unique_counts(input_path)
        return 0

    if not input_path.is_dir():
        print(f"ERROR: Path is neither a file nor a directory: {input_path}", file=sys.stderr)
        return 1

    tif_paths = _collect_tifs(input_path)
    if not tif_paths:
        print(f"ERROR: No .tif or .tiff files found in directory: {input_path}", file=sys.stderr)
        return 1

    for tif_path in tif_paths:
        _print_unique_counts(tif_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
