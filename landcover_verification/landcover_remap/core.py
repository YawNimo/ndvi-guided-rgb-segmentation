from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import tifffile

from remap_constants import LANDCOVER_CLASS_MAPPING

REMAP: dict[int, int] = LANDCOVER_CLASS_MAPPING
EXPECTED_INPUT_LABELS: set[int] = {0, 1, 2, 3, 4}
EXPECTED_OUTPUT_LABELS: set[int] = set(LANDCOVER_CLASS_MAPPING.values())


def collect_tif_files(input_dir: Path) -> list[Path]:
    """Return deterministically sorted .tif/.tiff files from input_dir."""
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    return sorted(files)


def _load_mask(mask_path: Path) -> np.ndarray:
    """Load a mask and normalize it to a 2D uint8 array."""
    arr = tifffile.imread(mask_path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape} for {mask_path}")
    return np.asarray(arr, dtype=np.uint8)


def scan_label_stats(mask_paths: list[Path]) -> Counter[int]:
    """Aggregate label counts across all mask files."""
    label_counts: Counter[int] = Counter()
    for mask_path in mask_paths:
        arr = _load_mask(mask_path)
        values, counts = np.unique(arr, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            label_counts[int(value)] += int(count)
    return label_counts


def validate_labels(found_labels: set[int], expected_labels: set[int], allow_extra_labels: bool) -> tuple[set[int], set[int]]:
    """Return (missing_labels, unexpected_labels)."""
    missing_labels = expected_labels - found_labels
    unexpected_labels = found_labels - expected_labels
    if unexpected_labels and not allow_extra_labels:
        raise ValueError(
            "Unexpected labels found: "
            f"{sorted(unexpected_labels)}; expected subset of {sorted(expected_labels)}. "
            "Use --allow-extra-labels to bypass."
        )
    return missing_labels, unexpected_labels


def summarize_counts(label_counts: Counter[int]) -> str:
    """Format label counts as compact text for CLI output."""
    if not label_counts:
        return "<none>"
    parts = [f"{label}:{label_counts[label]}" for label in sorted(label_counts)]
    return ", ".join(parts)


def remap_and_write_masks_with_mapping(
    mask_paths: list[Path], output_dir: Path, mapping: dict[int, int]
) -> int:
    """Remap labels with a caller-provided mapping and save masks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lut = np.arange(256, dtype=np.uint8)
    for src_label, dst_label in mapping.items():
        lut[src_label] = dst_label

    processed = 0
    for mask_path in mask_paths:
        arr = _load_mask(mask_path)
        remapped = lut[arr]
        out_path = output_dir / mask_path.name
        tifffile.imwrite(out_path, remapped)
        processed += 1
    return processed


def remap_and_write_masks(mask_paths: list[Path], output_dir: Path) -> int:
    """Remap labels with default landcover mapping and save masks."""
    return remap_and_write_masks_with_mapping(mask_paths, output_dir, REMAP)
