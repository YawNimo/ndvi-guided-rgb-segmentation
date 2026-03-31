from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
from PIL import Image


@dataclass
class PairingResult:
    tile_id: str
    gt_path: Path
    pred_path: Path


@dataclass
class SkippedItem:
    tile_id: str
    reason: str


def _tif_files(directory: Path) -> list[Path]:
    return sorted([p for p in directory.glob("*.tif") if p.is_file()])


def _build_lookup(files: list[Path]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for fp in files:
        stem = fp.stem
        lookup.setdefault(stem, fp)
        if stem.endswith("_mask"):
            lookup.setdefault(stem[:-5], fp)
    return lookup


def pair_ground_truth_and_predictions(gt_dir: Path, pred_dir: Path) -> tuple[list[PairingResult], list[SkippedItem]]:
    gt_files = _tif_files(gt_dir)
    pred_files = _tif_files(pred_dir)

    gt_lookup = _build_lookup(gt_files)
    pred_lookup = _build_lookup(pred_files)

    all_tile_ids = sorted(set(gt_lookup.keys()) | set(pred_lookup.keys()))

    pairs: list[PairingResult] = []
    skipped: list[SkippedItem] = []

    for tile_id in all_tile_ids:
        gt_fp = gt_lookup.get(tile_id)
        pred_fp = pred_lookup.get(tile_id)
        if gt_fp is None:
            skipped.append(SkippedItem(tile_id=tile_id, reason="missing_ground_truth"))
            continue
        if pred_fp is None:
            skipped.append(SkippedItem(tile_id=tile_id, reason="missing_prediction"))
            continue
        pairs.append(PairingResult(tile_id=tile_id, gt_path=gt_fp, pred_path=pred_fp))

    return pairs, skipped


def load_mask(mask_path: Path) -> np.ndarray:
    arr = np.array(Image.open(mask_path), dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape} for {mask_path}")
    return arr.astype(np.uint8)


def write_validation_csv(
    out_csv: Path,
    rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    skipped_rows: list[dict[str, str]],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tile_id",
        "status",
        "macro_dice",
        "dice_class_0_water",
        "dice_class_1_impervious",
        "dice_class_2_sparse_veg",
        "dice_class_3_dense_veg",
        "ground_truth_path",
        "prediction_path",
        "reason",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

        writer.writerow({})

        for row in summary_rows:
            writer.writerow(row)

        writer.writerow({})

        for row in skipped_rows:
            writer.writerow(row)
