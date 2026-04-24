#!/usr/bin/env python3
"""Step 3 of landcover verification: score predictions vs remapped GT.

The pipeline writes remapped masks to ``datasets/remapped_landcover_masks`` and
``runmodel/main.py`` writes predictions to ``datasets/pred_masks`` with four
semantic classes (same as ``num_classes=4`` in training/inference). This
script pairs tiles by stem, computes per-class F1/IoU and macro averages, and
writes a CSV. Override ``--pred-dir`` / ``--gt-dir`` if you use different
directories.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from validation.metrics import multiclass_f1_iou  # noqa: E402

# Matches ``runmodel/main.py`` / ``build_model(..., num_classes=4)`` and
# remapped LandCover.ai labels after ``convert_landcover_dataset.py`` (0–3).
DEFAULT_NUM_CLASSES = 4


@dataclass
class Pair:
    tile_id: str
    gt_path: Path
    pred_path: Path


def _datasets_dir() -> Path:
    return Path(__file__).resolve().parent / "datasets"


def _metric_csv_fieldnames(num_classes: int) -> list[str]:
    return (
        ["tile_id", "macro_f1"]
        + [f"f1_class_{c}" for c in range(num_classes)]
        + ["macro_iou"]
        + [f"iou_class_{c}" for c in range(num_classes)]
        + ["gt_path", "pred_path"]
    )


def parse_args() -> argparse.Namespace:
    datasets = _datasets_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Score F1 and IoU between predicted masks and remapped landcover GT "
            "(landcover verification pipeline step 3)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical usage matches run_landcover_verification.sh step 3:\n"
            "  %(prog)s --pred-dir landcover_verification/datasets/pred_masks \\\n"
            "           --gt-dir landcover_verification/datasets/remapped_landcover_masks \\\n"
            "           --out-csv landcover_verification/datasets/landcover_metrics_scores.csv"
        ),
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=datasets / "pred_masks",
        help="Directory of prediction TIFFs from runmodel (default: landcover_verification/datasets/pred_masks)",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=datasets / "remapped_landcover_masks",
        help="Directory of remapped GT masks (default: landcover_verification/datasets/remapped_landcover_masks)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help=(
            f"Number of classes 0..C-1 to score (default: {DEFAULT_NUM_CLASSES}, "
            "matching the verification U-Net and remapped label set)."
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=datasets / "landcover_metrics_scores.csv",
        help="CSV path for per-tile rows plus a __SUMMARY__ row",
    )
    args = parser.parse_args()
    if args.num_classes < 1:
        parser.error("--num-classes must be >= 1")
    return args


def _tifs(directory: Path) -> list[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])


def _lookup(files: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for fp in files:
        out[fp.stem] = fp
    return out


def _load_mask(fp: Path) -> np.ndarray:
    arr = tifffile.imread(fp)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {arr.shape} for {fp}")
    return arr.astype(np.uint8)


def pair_files(gt_dir: Path, pred_dir: Path) -> tuple[list[Pair], list[str]]:
    gt_lookup = _lookup(_tifs(gt_dir))
    pred_lookup = _lookup(_tifs(pred_dir))
    all_ids = sorted(set(gt_lookup) | set(pred_lookup))

    pairs: list[Pair] = []
    skipped: list[str] = []
    for tile_id in all_ids:
        gt = gt_lookup.get(tile_id)
        pred = pred_lookup.get(tile_id)
        if gt is None:
            skipped.append(f"{tile_id}:missing_gt")
            continue
        if pred is None:
            skipped.append(f"{tile_id}:missing_pred")
            continue
        pairs.append(Pair(tile_id=tile_id, gt_path=gt, pred_path=pred))
    return pairs, skipped


def _tile_row(
    pair: Pair,
    macro_f1: float,
    cls_f1: list[float],
    macro_iou: float,
    cls_iou: list[float],
    num_classes: int,
) -> dict[str, str]:
    row: dict[str, str] = {
        "tile_id": pair.tile_id,
        "macro_f1": f"{macro_f1:.6f}",
        "macro_iou": f"{macro_iou:.6f}",
        "gt_path": str(pair.gt_path),
        "pred_path": str(pair.pred_path),
    }
    for c in range(num_classes):
        row[f"f1_class_{c}"] = f"{cls_f1[c]:.6f}"
        row[f"iou_class_{c}"] = f"{cls_iou[c]:.6f}"
    return row


def _summary_row(
    macro_f1_mean: float,
    class_f1_means: list[float],
    macro_iou_mean: float,
    class_iou_means: list[float],
    num_classes: int,
    num_skipped: int,
) -> dict[str, str]:
    row: dict[str, str] = {
        "tile_id": "__SUMMARY__",
        "macro_f1": f"{macro_f1_mean:.6f}",
        "macro_iou": f"{macro_iou_mean:.6f}",
        "gt_path": "",
        "pred_path": f"skipped={num_skipped}",
    }
    for c in range(num_classes):
        row[f"f1_class_{c}"] = f"{class_f1_means[c]:.6f}"
        row[f"iou_class_{c}"] = f"{class_iou_means[c]:.6f}"
    return row


def main() -> int:
    args = parse_args()
    num_classes = args.num_classes
    fieldnames = _metric_csv_fieldnames(num_classes)

    if not args.gt_dir.exists() or not args.gt_dir.is_dir():
        print(f"ERROR: GT directory missing or invalid: {args.gt_dir}", file=sys.stderr)
        return 1
    if not args.pred_dir.exists() or not args.pred_dir.is_dir():
        print(f"ERROR: Pred directory missing or invalid: {args.pred_dir}", file=sys.stderr)
        return 1

    pairs, skipped = pair_files(gt_dir=args.gt_dir, pred_dir=args.pred_dir)
    if not pairs:
        print("ERROR: No matched GT/pred TIFF pairs found.", file=sys.stderr)
        return 1

    per_tile_rows: list[dict[str, str]] = []
    macro_f1_scores: list[float] = []
    macro_iou_scores: list[float] = []
    class_f1_scores: list[list[float]] = [[] for _ in range(num_classes)]
    class_iou_scores: list[list[float]] = [[] for _ in range(num_classes)]

    for pair in pairs:
        gt = _load_mask(pair.gt_path)
        pred = _load_mask(pair.pred_path)
        if gt.shape != pred.shape:
            skipped.append(f"{pair.tile_id}:shape_mismatch_gt{gt.shape}_pred{pred.shape}")
            continue

        macro_f1, cls_f1, macro_iou, cls_iou = multiclass_f1_iou(gt, pred, num_classes)
        macro_f1_scores.append(macro_f1)
        macro_iou_scores.append(macro_iou)
        for idx, score in enumerate(cls_f1):
            class_f1_scores[idx].append(score)
        for idx, score in enumerate(cls_iou):
            class_iou_scores[idx].append(score)

        per_tile_rows.append(_tile_row(pair, macro_f1, cls_f1, macro_iou, cls_iou, num_classes))

    if not macro_f1_scores or not macro_iou_scores:
        print("ERROR: No valid paired masks could be scored.", file=sys.stderr)
        return 1

    macro_f1_mean = float(np.mean(macro_f1_scores))
    macro_iou_mean = float(np.mean(macro_iou_scores))
    class_f1_means = [float(np.mean(scores)) if scores else 0.0 for scores in class_f1_scores]
    class_iou_means = [float(np.mean(scores)) if scores else 0.0 for scores in class_iou_scores]

    print(f"Scored tiles: {len(macro_f1_scores)}")
    print(f"Skipped tiles: {len(skipped)}")
    print(f"Macro F1: {macro_f1_mean:.6f}")
    for idx, score in enumerate(class_f1_means):
        print(f"F1 class {idx}: {score:.6f}")
    print(f"Macro IoU: {macro_iou_mean:.6f}")
    for idx, score in enumerate(class_iou_means):
        print(f"IoU class {idx}: {score:.6f}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_tile_rows:
                writer.writerow(row)
            writer.writerow({})
            writer.writerow(
                _summary_row(
                    macro_f1_mean,
                    class_f1_means,
                    macro_iou_mean,
                    class_iou_means,
                    num_classes,
                    len(skipped),
                )
            )
        print(f"CSV saved: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
