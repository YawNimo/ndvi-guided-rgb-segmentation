#!/usr/bin/env python3
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


@dataclass
class Pair:
    tile_id: str
    gt_path: Path
    pred_path: Path


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Score F1 and IoU between remapped prediction and GT masks.")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=base_dir / "datasets" / "remapped_pred_masks",
        help="Directory containing remapped prediction masks",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=base_dir / "datasets" / "remapped_landcover_masks",
        help="Directory containing remapped ground-truth masks",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of classes to score (default: 3 for labels 0,1,2)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=base_dir / "datasets" / "landcover_metrics_scores.csv",
        help="CSV output path for per-tile and summary metrics",
    )
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
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
    class_f1_scores: list[list[float]] = [[] for _ in range(args.num_classes)]
    class_iou_scores: list[list[float]] = [[] for _ in range(args.num_classes)]

    for pair in pairs:
        gt = _load_mask(pair.gt_path)
        pred = _load_mask(pair.pred_path)
        if gt.shape != pred.shape:
            skipped.append(f"{pair.tile_id}:shape_mismatch_gt{gt.shape}_pred{pred.shape}")
            continue

        macro_f1, cls_f1, macro_iou, cls_iou = multiclass_f1_iou(gt, pred, args.num_classes)
        macro_f1_scores.append(macro_f1)
        macro_iou_scores.append(macro_iou)
        for idx, score in enumerate(cls_f1):
            class_f1_scores[idx].append(score)
        for idx, score in enumerate(cls_iou):
            class_iou_scores[idx].append(score)

        row = {
            "tile_id": pair.tile_id,
            "macro_f1": f"{macro_f1:.6f}",
            "f1_class_0": f"{cls_f1[0]:.6f}" if len(cls_f1) > 0 else "",
            "f1_class_1": f"{cls_f1[1]:.6f}" if len(cls_f1) > 1 else "",
            "f1_class_2": f"{cls_f1[2]:.6f}" if len(cls_f1) > 2 else "",
            "macro_iou": f"{macro_iou:.6f}",
            "iou_class_0": f"{cls_iou[0]:.6f}" if len(cls_iou) > 0 else "",
            "iou_class_1": f"{cls_iou[1]:.6f}" if len(cls_iou) > 1 else "",
            "iou_class_2": f"{cls_iou[2]:.6f}" if len(cls_iou) > 2 else "",
            "gt_path": str(pair.gt_path),
            "pred_path": str(pair.pred_path),
        }
        per_tile_rows.append(row)

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
            fieldnames = [
                "tile_id",
                "macro_f1",
                "f1_class_0",
                "f1_class_1",
                "f1_class_2",
                "macro_iou",
                "iou_class_0",
                "iou_class_1",
                "iou_class_2",
                "gt_path",
                "pred_path",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_tile_rows:
                writer.writerow(row)
            writer.writerow({})
            writer.writerow(
                {
                    "tile_id": "__SUMMARY__",
                    "macro_f1": f"{macro_f1_mean:.6f}",
                    "f1_class_0": f"{class_f1_means[0]:.6f}" if len(class_f1_means) > 0 else "",
                    "f1_class_1": f"{class_f1_means[1]:.6f}" if len(class_f1_means) > 1 else "",
                    "f1_class_2": f"{class_f1_means[2]:.6f}" if len(class_f1_means) > 2 else "",
                    "macro_iou": f"{macro_iou_mean:.6f}",
                    "iou_class_0": f"{class_iou_means[0]:.6f}" if len(class_iou_means) > 0 else "",
                    "iou_class_1": f"{class_iou_means[1]:.6f}" if len(class_iou_means) > 1 else "",
                    "iou_class_2": f"{class_iou_means[2]:.6f}" if len(class_iou_means) > 2 else "",
                    "gt_path": "",
                    "pred_path": f"skipped={len(skipped)}",
                }
            )
        print(f"CSV saved: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
