#!/usr/bin/env python3
"""Create confusion matrices from landcover verification metrics CSV files.

This script scans CSV files (default: landcover_metrics_scores*.csv) under
``landcover_verification/datasets`` and reconstructs each model's confusion
matrix from the per-row ``gt_path`` / ``pred_path`` TIFF mask pairs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.constants import CLASS_NAMES  # noqa: E402
from validation.metrics import confusion_matrix_update  # noqa: E402
from validation.visualization import plot_confusion_matrix  # noqa: E402


def _default_datasets_dir() -> Path:
    return Path(__file__).resolve().parent / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create confusion matrix plots from verification metrics CSV files."
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=_default_datasets_dir(),
        help="Directory containing landcover_metrics_scores*.csv files.",
    )
    parser.add_argument(
        "--csv-glob",
        type=str,
        default="landcover_metrics_scores*.csv",
        help="Glob pattern used inside --datasets-dir to select CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for confusion matrix artifacts (default: <datasets-dir>/confusion_matrices).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of classes expected in GT/pred masks.",
    )
    return parser.parse_args()


def _load_mask(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask at {path}, got shape {arr.shape}")
    return arr.astype(np.int64, copy=False)


def _validate_classes(mask: np.ndarray, num_classes: int, label: str, path: Path) -> None:
    min_v = int(mask.min())
    max_v = int(mask.max())
    if min_v < 0 or max_v >= num_classes:
        raise ValueError(
            f"{label} mask values out of range at {path}: min={min_v}, max={max_v}, "
            f"expected [0, {num_classes - 1}]"
        )


def _iter_csv_pairs(csv_path: Path) -> list[tuple[str, Path, Path]]:
    rows: list[tuple[str, Path, Path]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"tile_id", "gt_path", "pred_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        for row in reader:
            tile_id = (row.get("tile_id") or "").strip()
            gt_str = (row.get("gt_path") or "").strip()
            pred_str = (row.get("pred_path") or "").strip()
            if not tile_id or tile_id == "__SUMMARY__":
                continue
            if not gt_str or not pred_str:
                continue
            rows.append((tile_id, Path(gt_str), Path(pred_str)))
    return rows


def _save_matrix_csv(out_path: Path, conf_mat: np.ndarray, class_names: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred", *class_names])
        for i, row in enumerate(conf_mat):
            writer.writerow([class_names[i], *row.tolist()])


def _csv_stem_label(csv_path: Path) -> str:
    stem = csv_path.stem
    prefix = "landcover_metrics_scores_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def process_csv(
    csv_path: Path,
    out_dir: Path,
    num_classes: int,
    class_names: list[str],
) -> tuple[int, int]:
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    pairs = _iter_csv_pairs(csv_path)

    used = 0
    skipped = 0
    for tile_id, gt_path, pred_path in pairs:
        try:
            if not gt_path.exists() or not pred_path.exists():
                raise FileNotFoundError(f"missing file(s): gt={gt_path.exists()}, pred={pred_path.exists()}")

            gt = _load_mask(gt_path)
            pred = _load_mask(pred_path)
            if gt.shape != pred.shape:
                raise ValueError(f"shape mismatch gt={gt.shape}, pred={pred.shape}")

            _validate_classes(gt, num_classes, "gt", gt_path)
            _validate_classes(pred, num_classes, "pred", pred_path)
            confusion_matrix_update(conf_mat, gt, pred, num_classes)
            used += 1
        except Exception as exc:
            skipped += 1
            print(f"[warn] {csv_path.name} | tile={tile_id} skipped: {exc}")

    label = _csv_stem_label(csv_path)
    _save_matrix_csv(out_dir / f"{label}_confusion_matrix_counts.csv", conf_mat, class_names)
    plot_confusion_matrix(
        out_dir / f"{label}_confusion_matrix_counts.png",
        conf_mat,
        class_names,
        normalize=False,
    )
    plot_confusion_matrix(
        out_dir / f"{label}_confusion_matrix_normalized.png",
        conf_mat,
        class_names,
        normalize=True,
    )
    return used, skipped


def main() -> int:
    args = parse_args()
    if args.num_classes < 1:
        print("ERROR: --num-classes must be >= 1", file=sys.stderr)
        return 1
    if not args.datasets_dir.exists() or not args.datasets_dir.is_dir():
        print(f"ERROR: datasets directory not found: {args.datasets_dir}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (args.datasets_dir / "confusion_matrices")
    csv_files = sorted(args.datasets_dir.glob(args.csv_glob))
    if not csv_files:
        print(
            f"ERROR: No CSV files matching '{args.csv_glob}' in {args.datasets_dir}",
            file=sys.stderr,
        )
        return 1

    class_names = CLASS_NAMES[: args.num_classes]
    if len(class_names) < args.num_classes:
        class_names = [f"class_{i}" for i in range(args.num_classes)]

    total_used = 0
    total_skipped = 0
    for csv_path in csv_files:
        used, skipped = process_csv(csv_path, out_dir, args.num_classes, class_names)
        total_used += used
        total_skipped += skipped
        print(
            f"[ok] {csv_path.name}: used={used}, skipped={skipped} -> {out_dir}"
        )

    print(
        f"Done. Processed {len(csv_files)} CSV files, used rows={total_used}, skipped rows={total_skipped}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
