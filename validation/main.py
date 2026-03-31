from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from common.constants import (  # noqa: E402
	CLASS_COLORS,
	CLASS_NAMES,
	DEFAULT_INPUT_DIR,
	DEFAULT_RESULTS_DIR,
	INPUT_IMAGES_SUBDIR,
	INPUT_MASKS_SUBDIR,
	OUTPUT_PRED_MASKS_SUBDIR,
)
from validation.io_utils import (  # noqa: E402
	SkippedItem,
	load_mask,
	pair_ground_truth_and_predictions,
	write_validation_csv,
)
from validation.metrics import confusion_matrix_update, multiclass_dice  # noqa: E402
from validation.visualization import (  # noqa: E402
	plot_confusion_matrix,
	plot_dice_histogram,
	plot_per_class_dice,
	plot_triplets,
)


def parse_args() -> argparse.Namespace:
	"""Parse CLI options for validation and visualization outputs.

	Returns:
		argparse.Namespace: Validation runtime configuration.
	"""
	parser = argparse.ArgumentParser(description="Validate predicted masks against ground truth masks with Dice metrics.")
	parser.add_argument("--model", type=str, required=True, help="Model name under results/<model>/")
	parser.add_argument("--input_base", type=str, default=DEFAULT_INPUT_DIR, help="Input root containing images/ and masks/")
	parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Root directory for results")
	parser.add_argument("--pred_subdir", type=str, default=OUTPUT_PRED_MASKS_SUBDIR, help="Prediction subdirectory under results/<model>")
	parser.add_argument("--masks_subdir", type=str, default=INPUT_MASKS_SUBDIR, help="Ground truth mask subdirectory under input_base")
	parser.add_argument("--images_subdir", type=str, default=INPUT_IMAGES_SUBDIR, help="RGB image subdirectory under input_base for triplet visualizations")
	parser.add_argument("--csv_name", type=str, default="validation_dice.csv", help="CSV filename written under results/<model>")
	parser.add_argument("--viz_subdir", type=str, default="validation_viz", help="Visualization output folder under results/<model>")
	parser.add_argument("--sample_viz_count", type=int, default=6, help="How many worst-performing tiles to include in triplet panel")
	parser.add_argument("--num_classes", type=int, default=len(CLASS_NAMES), help="Number of classes for Dice and confusion matrix")
	return parser.parse_args()


def _ensure_required_dirs(gt_dir: Path, pred_dir: Path, img_dir: Path) -> None:
	"""Validate required directories and warn when RGB folder is unavailable."""
	if not gt_dir.exists():
		raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
	if not pred_dir.exists():
		raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
	if not img_dir.exists():
		print(f"Warning: RGB image directory not found (triplet visualization may be skipped): {img_dir}")


def _validate_mask_values(mask: np.ndarray, num_classes: int, label: str, tile_id: str) -> None:
	"""Ensure a mask contains only valid class indices."""
	min_v = int(mask.min())
	max_v = int(mask.max())
	if min_v < 0 or max_v >= num_classes:
		raise ValueError(
			f"{label} has out-of-range values for tile {tile_id}: min={min_v}, max={max_v}, expected [0, {num_classes - 1}]"
		)


def _summary_rows(
	num_pairs: int,
	num_skipped: int,
	macro_scores: list[float],
	per_class_scores: list[list[float]],
) -> list[dict[str, str]]:
	"""Build aggregate CSV rows for validation metrics output."""
	mean_macro = float(np.mean(macro_scores)) if macro_scores else 0.0
	class_means = [float(np.mean(col)) if col else 0.0 for col in per_class_scores]

	return [
		{
			"tile_id": "__SUMMARY__",
			"status": "aggregate",
			"macro_dice": f"{mean_macro:.6f}",
			"dice_class_0_water": f"{class_means[0]:.6f}" if len(class_means) > 0 else "",
			"dice_class_1_impervious": f"{class_means[1]:.6f}" if len(class_means) > 1 else "",
			"dice_class_2_sparse_veg": f"{class_means[2]:.6f}" if len(class_means) > 2 else "",
			"dice_class_3_dense_veg": f"{class_means[3]:.6f}" if len(class_means) > 3 else "",
			"ground_truth_path": "",
			"prediction_path": "",
			"reason": f"paired_tiles={num_pairs};skipped_tiles={num_skipped}",
		}
	]


def _select_triplet_samples(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
	"""Select lowest macro-Dice tiles for triplet visualization."""
	ok_rows = [r for r in rows if r.get("status") == "ok"]
	ok_rows.sort(key=lambda r: float(r["macro_dice"]))
	return ok_rows[: max(0, limit)]


def main() -> None:
	"""Run full validation workflow and save metric visualizations."""
	args = parse_args()

	input_base = Path(args.input_base)
	results_base = Path(args.results_dir)
	model_dir = results_base / args.model

	gt_dir = input_base / args.masks_subdir
	img_dir = input_base / args.images_subdir
	pred_dir = model_dir / args.pred_subdir
	out_csv = model_dir / args.csv_name
	viz_dir = model_dir / args.viz_subdir
	viz_dir.mkdir(parents=True, exist_ok=True)

	_ensure_required_dirs(gt_dir=gt_dir, pred_dir=pred_dir, img_dir=img_dir)

	pairs, skipped = pair_ground_truth_and_predictions(gt_dir=gt_dir, pred_dir=pred_dir)
	if not pairs:
		raise RuntimeError("No valid GT/prediction pairs found after matching.")

	csv_rows: list[dict[str, str]] = []
	valid_macro_scores: list[float] = []
	valid_per_class_scores: list[list[float]] = [[] for _ in range(args.num_classes)]
	conf_mat = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

	gt_masks_for_triplets: list[np.ndarray] = []
	pred_masks_for_triplets: list[np.ndarray] = []
	img_paths_for_triplets: list[Path] = []

	for pair in pairs:
		try:
			gt_mask = load_mask(pair.gt_path)
			pred_mask = load_mask(pair.pred_path)
			if gt_mask.shape != pred_mask.shape:
				raise ValueError(f"Shape mismatch for tile {pair.tile_id}: gt={gt_mask.shape} pred={pred_mask.shape}")

			_validate_mask_values(gt_mask, args.num_classes, "ground_truth", pair.tile_id)
			_validate_mask_values(pred_mask, args.num_classes, "prediction", pair.tile_id)

			macro_dice, class_dice = multiclass_dice(gt_mask, pred_mask, args.num_classes)
			confusion_matrix_update(conf_mat, gt_mask, pred_mask, args.num_classes)

			valid_macro_scores.append(macro_dice)
			for class_idx, score in enumerate(class_dice):
				valid_per_class_scores[class_idx].append(score)

			row = {
				"tile_id": pair.tile_id,
				"status": "ok",
				"macro_dice": f"{macro_dice:.6f}",
				"dice_class_0_water": f"{class_dice[0]:.6f}" if len(class_dice) > 0 else "",
				"dice_class_1_impervious": f"{class_dice[1]:.6f}" if len(class_dice) > 1 else "",
				"dice_class_2_sparse_veg": f"{class_dice[2]:.6f}" if len(class_dice) > 2 else "",
				"dice_class_3_dense_veg": f"{class_dice[3]:.6f}" if len(class_dice) > 3 else "",
				"ground_truth_path": str(pair.gt_path),
				"prediction_path": str(pair.pred_path),
				"reason": "",
			}
			csv_rows.append(row)
		except Exception as exc:
			skipped.append(SkippedItem(tile_id=pair.tile_id, reason=f"invalid_pair:{exc}"))

	summary_rows = _summary_rows(
		num_pairs=len(valid_macro_scores),
		num_skipped=len(skipped),
		macro_scores=valid_macro_scores,
		per_class_scores=valid_per_class_scores,
	)

	skipped_rows = [
		{
			"tile_id": skip.tile_id,
			"status": "skipped",
			"macro_dice": "",
			"dice_class_0_water": "",
			"dice_class_1_impervious": "",
			"dice_class_2_sparse_veg": "",
			"dice_class_3_dense_veg": "",
			"ground_truth_path": "",
			"prediction_path": "",
			"reason": skip.reason,
		}
		for skip in skipped
	]

	write_validation_csv(
		out_csv=out_csv,
		rows=csv_rows,
		summary_rows=summary_rows,
		skipped_rows=skipped_rows,
	)

	plot_dice_histogram(viz_dir / "dice_histogram.png", valid_macro_scores)
	plot_confusion_matrix(viz_dir / "confusion_matrix.png", conf_mat, CLASS_NAMES[: args.num_classes], normalize=True)
	class_means = [float(np.mean(scores)) if scores else 0.0 for scores in valid_per_class_scores]
	plot_per_class_dice(viz_dir / "per_class_dice.png", CLASS_NAMES[: args.num_classes], class_means)

	sample_rows = _select_triplet_samples(csv_rows, args.sample_viz_count)
	for sample in sample_rows:
		tile_id = str(sample["tile_id"])
		img_fp = img_dir / f"{tile_id}.tif"
		gt_fp = gt_dir / f"{tile_id}.tif"
		pred_fp = pred_dir / f"{tile_id}.tif"
		if img_fp.exists() and gt_fp.exists() and pred_fp.exists():
			img_paths_for_triplets.append(img_fp)
			gt_masks_for_triplets.append(load_mask(gt_fp))
			pred_masks_for_triplets.append(load_mask(pred_fp))

	if img_paths_for_triplets:
		plot_triplets(
			out_png=viz_dir / "triplets_worst_tiles.png",
			image_paths=img_paths_for_triplets,
			gt_masks=gt_masks_for_triplets,
			pred_masks=pred_masks_for_triplets,
			class_names=CLASS_NAMES[: args.num_classes],
			class_colors=CLASS_COLORS[: args.num_classes],
			title=f"{args.model.upper()} Validation | Worst Macro Dice Tiles",
		)
	else:
		print("No eligible RGB/GT/pred triplets found for triplet visualization.")

	print(f"Validation complete for model: {args.model}")
	print(f"Paired tiles scored: {len(valid_macro_scores)}")
	print(f"Skipped tiles: {len(skipped)}")
	print(f"CSV saved: {out_csv}")
	print(f"Visualizations saved in: {viz_dir}")


if __name__ == "__main__":
	main()
