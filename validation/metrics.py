from __future__ import annotations

import numpy as np


def dice_for_class(gt: np.ndarray, pred: np.ndarray, class_idx: int) -> float:
    gt_bin = gt == class_idx
    pred_bin = pred == class_idx

    gt_count = int(gt_bin.sum())
    pred_count = int(pred_bin.sum())
    if gt_count == 0 and pred_count == 0:
        return 1.0

    intersection = int(np.logical_and(gt_bin, pred_bin).sum())
    denom = gt_count + pred_count
    if denom == 0:
        return 0.0
    return float((2.0 * intersection) / denom)


def multiclass_dice(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> tuple[float, list[float]]:
    class_dice = [dice_for_class(gt, pred, c) for c in range(num_classes)]
    macro = float(np.mean(class_dice))
    return macro, class_dice


def confusion_matrix_update(conf_mat: np.ndarray, gt: np.ndarray, pred: np.ndarray, num_classes: int) -> None:
    valid = (gt >= 0) & (gt < num_classes)
    gt_flat = gt[valid].reshape(-1)
    pred_flat = pred[valid].reshape(-1)
    bins = (num_classes * gt_flat.astype(np.int64)) + pred_flat.astype(np.int64)
    hist = np.bincount(bins, minlength=num_classes**2)
    conf_mat += hist.reshape(num_classes, num_classes)
