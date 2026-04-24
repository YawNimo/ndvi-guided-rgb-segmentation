from __future__ import annotations

import numpy as np


def dice_for_class(gt: np.ndarray, pred: np.ndarray, class_idx: int) -> float:
    """Compute Dice score for a single class index.

    Args:
        gt (np.ndarray): Ground-truth class mask.
        pred (np.ndarray): Predicted class mask.
        class_idx (int): Target class index.

    Returns:
        float: Dice score in ``[0, 1]`` for the class.
    """
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
    """Compute per-class Dice and macro-Dice across all classes."""
    class_dice = [dice_for_class(gt, pred, c) for c in range(num_classes)]
    macro = float(np.mean(class_dice))
    return macro, class_dice


def f1_for_class(gt: np.ndarray, pred: np.ndarray, class_idx: int) -> float:
    """Compute F1 score for a single class index.

    For one-vs-rest semantic segmentation masks, this is equivalent to Dice.
    """
    gt_bin = gt == class_idx
    pred_bin = pred == class_idx

    tp = int(np.logical_and(gt_bin, pred_bin).sum())
    fp = int(np.logical_and(~gt_bin, pred_bin).sum())
    fn = int(np.logical_and(gt_bin, ~pred_bin).sum())

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0

    denom = (2 * tp) + fp + fn
    if denom == 0:
        return 0.0
    return float((2.0 * tp) / denom)


def iou_for_class(gt: np.ndarray, pred: np.ndarray, class_idx: int) -> float:
    """Compute intersection-over-union score for a single class index."""
    gt_bin = gt == class_idx
    pred_bin = pred == class_idx

    intersection = int(np.logical_and(gt_bin, pred_bin).sum())
    union = int(np.logical_or(gt_bin, pred_bin).sum())
    if union == 0:
        return 1.0
    return float(intersection / union)


def multiclass_f1_iou(
    gt: np.ndarray, pred: np.ndarray, num_classes: int
) -> tuple[float, list[float], float, list[float]]:
    """Compute macro/per-class F1 and IoU across classes."""
    class_f1 = [f1_for_class(gt, pred, c) for c in range(num_classes)]
    class_iou = [iou_for_class(gt, pred, c) for c in range(num_classes)]
    macro_f1 = float(np.mean(class_f1))
    macro_iou = float(np.mean(class_iou))
    return macro_f1, class_f1, macro_iou, class_iou


def confusion_matrix_update(conf_mat: np.ndarray, gt: np.ndarray, pred: np.ndarray, num_classes: int) -> None:
    """Accumulate one batch of predictions into a confusion matrix."""
    valid = (gt >= 0) & (gt < num_classes)
    gt_flat = gt[valid].reshape(-1)
    pred_flat = pred[valid].reshape(-1)
    bins = (num_classes * gt_flat.astype(np.int64)) + pred_flat.astype(np.int64)
    hist = np.bincount(bins, minlength=num_classes**2)
    conf_mat += hist.reshape(num_classes, num_classes)
