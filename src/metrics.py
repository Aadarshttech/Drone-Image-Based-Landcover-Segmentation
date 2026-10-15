"""
Evaluation metrics for semantic segmentation of drone landcover imagery.
Includes Mean Intersection-over-Union (mIoU), per-class IoU, Dice/F1, and Pixel Accuracy.
"""

from typing import Dict, Tuple, Union
import torch
import torch.nn.functional as F


def compute_iou(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 3,
    threshold: float = 0.5,
    epsilon: float = 1e-7
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """
    Computes per-class IoU and mean IoU across a batch.

    Args:
        preds: Raw model logits of shape [B, C, H, W].
        labels: Ground truth of shape [B, H, W] (class indices) or [B, C, H, W] (one-hot/binary).
        num_classes: Number of semantic classes.
        threshold: Threshold if using sigmoid binary probabilities.
        epsilon: Small epsilon to prevent division by zero.

    Returns:
        mean_iou (torch.Tensor scalar), per_class_iou (dict mapping class_idx -> float).
    """
    # If predictions are logits [B, C, H, W]
    if preds.ndim == 4 and preds.shape[1] == num_classes:
        if num_classes > 1 and labels.ndim == 3:
            # Multi-class classification with CrossEntropy target
            pred_classes = torch.argmax(preds, dim=1)  # [B, H, W]
        else:
            # Multi-label or binary segmentation
            pred_prob = torch.sigmoid(preds)
            pred_classes = (pred_prob > threshold).float()
    else:
        pred_classes = preds

    per_class_iou = {}
    valid_ious = []

    # Case 1: Standard integer class maps [B, H, W]
    if pred_classes.ndim == 3 and labels.ndim == 3:
        for c in range(num_classes):
            p = (pred_classes == c)
            t = (labels == c)
            intersection = (p & t).float().sum()
            union = (p | t).float().sum()

            if union == 0:
                # Class not present in ground truth or prediction
                iou_c = 1.0 if intersection == 0 else 0.0
            else:
                iou_c = (intersection + epsilon) / (union + epsilon)
                iou_c = iou_c.item()
                valid_ious.append(iou_c)

            per_class_iou[c] = float(iou_c)

    # Case 2: Multi-channel masks [B, C, H, W]
    elif pred_classes.ndim == 4 and labels.ndim == 4:
        for c in range(min(num_classes, pred_classes.shape[1], labels.shape[1])):
            p = (pred_classes[:, c] > threshold).float()
            t = (labels[:, c] > 0.5).float()
            intersection = (p * t).sum()
            union = (p + t).sum() - intersection

            if union == 0:
                iou_c = 1.0 if intersection == 0 else 0.0
            else:
                iou_c = ((intersection + epsilon) / (union + epsilon)).item()
                valid_ious.append(iou_c)

            per_class_iou[c] = float(iou_c)
    else:
        # Fallback to flattened IoU
        p_flat = pred_classes.reshape(-1)
        t_flat = labels.reshape(-1)
        intersection = (p_flat == t_flat).float().sum()
        union = float(len(t_flat))
        mean_iou = intersection / union
        return mean_iou, {0: mean_iou.item()}

    if len(valid_ious) > 0:
        mean_iou = torch.tensor(sum(valid_ious) / len(valid_ious))
    else:
        mean_iou = torch.tensor(0.0)

    return mean_iou, per_class_iou


def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 3
) -> Dict[str, float]:
    """
    Computes a comprehensive dictionary of evaluation metrics:
    mIoU, Pixel Accuracy, and Per-Class IoU.
    """
    mean_iou, per_class = compute_iou(preds, labels, num_classes=num_classes)

    # Pixel accuracy
    if preds.ndim == 4:
        pred_indices = torch.argmax(preds, dim=1)
    else:
        pred_indices = preds

    if labels.ndim == 4:
        target_indices = torch.argmax(labels, dim=1)
    else:
        target_indices = labels

    correct_pixels = (pred_indices == target_indices).float().sum()
    total_pixels = float(target_indices.numel())
    pixel_acc = (correct_pixels / total_pixels).item() if total_pixels > 0 else 0.0

    metrics = {
        "mIoU": float(mean_iou.item() if isinstance(mean_iou, torch.Tensor) else mean_iou),
        "pixel_accuracy": pixel_acc,
    }

    for c, score in per_class.items():
        metrics[f"iou_class_{c}"] = score

    return metrics
