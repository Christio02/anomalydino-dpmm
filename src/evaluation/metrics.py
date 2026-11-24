"""Pixel, image, stratified, and runtime evaluation metrics (NumPy + scikit-learn)."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert a Tensor or array-like object to a float32 NumPy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _squeeze_maps(maps: Any) -> np.ndarray:
    """Normalize score maps or masks to shape (N, H, W)."""
    arr = _to_numpy(maps)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D (N, H, W) or 4D (N, 1, H, W) array, got shape {arr.shape}")
    return arr


def _lookup_threshold(thresholds: Mapping[Any, float], target_fpr: float) -> float:
    """Look up a threshold by float FPR key (e.g. 0.01) or formatted string key."""
    if target_fpr in thresholds:
        return float(thresholds[target_fpr])
    pct = int(round(target_fpr * 100))
    candidate_keys = [
        str(target_fpr),
        f"{target_fpr:.2f}",
        f"{pct}pct",
        f"fpr_{pct}pct",
    ]
    for key in candidate_keys:
        if key in thresholds:
            return float(thresholds[key])
    for k, v in thresholds.items():
        try:
            if abs(float(k) - target_fpr) < 1e-6:
                return float(v)
        except (TypeError, ValueError):
            continue
    raise KeyError(f"Threshold for FPR={target_fpr} not found in {list(thresholds.keys())}")


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    intersection = float(np.logical_and(pred_bool, gt_bool).sum())
    denom = float(pred_bool.sum() + gt_bool.sum())
    if denom == 0.0:
        return 0.0
    return float((2.0 * intersection) / denom)


def compute_pixel_metrics(
    score_maps: Any,
    gt_masks: Any,
    thresholds: Mapping[Any, float],
) -> dict[str, float]:
    """Compute pixel-level AUPR (primary), AUROC, and Dice at 1%, 5%, 10% FPR."""
    scores = _squeeze_maps(score_maps).astype(np.float64, copy=False)
    masks = (_squeeze_maps(gt_masks) > 0.5).astype(np.uint8)

    flat_scores = scores.ravel()
    flat_masks = masks.ravel()

    pixel_aupr = _safe_aupr(flat_masks, flat_scores)
    pixel_auroc = _safe_auroc(flat_masks, flat_scores)

    t_1 = _lookup_threshold(thresholds, 0.01)
    t_5 = _lookup_threshold(thresholds, 0.05)
    t_10 = _lookup_threshold(thresholds, 0.10)

    return {
        "pixel_aupr": pixel_aupr,
        "pixel_auroc": pixel_auroc,
        "dice_fpr_1pct": _compute_dice(scores >= t_1, masks),
        "dice_fpr_5pct": _compute_dice(scores >= t_5, masks),
        "dice_fpr_10pct": _compute_dice(scores >= t_10, masks),
    }


def compute_image_metrics(image_scores: Any, image_labels: Any) -> dict[str, float]:
    """Compute image-level AUPR and AUROC."""
    scores = _to_numpy(image_scores).astype(np.float64, copy=False).ravel()
    labels = (_to_numpy(image_labels) > 0.5).astype(np.uint8).ravel()

    return {
        "image_aupr": _safe_aupr(labels, scores),
        "image_auroc": _safe_auroc(labels, scores),
    }


def compute_stratified_metrics(
    score_maps: Any,
    gt_masks: Any,
    image_labels: Any,
    thresholds: Mapping[Any, float],
    area_buckets: Sequence[tuple[str, float, float]] = (
        ("tiny", 0.0, 0.0025),
        ("medium", 0.0025, 0.02),
        ("large", 0.02, 1.01),
    ),
) -> dict[str, dict[str, float | int]]:
    """Evaluate pixel AUPR and Dice across defect-area buckets paired with all normal test images."""
    scores = _squeeze_maps(score_maps)
    masks = (_squeeze_maps(gt_masks) > 0.5).astype(np.uint8)
    labels = (_to_numpy(image_labels) > 0.5).astype(np.uint8).ravel()

    normal_idx = np.where(labels == 0)[0]
    anom_idx = np.where(labels == 1)[0]
    area_fractions = masks.reshape(masks.shape[0], -1).mean(axis=1)

    results: dict[str, dict[str, float | int]] = {}
    for bucket_name, low, high in area_buckets:
        if low == 0.0:
            in_bucket = (area_fractions[anom_idx] > low) & (area_fractions[anom_idx] < high)
        else:
            in_bucket = (area_fractions[anom_idx] >= low) & (area_fractions[anom_idx] < high)
        bucket_anom_idx = anom_idx[in_bucket]
        n_anom = int(bucket_anom_idx.size)

        if n_anom == 0:
            results[bucket_name] = {
                "n_anomalous": 0,
                "pixel_aupr": float("nan"),
                "dice_fpr_1pct": float("nan"),
                "dice_fpr_5pct": float("nan"),
                "dice_fpr_10pct": float("nan"),
            }
            continue

        selected_idx = np.concatenate([normal_idx, bucket_anom_idx])
        px_metrics = compute_pixel_metrics(
            scores[selected_idx],
            masks[selected_idx],
            thresholds=thresholds,
        )
        results[bucket_name] = {
            "n_anomalous": n_anom,
            "pixel_aupr": px_metrics["pixel_aupr"],
            "dice_fpr_1pct": px_metrics["dice_fpr_1pct"],
            "dice_fpr_5pct": px_metrics["dice_fpr_5pct"],
            "dice_fpr_10pct": px_metrics["dice_fpr_10pct"],
        }

    return results


def _is_cuda_device(device: torch.device | str) -> bool:
    dev = torch.device(device) if isinstance(device, str) else device
    return dev.type == "cuda" and torch.cuda.is_available()


def measure_inference_latency_ms(
    fn: Callable[[Any], Any],
    sample_input: Any,
    device: torch.device | str,
    warmup: int = 3,
    runs: int = 10,
) -> float:
    """Measure average inference latency in milliseconds with CUDA synchronization."""
    use_cuda = _is_cuda_device(device)

    for _ in range(max(0, warmup)):
        _ = fn(sample_input)
        if use_cuda:
            torch.cuda.synchronize(device)

    if use_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(max(1, runs)):
        _ = fn(sample_input)
    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - t0
    return float((elapsed_s / max(1, runs)) * 1000.0)


def get_peak_vram_gb(device: torch.device | str) -> float:
    """Return peak allocated CUDA memory in GiB (0.0 on CPU)."""
    if not _is_cuda_device(device):
        return 0.0
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return float(peak_bytes / (1024.0 ** 3))
