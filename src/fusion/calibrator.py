"""Leak-free score map calibration, multi-scale fusion, and FPR thresholding."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_score_map_shape(score_map: torch.Tensor) -> torch.Tensor:
    """Ensure a score map has shape (B, 1, H, W)."""
    if score_map.ndim == 3:
        score_map = score_map.unsqueeze(1)
    if score_map.ndim != 4 or score_map.shape[1] != 1:
        raise ValueError(
            f"Expected score_map of shape (B, 1, H, W) or (B, H, W), got {tuple(score_map.shape)}."
        )
    return score_map


def _validate_target_size(target_size: tuple[int, int]) -> tuple[int, int]:
    """Validate and normalize (H, W) target size."""
    if len(target_size) != 2:
        raise ValueError(f"target_size must be a 2-tuple (H, W), got {target_size}.")
    h, w = int(target_size[0]), int(target_size[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"target_size dimensions must be positive, got {(h, w)}.")
    return (h, w)


def upsample_score_map(
    score_map: torch.Tensor,
    target_size: tuple[int, int],
    blur_sigma: float = 0.0,
) -> torch.Tensor:
    """Bilinearly upsample a score map and optionally apply separable Gaussian blur.

    Args:
        score_map: Score tensor of shape ``(B, 1, H_m, W_m)`` or ``(B, H_m, W_m)``.
        target_size: Target spatial resolution ``(H, W)``.
        blur_sigma: Standard deviation for separable 1D Gaussian smoothing. Disabled when ``0.0``.

    Returns:
        Upsampled (and optionally smoothed) tensor of shape ``(B, 1, H, W)``.
    """
    if blur_sigma < 0.0:
        raise ValueError(f"blur_sigma must be >= 0.0, got {blur_sigma}.")
    target_hw = _validate_target_size(target_size)
    out = _normalize_score_map_shape(score_map)
    if not out.is_floating_point():
        out = out.float()

    if tuple(out.shape[-2:]) != target_hw:
        out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)

    if blur_sigma > 0.0:
        radius = max(1, int(math.ceil(3.0 * blur_sigma)))
        coords = torch.arange(-radius, radius + 1, device=out.device, dtype=out.dtype)
        kernel_1d = torch.exp(-0.5 * (coords / blur_sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_x = kernel_1d.view(1, 1, 1, -1)
        kernel_y = kernel_1d.view(1, 1, -1, 1)

        # Replicate padding avoids boundary artifacts and works even if H or W < radius.
        out = F.pad(out, (radius, radius, 0, 0), mode="replicate")
        out = F.conv2d(out, kernel_x)
        out = F.pad(out, (0, 0, radius, radius), mode="replicate")
        out = F.conv2d(out, kernel_y)

    return out


def upsample_score_maps(
    score_maps: dict[str, torch.Tensor],
    target_size: tuple[int, int],
    blur_sigma: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Upsample a dictionary of per-stream score maps to ``target_size``."""
    return {
        stream: upsample_score_map(
            score_map, target_size=target_size, blur_sigma=blur_sigma
        )
        for stream, score_map in score_maps.items()
    }


class BaseCalibrator(ABC):
    """Abstract base class for leak-free per-stream score map calibrators."""

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted on normal validation maps."""
        return self._is_fitted

    @abstractmethod
    def fit(self, val_normal_maps: dict[str, torch.Tensor]) -> BaseCalibrator:
        """Fit calibration statistics strictly on normal validation score maps."""

    @abstractmethod
    def transform(self, score_maps: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply fitted calibration to each stream's score map."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary of calibrator parameters and state."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore calibrator parameters and state from ``state``."""

    def fit_transform(
        self, val_normal_maps: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fit on ``val_normal_maps`` and return their calibrated transforms."""
        self.fit(val_normal_maps)
        return self.transform(val_normal_maps)

    def fit_from_batches(
        self, batch_iter: Iterable[dict[str, torch.Tensor]]
    ) -> BaseCalibrator:
        """Fit by accumulating stream batches from an iterable along the batch dimension."""
        accumulated: dict[str, list[torch.Tensor]] = {}
        expected_keys: set[str] | None = None

        for batch in batch_iter:
            if not batch:
                raise ValueError("Encountered an empty batch in batch_iter.")
            batch_keys = set(batch.keys())
            if expected_keys is None:
                expected_keys = batch_keys
                for key in batch_keys:
                    accumulated[key] = []
            elif batch_keys != expected_keys:
                raise ValueError(
                    f"Inconsistent stream keys across batches: expected {sorted(expected_keys)}, "
                    f"got {sorted(batch_keys)}."
                )
            for key, tensor in batch.items():
                accumulated[key].append(tensor.detach())

        if expected_keys is None:
            raise ValueError("batch_iter yielded no batches.")

        merged = {
            key: torch.cat(tensors, dim=0) for key, tensors in accumulated.items()
        }
        return self.fit(merged)


class IdentityCalibrator(BaseCalibrator):
    """No-op pass-through calibrator (used when ``calibration='none'``)."""

    def __init__(self) -> None:
        super().__init__()
        self._is_fitted = True

    def fit(self, val_normal_maps: dict[str, torch.Tensor]) -> IdentityCalibrator:
        if not val_normal_maps:
            raise ValueError("val_normal_maps must not be empty.")
        self._is_fitted = True
        return self

    def transform(self, score_maps: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return dict(score_maps)

    def state_dict(self) -> dict[str, Any]:
        return {"type": "none", "is_fitted": self._is_fitted}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._is_fitted = bool(state.get("is_fitted", True))


class ZScoreCalibrator(BaseCalibrator):
    """Per-stream z-score calibrator using exact population mean and std."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        self.eps: float = float(eps)
        self.stats: dict[str, tuple[float, float]] = {}

    def fit(self, val_normal_maps: dict[str, torch.Tensor]) -> ZScoreCalibrator:
        if not val_normal_maps:
            raise ValueError("val_normal_maps must not be empty.")
        stats: dict[str, tuple[float, float]] = {}
        for stream, score_map in val_normal_maps.items():
            if score_map.numel() == 0:
                raise ValueError(f"Stream '{stream}' has no elements.")
            flat = score_map.detach().to(dtype=torch.float64).reshape(-1)
            mean = float(flat.mean().item())
            std = float(flat.std(unbiased=False).item())
            stats[stream] = (mean, std)
        self.stats = stats
        self._is_fitted = True
        return self

    def transform(self, score_maps: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self._is_fitted:
            raise RuntimeError(
                "ZScoreCalibrator must be fitted before calling transform()."
            )
        calibrated: dict[str, torch.Tensor] = {}
        for stream, score_map in score_maps.items():
            if stream not in self.stats:
                raise KeyError(f"Stream '{stream}' was not seen during fit().")
            mean, std = self.stats[stream]
            calibrated[stream] = (score_map - mean) / (std + self.eps)
        return calibrated

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "zscore",
            "eps": self.eps,
            "is_fitted": self._is_fitted,
            "stats": {
                stream: {"mean": mean, "std": std}
                for stream, (mean, std) in self.stats.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        eps = float(state["eps"])
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        self.eps = eps
        self.stats = {
            str(stream): (float(vals["mean"]), float(vals["std"]))
            for stream, vals in state.get("stats", {}).items()
        }
        self._is_fitted = bool(state.get("is_fitted", bool(self.stats)))


class RobustIQRCalibrator(BaseCalibrator):
    """Per-stream robust calibrator using median (Q50) and IQR (Q75 - Q25)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        self.eps: float = float(eps)
        self.stats: dict[str, tuple[float, float, float]] = {}

    def fit(self, val_normal_maps: dict[str, torch.Tensor]) -> RobustIQRCalibrator:
        if not val_normal_maps:
            raise ValueError("val_normal_maps must not be empty.")
        stats: dict[str, tuple[float, float, float]] = {}
        for stream, score_map in val_normal_maps.items():
            if score_map.numel() == 0:
                raise ValueError(f"Stream '{stream}' has no elements.")
            # Use NumPy on CPU float64 to avoid torch.quantile's 2^24 element limit.
            pixels = (
                score_map.detach().cpu().to(dtype=torch.float64).reshape(-1).numpy()
            )
            q25, q50, q75 = np.quantile(pixels, [0.25, 0.50, 0.75])
            stats[stream] = (float(q25), float(q50), float(q75))
        self.stats = stats
        self._is_fitted = True
        return self

    def transform(self, score_maps: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self._is_fitted:
            raise RuntimeError(
                "RobustIQRCalibrator must be fitted before calling transform()."
            )
        calibrated: dict[str, torch.Tensor] = {}
        for stream, score_map in score_maps.items():
            if stream not in self.stats:
                raise KeyError(f"Stream '{stream}' was not seen during fit().")
            q25, q50, q75 = self.stats[stream]
            calibrated[stream] = (score_map - q50) / ((q75 - q25) + self.eps)
        return calibrated

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "robust_iqr",
            "eps": self.eps,
            "is_fitted": self._is_fitted,
            "stats": {
                stream: {"q25": q25, "q50": q50, "q75": q75}
                for stream, (q25, q50, q75) in self.stats.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        eps = float(state["eps"])
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        self.eps = eps
        self.stats = {
            str(stream): (float(vals["q25"]), float(vals["q50"]), float(vals["q75"]))
            for stream, vals in state.get("stats", {}).items()
        }
        self._is_fitted = bool(state.get("is_fitted", bool(self.stats)))


def build_calibrator(calibration: str, eps: float = 1e-6) -> BaseCalibrator:
    """Factory for calibrators supporting ``'none'``, ``'zscore'``, and ``'robust_iqr'``."""
    mode = calibration.strip().lower()
    if mode == "none":
        return IdentityCalibrator()
    if mode == "zscore":
        return ZScoreCalibrator(eps=eps)
    if mode == "robust_iqr":
        return RobustIQRCalibrator(eps=eps)
    raise ValueError(
        f"Unsupported calibration mode '{calibration}'. "
        "Expected one of {'none', 'zscore', 'robust_iqr'}."
    )


class FusionHead:
    """Multi-scale score map fusion, image-level scoring, and FPR threshold calibration."""

    _VALID_MODES = {"max", "weighted_sum", "lse"}
    _VALID_AGGREGATIONS = {"max", "top_k_mean"}

    def __init__(
        self,
        mode: str = "max",
        weights: dict[str, float] | None = None,
        lse_tau: float = 2.0,
        image_score_aggregation: str = "top_k_mean",
        image_score_top_k_fraction: float = 0.005,
        fpr_targets: tuple[float, ...] | list[float] = (0.01, 0.05, 0.10),
    ) -> None:
        self._validate_config(
            mode=mode,
            lse_tau=lse_tau,
            image_score_aggregation=image_score_aggregation,
            image_score_top_k_fraction=image_score_top_k_fraction,
            fpr_targets=fpr_targets,
        )
        self.mode: str = mode
        self.weights: dict[str, float] | None = (
            {str(k): float(v) for k, v in weights.items()}
            if weights is not None
            else None
        )
        self.lse_tau: float = float(lse_tau)
        self.image_score_aggregation: str = image_score_aggregation
        self.image_score_top_k_fraction: float = float(image_score_top_k_fraction)
        self.fpr_targets: tuple[float, ...] = tuple(float(t) for t in fpr_targets)
        self.thresholds: dict[float, float] = {}

    @classmethod
    def _validate_config(
        cls,
        mode: str,
        lse_tau: float,
        image_score_aggregation: str,
        image_score_top_k_fraction: float,
        fpr_targets: Iterable[float],
    ) -> None:
        if mode not in cls._VALID_MODES:
            raise ValueError(
                f"mode must be in {sorted(cls._VALID_MODES)}, got '{mode}'."
            )
        if lse_tau <= 0.0:
            raise ValueError(f"lse_tau must be > 0, got {lse_tau}.")
        if image_score_aggregation not in cls._VALID_AGGREGATIONS:
            raise ValueError(
                f"image_score_aggregation must be in {sorted(cls._VALID_AGGREGATIONS)}, "
                f"got '{image_score_aggregation}'."
            )
        if not (0.0 < image_score_top_k_fraction <= 1.0):
            raise ValueError(
                f"image_score_top_k_fraction must be in (0, 1], got {image_score_top_k_fraction}."
            )
        for fpr in fpr_targets:
            if not (0.0 < float(fpr) < 1.0):
                raise ValueError(f"Each FPR target must be in (0, 1), got {fpr}.")

    def fuse(self, calibrated_maps: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse per-stream calibrated maps ``(B, 1, H, W)`` into a single ``(B, 1, H, W)`` map."""
        if not calibrated_maps:
            raise ValueError("calibrated_maps must not be empty.")

        keys = list(calibrated_maps.keys())
        tensors = [_normalize_score_map_shape(calibrated_maps[k]) for k in keys]
        ref_shape = tensors[0].shape
        for k, t in zip(keys[1:], tensors[1:]):
            if t.shape != ref_shape:
                raise ValueError(
                    f"All stream maps must share shape {tuple(ref_shape)}, "
                    f"but '{k}' has {tuple(t.shape)}."
                )

        stacked = torch.stack(tensors, dim=0)  # (M, B, 1, H, W)

        if self.mode == "max":
            return stacked.max(dim=0).values

        if self.mode == "weighted_sum":
            if self.weights is None:
                raw_weights = [1.0] * len(keys)
            else:
                missing = [k for k in keys if k not in self.weights]
                if missing:
                    raise ValueError(f"Missing weights for stream(s): {missing}.")
                raw_weights = [float(self.weights[k]) for k in keys]

            if any(w < 0.0 for w in raw_weights):
                raise ValueError(f"All stream weights must be >= 0, got {raw_weights}.")
            weight_sum = sum(raw_weights)
            if weight_sum <= 0.0:
                raise ValueError(
                    f"Sum of stream weights must be > 0, got {weight_sum}."
                )

            norm_weights = [w / weight_sum for w in raw_weights]
            weight_tensor = torch.tensor(
                norm_weights, device=stacked.device, dtype=stacked.dtype
            ).view(-1, 1, 1, 1, 1)
            return (stacked * weight_tensor).sum(dim=0)

        # self.mode == "lse"
        return torch.logsumexp(self.lse_tau * stacked, dim=0) / self.lse_tau

    def fit_thresholds(
        self,
        fused_val_normal_maps: torch.Tensor,
        fpr_targets: tuple[float, ...] | list[float] | None = None,
    ) -> dict[float, float]:
        """Calibrate pixel-level thresholds on fused normal validation maps for target FPRs."""
        if fused_val_normal_maps.numel() == 0:
            raise ValueError("fused_val_normal_maps must not be empty.")

        targets = (
            self.fpr_targets
            if fpr_targets is None
            else tuple(float(t) for t in fpr_targets)
        )
        for fpr in targets:
            if not (0.0 < fpr < 1.0):
                raise ValueError(f"Each FPR target must be in (0, 1), got {fpr}.")

        pixels = (
            fused_val_normal_maps.detach()
            .cpu()
            .to(dtype=torch.float64)
            .reshape(-1)
            .numpy()
        )
        for fpr in targets:
            thr = float(np.quantile(pixels, 1.0 - fpr))
            self.thresholds[float(fpr)] = thr

        return dict(self.thresholds)

    def predict_mask(
        self, fused_maps: torch.Tensor, fpr_target: float = 0.01
    ) -> torch.Tensor:
        """Threshold fused score maps into boolean anomaly masks for a fitted FPR target."""
        key = float(fpr_target)
        if key not in self.thresholds:
            raise KeyError(
                f"Threshold for fpr_target={fpr_target} is not fitted. "
                f"Available targets: {sorted(self.thresholds.keys())}."
            )
        return fused_maps > self.thresholds[key]

    def compute_image_scores(self, fused_maps: torch.Tensor) -> torch.Tensor:
        """Aggregate spatial fused maps ``(B, 1, H, W)`` into image-level scores ``(B,)``."""
        maps = _normalize_score_map_shape(fused_maps)
        batch_size, _, height, width = maps.shape
        flat = maps.reshape(batch_size, height * width)

        if self.image_score_aggregation == "max":
            return flat.max(dim=1).values

        k = max(1, int(round(height * width * self.image_score_top_k_fraction)))
        k = min(k, flat.shape[1])
        topk_vals = torch.topk(flat, k=k, dim=1, largest=True, sorted=False).values
        return topk_vals.mean(dim=1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "weights": dict(self.weights) if self.weights is not None else None,
            "lse_tau": self.lse_tau,
            "image_score_aggregation": self.image_score_aggregation,
            "image_score_top_k_fraction": self.image_score_top_k_fraction,
            "fpr_targets": list(self.fpr_targets),
            "thresholds": {float(k): float(v) for k, v in self.thresholds.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        mode = str(state["mode"])
        lse_tau = float(state["lse_tau"])
        image_score_aggregation = str(state["image_score_aggregation"])
        image_score_top_k_fraction = float(state["image_score_top_k_fraction"])
        fpr_targets = tuple(float(t) for t in state["fpr_targets"])
        self._validate_config(
            mode=mode,
            lse_tau=lse_tau,
            image_score_aggregation=image_score_aggregation,
            image_score_top_k_fraction=image_score_top_k_fraction,
            fpr_targets=fpr_targets,
        )
        raw_weights = state.get("weights")
        self.mode = mode
        self.weights = (
            {str(k): float(v) for k, v in raw_weights.items()}
            if raw_weights is not None
            else None
        )
        self.lse_tau = lse_tau
        self.image_score_aggregation = image_score_aggregation
        self.image_score_top_k_fraction = image_score_top_k_fraction
        self.fpr_targets = fpr_targets
        self.thresholds = {
            float(k): float(v) for k, v in state.get("thresholds", {}).items()
        }


class CalibrationFusionPipeline:
    """End-to-end upsampling, leak-free calibration, multi-scale fusion, and thresholding."""

    def __init__(
        self,
        target_size: tuple[int, int],
        blur_sigma: float = 0.0,
        calibrator: BaseCalibrator | None = None,
        fusion_head: FusionHead | None = None,
    ) -> None:
        if blur_sigma < 0.0:
            raise ValueError(f"blur_sigma must be >= 0.0, got {blur_sigma}.")
        self.target_size: tuple[int, int] = _validate_target_size(target_size)
        self.blur_sigma: float = float(blur_sigma)
        self.calibrator: BaseCalibrator = (
            calibrator if calibrator is not None else IdentityCalibrator()
        )
        self.fusion_head: FusionHead = (
            fusion_head if fusion_head is not None else FusionHead()
        )

    def fit(self, val_normal_raw_maps: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Upsample, fit calibrator, fuse normal validation maps, and calibrate FPR thresholds."""
        upsampled_maps = upsample_score_maps(
            val_normal_raw_maps,
            target_size=self.target_size,
            blur_sigma=self.blur_sigma,
        )
        calibrated_maps = self.calibrator.fit_transform(upsampled_maps)
        fused_map = self.fusion_head.fuse(calibrated_maps)
        thresholds = self.fusion_head.fit_thresholds(fused_map)
        image_scores = self.fusion_head.compute_image_scores(fused_map)
        binary_masks = {
            fpr: self.fusion_head.predict_mask(fused_map, fpr_target=fpr)
            for fpr in self.fusion_head.thresholds
        }
        return {
            "upsampled_maps": upsampled_maps,
            "calibrated_maps": calibrated_maps,
            "fused_map": fused_map,
            "image_scores": image_scores,
            "binary_masks": binary_masks,
            "thresholds": thresholds,
        }

    def predict(self, raw_maps: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Upsample, calibrate, fuse, and score raw multi-scale score maps."""
        upsampled_maps = upsample_score_maps(
            raw_maps, target_size=self.target_size, blur_sigma=self.blur_sigma
        )
        calibrated_maps = self.calibrator.transform(upsampled_maps)
        fused_map = self.fusion_head.fuse(calibrated_maps)
        image_scores = self.fusion_head.compute_image_scores(fused_map)
        binary_masks = {
            fpr: self.fusion_head.predict_mask(fused_map, fpr_target=fpr)
            for fpr in self.fusion_head.thresholds
        }
        return {
            "upsampled_maps": upsampled_maps,
            "calibrated_maps": calibrated_maps,
            "fused_map": fused_map,
            "image_scores": image_scores,
            "binary_masks": binary_masks,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "target_size": list(self.target_size),
            "blur_sigma": self.blur_sigma,
            "calibrator": self.calibrator.state_dict(),
            "fusion_head": self.fusion_head.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.target_size = _validate_target_size(tuple(state["target_size"]))
        blur_sigma = float(state["blur_sigma"])
        if blur_sigma < 0.0:
            raise ValueError(f"blur_sigma must be >= 0.0, got {blur_sigma}.")
        self.blur_sigma = blur_sigma

        cal_state = state["calibrator"]
        cal_type = cal_state.get("type")
        if cal_type is not None:
            expected_cls = {
                "none": IdentityCalibrator,
                "zscore": ZScoreCalibrator,
                "robust_iqr": RobustIQRCalibrator,
            }.get(cal_type)
            if expected_cls is not None and not isinstance(
                self.calibrator, expected_cls
            ):
                self.calibrator = build_calibrator(cal_type)
        self.calibrator.load_state_dict(cal_state)
        self.fusion_head.load_state_dict(state["fusion_head"])
