"""Evaluation metrics and diagnostic reporting utilities."""

from src.evaluation.diagnostics import (
    save_diagnostic_panels,
    write_ablation_reports,
    write_baseline_metrics,
)
from src.evaluation.metrics import (
    compute_image_metrics,
    compute_pixel_metrics,
    compute_stratified_metrics,
    get_peak_vram_gb,
    measure_inference_latency_ms,
)

__all__ = [
    "compute_image_metrics",
    "compute_pixel_metrics",
    "compute_stratified_metrics",
    "get_peak_vram_gb",
    "measure_inference_latency_ms",
    "save_diagnostic_panels",
    "write_ablation_reports",
    "write_baseline_metrics",
]
