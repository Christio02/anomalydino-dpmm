"""Diagnostic panel visualization and ablation/baseline report writers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_numpy(arr: Any) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _squeeze_maps(maps: Any) -> np.ndarray:
    arr = _to_numpy(maps)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr


def _image_to_display_rgb(img: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) ImageNet-normalized or (H, W, 3) image to [0, 1] RGB (H, W, 3)."""
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = img.astype(np.float32, copy=False)
    # Heuristic: if values fall outside [0, 1], assume ImageNet normalization.
    if img.min() < 0.0 or img.max() > 1.0:
        if img.max() > 10.0 and img.min() >= 0.0:
            img = img / 255.0
        else:
            img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def _render_7panel_figure(
    out_path: Path,
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    wrn_map: np.ndarray | None,
    dino_b4_map: np.ndarray | None,
    dino_b12_map: np.ndarray | None,
    fused_map: np.ndarray,
    binary_mask_1pct: np.ndarray,
    title: str,
) -> None:
    """Render and save a 7-panel diagnostic PNG using non-interactive FigureCanvasAgg."""
    fig = Figure(figsize=(24.5, 3.8), dpi=120)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 7)

    panels: list[tuple[str, np.ndarray | None, str, tuple[float, float] | None]] = [
        ("Input X-Ray", image_rgb, "gray", None),
        ("Ground Truth Mask", gt_mask, "gray", (0.0, 1.0)),
        ("wrn_l2l3 Map", wrn_map, "inferno", None),
        ("dino_b4 Map", dino_b4_map, "inferno", None),
        ("dino_b12 Map", dino_b12_map, "inferno", None),
        ("Fused Map", fused_map, "inferno", None),
        ("1% FPR Mask", binary_mask_1pct, "gray", (0.0, 1.0)),
    ]

    for ax, (panel_title, data, cmap, clim) in zip(axes, panels):
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")
        if data is None:
            ax.set_facecolor("#f0f0f0")
            ax.text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#555555",
                transform=ax.transAxes,
            )
            continue

        if panel_title == "Input X-Ray":
            ax.imshow(data)
        elif clim is not None:
            ax.imshow(data, cmap=cmap, vmin=clim[0], vmax=clim[1])
        else:
            ax.imshow(data, cmap=cmap)

    fig.suptitle(title, fontsize=11, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)


def save_diagnostic_panels(
    output_dir: str | Path,
    images: Any,
    gt_masks: Any,
    image_labels: Any,
    per_stream_maps: Mapping[str, Any],
    fused_maps: Any,
    binary_masks_1pct: Any,
    image_scores: Any,
    image_paths: Sequence[str],
    top_n: int = 10,
) -> dict[str, list[str]]:
    """Save 7-panel diagnostic PNGs for smallest defects and top false-positive normals."""
    out_root = Path(output_dir)
    imgs_np = _to_numpy(images)
    masks_np = (_squeeze_maps(gt_masks) > 0.5).astype(np.float32)
    labels_np = (_to_numpy(image_labels) > 0.5).astype(np.int64).ravel()
    fused_np = _squeeze_maps(fused_maps).astype(np.float32)
    bin_np = (_squeeze_maps(binary_masks_1pct) > 0.5).astype(np.float32)
    scores_np = _to_numpy(image_scores).astype(np.float32).ravel()

    stream_np: dict[str, np.ndarray] = {
        k: _squeeze_maps(v).astype(np.float32) for k, v in per_stream_maps.items()
    }

    # 1. Select top_n smallest-defect anomalous cases.
    anom_indices = np.where(labels_np == 1)[0]
    smallest_selected: list[int] = []
    if anom_indices.size > 0:
        pixel_counts = masks_np[anom_indices].reshape(anom_indices.size, -1).sum(axis=1)
        nonzero_mask = pixel_counts > 0
        if np.any(nonzero_mask):
            nz_indices = anom_indices[nonzero_mask]
            nz_counts = pixel_counts[nonzero_mask]
            order = np.argsort(nz_counts, kind="stable")
            smallest_selected = [int(nz_indices[i]) for i in order[:top_n]]
        else:
            # Fallback to lowest anomalous image scores when GT masks are absent.
            order = np.argsort(scores_np[anom_indices], kind="stable")
            smallest_selected = [int(anom_indices[i]) for i in order[:top_n]]

    # 2. Select top_n highest-scoring false-positive normal cases.
    normal_indices = np.where(labels_np == 0)[0]
    fp_selected: list[int] = []
    if normal_indices.size > 0:
        order = np.argsort(-scores_np[normal_indices], kind="stable")
        fp_selected = [int(normal_indices[i]) for i in order[:top_n]]

    saved: dict[str, list[str]] = {"smallest_defects": [], "false_positives": []}

    def _save_group(indices: list[int], subdir: str, key: str) -> None:
        for rank, idx in enumerate(indices, start=1):
            stem = Path(image_paths[idx]).stem if idx < len(image_paths) else f"sample_{idx}"
            px_area = int(masks_np[idx].sum())
            score_val = float(scores_np[idx])
            out_file = out_root / subdir / f"{rank:02d}_{stem}.png"
            title = f"{subdir} #{rank} | {stem} | score={score_val:.4f} | gt_px={px_area}"

            _render_7panel_figure(
                out_path=out_file,
                image_rgb=_image_to_display_rgb(imgs_np[idx]),
                gt_mask=masks_np[idx],
                wrn_map=stream_np["wrn_l2l3"][idx] if "wrn_l2l3" in stream_np else None,
                dino_b4_map=stream_np["dino_b4"][idx] if "dino_b4" in stream_np else None,
                dino_b12_map=stream_np["dino_b12"][idx] if "dino_b12" in stream_np else None,
                fused_map=fused_np[idx],
                binary_mask_1pct=bin_np[idx],
                title=title,
            )
            saved[key].append(str(out_file))

    _save_group(smallest_selected, "smallest_defects", "smallest_defects")
    _save_group(fp_selected, "false_positives", "false_positives")
    return saved


def _fmt_float(val: Any, precision: int = 4) -> str:
    if val is None:
        return "NaN"
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return str(val)
    if math.isnan(fval) or math.isinf(fval):
        return "NaN"
    return f"{fval:.{precision}f}"


def _sanitize_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for strict JSON compliance."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def write_ablation_reports(
    results: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> tuple[str, str]:
    """Write ablation_summary.md (main + stratified tables) and ablation_summary.csv."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "ablation_summary.md"
    csv_path = out_dir / "ablation_summary.csv"

    main_cols = [
        "experiment",
        "seed",
        "pixel_aupr",
        "pixel_auroc",
        "dice_fpr_1pct",
        "dice_fpr_5pct",
        "dice_fpr_10pct",
        "image_aupr",
        "image_auroc",
        "latency_ms",
        "peak_vram_gb",
    ]
    buckets = ("tiny", "medium", "large")
    strat_cols = []
    for b in buckets:
        strat_cols.extend(
            [
                f"{b}_n_anomalous",
                f"{b}_pixel_aupr",
                f"{b}_dice_fpr_1pct",
                f"{b}_dice_fpr_5pct",
                f"{b}_dice_fpr_10pct",
            ]
        )

    # Write CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=main_cols + strat_cols)
        writer.writeheader()
        for row in results:
            flat: dict[str, Any] = {col: row.get(col, "") for col in main_cols}
            strat = row.get("stratified", {}) or {}
            for b in buckets:
                b_data = strat.get(b, {}) or {}
                flat[f"{b}_n_anomalous"] = b_data.get("n_anomalous", 0)
                flat[f"{b}_pixel_aupr"] = b_data.get("pixel_aupr", float("nan"))
                flat[f"{b}_dice_fpr_1pct"] = b_data.get("dice_fpr_1pct", float("nan"))
                flat[f"{b}_dice_fpr_5pct"] = b_data.get("dice_fpr_5pct", float("nan"))
                flat[f"{b}_dice_fpr_10pct"] = b_data.get("dice_fpr_10pct", float("nan"))
            writer.writerow(flat)

    # Write Markdown
    lines: list[str] = [
        "# Ablation Summary",
        "",
        "## Main Metrics (Pixel AUPR Primary)",
        "",
        "| Experiment | Seed | Pixel AUPR | Pixel AUROC | Dice@1%FPR | Dice@5%FPR | Dice@10%FPR | Image AUPR | Image AUROC | Latency (ms) | Peak VRAM (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("experiment", "")),
                    str(row.get("seed", "")),
                    _fmt_float(row.get("pixel_aupr")),
                    _fmt_float(row.get("pixel_auroc")),
                    _fmt_float(row.get("dice_fpr_1pct")),
                    _fmt_float(row.get("dice_fpr_5pct")),
                    _fmt_float(row.get("dice_fpr_10pct")),
                    _fmt_float(row.get("image_aupr")),
                    _fmt_float(row.get("image_auroc")),
                    _fmt_float(row.get("latency_ms"), precision=2),
                    _fmt_float(row.get("peak_vram_gb"), precision=3),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Stratified Defect-Area Metrics",
            "",
            "| Experiment | Seed | Bucket | N Anom | Pixel AUPR | Dice@1%FPR | Dice@5%FPR | Dice@10%FPR |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        strat = row.get("stratified", {}) or {}
        for b in buckets:
            b_data = strat.get(b, {}) or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("experiment", "")),
                        str(row.get("seed", "")),
                        b,
                        str(b_data.get("n_anomalous", 0)),
                        _fmt_float(b_data.get("pixel_aupr")),
                        _fmt_float(b_data.get("dice_fpr_1pct")),
                        _fmt_float(b_data.get("dice_fpr_5pct")),
                        _fmt_float(b_data.get("dice_fpr_10pct")),
                    ]
                )
                + " |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(md_path), str(csv_path)


def write_baseline_metrics(
    seed_results: Sequence[Mapping[str, Any]],
    output_path: str | Path = "reports/baseline_metrics.json",
) -> str:
    """Write baseline per-seed metrics and mean/std summaries to JSON."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        "pixel_aupr",
        "pixel_auroc",
        "dice_fpr_1pct",
        "dice_fpr_5pct",
        "dice_fpr_10pct",
        "image_aupr",
        "image_auroc",
        "latency_ms",
        "peak_vram_gb",
    ]
    summary: dict[str, dict[str, float | None]] = {}
    for key in metric_keys:
        vals = [
            float(r[key])
            for r in seed_results
            if key in r and r[key] is not None and not math.isnan(float(r[key]))
        ]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
            }
        else:
            summary[key] = {"mean": None, "std": None}

    payload = {
        "per_seed": _sanitize_json(list(seed_results)),
        "summary": _sanitize_json(summary),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(out_path)
