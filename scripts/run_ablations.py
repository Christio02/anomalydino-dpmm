"""Controlled ablation runner (E0-E8) for hybrid WRN50 + DINOv2 multi-stream DPMM."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path regardless of current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

from src.data_multistream import build_multistream_dataloaders
from src.evaluation import (
    compute_image_metrics,
    compute_pixel_metrics,
    compute_stratified_metrics,
    get_peak_vram_gb,
    measure_inference_latency_ms,
    save_diagnostic_panels,
    write_ablation_reports,
    write_baseline_metrics,
)

ALL_EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8")
ALL_DINOV3_AB = ("AB0", "AB1", "AB2", "AB3")


def pin_all_seeds(seed: int) -> None:
    """Pin Python, NumPy, and PyTorch RNG seeds for deterministic execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    """Resolve 'auto' or explicit device string to a torch.device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML experiment configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        candidate = REPO_ROOT / path
        if candidate.exists():
            path = candidate
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_experiment_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the E0-E8 controlled ablation experiment matrix."""
    fusion_cfg = cfg.get("fusion", {})
    lse_tau = float(fusion_cfg.get("lse_tau", 2.0))
    weights = dict(fusion_cfg.get("weights", {}))

    return {
        "E0": {
            "name": "E0",
            "description": "DINOv2 block-12 single-stream baseline",
            "streams": ["dino_b12"],
            "wrn_metric": None,
            "calibration": "none",
            "fusion": "max",
            "lse_tau": None,
            "weights": None,
        },
        "E1": {
            "name": "E1",
            "description": "DINOv2 multi-layer (b4,b8,b12) + zscore + max",
            "streams": ["dino_b4", "dino_b8", "dino_b12"],
            "wrn_metric": None,
            "calibration": "zscore",
            "fusion": "max",
            "lse_tau": None,
            "weights": None,
        },
        "E2": {
            "name": "E2",
            "description": "DINOv2 multi-layer (b4,b8,b12) + robust_iqr + lse",
            "streams": ["dino_b4", "dino_b8", "dino_b12"],
            "wrn_metric": None,
            "calibration": "robust_iqr",
            "fusion": "lse",
            "lse_tau": lse_tau,
            "weights": None,
        },
        "E3": {
            "name": "E3",
            "description": "WRN50-2 L2+L3 single-stream (Euclidean)",
            "streams": ["wrn_l2l3"],
            "wrn_metric": "euclidean",
            "calibration": "none",
            "fusion": "max",
            "lse_tau": None,
            "weights": None,
        },
        "E4": {
            "name": "E4",
            "description": "WRN50-2 L2+L3 single-stream (Mahalanobis)",
            "streams": ["wrn_l2l3"],
            "wrn_metric": "mahalanobis",
            "calibration": "none",
            "fusion": "max",
            "lse_tau": None,
            "weights": None,
        },
        "E5": {
            "name": "E5",
            "description": "Hybrid WRN50+DINOv2 + zscore + max",
            "streams": ["wrn_l2l3", "dino_b4", "dino_b8", "dino_b12"],
            "wrn_metric": "best_of_e3_e4",
            "calibration": "zscore",
            "fusion": "max",
            "lse_tau": None,
            "weights": None,
        },
        "E6": {
            "name": "E6",
            "description": "Hybrid WRN50+DINOv2 + robust_iqr + max",
            "streams": ["wrn_l2l3", "dino_b4", "dino_b8", "dino_b12"],
            "wrn_metric": "best_of_e3_e4",
            "calibration": "robust_iqr",
            "fusion": "max",
            "lse_tau": None,
            "weights": None,
        },
        "E7": {
            "name": "E7",
            "description": "Hybrid WRN50+DINOv2 + best_calib + weighted_sum",
            "streams": ["wrn_l2l3", "dino_b4", "dino_b8", "dino_b12"],
            "wrn_metric": "best_of_e3_e4",
            "calibration": "best_of_e5_e6",
            "fusion": "weighted_sum",
            "lse_tau": None,
            "weights": weights,
        },
        "E8": {
            "name": "E8",
            "description": "Hybrid WRN50+DINOv2 + best_calib + lse(tau=2.0)",
            "streams": ["wrn_l2l3", "dino_b4", "dino_b8", "dino_b12"],
            "wrn_metric": "best_of_e3_e4",
            "calibration": "best_of_e5_e6",
            "fusion": "lse",
            "lse_tau": lse_tau,
            "weights": None,
        },
    }


def print_experiment_matrix(
    specs: dict[str, dict[str, Any]],
    selected: list[str],
    seeds: list[int],
    cfg: dict[str, Any],
) -> None:
    """Print the E0-E8 experiment matrix cleanly for dry-run inspection."""
    print(f"Experiment: {cfg['experiment']['name']} | image_size={cfg['experiment']['image_size']}")
    print(f"Seeds: {seeds} | Selected: {selected}")
    print("-" * 108)
    header = f"{'ID':<4} {'Streams':<38} {'WRN Metric':<14} {'Calibration':<14} {'Fusion':<14} {'Tau':<6}"
    print(header)
    print("-" * 108)
    for exp_id in ALL_EXPERIMENTS:
        spec = specs[exp_id]
        mark = "*" if exp_id in selected else " "
        streams_str = ",".join(spec["streams"])
        wrn_str = str(spec["wrn_metric"] or "-")
        cal_str = str(spec["calibration"])
        fus_str = str(spec["fusion"])
        tau_str = str(spec["lse_tau"] if spec["lse_tau"] is not None else "-")
        print(f"{exp_id:<3}{mark} {streams_str:<38} {wrn_str:<14} {cal_str:<14} {fus_str:<14} {tau_str:<6}")
    print("-" * 108)


def build_dinov3_ab_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the 4 head-to-head DINOv2 vs DINOv3 A/B benchmark specs (AB0-AB3)."""
    backbone_cfg = cfg.get("backbone", {})
    dpmm_cfg = cfg.get("dpmm") or cfg.get("streams", {}).get("dino_multilayer", {}).get("dpmm", {})
    l2_normalize = bool(backbone_cfg.get("l2_normalize", True))
    cache_embeddings = bool(dpmm_cfg.get("cache_embeddings", True))

    return {
        "AB0": {
            "name": "AB0",
            "description": "DINOv2 ViT-S/14 baseline (32x32 grid, 384-D)",
            "provider": "torchhub_dinov2",
            "model_name": "dinov2_vits14",
            "image_size": 448,
            "layers": [-1],
            "expected_grid": "32x32",
            "expected_dim": 384,
            "l2_normalize": l2_normalize,
            "cache_embeddings": cache_embeddings,
        },
        "AB1": {
            "name": "AB1",
            "description": "DINOv3 ViT-S/16 grid-matched (32x32 grid, 384-D)",
            "provider": "timm",
            "model_name": "vit_small_patch16_dinov3.lvd1689m",
            "image_size": 512,
            "layers": [-1],
            "expected_grid": "32x32",
            "expected_dim": 384,
            "l2_normalize": l2_normalize,
            "cache_embeddings": cache_embeddings,
        },
        "AB2": {
            "name": "AB2",
            "description": "DINOv3 ViT-S/16 high-res RoPE (48x48 grid, 384-D)",
            "provider": "timm",
            "model_name": "vit_small_patch16_dinov3.lvd1689m",
            "image_size": 768,
            "layers": [-1],
            "expected_grid": "48x48",
            "expected_dim": 384,
            "l2_normalize": l2_normalize,
            "cache_embeddings": cache_embeddings,
        },
        "AB3": {
            "name": "AB3",
            "description": "DINOv3 ViT-B/16 higher-capacity (32x32 grid, 768-D)",
            "provider": "timm",
            "model_name": "vit_base_patch16_dinov3.lvd1689m",
            "image_size": 512,
            "layers": [-1],
            "expected_grid": "32x32",
            "expected_dim": 768,
            "l2_normalize": l2_normalize,
            "cache_embeddings": cache_embeddings,
        },
    }


def print_dinov3_ab_matrix(
    specs: dict[str, dict[str, Any]],
    selected: list[str],
    seeds: list[int],
    cfg: dict[str, Any],
) -> None:
    """Print the AB0-AB3 DINOv3 A/B benchmark matrix cleanly for dry-run inspection."""
    print(f"Experiment: {cfg['experiment']['name']} (DINOv3 A/B Benchmark)")
    print(f"Seeds: {seeds} | Selected: {selected}")
    print("-" * 116)
    header = (
        f"{'ID':<5} {'Provider':<17} {'Model':<38} {'Res':<9} "
        f"{'Grid':<8} {'Dim':<6} {'Cache':<7} {'Layers'}"
    )
    print(header)
    print("-" * 116)
    for exp_id in ALL_DINOV3_AB:
        spec = specs[exp_id]
        mark = "*" if exp_id in selected else " "
        res_str = f"{spec['image_size']}x{spec['image_size']}"
        print(
            f"{exp_id:<4}{mark} {spec['provider']:<17} {spec['model_name']:<38} "
            f"{res_str:<9} {spec['expected_grid']:<8} {spec['expected_dim']:<6} "
            f"{str(spec['cache_embeddings']):<7} {spec['layers']}"
        )
    print("-" * 116)


def _resolve_dynamic_choices(
    spec: dict[str, Any],
    seed_metrics: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    """Resolve 'best_of_e3_e4' and 'best_of_e5_e6' using prior seed pixel_aupr."""
    wrn_metric = spec["wrn_metric"]
    if wrn_metric == "best_of_e3_e4":
        e3_aupr = seed_metrics.get("E3", {}).get("pixel_aupr", float("-inf"))
        e4_aupr = seed_metrics.get("E4", {}).get("pixel_aupr", float("-inf"))
        if np.isnan(e3_aupr):
            e3_aupr = float("-inf")
        if np.isnan(e4_aupr):
            e4_aupr = float("-inf")
        wrn_metric = "euclidean" if e3_aupr > e4_aupr else "mahalanobis"

    calibration = spec["calibration"]
    if calibration == "best_of_e5_e6":
        e5_aupr = seed_metrics.get("E5", {}).get("pixel_aupr", float("-inf"))
        e6_aupr = seed_metrics.get("E6", {}).get("pixel_aupr", float("-inf"))
        if np.isnan(e5_aupr):
            e5_aupr = float("-inf")
        if np.isnan(e6_aupr):
            e6_aupr = float("-inf")
        calibration = "zscore" if e5_aupr > e6_aupr else "robust_iqr"

    return wrn_metric, calibration


def _collect_split_batches(loader: torch.utils.data.DataLoader) -> dict[str, Any]:
    """Collect images, masks, labels, and paths from a DataLoader."""
    images_list, masks_list, labels_list, paths_list = [], [], [], []
    for imgs, masks, labels, paths in loader:
        images_list.append(imgs)
        masks_list.append(masks)
        labels_list.append(labels)
        paths_list.extend(list(paths))
    return {
        "images": torch.cat(images_list, dim=0),
        "masks": torch.cat(masks_list, dim=0),
        "labels": torch.cat(labels_list, dim=0).numpy(),
        "paths": paths_list,
    }


def _build_stream_extractor(
    stream_name: str,
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
) -> Any:
    """Instantiate the FeatureStreamExtractor for a single stream."""
    from src.features import DINOv2MultiLayerStream, WideResNet50Stream

    if stream_name == "wrn_l2l3":
        wrn_cfg = cfg["streams"]["wrn_l2l3"]
        reducer_cfg = wrn_cfg.get("reducer", {})
        return WideResNet50Stream(
            backbone=str(wrn_cfg.get("backbone", "wide_resnet50_2")),
            layers=tuple(wrn_cfg.get("layers", ["layer2", "layer3"])),
            local_pool_kernel=int(wrn_cfg.get("local_pool_kernel", 3)),
            reducer_type=str(reducer_cfg.get("type", "incremental_pca")),
            out_dim=int(reducer_cfg.get("out_dim", 384)),
            seed=seed,
            pretrained=True,
            device=device,
            stream_name="wrn_l2l3",
        )

    dino_cfg = cfg["streams"]["dino_multilayer"]
    block_idx = int(stream_name.replace("dino_b", ""))
    backbone = str(dino_cfg.get("backbone", "dinov2_vits14"))
    provider = str(dino_cfg.get("provider", "torchhub_dinov2"))
    l2_normalize = bool(dino_cfg.get("l2_normalize", True))
    if provider != "torchhub_dinov2" or "dinov3" in backbone.lower():
        from src.features import DINOv3Backbone

        return DINOv3Backbone(
            model_name=backbone,
            provider=provider,
            layers=[block_idx - 1],
            l2_normalize=l2_normalize,
            device=device,
        )
    return DINOv2MultiLayerStream(
        backbone=backbone,
        blocks=(block_idx,),
        l2_normalize=l2_normalize,
        device=device,
    )


def _score_loader(
    extractor: Any,
    dpmm: Any,
    loader: torch.utils.data.DataLoader,
    stream_name: str,
    metric_key: str,
    device: torch.device,
) -> torch.Tensor:
    """Score all images in a DataLoader for a single stream without mutating state."""
    maps = []
    with torch.no_grad():
        for imgs, _, _, _ in loader:
            imgs = imgs.to(device)
            scored = dpmm.score_images(extractor, imgs, metric_overrides={stream_name: metric_key})
            maps.append(scored[stream_name].cpu())
    return torch.cat(maps, dim=0)


def _train_or_get_stream_cache(
    seed: int,
    stream_name: str,
    wrn_metric: str | None,
    cfg: dict[str, Any],
    dataloaders: dict[str, Any],
    device: torch.device,
    model_cache: dict[tuple[int, str], tuple[Any, Any]],
    score_cache: dict[tuple[int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Train or reuse a per-stream DPMM and score val_normal and test splits."""
    from src.models import MultiStreamDPMM, StreamDPMMConfig

    metric_key = str(wrn_metric) if stream_name == "wrn_l2l3" else "cosine"
    score_key = (seed, stream_name, metric_key)
    if score_key in score_cache:
        return score_cache[score_key]

    model_key = (seed, stream_name)
    if model_key not in model_cache:
        extractor = _build_stream_extractor(stream_name, cfg, seed=seed, device=device)
        if stream_name == "wrn_l2l3":
            stream_section = cfg["streams"]["wrn_l2l3"]
            mahalanobis_eps = float(stream_section.get("mahalanobis_eps", 1e-4))
        else:
            stream_section = cfg["streams"]["dino_multilayer"]
            mahalanobis_eps = 1e-4

        dpmm_cfg = stream_section.get("dpmm", {})
        feature_dims = getattr(extractor, "feature_dims", None)
        stream_dim = (
            int(feature_dims[stream_name])
            if isinstance(feature_dims, dict) and stream_name in feature_dims
            else int(extractor.embed_dim)
        )
        stream_cfg = StreamDPMMConfig(
            name=stream_name,
            dim=stream_dim,
            max_components=int(dpmm_cfg.get("max_components", 500)),
            epochs=int(dpmm_cfg.get("epochs", 40)),
            gamma=float(dpmm_cfg.get("gamma", 0.2)),
            weight_threshold=float(dpmm_cfg.get("weight_threshold", 1e-6)),
            scoring_metric=metric_key,
            mahalanobis_eps=mahalanobis_eps,
        )
        dpmm = MultiStreamDPMM({stream_name: stream_cfg}, device=device)
        # Strictly fit reducer and DPMM on train_normal; select best epoch on val_normal.
        dpmm.fit(
            extractor,
            dataloaders["train_normal"],
            val_normal_loader=dataloaders["val_normal"],
            seed=seed,
            cache_embeddings=bool(dpmm_cfg.get("cache_embeddings", False)),
        )
        model_cache[model_key] = (extractor, dpmm)
    else:
        extractor, dpmm = model_cache[model_key]

    val_maps = _score_loader(extractor, dpmm, dataloaders["val_normal"], stream_name, metric_key, device)
    test_maps = _score_loader(extractor, dpmm, dataloaders["test"], stream_name, metric_key, device)

    entry = {
        "model": dpmm,
        "active_prototypes": dpmm.get_active_prototype_counts().get(stream_name, 0),
        "val_maps": val_maps,
        "test_maps": test_maps,
    }
    score_cache[score_key] = entry
    return entry


def run_dinov3_ab(args: argparse.Namespace) -> None:
    """Execute head-to-head DINOv2 vs DINOv3 A/B benchmark (AB0-AB3)."""
    config_path = args.config
    if config_path == "configs/hybrid_multistream.yaml" and (REPO_ROOT / "configs/dinov3.yaml").exists():
        config_path = "configs/dinov3.yaml"
    cfg = load_config(config_path)
    specs = build_dinov3_ab_specs(cfg)

    selected = [e.upper() for e in args.experiments] if args.experiments else list(ALL_DINOV3_AB)
    for exp_id in selected:
        if exp_id not in specs:
            raise ValueError(f"Unknown DINOv3 A/B experiment '{exp_id}'. Valid choices: {list(ALL_DINOV3_AB)}")

    seeds = list(args.seeds) if args.seeds else [int(cfg["experiment"].get("seed", 42))]

    if args.dry_run:
        print_dinov3_ab_matrix(specs, selected, seeds, cfg)
        return

    from src.features import DINOv3Backbone
    from src.fusion import CalibrationFusionPipeline, FusionHead, build_calibrator
    from src.models import MultiStreamDPMM, StreamDPMMConfig

    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dpmm_cfg = cfg.get("dpmm") or cfg.get("streams", {}).get("dino_multilayer", {}).get("dpmm", {})
    metric_key = str(dpmm_cfg.get("scoring_metric", "cosine"))
    fusion_cfg = cfg.get("fusion", {})
    calibration = str(fusion_cfg.get("calibration", "none"))
    fusion_mode = str(fusion_cfg.get("mode", "max"))
    lse_tau = float(fusion_cfg.get("lse_tau", 2.0))
    fpr_targets = tuple(float(x) for x in fusion_cfg.get("fpr_targets", [0.01, 0.05, 0.10]))
    image_score_agg = str(fusion_cfg.get("image_score_aggregation", "top_k_mean"))
    top_k_frac = float(fusion_cfg.get("image_score_top_k_fraction", 0.005))
    blur_sigma = float(fusion_cfg.get("gaussian_blur_sigma", 2.0))

    data_cache: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    all_results: list[dict[str, Any]] = []

    for seed in seeds:
        pin_all_seeds(seed)

        for exp_id in selected:
            spec = specs[exp_id]
            spec_image_size = int(spec["image_size"])
            if spec_image_size not in data_cache:
                exp_cfg = dict(cfg)
                exp_cfg["experiment"] = {**cfg.get("experiment", {}), "image_size": spec_image_size}
                dataloaders = build_multistream_dataloaders(exp_cfg)
                test_data = _collect_split_batches(dataloaders["test"])
                val_data = _collect_split_batches(dataloaders["val_normal"])
                data_cache[spec_image_size] = (dataloaders, test_data, val_data)
            else:
                dataloaders, test_data, val_data = data_cache[spec_image_size]

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            extractor = DINOv3Backbone(
                model_name=spec["model_name"],
                provider=spec["provider"],
                layers=spec["layers"],
                l2_normalize=spec["l2_normalize"],
                device=device,
            )
            stream_name = (
                next(iter(extractor.feature_dims.keys()))
                if getattr(extractor, "feature_dims", None)
                else getattr(extractor, "stream_name", "dinov3")
            )
            stream_cfg = StreamDPMMConfig(
                name=stream_name,
                dim=int(extractor.embed_dim),
                max_components=int(dpmm_cfg.get("max_components", 500)),
                epochs=int(dpmm_cfg.get("epochs", 40)),
                gamma=float(dpmm_cfg.get("gamma", 0.2)),
                weight_threshold=float(dpmm_cfg.get("weight_threshold", 1e-6)),
                scoring_metric=metric_key,
                mahalanobis_eps=float(dpmm_cfg.get("mahalanobis_eps", 1e-4)),
            )
            dpmm = MultiStreamDPMM({stream_name: stream_cfg}, device=device)
            dpmm.fit(
                extractor,
                dataloaders["train_normal"],
                val_normal_loader=dataloaders["val_normal"],
                seed=seed,
                cache_embeddings=bool(spec["cache_embeddings"]),
            )

            val_maps = _score_loader(extractor, dpmm, dataloaders["val_normal"], stream_name, metric_key, device)
            test_maps = _score_loader(extractor, dpmm, dataloaders["test"], stream_name, metric_key, device)
            val_stream_maps = {stream_name: val_maps}
            test_stream_maps = {stream_name: test_maps}

            pipeline = CalibrationFusionPipeline(
                target_size=(spec_image_size, spec_image_size),
                blur_sigma=blur_sigma,
                calibrator=build_calibrator(calibration),
                fusion_head=FusionHead(
                    mode=fusion_mode,
                    weights=None,
                    lse_tau=lse_tau,
                    image_score_aggregation=image_score_agg,
                    image_score_top_k_fraction=top_k_frac,
                    fpr_targets=fpr_targets,
                ),
            )
            pipeline.fit(val_stream_maps)
            pred = pipeline.predict(test_stream_maps)

            test_fused = pred["fused_map"]
            test_image_scores = pred["image_scores"]
            thresholds = pipeline.fusion_head.thresholds

            px_metrics = compute_pixel_metrics(test_fused, test_data["masks"], thresholds)
            img_metrics = compute_image_metrics(test_image_scores, test_data["labels"])
            strat_metrics = compute_stratified_metrics(
                test_fused, test_data["masks"], test_data["labels"], thresholds
            )

            sample_val = {k: v[:1] for k, v in val_stream_maps.items()}
            latency_ms = measure_inference_latency_ms(pipeline.predict, sample_val, device=device)
            peak_vram_gb = get_peak_vram_gb(device)

            record = {
                "experiment": exp_id,
                "seed": seed,
                "provider": spec["provider"],
                "model_name": spec["model_name"],
                "image_size": spec_image_size,
                "wrn_metric": None,
                "calibration": calibration,
                "fusion": fusion_mode,
                "active_prototypes": int(dpmm.get_active_prototype_counts().get(stream_name, 0)),
                **px_metrics,
                **img_metrics,
                "latency_ms": latency_ms,
                "peak_vram_gb": peak_vram_gb,
                "stratified": strat_metrics,
                "val_count": int(val_data["images"].shape[0]),
            }
            all_results.append(record)

            fpr_key = 0.01 if 0.01 in pred["binary_masks"] else min(pred["binary_masks"].keys())
            bin_1pct = pred["binary_masks"][fpr_key].to(torch.float32)
            save_diagnostic_panels(
                output_dir=out_dir / "diagnostics" / f"{exp_id}_seed{seed}",
                images=test_data["images"],
                gt_masks=test_data["masks"],
                image_labels=test_data["labels"],
                per_stream_maps=pred["upsampled_maps"],
                fused_maps=test_fused,
                binary_masks_1pct=bin_1pct,
                image_scores=test_image_scores,
                image_paths=test_data["paths"],
                top_n=10,
            )

    write_ablation_reports(all_results, out_dir)


def run_ablations(args: argparse.Namespace) -> None:
    """Execute selected ablation experiments across seeds and write reports."""
    if getattr(args, "dinov3_ab", False):
        run_dinov3_ab(args)
        return

    cfg = load_config(args.config)
    specs = build_experiment_specs(cfg)

    if args.baseline_only:
        selected = ["E0"]
    elif args.experiments:
        selected = [e.upper() for e in args.experiments]
    else:
        selected = list(ALL_EXPERIMENTS)

    for exp_id in selected:
        if exp_id not in specs:
            raise ValueError(f"Unknown experiment '{exp_id}'. Valid choices: {list(ALL_EXPERIMENTS)}")

    seeds = list(args.seeds) if args.seeds else [int(cfg["experiment"].get("seed", 42))]

    if args.dry_run:
        print_experiment_matrix(specs, selected, seeds, cfg)
        return

    from src.fusion import CalibrationFusionPipeline, FusionHead, build_calibrator

    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataloaders = build_multistream_dataloaders(cfg)
    test_data = _collect_split_batches(dataloaders["test"])
    val_data = _collect_split_batches(dataloaders["val_normal"])
    image_size = int(cfg["experiment"].get("image_size", 448))
    fusion_cfg = cfg.get("fusion", {})
    fpr_targets = tuple(float(x) for x in fusion_cfg.get("fpr_targets", [0.01, 0.05, 0.10]))
    image_score_agg = str(fusion_cfg.get("image_score_aggregation", "top_k_mean"))
    top_k_frac = float(fusion_cfg.get("image_score_top_k_fraction", 0.005))
    blur_sigma = float(fusion_cfg.get("gaussian_blur_sigma", 2.0))

    model_cache: dict[tuple[int, str], tuple[Any, Any]] = {}
    score_cache: dict[tuple[int, str, str], dict[str, Any]] = {}
    all_results: list[dict[str, Any]] = []
    baseline_seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        pin_all_seeds(seed)
        seed_metrics: dict[str, dict[str, Any]] = {}

        for exp_id in selected:
            spec = specs[exp_id]
            wrn_metric, calibration = _resolve_dynamic_choices(spec, seed_metrics)

            val_stream_maps: dict[str, torch.Tensor] = {}
            test_stream_maps: dict[str, torch.Tensor] = {}
            total_active_prototypes = 0

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            for s_name in spec["streams"]:
                cached = _train_or_get_stream_cache(
                    seed=seed,
                    stream_name=s_name,
                    wrn_metric=wrn_metric,
                    cfg=cfg,
                    dataloaders=dataloaders,
                    device=device,
                    model_cache=model_cache,
                    score_cache=score_cache,
                )
                val_stream_maps[s_name] = cached["val_maps"]
                test_stream_maps[s_name] = cached["test_maps"]
                total_active_prototypes += int(cached["active_prototypes"])

            pipeline = CalibrationFusionPipeline(
                target_size=(image_size, image_size),
                blur_sigma=blur_sigma,
                calibrator=build_calibrator(calibration),
                fusion_head=FusionHead(
                    mode=spec["fusion"],
                    weights=spec["weights"],
                    lse_tau=float(spec["lse_tau"]) if spec["lse_tau"] is not None else 2.0,
                    image_score_aggregation=image_score_agg,
                    image_score_top_k_fraction=top_k_frac,
                    fpr_targets=fpr_targets,
                ),
            )
            # Calibrate and fit FPR thresholds strictly on val_normal.
            pipeline.fit(val_stream_maps)
            pred = pipeline.predict(test_stream_maps)

            test_fused = pred["fused_map"]
            test_image_scores = pred["image_scores"]
            thresholds = pipeline.fusion_head.thresholds

            px_metrics = compute_pixel_metrics(test_fused, test_data["masks"], thresholds)
            img_metrics = compute_image_metrics(test_image_scores, test_data["labels"])
            strat_metrics = compute_stratified_metrics(
                test_fused, test_data["masks"], test_data["labels"], thresholds
            )

            sample_val = {k: v[:1] for k, v in val_stream_maps.items()}
            latency_ms = measure_inference_latency_ms(pipeline.predict, sample_val, device=device)
            peak_vram_gb = get_peak_vram_gb(device)

            record = {
                "experiment": exp_id,
                "seed": seed,
                "wrn_metric": wrn_metric,
                "calibration": calibration,
                "fusion": spec["fusion"],
                "active_prototypes": total_active_prototypes,
                **px_metrics,
                **img_metrics,
                "latency_ms": latency_ms,
                "peak_vram_gb": peak_vram_gb,
                "stratified": strat_metrics,
                "val_count": int(val_data["images"].shape[0]),
            }
            all_results.append(record)
            seed_metrics[exp_id] = record

            if exp_id == "E0":
                baseline_seed_results.append(record)

            # Save diagnostic panels for E0, E4, E8 (or any selected final model).
            if exp_id in {"E0", "E4", "E8"}:
                fpr_key = 0.01 if 0.01 in pred["binary_masks"] else min(pred["binary_masks"].keys())
                bin_1pct = pred["binary_masks"][fpr_key].to(torch.float32)
                save_diagnostic_panels(
                    output_dir=out_dir / "diagnostics" / f"{exp_id}_seed{seed}",
                    images=test_data["images"],
                    gt_masks=test_data["masks"],
                    image_labels=test_data["labels"],
                    per_stream_maps=pred["upsampled_maps"],
                    fused_maps=test_fused,
                    binary_masks_1pct=bin_1pct,
                    image_scores=test_image_scores,
                    image_paths=test_data["paths"],
                    top_n=10,
                )

    if args.baseline_only and baseline_seed_results:
        write_baseline_metrics(baseline_seed_results, out_dir / "baseline_metrics.json")
    else:
        write_ablation_reports(all_results, out_dir)
        if baseline_seed_results:
            write_baseline_metrics(baseline_seed_results, out_dir / "baseline_metrics.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled E0-E8 multi-stream ablations.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hybrid_multistream.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Subset of experiments to run (e.g. E0 E4 E8). Defaults to all E0-E8.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds to run. Defaults to experiment.seed in config.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only E0 baseline and write reports/baseline_metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for reports and diagnostic figures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('auto', 'cpu', 'cuda', 'cuda:0', ...).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config, build and print the experiment matrix without loading models or data.",
    )
    parser.add_argument(
        "--dinov3-ab",
        action="store_true",
        help="Run the head-to-head DINOv2 vs DINOv3 A/B benchmark matrix (AB0-AB3).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_ablations(parse_args())
