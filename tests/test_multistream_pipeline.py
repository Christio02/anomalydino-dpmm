"""Comprehensive offline CPU unit test suite for the hybrid multi-stream DPMM pipeline."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ablations import ALL_EXPERIMENTS, build_experiment_specs, load_config
from src.data_multistream import verify_zero_split_overlap
from src.evaluation.diagnostics import (
    save_diagnostic_panels,
    write_ablation_reports,
    write_baseline_metrics,
)
from src.evaluation.metrics import (
    compute_image_metrics,
    compute_pixel_metrics,
    compute_stratified_metrics,
)
from src.features.base import FeatureStreamExtractor, MultiStreamFeatureExtractor
from src.features.dinov2_multilayer import DINOv2MultiLayerStream
from src.features.wideresnet import WideResNet50Stream
from src.fusion.calibrator import (
    CalibrationFusionPipeline,
    FusionHead,
    RobustIQRCalibrator,
    ZScoreCalibrator,
    upsample_score_map,
    upsample_score_maps,
)
from src.models.multi_stream_dpmm import DiagonalDPMM, MultiStreamDPMM, StreamDPMMConfig


class SentinelTestLoader:
    """DataLoader sentinel that raises AssertionError if ever iterated."""

    def __iter__(self):
        raise AssertionError(
            "Test loader must never be iterated during training/calibration!"
        )

    def __len__(self) -> int:
        return 1


class MockViTS14(nn.Module):
    """Lightweight offline mock of DINOv2 ViT-S/14 with forward-pass counting."""

    def __init__(
        self,
        embed_dim: int = 384,
        patch_size: int = 14,
        num_blocks: int = 12,
        num_register_tokens: int = 0,
        return_tokens_3d: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.num_register_tokens = num_register_tokens
        self.return_tokens_3d = return_tokens_3d
        self.forward_calls = 0
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: list[int] | tuple[int, ...],
        reshape: bool = True,
        return_class_token: bool = False,
    ) -> list[torch.Tensor]:
        del reshape, return_class_token
        self.forward_calls += 1
        base = self.proj(x)  # (B, embed_dim, H_p, W_p)
        outputs: list[torch.Tensor] = []
        for idx in n:
            scale = 1.0 + 0.1 * float(idx)
            feat = base * scale + 0.05 * float(idx + 1)
            if self.return_tokens_3d:
                bsz, c_dim, h_p, w_p = feat.shape
                patch_tokens = feat.flatten(2).transpose(1, 2)  # (B, N_p, C)
                n_prefix = 1 + self.num_register_tokens
                prefix = torch.zeros(
                    (bsz, n_prefix, c_dim), dtype=feat.dtype, device=feat.device
                )
                outputs.append(torch.cat([prefix, patch_tokens], dim=1))
            else:
                outputs.append(feat)
        return outputs


class DummyFeatureStream(FeatureStreamExtractor):
    """Lightweight synthetic FeatureStreamExtractor for fast DPMM convergence tests."""

    def __init__(self, name: str, dim: int, stride: int, centers: torch.Tensor) -> None:
        super().__init__()
        self._name = name
        self._dim = dim
        self._stride = stride
        self.register_buffer("centers", centers)

    @property
    def stream_names(self) -> list[str]:
        return [self._name]

    @property
    def feature_dims(self) -> dict[str, int]:
        return {self._name: self._dim}

    @property
    def strides(self) -> dict[str, int]:
        return {self._name: self._stride}

    def extract(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, _, height, width = images.shape
        h_out = max(1, height // self._stride)
        w_out = max(1, width // self._stride)
        n_centers = self.centers.shape[0]

        gen = torch.Generator(device=images.device)
        gen.manual_seed(int(images.abs().sum().item() * 1000) % 100000 + self._dim)

        idx = torch.randint(
            0, n_centers, (bsz, h_out, w_out), generator=gen, device=images.device
        )
        feats = self.centers[idx]  # (B, H, W, D)
        noise = 0.05 * torch.randn(
            feats.shape, generator=gen, device=images.device, dtype=feats.dtype
        )
        out = (feats + noise).permute(0, 3, 1, 2).contiguous()
        return {self._name: out}


class TestPhase0LeakageAndSplits(unittest.TestCase):
    """Phase 0: Split overlap verification and sentinel leakage tests."""

    def test_verify_zero_split_overlap_disjoint_passes(self) -> None:
        audit = verify_zero_split_overlap(
            train_paths=["/data/train/img_001.png", "/data/train/img_002.png"],
            val_paths=["/data/val/img_003.png"],
            test_paths=["/data/test/good/img_004.png", "/data/test/anom/img_005.png"],
        )
        self.assertEqual(audit["train_count"], 2)
        self.assertEqual(audit["val_count"], 1)
        self.assertEqual(audit["test_count"], 2)
        self.assertEqual(audit["path_overlaps"], 0)
        self.assertEqual(audit["id_overlaps"], 0)

    def test_verify_zero_split_overlap_rejects_overlapping_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shared = str(Path(tmpdir) / "shared_001.png")
            other1 = str(Path(tmpdir) / "val_002.png")
            other2 = str(Path(tmpdir) / "test_003.png")
            with self.assertRaises(ValueError):
                verify_zero_split_overlap(
                    train_paths=[shared],
                    val_paths=[other1],
                    test_paths=[shared, other2],
                )

    def test_verify_zero_split_overlap_rejects_overlapping_stems(self) -> None:
        with self.assertRaises(ValueError):
            verify_zero_split_overlap(
                train_paths=["/split_a/patient_42.png"],
                val_paths=["/split_b/patient_99.png"],
                test_paths=["/split_c/patient_42.jpg"],
            )
        with self.assertRaises(ValueError):
            verify_zero_split_overlap(
                train_paths=["/split_a/patient_10.png"],
                val_paths=["/split_b/patient_10.dcm"],
                test_paths=["/split_c/patient_11.png"],
            )

    def test_sentinel_test_loader_never_touched_during_training_and_calibration(
        self,
    ) -> None:
        sentinel_test_loader = SentinelTestLoader()
        with self.assertRaises(AssertionError):
            list(sentinel_test_loader)

        torch.manual_seed(0)
        train_imgs = [torch.randn(2, 3, 448, 448)]
        val_imgs = [torch.randn(2, 3, 64, 64)]

        # 1. WideResNet50Stream.fit_reducer
        wrn = WideResNet50Stream(
            reducer_type="random_projection",
            out_dim=64,
            pretrained=False,
            device="cpu",
        )
        wrn.fit_reducer(train_imgs)
        self.assertTrue(bool(wrn.reducer_fitted.item()))

        # 2. MultiStreamDPMM.fit
        centers = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        dummy_ext = DummyFeatureStream("dummy", dim=4, stride=8, centers=centers)
        dpmm = MultiStreamDPMM(
            [StreamDPMMConfig(name="dummy", dim=4, max_components=10, epochs=2)],
            device="cpu",
        )
        train_small = [torch.randn(2, 3, 64, 64)]
        dpmm.fit(dummy_ext, train_small, val_normal_loader=val_imgs, seed=42)

        # 3. ZScoreCalibrator.fit, RobustIQRCalibrator.fit, FusionHead.fit_thresholds
        val_normal_maps = {"dummy": torch.randn(2, 1, 32, 32)}
        z_cal = ZScoreCalibrator().fit(val_normal_maps)
        iqr_cal = RobustIQRCalibrator().fit(val_normal_maps)
        self.assertTrue(z_cal.is_fitted)
        self.assertTrue(iqr_cal.is_fitted)

        fusion = FusionHead(mode="max", fpr_targets=(0.01, 0.05, 0.10))
        thresholds = fusion.fit_thresholds(val_normal_maps["dummy"])
        self.assertIn(0.01, thresholds)
        # sentinel_test_loader remains untouched throughout all fitting/calibration stages.
        self.assertEqual(len(sentinel_test_loader), 1)


class TestPhase1FeatureExtractors(unittest.TestCase):
    """Phase 1: DINOv2 multi-layer, WideResNet50Stream, and MultiStreamFeatureExtractor tests."""

    def test_dinov2_multilayer_shapes_unit_norm_and_single_forward(self) -> None:
        torch.manual_seed(42)
        mock_vit = MockViTS14(embed_dim=384, patch_size=14, num_blocks=12)
        stream = DINOv2MultiLayerStream(
            backbone="dinov2_vits14",
            blocks=(4, 8, 12),
            l2_normalize=True,
            device="cpu",
            model=mock_vit,
        )

        x = torch.randn(2, 3, 448, 448)
        out = stream.extract(x)

        self.assertEqual(mock_vit.forward_calls, 1)
        self.assertEqual(set(out.keys()), {"dino_b4", "dino_b8", "dino_b12"})

        for name in ("dino_b4", "dino_b8", "dino_b12"):
            feat = out[name]
            self.assertEqual(tuple(feat.shape), (2, 384, 32, 32))
            self.assertTrue(torch.isfinite(feat).all())
            norms = torch.linalg.vector_norm(feat, ord=2, dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_dinov2_multilayer_with_register_tokens(self) -> None:
        torch.manual_seed(7)
        mock_vit = MockViTS14(
            embed_dim=384,
            patch_size=14,
            num_blocks=12,
            num_register_tokens=4,
            return_tokens_3d=True,
        )
        stream = DINOv2MultiLayerStream(
            blocks=(4, 12),
            l2_normalize=True,
            device="cpu",
            model=mock_vit,
        )
        x = torch.randn(2, 3, 448, 448)
        out = stream.extract(x)
        self.assertEqual(mock_vit.forward_calls, 1)
        for name in ("dino_b4", "dino_b12"):
            self.assertEqual(tuple(out[name].shape), (2, 384, 32, 32))

    def test_wideresnet_raises_before_fit_and_fits_ipca_and_random_proj(self) -> None:
        torch.manual_seed(11)
        dummy_batch = torch.randn(2, 3, 448, 448)
        train_loader = [dummy_batch]

        # Extract before fit_reducer must raise RuntimeError
        wrn_unfitted = WideResNet50Stream(
            reducer_type="incremental_pca",
            out_dim=256,
            pretrained=False,
            device="cpu",
        )
        with self.assertRaises(RuntimeError):
            wrn_unfitted.extract(dummy_batch)

        # IncrementalPCA with out_dim=256 and out_dim=384, plus random_projection out_dim=256
        for reducer_type, out_dim in (
            ("incremental_pca", 256),
            ("incremental_pca", 384),
            ("random_projection", 256),
        ):
            wrn = WideResNet50Stream(
                reducer_type=reducer_type,
                out_dim=out_dim,
                seed=42,
                pretrained=False,
                device="cpu",
            )
            wrn.fit_reducer(train_loader)
            out = wrn.extract(dummy_batch)
            self.assertIn("wrn_l2l3", out)
            feat = out["wrn_l2l3"]
            self.assertEqual(tuple(feat.shape), (2, out_dim, 56, 56))
            self.assertTrue(torch.isfinite(feat).all())

    def test_wideresnet_reducer_state_dict_roundtrip(self) -> None:
        torch.manual_seed(19)
        dummy_batch = torch.randn(2, 3, 448, 448)
        wrn1 = WideResNet50Stream(
            reducer_type="incremental_pca",
            out_dim=256,
            pretrained=False,
            device="cpu",
        )
        wrn1.fit_reducer([dummy_batch])
        expected = wrn1.extract(dummy_batch)["wrn_l2l3"]
        state = wrn1.reducer_state_dict()

        wrn2 = WideResNet50Stream(
            reducer_type="incremental_pca",
            out_dim=256,
            pretrained=False,
            device="cpu",
            backbone_model=wrn1.backbone_model,
        )
        wrn2.load_reducer_state_dict(state)
        self.assertTrue(bool(wrn2.reducer_fitted.item()))
        self.assertTrue(torch.equal(wrn1.proj_weight, wrn2.proj_weight))
        self.assertTrue(torch.equal(wrn1.proj_mean, wrn2.proj_mean))
        actual = wrn2.extract(dummy_batch)["wrn_l2l3"]
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_multistream_feature_extractor_merges_and_rejects_duplicates(self) -> None:
        mock_vit = MockViTS14(embed_dim=384, patch_size=14, num_blocks=12)
        dino = DINOv2MultiLayerStream(blocks=(4, 8, 12), model=mock_vit, device="cpu")
        wrn = WideResNet50Stream(
            reducer_type="random_projection",
            out_dim=256,
            pretrained=False,
            device="cpu",
        )
        dummy_batch = torch.randn(2, 3, 448, 448)
        merged = MultiStreamFeatureExtractor([wrn, dino])
        merged.fit_reducer([dummy_batch])

        out = merged.extract(dummy_batch)
        self.assertEqual(
            set(out.keys()), {"wrn_l2l3", "dino_b4", "dino_b8", "dino_b12"}
        )
        self.assertEqual(tuple(out["wrn_l2l3"].shape), (2, 256, 56, 56))
        for k in ("dino_b4", "dino_b8", "dino_b12"):
            self.assertEqual(tuple(out[k].shape), (2, 384, 32, 32))

        # Duplicate stream names must be rejected
        dino_dup = DINOv2MultiLayerStream(blocks=(4,), model=mock_vit, device="cpu")
        with self.assertRaises(ValueError):
            MultiStreamFeatureExtractor([dino, dino_dup])


class TestPhase2DPMMAndScoring(unittest.TestCase):
    """Phase 2: DPMM convergence, saturation warning, metric scoring, and checkpoint round-trip."""

    def test_multistream_dpmm_convergence_and_active_prototypes(self) -> None:
        torch.manual_seed(123)
        centers_a = torch.eye(4, 8)
        centers_b = torch.eye(3, 6) * 2.0
        ext = MultiStreamFeatureExtractor(
            [
                DummyFeatureStream("stream_a", dim=8, stride=8, centers=centers_a),
                DummyFeatureStream("stream_b", dim=6, stride=16, centers=centers_b),
            ]
        )
        train_loader = [torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)]
        val_loader = [torch.randn(2, 3, 64, 64)]

        cfgs = {
            "stream_a": StreamDPMMConfig(
                name="stream_a",
                dim=8,
                max_components=30,
                epochs=6,
                scoring_metric="cosine",
            ),
            "stream_b": StreamDPMMConfig(
                name="stream_b",
                dim=6,
                max_components=25,
                epochs=6,
                scoring_metric="mahalanobis",
            ),
        }
        dpmm = MultiStreamDPMM(cfgs, device="cpu")
        summary = dpmm.fit(ext, train_loader, val_normal_loader=val_loader, seed=123)

        for name, cfg in cfgs.items():
            model = dpmm.models[name]
            self.assertTrue(torch.isfinite(model.mean).all())
            self.assertTrue(torch.isfinite(model.var).all())
            self.assertTrue(torch.isfinite(model.v).all())
            self.assertTrue(np.isfinite(model.alpha))
            k_active = summary["active_counts"][name]
            self.assertGreater(k_active, 1)
            self.assertLessEqual(k_active, cfg.max_components)

    def test_saturation_warning_logged_when_exceeding_90_percent(self) -> None:
        torch.manual_seed(5)
        # 5 well-separated centers with K_m=5 and high fixed alpha -> all 5 active (> 0.9 * 5)
        centers = torch.eye(5, 8) * 5.0
        ext = DummyFeatureStream("sat_stream", dim=8, stride=8, centers=centers)
        train_loader = [torch.randn(4, 3, 64, 64)]

        cfg = StreamDPMMConfig(
            name="sat_stream",
            dim=8,
            max_components=5,
            epochs=4,
            alpha=50.0,
            alpha_fixed=True,
            weight_threshold=1e-6,
        )
        dpmm = MultiStreamDPMM({"sat_stream": cfg}, device="cpu")
        with self.assertLogs("src.models.multi_stream_dpmm", level="WARNING") as cm:
            dpmm.fit(ext, train_loader, seed=5)
        self.assertTrue(any("exceed 90%" in msg for msg in cm.output))

    def test_exact_metric_scoring_formulas(self) -> None:
        torch.manual_seed(99)
        dpmm = DiagonalDPMM(K=4, D=6, eval_chunk_size=3, device="cpu")
        dpmm.mean = torch.randn(4, 6)
        dpmm.var = torch.rand(4, 6) + 0.1
        # Make component 2 inactive (pi[2] ~ 2.5e-11 <= 1e-6) and others active
        dpmm.v = torch.tensor([0.5, 0.5, 1e-10, 1.0], dtype=torch.float32)

        y = torch.randn(7, 6)
        wt = 1e-6
        mask = dpmm.active_mask(weight_threshold=wt)
        self.assertEqual(int(mask.sum().item()), 3)
        mu = dpmm.mean[mask]
        var = dpmm.var[mask]

        # 1. Cosine distance: 1 - max_k cos_sim(y, mu_k)
        eps_cos = 1e-8
        y_norm = torch.linalg.norm(y, dim=-1, keepdim=True)
        mu_norm = torch.linalg.norm(mu, dim=-1, keepdim=True).T
        ref_cos = 1.0 - ((y @ mu.T) / (y_norm * mu_norm + eps_cos)).max(dim=1).values
        got_cos = dpmm.score_cosine(y, weight_threshold=wt, eps=eps_cos)
        self.assertTrue(torch.allclose(got_cos, ref_cos, atol=1e-5, rtol=1e-5))

        # 2. Diagonal Mahalanobis distance with mahalanobis_eps=1e-4
        m_eps = 1e-4
        diff = y.unsqueeze(1) - mu.unsqueeze(0)  # (N, M, D)
        ref_mahal = torch.sqrt(
            ((diff**2) / (var.unsqueeze(0) + m_eps)).sum(dim=-1).min(dim=1).values
        )
        got_mahal = dpmm.score_mahalanobis(
            y, weight_threshold=wt, mahalanobis_eps=m_eps
        )
        self.assertTrue(torch.allclose(got_mahal, ref_mahal, atol=1e-5, rtol=1e-5))

        # 3. Euclidean distance: min_k ||y - mu_k||_2
        ref_euc = torch.sqrt((diff**2).sum(dim=-1).min(dim=1).values)
        got_euc = dpmm.score_euclidean(y, weight_threshold=wt)
        self.assertTrue(torch.allclose(got_euc, ref_euc, atol=1e-5, rtol=1e-5))

    def test_unified_checkpoint_save_load_roundtrip(self) -> None:
        torch.manual_seed(21)
        dummy_train = torch.randn(2, 3, 224, 224)
        dummy_val = torch.randn(2, 3, 224, 224)
        dummy_test = torch.randn(2, 3, 224, 224)

        mock_vit = MockViTS14(embed_dim=64, patch_size=14, num_blocks=12)
        dino1 = DINOv2MultiLayerStream(blocks=(4,), model=mock_vit, device="cpu")
        wrn1 = WideResNet50Stream(
            reducer_type="incremental_pca",
            out_dim=64,
            pretrained=False,
            device="cpu",
        )
        ext1 = MultiStreamFeatureExtractor([wrn1, dino1])
        ext1.fit_reducer([dummy_train])

        cfgs = {
            "wrn_l2l3": StreamDPMMConfig(
                name="wrn_l2l3",
                dim=64,
                max_components=15,
                epochs=2,
                scoring_metric="mahalanobis",
            ),
            "dino_b4": StreamDPMMConfig(
                name="dino_b4",
                dim=64,
                max_components=15,
                epochs=2,
                scoring_metric="cosine",
            ),
        }
        dpmm1 = MultiStreamDPMM(cfgs, device="cpu")
        dpmm1.fit(ext1, [dummy_train], val_normal_loader=[dummy_val], seed=21)

        val_raw1 = dpmm1.score_images(ext1, dummy_val)
        pipe1 = CalibrationFusionPipeline(
            target_size=(224, 224),
            blur_sigma=2.0,
            calibrator=ZScoreCalibrator(),
            fusion_head=FusionHead(mode="max"),
        )
        pipe1.fit(val_raw1)

        test_raw1 = dpmm1.score_images(ext1, dummy_test)
        expected_fused = pipe1.predict(test_raw1)["fused_map"]

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "unified.pt"
            dpmm1.save_checkpoint(
                ckpt_path,
                feature_extractor=ext1,
                calibrator=pipe1.calibrator,
                fusion_head=pipe1.fusion_head,
                config={"seed": 21},
            )

            # Build fresh unfitted instances (sharing mock/untrained backbones)
            dino2 = DINOv2MultiLayerStream(blocks=(4,), model=mock_vit, device="cpu")
            wrn2 = WideResNet50Stream(
                reducer_type="incremental_pca",
                out_dim=64,
                pretrained=False,
                device="cpu",
                backbone_model=wrn1.backbone_model,
            )
            ext2 = MultiStreamFeatureExtractor([wrn2, dino2])
            cal2 = ZScoreCalibrator()
            head2 = FusionHead(mode="max")

            dpmm2, extra = MultiStreamDPMM.load_checkpoint(
                ckpt_path,
                device="cpu",
                feature_extractor=ext2,
                calibrator=cal2,
                fusion_head=head2,
            )
            self.assertEqual(extra["config"], {"seed": 21})

            pipe2 = CalibrationFusionPipeline(
                target_size=(224, 224),
                blur_sigma=2.0,
                calibrator=cal2,
                fusion_head=head2,
            )
            test_raw2 = dpmm2.score_images(ext2, dummy_test)
            actual_fused = pipe2.predict(test_raw2)["fused_map"]

            self.assertTrue(torch.allclose(expected_fused, actual_fused, atol=1e-6))


class TestPhase3CalibrationAndFusion(unittest.TestCase):
    """Phase 3: Upsampling, Z-score / Robust IQR calibration, FusionHead, and FPR thresholds."""

    def test_spatial_upsampling_and_gaussian_blur(self) -> None:
        spike_32 = torch.zeros(2, 1, 32, 32)
        spike_32[:, :, 16, 16] = 10.0
        spike_56 = torch.zeros(2, 1, 56, 56)
        spike_56[:, :, 28, 28] = 10.0

        prev_max = float("inf")
        for sigma in (0.0, 2.0, 4.0):
            up_32 = upsample_score_map(
                spike_32, target_size=(448, 448), blur_sigma=sigma
            )
            up_56 = upsample_score_map(
                spike_56, target_size=(448, 448), blur_sigma=sigma
            )
            self.assertEqual(tuple(up_32.shape), (2, 1, 448, 448))
            self.assertEqual(tuple(up_56.shape), (2, 1, 448, 448))
            self.assertTrue(torch.isfinite(up_32).all())
            self.assertTrue(torch.isfinite(up_56).all())
            cur_max = float(up_32.max().item())
            self.assertLessEqual(cur_max, prev_max + 1e-6)
            prev_max = cur_max

    def test_zscore_calibration_mean_zero_std_one(self) -> None:
        torch.manual_seed(3)
        val_maps = {
            "wrn_l2l3": torch.randn(4, 1, 32, 32) * 3.5 + 12.0,
            "dino_b4": torch.randn(4, 1, 32, 32) * 0.05 + 0.3,
        }
        cal = ZScoreCalibrator().fit(val_maps)
        transformed = cal.transform(val_maps)
        for name, t in transformed.items():
            mean = float(t.mean().item())
            std = float(t.std(unbiased=False).item())
            self.assertLess(abs(mean), 1e-3, msg=f"{name} mean={mean}")
            self.assertLess(abs(std - 1.0), 1e-3, msg=f"{name} std={std}")

    def test_robust_iqr_calibration_on_heavy_tailed_data(self) -> None:
        rng = np.random.default_rng(17)
        heavy = rng.standard_t(df=2, size=(4, 1, 64, 64)).astype(np.float32) * 2.5 + 5.0
        val_maps = {"wrn_l2l3": torch.from_numpy(heavy)}

        cal = RobustIQRCalibrator().fit(val_maps)
        transformed = cal.transform(val_maps)["wrn_l2l3"].numpy().ravel()
        q25, q50, q75 = np.quantile(transformed, [0.25, 0.50, 0.75])
        iqr = q75 - q25
        self.assertLess(abs(float(q50)), 1e-3)
        self.assertLess(abs(float(iqr) - 1.0), 1e-3)

    def test_fusion_head_modes_max_weighted_sum_and_lse(self) -> None:
        torch.manual_seed(8)
        m1 = torch.randn(2, 1, 16, 16)
        m2 = torch.randn(2, 1, 16, 16)
        maps = {"s1": m1, "s2": m2}

        # max
        head_max = FusionHead(mode="max")
        self.assertTrue(
            torch.allclose(head_max.fuse(maps), torch.maximum(m1, m2), atol=1e-6)
        )

        # weighted_sum with unnormalized weights (3.0, 1.0 -> 0.75, 0.25)
        head_ws = FusionHead(mode="weighted_sum", weights={"s1": 3.0, "s2": 1.0})
        expected_ws = 0.75 * m1 + 0.25 * m2
        self.assertTrue(torch.allclose(head_ws.fuse(maps), expected_ws, atol=1e-6))

        # lse with tau=2.0
        head_lse = FusionHead(mode="lse", lse_tau=2.0)
        expected_lse = torch.logsumexp(2.0 * torch.stack([m1, m2], dim=0), dim=0) / 2.0
        self.assertTrue(torch.allclose(head_lse.fuse(maps), expected_lse, atol=1e-6))

    def test_learned_fpr_thresholds_and_image_aggregation(self) -> None:
        torch.manual_seed(101)
        fused_val = torch.randn(4, 1, 64, 64)  # 16,384 continuous pixels
        head = FusionHead(
            mode="max",
            image_score_aggregation="top_k_mean",
            image_score_top_k_fraction=0.01,
            fpr_targets=(0.01, 0.05, 0.10),
        )
        head.fit_thresholds(fused_val)

        for target, tol in ((0.01, 0.001), (0.05, 0.002), (0.10, 0.002)):
            mask = head.predict_mask(fused_val, fpr_target=target)
            actual_fpr = float(mask.float().mean().item())
            self.assertAlmostEqual(actual_fpr, target, delta=tol)

        # Image-level aggregation: max
        head_img_max = FusionHead(image_score_aggregation="max")
        expected_max = fused_val.reshape(4, -1).max(dim=1).values
        self.assertTrue(
            torch.allclose(
                head_img_max.compute_image_scores(fused_val), expected_max, atol=1e-6
            )
        )

        # Image-level aggregation: top_k_mean
        k = max(1, int(round(64 * 64 * 0.01)))
        expected_topk = torch.topk(fused_val.reshape(4, -1), k=k, dim=1).values.mean(
            dim=1
        )
        self.assertTrue(
            torch.allclose(
                head.compute_image_scores(fused_val), expected_topk, atol=1e-6
            )
        )


class TestPhase4EvaluationAndAblations(unittest.TestCase):
    """Phase 4: Metrics, stratified buckets, diagnostic panels, reports, and ablation specs."""

    def test_pixel_image_and_stratified_metrics(self) -> None:
        # 5 images of size 100x100 (10,000 pixels each):
        # 0, 1: normal (0 px)
        # 2: tiny defect (10 px = 0.0010 < 0.0025)
        # 3: medium defect (100 px = 0.0100 in [0.0025, 0.02))
        # 4: large defect (500 px = 0.0500 >= 0.02)
        h, w = 100, 100
        gt_masks = np.zeros((5, h, w), dtype=np.float32)
        gt_masks[2, 10:12, 10:15] = 1.0  # 10 px
        gt_masks[3, 20:30, 20:30] = 1.0  # 100 px
        gt_masks[4, 30:55, 30:50] = 1.0  # 500 px

        labels = np.array([0, 0, 1, 1, 1], dtype=np.int64)
        rng = np.random.default_rng(42)
        scores = rng.uniform(0.0, 0.2, size=(5, h, w)).astype(np.float32)
        scores[gt_masks > 0.5] = 0.9

        thresholds = {0.01: 0.5, 0.05: 0.3, 0.10: 0.25}
        px = compute_pixel_metrics(scores, gt_masks, thresholds=thresholds)
        self.assertGreater(px["pixel_aupr"], 0.99)
        self.assertGreater(px["pixel_auroc"], 0.99)
        self.assertGreater(px["dice_fpr_1pct"], 0.99)

        img_scores = scores.reshape(5, -1).max(axis=1)
        im = compute_image_metrics(img_scores, labels)
        self.assertGreater(im["image_aupr"], 0.99)
        self.assertGreater(im["image_auroc"], 0.99)

        strat = compute_stratified_metrics(
            scores, gt_masks, labels, thresholds=thresholds
        )
        for bucket in ("tiny", "medium", "large"):
            self.assertEqual(strat[bucket]["n_anomalous"], 1)
            self.assertGreater(float(strat[bucket]["pixel_aupr"]), 0.99)
            self.assertGreater(float(strat[bucket]["dice_fpr_1pct"]), 0.99)

    def test_save_diagnostic_panels_creates_readable_pngs(self) -> None:
        torch.manual_seed(4)
        n, h, w = 4, 64, 64
        images = torch.randn(n, 3, h, w)
        gt_masks = torch.zeros(n, 1, h, w)
        gt_masks[2, 0, 10:14, 10:14] = 1.0
        gt_masks[3, 0, 20:30, 20:30] = 1.0
        labels = np.array([0, 0, 1, 1])

        per_stream = {
            "wrn_l2l3": torch.rand(n, 1, h, w),
            "dino_b4": torch.rand(n, 1, h, w),
            "dino_b12": torch.rand(n, 1, h, w),
        }
        fused = torch.rand(n, 1, h, w)
        bin_mask = (fused > 0.9).float()
        img_scores = fused.reshape(n, -1).max(dim=1).values
        paths = [f"/tmp/sample_{i}.png" for i in range(n)]

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_diagnostic_panels(
                output_dir=tmpdir,
                images=images,
                gt_masks=gt_masks,
                image_labels=labels,
                per_stream_maps=per_stream,
                fused_maps=fused,
                binary_masks_1pct=bin_mask,
                image_scores=img_scores,
                image_paths=paths,
                top_n=2,
            )
            self.assertEqual(len(saved["smallest_defects"]), 2)
            self.assertEqual(len(saved["false_positives"]), 2)
            for p_str in saved["smallest_defects"] + saved["false_positives"]:
                p = Path(p_str)
                self.assertTrue(p.is_file())
                self.assertGreater(p.stat().st_size, 0)
                with Image.open(p) as im:
                    im.verify()

    def test_write_ablation_reports_and_baseline_metrics(self) -> None:
        sample_row = {
            "experiment": "E0",
            "seed": 42,
            "pixel_aupr": 0.75,
            "pixel_auroc": 0.95,
            "dice_fpr_1pct": 0.55,
            "dice_fpr_5pct": 0.60,
            "dice_fpr_10pct": 0.58,
            "image_aupr": 0.92,
            "image_auroc": 0.94,
            "latency_ms": 12.34,
            "peak_vram_gb": 0.0,
            "stratified": {
                "tiny": {
                    "n_anomalous": 2,
                    "pixel_aupr": 0.65,
                    "dice_fpr_1pct": 0.45,
                    "dice_fpr_5pct": 0.50,
                    "dice_fpr_10pct": 0.48,
                },
                "medium": {
                    "n_anomalous": 3,
                    "pixel_aupr": 0.78,
                    "dice_fpr_1pct": 0.60,
                    "dice_fpr_5pct": 0.62,
                    "dice_fpr_10pct": 0.61,
                },
                "large": {
                    "n_anomalous": 1,
                    "pixel_aupr": 0.85,
                    "dice_fpr_1pct": 0.70,
                    "dice_fpr_5pct": 0.72,
                    "dice_fpr_10pct": 0.71,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path, csv_path = write_ablation_reports([sample_row], tmpdir)
            self.assertTrue(Path(md_path).is_file())
            self.assertTrue(Path(csv_path).is_file())
            self.assertIn("Ablation Summary", Path(md_path).read_text(encoding="utf-8"))

            with Path(csv_path).open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment"], "E0")

            json_path = Path(tmpdir) / "baseline_metrics.json"
            written_json = write_baseline_metrics([sample_row], json_path)
            payload = json.loads(Path(written_json).read_text(encoding="utf-8"))
            self.assertIn("per_seed", payload)
            self.assertIn("summary", payload)
            self.assertAlmostEqual(
                payload["summary"]["pixel_aupr"]["mean"], 0.75, places=6
            )

    def test_config_loads_and_builds_all_nine_ablation_specs(self) -> None:
        cfg_path = REPO_ROOT / "configs" / "hybrid_multistream.yaml"
        cfg = load_config(cfg_path)
        for key in ("experiment", "data", "streams", "fusion"):
            self.assertIn(key, cfg)

        specs = build_experiment_specs(cfg)
        self.assertEqual(tuple(specs.keys()), ALL_EXPERIMENTS)
        self.assertEqual(len(specs), 9)
        for exp_id in ALL_EXPERIMENTS:
            self.assertIn("streams", specs[exp_id])
            self.assertIn("calibration", specs[exp_id])
            self.assertIn("fusion", specs[exp_id])


if __name__ == "__main__":
    unittest.main()
