"""Offline CPU unit tests for DINOv3Backbone, FP16 embedding cache, and A/B specs."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ablations import build_dinov3_ab_specs, load_config
from src.data_multistream import XRaySplitDataset
from src.features.base import MultiStreamFeatureExtractor
from src.features.dinov3_backbone import DINOv3Backbone
from src.features.wideresnet import WideResNet50Stream
from src.fusion.calibrator import CalibrationFusionPipeline, FusionHead, ZScoreCalibrator
from src.models.multi_stream_dpmm import MultiStreamDPMM, StreamDPMMConfig


class _MockPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)


class _MockTimmViT(nn.Module):
    """Lightweight mock timm ViT implementing ``forward_intermediates``."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 384,
        num_register_tokens: int = 4,
        num_blocks: int = 12,
        proj: nn.Conv2d | None = None,
    ) -> None:
        super().__init__()
        self.patch_embed = _MockPatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        if proj is not None:
            self.patch_embed.proj = proj
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self.call_count = 0

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: list[int],
        return_prefix_tokens: bool = False,
        norm: bool = True,
        output_fmt: str = "NCHW",
        intermediates_only: bool = True,
    ) -> list[torch.Tensor]:
        del return_prefix_tokens, norm, output_fmt, intermediates_only
        self.call_count += 1
        base = self.patch_embed.proj(x)
        return [base * float(idx + 1) for idx in indices]


class _MockTransformersViT(nn.Module):
    """Lightweight mock HuggingFace ViT returning ``hidden_states`` with prefix tokens."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 384,
        num_register_tokens: int = 4,
        num_blocks: int = 12,
        proj: nn.Conv2d | None = None,
    ) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            patch_size=patch_size,
            hidden_size=embed_dim,
            num_register_tokens=num_register_tokens,
            num_hidden_layers=num_blocks,
        )
        self.proj = (
            proj
            if proj is not None
            else nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        )
        self.num_register_tokens = num_register_tokens
        self.num_blocks = num_blocks
        self.call_count = 0

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> Any:
        del output_hidden_states
        self.call_count += 1
        base = self.proj(pixel_values)  # [B, C, H_p, W_p]
        bsz, c_dim, h_p, w_p = base.shape
        patch_tokens = base.permute(0, 2, 3, 1).reshape(bsz, h_p * w_p, c_dim)
        n_prefix = 1 + self.num_register_tokens
        prefix = torch.full(
            (bsz, n_prefix, c_dim),
            fill_value=999.0,
            dtype=patch_tokens.dtype,
            device=patch_tokens.device,
        )

        # hidden_states[0] is patch embedding; hidden_states[idx + 1] is block idx output.
        hidden_states = [torch.cat([prefix, patch_tokens], dim=1)]
        for idx in range(self.num_blocks):
            block_tokens = patch_tokens * float(idx + 1)
            hidden_states.append(torch.cat([prefix, block_tokens], dim=1))
        return types.SimpleNamespace(hidden_states=tuple(hidden_states))


class _MockTorchHubDINOv2(nn.Module):
    """Lightweight mock torch.hub DINOv2 ViT implementing ``get_intermediate_layers``."""

    def __init__(
        self,
        patch_size: int = 14,
        embed_dim: int = 384,
        num_register_tokens: int = 0,
        num_blocks: int = 12,
    ) -> None:
        super().__init__()
        self.patch_embed = _MockPatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self.call_count = 0

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: list[int],
        reshape: bool = True,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> list[torch.Tensor]:
        del reshape, return_class_token, norm
        self.call_count += 1
        base = self.patch_embed.proj(x)
        return [base * float(idx + 1) for idx in n]


class TestDINOv3Backbone(unittest.TestCase):
    """Offline CPU unit test suite for DINOv3Backbone and integrated pipeline."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def _assert_unit_l2_float32_finite(self, feat: torch.Tensor) -> None:
        self.assertEqual(feat.dtype, torch.float32)
        self.assertTrue(torch.isfinite(feat).all())
        norms = torch.linalg.vector_norm(feat, ord=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0))

    def test_timm_vits16_resolutions_norms_and_single_forward_multilayer(self) -> None:
        mock_timm = _MockTimmViT(patch_size=16, embed_dim=384, num_register_tokens=4)
        backbone = DINOv3Backbone(
            model_name="vit_small_patch16_dinov3.lvd1689m",
            provider="timm",
            layers=[3, 7, 11],
            l2_normalize=True,
            device="cpu",
            model=mock_timm,
        )

        x_512 = torch.randn(2, 3, 512, 512)
        out_512 = backbone.extract_patches(x_512)
        self.assertEqual(mock_timm.call_count, 1)
        self.assertEqual(set(out_512.keys()), {3, 7, 11})
        for layer_idx, feat in out_512.items():
            self.assertEqual(feat.shape, (2, 384, 32, 32), msg=f"layer={layer_idx}")
            self._assert_unit_l2_float32_finite(feat)

        x_768 = torch.randn(2, 3, 768, 768)
        stream_out_768 = backbone.extract(x_768)
        self.assertEqual(mock_timm.call_count, 2)
        self.assertEqual(list(stream_out_768.keys()), ["dino_b4", "dino_b8", "dino_b12"])
        for name, feat in stream_out_768.items():
            self.assertEqual(feat.shape, (2, 384, 48, 48), msg=f"stream={name}")
            self._assert_unit_l2_float32_finite(feat)

    def test_vitb16_dynamic_embed_dim_flows_into_multistream_dpmm(self) -> None:
        mock_timm_b = _MockTimmViT(patch_size=16, embed_dim=768, num_register_tokens=4)
        backbone = DINOv3Backbone(
            model_name="vit_base_patch16_dinov3.lvd1689m",
            provider="timm",
            layers=[-1],
            l2_normalize=True,
            device="cpu",
            model=mock_timm_b,
        )

        self.assertEqual(backbone.embed_dim, 768)
        self.assertEqual(backbone.feature_dims, {"dino_b12": 768})

        x = torch.randn(2, 3, 512, 512)
        feats = backbone.extract(x)
        self.assertEqual(feats["dino_b12"].shape, (2, 768, 32, 32))
        self._assert_unit_l2_float32_finite(feats["dino_b12"])

        stream_cfgs = {
            name: StreamDPMMConfig(name=name, dim=dim, max_components=8, epochs=2)
            for name, dim in backbone.feature_dims.items()
        }
        dpmm = MultiStreamDPMM(stream_cfgs, device="cpu")
        self.assertEqual(dpmm.models["dino_b12"].D, 768)

        train_loader = [(torch.randn(2, 3, 64, 64),)]
        dpmm.fit(backbone, train_loader, seed=42)
        scores = dpmm.score_images(backbone, x)
        self.assertEqual(scores["dino_b12"].shape, (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(scores["dino_b12"]).all())

    def test_resolution_divisibility_validation(self) -> None:
        mock_timm = _MockTimmViT(patch_size=16, embed_dim=384, num_register_tokens=4)
        backbone = DINOv3Backbone(
            model_name="vit_small_patch16_dinov3.lvd1689m",
            provider="timm",
            layers=[-1],
            device="cpu",
            model=mock_timm,
        )
        with self.assertRaises(ValueError):
            backbone.extract(torch.randn(2, 3, 500, 500))

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                XRaySplitDataset(tmp_dir, image_size=500, split="train_normal", patch_size=16)

    def test_cross_backend_parity_timm_vs_transformers(self) -> None:
        shared_proj = nn.Conv2d(3, 384, kernel_size=16, stride=16)
        mock_timm = _MockTimmViT(
            patch_size=16,
            embed_dim=384,
            num_register_tokens=4,
            num_blocks=12,
            proj=shared_proj,
        )
        mock_hf = _MockTransformersViT(
            patch_size=16,
            embed_dim=384,
            num_register_tokens=4,
            num_blocks=12,
            proj=shared_proj,
        )

        timm_bb = DINOv3Backbone(
            model_name="vit_small_patch16_dinov3.lvd1689m",
            provider="timm",
            layers=[3, -1],
            l2_normalize=True,
            device="cpu",
            model=mock_timm,
        )
        hf_bb = DINOv3Backbone(
            model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            provider="transformers",
            layers=[3, -1],
            l2_normalize=True,
            device="cpu",
            model=mock_hf,
        )

        x = torch.randn(2, 3, 512, 512)
        timm_out = timm_bb.extract(x)
        hf_out = hf_bb.extract(x)

        self.assertEqual(timm_out.keys(), hf_out.keys())
        prototypes = F.normalize(torch.randn(16, 384), dim=1)

        for stream_name in timm_out:
            f_timm = timm_out[stream_name]
            f_hf = hf_out[stream_name]
            self.assertEqual(f_timm.shape, (2, 384, 32, 32))
            self.assertEqual(f_hf.shape, (2, 384, 32, 32))
            self.assertTrue(torch.allclose(f_timm, f_hf, atol=1e-5))

            flat_timm = f_timm.permute(0, 2, 3, 1).reshape(-1, 384)
            flat_hf = f_hf.permute(0, 2, 3, 1).reshape(-1, 384)
            rank_timm = torch.argmax(flat_timm @ prototypes.T, dim=1)
            rank_hf = torch.argmax(flat_hf @ prototypes.T, dim=1)
            self.assertTrue(torch.equal(rank_timm, rank_hf))

    def test_torchhub_dinov2_backward_compatibility(self) -> None:
        mock_v2 = _MockTorchHubDINOv2(
            patch_size=14, embed_dim=384, num_register_tokens=0, num_blocks=12
        )
        backbone = DINOv3Backbone(
            model_name="dinov2_vits14",
            provider="torchhub_dinov2",
            layers=[-1],
            l2_normalize=True,
            device="cpu",
            model=mock_v2,
        )
        self.assertEqual(backbone.patch_size, 14)
        self.assertEqual(backbone.num_register_tokens, 0)

        out = backbone.extract(torch.randn(2, 3, 448, 448))
        feat = out["dino_b12"]
        self.assertEqual(feat.shape, (2, 384, 32, 32))
        self._assert_unit_l2_float32_finite(feat)

    def test_fp16_embedding_cache_ram_and_disk(self) -> None:
        train_ds = TensorDataset(torch.randn(4, 3, 64, 64))
        val_ds = TensorDataset(torch.randn(2, 3, 64, 64))
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
        expected_calls = len(train_loader) + len(val_loader)
        test_images = torch.randn(2, 3, 64, 64)

        # 1. RAM cache mode (cache_dir=None)
        mock_ram = _MockTimmViT(patch_size=16, embed_dim=384, num_register_tokens=4)
        bb_ram = DINOv3Backbone(
            model_name="vit_small_patch16_dinov3.lvd1689m",
            provider="timm",
            layers=[-1],
            device="cpu",
            model=mock_ram,
        )
        dpmm_ram = MultiStreamDPMM(
            [StreamDPMMConfig(name="dino_b12", dim=384, max_components=12, epochs=4)],
            device="cpu",
        )
        dpmm_ram.fit(
            bb_ram,
            train_loader,
            val_normal_loader=val_loader,
            seed=42,
            cache_embeddings=True,
            cache_dir=None,
        )
        self.assertEqual(mock_ram.call_count, expected_calls)
        dpmm_ram._validate_finite_parameters()
        ram_maps = dpmm_ram.score_images(bb_ram, test_images)
        self.assertEqual(ram_maps["dino_b12"].shape, (2, 1, 4, 4))
        self.assertTrue(torch.isfinite(ram_maps["dino_b12"]).all())

        # 2. Disk cache mode (cache_dir=tmp_dir)
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_disk = _MockTimmViT(patch_size=16, embed_dim=384, num_register_tokens=4)
            bb_disk = DINOv3Backbone(
                model_name="vit_small_patch16_dinov3.lvd1689m",
                provider="timm",
                layers=[-1],
                device="cpu",
                model=mock_disk,
            )
            dpmm_disk = MultiStreamDPMM(
                [StreamDPMMConfig(name="dino_b12", dim=384, max_components=12, epochs=4)],
                device="cpu",
            )
            dpmm_disk.fit(
                bb_disk,
                train_loader,
                val_normal_loader=val_loader,
                seed=42,
                cache_embeddings=True,
                cache_dir=tmp_dir,
            )
            self.assertEqual(mock_disk.call_count, expected_calls)
            dpmm_disk._validate_finite_parameters()

            cached_files = sorted(Path(tmp_dir).glob("*.pt"))
            self.assertEqual(len(cached_files), expected_calls)
            loaded_batch = torch.load(cached_files[0], map_location="cpu", weights_only=True)
            self.assertEqual(loaded_batch["dino_b12"].dtype, torch.float16)

            disk_maps = dpmm_disk.score_images(bb_disk, test_images)
            self.assertEqual(disk_maps["dino_b12"].shape, (2, 1, 4, 4))
            self.assertTrue(torch.isfinite(disk_maps["dino_b12"]).all())

    def test_end_to_end_hybrid_integration_512(self) -> None:
        mock_timm = _MockTimmViT(patch_size=16, embed_dim=384, num_register_tokens=4)
        dino_stream = DINOv3Backbone(
            model_name="vit_small_patch16_dinov3.lvd1689m",
            provider="timm",
            layers=[-1],
            device="cpu",
            model=mock_timm,
        )
        wrn_stream = WideResNet50Stream(
            pretrained=False,
            reducer_type="random_projection",
            out_dim=64,
            seed=42,
            device="cpu",
        )
        extractor = MultiStreamFeatureExtractor([dino_stream, wrn_stream])

        train_images = torch.randn(2, 3, 512, 512)
        val_images = torch.randn(2, 3, 512, 512)
        test_images = torch.randn(2, 3, 512, 512)
        train_loader = [(train_images,)]
        val_loader = [(val_images,)]

        stream_cfgs = {
            name: StreamDPMMConfig(name=name, dim=dim, max_components=16, epochs=2)
            for name, dim in extractor.feature_dims.items()
        }
        dpmm = MultiStreamDPMM(stream_cfgs, device="cpu")
        dpmm.fit(
            extractor,
            train_loader,
            val_normal_loader=val_loader,
            seed=42,
            cache_embeddings=True,
        )

        val_raw_maps = dpmm.score_images(extractor, val_images)
        self.assertEqual(val_raw_maps["dino_b12"].shape, (2, 1, 32, 32))
        self.assertEqual(val_raw_maps["wrn_l2l3"].shape, (2, 1, 64, 64))

        pipeline = CalibrationFusionPipeline(
            target_size=(512, 512),
            blur_sigma=2.0,
            calibrator=ZScoreCalibrator(),
            fusion_head=FusionHead(mode="max"),
        )
        pipeline.fit(val_raw_maps)

        test_raw_maps = dpmm.score_images(extractor, test_images)
        result = pipeline.predict(test_raw_maps)
        self.assertEqual(result["fused_map"].shape, (2, 1, 512, 512))
        self.assertEqual(result["image_scores"].shape, (2,))
        self.assertTrue(torch.isfinite(result["fused_map"]).all())

    def test_dinov3_config_and_ab_specs(self) -> None:
        cfg_path = REPO_ROOT / "configs" / "dinov3.yaml"
        cfg = load_config(cfg_path)
        self.assertEqual(cfg["experiment"]["image_size"], 512)
        self.assertEqual(cfg["backbone"]["provider"], "timm")
        self.assertTrue(cfg["dpmm"]["cache_embeddings"])

        specs = build_dinov3_ab_specs(cfg)
        self.assertEqual(list(specs.keys()), ["AB0", "AB1", "AB2", "AB3"])
        self.assertEqual(specs["AB0"]["provider"], "torchhub_dinov2")
        self.assertEqual(specs["AB0"]["image_size"], 448)
        self.assertEqual(specs["AB1"]["expected_grid"], "32x32")
        self.assertEqual(specs["AB2"]["image_size"], 768)
        self.assertEqual(specs["AB2"]["expected_grid"], "48x48")
        self.assertEqual(specs["AB3"]["expected_dim"], 768)


if __name__ == "__main__":
    unittest.main()
