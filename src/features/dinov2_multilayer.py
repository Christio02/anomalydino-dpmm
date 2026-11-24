"""Multi-layer DINOv2 spatial feature stream extractor."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .base import FeatureStreamExtractor


class DINOv2MultiLayerStream(FeatureStreamExtractor):
    """Extract spatial patch features from multiple transformer blocks of DINOv2.

    Each requested 1-indexed block produces an independent stream ``dino_b{b}``
    of shape ``(B, embed_dim, H_p, W_p)`` in a single forward pass.
    """

    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        blocks: Sequence[int] = (4, 8, 12),
        l2_normalize: bool = True,
        eps: float = 1e-12,
        device: str | torch.device = "cpu",
        model: nn.Module | None = None,
        patch_size: int | None = None,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        blocks_list = list(blocks)
        if not blocks_list:
            raise ValueError(
                "`blocks` must be a non-empty sequence of positive integers."
            )
        for b in blocks_list:
            if isinstance(b, bool) or not isinstance(b, int) or b < 1:
                raise ValueError(
                    f"All entries in `blocks` must be positive 1-indexed integers, got {b!r}."
                )
        if len(set(blocks_list)) != len(blocks_list):
            raise ValueError(
                f"`blocks` must contain unique indices, got {blocks_list}."
            )

        self.backbone = backbone
        self.blocks = tuple(blocks_list)
        self.block_indices = [b - 1 for b in self.blocks]
        self.l2_normalize = bool(l2_normalize)
        self.eps = float(eps)
        self.device = torch.device(device)

        if model is None:
            model = torch.hub.load("facebookresearch/dinov2", backbone)

        model.requires_grad_(False)
        model.eval()
        model.to(self.device)
        self.model = model

        self.patch_size = int(
            patch_size if patch_size is not None else getattr(model, "patch_size", 14)
        )
        self.embed_dim = int(
            embed_dim if embed_dim is not None else getattr(model, "embed_dim", 384)
        )

        self._stream_names = [f"dino_b{b}" for b in self.blocks]
        self._feature_dims = {name: self.embed_dim for name in self._stream_names}
        self._strides = {name: self.patch_size for name in self._stream_names}

    @property
    def stream_names(self) -> list[str]:
        return list(self._stream_names)

    @property
    def feature_dims(self) -> dict[str, int]:
        return dict(self._feature_dims)

    @property
    def strides(self) -> dict[str, int]:
        return dict(self._strides)

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    def extract(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract per-block spatial features from ``images`` ``[B, C, H, W]``."""
        if images.ndim != 4:
            raise ValueError(
                f"Expected 4D image tensor [B, C, H, W], got shape {images.shape}."
            )

        target_device = self._model_device()
        x = images.to(target_device)

        _, _, h, w = x.shape
        h_crop = h - (h % self.patch_size)
        w_crop = w - (w % self.patch_size)
        if h_crop == 0 or w_crop == 0:
            raise ValueError(
                f"Image spatial dimensions ({h}, {w}) are smaller than patch_size={self.patch_size}."
            )
        if h_crop != h or w_crop != w:
            x = x[:, :, :h_crop, :w_crop]

        h_p = h_crop // self.patch_size
        w_p = w_crop // self.patch_size
        n_patches = h_p * w_p

        with torch.no_grad():
            layer_outputs = self.model.get_intermediate_layers(
                x,
                n=self.block_indices,
                reshape=True,
                return_class_token=False,
            )

        if len(layer_outputs) != len(self._stream_names):
            raise RuntimeError(
                f"Expected {len(self._stream_names)} layer outputs, got {len(layer_outputs)}."
            )

        outputs: dict[str, torch.Tensor] = {}
        n_prefix = 1 + int(getattr(self.model, "num_register_tokens", 0))

        for name, feat in zip(self._stream_names, layer_outputs):
            if feat.ndim == 3:
                b_sz, n_tok, c_dim = feat.shape
                if n_tok > n_patches:
                    if n_tok - n_prefix != n_patches:
                        raise ValueError(
                            f"Token count mismatch for '{name}': got {n_tok}, expected "
                            f"{n_patches} patches (+ {n_prefix} prefix tokens)."
                        )
                    feat = feat[:, n_prefix:, :]
                elif n_tok < n_patches:
                    raise ValueError(
                        f"Token count {n_tok} for '{name}' is smaller than {n_patches} patches."
                    )
                feat = feat.transpose(1, 2).reshape(b_sz, c_dim, h_p, w_p).contiguous()
            elif feat.ndim != 4:
                raise ValueError(
                    f"Expected 3D token or 4D spatial tensor for '{name}', got {feat.ndim}D."
                )

            if self.l2_normalize:
                norm = torch.linalg.vector_norm(feat, ord=2, dim=1, keepdim=True)
                feat = feat / (norm + self.eps)

            outputs[name] = feat

        return outputs
