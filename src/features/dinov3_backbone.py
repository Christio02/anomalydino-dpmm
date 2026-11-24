"""Unified DINOv3 / DINOv2 spatial feature backbone extractor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from .base import FeatureStreamExtractor

VALID_PROVIDERS = ("timm", "transformers", "torchhub_dinov2", "torchhub_dinov3")

TIMM_MODEL_ALIASES: dict[str, str] = {
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_vits16plus": "vit_small_plus_patch16_dinov3.lvd1689m",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
}

TRANSFORMERS_MODEL_ALIASES: dict[str, str] = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}

# Known architecture defaults used only when dynamic inspection from the model
# object is unavailable.
KNOWN_MODEL_SPECS: dict[str, dict[str, int]] = {
    "vit_small_patch16_dinov3.lvd1689m": {
        "patch_size": 16,
        "embed_dim": 384,
        "num_register_tokens": 4,
        "num_blocks": 12,
    },
    "vit_small_plus_patch16_dinov3.lvd1689m": {
        "patch_size": 16,
        "embed_dim": 384,
        "num_register_tokens": 4,
        "num_blocks": 12,
    },
    "vit_base_patch16_dinov3.lvd1689m": {
        "patch_size": 16,
        "embed_dim": 768,
        "num_register_tokens": 4,
        "num_blocks": 12,
    },
    "facebook/dinov3-vits16-pretrain-lvd1689m": {
        "patch_size": 16,
        "embed_dim": 384,
        "num_register_tokens": 4,
        "num_blocks": 12,
    },
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": {
        "patch_size": 16,
        "embed_dim": 384,
        "num_register_tokens": 4,
        "num_blocks": 12,
    },
    "facebook/dinov3-vitb16-pretrain-lvd1689m": {
        "patch_size": 16,
        "embed_dim": 768,
        "num_register_tokens": 4,
        "num_blocks": 12,
    },
    "dinov2_vits14": {
        "patch_size": 14,
        "embed_dim": 384,
        "num_register_tokens": 0,
        "num_blocks": 12,
    },
    "dinov2_vitb14": {
        "patch_size": 14,
        "embed_dim": 768,
        "num_register_tokens": 0,
        "num_blocks": 12,
    },
    "dinov2_vitl14": {
        "patch_size": 14,
        "embed_dim": 1024,
        "num_register_tokens": 0,
        "num_blocks": 24,
    },
    "dinov2_vits14_reg": {
        "patch_size": 14,
        "embed_dim": 384,
        "num_register_tokens": 4,
        "num_blocks": 12,
    },
}


def resolve_model_name(model_name: str, provider: str) -> str:
    """Resolve short aliases to provider-specific model identifiers."""
    if provider == "timm":
        return TIMM_MODEL_ALIASES.get(model_name, model_name)
    if provider == "transformers":
        return TRANSFORMERS_MODEL_ALIASES.get(model_name, model_name)
    return model_name


def _as_int_scalar(value: Any) -> int | None:
    """Convert an int or tuple/list of ints (e.g. patch_size=(16, 16)) to int."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (tuple, list)) and len(value) > 0:
        return int(value[0])
    return None


class DINOv3Backbone(FeatureStreamExtractor):
    """Unified frozen DINOv3/DINOv2 backbone extracting multi-layer NCHW patch maps.

    Supports ``timm``, ``transformers``, ``torchhub_dinov2``, and ``torchhub_dinov3``
    providers while exposing a single ``FeatureStreamExtractor`` interface.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        provider: str = "timm",
        layers: Sequence[int] = (-1,),
        l2_normalize: bool = True,
        eps: float = 1e-8,
        device: str | torch.device = "cpu",
        pretrained: bool = True,
        use_amp: bool = True,
        model: nn.Module | None = None,
        patch_size: int | None = None,
        embed_dim: int | None = None,
        num_register_tokens: int | None = None,
        num_blocks: int | None = None,
        stream_prefix: str = "dino",
        repo_dir: str | None = None,
        weights_path: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        super().__init__()
        if provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported provider {provider!r}. Expected one of {VALID_PROVIDERS}."
            )

        self.model_name = model_name
        self.provider = provider
        self.resolved_model_name = resolve_model_name(model_name, provider)
        self.l2_normalize = bool(l2_normalize)
        self.eps = float(eps)
        self.device = torch.device(device)
        self.pretrained = bool(pretrained)
        self.use_amp = bool(use_amp)
        self.stream_prefix = str(stream_prefix)
        self.repo_dir = repo_dir
        self.weights_path = weights_path
        self.hf_token = hf_token

        if model is None:
            model = self._load_provider_model()

        model.requires_grad_(False)
        model.eval()
        model.to(self.device)
        self.model = model

        self._patch_size = self._infer_patch_size(patch_size)
        self._embed_dim = self._infer_embed_dim(embed_dim)
        self._num_register_tokens = self._infer_num_register_tokens(num_register_tokens)
        self._num_blocks = self._infer_num_blocks(num_blocks)

        layers_list = list(layers)
        if not layers_list:
            raise ValueError("`layers` must be a non-empty sequence of integers.")
        self.layers = tuple(layers_list)
        self._resolved_indices = [self._resolve_layer_index(idx) for idx in self.layers]
        if len(set(self._resolved_indices)) != len(self._resolved_indices):
            raise ValueError(
                f"`layers` resolved to duplicate block indices: {self._resolved_indices}."
            )

        self._stream_names = [
            f"{self.stream_prefix}_b{idx + 1}" for idx in self._resolved_indices
        ]
        self._feature_dims = {name: self._embed_dim for name in self._stream_names}
        self._strides = {name: self._patch_size for name in self._stream_names}

    def _load_provider_model(self) -> nn.Module:
        """Instantiate the backbone from the configured provider."""
        if self.provider == "timm":
            try:
                import timm
            except ImportError as exc:
                raise ImportError(
                    "The `timm` package (timm>=1.0.21) is required for provider='timm'."
                ) from exc

            try:
                model = timm.create_model(
                    self.resolved_model_name,
                    pretrained=self.pretrained,
                    num_classes=0,
                    dynamic_img_size=True,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "Failed to create timm model with `dynamic_img_size=True`. "
                    "Please upgrade to timm>=1.0.21."
                ) from exc

            if not hasattr(model, "forward_intermediates"):
                raise RuntimeError(
                    f"timm model {self.resolved_model_name!r} does not support "
                    "`forward_intermediates`. Please upgrade to timm>=1.0.21."
                )
            return model

        if self.provider == "transformers":
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "The `transformers` package is required for provider='transformers'."
                ) from exc

            return AutoModel.from_pretrained(
                self.resolved_model_name,
                token=self.hf_token,
            )

        if self.provider == "torchhub_dinov2":
            return torch.hub.load("facebookresearch/dinov2", self.resolved_model_name)

        if self.provider == "torchhub_dinov3":
            if not self.repo_dir:
                raise ValueError(
                    "`repo_dir` must be provided when provider='torchhub_dinov3'."
                )
            return torch.hub.load(
                self.repo_dir,
                self.resolved_model_name,
                source="local",
                weights=self.weights_path,
            )

        raise ValueError(f"Unsupported provider {self.provider!r}.")

    def _infer_patch_size(self, explicit: int | None) -> int:
        if explicit is not None:
            return int(explicit)

        patch_embed = getattr(self.model, "patch_embed", None)
        if patch_embed is not None:
            val = _as_int_scalar(getattr(patch_embed, "patch_size", None))
            if val is not None:
                return val

        val = _as_int_scalar(getattr(self.model, "patch_size", None))
        if val is not None:
            return val

        config = getattr(self.model, "config", None)
        if config is not None:
            val = _as_int_scalar(getattr(config, "patch_size", None))
            if val is not None:
                return val

        spec = KNOWN_MODEL_SPECS.get(self.resolved_model_name)
        if spec is not None:
            return spec["patch_size"]

        name_lower = self.resolved_model_name.lower()
        if self.provider == "torchhub_dinov2" or "dinov2" in name_lower:
            return 14
        return 16

    def _infer_embed_dim(self, explicit: int | None) -> int:
        if explicit is not None:
            return int(explicit)

        for attr in ("embed_dim", "num_features"):
            val = _as_int_scalar(getattr(self.model, attr, None))
            if val is not None:
                return val

        config = getattr(self.model, "config", None)
        if config is not None:
            val = _as_int_scalar(getattr(config, "hidden_size", None))
            if val is not None:
                return val

        spec = KNOWN_MODEL_SPECS.get(self.resolved_model_name)
        if spec is not None:
            return spec["embed_dim"]

        name_lower = self.resolved_model_name.lower()
        if "vit_large" in name_lower or "vitl" in name_lower:
            return 1024
        if "vit_base" in name_lower or "vitb" in name_lower:
            return 768
        return 384

    def _infer_num_register_tokens(self, explicit: int | None) -> int:
        if explicit is not None:
            return int(explicit)

        val = _as_int_scalar(getattr(self.model, "num_register_tokens", None))
        if val is not None:
            return val

        config = getattr(self.model, "config", None)
        if config is not None:
            val = _as_int_scalar(getattr(config, "num_register_tokens", None))
            if val is not None:
                return val

        spec = KNOWN_MODEL_SPECS.get(self.resolved_model_name)
        if spec is not None:
            return spec["num_register_tokens"]

        name_lower = self.resolved_model_name.lower()
        if (
            "dinov3" in name_lower
            or self.provider == "torchhub_dinov3"
            or "_reg" in name_lower
        ):
            return 4
        return 0

    def _infer_num_blocks(self, explicit: int | None) -> int:
        if explicit is not None:
            return int(explicit)

        blocks = getattr(self.model, "blocks", None)
        if blocks is not None and hasattr(blocks, "__len__"):
            return int(len(blocks))

        config = getattr(self.model, "config", None)
        if config is not None:
            val = _as_int_scalar(getattr(config, "num_hidden_layers", None))
            if val is not None:
                return val

        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            enc_layers = getattr(encoder, "layer", None)
            if enc_layers is not None and hasattr(enc_layers, "__len__"):
                return int(len(enc_layers))

        spec = KNOWN_MODEL_SPECS.get(self.resolved_model_name)
        if spec is not None:
            return spec["num_blocks"]

        name_lower = self.resolved_model_name.lower()
        if "vit_large" in name_lower or "vitl" in name_lower:
            return 24
        return 12

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def num_register_tokens(self) -> int:
        return self._num_register_tokens

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def grid_size(self, height: int, width: int) -> tuple[int, int]:
        """Compute patch grid ``(H_p, W_p)``, enforcing exact patch-size divisibility."""
        if height <= 0 or width <= 0:
            raise ValueError(
                f"Image height and width must be positive, got ({height}, {width})."
            )
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Image spatial dimensions ({height}, {width}) must be divisible by "
                f"patch_size={self.patch_size}."
            )
        return height // self.patch_size, width // self.patch_size

    def _resolve_layer_index(self, layer: int) -> int:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError(f"Layer index must be an int, got {layer!r}.")
        resolved = layer if layer >= 0 else self.num_blocks + layer
        if resolved < 0 or resolved >= self.num_blocks:
            raise ValueError(
                f"Layer index {layer} (resolved={resolved}) is out of range for "
                f"num_blocks={self.num_blocks}."
            )
        return resolved

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

    def _forward_timm(
        self,
        x_dev: torch.Tensor,
        unique_indices: list[int],
        h_p: int,
        w_p: int,
    ) -> dict[int, torch.Tensor]:
        try:
            intermediates = self.model.forward_intermediates(
                x_dev,
                indices=unique_indices,
                return_prefix_tokens=False,
                norm=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        except TypeError:
            result = self.model.forward_intermediates(
                x_dev,
                indices=unique_indices,
                return_prefix_tokens=False,
                norm=True,
                output_fmt="NCHW",
            )
            if isinstance(result, tuple) and len(result) == 2:
                intermediates = result[1]
            else:
                intermediates = result

        if len(intermediates) != len(unique_indices):
            raise RuntimeError(
                f"Expected {len(unique_indices)} intermediate tensors from timm, "
                f"got {len(intermediates)}."
            )

        out: dict[int, torch.Tensor] = {}
        n_patches = h_p * w_p
        n_prefix = 1 + self.num_register_tokens
        for idx, feat in zip(unique_indices, intermediates):
            if feat.ndim == 3:
                feat = self._tokens_to_nchw(feat, h_p, w_p, n_patches, n_prefix)
            elif feat.ndim != 4:
                raise ValueError(
                    f"Expected 4D NCHW or 3D token tensor from timm, got {feat.ndim}D."
                )
            out[idx] = feat
        return out

    def _forward_transformers(
        self,
        x_dev: torch.Tensor,
        unique_indices: list[int],
        h_p: int,
        w_p: int,
    ) -> dict[int, torch.Tensor]:
        outputs = self.model(pixel_values=x_dev, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("transformers model did not return `hidden_states`.")

        n_patches = h_p * w_p
        n_prefix = 1 + self.num_register_tokens
        out: dict[int, torch.Tensor] = {}
        for idx in unique_indices:
            if idx == self.num_blocks - 1 and len(hidden_states) == self.num_blocks + 1:
                h = hidden_states[-1]
            else:
                h = hidden_states[idx + 1]

            patch_tokens = h[:, n_prefix:, :]
            if patch_tokens.shape[1] != n_patches:
                raise ValueError(
                    f"Token count mismatch at block {idx}: expected {n_patches} patch "
                    f"tokens (after stripping {n_prefix} prefix tokens), got "
                    f"{patch_tokens.shape[1]}."
                )
            b_sz, _, c_dim = patch_tokens.shape
            feat = (
                patch_tokens.reshape(b_sz, h_p, w_p, c_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out[idx] = feat
        return out

    def _forward_torchhub(
        self,
        x_dev: torch.Tensor,
        unique_indices: list[int],
        h_p: int,
        w_p: int,
    ) -> dict[int, torch.Tensor]:
        intermediates = self.model.get_intermediate_layers(
            x_dev,
            n=unique_indices,
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        if len(intermediates) != len(unique_indices):
            raise RuntimeError(
                f"Expected {len(unique_indices)} intermediate tensors from torch.hub, "
                f"got {len(intermediates)}."
            )

        n_patches = h_p * w_p
        n_prefix = 1 + self.num_register_tokens
        out: dict[int, torch.Tensor] = {}
        for idx, feat in zip(unique_indices, intermediates):
            if feat.ndim == 3:
                feat = self._tokens_to_nchw(feat, h_p, w_p, n_patches, n_prefix)
            elif feat.ndim != 4:
                raise ValueError(
                    f"Expected 3D token or 4D NCHW tensor from torch.hub, got {feat.ndim}D."
                )
            out[idx] = feat
        return out

    @staticmethod
    def _tokens_to_nchw(
        tokens: torch.Tensor,
        h_p: int,
        w_p: int,
        n_patches: int,
        n_prefix: int,
    ) -> torch.Tensor:
        b_sz, n_tok, c_dim = tokens.shape
        if n_tok > n_patches:
            if n_tok - n_prefix != n_patches:
                raise ValueError(
                    f"Token count mismatch: got {n_tok}, expected {n_patches} patches "
                    f"(+ {n_prefix} prefix tokens)."
                )
            tokens = tokens[:, n_prefix:, :]
        elif n_tok < n_patches:
            raise ValueError(
                f"Token count {n_tok} is smaller than expected {n_patches} patches."
            )
        return tokens.reshape(b_sz, h_p, w_p, c_dim).permute(0, 3, 1, 2).contiguous()

    def extract_patches(
        self,
        x: torch.Tensor,
        layers: Sequence[int] | None = None,
        l2_normalize: bool | None = None,
    ) -> dict[int, torch.Tensor]:
        """Extract spatial NCHW float32 patch features for requested block layers.

        Runs all requested layers in a single forward pass under
        ``torch.inference_mode()`` with optional CUDA FP16 autocast, casting every
        returned tensor to ``torch.float32`` prior to optional L2 normalization.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape {x.shape}.")

        h_p, w_p = self.grid_size(int(x.shape[2]), int(x.shape[3]))
        req_layers = list(self.layers if layers is None else layers)
        if not req_layers:
            raise ValueError("`layers` must be a non-empty sequence of integers.")
        resolved_indices = [self._resolve_layer_index(idx) for idx in req_layers]
        # Preserve order while deduplicating for the single backbone forward pass.
        unique_indices = list(dict.fromkeys(resolved_indices))
        do_norm = self.l2_normalize if l2_normalize is None else bool(l2_normalize)

        target_device = self._model_device()
        x_dev = x.to(target_device)
        amp_enabled = target_device.type == "cuda" and self.use_amp

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            if self.provider == "timm":
                raw_by_idx = self._forward_timm(x_dev, unique_indices, h_p, w_p)
            elif self.provider == "transformers":
                raw_by_idx = self._forward_transformers(x_dev, unique_indices, h_p, w_p)
            elif self.provider in ("torchhub_dinov2", "torchhub_dinov3"):
                raw_by_idx = self._forward_torchhub(x_dev, unique_indices, h_p, w_p)
            else:
                raise ValueError(f"Unsupported provider {self.provider!r}.")

        outputs: dict[int, torch.Tensor] = {}
        for req_layer, idx in zip(req_layers, resolved_indices):
            feat = raw_by_idx[idx].to(torch.float32)
            if do_norm:
                norm = torch.linalg.vector_norm(feat, ord=2, dim=1, keepdim=True)
                feat = feat / (norm + self.eps)
            outputs[req_layer] = feat

        return outputs

    def extract(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract configured spatial feature streams `{stream_name: Tensor[B, C, H_p, W_p]}`."""
        layer_feats = self.extract_patches(
            images, layers=self.layers, l2_normalize=self.l2_normalize
        )
        return {
            stream_name: layer_feats[req_layer]
            for stream_name, req_layer in zip(self.stream_names, self.layers)
        }


BackboneExtractor = DINOv3Backbone
