"""WideResNet-50 multi-layer local-pooled feature stream with dimensionality reduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FeatureStreamExtractor

_VALID_REDUCER_TYPES = {"incremental_pca", "random_projection", "none"}


def _unpack_images_from_batch(batch: Any) -> torch.Tensor:
    """Extract an image tensor from a DataLoader batch (Tensor, tuple/list, or dict)."""
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)):
        if not batch or not isinstance(batch[0], torch.Tensor):
            raise TypeError("Expected first element of batch sequence to be a torch.Tensor.")
        return batch[0]
    if isinstance(batch, Mapping):
        for key in ("image", "images"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key]
        raise KeyError("Batch dict must contain an 'image' or 'images' tensor.")
    raise TypeError(f"Unsupported batch type for reducer fitting: {type(batch).__name__}.")


class WideResNet50Stream(FeatureStreamExtractor):
    """Extract multi-layer WideResNet-50 features with local pooling and linear projection.

    Extracts intermediate feature maps from ``layers`` (default ``layer2`` and
    ``layer3``), applies local average pooling, upsamples coarser maps to the
    finest layer's spatial resolution, concatenates along channels, and
    optionally projects via a fitted ``IncrementalPCA`` or
    ``GaussianRandomProjection`` reducer.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        local_pool_kernel: int = 3,
        reducer_type: str = "incremental_pca",
        out_dim: int = 384,
        seed: int = 42,
        pretrained: bool = True,
        device: str | torch.device = "cpu",
        stream_name: str = "wrn_l2l3",
        backbone_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if reducer_type not in _VALID_REDUCER_TYPES:
            raise ValueError(
                f"Invalid reducer_type {reducer_type!r}; expected one of {sorted(_VALID_REDUCER_TYPES)}."
            )
        layers_tuple = tuple(layers)
        if not layers_tuple:
            raise ValueError("`layers` must be a non-empty sequence of layer names.")

        self.backbone = backbone
        self.layers = layers_tuple
        self.local_pool_kernel = int(local_pool_kernel)
        self.reducer_type = reducer_type
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        self.stream_name = str(stream_name)
        self.device = torch.device(device)

        if backbone_model is None:
            if backbone != "wide_resnet50_2":
                raise ValueError(f"Unsupported backbone {backbone!r}; expected 'wide_resnet50_2'.")
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
            backbone_model = wide_resnet50_2(weights=weights)

        backbone_model.requires_grad_(False)
        backbone_model.eval()
        backbone_model.to(self.device)
        self.backbone_model = backbone_model

        if self.local_pool_kernel > 1:
            self.local_pool: nn.Module = nn.AvgPool2d(
                kernel_size=self.local_pool_kernel,
                stride=1,
                padding=self.local_pool_kernel // 2,
            )
        else:
            self.local_pool = nn.Identity()

        self._hook_outputs: dict[str, torch.Tensor] = {}
        self._hook_handles: list[Any] = []
        named_modules = dict(self.backbone_model.named_modules())
        for layer_name in self.layers:
            if layer_name not in named_modules:
                raise ValueError(f"Layer '{layer_name}' not found in backbone_model.")
            handle = named_modules[layer_name].register_forward_hook(
                self._make_hook(layer_name)
            )
            self._hook_handles.append(handle)

        # Dynamically infer concatenated channel dimension and spatial stride.
        with torch.no_grad():
            probe = torch.zeros(1, 3, 64, 64, device=self._model_device())
            raw_probe = self._extract_raw(probe)
        self._in_dim = int(raw_probe.shape[1])
        self._stride = max(1, 64 // int(raw_probe.shape[2]))

        self._effective_out_dim = (
            self._in_dim if self.reducer_type == "none" else self.out_dim
        )

        self.register_buffer(
            "proj_weight",
            torch.zeros(self._effective_out_dim, self._in_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "proj_mean",
            torch.zeros(self._in_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "reducer_fitted",
            torch.tensor(False, dtype=torch.bool),
        )

    def _make_hook(self, layer_name: str):
        def _hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            self._hook_outputs[layer_name] = output

        return _hook

    def _model_device(self) -> torch.device:
        try:
            return next(self.backbone_model.parameters()).device
        except StopIteration:
            return self.device

    @property
    def stream_names(self) -> list[str]:
        return [self.stream_name]

    @property
    def feature_dims(self) -> dict[str, int]:
        return {self.stream_name: self._effective_out_dim}

    @property
    def strides(self) -> dict[str, int]:
        return {self.stream_name: self._stride}

    def _extract_raw(self, images: torch.Tensor) -> torch.Tensor:
        """Run backbone forward pass and return concatenated, pooled spatial features."""
        if images.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [B, C, H, W], got shape {images.shape}.")

        x = images.to(self._model_device())
        self._hook_outputs.clear()
        try:
            with torch.no_grad():
                _ = self.backbone_model(x)
            captured = [self._hook_outputs[name] for name in self.layers]
        finally:
            # Clear captured activations immediately to avoid retaining graph/tensor refs.
            self._hook_outputs.clear()

        pooled = [self.local_pool(feat) for feat in captured]
        target_hw = pooled[0].shape[-2:]
        aligned = [
            feat
            if feat.shape[-2:] == target_hw
            else F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
            for feat in pooled
        ]
        return torch.cat(aligned, dim=1)

    def fit_reducer(
        self,
        train_normal_loader: Any,
        device: Any = None,
        max_samples: int | None = None,
    ) -> None:
        """Fit the linear dimensionality reducer on normal training images."""
        if device is not None:
            self.to(device)

        if self.reducer_type == "none":
            self.reducer_fitted.fill_(True)
            return

        if self.reducer_type == "incremental_pca":
            self._fit_incremental_pca(train_normal_loader, max_samples=max_samples)
        elif self.reducer_type == "random_projection":
            self._fit_random_projection(train_normal_loader, max_samples=max_samples)

    def _iter_flattened_vectors(
        self,
        train_normal_loader: Any,
        max_samples: int | None = None,
    ):
        total_vectors = 0
        for batch in train_normal_loader:
            if max_samples is not None and total_vectors >= max_samples:
                break
            images = _unpack_images_from_batch(batch)
            raw = self._extract_raw(images)
            vecs = (
                raw.permute(0, 2, 3, 1)
                .reshape(-1, self._in_dim)
                .detach()
                .cpu()
                .numpy()
            )
            if max_samples is not None and total_vectors + vecs.shape[0] > max_samples:
                vecs = vecs[: max_samples - total_vectors]
            if vecs.shape[0] == 0:
                continue
            total_vectors += vecs.shape[0]
            yield vecs

    def _fit_incremental_pca(
        self,
        train_normal_loader: Any,
        max_samples: int | None = None,
    ) -> None:
        from sklearn.decomposition import IncrementalPCA

        ipca = IncrementalPCA(n_components=self.out_dim)
        min_chunk_size = max(self.out_dim, 2048)

        buffer_chunks: list[np.ndarray] = []
        buffer_count = 0
        fitted_any = False

        for vecs in self._iter_flattened_vectors(train_normal_loader, max_samples=max_samples):
            buffer_chunks.append(vecs)
            buffer_count += vecs.shape[0]
            if buffer_count >= min_chunk_size:
                chunk = np.concatenate(buffer_chunks, axis=0)
                ipca.partial_fit(chunk)
                fitted_any = True
                buffer_chunks.clear()
                buffer_count = 0

        if buffer_count > 0:
            tail = np.concatenate(buffer_chunks, axis=0)
            if tail.shape[0] >= self.out_dim:
                ipca.partial_fit(tail)
                fitted_any = True
            elif not fitted_any:
                raise ValueError(
                    f"Insufficient feature vectors ({tail.shape[0]}) to fit "
                    f"IncrementalPCA with n_components={self.out_dim}."
                )

        if not fitted_any:
            raise ValueError(
                f"Insufficient feature vectors (0) to fit IncrementalPCA with n_components={self.out_dim}."
            )

        self._copy_projection_params(ipca.components_, ipca.mean_)

    def _fit_random_projection(
        self,
        train_normal_loader: Any,
        max_samples: int | None = None,
    ) -> None:
        from sklearn.random_projection import GaussianRandomProjection

        running_sum = np.zeros(self._in_dim, dtype=np.float64)
        total_count = 0
        sample_array: np.ndarray | None = None

        for vecs in self._iter_flattened_vectors(train_normal_loader, max_samples=max_samples):
            if sample_array is None:
                sample_array = vecs
            running_sum += vecs.sum(axis=0, dtype=np.float64)
            total_count += vecs.shape[0]

        if total_count == 0 or sample_array is None:
            raise ValueError("No feature vectors extracted from train_normal_loader.")

        empirical_mean = running_sum / float(total_count)
        grp = GaussianRandomProjection(n_components=self.out_dim, random_state=self.seed)
        grp.fit(sample_array)

        self._copy_projection_params(grp.components_, empirical_mean)

    def _copy_projection_params(self, components: np.ndarray, mean: np.ndarray) -> None:
        weight_tensor = torch.as_tensor(
            components, dtype=self.proj_weight.dtype, device=self.proj_weight.device
        )
        mean_tensor = torch.as_tensor(
            mean, dtype=self.proj_mean.dtype, device=self.proj_mean.device
        )
        self.proj_weight.copy_(weight_tensor)
        self.proj_mean.copy_(mean_tensor)
        self.reducer_fitted.fill_(True)

    def extract(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract and optionally project WideResNet-50 spatial features."""
        raw = self._extract_raw(images)
        if self.reducer_type == "none":
            return {self.stream_name: raw}

        if not bool(self.reducer_fitted.item()):
            raise RuntimeError(
                "WideResNet50Stream reducer is not fitted. "
                "Call fit_reducer() or load_reducer_state_dict() before extract()."
            )

        b_sz, c_in, h_2, w_2 = raw.shape
        x_flat = raw.permute(0, 2, 3, 1).reshape(-1, c_in)
        proj_mean = self.proj_mean.to(device=x_flat.device, dtype=x_flat.dtype)
        proj_weight = self.proj_weight.to(device=x_flat.device, dtype=x_flat.dtype)
        projected = (x_flat - proj_mean) @ proj_weight.t()
        out = (
            projected.reshape(b_sz, h_2, w_2, self._effective_out_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return {self.stream_name: out}

    def reducer_state_dict(self) -> dict[str, Any]:
        """Return serializable reducer buffers and metadata."""
        return {
            "proj_weight": self.proj_weight.detach().cpu().clone(),
            "proj_mean": self.proj_mean.detach().cpu().clone(),
            "reducer_fitted": self.reducer_fitted.detach().cpu().clone(),
            "reducer_type": self.reducer_type,
            "in_dim": self._in_dim,
            "out_dim": self._effective_out_dim,
        }

    def load_reducer_state_dict(self, state: dict[str, Any]) -> None:
        """Restore reducer buffers from a dictionary produced by :meth:`reducer_state_dict`."""
        if "reducer_type" in state and state["reducer_type"] != self.reducer_type:
            raise ValueError(
                f"Reducer type mismatch: expected {self.reducer_type!r}, "
                f"got {state['reducer_type']!r}."
            )
        if "in_dim" in state and int(state["in_dim"]) != self._in_dim:
            raise ValueError(
                f"Input dim mismatch: expected {self._in_dim}, got {state['in_dim']}."
            )
        if "out_dim" in state and int(state["out_dim"]) != self._effective_out_dim:
            raise ValueError(
                f"Output dim mismatch: expected {self._effective_out_dim}, got {state['out_dim']}."
            )

        self.proj_weight.copy_(
            torch.as_tensor(state["proj_weight"], dtype=self.proj_weight.dtype, device=self.proj_weight.device)
        )
        self.proj_mean.copy_(
            torch.as_tensor(state["proj_mean"], dtype=self.proj_mean.dtype, device=self.proj_mean.device)
        )
        self.reducer_fitted.copy_(
            torch.as_tensor(state["reducer_fitted"], dtype=self.reducer_fitted.dtype, device=self.reducer_fitted.device)
        )
