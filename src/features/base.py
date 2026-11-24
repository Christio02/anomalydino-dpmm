"""Base abstractions for single-stream and multi-stream feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn


class FeatureStreamExtractor(nn.Module, ABC):
    """Abstract base class for spatial feature stream extractors."""

    @property
    @abstractmethod
    def stream_names(self) -> list[str]:
        """Ordered list of stream names produced by this extractor."""

    @property
    @abstractmethod
    def feature_dims(self) -> dict[str, int]:
        """Mapping from stream name to channel dimension C_m."""

    @property
    @abstractmethod
    def strides(self) -> dict[str, int]:
        """Mapping from stream name to spatial downsampling stride."""

    @abstractmethod
    def extract(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract spatial feature maps `{name: Tensor[B, C_m, H_m, W_m]}`."""

    def fit_reducer(self, train_normal_loader: Any, device: Any = None) -> None:
        """Fit any dimensionality reducer on normal training images (no-op by default)."""

    def reducer_state_dict(self) -> dict[str, Any]:
        """Return serializable reducer state (empty dict by default)."""
        return {}

    def load_reducer_state_dict(self, state: dict[str, Any]) -> None:
        """Restore reducer state from a state dict (no-op by default)."""

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass delegating to :meth:`extract`."""
        return self.extract(images)


class MultiStreamFeatureExtractor(FeatureStreamExtractor):
    """Composite extractor wrapping independent :class:`FeatureStreamExtractor` modules."""

    def __init__(
        self,
        extractors: Sequence[FeatureStreamExtractor] | Mapping[str, FeatureStreamExtractor],
    ) -> None:
        super().__init__()
        if isinstance(extractors, Mapping):
            extractor_list = list(extractors.values())
        else:
            extractor_list = list(extractors)

        for ext in extractor_list:
            if not isinstance(ext, FeatureStreamExtractor):
                raise TypeError(
                    f"Expected FeatureStreamExtractor instance, got {type(ext).__name__}."
                )

        self.extractors = nn.ModuleList(extractor_list)
        # Validate uniqueness of stream names at construction time.
        _ = self.stream_names

    @property
    def stream_names(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for ext in self.extractors:
            for name in ext.stream_names:
                if name in seen:
                    raise ValueError(f"Duplicate stream name detected: '{name}'.")
                seen.add(name)
                names.append(name)
        return names

    @property
    def feature_dims(self) -> dict[str, int]:
        dims: dict[str, int] = {}
        for ext in self.extractors:
            for name, dim in ext.feature_dims.items():
                if name in dims:
                    raise ValueError(f"Duplicate stream name detected: '{name}'.")
                dims[name] = dim
        return dims

    @property
    def strides(self) -> dict[str, int]:
        merged: dict[str, int] = {}
        for ext in self.extractors:
            for name, stride in ext.strides.items():
                if name in merged:
                    raise ValueError(f"Duplicate stream name detected: '{name}'.")
                merged[name] = stride
        return merged

    def extract(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run each child extractor independently and merge stream dictionaries."""
        outputs: dict[str, torch.Tensor] = {}
        for ext in self.extractors:
            child_out = ext.extract(images)
            for name, feat in child_out.items():
                if name in outputs:
                    raise ValueError(f"Duplicate stream name detected: '{name}'.")
                outputs[name] = feat
        return outputs

    def fit_reducer(self, train_normal_loader: Any, device: Any = None) -> None:
        """Delegate reducer fitting to every child extractor."""
        for ext in self.extractors:
            ext.fit_reducer(train_normal_loader, device=device)

    def reducer_state_dict(self) -> dict[str, Any]:
        """Collect child reducer states keyed by comma-joined stream names."""
        state: dict[str, Any] = {}
        for ext in self.extractors:
            key = ",".join(ext.stream_names)
            state[key] = ext.reducer_state_dict()
        return state

    def load_reducer_state_dict(self, state: dict[str, Any]) -> None:
        """Restore child reducer states from a dictionary keyed by stream names."""
        for ext in self.extractors:
            key = ",".join(ext.stream_names)
            if key in state:
                ext.load_reducer_state_dict(state[key])
