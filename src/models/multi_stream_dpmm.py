"""Per-stream diagonal Dirichlet Process Mixture Model (DPMM) and multi-stream container."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Beta

logger = logging.getLogger(__name__)

_VALID_METRICS = ("cosine", "mahalanobis", "euclidean")
_VALID_SCHEDULES = ("ema", "exp")


@dataclass
class StreamDPMMConfig:
    """Configuration for a single-stream DiagonalDPMM."""

    name: str
    dim: int
    max_components: int = 500
    epochs: int = 40
    gamma: float = 0.2
    schedule: str = "ema"
    weight_threshold: float = 1e-6
    scoring_metric: str = "cosine"
    mahalanobis_eps: float = 1e-4
    cov_type: str = "diag"
    reg_covar: float = 1e-6
    alpha: float = 2.0
    alpha_fixed: bool = False
    eval_chunk_size: int = 16384

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"dim must be >= 1, got {self.dim}")
        if self.max_components < 1:
            raise ValueError(f"max_components must be >= 1, got {self.max_components}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.schedule.lower() not in _VALID_SCHEDULES:
            raise ValueError(
                f"schedule must be one of {_VALID_SCHEDULES}, got {self.schedule!r}"
            )
        if self.scoring_metric.lower() not in _VALID_METRICS:
            raise ValueError(
                f"scoring_metric must be one of {_VALID_METRICS}, got {self.scoring_metric!r}"
            )
        if self.alpha < 1.0:
            raise ValueError(f"alpha must be >= 1.0, got {self.alpha}")
        if self.eval_chunk_size < 1:
            raise ValueError(
                f"eval_chunk_size must be >= 1, got {self.eval_chunk_size}"
            )
        self.schedule = self.schedule.lower()
        self.scoring_metric = self.scoring_metric.lower()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StreamDPMMConfig:
        return cls(**dict(data))


def _resolve_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested (%s) but unavailable; falling back to CPU.", dev)
        return torch.device("cpu")
    return dev


def _extract_images_from_batch(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, Mapping):
        for key in ("image", "images", "img", "x"):
            if key in batch:
                return batch[key]
        raise KeyError(
            f"Could not find image tensor key in batch mapping: {list(batch.keys())}"
        )
    if isinstance(batch, (tuple, list)) and len(batch) > 0:
        return batch[0]
    raise TypeError(f"Unsupported batch type for image extraction: {type(batch)!r}")


class DiagonalDPMM:
    """Truncated stick-breaking Gaussian DPMM with diagonal covariances.

    Stores only diagonal variance vectors ``var (K, D)`` and
    ``second_moment_stat (K, D)`` so memory never scales as ``O(N * D^2)``.
    """

    def __init__(
        self,
        K: int,
        D: int,
        update_rate: float | None = 0.2,
        schedule: str = "ema",
        alpha: float = 2.0,
        alpha_fixed: bool = False,
        eps: float | None = None,
        reg_covar: float = 1e-6,
        eval_chunk_size: int = 16384,
        device: str | torch.device = "cpu",
    ) -> None:
        if alpha < 1.0:
            raise ValueError(f"alpha needs to be >= 1.0, got {alpha}")
        if K < 1 or D < 1:
            raise ValueError(f"K and D must be >= 1, got K={K}, D={D}")
        if schedule.lower() not in _VALID_SCHEDULES:
            raise ValueError(
                f"schedule must be one of {_VALID_SCHEDULES}, got {schedule!r}"
            )

        self.K = int(K)
        self.D = int(D)
        self.update_rate = update_rate
        self.schedule = schedule.lower()
        self.alpha = float(alpha)
        self.alpha_fixed = bool(alpha_fixed)
        self.eps = (
            float(eps) if eps is not None else float(torch.finfo(torch.float32).eps)
        )
        self.reg_covar = float(reg_covar)
        self.eval_chunk_size = int(eval_chunk_size)
        self.device = _resolve_device(device)

        self.initialize()

    def _clamp_v(self, v: torch.Tensor) -> torch.Tensor:
        """Clamp stick-breaking ratios so log(v) and log(1 - v) remain finite."""
        v_out = v.clone()
        if self.K > 1:
            v_out[:-1] = torch.clamp(v_out[:-1], min=1e-12, max=1.0 - 1e-12)
        v_out[-1] = 1.0
        return v_out

    def initialize(self, data: torch.Tensor | None = None) -> None:
        """Initialize sufficient statistics, stick-breaking weights, means, and variances."""
        self.iterations = 0
        self.resp_stat = torch.zeros((self.K,), dtype=torch.float32, device=self.device)
        self.mean_stat = torch.zeros(
            (self.K, self.D), dtype=torch.float32, device=self.device
        )
        self.second_moment_stat = torch.zeros(
            (self.K, self.D), dtype=torch.float32, device=self.device
        )

        beta_dist = Beta(1.0, self.alpha)
        breaking_ratios = beta_dist.sample((self.K,)).to(
            dtype=torch.float32, device=self.device
        )
        self.v = self._clamp_v(breaking_ratios)

        self.mean = torch.randn(
            (self.K, self.D), dtype=torch.float32, device=self.device
        )
        if data is None:
            self.var = torch.full(
                (self.K, self.D),
                1.0 + self.reg_covar,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            data_f = data.to(device=self.device, dtype=torch.float32)
            n_samples = min(self.K, data_f.shape[0])
            perm = torch.randperm(data_f.shape[0], device=data_f.device)[:n_samples]
            self.mean[:n_samples, :] = data_f[perm, :]
            # unbiased=False avoids NaN when data_f.shape[0] == 1.
            data_var = torch.var(data_f, dim=0, unbiased=False) + self.reg_covar
            self.var = data_var.unsqueeze(0).expand(self.K, self.D).clone()

    def state_dict(self) -> dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "iterations": int(self.iterations),
            "K": int(self.K),
            "D": int(self.D),
            "resp_stat": self.resp_stat.detach().clone(),
            "mean_stat": self.mean_stat.detach().clone(),
            "second_moment_stat": self.second_moment_stat.detach().clone(),
            "v": self.v.detach().clone(),
            "mean": self.mean.detach().clone(),
            "var": self.var.detach().clone(),
            "reg_covar": float(self.reg_covar),
            "update_rate": self.update_rate,
            "schedule": self.schedule,
            "alpha_fixed": bool(self.alpha_fixed),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.alpha = max(1.0, float(state["alpha"]))
        self.iterations = int(state["iterations"])
        self.K = int(state["K"])
        self.D = int(state["D"])
        self.resp_stat = (
            state["resp_stat"].to(device=self.device, dtype=torch.float32).clone()
        )
        self.mean_stat = (
            state["mean_stat"].to(device=self.device, dtype=torch.float32).clone()
        )
        self.second_moment_stat = (
            state["second_moment_stat"]
            .to(device=self.device, dtype=torch.float32)
            .clone()
        )
        self.v = self._clamp_v(state["v"].to(device=self.device, dtype=torch.float32))
        self.mean = state["mean"].to(device=self.device, dtype=torch.float32).clone()
        self.var = state["var"].to(device=self.device, dtype=torch.float32).clone()
        self.reg_covar = float(state["reg_covar"])
        self.update_rate = state["update_rate"]
        self.schedule = str(state["schedule"]).lower()
        self.alpha_fixed = bool(state["alpha_fixed"])

    def get_step_size(self) -> float:
        if self.update_rate is None:
            return 1.0
        if self.schedule == "ema":
            return float(self.update_rate)
        if self.schedule == "exp":
            return float(self.iterations ** (-self.update_rate))
        raise NotImplementedError(f"Unsupported schedule: {self.schedule!r}")

    def calculate_pi(self) -> torch.Tensor:
        """Compute normalized stick-breaking mixture weights of shape (K,)."""
        v_clamped = self._clamp_v(self.v)
        if self.K == 1:
            return torch.ones(1, dtype=v_clamped.dtype, device=v_clamped.device)
        prefix_prod = torch.cat(
            [
                torch.ones(1, dtype=v_clamped.dtype, device=v_clamped.device),
                torch.cumprod(1.0 - v_clamped[:-1], dim=0),
            ]
        )
        pi = prefix_prod * v_clamped
        pi_sum = pi.sum()
        if pi_sum > 0:
            pi = pi / pi_sum
        return pi

    def calculate_log_pi(self) -> torch.Tensor:
        """Compute log stick-breaking mixture weights of shape (K,)."""
        v_clamped = self._clamp_v(self.v)
        if self.K == 1:
            return torch.zeros(1, dtype=v_clamped.dtype, device=v_clamped.device)
        prefix_sum = torch.cat(
            [
                torch.zeros(1, dtype=v_clamped.dtype, device=v_clamped.device),
                torch.cumsum(torch.log(1.0 - v_clamped[:-1]), dim=0),
            ]
        )
        return prefix_sum + torch.log(v_clamped)

    def active_mask(self, weight_threshold: float = 1e-6) -> torch.Tensor:
        """Return boolean mask (K,) of components exceeding weight_threshold."""
        pi = self.calculate_pi()
        mask = pi > weight_threshold
        if not mask.any():
            logger.warning(
                "No component exceeded weight_threshold=%g; falling back to argmax(pi).",
                weight_threshold,
            )
            mask = torch.zeros_like(mask)
            mask[torch.argmax(pi)] = True
        return mask

    def num_active_components(self, weight_threshold: float = 1e-6) -> int:
        return int(self.active_mask(weight_threshold=weight_threshold).sum().item())

    def get_weighted_log_prob(
        self,
        data: torch.Tensor,
        weight_threshold: float = 0.0,
    ) -> torch.Tensor:
        """Compute log(pi_k * N(x_n | mu_k, diag(var_k))) of shape (N, K)."""
        data_f = data.to(device=self.device, dtype=torch.float32)
        n_samples = data_f.shape[0]
        log_pi = self.calculate_log_pi()

        if weight_threshold > 0.0:
            mask = self.active_mask(weight_threshold=weight_threshold)
        else:
            mask = torch.ones(self.K, dtype=torch.bool, device=self.device)

        mu = self.mean[mask]  # (M, D)
        var = torch.clamp(self.var[mask], min=self.reg_covar)  # (M, D)
        log_pi_m = log_pi[mask]  # (M,)

        inv_var = 1.0 / var  # (M, D)
        inv_var_t = inv_var.T  # (D, M)
        mu_inv_var_t = (mu * inv_var).T  # (D, M)
        mu_quad = (mu * mu * inv_var).sum(dim=-1).unsqueeze(0)  # (1, M)
        log_det = torch.log(var).sum(dim=-1).unsqueeze(0)  # (1, M)
        const = self.D * math.log(2.0 * math.pi)

        weighted_log_prob = torch.full(
            (n_samples, self.K), -torch.inf, dtype=torch.float32, device=self.device
        )

        chunk_size = self.eval_chunk_size
        for start in range(0, n_samples, chunk_size):
            x = data_f[start : start + chunk_size]
            mahal_sq = (x * x) @ inv_var_t - 2.0 * (x @ mu_inv_var_t) + mu_quad
            mahal_sq = torch.clamp(mahal_sq, min=0.0)
            log_prob = -0.5 * (const + log_det + mahal_sq)
            weighted_log_prob[start : start + chunk_size, mask] = (
                log_pi_m.unsqueeze(0) + log_prob
            )

        return weighted_log_prob

    def sample_score(
        self,
        data: torch.Tensor,
        weight_threshold: float = 0.0,
    ) -> torch.Tensor:
        """Compute per-sample log-likelihood log sum_k pi_k N(x_n | mu_k, var_k) of shape (N,)."""
        weighted_log_prob = self.get_weighted_log_prob(
            data, weight_threshold=weight_threshold
        )
        return torch.logsumexp(weighted_log_prob, dim=1)

    def score(
        self,
        data: torch.Tensor,
        weight_threshold: float = 0.0,
    ) -> torch.Tensor:
        """Compute mean log-likelihood across all N samples."""
        return self.sample_score(data, weight_threshold=weight_threshold).mean()

    def step(self, data: torch.Tensor) -> None:
        """Perform one batched EM step on data of shape (N, D)."""
        self.iterations += 1
        data_f = data.to(device=self.device, dtype=torch.float32)
        n_samples = data_f.shape[0]
        if n_samples == 0:
            raise ValueError("Cannot perform EM step on empty batch (N=0).")

        # E-step
        weighted_log_prob = self.get_weighted_log_prob(data_f, weight_threshold=0.0)
        log_resp = weighted_log_prob - torch.logsumexp(
            weighted_log_prob, dim=1, keepdim=True
        )
        resp = torch.exp(log_resp)  # (N, K)

        # M-step sufficient statistics via BLAS gemm (no N x K x D or N x 1 x D x D tensors)
        step_size = self.get_step_size()
        decay = 1.0 - step_size
        self.resp_stat.mul_(decay)
        self.mean_stat.mul_(decay)
        self.second_moment_stat.mul_(decay)

        self.resp_stat.add_(step_size * resp.mean(dim=0))
        self.mean_stat.add_(step_size * (resp.T @ data_f) / n_samples)
        self.second_moment_stat.add_(
            step_size * (resp.T @ (data_f * data_f)) / n_samples
        )

        denom = self.resp_stat[:, None] + self.eps
        self.mean = self.mean_stat / denom
        self.var = (
            torch.clamp(self.second_moment_stat / denom - self.mean**2, min=0.0)
            + self.reg_covar
        )

        # Stick-breaking ratio update
        double_sum = torch.cumsum(self.resp_stat.flip(0), 0).flip(0)
        double_sum[0] = 0.0
        double_sum = double_sum.roll(-1)
        self.v = self.resp_stat / (
            self.resp_stat + (self.alpha - 1.0) / n_samples + double_sum + self.eps
        )
        self.v = self._clamp_v(self.v)

        # Concentration parameter alpha digamma update
        if not self.alpha_fixed and self.K > 1:
            term1 = torch.special.digamma(
                n_samples * (self.resp_stat + double_sum) + self.alpha + 1.0
            )
            term2 = torch.special.digamma(n_samples * double_sum + self.alpha)
            denom_sum = (term1 - term2)[:-1].sum().item()
            if math.isfinite(denom_sum) and denom_sum > 0.0:
                new_alpha = (self.K - 1) / denom_sum
                if math.isfinite(new_alpha):
                    self.alpha = max(1.0, float(new_alpha))

    def score_cosine(
        self,
        data: torch.Tensor,
        weight_threshold: float = 1e-6,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute chunked 1 - max_k cosine_similarity(y_n, mu_k) over active components."""
        data_f = data.to(device=self.device, dtype=torch.float32)
        mask = self.active_mask(weight_threshold=weight_threshold)
        mu = self.mean[mask]  # (M, D)
        mu_norm_t = torch.linalg.norm(mu, dim=-1, keepdim=True).T  # (1, M)
        mu_t = mu.T  # (D, M)

        chunks: list[torch.Tensor] = []
        n_samples = data_f.shape[0]
        for start in range(0, n_samples, self.eval_chunk_size):
            y = data_f[start : start + self.eval_chunk_size]
            y_norm = torch.linalg.norm(y, dim=-1, keepdim=True)  # (B_c, 1)
            cos_sim = (y @ mu_t) / (y_norm * mu_norm_t + eps)
            chunks.append(1.0 - cos_sim.max(dim=1).values)
        return (
            torch.cat(chunks, dim=0) if chunks else torch.empty(0, device=self.device)
        )

    def score_mahalanobis(
        self,
        data: torch.Tensor,
        weight_threshold: float = 1e-6,
        mahalanobis_eps: float = 1e-4,
    ) -> torch.Tensor:
        """Compute chunked min_k diagonal Mahalanobis distance over active components."""
        data_f = data.to(device=self.device, dtype=torch.float32)
        mask = self.active_mask(weight_threshold=weight_threshold)
        mu = self.mean[mask]  # (M, D)
        inv_var = 1.0 / (self.var[mask] + mahalanobis_eps)  # (M, D)
        inv_var_t = inv_var.T  # (D, M)
        mu_inv_var_t = (mu * inv_var).T  # (D, M)
        mu_quad = (mu * mu * inv_var).sum(dim=-1).unsqueeze(0)  # (1, M)

        chunks: list[torch.Tensor] = []
        n_samples = data_f.shape[0]
        for start in range(0, n_samples, self.eval_chunk_size):
            y = data_f[start : start + self.eval_chunk_size]
            dist_sq = (y * y) @ inv_var_t - 2.0 * (y @ mu_inv_var_t) + mu_quad
            dist_sq = torch.clamp(dist_sq, min=0.0)
            chunks.append(torch.sqrt(dist_sq.min(dim=1).values))
        return (
            torch.cat(chunks, dim=0) if chunks else torch.empty(0, device=self.device)
        )

    def score_euclidean(
        self,
        data: torch.Tensor,
        weight_threshold: float = 1e-6,
    ) -> torch.Tensor:
        """Compute chunked min_k ||y_n - mu_k||_2 over active components."""
        data_f = data.to(device=self.device, dtype=torch.float32)
        mask = self.active_mask(weight_threshold=weight_threshold)
        mu = self.mean[mask]  # (M, D)

        chunks: list[torch.Tensor] = []
        n_samples = data_f.shape[0]
        for start in range(0, n_samples, self.eval_chunk_size):
            y = data_f[start : start + self.eval_chunk_size]
            dists = torch.cdist(y, mu, p=2.0)
            chunks.append(dists.min(dim=1).values)
        return (
            torch.cat(chunks, dim=0) if chunks else torch.empty(0, device=self.device)
        )

    def score_anomaly(
        self,
        data: torch.Tensor,
        metric: str,
        weight_threshold: float = 1e-6,
        mahalanobis_eps: float = 1e-4,
    ) -> torch.Tensor:
        """Dispatch anomaly scoring to the requested metric."""
        metric_l = metric.lower()
        if metric_l == "cosine":
            return self.score_cosine(data, weight_threshold=weight_threshold)
        if metric_l == "mahalanobis":
            return self.score_mahalanobis(
                data, weight_threshold=weight_threshold, mahalanobis_eps=mahalanobis_eps
            )
        if metric_l == "euclidean":
            return self.score_euclidean(data, weight_threshold=weight_threshold)
        raise ValueError(
            f"Unsupported metric {metric!r}; expected one of {_VALID_METRICS}"
        )


class MultiStreamDPMM:
    """Container managing one DiagonalDPMM per feature stream."""

    def __init__(
        self,
        stream_configs: dict[str, StreamDPMMConfig] | Sequence[StreamDPMMConfig],
        device: str | torch.device = "cpu",
    ) -> None:
        if isinstance(stream_configs, Mapping):
            self.stream_configs: dict[str, StreamDPMMConfig] = dict(stream_configs)
        else:
            self.stream_configs = {cfg.name: cfg for cfg in stream_configs}

        self.device = _resolve_device(device)
        self.models: dict[str, DiagonalDPMM] = {
            name: DiagonalDPMM(
                K=cfg.max_components,
                D=cfg.dim,
                update_rate=cfg.gamma,
                schedule=cfg.schedule,
                alpha=cfg.alpha,
                alpha_fixed=cfg.alpha_fixed,
                reg_covar=cfg.reg_covar,
                eval_chunk_size=cfg.eval_chunk_size,
                device=self.device,
            )
            for name, cfg in self.stream_configs.items()
        }

    def _validate_finite_parameters(self) -> None:
        for name, model in self.models.items():
            for attr_name in ("mean", "var", "v"):
                tensor = getattr(model, attr_name)
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError(
                        f"Non-finite values detected in stream '{name}' parameter '{attr_name}'."
                    )
            if not math.isfinite(model.alpha):
                raise FloatingPointError(
                    f"Non-finite alpha ({model.alpha}) detected in stream '{name}'."
                )

    def _evaluate_val_log_likelihood(
        self,
        feature_extractor: Any,
        val_normal_loader: Any,
    ) -> dict[str, float]:
        totals = {name: 0.0 for name in self.stream_configs}
        counts = {name: 0 for name in self.stream_configs}

        with torch.no_grad():
            for batch in val_normal_loader:
                images = _extract_images_from_batch(batch).to(self.device)
                feature_dict = feature_extractor(images)
                for name, cfg in self.stream_configs.items():
                    feat = feature_dict[name]
                    flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
                    n_patches = flat.shape[0]
                    if n_patches == 0:
                        continue
                    ll = self.models[name].score(flat, weight_threshold=0.0).item()
                    totals[name] += ll * n_patches
                    counts[name] += n_patches

        return {
            name: (totals[name] / counts[name] if counts[name] > 0 else float("-inf"))
            for name in self.stream_configs
        }

    def _extract_and_cache_batches(
        self,
        feature_extractor: Any,
        loader: Any,
        split_prefix: str,
        cache_dir: Path | None,
    ) -> list[dict[str, torch.Tensor] | Path]:
        """Extract stream embeddings once and store as CPU float16 in RAM or .pt files."""
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        cached: list[dict[str, torch.Tensor] | Path] = []
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                images = _extract_images_from_batch(batch).to(self.device)
                feature_dict = feature_extractor(images)
                batch_dict: dict[str, torch.Tensor] = {}
                for name in self.stream_configs:
                    feat = feature_dict[name]
                    batch_dict[name] = (
                        feat.permute(0, 2, 3, 1)
                        .reshape(-1, feat.shape[1])
                        .to(device="cpu", dtype=torch.float16)
                    )
                if cache_dir is not None:
                    batch_path = cache_dir / f"{split_prefix}_batch_{i:05d}.pt"
                    torch.save(batch_dict, batch_path)
                    cached.append(batch_path)
                else:
                    cached.append(batch_dict)
        return cached

    @staticmethod
    def _load_cached_batch(
        entry: dict[str, torch.Tensor] | Path,
    ) -> dict[str, torch.Tensor]:
        if isinstance(entry, Path):
            try:
                return torch.load(entry, map_location="cpu", weights_only=True)
            except TypeError:
                return torch.load(entry, map_location="cpu")
        return entry

    def _evaluate_val_log_likelihood_cached(
        self,
        val_cache: Sequence[dict[str, torch.Tensor] | Path],
    ) -> dict[str, float]:
        totals = {name: 0.0 for name in self.stream_configs}
        counts = {name: 0 for name in self.stream_configs}

        with torch.no_grad():
            for entry in val_cache:
                batch_dict = self._load_cached_batch(entry)
                for name in self.stream_configs:
                    flat = batch_dict[name].to(device=self.device, dtype=torch.float32)
                    n_patches = flat.shape[0]
                    if n_patches == 0:
                        continue
                    ll = self.models[name].score(flat, weight_threshold=0.0).item()
                    totals[name] += ll * n_patches
                    counts[name] += n_patches

        return {
            name: (totals[name] / counts[name] if counts[name] > 0 else float("-inf"))
            for name in self.stream_configs
        }

    def fit(
        self,
        feature_extractor: Any,
        train_normal_loader: Any,
        val_normal_loader: Any | None = None,
        seed: int = 42,
        show_progress: bool = False,
        cache_embeddings: bool = False,
        cache_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Fit per-stream reducer and DiagonalDPMMs on train_normal_loader."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if hasattr(feature_extractor, "fit_reducer"):
            feature_extractor.fit_reducer(train_normal_loader, device=self.device)

        if not self.stream_configs:
            return {"active_counts": {}, "best_epochs": {}, "val_losses": {}}

        resolved_cache_dir = Path(cache_dir) if cache_dir is not None else None
        train_cache: list[dict[str, torch.Tensor] | Path] | None = None
        val_cache: list[dict[str, torch.Tensor] | Path] | None = None
        if cache_embeddings:
            train_cache = self._extract_and_cache_batches(
                feature_extractor, train_normal_loader, "train", resolved_cache_dir
            )
            if val_normal_loader is not None:
                val_cache = self._extract_and_cache_batches(
                    feature_extractor, val_normal_loader, "val", resolved_cache_dir
                )

        max_epochs = max(cfg.epochs for cfg in self.stream_configs.values())
        best_val_ll: dict[str, float] = {
            name: float("-inf") for name in self.stream_configs
        }
        best_epochs: dict[str, int] = {
            name: cfg.epochs - 1 for name, cfg in self.stream_configs.items()
        }
        best_states: dict[str, dict[str, Any]] = {}
        val_losses: dict[str, list[float]] = {name: [] for name in self.stream_configs}

        epoch_iter: Any = range(max_epochs)
        if show_progress:
            try:
                from tqdm import tqdm

                epoch_iter = tqdm(
                    epoch_iter, desc="MultiStreamDPMM fit", total=max_epochs
                )
            except ImportError:
                pass

        for epoch in epoch_iter:
            if cache_embeddings:
                assert train_cache is not None
                for step_idx, entry in enumerate(train_cache):
                    batch_dict = self._load_cached_batch(entry)
                    for name, cfg in self.stream_configs.items():
                        if epoch >= cfg.epochs:
                            continue
                        flat = batch_dict[name].to(
                            device=self.device, dtype=torch.float32
                        )
                        if epoch == 0 and step_idx == 0:
                            self.models[name].initialize(flat)
                        self.models[name].step(flat)
            else:
                for step_idx, batch in enumerate(train_normal_loader):
                    images = _extract_images_from_batch(batch).to(self.device)
                    with torch.no_grad():
                        feature_dict = feature_extractor(images)

                    for name, cfg in self.stream_configs.items():
                        if epoch >= cfg.epochs:
                            continue
                        feat = feature_dict[name]
                        flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
                        if epoch == 0 and step_idx == 0:
                            self.models[name].initialize(flat)
                        self.models[name].step(flat)

            if val_normal_loader is not None:
                if cache_embeddings:
                    assert val_cache is not None
                    val_ll = self._evaluate_val_log_likelihood_cached(val_cache)
                else:
                    val_ll = self._evaluate_val_log_likelihood(
                        feature_extractor, val_normal_loader
                    )
                for name, cfg in self.stream_configs.items():
                    if epoch >= cfg.epochs:
                        continue
                    ll = val_ll[name]
                    val_losses[name].append(-ll)
                    if ll > best_val_ll[name]:
                        best_val_ll[name] = ll
                        best_epochs[name] = epoch
                        best_states[name] = self.models[name].state_dict()

        if val_normal_loader is not None:
            for name, state in best_states.items():
                self.models[name].load_state_dict(state)

        self._validate_finite_parameters()

        active_counts = self.get_active_prototype_counts()
        for name, cfg in self.stream_configs.items():
            count = active_counts[name]
            logger.info(
                "Stream '%s': %d / %d active prototypes (best_epoch=%d).",
                name,
                count,
                cfg.max_components,
                best_epochs[name],
            )
            if count > 0.9 * cfg.max_components:
                logger.warning(
                    "Stream '%s' active prototypes (%d) exceed 90%% of max_components (%d); "
                    "consider increasing max_components.",
                    name,
                    count,
                    cfg.max_components,
                )

        return {
            "active_counts": active_counts,
            "best_epochs": best_epochs,
            "val_losses": val_losses,
            "best_val_log_likelihood": best_val_ll
            if val_normal_loader is not None
            else {},
        }

    def score_features(
        self,
        feature_dict: Mapping[str, torch.Tensor],
        metric_overrides: Mapping[str, str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score pre-extracted feature maps {name: (B, C_m, H_m, W_m)} -> {name: (B, 1, H_m, W_m)}."""
        scores_out: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, cfg in self.stream_configs.items():
                feat = feature_dict[name]
                bsz, channels, height, width = feat.shape
                flat = feat.permute(0, 2, 3, 1).reshape(-1, channels)
                metric = (
                    metric_overrides[name]
                    if metric_overrides is not None and name in metric_overrides
                    else cfg.scoring_metric
                )
                flat_scores = self.models[name].score_anomaly(
                    flat,
                    metric=metric,
                    weight_threshold=cfg.weight_threshold,
                    mahalanobis_eps=cfg.mahalanobis_eps,
                )
                scores_out[name] = flat_scores.reshape(bsz, 1, height, width)
        return scores_out

    def score_images(
        self,
        feature_extractor: Any,
        images: torch.Tensor,
        metric_overrides: Mapping[str, str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract features and score images -> {stream_name: Tensor[B, 1, H_m, W_m]}."""
        with torch.no_grad():
            feature_dict = feature_extractor(images.to(self.device))
        return self.score_features(feature_dict, metric_overrides=metric_overrides)

    def get_active_prototype_counts(self) -> dict[str, int]:
        return {
            name: self.models[name].num_active_components(cfg.weight_threshold)
            for name, cfg in self.stream_configs.items()
        }

    def get_active_masks(self) -> dict[str, torch.Tensor]:
        return {
            name: self.models[name].active_mask(cfg.weight_threshold)
            for name, cfg in self.stream_configs.items()
        }

    def save_checkpoint(
        self,
        path: str | Path,
        feature_extractor: Any | None = None,
        calibrator: Any | None = None,
        fusion_head: Any | None = None,
        config: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save all stream DPMMs, active masks, configs, and optional components to a .pt file."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "stream_configs": {
                name: cfg.to_dict() for name, cfg in self.stream_configs.items()
            },
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "active_masks": {
                name: mask.detach().cpu()
                for name, mask in self.get_active_masks().items()
            },
            "config": config,
            "metadata": metadata or {},
        }

        if feature_extractor is not None and hasattr(
            feature_extractor, "reducer_state_dict"
        ):
            payload["reducer_state_dict"] = feature_extractor.reducer_state_dict()
        if calibrator is not None and hasattr(calibrator, "state_dict"):
            payload["calibrator_state_dict"] = calibrator.state_dict()
        if fusion_head is not None and hasattr(fusion_head, "state_dict"):
            payload["fusion_head_state_dict"] = fusion_head.state_dict()

        torch.save(payload, ckpt_path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        feature_extractor: Any | None = None,
        calibrator: Any | None = None,
        fusion_head: Any | None = None,
    ) -> tuple[MultiStreamDPMM, dict[str, Any]]:
        """Load MultiStreamDPMM and optional auxiliary states from a .pt checkpoint."""
        resolved_device = _resolve_device(device)
        try:
            payload = torch.load(path, map_location=resolved_device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=resolved_device)

        stream_configs = {
            name: StreamDPMMConfig.from_dict(cfg_dict)
            for name, cfg_dict in payload["stream_configs"].items()
        }
        instance = cls(stream_configs=stream_configs, device=resolved_device)

        for name, state in payload["models"].items():
            if name in instance.models:
                instance.models[name].load_state_dict(state)

        if (
            feature_extractor is not None
            and "reducer_state_dict" in payload
            and hasattr(feature_extractor, "load_reducer_state_dict")
        ):
            feature_extractor.load_reducer_state_dict(payload["reducer_state_dict"])

        if (
            calibrator is not None
            and "calibrator_state_dict" in payload
            and hasattr(calibrator, "load_state_dict")
        ):
            calibrator.load_state_dict(payload["calibrator_state_dict"])

        if (
            fusion_head is not None
            and "fusion_head_state_dict" in payload
            and hasattr(fusion_head, "load_state_dict")
        ):
            fusion_head.load_state_dict(payload["fusion_head_state_dict"])

        extra = {
            "active_masks": payload.get("active_masks", {}),
            "config": payload.get("config"),
            "metadata": payload.get("metadata", {}),
        }
        return instance, extra
