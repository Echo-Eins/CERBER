"""
Adaptive sigma scheduling for Langevin inference.

The energy model E(q, v, sigma) is sigma-conditioned. During inference,
using fixed sigma for all Langevin steps is usually suboptimal.

This module provides:
- geometric sigma schedule,
- adaptive distance-based sigma,
- hybrid schedule,
and optional synchronized noise annealing so Langevin noise can be
co-annealed with sigma.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SigmaScheduleConfig:
    """Configuration for sigma annealing during Langevin inference."""

    enabled: bool = True
    mode: str = "geometric"  # "geometric", "adaptive", "hybrid"
    sigma_max: float = 0.3
    sigma_min: float = 0.01
    # Hybrid blend: 0.0 = pure schedule, 1.0 = pure adaptive
    adaptive_blend: float = 0.5

    # Optional synchronized Langevin noise annealing.
    noise_anneal: bool = False
    noise_mode: str = "geometric"  # "geometric", "adaptive", "hybrid"
    noise_max: float = 0.15
    noise_min: float = 0.0002
    noise_sync_with_sigma: bool = True


def geometric_sigma(step: int, max_steps: int, sigma_max: float, sigma_min: float) -> float:
    """Geometric sigma schedule: sigma(t) = sigma_max * (sigma_min/sigma_max)^(t/T)."""
    if max_steps <= 1:
        return sigma_min
    t = min(step / (max_steps - 1), 1.0)
    return sigma_max * (sigma_min / max(sigma_max, 1e-12)) ** t


def adaptive_sigma_from_distance(v_current: Tensor, v_query: Tensor) -> Tensor:
    """Per-sample sigma estimated from relative distance to query."""
    with torch.no_grad():
        dist = (v_current - v_query).norm(dim=-1, keepdim=True)
        query_norm = v_query.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return dist / query_norm


def compute_sigma(
    step: int,
    max_steps: int,
    v_current: Tensor,
    v_query: Tensor,
    config: SigmaScheduleConfig,
) -> Tensor:
    """Compute sigma for current Langevin step."""
    batch = v_current.shape[0]
    device = v_current.device

    if config.mode == "geometric":
        s = geometric_sigma(step, max_steps, config.sigma_max, config.sigma_min)
        return torch.full((batch, 1), s, device=device, dtype=v_current.dtype)

    if config.mode == "adaptive":
        sigma = adaptive_sigma_from_distance(v_current, v_query)
        return sigma.clamp(min=config.sigma_min, max=config.sigma_max)

    if config.mode == "hybrid":
        sched_sigma = geometric_sigma(step, max_steps, config.sigma_max, config.sigma_min)
        adapt_sigma = adaptive_sigma_from_distance(v_current, v_query)
        adapt_sigma = adapt_sigma.clamp(min=sched_sigma * 0.5, max=sched_sigma * 2.0)

        alpha = config.adaptive_blend
        sched_tensor = torch.full_like(adapt_sigma, sched_sigma)
        log_sigma = (1.0 - alpha) * sched_tensor.log() + alpha * adapt_sigma.clamp(min=1e-8).log()
        return log_sigma.exp().clamp(min=config.sigma_min, max=config.sigma_max)

    raise ValueError(f"Unknown sigma schedule mode: {config.mode}")


class AdaptiveSigmaEnergyWrapper:
    """
    Wrap sigma-conditioned energy model with step-dependent sigma.

    In Langevin loop, `run_langevin` will call `set_step(step)` when available.
    """

    def __init__(self, energy_fn, config: SigmaScheduleConfig, max_steps: int):
        self.energy_fn = energy_fn
        self.config = config
        self.max_steps = max_steps
        self._current_step = 0

    def set_step(self, step: int) -> None:
        """Update current step index."""
        self._current_step = step

    def _get_sigma(self, v_query: Tensor, v_candidate: Tensor) -> Tensor:
        """Compute sigma for current step and batch."""
        return compute_sigma(
            self._current_step,
            self.max_steps,
            v_candidate,
            v_query,
            self.config,
        )

    def get_step_noise_scale(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        base_noise_scale: float,
    ) -> float:
        """
        Return step-dependent Langevin noise scale.

        If noise_anneal is disabled, returns base_noise_scale.
        If noise_sync_with_sigma is enabled, maps current sigma to [noise_min, noise_max]
        in log-space for consistent co-annealing.
        """
        if not self.config.noise_anneal:
            return float(base_noise_scale)

        n_min = max(float(self.config.noise_min), 0.0)
        n_max = max(float(self.config.noise_max), n_min + 1e-12)

        if self.config.noise_sync_with_sigma:
            sigma_t = float(self._get_sigma(v_query, v_candidate).mean().item())
            s_min = max(float(self.config.sigma_min), 1e-12)
            s_max = max(float(self.config.sigma_max), s_min + 1e-12)
            sigma_t = min(max(sigma_t, s_min), s_max)

            denom = max(math.log(s_max) - math.log(s_min), 1e-12)
            alpha = (math.log(sigma_t) - math.log(s_min)) / denom
            alpha = min(max(alpha, 0.0), 1.0)

            log_n_min = math.log(max(n_min, 1e-12))
            log_n_max = math.log(max(n_max, 1e-12))
            noise_t = math.exp(log_n_min + alpha * (log_n_max - log_n_min))
            return float(min(max(noise_t, n_min), n_max))

        noise_cfg = SigmaScheduleConfig(
            enabled=True,
            mode=self.config.noise_mode,
            sigma_max=n_max,
            sigma_min=max(n_min, 1e-12),
            adaptive_blend=self.config.adaptive_blend,
        )
        noise_t = compute_sigma(
            self._current_step,
            self.max_steps,
            v_candidate,
            v_query,
            noise_cfg,
        )
        return float(noise_t.mean().item())

    def __call__(self, v_query: Tensor, v_candidate: Tensor) -> Tensor:
        sigma = self._get_sigma(v_query, v_candidate)
        return self.energy_fn(v_query, v_candidate, sigma=sigma)

    def energy_and_grad(self, v_query: Tensor, v_candidate: Tensor) -> tuple[Tensor, Tensor]:
        sigma = self._get_sigma(v_query, v_candidate)
        return self.energy_fn.energy_and_grad(v_query, v_candidate, sigma=sigma)
