"""
Adaptive sigma scheduling for Langevin dynamics.

During inference, the energy model E(q, v, σ) is sigma-conditioned.
Using a FIXED sigma throughout all Langevin steps is suboptimal because:
- Early steps: sample is far from target → need large σ (broad landscape)
- Late steps: sample is close to target → need small σ (fine corrections)

This module provides sigma annealing strategies inspired by NCSN/diffusion:

1. **Geometric schedule** (NCSN, Song & Ermon 2019):
   σ(t) = σ_max · (σ_min / σ_max)^(t/T)
   Standard noise schedule that covers multiple scales exponentially.

2. **Sample-adaptive** (distance-based):
   σ_i(t) = ||v_current_i - v_query_i|| / ||v_query_i||
   Auto-estimated per-sample sigma based on current distance.

3. **Hybrid** (recommended):
   Geometric schedule as envelope, per-sample distance for fine-tuning.
   σ_i(t) = clip(adaptive_σ_i, schedule_σ(t)*0.5, schedule_σ(t)*2.0)

References:
    - Song & Ermon, "Generative Modeling by Estimating Gradients of the Data
      Distribution" (NeurIPS 2019)
    - Song et al., "Score-Based Generative Modeling through SDEs" (ICLR 2021)

Spec reference: §8 (Langevin dynamics)
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
    # Hybrid mode: blend factor between schedule and adaptive
    # 0.0 = pure schedule, 1.0 = pure adaptive
    adaptive_blend: float = 0.5


def geometric_sigma(step: int, max_steps: int, sigma_max: float, sigma_min: float) -> float:
    """
    Geometric (log-linear) sigma schedule: σ(t) = σ_max · (σ_min/σ_max)^(t/T).

    This is the standard NCSN noise schedule. Covers multiple noise scales
    exponentially, spending equal relative time at each scale.

    Args:
        step: Current Langevin step (0-indexed).
        max_steps: Total number of steps.
        sigma_max: Starting (largest) sigma.
        sigma_min: Final (smallest) sigma.

    Returns:
        Sigma value at this step.
    """
    if max_steps <= 1:
        return sigma_min
    t = min(step / (max_steps - 1), 1.0)
    return sigma_max * (sigma_min / max(sigma_max, 1e-12)) ** t


def adaptive_sigma_from_distance(
    v_current: Tensor,
    v_query: Tensor,
) -> Tensor:
    """
    Per-sample sigma estimated from current distance to query.

    σ_i = ||v_current_i - v_query_i|| / ||v_query_i||

    This naturally decreases as v_current approaches v_query during
    Langevin refinement, providing automatic sigma annealing.

    Args:
        v_current: [B, D] current candidate vectors.
        v_query: [B, D] query/anchor vectors.

    Returns:
        [B, 1] per-sample sigma estimates.
    """
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
    """
    Compute sigma for current Langevin step using configured strategy.

    Args:
        step: Current step (0-indexed).
        max_steps: Total steps.
        v_current: [B, D] current candidate.
        v_query: [B, D] query.
        config: Sigma schedule configuration.

    Returns:
        [B, 1] sigma tensor on same device as v_current.
    """
    B = v_current.shape[0]
    device = v_current.device

    if config.mode == "geometric":
        s = geometric_sigma(step, max_steps, config.sigma_max, config.sigma_min)
        return torch.full((B, 1), s, device=device, dtype=v_current.dtype)

    elif config.mode == "adaptive":
        sigma = adaptive_sigma_from_distance(v_current, v_query)
        return sigma.clamp(min=config.sigma_min, max=config.sigma_max)

    elif config.mode == "hybrid":
        # Geometric schedule as envelope
        sched_sigma = geometric_sigma(step, max_steps, config.sigma_max, config.sigma_min)
        # Per-sample adaptive
        adapt_sigma = adaptive_sigma_from_distance(v_current, v_query)
        # Clip adaptive to schedule range
        adapt_sigma = adapt_sigma.clamp(
            min=sched_sigma * 0.5,
            max=sched_sigma * 2.0,
        )
        # Blend: weighted average in log-space for smooth interpolation
        alpha = config.adaptive_blend
        sched_tensor = torch.full_like(adapt_sigma, sched_sigma)
        log_sigma = (1.0 - alpha) * sched_tensor.log() + alpha * adapt_sigma.clamp(min=1e-8).log()
        return log_sigma.exp().clamp(min=config.sigma_min, max=config.sigma_max)

    else:
        raise ValueError(f"Unknown sigma schedule mode: {config.mode}")


class AdaptiveSigmaEnergyWrapper:
    """
    Wraps a sigma-conditioned energy model with step-dependent adaptive sigma.

    Replaces the fixed-sigma `SigmaBoundEnergy` for Langevin inference.
    The wrapper is stateful — call `set_step()` before each Langevin step
    to update the current sigma.

    Usage in Langevin loop:
        wrapper = AdaptiveSigmaEnergyWrapper(energy_fn, config, max_steps)
        for step in range(max_steps):
            wrapper.set_step(step)
            energy, grad = wrapper.energy_and_grad(v_query, v_current)
            ...
    """

    def __init__(
        self,
        energy_fn,
        config: SigmaScheduleConfig,
        max_steps: int,
    ):
        self.energy_fn = energy_fn
        self.config = config
        self.max_steps = max_steps
        self._current_step = 0

    def set_step(self, step: int) -> None:
        """Update current Langevin step for sigma computation."""
        self._current_step = step

    def _get_sigma(self, v_query: Tensor, v_candidate: Tensor) -> Tensor:
        """Compute sigma for current step and sample pair."""
        return compute_sigma(
            self._current_step,
            self.max_steps,
            v_candidate,
            v_query,
            self.config,
        )

    def __call__(self, v_query: Tensor, v_candidate: Tensor) -> Tensor:
        sigma = self._get_sigma(v_query, v_candidate)
        return self.energy_fn(v_query, v_candidate, sigma=sigma)

    def energy_and_grad(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        sigma = self._get_sigma(v_query, v_candidate)
        return self.energy_fn.energy_and_grad(v_query, v_candidate, sigma=sigma)
