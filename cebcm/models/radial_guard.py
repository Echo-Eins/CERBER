"""
Analytical Radial Guard — manifold norm control for SONAR embeddings.

This module provides a **parameter-free** radial energy term that keeps
Langevin samples on the correct norm shell:

    E_rad(v) = scale × (‖v‖ − target_norm)²

Gradient (analytical, no autograd needed):

    ∇_v E_rad = 2 × scale × (‖v‖ − target_norm) × v / ‖v‖

Key properties:
  • Gradient is **purely radial** (∥ v̂) — orthogonal to the angular critic's
    tangential gradient.  Zero interference guaranteed.
  • Quadratic potential ⇒ linear restoring force ⇒ stable, predictable.
  • No trainable parameters ⇒ no training needed, no overfitting risk.
  • Analytical gradient ⇒ no autograd overhead, exact computation.

Usage with CompositeCritic:
    E_total = E_angular(q, v, ctx, σ) + λ_rad × E_radial(v)
    ∇E_total = ∇E_angular (tangential) + λ_rad × ∇E_radial (radial)

SONAR default: target_norm ≈ 0.2051 (empirical mean of SONAR sentence vectors).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class RadialGuardConfig:
    """Configuration for the Analytical Radial Guard."""

    target_norm: float = 0.2051
    """Target L2 norm for embeddings (SONAR empirical mean)."""

    scale: float = 1.0
    """Energy scale factor.  E = scale × (‖v‖ − target)².
    Higher ⇒ tighter shell constraint.  Typical: 1.0–10.0."""


class AnalyticalRadialGuard(nn.Module):
    """
    Parameter-free radial energy guard.

    E_rad(v) = scale × (‖v‖ − target_norm)²

    Provides both forward() for energy and energy_and_grad() for
    Langevin-compatible gradient computation.  All computations are
    analytical — no autograd graph is built.
    """

    def __init__(self, cfg: RadialGuardConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = RadialGuardConfig()
        self.cfg = cfg

        # Store as buffers so they move with .to(device) / .half() etc.
        self.register_buffer(
            "_target_norm",
            torch.tensor(cfg.target_norm, dtype=torch.float32),
        )
        self.register_buffer(
            "_scale",
            torch.tensor(cfg.scale, dtype=torch.float32),
        )

    @property
    def target_norm(self) -> float:
        return self._target_norm.item()

    @property
    def scale(self) -> float:
        return self._scale.item()

    def forward(self, v_candidate: Tensor) -> Tensor:
        """
        Compute radial energy.

        Args:
            v_candidate: [B, D] candidate embeddings.

        Returns:
            [B] energy scalars.  E = scale × (‖v‖ − target)².
            Minimum (E=0) when ‖v‖ = target_norm.
        """
        v_norm = v_candidate.norm(dim=-1)  # [B]
        delta = v_norm - self._target_norm  # [B]
        return self._scale * delta.pow(2)

    def energy_and_grad(
        self, v_candidate: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute energy and analytical gradient (no autograd).

        ∇_v E = 2 × scale × (‖v‖ − target) × v / ‖v‖
              = 2 × scale × (‖v‖ − target) × v̂

        The gradient is purely radial (parallel to v̂), so it is
        orthogonal to the angular critic's tangential gradient.

        Args:
            v_candidate: [B, D] candidate embeddings.

        Returns:
            (energy [B], grad [B, D])
        """
        v_norm = v_candidate.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
        delta = v_norm - self._target_norm  # [B, 1]

        energy = self._scale * delta.pow(2)  # [B, 1]
        # ∇E = 2 * scale * delta * v / ‖v‖
        grad = (2.0 * self._scale * delta / v_norm) * v_candidate  # [B, D]

        return energy.squeeze(-1), grad

    def extra_repr(self) -> str:
        return f"target_norm={self.target_norm:.4f}, scale={self.scale:.1f}"
