"""
Composite Critic — angular semantics + radial guard for SONAR QA.

Combines:
    1. ConditionalAngularCritic — learned semantic energy on unit sphere
    2. AnalyticalRadialGuard    — analytical norm-shell constraint

    E_total(q, v, ctx, σ) = E_angular(q, v, ctx, σ) + λ_rad × E_radial(v)

Gradient decomposition (provably orthogonal):
    ∇E_total = ∇E_angular  +  λ_rad × ∇E_radial
               ╰─tangent──╯    ╰──radial──╯

    ∇E_angular ⊥ v̂   (tangential, from normalisation Jacobian)
    ∇E_radial  ∥ v̂   (radial, analytical)
    ⟹  ⟨∇E_angular, ∇E_radial⟩ = 0   (zero interference)

This module is the top-level energy function used by:
    - Langevin dynamics (via _ContextWrappedEnergyFn in system_switching.py)
    - Training script (direct loss computation)

API matches the convention expected by Langevin infrastructure:
    forward(v_query, v_candidate, v_context?, sigma?) → [B]
    energy_and_grad(v_query, v_candidate, v_context?, sigma?) → ([B], [B,D])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from cebcm.models.conditional_angular_critic import (
    ConditionalAngularCritic,
    ConditionalAngularCriticConfig,
)
from cebcm.models.radial_guard import AnalyticalRadialGuard, RadialGuardConfig


@dataclass
class CompositeCriticConfig:
    """Configuration for the Composite Critic."""

    angular: ConditionalAngularCriticConfig = field(
        default_factory=ConditionalAngularCriticConfig
    )
    radial: RadialGuardConfig = field(default_factory=RadialGuardConfig)

    lambda_radial: float = 5.0
    """Weight of radial guard in total energy.  Higher ⇒ tighter shell.
    Typical range: 1.0–10.0.  Since E_angular ∈ [-50, 50] and
    E_radial ≈ scale×δ² (small δ for on-manifold), λ=5 keeps them balanced."""


class CompositeCritic(nn.Module):
    """
    Combined angular + radial energy critic.

    E_total = E_angular + λ_rad × E_radial

    The angular component is *learned* (trainable MLP on unit sphere).
    The radial component is *analytical* (quadratic norm penalty, no params).

    Trainable parameters: only in the angular critic (~13M default).
    """

    def __init__(self, cfg: CompositeCriticConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = CompositeCriticConfig()
        self.cfg = cfg

        self.angular = ConditionalAngularCritic(cfg.angular)
        self.radial = AnalyticalRadialGuard(cfg.radial)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── forward ──────────────────────────────────────────────────

    def forward(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        v_context: Tensor | None = None,
        sigma: Tensor | None = None,
    ) -> Tensor:
        """
        Compute total energy = E_angular + λ × E_radial.

        Args:
            v_query:     [B, D] question embedding
            v_candidate: [B, D] candidate answer embedding
            v_context:   [B, D] context.  Falls back to v_query if None.
            sigma:       [B, 1] noise level.  Auto-estimated if None.

        Returns:
            [B] total energy scalars.
        """
        e_ang = self.angular(v_query, v_candidate, v_context=v_context, sigma=sigma)
        e_rad = self.radial(v_candidate)
        return e_ang + self.cfg.lambda_radial * e_rad

    # ── gradient (Langevin API) ──────────────────────────────────

    def energy_and_grad(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        v_context: Tensor | None = None,
        sigma: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute total energy and gradient ∇_{v_candidate} E_total.

        Angular gradient: via autograd (tangential by construction).
        Radial gradient: analytical (radial by construction).
        Combined: orthogonal sum.

        Returns:
            (energy [B],  grad [B, D])
        """
        # Angular: autograd through normalisation
        e_ang, g_ang = self.angular.energy_and_grad(
            v_query, v_candidate, v_context=v_context, sigma=sigma,
        )

        # Radial: analytical (no autograd graph)
        e_rad, g_rad = self.radial.energy_and_grad(v_candidate)

        e_total = e_ang + self.cfg.lambda_radial * e_rad
        g_total = g_ang + self.cfg.lambda_radial * g_rad

        return e_total, g_total

    # ── training losses (delegate to angular) ────────────────────

    def compute_contrastive_loss(
        self,
        v_query: Tensor,
        v_positive: Tensor,
        v_negatives: Tensor,
        v_context: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Focal-InfoNCE from angular critic (radial guard is not trained)."""
        return self.angular.compute_contrastive_loss(
            v_query, v_positive, v_negatives, v_context=v_context,
        )

    def compute_direction_loss(
        self,
        v_query: Tensor,
        v_noisy: Tensor,
        v_target: Tensor,
        v_context: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Direction supervision from angular critic."""
        return self.angular.compute_direction_loss(
            v_query, v_noisy, v_target, v_context=v_context,
        )

    def compute_path_contrastive_loss(
        self,
        v_query: Tensor,
        v_answer: Tensor,
        v_context: Tensor | None = None,
        num_waypoints: int = 5,
        waypoint_noise: float = 0.02,
        margin: float = 0.1,
    ) -> tuple[Tensor, dict[str, float]]:
        """Path-contrastive loss from angular critic (1st-order, no Hessian)."""
        return self.angular.compute_path_contrastive_loss(
            v_query, v_answer, v_context=v_context,
            num_waypoints=num_waypoints,
            waypoint_noise=waypoint_noise,
            margin=margin,
        )

    def compute_cosine_loss(
        self,
        v_predicted: Tensor,
        v_target: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Cosine reconstruction metric."""
        return self.angular.compute_cosine_loss(v_predicted, v_target)
