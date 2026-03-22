"""
Langevin Dynamics for energy-based refinement in SONAR space.

Core loop:
    V_{t+1} = V_t - η·∇_V E(V_query, V_t) + √(2η)·ε_t

With momentum (optional):
    m_t = β·m_{t-1} + (1-β)·∇_V E
    V_{t+1} = V_t - η·m_t + √(2η)·ε_t

OOD projection after each step:
    V ← V / ||V|| × target_norm

Spec reference: §8, §10, Appendix A.1/A.3/A.4
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass


@dataclass
class LangevinResult:
    """Result of a Langevin dynamics run."""
    v_final: Tensor          # [B, D] refined vectors
    trajectory: list[float]  # energy at each step
    cos_trajectory: list[float]  # cos_sim to target (if provided) at each step
    num_steps: int           # actual steps taken
    stopped_early: bool      # whether early stopping triggered


def langevin_dynamics(
    energy_fn: torch.nn.Module,
    v_query: Tensor,
    v_init: Tensor,
    lr: float = 0.01,
    noise_scale: float = 0.003,
    max_steps: int = 100,
    target_norm: float | None = None,
    momentum_beta: float = 0.0,
    energy_threshold: float | None = None,
    plateau_patience: int = 10,
    plateau_delta: float = 1e-4,
    v_target: Tensor | None = None,
) -> LangevinResult:
    """
    Run Langevin dynamics to refine v_init guided by energy_fn.

    The energy function E(v_query, v_candidate) is minimized w.r.t. v_candidate.
    Lower energy = better candidate.

    Args:
        energy_fn:         E(v_query, v_candidate) → scalar energy
        v_query:           [B, D] anchor vector (frozen)
        v_init:            [B, D] starting point for refinement
        lr:                Step size η
        noise_scale:       Langevin noise magnitude (relative to lr)
        max_steps:         Maximum number of refinement steps
        target_norm:       If set, project V onto sphere of this radius after each step
        momentum_beta:     Momentum coefficient β (0 = no momentum)
        energy_threshold:  Stop if energy drops below this
        plateau_patience:  Stop if energy doesn't improve for this many steps
        plateau_delta:     Minimum improvement to reset plateau counter
        v_target:          [B, D] ground truth (for monitoring only, not used in optimization)

    Returns:
        LangevinResult with final vectors and diagnostics
    """
    device = v_init.device
    v_current = v_init.clone().detach()
    momentum = torch.zeros_like(v_current) if momentum_beta > 0 else None

    trajectory: list[float] = []
    cos_trajectory: list[float] = []
    best_energy = float("inf")
    plateau_counter = 0

    for step in range(max_steps):
        # Compute energy and gradient
        energy, grad = energy_fn.energy_and_grad(v_query, v_current)
        e_mean = energy.mean().item()
        trajectory.append(e_mean)

        # Track cosine similarity to target if available
        if v_target is not None:
            cos = F.cosine_similarity(v_current, v_target, dim=-1).mean().item()
            cos_trajectory.append(cos)

        # Early stopping: energy threshold
        if energy_threshold is not None and e_mean < energy_threshold:
            return LangevinResult(
                v_final=v_current,
                trajectory=trajectory,
                cos_trajectory=cos_trajectory,
                num_steps=step + 1,
                stopped_early=True,
            )

        # Early stopping: plateau detection
        if e_mean < best_energy - plateau_delta:
            best_energy = e_mean
            plateau_counter = 0
        else:
            plateau_counter += 1
            if plateau_counter >= plateau_patience:
                return LangevinResult(
                    v_final=v_current,
                    trajectory=trajectory,
                    cos_trajectory=cos_trajectory,
                    num_steps=step + 1,
                    stopped_early=True,
                )

        # Update with optional momentum
        if momentum is not None and momentum_beta > 0:
            momentum = momentum_beta * momentum + (1 - momentum_beta) * grad
            update = momentum
        else:
            update = grad

        # Tangent plane projection: remove radial component so the update
        # moves along the sphere rather than fighting the norm projection.
        if target_norm is not None:
            v_hat = F.normalize(v_current, dim=-1)
            radial = (update * v_hat).sum(dim=-1, keepdim=True) * v_hat
            update = update - radial

        # Langevin step: gradient descent + stochastic noise
        langevin_noise = torch.randn_like(v_current) * (2 * lr * noise_scale) ** 0.5
        v_current = v_current - lr * update + langevin_noise

        # OOD projection: keep vectors on the data manifold
        if target_norm is not None:
            v_current = F.normalize(v_current, dim=-1) * target_norm

    return LangevinResult(
        v_final=v_current,
        trajectory=trajectory,
        cos_trajectory=cos_trajectory,
        num_steps=max_steps,
        stopped_early=False,
    )
