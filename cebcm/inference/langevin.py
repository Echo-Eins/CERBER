"""
Langevin Dynamics variants for energy-based refinement in SONAR space.

Implements three Langevin dynamics methods:

1. **Overdamped** (classic):
       V_{t+1} = V_t - η·∇E + √(2η)·ε

2. **PID-Controlled** (PIDLD, arXiv:2511.12603):
       Uses PID controller theory to accelerate convergence.
       P = current gradient, I = integral (momentum with decay),
       D = derivative (gradient change rate / trend prediction).
       Drop-in replacement, no retraining needed.

3. **Underdamped** (second-order, GAUL-inspired):
       p_{t+1} = (1-γ)·p_t - η·∇E + √(2γη)·ε
       V_{t+1} = V_t + p_{t+1}
       Particles have inertia, can cross energy barriers.
       2-5× faster convergence than overdamped.

All variants support:
    - OOD projection (sphere constraint)
    - Tangent plane projection
    - Early stopping (energy threshold + plateau detection)
    - Trajectory logging

Spec reference: §8, §10, Appendix A.1/A.3/A.4
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from enum import Enum


class LangevinMethod(str, Enum):
    """Available Langevin dynamics methods."""
    OVERDAMPED = "overdamped"
    PID = "pid"
    UNDERDAMPED = "underdamped"


@dataclass
class LangevinResult:
    """Result of a Langevin dynamics run."""
    v_final: Tensor          # [B, D] refined vectors
    v_last: Tensor | None = None  # [B, D] actually reached state at the last executed step
    trajectory: list[float] = field(default_factory=list)  # energy at each step
    cos_trajectory: list[float] = field(default_factory=list)  # cos_sim to target at each step
    v_trajectory: list[Tensor] = field(default_factory=list)  # optional vector trajectory
    num_steps: int = 0       # actual steps taken
    stopped_early: bool = False  # whether early stopping triggered


def _check_early_stop(
    e_mean: float,
    best_energy: float,
    plateau_counter: int,
    energy_threshold: float | None,
    plateau_patience: int,
    plateau_delta: float,
) -> tuple[bool, float, int, bool]:
    """
    Check early stopping conditions. Returns (should_stop, new_best, new_counter).
    """
    best_before = best_energy
    improved = e_mean < best_before
    if improved:
        best_energy = e_mean

    if e_mean < best_before - plateau_delta:
        plateau_counter = 0
    else:
        plateau_counter += 1

    if energy_threshold is not None and e_mean < energy_threshold:
        return True, best_energy, plateau_counter, improved

    if plateau_counter >= plateau_patience:
        return True, best_energy, plateau_counter, improved

    return False, best_energy, plateau_counter, improved


def _project_to_sphere(v: Tensor, target_norm: float) -> Tensor:
    """Project vectors onto sphere of given radius."""
    return F.normalize(v, dim=-1) * target_norm


def _tangent_projection(update: Tensor, v_current: Tensor) -> Tensor:
    """Remove radial component so update moves along the sphere."""
    v_hat = F.normalize(v_current, dim=-1)
    radial = (update * v_hat).sum(dim=-1, keepdim=True) * v_hat
    return update - radial


# ============================================================
# Method 1: Classic Overdamped Langevin
# ============================================================

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
    track_vectors: bool = False,
) -> LangevinResult:
    """
    Classic overdamped Langevin dynamics.

    V_{t+1} = V_t - η·∇_V E(V_query, V_t) + √(2η)·ε_t

    With optional momentum:
        m_t = β·m_{t-1} + (1-β)·∇E
        V_{t+1} = V_t - η·m_t + √(2η)·ε_t

    Args:
        energy_fn:         E(v_query, v_candidate) → scalar energy
        v_query:           [B, D] anchor vector (frozen)
        v_init:            [B, D] starting point for refinement
        lr:                Step size η
        noise_scale:       Langevin noise magnitude (relative to lr)
        max_steps:         Maximum number of refinement steps
        target_norm:       If set, project V onto sphere of this radius
        momentum_beta:     Momentum coefficient β (0 = no momentum)
        energy_threshold:  Stop if energy drops below this
        plateau_patience:  Stop if energy doesn't improve for this many steps
        plateau_delta:     Minimum improvement to reset plateau counter
        v_target:          [B, D] ground truth (monitoring only)

    Returns:
        LangevinResult with final vectors and diagnostics
    """
    v_current = v_init.clone().detach()
    momentum = torch.zeros_like(v_current) if momentum_beta > 0 else None

    trajectory: list[float] = []
    cos_trajectory: list[float] = []
    v_trajectory: list[Tensor] = []
    best_energy = float("inf")
    v_best = v_current.clone()
    plateau_counter = 0
    if track_vectors:
        v_trajectory.append(v_current.detach().cpu().clone())

    for step in range(max_steps):
        energy, grad = energy_fn.energy_and_grad(v_query, v_current)
        e_mean = energy.mean().item()
        trajectory.append(e_mean)

        if v_target is not None:
            cos = F.cosine_similarity(v_current, v_target, dim=-1).mean().item()
            cos_trajectory.append(cos)

        # Early stopping
        should_stop, best_energy, plateau_counter, improved = _check_early_stop(
            e_mean, best_energy, plateau_counter,
            energy_threshold, plateau_patience, plateau_delta,
        )
        if improved:
            v_best = v_current.clone()
        if should_stop:
            return LangevinResult(
                v_final=v_best,
                v_last=v_current.clone(),
                trajectory=trajectory,
                cos_trajectory=cos_trajectory,
                v_trajectory=v_trajectory,
                num_steps=step + 1, stopped_early=True,
            )

        # Update with optional momentum
        if momentum is not None and momentum_beta > 0:
            momentum = momentum_beta * momentum + (1 - momentum_beta) * grad
            update = momentum
        else:
            update = grad

        # Tangent projection
        if target_norm is not None:
            update = _tangent_projection(update, v_current)

        # Langevin step
        langevin_noise = torch.randn_like(v_current) * (2 * lr * noise_scale) ** 0.5
        v_current = v_current - lr * update + langevin_noise

        # OOD projection
        if target_norm is not None:
            v_current = _project_to_sphere(v_current, target_norm)
        if track_vectors:
            v_trajectory.append(v_current.detach().cpu().clone())

    with torch.no_grad():
        final_energy = energy_fn(v_query, v_current).mean().item()
    if final_energy < best_energy:
        v_best = v_current.clone()

    return LangevinResult(
        v_final=v_best,
        v_last=v_current.clone(),
        trajectory=trajectory,
        cos_trajectory=cos_trajectory,
        v_trajectory=v_trajectory,
        num_steps=max_steps, stopped_early=False,
    )


# ============================================================
# Method 2: PID-Controlled Langevin Dynamics (PIDLD)
# ============================================================

def pid_langevin_dynamics(
    energy_fn: torch.nn.Module,
    v_query: Tensor,
    v_init: Tensor,
    lr: float = 0.01,
    noise_scale: float = 0.003,
    max_steps: int = 100,
    target_norm: float | None = None,
    energy_threshold: float | None = None,
    plateau_patience: int = 10,
    plateau_delta: float = 1e-4,
    v_target: Tensor | None = None,
    # PID coefficients
    kp: float = 1.0,
    ki: float = 0.3,
    kd: float = 0.1,
    integral_decay: float = 0.95,
    track_vectors: bool = False,
) -> LangevinResult:
    """
    PID-Controlled Langevin Dynamics (PIDLD).

    Reinterprets Langevin sampling through control theory:
        - P (proportional): current gradient ∇E — standard Langevin term
        - I (integral): accumulated gradient history — like momentum but with
          exponential decay, provides persistent drift direction
        - D (derivative): rate of gradient change — predicts gradient trend,
          allows faster deceleration near minima and aggressive movement far away

    update = Kp·∇E_t + Ki·I_t + Kd·D_t

    where I_t = decay·I_{t-1} + ∇E_t (integral with decay)
          D_t = ∇E_t - ∇E_{t-1}      (discrete derivative)

    This is a drop-in replacement for standard Langevin — no retraining needed.
    Significantly reduces iterations while maintaining sample quality.

    Reference: arXiv:2511.12603 "PID-controlled Langevin Dynamics for
    Faster Sampling of Generative Models"

    Args:
        energy_fn:       Energy model.
        v_query:         [B, D] anchor vector.
        v_init:          [B, D] starting point.
        lr:              Base step size.
        noise_scale:     Langevin noise magnitude.
        max_steps:       Maximum iterations.
        target_norm:     Sphere projection radius.
        energy_threshold: Early stop threshold.
        plateau_patience: Plateau detection patience.
        plateau_delta:   Minimum improvement.
        v_target:        [B, D] ground truth (monitoring only).
        kp:              Proportional gain (standard gradient weight).
        ki:              Integral gain (accumulated gradient weight).
        kd:              Derivative gain (gradient change rate weight).
        integral_decay:  Exponential decay for integral term (prevents windup).

    Returns:
        LangevinResult with final vectors and diagnostics.
    """
    v_current = v_init.clone().detach()

    # PID state
    integral = torch.zeros_like(v_current)    # I: accumulated gradient
    prev_grad: Tensor | None = None           # for computing D

    trajectory: list[float] = []
    cos_trajectory: list[float] = []
    v_trajectory: list[Tensor] = []
    best_energy = float("inf")
    v_best = v_current.clone()
    plateau_counter = 0
    if track_vectors:
        v_trajectory.append(v_current.detach().cpu().clone())

    for step in range(max_steps):
        energy, grad = energy_fn.energy_and_grad(v_query, v_current)
        e_mean = energy.mean().item()
        trajectory.append(e_mean)

        if v_target is not None:
            cos = F.cosine_similarity(v_current, v_target, dim=-1).mean().item()
            cos_trajectory.append(cos)

        # Early stopping
        should_stop, best_energy, plateau_counter, improved = _check_early_stop(
            e_mean, best_energy, plateau_counter,
            energy_threshold, plateau_patience, plateau_delta,
        )
        if improved:
            v_best = v_current.clone()
        if should_stop:
            return LangevinResult(
                v_final=v_best,
                v_last=v_current.clone(),
                trajectory=trajectory,
                cos_trajectory=cos_trajectory,
                v_trajectory=v_trajectory,
                num_steps=step + 1, stopped_early=True,
            )

        # PID components
        # P: proportional — current gradient
        p_term = grad

        # I: integral — accumulated gradient with decay (prevents windup)
        integral = integral_decay * integral + grad
        i_term = integral

        # D: derivative — gradient change rate
        if prev_grad is not None:
            d_term = grad - prev_grad
        else:
            d_term = torch.zeros_like(grad)
        prev_grad = grad.clone()

        # Combined PID update
        update = kp * p_term + ki * i_term + kd * d_term

        # Tangent projection
        if target_norm is not None:
            update = _tangent_projection(update, v_current)

        # Langevin step with PID-controlled gradient
        langevin_noise = torch.randn_like(v_current) * (2 * lr * noise_scale) ** 0.5
        v_current = v_current - lr * update + langevin_noise

        # OOD projection
        if target_norm is not None:
            v_current = _project_to_sphere(v_current, target_norm)
        if track_vectors:
            v_trajectory.append(v_current.detach().cpu().clone())

    with torch.no_grad():
        final_energy = energy_fn(v_query, v_current).mean().item()
    if final_energy < best_energy:
        v_best = v_current.clone()

    return LangevinResult(
        v_final=v_best,
        v_last=v_current.clone(),
        trajectory=trajectory,
        cos_trajectory=cos_trajectory,
        v_trajectory=v_trajectory,
        num_steps=max_steps, stopped_early=False,
    )


# ============================================================
# Method 3: Underdamped (Second-Order) Langevin Dynamics
# ============================================================

def underdamped_langevin_dynamics(
    energy_fn: torch.nn.Module,
    v_query: Tensor,
    v_init: Tensor,
    lr: float = 0.01,
    noise_scale: float = 0.003,
    max_steps: int = 100,
    target_norm: float | None = None,
    energy_threshold: float | None = None,
    plateau_patience: int = 10,
    plateau_delta: float = 1e-4,
    v_target: Tensor | None = None,
    # Underdamped parameters
    friction: float = 0.5,
    mass: float = 1.0,
    track_vectors: bool = False,
) -> LangevinResult:
    """
    Underdamped (second-order) Langevin Dynamics.

    Adds an auxiliary momentum variable p (like HMC but without
    the Metropolis-Hastings accept/reject step):

        p_{t+1} = (1 - γ)·p_t - η·∇_V E + √(2γη/m)·ε
        V_{t+1} = V_t + (η/m)·p_{t+1}

    where γ is the friction coefficient and m is the mass.

    Physics intuition: particles have inertia and can overshoot
    energy barriers, escaping local minima more effectively than
    overdamped dynamics. The friction γ controls the balance:
        - γ → 0: pure Hamiltonian (no dissipation, oscillatory)
        - γ → 1: strongly damped (approaches overdamped)
        - γ ≈ 0.3-0.7: optimal for most energy landscapes

    Expected 2-5× faster convergence than overdamped, especially
    for Deep Thinking (200-500 steps).

    References:
        - "Gradient-Adjusted Underdamped Langevin Dynamics" (SIAM JUQ 2025)
        - "Underdamped Diffusion Bridges" (arXiv:2503.01006)

    Args:
        energy_fn:       Energy model.
        v_query:         [B, D] anchor vector.
        v_init:          [B, D] starting point.
        lr:              Step size η.
        noise_scale:     Base noise magnitude.
        max_steps:       Maximum iterations.
        target_norm:     Sphere projection radius.
        energy_threshold: Early stop threshold.
        plateau_patience: Plateau detection patience.
        plateau_delta:   Minimum improvement.
        v_target:        [B, D] ground truth (monitoring only).
        friction:        Damping coefficient γ ∈ (0, 1].
                         Lower = more inertia, higher = more damping.
        mass:            Particle mass m. Higher mass = more inertia.

    Returns:
        LangevinResult with final vectors and diagnostics.
    """
    v_current = v_init.clone().detach()
    momentum = torch.zeros_like(v_current)  # auxiliary momentum variable p

    trajectory: list[float] = []
    cos_trajectory: list[float] = []
    v_trajectory: list[Tensor] = []
    best_energy = float("inf")
    v_best = v_current.clone()
    plateau_counter = 0
    if track_vectors:
        v_trajectory.append(v_current.detach().cpu().clone())

    for step in range(max_steps):
        energy, grad = energy_fn.energy_and_grad(v_query, v_current)
        e_mean = energy.mean().item()
        trajectory.append(e_mean)

        if v_target is not None:
            cos = F.cosine_similarity(v_current, v_target, dim=-1).mean().item()
            cos_trajectory.append(cos)

        # Early stopping
        should_stop, best_energy, plateau_counter, improved = _check_early_stop(
            e_mean, best_energy, plateau_counter,
            energy_threshold, plateau_patience, plateau_delta,
        )
        if improved:
            v_best = v_current.clone()
        if should_stop:
            return LangevinResult(
                v_final=v_best,
                v_last=v_current.clone(),
                trajectory=trajectory,
                cos_trajectory=cos_trajectory,
                v_trajectory=v_trajectory,
                num_steps=step + 1, stopped_early=True,
            )

        # Underdamped update:
        # p_{t+1} = (1-γ)·p_t - η·∇E + √(2γη/m)·ε
        thermal_noise = torch.randn_like(v_current) * (
            2 * friction * lr * noise_scale / mass
        ) ** 0.5
        momentum = (1.0 - friction) * momentum - lr * grad + thermal_noise

        # Position update: V_{t+1} = V_t + (η/m)·p_{t+1}
        position_update = (lr / mass) * momentum

        # Tangent projection (apply to position update, not momentum directly)
        if target_norm is not None:
            position_update = _tangent_projection(position_update, v_current)

        v_current = v_current + position_update

        # OOD projection
        if target_norm is not None:
            v_current = _project_to_sphere(v_current, target_norm)
            # Also project momentum to tangent space to prevent norm-fighting
            momentum = momentum - (
                (momentum * F.normalize(v_current, dim=-1)).sum(dim=-1, keepdim=True)
                * F.normalize(v_current, dim=-1)
            )
        if track_vectors:
            v_trajectory.append(v_current.detach().cpu().clone())

    with torch.no_grad():
        final_energy = energy_fn(v_query, v_current).mean().item()
    if final_energy < best_energy:
        v_best = v_current.clone()

    return LangevinResult(
        v_final=v_best,
        v_last=v_current.clone(),
        trajectory=trajectory,
        cos_trajectory=cos_trajectory,
        v_trajectory=v_trajectory,
        num_steps=max_steps, stopped_early=False,
    )


# ============================================================
# Unified API
# ============================================================

def run_langevin(
    method: str | LangevinMethod,
    energy_fn: torch.nn.Module,
    v_query: Tensor,
    v_init: Tensor,
    lr: float = 0.01,
    noise_scale: float = 0.003,
    max_steps: int = 100,
    target_norm: float | None = None,
    energy_threshold: float | None = None,
    plateau_patience: int = 10,
    plateau_delta: float = 1e-4,
    v_target: Tensor | None = None,
    track_vectors: bool = False,
    **method_kwargs,
) -> LangevinResult:
    """
    Unified entry point for all Langevin dynamics variants.

    Args:
        method: "overdamped", "pid", or "underdamped".
        **method_kwargs: Method-specific parameters:
            - overdamped: momentum_beta
            - pid: kp, ki, kd, integral_decay
            - underdamped: friction, mass
        (all other args same as individual functions)

    Returns:
        LangevinResult
    """
    method = LangevinMethod(method)
    common = dict(
        energy_fn=energy_fn, v_query=v_query, v_init=v_init,
        lr=lr, noise_scale=noise_scale, max_steps=max_steps,
        target_norm=target_norm, energy_threshold=energy_threshold,
        plateau_patience=plateau_patience, plateau_delta=plateau_delta,
        v_target=v_target,
        track_vectors=track_vectors,
    )

    if method == LangevinMethod.OVERDAMPED:
        return langevin_dynamics(**common, **method_kwargs)
    elif method == LangevinMethod.PID:
        return pid_langevin_dynamics(**common, **method_kwargs)
    elif method == LangevinMethod.UNDERDAMPED:
        return underdamped_langevin_dynamics(**common, **method_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
