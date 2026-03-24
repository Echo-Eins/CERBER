"""
Energy Matching loss functions for simulation-free EBM training.

Implements the Energy Matching objective (Balcerak et al., NeurIPS 2025,
arXiv:2504.10612) adapted for SONAR sentence embeddings (1024d, concentrated
near ||x|| ≈ 0.2051).

Three loss variants:
    1. energy_matching_loss     — Standard MSE: ||-∇E(x_t) - u_t||²
    2. energy_matching_cosine   — Cosine direction + magnitude: for 1-Lipschitz nets
    3. energy_matching_weighted — σ(t)-weighted: emphasizes near-data regime

Mathematical basis:
    Given data x₁ ~ p_data and prior x₀ ~ p_prior:
        x_t = (1-t)·x₀ + t·x₁           (linear OT path)
        u_t = x₁ - x₀                    (conditional velocity)

    Energy Matching trains E_θ such that:
        -∇_x E_θ(x_t) ≈ u_t

    Loss: L = E_{t,x₀,x₁}[ ||-∇E(x_t) - u_t||² ]

Spec reference: §10.5, tasks/todo.md §0.4
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor


def energy_matching_loss(
    energy_fn: torch.nn.Module,
    x_data: Tensor,
    prior_std: float = 0.00641,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> Tensor:
    """
    Standard Energy Matching loss (MSE variant).

    L = E_{t~U(t_min,t_max), x₁~data, x₀~prior} [ ||-∇E(x_t) - u_t||² ]

    Args:
        energy_fn: E(x) → [B] scalar energy.
        x_data: [B, D] clean SONAR embeddings (x₁).
        prior_std: Std of isotropic Gaussian prior. Default: target_norm/√d.
        t_min: Minimum interpolation time.
        t_max: Maximum interpolation time (< 1.0 avoids singularity at data).

    Returns:
        Scalar loss (mean over batch).
    """
    B, D = x_data.shape
    device = x_data.device

    # Sample t ~ U(t_min, t_max)
    t = torch.rand(B, 1, device=device) * (t_max - t_min) + t_min

    # Sample prior x₀ ~ N(0, σ²I)
    x0 = torch.randn(B, D, device=device) * prior_std

    # OT path: x_t = (1-t)·x₀ + t·x₁
    x_t = (1 - t) * x0 + t * x_data

    # Target velocity: u_t = x₁ - x₀
    u_t = x_data - x0

    # Compute ∇_x E(x_t)
    x_t_grad = x_t.detach().requires_grad_(True)
    energy = energy_fn(x_t_grad)
    grad_E = torch.autograd.grad(
        energy.sum(), x_t_grad, create_graph=True
    )[0]  # [B, D]

    # Loss: ||-∇E - u_t||²
    loss = ((-grad_E - u_t) ** 2).sum(dim=-1).mean()

    return loss


def energy_matching_cosine(
    energy_fn: torch.nn.Module,
    x_data: Tensor,
    prior_std: float = 0.00641,
    t_min: float = 0.0,
    t_max: float = 1.0,
    magnitude_weight: float = 0.1,
) -> Tensor:
    """
    Cosine Energy Matching loss — adapted for 1-Lipschitz networks.

    Since OrthoLinear + GroupSort caps ||∇E|| ≈ O(1) while ||u_t|| can be much
    larger, MSE loss is dominated by magnitude mismatch. Cosine loss separates
    direction from magnitude:

    L = (1 - cos(-∇E, u_t)) + λ·(||∇E|| - ||u_t||)²

    Direction matching (cosine) is scale-invariant and immediately trainable.
    Magnitude term is optional and down-weighted.

    Args:
        energy_fn: E(x) → [B] scalar energy.
        x_data: [B, D] clean SONAR embeddings.
        prior_std: Std of prior distribution.
        t_min: Min interpolation time.
        t_max: Max interpolation time.
        magnitude_weight: Weight λ for magnitude loss term.

    Returns:
        Scalar loss.
    """
    B, D = x_data.shape
    device = x_data.device

    t = torch.rand(B, 1, device=device) * (t_max - t_min) + t_min
    x0 = torch.randn(B, D, device=device) * prior_std
    x_t = (1 - t) * x0 + t * x_data
    u_t = x_data - x0

    x_t_grad = x_t.detach().requires_grad_(True)
    energy = energy_fn(x_t_grad)
    grad_E = torch.autograd.grad(
        energy.sum(), x_t_grad, create_graph=True
    )[0]

    neg_grad = -grad_E

    # Direction loss: 1 - cos(−∇E, u_t)
    cos_sim = F.cosine_similarity(neg_grad, u_t, dim=-1)
    direction_loss = (1 - cos_sim).mean()

    # Magnitude loss: (||∇E|| - ||u_t||)²
    mag_loss = ((grad_E.norm(dim=-1) - u_t.norm(dim=-1)) ** 2).mean()

    return direction_loss + magnitude_weight * mag_loss


def energy_matching_weighted(
    energy_fn: torch.nn.Module,
    x_data: Tensor,
    prior_std: float = 0.00641,
    t_min: float = 0.0,
    t_max: float = 1.0,
    near_data_weight: float = 3.0,
) -> Tensor:
    """
    Weighted Energy Matching — emphasizes near-data regime (t close to 1).

    The energy landscape near data is most critical for Langevin refinement.
    We weight the loss by w(t) = 1 + (near_data_weight - 1) · t², giving
    higher weight to samples closer to the data manifold.

    This addresses the "two-phase" behavior of EM: the near-data regime
    (EBM-like) is more important for CERBER than the transport regime.

    Args:
        energy_fn: E(x) → [B] scalar energy.
        x_data: [B, D] clean embeddings.
        prior_std: Prior std.
        t_min: Min time.
        t_max: Max time.
        near_data_weight: Weight multiplier at t=1. Default 3.0 (3x more weight).

    Returns:
        Scalar loss.
    """
    B, D = x_data.shape
    device = x_data.device

    t = torch.rand(B, 1, device=device) * (t_max - t_min) + t_min
    x0 = torch.randn(B, D, device=device) * prior_std
    x_t = (1 - t) * x0 + t * x_data
    u_t = x_data - x0

    x_t_grad = x_t.detach().requires_grad_(True)
    energy = energy_fn(x_t_grad)
    grad_E = torch.autograd.grad(
        energy.sum(), x_t_grad, create_graph=True
    )[0]

    # Per-sample loss
    per_sample_loss = ((-grad_E - u_t) ** 2).sum(dim=-1)  # [B]

    # Weight: w(t) = 1 + (w_max - 1) · t²
    weights = 1.0 + (near_data_weight - 1.0) * (t.squeeze(-1) ** 2)  # [B]

    return (weights * per_sample_loss).mean() / weights.mean()


def compute_ot_path(
    x_data: Tensor,
    t: Tensor,
    prior_std: float = 0.00641,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute OT interpolation path and target velocity.

    Args:
        x_data: [B, D] data samples (x₁).
        t: [B, 1] interpolation times.
        prior_std: Prior distribution std.

    Returns:
        (x_t [B,D], u_t [B,D], x0 [B,D]):
            x_t: interpolated points
            u_t: target velocity at x_t
            x0: prior samples used
    """
    B, D = x_data.shape
    device = x_data.device

    x0 = torch.randn(B, D, device=device) * prior_std
    x_t = (1 - t) * x0 + t * x_data
    u_t = x_data - x0

    return x_t, u_t, x0


@torch.no_grad()
def generate_samples_ode(
    energy_fn: torch.nn.Module,
    num_samples: int,
    dim: int = 1024,
    num_steps: int = 100,
    prior_std: float = 0.00641,
    target_norm: float | None = 0.2051,
    device: torch.device | str = "cuda",
) -> Tensor:
    """
    Generate samples by integrating dx/dt = -∇E(x) from prior.

    Args:
        energy_fn: Trained energy model.
        num_samples: Number of samples to generate.
        dim: Embedding dimension.
        num_steps: Euler integration steps.
        prior_std: Prior distribution std.
        target_norm: If set, project samples to sphere after each step.
        device: Compute device.

    Returns:
        [num_samples, dim] generated embeddings.
    """
    device = torch.device(device)
    dt = 1.0 / num_steps

    # Initialize from prior
    x = torch.randn(num_samples, dim, device=device) * prior_std

    for step in range(num_steps):
        x_grad = x.detach().requires_grad_(True)
        with torch.enable_grad():
            energy = energy_fn(x_grad)
            grad = torch.autograd.grad(energy.sum(), x_grad)[0]
        x = x - dt * grad  # Euler step: dx = -∇E · dt

        if target_norm is not None:
            x = F.normalize(x, dim=-1) * target_norm

    return x


@torch.no_grad()
def generate_samples_langevin(
    energy_fn: torch.nn.Module,
    num_samples: int,
    dim: int = 1024,
    num_steps: int = 200,
    lr: float = 0.01,
    noise_scale: float = 0.005,
    target_norm: float | None = 0.2051,
    device: torch.device | str = "cuda",
) -> Tensor:
    """
    Generate samples via Langevin dynamics from prior.

    x_{k+1} = x_k - η·∇E(x_k) + √(2η)·ε

    Args:
        energy_fn: Trained energy model.
        num_samples: Number of samples.
        dim: Embedding dimension.
        num_steps: Langevin steps.
        lr: Step size η.
        noise_scale: Noise multiplier.
        target_norm: Sphere projection target.
        device: Compute device.

    Returns:
        [num_samples, dim] generated embeddings.
    """
    device = torch.device(device)

    x = torch.randn(num_samples, dim, device=device) * (target_norm or 0.2051) / (dim ** 0.5)

    for step in range(num_steps):
        x_grad = x.detach().requires_grad_(True)
        with torch.enable_grad():
            energy = energy_fn(x_grad)
            grad = torch.autograd.grad(energy.sum(), x_grad)[0]

        noise = torch.randn_like(x) * noise_scale
        x = x - lr * grad + (2 * lr) ** 0.5 * noise

        if target_norm is not None:
            x = F.normalize(x, dim=-1) * target_norm

    return x
