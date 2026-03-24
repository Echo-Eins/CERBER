"""
Loss functions for CEBCM training.

Stage 1: Multi-Scale DSM with optional directional objective, sigma curricula,
and sphere-tangent projection support.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# Stage 1: MDSM (Multi-Scale Denoising Score Matching)
# ============================================================

def _sample_sigma(
    batch_size: int,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    sigma_sampling: str,
    edm_p_mean: float,
    edm_p_std: float,
) -> Tensor:
    if sigma_sampling == "loguniform":
        log_sigma = torch.rand(batch_size, 1, device=device) * (
            math.log(sigma_max) - math.log(sigma_min)
        ) + math.log(sigma_min)
        return log_sigma.exp()
    if sigma_sampling == "edm":
        return torch.exp(
            torch.randn(batch_size, 1, device=device) * edm_p_std + edm_p_mean
        ).clamp(min=sigma_min, max=sigma_max)
    raise ValueError(f"Unknown sigma_sampling: {sigma_sampling}")


def multiscale_dsm_loss(
    energy_fn: torch.nn.Module,
    v_clean: Tensor,
    sigma_min: float = 0.01,
    sigma_max: float = 0.5,
    relative_noise: bool = True,
    sigma_sampling: str = "loguniform",
    sigma_weighting: str = "sigma2",
    directional: bool = False,
    magnitude_aux_weight: float = 0.0,
    tangent_projection: bool = False,
    edm_p_mean: float = -1.2,
    edm_p_std: float = 1.2,
    cosine_eps: float = 1e-4,
    norm_floor: float = 1e-4,
) -> Tensor:
    """
    Multi-Scale DSM loss with optional directional mode.

    directional=False:
        Full score matching (direction + magnitude).
    directional=True:
        Cosine DSM on score direction with optional weak magnitude auxiliary.
    """
    batch_size, _ = v_clean.shape
    device = v_clean.device

    sigma = _sample_sigma(
        batch_size=batch_size,
        device=device,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_sampling=sigma_sampling,
        edm_p_mean=edm_p_mean,
        edm_p_std=edm_p_std,
    )

    noise = torch.randn_like(v_clean)
    if relative_noise:
        norms = v_clean.norm(dim=-1, keepdim=True).clamp(min=norm_floor)
        v_noisy = v_clean + noise * sigma * norms
        sigma_eff_sq = ((sigma * norms) ** 2).clamp(min=1e-6)
    else:
        v_noisy = v_clean + noise * sigma
        sigma_eff_sq = (sigma ** 2).clamp(min=1e-6)

    v_noisy_grad = v_noisy.detach().requires_grad_(True)
    energy = energy_fn(v_clean, v_noisy_grad, sigma=sigma.detach())
    grad_energy = torch.autograd.grad(
        energy.sum(),
        v_noisy_grad,
        create_graph=True,
    )[0]

    # For energy descent v <- v - lr * grad(E), the target gradient must point
    # from clean to noisy (opposite to denoising score).
    target_score = (v_noisy.detach() - v_clean) / sigma_eff_sq
    target_score = torch.nan_to_num(target_score, nan=0.0, posinf=1e4, neginf=-1e4)
    grad_energy = torch.nan_to_num(grad_energy, nan=0.0, posinf=1e4, neginf=-1e4)

    if tangent_projection:
        v_hat = F.normalize(v_noisy.detach(), dim=-1)
        target_score = target_score - (
            (target_score * v_hat).sum(dim=-1, keepdim=True) * v_hat
        )
        grad_energy = grad_energy - (
            (grad_energy * v_hat).sum(dim=-1, keepdim=True) * v_hat
        )

    if directional:
        cosine = F.cosine_similarity(
            grad_energy, target_score, dim=-1, eps=cosine_eps
        ).clamp(min=-1.0, max=1.0)
        loss_per_sample = 1.0 - cosine

        if magnitude_aux_weight > 0:
            grad_norm = grad_energy.norm(dim=-1).clamp(min=norm_floor, max=1e4)
            target_norm = target_score.norm(dim=-1).clamp(min=norm_floor, max=1e4)
            mag_aux = F.smooth_l1_loss(
                torch.log(grad_norm),
                torch.log(target_norm),
                reduction="none",
            )
            loss_per_sample = loss_per_sample + magnitude_aux_weight * mag_aux
    else:
        score_diff = grad_energy - target_score
        loss_per_sample = (score_diff ** 2).sum(dim=-1)

    loss_per_sample = torch.nan_to_num(loss_per_sample, nan=1e4, posinf=1e4, neginf=1e4)

    if sigma_weighting == "sigma2":
        weights = sigma_eff_sq.squeeze(-1)
    elif sigma_weighting == "uniform":
        weights = torch.ones_like(loss_per_sample)
    elif sigma_weighting == "inv_sigma2":
        weights = 1.0 / sigma_eff_sq.squeeze(-1).clamp(min=1e-8)
    else:
        raise ValueError(f"Unknown sigma_weighting: {sigma_weighting}")

    weights = torch.nan_to_num(weights, nan=1.0, posinf=1e4, neginf=1.0)
    # Keep weighting relative across sigma scales, but normalize absolute scale
    # so directional cosine losses do not collapse to near-zero gradient magnitudes.
    weights = weights / weights.mean().clamp(min=1e-8)
    return (weights * loss_per_sample).mean()


def dsm_loss_fixed_sigma(
    energy_fn: torch.nn.Module,
    v_clean: Tensor,
    sigma: float,
    relative_noise: bool = True,
    norm_floor: float = 1e-4,
) -> Tensor:
    """Single-scale DSM loss for debugging/ablation."""
    batch_size, _ = v_clean.shape
    device = v_clean.device

    noise = torch.randn_like(v_clean)

    if relative_noise:
        norms = v_clean.norm(dim=-1, keepdim=True).clamp(min=norm_floor)
        v_noisy = v_clean + noise * sigma * norms
        sigma_eff_sq = ((sigma * norms) ** 2).clamp(min=1e-6)
    else:
        v_noisy = v_clean + noise * sigma
        sigma_eff_sq = torch.full_like(v_noisy[:, :1], sigma**2).clamp(min=1e-6)

    sigma_tensor = torch.full((batch_size, 1), sigma, device=device)

    v_noisy_grad = v_noisy.detach().requires_grad_(True)
    energy = energy_fn(v_clean, v_noisy_grad, sigma=sigma_tensor)
    grad_energy = torch.autograd.grad(
        energy.sum(),
        v_noisy_grad,
        create_graph=True,
    )[0]

    target_score = (v_noisy.detach() - v_clean) / sigma_eff_sq
    target_score = torch.nan_to_num(target_score, nan=0.0, posinf=1e4, neginf=-1e4)
    grad_energy = torch.nan_to_num(grad_energy, nan=0.0, posinf=1e4, neginf=-1e4)
    score_diff = grad_energy - target_score
    return (score_diff ** 2).sum(dim=-1).mean()


# ============================================================
# Legacy: Margin Contrastive Loss
# ============================================================

def margin_contrastive_loss(
    e_pos: Tensor,
    e_neg: Tensor,
    margin: float = 1.0,
) -> Tensor:
    return F.relu(e_pos - e_neg + margin).mean()


# ============================================================
# Gradient Penalty
# ============================================================

def gradient_penalty(
    energy_fn: torch.nn.Module,
    v_query: Tensor,
    v_candidate: Tensor,
) -> Tensor:
    v_candidate = v_candidate.detach().requires_grad_(True)
    energy = energy_fn(v_query, v_candidate)
    grad = torch.autograd.grad(
        energy.sum(),
        v_candidate,
        create_graph=True,
    )[0]
    return (grad.norm(dim=-1) ** 2).mean()


# ============================================================
# Stage 2+: Focal-InfoNCE and Soft-InfoNCE
# ============================================================

def focal_infonce_loss(
    e_pos: Tensor,
    e_neg: Tensor,
    temperature: float = 0.07,
    focal_gamma: float = 2.0,
) -> Tensor:
    pos_logits = -e_pos / temperature
    neg_logits = -e_neg / temperature

    with torch.no_grad():
        neg_probs = F.softmax(neg_logits, dim=-1)
        focal_weights = neg_probs.pow(focal_gamma)
        focal_weights = focal_weights * (
            neg_logits.shape[-1] / focal_weights.sum(dim=-1, keepdim=True)
        )

    weighted_neg_logits = neg_logits + focal_weights.log()
    all_logits = torch.cat([pos_logits.unsqueeze(-1), weighted_neg_logits], dim=-1)
    labels = torch.zeros(e_pos.shape[0], dtype=torch.long, device=e_pos.device)
    return F.cross_entropy(all_logits, labels)


def soft_infonce_loss(
    e_pos: Tensor,
    e_neg: Tensor,
    temperature: float = 0.07,
    softness: float = 0.5,
) -> Tensor:
    pos_logits = -e_pos / temperature

    with torch.no_grad():
        neg_probs = F.softmax(-e_neg / temperature, dim=-1)
        tau_per_neg = temperature * (1.0 + softness * (1.0 - neg_probs))

    neg_logits = -e_neg / tau_per_neg
    all_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
    labels = torch.zeros(e_pos.shape[0], dtype=torch.long, device=e_pos.device)
    return F.cross_entropy(all_logits, labels)
