"""
Loss functions for CEBCM training.

Stage 1: Multi-Scale Denoising Score Matching (MDSM) + gradient penalty.
         Replaces margin contrastive loss with score matching for correct
         gradient learning across all noise levels.
Later stages: InfoNCE, Focal-InfoNCE, curriculum-aware losses.

References:
    - Li et al., "Learning Energy-Based Models in High-Dimensional Spaces
      with Multi-scale Denoising Score Matching" (Entropy 2023)
    - Song & Ermon, "Generative Modeling by Estimating Gradients of the
      Data Distribution" (NeurIPS 2019)
    - Hou & Li, "Improving Contrastive Learning of Sentence Embeddings
      with Focal-InfoNCE" (EMNLP 2023)

Spec reference: §5.4, §5.6, Appendix A.5
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# Stage 1: MDSM (Multi-Scale Denoising Score Matching)
# ============================================================

def multiscale_dsm_loss(
    energy_fn: torch.nn.Module,
    v_clean: Tensor,
    sigma_min: float = 0.01,
    sigma_max: float = 0.5,
    relative_noise: bool = True,
) -> Tensor:
    """
    Multi-Scale Denoising Score Matching loss for energy-based models.

    Trains the energy function so that ∇_V E(V_query, V_candidate) correctly
    points from noisy vectors back toward clean vectors at ALL noise levels.
    This is exactly what Langevin dynamics needs for correct refinement.

    The noise level σ is sampled continuously from LogUniform(σ_min, σ_max),
    ensuring the energy landscape is smooth across all scales — from coarse
    global structure (high σ) to fine-grained adjustments (low σ).

    DSM objective: ||∇_V E(V_q, V_noisy) - (V_clean - V_noisy) / σ²||²

    The score function ∇ log p_σ(V_noisy | V_clean) = -(V_noisy - V_clean) / σ²
    for Gaussian perturbation. We train the energy gradient to match this.

    Args:
        energy_fn: Energy model E(v_query, v_candidate) → scalar.
        v_clean: [B, D] clean embeddings (from SONAR encoder).
        sigma_min: Minimum noise scale (for fine structure).
        sigma_max: Maximum noise scale (for global structure).
        relative_noise: If True, scale noise relative to embedding norm.

    Returns:
        Scalar DSM loss (mean over batch).
    """
    B, D = v_clean.shape
    device = v_clean.device

    # Sample σ from LogUniform(σ_min, σ_max) — equal density per octave
    log_sigma = torch.rand(B, 1, device=device) * (
        math.log(sigma_max) - math.log(sigma_min)
    ) + math.log(sigma_min)
    sigma = log_sigma.exp()  # [B, 1]

    # Generate noise
    noise = torch.randn_like(v_clean)  # [B, D]

    if relative_noise:
        # Scale noise relative to embedding norm (as in CEBCM spec)
        norms = v_clean.norm(dim=-1, keepdim=True)  # [B, 1]
        v_noisy = v_clean + noise * sigma * norms
    else:
        v_noisy = v_clean + noise * sigma

    # Compute energy gradient w.r.t. v_noisy (the candidate)
    v_noisy_grad = v_noisy.detach().requires_grad_(True)
    energy = energy_fn(v_clean, v_noisy_grad)
    grad_energy = torch.autograd.grad(
        energy.sum(), v_noisy_grad, create_graph=True
    )[0]  # [B, D]

    # Target score: ∇ log p_σ(V_noisy | V_clean) = -(V_noisy - V_clean) / σ²
    # With relative noise, the effective σ_eff = σ * ||V_clean||
    if relative_noise:
        sigma_eff_sq = (sigma * norms) ** 2  # [B, 1]
    else:
        sigma_eff_sq = sigma ** 2  # [B, 1]

    target_score = -(v_noisy.detach() - v_clean) / sigma_eff_sq  # [B, D]

    # DSM loss: ||∇E - target_score||² with σ²-weighting for scale balance
    # Weight by σ² to equalize contribution across noise levels
    # (low σ → small gradients → needs amplification)
    score_diff = grad_energy - target_score  # [B, D]
    loss_per_sample = (score_diff ** 2).sum(dim=-1)  # [B]

    # Weight by σ² for balanced multi-scale learning
    weights = sigma_eff_sq.squeeze(-1)  # [B]
    weighted_loss = (weights * loss_per_sample).mean()

    return weighted_loss


def dsm_loss_fixed_sigma(
    energy_fn: torch.nn.Module,
    v_clean: Tensor,
    sigma: float,
    relative_noise: bool = True,
) -> Tensor:
    """
    Single-scale DSM loss — useful for debugging and ablation.

    Same as multiscale_dsm_loss but with a fixed noise level.

    Args:
        energy_fn: Energy model.
        v_clean: [B, D] clean embeddings.
        sigma: Fixed noise scale.
        relative_noise: Scale relative to norm.

    Returns:
        Scalar DSM loss.
    """
    B, D = v_clean.shape
    device = v_clean.device

    noise = torch.randn_like(v_clean)

    if relative_noise:
        norms = v_clean.norm(dim=-1, keepdim=True)
        v_noisy = v_clean + noise * sigma * norms
        sigma_eff_sq = (sigma * norms) ** 2
    else:
        v_noisy = v_clean + noise * sigma
        sigma_eff_sq = sigma ** 2

    v_noisy_grad = v_noisy.detach().requires_grad_(True)
    energy = energy_fn(v_clean, v_noisy_grad)
    grad_energy = torch.autograd.grad(
        energy.sum(), v_noisy_grad, create_graph=True
    )[0]

    target_score = -(v_noisy.detach() - v_clean) / sigma_eff_sq
    score_diff = grad_energy - target_score
    loss = (score_diff ** 2).sum(dim=-1).mean()

    return loss


# ============================================================
# Legacy: Margin Contrastive Loss (kept for backward compat)
# ============================================================

def margin_contrastive_loss(
    e_pos: Tensor,
    e_neg: Tensor,
    margin: float = 1.0,
) -> Tensor:
    """
    Margin-based contrastive loss: E_pos should be at least `margin` lower than E_neg.

    L = mean(relu(E_pos - E_neg + margin))

    NOTE: Superseded by MDSM for Stage 1. Kept for backward compatibility
    and potential use in ablation studies.

    Args:
        e_pos: [B] energy of positive pairs (should be low)
        e_neg: [B] energy of negative pairs (should be high)
        margin: minimum energy gap

    Returns:
        Scalar loss
    """
    return F.relu(e_pos - e_neg + margin).mean()


# ============================================================
# Gradient Penalty
# ============================================================

def gradient_penalty(
    energy_fn: torch.nn.Module,
    v_query: Tensor,
    v_candidate: Tensor,
) -> Tensor:
    """
    Gradient penalty on ∇_{V_candidate} E — enforces Lipschitz smoothness.

    Penalizes ||∇E||² to prevent sharp energy landscapes
    that break Langevin dynamics.

    NOTE: With orthonormalization + GroupSort/LipschitzSpline, the network
    is already 1-Lipschitz by construction. This penalty becomes optional
    but can still help with energy landscape smoothness beyond the Lipschitz
    constraint on the network itself.

    Spec reference: §5.6

    Args:
        energy_fn: Energy model
        v_query:     [B, D]
        v_candidate: [B, D]

    Returns:
        Scalar penalty (mean of squared gradient norms)
    """
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
    """
    Focal-InfoNCE loss — re-weights negatives to focus on hard examples.

    Standard InfoNCE treats all negatives equally, so easy negatives dominate
    the gradient. Focal-InfoNCE (Hou & Li, EMNLP 2023) applies focal weighting:
    harder negatives (higher similarity to anchor) get higher weight in the loss.

    L = -log( exp(-E_pos/τ) / (exp(-E_pos/τ) + Σ w_i · exp(-E_neg_i/τ)) )

    where w_i = (p_i)^γ, p_i = softmax(-E_neg_i/τ) is the "hardness" of negative i.
    Higher γ → more focus on hard negatives. γ=0 recovers standard InfoNCE.

    Args:
        e_pos: [B] energy of positive pairs.
        e_neg: [B, N] energy of N negative pairs per sample.
        temperature: Softmax temperature τ.
        focal_gamma: Focal weighting exponent γ. Higher = harder focus.

    Returns:
        Scalar loss.
    """
    # Logits: lower energy = better match = higher logit
    pos_logits = -e_pos / temperature  # [B]
    neg_logits = -e_neg / temperature  # [B, N]

    # Compute negative weights via focal mechanism
    with torch.no_grad():
        neg_probs = F.softmax(neg_logits, dim=-1)  # [B, N]
        focal_weights = neg_probs.pow(focal_gamma)  # [B, N]
        # Normalize weights to sum to N (preserve scale)
        focal_weights = focal_weights * (neg_logits.shape[-1] / focal_weights.sum(dim=-1, keepdim=True))

    # Weighted denominator
    weighted_neg_logits = neg_logits + focal_weights.log()  # [B, N]

    # Full logits: [B, 1 + N] — positive first
    all_logits = torch.cat([pos_logits.unsqueeze(-1), weighted_neg_logits], dim=-1)

    # Cross-entropy: positive is index 0
    labels = torch.zeros(e_pos.shape[0], dtype=torch.long, device=e_pos.device)
    return F.cross_entropy(all_logits, labels)


def soft_infonce_loss(
    e_pos: Tensor,
    e_neg: Tensor,
    temperature: float = 0.07,
    softness: float = 0.5,
) -> Tensor:
    """
    Soft-InfoNCE loss — individual temperature per negative pair.

    SoftCSE (CIKM 2024): instead of a fixed temperature for all negatives,
    each negative gets a temperature adjusted by its similarity to the anchor.
    Harder negatives get lower effective temperature (sharper focus).

    τ_i = τ · (1 + softness · (1 - hardness_i))

    where hardness_i = softmax(-E_neg_i/τ) measures how close the negative
    is to being a positive.

    Args:
        e_pos: [B] energy of positive pairs.
        e_neg: [B, N] energy of N negative pairs per sample.
        temperature: Base temperature τ.
        softness: How much to adjust temperature per negative.

    Returns:
        Scalar loss.
    """
    pos_logits = -e_pos / temperature  # [B]

    # Per-negative adaptive temperature
    with torch.no_grad():
        neg_probs = F.softmax(-e_neg / temperature, dim=-1)  # [B, N]
        # Higher prob = harder negative → lower temperature
        tau_per_neg = temperature * (1.0 + softness * (1.0 - neg_probs))  # [B, N]

    neg_logits = -e_neg / tau_per_neg  # [B, N]

    all_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
    labels = torch.zeros(e_pos.shape[0], dtype=torch.long, device=e_pos.device)
    return F.cross_entropy(all_logits, labels)
