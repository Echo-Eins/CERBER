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
    Multi-Scale Denoising Score Matching via direction matching + σ-conditioning.

    Trains ∇_V E(V_query, V_candidate, σ) to point from noisy vectors back
    toward clean vectors at ALL noise levels. The energy function receives σ
    as input (NCSN-style) for richer per-noise-level representation.

    Direction matching (cosine loss) is used because 1-Lipschitz hidden layers
    bound ||∇E|| ≤ exp(scale) × ||W_final|| ≈ O(10-100), while the raw DSM
    target has ||target|| = sqrt(D)/(σ_eff) ≈ O(2000-16000). MSE against this
    target is fundamentally untrainable. Cosine loss is scale-invariant and
    immediately trainable — gradient MAGNITUDE is controlled by Langevin lr.

    Args:
        energy_fn: Energy model E(v_query, v_candidate, sigma) → scalar.
        v_clean: [B, D] clean embeddings (from SONAR encoder).
        sigma_min: Minimum noise scale (for fine structure).
        sigma_max: Maximum noise scale (for global structure).
        relative_noise: If True, scale noise relative to embedding norm.

    Returns:
        Scalar cosine DSM loss (mean over batch), range [0, 2].
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
        norms = v_clean.norm(dim=-1, keepdim=True)  # [B, 1]
        v_noisy = v_clean + noise * sigma * norms
    else:
        v_noisy = v_clean + noise * sigma

    # Compute energy gradient w.r.t. v_noisy, conditioned on σ
    v_noisy_grad = v_noisy.detach().requires_grad_(True)
    energy = energy_fn(v_clean, v_noisy_grad, sigma=sigma.detach())
    grad_energy = torch.autograd.grad(
        energy.sum(), v_noisy_grad, create_graph=True
    )[0]  # [B, D]

    # Target direction: points from noisy back toward clean.
    # Cosine loss: 1 - cos(∇E, target_direction), range [0, 2].
    target_direction = v_clean - v_noisy.detach()  # [B, D]
    cos_sim = F.cosine_similarity(grad_energy, target_direction, dim=-1)  # [B]
    loss = (1 - cos_sim).mean()

    return loss


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
    else:
        v_noisy = v_clean + noise * sigma

    sigma_tensor = torch.full((B, 1), sigma, device=device)

    v_noisy_grad = v_noisy.detach().requires_grad_(True)
    energy = energy_fn(v_clean, v_noisy_grad, sigma=sigma_tensor)
    grad_energy = torch.autograd.grad(
        energy.sum(), v_noisy_grad, create_graph=True
    )[0]

    # Cosine direction loss (consistent with multiscale_dsm_loss)
    target_direction = v_clean - v_noisy.detach()
    cos_sim = F.cosine_similarity(grad_energy, target_direction, dim=-1)
    loss = (1 - cos_sim).mean()

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
    energy = energy_fn(v_query, v_candidate)  # sigma auto-estimated
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
