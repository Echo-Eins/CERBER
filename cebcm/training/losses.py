"""
Loss functions for CEBCM training.

Stage 1: Margin contrastive loss + gradient penalty.
Later stages: InfoNCE, curriculum-aware losses.

Spec reference: §5.4, §5.6, Appendix A.5
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def margin_contrastive_loss(
    e_pos: Tensor,
    e_neg: Tensor,
    margin: float = 1.0,
) -> Tensor:
    """
    Margin-based contrastive loss: E_pos should be at least `margin` lower than E_neg.

    L = mean(relu(E_pos - E_neg + margin))

    Args:
        e_pos: [B] energy of positive pairs (should be low)
        e_neg: [B] energy of negative pairs (should be high)
        margin: minimum energy gap

    Returns:
        Scalar loss
    """
    return F.relu(e_pos - e_neg + margin).mean()


def gradient_penalty(
    energy_fn: torch.nn.Module,
    v_query: Tensor,
    v_candidate: Tensor,
) -> Tensor:
    """
    Gradient penalty on ∇_{V_candidate} E — enforces Lipschitz smoothness.

    Penalizes ||∇E||² to prevent sharp energy landscapes
    that break Langevin dynamics.

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
