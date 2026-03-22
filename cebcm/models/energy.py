"""
SimpleEnergy — lightweight energy function for Stage 1 Denoising PoC.

Architecture: MLP over concatenated pair features [V_q; V_c; V_q-V_c; V_q*V_c].
Input:  two vectors of dim d → concatenated to 4d
Output: scalar energy E ∈ ℝ (lower = better match)

Spec reference: §13.1, §5.2 Mode A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimpleEnergy(nn.Module):
    """
    Pairwise energy function: E(V_orig, V_candidate) → scalar.

    Lower energy means V_candidate is a better match for V_orig.
    Uses four interaction features: concat, difference, element-wise product.

    Args:
        dim: Embedding dimension (1024 for SONAR).
        hidden_dims: List of hidden layer sizes.
        spectral_norm: Whether to apply spectral normalization for Lipschitz smoothness.
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dims: list[int] | None = None,
        spectral_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 512]

        input_dim = dim * 4  # [V_q; V_c; V_q - V_c; V_q * V_c]
        layers: list[nn.Module] = []

        prev_dim = input_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            if spectral_norm:
                linear = nn.utils.parametrizations.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Final projection to scalar
        final_linear = nn.Linear(prev_dim, 1)
        if spectral_norm:
            final_linear = nn.utils.parametrizations.spectral_norm(final_linear)
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, v_query: Tensor, v_candidate: Tensor) -> Tensor:
        """
        Compute energy for a pair of embeddings.

        Args:
            v_query:     [B, D] anchor/original embedding
            v_candidate: [B, D] candidate embedding

        Returns:
            [B] energy scalars
        """
        # Normalize to unit sphere — eliminates norm as a learnable feature,
        # forcing the model to learn directional (cosine) similarity.
        v_query = F.normalize(v_query, dim=-1)
        v_candidate = F.normalize(v_candidate, dim=-1)
        diff = v_query - v_candidate
        prod = v_query * v_candidate
        x = torch.cat([v_query, v_candidate, diff, prod], dim=-1)
        return self.net(x).squeeze(-1)

    def energy_and_grad(
        self, v_query: Tensor, v_candidate: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute energy and gradient w.r.t. v_candidate in one pass.

        Used by Langevin dynamics to get ∇_{V_candidate} E.

        Returns:
            (energy [B], grad [B, D])
        """
        v_candidate = v_candidate.detach().requires_grad_(True)
        energy = self.forward(v_query, v_candidate)
        grad = torch.autograd.grad(
            energy.sum(), v_candidate, create_graph=False
        )[0]
        return energy.detach(), grad.detach()
