"""
SimpleEnergy — lightweight energy function for Stage 1 Denoising PoC.

Architecture: MLP over concatenated pair features [V_q; V_c; V_q-V_c; V_q*V_c].
Input:  two vectors of dim d → concatenated to 4d
Output: scalar energy E ∈ ℝ (lower = better match)

Supports two normalization modes:
    - "spectral_norm": Classic spectral normalization (Lipschitz via σ_max only)
    - "orthonorm": Bjorck orthonormalization (all σ_i ≈ 1, no gradient attenuation)

Supports three activation modes:
    - "relu": Standard ReLU (legacy, not recommended with Lipschitz constraints)
    - "groupsort": GroupSort with group_size=2 (MaxMin), 1-Lipschitz by construction
    - "lipschitz_spline": Learnable piecewise-linear, 1-Lipschitz, most expressive

Spec reference: §13.1, §5.2 Mode A, §5.6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cebcm.models.activations import GroupSort, LipschitzLinearSpline
from cebcm.models.normalization import OrthoLinear


class SimpleEnergy(nn.Module):
    """
    Pairwise energy function: E(V_orig, V_candidate) → scalar.

    Lower energy means V_candidate is a better match for V_orig.
    Uses four interaction features: concat, difference, element-wise product.

    Args:
        dim: Embedding dimension (1024 for SONAR).
        hidden_dims: List of hidden layer sizes.
        norm_mode: Weight normalization — "spectral_norm" or "orthonorm".
        activation: Activation function — "relu", "groupsort", or "lipschitz_spline".
        spectral_norm: Legacy flag. If True and norm_mode is not set, uses spectral_norm.
                       Deprecated: use norm_mode instead.
        ortho_n_iters: Number of Bjorck iterations for orthonormalization.
        groupsort_size: Group size for GroupSort activation (2 = MaxMin).
        spline_num_knots: Number of knots for LipschitzLinearSpline.
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dims: list[int] | None = None,
        norm_mode: str = "orthonorm",
        activation: str = "groupsort",
        spectral_norm: bool | None = None,
        ortho_n_iters: int = 15,
        groupsort_size: int = 2,
        spline_num_knots: int = 4,
    ):
        super().__init__()

        # Handle legacy spectral_norm flag
        if spectral_norm is not None:
            if spectral_norm:
                norm_mode = "spectral_norm"
            else:
                norm_mode = "none"

        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512]

        self.norm_mode = norm_mode
        self.activation_name = activation

        input_dim = dim * 4  # [V_q; V_c; V_q - V_c; V_q * V_c]
        layers: list[nn.Module] = []

        prev_dim = input_dim
        for h_dim in hidden_dims:
            # Ensure hidden dim is divisible by groupsort_size
            if activation == "groupsort" and h_dim % groupsort_size != 0:
                h_dim = (h_dim // groupsort_size) * groupsort_size

            # Linear layer with chosen normalization
            linear = self._make_linear(prev_dim, h_dim, norm_mode, ortho_n_iters)
            layers.append(linear)

            # Activation
            act = self._make_activation(activation, h_dim, groupsort_size, spline_num_knots)
            layers.append(act)

            prev_dim = h_dim

        # Final projection to scalar (no activation)
        final_linear = self._make_linear(prev_dim, 1, norm_mode, ortho_n_iters)
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

    @staticmethod
    def _make_linear(
        in_dim: int,
        out_dim: int,
        norm_mode: str,
        ortho_n_iters: int,
    ) -> nn.Module:
        """Create a linear layer with the specified normalization."""
        if norm_mode == "orthonorm":
            return OrthoLinear(in_dim, out_dim, bias=True, n_iters=ortho_n_iters)
        elif norm_mode == "spectral_norm":
            linear = nn.Linear(in_dim, out_dim)
            return nn.utils.parametrizations.spectral_norm(linear)
        else:
            return nn.Linear(in_dim, out_dim)

    @staticmethod
    def _make_activation(
        activation: str,
        dim: int,
        groupsort_size: int,
        spline_num_knots: int,
    ) -> nn.Module:
        """Create the specified activation function."""
        if activation == "groupsort":
            return GroupSort(group_size=groupsort_size)
        elif activation == "lipschitz_spline":
            return LipschitzLinearSpline(num_features=dim, num_knots=spline_num_knots)
        elif activation == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

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
