"""
SimpleEnergy — lightweight energy function for Stage 1 Denoising PoC.

Architecture: MLP over concatenated pair features [V_q; V_c; V_q-V_c; V_q*V_c; σ_embed].
Input:  two vectors of dim d + noise level σ → concatenated to 4d + σ_embed_dim
Output: scalar energy E ∈ ℝ (lower = better match)

σ-conditioning (NCSN-style): the network sees the noise level σ, allowing it
to output scale-appropriate gradients. At small σ → large gradients (fine
corrections), at large σ → small gradients (coarse structure). This resolves
the magnitude mismatch between 1-Lipschitz hidden layers and DSM target scores.

Supports two normalization modes:
    - "spectral_norm": Classic spectral normalization (Lipschitz via σ_max only)
    - "orthonorm": Bjorck orthonormalization (all σ_i ≈ 1, no gradient attenuation)

Supports three activation modes:
    - "relu": Standard ReLU (legacy, not recommended with Lipschitz constraints)
    - "groupsort": GroupSort with group_size=2 (MaxMin), 1-Lipschitz by construction
    - "lipschitz_spline": Learnable piecewise-linear, 1-Lipschitz, most expressive

Spec reference: §13.1, §5.2 Mode A, §5.6
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cebcm.models.activations import GroupSort, LipschitzLinearSpline
from cebcm.models.normalization import OrthoLinear


# Number of sinusoidal frequencies for σ embedding
_SIGMA_EMBED_FREQS = 4
_SIGMA_EMBED_DIM = _SIGMA_EMBED_FREQS * 2  # sin + cos = 8 dims


class SimpleEnergy(nn.Module):
    """
    Pairwise energy function: E(V_orig, V_candidate, σ) → scalar.

    Lower energy means V_candidate is a better match for V_orig.
    Uses four interaction features plus sinusoidal σ embedding.

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

        # [V_q; V_c; V_q - V_c; V_q * V_c; σ_embed]
        input_dim = dim * 4 + _SIGMA_EMBED_DIM
        layers: list[nn.Module] = []

        # Register sinusoidal frequencies as buffer (not a parameter)
        freqs = torch.arange(1, _SIGMA_EMBED_FREQS + 1, dtype=torch.float32) * math.pi
        self.register_buffer("_sigma_freqs", freqs)

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

        # Final projection to scalar (no activation, unconstrained).
        # The hidden layers are 1-Lipschitz (OrthoLinear + GroupSort) for smooth
        # feature extraction. The final layer is a regular Linear so the energy
        # magnitude can be learned freely — with σ-conditioning the network can
        # output scale-appropriate gradients at each noise level.
        final_linear = nn.Linear(prev_dim, 1)
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

        # Learnable energy scale (log-parameterized for fast adaptation).
        # With σ-conditioning, the network adapts magnitude per noise level,
        # but a global scale factor still helps match the overall DSM target range.
        self.log_energy_scale = nn.Parameter(torch.tensor(0.0))

    def _embed_sigma(self, sigma: Tensor) -> Tensor:
        """
        Sinusoidal embedding of noise level σ.

        Maps log(σ) through multiple frequencies for rich representation
        of the noise scale. This allows the network to distinguish fine
        noise levels and adapt its gradient magnitude accordingly.

        Args:
            sigma: [B, 1] noise level (relative to embedding norm).

        Returns:
            [B, 8] sinusoidal features: [sin(π·logσ), cos(π·logσ), sin(2π·logσ), ...]
        """
        log_sigma = torch.log(sigma.clamp(min=1e-8))  # [B, 1]
        args = log_sigma * self._sigma_freqs  # [B, num_freqs]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 2*num_freqs]

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

    def forward(
        self, v_query: Tensor, v_candidate: Tensor, sigma: Tensor | None = None
    ) -> Tensor:
        """
        Compute energy for a pair of embeddings at noise level σ.

        Args:
            v_query:     [B, D] anchor/original embedding
            v_candidate: [B, D] candidate embedding
            sigma:       [B, 1] noise level. If None, estimated from
                         relative distance ||v_candidate - v_query|| / ||v_query||.

        Returns:
            [B] energy scalars
        """
        if sigma is None:
            sigma = self._estimate_sigma(v_query, v_candidate)

        diff = v_query - v_candidate
        prod = v_query * v_candidate
        sigma_emb = self._embed_sigma(sigma)  # [B, 8]
        x = torch.cat([v_query, v_candidate, diff, prod, sigma_emb], dim=-1)
        return self.log_energy_scale.exp() * self.net(x).squeeze(-1)

    def _estimate_sigma(self, v_query: Tensor, v_candidate: Tensor) -> Tensor:
        """
        Estimate relative noise level from distance between query and candidate.

        σ_est = ||v_candidate - v_query|| / ||v_query||

        This gives the relative noise level, which naturally decreases during
        Langevin refinement as v_candidate approaches v_query.
        """
        with torch.no_grad():
            dist = (v_candidate - v_query).norm(dim=-1, keepdim=True)
            query_norm = v_query.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return dist / query_norm

    def energy_and_grad(
        self, v_query: Tensor, v_candidate: Tensor, sigma: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Compute energy and gradient w.r.t. v_candidate in one pass.

        Used by Langevin dynamics to get ∇_{V_candidate} E.
        If σ is not provided, it is auto-estimated from relative distance.

        Returns:
            (energy [B], grad [B, D])
        """
        if sigma is None:
            sigma = self._estimate_sigma(v_query, v_candidate)

        v_candidate = v_candidate.detach().requires_grad_(True)
        energy = self.forward(v_query, v_candidate, sigma=sigma.detach())
        grad = torch.autograd.grad(
            energy.sum(), v_candidate, create_graph=False
        )[0]
        return energy.detach(), grad.detach()
