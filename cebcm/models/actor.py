"""
Latent denoising actor for Stage 1 Actor+Critic training.

The actor predicts a refinement update in SONAR space:
    v_next = v_current + step_size * actor(v_query, v_current, sigma)

Unlike energy-gradient objectives, actor training is first-order and avoids
second-order autograd overhead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cebcm.models.activations import GroupSort, LipschitzLinearSpline
from cebcm.models.normalization import OrthoLinear, make_cayley_linear


_SIGMA_EMBED_FREQS = 4
_SIGMA_EMBED_DIM = _SIGMA_EMBED_FREQS * 2


class LatentDenoiseActor(nn.Module):
    """
    Sigma-conditioned latent denoising actor.

    Input features:
        [v_query, v_current, v_query - v_current, v_query * v_current, sigma_embed]
    Output:
        delta update in latent space [B, D]
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dims: list[int] | None = None,
        norm_mode: str = "spectral_norm",
        activation: str = "silu",
        ortho_n_iters: int = 4,
        groupsort_size: int = 2,
        spline_num_knots: int = 4,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512]

        self.dim = dim
        self.norm_mode = norm_mode
        self.activation_name = activation

        freqs = torch.arange(1, _SIGMA_EMBED_FREQS + 1, dtype=torch.float32) * math.pi
        self.register_buffer("_sigma_freqs", freqs)

        input_dim = dim * 4 + _SIGMA_EMBED_DIM
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            if activation == "groupsort" and h_dim % groupsort_size != 0:
                h_dim = (h_dim // groupsort_size) * groupsort_size
            layers.append(self._make_linear(prev_dim, h_dim, norm_mode, ortho_n_iters))
            layers.append(
                self._make_activation(
                    activation,
                    dim=h_dim,
                    groupsort_size=groupsort_size,
                    spline_num_knots=spline_num_knots,
                )
            )
            prev_dim = h_dim
        layers.append(self._make_linear(prev_dim, dim, norm_mode, ortho_n_iters))
        self.net = nn.Sequential(*layers)

        # Learnable global step scaling for update magnitude calibration.
        self.log_step_scale = nn.Parameter(torch.tensor(-1.0))

    def _embed_sigma(self, sigma: Tensor) -> Tensor:
        log_sigma = torch.log(sigma.clamp(min=1e-8))
        args = log_sigma * self._sigma_freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def _estimate_sigma(self, v_query: Tensor, v_current: Tensor) -> Tensor:
        with torch.no_grad():
            dist = (v_current - v_query).norm(dim=-1, keepdim=True)
            query_norm = v_query.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return dist / query_norm

    @staticmethod
    def _make_linear(
        in_dim: int,
        out_dim: int,
        norm_mode: str,
        ortho_n_iters: int = 0,
    ) -> nn.Module:
        if norm_mode == "orthonorm":
            return make_cayley_linear(in_dim, out_dim, bias=True)
        if norm_mode == "spectral_norm":
            linear = nn.Linear(in_dim, out_dim)
            return nn.utils.parametrizations.spectral_norm(linear, n_power_iterations=5)
        return nn.Linear(in_dim, out_dim)

    @staticmethod
    def _make_activation(
        activation: str,
        dim: int,
        groupsort_size: int,
        spline_num_knots: int,
    ) -> nn.Module:
        if activation == "silu":
            return nn.SiLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "relu":
            return nn.ReLU()
        if activation == "groupsort":
            return GroupSort(group_size=groupsort_size)
        if activation == "lipschitz_spline":
            return LipschitzLinearSpline(num_features=dim, num_knots=spline_num_knots)
        raise ValueError(f"Unknown actor activation: {activation}")

    def forward(
        self,
        v_query: Tensor,
        v_current: Tensor,
        sigma: Tensor | None = None,
    ) -> Tensor:
        if sigma is None:
            sigma = self._estimate_sigma(v_query, v_current)

        diff = v_query - v_current
        prod = v_query * v_current
        sigma_emb = self._embed_sigma(sigma)
        x = torch.cat([v_query, v_current, diff, prod, sigma_emb], dim=-1)
        step_scale = torch.exp(self.log_step_scale.clamp(min=-6.0, max=3.0))
        return step_scale * self.net(x)

    def predict_step(
        self,
        v_query: Tensor,
        v_current: Tensor,
        sigma: Tensor | None = None,
        step_size: float = 1.0,
        target_norm: float | None = None,
        tangent_projection: bool = True,
    ) -> tuple[Tensor, Tensor]:
        delta = self.forward(v_query, v_current, sigma=sigma)
        if tangent_projection and target_norm is not None:
            v_hat = F.normalize(v_current, dim=-1)
            delta = delta - (delta * v_hat).sum(dim=-1, keepdim=True) * v_hat
        v_next = v_current + step_size * delta
        if target_norm is not None:
            v_next = F.normalize(v_next, dim=-1) * target_norm
        return v_next, delta
