"""
Geometrically specialized conditional energy critics for Stage 1.5.

This module provides:
- AngularEnergyCritic: semantic/tangential guidance on hypersphere.
- RadialEnergyCritic: norm/shell/OOD control.

Both critics expose the same API as SimpleEnergy:
    forward(v_query, v_candidate, sigma=None) -> [B]
    energy_and_grad(v_query, v_candidate, sigma=None) -> (energy, grad)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cebcm.models.activations import GroupSort, LipschitzLinearSpline
from cebcm.models.normalization import make_cayley_linear


_SIGMA_EMBED_FREQS = 4
_SIGMA_EMBED_DIM = _SIGMA_EMBED_FREQS * 2  # sin+cos


def _make_linear(
    in_dim: int,
    out_dim: int,
    norm_mode: str,
) -> nn.Module:
    if norm_mode == "orthonorm":
        return make_cayley_linear(in_dim, out_dim, bias=True)
    if norm_mode == "spectral_norm":
        linear = nn.Linear(in_dim, out_dim)
        return nn.utils.parametrizations.spectral_norm(linear, n_power_iterations=5)
    return nn.Linear(in_dim, out_dim)


def _make_activation(
    activation: str,
    dim: int,
    groupsort_size: int,
    spline_num_knots: int,
) -> nn.Module:
    if activation == "groupsort":
        return GroupSort(group_size=groupsort_size)
    if activation == "lipschitz_spline":
        return LipschitzLinearSpline(num_features=dim, num_knots=spline_num_knots)
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {activation}")


class _SigmaConditionedCritic(nn.Module):
    def __init__(
        self,
        hidden_dims: list[int],
        norm_mode: str,
        activation: str,
        groupsort_size: int = 2,
        spline_num_knots: int = 4,
        energy_output_clamp: float | None = None,
        trainable_energy_scale: bool = False,
        energy_scale_init_log: float = 0.0,
    ):
        super().__init__()
        self.norm_mode = norm_mode
        self.activation_name = activation
        self.energy_output_clamp = energy_output_clamp
        freqs = torch.arange(1, _SIGMA_EMBED_FREQS + 1, dtype=torch.float32) * math.pi
        self.register_buffer("_sigma_freqs", freqs)

        in_dim = self._feature_dim()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            if activation == "groupsort" and h % groupsort_size != 0:
                h = (h // groupsort_size) * groupsort_size
            h = max(2, int(h))
            layers.append(_make_linear(prev, h, norm_mode))
            layers.append(
                _make_activation(
                    activation=activation,
                    dim=h,
                    groupsort_size=groupsort_size,
                    spline_num_knots=spline_num_knots,
                )
            )
            prev = h
        layers.append(_make_linear(prev, 1, norm_mode))
        self.net = nn.Sequential(*layers)

        # Global energy scale: fixed by default; optionally trainable.
        init = torch.tensor(float(energy_scale_init_log))
        if trainable_energy_scale:
            self.log_energy_scale = nn.Parameter(init)
        else:
            self.register_buffer("log_energy_scale", init)

    def _feature_dim(self) -> int:
        raise NotImplementedError

    def _feature_build(self, v_query: Tensor, v_candidate: Tensor, sigma: Tensor) -> Tensor:
        raise NotImplementedError

    def _estimate_sigma(self, v_query: Tensor, v_candidate: Tensor) -> Tensor:
        with torch.no_grad():
            dist = (v_candidate - v_query).norm(dim=-1, keepdim=True)
            q_norm = v_query.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return dist / q_norm

    def _embed_sigma(self, sigma: Tensor) -> Tensor:
        log_sigma = torch.log(sigma.clamp(min=1e-8))
        args = log_sigma * self._sigma_freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        sigma: Tensor | None = None,
    ) -> Tensor:
        if sigma is None:
            sigma = self._estimate_sigma(v_query, v_candidate)
        x = self._feature_build(v_query=v_query, v_candidate=v_candidate, sigma=sigma)
        scale = torch.exp(self.log_energy_scale.clamp(min=-8.0, max=8.0))
        e = scale * self.net(x).squeeze(-1)
        if self.energy_output_clamp is not None:
            clip = float(self.energy_output_clamp)
            e = e.clamp(min=-clip, max=clip)
        return e

    def energy_and_grad(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        sigma: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if sigma is None:
            sigma = self._estimate_sigma(v_query, v_candidate)
        v_req = v_candidate.detach().requires_grad_(True)
        e = self.forward(v_query, v_req, sigma=sigma.detach())
        g = torch.autograd.grad(e.sum(), v_req, create_graph=False)[0]
        return e.detach(), g.detach()


class AngularEnergyCritic(_SigmaConditionedCritic):
    """
    Angular/tangential critic.

    Inputs:
    [q_hat, v_hat, q_hat-v_hat, q_hat*v_hat, cos(q,v), arccos(cos), sigma_embed]
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dims: list[int] | None = None,
        norm_mode: str = "none",
        activation: str = "silu",
        groupsort_size: int = 2,
        spline_num_knots: int = 4,
        energy_output_clamp: float | None = None,
        trainable_energy_scale: bool = False,
        energy_scale_init_log: float = 0.0,
    ):
        self.dim = int(dim)
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512]
        super().__init__(
            hidden_dims=hidden_dims,
            norm_mode=norm_mode,
            activation=activation,
            groupsort_size=groupsort_size,
            spline_num_knots=spline_num_knots,
            energy_output_clamp=energy_output_clamp,
            trainable_energy_scale=trainable_energy_scale,
            energy_scale_init_log=energy_scale_init_log,
        )

    def _feature_dim(self) -> int:
        # 4*dim + (cos, theta) + sigma_embed
        return self.dim * 4 + 2 + _SIGMA_EMBED_DIM

    def _feature_build(self, v_query: Tensor, v_candidate: Tensor, sigma: Tensor) -> Tensor:
        q_hat = F.normalize(v_query, dim=-1)
        v_hat = F.normalize(v_candidate, dim=-1)
        diff = q_hat - v_hat
        prod = q_hat * v_hat
        cos = (q_hat * v_hat).sum(dim=-1, keepdim=True).clamp(min=-0.999999, max=0.999999)
        theta = torch.acos(cos)
        sigma_emb = self._embed_sigma(sigma)
        return torch.cat([q_hat, v_hat, diff, prod, cos, theta, sigma_emb], dim=-1)


class RadialEnergyCritic(_SigmaConditionedCritic):
    """
    Radial/normal critic.

    Inputs are low-dimensional geometric controls focused on shell/manifold:
    [||q||, ||v||, delta_r, |delta_r|, ||v-q||, ||v-q||^2, cos(q,v), theta, sigma_embed]
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden_dims: list[int] | None = None,
        norm_mode: str = "none",
        activation: str = "silu",
        target_norm: float | None = None,
        groupsort_size: int = 2,
        spline_num_knots: int = 4,
        energy_output_clamp: float | None = None,
        trainable_energy_scale: bool = False,
        energy_scale_init_log: float = 0.0,
    ):
        self.dim = int(dim)
        self.target_norm = target_norm
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        super().__init__(
            hidden_dims=hidden_dims,
            norm_mode=norm_mode,
            activation=activation,
            groupsort_size=groupsort_size,
            spline_num_knots=spline_num_knots,
            energy_output_clamp=energy_output_clamp,
            trainable_energy_scale=trainable_energy_scale,
            energy_scale_init_log=energy_scale_init_log,
        )

    def _feature_dim(self) -> int:
        # 8 scalar geometry features + sigma embedding
        return 8 + _SIGMA_EMBED_DIM

    def _feature_build(self, v_query: Tensor, v_candidate: Tensor, sigma: Tensor) -> Tensor:
        q_norm = v_query.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_norm = v_candidate.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        r_target = q_norm if self.target_norm is None else torch.full_like(q_norm, float(self.target_norm))
        delta_r = v_norm - r_target
        pair_d = (v_candidate - v_query).norm(dim=-1, keepdim=True)
        q_hat = v_query / q_norm
        v_hat = v_candidate / v_norm
        cos = (q_hat * v_hat).sum(dim=-1, keepdim=True).clamp(min=-0.999999, max=0.999999)
        theta = torch.acos(cos)
        sigma_emb = self._embed_sigma(sigma)
        geom = torch.cat(
            [
                q_norm,
                v_norm,
                delta_r,
                delta_r.abs(),
                pair_d,
                pair_d.pow(2),
                cos,
                theta,
            ],
            dim=-1,
        )
        return torch.cat([geom, sigma_emb], dim=-1)
