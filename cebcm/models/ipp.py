"""
Initialization Point Predictor (IPP) with Conditional Flow Matching.

Generates the initial candidate vector V_init for Langevin refinement,
conditioned on the context vector V_context from the ContextEncoder.

Two implementations:
  1. FlowIPP (recommended): Conditional Flow Matching — generates samples
     from the answer distribution, avoiding mode-averaging. Uses ODE
     integration from noise → answer.
  2. MLPIPP (simple baseline): Direct MLP prediction. Fast, but averages
     modes when multiple valid answers exist.

Training (Flow Matching):
  - Sample t ~ U(0, 1)
  - Interpolate: V_t = (1-t)·V_noise + t·V_target
  - Target velocity: u_t = V_target - V_noise
  - Loss: ||v_θ(V_t, t, V_context) - u_t||²
  This is the Conditional Flow Matching (CFM) objective from Lipman et al.

Inference:
  - Start from V_0 ~ N(0, σ²I) (or smart init from context)
  - Integrate: V_{t+dt} = V_t + dt · v_θ(V_t, t, V_context)
  - V_1 = V_init (the initial proposal)

References:
  - Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
  - Latent-CFM (arXiv:2505.04486)
  - LIRF (arXiv:2509.19903)

Spec reference: §7.3
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class IPPConfig:
    """Configuration for IPP module."""
    d_model: int = 1024  # SONAR embedding dimension
    d_context: int = 1024  # Context vector dimension (from ContextEncoder)
    # Flow Matching architecture
    hidden_dims: list[int] | None = None  # MLP hidden dims for velocity net
    flow_activation: str = "silu"
    flow_norm: str = "layernorm"
    flow_dropout: float = 0.0
    flow_zero_init_last: bool = True
    flow_velocity_weight: float = 1.0
    n_integration_steps: int = 50  # ODE steps at inference
    # Time embedding
    d_time: int = 256  # Time embedding dimension
    time_embed_type: str = "sinusoidal"  # "sinusoidal" or "learned"
    # Noise schedule
    sigma_init: float = 0.05  # Noise scale for V_0 (SONAR norm ~0.2051, keep same order)
    # ODE solver
    solver: str = "euler"  # "euler" or "midpoint"
    # Optional endpoint supervision (improves sample quality, not just velocity fit)
    endpoint_loss_weight: float = 0.0
    endpoint_cos_weight: float = 0.5
    endpoint_steps: int = 20
    endpoint_target_norm: float | None = 0.2051
    # MLP baseline
    mlp_hidden_dims: list[int] | None = None
    mlp_activation: str = "silu"
    mlp_norm: str = "layernorm"
    mlp_dropout: float = 0.0
    mlp_mse_weight: float = 1.0
    mlp_cos_weight: float = 0.5
    mlp_contrastive_weight: float = 0.0
    mlp_temperature: float = 0.07

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [2048, 2048, 1024]
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [2048, 1024]


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding, same as used in diffusion models."""

    def __init__(self, d_embed: int):
        super().__init__()
        self.d_embed = d_embed

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: [B] or [B, 1] time values in [0, 1]

        Returns:
            [B, d_embed] time embeddings
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        half = self.d_embed // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, d_embed]


def _make_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "silu":
        return nn.SiLU()
    if key == "gelu":
        return nn.GELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def _make_norm(name: str, dim: int) -> nn.Module:
    key = name.lower()
    if key == "layernorm":
        return nn.LayerNorm(dim)
    if key in {"identity", "none"}:
        return nn.Identity()
    if key == "rmsnorm":
        if not hasattr(nn, "RMSNorm"):
            raise ValueError("RMSNorm is not available in this torch version.")
        return nn.RMSNorm(dim)
    raise ValueError(f"Unsupported norm: {name}")


class VelocityNet(nn.Module):
    """
    Velocity field network v_θ(V_t, t, V_context).

    Predicts the instantaneous velocity of the flow at position V_t,
    time t, conditioned on context V_context.

    Architecture:
      concat(V_t, V_context, time_embed) → MLP → velocity [D]
    """

    def __init__(self, cfg: IPPConfig):
        super().__init__()
        self.cfg = cfg

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(cfg.d_time)
        time_act = _make_activation(cfg.flow_activation)
        self.time_proj = nn.Sequential(
            nn.Linear(cfg.d_time, cfg.d_time),
            time_act,
            nn.Dropout(cfg.flow_dropout) if cfg.flow_dropout > 0 else nn.Identity(),
            nn.Linear(cfg.d_time, cfg.d_time),
        )

        # Input dim: V_t + V_context + time_embed
        input_dim = cfg.d_model + cfg.d_context + cfg.d_time

        # Build MLP
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in cfg.hidden_dims:
            norm = _make_norm(cfg.flow_norm, h_dim)
            act = _make_activation(cfg.flow_activation)
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                norm,
                act,
                nn.Dropout(cfg.flow_dropout) if cfg.flow_dropout > 0 else nn.Identity(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, cfg.d_model))

        # Zero-initialize last layer (start with zero velocity)
        if cfg.flow_zero_init_last:
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)

        self.net = nn.Sequential(*layers)

    def forward(self, v_t: Tensor, t: Tensor, v_context: Tensor) -> Tensor:
        """
        Predict velocity at (v_t, t) conditioned on context.

        Args:
            v_t: [B, D] current position in flow
            t: [B] time in [0, 1]
            v_context: [B, D_ctx] context vector

        Returns:
            velocity: [B, D] predicted velocity
        """
        t_embed = self.time_proj(self.time_embed(t))  # [B, d_time]
        x = torch.cat([v_t, v_context, t_embed], dim=-1)
        return self.net(x)


class FlowIPP(nn.Module):
    """
    IPP based on Conditional Flow Matching.

    Generates V_init by integrating a learned velocity field from
    noise to the answer distribution. Handles multimodality naturally
    (different noise samples → different answer modes).

    Training:
      loss = ||v_θ(V_t, t, V_context) - (V_target - V_noise)||²

    Inference:
      V_0 ~ N(0, σ²I)  or  smart_init(V_context)
      for t in linspace(0, 1, steps):
          V_{t+dt} = V_t + dt · v_θ(V_t, t, V_context)
      return V_1
    """

    def __init__(self, cfg: IPPConfig):
        super().__init__()
        self.cfg = cfg
        self.velocity_net = VelocityNet(cfg)

    def _integrate(
            self,
            v_start: Tensor,
            v_context: Tensor,
            n_steps: int,
    ) -> Tensor:
        """
        Integrate dV/dt = v_theta(V, t, context) from t=0 to t=1.
        """
        B = v_start.shape[0]
        device = v_start.device
        steps = max(1, int(n_steps))
        dt = 1.0 / steps
        v = v_start

        if self.cfg.solver == "midpoint":
            for i in range(steps):
                t = torch.full((B,), i * dt, device=device)
                t_mid = torch.full((B,), (i + 0.5) * dt, device=device)
                v_mid = v + 0.5 * dt * self.velocity_net(v, t, v_context)
                v = v + dt * self.velocity_net(v_mid, t_mid, v_context)
        else:  # Euler
            for i in range(steps):
                t = torch.full((B,), i * dt, device=device)
                velocity = self.velocity_net(v, t, v_context)
                v = v + dt * velocity

        return v

    def compute_loss(
            self,
            v_context: Tensor,  # [B, D_ctx] from ContextEncoder
            v_target: Tensor,  # [B, D] ground truth answer vector
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute Conditional Flow Matching loss.

        Samples random time t, interpolates between noise and target,
        and trains velocity net to predict the OT velocity.

        Args:
            v_context: [B, D_ctx] context vector
            v_target: [B, D] target answer vector

        Returns:
            loss: scalar
            metrics: dict with diagnostic values
        """
        B, D = v_target.shape
        device = v_target.device

        # Sample time: t ~ U(0, 1)
        t = torch.rand(B, device=device)

        # Sample noise: V_0 ~ N(0, σ²I)
        v_noise = torch.randn_like(v_target) * self.cfg.sigma_init

        # Interpolate: V_t = (1-t)·V_noise + t·V_target
        t_expand = t.unsqueeze(-1)  # [B, 1]
        v_t = (1 - t_expand) * v_noise + t_expand * v_target

        # Target velocity: u_t = V_target - V_noise (constant along OT path)
        u_target = v_target - v_noise

        # Predict velocity
        v_pred = self.velocity_net(v_t, t, v_context)

        # Flow matching loss on velocity
        loss_flow = F.mse_loss(v_pred, u_target)
        loss = loss_flow
        if self.cfg.flow_velocity_weight != 1.0:
            loss = float(self.cfg.flow_velocity_weight) * loss

        endpoint_mse = None
        endpoint_cos = None
        if self.cfg.endpoint_loss_weight > 0.0:
            # Explicitly supervise final integrated sample quality.
            v_end = self._integrate(
                v_start=v_noise,
                v_context=v_context,
                n_steps=self.cfg.endpoint_steps,
            )
            if self.cfg.endpoint_target_norm is not None:
                v_end = F.normalize(v_end, dim=-1) * float(self.cfg.endpoint_target_norm)
            endpoint_mse_t = F.mse_loss(v_end, v_target)
            endpoint_cos_t = (1 - F.cosine_similarity(v_end, v_target, dim=-1)).mean()
            endpoint_loss = endpoint_mse_t + float(self.cfg.endpoint_cos_weight) * endpoint_cos_t
            loss = loss + float(self.cfg.endpoint_loss_weight) * endpoint_loss
            endpoint_mse = endpoint_mse_t.item()
            endpoint_cos = 1.0 - endpoint_cos_t.item()

        # Diagnostics
        with torch.no_grad():
            flow_cos = F.cosine_similarity(v_pred, u_target, dim=-1).mean().item()
            vel_norm = v_pred.norm(dim=-1).mean().item()

        metrics = {
            "flow_loss": loss.item(),
            "flow_vel_mse": loss_flow.item(),
            "flow_cos": flow_cos,
            "vel_norm": vel_norm,
        }
        if endpoint_mse is not None and endpoint_cos is not None:
            metrics["flow_endpoint_mse"] = endpoint_mse
            metrics["flow_endpoint_cos"] = endpoint_cos
        return loss, metrics

    @torch.no_grad()
    def sample(
            self,
            v_context: Tensor,  # [B, D_ctx]
            target_norm: float | None = None,  # Project to SONAR sphere
            n_steps: int | None = None,
            v_init: Tensor | None = None,  # Optional starting point override
    ) -> Tensor:
        """
        Generate V_init by ODE integration of the learned velocity field.

        Args:
            v_context: [B, D_ctx] context vector
            target_norm: if set, project to this sphere radius after generation
            n_steps: override number of integration steps
            v_init: optional starting point (default: sample from N(0, σ²I))

        Returns:
            v_out: [B, D] generated initial proposal vector
        """
        B = v_context.shape[0]
        D = self.cfg.d_model
        device = v_context.device
        steps = n_steps or self.cfg.n_integration_steps

        # Start from noise (or provided init)
        if v_init is not None:
            v = v_init.clone()
        else:
            v = torch.randn(B, D, device=device) * self.cfg.sigma_init

        v = self._integrate(v_start=v, v_context=v_context, n_steps=steps)

        # Project to SONAR sphere if target_norm specified
        if target_norm is not None:
            v = F.normalize(v, dim=-1) * target_norm

        return v

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MLPIPP(nn.Module):
    """
    Simple MLP baseline IPP.

    Directly predicts V_init from V_context. Fast (single forward pass)
    but may mode-average when multiple valid answers exist.

    Use this for initial experiments; switch to FlowIPP for production.
    """

    def __init__(self, cfg: IPPConfig):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        prev_dim = cfg.d_context
        for h_dim in cfg.mlp_hidden_dims:
            norm = _make_norm(cfg.mlp_norm, h_dim)
            act = _make_activation(cfg.mlp_activation)
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                norm,
                act,
                nn.Dropout(cfg.mlp_dropout) if cfg.mlp_dropout > 0 else nn.Identity(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, cfg.d_model))
        self.net = nn.Sequential(*layers)

    def compute_loss(
            self,
            v_context: Tensor,
            v_target: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        MSE + cosine loss.

        Args:
            v_context: [B, D_ctx]
            v_target: [B, D]

        Returns:
            loss, metrics
        """
        v_pred = self.net(v_context)
        loss_mse = F.mse_loss(v_pred, v_target)
        loss_cos = (1 - F.cosine_similarity(v_pred, v_target, dim=-1)).mean()
        loss = float(self.cfg.mlp_mse_weight) * loss_mse + float(self.cfg.mlp_cos_weight) * loss_cos

        loss_nce = torch.tensor(0.0, device=v_pred.device)
        if self.cfg.mlp_contrastive_weight > 0.0 and v_pred.shape[0] > 1:
            z_pred = F.normalize(v_pred, dim=-1)
            z_tgt = F.normalize(v_target, dim=-1)
            logits = z_pred @ z_tgt.t()
            logits = logits / max(1e-6, float(self.cfg.mlp_temperature))
            labels = torch.arange(v_pred.shape[0], device=v_pred.device)
            loss_nce = F.cross_entropy(logits, labels)
            loss = loss + float(self.cfg.mlp_contrastive_weight) * loss_nce

        with torch.no_grad():
            cos = F.cosine_similarity(v_pred, v_target, dim=-1).mean().item()

        return loss, {
            "ipp_loss": loss.item(),
            "ipp_mse": loss_mse.item(),
            "ipp_cos": cos,
            "ipp_nce": loss_nce.item(),
        }

    @torch.no_grad()
    def sample(
            self,
            v_context: Tensor,
            target_norm: float | None = None,
            **kwargs,
    ) -> Tensor:
        v = self.net(v_context)
        if target_norm is not None:
            v = F.normalize(v, dim=-1) * target_norm
        return v

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
