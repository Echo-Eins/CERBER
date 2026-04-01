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
    n_integration_steps: int = 10  # ODE steps at inference
    # Time embedding
    d_time: int = 256  # Time embedding dimension
    time_embed_type: str = "sinusoidal"  # "sinusoidal" or "learned"
    # Noise schedule
    sigma_init: float = 0.5  # Noise scale for V_0 (starting distribution)
    # ODE solver
    solver: str = "euler"  # "euler" or "midpoint"
    # MLP baseline
    mlp_hidden_dims: list[int] | None = None

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
        self.time_proj = nn.Sequential(
            nn.Linear(cfg.d_time, cfg.d_time),
            nn.SiLU(),
            nn.Linear(cfg.d_time, cfg.d_time),
        )

        # Input dim: V_t + V_context + time_embed
        input_dim = cfg.d_model + cfg.d_context + cfg.d_time

        # Build MLP
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in cfg.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, cfg.d_model))

        # Zero-initialize last layer (start with zero velocity)
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

        # MSE loss on velocity
        loss = F.mse_loss(v_pred, u_target)

        # Diagnostics
        with torch.no_grad():
            cos = F.cosine_similarity(v_pred, u_target, dim=-1).mean().item()
            vel_norm = v_pred.norm(dim=-1).mean().item()

        return loss, {"flow_loss": loss.item(), "flow_cos": cos, "vel_norm": vel_norm}

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
        dt = 1.0 / steps

        # Start from noise (or provided init)
        if v_init is not None:
            v = v_init.clone()
        else:
            v = torch.randn(B, D, device=device) * self.cfg.sigma_init

        # Euler or midpoint integration
        if self.cfg.solver == "midpoint":
            for i in range(steps):
                t = torch.full((B,), i * dt, device=device)
                t_mid = torch.full((B,), (i + 0.5) * dt, device=device)
                # Half step
                v_mid = v + 0.5 * dt * self.velocity_net(v, t, v_context)
                # Full step using midpoint velocity
                v = v + dt * self.velocity_net(v_mid, t_mid, v_context)
        else:  # Euler
            for i in range(steps):
                t = torch.full((B,), i * dt, device=device)
                velocity = self.velocity_net(v, t, v_context)
                v = v + dt * velocity

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
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
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
        loss = loss_mse + 0.5 * loss_cos

        with torch.no_grad():
            cos = F.cosine_similarity(v_pred, v_target, dim=-1).mean().item()

        return loss, {"ipp_loss": loss.item(), "ipp_mse": loss_mse.item(), "ipp_cos": cos}

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