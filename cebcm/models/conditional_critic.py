"""
Conditional Energy Critic — context-aware energy function for QA.

Unlike SimpleEnergy (self-denoise: E(v, v+noise) → drives candidate to query),
this critic learns CONDITIONAL energy:

    E(v_query, v_candidate, v_context) → low iff v_candidate is a valid answer

The energy minimum sits at the CORRECT ANSWER, not at the query itself.
This requires training on (question, answer, distractors) triplets with context.

Architecture: MLP over concatenated features with optional chain context.
Input:  [v_query; v_candidate; v_query-v_candidate; v_query*v_candidate; v_context; σ_embed]
Output: scalar energy E ∈ ℝ (lower = better answer)

Training losses:
    1. Focal-InfoNCE contrastive: E(q, answer, ctx) < E(q, distractor, ctx)
    2. Cosine reconstruction: cos(v_langevin_output, v_target) → 1.0

Spec reference: Stage 3 audit conclusions → Option A: Conditional Critic
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Sinusoidal σ embedding (same as existing critics for compatibility)
_SIGMA_EMBED_FREQS = 4
_SIGMA_EMBED_DIM = _SIGMA_EMBED_FREQS * 2  # sin + cos = 8 dims


@dataclass
class ConditionalCriticConfig:
    """Configuration for the Conditional Energy Critic."""

    dim: int = 1024
    """Embedding dimension (SONAR = 1024)."""

    hidden_dims: list[int] = field(default_factory=lambda: [2048, 1024, 512])
    """Hidden layer sizes."""

    activation: str = "silu"
    """Activation function: silu, relu, gelu."""

    dropout: float = 0.1
    """Dropout rate between hidden layers."""

    energy_output_clamp: float | None = 100.0
    """Optional symmetric clamp for output energy. None = no clamp."""

    context_mode: str = "concat"
    """How to incorporate context: 'concat' (append v_context to features)
    or 'film' (FiLM-style modulation: scale + shift hidden activations)."""

    use_layer_norm: bool = True
    """Apply LayerNorm after each hidden layer (stabilizes training)."""

    # Contrastive loss config
    temperature: float = 0.07
    """InfoNCE temperature τ."""

    focal_gamma: float = 2.0
    """Focal-InfoNCE γ (0 = standard InfoNCE)."""

    num_negatives: int = 7
    """Number of negatives per positive in contrastive loss."""


class ConditionalCritic(nn.Module):
    """
    Context-aware energy critic for QA.

    E(v_query, v_candidate, v_context, σ) → scalar energy.

    Lower energy = v_candidate is a better answer to v_query given context.
    """

    def __init__(self, cfg: ConditionalCriticConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ConditionalCriticConfig()
        self.cfg = cfg

        # Feature dimension depends on context mode
        if cfg.context_mode == "concat":
            # [q; c; q-c; q*c; ctx; σ_embed] = 5*dim + σ_embed_dim
            input_dim = cfg.dim * 5 + _SIGMA_EMBED_DIM
        else:
            # FiLM: context modulates hidden layers, not concatenated to input
            input_dim = cfg.dim * 4 + _SIGMA_EMBED_DIM

        # Sinusoidal frequencies for σ embedding
        freqs = torch.arange(1, _SIGMA_EMBED_FREQS + 1, dtype=torch.float32) * math.pi
        self.register_buffer("_sigma_freqs", freqs)

        # Build MLP
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in cfg.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if cfg.use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(self._make_activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev_dim = h_dim

        # Output head: scalar energy
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        # FiLM conditioning layers (if enabled)
        if cfg.context_mode == "film":
            self.film_layers = nn.ModuleList()
            for h_dim in cfg.hidden_dims:
                # Context → (scale, shift) for each hidden dim
                self.film_layers.append(nn.Linear(cfg.dim, h_dim * 2))

        # Zero-init output layer for stable training start (E ≈ 0 initially)
        with torch.no_grad():
            out_layer = self.net[-1]
            if hasattr(out_layer, "weight"):
                nn.init.zeros_(out_layer.weight)
            if hasattr(out_layer, "bias") and out_layer.bias is not None:
                nn.init.zeros_(out_layer.bias)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        if name == "silu":
            return nn.SiLU()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unknown activation: {name}")

    def _embed_sigma(self, sigma: Tensor) -> Tensor:
        """Sinusoidal embedding of noise level σ."""
        log_sigma = torch.log(sigma.clamp(min=1e-8))
        args = log_sigma * self._sigma_freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def _estimate_sigma(self, v_query: Tensor, v_candidate: Tensor) -> Tensor:
        """Estimate relative noise level from query-candidate distance."""
        with torch.no_grad():
            dist = (v_candidate - v_query).norm(dim=-1, keepdim=True)
            q_norm = v_query.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return dist / q_norm

    def forward(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        v_context: Tensor | None = None,
        sigma: Tensor | None = None,
    ) -> Tensor:
        """
        Compute conditional energy for a QA pair.

        Args:
            v_query:     [B, D] question embedding
            v_candidate: [B, D] candidate answer embedding
            v_context:   [B, D] aggregated context embedding (from reasoning chain).
                         If None, uses v_query as context (no external context).
            sigma:       [B, 1] noise level. Auto-estimated if None.

        Returns:
            [B] energy scalars. Lower = better answer.
        """
        if v_context is None:
            v_context = v_query  # Self-context fallback

        if sigma is None:
            sigma = self._estimate_sigma(v_query, v_candidate)

        # Ensure sigma is [B, 1]
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)

        diff = v_query - v_candidate
        prod = v_query * v_candidate
        sigma_emb = self._embed_sigma(sigma)

        if self.cfg.context_mode == "concat":
            x = torch.cat([v_query, v_candidate, diff, prod, v_context, sigma_emb], dim=-1)
            raw = self.net(x).squeeze(-1)
        else:
            # FiLM mode: run MLP with context modulation
            x = torch.cat([v_query, v_candidate, diff, prod, sigma_emb], dim=-1)
            raw = self._forward_film(x, v_context)

        if self.cfg.energy_output_clamp is not None:
            clip = float(self.cfg.energy_output_clamp)
            raw = raw.clamp(min=-clip, max=clip)

        return raw

    def _forward_film(self, x: Tensor, v_context: Tensor) -> Tensor:
        """Forward pass with FiLM conditioning from context.

        FiLM modulation is applied only to HIDDEN Linear layers,
        never to the output layer (last element of self.net).
        """
        film_idx = 0
        last_layer_idx = len(self.net) - 1
        for i, layer in enumerate(self.net):
            x = layer(x)
            # Apply FiLM only to hidden Linear layers, NOT the output layer
            if (
                isinstance(layer, nn.Linear)
                and i < last_layer_idx
                and film_idx < len(self.film_layers)
            ):
                film_params = self.film_layers[film_idx](v_context)
                h_dim = x.shape[-1]
                if film_params.shape[-1] == h_dim * 2:
                    scale, shift = film_params.chunk(2, dim=-1)
                    x = x * (1 + scale) + shift  # FiLM modulation
                    film_idx += 1
        return x.squeeze(-1)

    def energy_and_grad(
        self,
        v_query: Tensor,
        v_candidate: Tensor,
        v_context: Tensor | None = None,
        sigma: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute energy and gradient w.r.t. v_candidate.

        Used by Langevin dynamics: ∇_{v_candidate} E.

        Returns:
            (energy [B], grad [B, D])
        """
        if sigma is None:
            sigma = self._estimate_sigma(v_query, v_candidate)

        v_req = v_candidate.detach().requires_grad_(True)
        energy = self.forward(v_query, v_req, v_context=v_context, sigma=sigma.detach())
        grad = torch.autograd.grad(energy.sum(), v_req, create_graph=False)[0]
        return energy.detach(), grad.detach()

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------

    def compute_contrastive_loss(
        self,
        v_query: Tensor,
        v_positive: Tensor,
        v_negatives: Tensor,
        v_context: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Focal-InfoNCE contrastive loss.

        E(q, answer, ctx) should be lower than E(q, distractor, ctx).

        Args:
            v_query:     [B, D] questions
            v_positive:  [B, D] correct answers
            v_negatives: [B, N, D] distractors (N = num_negatives)
            v_context:   [B, D] context embeddings (optional)

        Returns:
            (loss, metrics_dict)
        """
        B, N, D = v_negatives.shape
        tau = self.cfg.temperature
        gamma = self.cfg.focal_gamma

        # Compute energies (lower = better)
        E_pos = self.forward(v_query, v_positive, v_context=v_context)  # [B]

        # Expand query and context for negatives
        v_q_exp = v_query.unsqueeze(1).expand_as(v_negatives).reshape(B * N, D)
        v_neg_flat = v_negatives.reshape(B * N, D)
        v_ctx_exp = None
        if v_context is not None:
            v_ctx_exp = v_context.unsqueeze(1).expand(B, N, D).reshape(B * N, D)

        E_neg_flat = self.forward(v_q_exp, v_neg_flat, v_context=v_ctx_exp)  # [B*N]
        E_neg = E_neg_flat.reshape(B, N)  # [B, N]

        # InfoNCE with energy (lower energy = higher logit)
        # logit_pos = -E_pos/τ, logit_neg = -E_neg/τ
        logit_pos = -E_pos / tau  # [B]
        logits_neg = -E_neg / tau  # [B, N]

        # All logits: [B, 1+N] where first is positive
        all_logits = torch.cat([logit_pos.unsqueeze(1), logits_neg], dim=1)  # [B, 1+N]

        if gamma > 0:
            # Focal-InfoNCE: standard focal weighting (1-p)^γ
            # Easy examples (high p_pos) → (1-p)^γ small → downweighted
            # Hard examples (low p_pos)  → (1-p)^γ large → upweighted
            with torch.no_grad():
                probs = F.softmax(all_logits, dim=1)  # [B, 1+N]
                focal_weights = (1.0 - probs).pow(gamma)
                # Normalize to preserve loss scale
                focal_weights = focal_weights / focal_weights.sum(dim=1, keepdim=True) * (1 + N)

            # Weighted log-softmax
            log_probs = F.log_softmax(all_logits, dim=1)
            loss = -(focal_weights[:, 0] * log_probs[:, 0]).mean()
        else:
            # Standard InfoNCE
            loss = F.cross_entropy(all_logits, torch.zeros(B, dtype=torch.long, device=v_query.device))

        # Metrics
        with torch.no_grad():
            rank_acc = (E_pos.unsqueeze(1) < E_neg).float().mean().item()
            energy_gap = (E_neg.mean(dim=1) - E_pos).mean().item()

        metrics = {
            "contrastive_loss": loss.item(),
            "rank_acc": rank_acc,
            "E_pos_mean": E_pos.mean().item(),
            "E_neg_mean": E_neg.mean().item(),
            "energy_gap": energy_gap,
        }
        return loss, metrics

    def compute_cosine_loss(
        self,
        v_predicted: Tensor,
        v_target: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Cosine reconstruction loss: penalize predicted vectors far from target.

        Args:
            v_predicted: [B, D] Langevin output vectors
            v_target:    [B, D] ground truth answer embeddings

        Returns:
            (loss, metrics_dict)
        """
        cos_sim = F.cosine_similarity(v_predicted, v_target, dim=-1)  # [B]
        loss = (1.0 - cos_sim).mean()

        metrics = {
            "cosine_loss": loss.item(),
            "cos_sim_mean": cos_sim.mean().item(),
            "cos_sim_min": cos_sim.min().item(),
            "cos_sim_max": cos_sim.max().item(),
        }
        return loss, metrics
