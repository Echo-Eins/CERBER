"""
ChainGenerator — Autoregressive Transformer Decoder in SONAR 1024d space.

Pure QA neural network. NOT a denoiser.

Architecture:
    Input:  v_query [B, D]     — question embedding (SONAR 1024d)
    Output: chain   [B, N, D]  — reasoning steps + final answer

    Each generated step is a valid SONAR vector (on manifold, decodable to text).

    Decoder Block (Pre-Norm, repeated N_layers times):
        1. Causal Self-Attention with RoPE  — chain ordering
        2. Cross-Attention to v_query        — semantic grounding (no positional enc)
        3. FFN (SiLU, D → 4D → D)

    Output: Linear(D, D) → sphere projection (normalize × target_norm)

Inference modes:
    System 1: generate 1 step  (direct answer)
    System 2: generate N steps (visible reasoning chain → final answer)

Training:
    Teacher forcing on ground-truth reasoning chains (v_steps from HotpotQA).
    Loss: cosine similarity + MSE per step.

Design rationale:
    - RoPE for self-attention: position-content binding critical for chain ordering
      (same content at different positions → different attention patterns)
    - Cross-attention to v_query WITHOUT positional encoding: query is semantic
      anchor, not a sequence element — no position bias needed
    - Causal mask: autoregressive generation, step i sees only steps 0..i
    - Sphere projection: every output is on SONAR manifold (target_norm=0.2051)
    - SiLU activation: matches angular critic, smooth gradients
    - Pre-Norm: stable training with deep networks

CompositeCritic serves as optional reranker at inference (beam search / best-of-N).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cebcm.models.chain_head import _build_rope_cache, _apply_rope


# ─── Config ────────────────────────────────────────────────────────────

@dataclass
class ChainGeneratorConfig:
    """Configuration for the ChainGenerator."""

    d_model: int = 1024
    """SONAR embedding dimension."""

    n_heads: int = 8
    """Number of attention heads."""

    n_layers: int = 6
    """Number of decoder layers."""

    dim_feedforward: int = 4096
    """FFN hidden dimension (4× d_model)."""

    max_chain_len: int = 20
    """Maximum reasoning chain length."""

    dropout: float = 0.1
    """Dropout rate."""

    target_norm: float = 0.2051
    """SONAR manifold target L2 norm for output vectors."""

    # Training
    loss_cosine_weight: float = 1.0
    """Weight for cosine similarity loss."""

    loss_mse_weight: float = 0.1
    """Weight for MSE loss (scaled by 1/D for numerical balance)."""


# ─── Cross-Attention ───────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention to v_query.

    No positional encoding — v_query is a semantic anchor, not sequenced.
    Uses SDPA for Flash Attention support.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            x:       [B, L, D] decoder sequence (chain so far)
            context: [B, 1, D] query embedding (unsqueezed for KV)

        Returns:
            [B, L, D] attended output
        """
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # q: [B, H, L, Dh], k/v: [B, H, 1, Dh]

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            scale=self.head_dim ** -0.5,
        )  # [B, H, L, Dh]

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


# ─── Causal Self-Attention with RoPE ──────────────────────────────────

class CausalRoPESelfAttention(nn.Module):
    """
    Causal multi-head self-attention with RoPE.

    Combines:
      - RoPE for position-content binding (chain ordering)
      - Causal mask (autoregressive: step i sees steps 0..i only)
      - SDPA for Flash Attention

    Reuses RoPE infrastructure from chain_head.py.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

        rope_cache = _build_rope_cache(max_seq_len, self.head_dim, torch.device("cpu"))
        self.register_buffer("_rope_cache", rope_cache, persistent=False)
        self._cached_seq_len = max_seq_len

    def _get_rope(self, seq_len: int, device: torch.device) -> Tensor:
        if seq_len <= self._cached_seq_len:
            return self._rope_cache[:seq_len].to(device)
        rope = _build_rope_cache(seq_len, self.head_dim, device)
        self._rope_cache = rope
        self._cached_seq_len = seq_len
        return rope

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, L, D] sequence

        Returns:
            [B, L, D] with causal masking applied
        """
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE on Q and K
        rope = self._get_rope(L, x.device)
        q = _apply_rope(q, rope)
        k = _apply_rope(k, rope)

        # Causal attention via SDPA is_causal flag
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=True,
            scale=self.head_dim ** -0.5,
        )  # [B, H, L, Dh]

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


# ─── Decoder Block ────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Single decoder block: causal self-attn → cross-attn → FFN.

    Pre-Norm architecture (LayerNorm before each sub-layer).
    Residual connections throughout.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()

        # Causal self-attention with RoPE
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = CausalRoPESelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Cross-attention to query
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # FFN
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Args:
            x:       [B, L, D] chain sequence
            context: [B, 1, D] query embedding

        Returns:
            [B, L, D]
        """
        # Causal self-attention
        x = x + self.self_attn(self.norm_self(x))
        # Cross-attention to query
        x = x + self.cross_attn(self.norm_cross(x), context)
        # FFN
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ─── ChainGenerator ──────────────────────────────────────────────────

class ChainGenerator(nn.Module):
    """
    Autoregressive Transformer Decoder in SONAR embedding space.

    Generates reasoning chains conditioned on v_query.
    Each output step is a valid SONAR vector (decodable to text).

    NOT a denoiser. Direct prediction of answer embeddings.

    Training: teacher forcing with ground-truth chains.
    Inference: autoregressive generation (System 1: 1 step, System 2: N steps).
    """

    def __init__(self, cfg: ChainGeneratorConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ChainGeneratorConfig()
        self.cfg = cfg

        # Learned [START] token — initiates chain generation
        self.start_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # Decoder layers
        max_seq = cfg.max_chain_len + 1  # +1 for START token
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                max_seq_len=max_seq,
            )
            for _ in range(cfg.n_layers)
        ])

        # Final norm (Pre-Norm arch needs post-decoder norm)
        self.final_norm = nn.LayerNorm(cfg.d_model)

        # Output projection: D → D (stays in SONAR space)
        self.output_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self._init_weights()

    def _init_weights(self):
        """Xavier init. Small-scale output projection for stable start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Small-scale init for output projection (not zero — sphere projection
        # needs non-zero input to produce valid gradients through F.normalize)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.01)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def _sphere_project(self, v: Tensor) -> Tensor:
        """Project vectors onto SONAR manifold sphere."""
        return F.normalize(v, dim=-1) * self.cfg.target_norm

    # ── Training forward (teacher forcing) ──────────────────────────

    def forward(
        self,
        v_query: Tensor,
        v_target_chain: Tensor,
    ) -> Tensor:
        """
        Teacher-forced forward pass.

        Args:
            v_query:        [B, D] question embedding
            v_target_chain: [B, N, D] ground-truth chain (reasoning steps + answer)

        Returns:
            v_predicted: [B, N, D] predicted chain vectors (on sphere)

        During training, the input to the decoder is:
            [START, target_step_1, target_step_2, ..., target_step_{N-1}]
        and the output predicts:
            [pred_step_1, pred_step_2, ..., pred_step_N]

        This is standard teacher forcing: shifted input → output alignment.
        """
        B, N, D = v_target_chain.shape

        # Build decoder input: [START, step_1, ..., step_{N-1}]
        start = self.start_token.expand(B, -1, -1)  # [B, 1, D]
        decoder_input = torch.cat([start, v_target_chain[:, :-1, :]], dim=1)  # [B, N, D]

        # Context for cross-attention: v_query as single KV token
        context = v_query.unsqueeze(1)  # [B, 1, D]

        # Run through decoder layers
        x = decoder_input
        for layer in self.layers:
            x = layer(x, context)

        x = self.final_norm(x)

        # Project to SONAR space and onto sphere
        v_predicted = self.output_proj(x)  # [B, N, D]
        v_predicted = self._sphere_project(v_predicted)

        return v_predicted

    # ── Autoregressive generation (inference) ───────────────────────

    @torch.no_grad()
    def generate(
        self,
        v_query: Tensor,
        num_steps: int = 1,
    ) -> Tensor:
        """
        Autoregressive chain generation.

        Args:
            v_query:   [B, D] question embedding
            num_steps: number of chain steps to generate
                       1 = System 1 (direct answer)
                       N = System 2 (reasoning chain → answer)

        Returns:
            chain: [B, num_steps, D] generated chain (all on sphere)
        """
        B, D = v_query.shape
        device = v_query.device
        context = v_query.unsqueeze(1)  # [B, 1, D]

        # Start with [START] token
        chain = self.start_token.expand(B, -1, -1)  # [B, 1, D]

        generated = []
        for step in range(num_steps):
            # Run decoder on current chain
            x = chain
            for layer in self.layers:
                x = layer(x, context)

            x = self.final_norm(x)

            # Take last position output
            last_hidden = x[:, -1:, :]  # [B, 1, D]
            next_vec = self.output_proj(last_hidden)  # [B, 1, D]
            next_vec = self._sphere_project(next_vec)

            generated.append(next_vec)

            # Append to chain for next step
            chain = torch.cat([chain, next_vec], dim=1)  # [B, step+2, D]

        return torch.cat(generated, dim=1)  # [B, num_steps, D]

    # ── Loss computation ────────────────────────────────────────────

    def compute_loss(
        self,
        v_query: Tensor,
        v_target_chain: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute training loss with teacher forcing.

        Loss = w_cos × (1 - cos_sim) + w_mse × MSE

        Args:
            v_query:        [B, D] question embedding
            v_target_chain: [B, N, D] ground-truth chain

        Returns:
            loss: scalar
            metrics: diagnostic dict
        """
        v_pred = self.forward(v_query, v_target_chain)  # [B, N, D]

        # Cosine similarity loss: 1 - cos(pred, target), averaged over steps
        cos_sim = F.cosine_similarity(v_pred, v_target_chain, dim=-1)  # [B, N]
        cos_loss = (1.0 - cos_sim).mean()

        # MSE loss (normalized by D for numerical balance)
        mse_loss = F.mse_loss(v_pred, v_target_chain) / self.cfg.d_model

        loss = self.cfg.loss_cosine_weight * cos_loss + self.cfg.loss_mse_weight * mse_loss

        with torch.no_grad():
            # Per-step cosine for diagnostics
            cos_per_step = cos_sim.mean(dim=0)  # [N]
            metrics = {
                "loss": loss.item(),
                "cos_loss": cos_loss.item(),
                "mse_loss": mse_loss.item(),
                "cos_sim_mean": cos_sim.mean().item(),
                "cos_sim_last": cos_sim[:, -1].mean().item(),  # answer quality
                "cos_sim_first": cos_sim[:, 0].mean().item(),  # first step
                "pred_norm_mean": v_pred.norm(dim=-1).mean().item(),
            }

        return loss, metrics

    # ── Properties ──────────────────────────────────────────────────

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
