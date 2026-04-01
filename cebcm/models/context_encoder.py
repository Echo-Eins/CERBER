"""
Context Encoder: SSM backbone + Surprise-Driven Global Token Attention.

Processes a variable-length sequence of SONAR vectors (dialogue history)
and produces a fixed-size context vector V_context (1024d) that conditions
the IPP, Critic, and Actor.

Architecture:
  1. Type embedding (query=0, answer=1, compact=2) added to input
  2. SSM backbone (Mamba, O(N)) processes full sequence
  3. Global Token attention: high-surprise vectors get direct cross-attention
  4. Final projection → V_context [B, 1024]

The SSM handles the full context in O(N), while Global Tokens preserve
critical information that might be lost in the recurrent state.
At inference time, operates incrementally (one vector per step, O(1)).

Spec reference: §9.5, §9.6, §9.9
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cebcm.models.ssm import SSMConfig, SSMBackbone, RMSNorm


@dataclass
class ContextEncoderConfig:
    """Configuration for the Context Encoder."""
    d_model: int = 1024  # Must match SONAR embedding dim
    # SSM backbone
    ssm_d_state: int = 64  # SSM state dimension
    ssm_d_conv: int = 4  # Local convolution width
    ssm_expand: int = 2  # Inner dimension expansion
    ssm_n_layers: int = 2  # Number of SSM blocks
    ssm_dropout: float = 0.05  # Dropout between SSM layers
    # Global Token attention
    n_global_heads: int = 8  # Attention heads for global tokens
    global_attn_dropout: float = 0.1
    surprise_top_k_pct: float = 0.05  # Top 5% by surprise → global tokens
    # Type embeddings
    n_types: int = 3  # 0=query, 1=answer, 2=compact
    # Output
    output_dim: int = 1024  # Output context vector dimension
    use_alibi: bool = True  # ALiBi positional bias (preserves SONAR geometry)


class ALiBiAttention(nn.Module):
    """
    Multi-head attention with ALiBi positional encoding.

    ALiBi adds additive linear bias -m|i-j| to attention scores,
    where m is a per-head geometric sequence. This does NOT modify
    the Q/K/V vectors, preserving SONAR semantic distances.

    Reference: Press et al., "Train Short, Test Long" (ICLR 2022)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # ALiBi slopes: geometric sequence from 2^(-8/n_heads) to 2^(-8)
        slopes = self._get_alibi_slopes(n_heads)
        self.register_buffer("alibi_slopes", slopes)

    @staticmethod
    def _get_alibi_slopes(n_heads: int) -> Tensor:
        """Compute ALiBi slopes as geometric sequence."""

        def _get_slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n_heads).is_integer():
            slopes = _get_slopes_power_of_2(n_heads)
        else:
            # For non-power-of-2, interpolate
            closest_pow2 = 2 ** math.floor(math.log2(n_heads))
            slopes = _get_slopes_power_of_2(closest_pow2)
            extra = _get_slopes_power_of_2(2 * closest_pow2)
            slopes = slopes + extra[0::2][: n_heads - closest_pow2]
        return torch.tensor(slopes, dtype=torch.float32)

    def _get_alibi_bias(self, seq_len_q: int, seq_len_k: int, device: torch.device) -> Tensor:
        """Compute ALiBi bias matrix: -slope * |i - j|."""
        q_pos = torch.arange(seq_len_q, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, device=device).unsqueeze(0)
        dist = (q_pos - k_pos).abs().float()  # [Q, K]
        # [n_heads, Q, K]
        bias = -self.alibi_slopes.to(device)[:, None, None] * dist.unsqueeze(0)
        return bias

    def forward(
            self,
            query: Tensor,  # [B, Q, D]
            key: Tensor,  # [B, K, D]
            value: Tensor,  # [B, K, D]
            attn_mask: Tensor | None = None,  # [B, Q, K] or [B, 1, Q, K]
    ) -> Tensor:
        B, Q, _ = query.shape
        K = key.shape[1]

        q = self.q_proj(query).view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [B, H, L, D_head]

        # Scaled dot-product attention + ALiBi bias
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Q, K]

        # Add ALiBi bias
        alibi = self._get_alibi_bias(Q, K, query.device)  # [H, Q, K]
        attn_weights = attn_weights + alibi.unsqueeze(0)

        # Apply mask (e.g., for padding)
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, Q, K]
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, H, Q, D_head]
        out = out.transpose(1, 2).contiguous().view(B, Q, self.d_model)
        return self.out_proj(out)


class ContextEncoder(nn.Module):
    """
    Full context encoder: SSM + Global Token Attention.

    Processes dialogue history [V₁, V₂, ..., Vₙ] and produces
    a context vector V_context that conditions the rest of the pipeline.

    Two-stage processing:
      1. SSM backbone reads full sequence (O(N))
      2. Cross-attention from query to SSM output + Global Tokens (O(G²))

    At inference, SSM runs incrementally (O(1) per new vector).
    """

    def __init__(self, cfg: ContextEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Type embeddings: query=0, answer=1, compact=2
        self.type_embedding = nn.Embedding(cfg.n_types, cfg.d_model)

        # SSM backbone
        ssm_cfg = SSMConfig(
            d_model=cfg.d_model,
            d_state=cfg.ssm_d_state,
            d_conv=cfg.ssm_d_conv,
            expand=cfg.ssm_expand,
            n_layers=cfg.ssm_n_layers,
            dropout=cfg.ssm_dropout,
        )
        self.ssm = SSMBackbone(ssm_cfg)

        # Global Token cross-attention (query attends to global tokens)
        self.global_attn = ALiBiAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_global_heads,
            dropout=cfg.global_attn_dropout,
        )
        self.global_norm_q = RMSNorm(cfg.d_model)
        self.global_norm_kv = RMSNorm(cfg.d_model)

        # Fusion: combine SSM output + global attention output
        self.fusion = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.output_dim),
        )
        self.output_norm = RMSNorm(cfg.output_dim)

        # Incremental inference state
        self._ssm_states: list[tuple[Tensor, Tensor]] | None = None
        self._global_tokens: list[Tensor] = []
        self._surprise_history: list[float] = []

    def forward(
            self,
            context_vectors: Tensor,  # [B, L, D]
            type_ids: Tensor,  # [B, L] int — 0=query, 1=answer, 2=compact
            surprise_scores: Tensor | None = None,  # [B, L] float — surprise per vector
            query_vector: Tensor | None = None,  # [B, D] — current query (optional)
            lengths: Tensor | None = None,  # [B] — actual sequence lengths (for padding)
    ) -> Tensor:
        """
        Encode context sequence into a fixed-size context vector.

        Args:
            context_vectors: [B, L, D] SONAR vectors of dialogue history
            type_ids: [B, L] type of each vector (query/answer/compact)
            surprise_scores: [B, L] surprise score per vector (from SurprisePredictor)
            query_vector: [B, D] current user query (appended to context)
            lengths: [B] actual lengths for masking padded positions

        Returns:
            V_context: [B, output_dim] context vector for IPP/Critic/Actor
        """
        B, L, D = context_vectors.shape

        # Add type embeddings
        x = context_vectors + self.type_embedding(type_ids)

        # If query provided, append it to the sequence
        if query_vector is not None:
            q_type = torch.zeros(B, 1, dtype=torch.long, device=x.device)
            q_embed = query_vector.unsqueeze(1) + self.type_embedding(q_type)
            x = torch.cat([x, q_embed], dim=1)  # [B, L+1, D]
            L = L + 1

            if surprise_scores is not None:
                # Query has zero surprise (it's the input, not predicted)
                surprise_scores = torch.cat([
                    surprise_scores,
                    torch.zeros(B, 1, device=surprise_scores.device),
                ], dim=1)

            if lengths is not None:
                lengths = lengths + 1

        # Build padding mask if lengths provided
        pad_mask = None
        if lengths is not None:
            positions = torch.arange(L, device=x.device).unsqueeze(0)
            pad_mask = positions < lengths.unsqueeze(1)  # [B, L] bool

        # Stage 1: SSM processes full sequence
        ssm_out = self.ssm(x)  # [B, L, D]

        # Get the SSM summary: last valid position per batch
        if lengths is not None:
            # Gather last valid position
            last_idx = (lengths - 1).clamp(min=0).long()
            ssm_summary = ssm_out[
                torch.arange(B, device=ssm_out.device), last_idx
            ]  # [B, D]
        else:
            ssm_summary = ssm_out[:, -1, :]  # [B, D]

        # Stage 2: Global Token cross-attention
        if surprise_scores is not None:
            global_out = self._attend_to_globals(
                query=ssm_summary,
                ssm_hidden=ssm_out,
                surprise_scores=surprise_scores,
                pad_mask=pad_mask,
            )
        else:
            # No surprise info → just use SSM summary doubled
            global_out = ssm_summary

        # Fuse SSM summary + global attention
        fused = torch.cat([ssm_summary, global_out], dim=-1)  # [B, 2*D]
        v_context = self.fusion(fused)
        v_context = self.output_norm(v_context)

        return v_context  # [B, output_dim]

    def _attend_to_globals(
            self,
            query: Tensor,  # [B, D]
            ssm_hidden: Tensor,  # [B, L, D]
            surprise_scores: Tensor,  # [B, L]
            pad_mask: Tensor | None,  # [B, L]
    ) -> Tensor:
        """Cross-attention from query to high-surprise (global) tokens."""
        B, L, D = ssm_hidden.shape

        if pad_mask is None:
            pad_mask = torch.ones(B, L, dtype=torch.bool, device=ssm_hidden.device)

        pct = float(self.cfg.surprise_top_k_pct)
        pct = min(max(pct, 1e-6), 1.0)
        global_tokens_per_batch: list[Tensor] = []
        max_k = 1

        # Per-sample selection avoids pulling padded positions into top-k.
        for b in range(B):
            valid_idx = pad_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                global_tokens_per_batch.append(ssm_hidden[b:b + 1, :1, :].squeeze(0))
                max_k = max(max_k, 1)
                continue

            scores = surprise_scores[b, valid_idx]
            k_b = max(1, int(valid_idx.numel() * pct))
            k_b = min(k_b, int(valid_idx.numel()))

            top_local = scores.topk(k_b, dim=0).indices
            selected_idx = valid_idx[top_local]
            tokens_b = ssm_hidden[b, selected_idx, :]  # [k_b, D]
            global_tokens_per_batch.append(tokens_b)
            max_k = max(max_k, k_b)

        global_tokens = torch.zeros(B, max_k, D, device=ssm_hidden.device, dtype=ssm_hidden.dtype)
        global_mask = torch.zeros(B, max_k, dtype=torch.bool, device=ssm_hidden.device)
        for b, tokens_b in enumerate(global_tokens_per_batch):
            k_b = tokens_b.shape[0]
            global_tokens[b, :k_b] = tokens_b
            global_mask[b, :k_b] = True

        # Cross-attention: query → global tokens
        q = self.global_norm_q(query.unsqueeze(1))  # [B, 1, D]
        kv = self.global_norm_kv(global_tokens)  # [B, k, D]
        attn_mask = global_mask.unsqueeze(1)  # [B, 1, k]

        attn_out = self.global_attn(query=q, key=kv, value=kv, attn_mask=attn_mask)  # [B, 1, D]
        return attn_out.squeeze(1)  # [B, D]

    def reset_state(self) -> None:
        """Reset incremental inference state."""
        self._ssm_states = None
        self._global_tokens = []
        self._surprise_history = []

    def _is_global_surprise(self, surprise: float) -> bool:
        """
        Running-percentile global-token flag for incremental inference.

        Uses `surprise_top_k_pct` as a tail fraction (e.g. 0.05 → top-5%).
        """
        pct = float(self.cfg.surprise_top_k_pct)
        if pct <= 0.0:
            return False
        if pct >= 1.0:
            return True

        self._surprise_history.append(float(surprise))
        if len(self._surprise_history) > 4096:
            self._surprise_history = self._surprise_history[-4096:]

        hist = torch.tensor(self._surprise_history, dtype=torch.float32)
        q = max(0.0, min(1.0, 1.0 - pct))
        threshold = torch.quantile(hist, q).item()
        return float(surprise) >= threshold

    def step(
            self,
            v_new: Tensor,  # [B, D] new vector
            type_id: int,  # type of the new vector
            surprise: float = 0.0,  # surprise score for this vector
    ) -> Tensor:
        """
        Incremental inference: process one new vector, return updated context.

        Args:
            v_new: [B, D] new SONAR vector
            type_id: 0=query, 1=answer, 2=compact
            surprise: surprise score from SurprisePredictor

        Returns:
            V_context: [B, output_dim] updated context vector
        """
        B, D = v_new.shape

        # Add type embedding
        type_ids = torch.full((B,), type_id, dtype=torch.long, device=v_new.device)
        x = v_new + self.type_embedding(type_ids)

        # SSM step
        ssm_out, self._ssm_states = self.ssm.step(x, self._ssm_states)

        # Track global tokens using running percentile (top-k% tail), not fixed scalar.
        if self._is_global_surprise(surprise):
            self._global_tokens.append(ssm_out.detach())

        # Cross-attend to accumulated global tokens
        if len(self._global_tokens) > 0:
            globals_t = torch.stack(self._global_tokens, dim=1)  # [B, G, D]
            q = self.global_norm_q(ssm_out.unsqueeze(1))
            kv = self.global_norm_kv(globals_t)
            global_out = self.global_attn(query=q, key=kv, value=kv).squeeze(1)
        else:
            global_out = ssm_out

        # Fuse
        fused = torch.cat([ssm_out, global_out], dim=-1)
        v_context = self.fusion(fused)
        v_context = self.output_norm(v_context)
        return v_context

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
