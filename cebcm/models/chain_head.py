"""
EBT Chain Head: Transformer-based reasoning chain evaluator.

Evaluates quality of reasoning chains V₁ → V₂ → ... → Vₙ using
self-attention to detect global inconsistencies (e.g., step 5
contradicts step 2).

Architecture:
  [CLS] + [V₁, V₂, ..., Vₙ] + RoPE positional encoding
    → TransformerEncoder (2 layers, 8 heads, 2048 FFN, Pre-Norm)
    → CLS pooling
    → Energy head: Linear(1024, 512) + GELU + Linear(512, 1) → scalar E

Why RoPE (not ALiBi) for Chain Head (spec Appendix C, line 2232):
  Chain Head evaluates ORDER of reasoning steps (5-20 elements).
  Order detection requires position-content binding: the model must
  distinguish "concept A at position 3" from "concept A at position 4".

  - RoPE rotates Q/K vectors by position → same content at different
    positions produces different attention patterns → enables swap detection
  - ALiBi adds content-agnostic bias -m|i-j| → biases attention distance
    but does NOT bind position to content → cannot detect adjacent swaps

  ALiBi is reserved for Context Aggregation (§9.9) where preserving
  SONAR geometry across 50K+ vectors matters more than order.

  For short chains (5-20), RoPE distortion of SONAR geometry is minimal,
  and the position-content binding is critical for the ordering task.

Uses PyTorch SDPA (scaled_dot_product_attention) for automatic Flash
Attention support on compatible hardware (spec Appendix C, line 2231).

Energy semantics: lower E = more coherent chain.

Training: Focal-InfoNCE over positive vs negative chains.
Negatives: shuffled order, truncated, corrupted, wrong conclusion.

At inference: called every 5-10 Langevin steps during Deep Thinking
(System 2) mode to validate the reasoning chain being constructed.

Spec reference: §5.2 Mode B, §8.2 (Deep Thinking), Appendix C
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ChainHeadConfig:
    """Configuration for the Chain Head."""
    d_model: int = 1024        # SONAR embedding dim
    n_heads: int = 8           # Multi-head attention heads
    n_layers: int = 2          # Transformer encoder layers
    dim_feedforward: int = 2048  # FFN hidden dim (2× d_model)
    max_chain_len: int = 20    # Max reasoning chain length
    dropout: float = 0.1       # Attention + FFN dropout
    activation: str = "gelu"   # FFN activation
    # Energy head
    energy_hidden: int = 512   # Hidden dim of energy projection
    # Training
    temperature: float = 0.07  # InfoNCE temperature
    focal_gamma: float = 2.0   # Focal-InfoNCE exponent (0 = standard)
    # Regularization (spec §5.4: smooth landscape for Langevin)
    lambda_grad: float = 0.05  # Gradient penalty weight
    lambda_energy_norm: float = 0.01  # Energy magnitude regularization
    energy_norm_margin: float = 5.0   # Max allowed |E| before penalty


# ─── RoPE for Chain Head Self-Attention ──────────────────────────────
# RoPE (Rotary Position Embedding, Su et al. 2021) rotates Q/K by
# position-dependent angles, creating position-content binding.
# Critical for chain ordering: same content at pos 3 vs pos 4 produces
# different Q·K products → model can detect adjacent swaps.
#
# For short chains (5-20), RoPE distortion of SONAR geometry is minimal.
# ALiBi is reserved for Context Aggregation (long sequences, 50K+).


def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> Tensor:
    """
    Build RoPE frequency cache.

    Returns: [seq_len, head_dim] complex rotation factors
    """
    theta = 10000.0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, freqs)  # [L, head_dim//2]
    return torch.polar(torch.ones_like(angles), angles)  # [L, head_dim//2] complex


def _apply_rope(x: Tensor, rope_cache: Tensor) -> Tensor:
    """
    Apply RoPE to input tensor.

    Args:
        x: [B, H, L, D_head] real tensor
        rope_cache: [L, D_head//2] complex rotation factors
    Returns:
        [B, H, L, D_head] rotated tensor
    """
    # View as complex pairs: [B, H, L, D_head//2] complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Rotate: broadcast rope_cache [L, D_head//2] over B, H
    rope = rope_cache[:x.shape[2]]  # trim to actual seq_len
    rotated = x_complex * rope.unsqueeze(0).unsqueeze(0)  # [B, H, L, D_head//2]
    # Back to real: [B, H, L, D_head]
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)


class RoPESelfAttention(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embedding (RoPE).

    RoPE rotates Q/K vectors by position-dependent angles, creating
    position-content binding that is critical for chain ordering tasks.

    Uses PyTorch SDPA for automatic Flash Attention on compatible hardware.

    Properties:
      - Q/K rotated by position → enables adjacent swap detection
      - V vectors unchanged → output preserves content information
      - No learnable position parameters — deterministic rotations
      - Compatible with Flash Attention via torch SDPA
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

        # Pre-compute RoPE cache
        rope_cache = _build_rope_cache(max_seq_len, self.head_dim, torch.device("cpu"))
        self.register_buffer("_rope_cache", rope_cache, persistent=False)
        self._cached_seq_len = max_seq_len

    def _get_rope(self, seq_len: int, device: torch.device) -> Tensor:
        """Get RoPE cache, rebuilding if sequence exceeds cached length."""
        if seq_len <= self._cached_seq_len:
            return self._rope_cache[:seq_len].to(device)
        rope = _build_rope_cache(seq_len, self.head_dim, device)
        self._rope_cache = rope
        self._cached_seq_len = seq_len
        return rope

    def forward(
        self,
        x: Tensor,                         # [B, L, D]
        attn_mask: Tensor | None = None,   # [B, L] bool — True = valid
    ) -> Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [B, H, L, D_head]

        # Apply RoPE to Q and K (NOT V — preserves content)
        rope = self._get_rope(L, x.device)
        q = _apply_rope(q, rope)
        k = _apply_rope(k, rope)

        # Build SDPA-compatible attention mask (additive: 0 for valid, -inf for masked)
        sdpa_mask = None
        if attn_mask is not None:
            # [B, L] bool → [B, 1, L, L] key mask (broadcast over heads)
            key_mask = attn_mask[:, None, None, :]  # [B, 1, 1, L]
            sdpa_mask = torch.where(key_mask, 0.0, float("-inf"))
            sdpa_mask = sdpa_mask.expand(B, 1, L, L)

        # Use PyTorch SDPA — auto-selects Flash Attention when available
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            scale=self.head_dim ** -0.5,
        )  # [B, H, L, D_head]

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class ChainTransformerLayer(nn.Module):
    """
    Single Transformer encoder layer with RoPE attention + FFN.

    Pre-norm architecture (LayerNorm before attention/FFN) for stable
    training with contrastive losses. Residual connections throughout.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        # Pre-norm self-attention with residual
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class EBTChainHead(nn.Module):
    """
    Energy-Based Chain Head: evaluates reasoning chain coherence.

    Input:  [CLS, V₁, V₂, ..., Vₙ]  (learnable CLS prepended)
    Output: scalar energy E per chain (lower = more coherent)

    Architecture:
      - Learnable [CLS] token (1024d)
      - RoPE Transformer encoder (2 layers, 8 heads, Pre-Norm, SDPA)
      - CLS vector extraction after encoding
      - Energy head: Linear(1024, 512) + GELU + Linear(512, 1)

    The [CLS] token acts as a global aggregator — self-attention allows it
    to attend to all chain positions simultaneously, detecting contradictions
    between arbitrary steps. RoPE provides position-content binding critical
    for detecting ordering violations (adj_swap, truncation).

    ~17M parameters at default settings.
    """

    def __init__(self, cfg: ChainHeadConfig):
        super().__init__()
        self.cfg = cfg

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # Transformer encoder layers with ALiBi
        max_seq = cfg.max_chain_len + 1  # +1 for CLS
        self.layers = nn.ModuleList([
            ChainTransformerLayer(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation=cfg.activation,
                max_seq_len=max_seq,
            )
            for _ in range(cfg.n_layers)
        ])

        # Final LayerNorm (pre-norm arch needs post-encoder norm)
        self.final_norm = nn.LayerNorm(cfg.d_model)

        # Energy head: CLS vector → scalar energy
        self.energy_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.energy_hidden),
            nn.GELU(),
            nn.Linear(cfg.energy_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights following SOTA practices.

        Critical: zero-init output layer of energy head.
        From lessons.md: "Always zero-init output layer of energy networks —
        SOTA practice from score matching literature."
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-init final energy projection for stable early training
        # This ensures E ≈ 0 initially, preventing gradient explosion
        final_linear = self.energy_head[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        chain: Tensor,                        # [B, L, D] chain vectors
        chain_lengths: Tensor | None = None,  # [B] actual chain lengths (before padding)
    ) -> Tensor:
        """
        Compute energy for reasoning chains.

        Args:
            chain: [B, L, D] SONAR vectors forming the chain
            chain_lengths: [B] actual lengths (for padded batches)

        Returns:
            energy: [B] scalar energy per chain (lower = more coherent)
        """
        B, L, D = chain.shape

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, chain], dim=1)  # [B, L+1, D]

        # Build attention mask (CLS always valid)
        attn_mask = None
        if chain_lengths is not None:
            total_len = L + 1
            positions = torch.arange(total_len, device=chain.device).unsqueeze(0)
            valid_len = chain_lengths.unsqueeze(1) + 1  # +1 for CLS
            attn_mask = positions < valid_len  # [B, L+1] bool

        # Transformer encoder with ALiBi
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.final_norm(x)

        # Extract CLS representation (position 0)
        cls_out = x[:, 0, :]  # [B, D]

        # Energy head
        energy = self.energy_head(cls_out).squeeze(-1)  # [B]
        return energy

    def energy_and_grad(
        self,
        chain: Tensor,
        chain_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute energy and gradient w.r.t. chain vectors.

        Used during Langevin dynamics (System 2) to refine chain quality.

        Returns:
            (energy [B], grad [B, L, D])
        """
        chain = chain.detach().requires_grad_(True)
        energy = self.forward(chain, chain_lengths=chain_lengths)
        grad = torch.autograd.grad(
            energy.sum(), chain, create_graph=False
        )[0]
        return energy.detach(), grad.detach()

    # ─── Loss functions ───────────────────────────────────────────────

    def compute_infonce_loss(
        self,
        positive_chains: Tensor,       # [B, L, D]
        negative_chains: Tensor,       # [B, N, L, D]
        pos_lengths: Tensor | None = None,   # [B]
        neg_lengths: Tensor | None = None,   # [B, N]
        neg_types: list[list[str]] | None = None,  # [B][N] per-neg type labels
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Focal-InfoNCE loss over positive vs negative chains.

        E_pos should be LOWER than E_neg (positive chains are more coherent).

        Loss = -log( exp(-E_pos / τ) / (exp(-E_pos / τ) + Σ wᵢ exp(-E_negᵢ / τ)) )

        where wᵢ = pᵢ^γ (focal weighting on hard negatives).

        Args:
            positive_chains: [B, L, D] positive reasoning chains
            negative_chains: [B, N, L, D] negative chains (N negatives per sample)
            pos_lengths: [B] lengths of positive chains
            neg_lengths: [B, N] lengths of negative chains

        Returns:
            loss: scalar
            metrics: diagnostic dict
        """
        B, N, L, D = negative_chains.shape
        tau = self.cfg.temperature
        gamma = self.cfg.focal_gamma

        # Positive energies
        E_pos = self.forward(positive_chains, chain_lengths=pos_lengths)  # [B]

        # Negative energies: reshape to [B*N, L, D], compute, reshape back
        neg_flat = negative_chains.reshape(B * N, L, D)
        neg_len_flat = None
        if neg_lengths is not None:
            neg_len_flat = neg_lengths.reshape(B * N)
        E_neg = self.forward(neg_flat, chain_lengths=neg_len_flat).reshape(B, N)  # [B, N]

        # Logits: [-E_pos, -E_neg_1, ..., -E_neg_N] / tau
        pos_logits = (-E_pos / tau).unsqueeze(1)  # [B, 1]
        neg_logits = -E_neg / tau                  # [B, N]

        if gamma > 0:
            # Focal weighting: upweight hard negatives (low E_neg)
            with torch.no_grad():
                neg_probs = F.softmax(neg_logits, dim=-1)  # [B, N]
                focal_weights = neg_probs.pow(gamma)       # [B, N]
                focal_weights = focal_weights * (N / focal_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8))
            neg_logits_weighted = neg_logits + focal_weights.log()
            logits = torch.cat([pos_logits, neg_logits_weighted], dim=1)  # [B, 1+N]
        else:
            logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, 1+N]

        labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        nce_loss = F.cross_entropy(logits, labels)

        # Gradient penalty: ||∇_chain E||² (smooth landscape for Langevin)
        # Compute on positive chains (they represent valid data region)
        # Skip when grad is disabled (eval mode with torch.no_grad())
        grad_penalty = torch.tensor(0.0, device=positive_chains.device)
        if self.cfg.lambda_grad > 0 and torch.is_grad_enabled():
            pos_for_grad = positive_chains.detach().requires_grad_(True)
            E_for_grad = self.forward(pos_for_grad, chain_lengths=pos_lengths)
            grads = torch.autograd.grad(
                E_for_grad.sum(), pos_for_grad, create_graph=True,
            )[0]
            grad_penalty = grads.pow(2).sum(dim=-1).mean()

        # Energy norm regularization: penalize |E| > margin
        energy_norm_reg = torch.tensor(0.0, device=positive_chains.device)
        if self.cfg.lambda_energy_norm > 0:
            all_E = torch.cat([E_pos, E_neg.reshape(-1)])
            energy_norm_reg = F.relu(
                all_E.abs() - self.cfg.energy_norm_margin
            ).pow(2).mean()

        loss = (
            nce_loss
            + self.cfg.lambda_grad * grad_penalty
            + self.cfg.lambda_energy_norm * energy_norm_reg
        )

        # Metrics
        with torch.no_grad():
            # Per-pair correctness: [B, N] bool
            pair_correct = (E_pos.unsqueeze(1) < E_neg).float()
            correct = pair_correct.mean()
            energy_gap = (E_neg.mean(dim=1) - E_pos).mean()
            metrics = {
                "chain_loss": loss.item(),
                "chain_nce_loss": nce_loss.item(),
                "chain_grad_penalty": grad_penalty.item(),
                "chain_energy_norm_reg": energy_norm_reg.item(),
                "chain_rank_acc": correct.item(),
                "chain_E_pos_mean": E_pos.mean().item(),
                "chain_E_neg_mean": E_neg.mean().item(),
                "chain_energy_gap": energy_gap.item(),
            }

            # Per-type accuracy (diagnostic: which negatives are too easy?)
            if neg_types is not None:
                type_correct: dict[str, list[float]] = {}
                for b in range(B):
                    for n_idx in range(N):
                        if n_idx < len(neg_types[b]):
                            ntype = neg_types[b][n_idx]
                            if ntype not in type_correct:
                                type_correct[ntype] = []
                            type_correct[ntype].append(pair_correct[b, n_idx].item())
                for ntype, vals in type_correct.items():
                    metrics[f"chain_acc_{ntype}"] = sum(vals) / len(vals)

        return loss, metrics

    def compute_margin_loss(
        self,
        positive_chains: Tensor,       # [B, L, D]
        negative_chains: Tensor,       # [B, N, L, D]
        pos_lengths: Tensor | None = None,
        neg_lengths: Tensor | None = None,
        margin: float = 1.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Margin-based ranking loss: E_pos + margin < E_neg.

        Simpler alternative to InfoNCE for early training stability.

        Returns:
            loss: scalar
            metrics: diagnostic dict
        """
        B, N, L, D = negative_chains.shape

        E_pos = self.forward(positive_chains, chain_lengths=pos_lengths)  # [B]
        neg_flat = negative_chains.reshape(B * N, L, D)
        neg_len_flat = neg_lengths.reshape(B * N) if neg_lengths is not None else None
        E_neg = self.forward(neg_flat, chain_lengths=neg_len_flat).reshape(B, N)

        loss = F.relu(E_pos.unsqueeze(1) - E_neg + margin).mean()

        with torch.no_grad():
            correct = (E_pos.unsqueeze(1) < E_neg).float().mean()
            metrics = {
                "chain_margin_loss": loss.item(),
                "chain_rank_acc": correct.item(),
                "chain_E_pos_mean": E_pos.mean().item(),
                "chain_E_neg_mean": E_neg.mean().item(),
            }

        return loss, metrics

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
