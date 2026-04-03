"""
EBT Chain Head: Transformer-based reasoning chain evaluator.

Evaluates quality of reasoning chains V₁ → V₂ → ... → Vₙ using
self-attention to detect global inconsistencies (e.g., step 5
contradicts step 2).

Architecture:
  [CLS] + [V₁, V₂, ..., Vₙ] + ALiBi positional bias
    → TransformerEncoder (2 layers, 8 heads, 2048 FFN, Pre-Norm)
    → CLS pooling
    → Energy head: Linear(1024, 512) + GELU + Linear(512, 1) → scalar E

Why ALiBi (not RoPE):
  RoPE rotates Q/K vectors, breaking SONAR semantic geometry.
  ALiBi only adds additive bias -m|i-j| to attention scores —
  Q, K, V vectors stay untouched, preserving SONAR distances.
  For short chains (5-20 elements), ALiBi slopes provide sufficient
  order signal without modifying the embedding space.

Energy semantics: lower E = more coherent chain.

Training: Focal-InfoNCE over positive vs negative chains.
Negatives: shuffled order, truncated, corrupted, wrong conclusion.

At inference: called every 5-10 Langevin steps during Deep Thinking
(System 2) mode to validate the reasoning chain being constructed.

Spec reference: §5.2 Mode B, §8.2 (Deep Thinking), §9.9 (ALiBi)
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


# ─── ALiBi for Chain Head Self-Attention ──────────────────────────────
# ALiBi adds -m|i-j| bias to attention scores (Press et al., ICLR 2022).
# Does NOT modify Q/K/V — preserves SONAR geometry fully.
# Slopes m follow a geometric sequence per head.


def _get_alibi_slopes(n_heads: int) -> Tensor:
    """
    Compute ALiBi slopes as geometric sequence.

    For n_heads=8: slopes = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
    Closest heads focus on local order, farthest on global structure.
    """
    def _power_of_2_slopes(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = _power_of_2_slopes(n_heads)
    else:
        closest_pow2 = 2 ** math.floor(math.log2(n_heads))
        slopes = _power_of_2_slopes(closest_pow2)
        extra = _power_of_2_slopes(2 * closest_pow2)
        slopes = slopes + extra[0::2][: n_heads - closest_pow2]
    return torch.tensor(slopes, dtype=torch.float32)


def _build_alibi_bias(seq_len: int, n_heads: int, device: torch.device) -> Tensor:
    """
    Build ALiBi attention bias matrix.

    Returns: [n_heads, seq_len, seq_len] additive bias
    """
    slopes = _get_alibi_slopes(n_heads).to(device)
    positions = torch.arange(seq_len, device=device)
    # |i - j| distance matrix
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs().float()  # [L, L]
    # Per-head bias: -slope * |i-j|
    bias = -slopes[:, None, None] * dist.unsqueeze(0)  # [H, L, L]
    return bias


class ALiBiSelfAttention(nn.Module):
    """
    Multi-head self-attention with ALiBi positional bias.

    ALiBi adds additive bias -m|i-j| to attention scores without
    modifying Q/K/V vectors. This preserves SONAR embedding geometry
    while encoding positional information for chain ordering.

    Properties:
      - Q, K, V vectors unchanged → SONAR cosine distances preserved
      - Larger slopes (head 0) = strong local bias (order-sensitive)
      - Smaller slopes (head 7) = weak global bias (content-sensitive)
      - No learnable position parameters — zero extra params
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

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # Pre-compute ALiBi slopes (constant, no grad)
        slopes = _get_alibi_slopes(n_heads)
        self.register_buffer("alibi_slopes", slopes, persistent=False)
        # Pre-compute bias for max length
        bias = _build_alibi_bias(max_seq_len, n_heads, torch.device("cpu"))
        self.register_buffer("_alibi_bias_cache", bias, persistent=False)
        self._cached_seq_len = max_seq_len

    def _get_alibi_bias(self, seq_len: int, device: torch.device) -> Tensor:
        """Get ALiBi bias for given sequence length, rebuilding cache if needed."""
        if seq_len <= self._cached_seq_len:
            return self._alibi_bias_cache[:, :seq_len, :seq_len].to(device)
        # Rebuild for longer sequence
        bias = _build_alibi_bias(seq_len, self.n_heads, device)
        self._alibi_bias_cache = bias
        self._cached_seq_len = seq_len
        return bias

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

        # Scaled dot-product + ALiBi bias
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L]

        # Add ALiBi positional bias (does NOT touch Q/K vectors)
        alibi = self._get_alibi_bias(L, x.device)  # [H, L, L]
        attn_weights = attn_weights + alibi.unsqueeze(0)  # broadcast over batch

        # Apply padding mask
        if attn_mask is not None:
            mask_kv = attn_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_weights = attn_weights.masked_fill(~mask_kv, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, H, L, D_head]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class ChainTransformerLayer(nn.Module):
    """
    Single Transformer encoder layer with ALiBi attention + FFN.

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
        self.attn = ALiBiSelfAttention(
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
      - ALiBi-biased Transformer encoder (2 layers, 8 heads, Pre-Norm)
      - CLS vector extraction after encoding
      - Energy head: Linear(1024, 512) + GELU + Linear(512, 1)

    The [CLS] token acts as a global aggregator — self-attention allows it
    to attend to all chain positions simultaneously, detecting contradictions
    between arbitrary steps. ALiBi provides order sensitivity without
    modifying the SONAR vectors.

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
        loss = F.cross_entropy(logits, labels)

        # Metrics
        with torch.no_grad():
            correct = (E_pos.unsqueeze(1) < E_neg).float().mean()
            energy_gap = (E_neg.mean(dim=1) - E_pos).mean()
            metrics = {
                "chain_loss": loss.item(),
                "chain_rank_acc": correct.item(),
                "chain_E_pos_mean": E_pos.mean().item(),
                "chain_E_neg_mean": E_neg.mean().item(),
                "chain_energy_gap": energy_gap.item(),
            }

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
