from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cebcm.models.chain_head import _apply_rope, _build_rope_cache


@dataclass
class ChainGeneratorConfig:
    """Configuration for autoregressive chain generation in SONAR space."""

    d_model: int = 1024
    n_heads: int = 8
    n_layers: int = 6
    dim_feedforward: int = 4096
    max_chain_len: int = 20
    dropout: float = 0.1
    target_norm: float = 0.2051

    # Supervised teacher-forcing step loss.
    loss_cosine_weight: float = 1.0
    loss_mse_weight: float = 0.1

    # Diffusion Forcing (continuous SONAR-space denoising objective).
    diffusion_timesteps: int = 64
    diffusion_beta_schedule: str = "cosine"
    # 0.0 means target_norm / sqrt(d_model), i.e. unit Gaussian direction
    # rescaled to the SONAR hypersphere scale instead of raw N(0, I).
    diffusion_noise_scale: float = 0.0

    # ── v-prediction (Salimans & Ho 2022) ──
    # "x0" = predict clean target, "v" = predict velocity, "eps" = predict noise.
    prediction_type: str = "v"

    # ── Classifier-Free Guidance ──
    # During training, drop context with this probability to learn unconditional.
    cfg_dropout_prob: float = 0.1

    # ── Architecture extensions ──
    # "swiglu" replaces SiLU FFN with gated SwiGLU (Shazeer 2020).
    ffn_type: str = "swiglu"
    # "ada_rmsnorm" = AdaLN with RMSNorm base for DF conditioning.
    # "layernorm" = original LayerNorm (backward compatible).
    norm_type: str = "ada_rmsnorm"

    # ── ResNet / DiT stabilization practices ──
    # Zero-init the final projector of every residual sublayer (self-attn
    # out_proj, cross-attn out_proj, FFN down-projection).  At init each
    # block is a perfect identity, which creates a clean gradient highway
    # from the loss down to layer 0 (FixUp / DeepNet / DiT).
    zero_init_residual: bool = True
    # Per-residual learnable gate γ_l (CaiT LayerScale).  Each sublayer
    # output is multiplied by a [D]-shaped parameter initialized to
    # ``layerscale_init`` before the residual add.  With a small init the
    # block starts near-identity and "opens up" as training progresses,
    # dampening the early-training chaos that Diffusion Forcing is
    # particularly prone to.
    use_layerscale: bool = True
    layerscale_init: float = 1e-4

    # ── Residual-branch norm clamp (bf16-overflow safety valve) ──
    # Soft-clamp the per-token L2 norm of every residual sublayer output
    # (self-attn, cross-attn, FFN) to this bound *before* the residual add.
    # The FFN path is the main offender: gated ``W_down(SiLU(W_g x) ⊙ W_u x)``
    # can grow multiplicatively without any built-in bound, and once any single
    # residual contribution pushes per-element values past ~76 in bf16 the
    # subsequent QK logits overflow (bf16 max = 65504).  LayerScale + weight
    # decay slow this but do not prevent it.  A direct, differentiable
    # ``min(1, C/||y||)`` rescaling is the SOTA safety valve — it is a no-op
    # in the healthy regime and only activates when a branch tries to emit
    # an out-of-manifold spike.  Recommended default: ``8 * sqrt(d_model)``.
    # ``None`` or ``<= 0`` disables (backward compatible).
    residual_norm_clamp: float | None = None

    # ── AdaLN scale clamp (FFN explosion safety valve) ──
    # Soft-bound the AdaLN modulation scale output via tanh so the gate
    # (1 + scale) stays in [GATE_MIN, 1 + adaln_scale_clamp].  Without
    # this, the modulation MLP can produce arbitrarily large positive
    # scales, which SwiGLU amplifies quadratically (output ∝ input²),
    # causing bf16 overflow at horizon transitions where cold positions
    # see extreme residual-stream statistics.  Default 1.0 caps the gate
    # at 2.0× (ample for noise-level conditioning).  None disables.
    adaln_scale_clamp: float | None = 1.0

    # Deep supervision / auxiliary heads. Layer numbers are 1-based to match
    # human-facing configs ("layer 2", "layer 4"). Empty disables the feature.
    aux_head_layers: tuple[int, ...] = ()
    aux_head_hidden_dim: int = 1024
    aux_head_dropout: float = 0.0


# ---------------------------------------------------------------------------
# Normalization & FFN building blocks
# ---------------------------------------------------------------------------

class AdaRMSNorm(nn.Module):
    """RMSNorm with optional Adaptive Layer Normalization (AdaLN) conditioning.

    Base: ``RMSNorm(x) = x / RMS(x) * γ``  (Zhang & Sennrich 2019)
    AdaLN: ``AdaRMSN(x, s, b) = RMSNorm(x) * gate + b``
    where *gate = (1 + s)* and *s, b* are produced by a timestep MLP.

    Two numerical-stability mechanisms (lessons.md 2026-04-20 evening):

    1. **Full FP32 modulation chain.**  bf16 has only ~7 bits of mantissa,
       which becomes catastrophically coarse when the modulation gate
       ``(1 + scale)`` drifts close to zero (a known AdaLN failure mode).
       We keep the entire normalize → gain → modulate → shift pipeline
       in FP32 and only down-cast at the final return.

    2. **Soft floor on the AdaLN gate.**  ``(1 + scale)`` is unbounded
       below in vanilla AdaLN, and the user's GB10 runs showed catastrophic
       gradient suppression when it dipped near 0 (signal annihilation).
       We clamp the gate to a strictly-positive floor so the residual
       branch can be strongly suppressed but never inverted or zeroed out.

    When *scale* and *shift* are ``None`` the layer is a standard RMSNorm,
    so teacher-forcing (no timestep) works unchanged.
    """

    # Gate floor: (1 + scale) is clamped to >= GATE_MIN.  At GATE_MIN=1e-3
    # the model can still attenuate the signal by 1000x — plenty for any
    # real "suppress this position" use case — without ever hitting the
    # NaN-prone 0 region or flipping sign.
    GATE_MIN: float = 1e-3

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(
            self,
            x: Tensor,
            scale: Tensor | None = None,
            shift: Tensor | None = None,
    ) -> Tensor:
        in_dtype = x.dtype
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        # Stay in FP32 through γ-multiplication; bf16 here loses precision
        # whenever the post-norm signal's per-channel scale is small.
        out = x_f * rms * self.weight.float()
        if scale is not None:
            gate = (1.0 + scale.float()).clamp(min=self.GATE_MIN)
            out = out * gate
        if shift is not None:
            out = out + shift.float()
        return out.to(in_dtype)


class SwiGLUFFN(nn.Module):
    """Gated Linear Unit with SiLU gate (Shazeer 2020, LLaMA-style).

    ``SwiGLU(x) = W_down( SiLU(W_gate(x)) ⊙ W_up(x) )``

    Three projections instead of two; no biases (standard practice).

    Two anti-collapse mechanisms (lessons.md 2026-04-20 evening):

    1. **Orthogonal init for W_gate / W_up.**  Xavier-uniform tends to
       cluster gate inputs around zero, so ``SiLU(gate)`` lives in the
       near-linear regime — exactly the "dead zone" the third-party
       analysis warned about, where the gradient through the gate path
       is ~0.5 and any drift toward the saturating tail kills training.
       Orthogonal init spreads the gate inputs across the SiLU non-linear
       region, preserving signal entropy and gradient flow at start.

    2. **In-band activation telemetry.**  We record gate-input mean/std
       and post-down output norm as non-persistent buffers, so the
       training loop can poll them every ``probe_every_steps`` and
       detect dead-zone drift before it kills training.  No per-step
       cost beyond a single ``.detach().mean()`` when tracking is on.
    """

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_up = nn.Linear(d_model, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Activation telemetry — populated when ``track_stats`` is True.
        self.track_stats: bool = False
        self.register_buffer("_gate_in_abs_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("_gate_in_std", torch.tensor(0.0), persistent=False)
        self.register_buffer("_silu_kurtosis", torch.tensor(0.0), persistent=False)
        self.register_buffer("_out_norm_mean", torch.tensor(0.0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        gate_in = self.w_gate(x)
        silu_gate = F.silu(gate_in)
        out = self.dropout(self.w_down(silu_gate * self.w_up(x)))
        if self.track_stats and self.training:
            # Stay on-device: copy_ keeps the values as tensors so we do
            # NOT trigger a GPU→CPU sync on every forward.  The training
            # loop pays the .item() cost only when polling at probe time.
            with torch.no_grad():
                gi = gate_in.detach().float()
                self._gate_in_abs_mean.copy_(gi.abs().mean())
                self._gate_in_std.copy_(gi.std())
                # Excess kurtosis of post-SiLU values: high → spiky/dead,
                # ~0 → Gaussian-like (healthy).  Detects the "лес из нулей
                # и редких пиков" failure mode the analysis warned about.
                sg = silu_gate.detach().float()
                centered = sg - sg.mean()
                var = centered.pow(2).mean().clamp(min=1e-12)
                self._silu_kurtosis.copy_(centered.pow(4).mean() / var.pow(2) - 3.0)
                self._out_norm_mean.copy_(out.detach().float().norm(dim=-1).mean())
        return out


def _soft_clamp_residual_norm(
    y: Tensor, max_norm: float | None, eps: float = 1e-6
) -> Tensor:
    """Soft-clamp the L2 norm of ``y`` along the last (feature) dim.

    For each token (position in the sequence), if ``||y||_2 <= max_norm`` the
    vector passes through unchanged; if ``||y||_2 > max_norm`` it is uniformly
    rescaled to have norm exactly ``max_norm``.  This is the standard
    gradient-clipping construction applied to forward activations and is
    differentiable everywhere (in the sub-gradient sense at the boundary).

    The norm is computed in FP32 — crucial under bf16 autocast because the
    very thing we are trying to bound can already be near bf16's 65504 ceiling,
    and computing ``.norm()`` in bf16 would itself overflow to inf.

    Args:
        y: [..., D] residual sublayer output (any leading dims allowed).
        max_norm: positive scalar bound.  ``None`` or non-positive is a no-op.
        eps: floor on the denominator to avoid division by zero on exactly-
             zero tokens (the rescale factor becomes ``max_norm / eps`` but
             is then clamped to <= 1, so the result is still ``y``).

    Returns:
        Tensor with the same shape/dtype as ``y`` and per-token L2 norm
        bounded above by ``max_norm``.
    """
    if max_norm is None or float(max_norm) <= 0.0:
        return y
    y32 = y.float()
    norm = y32.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = (float(max_norm) / norm).clamp_max(1.0)
    return (y32 * scale).to(y.dtype)


class AuxiliaryPredictionHead(nn.Module):
    """Small per-layer prediction head for deep supervision.

    The head maps an intermediate residual-stream state back to SONAR-vector
    space. It is intentionally independent from the main output projection:
    auxiliary losses should diagnose and regularize intermediate layers, not
    silently couple them to the final decoder head.
    """

    def __init__(
            self,
            d_model: int,
            hidden_dim: int = 1024,
            dropout: float = 0.0,
            norm_type: str = "ada_rmsnorm",
    ) -> None:
        super().__init__()
        self.norm = AdaRMSNorm(d_model) if norm_type == "ada_rmsnorm" else nn.LayerNorm(d_model)
        hidden = int(hidden_dim)
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, d_model),
            )
        else:
            self.net = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.norm(x))

    def init_output(self, gain: float = 0.01) -> None:
        """Initialize the final projection small, matching the main head."""
        modules = list(self.net.modules()) if isinstance(self.net, nn.Module) else []
        for module in reversed(modules):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                return


class CrossAttention(nn.Module):
    """Multi-head cross-attention over context bank [B, K, D]."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: [B, L, D]
            context: [B, K, D]
            context_mask: optional [B, K] bool/float (1=valid, 0=masked)
        """
        bsz, seq_len, d_model = x.shape
        ctx_len = context.shape[1]

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(bsz, ctx_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(bsz, ctx_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if context_mask is not None:
            cm = context_mask.to(device=x.device)
            if cm.shape != (bsz, ctx_len):
                raise ValueError(
                    f"context_mask shape {tuple(cm.shape)} does not match context shape {(bsz, ctx_len)}"
                )
            # SDPA additive mask.  CRITICAL: using ``float("-inf")`` in
            # bfloat16 triggers NaN in SDPA on CUDA (flash / mem-efficient
            # backends) — softmax over a row of all ``-inf`` produces
            # ``0/0 = NaN`` and ``-inf + -inf = -inf`` poisons gradients.
            # The standard safe pattern (HuggingFace transformers) is to
            # use ``finfo(dtype).min``: ~-3.4e38 in fp32, ~-3.4e38 cast to
            # bf16 clips to ~-3.3e38 — large enough that ``exp(logit + min)``
            # underflows to 0, but never produces ``inf - inf = NaN``.
            invalid = cm <= 0
            # Additionally, if an ENTIRE row is masked out (rare but not
            # impossible with context_bank_size=4), SDPA still NaNs.  Force
            # such rows to be fully visible: they'll attend uniformly to
            # whatever is there, which is strictly better than NaN and the
            # downstream loss will correctly mask the position anyway.
            all_invalid = invalid.all(dim=-1, keepdim=True)  # [B, 1]
            invalid = invalid & ~all_invalid
            mask_min = torch.finfo(q.dtype).min
            attn_mask = torch.zeros(
                (bsz, self.n_heads, seq_len, ctx_len),
                device=x.device,
                dtype=q.dtype,
            )
            attn_mask = attn_mask.masked_fill(
                invalid[:, None, None, :], mask_min
            )

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=self.head_dim ** -0.5,
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.out_proj(out)


class CausalRoPESelfAttention(nn.Module):
    """Causal self-attention with RoPE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

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
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        rope = self._get_rope(seq_len, x.device)
        q = _apply_rope(q, rope)
        k = _apply_rope(k, rope)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=True,
            scale=self.head_dim ** -0.5,
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class DecoderBlock(nn.Module):
    """Pre-norm decoder block: self-attn → cross-attn → FFN.

    Supports two norm types (configured at construction):
    - ``"layernorm"``: standard ``nn.LayerNorm`` (original, no timestep cond.)
    - ``"ada_rmsnorm"``: ``AdaRMSNorm`` with optional per-position scale/shift
      from a diffusion timestep MLP (DiT-style AdaLN-Zero conditioning).

    FFN type:
    - ``"silu"``: ``Linear → SiLU → Linear`` (original)
    - ``"swiglu"``: ``SwiGLUFFN`` with gated activation (LLaMA-style)

    Stabilization (enabled by default):
    - **Zero-init residual projectors**: the final projection of every
      sublayer (``self_attn.out_proj``, ``cross_attn.out_proj``, the FFN
      down-projection) is zeroed at init so each block starts as an exact
      identity — the "gradient highway" trick from FixUp / DeepNet / DiT.
    - **LayerScale** (CaiT, Touvron et al. 2021): each sublayer's output is
      scaled by a learnable per-channel gate γ_l initialized to ``1e-4``
      before the residual add, preventing early-training blow-ups in
      Diffusion Forcing.

    The actual zeroing of residual projectors is applied by the parent
    ``ChainGenerator._init_weights`` *after* its xavier sweep; doing it
    here in ``__init__`` would be silently overwritten.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_seq_len: int = 32,
        norm_type: str = "ada_rmsnorm",
        ffn_type: str = "swiglu",
        use_layerscale: bool = True,
        layerscale_init: float = 1e-4,
        residual_norm_clamp: float | None = None,
        adaln_scale_clamp: float | None = 1.0,
    ):
        super().__init__()
        self._norm_type = norm_type
        self._ffn_type = ffn_type
        self._use_layerscale = bool(use_layerscale)
        # Store as Python float (or None) — read on every forward, so keep it
        # as a plain scalar to avoid tensor-creation overhead in the hot path.
        if residual_norm_clamp is None or float(residual_norm_clamp) <= 0.0:
            self._residual_norm_clamp: float | None = None
        else:
            self._residual_norm_clamp = float(residual_norm_clamp)
        if adaln_scale_clamp is not None and float(adaln_scale_clamp) > 0.0:
            self._adaln_scale_clamp: float | None = float(adaln_scale_clamp)
        else:
            self._adaln_scale_clamp = None

        # ── Normalization ──
        if norm_type == "ada_rmsnorm":
            self.norm_self = AdaRMSNorm(d_model)
            self.norm_cross = AdaRMSNorm(d_model)
            self.norm_ffn = AdaRMSNorm(d_model)
            # AdaLN modulation: 6 vectors (scale+shift for each of 3 norms).
            # NOTE on init: we ask for zero-weight + zero-bias here so the
            # block is identity-through-norm (scale=0→γ=1, shift=0→β=0) at
            # start.  But ``ChainGenerator._init_weights`` runs a xavier
            # sweep over every ``nn.Linear`` *after* construction, which
            # silently overwrites this.  The parent's ``_init_weights``
            # explicitly re-zeros ``adaln_modulation[-1]`` after the sweep;
            # do NOT remove that call — otherwise we train with xavier
            # adaLN and lose the AdaLN-Zero property entirely.
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, 6 * d_model, bias=True),
            )
            nn.init.zeros_(self.adaln_modulation[-1].weight)
            nn.init.zeros_(self.adaln_modulation[-1].bias)
        else:
            self.norm_self = nn.LayerNorm(d_model)
            self.norm_cross = nn.LayerNorm(d_model)
            self.norm_ffn = nn.LayerNorm(d_model)
            self.adaln_modulation = None

        # ── Sub-layers ──
        self.self_attn = CausalRoPESelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.cross_attn = CrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # ── FFN ──
        if ffn_type == "swiglu":
            self.ffn = SwiGLUFFN(d_model, dim_feedforward, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )

        # ── LayerScale (CaiT) ──
        # One per-channel gate γ_l per sublayer, initialized to a small
        # constant so each residual branch contributes ~0 at start and the
        # identity path from zero-init residual dominates.
        if self._use_layerscale:
            init_val = float(layerscale_init)
            self.ls_self = nn.Parameter(torch.full((d_model,), init_val))
            self.ls_cross = nn.Parameter(torch.full((d_model,), init_val))
            self.ls_ffn = nn.Parameter(torch.full((d_model,), init_val))
        else:
            self.register_parameter("ls_self", None)
            self.register_parameter("ls_cross", None)
            self.register_parameter("ls_ffn", None)

    def _scale(self, y: Tensor, gamma: Tensor | None) -> Tensor:
        """Apply LayerScale gate if enabled."""
        return y * gamma if gamma is not None else y

    def _zero_init_residual_projectors(self) -> None:
        """Zero the final projector weight of every residual sublayer.

        After this call the block is an **exact identity** at initialization:
        ``self_attn(norm(x)) = 0``, ``cross_attn(norm(x)) = 0``, ``ffn(norm(x)) = 0``,
        so ``x → x + 0 + 0 + 0 = x`` through every layer.  This is the
        "zero-init residual" trick from FixUp / DeepNet / DiT and is what
        allows deep residual networks to receive a clean, full-magnitude
        gradient on every parameter from the very first step.
        """
        nn.init.zeros_(self.self_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        if self._ffn_type == "swiglu":
            nn.init.zeros_(self.ffn.w_down.weight)
        else:
            # nn.Sequential: [Linear, SiLU, Dropout, Linear, Dropout].
            # Zero the last Linear (the down-projection).
            for m in reversed(list(self.ffn)):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    break

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        context_mask: Tensor | None = None,
        t_emb: Tensor | None = None,
    ) -> Tensor:
        """Forward with optional timestep conditioning.

        Args:
            x: [B, L, D] input sequence.
            context: [B, K, D] cross-attention context.
            context_mask: [B, K] bool mask for context.
            t_emb: [B, L, D] per-position timestep embedding (DF mode).
                   ``None`` for teacher-forcing / generation without diffusion.
        """
        clamp = self._residual_norm_clamp  # Python float or None — hot-path read.
        if self.adaln_modulation is not None and t_emb is not None:
            # AdaLN-Zero: produce per-position scale/shift for each norm.
            mod = self.adaln_modulation(t_emb)  # [B, L, 6D]
            s_sa, sh_sa, s_ca, sh_ca, s_ff, sh_ff = mod.chunk(6, dim=-1)
            sc = self._adaln_scale_clamp
            if sc is not None:
                s_sa = torch.tanh(s_sa) * sc
                s_ca = torch.tanh(s_ca) * sc
                s_ff = torch.tanh(s_ff) * sc
            sa_out = self.self_attn(self.norm_self(x, scale=s_sa, shift=sh_sa))
            sa_out = _soft_clamp_residual_norm(sa_out, clamp)
            x = x + self._scale(sa_out, self.ls_self)

            ca_out = self.cross_attn(
                self.norm_cross(x, scale=s_ca, shift=sh_ca),
                context,
                context_mask=context_mask,
            )
            ca_out = _soft_clamp_residual_norm(ca_out, clamp)
            x = x + self._scale(ca_out, self.ls_cross)

            ffn_out = self.ffn(self.norm_ffn(x, scale=s_ff, shift=sh_ff))
            ffn_out = _soft_clamp_residual_norm(ffn_out, clamp)
            x = x + self._scale(ffn_out, self.ls_ffn)
        else:
            # Standard pre-norm (no conditioning).
            sa_out = self.self_attn(self.norm_self(x))
            sa_out = _soft_clamp_residual_norm(sa_out, clamp)
            x = x + self._scale(sa_out, self.ls_self)

            ca_out = self.cross_attn(
                self.norm_cross(x), context, context_mask=context_mask
            )
            ca_out = _soft_clamp_residual_norm(ca_out, clamp)
            x = x + self._scale(ca_out, self.ls_cross)

            ffn_out = self.ffn(self.norm_ffn(x))
            ffn_out = _soft_clamp_residual_norm(ffn_out, clamp)
            x = x + self._scale(ffn_out, self.ls_ffn)
        return x


class ChainGenerator(nn.Module):
    """Autoregressive transformer decoder in SONAR embedding space."""

    def __init__(self, cfg: ChainGeneratorConfig | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else ChainGeneratorConfig()

        # Initialize start_token to have standard normal variance (norm ≈ sqrt(D) ≈ 32)
        # to correctly match the scaled residual stream magnitude.
        self.start_token = nn.Parameter(torch.randn(1, 1, self.cfg.d_model))

        # Learnable null-context token for Classifier-Free Guidance (CFG).
        # During training, context is replaced with this token with prob
        # cfg_dropout_prob; at inference, used for unconditional forward pass.
        self.null_context_token = nn.Parameter(torch.zeros(1, 1, self.cfg.d_model))

        max_seq = self.cfg.max_chain_len + 1
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=self.cfg.d_model,
                    n_heads=self.cfg.n_heads,
                    dim_feedforward=self.cfg.dim_feedforward,
                    dropout=self.cfg.dropout,
                    max_seq_len=max_seq,
                    norm_type=self.cfg.norm_type,
                    ffn_type=self.cfg.ffn_type,
                    use_layerscale=self.cfg.use_layerscale,
                    layerscale_init=self.cfg.layerscale_init,
                    residual_norm_clamp=self.cfg.residual_norm_clamp,
                    adaln_scale_clamp=self.cfg.adaln_scale_clamp,
                )
                for _ in range(self.cfg.n_layers)
            ]
        )

        # Final norm matches the decoder block norm type.
        if self.cfg.norm_type == "ada_rmsnorm":
            self.final_norm = AdaRMSNorm(self.cfg.d_model)
        else:
            self.final_norm = nn.LayerNorm(self.cfg.d_model)

        self.output_proj = nn.Linear(self.cfg.d_model, self.cfg.d_model)

        self._aux_layer_numbers = self._normalize_aux_layers(self.cfg.aux_head_layers)
        self.aux_heads = nn.ModuleDict(
            {
                str(layer_num): AuxiliaryPredictionHead(
                    d_model=self.cfg.d_model,
                    hidden_dim=self.cfg.aux_head_hidden_dim,
                    dropout=self.cfg.aux_head_dropout,
                    norm_type=self.cfg.norm_type,
                )
                for layer_num in self._aux_layer_numbers
            }
        )

        # Timestep MLP for diffusion noise-level conditioning.
        self.diffusion_time_mlp = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.SiLU(),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
        )

        self._build_diffusion_schedule()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters for a deep, diffusion-stable transformer.

        Order matters — we first do a global xavier sweep over every
        ``nn.Linear``, then selectively re-initialize a handful of projectors
        to enforce the three "ResNet for diffusion" stabilization tricks:

        1. **Zero-init residual projectors** (FixUp / DeepNet / DiT): the
           final projection of every residual sublayer is zeroed, making
           each ``DecoderBlock`` start as an exact identity.  Combined with
           LayerScale (initialized tiny), this guarantees a clean gradient
           highway from the loss straight to layer 0 at step 0.
        2. **AdaLN-Zero**: the final Linear of every block's timestep
           modulation MLP is zeroed so the AdaRMSNorm layers start as plain
           RMSNorm (scale=0 → γ=1, shift=0 → β=0).  Without the re-zeroing
           below, the xavier sweep overwrites the zeros set inside
           ``DecoderBlock.__init__``, silently breaking AdaLN-Zero.
        3. **Small output head**: ``output_proj`` uses xavier with gain
           0.01 so initial predictions are small in SONAR space (compatible
           with the 0.2051 target norm).
        """
        # ── 1. Global xavier sweep ──
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # ── 2. Re-enforce AdaLN-Zero on every block ──
        for block in self.layers:
            if getattr(block, "adaln_modulation", None) is not None:
                final = block.adaln_modulation[-1]
                nn.init.zeros_(final.weight)
                if final.bias is not None:
                    nn.init.zeros_(final.bias)

        # ── 3. Zero-init residual projectors (FixUp / DeepNet / DiT) ──
        if self.cfg.zero_init_residual:
            for block in self.layers:
                block._zero_init_residual_projectors()

        # ── 4. Small output head ──
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.01)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        for head in self.aux_heads.values():
            head.init_output(gain=0.01)

        # ── 5. Orthogonal init for SwiGLU gate / up projections ──
        # Xavier-uniform clusters gate inputs near zero (where SiLU is
        # near-linear, gradient ~0.5), creating a "dead zone" that the
        # third-party scale-collapse analysis flagged.  Orthogonal
        # weights spread the gate inputs across SiLU's non-linear
        # region, preserving signal entropy and gradient flow at start.
        # Skip ``w_down`` — it's zero-initialized just above by
        # _zero_init_residual_projectors() and we must NOT overwrite it.
        for block in self.layers:
            ffn = getattr(block, "ffn", None)
            if isinstance(ffn, SwiGLUFFN):
                nn.init.orthogonal_(ffn.w_gate.weight, gain=1.0)
                nn.init.orthogonal_(ffn.w_up.weight, gain=1.0)

    def _normalize_aux_layers(self, layers: tuple[int, ...] | list[int] | int | None) -> tuple[int, ...]:
        """Return sorted unique 1-based layer numbers that are valid."""
        if layers is None:
            return ()
        if isinstance(layers, int):
            raw = [layers]
        else:
            raw = list(layers)
        out: list[int] = []
        for value in raw:
            layer_num = int(value)
            if layer_num < 1 or layer_num > int(self.cfg.n_layers):
                raise ValueError(
                    f"aux_head layer {layer_num} is outside valid range 1..{self.cfg.n_layers}"
                )
            if layer_num not in out:
                out.append(layer_num)
        return tuple(sorted(out))

    def _maybe_aux_predict(self, layer_num: int, x: Tensor, aux: dict[str, Tensor] | None) -> None:
        if aux is None:
            return
        key = str(layer_num)
        if key in self.aux_heads:
            aux[key] = self.aux_heads[key](x)

    def _build_diffusion_schedule(self) -> None:
        """Register DDPM schedule buffers used by Diffusion Forcing.

        Timesteps are indexed 0..K-1.  t=0 is near-clean, t=K-1 is the
        noisiest state.  Noise itself is SONAR-scaled in q_sample, not raw
        N(0, I), so the vector norm stays compatible with the hypersphere.
        """
        timesteps = max(2, int(self.cfg.diffusion_timesteps))
        schedule = str(self.cfg.diffusion_beta_schedule).lower()

        if schedule == "cosine":
            s = 0.008
            x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).pow(2)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0].clamp(min=1e-12)
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1].clamp(min=1e-12))
            betas = betas.clamp(min=1e-5, max=0.999)
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, timesteps, dtype=torch.float64)
        else:
            raise ValueError(f"Unsupported diffusion_beta_schedule: {self.cfg.diffusion_beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).float()
        snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)

        self.register_buffer("_df_sqrt_alphas_cumprod", alphas_cumprod.sqrt(), persistent=True)
        self.register_buffer(
            "_df_sqrt_one_minus_alphas_cumprod",
            (1.0 - alphas_cumprod).clamp(min=0.0).sqrt(),
            persistent=True,
        )
        self.register_buffer("_df_alphas_cumprod", alphas_cumprod.float(), persistent=True)
        self.register_buffer("_df_snr", snr.float(), persistent=True)

    def _diffusion_noise_scale(self) -> float:
        if float(self.cfg.diffusion_noise_scale) > 0.0:
            return float(self.cfg.diffusion_noise_scale)
        return float(self.cfg.target_norm) / math.sqrt(float(self.cfg.d_model))

    def _diffusion_timestep_embedding(self, noise_levels: Tensor) -> Tensor:
        """Sinusoidal timestep embedding projected to residual-stream scale."""
        half = self.cfg.d_model // 2
        if half <= 0:
            raise ValueError("d_model must be >= 2 for diffusion timestep embeddings")

        # Use raw diffusion level, not [0, 1] normalization.  Standard
        # sinusoidal timestep embeddings rely on the timestep range itself;
        # compressing 0..K-1 into 0..1 makes most frequencies nearly constant
        # and weakens noise-level conditioning.
        t = noise_levels.float()
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=noise_levels.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.cfg.d_model:
            emb = F.pad(emb, (0, self.cfg.d_model - emb.shape[-1]))
        emb = emb.to(dtype=self.diffusion_time_mlp[0].weight.dtype)
        return self.diffusion_time_mlp(emb).to(dtype=self.start_token.dtype)

    def diffusion_q_sample(
        self,
        x_start: Tensor,
        noise_levels: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """SONAR-safe forward diffusion: x_t = sqrt(a)*x0 + sqrt(1-a)*eps.

        The returned epsilon has expected norm near target_norm, not sqrt(D).
        This is the key adaptation to SONAR geometry and prevents raw DDPM
        noise from dwarfing 0.205-norm semantic vectors.
        """
        if noise_levels.shape != x_start.shape[:2]:
            raise ValueError(
                f"noise_levels shape {tuple(noise_levels.shape)} must match x_start[:2]={tuple(x_start.shape[:2])}"
            )
        levels = noise_levels.to(device=x_start.device, dtype=torch.long).clamp(
            min=0,
            max=max(int(self.cfg.diffusion_timesteps) - 1, 0),
        )
        if noise is None:
            noise = torch.randn_like(x_start) * self._diffusion_noise_scale()
        sqrt_alpha = self._df_sqrt_alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype)[levels]
        sqrt_one_minus = self._df_sqrt_one_minus_alphas_cumprod.to(
            device=x_start.device, dtype=x_start.dtype
        )[levels]
        x_noisy = sqrt_alpha.unsqueeze(-1) * x_start + sqrt_one_minus.unsqueeze(-1) * noise
        return x_noisy, noise

    def diffusion_v_target(
            self,
            x_start: Tensor,
            noise: Tensor,
            noise_levels: Tensor,
    ) -> Tensor:
        """Compute v-prediction target: v_t = √α̅_t · ε − √(1−α̅_t) · x₀.

        This is the "velocity" in the diffusion ODE interpretation
        (Salimans & Ho 2022).  At t≈0, v ≈ ε (all noise); at t≈T, v ≈ −x₀
        (all signal).  Both extremes are easy to predict, concentrating
        difficulty at intermediate noise levels.
        """
        levels = noise_levels.to(device=x_start.device, dtype=torch.long).clamp(
            min=0, max=max(int(self.cfg.diffusion_timesteps) - 1, 0),
        )
        sqrt_alpha = self._df_sqrt_alphas_cumprod.to(
            device=x_start.device, dtype=x_start.dtype,
        )[levels].unsqueeze(-1)
        sqrt_one_minus = self._df_sqrt_one_minus_alphas_cumprod.to(
            device=x_start.device, dtype=x_start.dtype,
        )[levels].unsqueeze(-1)
        return sqrt_alpha * noise - sqrt_one_minus * x_start

    def predict_x0(
            self,
            x_t: Tensor,
            model_output: Tensor,
            noise_levels: Tensor,
    ) -> Tensor:
        """Recover clean x₀ from model output, respecting ``prediction_type``.

        - x₀-prediction:  x₀ = model_output
        - v-prediction:   x₀ = √α̅_t · x_t − √(1−α̅_t) · v_pred
        - ε-prediction:   x₀ = (x_t − √(1−α̅_t) · ε_pred) / √α̅_t
        """
        pt = self.cfg.prediction_type
        if pt == "x0":
            return model_output
        levels = noise_levels.to(device=x_t.device, dtype=torch.long).clamp(
            min=0, max=max(int(self.cfg.diffusion_timesteps) - 1, 0),
        )
        sa = self._df_sqrt_alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)[levels].unsqueeze(-1)
        som = self._df_sqrt_one_minus_alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)[levels].unsqueeze(-1)
        if pt == "v":
            return sa * x_t - som * model_output
        if pt == "eps":
            return (x_t - som * model_output) / sa.clamp(min=1e-8)
        raise ValueError(f"Unknown prediction_type: {pt!r}")

    def predict_eps(
            self,
            x_t: Tensor,
            model_output: Tensor,
            noise_levels: Tensor,
    ) -> Tensor:
        """Recover noise ε from model output, respecting ``prediction_type``."""
        pt = self.cfg.prediction_type
        if pt == "eps":
            return model_output
        levels = noise_levels.to(device=x_t.device, dtype=torch.long).clamp(
            min=0, max=max(int(self.cfg.diffusion_timesteps) - 1, 0),
        )
        sa = self._df_sqrt_alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)[levels].unsqueeze(-1)
        som = self._df_sqrt_one_minus_alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)[levels].unsqueeze(-1)
        if pt == "v":
            return som * x_t + sa * model_output
        if pt == "x0":
            return (x_t - sa * model_output) / som.clamp(min=1e-8)
        raise ValueError(f"Unknown prediction_type: {pt!r}")

    def diffusion_target(
            self,
            x_start: Tensor,
            noise: Tensor,
            noise_levels: Tensor,
    ) -> Tensor:
        """Return the supervision target for the current ``prediction_type``."""
        pt = self.cfg.prediction_type
        if pt == "x0":
            return x_start
        if pt == "eps":
            return noise
        if pt == "v":
            return self.diffusion_v_target(x_start, noise, noise_levels)
        raise ValueError(f"Unknown prediction_type: {pt!r}")

    def _safe_normalize(self, v: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
        """Numerically safe normalization that prevents NaN gradients.

        F.normalize with default eps=1e-12 can produce exploding gradients
        when input norm approaches zero (especially in bfloat16). We use a
        larger eps and clamp the norm to prevent this.  We also scrub any
        non-finite values from the input so a poisoned row cannot leak NaN
        through the downstream division (see tasks/lessons.md 2026-04-11
        rule #3 for the NaN*0 propagation trap).
        """
        v_float = v.float()
        v_float = torch.nan_to_num(v_float, nan=0.0, posinf=0.0, neginf=0.0)
        norms = v_float.norm(dim=dim, keepdim=True).clamp(min=eps)
        return (v_float / norms).to(dtype=v.dtype)

    def _sphere_project(self, v: Tensor) -> Tensor:
        """Project ``v`` onto the SONAR hypersphere of radius ``target_norm``.

        Critically, the output is **guaranteed** to lie on the sphere — even
        when the input is non-finite or zero-norm.  Without this guarantee,
        ``_safe_normalize`` would produce a zero vector for a degenerate row
        (NaN → nan_to_num(0) → norm clamped to eps → 0/eps = 0), and that
        off-manifold zero would feed back as the next step's history,
        skewing attention statistics and triggering a NaN cascade across
        the chain.  We detect collapsed rows and substitute the canonical
        e₀ basis vector scaled to ``target_norm``.
        """
        v_safe = self._safe_normalize(v, dim=-1)  # unit-norm or ~0 for degenerate rows
        # Detect rows that collapsed to ~0 (norm < 0.5 means we lost the unit
        # constraint — under normal conditions safe_normalize returns norm=1).
        with torch.no_grad():
            row_norm = v_safe.float().norm(dim=-1, keepdim=True)
            collapsed = row_norm < 0.5  # broadcast over last dim
        if collapsed.any():
            fallback = torch.zeros_like(v_safe)
            fallback[..., 0] = 1.0  # canonical e₀ direction, on the unit sphere
            v_safe = torch.where(collapsed, fallback, v_safe)
        return v_safe * self.cfg.target_norm

    def _ste_sphere_project(self, v: Tensor) -> Tensor:
        """Straight-Through Estimator sphere projection for BPTT.

        Forward: identical to :meth:`_sphere_project` (output on the SONAR
        hypersphere at exactly ``target_norm``).
        Backward: identity Jacobian — gradient flows through ``v`` directly,
        bypassing the normalization Jacobian (gain = target_norm/||v||) that
        causes exponential gradient vanishing over multi-step chains.

        This ensures the BPTT chain sees the SAME input distribution as eval
        (hard sphere projection), eliminating the train/eval distribution
        mismatch that made the previous soft-projection BPTT ineffective.
        """
        v_clean = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        hard = self._sphere_project(v_clean)
        return v_clean + (hard - v_clean).detach()

    def _to_residual_space(self, sonar_vectors: Tensor) -> Tensor:
        """
        SONAR embeddings have norm ~0.2 (variance ~4e-5). If fed directly into 
        the residual stream or cross-attn, they are completely obliterated by 
        LayerNorm-scaled self-attention updates (norm ~32, variance ~1.0).
        This scales them up to naturally match standard Transformer variance.
        """
        import math
        scale = math.sqrt(self.cfg.d_model) / self.cfg.target_norm
        return sonar_vectors * scale

    def _prepare_context(
        self,
        v_query: Tensor,
        v_context_bank: Tensor | None,
        context_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        bsz, d_model = v_query.shape

        if v_context_bank is None:
            context = v_query.unsqueeze(1)
            context = self._to_residual_space(context)
            mask = torch.ones((bsz, 1), device=v_query.device, dtype=torch.bool)
            return context, mask

        context = v_context_bank
        context = self._to_residual_space(context)
        if context.dim() == 2:
            context = context.unsqueeze(1)
        if context.dim() != 3:
            raise ValueError(f"v_context_bank must have shape [B,K,D] or [B,D], got {tuple(context.shape)}")
        if context.shape[0] != bsz or context.shape[2] != d_model:
            raise ValueError(
                f"v_context_bank shape {tuple(context.shape)} incompatible with v_query {(bsz, d_model)}"
            )

        if context_mask is None:
            mask = torch.ones((bsz, context.shape[1]), device=v_query.device, dtype=torch.bool)
        else:
            mask = context_mask.to(device=v_query.device)
            if mask.shape != (bsz, context.shape[1]):
                raise ValueError(
                    f"context_mask shape {tuple(mask.shape)} does not match context shape {(bsz, context.shape[1])}"
                )
            mask = mask > 0
        return context, mask

    def forward(
        self,
        v_query: Tensor,
        v_target_chain: Tensor,
        v_context_bank: Tensor | None = None,
        context_mask: Tensor | None = None,
        scheduled_sampling_prob: float = 0.0,
        tf_noise_std: float = 0.0,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Teacher-forced forward pass with optional scheduled sampling.

        When scheduled_sampling_prob > 0 and training, each position (except
        the first) independently uses the model's own prediction instead of
        ground truth with probability ``scheduled_sampling_prob``.
        This bridges the teacher-forcing / free-run distribution gap.

        Noisy teacher forcing (tf_noise_std > 0, training only):
        Adds Gaussian noise to the teacher-forced prefix, scaled per-sample
        from U[0, tf_noise_std]. This is the continuous-space analogue of
        training a diffusion model at multiple noise levels — the model
        learns to predict correctly from imperfect contexts, reducing
        autoregressive error accumulation at eval.  The noise is in SONAR
        space (pre-residual-scaling), so tf_noise_std=0.01 corresponds to
        perturbation relative to target_norm≈0.2051 (~5% relative).
        """
        bsz, num_steps, _ = v_target_chain.shape
        context, ctx_mask = self._prepare_context(v_query, v_context_bank, context_mask)

        ss_prob = float(scheduled_sampling_prob)
        use_ss = self.training and ss_prob > 0.0 and num_steps > 1

        if not use_ss:
            # Pure teacher forcing (original path).
            start = self.start_token.expand(bsz, -1, -1)
            scaled_target = self._to_residual_space(v_target_chain)

            # Noisy teacher forcing: add per-sample scaled noise to the
            # prefix context.  Noise is added in SONAR space before residual
            # scaling so that tf_noise_std is interpretable relative to
            # target_norm.  Each sample gets a uniformly random noise level
            # in [0, tf_noise_std], simulating diffusion-style multi-level
            # training.
            if self.training and tf_noise_std > 0.0 and num_steps > 1:
                prefix_sonar = v_target_chain[:, :-1, :]  # [B, T-1, D]
                # Per-sample noise level ~ U[0, tf_noise_std]
                sigma = torch.rand(bsz, 1, 1, device=prefix_sonar.device) * tf_noise_std
                noisy_prefix = prefix_sonar + sigma * torch.randn_like(prefix_sonar)
                scaled_prefix = self._to_residual_space(noisy_prefix)
                decoder_input = torch.cat([start, scaled_prefix], dim=1)
            else:
                decoder_input = torch.cat([start, scaled_target[:, :-1, :]], dim=1)

            x = decoder_input
            aux: dict[str, Tensor] | None = {} if return_aux and len(self.aux_heads) > 0 else None
            for layer_idx, layer in enumerate(self.layers, start=1):
                x = layer(x, context, context_mask=ctx_mask)
                self._maybe_aux_predict(layer_idx, x, aux)

            x = self.final_norm(x)
            v_pred = self.output_proj(x)

            # ── NaN gate for teacher-forcing path ─────────────────
            # Mirror the scheduled-sampling NaN gate (line ~975).
            # Replace non-finite predictions with ground truth so
            # the loss sees 0 (pred == target) instead of the
            # nan_to_num(0) artifact that gives cos_loss=1.0 with
            # zero gradient — the root cause of the NaN cascade
            # where the unstable parameter region silently expands
            # until every output is NaN (lesson 2026-04-16).
            if self.training:
                bad = ~torch.isfinite(v_pred)
                if bad.any():
                    # Count samples (not elements) with any NaN.
                    n_bad = int(bad.any(dim=-1).any(dim=-1).sum().item())
                    self._nan_gate_count = n_bad
                    v_pred = torch.where(bad, v_target_chain, v_pred)
                else:
                    self._nan_gate_count = 0
            else:
                self._nan_gate_count = 0

            if aux is not None:
                return v_pred, aux
            return v_pred

        # ── Scheduled sampling: step-by-step with token mixing ──
        self._nan_gate_count = 0  # reset before rollout
        seq = self.start_token.expand(bsz, -1, -1)  # [B, 1, D]
        preds: list[Tensor] = []
        use_tf_noise = self.training and tf_noise_std > 0.0

        aux_steps: dict[str, list[Tensor]] | None = (
            {str(layer_num): [] for layer_num in self._aux_layer_numbers}
            if return_aux and len(self.aux_heads) > 0
            else None
        )

        for t in range(num_steps):
            x = seq
            for layer_idx, layer in enumerate(self.layers, start=1):
                x = layer(x, context, context_mask=ctx_mask)
                if aux_steps is not None and str(layer_idx) in aux_steps:
                    aux_steps[str(layer_idx)].append(self.aux_heads[str(layer_idx)](x[:, -1:, :]))

            x = self.final_norm(x)
            raw = self.output_proj(x[:, -1:, :])  # [B, 1, D]

            # ── Per-sample NaN gate ──────────────────────────────────
            # If any dimension is non-finite for a sample, replace that
            # sample's raw output with ground truth.  This prevents a
            # corrupted prediction from being fed back as context in
            # subsequent rollout steps.  The loss sees GT for that
            # sample (≈ zero loss, correct: no training signal from a
            # broken forward pass).  Detach-free: GT has no grad path.
            has_bad = ~torch.isfinite(raw).all(dim=-1, keepdim=True)  # [B, 1, 1]
            if has_bad.any():
                n_bad = int(has_bad.sum().item())
                self._nan_gate_count = getattr(self, "_nan_gate_count", 0) + n_bad
                gt_t = v_target_chain[:, t : t + 1, :]
                raw = torch.where(has_bad.expand_as(raw), gt_t, raw)

            pred_t = self._sphere_project(raw)
            preds.append(raw)

            if t < num_steps - 1:
                # Decide per-sample: use own prediction or ground truth.
                rand_vals = torch.rand(bsz, 1, 1, device=x.device)
                use_pred = rand_vals < ss_prob

                # GT token — optionally noised (diffusion-inspired).
                gt_sonar = v_target_chain[:, t : t + 1, :]
                if use_tf_noise:
                    sigma = torch.rand(bsz, 1, 1, device=x.device) * tf_noise_std
                    gt_sonar = gt_sonar + sigma * torch.randn_like(gt_sonar)
                scaled_gt = self._to_residual_space(gt_sonar)
                # DETACH the prediction being used as context to prevent recursive BPTT
                # across Transformer layers!
                scaled_noisy_pred = self._to_residual_space(pred_t).detach()
                next_vec = torch.where(use_pred, scaled_noisy_pred, scaled_gt)

                seq = torch.cat([seq, next_vec], dim=1)
        v_pred = torch.cat(preds, dim=1)
        if aux_steps is not None:
            aux = {k: torch.cat(v, dim=1) for k, v in aux_steps.items() if v}
            return v_pred, aux
        return v_pred

    def diffusion_snr(self, noise_levels: Tensor) -> Tensor:
        """Return schedule SNR for each noise level."""
        levels = noise_levels.to(dtype=torch.long, device=self._df_snr.device).clamp(
            min=0,
            max=max(int(self.cfg.diffusion_timesteps) - 1, 0),
        )
        return self._df_snr[levels].to(device=noise_levels.device)

    def _apply_cfg_dropout(
            self,
            context: Tensor,
            ctx_mask: Tensor | None,
            bsz: int,
    ) -> tuple[Tensor, Tensor | None]:
        """Replace context with null token for random samples (CFG training).

        During training, each sample in the batch independently has its context
        replaced with the learnable ``null_context_token`` with probability
        ``cfg_dropout_prob``.  This teaches the model to generate without
        context so that at inference classifier-free guidance can interpolate
        between conditional and unconditional predictions.
        """
        p = float(self.cfg.cfg_dropout_prob)
        if not self.training or p <= 0.0:
            return context, ctx_mask
        drop = torch.rand(bsz, 1, 1, device=context.device) < p  # [B, 1, 1]
        null_ctx = self.null_context_token.expand(bsz, context.shape[1], -1)
        null_ctx = self._to_residual_space(null_ctx)
        context = torch.where(drop, null_ctx, context)
        # Null context is always "valid" — keep mask unchanged.
        return context, ctx_mask

    def forward_diffusion_forcing(
            self,
            v_query: Tensor,
            v_target_chain: Tensor,
            noise_levels: Tensor,
            v_context_bank: Tensor | None = None,
            context_mask: Tensor | None = None,
            noise: Tensor | None = None,
            return_noisy: bool = False,
            return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, dict[str, Tensor]] | tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Diffusion Forcing denoising pass over the whole chain.

        Each valid position receives its own diffusion noise level.  Unlike
        teacher forcing, the noised token at position i is fed at position i
        and the causal mask prevents future leakage.  This trains the decoder
        to repair partially corrupted self-generated prefixes instead of only
        one clean shifted prefix.

        With ``norm_type="ada_rmsnorm"``, the timestep embedding is injected
        via AdaLN conditioning in every DecoderBlock, instead of a simple
        addition to the input.  This provides richer per-layer modulation.
        """
        bsz, num_steps, d_model = v_target_chain.shape
        if noise_levels.shape != (bsz, num_steps):
            raise ValueError(
                f"noise_levels shape {tuple(noise_levels.shape)} does not match {(bsz, num_steps)}"
            )
        if d_model != self.cfg.d_model:
            raise ValueError(f"target dim {d_model} != config d_model {self.cfg.d_model}")

        context, ctx_mask = self._prepare_context(v_query, v_context_bank, context_mask)
        # CFG training: randomly drop context for some samples.
        context, ctx_mask = self._apply_cfg_dropout(context, ctx_mask, bsz)

        v_noisy, eps = self.diffusion_q_sample(v_target_chain, noise_levels, noise=noise)

        x = self._to_residual_space(v_noisy)
        t_emb = self._diffusion_timestep_embedding(noise_levels).to(device=x.device, dtype=x.dtype)

        aux: dict[str, Tensor] | None = {} if return_aux and len(self.aux_heads) > 0 else None

        if self.cfg.norm_type == "ada_rmsnorm":
            # AdaLN path: per-layer modulation only (DiT design).
            for layer_idx, layer in enumerate(self.layers, start=1):
                x = layer(x, context, context_mask=ctx_mask, t_emb=t_emb)
                self._maybe_aux_predict(layer_idx, x, aux)
        else:
            # Legacy: simple additive conditioning.
            x = x + t_emb
            for layer_idx, layer in enumerate(self.layers, start=1):
                x = layer(x, context, context_mask=ctx_mask)
                self._maybe_aux_predict(layer_idx, x, aux)

        x = self.final_norm(x)
        v_pred = self.output_proj(x)
        if return_noisy and aux is not None:
            return v_pred, v_noisy, eps, aux
        if return_noisy:
            return v_pred, v_noisy, eps
        if aux is not None:
            return v_pred, aux
        return v_pred

    def generate(
        self,
        v_query: Tensor,
        num_steps: int = 1,
        v_context_bank: Tensor | None = None,
        context_mask: Tensor | None = None,
        temperature: float = 1.0,
        latent_noise_std: float = 0.0,
        start_noise_std: float = 0.0,
        repeat_penalty: float = 0.0,
        repeat_cos_threshold: float = 0.98,
        repeat_ban_threshold: float = 0.995,
        repeat_ban_max_retries: int = 4,
        energy_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        target_vec: Tensor | None = None,
        stagnation_patience: int = 0,
        stagnation_delta_energy: float = 1e-4,
        stagnation_delta_cos: float = 1e-4,
        convergence_cos: float = 0.0,
        convergence_window: int = 2,
        chain_prefix: Tensor | None = None,
        return_info: bool = False,
        num_candidates: int = 1,
        oracle_guide: Tensor | None = None,
        oracle_max_retries: int = 0,
        oracle_prob: float = 0.0,
        ddim_steps: int = 0,
        cfg_scale: float = 1.0,
    ) -> Tensor | tuple[Tensor, dict[str, float | int | bool]]:
        """
        Autoregressive generation with adaptive stopping and resume support.

        Stopping modes (checked in order):
          1. convergence_cos > 0: stop when the last `convergence_window`
             consecutive outputs all have pairwise cosine > convergence_cos.
             This is the SONAR-space equivalent of EOS — the model has
             converged to its answer and keeps repeating it.
          2. stagnation_patience > 0: stop when energy/cosine metrics are flat.
          3. num_steps reached (hard cap).

        Resume mode:
          If chain_prefix is provided [B, P, D], generation resumes from that
          chain instead of starting from start_token. This enables the
          "append final vector and keep going" workflow.

        Returns:
            chain [B, T, D] or (chain, info) when return_info=True.
        """
        bsz, _ = v_query.shape
        context, ctx_mask = self._prepare_context(v_query, v_context_bank, context_mask)

        temp = max(float(temperature), 1e-4)
        latent_noise_std = max(float(latent_noise_std), 0.0)
        start_noise_std = max(float(start_noise_std), 0.0)
        repeat_penalty = max(float(repeat_penalty), 0.0)
        repeat_cos_threshold = float(repeat_cos_threshold)
        repeat_ban_threshold = float(repeat_ban_threshold)
        repeat_ban_max_retries = max(int(repeat_ban_max_retries), 0)
        stagnation_patience = max(int(stagnation_patience), 0)
        convergence_cos = float(convergence_cos)
        convergence_window = max(2, int(convergence_window))

        # Resume from prefix or start fresh.
        if chain_prefix is not None:
            if chain_prefix.dim() == 2:
                chain_prefix = chain_prefix.unsqueeze(0)  # [1, P, D]
            if chain_prefix.shape[0] == 1 and bsz > 1:
                chain_prefix = chain_prefix.expand(bsz, -1, -1)
            start = self.start_token.expand(bsz, -1, -1)
            chain = torch.cat([start, chain_prefix], dim=1)
            # Pre-seed generated list with prefix vectors for anti-loop checks.
            generated: list[Tensor] = [
                chain_prefix[:, i : i + 1, :] for i in range(chain_prefix.shape[1])
            ]
        else:
            chain = self.start_token.expand(bsz, -1, -1)
            if start_noise_std > 0.0:
                chain = chain + start_noise_std * torch.randn_like(chain)
            generated = []

        energy_hist: list[float] = []
        cos_hist: list[float] = []
        raw_norms: list[float] = []
        early_stop = False
        early_stop_step = -1
        early_stop_reason = ""
        repeat_resamples = 0

        t_target = None
        if target_vec is not None:
            t_target = target_vec
            if t_target.dim() == 1:
                t_target = t_target.unsqueeze(0)
            t_target = t_target.to(device=v_query.device)

        ddim_steps = max(0, int(ddim_steps))
        cfg_scale = max(1.0, float(cfg_scale))

        steps = max(1, int(num_steps))
        for step_idx in range(steps):
            # ── DDIM multi-step refinement mode ──
            if ddim_steps > 0 and not self.training:
                ddim_result = self._ddim_denoise_position(
                    chain, context, ctx_mask,
                    ddim_steps=ddim_steps,
                    cfg_scale=cfg_scale,
                )
                # ddim_result is [B, 1, D] in SONAR space, already sphere-projected.
                raw_next = ddim_result
            else:
                x = chain
                for layer in self.layers:
                    x = layer(x, context, context_mask=ctx_mask)

                x = self.final_norm(x)
                raw_next = self.output_proj(x[:, -1:, :])

            raw_norms.append(float(raw_next.detach().norm(dim=-1).mean().item()))

            # Repeat penalty: ONLY at inference. During training, this pushes
            # raw_next toward zero norm, causing F.normalize gradient explosion
            # in bfloat16 (the primary NaN collapse trigger at E13+).
            # The penalty provides no useful gradient (history is detached), and
            # Diversity must be handled by scheduled sampling/noise in training.
            if generated and repeat_penalty > 0.0 and not self.training:
                hist = torch.cat(generated, dim=1).detach()
                cos_hist_tensor = (
                    self._safe_normalize(raw_next.expand(-1, hist.shape[1], -1), dim=-1)
                    * self._safe_normalize(hist, dim=-1)
                ).sum(dim=-1)
                max_cos, max_idx = cos_hist_tensor.max(dim=1)
                over = (max_cos - repeat_cos_threshold).clamp(min=0.0)
                if torch.any(over > 0):
                    repel = hist[torch.arange(bsz, device=hist.device), max_idx].unsqueeze(1)
                    raw_next = raw_next - repeat_penalty * over.view(bsz, 1, 1) * repel

            # Scale noise by temperature: higher temp → more noise, lower temp → less.
            noise_std = latent_noise_std * temp

            clean_next_vec = self._sphere_project(raw_next)

            # Oracle-guided DAgger with probability decay.
            # oracle_prob < 1.0 means some samples skip oracle guidance entirely,
            # forcing the model to learn robust generation without oracle dependency.
            # This prevents val roll_cos_last degradation caused by train-only
            # oracle reliance (oracle is disabled at eval since self.training=False).
            import random as _random
            use_oracle = (
                oracle_guide is not None
                and oracle_max_retries > 0
                and self.training
                and noise_std > 0.0
                and _random.random() < oracle_prob
            )
            if use_oracle:
                t_idx = min(step_idx, oracle_guide.shape[1] - 1)
                t_step = oracle_guide[:, t_idx, :]  # [B, D]

                best_cand = clean_next_vec.clone()
                best_cos = (
                    self._safe_normalize(best_cand.squeeze(1), dim=-1)
                    * self._safe_normalize(t_step, dim=-1)
                ).sum(dim=-1)
                jitter_std = max(noise_std, 0.01)

                for _ in range(oracle_max_retries):
                    cand_noisy = raw_next + jitter_std * torch.randn_like(raw_next)
                    cand_proj = self._sphere_project(cand_noisy)
                    cand_cos = (
                        self._safe_normalize(cand_proj.squeeze(1), dim=-1)
                        * self._safe_normalize(t_step, dim=-1)
                    ).sum(dim=-1)

                    improved = cand_cos > best_cos
                    if improved.any():
                        best_cos = torch.where(improved, cand_cos, best_cos)
                        best_cand = torch.where(improved.view(bsz, 1, 1), cand_proj, best_cand)

                next_vec_for_chain = best_cand

            elif num_candidates > 1 and energy_fn is not None and noise_std > 0.0:
                best_noisy_vecs = []
                for b in range(bsz):
                    raw_b = raw_next[b : b + 1]  # [1, 1, D]
                    raw_k = raw_b.expand(int(num_candidates), -1, -1).clone()
                    raw_k_noisy = raw_k + noise_std * torch.randn_like(raw_k)
                    cand_vecs = self._sphere_project(raw_k_noisy)  # [K, 1, D]
                    
                    q_b = v_query[b : b + 1].expand(int(num_candidates), -1)  # [K, D]
                    with torch.no_grad():
                        e_vals = energy_fn(q_b, cand_vecs.squeeze(1))  # [K]
                    
                    best_idx = e_vals.argmin()
                    best_noisy_vecs.append(cand_vecs[best_idx : best_idx + 1])
                
                next_vec_for_chain = torch.cat(best_noisy_vecs, dim=0)
            elif noise_std > 0.0:
                raw_next_noisy = raw_next + noise_std * torch.randn_like(raw_next)
                next_vec_for_chain = self._sphere_project(raw_next_noisy)
            else:
                next_vec_for_chain = clean_next_vec

            # Repeat-ban is inference-only. In training it creates a second
            # train/eval mismatch and adds stochastic resampling to the rollout
            # context while the loss still supervises the clean raw prediction.
            if (
                generated
                and repeat_ban_threshold < 1.0
                and repeat_ban_max_retries > 0
                and not self.training
            ):
                hist = torch.cat(generated, dim=1)
                jitter_std = max(noise_std, 0.01)
                for _ in range(repeat_ban_max_retries):
                    cos_to_hist = (
                        self._safe_normalize(next_vec_for_chain.expand(-1, hist.shape[1], -1), dim=-1)
                        * self._safe_normalize(hist, dim=-1)
                    ).sum(dim=-1)
                    max_cos = cos_to_hist.max(dim=1).values
                    repeat_mask = max_cos > repeat_ban_threshold
                    if not torch.any(repeat_mask):
                        break

                    candidate_noisy = raw_next + jitter_std * torch.randn_like(raw_next)
                    candidate = self._sphere_project(candidate_noisy)
                    mask3 = repeat_mask.view(bsz, 1, 1)
                    next_vec_for_chain = torch.where(mask3, candidate, next_vec_for_chain)
                    repeat_resamples += int(repeat_mask.sum().item())

            # NaN guard: check raw_next BEFORE storing.  _sphere_project
            # silently cleans NaN via nan_to_num(0) inside _safe_normalize,
            # so checking next_vec_for_chain (post-projection) never fires.
            # We must check raw_next (pre-projection) to catch the actual NaN.
            raw_has_nan = not torch.isfinite(raw_next).all()

            # ── Train/eval rollout alignment (lessons 2026-04-26) ──
            # During training we previously stored raw_next (pre-projection)
            # for loss/metrics, while eval stored next_vec_for_chain (post-
            # projection at exact target_norm). This made train roll_cos
            # incomparable to eval roll_cos — the two paths sampled
            # different distributions.
            #
            # Fix: in training, store the STE-projected vector. Forward gives
            # the same distribution as eval (sphere-projected at target_norm),
            # backward flows the gradient through the identity (no Jacobian
            # attenuation from normalization). This unifies train and eval
            # without sacrificing gradient signal.
            if self.training:
                if raw_has_nan:
                    # Store clean sphere-projected vec instead of NaN-contaminated raw.
                    # Downstream loss sees clean_pred ≈ GT direction → loss ≈ 0.
                    generated.append(clean_next_vec)
                else:
                    # STE projection: forward = sphere_project(raw_next),
                    # backward = identity through raw_next.
                    generated.append(self._ste_sphere_project(raw_next))
            else:
                generated.append(next_vec_for_chain)

            if raw_has_nan:
                if len(generated) >= 2:
                    fallback = generated[-2].detach().clone()
                    next_vec_for_chain = self._sphere_project(fallback)
                else:
                    next_vec_for_chain = self._sphere_project(
                        torch.randn(bsz, 1, self.cfg.d_model, device=v_query.device)
                    )
                if not self.training:
                    generated[-1] = next_vec_for_chain

            # Fix #3: Detach before appending so backward() through L_roll
            # only goes one step deep, not through the entire autoregressive chain.
            # Scale the next_vec_for_chain UP to residual space before appending!
            scaled_next = self._to_residual_space(next_vec_for_chain)
            chain = torch.cat([chain, scaled_next.detach()], dim=1)

            # ── Convergence check (SONAR-space EOS) ──
            # If the last W outputs are all mutually similar (cos > threshold),
            # the model has converged to its answer. This is trained via
            # answer-repeat padding: [steps..., answer, answer, answer].
            if convergence_cos > 0 and len(generated) >= convergence_window:
                window = torch.cat(generated[-convergence_window:], dim=1)  # [B, W, D]
                # Check all consecutive pairs in window.
                w1 = window[:, :-1, :]   # [B, W-1, D]
                w2 = window[:, 1:, :]    # [B, W-1, D]
                pair_cos = (self._safe_normalize(w1, dim=-1) * self._safe_normalize(w2, dim=-1)).sum(dim=-1)
                # Converged if ALL pairs exceed threshold (mean across batch).
                min_pair_cos = pair_cos.min(dim=1).values.mean().item()
                if min_pair_cos > convergence_cos:
                    early_stop = True
                    early_stop_step = step_idx + 1
                    early_stop_reason = "convergence"
                    break

            if energy_fn is not None:
                try:
                    e_val = energy_fn(v_query, clean_next_vec.squeeze(1))
                    if torch.is_tensor(e_val):
                        energy_hist.append(float(e_val.detach().mean().item()))
                    else:
                        energy_hist.append(float(e_val))
                except Exception as exc:
                    import warnings
                    if step_idx == 0:
                        warnings.warn(f"energy_fn failed at step 0: {exc}", stacklevel=2)

            if t_target is not None:
                cos_val = (
                    self._safe_normalize(clean_next_vec.squeeze(1), dim=-1)
                    * self._safe_normalize(t_target, dim=-1)
                ).sum(dim=-1)
                cos_hist.append(float(cos_val.mean().item()))

            if stagnation_patience > 0 and step_idx + 1 >= (stagnation_patience + 1):
                checks = []
                if len(energy_hist) > stagnation_patience:
                    de = abs(energy_hist[-1] - energy_hist[-1 - stagnation_patience])
                    checks.append(de <= stagnation_delta_energy)
                if len(cos_hist) > stagnation_patience:
                    dc = abs(cos_hist[-1] - cos_hist[-1 - stagnation_patience])
                    checks.append(dc <= stagnation_delta_cos)
                stagnated = len(checks) > 0 and all(checks)
                if stagnated:
                    early_stop = True
                    early_stop_step = step_idx + 1
                    early_stop_reason = "stagnation"
                    break

        # Only return newly generated steps (exclude prefix).
        prefix_len = chain_prefix.shape[1] if chain_prefix is not None else 0
        new_generated = generated[prefix_len:]
        chain_out = torch.cat(new_generated, dim=1) if new_generated else torch.zeros(
            bsz, 0, v_query.shape[-1], device=v_query.device
        )

        if not return_info:
            return chain_out

        raw_norm_mean = sum(raw_norms) / len(raw_norms) if raw_norms else 0.0

        info: dict[str, float | int | bool | str] = {
            "early_stop": early_stop,
            "early_stop_step": int(early_stop_step),
            "early_stop_reason": early_stop_reason,
            "repeat_resamples": int(repeat_resamples),
            "steps_generated": int(chain_out.shape[1]),
            "prefix_len": int(prefix_len),
            "temperature": float(temp),
            "latent_noise_std": float(latent_noise_std),
            "start_noise_std": float(start_noise_std),
            "repeat_penalty": float(repeat_penalty),
            "raw_norm_mean": float(raw_norm_mean),
        }
        if energy_hist:
            info["energy_final"] = float(energy_hist[-1])
        if cos_hist:
            info["cos_final"] = float(cos_hist[-1])

        return chain_out, info

    # ------------------------------------------------------------------
    # Truncated BPTT chain generation
    # ------------------------------------------------------------------

    def generate_with_bptt(
        self,
        v_query: Tensor,
        num_steps: int,
        bptt_steps: int = 2,
        v_context_bank: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate chain with truncated BPTT via Straight-Through Estimator.

        The first ``num_steps - bptt_steps`` positions are generated with
        :meth:`_sphere_project` + detach (identical to eval :meth:`generate`).
        The last ``bptt_steps`` positions use :meth:`_ste_sphere_project`:
        forward = exact sphere projection (norm = target_norm), backward =
        identity Jacobian (full gradient flow).  This eliminates the
        train/eval distribution mismatch that made soft-projection BPTT
        ineffective.

        Use ``bptt_steps=-1`` to cover all steps (full-chain BPTT).

        Args:
            v_query: ``[B, D]`` query embeddings.
            num_steps: total number of chain steps to generate.
            bptt_steps: trailing steps with gradient flow (-1 = all).
            v_context_bank: ``[B, K, D]`` optional cross-attention context.
            context_mask: ``[B, K]`` mask for context_bank.

        Returns:
            all_preds: ``[B, T, D]`` STE-projected predictions at every
                position (SONAR space, norm = target_norm).
            bptt_preds: ``[B, K, D]`` BPTT-window predictions with
                through-chain gradient (STE-projected).
            bptt_raw_norms: ``[B, K]`` raw output norms (pre-projection)
                for the norm penalty that keeps STE accurate.
        """
        bsz, _ = v_query.shape
        context, ctx_mask = self._prepare_context(
            v_query, v_context_bank, context_mask,
        )

        steps = max(1, int(num_steps))
        bptt_k = steps if int(bptt_steps) < 0 else max(0, min(int(bptt_steps), steps))
        prefix_steps = steps - bptt_k

        chain = self.start_token.expand(bsz, -1, -1)
        all_preds: list[Tensor] = []
        bptt_nan_count = 0

        # Phase 1: Prefix — detached, identical to eval generate().
        for _ in range(prefix_steps):
            x = chain
            for layer in self.layers:
                x = layer(x, context, context_mask=ctx_mask)
            x = self.final_norm(x)
            raw = self.output_proj(x[:, -1:, :])

            proj = self._sphere_project(raw)
            all_preds.append(proj)
            chain = torch.cat(
                [chain, self._to_residual_space(proj).detach()], dim=1,
            )

        # Phase 2: BPTT window — STE projection, gradient flows.
        bptt_pred_list: list[Tensor] = []
        bptt_raw_norm_list: list[Tensor] = []
        for _ in range(bptt_k):
            x = chain
            for layer in self.layers:
                x = layer(x, context, context_mask=ctx_mask)
            x = self.final_norm(x)
            raw = self.output_proj(x[:, -1:, :])

            nan_detected = not torch.isfinite(raw).all()
            if nan_detected:
                bptt_nan_count += 1

            ste = self._ste_sphere_project(raw)
            all_preds.append(ste)
            bptt_pred_list.append(ste)

            raw_clean = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            bptt_raw_norm_list.append(raw_clean.squeeze(1).norm(dim=-1))

            scaled = self._to_residual_space(ste)
            if nan_detected:
                chain = torch.cat([chain, scaled.detach()], dim=1)
            else:
                chain = torch.cat([chain, scaled], dim=1)

        self._bptt_nan_gate_count = bptt_nan_count

        if not bptt_pred_list:
            empty_preds = v_query.new_zeros(bsz, 0, self.cfg.d_model)
            empty_norms = v_query.new_zeros(bsz, 0)
            all_predictions = (
                torch.cat(all_preds, dim=1) if all_preds
                else v_query.new_zeros(bsz, 0, self.cfg.d_model)
            )
            return all_predictions, empty_preds, empty_norms

        all_predictions = torch.cat(all_preds, dim=1)
        bptt_predictions = torch.cat(bptt_pred_list, dim=1)
        bptt_raw_norms = torch.stack(bptt_raw_norm_list, dim=1)

        return all_predictions, bptt_predictions, bptt_raw_norms

    # ------------------------------------------------------------------
    # DDIM iterative refinement at each autoregressive position
    # ------------------------------------------------------------------

    def _ddim_denoise_position(
            self,
            chain_residual: Tensor,
            context: Tensor,
            ctx_mask: Tensor | None,
            ddim_steps: int = 3,
            cfg_scale: float = 1.0,
    ) -> Tensor:
        """Refine the next chain position via DDIM multi-step denoising.

        Instead of one-shot prediction, this starts from pure noise and
        iteratively denoises through ``ddim_steps`` diffusion steps, using
        the model's DF pathway.  The chain prefix is treated as clean (t=0).

        Args:
            chain_residual: [B, L, D] existing chain in residual space
                            (start_token + generated so far).
            context: [B, K, D] cross-attention context (residual space).
            ctx_mask: [B, K] context mask.
            ddim_steps: number of DDIM denoising steps.
            cfg_scale: classifier-free guidance scale (1.0 = no guidance).

        Returns:
            x0_pred: [B, 1, D] predicted clean vector in **SONAR** space.
        """
        bsz = chain_residual.shape[0]
        K = max(2, int(self.cfg.diffusion_timesteps))

        # Build step schedule: evenly spaced from t_max down to 0.
        # E.g. ddim_steps=3, K=64 → [63, 42, 21, 0]
        schedule = torch.linspace(K - 1, 0, ddim_steps + 1).long().tolist()

        # Start from pure SONAR-scaled noise for the new position.
        noise_scale = self._diffusion_noise_scale()
        x_t = noise_scale * torch.randn(
            bsz, 1, self.cfg.d_model,
            device=chain_residual.device,
            dtype=chain_residual.dtype,
        )

        for i in range(ddim_steps):
            t_cur = schedule[i]
            t_next = schedule[i + 1]

            # Build levels: 0 for clean prefix, t_cur for the new position.
            prefix_len = chain_residual.shape[1]
            levels_prefix = torch.zeros(bsz, prefix_len, device=chain_residual.device, dtype=torch.long)
            levels_new = torch.full((bsz, 1), t_cur, device=chain_residual.device, dtype=torch.long)
            levels = torch.cat([levels_prefix, levels_new], dim=1)  # [B, L+1]

            # Concatenate prefix + noisy new position.
            x_new_res = self._to_residual_space(x_t)
            full_seq = torch.cat([chain_residual, x_new_res], dim=1)  # [B, L+1, D]

            # Timestep embedding for the full sequence.
            t_emb = self._diffusion_timestep_embedding(levels).to(
                device=full_seq.device, dtype=full_seq.dtype,
            )

            # Forward through transformer with timestep conditioning.
            if self.cfg.norm_type == "ada_rmsnorm":
                x = full_seq + t_emb
                for layer in self.layers:
                    x = layer(x, context, context_mask=ctx_mask, t_emb=t_emb)
            else:
                x = full_seq + t_emb
                for layer in self.layers:
                    x = layer(x, context, context_mask=ctx_mask)

            x = self.final_norm(x)
            model_out = self.output_proj(x[:, -1:, :])  # [B, 1, D]

            # CFG: conditional + unconditional interpolation (at the prediction,
            # not x0 — mathematically equivalent for linear prediction types).
            if cfg_scale > 1.0 and hasattr(self, "null_context_token"):
                null_ctx = self.null_context_token.expand(bsz, context.shape[1], -1)
                null_ctx = self._to_residual_space(null_ctx)
                if self.cfg.norm_type == "ada_rmsnorm":
                    x_uc = full_seq + t_emb
                    for layer in self.layers:
                        x_uc = layer(x_uc, null_ctx, context_mask=ctx_mask, t_emb=t_emb)
                else:
                    x_uc = full_seq + t_emb
                    for layer in self.layers:
                        x_uc = layer(x_uc, null_ctx, context_mask=ctx_mask)
                x_uc = self.final_norm(x_uc)
                model_out_uc = self.output_proj(x_uc[:, -1:, :])
                model_out = model_out_uc + cfg_scale * (model_out - model_out_uc)

            # Recover x₀ and ε from model output.
            level_new = levels_new  # [B, 1]
            x0_pred = self.predict_x0(x_t, model_out, level_new)
            eps_pred = self.predict_eps(x_t, model_out, level_new)

            # DDIM deterministic step: x_{t'} = √α̅_{t'} · x₀ + √(1-α̅_{t'}) · ε
            if t_next > 0:
                sa_next = self._df_sqrt_alphas_cumprod[t_next].to(
                    device=x_t.device, dtype=x_t.dtype,
                )
                som_next = self._df_sqrt_one_minus_alphas_cumprod[t_next].to(
                    device=x_t.device, dtype=x_t.dtype,
                )
                x_t = sa_next * x0_pred + som_next * eps_pred
            else:
                x_t = x0_pred

        # Project to SONAR sphere.
        return self._sphere_project(x_t)

    def beam_generate(
        self,
        v_query: Tensor,
        num_steps: int = 1,
        beam_width: int = 4,
        num_candidates: int = 4,
        v_context_bank: Tensor | None = None,
        context_mask: Tensor | None = None,
        temperature: float = 1.0,
        noise_std: float = 0.05,
        energy_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> Tensor:
        """
        Energy-Guided Beam Search for Autoregressive Generation.

        Maintains `beam_width` active reasoning trajectories. At each step, expands
        each trajectory with `num_candidates` noisy extensions, evaluates them
        using `energy_fn`, and prunes back to `beam_width` by cumulative energy.
        """
        bsz, _ = v_query.shape
        if energy_fn is None:
            raise ValueError("beam_generate requires energy_fn to score paths")

        context, ctx_mask = self._prepare_context(v_query, v_context_bank, context_mask)
        temp = max(float(temperature), 1e-4)
        n_std = max(float(noise_std) * temp, 0.0)

        # active_chains: [B, W, t, D]. Starts at t=1 (start_token only)
        active_chains = self.start_token.expand(bsz, 1, 1, -1)  # [B, 1, 1, D]
        active_energies = torch.zeros(bsz, 1, device=v_query.device)  # [B, 1]
        
        W = 1
        steps = max(1, int(num_steps))
        K = max(1, int(num_candidates))
        
        for step_idx in range(steps):
            B_W = bsz * W
            flat_chains = active_chains.reshape(B_W, -1, self.cfg.d_model)
            
            flat_context = context.repeat_interleave(W, dim=0)
            flat_ctx_mask = ctx_mask.repeat_interleave(W, dim=0)

            x = flat_chains
            for layer in self.layers:
                x = layer(x, flat_context, context_mask=flat_ctx_mask)

            x = self.final_norm(x)
            raw_next = self.output_proj(x[:, -1:, :])  # [B*W, 1, D]

            new_W = W * K
            raw_k = raw_next.reshape(bsz, W, 1, self.cfg.d_model).unsqueeze(2).expand(-1, -1, K, -1, -1).clone()

            if n_std > 0.0:
                raw_k_noisy = raw_k + n_std * torch.randn_like(raw_k)
            else:
                raw_k_noisy = raw_k

            # Sphere-project to SONAR space for energy scoring and output.
            cand_vecs = self._sphere_project(raw_k_noisy)  # [B, W, K, 1, D] — SONAR scale

            flat_cands = cand_vecs.reshape(bsz * new_W, self.cfg.d_model)
            flat_qs = v_query.repeat_interleave(new_W, dim=0)

            with torch.no_grad():
                step_energies = energy_fn(flat_qs, flat_cands)  # [B*W*K]

            step_energies = step_energies.reshape(bsz, W, K)

            # Cumulative energy over the chain path
            cum_energies = active_energies.unsqueeze(2) + step_energies  # [B, W, K]
            cum_energies_flat = cum_energies.reshape(bsz, new_W)

            next_W = min(int(beam_width), new_W)
            top_energies, top_indices = torch.topk(cum_energies_flat, next_W, dim=1, largest=False)  # [B, next_W]

            # Scale candidates to residual space BEFORE appending to the chain.
            # The chain is built in residual space (norm~32). SONAR-scale vectors
            # (norm~0.2051) would be 156x smaller than the start_token and produce
            # degenerate attention weights across beam steps.
            cand_vecs_res = self._to_residual_space(cand_vecs)  # [B, W, K, 1, D] — residual scale
            reconstructed_cands = cand_vecs_res.reshape(bsz, new_W, 1, self.cfg.d_model)
            t = active_chains.shape[2]
            history = active_chains.unsqueeze(2).expand(-1, -1, K, -1, -1).reshape(bsz, new_W, t, self.cfg.d_model)

            next_chains = []
            for b in range(bsz):
                idx = top_indices[b]
                b_hist = history[b, idx]  # [next_W, t, D]
                b_cand = reconstructed_cands[b, idx]  # [next_W, 1, D]
                b_new = torch.cat([b_hist, b_cand], dim=1)  # [next_W, t+1, D]
                next_chains.append(b_new)
                
            active_chains = torch.stack(next_chains, dim=0)  # [B, next_W, t+1, D]
            active_energies = top_energies
            W = next_W
            
        # Return best chain per batch item (idx 0), excluding start_token (pos 0)
        best_chains = active_chains[:, 0, 1:, :]  # [B, num_steps, D]
        return best_chains

    def compute_loss(
        self,
        v_query: Tensor,
        v_target_chain: Tensor,
        loss_mask: Tensor | None = None,
        v_context_bank: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Masked teacher-forcing loss."""
        v_pred = self.forward(
            v_query,
            v_target_chain,
            v_context_bank=v_context_bank,
            context_mask=context_mask,
        )

        bsz, num_steps, _ = v_pred.shape
        if loss_mask is None:
            mask = torch.ones(bsz, num_steps, device=v_pred.device, dtype=v_pred.dtype)
        else:
            mask = loss_mask.to(device=v_pred.device, dtype=v_pred.dtype)
            if mask.shape != (bsz, num_steps):
                raise ValueError(
                    f"loss_mask shape {tuple(mask.shape)} does not match (B,N)=({bsz},{num_steps})"
                )

        mask_bool = mask > 0
        mask_sum = mask.sum().clamp(min=1.0)

        v_pred_f = v_pred.float()
        v_target_f = v_target_chain.float()
        cos_sim = (
            self._safe_normalize(v_pred_f, dim=-1)
            * self._safe_normalize(v_target_f, dim=-1)
        ).sum(dim=-1)
        cos_term = torch.where(mask_bool, 1.0 - cos_sim, torch.zeros_like(cos_sim))
        cos_loss = cos_term.sum() / mask_sum

        mse_per_step = (v_pred_f - v_target_f).pow(2).sum(dim=-1)
        mse_term = torch.where(mask_bool, mse_per_step, torch.zeros_like(mse_per_step))
        mse_loss = mse_term.sum() / mask_sum

        loss = self.cfg.loss_cosine_weight * cos_loss + self.cfg.loss_mse_weight * mse_loss

        with torch.no_grad():
            valid_per_step = mask.sum(dim=0).clamp(min=1.0)
            sample_valid_counts = mask.sum(dim=1).long().clamp(min=1)
            last_idx = (sample_valid_counts - 1).clamp(min=0)
            cos_masked = torch.where(mask_bool, cos_sim, torch.zeros_like(cos_sim))
            pred_norm = v_pred.norm(dim=-1)
            pred_norm_masked = torch.where(mask_bool, pred_norm, torch.zeros_like(pred_norm))
            cos_last = cos_sim.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean().item()
            cos_first = cos_sim[:, 0].mean().item()

            metrics = {
                "loss": float(loss.item()),
                "cos_loss": float(cos_loss.item()),
                "mse_loss": float(mse_loss.item()),
                "cos_sim_mean": float((cos_masked.sum() / mask_sum).item()),
                "cos_sim_last": float(cos_last),
                "cos_sim_first": float(cos_first),
                "pred_norm_mean": float((pred_norm_masked.sum() / mask_sum).item()),
                "valid_tokens": float(mask_sum.item()),
                "valid_tokens_per_sample": float(sample_valid_counts.float().mean().item()),
                "cos_step0": float((
                    torch.where(mask_bool[:, 0], cos_sim[:, 0], torch.zeros_like(cos_sim[:, 0])).sum()
                    / mask[:, 0].sum().clamp(min=1.0)
                ).item()),
                "cos_step_last_masked": float(cos_last),
                "cos_step_mean_masked": float((cos_masked.sum() / mask_sum).item()),
                "valid_steps": float(mask_sum.item()),
                "valid_steps_per_sample": float(sample_valid_counts.float().mean().item()),
            }

        return loss, metrics

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
