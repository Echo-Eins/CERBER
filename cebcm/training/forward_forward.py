"""Forward-Forward (FF) training utilities for CERBER chain generator.

Implements the layer-local Forward-Forward algorithm (Hinton 2022) with:
- SymBa loss (Lee & Song 2023) instead of original logistic — no threshold θ,
  balanced gradient on both pos/neg, no saturation at depth.
- Activity normalization (L2-norm rescale) between blocks to prevent
  amplitude explosion while preserving directional information.
- Goodness = per-sample mean squared activation of the block's **full
  hidden state** x_out (Hinton 2022 original), measured on the pre-next-
  layer-norm tensor (Brenig & Timofte 2023) to avoid the norm-kills-signal
  failure.  Using block delta (x_out − x_in) was considered but rejected:
  zero-init residual projectors make δ=0 at init → ∂||δ||²/∂δ = 0
  (vanishing gradient at initialization).
- Negative chain generation with curriculum: gaussian → shuffle → hard-NN.

All goodness / loss computations run in FP32 to avoid bf16 overflow — same
principle as ``_soft_clamp_residual_norm`` in the model.

Reference formulas
------------------
SymBa:   L = log(1 + exp(−α · (g⁺ − g⁻)))           (Lee & Song 2023)
Goodness: g = mean_over_valid_tokens( ||δ_l||² )      per sample
Activity norm:  ĥ = h / (||h||₂ + ε) · √D            Hinton 2022 §3.2
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. Goodness metric
# ---------------------------------------------------------------------------

def compute_goodness(
    block_delta: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Per-sample goodness from a decoder block's residual contribution.

    Args:
        block_delta: [B, L, D]  x_block_out − x_block_in  (the block's
            additive contribution to the residual stream).
        mask: [B, L] bool — True for valid (non-pad) positions.
            ``None`` means all positions valid.

    Returns:
        [B] per-sample goodness  g_i = mean_{valid tokens t} ||δ_{i,t}||²

    Math (FP32):
        ||δ||² per token = sum_d δ_d²      → [B, L]
        masked mean over L                  → [B]
    """
    delta_f = block_delta.float()
    sq_norm = delta_f.pow(2).sum(dim=-1)  # [B, L]

    if mask is not None:
        mask_f = mask.float()
        count = mask_f.sum(dim=-1).clamp(min=1.0)  # [B]
        return (sq_norm * mask_f).sum(dim=-1) / count
    return sq_norm.mean(dim=-1)  # [B]


# ---------------------------------------------------------------------------
# 2. SymBa loss (symmetric, no threshold)
# ---------------------------------------------------------------------------

def symba_loss(
    g_pos: Tensor,
    g_neg: Tensor,
    alpha: float = 2.0,
) -> Tensor:
    """SymBa contrastive loss on goodness values (Lee & Song 2023).

    L = mean_i  log(1 + exp(−α · (g⁺_i − g⁻_i)))

    Properties vs. Hinton's original:
    - No threshold θ to tune.
    - Gradient is balanced: pos and neg get equal-magnitude push regardless
      of which side is already well-separated (fixes saturation).
    - α controls margin sharpness.  α=2 is the recommended default (paper).

    Args:
        g_pos: [B] per-sample goodness on positive data.
        g_neg: [B] per-sample goodness on negative data.
        alpha: margin temperature (≥1 recommended).

    Returns:
        Scalar loss (mean over batch).
    """
    diff = (g_pos.float() - g_neg.float()) * float(alpha)
    return F.softplus(-diff).mean()


# ---------------------------------------------------------------------------
# 3. Activity normalization between blocks
# ---------------------------------------------------------------------------

def activity_normalize(
    h: Tensor,
    target_rms: float | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """L2-normalize then rescale to target RMS per token (Hinton §3.2).

    Strips the magnitude cue (which IS the training signal for the current
    block) so the next block must discover its own features rather than
    amplify the previous block's norm.

    After this:  RMS(ĥ) ≈ 1.0,  ||ĥ||₂ ≈ √D   (healthy for AdaRMSNorm).

    If ``target_rms`` is None, the natural RMS=1 (||ĥ||₂ = √D) is used.

    Math (FP32):
        norm = ||h||₂  per token  (dim=-1)
        ĥ = h / (norm + ε)  ·  scale
        scale = √D  if target_rms is None, else target_rms · √D

    Args:
        h: [B, L, D] hidden states.
        target_rms: desired per-element RMS after normalization.
            ``None`` → RMS=1.0 (standard).
        eps: floor on denominator.

    Returns:
        [B, L, D] normalized tensor, same dtype as input.
    """
    d = h.shape[-1]
    h_f = h.float()
    norm = h_f.norm(dim=-1, keepdim=True).clamp_min(eps)

    if target_rms is None:
        scale = d ** 0.5
    else:
        scale = float(target_rms) * d ** 0.5

    return ((h_f / norm) * scale).to(h.dtype)


# ---------------------------------------------------------------------------
# 4. Negative chain generation
# ---------------------------------------------------------------------------

def generate_negatives_gaussian(
    v_chain: Tensor,
    sigma: float = 0.3,
    mask: Tensor | None = None,
) -> Tensor:
    """Gaussian-perturbed negatives: chain + N(0, σ²).

    σ is relative to per-element std of the chain. Easy negatives —
    good for warmup curriculum stage.

    Args:
        v_chain: [B, L, D] positive chain embeddings.
        sigma: noise std relative to input element-wise std.
        mask: [B, L] bool — if provided, only perturb valid positions.

    Returns:
        [B, L, D] negatives, same device/dtype.
    """
    noise = torch.randn_like(v_chain) * float(sigma) * v_chain.float().std().item()
    if mask is not None:
        noise = noise * mask.unsqueeze(-1).float()
    return v_chain + noise.to(v_chain.dtype)


def generate_negatives_shuffle(
    v_chain: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Shuffle chain step order within each sample (medium negatives).

    Permutes the L dimension independently per sample. Only permutes valid
    (non-pad) positions; padding stays in place.

    Args:
        v_chain: [B, L, D].
        mask: [B, L] bool.

    Returns:
        [B, L, D] with permuted step order.
    """
    B, L, D = v_chain.shape
    out = v_chain.clone()
    for i in range(B):
        if mask is not None:
            valid = int(mask[i].sum().item())
        else:
            valid = L
        if valid <= 1:
            continue
        perm = torch.randperm(valid, device=v_chain.device)
        out[i, :valid] = v_chain[i, perm]
    return out


def generate_negatives_batch_swap(
    v_chain: Tensor,
) -> Tensor:
    """Swap chains between samples in the batch (hard negatives).

    Sample i receives the chain of sample (i+1) % B. This creates
    semantically mismatched (query, chain) pairs — strong negative.

    Args:
        v_chain: [B, L, D].

    Returns:
        [B, L, D] with rolled batch dimension.
    """
    return v_chain.roll(-1, dims=0)


def generate_negatives(
    v_chain: Tensor,
    strategy: str = "shuffle",
    mask: Tensor | None = None,
    sigma: float = 0.3,
) -> Tensor:
    """Dispatch to the requested negative generation strategy.

    Args:
        v_chain: [B, L, D] positive chain.
        strategy: ``"gaussian"`` | ``"shuffle"`` | ``"batch_swap"``.
        mask: [B, L] bool chain mask.
        sigma: noise std for ``"gaussian"`` strategy.

    Returns:
        [B, L, D] negative chain.
    """
    if strategy == "gaussian":
        return generate_negatives_gaussian(v_chain, sigma=sigma, mask=mask)
    if strategy == "shuffle":
        return generate_negatives_shuffle(v_chain, mask=mask)
    if strategy == "batch_swap":
        return generate_negatives_batch_swap(v_chain)
    raise ValueError(f"Unknown negative strategy: {strategy!r}")


def curriculum_strategy(epoch: int, schedule: dict[str, int] | None = None) -> str:
    """Select negative strategy based on epoch (curriculum).

    Default schedule:
        epochs 0-4:   gaussian (easy)
        epochs 5-14:  shuffle (medium)
        epochs 15+:   batch_swap (hard)

    Args:
        epoch: current epoch (0-indexed).
        schedule: optional dict with keys ``"shuffle_start"``, ``"hard_start"``.

    Returns:
        Strategy name string.
    """
    if schedule is None:
        schedule = {}
    shuffle_start = int(schedule.get("shuffle_start", 5))
    hard_start = int(schedule.get("hard_start", 15))
    if epoch >= hard_start:
        return "batch_swap"
    if epoch >= shuffle_start:
        return "shuffle"
    return "gaussian"


# ---------------------------------------------------------------------------
# 5. Layer-local Forward-Forward pass through ChainGenerator
# ---------------------------------------------------------------------------

def ff_forward(
    model,
    v_query: Tensor,
    v_chain: Tensor,
    v_context_bank: Tensor | None = None,
    context_mask: Tensor | None = None,
    chain_mask: Tensor | None = None,
) -> tuple[list[Tensor], Tensor]:
    """Layer-local FF forward: stop_gradient between blocks, collect goodness.

    Each decoder layer receives a **detached** hidden state so its
    parameters are trained only by the local FF loss, not by any
    downstream layer or the output head.

    After every layer the hidden state is **activity-normalized** (L2-norm
    rescale to √D) to strip the magnitude cue — which IS that layer's
    training signal — so the next layer must learn its own features.

    The ``output_proj`` head receives the last detached + final_norm'd
    state and is trained with a standard supervised loss (FFCL hybrid:
    FF on hidden layers, BP on output head only).

    Gradient-flow diagram::

        decoder_input ─detach→ layer0 ─activity_norm─detach→ layer1 ─…─ layerN
                                 │                              │          │
                              goodness[0]                  goodness[1] goodness[N]
                                                                       │
                                                               ─detach→ final_norm → output_proj → v_pred
                                                                           ↑          ↑
                                                                     head_loss gradients only

    Args:
        model: ``ChainGenerator`` instance (used read-only for its layers,
            norms, projections, and ``_prepare_context`` / ``_to_residual_space``).
        v_query: [B, D] query vectors (SONAR-space).
        v_chain: [B, L, D] chain to evaluate (positive or negative).
        v_context_bank: [B, K, D] optional context bank (first slot = v_query
            conventionally).
        context_mask: [B, K] bool mask for context bank.
        chain_mask: [B, L] bool mask — True for valid (non-pad) chain positions.
            Used for goodness aggregation (padded positions excluded).

    Returns:
        goodness_list: ``list[Tensor]`` of length ``n_layers``, each [B].
        v_pred: [B, L, D] output_proj prediction in SONAR space (for FFCL
            supervised head loss via cosine / MSE).
    """
    bsz, num_steps, _ = v_chain.shape

    # ── Prepare context (frozen data path — no learnable params) ──
    context, ctx_mask = model._prepare_context(
        v_query, v_context_bank, context_mask,
    )
    # Detach context to guarantee no gradient leaks across layers through
    # shared cross-attention keys/values.
    context = context.detach()
    if ctx_mask is not None:
        ctx_mask = ctx_mask.detach()

    # ── Build teacher-forced decoder input: [start, chain[:-1]] ──
    start = model.start_token.detach().expand(bsz, -1, -1)
    scaled_chain = model._to_residual_space(v_chain).detach()
    decoder_input = torch.cat([start, scaled_chain[:, :-1, :]], dim=1)

    # ── Layer-local forward with stop_gradient ──
    x = decoder_input  # already detached above
    goodness_list: list[Tensor] = []

    for layer in model.layers:
        x_in = x.detach()  # stop_gradient at block boundary
        x_out = layer(x_in, context, context_mask=ctx_mask)
        # t_emb=None — no diffusion conditioning in FF mode;
        # AdaLN-Zero acts as plain RMSNorm (scale=0 → γ=1).

        # Goodness on FULL hidden state (Hinton 2022 original), NOT delta.
        # Why not delta: zero-init residual → δ = 0 → ∂||δ||²/∂δ = 2δ = 0 →
        # vanishing gradient at init.  Full-state goodness has ∂g/∂x_out = 2·x_out ≠ 0
        # because x_out ≈ x_in ≠ 0 (contains chain content).
        # Measuring on pre-next-layer-norm state (Brenig & Timofte 2023) avoids
        # the "norm kills signal" failure — AdaRMSNorm hasn't been applied yet.
        g = compute_goodness(x_out, mask=chain_mask)
        goodness_list.append(g)

        # Activity normalize: strip magnitude cue for next layer.
        x = activity_normalize(x_out)

    # ── FFCL head: final_norm + output_proj (BP-trained) ──
    # Detach from FF layers so head loss doesn't flow into them.
    x_head = x.detach()
    x_normed = model.final_norm(x_head)
    v_pred = model.output_proj(x_normed)

    return goodness_list, v_pred


def ff_compute_losses(
    model,
    v_query: Tensor,
    v_chain_pos: Tensor,
    v_chain_neg: Tensor,
    v_context_bank: Tensor | None = None,
    context_mask: Tensor | None = None,
    chain_mask: Tensor | None = None,
    symba_alpha: float = 2.0,
    head_cosine_weight: float = 1.0,
    head_mse_weight: float = 0.1,
) -> dict[str, Tensor]:
    """Full FF training step: compute all losses for one batch.

    Runs ``ff_forward`` twice (positive + negative), computes per-layer
    SymBa loss and the FFCL output head loss.

    The dict keys are designed for easy logging and per-layer optimizer routing.

    Returns dict:
        ``"ff_layer_{i}"`` → scalar SymBa loss for decoder layer i (0-indexed).
        ``"ff_total"``     → mean SymBa loss across layers.
        ``"head_cosine"``  → cosine distance loss on output_proj vs target.
        ``"head_mse"``     → MSE loss on output_proj vs target.
        ``"head_total"``   → weighted sum of head_cosine + head_mse.
        ``"loss_total"``   → ff_total + head_total (for logging, NOT for
                             single backward — each component is backward'd
                             separately to its own parameters).
        ``"goodness_pos"`` → [n_layers] mean positive goodness per layer.
        ``"goodness_neg"`` → [n_layers] mean negative goodness per layer.
    """
    # ── Forward both positive and negative chains ──
    g_pos_list, v_pred_pos = ff_forward(
        model, v_query, v_chain_pos,
        v_context_bank, context_mask, chain_mask,
    )
    g_neg_list, _ = ff_forward(
        model, v_query, v_chain_neg,
        v_context_bank, context_mask, chain_mask,
    )

    # ── Per-layer SymBa losses ──
    n_layers = len(g_pos_list)
    layer_losses: list[Tensor] = []
    g_pos_means: list[float] = []
    g_neg_means: list[float] = []

    for i in range(n_layers):
        ll = symba_loss(g_pos_list[i], g_neg_list[i], alpha=symba_alpha)
        layer_losses.append(ll)
        g_pos_means.append(g_pos_list[i].mean().item())
        g_neg_means.append(g_neg_list[i].mean().item())

    ff_total = torch.stack(layer_losses).mean()

    # ── FFCL head loss (supervised, on positive chain only) ──
    # Target = the original SONAR-space chain (not residual-scaled).
    target = v_chain_pos
    if chain_mask is not None:
        mask_f = chain_mask.unsqueeze(-1).float()
        count = mask_f.sum().clamp(min=1.0)
        cos_sim = F.cosine_similarity(v_pred_pos.float(), target.float(), dim=-1)
        head_cosine = ((1.0 - cos_sim) * chain_mask.float()).sum() / chain_mask.float().sum().clamp(min=1.0)
        head_mse = ((v_pred_pos.float() - target.float()).pow(2) * mask_f).sum() / count
    else:
        cos_sim = F.cosine_similarity(v_pred_pos.float(), target.float(), dim=-1)
        head_cosine = (1.0 - cos_sim).mean()
        head_mse = (v_pred_pos.float() - target.float()).pow(2).mean()

    head_total = float(head_cosine_weight) * head_cosine + float(head_mse_weight) * head_mse

    results = {
        "ff_total": ff_total,
        "head_cosine": head_cosine,
        "head_mse": head_mse,
        "head_total": head_total,
        "loss_total": ff_total.detach() + head_total.detach(),
        "goodness_pos": g_pos_means,
        "goodness_neg": g_neg_means,
    }
    for i, ll in enumerate(layer_losses):
        results[f"ff_layer_{i}"] = ll

    return results
