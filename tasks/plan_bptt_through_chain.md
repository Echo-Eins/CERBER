# Plan: BPTT Through Chain Generation (Closing the Exposure Bias Gap)

## Problem Statement

`roll_cos_last` = 0.616 at `target_steps=5`. Baseline (no DF/SS/TF-noise) gives 0.616 too.
Three interventions (SS=0.30, noisy TF=0.002, DF=0.15) each gave < 0.01 improvement.

**Root cause**: `generate()` detaches at every step (line ~1570). The model never receives
gradient for *cascading* multi-step drift — only independent per-step errors.

## Pre-Requisites: Three Blockers for BPTT

Before enabling gradient flow through the chain, three mechanisms must be made
BPTT-safe. Without this, gradients either vanish exponentially or explode.

### Blocker 1: `sphere_project` Norm Mismatch → Exponential Gradient Vanishing

**Current**: `sphere_project` rescales `raw_next` (norm ≈ 0.25) to `target_norm` (0.2051).
The Jacobian of `v/||v|| * target_norm` has gain `target_norm / ||v||` per step.
At raw_norm=0.25: gain = 0.82/step → **0.82⁵ = 0.37** over 5 steps (63% gradient lost).

**Fix**: Replace `sphere_project` in the BPTT path with a **differentiable soft projection**
that does NOT rescale but only penalizes norm deviation:

```python
def _soft_sphere_project(self, v: Tensor) -> Tensor:
    """Differentiable near-sphere projection for BPTT-safe chain generation.
    
    Instead of hard rescaling (gain = target_norm/||v||, causing gradient 
    vanishing when ||v|| > target_norm), use the raw output directly and 
    add an auxiliary norm-matching loss. The gradient through the chain is
    then a pure Jacobian of the transformer — no multiplicative attenuation.
    """
    return v  # pass-through; norm penalty applied in loss
```

The norm penalty is added as a separate loss term:
```
L_norm = λ_norm * mean((||raw_next|| - target_norm)² )
```

This gives the model gradient signal to *learn* to produce `target_norm`-scale outputs
(addressing the raw_norm drift problem!) rather than forcibly rescaling and destroying
gradient flow.

**Eval path unchanged**: eval still uses hard `sphere_project` for SONAR compatibility.

### Blocker 2: `_to_residual_space` × `output_proj` Jacobian Stability

**Forward**: SONAR(0.2) → ×156 → residual(32) → layers → output_proj → SONAR(~0.25)
**Backward**: The gradient through `output_proj.T @ _to_residual_space.T` has gain ≈ 1.0
(156 × 0.008 ≈ 1.25). This is nearly unity and stable.

**BUT**: `final_norm` (RMSNorm) is in the path. RMSNorm Jacobian:
`d/dx (x/rms) = (I - x⊗x / ||x||²) / rms`
This is a projection operator with spectral radius 1.0 (idempotent).
Over K steps, it doesn't compound — safe.

**Conclusion**: No fix needed. The `_to_residual_space` ↔ `output_proj` pair is
naturally balanced for BPTT. The 156× scaling in `_to_residual_space` is exactly
compensated by the learned ~0.008× gain of `output_proj`.

### Blocker 3: Attention Sink on Start Token

**Current**: `generate()` builds chain as `[start_token, gen_1, gen_2, ...]`. With
causal self-attention, every position can attend to `start_token`. If it develops
high attention weight (known "sink" pattern), the gradient from step t flows
primarily to start_token rather than to intermediate chain positions 1..t-1.

**Fix for BPTT**: Not a blocker per se — the gradient still flows through the 
residual stream (skip connection) even if attention weights favor position 0.
The residual connection `x + attn(norm(x))` means `d(out)/d(input) = I + d(attn)/d(input)`.
Even if `d(attn)/d(chain_k)` is small, the identity term `I` provides gradient flow.

However, for the attention path specifically, **truncated BPTT** (see below) naturally
limits the distance over which we ask attention to propagate gradient, sidestepping
the sink issue.

## The Approach: Truncated BPTT with K-Step Rollouts

### Core Idea

Instead of detaching at every step, allow gradient to flow through the **last K steps**
of the generated chain. This is standard truncated BPTT from RNN training, adapted
to our autoregressive transformer in continuous vector space.

### Design

```
Chain positions:    [start, p₁, p₂, p₃, p₄, p₅]
                     det   det  det  ←── grad ──→
                                     BPTT window K=3
```

For each training step with `target_steps = T`:
1. Generate positions 1...(T-K) with `.detach()` as today (warm-up prefix)
2. Generate positions (T-K+1)...T **without detach** (BPTT window)
3. Compute loss only on positions within the BPTT window
4. Gradients flow through K consecutive transformer forward passes

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `bptt_steps` | 2-3 | Start conservative; K=2 means gradient through 2 chained steps |
| `bptt_warmup_epochs` | After system1 | Only enable when model is stable |
| `lambda_bptt` | 0.5 | Separate weight from `lambda_roll` |
| `bptt_grad_clip` | 0.5 | Tighter clip than main (1.0) — BPTT gradients can spike |
| `bptt_use_soft_sphere` | true | Use soft projection (Blocker 1 fix) in BPTT window |
| `lambda_norm_penalty` | 0.01 | Weight for the norm-matching auxiliary loss |

### Implementation Plan

#### Step 1: `_soft_sphere_project` method in ChainGenerator

Add alongside existing `_sphere_project`. Returns raw output without rescaling.
Add NaN scrubbing (same as existing) but no norm change.

```python
def _soft_sphere_project(self, v: Tensor) -> Tensor:
    v_safe = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    row_norm = v_safe.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
    collapsed = row_norm < 1e-5
    if collapsed.any():
        fallback = torch.zeros_like(v_safe)
        fallback[..., 0] = self.cfg.target_norm
        v_safe = torch.where(collapsed, fallback, v_safe)
    return v_safe
```

#### Step 2: `generate_with_bptt` method in ChainGenerator

New method alongside existing `generate()`. Key differences:
- Takes `bptt_steps` parameter
- First `T - bptt_steps` positions: same as current (detach)
- Last `bptt_steps` positions: NO detach, uses `_soft_sphere_project`
- Returns both the chain and per-step raw norms for the norm penalty
- **No repeat_penalty** during BPTT (it uses detached history, would break grad flow)
- **No sphere_project in BPTT window** (use soft version)
- **No noise** in BPTT window (must be deterministic for clean gradient)

```python
def generate_with_bptt(
    self,
    v_query: Tensor,
    num_steps: int,
    bptt_steps: int = 2,
    v_context_bank: Tensor | None = None,
    context_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """Generate chain with gradient flow through last bptt_steps positions.
    
    Returns:
        chain_predictions: [B, T, D] — all step predictions (SONAR space)
        bptt_predictions: [B, K, D] — raw (un-projected) predictions in BPTT window
        raw_norms: list of per-step raw output norms for norm penalty
    """
```

**Implementation sketch**:
```python
# Phase 1: Prefix generation (detached, same as current generate())
prefix_steps = max(0, num_steps - bptt_steps)
chain = self.start_token.expand(bsz, -1, -1)
prefix_preds = []
for t in range(prefix_steps):
    x = chain
    for layer in self.layers:
        x = layer(x, context, context_mask=ctx_mask)
    x = self.final_norm(x)
    raw = self.output_proj(x[:, -1:, :])
    proj = self._sphere_project(raw)
    prefix_preds.append(raw)
    chain = torch.cat([chain, self._to_residual_space(proj).detach()], dim=1)

# Phase 2: BPTT window (gradient flows through)
bptt_preds = []
bptt_raw_norms = []
for t in range(bptt_steps):
    x = chain  # chain includes detached prefix + attached recent steps
    for layer in self.layers:
        x = layer(x, context, context_mask=ctx_mask)
    x = self.final_norm(x)
    raw = self.output_proj(x[:, -1:, :])
    
    bptt_preds.append(raw)
    bptt_raw_norms.append(raw.detach().norm(dim=-1).mean())
    
    # Soft projection: pass-through (no rescaling, no detach!)
    soft_proj = self._soft_sphere_project(raw)
    scaled = self._to_residual_space(soft_proj)
    chain = torch.cat([chain, scaled], dim=1)  # NO .detach()!

all_preds = torch.cat([torch.cat(prefix_preds, dim=1), 
                        torch.cat(bptt_preds, dim=1)], dim=1)
bptt_out = torch.cat(bptt_preds, dim=1)
return all_preds, bptt_out, bptt_raw_norms
```

#### Step 3: BPTT Loss in `compute_composite_objective`

```python
if bptt_enabled:
    all_preds, bptt_preds, bptt_norms = model.generate_with_bptt(
        v_q, steps, bptt_steps=bptt_k,
        v_context_bank=context_banks, context_mask=context_mask,
    )
    
    # BPTT loss: compare last K positions against GT
    bptt_targets = chains[:, -bptt_k:, :]  # last K GT positions
    bptt_mask = chain_mask[:, -bptt_k:]
    l_bptt, bptt_stats = _masked_step_losses(
        bptt_preds, bptt_targets, bptt_mask, d_model, w_cos, w_mse
    )
    
    # Norm penalty: encourage raw_norm → target_norm
    norm_targets = torch.full_like(bptt_norms_tensor, model.cfg.target_norm)
    l_norm = F.mse_loss(bptt_norms_tensor, norm_targets)
    
    loss = loss + lambda_bptt * l_bptt + lambda_norm * l_norm
```

#### Step 4: Gradient Safety

1. **Per-step gradient clipping inside BPTT**: After the BPTT backward,
   clip gradients to `bptt_grad_clip` before the optimizer step.
   This is already handled by the global `clip_grad_norm`, but we may
   want a tighter value when BPTT is active.

2. **NaN gate inside BPTT window**: If any step in the BPTT window produces
   NaN, fall back to detached mode for that step (break the grad chain).
   
3. **AMP compatibility**: The BPTT window runs through the model K times.
   Each forward pass is under `torch.autocast`. The backward through all K
   passes must also be under autocast. This works naturally since the entire
   `compute_composite_objective` is already under `torch.autocast`.

4. **Memory**: K=2 means 2 extra forward passes stored in the computation graph.
   Memory ≈ 2× one forward pass ≈ 2 × (batch_size × seq_len × d_model × n_layers × ~4 tensors).
   For B=32, L=6, D=1024, 6 layers: ~150 MB per step. K=2 → +300 MB. Manageable.

### Step 5: Config Changes

```json
{
    "training": {
        "bptt_enabled": true,
        "bptt_steps": 2,
        "bptt_warmup_epochs": 5,
        "loss_lambda_bptt": 0.5,
        "loss_lambda_norm_penalty": 0.01,
        "bptt_grad_clip": 0.5
    }
}
```

### Step 6: Experiment Protocol

1. **Baseline confirmation**: Run current config (DF FIXED) for 5 more epochs, confirm plateau.
2. **BPTT K=2 only**: Enable BPTT with K=2, disable DF (isolate the effect). 
   Watch: bptt_loss, norm_penalty, raw_norm convergence, roll_cos_last.
3. **BPTT K=2 + DF**: Re-enable DF alongside BPTT if K=2 shows improvement.
4. **BPTT K=3**: If K=2 works, try K=3 for deeper correction signal.

### Expected Outcomes

| Metric | Current (E22) | Expected with BPTT K=2 |
|--------|--------------|------------------------|
| roll_cos_last (ts=5) | 0.616 | 0.65-0.70 |
| raw_norm | 0.25 | 0.205 (converges to target!) |
| tf_cos | 0.873 | ~0.87 (unchanged) |
| gap (tf-roll_last) | 0.256 | 0.17-0.22 |

The norm penalty simultaneously fixes the raw_norm drift problem AND enables
stable BPTT by making `sphere_project` gain ≈ 1.0.

## Risk Analysis

| Risk | Severity | Mitigation |
|------|----------|------------|
| BPTT gradient explosion | HIGH | bptt_grad_clip=0.5, K≤3, NaN gate per step |
| Memory OOM | MEDIUM | K=2 adds ~300MB; reduce batch_size if needed |
| BPTT fights TF loss | MEDIUM | Separate lambda; BPTT only on last K positions |
| Attention sink kills BPTT signal | LOW | Residual skip provides gradient path regardless |
| Norm penalty destabilizes early training | LOW | bptt_warmup_epochs=5 delays activation |

## Summary

The approach is conceptually simple: **let the gradient flow through the last 2-3 steps
of generation**. The main engineering challenge is making `sphere_project` BPTT-safe
via soft projection + norm penalty. This simultaneously solves the raw_norm drift issue
that has persisted since the beginning of training.
