# Stage 1 Improvement Plan (2026-03-23)

## Goal
Implement the agreed Stage 1 upgrades for speed, stability, and reproducibility while preserving current architecture semantics.

## Checklist
- [x] Add config support for Bjorck schedule, DSM geometry options, and systems tuning
- [x] Implement DSM extensions: sigma sampling/weighting, tangent projection, directional mode, magnitude auxiliary
- [x] Fix Langevin best-state tracking bug and early-stop behavior
- [x] Add true trajectory capture in Langevin API
- [x] Align visualization trajectory with actual sampler trajectory
- [x] Rewrite training loop with deterministic seeds + fixed eval subset
- [x] Add AMP / compile / dataloader throughput options
- [x] Skip unnecessary GP computation when lambda=0
- [x] Save full Stage1 config in checkpoints
- [x] Make evaluate script consume checkpoint Stage1 config to avoid train/eval drift
- [x] Add NaN hotfixes after first real run feedback
- [x] Add finite-gradient guard + fail-fast/backoff + robust Björck update after second run feedback
- [x] Fix LR-collapse coupling (backoff vs warmup) and harden MDSM numerics for low-norm outliers
- [x] Full Stage1 math audit against training logs (`transfer_note`) with contradiction fixes

## Review
### Implemented files
- `configs/base.py`
- `cebcm/models/energy.py`
- `cebcm/training/losses.py`
- `cebcm/inference/langevin.py`
- `experiments/01_denoising_poc/train.py`
- `experiments/01_denoising_poc/evaluate.py`
- `experiments/01_denoising_poc/visualize_landscape.py`

### NaN hotfixes
- MDSM second-order path forced to FP32 by default (`mdsm_force_fp32=True`).
- `log_energy_scale` exponential is clamped in forward to avoid inf/nan cascade.
- `energy_scale` LR multiplier reduced/configurable (`energy_scale_lr_multiplier=20`).
- Non-finite batch guard with skip-and-continue (`skip_non_finite_batches=True`).
- Added finite-gradient guard before `optimizer.step` to prevent parameter corruption.
- Added consecutive non-finite fail-fast and LR backoff controls.
- Corrected Björck update to row/column-consistent form (`WW^T` for wide matrices) with spectral pre-normalization.
- Fixed LR-collapse bug: non-finite backoff no longer mutates `initial_lr` (warmup anchor).
- Backoff now triggers only on short consecutive streaks (default >=3), not on isolated events.
- Hardened MDSM numerics: norm/sigma floors, safer cosine epsilon, finite sanitization, and skip-rate metrics.
- Fixed objective/inference sign mismatch: Stage1 MDSM now trains target gradient with the sign consistent to `v <- v - lr * ∇E`.
- Disabled unsafe default early-stop threshold (`energy_threshold=None`) to prevent no-op Langevin refinement.
- Warmup/backoff now uses optimizer-update steps only (skipped batches no longer advance warmup), and backoff persists during warmup via per-group LR scale.

### Validation status
- Runtime training execution is still blocked in this environment because `torch` is not installed in the active Python interpreter.
- Verification performed via static inspection, compile checks, and user-provided runtime logs.

---

## Remediation Pass (2026-03-23, Transfer Log Recheck)

### Goal
Eliminate remaining math/stability contradictions from `transfer_note`: LR perception under instability, scale-identifiability gap in directional MDSM, and final-state selection in Langevin.

### Checklist
- [x] Re-audit `transfer_note` and Stage 1 code paths for unresolved contradictions
- [x] Fix directional MDSM weighting scale collapse (normalize sigma weights to mean 1)
- [x] Add directional-scale identifiability guard in trainer (`directional=True` + no magnitude auxiliary)
- [x] Tune Stage 1 default stability profile (Bjorck schedule, magnitude auxiliary, non-finite backoff behavior)
- [x] Fix Langevin best-state selection to include terminal state
- [x] Update project lessons with new failure patterns
- [x] Re-check syntax/compile on edited files
- [ ] Re-run full Stage 1 train/eval on target CUDA environment and verify denoising improvement

### Files updated in remediation pass
- `configs/base.py`
- `cebcm/training/losses.py`
- `experiments/01_denoising_poc/train.py`
- `cebcm/inference/langevin.py`

---

## Actor + Critic Track (2026-03-23)

### Goal
Implement a full Stage 1 `Actor + EBM Critic` pipeline to reduce training fragility and improve denoising convergence while preserving CERBER's energy-verifier architecture.

### Plan
- [x] Add dedicated latent denoising Actor model with sigma conditioning
- [x] Extend Stage1 config for actor_critic training/eval controls
- [x] Implement joint training loop (single optimizer, dual-network losses for actor and critic)
- [x] Integrate actor-then-critic evaluation path in train/evaluate scripts
- [x] Extend checkpoint/resume logic to persist both components
- [x] Run static validation (compile checks) and document usage
- [ ] Run full CUDA training/evaluation for actor_critic and tune initial hyperparameters from first logs

---

# Energy Matching Pipeline — Mathematically Verified Implementation Plan

**Date:** 2026-03-23
**Status:** Implementation complete, pending CUDA validation
**Branch:** `claude/review-tech-spec-xGHfW`

---

## 0. Mathematical Foundation

### 0.1 Energy Matching Core Idea (Balcerak et al., NeurIPS 2025)

Energy Matching trains a **time-invariant scalar energy** E_θ(x) : ℝ^d → ℝ such that
its negative gradient -∇_x E_θ(x) approximates the velocity field of an optimal
transport flow from noise to data.

**Key insight:** unlike flow matching (which learns a vector field v_θ(x,t)), Energy
Matching learns a *conservative* vector field derived from a scalar potential.
This guarantees:
- Path independence (the energy landscape is well-defined)
- Thermodynamic consistency (Boltzmann distribution at equilibrium)
- No need for time conditioning

### 0.2 The Conditional Optimal Transport (OT) Path

Given data point x₁ ~ p_data and noise x₀ ~ p_prior (typically N(0,I)):

```
x_t = (1 - t) · x₀ + t · x₁,  t ∈ [0, 1]
```

The conditional velocity field (ground truth):

```
u_t(x_t | x₁) = x₁ - x₀ = (x₁ - x_t) / (1 - t)
```

### 0.3 Flow Matching Loss (Lipman et al., ICLR 2023)

Standard flow matching trains v_θ(x_t, t) to match u_t:

```
L_FM = E_{t~U(0,1), x₁~p_data, x₀~p_prior} [ ||v_θ(x_t, t) - u_t(x_t | x₁)||² ]
```

### 0.4 Energy Matching Adaptation

Energy Matching replaces the free vector field v_θ(x_t, t) with the
**negative gradient of a scalar energy** -∇_x E_θ(x):

```
L_EM = E_{t~U(0,1), x₁~p_data, x₀~p_prior} [ ||-∇_x E_θ(x_t) - u_t(x_t | x₁)||² ]
```

**Critical property:** E_θ has NO time conditioning. The single energy landscape
must simultaneously encode the correct velocity at all points along all OT paths.

### 0.5 Two-Phase Behavior

The paper identifies that the loss naturally separates into two regimes:

**Phase 1 (t ≈ 0, far from data):**
- x_t ≈ x₀ (near noise)
- Target velocity ≈ x₁ - x₀ (points toward data)
- Energy gradient learns OT transport directions
- This is the "flow matching" regime

**Phase 2 (t ≈ 1, near data):**
- x_t ≈ x₁ (near data manifold)
- Velocity field converges to score function: ∇ log p(x)
- Energy learns Boltzmann-like landscape near data
- This is the "EBM" regime

### 0.6 Sampling from Trained Model

After training, generate samples via ODE integration:

```
dx/dt = -∇_x E_θ(x),  x(0) ~ p_prior
```

Discretized (Euler):
```
x_{k+1} = x_k + Δt · (-∇_x E_θ(x_k))
```

Or with Langevin noise for stochastic sampling:
```
x_{k+1} = x_k - η · ∇_x E_θ(x_k) + √(2η) · ε,  ε ~ N(0,I)
```

### 0.7 Adaptation for SONAR Embeddings (our contribution)

**Key differences from image domain:**

1. **Hypersphere geometry:** SONAR embeddings have ||x|| ≈ 0.2051 (not unit norm,
   but concentrated). After each integration step, project back:
   ```
   x_{k+1} = normalize(x_{k+1}) · target_norm
   ```

2. **Prior distribution:** Instead of N(0,I), use N(0, σ²I) matched to data distribution:
   ```
   σ_prior = target_norm / √d ≈ 0.2051 / √1024 ≈ 0.00641
   ```
   This ensures prior samples have similar norm to data.

3. **Relative noise scaling:** Following CERBER convention, noise is relative to norm:
   ```
   x_t = (1-t) · x₀ + t · x₁  where  x₀ = x₁ + ε,  ε ~ N(0, σ²·||x₁||²·I)
   ```
   This preserves the SONAR geometry better than absolute noise.

4. **1-Lipschitz constraint:** We keep OrthoLinear + GroupSort for the energy network
   to ensure smooth gradients. The final layer is unconstrained for energy magnitude.

### 0.8 Mathematical Verification Checklist

- [x] OT path x_t is well-defined: linear interpolation, ∂x_t/∂t = x₁ - x₀ ✓
- [x] Conditional velocity u_t = (x₁ - x₀) is correct: by definition of linear OT ✓
- [x] Loss L_EM minimizes ||∇E + u_t||²: convex in function space ✓
- [x] At convergence, -∇E_θ = u_t almost everywhere: by optimality of L² loss ✓
- [x] Conservative field guarantee: v = -∇E is curl-free by construction ✓
- [x] Sampling via ODE follows learned flow: by definition of gradient flow ✓
- [x] Sphere projection preserves tangent dynamics: projects only radial component ✓
- [x] Prior norm matches data norm: by construction of σ_prior ✓

---

## 1. Implementation

### 1.1 Implemented Files

```
cebcm/models/energy_unconditional.py     — E(x) → scalar, no pairwise, no σ (~3.7M params)
cebcm/training/energy_matching.py        — EM loss (MSE, cosine, weighted) + OT path + sampling
cebcm/training/negative_buffer.py        — Replay buffer + NCE loss (full & simple)
configs/energy_matching.py               — EnergyMatchingConfig dataclass
experiments/02_energy_matching/train.py   — Training script (3 modes)
experiments/02_energy_matching/evaluate.py — Evaluation (denoise + sample quality + SONAR decode)
```

### 1.2 Training Modes

1. **`nce_warmstart_em`** (recommended) — NCE (10 epochs) → Cosine EM fine-tune
2. **`energy_matching`** — Pure MSE EM from scratch
3. **`cosine_em`** — Cosine direction EM (for 1-Lipschitz networks)
4. **`weighted_em`** — Near-data weighted EM

---

## 2. Verification Against Known Failure Modes

| Failure Mode | Mitigation |
|---|---|
| Energy collapse (flat E) | NCE warmstart creates initial landscape; EM refines |
| Score matching mode blindness | NCE explicitly learns p(x)/p_n(x) ratio |
| Gradient magnitude mismatch | Cosine EM variant; unconstrained final layer |
| Prior mismatch | Matched prior σ = target_norm/√d |
| OOD sampling | Sphere projection after each step |
| Hessian cost at 1024d | Not needed — EM uses first-order only |

---

## 3. Task Checklist

- [x] Fix visualization device mismatch bug
- [x] Create `cebcm/models/energy_unconditional.py`
- [x] Create `cebcm/training/energy_matching.py`
- [x] Create `cebcm/training/negative_buffer.py`
- [x] Create `configs/energy_matching.py`
- [x] Create `experiments/02_energy_matching/train.py`
- [x] Create `experiments/02_energy_matching/evaluate.py`
- [ ] Run full CUDA training/evaluation and tune from first logs

---

## 4. Architecture Review: Actor + Critic Pattern

**Current CERBER architecture (from Tech Spec §3.1):**

| Component | Role | Training |
|-----------|------|----------|
| SONAR Encoder | Text → V (1024d) | Frozen |
| IPP | Predicts V_init (the "Actor") | Stage 2+ |
| EBT (SimpleEnergy) | Evaluates quality (the "Critic") | Stage 1+ |
| Langevin Dynamics | Refines V_init → V_answer | No training (uses ∇E) |
| SONAR Decoder | V → Text | Frozen |

**The Actor-Critic analogy:**
- **Critic = E_θ** — evaluates "how good is this candidate?"
- **Actor = IPP** — proposes initial answer
- **Refinement = Langevin** — uses Critic's gradients to improve Actor's proposal

**Energy Matching strengthens the Critic** by training it to model the full
data distribution p(x), not just local denoising directions. This gives:
1. Absolute quality evaluation (not just relative)
2. Sample generation capability (new)
3. Theoretically optimal score function near data
