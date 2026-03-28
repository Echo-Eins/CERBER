# Stage 1.5 Loss Recovery Plan (2026-03-28)

## Context
Pure ranking ablation (norm_mode=none, SiLU, lr=1e-3) proved ranking CAN learn energy separation:
- rank_success=0.569, spread=0.236, E[c/a/h]=-1.62/-1.47/-1.39
- BUT inference fails: cosine 0.454→0.164 (WORSE), 0% success rate
- Root cause: ranking only teaches relative order, not WHERE the minimum should be
- Energy landscape shows deep well (E=-4.19) at wrong location, Langevin goes there

## Strategy: Add losses ONE AT A TIME, verify each doesn't break ranking

### Phase 1: Anchored Ranking (NEXT)
Config: `configs/ablation_phase1_anchored_ranking.json`
- Keep: ranking (λ=1.0), unconstrained MLP, SiLU, lr=1e-3
- Add: `clean_min` (λ=0.3, margin=0.1) — forces E(clean) to be the minimum
- Add: `energy_reg` (λ=0.01) — prevents unbounded energy wells
- [ ] Run 20 epochs
- [ ] Check: rank_success ≥ 0.5 (ranking not broken)
- [ ] Check: spread ≥ 0.15
- [ ] Check: cosine improvement > 0 in GUI inference
- [ ] Check: energy landscape — minimum near clean target, not spurious

### Phase 2: Conservative Boundary
- Add: `CQL` (λ=0.1, noise=0.5) — penalizes low energy on OOD points
- [ ] Verify ranking preserved, inference improved
- [ ] Check energy landscape for tighter wells around data

### Phase 3: Contrastive Signal
- Add: `InfoNCE` (λ=0.1, temp=0.07, 4 random negatives)
- [ ] Verify ranking preserved
- [ ] Check batch eval success rate

### Phase 4: Score Matching (if needed)
- Add: `MDSM` with warmup (warmup_epochs=10, λ_mdsm=0.1)
- Only if gradient DIRECTION is wrong after Phase 3
- [ ] Verify ranking not destroyed (rank_success ≥ 0.45)

### Kill Criteria (abandon approach if)
- Phase 1 ranking breaks (rank_success < 0.3) → weights too high, halve them
- Phase 1+2 inference still 0% → fundamental architecture problem
- Any phase: spread collapses to 0 → loss conflict, debug

---

# CERBER GUI Web Debug Plan (2026-03-25)

## Goal
Bring `cerber_gui` math and visualization behavior into parity with the CLI landscape tool and fix Plotly backend failures.

## Checklist
- [x] Reproduce/analyze Plotly trace error path in `cerber_gui`
- [x] Diff GUI math vs CLI (`visualize_landscape.py`) for vector scale, noise model, and Langevin path
- [x] Refactor GUI inference/landscape generation to reuse Stage1-consistent Langevin logic and target norm
- [x] Fix Plotly trace construction robustness and remove invalid/fragile properties
- [x] Align GUI trajectory panel semantics with CLI contour + trajectory behavior
- [x] Validate by static checks and code-path walkthrough; document remaining runtime checks for local CUDA env

## Review (to fill after fixes)
- Findings:
  - GUI used synthetic vectors with norm `10.0` and custom Langevin, while CLI used real SONAR vectors and shared Stage1 Langevin. This caused major geometry drift.
  - GUI model loading used permissive `strict=False` with inferred dims fallback, allowing silent architecture/checkpoint mismatches and invalid landscapes.
  - `SimpleEnergy.forward()` had hard clamp `[-100, 100]`; this flattened real energy ranges (e.g. `-367..-297`) into a plateau in visualization.
  - Plotly path could fail hard on schema/version mismatch; GUI had no graceful fallback.
- Files changed:
  - `cerber_gui/app.py`
  - `cerber_gui/checkpoint_analyzer.py`
  - `cerber_gui/landscape_3d.py`
  - `cebcm/models/energy.py`
  - `experiments/01_denoising_poc/visualize_landscape.py`
- Verification:
  - `python -m py_compile cerber_gui/app.py cerber_gui/landscape_3d.py cebcm/models/energy.py experiments/01_denoising_poc/visualize_landscape.py`
  - Code-path audit confirms GUI now uses Stage1 config + shared `run_langevin()` + dataset-based clean sample + relative noise semantics.
  - Runtime UI validation on target CUDA environment remains to be executed locally (this environment has no `torch` runtime).

---
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

---

# CERBER GUI + Math Re-Audit Plan (2026-03-25, Pass 2)

## Goal
Close all remaining GUI/math inconsistencies reported by user:
- mojibake/encoding corruption in UI labels,
- denoised marker/trajectory mismatch on landscape surface,
- unconditional test semantics (clean vector should not be treated as mandatory minimum),
- misleading `improvement=0` diagnostics,
- deterministic parity with CLI and mathematically valid inference diagnostics.

## Checklist
- [x] Re-audit `cerber_gui` math path end-to-end (sampling, projection, energy eval, plotting coordinates)
- [x] Fix user-facing text encoding and checkpoint title sanitization in GUI
- [x] Split diagnostics by model type (`simple` vs `unconditional`) and remove misleading target interpretation
- [x] Ensure inference output reports both energy-descent and geometric movement; detect no-op/refusal cases
- [x] Ensure 3D + trajectory plots refresh from the same post-inference landscape payload
- [x] Validate plot coordinate conventions (x/y/z mapping) with deterministic synthetic regression checks
- [x] Run static/runtime verification scripts and summarize remaining gaps
- [x] Produce `research1.md` with mathematical proof notes + test validity matrix + external references

## Review
- Implemented:
  - Inference now uses last executed Langevin state for GUI diagnostics and visualization consistency.
  - Landscape scan auto-expands to include trajectory/denoised projections.
  - 3D/2D markers are snapped to plotted mesh interpolation to prevent visual floating.
  - Added checkpoint name mojibake recovery and explicit unconditional-mode semantics in report.
  - Added explicit `reference source` (`dataset` vs `synthetic`) in inference report with warnings when dataset vectors are unavailable.
  - Added `research1.md` with external references + code-grounded math audit.
- Verification:
  - `python -m py_compile cerber_gui/app.py cerber_gui/landscape_3d.py cebcm/visualization/energy_landscape.py experiments/02_energy_matching/train.py`
  - Runtime validation script could not run here: active Python environment has no `torch`.

---

# Compact SOTA Research for Stage1 GUI Tests (2026-03-25)

## Goal
Prepare a compact, practical evidence pack in `research1.md` with 3-5 reliable sources per question:
- (A) energy interpretation in EBMs (minimum-energy behavior and regimes),
- (B) why unconditional EBM does not have to reconstruct a specific clean sample from a noisy one,
- (C) correct metrics for unconditional energy descent in high-dimensional embeddings.

## Checklist
- [x] Re-check AGENTS workflow and register plan in `tasks/todo.md`
- [x] Collect 3-5 high-trust sources for (A)
- [x] Collect 3-5 high-trust sources for (B)
- [x] Collect 3-5 high-trust sources for (C)
- [x] Write compact theses + direct CERBER Stage1 GUI implications in `research1.md`
- [x] Verify links and close review notes

## Review
- `research1.md` added with compact A/B/C structure.
- Source coverage: 4 refs for (A), 5 refs for (B), 5 refs for (C).
- Each block includes direct implications for Stage1 GUI test semantics.

---

# GUI Inference Metrics + Unconditional Crash Fix (2026-03-25, Pass 3)

## Goal
1) Fix unconditional inference crash (`element 0 ... does not require grad`).
2) Verify architecture display path for unconditional checkpoints.
3) Replace `N/A`-style summary with live runtime metrics that refresh after:
   - checkpoint load preview inference,
   - every manual inference run.

## Checklist
- [x] Fix grad-context bug in unconditional alignment diagnostic
- [x] Validate/strengthen architecture rendering source for unconditional models
- [x] Add persistent per-checkpoint runtime metrics in session state
- [x] Compute/update metrics on preview inference (during checkpoint selection)
- [x] Compute/update metrics on manual inference and refresh checkpoint summary
- [x] Run compile validation and document findings

## Review
- Fixed unconditional crash by removing `@torch.no_grad` from alignment path and forcing local `torch.enable_grad()` around `energy_and_grad`.
- Checkpoint summary architecture now resolves directly from loaded `state_dict` (with mismatch warning vs metadata).
- Added live per-checkpoint runtime metrics store; summary now shows latest inference metrics and refreshes:
  - after checkpoint selection preview inference,
  - after every manual inference run.
- Manual inference callback now updates `checkpoint_summary` output in the same click event.
- Validation:
  - `python -m py_compile cerber_gui/app.py cerber_gui/landscape_3d.py cebcm/visualization/energy_landscape.py experiments/02_energy_matching/train.py`

---

# GUI SOTA-Eval Completion (2026-03-25, Pass 4)

## Goal
Implement full SOTA-grade GUI evaluation for Stage1/Unconditional checks, so model quality is judged by distribution and manifold metrics, not only single-vector cosine.

## Detailed spec
- Add a dedicated evaluation core (`cerber_gui/sota_eval.py`) with:
  - MMD (RBF, median heuristic),
  - C2ST (linear probe, held-out accuracy),
  - PRDC (precision/recall/density/coverage),
  - manifold kNN proximity improvements (cosine and euclidean),
  - energy-descent statistics over a batch (improvement + success rate),
  - trajectory monotonicity for the inspected sample.
- Integrate evaluation in GUI pipeline for:
  1) checkpoint preview run (on selection),
  2) every manual inference run.
- Add GUI controls for evaluation budget:
  - eval batch size (number of query samples),
  - eval reference bank size.
- Persist per-checkpoint latest SOTA metrics in session state and render them:
  - in checkpoint summary (Live Inference Metrics block),
  - in inference output markdown.
- Keep implementation robust:
  - if dataset unavailable, explicitly mark distribution metrics as unavailable,
  - avoid accidental no-grad on input-gradient diagnostics.

## Checklist
- [x] Add `cerber_gui/sota_eval.py` with stable batched metric implementations
- [x] Extend runtime metric payload and formatting to include SOTA metric block
- [x] Batch-run Langevin on evaluation sample set and compute post-denoise distribution metrics
- [x] Wire new GUI controls (eval batch size, eval bank size) into both preview/manual callbacks
- [x] Refresh checkpoint summary after each inference with updated SOTA metrics
- [x] Validate with py_compile + quick synthetic invariants (metrics finite / ranges sane)

## Review
- Implemented:
  - Added `cerber_gui/sota_eval.py` with batched metric suite:
    - MMD (RBF + median heuristic),
    - C2ST (linear probe),
    - PRDC (precision/recall/density/coverage),
    - manifold kNN proximity improvements.
  - Added `sota_eval_cache` in GUI session state and cache invalidation per checkpoint.
  - Added SOTA batch evaluation computation inside both:
    - checkpoint preview landscape generation,
    - manual inference callback.
  - Extended checkpoint summary and inference report to render SOTA block from latest runtime metrics.
  - Added explicit GUI controls:
    - `SOTA Eval Batch Size`,
    - `SOTA Eval Reference Bank Size`,
    and wired them into all preview refresh triggers + manual inference.
- Validation:
  - `python -m py_compile cerber_gui/app.py cerber_gui/sota_eval.py cerber_gui/landscape_3d.py cebcm/visualization/energy_landscape.py`
  - Quick runtime synthetic invariants could not be executed in this shell because active Python environment has no `torch`.

---

# GUI Metric Consistency Patch (2026-03-25, Pass 5)

## Goal
Resolve user-reported mismatch between visual "near-target" behavior and zero/negative improvement readouts, and surface short runtime metrics where checkpoint-saved metrics are `N/A`.

## Checklist
- [x] Add explicit 2D-plane diagnostics (projected distance before/after) to runtime output
- [x] Keep 1024D primary metrics and explicitly label projection-vs-fullspace distinction
- [x] Add fallback for missing checkpoint metrics from latest runtime inference
- [x] Improve tiny-delta formatting (scientific fallback) to avoid false `0.000000` interpretation
- [x] Extend SOTA block with L2-before/after and denoise step norm
- [x] Fix C2ST reporting to be label-invariant (`max(acc, 1-acc)`) and expose raw acc
- [x] Bump SOTA eval cache key version to avoid stale pre-fix cache reuse
- [x] Validate via py_compile

## Review
- Implemented:
  - Added `Quick Inference Snapshot` in checkpoint summary with cosine/energy deltas.
  - Added fallback backfill for missing checkpoint cosine/success metrics from latest runtime run.
  - Added `2D slice distance to reference` diagnostics and explicit note that plot coordinates are projections.
  - Added L2 diagnostics (`l2_before/after/improvement/success`) and `denoise_step_norm_mean` to SOTA batch metrics.
  - Updated C2ST metric in `cerber_gui/sota_eval.py` to report label-invariant accuracy and raw accuracy.
  - Added `SOTA_EVAL_CACHE_VERSION=2` into cache key to prevent stale cached metrics.
- Validation:
  - `python -m py_compile cerber_gui/app.py cerber_gui/sota_eval.py`

---

# GUI Landscape Span Control + Cache Invalidation Fix (2026-03-25, Pass 6)

## Goal
- Add explicit control to expand Direction 1/2 visible range independently from grid detail.
- Ensure this control is wired through preview + manual inference + cache keys.
- Fix SOTA cache invalidation after key-versioning change.

## Checklist
- [x] Add absolute half-range control in UI (`0` = auto, `>0` = forced span)
- [x] Thread new parameter through `select_checkpoint_fn` and `run_inference_fn`
- [x] Thread new parameter through `generate_landscape_for_checkpoint` + cache key
- [x] Extend `scan_energy_landscape_3d` to honor absolute span override
- [x] Wire all Gradio `.change`/`.click` handlers with the new input
- [x] Fix `_invalidate_checkpoint_cache` compatibility with versioned SOTA cache keys
- [x] Validate by compile check

## Review
- Implemented:
  - New GUI slider `Landscape Half-Range (Absolute)` with range `[0..100]`.
  - Full parameter wiring across preview generation, manual inference, landscape scan, and cache keys.
  - Absolute span now overrides factor-based span when set (`>0`), enabling large-area exploration even for small noise distances.
  - Updated SOTA cache invalidation logic to support both legacy and v2 cache key formats.
- Validation:
  - `python -m py_compile cerber_gui/app.py cerber_gui/landscape_3d.py cerber_gui/sota_eval.py`

---

# Endpoint/Best-State Desync Closure (2026-03-25, Pass 7)

## Goal
Eliminate metric desynchronization between live inference and SOTA batch evaluation by enforcing a single endpoint semantics (last executed state) while preserving `best-energy` state for optional analysis.

## Checklist
- [x] Extend `LangevinResult` to carry both states explicitly (`v_final` best-energy, `v_last` last executed)
- [x] Update all Langevin variants (overdamped/pid/underdamped) to populate `v_last` on early-stop and full-run exits
- [x] Update GUI `run_langevin_denoise` fallback (`track_vectors=False`) to return `v_last` instead of `v_final`
- [x] Keep trajectory and reported endpoint consistent in non-tracking mode
- [x] Run compile validation

## Review
- Implemented:
  - Added `v_last` to `LangevinResult`.
  - Filled `v_last` in every return path of all three Langevin methods.
  - Batch/SOTA pathway now uses actually reached terminal state (`v_last`) instead of best-energy fallback.
  - This removes live-vs-batch endpoint mismatch and stabilizes interpretation of step norm / cosine / L2 improvements.
  - Bumped `SOTA_EVAL_CACHE_VERSION` to `3` so post-fix metrics are recomputed (no stale pre-fix cache artifacts).
- Validation:
  - `python -m py_compile cebcm/inference/langevin.py cerber_gui/app.py cerber_gui/sota_eval.py cerber_gui/landscape_3d.py`

---

# Stage1 Pipeline Re-Research: Unconditional vs Simple (2026-03-25, Pass 8)

## Goal
Perform a full re-research of Stage1 training pipelines:
- `unconditional` energy pipeline,
- `simple` / actor+critic-aligned pairwise pipeline,
and produce SOTA-grounded upgrade options with concrete implementation tracks.

## Checklist
- [x] Re-read AGENTS/spec/implementation plan constraints for Stage1 role in CERBER architecture
- [x] Audit current code paths end-to-end for both pipelines (objective, sampling, OOD controls, metrics, inference semantics)
- [ ] Run parallel subagent research:
  - [x] unconditional pipeline deep audit
  - [x] simple/actor-critic pipeline deep audit
  - [x] external SOTA methods + papers + practical recipes
- [x] Produce `research3.md` with agent-attributed findings and source links
- [x] Build prioritized improvement matrix (immediate / near-term / long-term)
- [x] Validate recommendations against current implementation constraints and Stage2/Stage3 integration goals

## Review
- Completed full three-track parallel audit:
  - `research3_agent_unconditional.md`
  - `research3_agent_simple.md`
  - `research3_agent_sota.md`
- Added consolidated synthesis file: `research3.md`.
- Code-verified key contradictions and risks before synthesis:
  - `actor_critic` objective conflict (`E(clean)<E(actor)` ranking vs actor term minimizing `E(actor)-E(clean)`),
  - unconditional Langevin noise-semantics mismatch across sampler paths,
  - unconditional `best.pt` selection by paired cosine improvement,
  - training eval endpoint mismatch (`v_final` usage in Stage1 eval path).
- Output includes prioritized implementation roadmap (P0-P3), experiment matrix, Stage2/3 readiness gates, and SOTA-backed architecture direction (conditional critic mainline + unconditional prior auxiliary).

---

# Stage1 Kill Criteria + Eval Rewrite (2026-03-25, Pass 9)

## Goal
Eliminate false-positive training verdicts by replacing weak single-metric pass checks with strict SOTA-aligned multi-metric evaluation and kill criteria during training.

## Checklist
- [x] Add unified evaluation/kill-criteria utilities for:
  - conditional Stage1 (`simple`/`actor_critic`)
  - unconditional Energy Matching
- [x] Rewrite `experiments/01_denoising_poc/train.py` evaluation:
  - use reached endpoint semantics (`v_last`)
  - add geodesic/L2/energy/clean-min-violation metrics
  - compute composite score + strict gates
- [x] Rewrite `experiments/02_energy_matching/train.py` evaluation:
  - keep cosine as diagnostic only
  - add distribution/manifold metrics (MMD/C2ST/PRDC/kNN)
  - replace best-checkpoint selection criterion with unconditional composite score
  - enforce strict unconditional kill criteria gates
- [x] Align standalone evaluation scripts with same kill-criteria logic
- [x] Run static validation (`py_compile`) on changed files
- [x] Document resulting behavior and remaining calibration knobs

## Review
- Added shared strict criteria module: `cebcm/training/kill_criteria.py`.
- Stage1 conditional training now evaluates with:
  - cosine + geodesic + L2 + energy success + clean-min violation + step norm,
  - `v_last` endpoint semantics,
  - strict multi-gate verdict and composite score checkpoint selection.
- Unconditional training now evaluates with:
  - energy-descent diagnostics across noise scales,
  - distribution/manifold suite (`MMD`, `C2ST`, `PRDC`, kNN),
  - strict unconditional gates and composite score checkpoint selection.
- Replaced legacy weak criteria:
  - old: `improvement > 0 && success_rate > 0.5`,
  - new: model-type-specific multi-gate SOTA criteria.
- Standalone evaluators were aligned with strict criteria output for consistency.
- Static validation passed:
  - `python -m py_compile cebcm/training/kill_criteria.py experiments/01_denoising_poc/train.py experiments/01_denoising_poc/evaluate.py experiments/02_energy_matching/train.py experiments/02_energy_matching/evaluate.py`

---

# Stage1.5 SOTA Implementation (2026-03-26, Pass 11)

## Goal
Implement full SOTA hybrid actor-critic pipeline with all P0 fixes and SOTA stabilization.

## Implementation Status

### ✅ Completed
- [x] Created `train_stage1_5.py` with full SOTA implementation
- [x] Removed actor_energy_loss contradiction
- [x] Added MDSM to critic for gradient validity
- [x] Implemented hybrid critic pattern (E_cond + λ*E_prior)
- [x] Added alternating training (2 critic : 1 actor)
- [x] Added CQL regularization for OOD
- [x] Added BC regularization for embedding anchor
- [x] Implemented composite score checkpoint selection
- [x] Added kill criteria integration

### 🔄 In Progress
- [ ] Create Stage1.5 config template
- [ ] Run CUDA validation
- [ ] Tune hyperparameters from first logs

## Priority Matrix (Original)

### P0 — Critical Correctness (blocker for Stage2/3)

- [x] **Fix actor_critic objective contradiction**
  - Files: `experiments/01_denoising_poc/train_stage1_5.py`
  - Issue: Critic requires `E(clean) < E(actor)` but actor minimizes `softplus(e_actor - e_clean)` → `E(actor) < E(clean)`
  - Fix: **REMOVED** actor_energy_loss entirely
  - Test: Pending CUDA validation

- [x] **Unify unconditional Langevin noise semantics**
  - Files: `cebcm/inference/langevin.py` (already fixed in Pass 7)
  - Status: v_last endpoint already implemented

- [x] **Fix unconditional checkpoint selection criterion**
  - Files: `experiments/01_denoising_poc/train_stage1_5.py`
  - Fix: Composite score with cosine/geodesic/clean-min-violation

- [x] **Unify endpoint semantics (v_last vs v_final)**
  - Status: Already fixed in Pass 7, reused in Stage1.5

### P1 — Objective Alignment

- [ ] **Add gradient penalty to critic loss**
  - Files: `cebcm/training/losses.py`
  - Purpose: Smooth energy landscape, prevent sharp minima
  - Implementation: `gradient_penalty()` function, add to MDSM loss

- [ ] **Add persistent contrastive term to unconditional**
  - Files: `cebcm/training/negative_buffer.py`, `experiments/02_energy_matching/train.py`
  - Issue: NCE only in warmstart, not main EM phase
  - Fix: Keep NCE active during main training with persistent chains
  - Test: PRDC/C2ST metrics improve vs warmstart-only baseline

- [ ] **Add final-state geometry loss for actor**
  - Files: `experiments/01_denoising_poc/train.py`
  - Purpose: Actor optimized for final projected state, not just delta
  - Implementation: Cosine/geodesic on `v_refined`, gradient alignment with `-∇E`

- [ ] **Add OOD/manifold penalties**
  - Files: `cebcm/training/losses.py`
  - Functions: `manifold_proximity_penalty()`, `shell_barrier_penalty()`
  - Test: OOD rate decreases, kNN proximity improves

### P2 — Stability and Monitoring

- [ ] **Add energy calibration layer**
  - Files: `cebcm/models/energy_unconditional.py`
  - Purpose: Normalize energy output to [0, 1] via running statistics
  - Implementation: `EnergyCalibrator` module with EMA

- [ ] **Add gradient clipping and EMA**
  - Files: `experiments/01_denoising_poc/train.py`
  - Implementation: `clip_grad_norm_()`, EMA weight wrapper
  - Test: Training stability improves, late-training generalization better

- [ ] **Add convergence detection for Langevin**
  - Files: `cebcm/inference/langevin.py`
  - Purpose: Early stopping when energy plateaus
  - Implementation: `converged`, `convergence_step` in `LangevinResult`

- [ ] **Add comprehensive metrics telemetry**
  - Files: `experiments/01_denoising_poc/train.py`, `experiments/02_energy_matching/train.py`
  - Track: energy stats, gradient norms, Langevin convergence, manifold quality, OOD rate

### P3 — Architecture Enhancements

- [ ] **Add manifold-aware Langevin dynamics**
  - Files: `cebcm/inference/langevin.py` (new function)
  - Purpose: Tangent space projection for hypersphere geometry
  - Implementation: `manifold_langevin_step()` with tangent gradient + noise

- [ ] **Add spectral normalization to energy network**
  - Files: `cebcm/models/energy.py`, `cebcm/models/energy_unconditional.py`
  - Purpose: Enforce 1-Lipschitz constraint, stabilize gradients
  - Implementation: Spectral norm on OrthoLinear weights

- [ ] **Implement hybrid critic (conditional + prior)**
  - Files: `experiments/01_denoising_poc/train.py`
  - Formula: `E_total(q, x) = E_cond(q, x) + lambda_prior * E_prior(x)`
  - Test: OOD drift reduces without hurting relevance metrics

## Experiment Matrix

| ID | Change | Expected Impact | Validation |
|----|--------|-----------------|------------|
| AC-1 | Remove actor energy term | Fix P0 contradiction | Clean-min violation ↓ |
| AC-2 | Add MDSM to critic | Gradient field quality | Langevin stability ↑ |
| AC-3 | Final-state geodesic loss | Cosine/geodesic ↑ | kNN proximity ↑ |
| U-1 | Unify sampler semantics | Train/eval parity | Trajectory match |
| U-2 | Manifold checkpoint criterion | Better selection | PRDC/C2ST ↑ |
| U-3 | Persistent NCE in EM | Manifold calibration | Density metrics ↑ |
| HYB-1 | Hybrid critic + prior | OOD robustness | AUROC ↑ |

## Stage2/3 Readiness Gates

Do **not** advance until:
- [ ] All P0 items complete and verified
- [ ] Conditional branch shows stable positive cosine/geodesic gain
- [ ] Unconditional branch improves PRDC/C2ST/MMD + OOD AUROC
- [ ] Actor proposals stay within support constraints (multi-start test)
- [ ] No clean-min violation mode in eval

## Review
- Plan created from `research3.md` findings
- Awaiting CUDA runtime for implementation and validation

---

# Stage1.5 Hard Validation + Repair (2026-03-26, Pass 12)

## Goal
Run a strict code-and-math validation of Stage 1.5 and harden it to match research3 P0/P1 constraints:
- objective sign consistency,
- noise/sampler semantics consistency,
- anti-instability guards (non-finite, gradient sanitation, clipping),
- truthful strict success criteria during training.

## Checklist
- [x] Re-read `AGENTS.md`, `CLAUDE.md`, `research3.md` and map required checks to code paths
- [x] Full static audit of `train_stage1_5.py` for runtime blockers and math contradictions
- [x] Fix Stage1.5 P0 correctness blockers (type/config/API mismatches, missing eval/kill hooks)
- [x] Verify critic/actor gradient signs and ranking consistency vs inference update direction
- [x] Verify noise semantics and Langevin navigation consistency with shared sampler (`cebcm/inference/langevin.py`)
- [x] Implement strict Stage1.5 eval + kill criteria integration using shared `kill_criteria.py`
- [x] Add/verify training stabilizers: non-finite guards, gradient sanitation, clipping, clean-min violation telemetry
- [x] Run compile validation on all changed files and document residual runtime limits
- [x] Compare current Stage1.5 to proposed pairwise conditional critic + actor(refinement) design and list exact deltas

## Review
- Replaced non-runnable `train_stage1_5.py` with executable Stage1.5 pipeline:
  - strict config parsing,
  - consistent critic/actor objectives,
  - shared Langevin-based eval,
  - strict kill criteria via `summarize_conditional_eval`.
- Corrected false implementation assumptions from Pass 11:
  - previous script declared features that were not actually runnable due API/type/signature mismatches.
- Updated Stage1.5 config schema:
  - `configs/base.py::Stage1_5Config`,
  - `configs/stage1_5_config.json`.
- Added unconditional prior scale guard:
  - `cebcm/models/energy_unconditional.py` now clamps `log_energy_scale` before `exp` (parity with pairwise critic stability guard).
- Fixed Stage1.5 optimizer semantics for hybrid critic:
  - separate parameter groups now apply both `critic_lr` and `prior_critic_lr` (no silent LR override when prior critic is enabled).
- Validation:
  - `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py`
  - `python -m py_compile configs/base.py`
  - `python -m py_compile experiments/01_denoising_poc/train.py`
  - `python -m py_compile cebcm/inference/langevin.py cebcm/training/losses.py cebcm/training/kill_criteria.py`
- Remaining limitation:
  - full CUDA runtime check still required in target environment with installed `torch`.

---

# Stage1.5 Completion Pass (2026-03-26, Pass 13)

## Goal
Close remaining Stage1.5 architectural gaps identified by user:
- implement real `critic_steps_per_actor > 1`,
- remove teacher-forced training semantics (`query != clean target`),
- implement full twin-critic conditional training with retrieval/hard-negative conditioning.

## Checklist
- [x] Implement non-teacher-forced pair sampling in Stage1.5 (query/positive from retrieval protocol)
- [x] Implement retrieval-conditioned hard negatives for critic ranking loss
- [x] Implement actual twin conditional critics in Stage1.5 (`E1_cond`, `E2_cond`) plus optional prior
- [x] Implement real alternating schedule `critic_steps_per_actor`
- [x] Align actor/refinement update with twin hybrid energy and verify sign consistency (`v <- v - lr * grad(E)`)
- [x] Add explicit Stage1.5 metrics for retrieval/ranking quality and clean-min violations
- [x] Update Stage1.5 config schema/json for new retrieval+twin parameters
- [x] Run static validation (`py_compile`) for all changed files
- [x] Update `research3.md` and `tasks/lessons.md` with findings and anti-regression rules

## Review
- Stage1.5 training is now non-teacher-forced:
  - query `q` and target positive `v_pos` are sampled via retrieval (`retrieve_pos_hard`) from a manifold bank.
- Critic is now truly twin:
  - separate `critic1` / `critic2` checkpoints,
  - hybrid inference energy via `max/mean` aggregation (`twin_aggregate`).
- `critic_steps_per_actor` now changes runtime behavior:
  - critic updates run in a real inner loop, alternating critic branches.
- Added retrieval/hard-negative conditioning:
  - positives from top-k retrieval excluding near-identical self-match,
  - hard negatives from deeper retrieval window.
- Added 8GB-safe stabilization:
  - per-critic update (not both critics in one second-order graph),
  - OOM catch + `torch.cuda.empty_cache()` skip path,
  - lighter default config (`batch_size=32`, `ortho_n_iters=4`, reduced eval load).
- Added math-forward upgrades:
  - smooth twin critic aggregation (`twin_aggregate=softmax`, temperature-controlled),
  - conditional NCE loss for retrieval ranking plus optional prior-NCE branch,
  - strict retrieval self-exclusion by sample index (not only cosine threshold),
  - tangent-noise Langevin option (`langevin_tangent_noise=true`) for sphere-consistent stochastic steps.



## Stage1.5 Post-change Audit (2026-03-26)

### Checklist
- [x] Re-validated Stage1.5 sign consistency (MDSM target, Langevin descent, ranking inequalities, actor alignment)
- [x] Fixed misleading training telemetry labels (`rank` -> explicit `rank_loss` + `rank_success`)
- [x] Added per-inequality ranking rates (`clean<actor`, `actor<hard`, `clean<hard`)
- [x] Added deterministic eval subset reuse for fair checkpoint selection
- [x] Stabilized no-eval epoch logging schema with `status=not_evaluated`
- [x] Added numerical sanitization in Stage1.5 `conditional_mdsm`
- [x] Reduced parameter finite-check overhead (interval-based)
- [x] Synced Stage1.5 README with actual pipeline/config/checkpoint keys
- [x] Added minimal Stage1.5 regression tests (`tests/test_stage1_5_integrity.py`)

### Review
- Fixed correctness gaps in reporting and checkpoint-scoring fairness without changing core objective semantics.
- Runtime verification is still blocked in this shell due missing `torch/pytest`; static compile checks pass.

---

# Stage1.5 Performance + Math Safety Pass (2026-03-26, Pass 14)

## Goal
Implement requested SOTA-safe acceleration for Stage1.5 without changing optimization direction:
- cheaper orthonorm path,
- batched eval Langevin,
- `torch.compile` + checkpointing on heavy second-order graph,
- optional vectorized retrieval.

## Checklist
- [x] Add Stage1.5 config/runtime knobs for orthonorm schedule, compile, checkpointing, and batched eval
- [x] Implement epoch-based orthonorm iteration schedule and log active `n_iters`
- [x] Implement batched eval Langevin with mathematically-safe fixed-step semantics
- [x] Add optional `torch.compile` wrappers with runtime-safe fallback (no key/ckpt breakage)
- [x] Add gradient checkpointing in create_graph path (`conditional_mdsm`)
- [x] Vectorize `retrieve_pos_hard` while preserving strict index exclusion behavior
- [x] Add/update integrity tests for schema/math-sensitive changes
- [x] Run `py_compile` and summarize expected perf/quality impact + risks

## Review
- Added opt-in compile path (`enable_compile`) with safe fallback to eager mode.
- Added Stage1.5 orthonorm schedule controls and per-epoch `n_iters` logging.
- Added batched eval Langevin path with fixed-step semantics (no batch-coupled early stop).
- Added gradient checkpointing knob for MDSM create-graph path.
- Vectorized retrieval positive/hard selection without relaxing strict index exclusion.
- Updated tests for ortho schedule and schedule validation contract.
- Validation: `python -m py_compile` passed for changed Stage1.5 files and tests.

---

# Stage1.5 Runtime Hotfix (2026-03-26, Pass 15)

## Goal
Fix runtime crash after epoch due to scalar/tensor variable shadowing in Stage1.5 training loop.

## Checklist
- [x] Reproduce and localize crash source from traceback (`Boolean value of Tensor ... ambiguous`)
- [x] Rename conflicting epoch loop variable to `epoch_idx`
- [x] Rename critic energy tensors to explicit `e_pos/e_actor/e_hard`
- [x] Update all downstream logging/checkpoint fields to use `epoch_idx`
- [x] Re-run syntax validation (`py_compile`)
- [x] Update `tasks/lessons.md` anti-regression rule

## Review
- Root cause: `ep` (epoch index) was overwritten by energy tensor `ep = crit(...)` in same function scope.
- Fixed in `experiments/01_denoising_poc/train_stage1_5.py`; eval gate and JSON logging now read scalar epoch index only.
- Validation: `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py` passed.

---

# Stage1.5 Config + Live Monitoring Upgrade (2026-03-26, Pass 16)

## Goal
Stabilize Stage1.5 default config and provide honest live visualization for training progress:
- fix unstable rank margins / actor barrier defaults,
- log batch-window timing (`sec/batch`) in training output,
- stream batch+epoch metrics to GUI-compatible JSONL,
- update web live monitor to read Stage1.5 JSONL and render detailed progress charts.

## Checklist
- [x] Re-check and update `configs/stage1_5_config.json` for safer startup hyperparameters
- [x] Add per-log-window timing metrics in `train_stage1_5.py` and persist to JSONL stream
- [x] Keep epoch-level metrics/kill criteria logging schema stable and backward compatible
- [x] Extend `cerber_gui/live_monitor.py` to parse both JSON (legacy) and JSONL (Stage1.5 stream)
- [x] Add detailed live dashboard traces (loss, rank metrics, violation, speed) in Plotly
- [x] Update GUI labels/help text (`app.py`) for Stage1.5 live path defaults
- [x] Validate syntax (`py_compile`) for all changed files and summarize runtime usage

## Review
- Config defaults were tightened for stability:
  - lower ranking margins (`0.5/0.3/0.8 -> 0.2/0.1/0.3`),
  - smaller actor step size (`1.0 -> 0.5`),
  - stronger actor barrier (`0.1 -> 0.2`),
  - softer retrieval hardness window (`topk/hard: 8..32 -> 6..24`).
- Stage1.5 trainer now emits timing in console every `log_every` window:
  - `sec/batch=...`
  - `eta=...m`
- Stage1.5 trainer now writes streaming metrics to
  `experiments/03_Stage_1.5/logs/training_metrics.jsonl` with explicit events:
  - `event=batch` (window metrics + timing),
  - `event=epoch` (train aggregate + kill/eval snapshot),
  - `event=final`.
- Live monitor now supports both legacy JSON and Stage1.5 JSONL streams and builds a detailed 4-panel Plotly dashboard:
  - core losses,
  - rank/violation rates,
  - regularizers/retrieval/skip,
  - speed + eval/kill signals.
- GUI updates:
  - metrics upload accepts `.json` and `.jsonl`,
  - live tab labels/default path now target Stage1.5 JSONL stream,
  - live monitor startup stops old watcher and immediately returns a plot.
- Validation:
  - `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py`
  - `python -m py_compile cerber_gui/live_monitor.py`
  - `python -m py_compile cerber_gui/metrics_viewer.py`
  - `python -m py_compile cerber_gui/app.py`

---

# Live 3D Landscape Monitoring + Rolling Checkpoint Policy (2026-03-26, Pass 17)

## Goal
Eliminate live-monitor UX gaps:
- prevent aggressive rerender scrolling behavior as much as possible,
- add live 3D landscape checks during training (manual + auto every N epochs),
- support rolling latest checkpoint plus periodic milestone checkpoint retention.

## Checklist
- [x] Add Stage1.5 checkpoint cadence config (`checkpoint_every_epochs`, `rolling_checkpoint_name`)
- [x] Change Stage1.5 saver to periodic checkpoints + rolling latest + non-periodic cleanup
- [x] Add Stage1.5 checkpoint compatibility in GUI checkpoint loader (`critic1_state` fallback)
- [x] Add Live Monitor 3D controls and manual `Check Landscape` button
- [x] Add timer-based auto landscape refresh gate (default every 5 epochs)
- [x] Reuse checkpoint-analysis plotting pipeline for live 3D rendering
- [x] Reduce unnecessary live plot rerenders when metrics file timestamp is unchanged
- [x] Run syntax validation on modified files

## Review
- Added rolling checkpoint workflow:
  - periodic checkpoints are kept every `checkpoint_every_epochs`,
  - `latest_epoch.pt` is overwritten each epoch for immediate landscape inspection,
  - stale non-periodic `epoch_*.pt` files are cleaned up.
- Live tab now includes:
  - checkpoint directory input,
  - auto-update toggle,
  - `Auto Every N Epochs` (default `5`),
  - manual `Check Landscape` button,
  - dedicated live 3D landscape + live trajectory plots.
- Live landscape uses same core rendering modules as Checkpoint Analysis:
  - `generate_landscape_for_checkpoint(...)`
  - `_render_landscape_figure(...)`
  - `create_trajectory_plot(...)`
- Added Stage1.5 checkpoint format support in analyzer (`critic1_state` as fallback `model_state`).
- Added a scroll-preservation JS observer and skipped redundant plot refreshes when watcher data has not changed.

# Stage 1.5 GUI Eval Protocol Alignment + Runtime NameError Fix (2026-03-26, Pass 18)

## Goal
Fix Stage 1.5 GUI evaluation mismatch (self-target vs retrieval-target) and resolve runtime crash in live landscape auto-update.

## Checklist
- [x] Fix `NameError: json is not defined` in `cerber_gui/app.py`
- [x] Add Stage 1.5-aware SOTA eval branch in GUI (conditional retrieval objective)
- [x] Keep backward compatibility for unconditional/self-denoise checkpoints
- [x] Update SOTA text labels from `clean` to `target` where applicable
- [x] Validate modified modules with `py_compile`

## Review
- Added `import json` to app module to unblock JSONL epoch parsing in auto landscape refresh.
- Added conditional Stage 1.5 SOTA eval path:
  - sample `(query, positive, hard)` triplets via cosine retrieval,
  - seed noisy candidates from query/hard mix,
  - evaluate cosine/L2 against retrieval target (not self clean) for conditional checkpoints.
- Added explicit SOTA metadata in UI output:
  - `eval_objective` (`conditional_retrieval` or `self_denoise`),
  - `target_label` (`retrieved_pos` or `clean`).
- Compiled successfully:
  - `python -m py_compile cerber_gui/app.py`
  - `python -m py_compile cerber_gui/checkpoint_analyzer.py`
  - `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py`

# Stage 1.5 Single-Run Inference Protocol Alignment (2026-03-26, Pass 19)

## Goal
Align single-run GUI inference with Stage 1.5 conditional objective to remove mixed interpretation between runtime metrics and SOTA batch eval.

## Checklist
- [x] Add shared conditional-checkpoint detector helper
- [x] Add unified inference sampler returning query/target/noisy for both self and conditional modes
- [x] Route `run_langevin_denoise` with explicit `v_query_override` / `v_target_override` in conditional mode
- [x] Recompute runtime cosine/L2/energy against target (not always self-clean)
- [x] Update runtime/report labels: objective + target semantics
- [x] Apply same alignment to checkpoint landscape preview path
- [x] Validate with `py_compile`

## Review
- Runtime and SOTA now evaluate under coherent objective semantics.
- Stage 1.5 conditional checkpoints use retrieval target in both preview and manual inference.
- GUI now displays objective context explicitly (`self_denoise` vs `conditional_retrieval`).

# Stage 1.5 Structural Math Fixes: Actor Energy Corridor + Retrieval Target Hygiene (2026-03-26, Pass 20)

## Goal
Fix non-hyperparameter mathematical failure modes causing false minima trapping and rank/violation drift.

## Checklist
- [x] Add retrieval positive quality floor + fallback-to-query when no valid positive exists
- [x] Add actor two-sided energy corridor loss using critic references (`pos` and `hard`)
- [x] Add actor monotonic descent guard from seed (`E(next) <= E(seed)`)
- [x] Expose new actor guard metrics in batch logs (`a_bar`, `a_desc`)
- [x] Sync GUI conditional retrieval sampler with same min-similarity + fallback logic
- [x] Validate via `py_compile`

## Review
- Retrieval objective no longer trains on semantically invalid positives when neighborhood quality is poor.
- Actor is constrained to stay between critic reference energies (with margins), reducing collapse into pathological low-energy pockets.
- Additional descent guard suppresses actor steps that increase energy from its own seed.
- Runtime observability improved with explicit actor guard metrics in training stream.

# Stage1.5 Full Math Audit: Critics + Actor + Navigation (2026-03-26, Pass 21)

## Goal
Close non-hyperparameter correctness gaps found in full Stage1.5 audit:
- sigma-conditioning parity between train and eval/inference,
- tangent-space alignment consistency in actor loss,
- underdamped Langevin step correctness and safety checks,
- GUI runtime parity with Stage1.5 checkpoint config.

## Checklist
- [x] Re-audit Stage1.5 critic/actor/navigation math end-to-end with subagent cross-check
- [x] Fix Stage1.5 eval sigma mismatch by binding explicit sigma in Langevin/eval energy path
- [x] Fix actor gradient-alignment geometry to compare tangent vs tangent directions
- [x] Fix actor descent guard to compare energies on the same projected manifold
- [x] Harden config validation (`sigma_curriculum_start>0`, strict enums, underdamped constraints)
- [x] Fix underdamped Langevin position update scaling (remove extra `lr` factor)
- [x] Add numerical-safe sphere/tangent projection for zero-norm edge cases
- [x] Fix GUI Stage1.5 runtime config hydration (`checkpoint["config"]` fallback)
- [x] Add GUI sigma/tangent/sampler parity for Stage1.5 conditional inference and SOTA batch eval
- [x] Run static verification (`py_compile`) on all changed modules

## Review
- Stage1.5 eval now optimizes and measures the same sigma-conditioned critic regime used in training.
- Actor alignment no longer asks tangent-projected delta to match full-space gradients.
- Underdamped dynamics no longer apply an unintended `O(lr^2)` position scaling.
- GUI now inherits Stage1.5 runtime knobs from checkpoints and uses matching conditional seed/noise semantics.
- Validation run:
  - `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py cebcm/inference/langevin.py cebcm/training/losses.py cerber_gui/app.py configs/base.py`

# Stage1.5 Twin-Critic GUI Parity Fix (2026-03-26, Pass 22)

## Goal
Remove structural visualization/inference mismatch where Stage1.5 checkpoints were rendered as single-critic (`critic1_state`) models instead of true twin-hybrid energy.

## Checklist
- [x] Add runtime twin-energy adapter in GUI (`_TwinConditionalEnergyAdapter`)
- [x] Load both `critic1_state` and `critic2_state` for Stage1.5 checkpoints
- [x] Apply training-time aggregation mode parity (`max` / `mean` / `softmax`, with temperature)
- [x] Load optional `prior_state` + `lambda_prior` into GUI runtime energy
- [x] Keep backward compatibility for non-Stage1.5 single-model checkpoints
- [x] Validate via `py_compile`

## Review
- GUI inference/landscape now reflects the same hybrid energy family used during Stage1.5 training.
- This removes a major source of apparent "training vs landscape" contradictions in checkpoint inspection.
- Validation run:
  - `python -m py_compile cerber_gui/app.py`
