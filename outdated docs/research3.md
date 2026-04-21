# CERBER Stage1 Re-Research (Pass 8): Unconditional vs Simple/Actor+Critic

Date: 2026-03-25  
Scope: full re-audit of Stage1 training pipelines + SOTA options for next iteration

---

## 0) Method and Inputs

This synthesis combines:

1. **Code-grounded audit (current repo)**:
   - `experiments/01_denoising_poc/train.py`
   - `experiments/02_energy_matching/train.py`
   - `cebcm/training/losses.py`
   - `cebcm/training/energy_matching.py`
   - `cebcm/inference/langevin.py`
   - `cebcm/models/energy.py`
   - `cebcm/models/energy_unconditional.py`
   - `cebcm/training/negative_buffer.py`
   - `cerber_gui/app.py`
   - `cerber_gui/sota_eval.py`

2. **Parallel subagent reports**:
   - `research3_agent_unconditional.md`
   - `research3_agent_simple.md`
   - `research3_agent_sota.md`

3. **Spec alignment checks**:
   - `CEBCM_Technical_Specification.md`
   - `IMPLEMENTATION_PLAN.md`
   - `AGENTS.md`

---

## 1) Executive Verdict

### 1.1 Main conclusion

Do **not** choose a single winner as "unconditional-only" or "simple-only".

Best SOTA-aligned direction for CERBER:

- keep **pairwise conditional critic path** as main backbone for Stage2/3 relevance,
- keep **unconditional energy** as auxiliary manifold prior / reranker / OOD regularizer,
- make actor explicit as **support-constrained proposal model**, then short-run critic refinement.

### 1.2 Why current results look contradictory

Observed pattern ("energy improves, cosine worsens", "2D looks good, 1024D bad") is explained by real code-level issues, not by one visualization bug:

1. Objective contradictions in `actor_critic`.
2. Sampler semantics mismatch across unconditional paths.
3. Evaluation metric mismatch for unconditional checkpoints.
4. Geometry mismatch (Euclidean training vs sphere-constrained inference).
5. Weak manifold/OOD control in both pipelines.

---

## 2) Hard Code Findings (Verified)

## 2.1 `actor_critic` has a real objective contradiction (P0)

In `experiments/01_denoising_poc/train.py`:

- Critic ranking requires `E(clean) < E(actor)` via hinge terms:
  - `F.relu(e_clean - e_actor + margin_clean_actor)` etc.
- Actor energy coupling minimizes `softplus(e_actor - e_clean)`, i.e. pushes `E(actor) < E(clean)`.

These two gradients fight each other by construction.

Consequence:
- model can learn sub-clean attractors;
- energy/L2 can improve while semantic cosine gets worse.

## 2.2 Training-time eval still uses `v_final` (best-energy), not reached endpoint (P1)

In `experiments/01_denoising_poc/train.py` eval path:
- `v_final = result.v_final`
- metrics computed on `v_final`

But in diagnostics you often want **actual reached state** (`v_last`) for consistency with trajectory semantics.

Consequence:
- train/eval metrics can drift from live behavior.

## 2.3 Unconditional Langevin noise semantics are inconsistent (P0)

Different code paths use different parameterizations:

- In unconditional helpers (`energy_matching.py`, `energy_unconditional.py`):
  - `noise = randn * noise_scale`
  - update: `x - lr*grad + sqrt(2*lr)*noise`
  - effective std: `sqrt(2*lr) * noise_scale`

- In shared inference (`cebcm/inference/langevin.py`):
  - update noise `randn * sqrt(2*lr*noise_scale)`
  - effective std: `sqrt(2*lr*noise_scale)`

These are not equivalent and can differ by order-of-magnitude.

Consequence:
- train/eval/GUI behavior mismatch even with "same" hyperparameters.

## 2.4 Unconditional `best.pt` selection metric is currently misaligned (P0)

In `experiments/02_energy_matching/train.py`:
- `best.pt` is chosen by average paired cosine denoising improvement.

For unconditional `E(x)` this is secondary diagnostic at best, not primary objective.

Consequence:
- checkpoint selection can systematically pick the wrong model.

## 2.5 Current unconditional EM phase is OT-dominant and drops persistent equilibrium shaping (P1)

`nce_warmstart_em` applies NCE in warmstart phase, then main training is EM-only variants.
No persistent contrastive/equilibrium term in main phase.

Consequence:
- weaker manifold density calibration and mode weighting.

## 2.6 `UnconditionalEnergy.forward` has no scale clamp guard unlike pairwise model (P2)

- `energy_unconditional.py`: `exp(log_energy_scale) * net(x)`
- `energy.py`: clamp on `log_energy_scale` before `exp`

Consequence:
- unconditional scale is less guarded against runaway.

---

## 3) Pipeline-Specific Analysis

## 3.1 Unconditional Pipeline (`experiments/02_energy_matching`)

### Strengths

- mathematically clean scalar potential;
- no query leakage by design;
- useful as global manifold prior candidate.

### Current blockers

1. Geometry mismatch:
   - training uses Euclidean OT chords (`x_t = (1-t)x0 + tx1`),
   - inference enforces sphere/tangent dynamics in SONAR space.

2. Sampler mismatch:
   - inconsistent noise semantics and update implementation.

3. Eval mismatch:
   - paired denoise cosine used as best-checkpoint criterion.

4. Weak persistent equilibrium/OOD shaping:
   - NCE is warmstart-only in typical run.

### Net result

Unconditional model can look locally plausible but still fail manifold-level metrics and semantic diagnostics.

## 3.2 Simple / Actor+Critic Pipeline (`experiments/01_denoising_poc`)

### Strengths

- closer to Stage2/3 goal (query-conditioned critic + proposal/refinement idea),
- can express relevance dynamics that unconditional cannot.

### Current blockers

1. P0 loss contradiction (critic ordering vs actor energy coupling).
2. Critic objective mismatch with inference:
   - ranking-only critic in actor_critic mode,
   - but inference depends on gradient field quality.
3. Actor task mismatch:
   - actor sees clean target in Stage1 setup (teacher-forced denoise),
   - this is not Stage2 proposal task.
4. Geometry mismatch in actor loss:
   - optimized in delta-space before projection,
   - evaluated in projected final-state geometry.
5. OOD/manifold control too weak.

### Net result

Can produce low-energy and lower L2 endpoints while hurting cosine/manifold quality.

---

## 4) Cross-Pipeline Root Causes

1. **Objective/inference mismatch**: losses teach scalar ordering or local deltas, while inference relies on stable global gradient field.
2. **Support blindness**: norm projection is not enough; true data-manifold support constraints are missing.
3. **Metric mismatch**: pairwise denoising metrics mixed with unconditional evaluation.
4. **Endpoint semantics drift**: best-energy vs last-step mixing across paths.
5. **Geometry inconsistency**: Euclidean training with sphere-constrained deployment.

---

## 5) SOTA-Backed Design Direction for CERBER

## 5.1 Recommended architecture pattern

Use a **hybrid**:

- `E_cond(q, x)` pairwise critic (main relevance signal),
- `E_prior(x)` unconditional critic (manifold prior / OOD barrier),
- actor/initializer proposes in-support candidates,
- short-run bounded refinement by critic gradients.

Combined energy:

`E_total(q, x) = E_cond(q, x) + lambda_prior * E_prior(x)`

## 5.2 Why this fits spec and SOTA

- Matches CERBER spec direction (IPP + EBT + Langevin/refinement).
- Aligns with correction-model pattern from modern literature:
  - actor/initializer proposes,
  - energy refines/reranks,
  - not "energy alone solves everything".

---

## 6) Concrete Upgrade Plan

## Phase P0: Correctness (must do first)

1. **Fix actor_critic contradiction**
   - remove or redesign current `actor_energy_loss`.
   - keep consistent ordering objective.

2. **Unify sampler semantics**
   - one shared unconditional sampler backend,
   - one noise parameterization everywhere.

3. **Unify endpoint semantics**
   - report `v_last` as default user-facing endpoint everywhere.

4. **Fix unconditional checkpoint selection**
   - stop using paired cosine gain as primary best metric.

## Phase P1: Objective alignment

1. **Hybrid critic for actor_critic mode**
   - `L_critic = lambda_dsm * L_MDSM + lambda_rank * L_rank + lambda_gp * L_GP`.

2. **Actor final-state geometry losses**
   - geodesic/cosine on final projected state,
   - optional alignment with `-grad E`,
   - trust-region step penalty.

3. **Persistent equilibrium term in unconditional**
   - keep NCE/contrastive component active during main EM phase (not warmstart-only).

## Phase P2: Manifold/OOD controls

1. Add explicit support penalties:
   - kNN-distance penalty to reference manifold,
   - shell/radial barrier,
   - hard OOD negatives.

2. Train with self-refined states:
   - short-run refinement states must be seen during training.

## Phase P3: Performance/system

1. Orthonorm schedule / spectral ablation.
2. AMP/compile/checkpointing/EMA for stable throughput.
3. Per-layer grad-norm telemetry and non-finite root-cause logging.

---

## 7) Metrics That Must Become Primary

For unconditional:

- energy descent success rate,
- PRDC, MMD, C2ST,
- support/OOD AUROC,
- shell deviation,
- multi-start consistency.

For conditional actor+critic:

- cosine/geodesic improvement (primary),
- clean-minimum violation rate,
- kNN manifold proximity delta,
- energy/cosine monotonicity over trajectory,
- support trust-region violations.

Common:

- always report both `v_last` and `v_best` (if used), never mix silently.

---

## 8) Stage2/Stage3 Readiness Criteria (Proposed)

Do **not** advance on Stage1 loss alone.
Advance when all gates pass:

1. No P0 contradictions remain.
2. Sampler/eval semantics unified.
3. Conditional branch shows stable positive angular gain on held-out without target leakage mode.
4. Unconditional prior branch improves manifold/OOD metrics (PRDC/C2ST/MMD + AUROC).
5. Actor proposals stay within support constraints under multi-start tests.

---

## 9) Immediate Experiment Matrix

1. **AC-1**: Remove current actor energy term; keep ranking critic.
   - Check clean-min violation and cosine recovery.

2. **AC-2**: Add MDSM back to critic in actor_critic.
   - Check gradient alignment and Langevin stability.

3. **AC-3**: Swap actor delta-loss priority to final-state geodesic priority.
   - Check cosine/geodesic and kNN improvements.

4. **U-1**: Unify unconditional sampler semantics.
   - Verify train/eval/GUI parity.

5. **U-2**: Replace unconditional best checkpoint criterion with manifold composite score.
   - Compare selected checkpoint vs old criterion.

6. **U-3**: Keep NCE-equilibrium term during main EM training.
   - Check PRDC/C2ST improvements vs warmstart-only baseline.

7. **HYB-1**: Attach `E_prior` to conditional critic with small `lambda_prior`.
   - Check OOD drift reduction without hurting relevance.

---

## 10) What Was Missing for Future Inference (Important)

The biggest missing piece was not "more layers"; it is **support-aware proposal and trust control**.

Future full inference needs:

1. actor proposal distribution constrained to data support,
2. critic that is both relevance-aware and manifold-aware,
3. bounded refinement (trust region), not unconstrained long chains,
4. calibration metrics linking energy rank to downstream semantic quality.

Without this, system can optimize the wrong geometry while looking numerically "better" in local plots.

---

## 11) Final Recommendation

Primary path:

1. Fix P0 correctness issues now.
2. Keep conditional pairwise critic as main branch.
3. Promote unconditional model to auxiliary prior role.
4. Train actor as support-constrained proposal model and distill from short-run critic refinement.
5. Gate progress by manifold + semantic metrics, not by one scalar loss.

This is the most defensible SOTA-aligned route from current Stage1 to stable Stage2/3 CERBER.

---

## 11.5) Concrete Daboration Plans Per Pipeline

### Plan A: Unconditional Energy Pipeline Improvements

**Goal:** Make unconditional training more stable and manifold-aware.

#### A1. Add Energy Calibration Layer
**File:** `cebcm/models/energy_unconditional.py`

```python
class EnergyCalibrator(torch.nn.Module):
    """Normalizes energy output using running statistics."""
    def __init__(self, ema_decay=0.999):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_std", torch.ones(1))
        self.ema_decay = ema_decay

    def forward(self, energy: torch.Tensor, training: bool = False) -> torch.Tensor:
        if training:
            batch_mean = energy.mean()
            batch_std = energy.std()
            self.running_mean.lerp_(batch_mean, 1 - self.ema_decay)
            self.running_std.lerp_(batch_std, 1 - self.ema_decay)

        z = (energy - self.running_mean) / (self.running_std + 1e-8)
        return torch.sigmoid(z)  # Normalize to [0, 1]
```

**Integration:** Wrap `UnconditionalEnergy.forward()` output.

---

#### A2. Add Manifold-Aware Langevin Dynamics
**File:** `cebcm/inference/langevin.py` (new function)

```python
def manifold_langevin_step(
    v: torch.Tensor,  # [batch, dim] on hypersphere
    grad: torch.Tensor,
    lr: float,
    noise_scale: float,
    target_norm: float,
) -> torch.Tensor:
    """Langevin dynamics constrained to hypersphere manifold."""
    # 1. Project gradient to tangent space
    grad_tangent = grad - (grad * v).sum(dim=-1, keepdim=True) * v

    # 2. Tangent space noise (not isotropic!)
    noise = torch.randn_like(grad_tangent)
    noise_tangent = noise - (noise * v).sum(dim=-1, keepdim=True) * v

    # 3. Move in tangent space
    v_tangent = v - lr * grad_tangent + (2 * lr * noise_scale) ** 0.5 * noise_tangent

    # 4. Project back to sphere (exponential map approximation)
    v_new = torch.nn.functional.normalize(v_tangent, dim=-1) * target_norm

    return v_new
```

**Citation:** Mathieu & Nickel (2020) "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders"

---

#### A3. Add Convergence Detection
**File:** `cebcm/inference/langevin.py`

Extend `LangevinResult`:
```python
@dataclass
class LangevinResult:
    v_final: torch.Tensor
    v_last: torch.Tensor  # Already added in Pass 7
    converged: bool = False  # NEW
    convergence_step: int | None = None  # NEW
    final_energy: float | None = None  # NEW
```

Add early stopping in Langevin loop:
```python
plateau_counter = 0
prev_energy = float("inf")
energy_tolerance = 1e-4
patience = 10

for step in range(max_steps):
    energy = compute_energy(v_current)

    if abs(energy.item() - prev_energy) < energy_tolerance:
        plateau_counter += 1
        if plateau_counter >= patience:
            result.converged = True
            result.convergence_step = step
            break
    else:
        plateau_counter = 0
    prev_energy = energy.item()
```

---

#### A4. Add KL Prior Matching Loss
**File:** `cebcm/training/energy_matching.py`

```python
def kl_prior_penalty(v: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Penalize deviation from target norm shell."""
    actual_norm = v.norm(dim=-1)
    return torch.mean((actual_norm - target_norm) ** 2)

# In training loop:
loss += lambda_kl * kl_prior_penalty(v_samples, target_norm)
```

---

#### A5. Add Persistent Contrastive Term
**File:** `cebcm/training/negative_buffer.py` (extend existing)

Keep NCE active during main EM phase, not just warmstart:
```python
# In main training loop (not just warmstart):
if config.use_persistent_contrastive:
    negative_samples = buffer.sample(batch_size)
    nce_loss = compute_nce_loss(energy, positive_samples, negative_samples)
    loss += config.lambda_nce * nce_loss
    buffer.add(v_current.detach())  # Maintain persistent chains
```

---

### Plan B: Simple/Actor-Critic Pipeline Improvements

**Goal:** Fix P0 contradictions and add missing regularizations.

#### B1. Fix Actor-Critic Contradiction (P0 CRITICAL)
**File:** `experiments/01_denoising_poc/train.py`

**Current broken code:**
```python
# Critic requires E(clean) < E(actor)
critic_loss = F.relu(e_clean - e_actor + margin)

# Actor minimizes softplus(e_actor - e_clean), i.e. E(actor) < E(clean)
actor_loss = torch.nn.functional.softplus(e_actor - e_clean)
```

**Fix option 1 (remove actor energy coupling):**
```python
# Remove actor_energy_loss entirely
# Actor learns only from:
# - L2/denoise loss to clean target
# - Optional: gradient alignment with critic
total_loss = critic_ranking_loss + actor_denoise_loss
```

**Fix option 2 (align objectives):**
```python
# Make actor help critic by pushing E(actor) HIGHER than E(clean)
# This matches critic ranking objective
actor_energy_loss = torch.nn.functional.softplus(e_clean - e_actor)  # Flipped sign
# Now both critic and actor agree: E(clean) < E(actor)
```

---

#### B2. Add Gradient Penalty to Critic
**File:** `cebcm/training/losses.py`

```python
def gradient_penalty(energy_fn, x: torch.Tensor, lambda_gp: float = 1.0) -> torch.Tensor:
    """Penalize large gradient norms for smooth energy landscape."""
    grad = torch.autograd.grad(
        energy_fn(x).sum(),
        x,
        create_graph=True,
        only_inputs=True,
    )[0]
    return lambda_gp * torch.mean((grad.norm(dim=-1) - 1.0) ** 2)

# Add to MDSM loss:
total_loss = mdsm_loss + gradient_penalty(...)
```

---

#### B3. Add OOD/Manifold Penalties
**File:** `cebcm/training/losses.py` (new function)

```python
def manifold_proximity_penalty(
    v: torch.Tensor,
    reference_bank: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """Penalize points far from data manifold via kNN distance."""
    # Compute min distance to any reference point
    # reference_bank: [N, dim], v: [batch, dim]
    distances = torch.cdist(v, reference_bank, p=2)  # [batch, N]
    min_distances = distances.min(dim=-1).values  # [batch]
    return torch.mean(torch.relu(min_distances - threshold) ** 2)

def shell_barrier_penalty(v: torch.Tensor, target_norm: float, margin: float = 0.1) -> torch.Tensor:
    """Penalize points outside norm shell."""
    norm = v.norm(dim=-1)
    lower = target_norm * (1 - margin)
    upper = target_norm * (1 + margin)
    return torch.mean(torch.relu(lower - norm) ** 2 + torch.relu(norm - upper) ** 2)
```

---

#### B4. Add Final-State Geometry Loss for Actor
**File:** `experiments/01_denoising_poc/train.py`

```python
# After Langevin refinement:
v_refined = run_langevin(v_actor_init, critic, steps=20)

# Add loss on final refined state (not just delta)
cosine_final = cosine_similarity(v_refined, v_clean)
actor_final_loss = 1 - cosine_final.mean()

# Optional: align actor output with critic gradient direction
with torch.enable_grad():
    v_actor_init.requires_grad_(True)
    _, grad = critic.energy_and_grad(v_actor_init)
grad_alignment = torch.nn.functional.cosine_similarity(
    (v_refined - v_actor_init).detach(),
    -grad.detach()
)
actor_grad_align_loss = 1 - grad_alignment.mean()

total_actor_loss = actor_denoise_loss + actor_final_loss + actor_grad_align_loss
```

---

#### B5. Add Gradient Clipping and EMA
**File:** `experiments/01_denoising_poc/train.py`

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# EMA of weights
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k].lerp_(v, 1 - self.decay)

    def apply(self):
        self.model.load_state_dict(self.shadow)

# Use EMA for evaluation
ema = EMA(model, decay=0.999)
# After each training step:
ema.update()
# Before eval:
ema.apply()
```

---

#### B6. Add Comprehensive Metrics Telemetry
**File:** `experiments/01_denoising_poc/train.py`

Track per epoch:
```python
metrics = {
    # Energy statistics
    "energy_mean": energy.mean().item(),
    "energy_std": energy.std().item(),
    "energy_min": energy.min().item(),
    "energy_max": energy.max().item(),

    # Gradient statistics
    "grad_norm_mean": grad_norm.mean().item(),
    "grad_norm_max": grad_norm.max().item(),

    # Langevin diagnostics
    "langevin_convergence_rate": converged_count / total_samples,
    "avg_convergence_step": avg_step.item(),

    # Manifold quality
    "shell_deviation": shell_penalty.item(),
    "knn_proximity": knn_metric.item(),

    # OOD detection
    "ood_rate": ood_count / total_samples,
}
```

---

## 12) Agent Reports (Detailed Findings)

### 12.1 Unconditional Pipeline Audit (Haiku 4.5)

**Architecture findings:**
- `UnconditionalEnergy`: MLP [2048, 1024, 512], ~6.7M params, no `_sigma_freqs` buffer
- Uses `energy_and_grad()` for efficient joint computation
- Sphere projection implemented: `F.normalize(v, dim=-1) * target_norm`

**Missing regularizations:**
1. No KL divergence to prior distribution
2. No manifold constraint loss (decoder fidelity not checked)
3. No curvature/Hessian regularization for local convexity
4. No energy calibration layer (values unbounded, range -100 to -370+)

**Missing metrics:**
- Energy distribution per epoch
- Gradient norm statistics
- Langevin convergence rate
- OOD detection rate (how often projection fires)
- Manifold coverage metrics

**Key proposals:**
1. Manifold-aware Langevin (tangent space projection)
2. Energy calibration module (sigmoid normalization)
3. Convergence detection with early stopping

---

### 12.2 SOTA Methods Research (Opus 4.5)

**Priority recommendations:**

**Immediate (Low complexity, High impact):**
- Gradient penalty to Stage1 loss
- Gradient clipping in optimizer
- Energy/gradient monitoring telemetry
- AdamW with weight decay (0.01-0.1)

**Short-term (Medium complexity, High impact):**
- Persistent MCMC chains (not fresh noise each batch)
- Contrastive OOD penalty
- EMA of weights
- Cosine decay LR after warmup

**Medium-term:**
- Spectral normalization for energy network
- Manifold quality metrics (PRDC, C2ST, MMD)
- Actor-critic gradient flow optimization
- Hierarchical energy modeling

**Key references:**
- Song et al. (2020) Score-Based Generative Modeling
- Du et al. (2020) Improved Contrastive Divergence for EBMs
- Liu et al. Energy-based OOD Detection
- Energy Matching (Balcerak et al., NeurIPS 2025)

---

### 12.3 Simple/Actor-Critic Pipeline Status

**Note:** Full code-grounded audit was blocked by file access issues during subagent execution. However, the main synthesis pass (Section 2-11 above) identified all critical P0-P2 issues from direct code inspection.

---

## 13) Sources (Primary)

1. Energy Matching (NeurIPS 2025):
   https://arxiv.org/abs/2504.10612

2. Energy Discrepancies (2023):  
   https://arxiv.org/abs/2307.06431

3. Diffusion Recovery Likelihood (ICLR 2021):  
   https://arxiv.org/abs/2012.08125

4. Cooperative DRL (2023):  
   https://arxiv.org/abs/2309.05153

5. Should EBMs model energy or score? (OpenReview):  
   https://openreview.net/forum?id=9AS-TF2jRNb

6. Multiscale DSM for High-D EBMs (Entropy 2023):  
   https://www.mdpi.com/1099-4300/25/10/1367

7. Flow Matching (ICLR 2023):  
   https://arxiv.org/abs/2210.02747

8. Rectified Flow (ICLR 2023):  
   https://arxiv.org/abs/2209.03003

9. Consistency Models (ICML 2023):  
   https://arxiv.org/abs/2303.01469

10. PRDC metrics:  
    https://arxiv.org/abs/2002.09797  
    https://github.com/clovaai/generative-evaluation-prdc

11. C2ST references:  
    https://arxiv.org/abs/1610.06545  
    https://proceedings.mlr.press/v108/kirchler20a.html

12. MMD reference:  
    https://www.jmlr.org/papers/v13/gretton12a.html

---

## 14) Stage1.5 Hard Validation (Pass 12, 2026-03-26)

### 14.1 Critical runtime blockers found in previous `train_stage1_5.py`

1. Non-runnable API mismatches:
   - `Stage1Config` type used in Stage1.5 function signature.
   - `config.get(...)` used on dataclass object.
   - `evaluate_denoising(...)` called but not defined/imported.
   - `check_conditional_kill(...)` imported but absent in `kill_criteria.py`.
2. Incorrect model API usage:
   - `UnconditionalEnergy` called as `prior_critic(v, sigma=...)` (unsupported).
3. Data path mismatch:
   - Loop expected `batch["v"]`, but `SONARVectorDataset` returns tensor.
4. Loss API mismatch:
   - `multiscale_dsm_loss(...)` called with unsupported args (`v_noisy`, `sigma`, `force_fp32`).
   - `gradient_penalty(...)` called with unsupported signature (`lambda_gp=` kwarg).
5. Config/schema mismatch:
   - JSON carried metadata + nested kill block + fields absent in dataclass.

### 14.2 Fixes implemented

1. Replaced Stage1.5 training script with a runnable and internally consistent pipeline:
   - `experiments/01_denoising_poc/train_stage1_5.py`
2. Added strict config normalization for legacy JSON:
   - metadata stripping (`_comment`, `_version`, ...),
   - nested `kill_criteria` mapping,
   - `sigma_weighting` bool -> string normalization.
3. Enforced objective consistency:
   - critic: `MDSM + ranking + optional CQL/GP/shell`,
   - actor: geodesic + alignment + BC + clean-barrier (`softplus(E_clean - E_actor)`),
   - no contradictory `E(actor) < E(clean)` actor term.
4. Unified evaluation semantics with shared Langevin:
   - uses `run_langevin(...)`,
   - reports terminal endpoint (`v_last` if present),
   - strict pass/fail from `summarize_conditional_eval(...)`.
5. Hardened training safety:
   - non-finite guards,
   - gradient sanitation (`nan_to_num`),
   - grad clipping,
   - LR backoff on bad-batch streaks,
   - fail-fast after max consecutive bad batches,
   - optional loss-spike guard.
6. Updated config schema to match code:
   - `configs/base.py::Stage1_5Config`
   - `configs/stage1_5_config.json`
7. Added stability parity for prior critic:
   - `UnconditionalEnergy.forward()` now clamps `log_energy_scale` before exponentiation.
8. Fixed optimizer parameter-group semantics:
   - Stage1.5 now uses separate LR groups for conditional critic and optional prior critic
     (`critic_lr` and `prior_critic_lr` both become effective, no silent override).

### 14.3 Remaining architectural gap vs target pairwise proposal/refinement design

Current Stage1.5 is now internally consistent and debuggable, but still not full Stage2 pairwise semantics:

1. Stage1.5 still uses teacher-forced denoising setup (`v_query == v_clean`) for supervision.
2. True proposal/refinement path for Stage2 requires:
   - query/context-driven actor proposals without direct clean exposure,
   - batchwise/hard negative conditioning,
   - support-aware trust region and retrieval-grounded metrics.
3. Hybrid prior (`E_prior`) exists as optional branch and should remain auxiliary, not primary relevance signal.

---

## 15) Stage1.5 Completion Pass (Pass 13, 2026-03-26)

### 15.1 Closed gaps

1. `critic_steps_per_actor` now changes runtime behavior:
   - real inner-loop critic updates (no forced fallback to 1).
2. Teacher-forced setup removed:
   - training now uses retrieval pairs `q -> v_pos` instead of `q == clean target`.
3. Twin conditional critic implemented:
   - separate `critic1`, `critic2`, with hybrid aggregation (`max`/`mean`) in actor/inference path.
4. Retrieval-conditioned hard negatives implemented:
   - positives from top-k neighbors with self-match exclusion,
   - hard negatives sampled from deeper retrieval window.

### 15.2 8GB VRAM stabilization changes

1. Memory peak reduction:
   - per-step update of a single critic branch (alternating c1/c2) instead of one joint second-order graph for both.
2. OOM guard:
   - `torch.OutOfMemoryError` catch + `torch.cuda.empty_cache()` + bad-batch skip/backoff path.
3. Safer defaults:
   - `batch_size=32`,
   - `ortho_n_iters=4`,
   - reduced eval load (`eval_num_samples=64`, shorter eval Langevin rollout).

### 15.3 Performance interpretation

- Long epoch time in logs is expected under:
  - second-order MDSM (`create_graph=True`),
  - orthonormal layers with Björck iterations,
  - twin critics + retrieval sampling + per-epoch evaluation.
- Runtime is compute-bound and memory-sensitive on 8GB GPUs; stabilization changes above are necessary baseline.

---

## 16) Stage1.5 Math Upgrades (Pass 14, 2026-03-26)

### 16.1 Implemented

1. Smooth twin critic aggregation:
   - added `softmax` aggregate (`tau * logsumexp(E_i/tau)`) with temperature control.
   - avoids non-smooth gradient switching of hard `max`.
2. Conditional NCE objective:
   - added InfoNCE-style loss over `(positive, hard negative, random negatives)` per query.
   - integrated as additive critic term (`lambda_nce`), with optional prior-NCE (`lambda_prior_nce`).
3. Strict retrieval self-exclusion:
   - positive/hard candidate selection can exclude exact same sample by index, not just cosine threshold.
4. Tangent-noise Langevin:
   - added optional tangent projection for stochastic noise in overdamped/PID/underdamped updates.
   - exposed through `run_langevin(..., tangent_noise=...)` and Stage1.5 config.

### 16.2 Rationale

- Smooth aggregator improves gradient continuity for actor alignment and Langevin refinement.
- NCE complements hinge ranking by calibrating relative energies over small candidate sets.
- Index-level self-exclusion prevents trivial retrieval leakage.
- Tangent-noise keeps stochastic exploration closer to sphere geometry assumptions.

## 2026-03-26 Stage1.5 optimization re-audit (post-modification)

Primary bottlenecks confirmed:
1. Bjork orthonormalization in-forward remains dominant compute/memory cost under second-order MDSM.
2. Actor using orthonorm multiplies overhead; critic and actor can be decoupled by norm mode.
3. Retrieval path had Python `.item()` synchronization pressure; vectorization is next high-impact step.
4. Eval jitter from changing sample subset can bias checkpoint selection (now fixed via deterministic eval subset).

External references checked:
- CVPR 2024 benchmark comparing 1-Lipschitz layers and practical guidelines:  
  https://openaccess.thecvf.com/content/CVPR2024/papers/Prach_1-Lipschitz_Layers_Compared_Memory_Speed_and_Certifiable_Robustness_CVPR_2024_paper.pdf
- PyTorch orthogonal parametrization docs (`matrix_exp` / `cayley` / `householder` tradeoffs):  
  https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.orthogonal.html
- PyTorch `torch.compile` troubleshooting for graph-break profiling and selective disable:  
  https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html
- PyTorch CUDA memory-management environment knobs (`PYTORCH_CUDA_ALLOC_CONF`):  
  https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management

Actionable next optimization queue:
- Introduce Stage1.5 orthonorm schedule (`4->2->1`) with strict eval gates.
- Add actor-specific norm mode (`spectral_norm`) while keeping critic orthonorm.
- Vectorize retrieval hard/positive selection to remove Python-side sync loops.
- Batch eval Langevin path to reduce validation overhead.
