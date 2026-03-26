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

## 12) Sources (Primary)

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

