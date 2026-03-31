# Lessons

## 2026-03-31 - "Twin critic" must not be mislabeled as radial+angular without explicit specialization

### Pattern
User requested a true radial+angular twin-critic architecture. Current Stage1.5 had two homogeneous critics (`SimpleEnergy` + same features/losses), which is an ensemble, not geometric decomposition.

### Rule
1. Never call architecture "radial+angular" unless critics are explicitly specialized by design.
2. Radial critic must consume radius/displacement features (or equivalent) and be supervised by radial objectives.
3. Angular critic must consume normalized/geodesic features and be supervised by angular objectives.
4. Trainer/eval must report per-head metrics (radial vs angular), not only aggregated twin score.

### Verification checklist before claiming radial+angular
1. Distinct critic modules/classes or distinct head pathways exist in code.
2. Distinct loss terms are active and mapped to respective heads.
3. Aggregator mixes the two heads in inference and training consistently.
4. Ablations can independently disable radial or angular head.

## 2026-03-29 - CRITICAL: Never change multiple variables at once (Phase 2f post-mortem)

### Pattern
Phase 2f changed 5 things simultaneously from Phase 2b: norm_mode (none→orthonorm), activation (silu→groupsort), critic_lr (0.001→0.0003), direction_loss (removed), energy_reg_universal (false→true). Result: total failure (rank_success=0.001, spread=-0.001). Impossible to diagnose which change caused the collapse.

### Evidence
- Phase 2b (norm_mode=none, silu, lr=0.001): spread=0.56, rank_success=88%, E range [-0.03, 0.53]
- Phase 2f (orthonorm, groupsort, lr=0.0003): spread=0.002, rank_success=0.1%, E range [0.23, 0.36]
- 1-Lipschitz (orthonorm+groupsort) crushed energy capacity to 0.13 range (4× less than Phase 2b)
- energy_reg_universal=true was ALREADY known to kill ranking (Phase 2e lesson!)

### Rule
1. **ONE change per experiment**. If Phase 2b is the baseline, the next experiment changes ONLY lambda_mdsm
2. NEVER reuse a parameter combination that already failed (energy_reg_universal=true)
3. If an experiment fails, identify which single variable caused it before trying the next
4. Architecture changes (norm_mode, activation) are the MOST impactful — never combine with loss changes

## 2026-03-29 - Phase 2b training success ≠ inference success

### Pattern
Phase 2b achieved 88% rank_success, 0.56 spread, dir=0.643 in training. But strict inference evaluation: mean_cos_success=19%, cos_improvement at noise=0.05: -0.107 (NEGATIVE). Training metrics can look excellent while the actual Langevin navigation fails completely.

### Why
- Ranking trains energy VALUES at discrete training points
- Direction loss trains gradient DIRECTION at sampled noisy points
- Neither guarantees smooth gradient field BETWEEN training points
- Unconstrained MLP (norm_mode=none) creates wild gradients in unexplored regions
- At fine noise (0.05), Langevin is purely gradient-driven → navigates the untrained wild field
- At coarse noise (0.3), random walk component dominates → partially compensates bad gradients

### Rule
1. **Never trust training rank_success for inference quality** — always check strict Langevin eval
2. The gap between training and inference = gradient field smoothness problem
3. Solutions: (a) smooth the field (soft Lipschitz), (b) supervise the field (MDSM), (c) bypass the field (flow matching, score distillation)
4. Test at noise_scale=0.05 to expose gradient field quality (removes random walk compensation)

## 2026-03-29 - CRITICAL: Always enable gradient field supervision (MDSM) for Langevin dynamics

### Pattern
If Langevin dynamics follows -∇E, the gradient field ∇E MUST be explicitly trained. Ranking loss trains energy VALUES at specific training points but says NOTHING about the gradient field between them. Direction loss is a partial fix (teaches direction but not magnitude). Only full MDSM (denoising score matching) trains both direction and magnitude of ∇E.

### The Bug
`lambda_mdsm=0.0` was set in ALL configs across ALL phases. The gradient field was never trained. Langevin dynamics navigated an untrained gradient landscape. Changing the Langevin variant (overdamped, PID, underdamped) made zero difference because the underlying gradient field was the same untrained garbage.

### Root Cause of Confusion
- MDSM was labeled "last resort, unbounded MSE is dangerous" in the plan
- But directional mode (cosine similarity) IS bounded [0,2] — the danger only applies to L2 DSM
- This fear caused MDSM to be deferred indefinitely while other losses were tried
- Direction loss was treated as sufficient, but it only teaches direction, not magnitude

### Rule
**Never run Langevin inference without lambda_mdsm > 0 (or equivalent gradient field supervision).** Ranking/NCE/CQL/energy_reg are all VALUE-based losses — they cannot teach the gradient field. If the inference method uses ∇E, the training MUST include a loss on ∇E.

### Diagnostic Signs
- E[c/a/h] nearly identical (spread < 0.05) despite ranking loss converging
- Langevin goes AWAY from clean target (inverted landscape)
- noise_scale=0.5 works but noise_scale=0.0002 doesn't (random walk vs gradient-driven)
- PID and underdamped produce identical results (dynamics variant doesn't matter if gradients are untrained)

## 2026-03-29 - Removing ALL energy_reg causes scale inflation → crash (Phase 2f v1)

### Pattern
Phase 2f v1 disabled energy_reg entirely (λ=0). Without ANY scale anchor, unconstrained MLP energies grew exponentially: E=0.13 (epoch 1) → 2678 (epoch 11) → crash. Ranking hinge margins (0.1/0.05/0.15) are FIXED — at E=2678, margin 0.15 is 0.005% of scale → zero gradient → ranking stops teaching.

### Key Distinction
- **BAD**: `energy_reg_universal=true` + `interp_gp` (Phase 2e) — flattens separation
- **GOOD**: `energy_reg=true, λ=0.01, universal=false` (Phase 2b) — mild scale anchor on clean points only
- **BAD**: `energy_reg=false` entirely (Phase 2f v1) — unconstrained scale explosion

### Rule
1. **ALWAYS keep clean-only energy_reg at λ=0.01** as scale anchor (NOT universal)
2. **Enable rank_normalize_by_std=true** as second defense — makes margins relative to batch std
3. Never confuse "universal ereg kills ranking" with "clean-only ereg kills ranking" — they are different
4. Monitor E[clean] growth rate: if doubling every 2 epochs, scale is unconstrained

## 2026-03-29 - energy_reg_universal + interp_gp = plateau+cliff landscape that kills Langevin

### Pattern
Phase 2e combined `energy_reg_universal` (penalize E² at clean, actor, AND hard) with `interp_gp` (WGAN-GP along clean→hard corridor). Together they created a flat plateau (energies ~0) with steep cliff edges (wells to -300). Langevin dynamics stuck on plateau — gradient magnitude crushed by GP while direction_loss teaches only direction. Over 50 epochs: spread grew to only 0.198 (vs Phase 2b's 0.559), inference cosine improvement = -0.148 (negative!), success rate = 0%.

### Diagnosis Signals
- Energy landscape: range [-299, 0.4] with plateau + cliff visible in 3D plots
- `igp` growing every epoch (0.10→0.30) — GP punishment increasing = gradients being flattened more
- `ereg` growing (0.003→0.051) — energy regularization fighting ranking
- `dir` plateauing at 0.509 — direction learned but magnitude insufficient
- E[clean] → -0.01, near zero — ereg successfully crushed clean energy

### Rule
1. **NEVER combine energy_reg with ranking losses** — THIRD time this lesson is recorded (2026-03-28 twice, now again)
2. **NEVER use gradient penalty along the Langevin inference corridor** — it flattens exactly the gradients Langevin needs
3. If you need Lipschitz-like stability without killing capacity, use Tamed Langevin (inference-side fix) not GP (training-side cripple)
4. PID gains kp < 1.0 are dangerous with flat landscapes — standard kp=1.0 unless specific reason to dampen

## 2026-03-28 - Tamed Langevin as safety net for non-Lipschitz or poorly-conditioned gradients

### Pattern
Standard Langevin dynamics `v += -lr * grad_E + noise` assumes bounded gradients. When gradients explode (due to weak Lipschitz constraint, low ortho iterations, or out-of-distribution inputs), the step diverges catastrophically. Tamed Langevin (Benko et al., AAAI 2025) replaces raw gradient with `grad_tamed = grad / (1 + lr * ||grad||)`, automatically bounding the step size.

### Key Properties
- **Convergence guarantee**: Proven convergence even with superlinear (non-Lipschitz) gradients
- **Trivial implementation**: One line change in Langevin update
- **No architectural constraint**: Works with any energy network, no orthonormalization required
- **Magnitude-only**: Only bounds gradient magnitude, does NOT fix gradient direction
- **Compatible with sphere projection**: Taming happens before projection, so target_norm constraint still applies

### When to Use
1. As a **safety fallback** in all Langevin inference (costs nothing when gradients are already bounded)
2. If relaxing Lipschitz constraint (e.g., switching critic to spectral norm only)
3. If experimenting with unconstrained architectures (attention-based energy, etc.)

### When NOT Sufficient Alone
1. Taming does not help if gradient DIRECTION is wrong (model not trained well)
2. Does not replace proper noise_scale calibration (noise is not tamed)
3. Training stability still benefits from Lipschitz — taming is primarily an inference technique
4. Score matching loss targets can have wildly varying magnitudes without Lipschitz

### Implementation
```python
# In langevin.py, after computing grad:
grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
grad = grad / (1.0 + lr * grad_norm)  # tamed gradient
```

### References
- Benko et al., "Kinetic Langevin MCMC sampling without gradient Lipschitz continuity" (AAAI 2025)
- Also: "Langevin Monte Carlo Beyond Lipschitz Gradient Continuity" (J. Complexity, 2024)

## 2026-03-28 - PyTorch Cayley parametrization superior to Björck for our architecture

### Pattern
Custom Björck orthonormalization with 15 iterations costs ~30 matmuls per layer per forward pass, creates deep autograd graph under create_graph=True (MDSM loss), and provides only approximate orthogonality. PyTorch's built-in `torch.nn.utils.parametrizations.orthogonal` with `cayley` map provides exact orthogonality at ~3 matmul-equivalent cost.

### Rule
1. Prefer `torch.nn.utils.parametrizations.orthogonal(linear, orthogonal_map="cayley")` over custom Björck
2. For very rectangular matrices (e.g., 512×1), use `orthogonal_map="householder"`
3. Dynamic trivialization (built-in) improves optimizer convergence
4. Cayley cannot represent det=-1 matrices (eigenvalue=-1), but this is dense in O(n) — not a practical issue
5. When migrating: old Björck checkpoints need weight key remapping (parametrizations changes key structure)

### Evidence
- Björck-15: ~30 matmuls/layer, approximate, deep create_graph graph
- Cayley: ~3 matmul-equiv/layer, exact, shallow create_graph graph
- Sources: CVPR 2024 "1-Lipschitz Layers Compared", ICML 2019 "Cheap Orthogonal Constraints"

## 2026-03-26 - Björck ortho_n_iters=1 causes immediate rank degradation

### Pattern
Training logs epochs 30-50 proved that when ortho schedule drops to n_iters=1, rank_success immediately plummets (0.615→0.451 at epoch 33) and never recovers. The Björck orthonormalization needs at least 2 iterations to maintain the singular-value control required for stable MDSM gradients.

### Rule
1. NEVER allow ortho_n_iters < 2 in the schedule. Minimum is 2.
2. Both config default AND resolve_ortho_n_iters() must enforce floor of 2.
3. If training speed is a concern, optimize the Björck implementation itself rather than reducing iterations below 2.
4. When user provides training logs proving a parameter causes degradation, treat that as ground truth — don't debate it.

### Evidence
- Epoch 30 (n_iters=2): rank_success=0.615, rank(c<a)=0.878, viol=0.028
- Epoch 33 (n_iters=1): rank_success=0.451 (immediate DROP), metrics destabilized
- By epoch 50: rank_success never recovered above ~0.54

## 2026-03-26 - Sub-clean attractors cause cosine degradation despite energy improvement

### Pattern
Cosine similarity degrades (-30 to -60 range) while energy values improve. Root cause: critic creates energy minima BELOW clean target energy. Langevin dynamics follows gradient to these sub-clean attractors, overshooting past clean target. Energy looks good (lower = "better") but cosine to clean target worsens.

### Rule
1. Always include clean-minimum penalty: L = relu(E_clean - E_actor + margin) to prevent sub-clean attractors
2. Monitor viol rate (E_actor < E_clean) — this is the most direct indicator of this problem
3. Auto-inference with large noise (0.15) can mask this issue since starting point is far from clean; manual inference with small noise (0.05) exposes it
4. When energy improves but cosine degrades, suspect sub-clean attractors first

## 2026-03-26 - Do not claim proposal/refinement when training is still teacher-forced

### Pattern
User pointed out that Stage1.5 still had `query == clean` semantics, `critic_steps_per_actor` effectively disabled, and only partial twin-critic behavior despite claims of fuller architecture.

### Rule
Before marking Stage1.5+ as "proposal/refinement":
1) verify `query != positive target` in the actual batch construction,
2) verify `critic_steps_per_actor > 1` changes runtime behavior (not just config),
3) verify twin-critic aggregation is actually used in actor/inference paths,
4) verify hard-retrieval negatives are part of critic loss, not only Gaussian OOD.

## 2026-03-26 - Twin critic + MDSM on 8GB requires explicit memory budget

### Pattern
Runtime logs showed late-epoch CUDA OOM despite finite losses when both critics were trained in one second-order graph with Björck layers.

### Rule
For 8GB-class GPUs in Stage1.5:
1) update critic branches sequentially (not both in the same MDSM graph),
2) keep OOM-safe skip path (`empty_cache` + backoff),
3) reduce default eval cadence/load (`eval_every_epochs`, sample count, Langevin eval steps),
4) keep conservative baseline defaults (`batch_size`, `ortho_n_iters`, retrieval bank size) in config.

## 2026-03-23 - User requested repeated AGENTS reread and restart from scratch

### Pattern
When user explicitly says "read AGENTS again" or "restart research from scratch", partial continuation is not enough.

### Rule
Before continuing, immediately:
1) re-open AGENTS.md,
2) re-open core project docs (spec + implementation plan),
3) restate and execute a fresh plan,
4) rewrite research artifact (`research.md`) from a clean structure.

## 2026-03-23 - Context preservation requirement

### Pattern
User asked to persist findings if context approaches limit.

### Rule
Write findings incrementally to `research.md` during exploration, not only at the end of the run.

## 2026-03-23 - Scope discipline

### Pattern
User explicitly asked to read all Stage 1 modules.

### Rule
For architecture research tasks, always include code-grounded analysis of all modules in scope before proposing SOTA changes.

## 2026-03-23 - Non-finite handling in training loops

### Pattern
Guarding only the loss is insufficient: gradients can become non-finite while loss is finite, and one optimizer step can irreversibly corrupt parameters.

### Rule
For all training loops with second-order paths, always add:
1) finite-gradient checks before optimizer step,
2) fail-fast threshold for consecutive non-finite batches,
3) adaptive LR backoff on non-finite events,
4) orthonormalization stability guards (spectral pre-normalization + correct rectangular update form).

## 2026-03-23 - Backoff/warmup coupling trap

### Pattern
Applying LR backoff to both `lr` and `initial_lr` silently breaks warmup logic and can collapse training LR to near-zero after sparse warnings.

### Rule
When warmup uses `lr = initial_lr * warmup_factor`, non-finite backoff must never mutate `initial_lr`. Apply backoff only to current `lr` and preferably only after short consecutive non-finite streaks.

## 2026-03-23 - MDSM numeric floor requirements

### Pattern
Relative-noise DSM can hit non-finite values on low-norm embedding outliers due to tiny `sigma_eff_sq` and unstable cosine/log derivatives.

### Rule
For MDSM on real embedding corpora, always enforce norm/sigma floors and a safer cosine epsilon, and sanitize non-finite intermediates before reduction.

## 2026-03-23 - Objective/inference sign consistency

### Pattern
Energy can train to low loss but inference fails when the target field sign in training is inconsistent with Langevin update direction.

### Rule
For every new objective, explicitly verify sign consistency end-to-end:
1) target field definition,
2) whether model learns `grad(E)` or `score`,
3) inference update (`v <- v - lr*grad(E)` vs `v <- v + lr*score`).

## 2026-03-23 - Early-stop calibration in energy samplers

### Pattern
A fixed absolute `energy_threshold` can instantly stop refinement when energy scale changes during model iterations, causing no-op denoising.

### Rule
Default to `energy_threshold=None` unless the threshold is explicitly calibrated on current checkpoints; prefer plateau-based stopping as the safe default.

## 2026-03-23 - Directional DSM scale identifiability

### Pattern
With cosine-only directional MDSM (`directional=True`, `magnitude_aux_weight=0`), the loss is nearly invariant to global gradient scale. This makes `log_energy_scale` effectively unidentifiable and can freeze at initialization.

### Rule
When using directional MDSM, always either:
1) enable a non-zero magnitude auxiliary term, or
2) explicitly disable energy-scale optimization and treat inference step size as the only scale control.

## 2026-03-23 - Sigma weighting normalization

### Pattern
Applying `sigma2` weighting without normalization can shrink directional DSM loss magnitude by orders of magnitude, hiding weak optimization dynamics behind tiny scalar losses.

### Rule
Keep sigma weighting relative, but normalize weights to mean 1 before reduction so loss scale remains interpretable and gradients do not collapse numerically.

## 2026-03-23 - Respect non-minimal refactor requests

### Pattern
When the user explicitly allows non-minimal changes for architecture-level stability, continuing with incremental patches can delay the right solution.

### Rule
If user authorizes broad refactoring, prioritize a coherent end-to-end architecture upgrade over local patching; keep compatibility, but do not artificially constrain scope.

## 2026-03-25 - GUI/CLI parity and energy-range integrity

### Pattern
Web visualization diverged from CLI because GUI had its own denoising math, synthetic vector scaling, and permissive checkpoint loading (`strict=False` with fallback dims). This produced fake landscapes and hid real energy ranges behind hard clipping.

### Rule
For any visualization/debug path:
1) reuse the same inference core (`run_langevin`) and data/noise semantics as CLI,
2) never silently accept model/checkpoint mismatch; fail fast with explicit missing/unexpected keys,
3) avoid hard energy clipping in analysis paths; preserve real energy values unless user explicitly requests clipping.

## 2026-03-25 - Do not mix best-state with full trajectory in diagnostics

### Pattern
When sampler returns `v_best` but UI plots full executed trajectory, denoised marker/metrics can contradict the shown path (`improvement=0` while trajectory clearly moved).

### Rule
For interactive diagnostics, always align reported denoised state with the displayed trajectory endpoint (or explicitly display both with labels). Never compute user-facing improvement from a state that is different from the plotted endpoint.

## 2026-03-25 - Model-type-specific test semantics are mandatory

### Pattern
Applying cosine-to-clean as a primary metric to unconditional `E(x)` models leads to false failure conclusions.

### Rule
Branch evaluation by model type:
1) `simple` (conditional pairwise): denoising-to-clean metrics are primary.
2) `unconditional` (distributional EBM): energy descent + manifold/distribution metrics are primary; cosine-to-clean is diagnostic only.

## 2026-03-25 - Never silently fallback to synthetic reference in GUI diagnostics

### Pattern
If GUI cannot load dataset vectors and silently uses synthetic clean vectors, users can misread denoising/cosine outcomes as model failures.

### Rule
When fallback to synthetic reference is used:
1) report this explicitly in the inference output,
2) downgrade cosine-to-clean interpretation,
3) prefer energy-descent metrics for decision making.

## 2026-03-25 - Do not call input-gradient diagnostics under no_grad

### Pattern
`energy_and_grad(x)` for unconditional EBM crashed in GUI because diagnostic helper was wrapped with `@torch.no_grad`, disabling autograd graph construction for input gradients.

### Rule
For diagnostics that require `∂E/∂x`, never use `@torch.no_grad`; run them under `torch.enable_grad()` even in eval mode.

## 2026-03-25 - GUI summary metrics must be live, not only checkpoint-saved

### Pattern
For checkpoints without embedded metrics, GUI summary showed only `N/A`, hiding useful inference evidence and confusing quality assessment.

### Rule
Persist and display latest runtime inference metrics per checkpoint, update them on preview inference and every manual inference, and refresh summary in the same UI callback.

## 2026-03-25 - Callback wiring completeness for GUI feature extensions

### Pattern
Function signatures were extended (SOTA eval parameters), but not all Gradio event handlers passed new inputs. This left the feature partially integrated and behavior inconsistent.

### Rule
After changing any callback/function signature in GUI code:
1) update every event binding (`.click`, `.change`, `.submit`) that calls it,
2) add/verify matching UI controls exist for each new argument,
3) run compile checks and inspect handler arity end-to-end before considering integration complete.

## 2026-03-25 - Never equate 2D landscape proximity with 1024D improvement

### Pattern
User observed denoised point near target on contour plot while global cosine/energy did not improve. Root cause: plot shows projection to 2D scan plane; optimization metrics are computed in full latent space.

### Rule
For every GUI landscape run:
1) always report both projection-space diagnostics and full-space metrics,
2) explicitly label 2D values as projection-only,
3) avoid rounding tiny deltas to ambiguous zero; use scientific format fallback,
4) version metric caches when schema/metric definitions change.

## 2026-03-25 - Separate landscape resolution from landscape span

### Pattern
`Landscape Grid Size` was used as if it controlled visible area, while it only changes mesh detail. Users needed larger Direction 1/2 field-of-view but had no explicit span control.

### Rule
Always expose two independent controls in landscape UI:
1) resolution control (`grid size`) for sampling detail,
2) span control (absolute half-range or factor) for explored domain size.
When adding new scan parameters, propagate them through all callbacks and cache keys.

## 2026-03-25 - Keep endpoint semantics identical across live and batch evaluation

### Pattern
Live diagnostics used the trajectory endpoint, while batch SOTA evaluation could use `best-energy` fallback. This produced contradictory metrics (e.g., visible movement in live run but near-zero batch step norm).

### Rule
Store and propagate both states from samplers:
1) `v_last` = actually reached terminal state (default for user-facing/live/batch comparisons),
2) `v_best`/`v_final` = best-energy state (optional analytical metric).
Never mix these semantics across reporting paths.

## 2026-03-25 - Subagent synthesis must be code-verified before consolidation

### Pattern
Parallel subagent reports can be directionally correct but still require hard verification against the live codebase before they are promoted to project-level recommendations.

### Rule
Before writing a consolidated research artifact (`research*.md`) from subagent outputs:
1) verify every P0/P1 claim against concrete code lines,
2) separate "code-verified" findings from "external/SOTA hypotheses",
3) only then publish the final prioritized roadmap.

## 2026-03-25 - Kill criteria must be multi-gate and model-type aware

### Pattern
Single-threshold checks like `improvement > 0` and `success_rate > 50%` produced false-positive PASS verdicts and masked real model failures.

### Rule
For Stage1/Stage2 training verdicts:
1) never use one-metric kill criteria,
2) use strict multi-gate checks with model-type semantics:
   - conditional models: angular/geodesic + L2 + energy + clean-min violation gates,
   - unconditional models: energy descent + distribution/manifold gates (MMD/C2ST/PRDC),
3) select `best.pt` by composite score aligned to objective, not by paired cosine alone.

## 2026-03-25 - Actor-Critic objective alignment is critical (P0)

### Pattern
In `actor_critic` training, critic ranking loss (`E(clean) < E(actor)`) was combined with actor energy loss minimizing `softplus(e_actor - e_clean)` which pushes `E(actor) < E(clean)`. These gradients fight by construction, producing:
- Low energy but worse cosine
- "Improved" L2 but degraded manifold quality
- Sub-clean attractors in energy landscape

### Rule
For actor-critic architectures:
1) verify objective alignment before training: critic and actor must agree on energy ordering,
2) if critic ranks `E(clean) < E(actor)`, actor must NOT minimize `E(actor) - E(clean)`,
3) alternative: actor learns from final-state geometry (cosine/geodesic) + gradient alignment with `-∇E`,
4) always verify clean-min violation rate during eval as sanity check.

## 2026-03-25 - Unconditional sampler semantics must be unified

### Pattern
Different Langevin implementations used incompatible noise parameterizations:
- Path A: `noise = randn * noise_scale` + `sqrt(2*lr)*noise` → std = `sqrt(2*lr) * noise_scale`
- Path B: `noise = randn * sqrt(2*lr*noise_scale)` → std = `sqrt(2*lr*noise_scale)`

These differ by factor of `sqrt(noise_scale)` and can cause order-of-magnitude mismatches.

### Rule
For Langevin dynamics across codebase:
1) one canonical sampler backend (shared `cebcm/inference/langevin.py`),
2) single noise parameterization with explicit documentation,
3) verify train/eval/GUI produce identical trajectories with same seed,
4) never duplicate sampler logic; always re-export from central module.

## 2026-03-25 - Unconditional checkpoint selection requires manifold metrics

### Pattern
Selecting `best.pt` for unconditional `E(x)` by paired cosine improvement (denoising task) is category error:
- Unconditional model learns data distribution, not pairwise denoising
- Cosine to clean is diagnostic, not primary objective
- Checkpoint selection systematically picks wrong models

### Rule
For unconditional energy model checkpointing:
1) primary metric: energy descent success rate across noise scales,
2) distribution metrics: MMD, C2ST, PRDC (precision/recall/density/coverage),
3) manifold metrics: kNN proximity, shell deviation,
4) composite score for `best.pt` selection weighted toward manifold quality,
5) paired cosine only as secondary diagnostic.

## 2026-03-25 - Endpoint semantics must be consistent (v_last vs v_best)

### Pattern
Training evaluation used `v_final` (best-energy state) while live diagnostics used trajectory endpoint. This caused:
- Metric drift between train and eval
- Confusing "improvement=0" reports while trajectory clearly moved
- False-negative training verdicts

### Rule
For sampler result propagation:
1) `v_last` = actually reached terminal state (default for user-facing metrics),
2) `v_best` = best-energy state along trajectory (analytical metric only),
3) never mix semantics across reporting paths,
4) explicitly label which endpoint is used in each context.

## 2026-03-25 - Hybrid critic architecture (conditional + prior)

### Pattern
Pure conditional critic `E(q, x)` can learn relevance but lacks manifold awareness. Pure unconditional `E(x)` knows manifold but not query relevance. Using either alone leads to:
- Conditional: low energy but OOD drift
- Unconditional: good manifold but no query conditioning

### Rule
For robust Stage2/3 architecture:
1) hybrid energy: `E_total(q, x) = E_cond(q, x) + lambda_prior * E_prior(x)`,
2) train conditional as main relevance signal,
3) train unconditional as auxiliary manifold prior / OOD barrier,
4) actor proposes within support constraints,
5) short-run refinement uses combined gradient field.

## 2026-03-25 - Persistent contrastive term required for manifold calibration

### Pattern
NCE warmstart-then-EM-only training produced weaker manifold density calibration. Persistent equilibrium shaping (keeping contrastive term active during main phase) is required for:
- Proper mode weighting
- OOD penalty during training
- Stable MCMC chain behavior

### Rule
For unconditional energy training:
1) keep NCE/contrastive term active during main EM phase (not warmstart-only),
2) maintain persistent MCMC chains across batches (short-run MCMC),
3) add negative buffer replay for hard negatives,
4) monitor PRDC/C2ST as primary manifold quality indicators.

## 2026-03-26 - Stage-level claims must be runtime-validated, not just “implemented” on paper

### Pattern
`train_stage1_5.py` previously claimed SOTA completion but contained unresolved runtime/API mismatches (wrong config access, missing functions, incompatible loss/model signatures). The code compiled but was not executable as a coherent training path.

### Rule
Before marking any stage implementation as complete:
1) run a strict runtime-readiness audit (config schema, dataloader shape contract, model/loss signatures, eval hooks),
2) verify that declared features are actually reachable in code paths,
3) treat “py_compile passes” as syntax-only check, never as execution proof,
4) record unresolved runtime blockers explicitly in `tasks/todo.md`.

## 2026-03-26 - Stage1.5 telemetry labels must match actual metrics

### Pattern
Training console printed `rank=...`, but the value came from `rank_success`, not ranking loss.

### Rule
For every training metric:
1) ensure the printed label matches the exact tensor/statistic being logged,
2) if a metric has both loss and success-rate versions, print both with explicit names (`*_loss`, `*_success`),
3) avoid overloaded short labels that can be interpreted as objective values.

## 2026-03-26 - Fair checkpoint selection requires fixed eval subset

### Pattern
Sampling a new random eval subset each epoch adds score jitter and can select the wrong `best.pt`.

### Rule
For checkpoint scoring:
1) freeze a deterministic eval index subset at run start (seeded generator),
2) reuse the same subset for all periodic eval calls and final eval,
3) explicitly log when eval is skipped and keep schema stable (`status=not_evaluated`).

## 2026-03-26 - Never reuse epoch loop variable names for tensor metrics

### Pattern
In `train_stage1_5.py`, `ep` was used both as epoch index and as tensor `E(pos)`.
Python function scope allowed reassignment, so later boolean logic (`do_eval`) read a tensor and crashed with:
`RuntimeError: Boolean value of Tensor with more than one value is ambiguous`.

### Rule
For all training loops:
1) use explicit names for loop indices (`epoch_idx`, `batch_idx`, `critic_step_idx`),
2) reserve energy tensor names as `e_pos`, `e_actor`, `e_hard` (never `ep`, `ea`, `eh` if loop vars can collide),
3) run a post-edit grep/lint check for reused short symbols before launch,
4) treat tensor-vs-scalar name collision as P0 runtime blocker.

## 2026-03-26 - Live monitoring requests must include visualization parity, not only scalar metrics

### Pattern
User requested live monitoring improvements, but initial implementation covered only metric dashboards and not the requested 3D landscape parity with Checkpoint Analysis.

### Rule
For any "live monitor / web dashboard" request:
1) confirm whether user expects scalar charts only or full model-state visualizations (e.g., 3D landscape),
2) if 3D parity is expected, reuse the same rendering pipeline as checkpoint analysis (no simplified substitute),
3) add both manual trigger and policy-driven auto-refresh controls (e.g., every N epochs),
4) ensure checkpoint policy supports near-real-time visualization (rolling latest + periodic milestones).

## 2026-03-26 - Evaluation protocol must match training objective

### Pattern
Stage 1.5 conditional training used retrieval positives (query -> pos), while GUI SOTA eval measured self-denoise (query -> clean self). This produced misleading cosine failures despite energy descent.

### Rule
Before trusting eval metrics:
1) verify eval target distribution matches train target distribution,
2) report eval objective explicitly in UI/logs (`self_denoise` vs `conditional_retrieval`),
3) avoid hard-coded metric labels (`L2(clean,x)`) when target is dynamic,
4) treat objective/eval mismatch as P0 diagnostics bug.

## 2026-03-26 - Add imports for new exception types in the same patch

### Pattern
A new `except json.JSONDecodeError` branch was added without importing `json`, causing runtime NameError in Gradio callback.

### Rule
For every new symbol in exception handling or typed branches:
1) run a symbol import check before commit,
2) run at least one module `py_compile` after patching callbacks,
3) treat missing-import callback crashes as P0 regressions.

## 2026-03-26 - Structural objective conflicts beat raw loss reduction

### Pattern
Global training loss decreased while rank ordering quality (`rank_success`) collapsed and clean-min violation grew. This came from objective mismatch: actor could minimize geometry terms while falling into critic-invalid energy pockets.

### Rule
For actor-critic EBMs:
1) enforce explicit actor energy interval constraints relative to critic references,
2) add monotonic energy guard from actor seed,
3) never trust decreasing aggregate loss without rank/violation diagnostics,
4) treat retrieval-positive quality failures as objective corruption and fallback safely.

## 2026-03-26 - Sigma-conditioned critics require sigma parity in eval/inference

### Pattern
Stage1.5 trained critics with explicit sampled `sigma`, but eval/Langevin/inference paths omitted `sigma`, silently falling back to distance-estimated sigma. This made train objective and runtime navigation optimize different fields.

### Rule
For any sigma-conditioned energy model:
1) pass explicit sigma in train, eval, and inference consistently,
2) if runtime uses adaptive sigma, train with the same adaptive rule,
3) never mix fixed-sigma training with implicit-sigma inference without explicit ablation.

## 2026-03-26 - Compare tangent quantities to tangent quantities

### Pattern
Actor delta was tangent-projected, but alignment loss compared it to full-space `-gradE` including radial component, creating impossible targets.

### Rule
When tangent projection is enabled:
1) project all direction targets (gradients/noise/updates) to the same tangent plane before directional losses,
2) keep geometry/energy comparisons on the same manifold (project both states if needed).

## 2026-03-26 - Underdamped discretization must avoid hidden extra step-size factors

### Pattern
Underdamped update used momentum with `-lr*grad` and then multiplied momentum by `lr` again in position update, yielding unintended `lr^2` scaling in drift.

### Rule
For second-order samplers:
1) define one consistent discretization (velocity-like vs momentum-like state),
2) verify drift/noise scaling dimensions once in code and doc,
3) add parameter guards (`mass > 0`, friction range) and fail fast.

## 2026-03-27 - Langevin noise_scale in high-D destroys angular information

### Pattern
With `noise_scale=0.05` and `lr=0.001` in 1024D SONAR space (sphere radius ≈ 0.2051), the per-step noise norm was:
`sqrt(2 * lr * noise_scale) * sqrt(D) = sqrt(2 * 0.001 * 0.05) * sqrt(1024) ≈ 0.32`
This is 1.56x the sphere radius per step. After projection, each step was effectively a ~70° random rotation, completely destroying angular information in ~5 steps. Noise-to-signal ratio was 3.3:1 (noise_norm 0.32 vs gradient_step_norm 0.096).

With the GUI default (`noise_scale=0.15`), the ratio was 6:1 — even worse.

### Rule
1. ALWAYS compute effective noise norm in high-D before setting noise_scale: `noise_norm = sqrt(2 * lr * noise_scale * D)`
2. Effective noise norm must be << sphere radius (target_norm). A ratio of noise_norm/target_norm < 0.1 is safe.
3. For D=1024, target_norm=0.2051: noise_scale should be ~0.0002 (not 0.05).
4. When user reports "cosine always degrades to ~0", check noise scale FIRST — this is the most common cause.
5. GUI slider ranges must match safe operational ranges, not arbitrary [0, 1].
6. Training eval uses the same Langevin params — broken noise_scale makes eval results meaningless even if model learns correctly.

### Evidence
- noise_scale=0.05, lr=0.001: cosine 0.303→0.012 (destruction)
- noise_scale=0.05, lr=0.005: cosine 0.303→-0.053 (even worse, sqrt(lr) amplifies)
- Mathematical proof: noise_norm/sphere_radius = 0.32/0.205 = 1.56x per step

## 2026-03-27 - Energy scale inflation requires explicit regularization

### Pattern
Energy values grew from ~7 (epoch 30) to ~16 (epoch 50) without bound. Unbounded energy scale makes Langevin step sizes miscalibrated (gradient magnitude grows proportionally) and complicates hyperparameter tuning across checkpoints.

### Rule
1. Add energy scale regularization: `lambda_energy_reg * E(clean)^2` to prevent unbounded growth
2. Monitor absolute energy values across epochs — monotonic growth signals missing regularization
3. When energy at epoch N is 2x+ energy at epoch N-20, this is a P1 issue requiring intervention
4. Default lambda_energy_reg=0.01 is a light touch — increase if growth continues

## 2026-03-26 - Stage1.5 GUI must load twin critic, not critic1 fallback

### Pattern
Checkpoint analyzer fallback mapped Stage1.5 checkpoints to `model_state=critic1_state`; GUI then rendered/inferred with a single critic while training used twin aggregation, causing misleading landscape behavior.

### Rule
For multi-head/multi-critic checkpoints:
1) GUI/runtime evaluators must reconstruct the exact training aggregation graph,
2) keep fallback-to-single-model only for genuinely single-model checkpoints,
3) treat visualization/runtime architecture mismatch as P0 diagnostics defect.

## 2026-03-28 - Conflicting loss terms cause EBM training collapse

### Pattern
When an EBM critic has 8+ loss terms with antagonistic gradient directions, training enters an unstable equilibrium:
- **energy_reg (L2 penalty on E²)** fights ranking losses (which need energy separation)
- **direction_loss** duplicates MDSM gradient supervision → conflicting backprop signals
- **clean_min_penalty** duplicates ranking loss constraint → redundant gradient pressure
- Result: energy magnitudes oscillate chaotically, rank_success stuck at random (5%), inference cosine collapses

### Diagnosis Signals
- `ereg` growing exponentially (0.03 → 2.15 over 6 epochs) while energy range stays flat ([1.38, 1.60])
- `rank_success` stuck at ~5% (random chance for triple ordering)
- `rank(c<a)` FALLING — critic can't distinguish clean from actor
- `viol` INCREASING — more ordering violations over time
- Inference cosine degrading (0.57 → 0.04)

### Root Cause
energy_reg wants ALL energies → 0. Ranking wants E_clean < E_actor < E_hard with margins.
These are mathematically incompatible. The normalized ranking loss creates a moving target:
as energy_reg shrinks magnitudes, normalized margins also shrink, so ranking loss grows,
which pushes energies larger, which makes energy_reg grow → positive feedback loop.

### Fix Applied
1. **Disable energy_reg entirely** (λ=0.01 → 0.0) — energy should scale freely
2. **Disable direction_loss** (λ=0.15 → 0.0) — redundant with MDSM
3. **Reduce clean_min** (λ=0.3 → 0.05) — partially redundant with ranking
4. **Reduce margins** (0.2/0.1/0.3 → 0.1/0.05/0.15) — smaller targets, easier to satisfy
5. **Align sigma range** (0.5 → 0.3) — match training to eval distribution
6. **Soften twin temperature** (0.1 → 0.3) — smoother gradient flow through logsumexp

### Rule
- **Never add L2 energy regularization when using ranking losses** — they have antagonistic objectives
- **Never duplicate gradient supervision** (MDSM + direction = double supervision → conflict)
- **Start with MDSM + RANK only**, add auxiliary losses one at a time, verify each helps
- **Log E(clean)/E(actor)/E(hard)/spread** — monitor energy separation, not just loss values
- **Check that energy spread grows over training** — flat spread = critic not discriminating

## 2026-03-28 - Cayley + GroupSort collapses to constant function: spread=0.000

### Pattern
After removing conflicting losses (energy_reg, direction_loss), retrained from scratch with
Cayley orthogonal parametrization + GroupSort activation (strict L=1 Lipschitz). Result:
energy magnitudes grew (0.14 → 1.93 over 4 epochs) but **spread stayed at 0.000** — the
network outputs identical energy for clean, actor, and hard negatives. rank_success fell
from 13% to 0.9%. The critic learned a constant function scaled by `log_energy_scale`.

### Root Cause Analysis
1. **GroupSort preserves information but doesn't create new features**: GroupSort(2) outputs
   (max(a,b), min(a,b)) — an isometry that reorders but cannot create asymmetric nonlinear
   responses. For inputs on a sphere (SONAR R≈0.2051) that are close in L2, GroupSort can't
   amplify small differences into large energy differences.

2. **MDSM dominates rank loss (4:1 ratio)**: MDSM (λ=1.0) trains gradients at noisy points.
   Rank loss (λ=0.25) trains absolute energy values. With MDSM dominating, the network
   prioritizes correct gradient direction while energy VALUES collapse to a constant.

3. **rank_std_floor=0.001 causes gradient explosion during collapse**: When all energies are
   equal, std→0, clipped to 0.001. Normalized gradient = 1/0.001 = 1000x amplification.
   This creates chaotic updates that prevent recovery from the collapsed state.

### Diagnosis Signals
- `spread=0.000` — THE smoking gun. Energy values grow but spread stays zero
- `E[c/a/h]=1.93/1.92/1.93` — all three nearly identical
- `rank_success` falling (13% → 0.9%) — worse than random
- `viol` rising (33% → 56%) — ordering degrading
- `log_energy_scale` growing — network scales output but underlying function is constant

### Fix Applied
1. **Switch from Cayley+GroupSort to SpectralNorm+SiLU** — SiLU is not strictly 1-Lipschitz
   (L≈1.1) but can create asymmetric nonlinear features. Spectral norm gives soft L≤1 per
   layer. More expressive, can actually separate energies.
2. **Rebalance λ_rank: 0.25 → 1.0** — equal weight with MDSM so both objectives matter
3. **Increase rank_std_floor: 0.001 → 0.1** — cap normalization gradient at 10x instead of 1000x
4. **actor_step_size: 0.5 → 0.3** — more conservative steps for stability

### Rule
- **Monitor `spread` as primary health metric** — if spread=0 for >50 batches, architecture is wrong
- **Never let MDSM dominate rank loss** — they train different aspects (gradients vs values);
  keep λ_mdsm ≈ λ_rank
- **rank_std_floor must be ≥ 0.05** to prevent gradient explosion during energy collapse
- **When changing architecture, always start from scratch** — old checkpoints encode wrong
  energy landscape patterns

## 2026-03-28 - SpectralNorm+SiLU ALSO collapses: log_energy_scale is the root cause

### Pattern
After switching from Cayley+GroupSort to SpectralNorm+SiLU, the EXACT same collapse occurred:
spread=0.000, E[c/a/h] growing in lockstep (0.02→7.26 over 11 epochs), rank_success=0.000.
The architecture change was irrelevant — the real culprit is `log_energy_scale`.

### Root Cause
`log_energy_scale` is a learnable scalar `nn.Parameter(torch.tensor(0.0))` that multiplies ALL
energy outputs: `E = exp(log_scale) * net(x)`. This creates a fatal decoupling:

1. **MDSM wants large gradients**: target score `(noisy-clean)/σ²` has large magnitude.
   MDSM loss pushes `log_energy_scale` upward to match gradient magnitude → all energies inflate.
2. **RANK wants value separation**: `E_clean < E_actor < E_hard`. But `log_energy_scale` multiplies
   ALL outputs equally → spread stays exactly 0 regardless of scale.
3. **Result**: MDSM is happy (gradient direction correct, magnitude grows via scale), rank is dead
   (all values identical, just scaled up). Energy 0.02→7.26 but spread=-0.001 to +0.001.

The network's internal function `net(x)` outputs ~constant for all inputs. A global scalar cannot
fix this — it can only inflate the constant. The network has no incentive to create value
separation because MDSM doesn't require it.

### Diagnosis Signals
- Energy magnitudes growing monotonically across epochs (log_energy_scale learning)
- spread ≈ 0 throughout (net(x) is constant)
- rank_success = 0.000 (zero separation)
- rank(c<a) falling toward 0 (ranking impossible with equal energies)
- Same failure with Cayley+GroupSort AND SpectralNorm+SiLU → architecture-independent

### Fix Applied
- **Freeze log_energy_scale**: change from `nn.Parameter` to `register_buffer` (scale=1.0 fixed)
- Forces the network weights themselves to learn energy separation
- MDSM must achieve gradient quality through weight updates, not through a global scalar shortcut

### Rule
- **NEVER use a learnable global energy scale in EBMs with mixed MDSM+ranking losses** — it creates
  a shortcut where MDSM inflates the scale while ranking gets zero gradient signal
- **Learnable scalars that multiply outputs are dangerous** — they decouple gradient-based and
  value-based losses, making one trivially satisfiable without helping the other
- **If spread=0 persists across architecture changes**, the problem is NOT the architecture —
  look for global parameters (scales, biases) that affect all outputs uniformly

## 2026-03-28 - MDSM second-order gradients dominate ranking first-order gradients

### Pattern
Even after freezing `log_energy_scale`, SpectralNorm+SiLU with `lambda_mdsm=1.0` and
`lambda_rank=1.0` still produces spread=0.000 over 7+ epochs. Energies grow identically
for clean/actor/hard (E[c/a/h]=2.61/2.60/2.61). ibnce=3.466=ln(32) confirms the network
is a near-constant function over candidate inputs.

### Root Cause
MDSM (directional, `create_graph=True`) produces **second-order gradients** (Hessian-vector
products) that dominate ranking's first-order gradients in the shared parameter space:
1. MDSM loss ≈ 0.7-1.0, ranking loss ≈ 0.3, but MDSM gradient magnitude is amplified
   10-100x by the second-order chain rule through the Hessian
2. MDSM trains gradient DIRECTION at noisy points — does NOT require energy VALUE separation
3. Ranking trains value ordering E_clean < E_actor < E_hard — needs value separation
4. MDSM's dominant gradient reshapes the network faster than ranking can establish separation
5. Result: network learns correct local gradient directions but constant energy values

### Diagnosis Signals
- rank_loss ≈ 0.3 = sum(margins) = constant — all hinge terms ALWAYS active but can't move weights
- ibnce = 3.466 = ln(batch_size) — random chance, zero discrimination
- rank(c<a) dropping from 0.882 to 0.095 — getting WORSE over training
- E[c/a/h] growing but always equal — network depends on (q, σ), ignores candidate
- Same pattern with Cayley+GroupSort AND SpectralNorm+SiLU AND frozen log_energy_scale

### Fix Applied
- **MDSM warmup curriculum**: `mdsm_warmup_epochs=5` — first 5 epochs pure ranking
  (effective_lambda_mdsm=0.0), then linear ramp to target lambda_mdsm
- Gives ranking loss exclusive access to network weights initially
- Once energy separation is established (spread > 0), MDSM can refine gradient directions
  without destroying the ranking structure

### Rule
- **NEVER combine second-order (MDSM/score matching) and first-order (ranking/contrastive)
  losses from the start** — second-order gradients will dominate and prevent value learning
- **Always use a warmup curriculum** when mixing gradient-matching and value-matching objectives
- **Monitor MDSM and ranking loss separately** — if ranking loss stays constant while MDSM
  decreases, MDSM is dominating the gradient
- **ibnce = ln(batch_size) is a red flag** — means the network is effectively constant

## 2026-03-28 - Unconstrained MLP dramatically outperforms Lipschitz-constrained architectures for ranking

### Pattern
Cayley+GroupSort (exact orthogonal + 1-Lipschitz activation) makes optimization on the orthogonal manifold extremely slow: 1700 sec/epoch vs 10 sec/epoch unconstrained, and spread grows from 0.001→0.006 over 5 epochs (4% of needed separation). Plain nn.Linear + SiLU with 10x higher lr (1e-3 vs 1e-4) achieves rank_success=0.569, spread=0.236 in 10 epochs.

### Evidence
| Config | 5 epochs | 10 epochs | sec/epoch |
|--------|----------|-----------|-----------|
| Cayley+GroupSort, lr=1e-4 | spread=0.006, rs=0.265 | N/A | 1700 |
| None+SiLU, lr=1e-3 | spread=0.100, rs=0.445 | spread=0.236, rs=0.569 | 10 |

### Rule
1. Start with unconstrained MLP (norm_mode="none", activation="silu") for all new experiments
2. Only add Lipschitz constraints AFTER ranking is established and for specific reasons (inference stability)
3. Higher lr (1e-3) is critical for unconstrained — orthonorm constrains the landscape, plain Linear needs faster exploration
4. If inference Langevin diverges with unconstrained critic, use Tamed Langevin as safety net instead of constraining architecture

## 2026-03-28 - Auxiliary losses actively destroy ranking when added simultaneously

### Pattern
With 8+ auxiliary losses active (CQL, NCE, in-batch NCE, clean_min, support, barrier, descent, bc_reg, geo, align), rank_success DECREASES over training (0.206→0.071). Each loss competes for gradient space, and the combined signal overwhelms the ranking objective.

### Evidence
- Pure ranking only: rank_success 0.247→0.569 over 10 epochs
- All losses active (stage1_5_config.json): rank_success 0.206→0.071 (WORSE than random)
- Disabling all but ranking immediately fixed training

### Rule
1. **NEVER activate all losses simultaneously** — start with ranking only, add ONE loss at a time
2. Each new loss must be validated: rank_success must not drop more than 5% when added
3. If rank_success drops when adding a loss, the loss weight is too high OR the loss is fundamentally conflicting
4. Prioritize losses by their direct contribution to the end goal (inference quality), not by theoretical appeal
5. The "kitchen sink" approach to losses is an anti-pattern — more losses ≠ better training

## 2026-03-28 - rank_normalize_by_std + clip_grad_norm creates gradient bottleneck

### Pattern
When `rank_normalize_by_std=true` and `rank_std_floor=0.01`, the normalization amplifies gradients by ~100x (dividing by a small std). Combined with `clip_grad_norm=1.0`, the effective learning rate becomes lr/100, making ranking unable to learn.

### Rule
1. Disable `rank_normalize_by_std` unless there's a specific reason (e.g., highly varying energy scales)
2. If normalization is needed, use std_floor ≥ 1.0 or adjust clip_grad_norm proportionally
3. Always check effective gradient magnitude after normalization + clipping

## 2026-03-28 - Ranking teaches VALUES not GRADIENTS — direction_loss is essential for Langevin inference

### Pattern
Ranking loss (triplet hinge) teaches E(clean) < E(actor) < E(hard) — correct value ordering.
But Langevin inference follows -∇E, so it needs correct gradient DIRECTION, not just values.
Without gradient supervision, the energy surface between training points has arbitrary shape.
Result: Phase 1 (ranking + clean_min + energy_reg) gets rank_success=0.714 but cosine success=0.39%.

### Evidence
- Phase 1 (ranking only): cosine improvement = -0.232, success = 0.39%
- Phase 1.5 (+ direction_loss λ=0.3): cosine improvement = +0.011 (batch), success = 60.55%
- direction_loss: `L = (1 - cos(-∇E, clean - noisy)).mean()` — cosine-based, bounded [0,2]
- Direction loss is safer than MDSM: bounded output → bounded Hessian-vector products
- But direction_loss converges slowly: 0.76 → 0.69 over 20 epochs (max=2.0, random=1.0)

### Rule
1. **Always include gradient direction supervision** when training an energy function for Langevin inference
2. `direction_loss` (cosine) preferred over MDSM (MSE) because bounded output prevents gradient dominance
3. Ranking alone is never sufficient for inference — it only teaches at training points
4. If direction_loss stalls, increase its weight or train longer — do NOT add landscape-flattening losses (CQL, strong energy_reg)

## 2026-03-28 - CQL + strong energy_reg FLATTEN the energy landscape and suppress direction_loss

### Pattern
CQL penalizes low energy on OOD points (`softplus(-E_ood)`), and strong energy_reg penalizes `E_clean²`.
Together they push ALL energies toward zero, creating a flat landscape with weak gradients.
Direction_loss needs strong gradients to teach direction — flattening destroys its signal.

### Evidence
- Phase 1.5 (direction_loss only): cosine success = 60.55%, batch improvement = +0.011
- Phase 2 (+ CQL λ=0.1, energy_reg λ=0.01→0.1): cosine success = 40.23%, batch improvement = -0.015
- Phase 2 direction_loss converged WORSE: 0.703 vs 0.688 (Phase 1.5)
- Phase 2 energy spread SMALLER: 0.339 vs 0.408 (Phase 1.5) — confirming flattening
- Phase 2 rank_success also dropped: 0.708 vs 0.747

### Rule
1. **Never add CQL or strong energy_reg alongside direction_loss** — they compete for landscape shape
2. energy_reg λ=0.01 is safe (prevents unbounded wells), λ=0.1 is too strong (flattens gradients)
3. CQL is designed for offline RL where Q-values explode — EBM ranking doesn't have that problem
4. When adding a new loss, check energy SPREAD — if it decreases, the loss is flattening the landscape
5. Test ONE change at a time. Phase 2 changed TWO things (CQL + 10× energy_reg) making diagnosis harder

## 2026-03-29 - Spectral norm is TOO restrictive for EBM ranking — kills all energy separation

### Pattern
Spectral norm bounds σ_max(W) ≤ 1 per layer. For a 4-layer MLP, total Lipschitz ≤ 1.
This means |E(x) - E(y)| ≤ ||x - y||. With SONAR norms ~0.2, max energy spread ≈ 0.
Result: spread=0.000, rank_success=0.000, model cannot learn ANY energy ordering.

### Evidence
- Phase 2c (spectral_norm, n_power_iterations=5): spread=0.000 from epoch 1 through 5+
- E[c/a/h] = 0.05/0.05/0.05 — perfectly flat, no separation at all
- rank(c<a) dropping: 0.955 → 0.198 (random chance, model can't distinguish)
- direction_loss still improving (0.734→0.646) — gradients CAN be learned, but have zero magnitude
- Phase 2b (none): spread=0.559, rank_success=0.884 — unconstrained works for ranking

### Rule
1. **Never use spectral_norm for EBM ranking** — it hard-caps energy range too aggressively
2. Lipschitz constraint spectrum: spectral_norm (too hard) → gradient_penalty (soft, tunable) → none (too free)
3. For Lipschitz control, prefer gradient penalty: penalizes ||∇E||² without hard-bounding capacity
4. If spread=0 after epoch 1, the constraint is too tight — don't wait for more epochs

---

## Lesson: Random probing is useless in 1024D (Phase 2h, 2026-03-30)

### Summary
Sampling 64-512 random points on the 1024D sphere NEVER finds structured energy wells.
efloor=0.000 for all 50 epochs despite deep wells (E=-31) in the landscape.

### Pattern
In high-dimensional spaces (D=1024), the probability of a random point landing near
a structured energy well is essentially zero. Wells occupy negligible volume relative
to the sphere surface. Random probing is a low-dimensional intuition that fails at D>100.

### Evidence
- Phase 2h: energy_floor with 64 random sphere points → efloor=0.000 every epoch
- Energy landscape has wells at E=-31 (from Phase 2h analysis)
- Increasing to 512 random points still gives efloor=0.000

### Rule
1. **Never use random probing to find wells in high dimensions** — use adversarial probing (gradient descent) or Contrastive Divergence
2. CD (Langevin in critic loop) is the principled EBM approach: model's own dynamics find wells
3. Adversarial probing (gradient descent from random starts) works because it FOLLOWS gradients into wells

---

## Lesson: Contrastive Divergence works for well suppression (Phase 2i, 2026-03-30)

### Summary
CD (run Langevin in critic loop, push up energy at endpoints) successfully suppresses
spurious wells without hurting ranking. Combined with adversarial probing and underdamped
inference, achieves 100% cosine success at noise=0.15.

### Evidence
- Phase 2i: cd≈0.003, efloor≈0.005 at end of training (both active and >0)
- rank_success=89.5%, spread=0.674 (no regression from Phase 2g)
- noise=0.15: 100% cosine success (+0.332 improvement)
- Wells shallower than Phase 2h (E=-31 → much less)

### Rule
1. CD is safe to combine with ranking + MDSM + direction_loss
2. Use softplus penalty with threshold (E < -5 only), NOT penalizing all E<0
3. CD threshold must be well below training energy range (E[c]≈-0.03) to avoid conflicting with ranking

---

## Lesson: Training σ range must cover inference σ (Phase 2i→2j, 2026-03-30)

### Summary
If training uses σ∈[0.01, 0.3] but inference runs at σ=0.0002, the critic
has never learned scores at that noise scale. MDSM teaches ∇E only for
trained σ range. At untrained σ, gradients are extrapolation noise.

### Evidence
- Phase 2i: noise=0.15 (within training range) → 100% success
- Phase 2i: noise=0.0002 (50x below training min) → 64% success
- Energy goes to -1.456 at low noise — critic creates untrained wells at fine scale
- σ-conditioned critic passes σ to all energy evaluations — at unseen σ, output is undefined

### Rule
1. **sigma_curriculum_start must be ≤ inference noise_scale** (or close to it)
2. Sigma annealing at inference (NCSN-style) should stay within trained σ range
3. If extending σ range, check numerical stability: sigma_eff_sq clamped at 1e-6 prevents overflow
4. loguniform sampling naturally allocates density across scales — extending range costs minimal compute

---

## Lesson: GroupSort, NCE, CQL, inbatch_negatives — all disabled in successful phases (2026-03-30)

### Summary
Phase 2g (best baseline) and Phase 2i (best overall) both run with ALL of these disabled.
Enabling any of them violates single-variable discipline and risks known failure modes.

### Evidence
- **GroupSort**: Phase 2f collapse (rank_success=0.001). With orthonorm, creates 1-Lipschitz = spread≈0.
  Phase 2c (spectral_norm+groupsort): spread=0.000. Both successful phases use SiLU.
- **NCE**: "VALUE-based loss — cannot teach gradient field" (root cause analysis).
  Phase 2g/2i: lambda_nce=0.0, use_nce=false.
- **CQL**: Phase 2 regression (74%→40%). Flattens landscape like energy_reg.
  CD is strictly superior for the same purpose (finds real wells, not random OOD).
- **inbatch_negatives**: Value-based contrastive. Phase 2g/2i: lambda_inbatch_nce=0.0.

### Rule
1. **Do NOT enable groupsort** — kills energy range on unconstrained MLP
2. **Do NOT enable NCE/CQL** — value-based losses conflict with MDSM gradient supervision
3. **Do NOT enable inbatch_negatives** — same category as NCE
4. If ranking needs improvement, tune lambda_rank or margins, not add auxiliary contrastive losses

## Lesson: Extended σ curriculum is a dead-end due to sigma_eff_sq clamp (Phase 2j, 2026-03-31)

### Summary
Phase 2j extended training σ from [0.01, 0.3] to [0.001, 0.3]. Result: no improvement at low noise (65% vs 64%), slight regression at high noise. The sigma_eff_sq clamp at 1e-6 makes training at σ<0.005 mathematically impossible.

### Root Cause
```python
sigma_eff_sq = ((sigma * nrm) ** 2).clamp(min=1e-6)  # train_stage1_5.py:243
```
At σ=0.001, nrm≈0.2051: `(0.001 × 0.2051)² = 4.2e-8` → clamped to `1e-6` (24× inflation).
- MDSM target `tgt = displacement / sigma_eff_sq` is 24× smaller than correct value
- sigma2 weight `w = sigma_eff_sq = 1e-6` → near-zero contribution to loss
- Double suppression: wrong target AND near-zero weight = critic learns NOTHING at σ<0.005
- Additionally, loguniform over [0.001, 0.3] = 2.5 decades → less density per decade at important σ=[0.01, 0.3]

### Evidence
- Phase 2i (σ=[0.01, 0.3]): noise=0.0002 success 64%, noise=0.15 success 100%
- Phase 2j (σ=[0.001, 0.3]): noise=0.0002 success 65%, noise=0.15 regressed (cosine -0.115 less improvement)

### Rule
1. **Never extend σ below 0.005** with current sigma_eff_sq clamp — the samples are dead weight
2. If low-σ training is needed, fix the clamp first (adaptive floor or log-space MDSM formulation)
3. Wider σ range with loguniform = diluted training density — always check samples-per-decade
4. Extending training range is wrong lever when inference σ-conditioning doesn't match actual noise

## Lesson: Stronger CD improves well suppression but competes with direction learning (Option A, 2026-03-31)

### Summary
Option A (cd_num_samples=64, cd_num_steps=40, lambda_cd=0.3) improved low-noise success 64%→75% but direction loss regressed significantly (dir 0.77→0.62). CD and direction loss have conflicting gradient objectives at overlapping spatial regions.

### Root Cause
CD pushes energy UP at Langevin endpoints (wells). Direction loss teaches gradient DIRECTION at noisy points (σ-perturbed training data). When CD particles land near training points, CD wants to flatten the landscape there while direction loss wants specific gradient orientations. Stronger CD = more particles competing for the same gradient space = worse direction learning.

### Evidence
- Phase 2i (CD: 32 samples, 10 steps, λ=0.1): dir=0.77, noise=0.0002 success 64%
- Option A (CD: 64 samples, 40 steps, λ=0.3): dir=0.624, noise=0.0002 success 75.39%
- Energy success still only 3.91% — wells near clean target persist despite stronger CD

### Rule
1. CD has diminishing returns — going from λ=0.1→0.3 gives +11% cosine success but -0.15 direction quality
2. CD and direction/MDSM losses compete for the energy landscape shape near training points
3. If CD is increased, expect direction loss regression — monitor both metrics together
4. Energy success ~3% means clean target is NOT the energy minimum — CD alone cannot fix landscape topology

## Lesson: More Langevin steps = deeper well trapping, not better convergence (Option B, 2026-03-31)

### Summary
Option B (500 Langevin steps instead of 100) produced WORSE results: 60.55% vs 65% cosine success at noise=0.0002. More steps gives more time to fall into and get trapped in structural local minima.

### Evidence
- 100 steps: noise=0.0002 success ~65%
- 500 steps: noise=0.0002 success 60.55%, energy -1.121 (deeper than 100-step endpoint)
- Energy success 2.73% — still overwhelmingly falling into wrong wells

### Root Cause
The energy landscape has structural local minima near clean targets (E_well < E_clean in 97% of cases). With noise_scale=0.0002, Langevin is deterministic gradient descent. More steps = deeper descent into the nearest well. The wells are structural features of the unconstrained MLP, not noise artifacts.

### Rule
1. **Do NOT increase Langevin steps** as a fix for low-noise inference — it makes things worse
2. At noise_scale=0.0002, dynamics is purely gradient-driven → more steps = deeper well trapping
3. If 100 steps don't converge, the problem is landscape topology, not insufficient iteration
4. The only way more steps could help is with noise annealing (high→low) so early steps escape wells

## CRITICAL Lesson: σ-conditioning is semantically broken at inference (2026-03-31)

### Summary
During training, σ truthfully describes the sample's noise level: `noisy = pos + noise * σ * norm`. During inference, σ is a schedule value (geometric anneal from σ_max→σ_min) that has NO relation to the sample's actual distance from clean. The critic learned `score(q, x, σ=actual_noise_level)` but inference asks for `score(q, x, σ=schedule_value)`.

### Evidence
- Training: σ sampled from [0.01, 0.3], `noisy` is literally at distance σ×norm from clean
- Inference: σ_anneal=true anneals σ from 0.3→0.01, but Langevin noise_scale FIXED at 0.0002
- The sample's actual distance from clean is unknown and constantly changing
- At noise=0.15 (high Langevin noise): mismatch tolerable because random walk dominates
- At noise=0.0002 (deterministic): mismatch fatal because gradients depend entirely on σ-conditioning

### Root Cause
This is the NCSN/diffusion inference paradigm done incorrectly:
- NCSN anneals BOTH the σ-conditioning AND the sampling noise together
- Our system anneals σ-conditioning but keeps sampling noise fixed at 0.0002
- Result: critic receives σ=0.3 (early steps) but sample may be at distance 0.05 from clean → wrong gradients

### Rule
1. **σ-conditioning must match actual sample state** — either:
   a) Anneal Langevin noise_scale in sync with σ-conditioning (true NCSN sampling)
   b) Use distance-adaptive σ (sigma_schedule.py's "adaptive" mode) so σ reflects reality
2. Fixed noise_scale + annealed σ-conditioning = semantic lie → critic gives wrong gradients
3. High-noise success (100% at noise=0.15) is stochastic search DESPITE bad gradients, not gradient-guided
4. The most promising fix: anneal noise FROM 0.15 TO 0.0002 synced with σ, leveraging the 100% success regime

## Lesson: Energy success ~3% proves clean target is not energy-minimal (2026-03-31)

### Summary
Across ALL configurations (Phase 2i/2j/Option A/Option B), energy success rate at noise=0.0002 is 2.7-3.9%. This means the Langevin endpoint has HIGHER energy than the clean target in 96-97% of cases. The clean target is structurally not the energy minimum in its local neighborhood.

### Evidence
| Config | Energy Success | E_final |
|--------|---------------|---------|
| Phase 2i | low | -1.456 |
| Phase 2j | 2.73% | -1.606 |
| Option A | 3.91% | -1.104 |
| Option B | 2.73% | -1.121 |

### Root Cause
An MLP with [2048, 1024, 512] hidden dims and SiLU activation creates exponentially many local minima in 1024D. clean_min_penalty only sees actor outputs during training, not the full neighborhood. CD explores a vanishing fraction of 1024D per step. Wells that neither actor nor CD finds during training persist at inference.

### Rule
1. 3% energy success = the energy landscape is fundamentally wrong near clean targets
2. No amount of CD/efloor can exhaustively suppress wells in 1024D — it's a whack-a-mole problem
3. This points to an architectural limitation of unconstrained MLP for EBM in high dimensions
4. Potential fixes require architectural change: dual-critic decomposition, score distillation, or flow matching

## Process Lesson: Capture user diagnosis into TODO before deeper work (2026-03-31)

### Summary
When the user provides a concrete root-cause diagnosis and asks to continue, first convert that diagnosis into explicit checkboxes in `tasks/todo.md`, then proceed to analysis/implementation.

### Rule
1. User diagnosis/corrections are actionable requirements, not just discussion.
2. Immediately write a dedicated TODO block with traceable items and priorities.
3. Only after TODO capture continue with deeper technical analysis.

## Process Lesson: Validate active run/log source before diagnostics (2026-03-31)

### Summary
If the user states that the active logs are those pasted in chat, do not infer root-cause from a different local run artifact even if it exists in `logs/`.

### Rule
1. Before drawing conclusions, confirm the exact log source for this diagnosis: in-chat stream vs local file path.
2. If sources diverge, prioritize the user-provided active run and label local artifacts as potentially stale/different-run.
3. Reflect the active-run conclusions in `tasks/todo.md` before continuing implementation.
