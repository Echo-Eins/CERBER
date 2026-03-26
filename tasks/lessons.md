# Lessons

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
