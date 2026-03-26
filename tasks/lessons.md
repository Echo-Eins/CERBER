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
