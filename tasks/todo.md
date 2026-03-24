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
