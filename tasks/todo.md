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

### Validation status
- Runtime execution is not possible in this environment due unavailable Python runtime.
- Verification performed via static inspection and user-provided runtime logs.
