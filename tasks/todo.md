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

## Review
### Implemented files
- `configs/base.py`
- `cebcm/training/losses.py`
- `cebcm/inference/langevin.py`
- `experiments/01_denoising_poc/train.py`
- `experiments/01_denoising_poc/evaluate.py`
- `experiments/01_denoising_poc/visualize_landscape.py`

### Major outcomes
- Main Stage 1 bottlenecks addressed at implementation level:
  - Bjorck schedule hooks are now available,
  - training/evaluation are reproducible and config-consistent,
  - systems acceleration path is integrated,
  - Langevin and visualization correctness gaps are fixed.

### Validation status
- Runtime execution was not possible in this environment due unavailable Python runtime.
- Code was validated via static inspection only.
