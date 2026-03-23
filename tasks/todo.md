# SOTA Research Plan - CEBCM/CERBER (2026-03-23)

## Goal
Run a deep SOTA research pass for CEBCM/CERBER focused on:
1) faster training,
2) more predictable convergence,
3) lower risk of converging to wrong behavior.

## Checklist (Pass 2 - extended)
- [x] Re-read updated `AGENTS.md`
- [x] Re-read full specification and implementation plan
- [x] Read code of all Stage 1 modules
- [x] Re-open existing `research.md`
- [x] Extend SOTA for simulation-free / few-step EBM alternatives
- [x] Extend SOTA for sampler acceleration and distillation pipelines
- [x] Extend SOTA for latent reasoning models beyond baseline JEPA/LCM references
- [x] Extend SOTA for long-context memory efficiency frontier
- [x] Extend single-GPU optimization stack with concrete implementation recommendations
- [x] Append all new findings to `research.md` with source links
- [x] Finalize updated review section

## Review
### Completed in pass 2
- Added deep addendum to `research.md` (sections 12-14).
- Expanded SOTA coverage with additional frontier lines:
  - simulation-free / near-simulation-free samplers,
  - few-step / one-step distillation families,
  - language-specific EBM diffusion (EDLM),
  - SONAR-native LM direction,
  - long-context memory frontier (Samba, Gated DeltaNet, RWKV-7, Titans/Atlas/Kimi).
- Added practical single-GPU optimization order grounded in official docs.
- Added updated decision framework for when to pivot away from incremental Stage 1 tuning.

### Current status
- Research document now contains both strategic and execution-level recommendations.
- Ready to move from research to implementation of Phase 0 speed stack.
