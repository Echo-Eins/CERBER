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
