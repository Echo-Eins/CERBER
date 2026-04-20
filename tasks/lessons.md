# Lessons

## 2026-04-20 - TRUE root cause: nan_to_num(0) "loss bomb" in 6 code paths

### Pattern
Deep re-verification (Opus re-audit) found that the documented "nan_to_num(x, nan=0.0) is a loss bomb" rule from 2026-04-11 was NEVER applied to the actual loss computation code. The same anti-pattern existed in 6 places, creating a positive feedback loop that accelerated the NaN cascade in both GB10_1 and GB10_2.

### Root Cause: The NaN Cascade Positive Feedback Loop

**Exact mechanism (traced line-by-line):**
1. `generate()` line 1244: `raw_next = output_proj(x[:,-1:,:])` — can contain NaN (bfloat16 overflow at cold positions)
2. `generate()` line 1268: `_sphere_project(raw_next)` calls `_safe_normalize` which does `nan_to_num(x, nan=0.0)` BEFORE normalizing — output is ALWAYS finite
3. `generate()` line 1373 (OLD): NaN guard checked `next_vec_for_chain` (post-projection, already clean) — **DEAD CODE, never fired**
4. `generate()` line 1365 (OLD): `generated.append(raw_next)` stored NaN-contaminated raw vector
5. `compute_composite_objective` line 1210 (OLD): `nan_to_num(v_roll, nan=0.0)` replaced NaN with **zero vector**
6. `_masked_step_losses` line 332 (OLD): `nan_to_num(pred, nan=0.0)` — second layer of same bug
7. Loss computation: `cos_sim(0-vector, GT) = 0` → `cos_loss = 1 - 0 = 1.0` per NaN position — the **"loss bomb"**
8. Gradient: `nan_to_num` backward = 0 for NaN entries → NaN-producing parameters get **zero gradient** → they **never self-heal**
9. Shared parameters updated from healthy positions → **worsens cold positions** → MORE NaN → positive feedback

**Contrast with forward() (which was CORRECT all along):**
- `forward()` NaN gate: NaN → replace with GT → loss ≈ 0 → grad ≈ 0 ✓ (no training signal from broken step)
- `generate()` path: NaN → replace with 0 → cos_loss = 1.0 → inflated loss metric ✗

**The same nan_to_num(0) anti-pattern existed in 6 places:**
1. `train_chain_generator.py:1147` — `v_tf` sanitization
2. `train_chain_generator.py:1210` — `v_roll` sanitization
3. `train_chain_generator.py:332` — `_masked_step_losses` pred sanitization
4. `train_chain_generator.py:580` — `_masked_weighted_step_losses` pred sanitization
5. `chain_generator.py:1373` — dead NaN guard (checked already-clean vec)
6. `chain_generator.py:1365` — stored NaN-contaminated raw_next in generated[]

### Fixes Applied
1. **v_tf / v_roll**: `nan_to_num(x, nan=0.0)` → `torch.where(~isfinite(x), chains_GT, x)` — loss ≈ 0 not 1.0
2. **_masked_step_losses / _masked_weighted_step_losses**: `nan_to_num(pred, nan=0.0)` → `torch.where(~isfinite(pred), target, pred)`
3. **generate() NaN guard**: now checks `raw_next` (pre-projection), stores `clean_next_vec` when NaN detected instead of NaN-contaminated `raw_next`

### Rules
1. **NEVER replace NaN predictions with zero vectors.** Always replace with GT (target). `cos_sim(0, GT) = 0 → cos_loss = 1.0` is a loss bomb. `cos_sim(GT, GT) = 1.0 → cos_loss = 0` is correct (no signal from broken step).
2. **Check for NaN BEFORE `_sphere_project` / `_safe_normalize`.** These functions silently clean NaN via `nan_to_num(x, nan=0.0)` before normalizing. Checking the output never detects the original NaN.
3. **Every nan_to_num in the loss path must be audited.** If the replacement value participates in loss computation, it MUST match the supervision target, not be 0.

---

## 2026-04-19 - Architecture audit: 10 bugs found across GB10_1 / GB10_2 experiments

### Pattern
Full architecture audit triggered by GB10_1 (no-SADT/DF/EMA/softSS exp_conf) and GB10_2 (orig_conf) training runs. GB10_1 showed excellent System2 learning (tf_cos 0.46→0.63 over 3200 steps) before zombie guard killed it. GB10_2 catastrophically collapsed in <200 System2 steps due to norm explosion. Deep code review found 10 issues.

### Root Causes & Bugs Found

**BUG 1 (CRITICAL): `beam_generate()` appended SONAR-scale vectors to residual-scale chain**
- Lines 1671-1679: chain was built as [start_token(residual, norm~32), cand_vecs(SONAR, norm~0.2051), ...]
- 156x scale mismatch caused degenerate attention weights across beam steps
- FIX: scale candidates via `_to_residual_space(cand_vecs)` before `torch.cat`.

**BUG 2: Dead code in `generate()` DDIM path**
- Line 1238: `raw_next = ddim_result / target_norm * output_proj.weight.data.norm()` was immediately overwritten on line 1240 by `raw_next = ddim_result`
- Dead computation with side effect of calling `.data.norm()` (breaks autograd assumptions)
- FIX: removed the dead line.

**BUG 3: Zombie guard killed genuine learning (GB10_1 crash)**
- `hard_fail_zero_grad_streak=25` fired at E3 S3200 while `tf_cos=0.63` was still improving (from 0.46 at epoch start)
- NaN gate was correctly handling the NaN outputs; zero_grad_streak accumulated from NaN-gated batches where all outputs were replaced with GT → loss≈0 → grad≈0 but model state was fine
- FIX: added `enable_zombie_guard` config flag (default `true` = old behavior). Set to `false` in exp_conf to allow survival through NaN cascades that the NaN gates are already handling.

**ISSUE 4: `preds.append(raw)` in SS path stores pre-projection outputs**
- Loss computed as MSE(raw_output, sonar_GT) where raw_output norm >> sonar_GT norm
- NaN gate replaces some `raw` entries with GT (norm~0.2051) but other entries are raw (norm~0.1-5)
- MSE penalizes the scale difference, creating an implicit "scale penalty" that conflicts with the training objective. Cosine loss is scale-invariant and correct; MSE should either be disabled or computed on sphere-projected predictions.

**ISSUE 5: MSE loss is geometrically wrong for hypersphere targets**
- Targets live on S^(d-1) (radius 0.2051). MSE measures chord (Euclidean) distance, not arc distance.
- Correct geometry: cosine loss = 1 - cos_sim = chord²/2 (at fixed radius) = monotone with geodesic arc distance
- MSE combines with cosine gives mixed metric that is neither spherical nor Euclidean consistently
- Mitigation: reduce MSE weight to near-zero and rely primarily on cosine loss, OR project predictions to sphere before MSE.

**ISSUE 6: Double timestep conditioning in DF forward pass**
- Line 1108 (`x = x + t_emb`): additive injection before transformer layers
- Lines 1110-1111 (`layer(x, ..., t_emb=t_emb)`): AdaLN modulation in each layer
- Result: timestep info applied twice — once as additive bias to noisy vectors AND once as per-layer scale+shift
- AdaLN alone is sufficient (DiT design). The additive injection is redundant and adds a timestep-dependent bias the AdaLN must fight against. Potential source of instability at high noise levels.

**ISSUE 7: Horizon warmup missing from exp_conf → [NaN:6] at System2 first step**
- GB10_1 exp_conf lacked `horizon_warmup_steps` and `nan_gate_window_size` configs
- Going from target_steps=1 to target_steps=4 with NO LR warmup → [NaN:6] at S50 from cold transformer positions 2-3-4 never seen before
- NaN gate contained it, but the absence of NaN cascade detector meant no LR response either
- FIX: added these fields to exp_conf with values from orig_conf.

**ISSUE 8: `_safe_normalize` uses `nan_to_num(x, nan=0.0)` → arbitrary direction from near-zero inputs**
- When input norm → 0 (after nan_to_num zeros out NaN dims), division by `clamp(min=eps)` gives a direction from numerical noise
- Semantically: the "normalized" vector has random direction, not a meaningful direction
- For NaN-gated inputs this is moot (NaN gate replaced them with GT before _sphere_project), but for near-zero legitimate outputs it produces garbage directions
- Better: use the `nan_to_num` only as a last-resort safety net and rely on NaN gates upstream.

**ISSUE 9: `null_context_token` learns large residual-scale norms**
- Initialized to zeros, then `_to_residual_space(null_ctx)` scales by 156x at inference
- As null_context_token trains away from zero, it gets amplified 156x, making it much louder in cross-attention than SONAR query embeddings (norm~0.2051 scaled to ~32)
- CFG interpolation between conditional (SONAR scale) and unconditional (potentially 156x larger) predictions is numerically asymmetric
- Watch for: null_ctx.norm() growing >> target_norm during training.

**ISSUE 10: Constant [NaN:6] in GB10_1 System2 is the "cold position" problem**
- System1 only trains positions [0] of the chain. Positions [1,2,3] are never in the autoregressive context.
- When System2 starts with target_steps=4, the transformer processes sequence length 4 for the first time
- Positions 1,2,3 have never had gradient flow through their RoPE/attention weights at those offsets
- bfloat16 + LayerScale + AdaLN at untrained positions → NaN on first pass
- NaN gate handles the outputs, but parameter gradients are still NaN-contaminated (0 × NaN = NaN in backward)
- A curriculum of [1→2→3→4 steps, 1 epoch each] would prevent this cold-position shock

### Rules
1. **Zombie guard should be disableable**. If the NaN gates are handling NaN outputs correctly (loss→0, not loss→NaN), the zero_grad_streak is an artifact of gated batches, not a sign of model death. Add `enable_zombie_guard=false` when genuine learning is observed through NaN streaks. NaN gates always stay enabled.
2. **`beam_generate()` must scale vectors to residual space before chain concatenation**. `_sphere_project` returns SONAR scale; `_to_residual_space` must be called before `torch.cat([chain, next_vec], dim=1)`. Failing to do so creates a 156x scale mismatch in attention.
3. **MSE on hypersphere targets is geometrically inconsistent**. Use cosine loss as primary, set MSE weight low or compute MSE on sphere-projected outputs. Do not mix raw-output-scale MSE with SONAR-scale targets.
4. **Double timestep conditioning is redundant and potentially harmful**. AdaLN-Zero modulation per layer is sufficient. The additive `x = x + t_emb` before layers creates a conflicting signal the AdaLN must overcome.
5. **System2 first steps see positions 1..N-1 for the first time**. These "cold positions" have untrained RoPE offsets and LayerScale values, causing NaN on first pass. Mitigate with: (a) curriculum [1,2,3,4] steps, (b) horizon warmup LR, (c) NaN gates + enable_zombie_guard=false.

---

## 2026-04-16 - NaN cascade in pure teacher-forcing path + System2 gradient shock

### Pattern
Clean experiment (no SADT, no DF, no EMA, soft SS=0.15) showed excellent System1 training (val=0.7511, best ever), genuine System2 learning (tf_cos recovering 0.46→0.63), but NaN cascade killed training at E3 S3000: NaN count exploded 5→6→8→11→22→33→crash (zero_grad_streak=25).

### Root Causes
1. **NaN gate only existed in the scheduled-sampling path** (lines 968-978 in chain_generator.py). The pure teacher-forcing path (lines 914-945, used when `ss_prob=0.0`) had ZERO NaN protection. When output contained NaN, `nan_to_num(v_tf, nan=0.0)` at line 1147 replaced with 0, giving cos_loss=1.0 but zero gradient through nan_to_num. Parameters in NaN-producing zones received no corrective signal, silently expanding the unstable region until every output was NaN.
2. **System1→System2 gradient shock**: Transition from 1-step to 4-step chains caused a ~50x gradient norm spike (0.39→19.67), pushing parameters into bfloat16-fragile zones. Even with clip_grad_norm=1.0, the initial batches at extreme gradient scale damaged the parameter landscape. The 5 initial NaN events at E3 S50 were the seed that later grew into the full cascade.
3. **`0 × NaN = NaN` in PyTorch autograd**: Even with output-level NaN gates (either nan_to_num or torch.where), the backward pass through shared parameters computes `∂L/∂W = grad_output × activations^T`. When activations stored in the graph are NaN and grad_output is 0 for those positions, IEEE 754 gives `0 × NaN = NaN`, contaminating parameter gradients. The NaN gate cleans the loss but cannot prevent NaN gradients from the computation graph.

### Fix
1. **NaN gate in TF path**: After `output_proj(x)`, replace non-finite predictions with ground truth via `torch.where(bad, v_target_chain, v_pred)`. Loss for NaN positions = 0 (pred==target), no inflated cos_loss=1.0 artifact.
2. **Horizon transition LR warmup**: When target_steps changes, multiply LR by a factor (default 0.1) that linearly ramps to 1.0 over N steps (default 200). Config: `horizon_warmup_steps`, `horizon_warmup_factor`. Prevents the gradient shock from pushing parameters into fragile zones.
3. **NaN cascade detector**: Track NaN gate activations per step in a rolling window. When rate exceeds threshold (`nan_gate_max_rate` events in `nan_gate_window_size` steps), halve LR to slow parameter drift. Logged as `[gate:N]` in training output.
4. **Model-side NaN counter**: `model._nan_gate_count` attribute updated each forward pass, consumed by training loop for metrics and cascade detection.

### Rules
1. **NaN gates must cover ALL forward paths, not just the fancy one**. The teacher-forcing path is the "simple" default — it must have the same NaN protection as the scheduled-sampling path. When adding a safety mechanism to one code path, grep for all paths that produce the same output and apply the same guard.
2. **`nan_to_num(x, nan=0.0)` is a loss bomb, not a fix**. Replacing NaN with 0 gives cos(0, target) = 0 → cos_loss = 1.0, inflating the loss. Replacing with GT gives cos(GT, GT) = 1.0 → cos_loss = 0.0, which is the correct "no signal" response. Always replace with GT, never with 0.
3. **Phase transitions need LR warmup**. System1→System2 is a task-complexity discontinuity that causes gradient norm spikes. Without LR dampening, the spike pushes parameters into numerically fragile regions. Apply the same principle to any training phase transition (curriculum steps, loss function changes, etc.).
4. **NaN cascades are exponential, not linear**. Once started, the NaN region expands because corrupted parameters don't receive corrective gradient (0 × NaN = NaN in backward). Detection must look at RATE of NaN increase, not just count. A constant low rate (3 in 5000 steps) is fine; an accelerating rate (22 in 250 steps) is catastrophic.

---

## 2026-04-15 - ChainGenerator "double zombie": SDPA `-inf` mask NaN root cause + Adam momentum persistence

### Pattern
After deploying the first-round Adam-zombie fix (grad sanitize + per-param Adam scrub + post-step EMA restore), a new run STILL zombified at E2 S3650: grad degraded over ~200 steps (20 → 13 → 8 → 0.6 → 0), then every step was a NaN-skip from S3800 onward through E6+ with val permanently stuck at 0.7405. The first-round fix was **necessary but not sufficient**.

### Root Causes
1. **`CrossAttention` used `float("-inf")` as an additive mask fill value**. This is a well-documented SDPA footgun on CUDA — `-inf` in bf16/fp16 triggers NaN in the flash/mem-efficient backends whenever softmax sees a row of all `-inf` (`0/0 = NaN`), AND `-inf - scale = -inf` corrupts gradient accumulation. The HuggingFace transformers library uses `torch.finfo(dtype).min` for exactly this reason. Any batch with a fully-masked context row (rare but possible with `context_bank_size=4` + short contexts) instantly produced cross-attention NaN, which propagated through every subsequent layer and every parameter's backward.
2. **Residual Adam momentum kept the zombie alive even after grad sanitation**. My first-round fix zeroed NaN gradients but only scrubbed Adam `exp_avg`/`exp_avg_sq` *if those buffers themselves were already non-finite*. For the much more common case of "finite momentum from a healthy prior step + zero'd current gradient", Adam computed `exp_avg ← β1·exp_avg + (1−β1)·0 = β1·exp_avg`, preserving the **pre-corruption direction** and applying `lr · exp_avg / sqrt(exp_avg_sq)` — the param kept drifting toward the bad basin with decaying but nonzero speed for thousands of steps.
3. **EMA was still updated on steps where grad sanitation fired** (my first-round guard only checked `params_restored == 0`). So the "subtle drift" during bad steps leaked into the shadow, and the rescue source slowly became the source of future corruption.
4. **`torch.isfinite` misses huge-but-finite drift**. A parameter at ±1e30 is still `isfinite=True`, so post-step sanity never triggered restore for the most common failure mode (numerical drift under corrupted momentum, not outright NaN).
5. **No escape from sustained corruption**. Once 100% of steps were sanitized, the first-round fix had no mechanism to step outside the corrupted basin — every step zeroed grads and Adam kept the same momentum loop.

### Fix
1. **Replace `float("-inf")` with `torch.finfo(q.dtype).min`** in `cebcm/models/chain_generator.py::CrossAttention.forward`, and detect the all-invalid-row edge case: for any batch row where every context slot is masked, force it to be fully visible. The downstream loss will mask the position correctly; non-NaN uniform attention is strictly better than NaN propagation.
2. **Unconditional Adam zero on grad sanitation**: in `train_step`, whenever `p.grad` had NaN/Inf, always `buf.zero_()` for `exp_avg`/`exp_avg_sq`/`max_exp_avg_sq` — not conditional on whether the buffers were themselves corrupt. This kills the momentum loop.
3. **EMA update ONLY on fully clean steps**: guard is now `if not grad_had_nan and params_restored == 0`. Any step with sanitation OR restore does not touch the shadow.
4. **Huge-but-finite drift guard**: post-step sanity now also checks `abs(p.data).amax() > param_abs_max` (default 1e4) and restores from EMA when exceeded. This catches the common "finite explosion under corrupted momentum" failure mode.
5. **Zombie streak detector + hard reset**: a streak counter (stored on the EMA object) increments on every sanitation step and resets on every clean step. When the streak hits `zombie_reset_threshold` (default 15 consecutive bad steps), hard-reset ALL parameters from the EMA shadow AND zero ALL Adam state AND reset the streak. This is the break-glass path that guarantees escape from any basin, regardless of root cause.

### Rules
1. **NEVER use `float("-inf")` as an attention mask fill value**. Always use `torch.finfo(dtype).min`. The bf16/fp16 + flash-SDPA combination turns `-inf` into NaN under masked-row edge cases. This is the single most common cause of mid-training bf16 attention NaN in modern transformer training. Grep for `-inf` in any attention/softmax path as a standing check.
2. **An all-masked row produces NaN under SDPA softmax**. Detect this explicitly (`invalid.all(dim=-1)`) and either (a) force the row to be fully visible, (b) add a sentinel visible token, or (c) short-circuit the sub-layer. Never let softmax see a row of pure `-inf`/`finfo.min` unless you want NaN.
3. **Grad sanitation without Adam momentum zeroing is insufficient**. With `grad=0` and live `exp_avg`, Adam computes `lr · exp_avg / sqrt(exp_avg_sq)` and keeps applying the PRE-corruption direction. For any "skip this grad" path, also zero the optimizer momentum for the affected parameter — otherwise the momentum persists the original bad direction for hundreds of steps.
4. **EMA shadow integrity requires conservative update gating**. Update the EMA only on FULLY clean steps — no sanitation, no restore, no anomaly. The decay factor does NOT save you from subtle drift leaking in over time: `0.9999·shadow + 0.0001·bad = slightly-bad`, and over 1000 bad steps the shadow becomes 10% bad. A dirty shadow means your break-glass source is poisoned.
5. **`isfinite` is not enough for param sanity**. Add a magnitude check (`abs(p).amax() > threshold`) and an optional global drift check (`||p||` growth rate). Finite-but-exploded values are the dominant failure mode under corrupted Adam momentum.
6. **Always have a break-glass "zombie reset" mechanism**. A per-step scrub breaks single-step errors; it does NOT break sustained corruption from a numerically fragile basin. Track the consecutive-sanitation streak and, once it crosses a threshold, force-restore all params from the EMA shadow and zero ALL Adam state. This is the only path that guarantees escape from a zombie basin that per-step scrubs cannot fix.
7. **When the first-round NaN fix "seems to work" but training still plateaus, instrument more metrics BEFORE iterating**. Add `grad_sanitized` (count per step), `param_restored` (count per step), `zombie_streak` (running counter), and `zombie_resets_total` — these metrics tell you whether you're fighting the right battle. A plateau with `grad_sanitized > 0` every step is a completely different bug than a plateau with `optimizer_stepped = 0.8` every step.

---

## 2026-04-14 - ChainGenerator "frozen zombie" model: Adam momentum corruption + skip-on-NaN trap

### Pattern
After the first round of NaN-collapse fixes, training reached E9 and then locked into a "zombie" state:
- `loss=0.73` (finite), `grad=0.0000` **every step**,
- `NaN:N` counter incrementing `+1` each step (100% NaN-skip rate),
- `lr` frozen, geometry probe values **byte-identical** across hundreds of steps,
- `val_roll_cos` plateaued at `0.7237` from E2 onward.

Training appeared to run (forward produced finite loss via defensive `nan_to_num` scrubs) but NO parameter updates occurred. Fresh corruption on every batch, perpetually skipped.

### Root Causes
1. **Skip-on-NaN-grad is a trap**. `train_step` pattern `if not torch.isfinite(total_norm): optimizer.zero_grad(); return` means HEALTHY parameters never update either — any batch that has one sick gradient poisons ALL updates that step. Over time: 100% skip rate, zero progress.
2. **Adam momentum state is sticky**. `optimizer.zero_grad()` only clears `.grad`; it does NOT touch `exp_avg` and `exp_avg_sq`. A single NaN that made it into Adam buffers persists forever. On the next step Adam computes `new_exp_avg = β1·old_exp_avg + (1−β1)·grad` → still NaN → output param becomes NaN → next forward uses `nan_to_num` to scrub the value but **the parameter is still NaN in memory**, and its gradient will be NaN again. Self-propagating.
3. **`nan_to_num` is NOT a gradient barrier**. Scrubbing `model_out` after forward only fixes the forward value, not the backward graph. If any weight in the graph is NaN, d(loss)/d(weight) is still NaN even though `loss` itself became `0.0` via scrubbing.
4. **`clamp(min=1e-8)` is a no-op on NaN**. NaN passes through `clamp` unchanged. `weight_sum = weights.sum().clamp(min=1.0)` still yields NaN when `weights.sum()` is NaN, which then makes `loss = cos_term.sum() / NaN → NaN`.
5. **bf16 denormal underflow**. `snr.to(bf16).clamp(min=1e-8)` is ineffective: bf16 has no denormals, so any value `< ~1.17e-38` silently underflows to 0 BEFORE clamp sees it. Then `1/snr → Inf → NaN`. Min-SNR must be computed entirely in fp32.

### Fix
In `experiments/13_chain_generator/train_chain_generator.py`:

1. **Sanitize gradients, don't skip the step**: in `train_step`, replace `if NaN: skip` with an in-place `.grad.masked_fill_(bad, 0.0)` scrub for each parameter whose gradient contains NaN/Inf. Healthy parameters keep training, sick ones get a zero-update this step. Log `grad_sanitized` count.
2. **Rescue Adam state on corruption**: for every parameter whose `.grad` was sanitized, walk `optimizer.state[p]` and `torch.nan_to_num(buf, out=buf)` for `exp_avg`, `exp_avg_sq`, `max_exp_avg_sq`. This breaks the self-propagation loop.
3. **Post-step parameter sanity + EMA rescue**: after `optimizer.step()`, scan `model.named_parameters()`. For any parameter with non-finite values, `copy_(ema.shadow[name])` as a break-glass restore. EMA is our last known-good copy. Skip `ema.update(model)` on the step where a param was restored, so we don't pollute the shadow.
4. **SNR entirely in fp32 with autocast disabled**: wrap `model.diffusion_snr(...)` in `torch.autocast(device_type=..., enabled=False)`, cast output `.to(dtype=torch.float32)`, `nan_to_num` with `posinf=1e4` BEFORE clamp, then `clamp(min=1e-6, max=1e4)`. Never allow bf16 to see SNR numerics.
5. **Scrub `.sum()` before `.clamp()`**: replace `weights.sum().clamp(min=1.0)` with `torch.nan_to_num(weights.sum(), nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1.0)` in both `_masked_step_losses` and `_masked_weighted_step_losses`.
6. **Scrub loss before backward**: final `loss = torch.nan_to_num(loss)` in `train_step`, and skip backward only when the sanitized value is exactly zero (no signal to propagate).
7. **Config**: bump `df_warmup_epochs` 3→5 and lower `loss_lambda_diffusion` 0.25→0.15 to give the critic more headroom before DF supervision kicks in at full strength.

### Rules
1. **NEVER `skip step` on NaN gradient**. Sanitize in place, log the count, let healthy params train. The skip-pattern is an anti-pattern that converts transient errors into permanent plateaus.
2. **Optimizer state must be scrubbed alongside grads**. `optimizer.zero_grad()` does NOT touch momentum buffers. Any NaN protection that only scrubs `.grad` is incomplete — Adam's `exp_avg`/`exp_avg_sq` must be scrubbed too, otherwise the next step recreates the NaN.
3. **Clamp is not NaN-safe**. `clamp(min=ε)` passes NaN through unchanged. ALWAYS `nan_to_num` before `clamp` when the input could be non-finite. Same for `clip_grad_norm_` — check the returned norm for `isfinite`, don't assume clipping sanitized it.
4. **Keep Min-SNR / any numerics-sensitive math in fp32**. Use `torch.autocast(..., enabled=False)` inside the helper. bf16 has no denormals → silent underflow → div-by-zero → NaN.
5. **EMA is break-glass recovery, not just "nicer eval weights"**. Maintain it, check parameter finiteness after every `optimizer.step()`, restore from shadow on corruption. A single corrupt step without rescue means the run is dead.
6. **`nan_to_num` is a forward-only value scrub, not a gradient barrier**. If a weight is NaN, its gradient is NaN regardless of downstream scrubs. Fix at the source (the parameter/optimizer state) not just at the loss.
7. **When every step reports `grad=0` and `NaN:N` climbs linearly, it is not a plateau — it is a zombie model**. Distinguish this from genuine convergence by checking: (a) probe values byte-identical across steps, (b) `optimizer_stepped` metric → 0, (c) `nan_grad_skipped` metric → step count. If all three are true, parameters are frozen / corrupt, not converged.

---

## 2026-04-14 - ChainGenerator probe: v-prediction misread as x₀ caused antipodal `df_cos_mean` in GUI

### Pattern
After the NaN-collapse fixes, the GUI showed:
- `train df_cos = 0.72` (training metric — positive, climbing)
- `probe df_cos_mean = −0.68` (geometry probe — negative, "going more wrong")
- 3D view: "DF pred_x0" marker antipodal to the clean target along PC1

User reported "модель идёт в обратную сторону". In reality the model was learning correctly; the probe was lying.

### Root Cause
`write_training_probe_snapshot` took the raw output of `model.forward_diffusion_forcing(...)` and stored/compared it as `pred_x0`. Under `prediction_type="v"` the raw output is the velocity `v = √ᾱ_t · ε − √(1−ᾱ_t) · x₀`, not x₀. At mid-range noise levels (`t≈32` with cosine schedule, √ᾱ ≈ √(1−ᾱ) ≈ 0.707) the expectation of `cos(v_pred, x₀_clean)` when the model is *perfect* is approximately `−√(1−ᾱ_t) ≈ −0.707`. The observed `−0.64 → −0.68` was converging toward that asymptote — i.e. evidence of correct learning, not regression.

Meanwhile the per-step `df_cos` logged by `_diffusion_forcing_objective` is computed AFTER `predict_x0` decoding in x₀-space, so it reads positive. The probe and the training metric were literally in different spaces.

### Fix
In `experiments/13_chain_generator/train_chain_generator.py::write_training_probe_snapshot`:
```python
v_df_raw, v_noisy, eps = model.forward_diffusion_forcing(..., return_noisy=True)
v_df = model.predict_x0(v_noisy, v_df_raw, levels).to(dtype=v_df_raw.dtype)
```
All downstream uses (`projected["pred_x0"]`, `df_cos`, `df_l2`, `pred_norm`, PCA basis fitting in `_masked_probe_points`, `raw["pred_x0"]`, metric `df_cos_mean`) now consume the x₀-decoded tensor, consistent with the clean target and with the training-side `df_cos` metric.

`v_noisy` stays untouched — it already lives in the clean/x_t space, which is what `noisy_cos_mean` and the `Clean → Noisy → Pred x₀` trajectory actually want.

### Rules
1. **Every "cos to clean" in training/probe code must live in x₀-space**. Whenever `prediction_type ∈ {"v", "eps"}`, route the raw model output through `predict_x0(x_t, model_out, t)` *before* any cosine/L2/projection comparison against `x₀`. Grep for `forward_diffusion_forcing` call sites every time you touch the diffusion math.
2. **When the training metric and the probe metric diverge in sign, the bug is in the one that is not the training metric** (usually). Training metrics have been debugged across many runs; probe code is newer and drifts. Start suspicion there.
3. **Probe export names are load-bearing**. If a field is called `pred_x0`, it MUST be in x₀-space. Mislabelled fields become silent time bombs for downstream GUI math (heatmaps, PCA basis, distance metrics) and confuse the user into thinking the model is broken.
4. **Expected asymptotes for a correct v-prediction model** (cosine schedule, `d_model=1024`, `T=64`):
   - At `t≈0`: `cos(v_pred, x₀) → 0` (v ≈ ε, orthogonal to x₀ in expectation)
   - At `t≈T/2`: `cos(v_pred, x₀) → −√(1−ᾱ_t) ≈ −0.707`
   - At `t≈T`: `cos(v_pred, x₀) → −1`
   If you see any of these numbers where you expected `+1`, you forgot to decode v.

---

## 2026-04-14 - ChainGenerator NaN collapse (E3→E4): NaN×0 trap recurrence, Min-SNR x₀/ε swap, defense-in-depth

### Pattern
Training log `Arch 14_01_26 full training log.txt` showed healthy convergence through E3 (val_roll_cos_last=0.7554), then a single NaN in `loss_df` at the end of E3, complete loss collapse from E4 (grad_norm=0.0000 for 20 consecutive epochs), plus a secondary `ans_coverage→0` collapse at System1→System2 transition (E10+). Early-stopped at E24 vs planned E50.

### Root Causes
1. **NaN × 0 = NaN recurrence in `_masked_weighted_step_losses`**. The lessons entry from the original NaN bug (L57 pattern) says "masked losses must use `torch.where(mask, term, 0)`, never multiplication by mask". When the positionally-weighted variant was added for Diffusion Forcing, it reintroduced the exact same bug: `cos_loss = ((1 - cos) * weights * mask).sum() / denom`. Any single token with NaN in `cos_sim`/`mse_per` (e.g. `target=0` row) poisoned the entire batch loss and every downstream gradient.
2. **Min-SNR x₀/ε formulas were swapped** in `_diffusion_forcing_weights`. Correct derivations:
   - x₀-prediction: `w = clipped` (NOT `clipped/snr` — that double-counts the 1/SNR already in the loss)
   - ε-prediction: `w = clipped/snr`
   - v-prediction: `w = clipped/(snr+1)`
   Code had x₀ and ε inverted. Latent because v-prediction is the default, but a footgun for anyone switching.
3. **`_safe_normalize` divergence**: train-side helper used unguarded `v/v.norm().clamp(1e-6)` without first scrubbing inf/NaN. A single inf-slot propagated through every downstream normalization.
4. **No nan_to_num at forward-pass boundaries** (`v_tf`, `v_roll`, `model_out`, `target`, `weights`). Bfloat16 + high-SNR regime at `t≈0` occasionally produced inf SNR and NaN model outputs that were not caught until they had already been masked-multiplied into the loss.
5. **DF lambda=0.5 applied from step 0** against an un-warmed backbone, combined with `df_noise_level_min=0` allowing trivially-clean samples where SNR→∞.
6. **System1 phase too short**: with `system1_epochs=10`, the model did not see enough single-step coverage before the much harder System2 phase, and positional weights amplified the answer-slot loss past the backbone's ability to keep up, collapsing `ans_coverage` to zero.

### Fixes (applied in this session)
- **Defense-in-depth `nan_to_num` scrub** at every boundary: `_safe_normalize` input, `cos_sim`, `mse_per`, `weights`, `v_tf`, `v_roll`, `model_out`, `target`, SNR clamp `max=1e4`.
- **`_masked_weighted_step_losses` rewritten** to use `torch.where(mask_bool, term, zeros)` for both cosine and MSE terms — no more `* mask` multiplication on potentially-NaN tensors.
- **Min-SNR x₀/ε formulas corrected** in `_diffusion_forcing_weights` per derivation above; `pt="v"` branch unchanged.
- **DF lambda warm-up ramp**: added `df_warmup_epochs=3` config + training-loop logic that scales `loss_lambda_diffusion` from `0 → base` over the first N epochs. Base lowered from `0.5 → 0.25`.
- **`df_noise_level_min: 0 → 2`** — skip the near-clean regime where SNR is numerically unstable and the objective is trivial.
- **`system1_epochs: 10 → 15`** — longer single-step phase so the backbone stabilizes before the System1→System2 transition.
- **Rolling `ans_coverage` warning** log: prints an explicit `[WARN]` when `val_answer_coverage` rolling-mean over last N epochs drops below threshold — early signal of the collapse pattern.
- **`df_lam` added to per-step training log** so the warm-up is visible in logs.

### Rules (carry forward)
1. **NaN × 0 = NaN is recurrent**. Every time a new masked/weighted loss is added, grep for `* mask` / `* weights` patterns and force them through `torch.where`. Add this to the PR checklist.
2. **Min-SNR-γ cheat sheet** (for the ChainGenerator DF loss formulation):
   - x₀ → `w = clip(snr, γ)`
   - ε  → `w = clip(snr, γ) / snr`
   - v  → `w = clip(snr, γ) / (snr + 1)`
   Do NOT memorise from papers written against a different parameterisation; re-derive against OUR loss every time.
3. **Always `nan_to_num` at forward-pass boundaries** when training in bfloat16 with a diffusion objective. SNR blow-ups at `t≈0` are a known failure mode; the cost of defensive sanitization is < 0.1% of step time.
4. **Never jam two new regimes at once**. Launching `System1→System2` AND full-strength DF loss simultaneously at E10 caused a double-shock. Use warm-up ramps for any auxiliary loss that can reach >10% of the primary loss magnitude.
5. **Log whatever you will want to investigate**. `df_lam`, `ans_coverage`, and grad_norm must all be in per-step logs; otherwise the post-mortem needs guesses instead of evidence.
6. **`safe_normalize` is a public API** — both the model and the training script must use the SAME implementation. Duplicate helpers drift; consolidate.

---

## 2026-04-13 - Diffusion Forcing audit: Min-SNR weight formula inversion and grad_norm logging gap

### Pattern
Full code review of Diffusion Forcing implementation (1760 lines added across 8 files) found one mathematical bug and one monitoring gap.

### Root Causes
1. **Min-SNR weight formula inverted for x₀-prediction**: Code had `min(SNR, γ) / γ` which gives weight ~1 for clean tokens (high SNR) and ~0 for noisy tokens. For x₀-prediction, correct formula is `min(SNR, γ) / SNR` which downweights the trivially easy clean regime. Bug was latent (γ=0 = disabled by default).
2. **grad_norm not logged**: Training dashboard couldn't show gradient norm over time, essential for diagnosing training instability.

### Fixes
1. Changed Min-SNR weights from `min(SNR, γ) / γ` to `min(SNR, γ) / SNR` in `_diffusion_forcing_weights`.
2. Added `grad_norm` to train_step metrics, console output, JSONL log, and GUI dashboard.
3. Fixed BOM (U+FEFF) in training_geometry.py.

### Rules
1. **For x₀-prediction diffusion, Min-SNR-γ weight = min(SNR, γ) / SNR**. For ε-prediction it's the same formula. The form `min(SNR, γ) / γ` is WRONG — it inverts the weighting.
2. **Always log grad_norm** — it's the earliest indicator of training instability before loss NaN appears.
3. **When reviewing latent bugs behind disabled features**: even if a flag is off, fix the underlying code. Someone will enable it later and get silent corruption.

## 2026-04-13 - Noise calibration in high-dim continuous space and diffusion-inspired exposure bias fix

### Pattern
Training log analysis revealed TWO compounding causes for the val_roll_cos_last ceiling at 0.60:
1. Oracle-assisted rollout inflated train_roll_cos (fixed in 5cced91)
2. Eval generate() received `noise_std=0.01` which in d=1024 produces noise norm ≈ sqrt(1024)*0.01 ≈ 0.32, comparable to raw_next norm ≈ 0.4 (SNR=1.25). This destroyed eval cosine by ~0.16.

Additionally, model has val_tf_cos=0.87 (single-step accuracy) but val_roll_cos=0.47 (multi-step accuracy) — classic exposure bias where the model never sees its own imperfect outputs during training.

### Root Causes
1. **Noise uncalibrated for dimensionality**: `noise_std` is per-dimension, but total noise norm scales as `sqrt(d) * std`. In d=1024, even small per-dim std=0.01 creates devastating total perturbation.
2. **No noise robustness in teacher forcing**: forward() gives the model perfect GT prefix. At eval, generate() feeds back model's own errors, causing error accumulation over multi-step rollout.
3. **Train generate() noise was pointless with oracle disabled**: Without oracle to select from noisy candidates, noise only degrades training quality and creates train/eval mismatch.

### Fixes
1. Set `free_run_noise_std=0.0` in both train and eval (no more noise in generate())
2. Added **Noisy Teacher Forcing** (diffusion-inspired): forward() adds per-sample noise from U[0, tf_noise_std] to GT prefix during training. Model learns to predict from imperfect contexts.
3. Noise calibration: tf_noise_std_max=0.005 in SONAR space (target_norm=0.2051). At max: noise_norm ≈ sqrt(1024)*0.005 ≈ 0.16, ratio=0.78, angular perturbation ≈ 38° — matches model's late-training error of arccos(0.87) ≈ 30°.
4. Noisy TF also applies inside scheduled sampling (GT tokens get noised too).

### Rules
1. **ALWAYS scale noise by sqrt(d)** to understand its real magnitude. noise_std=0.01 in d=1024 is NOT "small noise".
2. **If a noise mechanism exists only for a disabled feature (oracle), remove the noise too.**
3. **Exposure bias in continuous AR = diffusion at noise level 0 only.** Fix by training at multiple noise levels (noisy teacher forcing = multi-level denoising training).
4. **Calibrate noise to model error level**: tf_noise_std should produce angular perturbation ≈ arccos(current_tf_cos). Use tf_noise_std ≈ target_norm * tan(arccos(tf_cos)) / sqrt(d).

## 2026-04-11 - Oracle/DAger can fake rollout quality and disabled rank loss can still create NaNs

### Pattern
Latest ChainGenerator run (`Arch(11-04-26) training log.txt`) reached train `tf_cos≈0.89` and train `roll_cos≈0.87`, while eval stayed near `roll_cos≈0.50` and `roll_cos_last≈0.47`. The best eval point remained early System1 (`val_roll_cos_last≈0.5993`). NaN-skipped batches also appeared across most epochs even after earlier guards.

### Root Causes
1. **Oracle-guided DAger train/eval mismatch**: train rollout used `oracle_guide=chains` with nonzero `oracle_prob`; eval had no oracle because `self.training=False`. Train rollout therefore measured an oracle-assisted trajectory, not the actual model policy.
2. **Disabled rank loss was still computed**: `loss_lambda_rank=0.0`, but in-batch contrastive rank still ran through normalization. If it produced NaN, `0.0 * NaN` contaminated total loss.
3. **Unsafe normalization in rank path**: rank loss used `F.normalize` default epsilon instead of the project safe-normalize rule.
4. **Masked loss used multiplication by zero**: `NaN * 0` can stay NaN; masked losses must use `torch.where(mask, value, 0)` before reduction.
5. **Repeat-ban in training reintroduced stochastic rollout mismatch**: repeat-ban is an inference safety mechanism and must not perturb the training prefix.
6. **Scheduler advanced on skipped optimizer steps**: when NaN/Inf skipped a step, the LR scheduler still stepped, causing schedule drift and PyTorch warnings.

### Fixes
- Disable oracle/DAger by default with explicit `enable_oracle_dagger=false`, `oracle_max_retries=0`, `oracle_prob_max=0.0`.
- Gate `oracle_guide` in the train objective; only pass it when explicitly enabled for ablations.
- Compute rank loss lazily: if `loss_lambda_rank==0`, do not include it in the graph and never multiply zero by a possibly non-finite tensor.
- Replace contrastive rank normalization with safe normalization.
- Replace latent-vector cosine checks in generation with the same safe-normalize rule.
- Compute masked cosine/MSE reductions with `torch.where`, not value-by-mask multiplication.
- Disable repeat-ban during training.
- Return `optimizer_stepped` from `train_step` and only advance the scheduler after a real optimizer step.
- Add dataset/horizon diagnostics and explicit `ans_cov` / `roll_ans` metrics so `roll_cos_last` is not mistaken for answer quality when the current horizon has not reached the answer.

### Rules
1. If train rollout uses an expert/oracle and eval rollout does not, train rollout metrics are not valid quality metrics.
2. Disabled loss terms must not execute unstable graph code; never rely on `0.0 * loss`.
3. Masked losses must avoid `NaN * 0`; use `torch.where(mask, term, 0)`.
4. Inference anti-loop mechanisms (`repeat_ban`, repeat penalty, stochastic rerolling) must be guarded out of training unless the same mechanism is explicitly part of the train objective and metric.
5. LR schedulers should step only when the optimizer actually stepped.
6. Always log answer coverage for curriculum horizons; `roll_cos_last` is only an answer metric when the answer is inside the selected training window.

## 2026-04-10 - ChainGenerator NaN collapse at E13 and val roll_cos_last overfitting

### Pattern
Training collapses to all-NaN at E13 (target_steps=5, ss_prob=0.15) after sporadic NaN first seen at E7. Val roll_cos_last peaks at 0.5991 (E1) then DEGRADES to 0.5063 (E6) despite train tf_cos improving 0.29→0.85.

### Root Causes

**NaN collapse (3 coupled mechanisms):**
1. **repeat_penalty pushes raw_next toward zero norm**: `raw_next -= penalty * over * repel` subtracts detached historical vectors (norm ~32) from raw logits, driving raw_norm from 0.28→0.15→0. When `F.normalize()` receives near-zero input in bfloat16, gradient through division explodes to NaN.
2. **Oracle DAgger + repeat_ban compound the issue**: 5 oracle retries + 3 repeat_ban retries = 8 `F.normalize()` calls per step on increasingly corrupted vectors.
3. **No NaN guard before backward()**: Single NaN in loss → NaN in all gradients → `clip_grad_norm(NaN)=NaN` (IEEE 754) → NaN weights forever.

**SADT amplification**: At System2 transition, metrics naturally drop (harder task). SADT with tolerance=0.05 and OR-gate (`tf OR roll bad`) throttles LR repeatedly (halving by 0.5), masking symptoms without preventing NaN. EMA not reset at transition.

**Val overfitting:**
1. **Oracle-guided DAgger train/eval mismatch**: Oracle is gated by `self.training`, active during training, absent at eval. Model learns to depend on oracle crutch → val crashes when oracle is removed.
2. **Weak regularization**: 101M params on ~90k samples with dropout=0.1, weight_decay=1e-4 → severe memorization.

### Fixes
1. **safe_normalize**: Replace `F.normalize(v)` with `v / v.norm().clamp(min=1e-6)` — prevents gradient explosion on near-zero vectors.
2. **Disable repeat_penalty in training**: Guard with `not self.training`. It provides zero useful gradient (history detached) but destabilizes raw_next norm.
3. **NaN guard in train_step**: Check `torch.isfinite(loss)` before backward; check `torch.isfinite(total_norm)` after unscale. Skip step on NaN.
4. **NaN guard in generate()**: If generated vector is NaN, replace with previous valid vector.
5. **Oracle probability decay**: New `oracle_prob` param decays from 1.0→0.0 over 20 epochs. Forces model to learn robust generation without oracle dependency.
6. **SADT cooldown at horizon transition**: Reset EMA and add cooldown period when target_steps changes. Change OR→AND gate for degradation detection.
7. **Increased regularization**: weight_decay 1e-4→5e-4, sadt_tolerance 0.05→0.10, noise_std_max 0.05→0.03, oracle_max_retries 5→3.

### Rules
1. **NEVER modify raw logits in-place during training** — repeat_penalty, repulsion, etc. can drive vectors to zero norm, causing F.normalize gradient explosion in low-precision (bf16).
2. **Any F.normalize in training path must use safe normalization** with `clamp(min=1e-6)`.
3. **Always guard backward() with NaN check** — a single NaN infects all weights permanently. Skip the step, don't try to clip NaN gradients.
4. **Train-only mechanisms create eval mismatch** — if oracle/noise/etc. are gated by `self.training`, the model learns a different distribution than what it sees at eval. Decay such mechanisms to zero.
5. **SADT must reset EMA at curriculum transitions** — metric drops from harder tasks are not degradation.
6. **Use AND-gate (both metrics bad) for LR throttle, not OR-gate** — single-metric noise causes false throttling.

## 2026-04-09 - Inference pipeline missing convergence_cos → model never uses trained EOS

### Pattern
Model was trained with answer-repeat padding to learn convergence stopping (SONAR-space EOS), but the inference diagnostics code never passed `convergence_cos` to `model.generate()`. The parameter defaulted to 0.0, disabling convergence detection entirely. All early stops were from stagnation_patience, not convergence.

### Root Cause
`_generate_stochastic_chain()` didn't accept or forward `convergence_cos` / `convergence_window` parameters. The training code taught the model to converge, but inference never checked for it.

### Fix
Added `convergence_cos` and `convergence_window` to `_generate_stochastic_chain()` and `run_generation()`. Default: `convergence_cos=0.995, convergence_window=2` for multi-step.

### Rule
When adding a feature to training (convergence stopping, new loss, etc.), immediately verify the inference pipeline uses it too. Train ↔ inference feature parity must be checked explicitly.

## 2026-04-09 - 1-word outputs are a data/quality issue, not missing autoregressor

### Pattern
Model generated chain_texts like ["France", "France", "France and"] — each SONAR step decoded to 1-2 words instead of full sentences. User suspected missing autoregressive capability.

### Root Causes
1. HotpotQA answers are 1-3 word named entities ("Paris", "Satan"). SONAR embedding of short text decodes to short text. The model correctly learned to produce short answers.
2. Suffix alignment (old bug #6) prevented learning meaningful step-by-step reasoning.
3. cos ≈ 0.73 is too low for faithful SONAR sentence reconstruction — loses syntax, keeps only topic.
4. Answer-repeat padding trains fast convergence → model skips reasoning, goes straight to answer keyword.

### Rule
1. Each SONAR vector IS a full sentence embedding. "1-word output" means the model produces an impoverished vector, not a missing feature.
2. To get multi-sentence responses: deduplicate chain_texts and concatenate unique steps (assemble_response).
3. Data determines output format: HotpotQA → short answers. For longer responses, need different training data.

## 2026-04-09 - Autoregressive chain training: 8 bugs causing tf/roll gap and gradient collapse

### Pattern
Chain generator training showed zero tf/roll cosine gap during System1 (1 step), then instant 7–10% gap at System2 transition. Deeper analysis revealed 8 coupled bugs, 3 previously unknown.

### Root Causes & Fixes

1. **Suffix-aligned targets mismatched generate() start point** (Critical): `select_training_targets` took the LAST N tokens but generate() always starts from position 0. Model was asked to produce late-chain vectors from cold start — impossible task. Fix: prefix-aligned targets `chains[i, :ti]`.

2. **No scheduled sampling**: `forward()` always fed ground truth. Model never saw its own predictions. Fix: added `scheduled_sampling_prob` parameter with epoch-based linear ramp (0→0.5).

3. **Deep autoregressive gradient corruption**: `generate()` built full autoregressive graph through concatenation. Fix: `next_vec.detach()` before appending to chain — each step gets direct gradient from its target comparison, not through all subsequent steps.

4. **`repeat_ban` `torch.where` during training**: Piecewise gradient from resampling with second `randn`. Fix: skip `repeat_ban` when `self.training`.

5. **Noise std 0.005 was cosmetic**: Angular perturbation ≈ 1.5° ≈ 0.0003 cosine deviation (300× smaller than the 7–10% gap). Fix: raised to 0.05.

6. **L_ans double-counted the answer position**: L_step included all positions, L_ans re-applied loss on the last — answer got 2× gradient vs intermediate steps. Fix: exclude last position from L_step mask.

7. **Aggressive horizon ramp (1→2→4→6→8)**: `int()` truncation created step-function jumps. Fix: linear ramp — each epoch adds exactly 1 step.

8. **No per-step diagnostics**: Couldn't tell if front-loaded or uniform quality. Fix: log `tf_cos_first`, `roll_cos_first` alongside mean/last.

### Rules
1. **Autoregressive training targets must align with generation start point.** If generate() starts from position 0, targets must be prefix-aligned, not suffix-aligned.
2. **Scheduled sampling is mandatory for multi-step autoregressive training** — pure teacher forcing creates exposure bias proportional to chain length.
3. **Detach autoregressive context in generate()** — only the per-step output→target comparison should carry gradient, not the full chain.
4. **Any inference-only mechanism (repeat_ban, convergence stop) must be guarded by `not self.training`.**
5. **Noise std for rollout must be calibrated against the tf/roll gap magnitude.** If gap is 7–10%, noise should produce at least 1–2% deviation.
6. **Never apply two losses to the same position without explicit deduplication.**
7. **Linear is better than aggressive for curriculum ramps** — model needs time to stabilize at each chain length before extending.

## 2026-04-08 - Autoregressor needs convergence training and adaptive stopping

### Pattern
ChainGenerator had no EOS equivalent. At inference, the model couldn't signal "I'm done reasoning." Generating past the training horizon (N steps) produced OOD garbage. The model memorized fixed-length chains but couldn't adapt to variable reasoning depth.

### Root Causes
1. Training chain = [steps..., answer] with no continuation signal after answer
2. generate() had fixed `num_steps` with no adaptive stopping
3. No way to resume generation from a previous chain (no "continue thinking")
4. compute_loss MSE was double-divided by d_model (0.008% contribution = dead)

### Fixes
1. **Answer-repeat padding**: Chain becomes [steps..., answer, answer, answer]. Model learns: after finding the answer, keep outputting it. Consecutive similarity at inference = convergence signal.
2. **Convergence stopping**: `convergence_cos > 0` in generate() — stop when last W outputs have pairwise cosine > threshold. SONAR-space EOS.
3. **Chain prefix (resume)**: `chain_prefix` parameter lets you feed a previous chain and continue generating. Enables "append final vector and keep thinking" workflow.
4. **MSE fix**: Removed redundant `/d_model` division.

### Rules
1. An autoregressor MUST have a stopping criterion — either learned (answer-repeat convergence) or external (critic energy threshold).
2. Train with answer-repeat padding to teach convergence behavior. Without it, the model is OOD after the answer token.
3. Free-run rollout must use non-zero noise; otherwise it's a copy of teacher-forcing loss.
4. Always test that generate() preserves gradient flow for exposure-bias correction.

## 2026-04-08 - Post-review bugfixes for Stage 13/14 pipeline

### Bugs Found and Fixed

1. **Temperature noise scaling asymmetry** (`chain_generator.py:generate`): `temp>1` added 0.01 (negligible), `temp<1` multiplied (zeroed noise). Fixed: `noise_std = latent_noise_std * temp` — consistent multiplicative scaling.

2. **MSE double-division by d_model** (`train_chain_generator.py`): `.mean(dim=-1)` already averages over D=1024, then code divided by `d_model` again. Result: MSE contributed 0.008% of total loss — effectively dead. Fixed: removed redundant division. MSE now contributes meaningfully (~0.85%).

3. **Deterministic free-run defaults** (`chain_generator_config.json`): `free_run_noise_std=0.0, temperature=1.0` made rollout identical to greedy teacher-forcing. L_free_run degenerated into a copy of L_step — zero exposure-bias correction. Fixed: `noise_std=0.005, temperature=1.05`.

4. **Stagnation check logic** (`chain_generator.py`): When `energy_fn` failed silently, `energy_hist` was empty but `cos_hist` had data. AND logic `stagnated = False AND (cos_check)` → never triggered. Fixed: check ALL available metrics independently with proper length guards.

5. **Critic context mismatch** (`train_chain_critic.py`): Critic trained on `mean(v_steps)` but generator uses context bank `[query, evidence_1, ..., evidence_K]`. At reranking time, critic sees different semantic grounding. Fixed: critic dataset now builds context = `mean([query, evidence_slots])` matching generator's bank.

6. **Generator hard negative redundancy** (`train_chain_critic.py`): Loop filled k rightmost slots with identical `v_gen` vector. Fixed: single slot fill since only one generate call is made.

### Rules
1. Temperature must scale noise consistently — use multiplication, not mixed additive/multiplicative.
2. When loss = `f(x).mean(dim=-1)`, do NOT divide by `dim_size` again. Check loss magnitudes against other components.
3. Free-run rollout MUST have non-zero noise during training — otherwise it provides zero exposure-bias correction.
4. Stagnation/early-stop logic must handle partial metric availability (some trackers may fail).
5. Critic and generator MUST use the same context construction at train and inference time.

## 2026-04-08 - Autoregressive QA: never train on padded tokens or prefix-only curriculum that drops answer

### Pattern
ChainGenerator plateau/collapse was amplified by three coupled pipeline mistakes:
- padding mask was computed but not used in loss;
- sequence truncation could drop the final answer token;
- `System1` curriculum (`target_steps=1`) trained on the first reasoning step instead of answer.

### Root Cause
Training objective was misaligned with inference target:
- model optimized padded zeros and early-chain prefixes;
- direct-answer mode was not actually direct-answer;
- validation repeated the same masking bug.

### Fix
- Add `loss_mask` to `ChainGenerator.compute_loss` and use masked reductions.
- Preserve answer on truncation (`keep_last` policy).
- Build curriculum targets as suffix ending at answer (`System1 => answer-only`).
- Apply masking in both train and validation paths.

### Rule
1. For variable-length autoregressive chains, masking must be part of the model loss API, not only data loader logic.
2. Any truncation policy must explicitly preserve supervision target (final answer token).
3. Curriculum labels must be audited against declared mode semantics (`System1` must optimize answer directly).
4. Best-of-N reranking is invalid if candidate generation is deterministic; enforce stochastic diversity.

## 2026-04-08 - Critic parity and horizon discipline must be enforced in GUI/runtime

### Pattern
Online diagnostics diverged from training behavior:
- critic was trained with `v_context` but inference reranking called critic without context;
- GUI allowed long rollouts beyond training horizon;
- critic training relied on easy random negatives, which inflated rank metrics.

### Root Cause
Evaluation path did not preserve the same conditioning and hardness assumptions as training.

### Fix
- Pass `v_context` through all critic calls in diagnostics (rerank, per-step energy, chain analysis, landscape).
- Add compatibility wrapper for legacy critics that do not support `v_context`.
- Clamp inference steps by checkpoint-trained horizon (`training.max_chain_steps`) and architectural cap (`max_chain_len`).
- Add mixed-negative strategy in critic training: random negatives + in-batch hard negatives.
- Add optional generator-hard negatives (from current generator checkpoint) with false-negative guard by cosine threshold.

### Rule
1. Critic train/eval/inference signatures must be context-parity compatible.
2. Runtime inference must be horizon-capped by what the model actually saw during training.
3. Do not accept rank metrics from easy random-negative-only training.
4. Every GUI export should expose whether runtime safety caps were applied.
5. For critic reranking quality, train with mixed negatives: random + in-batch hard + generator-hard (when checkpoint is available).

## 2026-04-06 - GUI checkpoint loader must migrate old/new parametrization key layouts

### Pattern
Web GUI landscape failed on old checkpoints with mismatch:
model expected `net.*.parametrizations.weight.*` keys, checkpoint had plain `net.*.weight`.

### Root Cause
Loader validated `load_state_dict` mismatch strictly but had no compatibility migration path.

### Fix
- Add bidirectional migration in GUI loader:
  - plain -> parametrized (`weight` -> `parametrizations.weight.original`, keep `.0.base` defaults)
  - parametrized -> plain (map `...original` to `weight`, drop aux parametrization keys)
- Retry load after migration before raising mismatch.

### Rule
Any checkpoint-facing loader must support at least one backward-compat migration path across known architecture serialization changes.
## 2026-04-06 - Direction loss path interpolation CONFLICTS with InfoNCE (2nd-order dominance)

### Pattern
Full-path interpolation for direction loss (v_noisy sampled uniformly between query→answer) dramatically improved direction_cos (0.11→0.53) but DESTROYED cos_sim (0.51→0.20). The direction loss's 2nd-order gradients (create_graph=True) dominated InfoNCE's 1st-order gradients when covering the full landscape.

### Evidence
- Run 2 (near-answer, noise=0.01): direction_cos=0.20, val cos_sim=0.51 ✓
- Run 3 (interpolation): direction_cos=0.61, val cos_sim=0.20 ✗
- Pattern matches lessons L1151: "MDSM 2nd-order gradients dominate ranking's 1st-order"

### Fix
Keep near-answer sampling but increase noise_scale (0.01→0.1) + sphere projection:
```python
v_noisy = add_noise(v_a, direction_noise_scale)  # 0.1, not 0.01
v_noisy = F.normalize(v_noisy, dim=-1) * target_norm
```

### Rule
1. **NEVER use full-path interpolation for direction loss with InfoNCE** — 2nd-order gradient dominance
2. Widen supervision radius via noise_scale, not via sampling position
3. Always sphere-project v_noisy when Langevin operates on sphere
4. If direction_cos ↑ but cos_sim ↓ → gradient conflict, reduce direction loss coverage

## 2026-04-06 - Chain Head trained on old critic is incompatible with new critic — retrain from scratch

### Pattern
Joint training of ConditionalCritic + old Chain Head: chain_rank_acc collapsed 0.9081→0.5321 (random) within 1 epoch. cos_sim stuck at 0.06 (random).

### Root Cause
Old Chain Head was trained on self-denoise critic where "good chain" = sequence leading toward the QUERY. New ConditionalCritic leads toward the ANSWER. These are fundamentally different energy landscapes. The old chain head's knowledge is not transferable.

### Fix
Three-phase training, each with separate scripts:
1. **Phase 1**: Train ConditionalCritic solo (`train_critic.py`) → rank_acc ≥ 0.95, cos_sim ≥ 0.25
2. **Phase 2**: Train Chain Head v2 from scratch on frozen critic (`train_chain_v2.py`)
3. **Phase 3**: Joint fine-tuning (existing `train_autoregressor.py`)

### Rule
When replacing the critic architecture, ALL downstream components trained on the old critic must be retrained from scratch. Never assume transfer learning works across fundamentally different energy landscapes.

### Additional Fix
`langevin_lr` was 0.01 — far too small for SONAR space. With E_pos≈-0.12 and gradients≈0.03, displacement after 10 steps was 0.003 (1.5% of v_query norm). Increased to 0.3 with 30 steps.
## 2026-04-06 - Self-denoise critic CANNOT solve QA: energy minimum is at v_query, not v_answer

### Pattern
Stage 3 pipeline achieved cos_final ≈ 0.35-0.40 and 97% energy reduction simultaneously. Energy dropped correctly but cosine to target plateaued.

### Root Cause
Self-denoise training teaches E(v_clean, v_noisy) → minimum at v_candidate = v_query. At QA inference, v_query ≠ v_answer, so the critic drives candidate toward the question, not the answer. The 0.35 cosine is roughly cos(query, answer) plus noise drift — critic is working as designed, but it's the wrong design for QA.

### Fix
Replace self-denoise critic with ConditionalCritic that takes (v_query, v_candidate, v_context) and is trained on QA triplets: E(q, correct_answer, ctx) < E(q, distractor, ctx). The energy minimum is now at the correct answer.

### Rule
Never use a self-denoise critic for conditional generation tasks. If the target differs from the query, the critic MUST be trained with (query, target) pairs, not (target, target+noise). The critic's energy minimum determines where Langevin converges, and self-denoise always converges to the query.

## 2026-04-06 - System 2 v_final selection must prefer cosine-best over chain-best

### Pattern
System 2 always selected `v_best_chain` (lowest chain head energy) as the final output, even when `v_best_cos` (highest cosine to target) was better.

### Root Cause
Chain head energy evaluates chain coherence, not answer quality. An untrained or weak chain head may assign low energy to vectors that are poor answers.

### Fix
Changed priority: cosine-best > chain-best > pairwise-energy-best. When v_target is available, always prefer the state that was closest to the answer.

### Rule
For any multi-objective inference (chain + pairwise + cosine), the selection priority should match the actual goal. For QA, cosine-to-answer is the primary signal.

## 2026-04-05 - SONAR text decode must not rely on `v_final` only in self-denoise diagnostics

### Pattern
Text diagnostics produced garbage even when trajectory briefly reached better cosine states.

### Root Cause
`run_text_inference` decoded only the last state (`v_final`). In System2, late steps can move away from the best state, and decode quality is highly sensitive to off-manifold drift.

### Fix
- Decode through `decode_safe` for robustness.
- In self-denoise diagnostics, select the best trajectory step by cosine to known target embedding and decode that state.
- Keep transparency by reporting both decoded selected state and decoded final state, plus `decode_source`.

### Rule
For diagnostic modes where target embedding is known, never decode only terminal state. Always log terminal metrics, but allow decode readout from the best validated trajectory state.

## 2026-04-05 - Never assume one dataset schema across stages (legacy `embeddings` vs sequence payloads)

### Pattern
Old WikiText artifacts (`wikitext_sonar_10k.pt`) use flat payload
`{"embeddings": Tensor[N, D], "texts": list[str]}`, while Stage2/Stage3 loaders expected sequence payloads (`"sequences"` or `"vectors"`), causing load/parse failures.

### Root Cause
Dataset schema evolved, but loaders were not backward-compatible and had duplicated parsing logic in several scripts.

### Fix
- Add unified loader `cebcm/data/sequence_loading.py`.
- Support `sequences`, `vectors+lengths`, `embeddings` (legacy auto-windowing).
- Route Stage2/Stage3/GUI loaders through this one parser.
- Add clearer load-time diagnostics for corrupted/LFS-pointer files.

### Rule
When data format changes, keep one canonical parser and reuse it everywhere. Never duplicate format parsing across scripts.

## 2026-04-05 - CE global-token bottleneck on short sequences (top_k_pct alone can collapse to k=1)

### Pattern
With SQuAD mean sequence length around 7, `surprise_top_k_pct=0.05` yields `int(L*pct)=0` for most samples, so CE global branch falls back to `k=1` almost always.

### Root Cause
Percent-only top-k on short sequences discretizes to one token, reducing cross-token evidence and limiting context fusion capacity.

### Fix
- Add `surprise_top_k_min_tokens` and enforce `k >= min_tokens`.
- Add `global_include_last_token` so the last valid token (question in CE pretrain) is always visible in global attention.
- Log approximate effective `k` at train start.

### Rule
For short-sequence regimes, never rely on percent-only top-k selection. Always enforce a minimum token count.

## 2026-04-05 - High flow_cos can coexist with near-random eval_cos when sigma_init is mis-scaled

### Pattern
Flow IPP run reached `flow_cos ~0.93`, but `eval_cos_mean ~0.016` and `eval_l2 ~0.298` (near random on SONAR sphere).

### Root Cause
`sigma_init=0.5` is out-of-scale for SONAR norms (~0.2051). Velocity fitting improves on training interpolation, while finite-step sampling from oversized noise fails to land near targets.

### Rule
For SONAR-space Flow IPP, keep `sigma_init` in the same scale band (`~0.03..0.08`) and treat high `flow_cos` as insufficient without endpoint/sample metrics.

## 2026-04-04 - Respect requested IPP regime (MLP vs Flow) and lock mode explicitly

### Pattern
User asked for MLP-focused analysis, but changes were made in FlowIPP path first, causing mismatch with requested experiment track.

### Fix
- Set `ipp.mode` explicitly in active stage2 configs for the intended run.
- Print effective IPP mode/class at training start in all stage2 trainers.
- Treat Flow and MLP as separate experiment branches; do not silently mix.

### Rule
Before changing IPP math, verify the active regime (`mlp`/`flow`) in config and logs. If user requests one regime, all proposed code/config changes must target that regime first.

## 2026-04-04 - FlowIPP needs endpoint supervision, not only velocity supervision

### Pattern
`FlowIPP` could optimize velocity-field loss (`flow_cos`) while `eval_cos_mean` stayed near the ~0.60 ceiling.

### Root Cause
Pure CFM objective supervises local velocity at random `(V_t, t)` points, but does not directly penalize final integration error of `V_init` after finite-step ODE rollout.

### Fix
- Add endpoint loss during training: integrate from sampled `V_noise` to `t=1`, then supervise `(V_end, V_target)` with MSE+cosine.
- Keep CFM loss as base objective; endpoint term is auxiliary and config-controlled.
- Add best-of-k eval diagnostics to distinguish "single-sample quality" from "mode coverage".

### Rule
For flow-based IPP, always monitor and, when needed, optimize endpoint sample quality explicitly; velocity fit alone is insufficient for retrieval-level cosine targets.

## 2026-04-04 - Joint CE+IPP: never assume freeze state, log it explicitly

### Pattern
Joint training plateaued around cosine ~0.60, and there was uncertainty whether IPP was actually frozen or updating.

### Fix
- In joint trainer, explicitly force `requires_grad=True` for CE/IPP by default after checkpoint load.
- Print trainable parameter counts for both modules before training.
- Log `ce_grad_norm` and `ipp_grad_norm` during training to verify updates are real.

### Rule
If a module should be trainable in a multi-module stage, always prove it in logs (trainable params + grad norm), not by assumption.

## 2026-04-04 - CRITICAL: Chain Head energy_norm_margin=5.0 causes energy explosion and learning collapse

### Pattern
Phase A training with hard negatives showed:
- E_pos: 0.04 → 0.16 → 0.51 → 0.90 → 1.63 → 3.18 → 4.23 → 4.59 (8 epochs)
- Loss stuck at ~2.7 (ln(16)=2.77 = random for 1+15 classes). **Barely above random!**
- adj_swap accuracy: 0.63 → 0.55 → 0.46 → 0.40 (anti-learning, below random)
- energy_gap: 0.005 → 0.16 (negligible relative to E=4.8)

### Root Cause
`energy_norm_margin=5.0` allows energy to grow far before regularization kicks in.
With τ=0.07 InfoNCE: logits = -E/τ. When E=4.8, logit = -69. All logits are huge negative with tiny gap → softmax outputs near-uniform → zero learning signal for subtle differences (adj_swap).

### Fix
- `energy_norm_margin`: 5.0 → **1.0** (energy stays in workable range for τ=0.07)
- `lambda_energy_norm`: 0.01 → **0.05** (stronger scale anchor)
- With E ∈ [-1, 1] and τ=0.07: logits ∈ [-14, 14] — normal softmax operating range

### Rule
1. For InfoNCE with temperature τ, energy magnitude must stay << 1/τ to avoid softmax saturation
2. energy_norm_margin should be ~1/τ × 0.1 = **1.0** for τ=0.07
3. Monitor: if E_pos and E_neg grow together with tiny gap, scale is uncontrolled
4. Loss near ln(1+N) = constant means the model is at random → check energy scale first

## 2026-04-04 - CRITICAL: Checkpoint architecture mismatch — Stage 1.5 uses radial_angular, not SimpleEnergy

### Pattern
Stage 3 Phase B `build_pairwise()` created `SimpleEnergy`, but Stage 1.5 trained `AngularEnergyCritic` + `RadialEnergyCritic` (radial_angular architecture). Checkpoint keys: `critic1_state`, `critic2_state` — not `model` or `model_state_dict`.

Additional errors:
1. Checkpoint filename: `best.pt` (actual) vs `best_critic.pt` (in config)
2. `load_pairwise` in test_inference.py crashed with KeyError because format didn't match

### Rule
1. Before referencing ANY checkpoint, verify: (a) filename, (b) state dict keys, (c) model architecture class
2. Stage 1.5 = `radial_angular` → use `AngularEnergyCritic` with `critic1_state` key
3. Never default to a different model class than what produced the checkpoint

## 2026-04-03 - CRITICAL: Stage 3 config must match actually trained models, not aspirational architecture

### Pattern
`stage3_config.json` specified `norm_mode: "orthonorm"` and `activation: "groupsort"` for the pairwise critic, but:
1. The actually trained Stage 1.5 critic used `norm_mode: "none"`, `activation: "silu"`, architecture: `radial_angular`
2. Phase 2f proved orthonorm+groupsort **crushes energy capacity** to 0.13 range (E∈[0.23, 0.36]), rank_success=0.1%
3. Phase 2b (norm_mode=none, silu) achieved spread=0.56, rank_success=88%
4. Config mismatch would cause checkpoint loading failure or silent architecture incompatibility

### Root Cause
Config was written based on theoretical spec (v1.4 recommends orthonorm) rather than experimental results. Spec and practice diverged after Phase 2f failure.

### Rule
1. **Config MUST match the architecture of the actually trained checkpoint** — not the theoretical ideal
2. Before writing any model config, CHECK the training config of the referenced checkpoint
3. Phase 2f post-mortem is definitive: orthonorm+groupsort kills energy range in practice for this problem
4. Cross-reference `tasks/lessons.md` for known failures before choosing architecture settings

## 2026-04-03 - Chain Head rank_acc 0.93 in 4 epochs = shortcuts, not reasoning

### Pattern
Chain Head achieved rank_acc=0.928 by epoch 4 with original negatives. All 4 negative types were trivially detectable by cosine-distance heuristics, not chain coherence reasoning:
- Shuffled (full permutation): cos between neighbors drops from ~0.8 to ~0.4
- Corrupted (random vector on sphere): cos~0.0 with everything — trivial outlier
- Wrong conclusion (cross-document): cos to chain prefix ~0.3 instead of ~0.8
- Truncated: chain is shorter, visible in attention mask

### Fix
Hard negatives that defeat cosine shortcuts:
1. **Adjacent-swap** (not full shuffle): swap 1-2 neighboring pairs only
2. **Interpolated corruption** (not random noise): blend α=0.3-0.7 toward pool vector, stays on-manifold
3. **Same-document wrong conclusion**: replace last vector with another from SAME sequence (cos to prefix stays ~0.7-0.9)
4. **Increased negatives**: 7→15 per positive
5. **Dual gate**: rank_acc > 0.85 AND energy_gap > 1.0

### Rule
**Never use negatives that can be detected by simple feature statistics (mean cosine, norm outliers).** Always verify: "can a linear probe on [mean neighbor cos, min cos, norm variance] achieve similar accuracy?" If yes, negatives are trivial.

## 2026-04-03 - ALiBi not RoPE for attention over SONAR vectors

### Pattern
Chain Head initially implemented with RoPE (Rotary Position Embeddings). RoPE rotates Q and K vectors, which breaks SONAR semantic geometry — cosine distances between SONAR embeddings are no longer preserved after rotation.

### Fix
Use ALiBi (Attention with Linear Biases) instead. ALiBi adds additive bias -m|i-j| to attention scores without modifying Q, K, or V vectors. SONAR distances are fully preserved. ALiBi slopes provide sufficient order sensitivity for short chains (5-20 elements).

### Rule
**Never use RoPE on SONAR embeddings.** Any positional encoding that modifies the embedding vectors (RoPE, learned position embeddings added to input) corrupts SONAR geometry. Use ALiBi or other score-level biases that leave vectors untouched.

## 2026-04-02 - FlowIPP sigma_init must match SONAR embedding scale

### Pattern
FlowIPP with sigma_init=0.5 and 10 integration steps: train velocity cos=0.89 but eval sample cos=0.011 (100x gap). Velocity prediction was excellent but ODE integration from noise→target was completely broken.

### Root Causes
1. **sigma_init=0.5 vs target_norm=0.2051**: Starting noise V_0 ~ N(0, 0.25·I) has norm ≈ √1024·0.5 ≈ 16.0, while target lives on sphere of radius 0.2051. Signal-to-noise ratio = 80:1. 10 Euler steps cannot traverse this distance.
2. **10 integration steps**: Far too few for the long noise→target path. Discretization error accumulates.
3. **No target_norm projection at eval**: Results not projected to SONAR sphere after generation.
4. **No direct sample-quality loss**: Training only supervises velocity at random interpolation points V_t, never the final sample V_1. Errors at boundary (t→0, t→1) accumulate during integration.

### Fix
- sigma_init: 0.5 → 0.05 (same order as SONAR norms)
- n_integration_steps: 10 → 50
- Always call sample(target_norm=0.2051)
- Consider midpoint solver instead of Euler

### Rule
**For any flow matching model, sigma_init MUST be calibrated to the data scale.** If targets have norm ~0.2, starting noise should have norm ~0.05-0.1, not ~16.0. The train-eval gap in flow matching = ODE integration quality gap.

## 2026-04-02 - SurprisePredictor overfits without dropout/early-stopping

### Pattern
SP standalone 30 epochs: train loss 0.30→0.15, val loss improving until epoch 4 (0.3075) then steadily increasing (→0.3522). No dropout, no early stopping. SSM with d_state=64 memorizes sequences.

### Fix
- Add dropout=0.1 to SSM and prediction head
- Early stopping with patience=5 on val loss
- SP is frozen forever (per spec) — val loss quality at freeze point = permanent quality

### Rule
SP should train for ~5-8 epochs max (early stopping by val loss). Overfitting is permanent since SP is never unfrozen.

## 2026-04-02 - CE pretrain cos=0.60 is an MSE mode-averaging ceiling, not a bug

### Pattern
ContextEncoder pretrain with MSE head plateaus at cos=0.60 within 2-3 epochs. L2 keeps improving (1.93→0.84) but cosine doesn't. This is EXPECTED: MSE averages multiple valid continuations in WikiText, the geometric average of modes has cos≈0.60.

### Rule
CE pretrain cos=0.60 is the architectural ceiling for MSE-based pretrain. The real test is whether FlowIPP can EXCEED this ceiling by generating specific modes rather than averages. If FlowIPP (with fixed sigma/steps) gives cos>0.60, flow matching is working.

## 2026-03-31 - Standard MALA makes well-trapping WORSE, not better

### Pattern
Researched MALA (Metropolis-Adjusted Langevin) as inference improvement. Standard MALA rejects uphill moves (energy increases). But our failure mode is the OPPOSITE: particles fall into spurious low-energy wells. Standard MALA would ALWAYS accept steps into wells (energy decreases, α=1) and REJECT escape attempts (energy increases, α≈0).

### Solution
Trust-Region Metropolis (TRM): two-sided acceptance filter.
- Standard MH part: reject discretization errors going uphill.
- Trust region part: reject suspiciously large downhill jumps (well entry detection).
- Combined: band-pass filter on per-step energy changes.

### Rule
- Before implementing any sampling algorithm, verify its assumptions match your failure mode.
- For EBMs with spurious wells, standard MALA is counterproductive.
- The trust-region bound (max descent per step) is the key ingredient for well prevention.

## 2026-03-31 - Hybrid dual-critic v1 diagnosis: 6 bugs causing dir plateau and E=147

### Symptoms
- dir metric plateaus at ~0.313 (Phase 2i achieved 0.77)
- rank_success plateaus at ~0.805 (Phase 2i achieved 0.895)
- E_start=147 at inference (should be O(1))
- ereg spikes to 0.405

### Root Causes & Fixes
1. **No output layer zero-init**: Default Kaiming init on angular (4106-d input) produces large E at OOD points. Fix: `nn.init.zeros_` on output layer weight/bias.
2. **CD routed to radial only**: Angular head had NO well suppression. Angular wells (weight 0.45-0.75 in combined E) dominate inference. Fix: `route_cd_to_radial_only=false`.
3. **lambda_direction_angular=0.1**: 3x weaker than proven Phase 2i (0.3). Directly explains dir plateau. Fix: increase to 0.3.
4. **energy_reg_universal=false**: Only E(clean) penalized, inference starting points uncontrolled. Fix: enable universal.
5. **No energy_output_clamp**: E=147 causes gradient explosion in Langevin. Fix: add clamp=50.0.
6. **critic_steps_per_actor=2**: Each head gets only 1 update/actor step. Fix: increase to 4 (2 per head).

### Rule
- When splitting losses across multiple heads, VERIFY each head individually gets sufficient gradient signal.
- Well suppression (CD/floor) must apply to ALL heads that contribute to inference energy.
- Always zero-init output layer of energy networks — SOTA practice from score matching literature.
- After implementing multi-head architecture, re-derive the effective per-head lambda vs single-critic baseline.

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

## Process Lesson: Keep legacy trainer intact when user requests modular add-ons (2026-04-01)

### Summary
If the user asks for modular/standalone pipelines and explicitly says not to split an existing trainer, keep the current trainer file unchanged and add new scripts/configs around it.

### Rule
1. Treat "do not split existing trainer" as a hard compatibility requirement.
2. Implement new standalone entrypoints (`train_*`) instead of refactoring the existing monolithic script.
3. Preserve checkpoint key compatibility (`surprise_predictor`, `context_encoder`, `ipp`) across old and new scripts.

## 2026-04-04 - CRITICAL: Chain Head used ALiBi instead of RoPE — spec mismatch causing adj_swap failure

### Pattern
Chain Head implementation used ALiBi (content-agnostic distance bias) instead of RoPE (position-content binding) as specified in Appendix C line 2232. This caused:
- adj_swap accuracy: 0.6165 (barely above random 0.5) — model CANNOT detect adjacent swaps
- energy_gap: 0.024 (target 1.0) — model cannot separate positive from negative chains
- Val rank_acc peaked at epoch 1 then degraded — overfitting without learning position

### Root Cause
ALiBi adds `-m|i-j|` bias — content-agnostic, only biases attention by distance. When adjacent elements swap, the distance matrix barely changes → ALiBi cannot detect swaps. RoPE rotates Q/K vectors by position → same content at different positions produces different dot products → position-content binding enables swap detection.

### Spec Decision (Appendix C)
- **Chain Head (5-20 elements):** RoPE — order is CRITICAL, short chains minimize geometry distortion
- **Context Aggregation (50K+ vectors):** ALiBi — preserves SONAR geometry, distance bias sufficient

### Fix
Replaced ALiBiSelfAttention with RoPESelfAttention + PyTorch SDPA (Flash Attention).

### Rule
1. ALWAYS check the spec's Appendix C (Architecture Decisions) before implementing attention mechanisms
2. For order-critical tasks: use position-content binding (RoPE, learnable embeddings) — NOT content-agnostic bias (ALiBi)
3. ALiBi is ONLY for Context Aggregation where preserving SONAR vector geometry matters more than ordering
4. Use PyTorch SDPA (`F.scaled_dot_product_attention`) for automatic Flash Attention — never manual matmul+softmax

## 2026-04-04 - Chain Head Phase A overfitting: gradient penalty fighting discrimination + data overlap

### Pattern
After RoPE fix, Phase A showed: train adj_swap 0.60->0.72, val adj_swap 0.68->0.61 (overfitting). Gradient penalty grew 0.3->2.3 over training, adding up to 0.115 to loss — nearly canceling NCE improvement of 0.1. target_energy_gap=1.0 was unreachable (requires logit gap 14 with tau=0.07).

### Root Causes
1. **lambda_grad=0.05 too strong** — gradient penalty penalizes sharp energy landscapes, but discrimination between positive and near-identical adj_swap chains REQUIRES sharp gradients
2. **Pre-extracted overlapping chains** — 28964 chains from 10000 seqs via sliding window → heavy vector overlap → memorization
3. **adj_swap too subtle** — 1 swap in 15-vec chain = 7% disruption. SONAR adjacent sentences cos~0.8-0.9, single swap nearly undetectable
4. **target_energy_gap=1.0 unrealistic** — with tau=0.07, E_gap ~0.19 gives 85% accuracy. 1.0 would be logit gap 14

### Fix
- lambda_grad: 0.05 -> 0.01 (allow sharper discrimination in Phase A)
- dropout: 0.1 -> 0.2 (regularize overfitting)
- target_energy_gap: 1.0 -> 0.2 (realistic for tau=0.07)
- adj_swap: scale swap count with chain length (min=L//4, ~25% disruption)
- On-the-fly chain extraction: random chain per sequence per access (not pre-extracted overlapping windows)

### Rule
1. Gradient penalty must be WEAK during contrastive learning phase — smooth landscape is a Phase B concern (Langevin dynamics)
2. target_energy_gap should be ~3*tau for realistic convergence
3. Negative difficulty must scale with chain length — fixed count of perturbations dilutes as L grows
4. Random data augmentation per access beats pre-extracted fixed samples for small datasets

## 2026-04-04 - CRITICAL: System2 must share Langevin math path with System1 and chain-head must guide gradients

### Pattern
Stage3 diagnostics showed `system1` consistently outperforming `system2` despite higher step budget in `system2`.
Root cause was architectural mismatch:
- `system1` used PID Langevin (`run_langevin`) with robust stopping behavior.
- `system2` used a separate plain GD+noise loop and only used Chain Head for periodic eval/backtrack.
- With `backtracks=0`, Chain Head had near-zero control over trajectory, so long runs drifted from best cosine point.

Additional metric issue:
- Text-mode reported `cos(final,target)` against the input embedding (self-denoise), which can punish semantically valid QA-style outputs.

### Fix
- Reworked `system2` to PID-style updates with the same mathematical components as `run_langevin` path.
- Added chain-guided gradient term to the update objective (`pairwise + w * chain`).
- Kept chain eval/backtrack as secondary control.
- Enforced safe limits: `system1=10 steps`, `system2<=50 steps`, `max_chain_len<=20`.
- Made metric semantics explicit (`target_objective=qa|self_denoise`) and fixed text-mode `both` handling.
- Fixed data target selection to sequence endpoint (`seq[-1]`) for QA-style diagnostics.

### Rule
1. Never maintain separate optimization math for System1/System2 unless explicitly required by spec and validated by ablation.
2. If Chain Head is part of System2, it must influence gradients, not only monitor/backtrack.
3. Cap inference hyperparameters to chain-head training distribution (avoid OOD chain length).
4. Every cosine metric must declare target semantics (`qa` vs `self_denoise`) to avoid false regressions.

## 2026-04-04 - Stage3 gating must be strict and deterministic

### Pattern
Using only random weighted sampling for mode selection can violate hard policy requirements
("no System2 before threshold", "30/70 after unlock") due sampling variance.

### Rule
1. If mode policy is contractual, encode it as explicit state machine (`locked -> unlocked`) with checkpoint persistence.
2. Unlock `System2` only from validation metric threshold (default simple-task `pairwise_rank_acc >= 0.95`), optionally with consecutive eval hits.
3. After unlock, use an exact per-epoch ratio schedule (counted batches + shuffle), not just probabilistic weights.
4. Add post-unlock quality floor monitoring with fail-streak tracking and warnings in logs.

## 2026-04-04 - Keep fallback defaults aligned with model/spec math

### Pattern
Utility-layer fallback defaults in `build_ipp` drifted from model/spec defaults
(`sigma_init=0.5`, `n_integration_steps=10`) and could silently destabilize training
when a config misses these fields.

### Rule
1. Builder defaults must match model dataclass defaults unless there is an explicit documented override.
2. For SONAR-space IPP, keep fallback `sigma_init` in same norm scale (`~0.05`, not `0.5`).
3. Integration step defaults should preserve expected solver behavior (`50` for Flow IPP baseline).

## 2026-04-04 - Context Global Attention needs explicit position-content signal

### Pattern
Relying on ALiBi-only bias in global-token attention is insufficient for learning
position-content binding. ALiBi is distance bias, not content-anchored positional encoding.

### Rule
1. For context global-token attention, inject explicit absolute positional encoding into Q/KV.
2. Preserve true token positions for selected global tokens; do not lose chronology when top-k filtering.
3. Keep ALiBi optional as extra bias, never as the only positional mechanism for this head.

## 2026-04-06 - Hand-written Langevin loops MUST include sphere projection

### Pattern
CompositeCritic training script had hand-written Langevin assessment loops (train_step and eval_step) that omitted tangent projection and sphere projection. The production langevin.py applies both when target_norm is set. Without projection, ||v|| drifted from 0.2051 to 0.40+ over 30 steps despite the AnalyticalRadialGuard (λ=5 too weak vs discrete step accumulation).

### Root Cause
Angular gradient is tangential at the computation point, but after a discrete Langevin step the norm changes slightly. Over 30 steps this drift accumulates. The radial guard's restoring force is weak (quadratic near target) and tamed gradient further dampens it.

### Rule
1. NEVER write Langevin loops without sphere projection when target_norm is known. Always: `v = F.normalize(v) * target_norm` after each step.
2. ALWAYS add tangent projection before the step: remove radial component from gradient before applying update.
3. Prefer using the production `run_langevin()` from `cebcm/inference/langevin.py` instead of hand-writing loops — it already handles all projections correctly.
4. If you must hand-write a loop (e.g., for training with create_graph=True), copy the exact projection pattern from langevin.py.

## 2026-04-06 - Langevin/Flow/ODE navigation is fundamentally broken in 1024D multi-basin QA

### Pattern
ALL iterative navigation methods fail for conditional QA search in SONAR 1024d:
- Langevin: -∇E points to NEAREST basin, not TARGET. With 81k answer basins, nearest ≠ target.
- Path-contrastive: perfect energy ordering (violations→0.001) but cos_sim FELL from 0.20 to 0.03.
  Proved that the problem is NOT 2nd-order dominance — path-contrastive is purely 1st-order.
- Flow/ODE: compounding integration error (train cos=0.93, eval cos=0.017).
- Direction loss: 2nd-order gradient dominates 1st-order in MDSM, but NOT the root cause.

### Root Cause
In 1024D with 81k competing answer basins, the gradient landscape has too many local attractors.
A ranking critic can perfectly DISCRIMINATE (rank_acc=0.99) but cannot NAVIGATE because:
- "Ranking teaches VALUES not GRADIENTS" (lesson L1252)
- Energy landscape is locally smooth but globally multi-modal
- Any iterative method (Langevin, Flow, ODE) gets trapped by nearest-basin gravity

### Rule
1. NEVER use iterative navigation (Langevin, Flow, ODE, direction loss) for conditional QA in high-D multi-basin spaces.
2. Use DIRECT PREDICTION (autoregressive generation) for answer synthesis.
3. Keep the trained critic as a RERANKER only (it ranks perfectly, just can't navigate).
4. For QA: ChainGenerator (autoregressive Transformer decoder) + CompositeCritic (reranker).

## 2026-04-06 - CE/IPP/SP are dead ends for SONAR QA

### Pattern
Extensive experimentation proved all three approaches hit hard ceilings:
- CE (Context Encoder): cos_sim plateaus at ≈0.60 — information ceiling
- IPP (FlowIPP): train cos=0.93, eval cos=0.017 — catastrophic generalization failure
- IPP (MLPIPP): just matches CE ceiling, adds nothing
- Joint CE+IPP: no improvement over CE alone
- SP (Surprise Predictor): overfits

### Rule
1. Do NOT use CE, IPP, or SP in new architectures. They are dead code.
2. For QA answer generation, use autoregressive prediction (ChainGenerator), not encoding/denoising.
3. The only surviving component is CompositeCritic (Angular + Radial Guard) as a reranker.

## 2026-04-07 - ChainGenerator V1 training analysis: autoregressive collapse after step 2-3

### Pattern
ChainGenerator (101.8M params, 6-layer decoder) trained for 35 epochs on HotpotQA.
- System 1 (1 step): tf_cos=0.245, gen_cos=0.195 — genuine learning but low
- System 2 (3 steps): gen_cos=0.68, tf_cos=0.37 — phase transition at E15!
- System 2 (5 steps): gen_cos degrades from 0.687 to 0.640, cos_last=0.11

At inference, decoded chain shows collapse:
```
Step 1: "The Theoretical theory of relativity..." (coherent)
Step 2: "Theoretical Theory of Quantum Physics..." (still connected)
Step 3: "Scientology" (one word)
Step 4-20: "Theoretical", "Theoretical"... (fixed point loop)
```

### Root Causes Identified from Attention Analysis
1. **Cross-attention is trivial**: single KV token (v_query) → softmax always = 1.0 → cross-attn reduces to a fixed linear transform of v_query, identical for every position. No dynamic conditioning.
2. **Self-attention degenerates**: most heads learn identity (diagonal) or "look at previous only". No long-range patterns. Teacher forcing removes incentive to learn deep chain analysis.
3. **Error accumulation**: at generation time, error from step 1 feeds into step 2 etc. By step 3-4, model enters attractor basin (fixed point like "Theoretical").
4. **tf_cos vs gen_cos anomaly explained**: tf_cos = mean cos over ALL chain steps (including hard intermediates). gen_cos = cos of LAST generated step to GT answer. gen_cos >> tf_cos because the final step metric is different from the average.

### Rule
1. Single-token cross-attention is degenerate. For meaningful conditioning, either:
   a. Project v_query through multiple "pseudo-tokens" (learned query decomposition)
   b. Use v_query as additive bias instead of cross-attention
   c. Inject v_query at multiple points (not just cross-attn)
2. Teacher forcing alone causes exposure bias in SONAR space. Consider:
   a. Scheduled sampling (mix GT and predicted inputs during training)
   b. Add noise to teacher-forced inputs to simulate generation errors
3. Chain length curriculum must be more conservative. 3 steps was the sweet spot; 4-5 broke the model. Start with 1-3 and stay there until metrics stabilize.
4. Monitor cos_last (final step quality) as the PRIMARY metric, not cos_sim_mean.

## 2026-04-08 - Free-run objective and memory-bank context are mandatory for autoregressive QA

### Pattern
Teacher-forced-only chain supervision can look stable in logs but collapses in real rollout (loops/repetition) and gives degenerate best-of-N candidates.

### Root Cause
- Exposure bias: model never optimized on its own generated trajectories.
- Cross-attention with a single context token cannot learn useful selection over evidence.
- Deterministic candidate generation makes reranking ineffective.

### Fix
- Use composite objective with free-run and in-batch contrastive terms:
  - `L = lambda_step*L_step_masked + lambda_ans*L_final_answer + lambda_roll*L_free_run + lambda_rank*L_inbatch_contrastive`.
- Feed a context memory bank (`query + evidence slots`) into cross-attention with proper mask.
- Make system2 candidate generation stochastic and anti-loop constrained (repeat penalty, repeat-ban, stagnation early-stop).

### Rule
1. For autoregressive QA, never rely on teacher-forcing loss alone.
2. If cross-attention key length is 1, do not treat attention plots as evidence of context reasoning.
3. Reranker quality requires candidate diversity; deterministic best-of-N is a no-op.
4. Train and inference horizons must be aligned or explicitly capped.

## 2026-04-13 - Diffusion Forcing in SONAR space must use SONAR-scaled noise

### Pattern
Diffusion Forcing uses DDPM-style per-position noise levels, but raw DDPM epsilon `N(0, I)` is mathematically wrong for 1024D SONAR vectors with norm near 0.205. Raw epsilon has expected norm near 32 and recreates the same failure mode as uncalibrated rollout noise: the noise dominates semantic signal before attention can use it.

### Rule
1. In SONAR-space diffusion objectives, default epsilon scale must be `target_norm / sqrt(d_model)`, not 1.0 per dimension.
2. Keep the AR/free-run objective active when adding Diffusion Forcing. DF is an additional robustness objective, not proof that rollout works.
3. Train with independent per-position noise levels, but validate DF at a fixed noise level so diagnostics are comparable across epochs.
4. Monitor `roll_cos`, `roll_ans`, and `ans_cov` as the real QA rollout metrics. `df_cos` only proves denoising skill at a chosen noise level.

## 2026-04-13 - Diffusion timestep embeddings and Min-SNR weights must match objective type

### Pattern
A diffusion timestep embedding normalized to [0, 1] weakens sinusoidal conditioning for small K (e.g. K=64): most frequency channels become nearly constant, so the model can under-use the noise level. Also Min-SNR weights depend on the prediction target. `clipped_snr/snr` is for epsilon/pred-noise style objectives, not pred-x0.

### Rule
1. Use raw diffusion levels (0..K-1) for sinusoidal timestep embeddings unless the embedding implementation explicitly expects normalized continuous log-SNR.
2. For pred-x0 DF loss, use clipped-SNR style weights (`min(snr, gamma)` optionally normalized by gamma), not `min(snr, gamma) / snr`.
3. Keep Min-SNR disabled by default until an ablation proves it improves rollout metrics, not just DF denoising metrics.

## 2026-04-13 - Diffusion eval must fix both timestep and epsilon noise for stable diagnostics

### Pattern
Fixing only `df_eval_noise_level` is not enough for stable validation metrics. If epsilon is still sampled from the global RNG, `val_df_cos` changes across epochs even when the model is unchanged, making DF diagnostics harder to interpret.

### Rule
1. For DF validation diagnostics, use a fixed noise level and deterministic Gaussian epsilon per validation batch.
2. Do not use deterministic eval noise for training; train still needs independent stochastic noise levels and epsilon samples.
3. Keep rollout metrics (`val_roll_cos_last`, `val_roll_ans`) as the selection metric; deterministic DF eval is a diagnostic, not the final QA metric.

## 2026-04-13 - Answer-repeat padding must not drive System1/System2 answer supervision

### Pattern
`answer_repeat_pad` makes the last valid chain position an answer duplicate, not necessarily the first answer position. If answer coverage is inferred from `chain_len <= target_steps`, System2 can delay `L_ans`, rollout-answer metrics, and rank diagnostics until all repeated answer pads enter the horizon. That makes multi-step training misreport whether the actual answer is supervised.

### Rule
1. Store and propagate the first answer position (`answer_pos`) separately from `chain_len`.
2. System1 (`target_steps=1`) must train directly on `chains[answer_pos]`, not on the first reasoning step or the final repeat pad.
3. System2 should use prefix-aligned targets, but activate answer-specific losses and metrics as soon as `answer_pos < target_steps`.
4. Rank diagnostics and answer metrics should compare rollout at `answer_pos` to the first answer vector, not to an arbitrary last valid repeat.

## 2026-04-15 - Adam state reset is not neutral
- Do not treat Adam/AdamW moment zeroing as a harmless recovery action. After `exp_avg` and `exp_avg_sq` are cleared, the next finite micro-gradient can produce an almost sign-like full-LR update because Adam normalizes by the freshly tiny second moment (with bias correction, `g / (|g| + eps)`). This can create NaN -> dead -> wake -> NaN oscillations.
- Always verify recovery conclusions against JSONL metrics, not only terminal logs. Terminal logs may omit `zombie_reset`, `grad_sanitized`, or cumulative recovery counters.
- If answer coverage recovers only when horizon reaches a minimum length, do not start System2 below that horizon; too-short prefixes can make QA answer supervision mathematically absent.

## 2026-04-16 - Adam-zombie root cause: momentum zeroing, not NaN itself
### Context
NaN in gradients is a transient numerical event (bf16 overflow, bad batch).
The real damage comes from the RESPONSE to NaN, not NaN itself.
### Problem
Zeroing Adam exp_avg/exp_avg_sq on NaN grad creates a catastrophic state:
- Next non-zero gradient produces update ≈ lr · g / √(ε) ≈ lr · g · 1e4
- This 10000× amplified step destabilizes the model
- val metrics freeze at exact values because EMA shadow stops receiving updates
- Model enters "zombie" state: finite loss, finite params, but grad_norm=0
### Rule
1. On NaN grad: scrub grad to zero, but NEVER touch Adam momentum buffers.
   Zero grad → Adam decays momentum by β₁/β₂ → natural "no signal" handling.
2. Only zero Adam buffers that are themselves non-finite (defense-in-depth).
3. Zombie reset (restore from EMA shadow) should preserve healthy momentum.
4. For fixed-horizon curriculum: use `horizon_schedule` config key with
   `[[steps, epochs], ...]` to train at each horizon for a fixed duration,
   instead of continuously incrementing (which doesn't let the model converge
   at any single horizon before moving to a harder one).
