# Research3 Agent Report: Simple Pairwise / Actor+Critic Stage1 Audit

## Scope

This report audits the current Stage1 `simple` pipeline and the current `actor_critic`
variant that is built on top of the pairwise critic.

Audited code:

- `C:/Coding/Python/CERBER/cebcm/models/energy.py`
- `C:/Coding/Python/CERBER/cebcm/models/actor.py`
- `C:/Coding/Python/CERBER/cebcm/training/losses.py`
- `C:/Coding/Python/CERBER/experiments/01_denoising_poc/train.py`
- `C:/Coding/Python/CERBER/cebcm/inference/langevin.py`
- `C:/Coding/Python/CERBER/cerber_gui/sota_eval.py`
- `C:/Coding/Python/CERBER/cerber_gui/app.py`

External sources used:

- Li et al., "Learning Energy-Based Models in High-Dimensional Spaces with Multiscale Denoising-Score Matching" (Entropy 2023): <https://www.mdpi.com/1099-4300/25/10/1367>
- Gao et al., "Learning Energy-Based Models by Diffusion Recovery Likelihood" (ICLR 2021): <https://arxiv.org/abs/2012.08125>
- "Should EBMs Model the Energy or the Score?" (ICLR EBM workshop submission): <https://openreview.net/pdf?id=9AS-TF2jRNb>
- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM, 2022): <https://arxiv.org/abs/2206.00364>
- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023): <https://arxiv.org/abs/2210.02747>
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (ICLR 2023): <https://arxiv.org/abs/2209.03003>
- Song et al., "Consistency Models" (ICML 2023): <https://arxiv.org/abs/2303.01469>
- Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019): <https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf>
- Xu and Durrett, "Spherical Latent Spaces for Stable Variational Autoencoders" (EMNLP 2018): <https://aclanthology.org/D18-1480.pdf>
- Ta et al., "Conditional Energy-Based Models for Implicit Policies: The Gap between Theory and Practice" (2022): <https://arxiv.org/abs/2207.05824>

---

## Executive Verdict

The current `simple` / `actor_critic` Stage1 pipeline has three major structural problems:

1. The `actor_critic` critic is trained with local ranking only, but inference uses its gradient field.
   This is mathematically misaligned. Ranking losses can learn scalar ordering without learning a useful
   score field for Langevin.

2. The current `actor_energy_loss` directly conflicts with the critic ranking constraints.
   The critic is trained to keep `E(clean) < E(actor) < E(noisy)`, while the actor coupling term pushes
   `E(actor) < E(clean)`. This creates "sub-clean" attractors and explains why energy can improve while
   semantic quality degrades.

3. The actor is trained as a supervised denoiser with direct access to the clean target embedding as
   `v_query`. That is not the future CERBER actor task. In Stage2+, the actor must propose candidates
   from query/context, not from the answer embedding itself.

Bottom line:

- The current pipeline is a valid local denoising sandbox, but not yet a valid training basis for a
  future stable CERBER actor+critic system.
- More layers or more parameters are not the first fix.
- The first fix is objective alignment: critic must learn gradients that are valid for inference, and
  actor must be trained against the same geometry and endpoint semantics that will exist at inference.

---

## Current Pipeline, Precisely

### 1. Pairwise critic (`SimpleEnergy`)

`SimpleEnergy` takes:

- `v_query`
- `v_candidate`
- pairwise interaction features `[v_q, v_c, v_q - v_c, v_q * v_c, sigma_embed]`

and outputs a scalar energy.

Current Stage1 modes:

- `mdsm`: critic is trained via multiscale score matching.
- `margin_contrastive`: critic is trained via scalar ranking only.
- `actor_critic`: actor is trained by direct denoising regression, critic is trained via scalar triplet ranking only.

Important point:

- In `mdsm`, the critic is trained for gradient correctness.
- In `actor_critic`, the critic is not trained for gradient correctness, but Langevin still uses `grad E`.

This is the first major contradiction.

Relevant code:

- `multiscale_dsm_loss(...)` in `cebcm/training/losses.py`
- `train_epoch_actor_critic(...)` in `experiments/01_denoising_poc/train.py`

### 2. Actor

The actor predicts a latent update:

`v_next = v_current + step_size * actor(v_query, v_current, sigma)`

It uses:

- the same pairwise feature template as the critic,
- sigma conditioning,
- optional tangent projection,
- sphere projection to `target_norm`.

This is a first-order refinement model, which is computationally attractive.

But the actor is currently trained with access to `v_clean` as `v_query`, i.e. it is solving:

"Given the target embedding and a noisy version of it, output the denoising step."

That is not the future CERBER actor problem.

### 3. Inference

Inference uses:

- optional actor rollout,
- then critic-guided Langevin with `grad E`,
- tangent projection + sphere projection.

The entire system therefore assumes that:

1. the critic gradient is meaningful,
2. the actor update geometry matches the projected inference dynamics,
3. the energy minimum structure is semantically aligned.

Today, these assumptions are not jointly satisfied.

---

## Why Energy and L2 Can Improve While Cosine / Manifold Metrics Degrade

This is not one bug. It is a combined effect of objective mismatch, sphere geometry, and weak manifold control.

### A. Raw L2 and cosine are not equivalent here

For arbitrary vectors:

`||x - y||^2 = ||x||^2 + ||y||^2 - 2 ||x|| ||y|| cos(x, y)`

So L2 and cosine only become monotonic with each other if norms are controlled.

In this codebase:

- `v_clean` comes from SONAR data and has varying norm.
- the actor and Langevin path can project outputs to a fixed `target_norm`.

Therefore:

- the model can reduce raw L2 partly by norm adjustment,
- while worsening angle / cosine,
- which is much closer to the actual semantic relation in SONAR-style latent spaces.

On a common sphere, L2 and cosine are tied; off a common sphere, they can diverge strongly.

Implication:

- raw L2 must not be treated as the primary semantic metric here.
- geodesic / angular metrics should dominate.

This is exactly why angular-margin methods such as ArcFace emphasize geodesic/angular separation rather than
plain Euclidean objectives in normalized embedding spaces. ArcFace explicitly motivates additive angular margins
because geodesic distance on the hypersphere is the right geometry for normalized embeddings:
<https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf>

### B. The actor loss is defined in delta-space, not final-state geometry

Current actor training compares:

- `pred_delta = v_actor - v_noisy`
- `target_delta = v_clean - v_noisy`

with:

- directional cosine loss on deltas,
- vector MSE on deltas,
- log-magnitude loss on delta norms.

But the actual update path is:

1. predict delta
2. optionally tangent-project delta
3. add it to current vector
4. project result to sphere

This means the training loss is not applied on the final state after projection.

So even if delta loss looks good:

- final angular error can still be bad,
- final manifold location can still be bad,
- the critic can still assign lower energy to a point that is semantically worse.

This is a geometry mismatch.

### C. The critic is only locally ordered, not globally shaped

In `actor_critic` mode, the critic is trained only on three points:

- clean
- actor output
- noisy

with margin constraints:

- `E(clean) < E(actor)`
- `E(actor) < E(noisy)`
- `E(clean) < E(noisy)`

This is local ordering.

It does not enforce:

- correct gradient direction away from arbitrary noisy points,
- global manifold structure,
- calibration against other clean samples,
- repulsion from OOD regions,
- angular semantics across samples.

So it is fully possible to get:

- "good" local scalar ordering,
- lower energy after refinement,
- better raw L2,
- but worse cosine, PRDC, C2ST, and kNN manifold metrics.

That is exactly what your observed logs show.

---

## Critical Findings

### Finding 1. `actor_critic` critic training is mathematically inconsistent with Langevin inference

Severity: P0

In `actor_critic` mode, the critic is trained by hinge ranking only:

- `critic_per_sample = relu(e_clean - e_actor + margin) + ...`

This teaches scalar ordering.

But Langevin inference uses:

- `v <- v - lr * grad E`

which assumes the critic gradient itself is meaningful.

The spec already identified this issue: Stage1 was moved from margin loss to MDSM because margin contrastive
does not teach a correct score field.

Yet `actor_critic` reintroduces exactly that failure mode.

Consequences:

- The critic can sort points correctly while having a useless or misleading gradient field.
- Langevin can descend to lower energy states that are not semantically closer to the target.
- Energy success rate can look good while semantic metrics collapse.

This is one of the main reasons the current actor+critic pipeline is not trustworthy.

Relevant code:

- `experiments/01_denoising_poc/train.py:628-647`
- `cebcm/inference/langevin.py`

### Finding 2. The actor-energy coupling has the wrong sign relative to the critic constraints

Severity: P0

Current critic ranking requires:

- `E(clean) <= E(actor) - margin_clean_actor`

because zero critic loss requires `e_actor >= e_clean + margin`.

But current actor coupling is:

- `actor_energy_loss = softplus(e_actor_live - e_clean.detach())`

This pushes:

- `e_actor < e_clean`

So the actor is rewarded for moving to a state that the critic ranking says should be lower than the clean target.

This is a direct objective contradiction.

Observed empirical signature:

- final state often ends up with lower energy than clean,
- while cosine becomes worse,
- i.e. the system learns "sub-clean" attractors.

This exactly matches your logs where:

- `Energy(final) < Energy(clean)`
- but cosine after refinement becomes worse.

This is not a mysterious geometry artifact. The objective literally asks for it.

Relevant code:

- critic ranking: `experiments/01_denoising_poc/train.py:633-638`
- actor energy term: `experiments/01_denoising_poc/train.py:640-647`

### Finding 3. The actor is solving the wrong task for future CERBER

Severity: P1

The current actor receives the clean target as `v_query` during training.

That means it solves:

"Given the answer embedding and a noisy perturbation of that answer embedding, predict a denoising step."

Future CERBER actor must instead solve something closer to:

"Given question/context/state, propose an initial answer embedding."

Those are not the same task.

As a result:

- current actor success does not transfer to Stage2 proposal generation,
- the actor can overfit local denoising geometry without learning answer proposal structure,
- strong Stage1 actor performance can still be a dead end for Stage2.

This is a classic train-task / deployment-task mismatch.

### Finding 4. The current actor loss does not optimize geodesic final-state quality

Severity: P1

The actor is trained on ambient-space delta errors before final sphere projection.

But your downstream semantics are effectively hyperspherical:

- cosine is the main semantic diagnostic,
- `target_norm` projection is used during inference,
- SONAR vectors are handled as normalized-direction objects in many decisions.

Therefore the actor should be trained primarily on:

- final-state cosine / geodesic loss,
- not ambient delta MSE as the main target.

ArcFace and related hyperspherical methods are strong evidence that angular objectives matter when the underlying
representation lives on or near a sphere:
<https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf>

Additional sphere-latent evidence from NLP:
<https://aclanthology.org/D18-1480.pdf>

### Finding 5. OOD control is too weak

Severity: P1

Current OOD control is mostly:

- sphere projection to `target_norm`,
- optional tangent projection,
- orthonormal / spectral normalization,
- optional gradient penalty.

What is missing:

- explicit manifold proximity penalty,
- explicit repulsion from low-density / OOD regions,
- nearest-neighbor or bank-based trust region,
- unconditional prior energy or score regularizer.

Result:

- the system can stay near the correct radius while still leaving the data manifold,
- which is exactly what zero/near-zero PRDC precision and coverage are telling you.

Norm control is necessary, but not sufficient.

### Finding 6. Training/eval endpoint semantics are still inconsistent in the training script

Severity: P1

The GUI endpoint desync was fixed, but `evaluate_denoising(...)` in `train.py` still uses:

- `v_final = result.v_final`

where `v_final` is the best-energy state, not necessarily the last executed state.

So Stage1 training checkpoints / kill criteria can still be evaluated on a different endpoint semantics than
what the live trajectory actually reached.

This can distort:

- cosine after,
- energy after,
- success rate,
- comparisons with GUI diagnostics.

Relevant code:

- `experiments/01_denoising_poc/train.py:843-858`

### Finding 7. `simple` MDSM critic is the better mathematical base, but it is still incomplete

Severity: P2

Compared with `actor_critic`, the pure `mdsm` critic is much better aligned with Langevin, because it trains
the gradient field directly.

That is consistent with the Stage1 spec and with Li et al. 2023:
<https://www.mdpi.com/1099-4300/25/10/1367>

But it is still incomplete because:

- it is local denoising supervision, not global conditional relevance,
- it does not solve mode proportion / conditional generalization by itself,
- it does not include manifold prior modeling,
- it still relies on second-order autograd and expensive orthonormalization.

So the correct interpretation is:

- `mdsm` is the more correct critic pretraining base,
- but not the finished CERBER critic.

---

## Why the Current Actor+Critic Looks "Better" in a 2D Well but Fails Semantically

The current actor+critic often creates a visually nice well-shaped basin in a local 2D slice.

That does not prove global correctness.

Reasons:

1. A 2D slice can hide off-plane drift.
2. Local energy ordering does not guarantee correct angular semantics.
3. The critic is not constrained against other clean samples or manifold bank statistics.
4. The actor-energy term explicitly rewards states below the clean target energy.

So the current well can be understood as:

- a locally coherent attractor,
- but not necessarily an attractor centered on semantically valid states.

This is exactly compatible with:

- lower energy,
- lower raw L2,
- worse cosine,
- disastrous PRDC / kNN metrics.

---

## Recommended Upgrade Tracks

## Track A. Immediate Fixes (do these first)

### A1. Put MDSM back into the critic in `actor_critic` mode

Recommendation:

- critic loss should be hybrid:

`L_critic = lambda_dsm * L_MDSM + lambda_rank * L_rank + lambda_gp * L_GP`

Reason:

- MDSM teaches gradient field correctness.
- ranking teaches relative energy ordering around actor/noisy/clean states.
- together they align scalar energy and inference gradients.

Suggested starting weights:

- `lambda_dsm = 1.0`
- `lambda_rank = 0.25`
- `lambda_gp = 0.0 .. 0.05` only if orthonorm is relaxed

This is the highest-priority correction.

### A2. Remove the clean-vs-actor sign contradiction

Replace current actor energy coupling with a banded objective consistent with the critic order:

Option 1:

- encourage actor to beat noisy but not beat clean:

`L_actor_energy = softplus(e_actor - e_noisy) + beta * softplus(e_clean - e_actor)`

This means:

- actor should reduce energy relative to noisy,
- but should not go below clean.

Option 2:

- no actor-energy term at all initially,
- let actor learn from trajectory teacher / final-state losses only.

For debugging, Option 2 is cleaner.

### A3. Evaluate using the actually reached endpoint

Training script should use:

- `result.v_last` for user-facing denoising metrics,
- optionally log `result.v_final` separately as "best energy state".

Without this, kill criteria remain partially misleading.

### A4. Replace raw L2 as a primary Stage1 metric

Primary metrics for this latent space should be:

- cosine / angular improvement,
- geodesic distance on normalized sphere,
- manifold kNN improvement,
- OOD / prior violation metrics.

Raw L2 can stay as a secondary diagnostic only.

### A5. Log clean-minimum violation explicitly

Add:

- `below_clean_rate = mean(E(actor_or_final) < E(clean))`
- `energy_gap_clean_actor = mean(E(actor) - E(clean))`
- `energy_gap_clean_final = mean(E(final) - E(clean))`

If Stage1 denoising wants clean as the target minimum, this rate should be near zero.

Today it is not.

---

## Track B. Near-Term Architecture Upgrade

### B1. Distill the actor from critic-guided trajectories, not from clean deltas

The actor should learn:

"What update would a good multi-step refinement policy take from this state?"

not:

"What is the raw Euclidean delta from noisy to clean?"

Practical recipe:

1. Run critic-guided teacher trajectories (using the hybrid critic from A1).
2. Sample intermediate states `x_t` from those trajectories.
3. Train actor to predict:
   - next-step teacher update,
   - or final denoised state,
   - or consistency target between two points on the same trajectory.

This is much closer to future deployment.

Best SOTA directions for this:

- Flow Matching: <https://arxiv.org/abs/2210.02747>
- Rectified Flow: <https://arxiv.org/abs/2209.03003>
- Consistency Models: <https://arxiv.org/abs/2303.01469>

Interpretation for CERBER:

- critic = scalar evaluator / prior shaper,
- actor = distilled vector field / few-step policy.

This is cleaner than forcing the critic alone to solve everything with Langevin at inference.

### B2. Add actor-critic gradient alignment

Add a regularizer:

`L_align = 1 - cos(actor_delta, -grad_E(v_noisy))`

or at later rollout points.

This forces the actor to agree with the critic's descent direction, but only after the critic itself
has score supervision again.

Without MDSM or another score-valid objective, this term is dangerous.

### B3. Use final-state angular loss, not only delta loss

Recommended actor loss mix:

- geodesic / cosine loss on `v_actor_final`
- small vector loss on normalized state
- optional delta-direction loss
- trust-region penalty on step norm

Example:

`L_actor = w_geo * (1 - cos(v_actor_final, v_clean))`
`        + w_step * ||delta||_2`
`        + w_align * L_align`
`        + w_barrier * softplus(E(clean) - E(actor_final))`

This is much more aligned with actual evaluation.

### B4. Add OOD manifold prior to the critic

The cleanest version is a dual critic:

- conditional pairwise head: `E_cond(query, candidate)`
- unconditional prior head: `E_prior(candidate)`

Use:

`E_total = E_cond + lambda_prior * E_prior`

Benefits:

- pairwise head learns relevance,
- prior head learns "is this candidate on the data manifold?",
- inference no longer relies on norm projection alone for OOD control.

This is the single most promising way to reuse what is useful from the unconditional line.

Important nuance:

- the current unconditional model is not good enough to replace the pairwise critic,
- but it can still be useful as a manifold regularizer / auxiliary prior if retrained properly.

This is much more defensible than choosing one pipeline and discarding the other.

### B5. Replace local triplet ranking with batchwise conditional contrastive learning

Today the critic only sees:

- clean,
- actor,
- noisy.

It should also see:

- in-batch negatives from other clean targets,
- hard negatives from nearest-neighbor retrieval,
- same-query semantically related but wrong answers,
- teacher-trajectory negatives from actor overshoot.

The spec already points toward Focal-InfoNCE later. That direction is correct.

For Stage1.5 / Stage2-prep:

- keep MDSM for local score learning,
- add contrastive conditioning for global conditional discrimination.

Relevant warning from conditional EBM literature:

<https://arxiv.org/abs/2207.05824>

Blindly porting unconditional EBM techniques to conditional regression-like tasks is not enough.

---

## Track C. Longer-Term CERBER-Compatible Design

### C1. Critic should become a product-of-experts style energy

Future CERBER likely needs at least two constraints:

1. conditional relevance to the query/context,
2. manifold validity / linguistic plausibility of the candidate itself.

That naturally suggests:

- `E_total = E_relevance(query, candidate) + lambda * E_prior(candidate)`

This is much closer to the actual inference problem than a single pairwise scalar alone.

### C2. Actor should be a few-step policy, not a one-step denoiser

The future actor should be trained as:

- a proposal policy,
- or a distilled few-step refinement policy,
- conditioned on query/context and current latent state.

Good future targets:

- flow-matching actor,
- consistency-distilled actor,
- rectified-flow actor.

These directly address the real inference need:

- fewer critic backward passes,
- more stable refinement,
- actor and critic working in the same state distribution.

### C3. Stage1 must stop pretending "clean target is available" is enough

Current Stage1 actor learns with target exposure.

That is acceptable only as a local geometry probe.

For Stage2 readiness, you need a second benchmark:

- actor gets `query` and initial candidate,
- not the clean target embedding.

Otherwise Stage1 can pass while Stage2 still fails immediately.

---

## What To Borrow From the Unconditional Pipeline

Use the unconditional line selectively, not wholesale.

### What is worth borrowing

1. Manifold prior idea
   - unconditional energy can act as a prior term on candidate quality
   - best integrated as auxiliary head or auxiliary loss

2. Distribution metrics
   - PRDC
   - C2ST
   - MMD
   - kNN manifold proximity
   - OOD success diagnostics

3. Replay / negative-bank style thinking
   - useful for hard negatives and off-manifold detection

4. More explicit generative objectives for the actor
   - flow / consistency / rectified trajectory supervision

### What should not be copied blindly

1. Replacing the conditional critic with unconditional energy
   - unconditional alone does not encode query-conditioned relevance

2. Assuming unconditional stability means conditional readiness
   - conditional EBMs have different failure modes
   - see: <https://arxiv.org/abs/2207.05824>

---

## Concrete Metric Upgrade Plan

Current metric set is not enough for staff-engineering-grade iteration.

Add these metrics per epoch:

1. `cos_before`, `cos_after`, `cos_improvement`
2. normalized geodesic distance:
   - `acos(clamp(cos(x, y), -1, 1))`
3. raw L2 only as secondary
4. norm statistics:
   - `||clean||`, `||noisy||`, `||actor||`, `||final||`
5. clean-minimum violation:
   - rate of `E(actor) < E(clean)`
   - rate of `E(final) < E(clean)`
6. gradient alignment:
   - `cos(actor_delta, -gradE(noisy))`
7. actor trust-region:
   - mean `||delta||`
   - fraction of steps over threshold
8. manifold metrics:
   - kNN cosine top1 / topk
   - PRDC-lite on a fixed bank
9. trajectory metrics:
   - energy monotonicity rate
   - cosine monotonicity rate
   - off-plane residual only as plot diagnostic

The key rule:

- never let raw L2 success dominate angular/manifold failure.

---

## Concrete Loss Upgrade Proposal

## Version 1: Minimal, coherent repair

Critic:

`L_critic = L_MDSM + 0.25 * L_rank`

Actor:

`L_actor = 1.0 * L_geo_final`
`        + 0.25 * L_align`
`        + 0.05 * L_step_norm`
`        + 0.10 * L_energy_barrier`

Where:

- `L_geo_final = 1 - cos(v_actor_final, v_clean)`
- `L_align = 1 - cos(delta_actor, -gradE(v_noisy))`
- `L_step_norm = smooth_l1(||delta||, c * sigma * ||v_clean||)`
- `L_energy_barrier = softplus(E(clean) - E(actor_final))`

Note:

- this barrier only prevents sub-clean overshoot;
- it does not force actor below clean.

Inference:

- actor few steps first,
- then short critic Langevin,
- evaluate on `v_last`, not `v_best`, for user-facing metrics.

## Version 2: Better medium-term design

Critic:

- conditional pairwise MDSM + batchwise InfoNCE
- optional unconditional prior head

Actor:

- distill teacher trajectory via consistency or flow matching

This is the first version that starts to look like a real CERBER pre-Stage2 system.

---

## Experiment Matrix

## Phase 0. Debugging ablations

1. Remove `actor_energy_loss` entirely.
2. Keep current ranking critic.
3. Compare:
   - energy gaps
   - cosine
   - geodesic
   - clean-minimum violation

Expected result:

- overshoot below clean should drop.

## Phase 1. Restore score-valid critic

1. Add MDSM back into critic loss.
2. Keep ranking as auxiliary.
3. Train actor with final-state angular objective.

Expected result:

- Langevin becomes more semantically aligned,
- energy descent should stop fighting cosine.

## Phase 2. Distill actor from trajectories

1. Generate teacher trajectories using critic.
2. Train actor on:
   - trajectory next-step prediction,
   - or consistency target,
   - or flow target.

Compare:

- one-step actor,
- 2-step actor,
- actor + 10 critic steps,
- critic-only 30 steps.

Expected result:

- much stronger speed/quality tradeoff,
- better Stage2 transfer.

## Phase 3. Add manifold prior

1. Train / reuse an unconditional prior head.
2. Combine with conditional energy.
3. Re-evaluate PRDC / kNN / C2ST.

Expected result:

- lower OOD drift,
- better manifold coverage,
- less need for hard sphere-only projection.

---

## Staff-Level Conclusions

### What is fundamentally wrong right now

The current actor+critic variant is not failing because the MLP is too small.

It is failing because:

1. the critic is not trained for the gradients used at inference,
2. the actor and critic objectives contradict each other on whether clean is the minimum,
3. the actor is trained on the wrong task for future CERBER deployment,
4. the geometry of the loss is mismatched to the geometry of the latent space.

### What is salvageable

The pairwise critic idea is salvageable.

The actor idea is also salvageable.

But the current implementation should be reframed as:

- a local denoising research scaffold,
- not a valid final Stage1 actor+critic foundation.

### What I would do next

Priority order:

1. Fix the objective contradiction (`actor_energy_loss`).
2. Reintroduce score-valid critic training in `actor_critic` mode.
3. Align all train/eval metrics to reached endpoint semantics.
4. Move actor loss from delta-MSE to final-state angular + critic-alignment terms.
5. Add unconditional prior head as manifold regularizer.
6. Distill actor from critic trajectories via consistency / flow / rectified-flow style training.

This is the most coherent path from current Stage1 to a future stable CERBER actor+critic system.

---

## Short Checklist

- [ ] Remove current `actor_energy_loss` sign conflict
- [ ] Use `v_last` in training-time evaluation, not only `v_final`
- [ ] Make critic in `actor_critic` mode hybrid: `MDSM + ranking`
- [ ] Replace primary actor objective with final-state angular/geodesic loss
- [ ] Add clean-minimum violation metrics
- [ ] Add kNN / PRDC-lite / geodesic metrics to the training loop
- [ ] Prototype `E_total = E_pair + lambda * E_prior`
- [ ] Prototype actor distillation from critic trajectories

## Final One-Line Verdict

Do not scale the current `actor_critic` formulation as-is. Fix objective alignment first; then use the
pairwise critic as a score-valid conditional head and the actor as a trajectory-distilled few-step policy.
