# Deep Audit: CURRENT Unconditional Energy Pipeline

Date: 2026-03-25
Author: Codex sub-agent audit
Scope:
- `cebcm/models/energy_unconditional.py`
- `cebcm/training/energy_matching.py`
- `cebcm/training/negative_buffer.py`
- `experiments/02_energy_matching/train.py`
- `experiments/02_energy_matching/evaluate.py`
- `configs/energy_matching.py`
- related unconditional paths in `cerber_gui/app.py`, `cerber_gui/sota_eval.py`, `cebcm/inference/langevin.py`, Stage1 visualization/eval entrypoints

Goal: explain root causes behind the current unconditional logs and behavior:
- slow learning,
- instability / skipped steps,
- energy descent mismatch,
- manifold mismatch,
- contradictions between GUI / eval / training metrics,
- and propose implementation-ready SOTA upgrades.

---

## 1. Executive verdict

### Short version
The current `unconditional` pipeline is **not a faithful realization of full Energy Matching**, is **geometrically inconsistent with the SONAR hyperspherical manifold**, and uses **evaluation criteria that are partially wrong for an unconditional EBM**.

The main failures are not one bug, but a stack of mutually reinforcing mismatches:

1. **Objective mismatch**:
   current code trains mostly an **OT velocity-matching potential**, but does **not** keep an equilibrium / contrastive term active during the main phase, unlike the strongest Energy Matching recipe in the paper. That makes manifold fidelity weak.

2. **Geometry mismatch**:
   training is done on **Euclidean OT chords** between Gaussian prior and data, while inference is run with **sphere projection + tangent projection** in SONAR space. The model is trained in one geometry and sampled in another.

3. **Sampler mismatch**:
   unconditional codebase currently contains **multiple Langevin implementations with different noise semantics** and different update constraints.

4. **Eval mismatch**:
   training selects `best.pt` by **paired denoising cosine improvement to a specific clean sample**, but for an unconditional EBM this is **not the primary objective** and can select the wrong checkpoint.

5. **OOD / manifold control gap**:
   there is no explicit regularization that says "stay on the sentence manifold" beyond hard norm projection at inference time.

6. **Optimization bottleneck**:
   `create_graph=True` second-order training + Björck orthonormalization with `n_iters=15` at every forward is the dominant compute tax.

### Practical consequence
Right now this model can easily look "locally plausible" on a 2D slice and still fail in 1024D:
- `2D slice distance` improves,
- while `cosine` worsens,
- `off-plane residual` grows,
- and distribution metrics remain poor.

That is consistent with the current code. It is not just a GUI illusion.

---

## 2. What the current pipeline actually trains

## 2.1 Model

`UnconditionalEnergy` implements a scalar potential:

`E(x): R^1024 -> R`

Architecture in code:
- `1024 -> 2048 -> 1024 -> 512 -> 1`
- hidden layers optionally `OrthoLinear + GroupSort`
- final scalar layer unconstrained
- global learnable scale `exp(log_energy_scale)`

Files:
- `cebcm/models/energy_unconditional.py`

### Important implication
This model is **unconditional**. It does not see a query, target, sigma, time embedding, or task context. Therefore it can only learn:
- a manifold prior,
- a scalar energy field over latent space,
- and gradients that point toward lower-energy regions.

It **cannot** by itself represent "the correct answer to this specific prompt/query/state". That future role requires a conditional critic later.

---

## 2.2 Training modes that exist in code

### A. `energy_matching`
Loss in code:
- sample `x0 ~ N(0, sigma_prior^2 I)`
- sample `t ~ U[t_min, t_max]`
- set `x_t = (1-t)x0 + t x_data`
- target velocity `u_t = x_data - x0`
- optimize `|| -grad E(x_t) - u_t ||^2`

File:
- `cebcm/training/energy_matching.py::energy_matching_loss`

### B. `cosine_em`
Same path, but optimize direction:
- `1 - cos(-grad E, u_t)`
- plus weak magnitude term

File:
- `cebcm/training/energy_matching.py::energy_matching_cosine`

### C. `weighted_em`
Same OT path, but reweights near-data region by `w(t)`.

File:
- `cebcm/training/energy_matching.py::energy_matching_weighted`

### D. `nce_warmstart_em`
There is a short NCE pre-phase using a replay buffer, then training switches to EM.

Files:
- `experiments/02_energy_matching/train.py`
- `cebcm/training/negative_buffer.py`

---

## 3. First root cause: this is not full Energy Matching as reported in the paper

The Energy Matching paper reports strongest results for a combined objective with an equilibrium / contrastive term, not pure OT-only matching. See the paper and repository for the two-part training idea (`OT` term plus contrastive / CD term) [Balcerak et al., 2025](https://arxiv.org/abs/2504.10612).

### Current repo deviation
In this repo:
- main phase is either `energy_matching`, `cosine_em`, or `weighted_em`
- the NCE / negative-buffer term is used only as a **warmstart**
- after warmstart, the main phase drops the contrastive equilibrium pressure

That means the current model is optimized mostly to match a transport field, but not strongly enough to establish the correct stationary energy geometry around the real data manifold.

### Why this matters
OT-style velocity matching is good for learning a direction field, but unconditional EBM quality depends on more than direction:
- you also need **correct relative density geometry**,
- especially mode weights and equilibrium structure.

This gap is exactly why pure Fisher / score-style objectives can get local gradients right while still missing global mode proportions; this issue is part of the broader energy-vs-score discussion [Hyvärinen, 2005](https://jmlr.csail.mit.edu/papers/v6/hyvarinen05a.html), [Salimans & Ho, 2021](https://openreview.net/forum?id=9AS-TF2jRNb).

### Concrete recommendation
Do **not** keep the current regime as `warmstart -> pure EM`.

Instead implement:

`L_total = L_OT + lambda_eq * L_eq`

where `L_eq` is one of:
- contrastive divergence / replay-buffer contrastive term,
- principled NCE with explicit `log p_noise`,
- or a short-run negative phase maintained throughout training.

Recommended schedule:
- epochs 0-5: `lambda_eq = 1.0`, heavy equilibrium shaping
- epochs 5-20: `lambda_eq = 0.3`
- later: `lambda_eq = 0.1` or adaptive based on manifold metrics

This is much closer to the actual EM spirit than the current hard handoff.

---

## 4. Second root cause: geometry mismatch between training and inference

This is one of the most important findings.

## 4.1 What training assumes
Training loss uses:
- Gaussian prior in `R^1024`
- straight Euclidean interpolation `x_t = (1-t)x0 + t x_data`
- target velocity `u_t = x_data - x0`

Files:
- `cebcm/training/energy_matching.py`

## 4.2 What inference assumes
Inference and GUI unconditional refinement use:
- `target_norm` sphere projection,
- tangent projection of updates,
- Stage1 Langevin machinery via `_UnconditionalEnergyAdapter`

Files:
- `cerber_gui/app.py::run_langevin_denoise`
- `cebcm/inference/langevin.py`

## 4.3 Why this is mathematically inconsistent
SONAR embeddings are concentrated near a shell with target norm around `0.2051`. The current inference pipeline therefore treats the latent geometry as approximately hyperspherical.

But the EM training path is still Euclidean:
- prior is sampled in ambient Euclidean space,
- interpolation follows straight lines through the ambient space,
- loss target contains radial components,
- while inference later projects motion back to the sphere.

So the model is trained to follow one field, but sampled under another constraint set.

This directly explains the pattern seen in logs:
- 2D projected distance to the clean reference improves,
- but 1024D cosine degrades,
- and off-plane residual grows.

The model is allowed during training to use radial shortcuts that the deployed sampler later removes or distorts.

### Concrete SOTA-grade fix
Move to a **sphere-aware EM variant** for SONAR.

#### Recommended implementation
1. Replace Gaussian `x0` with sphere prior:
   - sample `z ~ N(0, I)`
   - set `x0 = target_norm * normalize(z)`
   - optional better version: vMF prior on the sphere

2. Replace linear interpolation with **SLERP / geodesic interpolation**:
   - `x_t = slerp(x0, x1, t)` on the sphere

3. Replace full velocity target with **tangent velocity target**:
   - project target onto tangent space at `x_t`
   - train against tangent component of `-grad E(x_t)`

4. Keep a separate radial penalty if needed:
   - `L_radial = ((||x|| - r0)^2)` or barrier around shell

This makes training geometry consistent with deployed inference.

Related sphere representation intuition: hyperspherical alignment/uniformity is a meaningful inductive bias when embeddings live on a normalized manifold [Wang & Isola, 2020](https://arxiv.org/abs/2005.10242).

---

## 5. Third root cause: sampler mismatch inside the repo itself

There is not one unconditional sampler in the codebase; there are several, and they do not implement the same SDE discretization.

## 5.1 Mismatch A: Langevin noise semantics are inconsistent

### In `cebcm/training/energy_matching.py::generate_samples_langevin`
Update is:

`x <- x - lr * grad + sqrt(2 * lr) * (noise_scale * eps)`

So the effective noise std is:

`noise_std = sqrt(2 * lr) * noise_scale`

### In `cebcm/inference/langevin.py`
Update is:

`x <- x - lr * grad + sqrt(2 * lr * noise_scale) * eps`

So the effective noise std is:

`noise_std = sqrt(2 * lr * noise_scale)`

### Consequence
If `noise_scale = 0.005`, these differ by a factor of roughly:

`sqrt(0.005) / 0.005 ~= 14.14`

That is not a small tuning difference. It is a different stochastic process.

This is a direct candidate root cause for:
- energy descent mismatch,
- manifold mismatch,
- different behavior between training-time eval and GUI inference,
- and contradictory conclusions from otherwise "same" hyperparameters.

### Required fix
Define one canonical parameterization and use it everywhere.

Recommended standard:
- expose `diffusion_coeff` or `noise_std`
- not ambiguous `noise_scale`

For example:

`x_{k+1} = x_k - eta * gradE + sqrt(2 * eta) * sigma * eps`

Then enforce the same semantics in:
- `generate_samples_langevin`
- `NegativeBuffer.refresh`
- `UnconditionalEnergy.sample_langevin`
- `cebcm/inference/langevin.py`
- GUI runtime inference
- all eval scripts

---

## 5.2 Mismatch B: tangent projection exists in one path, not in another

`run_langevin()` in Stage1 inference:
- projects updates to tangent space before stepping on the sphere

But training/eval unconditional generation functions do not use the same tangent update rule consistently.

That means even after fixing noise semantics, the sampler family is still inconsistent.

### Required fix
Refactor all unconditional sampling to one shared implementation:

`cebcm/inference/unconditional_sampler.py`

with explicit switches:
- `project_to_sphere: bool`
- `tangent_project: bool`
- `noise_std`
- `track_vectors`
- `return_best` vs `return_last`

Then make:
- GUI,
- training evaluation,
- negative buffer refresh,
- standalone eval,
- and model helper methods
all call that same sampler.

---

## 6. Fourth root cause: evaluation is partially wrong for unconditional EBM

## 6.1 The most serious contradiction
`best.pt` is currently selected by average paired denoising improvement:

- in `train.py`, `avg_improvement` is mean cosine gain over noisy/clean pairs
- that metric decides the best checkpoint

This is wrong for unconditional mode.

### Why wrong
An unconditional EBM is not trained to map `x_noisy -> x_clean_i` for the original paired sample. It is trained to model the data distribution / scalar energy landscape. Exact recovery of the one original clean point is a **conditional denoising metric**, not the primary unconditional objective [Vincent, 2011](https://direct.mit.edu/neco/article/23/7/1661/7677/A-Connection-Between-Score-Matching-and-Denoising), [Song & Ermon, 2019](https://arxiv.org/abs/1907.05600).

### Resulting failure mode
A checkpoint can be:
- better as an unconditional energy model,
- but worse at paired cosine recovery,
- and the current training loop will reject it.

This can fully invert model selection.

## 6.2 Standalone eval repeats the same mistake
`experiments/02_energy_matching/evaluate.py` still treats deterministic gradient following toward paired clean vectors as a headline denoising metric.

That is useful as a debug probe, but not as the main criterion.

## 6.3 GUI is improved but still mixed
The GUI now explicitly labels cosine-to-clean as secondary for unconditional mode, which is correct. But the underlying SOTA batch eval still includes paired denoising-style metrics that can dominate human interpretation if not clearly separated.

### Required fix: split eval into two regimes

#### Regime A: unconditional manifold / sampling evaluation (primary)
Primary metrics should be:
- mean `delta_energy = E(x0) - E(xT)` on held-out noisy starts
- energy monotonicity ratio along trajectory
- final-energy percentile relative to train/test data energy histogram
- PRDC / MMD / C2ST between refined/generated samples and held-out data
- kNN proximity to data bank
- radial deviation / shell deviation
- OOD separation (see section 7)

#### Regime B: paired local denoise diagnostic (secondary)
Keep, but label as secondary:
- cosine to original clean
- L2 to original clean
- 2D projection diagnostics

### Required checkpoint selection rule
Use a composite score for unconditional best-checkpoint selection, e.g.:

`score = 0.35 * energy_success + 0.20 * knn_cos_improvement - 0.20 * C2ST_sep - 0.15 * MMD - 0.10 * shell_error`

Or simpler at first:
- primary = energy success + PRDC coverage + MMD / C2ST
- secondary = cosine gain debug only

Do not use average paired cosine gain as the best-checkpoint criterion.

---

## 7. Fifth root cause: OOD / manifold control is too weak

Right now the unconditional model is mostly expected to learn the manifold implicitly, but the code gives it very little direct help.

## 7.1 Current OOD controls
Current controls are weak and mostly indirect:
- replay-buffer negatives during NCE warmstart
- norm projection during sampling
- orthonorm / GroupSort smoothness constraints

What is missing:
- explicit off-manifold penalties,
- radial barriers,
- hard OOD negatives,
- calibration metrics for data vs OOD energy.

## 7.2 Why this matters
A future CERBER critic must not only say "lower energy somewhere".
It must also say:
- these points are in-manifold,
- these are OOD,
- these are semantically plausible,
- and these are invalid despite maybe having low local energy.

Without explicit OOD shaping, the model can create attractive low-energy false basins.
That is exactly the kind of behavior that makes batch energy metrics look decent while PRDC / C2ST stay bad.

### SOTA-grade fixes

#### A. Radial shell regularization
Add explicit shell prior:

`L_shell = alpha * (||x_data|| - r0)^2 + beta * relu(m - | ||x_ood|| - r0 | )`

Practical simpler version:
- model energy becomes
  `E_total(x) = E_dir(normalize(x)) + lambda_r * (||x|| - r0)^2`

This is a very strong fit for SONAR.

#### B. OOD margin loss
Sample OOD negatives from multiple sources:
- larger-norm / smaller-norm radial corruptions
- far-bank shuffled vectors
- random sphere points far from nearest neighbors
- replay-buffer negatives

Train:

`L_ood = mean(relu(margin + E_data - E_ood))`

This gives the energy a real barrier outside the sentence manifold.

#### C. Persistent negative mixture
Do not rely on one buffer type only.
Use a mixture batch:
- 40% replay buffer
- 30% random sphere prior
- 20% radial shell violations
- 10% hard negatives chosen by low current energy

This is closer to robust modern EBM practice than one stale buffer source [Du & Mordatch, 2019](https://arxiv.org/abs/1903.08689), [Du et al., 2021](https://arxiv.org/abs/2101.03288).

---

## 8. Sixth root cause: optimization is expensive and some safeguards are mathematically blunt

## 8.1 Main compute bottlenecks

### A. Second-order graph is unavoidable here
EM loss differentiates through `grad E(x_t)`, so `create_graph=True` is necessary.
That is the dominant unavoidable cost.

### B. Björck orthonormalization inside each forward
With `ortho_n_iters=15`, wide hidden layers, and second-order training, this is a huge multiplier.

The GroupSort + norm-constrained family is theoretically well-motivated for Lipschitz function approximation [Anil et al., 2019](https://proceedings.mlr.press/v97/anil19a.html), but the current cost is very high.

### C. No systems acceleration stack
Current unconditional train script has no:
- AMP / bf16
- `torch.compile`
- gradient checkpointing
- EMA model for stable eval
- per-parameter optimization policy

## 8.2 Current stability safeguards are too blunt
The training loop currently:
- `nan_to_num`s every gradient,
- clips all grads to `1.0`,
- skips if loss > `100.0`.

These are understandable emergency guards, but they also hide failure modes.

### Problems
1. `nan_to_num` can silently convert real explosions into fake zero gradients.
2. Global clip `1.0` may undertrain the useful parts while one bad parameter dominates the threshold.
3. Hard `loss > 100` skip can create biased training if spikes are systematic in certain `t` regions.

### Better implementation
- Track per-layer grad norms instead of silently flattening them.
- Use adaptive clipping or percentile-based clipping.
- Separate optimizer groups:
  - no weight decay for `log_energy_scale`
  - lower LR for the final scalar layer and scale parameter
- Keep an EMA copy for evaluation.
- Log bad-batch signatures (`t` bin, energy scale, grad norm, input norm) before skipping.

---

## 9. Seventh root cause: the architecture is too generic for SONAR geometry

The current network is a generic MLP energy over `x`.
For SONAR this is leaving performance on the table.

## 9.1 Missing structural bias
SONAR embeddings are not arbitrary `R^1024` vectors. In your own pipeline they are repeatedly treated as living near a fixed-radius shell.

But the model sees only the raw vector.

### Better architecture for this exact problem
Use explicit directional + radial decomposition:

- `x_dir = normalize(x)`
- `r = ||x||`
- features: `[x_dir, (r-r0), (r-r0)^2]`
- energy:
  `E(x) = E_dir(x_dir) + lambda1 * (r-r0) + lambda2 * (r-r0)^2`

Benefits:
- cleaner shell control,
- more interpretable OOD barrier,
- reduced need for the model to reinvent radial geometry,
- more stable sampling under sphere projection.

## 9.2 Residual parameterization
Also consider replacing plain sequential MLP with residual blocks:
- pre-norm residual MLP,
- optional spectral norm instead of full Björck in early ablations,
- controlled skip scaling.

This usually improves optimization depth-wise without needing to widen blindly.

---

## 10. Missing metrics: what should be tracked but currently is not

Current unconditional pipeline still misses several metrics that are essential for debugging an energy model.

## 10.1 Training metrics missing
Track these every epoch or every N steps:

1. **Per-t-bin alignment**
   - split `t` into bins, log:
   - `cos(-gradE, u_t)`
   - `||gradE||`
   - MSE magnitude error

2. **Energy histograms**
   - `E(data)`
   - `E(prior)`
   - `E(buffer)`
   - `E(refined)`
   - `E(ood_radial)`

3. **Shell deviation**
   - `mean | ||x|| - r0 |` on refined and generated samples

4. **Best-vs-last sampler gap**
   - `E(last) - E(best)`
   - `||x_last - x_best||`

5. **Replay buffer health**
   - age distribution
   - fraction reinitialized
   - nearest-neighbor novelty to train bank
   - low-energy occupancy fraction

6. **Scale / norm monitoring**
   - `exp(log_energy_scale)`
   - final layer norm
   - per-layer grad norm
   - spectral / orthogonality diagnostics

## 10.2 Eval metrics missing or underused
Primary unconditional eval should include:
- MMD [Gretton et al., 2012](https://www.jmlr.org/beta/papers/v13/gretton12a.html)
- C2ST [Lopez-Paz & Oquab, 2016](https://arxiv.org/abs/1610.06545)
- PRDC [Kynkäänniemi et al., 2019](https://arxiv.org/abs/1904.06991), [Naeem et al., 2020](https://arxiv.org/abs/2002.09797)
- kNN manifold proximity
- energy descent success
- shell error
- OOD AUROC / AUPRC for `E(data)` vs `E(ood)`

These should be emitted into checkpoint JSON and `training_metrics.json`, not live only in GUI.

---

## 11. Concrete implementation plan: unconditional pipeline v2

## P0. Correctness and consistency fixes (must do first)

1. **Unify all unconditional samplers**
   - single implementation
   - single noise convention
   - single tangent/sphere logic
   - single endpoint semantics (`last`, `best` both returned)

2. **Stop selecting `best.pt` by paired cosine gain**
   - replace with unconditional composite metric

3. **Split eval reports into primary vs secondary**
   - primary: manifold / energy / distribution
   - secondary: paired denoise debug

4. **Move GUI and eval scripts to the same sampler backend**

## P1. Geometry-aware training (highest expected quality gain)

1. Replace Euclidean prior with spherical prior
2. Replace linear interpolation with SLERP / geodesic path
3. Match tangent velocity only
4. Add shell penalty / radial barrier

This is the single highest-leverage model-quality fix for SONAR geometry.

## P2. Restore equilibrium shaping during EM

Implement continuous joint training:

`L = L_sphere_OT + lambda_eq * L_eq + lambda_ood * L_ood + lambda_shell * L_shell`

Suggested starting coefficients:
- `lambda_eq = 0.25`
- `lambda_ood = 0.10`
- `lambda_shell = 1.0`

Ablate from there.

## P3. Systems + optimization

1. AMP/bf16
2. `torch.compile`
3. gradient checkpointing for the energy net
4. ortho ablation:
   - `orthonorm 15`
   - `orthonorm schedule 15 -> 8 -> 4`
   - `spectral_norm`
5. EMA model for evaluation
6. separate LR groups:
   - base net
   - final layer
   - `log_energy_scale`

## P4. Better negative phase

Replace current one-source replay buffer warmstart with sustained mixed negatives:
- replay buffer
- random sphere prior
- radial OOD corruptions
- hard negatives by current low energy

And keep the negative phase active throughout training.

---

## 12. Ablation matrix (implementation-ready)

| ID | Change | Why | Primary metrics | Pass criterion |
|---|---|---|---|---|
| U1 | Fix sampler noise semantics everywhere | remove hidden train/eval mismatch | mean dE, trajectory monotonicity, MMD/C2ST | metrics stable across GUI vs eval |
| U2 | Replace best-checkpoint metric | stop selecting wrong models | PRDC, C2ST, dE success | new best differs from old best on same run |
| U3 | Sphere prior + SLERP + tangent target | geometry alignment | shell error, PRDC, kNN, off-plane residual | shell error down, PRDC up |
| U4 | Joint EM + equilibrium loss | recover manifold density | PRDC, C2ST, energy histograms | C2ST closer to 0.5, PRDC precision/coverage up |
| U5 | OOD margin loss | prevent false low-energy basins | OOD AUROC, energy gap data-vs-OOD | AUROC > 0.8 |
| U6 | Orthonorm schedule / spectral ablation | speed vs quality | step_time, VRAM, PRDC, dE success | >=1.8x speedup with no PRDC collapse |
| U7 | Directional-radial energy decomposition | better shell inductive bias | shell error, PRDC, kNN | shell error materially lower |
| U8 | EMA eval | stabilize checkpoint selection | variance across evals | lower metric jitter |

---

## 13. What the current unconditional model is useful for, and what it is not

### It can become useful as:
- an unconditional manifold prior critic,
- a latent-space plausibility / OOD detector,
- a local refinement energy for generic sentence-like vectors.

### It is not enough for future CERBER inference as the only critic
Because it is unconditional, it cannot answer:
- "is this latent good **for this query/context/task**?"

So even a fixed unconditional pipeline will still only cover the **prior critic** role.
A future full CERBER system will need at least:
- `E_prior(x)` or equivalent manifold prior,
- plus a **conditional** critic / actor-conditioned energy later.

That is not a weakness of this implementation specifically; it is a limit of unconditional modeling itself.

---

## 14. Priority order

### Immediate
1. unify sampler semantics
2. replace wrong checkpoint-selection metric
3. move eval to unconditional-first metric suite

### Next
4. sphere-aware EM path
5. sustained equilibrium / OOD terms during main training
6. orthonorm speed ablation + AMP/compile/EMA

### After that
7. structural rewrite to radial+directional energy decomposition
8. promote unconditional model into `prior critic` role and stop overloading it as a conditional denoiser

---

## 15. External sources

### Primary papers / docs
1. Balcerak et al. **Energy Matching** (NeurIPS 2025).  
   https://arxiv.org/abs/2504.10612

2. Salimans, Ho. **Should EBMs Model the Energy or the Score?**  
   https://openreview.net/forum?id=9AS-TF2jRNb

3. Du, Mordatch. **Implicit Generation and Modeling with Energy-Based Models** (NeurIPS 2019).  
   https://arxiv.org/abs/1903.08689

4. Du et al. **How to Train Your Energy-Based Models** (ICLR 2021).  
   https://arxiv.org/abs/2101.03288

5. Hyvarinen. **Estimation of Non-Normalized Statistical Models by Score Matching** (JMLR 2005).  
   https://jmlr.csail.mit.edu/papers/v6/hyvarinen05a.html

6. Vincent. **A Connection Between Score Matching and Denoising Autoencoders** (2011).  
   https://direct.mit.edu/neco/article/23/7/1661/7677/A-Connection-Between-Score-Matching-and-Denoising

7. Song, Ermon. **Generative Modeling by Estimating Gradients of the Data Distribution** (NeurIPS 2019).  
   https://arxiv.org/abs/1907.05600

8. Anil et al. **Sorting Out Lipschitz Function Approximation** (ICML 2019).  
   https://proceedings.mlr.press/v97/anil19a.html

9. Wang, Isola. **Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere** (ICML 2020).  
   https://arxiv.org/abs/2005.10242

10. Gretton et al. **A Kernel Two-Sample Test** (JMLR 2012).  
    https://www.jmlr.org/beta/papers/v13/gretton12a.html

11. Lopez-Paz, Oquab. **Revisiting Classifier Two-Sample Tests** (2016).  
    https://arxiv.org/abs/1610.06545

12. Kynkäänniemi et al. **Improved Precision and Recall Metric for Assessing Generative Models** (NeurIPS 2019).  
    https://arxiv.org/abs/1904.06991

13. Naeem et al. **Reliable Fidelity and Diversity Metrics for Generative Models** (ICML 2020).  
    https://arxiv.org/abs/2002.09797

---

## 16. Final conclusion

If the goal is to salvage the unconditional pipeline rather than discard it, the correct move is **not** to do another blind hyperparameter sweep.

The correct move is:
1. fix sampler/eval correctness,
2. align training geometry with SONAR's sphere-like manifold,
3. keep an equilibrium / contrastive term alive during EM,
4. add explicit OOD / shell control,
5. only then re-run serious ablations.

In its current state, the unconditional pipeline is best understood as:
- a partially correct transport-field learner,
- but not yet a calibrated manifold EBM.
