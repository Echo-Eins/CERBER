# CERBER Stage1: Compact SOTA Research (A/B/C)

Date: 2026-03-25
Scope: practical guidance for Stage1 GUI tests of energy-landscape and Langevin behavior.

---

## A) How to interpret energy in EBM (minimum energy and regimes)

### Core point
In an EBM, energy `E(x)` defines a ranking over configurations: lower energy means more preferred / higher unnormalized density. Usually `p(x) propto exp(-E(x))`. Absolute energy value is not important; landscape shape and gradients are.

### Reliable sources (3-5)
1. LeCun, Chopra, Hadsell. *A Tutorial on Energy-Based Learning* (2006).  
   https://yann.lecun.org/exdb/publis/pdf/lecun-06.pdf  
   Thesis: inference in EBM is done by minimizing energy over variables; training lowers energy on desired configurations and raises it on undesired ones.

2. Du et al. *How to Train Your Energy-Based Models* (ICLR 2021).  
   https://arxiv.org/abs/2101.03288  
   Thesis: modern EBM can be treated as an unnormalized density model; energy is linked to log-density up to an additive constant.

3. Du, Mordatch. *Implicit Generation and Modeling with Energy-Based Models* (NeurIPS 2019).  
   https://arxiv.org/abs/1903.08689  
   Thesis: sampling is done via gradient dynamics / Langevin on the energy landscape; basin/mode structure matters more than absolute scale.

4. Balcerak et al. *Energy Matching* (NeurIPS 2025).  
   https://arxiv.org/abs/2504.10612  
   Thesis: one scalar potential can exhibit different effective regimes (global transport and near-data Boltzmann behavior).

5. Salimans, Ho. *Should EBMs model the energy or the score?* (ICLR 2021 EBM Workshop).  
   https://openreview.net/forum?id=9AS-TF2jRNb  
   Thesis: compares constrained energy-derived scores vs unconstrained score models; useful for deciding how strict energy constraints should be in practice.

### CERBER Stage1 GUI implications
- Do not hard-assume fixed absolute energy ranges (for example `[-100,100]`).
- Evaluate relative levels, local gradients, and trajectory behavior.
- "Well" vs "hill" visual shape is not by itself a bug; a bug is a mismatch between `-grad(E)` dynamics and plotted trajectory/projection.

---

## B) Why unconditional EBM is not required to recover a specific clean sample from noisy input

### Core point
An unconditional EBM learns a distribution (or score field), not a deterministic map `x_noisy -> x_clean_i`. One noisy point can correspond to many plausible clean points in a multimodal data manifold.

### Reliable sources (3-5)
1. Hyvarinen. *Estimation of Non-Normalized Statistical Models by Score Matching* (JMLR 2005).  
   https://jmlr.csail.mit.edu/papers/v6/hyvarinen05a.html  
   Thesis: the objective is distribution-level (score matching), not pairwise sample reconstruction.

2. Vincent. *A Connection Between Score Matching and Denoising Autoencoders* (Neural Computation 2011).  
   https://direct.mit.edu/neco/article/23/7/1661/7677/A-Connection-Between-Score-Matching-and-Denoising  
   Thesis: denoising-score objectives recover score structure, not necessarily exact identity recovery of a specific clean sample.

3. Song, Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution* (NeurIPS 2019).  
   https://arxiv.org/abs/1907.05600  
   Thesis: sampling via Langevin targets the data distribution, not the original paired clean instance.

4. Song et al. *Score-Based Generative Modeling through SDEs* (ICLR 2021).  
   https://arxiv.org/abs/2011.13456  
   Thesis: reverse-time dynamics recover distributional structure, not guaranteed per-sample inversion.

5. Efron. *Tweedie's Formula and Selection Bias* (JASA 2011).  
   https://pubmed.ncbi.nlm.nih.gov/22505788/  
   Thesis: in noisy settings, posterior-mean style estimators are natural; optimal denoising does not imply exact original-sample recovery.

### CERBER Stage1 GUI implications
- For unconditional mode, "cos(clean, denoised) must increase for this specific pair" is not a valid primary criterion.
- Split tests into:
  - `conditional denoise test` (paired target exists),
  - `unconditional energy-descent test` (model learns `E(x)` over manifold).
- `improvement = 0` on a specific clean pair does not by itself mean unconditional EBM failure.

---

## C) Correct metrics for unconditional energy descent in high-dimensional embeddings

### Core point
Use distribution-level and trajectory-level metrics, not only pairwise reconstruction metrics. In high-dimensional embeddings, single-metric evaluation is unreliable.

### Reliable sources (3-5)
1. Theis et al. *A Note on the Evaluation of Generative Models* (2015).  
   https://arxiv.org/abs/1511.01844  
   Thesis: no single metric is sufficient; generative evaluation must be multi-criteria.

2. Kynkaanniemi et al. *Improved Precision and Recall Metric for Assessing Generative Models* (NeurIPS 2019).  
   https://arxiv.org/abs/1904.06991  
   Thesis: separate fidelity vs coverage; one scalar quality score hides different failure modes.

3. Naeem et al. *Reliable Fidelity and Diversity Metrics for Generative Models* (ICML 2020).  
   https://arxiv.org/abs/2002.09797  
   Thesis: PR metrics can be unstable; PRDC (density/coverage) is more reliable.

4. Gretton et al. *A Kernel Two-Sample Test* (JMLR 2012).  
   https://www.jmlr.org/beta/papers/v13/gretton12a.html  
   Thesis: MMD is a principled two-sample test for high-dimensional distribution comparison.

5. Lopez-Paz, Oquab. *Revisiting Classifier Two-Sample Tests* (2016).  
   https://arxiv.org/abs/1610.06545  
   Thesis: C2ST gives an interpretable, practical test for distribution mismatch in complex high-dimensional data.

### CERBER Stage1 GUI implications
- Minimal valid metric set for unconditional mode:
  1. `energy_monotonicity_ratio` along trajectory: fraction of steps with `E_{t+1} <= E_t`.
  2. `final_energy_percentile` relative to train-energy distribution.
  3. `manifold_proximity`: kNN cosine/distance to train embeddings before vs after inference.
  4. Batch-level tests: `MMD` plus `C2ST` (or `PRDC`) between denoised batch and train batch.
- Keep pairwise cosine-to-clean as a secondary debug indicator, not as the main kill criterion for unconditional EBM.

---

## Practical summary for Stage1 GUI tests

1. Split testing modes: `conditional` and `unconditional`.
2. For unconditional mode, remove the false requirement to return to one specific clean target.
3. Report trajectory/distribution metrics first (energy descent, manifold proximity, two-sample tests).
4. Treat visualization as valid only when points and trajectory are projected with the same grid/tensor convention used to build the surface.

---

## D) Code-grounded math audit (CERBER Stage1 GUI)

### D1. What objective each model really optimizes

1. `SimpleEnergy` (`E(v_query, v_candidate, sigma)`) with Stage1 MDSM:
   - In code, target gradient is
     `target_score = (v_noisy - v_clean) / sigma_eff_sq`
     (`cebcm/training/losses.py`, `multiscale_dsm_loss`).
   - Langevin update in code is `v <- v - lr * grad(E) + noise`
     (`cebcm/inference/langevin.py`).
   - Therefore, when `grad(E) ~ (v_noisy - v_clean)`, descent moves toward `v_clean`.
   - Conclusion: for `simple` model, cosine-to-clean and denoising trajectory to clean are valid primary diagnostics.

2. `UnconditionalEnergy` (`E(x)`) with Energy Matching / unconditional training:
   - Model has no query conditioning, only `forward(x)` (`cebcm/models/energy_unconditional.py`).
   - It defines a scalar field over the manifold; one specific clean vector is not an obligatory attractor.
   - Conclusion: for `unconditional`, primary diagnostic is energy descent and manifold-level metrics; cosine-to-one-clean is secondary only.

### D2. Why the old GUI behavior looked contradictory

1. `best-state` vs `trajectory` mismatch:
   - Core sampler returns best-energy state (`v_best`) while trajectory stores all executed states.
   - GUI previously reported denoised point from returned state and plotted full path, causing apparent contradictions (including `improvement=0` with nontrivial path).

2. Surface/point mismatch:
   - 3D mesh is a finite 2D slice approximation.
   - If marker/trajectory `z` uses exact 1024D energy while surface uses coarse grid values, points can visually float above/below mesh.
   - If trajectory exits scanned range, marker can also appear outside valid surface domain.

3. Unconditional metric mismatch:
   - Old GUI printed cosine-to-clean as if it were a kill metric for unconditional mode.
   - This is mathematically not aligned with unconditional objective.

### D3. Fixes applied in this pass

1. GUI now uses executed final step for visualization/metrics, and reports early-stop/full-step flags.
2. Inference path enforces full-step debug mode for deterministic sensitivity to step/lr controls.
3. Landscape scan now expands range to include noisy/denoised/trajectory projections.
4. Surface/contour marker energies are snapped to plotted mesh interpolation for visual consistency.
5. Checkpoint title display includes best-effort mojibake recovery.
6. Inference report now explicitly separates unconditional semantics:
   - primary: energy descent,
   - secondary: cosine-to-clean as local diagnostic.
7. 2D landscape basis is deterministic now; repeated scans with same inputs stay in the same plane.

### D4. Remaining caveats after fixes

1. This environment does not have `torch`, so runtime CUDA/Gradio validation must be run on your machine.
2. Some legacy comments/strings in old files still contain mojibake but are non-user-facing.
3. For production-quality unconditional evaluation, add batch-level metrics (MMD/C2ST/PRDC), not only single-sample diagnostics.
