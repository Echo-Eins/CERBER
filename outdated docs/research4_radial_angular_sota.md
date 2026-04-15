# Research 4: SOTA Blueprint for Radial + Angular Twin Critic (CERBER Stage 1.5+)

Date: 2026-03-31
Scope: design a mathematically coherent, implementation-ready Stage 1.5 architecture with explicit radial/angular critic specialization for SONAR hyperspherical embeddings.

---

## 1) Problem Restatement

Current Stage 1.5 uses twin critics, but both critics are homogeneous (`SimpleEnergy` with identical features/objectives). This is an ensemble, not geometric decomposition.

Observed failure pattern from prior runs:
- good 2D trajectory visuals but inconsistent 1024D cosine gains,
- low-noise inference instability,
- local wells and objective interference.

Target architecture must explicitly decompose guidance into:
- angular/tangential refinement (semantic direction on hypersphere),
- radial/normal control (distance to manifold/shell and energy barriers).

---

## 2) Evidence from SOTA Literature (Primary Sources)

### 2.1 Tangent/normal score decomposition is a valid and useful principle
1. SDDM (ICML 2023) defines score decomposition into tangent and normal parts, with manifold-aware optimization stages.
- Source: https://proceedings.mlr.press/v202/sun23n.html
- PDF: https://proceedings.mlr.press/v202/sun23n/sun23n.pdf
- Key point: s(x) = s_r(x) + s_d(x), where tangent component refines on-manifold content and normal component handles off-manifold transitions.

2. Diffusion manifold theory confirms geometry-dependent score structure and decomposition relevance.
- Source: https://arxiv.org/abs/2603.20645

3. Euclidean diffusion on manifold data identifies tangential/normal score singularity behavior and proposes tangential-only training variants.
- Source: https://arxiv.org/abs/2505.09922

4. "Your diffusion model secretly knows the dimension of the data manifold" supports that low-noise score aligns with normal bundle information.
- Source: https://arxiv.org/abs/2212.12611

### 2.2 Hypersphere constraints are not cosmetic; they improve stability
5. NormFace formalizes benefits of hyperspherical embeddings and norm control.
- Source: https://arxiv.org/abs/1704.06369

6. ArcFace shows angular-margin objectives can outperform mixed Euclidean objectives when identity/semantic discrimination is angular.
- Source: https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html
- PDF: https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf

7. Deep Metric Learning with Spherical Embedding shows explicit norm regularization improves angular training consistency.
- Source: https://arxiv.org/abs/2011.02785

8. Hyperspherical latent constraints in modern AR generators show strong stability/quality gains by removing unstable scale component.
- Source: https://arxiv.org/abs/2509.24335

### 2.3 EBM-specific operational constraints
9. "Should EBMs model the energy or the score?" highlights practical tradeoff: unconstrained score often wins empirically, but conservative scalar energy remains useful when downstream requires potential-based navigation.
- Source: https://openreview.net/forum?id=9AS-TF2jRNb

10. Energy Matching provides strong scalar-energy training results but does not remove mixed-derivative requirements when supervising via input gradients.
- Source: https://arxiv.org/abs/2504.10612
- Repo: https://github.com/m1balcerak/EnergyMatching

11. EBM training literature (NeurIPS 2021 bidirectional bounds) reinforces need for bounded/regularized training and OOD controls.
- Source PDF: https://www2.compute.dtu.dk/~sohau/papers/neurips2021/Bounds_all_around__training_energy_based_models_with_bidirectional_bounds.pdf

### 2.4 Twin-critic stabilization principle
12. Twin-critic with clipped aggregation is established for bias/variance reduction in actor-critic systems.
- Source: https://proceedings.mlr.press/v80/fujimoto18a.html
- PDF: https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf

Inference for CERBER: twin setup should be preserved, but critics must be functionally specialized, not duplicated.

---

## 3) Architecture Decision: Radial + Angular Twin Critic

## 3.1 State representation
For vectors q (query) and v (candidate):
- q_hat = q / ||q||
- v_hat = v / ||v||
- cosine c = <q_hat, v_hat>
- geodesic theta = arccos(clamp(c))
- radial delta_r = ||v|| - r_target (or relative shell deviation)
- pair distance d = ||v - q||
- sigma embedding phi(sigma)

## 3.2 Critic heads
1. Angular critic E_ang(q, v, sigma)
- Inputs should prioritize directional geometry:
  [q_hat, v_hat, q_hat - v_hat, q_hat * v_hat, c, theta, phi(sigma)]
- Objective focus: semantic alignment and tangent guidance.

2. Radial critic E_rad(q, v, sigma)
- Inputs should prioritize norm/shell/OOD geometry:
  [||q||, ||v||, delta_r, d, d^2, phi(sigma)] plus optional low-dim pair context.
- Objective focus: shell adherence, support consistency, anti-well shaping.

3. Total energy
E_total = w_ang(sigma) * E_ang + w_rad(sigma) * E_rad + lambda_prior * E_prior(v)

Recommended w schedule:
- high sigma: higher angular weight (global direction),
- low sigma: increase radial/support weight to prevent micro-wells and off-shell drift.

---

## 4) Training Objectives (Specialized)

## 4.1 Angular head losses
1. Tangent-MDSM (core)
- Compute grad wrt v, project to tangent space at v:
  g_tan = g - <g, v_hat> v_hat
- Match tangent target from noisy-clean displacement projected to tangent.

2. Angular ranking
- Enforce E_ang(clean) < E_ang(actor) < E_ang(hard) on normalized embeddings.

3. Geodesic actor supervision
- Actor update should reduce geodesic distance to positive target.

## 4.2 Radial head losses
1. Shell/support penalty
- Penalize | ||v|| - r_target | beyond margin.

2. Energy floor / CD endpoint penalties
- Apply primarily through E_rad branch to raise energies in discovered spurious wells.

3. Clean-minimum radial margin
- Enforce E_rad(clean) <= E_rad(actor) - m_rad (or equivalent barrier).

## 4.3 Cross-head specialization regularizers
To avoid collapse of E_ang and E_rad into same function:
1. Gradient orthogonality regularizer
- Encourage angular gradient to be tangent, radial gradient to align with radial axis.

2. Feature-drop specialization
- During training, randomly mask radial-only features in angular head and angular-only features in radial head.

3. Optional disagreement floor
- Avoid identical outputs by penalizing too-high correlation of head logits over batch.

---

## 5) Inference Dynamics (Critical)

Use projected Langevin/PID with synchronized sigma+noise annealing:
- sigma_t: high -> low
- noise_t: high -> low, synchronized with sigma_t

At each step:
1. compute g_ang = grad_v E_ang, g_rad = grad_v E_rad
2. decompose by projection:
- g_tan = P_t(v) g_ang, P_t(v)=I-v_hat v_hat^T
- g_rad = <g_rad, v_hat> v_hat
3. combined update:
- g = alpha_t * g_tan + beta_t * g_rad
- optional tamed gradient scaling
4. add annealed noise and reproject to target sphere (if fixed-norm policy)

Key point: angular guidance should not be forced to solve radial OOD control; radial head should not dominate semantic direction.

---

## 6) Evaluation Protocol (SOTA-Strict)

Must report both aggregate and per-head metrics.

1. Global metrics
- cosine improvement, geodesic improvement,
- L2 to target,
- energy descent success,
- clean-min violation,
- step norm,
- distribution metrics (MMD, C2ST, PRDC).

2. Head diagnostics
- head-specific ranking success (angular/radial),
- gradient decomposition quality:
  - cos(g_ang, tangent_target),
  - cos(g_rad, radial_axis),
- head correlation (to detect collapse).

3. Noise regime split
- evaluate at low/noisy regimes separately (e.g. 0.0002, 0.01, 0.05, 0.15).

4. Checkpoint policy
- `best.pt`: strict pass only,
- `best_any.pt`: best composite for debugging.

---

## 7) Proposed Implementation Roadmap

## Phase A (minimal high-impact)
1. Introduce `AngularEnergyCritic` and `RadialEnergyCritic` classes.
2. Replace homogeneous twin in Stage1.5 trainer with explicit two-head wiring.
3. Add gradient decomposition diagnostics.
4. Keep existing actor and scheduler infra.

## Phase B (stability)
1. Add specialization regularizers (gradient orthogonality + head correlation guard).
2. Route CD/floor penalties mainly to radial critic.
3. Tune w_ang/w_rad schedule by sigma.

## Phase C (quality)
1. Add retrieval hard-negative curriculum by angular hardness and radial OOD hardness.
2. Add ablation matrix and auto-reporting in live GUI.

---

## 8) Ablation Matrix (required before declaring success)

A1: homogeneous twin vs radial+angular twin
A2: with/without specialization regularizers
A3: with/without sigma-noise synchronized annealing
A4: CD on both heads vs CD on radial only
A5: fixed vs sigma-dependent head weights
A6: actor with/without gradient decomposition alignment term

Success gate for migration to next stage:
- strict pass on all global gates,
- positive mean cosine improvement at low-noise regime,
- reduced clean-min violation,
- no head-collapse (correlation below configured threshold),
- consistent gain on held-out evaluation bank.

---

## 9) Explicit Non-Claims

1. This is not yet implemented in code; this is the SOTA research blueprint.
2. There is no direct benchmark proving this exact architecture on SONAR 1024d text embeddings; domain transfer risk remains.
3. Design is evidence-guided from manifold diffusion/metric learning/EBM training principles plus CERBER failure traces.

---

## 10) Practical Verdict

Given current CERBER behavior, radial+angular specialization is the most justified next architecture step:
- it directly targets the known failure mode (semantic direction vs radial well control conflict),
- keeps scalar-energy inference paradigm intact,
- remains compatible with current actor + Langevin infrastructure.
