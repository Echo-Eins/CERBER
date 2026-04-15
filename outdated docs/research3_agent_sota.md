# External SOTA Sweep for CERBER Stage1/2

Date: 2026-03-25
Author: Codex subagent (`research3_agent_sota`)
Scope: external primary-source research for energy-based modeling and actor-critic-like latent refinement, mapped onto current CERBER Stage1/2 code and failure modes.

## 1. Executive Verdict

The external literature points to a clear conclusion for CERBER:

1. Pure standalone `unconditional E(x)` on raw 1024d SONAR embeddings is the least justified near-term path.
2. A pairwise `critic` tied to a proposal/initializer (`actor`, `initializer`, `backbone`, `latent policy`) is much closer to what modern successful EBM-adjacent systems actually do.
3. The best modern EBM results do not come from "just train a scalar energy and Langevin on raw high-d data". They come from one of three patterns:
   - multiscale/noise-ladder objectives with auxiliary initializers or cooperative models,
   - latent/backbone/correction-model EBMs, where the energy corrects a simpler generator,
   - hybrid systems where the critic is used primarily for reranking, refinement, or constrained guidance rather than as the only generative mechanism.
4. For CERBER specifically, the most defensible SOTA path is not `simple OR unconditional`, but a hybrid:
   - pairwise critic remains central,
   - actor/initializer becomes explicit and support-constrained,
   - unconditional/global energy becomes a secondary regularizer or prior, not the sole Stage1 engine.

## 2. Primary Sources

Core EBM objective papers:

- [Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](https://arxiv.org/abs/2504.10612)
- [Energy Discrepancies: A Score-Independent Loss for Energy-Based Models](https://arxiv.org/abs/2307.06431)
- [Learning Energy-Based Models by Diffusion Recovery Likelihood](https://arxiv.org/abs/2012.08125)
- [Learning Energy-Based Models by Cooperative Diffusion Recovery Likelihood](https://arxiv.org/abs/2309.05153)
- [Improved Contrastive Divergence Training of Energy Based Models](https://arxiv.org/abs/2012.01316)

Energy vs score / latent-space tradeoffs:

- [Should EBMs model the energy or the score?](https://openreview.net/forum?id=9AS-TF2jRNb)
- [Learning Energy-Based Models in High-Dimensional Spaces with Multiscale Denoising-Score Matching](https://www.mdpi.com/1099-4300/25/10/1367)
- [Score-based Generative Modeling in Latent Space](https://arxiv.org/abs/2106.05931)
- [Riemannian Score-Based Generative Modelling](https://arxiv.org/abs/2202.02763)

Latent refinement / correction-model / guidance papers:

- [MCMC Should Mix: Learning Energy-Based Model with Neural Transport Latent Space MCMC](https://arxiv.org/abs/2006.06897)
- [Your GAN is Secretly an Energy-based Model and You Should use Discriminator Driven Latent Sampling](https://arxiv.org/abs/2003.06060)
- [Refining Deep Generative Models via Discriminator Gradient Flow](https://arxiv.org/abs/2012.00780)
- [Latent Space Energy-Based Model of Symbol-Vector Coupling for Text Generation and Classification](https://arxiv.org/abs/2108.11556)

OOD/support-constrained policy analogies that map well to actor design:

- [PLAS: Latent Action Space for Offline Reinforcement Learning](https://arxiv.org/abs/2011.07213)
- [Policy Regularization with Dataset Constraint for Offline Reinforcement Learning](https://arxiv.org/abs/2306.06569)

Evaluation / diagnostics:

- [Reliable Fidelity and Diversity Metrics for Generative Models](https://arxiv.org/abs/2002.09797)
- [Official PRDC code](https://github.com/clovaai/generative-evaluation-prdc)
- [Two-sample Testing Using Deep Learning](https://proceedings.mlr.press/v108/kirchler20a.html)
- [Learning Deep Kernels for Non-Parametric Two-Sample Tests](https://proceedings.mlr.press/v119/liu20m.html)

## 3. What the Literature Says About Objectives Beyond Plain DSM / EM

### 3.1 Energy Matching is real progress, but it does not solve CERBER's main bottleneck

What it gives:

- Strong image-generation results among EBMs.
- Single scalar potential field.
- Better integration with inverse-problem priors.

What it does not give:

- It does not remove second-order autograd when training `E(x)` through `?E(x)` matching.
- It does not solve high-dimensional manifold mismatch by itself.
- It is not validated on 1024d sentence embeddings on a sphere.

Direct implication for CERBER:

- EM is a valid research branch, but not the fastest route to a stable Stage1/2 system.
- If adopted, it should be coupled with stronger support control and likely an initializer/backbone, not used as a naked raw-space energy learner.

### 3.2 Energy Discrepancy is theoretically attractive but warns against raw high-dimensional manifold training

`Energy Discrepancy (ED)` is attractive because it avoids score computation and expensive MCMC in the objective. But the paper explicitly notes that in high-dimensional image settings the manifold hypothesis limits raw ED, and they recover usefulness by training the EBM as a prior over a variational decoder latent.

Direct implication for CERBER:

- ED is not an argument for training `E(x)` directly on raw SONAR embeddings with no learned manifold model.
- It is an argument for: if you want a score-independent EBM objective, first give it a better latent manifold than raw Stage1 SONAR.

### 3.3 DRL / CDRL are more relevant to CERBER than plain EM if refinement is central

`Diffusion Recovery Likelihood (DRL)` and `Cooperative DRL (CDRL)` are important because they operationalize a pattern very close to CERBER's needs:

- multiscale noisy distributions,
- a sequence of easier conditional denoising/refinement tasks,
- an initializer model paired with the energy model,
- short-run refinement from meaningful initial states instead of blind MCMC from arbitrary points.

This is much closer to `actor + critic + refinement` than pure EM.

Direct implication for CERBER:

- The strongest external support is for building a cooperative system where:
  - actor/initializer proposes in-support latent states,
  - critic/energy corrects and reranks,
  - training sees short-run refined endpoints, not only analytic local targets.

### 3.4 Contrastive / NCE-style terms remain necessary when mode proportions matter

The literature repeatedly shows that score-only objectives can be locally correct while globally wrong on mass allocation. NCE-like objectives remain useful because they directly train density-ratio discrimination.

Direct implication for CERBER:

- Pure DSM-like training is insufficient if you care about semantic frequency, support, and staying on the true latent manifold.
- CERBER should include a contrastive / replay-buffer / hard-negative term even if a score-like auxiliary remains.

## 4. Score vs Energy in High-Dimensional Embeddings

### 4.1 Score models are easier; energy models are more structured

`Should EBMs model the energy or the score?` gives the cleanest framing:

- unconstrained score models are easier to optimize,
- energy models impose conservative-field structure,
- with comparable architecture, energy can match score quality, but there is no free lunch in optimization cost.

For CERBER this means:

- if you insist on a scalar energy because future inference/planning depends on it, you should expect training difficulty and pay for it with extra structure, better initialization, and stronger regularization.
- if the immediate goal is only denoising/refinement quality, score-like auxiliaries are still the practical baseline.

### 4.2 High-dimensional raw spaces are hostile to direct EBM training

Several sources converge on the same point:

- raw high-dimensional spaces are hard for MCMC and hard for energy estimation,
- multiscale objectives help but do not remove the issue,
- latent-space score/energy models are easier to train and mix.

Direct implication for CERBER:

- Raw `1024d SONAR` is already a latent space, but not necessarily the right one for energy navigation.
- Stage1 may be missing an intermediate learned support model that reshapes the latent geometry for refinement.

### 4.3 The key missed idea: correction model, not sole generator

The strongest practical theme across modern work is:

- train a backbone/initializer/generator/policy,
- train energy as a correction, reranker, or local guide.

Examples:

- neural transport latent-space MCMC,
- DDLS,
- DGflow,
- latent action constraints in offline RL.

Direct implication for CERBER:

- Treat the critic as a local optimizer and quality estimator around actor proposals.
- Stop expecting the critic alone to globally generate or globally denoise from arbitrary points in a stable way.

## 5. Manifold and OOD-Safe Latent Dynamics

### 5.1 Sphere projection is necessary but weak

Current CERBER already uses sphere/tangent projection. The literature says that basic projection helps, but it is only a coarse geometric prior. It does not guarantee that steps remain on the true data manifold.

The real issue is support, not just norm.

### 5.2 Better support control comes from one of three mechanisms

1. Learn a support manifold explicitly.
   - VAE/flow/decoder-backed latent model.
   - Energy acts as prior or correction.

2. Constrain refinement to the actor/generator latent support.
   - This is the lesson from PLAS and related offline RL: move in a latent action/support space, not arbitrary ambient space.

3. Add dataset-constraint regularization.
   - PRDC-style nearest-neighbor support analogies from offline RL are directly relevant.
   - Penalize refined states that move away from local dataset neighborhoods.

Direct implication for CERBER:

- Norm projection is not enough.
- You likely need one or both of:
  - a learned projector / support model,
  - a nearest-neighbor or retrieval-based support penalty during actor and critic training.

### 5.3 If the latent really lives on a manifold, the dynamics should know it

`Riemannian Score-Based Generative Modelling` and manifold MCMC literature suggest that if geometry matters, Euclidean Langevin is only an approximation.

For CERBER this does not mean implementing full Riemannian MCMC tomorrow. It does mean:

- project both states and update directions consistently,
- consider manifold-aware noise scaling,
- use local metric or support-aware preconditioning instead of a single global `lr`.

Practical consequence:

- future CERBER Langevin should likely use adaptive preconditioning or trust-region control based on local support / uncertainty, not only PID coefficients.

## 6. Practical Diagnostics the Repo Should Add

The current metric stack is a good start, but the literature suggests more targeted diagnostics.

### 6.1 Keep existing metrics

Keep:

- PRDC,
- MMD,
- C2ST,
- cosine/L2 before-after,
- energy success rate.

These are useful.

### 6.2 Add the missing diagnostics that matter for CERBER

1. Support/OOD rate.
   - kNN distance to train bank.
   - fraction of refined samples outside local support percentile.

2. Actor-to-critic trust-region stats.
   - mean `||x_T - x_0||`, but also percentile tails.
   - rate of steps exceeding local support radius.

3. Energy-calibration diagnostics.
   - energy rank correlation with retrieval quality or semantic relevance.
   - monotonicity: does lower energy correspond to better downstream retrieval/decoding?

4. Long-run vs short-run divergence.
   - compare 5, 20, 50, 200-step refinement quality.
   - if short-run helps and long-run hurts, critic is local-only and must be used as such.

5. Off-plane / off-support diagnostics.
   - current off-plane metric is useful.
   - extend it with off-neighborhood support residual in 1024D.

6. Multi-start consistency.
   - same clean target, multiple noisy initializations.
   - does refinement converge to similar in-support solutions or collapse to one attractor?

7. Retrieval-grounded evaluation.
   - top-k nearest-neighbor semantic consistency before/after.
   - if L2 improves but nearest-neighbor semantics worsen, the model is overfitting geometry not meaning.

### 6.3 Why this matters

Your current failure mode is already exactly what the literature warns about:

- local geometric improvement,
- poor manifold coverage,
- visually plausible slices,
- globally wrong support.

## 7. Architecture and Training Tricks Most Relevant to CERBER

### 7.1 Replace "critic alone denoises" with "actor proposes, critic locally improves"

This is the single most important architectural lesson.

Recommended split:

- Actor / initializer predicts `x_init` or delta from context.
- Critic provides:
  - local energy guidance,
  - reranking,
  - rejection / acceptance / trust score,
  - optional short-run refinement.

Do not ask the critic to do all of:

- density estimation,
- support modeling,
- denoising,
- generation,
- global search.

### 7.2 Train the actor under explicit support constraints

Borrow from offline RL support-constrained policy learning:

- actor loss should include nearest-dataset or local-manifold constraint,
- critic-guided actor updates should be clipped by support radius,
- actor should output in tangent space or in a learned low-rank / latent subspace, not arbitrary full-space deltas.

### 7.3 Use the critic as a correction model over a simpler prior

Strongest options from the literature:

1. Critic over actor proposals.
2. Critic over retrieval-initialized candidates.
3. Critic over learned latent prior / autoencoder latent.

This is much more defensible than raw unconditional energy on SONAR embeddings.

### 7.4 Hybrid objective is more justified than any single objective

For CERBER, the most evidence-backed objective stack is:

- pairwise ranking / contrastive term for semantic ordering,
- local denoising/score term for directional guidance,
- support/OOD penalty for manifold safety,
- short-run refinement consistency loss,
- optional unconditional prior term later.

This is more SOTA-aligned than "just MDSM" or "just EM".

### 7.5 Short-run refinement consistency is missing and important

CDRL/DRL-style work suggests training should expose the model to its own short-run refined states. CERBER should not train only on analytic target directions and then hope long-run inference behaves.

Practical version:

- generate short trajectories during training,
- penalize divergence from support after refinement,
- train actor and critic on the actually reached states.

### 7.6 Stabilization tricks worth adopting

High-value practical tricks from the literature and broader generative-model practice:

- EMA weights for critic checkpoints.
- Persistent replay buffer for negatives or refined samples.
- Multi-scale or noise-ladder training.
- Data augmentation / corruption diversity for the critic.
- Hard-negative mining with retrieval and adversarial negatives.
- Ensemble or uncertainty-aware critics for trust-region control.
- Progressive refinement horizons during training.
- Correction-model view: train for short-run quality first, not perfect asymptotic MCMC.

## 8. Direct Applicability to CERBER Pipelines

### 8.1 Unconditional pipeline

Best reading from the literature:

- stable optimization is possible,
- but raw standalone `E(x)` on 1024d sentence embeddings is weakly supported,
- better formulations use latent backbones, noise ladders, cooperative initializers, or prior/correction decomposition.

Recommendation:

- Do not make standalone unconditional energy the main Stage1/2 path.
- Reuse it only as one of:
  - global prior regularizer,
  - reranker over actor candidates,
  - support-shaping auxiliary loss.

### 8.2 Simple / pairwise critic pipeline

This is much closer to the long-term CERBER design.

But it is currently under-specified because it lacks:

- mass/support calibration,
- explicit actor support constraints,
- short-run training on self-generated endpoints,
- retrieval-grounded semantic calibration.

Recommendation:

- Keep pairwise critic as the main branch.
- Upgrade it with hybrid losses and support-aware actor coupling.

### 8.3 Actor design implications

The literature strongly suggests the actor should be more than "query + noise":

- actor should learn proposal distribution inside support,
- actor should likely operate in a constrained latent delta space,
- actor updates should be regularized toward dataset support or nearest neighbors,
- critic refinement should be short-run and trust-region bounded.

## 9. Prioritized Recommendation Matrix

| Priority | Recommendation | Why it is supported externally | Direct CERBER effect |
|---|---|---|---|
| P0 | Keep `simple/pairwise critic` as the main branch; demote standalone unconditional to auxiliary role | Most successful refinement-like systems use energy as correction or guide, not sole generator | Aligns Stage1/2 with architecture goal |
| P0 | Add support/OOD constraints to actor proposals and refinement endpoints | PLAS/PRDC-style constraint logic directly addresses OOD drift | Reduces off-manifold refinement |
| P0 | Replace single-objective training with hybrid objective: contrastive/ranking + local denoise + support penalty | Literature repeatedly shows local score objectives miss global support and mode allocation | Fixes current "L2 good / cosine bad / manifold bad" split |
| P0 | Train on short-run self-refined states, not only analytic targets | DRL/CDRL/cooperative refinement evidence | Makes inference behavior closer to train behavior |
| P1 | Introduce persistent replay / hard negatives / NCE-like calibration for critic | NCE and improved CD remain key for density-ratio and mode calibration | Better semantic ranking and support awareness |
| P1 | Add EMA critic, multi-start eval, and trust-region metrics | Standard stabilization; directly addresses noisy refinement artifacts | More reliable checkpoint selection |
| P1 | Learn a better initializer/actor explicitly before expecting long Langevin loops to work | Modern successful systems almost always use an initializer/backbone | Fewer steps, better support, more stable refinement |
| P2 | Explore unconditional energy only as global prior or reranker, not sole denoiser | Raw high-d standalone EBM remains weakly supported here | Preserves useful pieces without overcommitting |
| P2 | If pursuing an unconditional branch, move toward CDRL-style cooperative training or latent-backbone correction | Better supported than naked EM on raw embeddings | Higher chance of meaningful unconditional energy |
| P3 | Long-term: manifold-aware or learned-metric refinement instead of plain Euclidean Langevin | Supported by Riemannian score/MCMC literature | Better geometry for System 2 / deep thinking |

## 10. Concrete Research-Backed Design for Next CERBER Iteration

### 10.1 Recommended Stage1.5 design

Train three coupled pieces:

1. Actor / initializer
   - predicts `x_init` or `delta_init` from query/context,
   - regularized to stay near train support.

2. Pairwise critic
   - scores `(query, candidate)`,
   - trained with:
     - ranking/contrastive semantic loss,
     - local denoising direction loss,
     - support penalty on refined endpoints,
     - hard negatives.

3. Optional unconditional prior
   - trained as auxiliary support prior or reranker,
   - not responsible for primary denoising success.

### 10.2 Inference design

At inference:

1. Actor proposes several candidates.
2. Critic reranks them.
3. Critic performs short-run bounded refinement.
4. Support gate rejects OOD moves.
5. Decoder sees top candidate(s).

This is far more consistent with the external evidence than a single long unconstrained Langevin chain.

## 11. Bottom Line

The biggest missing ingredient in the current CERBER Stage1 thinking is not another scalar loss term. It is the lack of an explicit support-aware proposal model.

The literature does not support the expectation that a raw energy function alone will stably learn:

- semantic ranking,
- support preservation,
- denoising directions,
- global generation,
- and long-run refinement dynamics,

all at once in a 1024d embedding space.

The strongest SOTA-aligned answer is:

- actor/initializer for in-support proposals,
- pairwise critic for local energy guidance and ranking,
- auxiliary unconditional/support prior if useful,
- support-aware metrics and trust-region controls,
- training on the same short-run refinement regime used at inference.

## 12. Recommended Immediate Decisions

1. Make `pairwise critic + explicit actor + support constraints` the primary architectural path.
2. Stop treating standalone unconditional EBM as the default next-step solution.
3. Add support/OOD penalties and short-run self-refinement training before increasing model size.
4. Evaluate with support and retrieval semantics, not only energy and L2.
5. If unconditional research continues, pursue `CDRL-style cooperative branch` or `energy-as-prior-over-better-latent-space`, not raw standalone EM-only training.
