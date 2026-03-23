# Alternatives Analysis: Bypassing Scalar EBM Landscape Problems

**Date:** 2026-03-23
**Context:** Arch_v1 (MDSM-trained EBM) failed at low noise — converged to "norm attractor" (cos_after ≈ 0.25 regardless of input). The scalar energy landscape in 1024d is fundamentally hard to shape. This report analyzes alternatives that achieve CERBER's goals while minimizing or eliminating scalar landscape problems.

---

## 0. What the EBM Was Supposed to Do

CERBER's core loop:
```
V_init → [Langevin: V ← V - η∇E(V_query, V) + noise] → V_answer → SONAR decode → text
```

The EBM must provide:
1. **Directional gradients** — ∇E points from bad answers toward good answers
2. **Quality scoring** — E(good) < E(bad), with enough energy gap for discrimination
3. **Multi-scale sensitivity** — works near data (fine corrections) and far from data (coarse transport)

**Root problem with scalar EBM:** A single scalar function E: ℝ^1024 → ℝ must simultaneously encode millions of "basins of attraction" (one per valid answer to each query). In practice, gradient-based training tends to collapse to trivial solutions (norm attractor, flat landscape, mode averaging).

---

## 1. Candidate Approaches

### 1.1 Vector Field Model (Flow Matching / Score Network)

**Idea:** Replace scalar E(x) with a learned vector field v_θ(x_query, x) : ℝ^1024 → ℝ^1024 that directly predicts the denoising direction. No energy landscape needed.

**Training:**
- Flow Matching (Lipman et al., 2023): v_θ learns the OT velocity u_t = x_clean - x_noisy
- Score Matching: v_θ learns ∇ log p(x | x_query)
- Loss: ||v_θ(x_query, x_noisy) - (x_clean - x_noisy)||²

**Inference:**
```
V ← V + η · v_θ(V_query, V)   # deterministic refinement
```

**Pros:**
- **No landscape problem** — directly predicts direction, not scalar
- **1024× more capacity** — output is 1024d vector, not 1d scalar
- Proven at scale (Stable Diffusion 3, Flux use flow matching)
- Can use same 1-Lipschitz architecture (OrthoLinear + GroupSort) for stability
- Training is simulation-free and stable (MSE loss)

**Cons:**
- Loses thermodynamic consistency (no Boltzmann distribution guarantee)
- No energy-based quality scoring — can predict direction but not "how good is this?"
- Path dependent (curl-free property lost)

**Verdict: ★★★★☆ — Best for pure denoising/refinement. Loses scoring capability.**

---

### 1.2 Actor + Critic (Current Arch_v2)

**Idea:** Separate the two roles:
- **Actor** (LatentDenoiseActor): predicts denoising direction (vector field)
- **Critic** (SimpleEnergy): evaluates quality (scalar energy, trained with easier objectives)

**Training:**
- Actor: MSE loss on denoising delta, σ-conditioned
- Critic: margin contrastive loss (clean vs noisy), no need for score matching
- Joint: shared optimizer, alternating or combined loss

**Inference:**
```
V ← V + η · Actor(V_query, V, σ)   # actor proposes step
score = Critic(V_query, V)            # critic evaluates (for selection/termination)
```

**Pros:**
- **Division of labor** — actor handles hard directional prediction, critic only does binary comparison (much easier)
- Critic doesn't need gradients for refinement — only for evaluation/ranking
- Actor training is straightforward supervised learning (no landscape shaping)
- Already partially implemented

**Cons:**
- Two models to train and tune
- Actor may overfit to training distribution
- Critic is still a scalar EBM (but with relaxed requirements — no gradient quality needed)

**Verdict: ★★★★★ — Most practical. Already in progress. Critic role is simplified.**

---

### 1.3 Denoising Autoencoder / Predictor (Direct Regression)

**Idea:** Skip energy entirely. Train a deterministic model f_θ(x_query, x_noisy) → x_clean that directly predicts the clean embedding.

**Training:**
```
L = ||f_θ(x_query, x_noisy) - x_clean||²
```

**Inference:**
```
V_answer = f_θ(V_query, V_noisy)   # single forward pass, no iteration
```

**Pros:**
- **Simplest possible approach** — one forward pass, no Langevin, no landscape
- Trivially trainable (supervised MSE)
- Can use any architecture (doesn't need Lipschitz constraint)
- Fast inference

**Cons:**
- **Mode averaging** — if multiple valid answers exist, predicts their mean (blurry)
- No iterative refinement — single shot must be perfect
- Doesn't scale to multi-step reasoning (no CoT)
- Doesn't align with CERBER's energy-based philosophy

**Verdict: ★★☆☆☆ — Too simple for CERBER's goals. Good baseline to beat.**

---

### 1.4 Contrastive Representation + kNN (Non-Parametric Critic)

**Idea:** Train a contrastive encoder h_θ(x_query, x_candidate) → ℝ^d that maps QA pairs to a metric space where correct answers are close to their queries. Score = negative distance.

**Training:**
- InfoNCE / SimCLR style: pull correct QA pairs together, push incorrect apart
- No scalar energy landscape needed

**Inference:**
```
score = -||h_θ(V_query) - h_θ(V_candidate)||²
∇V score = 2 · (h_θ(V_query) - h_θ(V_candidate)) · ∇V h_θ(V_candidate)
V ← V + η · ∇V score
```

**Pros:**
- Contrastive training is extremely well-understood and stable
- Metric space has geometric structure (not arbitrary scalar)
- Gradients are meaningful by construction (point toward query embedding)
- Scales to hard negatives naturally

**Cons:**
- Still needs gradient through the network for refinement (same Lipschitz concerns)
- Projection to metric space may lose information
- Quality evaluation requires distance computation, not absolute energy

**Verdict: ★★★☆☆ — Interesting but indirect. Gradient quality still uncertain.**

---

### 1.5 Energy Matching (Implemented, Untested)

**Idea:** Train scalar E(x) via simulation-free flow matching loss. Time-invariant energy whose gradient matches the OT velocity field.

**Already implemented in `experiments/02_energy_matching/`.**

**Pros:**
- Theoretically optimal — at convergence, -∇E = OT velocity
- Conservative field (curl-free) — thermodynamically consistent
- No Langevin chains during training (simulation-free)
- NCE warmstart fixes mode proportion blindness

**Cons:**
- **Still a scalar landscape** — all the same fundamental challenges
- 1024d is far beyond validated regimes (paper tested on images ≤128d latent)
- Requires the energy landscape to encode ALL transport directions simultaneously
- Training convergence on concentrated manifolds (SONAR, ||x|| ≈ 0.2051) is untested

**Verdict: ★★★☆☆ — Theoretically elegant but may hit the same wall as Arch_v1.**

---

### 1.6 Diffusion-Based Refinement (Conditional DDPM/DDIM in Latent Space)

**Idea:** Train a conditional denoiser ε_θ(x_t, t, x_query) that predicts noise at each diffusion timestep. No scalar energy needed.

**Training:**
```
L = ||ε_θ(x_t, t, x_query) - ε||²   where x_t = √ᾱ_t · x_clean + √(1-ᾱ_t) · ε
```

**Inference (DDIM for deterministic):**
```
x_{t-1} = √ᾱ_{t-1} · x̂_0(x_t) + √(1-ᾱ_{t-1}) · ε_θ(x_t, t, x_query)
```

**Pros:**
- **State-of-the-art generation quality** — proven in latent diffusion (Stable Diffusion)
- No scalar landscape — directly learns noise/denoising at each scale
- Multi-scale by construction (t ∈ [0,1])
- Can condition on query trivially (cross-attention or concat)
- DDIM allows controllable compute (few steps for speed, many for quality)

**Cons:**
- Time-conditioned (breaks time-invariance)
- No absolute quality scoring (needs separate classifier/critic)
- More complex architecture (requires t-embedding)
- Not energy-based (loses verifier semantics)

**Verdict: ★★★★☆ — Very powerful for generation. Needs separate verifier for ranking.**

---

## 2. Recommended Strategy

### Primary: Actor + Critic (1.2) — already in progress

This is the right architecture because:

1. **Actor handles the hard part** (directional prediction) via simple supervised loss
2. **Critic only needs to rank** (not provide gradients for refinement), which is dramatically easier than shaping a full landscape
3. **Modular** — can swap actor (MSE → flow matching → diffusion) independently of critic
4. **Preserves CERBER philosophy** — still has energy-based verification

### Enhancement: Vector Field Actor (1.1 inside 1.2)

Replace the current MLP actor with a proper **conditional flow matching** actor:

```python
# Current actor:  delta = Actor(v_query, v_current, sigma)
# Enhanced actor:  delta = FlowActor(v_query, v_current, t)
```

Training with flow matching loss instead of MSE on delta — this handles multi-scale denoising more naturally.

### Critic Training Strategy (Simplified)

Since the critic no longer provides refinement gradients (that's the actor's job), train it with:

1. **Margin contrastive loss** — E(query, clean) < E(query, noisy) - margin
2. **Curriculum negatives** — gradually harder pairs
3. **No MDSM/score matching needed** — the critic doesn't need correct ∇E

This completely sidesteps the scalar landscape problem because the critic only needs **monotonic ordering**, not a full energy landscape.

---

## 3. What Changes from Current Plan

| Component | Current Plan | Recommended Change |
|-----------|-------------|-------------------|
| Actor training | MSE on denoising delta | **Keep, but consider flow matching variant** |
| Critic training | MDSM (score matching) | **Switch to margin contrastive only** |
| Critic role at inference | Provides ∇E for Langevin | **Only ranks/selects/terminates** |
| Refinement loop | Langevin dynamics (∇E) | **Actor predictions (iterative)** |
| Quality scoring | E(query, candidate) | **Keep — but as ranking signal only** |
| Kill criterion | Gradient-based denoising improvement | **Actor-based denoising improvement** |

---

## 4. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Actor mode-averages for ambiguous queries | Medium | Medium | Iterative refinement (multiple steps), diversity via noise |
| Critic fails to rank correctly | Low | High | Margin contrastive is proven; curriculum helps |
| Actor doesn't generalize to unseen queries | Medium | High | Stage 2 trains on diverse QA pairs; regularization |
| Two-model coordination overhead | Low | Low | Single optimizer, shared data pipeline |
| Scalar landscape problems return in Stage 2+ | Medium | Medium | Critic only ranks (no gradient requirements) |

---

## 5. Conclusion

**The Actor + Critic split is the correct answer to "how to do what EBM was supposed to do without the landscape problems."**

The key insight: CERBER's original design asked one model (EBM) to do two things:
1. Predict which direction to move (via ∇E) — **HARD for a scalar function in 1024d**
2. Evaluate quality (via E value) — **EASY for a scalar function**

Splitting these into Actor (1) and Critic (2) lets each model solve the easier version of its problem. The critic never needs good gradients again — it just needs monotonic ordering. The actor is a vector field, which naturally has 1024× more capacity for directional prediction.

This is not a compromise — it's architecturally superior to the pure-EBM approach.
