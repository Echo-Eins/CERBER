# CERBER SOTA Research (Restarted from Zero)

Date: 2026-03-23  
Status: In progress, now with substantive findings  
Goal: find absolute SOTA approaches to solve two core problems:
1) Stage 1 learns too slowly,  
2) no confidence that current training converges to useful energy geometry.

---

## 0) What was re-read before research

Reviewed locally from scratch:
- `AGENTS.md` (latest revision)
- `CEBCM_Technical_Specification.md`
- `IMPLEMENTATION_PLAN.md`
- Stage 1 code modules:
  - `cebcm/models/activations.py`
  - `cebcm/models/normalization.py`
  - `cebcm/models/energy.py`
  - `cebcm/training/losses.py`
  - `cebcm/inference/langevin.py`
  - `cebcm/data/dataset.py`
  - `cebcm/data/encode_dataset.py`
  - `cebcm/models/sonar_wrapper.py`
  - `cebcm/visualization/energy_landscape.py`
  - `experiments/01_denoising_poc/train.py`
  - `experiments/01_denoising_poc/evaluate.py`
  - `experiments/01_denoising_poc/visualize_landscape.py`
  - `configs/base.py`

---

## 1) Current bottlenecks in Stage 1 (code-grounded)

### 1.1 Major training cost drivers

1. **Second-order autograd in every batch**
   - In `multiscale_dsm_loss`, gradient of energy wrt input is computed with `create_graph=True` (required for backprop through score loss).
   - Code reference: `cebcm/training/losses.py` (`torch.autograd.grad(..., create_graph=True)`).
   - This is computationally expensive and memory-heavy in 1024d.

2. **Bjorck orthonormalization inside each forward pass**
   - `OrthoLinear.forward()` orthonormalizes weight matrices every pass.
   - Code reference: `cebcm/models/normalization.py` (`bjorck_orthonormalize`, called in `forward`).
   - `n_iters=15` by default.
   - For large layers (input `4*1024+8`, hidden 2048/1024/512), this is a major FLOP bottleneck.

3. **No mixed precision / compile optimizations in train loop**
   - Current `train.py` does not use `torch.amp`, gradient scaling, `torch.compile`, CUDA graphs, fused optimizers, etc.
   - Code reference: `experiments/01_denoising_poc/train.py` main training loop uses standard FP training path.

4. **Langevin eval is fully iterative and per-sample**
   - Evaluation uses many refinement steps; useful for quality checks, but expensive for frequent iteration.

### 1.2 What this implies (inference from code)

Inference: slow learning is not only a hyperparameter issue; the current objective+architecture pair is intrinsically expensive:
- MDSM with gradient-of-energy supervision (second-order path),
- plus strict orthonormalization cost,
- plus missing systems-level acceleration stack.

---

## 2) SOTA map for CERBER-relevant research axes

## 2.1 EBM objectives and training strategy

### A) Multi-scale score matching family (your current direction)

- NCSN/score-based direction remains a strong theoretical base for denoising fields and Langevin guidance.
- Practical downside for your setup: computational burden from gradient-based score supervision on high-dimensional embeddings.

When to keep:
- If geometric correctness of local score field is non-negotiable and you can afford heavy training.

### B) Flow Matching (FM)

- Flow Matching provides simulation-free vector-field training and often better optimization behavior than classical diffusion objectives.
- Useful as teacher process or alternative trajectory model for latent navigation.

Why relevant:
- Could reduce uncertainty of convergence by training a direct transport field between noisy and clean embeddings.

### C) Energy Matching (2025 direction)

- Newer EBM direction explicitly targeting simulation-free training while retaining scalar-potential advantages.
- Reported as strong generative performance in source claims (vision benchmarks).

Inference for CERBER:
- This is one of the most promising "replace MDSM core" candidates if adapted to sentence embedding manifold constraints.

### D) Contrastive objectives after Stage 1

- Focal-InfoNCE remains useful for hard-negative emphasis in pairwise retrieval/ranking phases.
- Not a replacement for score supervision in denoising phase, but useful for later discriminative sharpening.

---

## 2.2 Sampling and inference dynamics

### A) PID-controlled Langevin (already in your codebase)

- Good practical direction for accelerating convergence without retraining model objective.
- Drop-in replacement advantage is strong for iterative research.

### B) Underdamped second-order Langevin

- Strong candidate for deep/refinement regimes with many steps and rugged landscapes.
- Better basin traversal can help when energy surface has local traps.

### C) Distillation pathway (SOTA pattern)

- Modern trend: distill iterative samplers into few-step or one-step samplers (consistency/flow-distillation style).

Inference for CERBER:
- Keep iterative Langevin as teacher.
- Train student latent updater for 1-4 steps.
- This is likely the highest leverage path to practical latency reduction after a stable critic exists.

---

## 2.3 Latent concept/world-model layer

### A) SONAR as pragmatic multilingual latent foundation

- Still a practical choice for PoC due existing encoder/decoder tooling.
- Bottleneck is not SONAR itself now; bottleneck is critic training dynamics on top of SONAR.

### B) JEPA line and concept-level prediction

- JEPA philosophy (predict abstract representations, not tokens) aligns directly with CEBCM goals.
- Supports your architectural bet that reasoning/planning in latent space is viable.

### C) Large Concept Models (LCM)

- Strong evidence that sentence/concept-level latent modeling can scale and generate coherent long-form outputs.
- Highly relevant as a north-star architecture reference for CERBER evolution beyond Stage 1.

---

## 2.4 Long-context memory and linear-time sequence modeling

### A) Mamba-2 / SSD family

- Important for linear-time context processing with competitive quality.
- Useful when you move from Stage 1 denoising to sequence-level memory/predictor tracks.

### B) Gated DeltaNet (ICLR 2025)

- Strong recurrent memory-control architecture with reported gains over Mamba2/DeltaNet on several long-context tasks.
- Hardware-efficient chunkwise training story is relevant for scale.

### C) Titans / Atlas (test-time memory)

- Directly aligned with your "surprise memory" and context compaction direction.
- Suggests explicit test-time memory update mechanisms can outperform static-context-only modeling.

### D) Kimi Linear (new long-context direction)

- Another strong signal that hybrid linear attention + memory-efficient kernels is currently SOTA frontier for long contexts.

Inference:
- Your Stage 4 memory roadmap is strategically correct.
- Biggest risk is execution complexity; should gate adoption via strict ablations.

---

## 3) High-impact findings for your immediate problem

## 3.1 Why training is slow right now

Root causes (not guesswork):
- score-matching with second-order graph,
- expensive orthonormalization each forward at large width,
- missing systems optimizations.

## 3.2 Why convergence is uncertain

- Stage 1 loss ensures local denoising direction, but not necessarily globally well-shaped energy landscape for downstream QA-style reasoning.
- Current pipeline needs stronger intermediate diagnostics than final cos improvement only.

---

## 4) Recommended prioritized roadmap (SOTA-first)

## Phase 0 (immediate, 1-2 days): speed and observability baseline

1. Add systems optimizations without changing objective:
   - AMP (`torch.amp`)
   - `torch.compile` where stable
   - dataloader tuning (`num_workers`, `pin_memory`, persistent workers)
2. Add profiler traces:
   - per-step time split: forward, loss, backward, optimizer, data
3. Add landscape diagnostics each N epochs:
   - gradient norm distribution
   - Hessian proxy / smoothness proxy
   - denoise success across sigma bins

Kill criterion:
- If <1.7x speedup in tokens/sec or steps/sec from pure systems tuning, move faster to Phase 1 objective redesign.

## Phase 1 (short term, 3-7 days): objective ablation to reduce instability

Run controlled matrix:
1. `MDSM + orthonorm` (baseline current)
2. `MDSM + spectral_norm`
3. `MDSM + orthonorm` but lower Bjorck iterations schedule
4. `contrastive warmstart -> MDSM finetune`

Metrics:
- wall-clock to reach fixed denoise threshold
- success rate by noise bucket
- energy monotonicity along Langevin path
- variance across seeds

Decision:
- choose fastest stable recipe; do not optimize everything simultaneously.

## Phase 2 (1-3 weeks): integrate frontier method

Top candidate order:
1. **Energy Matching adaptation** for 1024d sentence embeddings (highest novelty/impact)
2. **Flow Matching auxiliary teacher** for trajectory guidance
3. **Sampler distillation** (iterative teacher -> few-step student)

Target:
- preserve or improve denoise quality with 3-10x fewer iterative steps at inference.

---

## 5) Research-backed hypotheses to test next

H1. Most of current slowdown comes from Bjorck + second-order graph, not from model width alone.  
H2. A two-stage curriculum (`cheap objective warmup -> precise score objective`) gives faster and more stable convergence.  
H3. Distilled latent updater can replace long Langevin loops for fast-shot mode while keeping full Langevin as deep-thinking fallback.  
H4. Long-context stack should be introduced only after critic stability, otherwise debugging becomes intractable.

---

## 6) Updated risk register (practical)

1. **Over-optimizing architecture before profiling**
   - Mitigation: strict profiler-first phase.
2. **Using newest papers without adaptation to embedding manifold**
   - Mitigation: small controlled prototypes, manifold/radius constraints preserved.
3. **Mixing too many innovations at once**
   - Mitigation: one major change per experiment family.
4. **False positive progress from average metrics**
   - Mitigation: noise-bucketed metrics + trajectory diagnostics + seed variance.

---

## 7) Source index (primary sources)

Note: links below are primary papers/docs used for this research pass.

### Core CEBCM-relevant
- Energy Matching (2025): https://arxiv.org/abs/2504.10612
- Flow Matching (ICLR 2023): https://arxiv.org/abs/2210.02747
- PID-controlled Langevin Dynamics: https://arxiv.org/abs/2511.12603
- Underdamped diffusion bridges: https://arxiv.org/abs/2503.01006
- SONAR (Meta): https://arxiv.org/abs/2308.11466
- Large Concept Models: https://arxiv.org/abs/2412.08821
- I-JEPA: https://arxiv.org/abs/2301.08243
- A Path Towards Autonomous Machine Intelligence (LeCun): https://openreview.net/pdf/315d43ba26f55357a84cec9a7ed15a6610094f79.pdf

### Long-context and memory frontier
- Titans (test-time memory): https://arxiv.org/abs/2501.00663
- Atlas (test-time memory): https://arxiv.org/abs/2505.23735
- Kimi Linear: https://arxiv.org/abs/2510.26692
- Gated DeltaNet (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf
- Mamba-2 / SSD: https://arxiv.org/abs/2405.21060

### Training/systems acceleration
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- 8-bit Optimizers: https://arxiv.org/abs/2110.02861
- PyTorch docs hub (compile/AMP/checkpoint): https://pytorch.org/docs/stable/

### Contrastive refinement
- Focal-InfoNCE (EMNLP Findings 2023): https://aclanthology.org/2023.findings-emnlp.315/

---

## 8) Short conclusion for current stage

Current CEBCM Stage 1 is conceptually strong but computationally expensive by construction.  
Fastest path to practical progress:
1) systems-speed baseline first,  
2) objective ablation second,  
3) Energy Matching / Flow-Matching-assisted redesign third,  
4) sampler distillation once critic becomes stable.

---

## 9) Concrete experiment matrix (next practical loop)

## 9.1 Phase 0 - systems speed matrix

Run each toggle independently first, then best-combined run:
1. Baseline (current)
2. + AMP only
3. + `torch.compile` only
4. + dataloader tuning only
5. + AMP + compile + dataloader tuning

Track:
- step_time_ms
- peak_vram_mb
- steps_to_target_denoise
- train instability events (NaN, exploding norm, divergence)

## 9.2 Phase 1 - objective/architecture matrix

1. MDSM + orthonorm (`n_iters=15`) [current]
2. MDSM + orthonorm (`n_iters` schedule: 15 -> 8 -> 4)
3. MDSM + spectral norm + GroupSort
4. Margin warmstart (short) -> MDSM finetune
5. MDSM + lower hidden width (for speed baseline sanity)

Track:
- wall-clock to hit target quality
- denoise success by noise bins (5/10/20/30%)
- energy monotonicity along trajectories
- seed variance (min 3 seeds)

## 9.3 Phase 2 - frontier replacement candidates

1. Energy Matching prototype on SONAR vectors
2. Flow-Matching auxiliary field teacher
3. Distilled latent updater (teacher: iterative Langevin)

Decision rule:
- Keep only methods that improve either:
  - time-to-quality by >=2x, or
  - inference steps by >=3x at similar quality.

---

## 10) SOTA-to-CERBER fit ranking (current)

1. **Energy Matching adaptation**  
Reason: directly targets simulation-free EBM training pain point.

2. **Distillation of iterative sampler**  
Reason: likely biggest practical inference speedup once teacher is stable.

3. **Flow Matching auxiliary training**  
Reason: strong optimization behavior and trajectory supervision utility.

4. **Gated DeltaNet/Mamba2 stack for memory tracks**  
Reason: high value for Stage 4+, but should not block Stage 1 stability.

5. **Titans/Atlas-style test-time memory modules**  
Reason: strategic for long-context capability, but adds system complexity.

---

## 11) Explicit assumptions and uncertainty

1. Some recent papers are validated mostly on vision or generic sequence domains, not sentence-embedding EBMs on SONAR sphere; adaptation risk is real.
2. Reported SOTA gains are benchmark-specific; CERBER-specific reproduction is required before architectural commitment.
3. Inference sections above include planned extrapolation from literature to CERBER; these are marked as inference and should be validated experimentally.

---

## 12) Pass 2 Addendum (deeper SOTA pass, 2026-03-23)

This section extends the first pass with additional frontier methods and more concrete transfer paths to CERBER.

## 12.1 Simulation-free and near-simulation-free EBM training frontier

### A) iEFM (Iterated Energy-based Flow Matching)
- Source: arXiv:2408.16249
- Key idea: off-policy CNF training from unnormalized densities with energy-based flow matching objective.
- Why it matters for CERBER: this is close to your setting (energy-defined targets + expensive trajectory simulation).
- Transfer hypothesis: use iEFM-style objective to learn latent transport in SONAR space while keeping scalar-energy critic for regularization.

### B) "No Trick, No Treat" (survey + analysis of simulation-free samplers)
- Source: arXiv:2502.06685
- Key claim: many simulation-free neural samplers still collapse without additional preconditioning; Langevin-like preconditioning remains central.
- Why it matters for CERBER: confirms that removing Langevin entirely too early is risky; better path is hybrid teacher-student transition.

### C) Energy discrepancies / dual score approaches
- Sources: arXiv:2307.06431 (Energy Discrepancy), arXiv:2506.05310 (dual score matching)
- Practical relevance: both attempt to avoid expensive/unstable EBM training components and improve normalization consistency.
- Transfer caution: mostly image-centric evidence; should be treated as medium-priority exploration.

---

## 12.2 Few-step / one-step sampler acceleration (critical for latency)

### A) Progressive distillation
- Source: arXiv:2202.00512
- Key idea: repeatedly halve sampler steps (N -> N/2 -> ...), retaining quality.
- CERBER mapping: distill long Langevin chains into short latent updater for fast-shot mode.

### B) Consistency models
- Source: arXiv:2303.01469
- Key idea: direct noise-to-data mapping with one-step generation option and multistep refinement fallback.
- CERBER mapping: consistency-style latent denoiser head that approximates several Langevin iterations in one pass.

### C) EMD / MSD / Bregman distillation line
- Sources: arXiv:2405.16852 (EM Distillation), arXiv:2410.23274 (Multi-student Distillation), arXiv:2510.16983 (Di-Bregman)
- Main value: mature toolkit for converting slow iterative generators into one-step generators.
- CERBER mapping: keep full iterative critic for quality ceiling, train compact student for serving path.

### D) Fast ODE/SDE samplers
- Sources: arXiv:2211.01095 (DPM-Solver++), arXiv:2209.03003 (Rectified Flow)
- Why relevant: stable low-step integration methods reduce required iterations even before full distillation.

---

## 12.3 Language-specific energy/diffusion evidence (closer to CERBER domain)

### A) EDLM (Energy-Based Diffusion Language Model)
- Source: arXiv:2410.21357 (ICLR 2025)
- Key claim: sequence-level residual EBM at each diffusion step narrows gap to autoregressive LMs and reports ~1.3x sampling speedup vs diffusion baselines.
- CERBER relevance: this is direct evidence that EBM components can improve text diffusion quality/speed tradeoff.

### B) SONAR-LLM
- Source: arXiv:2508.05305
- Key idea: model reasons in SONAR sentence-embedding space but trains via token-level cross-entropy through frozen decoder.
- CERBER relevance: strong candidate for stabilizing Stage 1/2 uncertainty by adding likelihood-grounded supervision in SONAR latent pipelines.

Inference:
- A hybrid objective (latent energy + token CE path) may improve convergence reliability compared with pure score-field supervision.

---

## 12.4 Long-context/memory frontier (expanded)

### A) Samba (ICLR 2025)
- Source: arXiv:2406.07522
- Reported highlights: 3.8B model at 3.2T tokens, zero-shot extrapolation to very long contexts, and throughput gains (reported 3.73x for long prompts in paper abstract).
- CERBER use: strong blueprint for Stage 4 memory stack where hybrid local attention + recurrent memory is needed.

### B) Gated DeltaNet
- Source: arXiv:2412.06464 (ICLR 2025)
- Key contribution: gated delta rule + parallel training algorithm; consistent gains over Mamba2/DeltaNet in cited benchmarks.
- CERBER use: candidate replacement for raw Mamba-only memory module in long-context tracks.

### C) RWKV-7
- Source: arXiv:2503.14456
- Claim: constant memory / constant per-token inference time with competitive 3B-scale performance.
- CERBER use: useful for low-resource deployment scenarios where long-session memory is required.

### D) Titans / Atlas / Kimi Linear
- Sources: arXiv:2501.00663, arXiv:2505.23735, arXiv:2510.26692
- Combined signal: explicit test-time memory and hybrid linear attention are active SOTA directions for long-context efficiency.

---

## 12.5 Practical single-GPU optimization stack (official docs grounded)

Primary operational references:
- PyTorch Performance Tuning Guide (updated 2025-07-09 in docs)
- torch.compile docs
- torch.amp docs
- torch.utils.checkpoint docs
- SDPA docs (FlashAttention-2 / memory-efficient kernels)
- bitsandbytes repo + 8-bit optimizer paper

### Recommended implementation order for CERBER Stage 1

1. Data path
- `DataLoader(num_workers>0, pin_memory=True)`
- remove sync bottlenecks and unnecessary host/device transfers.

2. Precision
- wrap forward/loss in `torch.amp.autocast("cuda")`
- use `torch.amp.GradScaler("cuda")` where needed.

3. Compile
- `torch.compile` on hot modules (start with `SimpleEnergy` and loss path).
- benchmark modes: `default` vs `reduce-overhead`.

4. Memory-vs-compute tradeoff
- activation checkpointing on the deepest expensive blocks.
- especially relevant with second-order autograd pressure.

5. Optimizer state memory
- evaluate 8-bit optimizer states for larger effective batch or width without OOM.

6. SDPA kernels (for later attention-based modules)
- leverage fused FlashAttention-2/memory-efficient attention through SDPA interface.

---

## 13) Updated decision framework (what to do if Stage 1 remains slow)

If after Phase 0 optimization speed gain is still insufficient:

1. Keep full MDSM only as reference run.
2. Introduce a "stability route":
   - warmstart objective with cheaper surrogate,
   - then switch to precise score objective.
3. Start distilled fast-shot student in parallel track:
   - teacher = iterative Langevin critic,
   - student = 1-4 step latent updater.
4. Add token-grounded auxiliary supervision path (SONAR-LLM style idea) to reduce convergence ambiguity.

Hard pivot criterion:
- if wall-clock to target quality remains >2x worse than planned budget after objective ablations, prioritize Energy Matching / iEFM-inspired redesign over incremental tuning.

---

## 14) Extended source index (pass 2 additions)

### Additional EBM/sampling sources
- No Trick, No Treat (simulation-free samplers): https://arxiv.org/abs/2502.06685
- Iterated Energy-based Flow Matching: https://arxiv.org/abs/2408.16249
- Rectified Flow: https://arxiv.org/abs/2209.03003
- DPM-Solver++: https://arxiv.org/abs/2211.01095
- Consistency Models: https://arxiv.org/abs/2303.01469
- Progressive Distillation: https://arxiv.org/abs/2202.00512
- EM Distillation: https://arxiv.org/abs/2405.16852
- Multi-student Distillation: https://arxiv.org/abs/2410.23274
- Di-Bregman one-step distillation: https://arxiv.org/abs/2510.16983
- Energy Discrepancy: https://arxiv.org/abs/2307.06431
- Dual Score Matching (normalized energies): https://arxiv.org/abs/2506.05310

### Additional language/latent sources
- EDLM (ICLR 2025): https://arxiv.org/abs/2410.21357
- SONAR-LLM: https://arxiv.org/abs/2508.05305
- Scaling test-time compute in latent reasoning: https://arxiv.org/abs/2502.05171

### Additional long-context sources
- Samba (ICLR 2025): https://arxiv.org/abs/2406.07522
- RWKV-7: https://arxiv.org/abs/2503.14456

### Systems/engineering sources
- PyTorch Performance Tuning Guide: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- `torch.compile`: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- `torch.amp`: https://docs.pytorch.org/docs/stable/amp.html
- `torch.utils.checkpoint`: https://docs.pytorch.org/docs/stable/checkpoint.html
- SDPA API: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- bitsandbytes repo: https://github.com/bitsandbytes-foundation/bitsandbytes

---

## 15) Remediation update (2026-03-23)

### 15.1 Code-level math/stability fixes applied in Stage 1
- Directional MDSM weighting now keeps relative sigma weighting but normalizes weight mean to 1.
  - Motivation: avoid near-zero scalar losses and weak gradients when `sigma2` weights are tiny.
- Added explicit trainer guard for directional cosine MDSM scale ambiguity:
  - if `directional=True` and `magnitude_aux_weight==0`, warn and neutralize aggressive energy-scale LR multiplier.
- Updated Stage 1 defaults toward stability and lower orthonorm overhead:
  - Bjorck defaults from `15 -> 8`, schedule `[8,4,2]`,
  - `mdsm_magnitude_aux_weight=0.05`,
  - `energy_scale_lr_multiplier=5.0`,
  - non-finite backoff defaults to mild per-event (`0.99`, trigger `1`).
- Langevin best-state tracking now evaluates terminal state explicitly before returning.

### 15.2 Why training looked contradictory in `transfer_note`
- LR increase during early warnings was mostly warmup behavior; with sparse, non-consecutive non-finite events, the old backoff policy rarely activated.
- `E_scale` staying at `1.00` is expected when directional cosine loss has zero magnitude auxiliary (scale nearly unidentifiable).
- Tiny reported DSM numbers with no denoising gains were consistent with a low-effective-loss-scale regime, not a reliable sign of useful field learning.

### 15.3 Latest external reality check (is this already solved?)
- EBT-style energy minimization is still an active frontier, not obsolete:
  - Energy-Based Transformers (2025) report strong scaling/inference-time-thinking results.
- Fast alternatives that avoid direct scalar-energy gradient matching are mature:
  - Flow Matching, Score-SDE, and Consistency Models are well-established high-throughput families.
- Hybrid latent-language pipelines are emerging:
  - SONAR-LLM (2025) shows sentence-embedding-space reasoning with token-level likelihood supervision.

### 15.4 Practical implication for CERBER Stage 1
- Keep current energy path as the main CERBER identity track.
- In parallel, treat score/flow/consistency-style models as a speed-risk hedge track.
- For near-term execution, prioritize stabilizing current Stage 1 training (done in code), then run controlled ablations against one fast non-energy baseline.
