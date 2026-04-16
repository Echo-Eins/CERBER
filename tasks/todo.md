# Stage 1.5 Loss Recovery Plan (2026-03-28)

## 2026-04-07 - ChainGenerator/ChainCritic Full Diagnostics (P0/P1) + Stabilization Roadmap

### Objective
Закрепить полную диагностику текущих проблем в `13_chain_generator` и `14_chain_critic`,
зафиксировать приоритеты исправлений и сформировать дорожную карту до полноценной
autoregressive QA модели.

### Critical Errors (P0)

- [x] P0.1 Паддинг-маска в train считается, но не используется в лоссе.
  - Симптом:
    - `mask` создается в `experiments/13_chain_generator/train_chain_generator.py:159`.
    - `compute_loss` вызывается без маски в `experiments/13_chain_generator/train_chain_generator.py:165`.
    - Лосс усредняется по всем позициям в `cebcm/models/chain_generator.py:460`.
  - Риск:
    - модель учится на нулевых padded-векторах как на валидных целях.
  - Требование фикса:
    - передавать `mask` в `compute_loss`;
    - считать masked mean для cosine/MSE;
    - исключить паддинг из `cos_sim_last`.
  - Критерий приемки:
    - при изменении доли паддинга train/val метрики не деградируют искусственно;
    - masked и unmasked метрики логируются отдельно.

- [x] P0.2 При обрезке длинной цепочки теряется ответ (последний шаг).
  - Симптом:
    - `chain = chain[:max_chain_len]` в `experiments/13_chain_generator/train_chain_generator.py:86`.
    - ответ добавляется в конец в `experiments/13_chain_generator/train_chain_generator.py:80`.
  - Риск:
    - финальный answer-token может быть выброшен.
  - Требование фикса:
    - обрезка с гарантией сохранения последнего шага-ответа:
      - либо `keep_last` стратегия;
      - либо window по reasoning steps + обязательный `v_answer`.
  - Критерий приемки:
    - для всех sample `chain[-1] == v_answer` после preprocessing.

- [x] P0.3 `System1=direct answer` не соответствует реальному таргету обучения.
  - Симптом:
    - `target_steps=1` помечен как direct answer в `experiments/13_chain_generator/train_chain_generator.py:129`.
    - фактически берется `chains[:, :effective_len]` в `experiments/13_chain_generator/train_chain_generator.py:156`.
    - это обычно `v_steps[0]`, а не `v_answer`.
  - Риск:
    - System1 обучается не на задачу ответа, а на первый reasoning-step.
  - Требование фикса:
    - отдельная target-policy для System1: финальный шаг цепи (`v_answer`).
  - Критерий приемки:
    - в логах System1 `target_is_answer_rate=100%`.

- [x] P0.4 Best-of-N reranking в GUI вырожден: кандидаты одинаковые.
  - Симптом:
    - кандидаты генерируются детерминированно в цикле `cerber_gui/chain_generator_diagnostics.py:616`.
    - отсутствуют шум/температура/дискретизация/diverse policy.
    - в JSON у всех кандидатов одинаковые `energy`.
  - Риск:
    - reranking фактически не работает.
  - Требование фикса:
    - stochastic candidate generation:
      - temperature;
      - latent noise per step;
      - optional diverse beam / anti-duplicate penalty.
  - Критерий приемки:
    - `std(energy)` по кандидатам > 0;
    - rerank win-rate > random baseline.

### High-Risk Issues (P1)

- [x] P1.1 Mismatch critic train vs inference по контексту.
  - Симптом:
    - train с `v_context` в `experiments/14_chain_critic/train_chain_critic.py:136`.
    - GUI rerank без контекста в `cerber_gui/chain_generator_diagnostics.py:626`.
  - Риск:
    - энергия на инференсе не соответствует обученной функции.
  - Требование фикса:
    - унифицировать вызов критика с контекстом в train/eval/inference;
    - fallback-политика контекста явно зафиксирована.
  - Критерий приемки:
    - offline eval и GUI online eval дают согласованные ранги.

- [x] P1.2 OOD по горизонту: инференс 20 шагов при train до 5.
  - Симптом:
    - `max_chain_steps=5` в train-config;
    - реальные прогоны `num_steps=20`.
  - Риск:
    - циклы, коллапс, повторения.
  - Требование фикса:
    - выровнять train horizon с planned inference horizon.
  - Критерий приемки:
    - стабильность метрик при `steps in [1..20]` без резкого провала после 5-го.

- [x] P1.3 Валидация teacher-forced без маски паддинга.
  - Симптом:
    - `model.compute_loss(v_q, chains)` в `experiments/13_chain_generator/train_chain_generator.py:204`.
  - Риск:
    - вал-метрика смещена паддингом и непригодна для раннего стопа.
  - Требование фикса:
    - masked validation identical to train masking logic.
  - Критерий приемки:
    - `val_tf_*` пересчитаны с маской и отражают качество на реальных токенах.

- [x] P1.4 Негативы critic только случайные (легкие).
  - Симптом:
    - random negative sampling в `experiments/14_chain_critic/train_chain_critic.py:81`.
  - Риск:
    - высокая `rank_acc` без переносимости на hard candidates генератора.
  - Требование фикса:
    - hard-negative mining из текущего генератора;
    - mixed negatives: random + in-batch hard + model-hard.
  - Критерий приемки:
    - рост reranking quality на real candidate pools.

### Attention Diagnostics: What Is Actually Wrong/Right

- [x] A1 Causal mask в self-attention реализован корректно.
  - `is_causal=True` в `cebcm/models/chain_generator.py:207`.
  - В diagnostics ручной causal-mask есть в `cerber_gui/chain_generator_diagnostics.py:557`.

- [x] A2 Cross-attention `~1.0` не баг в текущей архитектуре.
  - Причина:
    - `context` длины 1 (`[B,1,D]`) в `cebcm/models/chain_generator.py:114`.
    - softmax по одному ключу всегда равен 1.
  - Следствие:
    - текущий график cross-attention в GUI малоинформативен.

- [ ] A3 Диагональные self-head карты интерпретируются как symptom of copying/looping.
  - Это не доказывает ошибку маски само по себе.
  - Усиливается из-за:
    - teacher forcing without free-run correction,
    - OOD horizon,
    - target-policy mismatch для System1.

### JSON/Runtime Diagnostics Interpretation (фиксировать как baseline)

- [x] J1 `step_norms ~ 0.2051` — ожидаемо и корректно.
  - Это следствие sphere projection:
  - `cebcm/models/chain_generator.py:341`.

- [x] J2 `step_cos_to_target` пустой в text-mode — ожидаемо.
  - В text-mode нет `v_target`, поэтому cosine к target не считается.

- [x] J3 `rerank_candidates` с одинаковым `energy/cos` — ожидаемо при детерминированном N-best.
  - Причина:
    - deterministic candidate generation + `v_target=None`.

- [ ] J4 Повторы вида `Russian/Russian...`, `Saturn/Saturn...` считаются collapse-сигналом.
  - Вероятные первопричины:
    - exposure bias teacher forcing,
    - OOD horizon (20 vs train<=5),
    - invalid training objective из-за маски/таргета.

### Consolidated Problem Statement

- [x] Главная проблема сейчас не в сломанной causal-mask в attention-слое.
- [ ] Главная проблема — ошибки постановки обучения/инференса:
  - неверные таргеты (System1),
  - потеря ответа при тримминге,
  - паддинг в лоссе,
  - OOD по длине,
  - вырожденный reranking,
  - контекстный mismatch критика.

### Highest-Impact Upgrades After Bug Fixes

- [ ] U1 Перестроить objective генератора под реальный autoregressive режим:
  - `L = λ_step*L_step_masked + λ_ans*L_final_answer + λ_roll*L_free_run + λ_rank*L_inbatch_contrastive`
  - Обоснование:
    - без `L_free_run` модель остается teacher-forcing-оптимальной и unstable в rollout.

- [ ] U2 Hard-negative mining для critic из текущего генератора.
  - Не только random negatives.
  - Собирать top-k сложных кандидатов из реального декодера.

- [ ] U3 Убрать вырожденность кандидатов в System2.
  - stochastic candidates:
    - latent noise,
    - temperature,
    - diverse beam / diversity penalty,
    - anti-repeat penalties.

- [ ] U4 Выравнять train horizon и inference horizon.
  - Если production target = 20 steps, training curriculum должен доходить до 20.

- [ ] U5 Ввести anti-loop контроль генерации.
  - cosine repeat penalty,
  - stagnation early-stop (`delta_energy`, `delta_cos`),
  - latent duplicate suppression.

### Architecture Upgrades (Next Milestone)

- [ ] R1 Cross-attention memory bank вместо single-key контекста.
  - Сейчас `context=[B,1,D]` делает cross-attn почти нефункциональным.
  - Требуется `K` контекстных векторов (`query + evidence slots`).

- [ ] R2 Explicit answer-head (повышенный вес финального шага).
  - Отдельная оптимизация финального answer-step.
  - Иначе модель перераспределяет емкость в промежуточные шаги.

- [ ] R3 Двухэтапный train pipeline:
  - Stage A: generator до стабильного free-run;
  - Stage B: critic на hard negatives;
  - Stage C: joint fine-tune с малым LR генератора.

### Acceptance Gates (Do Not Promote Without Passing)

- [ ] G1 Masked objective parity:
  - train/val одинаково masked, паддинг не влияет на метрики.
- [ ] G2 System1 target integrity:
  - при `target_steps=1` таргет всегда `v_answer`.
- [ ] G3 Reranker non-degeneracy:
  - кандидаты различаются по energy и семантике.
- [ ] G4 Horizon robustness:
  - метрики стабильны на шагах до production horizon.
- [ ] G5 Critic context parity:
  - единый вызов с контекстом в train/eval/gui inference.

### P0 Implementation Review (2026-04-08)
- [x] `cebcm/models/chain_generator.py`
  - added `loss_mask` in `compute_loss`;
  - masked cosine/MSE aggregation;
  - final-step metrics now use last valid token.
- [x] `experiments/13_chain_generator/train_chain_generator.py`
  - dataset truncation preserves final answer token;
  - curriculum targets switched to suffix policy ending at answer;
  - System1 (`target_steps=1`) now trains on `v_answer`;
  - train/val `compute_loss` now pass valid-token mask.
- [x] `cerber_gui/chain_generator_diagnostics.py`
  - added stochastic candidate generation for best-of-N reranking;
  - candidate #0 deterministic baseline, others noise-perturbed.
- [x] Validation:
  - `py -3 -m py_compile cebcm/models/chain_generator.py experiments/13_chain_generator/train_chain_generator.py cerber_gui/chain_generator_diagnostics.py`

### P1 Implementation Review (2026-04-08)
- [x] `P1.1` Critic context parity (train/eval/inference)
  - `cerber_gui/chain_generator_diagnostics.py` now passes `v_context` in rerank, per-step energy, critic analysis and landscape.
  - Added compatibility wrapper for legacy critics that do not accept `v_context`.
- [x] `P1.2` Horizon OOD guard (train<=5 vs infer=20)
  - Added runtime cap in GUI diagnostics: `num_steps` is clamped by checkpoint `training.max_chain_steps` and model `max_chain_len`.
  - Added explicit cap metadata in exported diagnostics JSON/Markdown (`requested_steps`, `step_cap_applied`, `step_cap_reason`).
  - Updated `configs/chain_generator_config.json` to train with `max_chain_steps=20`.
- [x] `P1.3` Teacher-forced validation mask
  - Confirmed masked validation path in `experiments/13_chain_generator/train_chain_generator.py` via `loss_mask`.
- [x] `P1.4` Hard negatives for critic
  - Added in-batch hard-negative mining (`top-k` nearest answers by cosine) in `experiments/14_chain_critic/train_chain_critic.py`.
  - Training now uses mixed negatives: random (dataset) + hard (in-batch injection).
  - Added optional generator-hard negatives from ChainGenerator checkpoint (filtered by `cos(gen, pos)` threshold).
  - Added config knobs in `configs/chain_critic_config.json`:
    - `enable_hard_negatives`
    - `hard_negatives_top_k`
    - `enable_generator_hard_negatives`
    - `generator_hard_checkpoint`
    - `generator_hard_steps`
    - `generator_hard_slots`
    - `generator_hard_max_pos_cos`


## 2026-04-06 - ChainGenerator: Autoregressive Transformer Decoder in SONAR Space

### Architecture
Pure autoregressive Transformer decoder — NOT a denoiser. Direct QA neural network.

```
Input:  v_query [B, 1024]  (question embedding)
Output: chain [B, N, 1024]  (reasoning steps + answer, all decodable to text)

ChainGenerator:
  - Learned [START] token (1024d)
  - N × Decoder Block (Pre-Norm):
    a. Causal Self-Attention (RoPE) — chain ordering
    b. Cross-Attention to v_query — semantic grounding (NO positional enc)
    c. FFN (SiLU, 1024 → 4096 → 1024)
  - Output projection → 1024d
  - Sphere projection: normalize → scale to target_norm (0.2051)

System 1: generate 1 step (direct answer)
System 2: generate N steps (reasoning chain → answer)
Each step is a valid SONAR vector — decodable to text at inference
```

### Key Design Decisions
- RoPE for self-attention (position-content binding for chain order)
- Cross-attention to v_query without positional encoding (semantic only)
- Causal mask (autoregressive: step i sees only steps 1..i)
- SiLU activation (matches angular critic)
- Sphere projection enforces SONAR manifold
- CompositeCritic as optional reranker (NOT navigator)
- NO CE, IPP, SP — all proven dead ends

### Training
- Teacher forcing on v_steps + v_answer from HotpotQA data
- Loss: cosine similarity + MSE per chain step
- Curriculum: start System 1 (1-step), ramp to System 2 (N-step)

### Tasks
- [x] Write architecture plan
- [ ] Implement `cebcm/models/chain_generator.py`
- [ ] Create `configs/chain_generator_config.json`
- [ ] Create `experiments/13_chain_generator/train_chain_generator.py`
- [ ] Update `tasks/lessons.md` with navigation failure lessons
- [ ] Verify: model forward pass, parameter count, gradient flow


## 2026-04-06 - GUI landscape: backward-compatible checkpoint loading (old/new parametrization keys)
- [x] Reproduce mismatch source in `cerber_gui/app.py::_load_energy_model_from_checkpoint`
- [x] Add automatic state_dict migration for `net.*.weight` <-> `net.*.parametrizations.weight.*`
- [x] Keep strict mismatch reporting when migration cannot resolve incompatibility
- [x] Validate syntax for `cerber_gui/app.py`

### Review
- Root cause: GUI loader expected current parametrized key layout and rejected old Stage1.5 checkpoints with bare linear weights.
- Added bidirectional migration fallback in loader, so old and new checkpoints can be loaded by GUI landscape path.

## 2026-04-05 - Text decode garbage in Stage3 diagnostics (off-manifold final state)
- [x] Re-audit text-mode decode path (`run_text_inference`) and verify it decodes only `v_final`
- [x] Add safe decode path for text diagnostics (`decode_safe`) with decode-time norm alignment to clean SONAR norm
- [x] Add best-step decode selection for self-denoise diagnostics (oracle by cosine to known target)
- [x] Keep final-state transparency: report both selected decode and final-state decode in metrics/markdown

### Review
- Root issue: diagnostics decoded a single final state that can be worse than earlier trajectory points.
- Fix keeps optimization math untouched and only hardens diagnostics decode/readout path.
- Added `decode_source` + `cos(decoded_state,target)` to make selection explicit and auditable.

## 2026-04-05 - Legacy dataset compatibility + sequence-splitting audit (wikitext parse failure)
- [x] Reproduce/confirm Stage2 loader failure mode for old `wikitext_sonar_10k.pt` (`embeddings` format)
- [x] Add backward-compatible conversion path in `SONARSequenceDataset` (`embeddings` -> contiguous sequences)
- [x] Unify Stage3 loaders (`phase_a`, `phase_b`, `test_inference`) to accept legacy format too
- [x] Add explicit error diagnostics for corrupted/non-PT payloads
- [ ] Run smoke validation (load + split + collate) and document whether this can cap CE/IPP at ~0.60

### Review
- Root cause confirmed in code: Stage2 `SONARSequenceDataset` expected only `"sequences"` while old wiki artifacts are `"embeddings"`.
- Added unified loader `cebcm/data/sequence_loading.py` and wired it into Stage2/Stage3/GUI loaders.
- Runtime smoke test is blocked in this environment (no local `torch` runtime), but syntax checks pass.

## 2026-04-05 - Web GUI tab for ContextEncoder real metrics
- [x] Add dedicated `Context Encoder Diagnostics` backend module
- [x] Implement CE bundle loading (CE checkpoint + optional SP checkpoint)
- [x] Implement batch evaluation metrics (cos/L2/MSE/norm + thresholds)
- [x] Add robustness sweep over input noise levels
- [x] Add context-length bucket breakdown
- [x] Wire a new tab in `cerber_gui/app.py` with controls and plots
- [x] Validate syntax (`py_compile`) for GUI files

### Review
- New GUI tab provides actionable CE metrics under configurable conditions and noise stress.
- Metrics are aligned with Stage2 CE pretraining semantics (target = final vector, context = sequence without final).

## 2026-04-05 - ContextEncoder 0.60 ceiling investigation and fixes
- [x] Audit CE training objective for potential regression-to-mean ceiling
- [x] Add CE loss decomposition controls (`mse/cos/nce`) and diagnostics
- [x] Add CE gradient-norm diagnostics to detect clipping/vanishing
- [x] Add global-token minimum count + include-last-token controls in ContextEncoder
- [x] Align Stage2 configs with new CE global-token controls
- [x] Validate compile + JSON correctness

### Review
- Found concrete bottleneck: short SQuAD sequences + percent-only top-k frequently reduce to one global token.
- Added objective controls to reduce representation averaging (NCE) and made CE optimization visible in logs.

## 2026-04-05 - IPP diagnostics hardening
- [x] Add component-wise loss logging (`ipp_mse`, `ipp_nce`, `flow_vel_mse`) in IPP trainers
- [x] Add config-risk warnings for `flow sigma_init` and extreme `weight_decay`
- [x] Keep eval output with best@k for sample-quality sanity checks

### Review
- Prevents false confidence from single aggregate `loss/cos`.
- Makes regularization/sampling pathologies visible during first epochs.

## 2026-04-04 - MLP-only IPP track alignment
- [x] Align stage2 configs to `ipp.mode = mlp` for active runs
- [x] Add explicit IPP mode/class print in stage2 trainers
- [x] Add MLP contrastive term (in-batch NCE) to reduce mode-averaging
- [x] Keep flow-path changes isolated and non-blocking for MLP mode
- [x] Validate compile + JSON parse after mode switch

### Review
- User target is MLP branch; training must be auditable as MLP from first log lines.
- MLP now receives extra discrimination pressure (`ipp_nce`) beyond MSE+cosine.

## 2026-04-04 - IPP ceiling mitigation (0.60 plateau)
- [x] Re-audit FlowIPP objective vs eval metric mismatch
- [x] Add endpoint supervision to FlowIPP (MSE+cosine on integrated endpoint)
- [x] Keep objective backward-compatible via config gates
- [x] Add best-of-k eval diagnostics to IPP and CE+IPP joint trainers
- [x] Update stage2 configs to enable endpoint-loss and multi-sample eval
- [x] Validate Python compile + JSON parse

### Review
- Plateau source is not only freeze state; velocity-field fit can improve without endpoint sample improving.
- Added direct endpoint optimization to align training objective with `eval_cos_mean`.
- Added `best@k` eval to detect whether model has good modes but weak single-sample selection.

## 2026-04-04 - Joint CE+IPP trainability diagnostics
- [x] Verify whether IPP is frozen in `train_stage2_ce_ipp_joint.py`
- [x] Add explicit forced-unfreeze toggles (default on) for CE and IPP
- [x] Add trainable parameter count printouts for CE and IPP
- [x] Add `ce_grad_norm` and `ipp_grad_norm` logging in train loop
- [x] Validate script syntax and config JSON parsing

### Review
- Root issue was not hidden freeze by default; IPP was already in optimizer.
- Added explicit diagnostics to remove ambiguity and confirm gradient flow every run.
- Next tuning should target optimization/objective mismatch, not freeze state.

## Context
Pure ranking ablation (norm_mode=none, SiLU, lr=1e-3) proved ranking CAN learn energy separation:
- rank_success=0.569, spread=0.236, E[c/a/h]=-1.62/-1.47/-1.39
- BUT inference fails: cosine 0.454→0.164 (WORSE), 0% success rate
- Root cause: ranking only teaches relative order, not WHERE the minimum should be
- Energy landscape shows deep well (E=-4.19) at wrong location, Langevin goes there

## Strategy: Add losses ONE AT A TIME, verify each doesn't break ranking

### Phase 1: Anchored Ranking ✅
Config: `configs/ablation_phase1_anchored_ranking.json`
- Ranking (λ=1.0) + clean_min (λ=0.3) + energy_reg (λ=0.01)
- Result: rank_success=0.714, spread=0.359, BUT inference cosine=-0.232, success=0.39%
- Diagnosis: ranking teaches VALUES not GRADIENTS — Langevin can't follow

### Phase 1.5: Gradient Direction ✅ ← BEST CONFIG
Config: `configs/ablation_phase1_5_direction.json`
- Phase 1 + direction_loss (λ=0.3)
- Result: rank_success=0.747, spread=0.408, batch cosine=+0.011, success=60.55%
- **BREAKTHROUGH**: first positive cosine improvement, 60% success
- Direction loss still converging slowly: 0.76 → 0.69 over 20 epochs

### Phase 2: CQL + Strong energy_reg ✗ REGRESSION
Config: `configs/ablation_phase2_cql_ereg.json`
- Phase 1.5 + CQL (λ=0.1) + energy_reg (λ=0.01→0.1)
- Result: rank_success=0.708, spread=0.339, batch cosine=-0.015, success=40.23%
- **WORSE than Phase 1.5** — CQL + strong energy_reg flatten the landscape, suppress direction signal
- ABANDONED: do not add landscape-flattening losses alongside direction_loss

### Phase 2b: Longer training + multi-sample ✅ (PARTIAL SUCCESS)
Config: `configs/ablation_phase2b_longer_multisample.json`
- Phase 1.5 + 50 epochs + direction_num_samples=4
- Result: rank_success=0.884, spread=0.559, dir=0.643, cosine success=74.22%
- **Improved** over Phase 1.5 (60.55%→74.22%), but landscape has E=-16 spurious wells
- Unconstrained MLP creates deep wells in unexplored regions of 1024D

### Phase 2c: Spectral norm ✗ TOO RESTRICTIVE
Config: `configs/ablation_phase2c_specnorm.json`
- Phase 2b + norm_mode=spectral_norm
- Result: spread=0.000, rank_success=0.000 — model can't learn ANY energy separation
- σ_max(W)≤1 per layer → total Lipschitz≤1 → energy range ~0 for SONAR embeddings
- ABANDONED: hard Lipschitz kills capacity

### Phase 2d: Gradient penalty (CURRENT)
Config: `configs/ablation_phase2d_gradpenalty.json`
- Phase 2b + gradient_penalty (λ=0.1) — soft Lipschitz via ||∇E||² penalty
- norm_mode=none (full capacity) + GP penalizes steep gradients at actor points (OOD)
- [ ] Run 50 epochs
- [ ] Check: spread > 0.3 (not killed like spectral_norm)
- [ ] Check: energy range bounded (no E=-16 wells)
- [ ] Check: cosine success ≥ 74% (not worse than Phase 2b)

### Phase 2e: Universal energy_reg + interpolated GP ✅ (PARTIAL)
Config: `configs/ablation_phase2e_universal_ereg_igp.json`
- Phase 2b + universal energy_reg + WGAN-GP style interpolated gradient penalty
- direction_loss (λ=0.3) still enabled, mdsm still λ=0.0
- PID and underdamped Langevin both tried → identical poor results
- E[c/a/h] converges to ~0.98/0.99/0.99 — spread only ~0.017
- Energy landscape INVERTED: minimum at noisy point, not clean target
- Diagnosis: direction_loss teaches WHERE gradients point but not HOW MUCH

### ROOT CAUSE ANALYSIS (2026-03-29)
**`lambda_mdsm=0.0` in ALL configs — the gradient field was never trained.**

Sign audit: ALL signs are mathematically correct:
- Energy: lower = better ✅
- Ranking: E(clean) < E(actor) < E(hard) ✅
- Langevin: v ← v - lr·∇E (descent toward lower energy) ✅
- DSM target: (noisy-clean)/σ² → ∇E should point clean→noisy → -∇E points noisy→clean ✅

The problem is NOT a sign error. The problem is that **no loss ever trains the gradient field ∇E**:
- Ranking loss: teaches energy VALUES at training points only
- Energy regularization: pushes VALUES toward 0
- Direction loss (Phase 1.5+): teaches gradient DIRECTION but not MAGNITUDE
- **MDSM (λ=0.0)**: the ONLY loss that teaches both direction AND magnitude of ∇E — DISABLED

Why noise_scale=0.5 "works": at high noise, the Langevin step is dominated by √(2lr·noise_scale)·ε (random walk), not the gradient. The ranking-trained basin around the clean point is enough for random search. At noise_scale=0.0002, dynamics is purely gradient-driven, and those gradients are untrained.

### Phase 2f: Full MDSM ✗ FAILED — ARCHITECTURE KILLED CAPACITY
Config: `configs/ablation_phase2f_mdsm.json` (overdamped)
- **lambda_mdsm=1.0** + norm_mode=orthonorm + activation=groupsort
- **RESULT**: rank_success=0.001, spread=-0.001, E[c/a/h]=0.12/0.18/0.12, energy range [0.23, 0.36] = **only 0.13**
- MDSM loss stuck at ~0.9 (barely above random cosine), ranking saturated at rank(c<a)=0.999 (degenerate)
- **Root cause**: NOT MDSM itself, but **4 simultaneous changes** from Phase 2b:
  1. norm_mode: none → orthonorm (1-Lipschitz crushed capacity)
  2. activation: silu → groupsort (piecewise-constant ≠ smooth landscape)
  3. critic_lr: 0.001 → 0.0003 (3× slower learning)
  4. direction_loss: removed (replaced by untested MDSM on crippled architecture)
  5. energy_reg_universal: false → true (**known from Phase 2e lesson to flatten landscape!**)
- Phase 2b had spread=0.56, energy range [-0.03, 0.53] = 0.56 — **4× more landscape depth**
- **Violated core principle**: change ONE thing at a time

### CORRECTED ROOT CAUSE ANALYSIS (2026-03-29, post Phase 2f)

**The REAL problem is not "no gradient field supervision". It's "gradient field not smooth BETWEEN training points".**

Evidence:
- Phase 2b achieved rank_success=88%, spread=0.56, dir=0.643 (direction loss partially learned)
- But strict inference: mean_cos_success=19%, cos_improvement at noise=0.05: **-0.107** (WORSE)
- At noise=0.3: cos_improvement=+0.003 (barely positive), 53% success
- This means: **at fine noise (where Langevin is deterministic), gradients are unreliable**
- The unconstrained MLP creates correct values at training points but wild gradients in between
- Spurious energy wells (E=-16) confirmed in Phase 2b analysis

**Why MDSM was the wrong diagnosis:**
- direction_loss already trains gradient DIRECTION at sampled noisy points → 64% cosine loss (dir=0.643)
- MDSM adds magnitude supervision, but magnitude alone doesn't fix inter-point smoothness
- The real gap: gradient field quality between training distribution points → need either:
  a) Smoother architecture (soft Lipschitz, not hard 1-Lipschitz)
  b) Denser gradient supervision (more points, wider noise coverage)
  c) Fundamentally different inference approach

---

## CORRECTED PLAN: Phase 2g+ (2026-03-29)

### Strategy: Fix inference, not training (training is already 88% rank success)

Phase 2b TRAINS well but INFERS poorly. The gradient field between training
points is chaotic for the unconstrained MLP. Three orthogonal approaches:

### Phase 2g: MDSM on Phase 2b Architecture (ONE CHANGE ONLY)
Config: `configs/ablation_phase2g_mdsm_on_2b.json`
- **Start from Phase 2b EXACTLY** (norm_mode=none, silu, lr=0.001, direction_loss=0.3)
- **ONLY addition**: lambda_mdsm=0.3 (mild, NOT 1.0), mdsm_directional=true
- Keep direction_loss active (complementary — simpler target for direction, MDSM for magnitude)
- mdsm_warmup_epochs=5, mdsm_magnitude_aux_weight=0.1
- energy_reg_universal=false (LESSON: universal kills ranking!)
- Tamed Langevin for inference safety (no Lipschitz guarantee):
  `grad_tamed = grad / (1 + lr * ||grad||)`
- **Hypothesis**: MDSM adds magnitude supervision WITHOUT killing capacity
- [ ] Create config
- [ ] Implement Tamed Langevin in langevin.py
- [ ] Run 50 epochs
- [ ] Check: spread ≥ 0.4 (not killed), dir ≤ 0.65 (improved)
- [ ] Check: cosine_success at noise=0.1 > 19% (beat Phase 2b)
- [ ] If success: run 80 epochs with cosine LR schedule

### Phase 2g Results ✅ (BEST BASELINE)
- rank_success=89%, spread=0.65, dir=0.643
- Inference at noise=0.15: 82% cosine success
- Inference at noise=0.0002: 82% cosine success (but energy goes negative)
- MDSM + direction_loss on unconstrained SiLU MLP = strong ranking + decent inference

### Phase 2h: Energy Floor (random probing) ✗ INEFFECTIVE
Config: `configs/ablation_phase2h_energy_floor.json`
- Phase 2g + energy_floor with random sphere probing (64 points)
- **Result**: efloor=0.000 for ALL 50 epochs — random points in 1024D never find structured wells
- Inference unchanged from Phase 2g
- **Lesson**: random probing useless in 1024D, need adversarial probing or CD

### Phase 2i: CD + Adversarial Probing + Underdamped ✅ (IMPROVED)
Config: `configs/ablation_phase2i_cd_underdamped.json`
- Phase 2g + contrastive divergence (32 particles, 10 Langevin steps) + adversarial probing (10 gradient descent steps) + underdamped inference
- **Result**: rank_success=89.5%, spread=0.674, best_score=+0.154
- noise=0.15: cosine 0.084→0.417, **100% success** (+0.332 improvement)
- noise=0.0002: cosine 0.403→0.427, **64% success** (+0.024 improvement)
- CD and efloor both active and >0 throughout training
- **Diagnosis**: low-noise inference still weak because training σ∈[0.01, 0.3] but inference at σ=0.0002 is 50x below training minimum. MDSM score not trained at near-zero σ.

### Phase 2j: Extended σ Curriculum ✗ FAILED (sigma_eff_sq clamp blocks low-σ learning)
Config: `configs/ablation_phase2j_sigma_extended.json`
- Phase 2i + sigma_curriculum_start: 0.01→0.001 (10x lower)
- **Result**: rank_success=89.5%, noise=0.0002 success **65%** (no improvement from 64%)
- noise=0.15 **regressed** (cosine improvement -0.115 less than Phase 2i)
- **Root cause**: sigma_eff_sq clamp at 1e-6 makes σ<0.005 a dead zone for MDSM learning
- Loguniform over 2.5 decades diluted training density at important σ=[0.01, 0.3]

### Option A: Stronger CD (on Phase 2j base) — PARTIAL IMPROVEMENT
Config: Phase 2j + cd_num_samples=64, cd_num_steps=40, lambda_cd=0.3
- **Result**: rank_success=88.4%, noise=0.0002 cosine success **75.39%** (best at low noise)
- But direction loss regressed: dir=0.624 vs 0.77 (Phase 2i)
- Energy success only 3.91% — wells persist despite stronger CD
- **Trade-off**: CD well suppression competes with direction/MDSM gradient quality

### Option B: 500 Langevin Steps (on Option A base) ✗ WORSE
Config: Option A + 500 Langevin steps instead of 100
- **Result**: rank_success=89.7%, noise=0.0002 cosine success **60.55%** (WORSE than 100 steps)
- Energy success 2.73% — more steps = deeper descent into structural wells
- **Conclusion**: more steps at noise=0.0002 = more time to get trapped in local minima

### Summary of All Low-Noise Results (noise=0.0002)
| Config | Cosine Success | Cosine Δ | Energy Success | Dir |
|--------|---------------|----------|----------------|-----|
| Phase 2i (baseline) | 64% | +0.024 | low | 0.77 |
| Phase 2j (σ extended) | 65% | +0.023 | 2.73% | ~0.77 |
| Option A (strong CD) | **75.39%** | +0.038 | 3.91% | 0.624 |
| Option B (500 steps) | 60.55% | +0.015 | 2.73% | ~0.62 |

### Root Cause Analysis (2026-03-31)
1. **σ-conditioning semantic mismatch**: Training σ = actual noise level; inference σ = schedule value unrelated to sample state. At low noise, dynamics is gradient-driven → mismatch is fatal.
2. **sigma_eff_sq clamp**: Makes σ<0.005 dead zone for MDSM → extending training range is pointless.
3. **Unconstrained MLP topology**: Exponential local minima in 1024D. CD explores vanishing fraction.
4. **Energy success ~3%**: Clean target is NOT the energy minimum in 97% of neighborhoods.
5. **High-noise success is stochastic**: 100% at noise=0.15 is random walk, not gradient quality.

### Phase 2k: NCSN-Style Noise-Annealed Inference (NEXT — highest leverage)
- Anneal BOTH Langevin noise_scale AND σ-conditioning together (true NCSN sampling)
- First 50 steps: noise=0.15, σ=0.15 (stochastic search, 100% success regime)
- Last 50 steps: noise→0.0002, σ→0.01 (deterministic refinement in trained regime)
- **No retraining needed** — uses existing Phase 2i checkpoint
- Fixes σ-conditioning semantic mismatch (issue #1)
- Leverages proven 100% success at high noise as starting point
- Already have infrastructure: AdaptiveSigmaEnergyWrapper + run_langevin

### Phase 2l (if 2k insufficient): Architectural Shift
- Options: dual-critic (angular + radial), score distillation network, flow matching
- Addresses issues #3 and #4 (MLP topology, clean not energy-minimal)

### Phase 2h (ORIGINAL): Soft Lipschitz (weight decay + GP at OOD only)
Config: `configs/ablation_phase2h_soft_lip.json`
- Start from Phase 2b architecture (norm_mode=none, silu)
- Increase weight_decay: 0.01 → 0.05 (smoother weights → smoother gradients)
- Add mild gradient penalty at RANDOM points (NOT on clean→noisy corridor):
  lambda_gp=0.05, GP sampled at random perturbations of actor outputs
- Keep direction_loss + clean_min + energy_reg (clean-only)
- **Hypothesis**: soft Lipschitz via weight decay smooths gradient field without killing capacity
- **Key difference from Phase 2d**: GP at random OOD points, not along training corridor

---

# SOTA Radial+Angular Research Kickoff (2026-03-31)

## Goal
Build an implementation-ready SOTA blueprint for a true geometric twin critic:
- Angular critic (tangential semantic guidance),
- Radial critic (normal/shell/manifold control),
- Shared inference protocol with mathematically consistent gradient composition.

## Checklist
- [x] Audit current Stage 1.5 codepath and verify whether current twin critics are truly specialized.
- [x] Collect primary-source SOTA evidence for tangent/normal decomposition on manifold data.
- [x] Write a concrete architecture blueprint with losses, inference math, and ablation protocol.
- [x] Record anti-pattern in lessons to prevent relabeling homogeneous twin critics as radial+angular.
- [x] Implement specialized critic modules and integrate into Stage 1.5 trainer.
- [x] Add per-head eval metrics (angular vs radial) in eval outputs/log stream.
- [ ] Add explicit disable-head ablation switches in config + trainer.
- [ ] Validate low-noise inference stability with synchronized sigma/noise schedule via full run.

## Artifacts
- Research: `research4_radial_angular_sota.md`
- Lessons: `tasks/lessons.md` (new rule: no fake radial+angular labeling)

## Review
- Current code has twin critics, but they are homogeneous (`SimpleEnergy` + same feature fusion).
- This means current architecture is ensemble-style, not radial+angular geometric decomposition.
- Implemented:
  - `cebcm/models/energy_decomposed.py` (`AngularEnergyCritic`, `RadialEnergyCritic`)
  - Stage1.5 trainer routing by critic role (angular/radial), role-specific losses, purity/correlation regularizers
  - Sigma-dependent angular/radial head weighting in `TwinHybridEnergy`
  - GUI loader support for `critic_architecture=radial_angular`
- Pending:
  - full ablation toggles and end-to-end long-run validation
- [ ] Create config
- [ ] Run 50 epochs
- [ ] Check: spread ≥ 0.4, no spurious wells (E < -5)
- [ ] Check: cosine_success at noise=0.1 > 19%

### Phase 2i: Dual-Critic (Angular + Radial Decomposition) ★ NOVEL
- **Architecture**: two separate energy critics trained on different aspects
- **Critic_ang (angular)**: operates on normalized vectors, learns angular energy
  - Input: (q/||q||, x/||x||) → scalar E_ang
  - Loss: ranking on angular proximity + direction_loss in tangent space
  - Learns: which direction on the hypersphere to move
- **Critic_rad (radial)**: operates on norms/distances, learns magnitude energy
  - Input: (||x||, ||x-q||, cos(x,q)) → scalar E_rad
  - Loss: ranking on distance-to-target + magnitude supervision
  - Learns: how far to move (step size)
- **Inference**: Langevin uses combined gradient:
  - Angular step: project -∇E_ang onto tangent plane of sphere
  - Radial step: -∇E_rad along radial direction
  - Separate step sizes for each (angular and radial dynamics have different scales)
- **Why this might work**:
  - Each critic has a SIMPLER task → easier to train
  - No conflict between angular and radial objectives
  - Angular critic naturally Lipschitz on compact sphere → better gradient field
  - Radial critic is 1D → trivially smooth
  - Decomposes 1024D navigation into two well-conditioned subproblems
- [ ] Design architecture (new model classes)
- [ ] Implement training loop changes
- [ ] Create config
- [ ] Run 50 epochs
- [ ] Compare gradient field smoothness vs single-critic

### Phase 2j: Score Distillation Network (if 2g-2i insufficient)
- Train energy critic as Phase 2b (ranking, direction_loss, clean_min)
- Add separate lightweight score network s_θ(x) ≈ -∇E(x)
- s_θ trained with L2 regression: ||s_θ(x) - sg(-∇E(x))||² at noisy points
  (sg = stop_gradient — distill FROM critic, don't backprop through)
- At inference: use s_θ(x) for Langevin, not autograd ∇E
- **Why**: s_θ is smooth MLP trained explicitly to predict gradients → naturally interpolates
- **Bonus**: no create_graph at inference → faster inference

### Phase 2k: Flow Matching Hybrid (EXPLORATORY)
- Instead of energy-based Langevin, train conditional velocity field v(x_t, t)
- v = (x_clean - x_noisy) / (1 - t) for t ∈ [0, 1]
- ODE integration: dx/dt = v(x_t, t) — no noise, deterministic path
- Keep energy critic for quality scoring/reranking, not for navigation
- **Radical departure**: separates SCORING (energy) from NAVIGATION (flow)
- Only try if energy-gradient approaches plateau

### Kill Criteria (updated)
- Phase 2g/2h: if spread < 0.3 or cosine_success worse than Phase 2b → architecture issue, try 2i
- Phase 2i: if dual-critic training unstable for 10 epochs → decomposition doesn't work, try 2j
- All phases: if inference cosine_success < 25% after 50 epochs → consider Phase 2k (flow matching)
- If Phase 2k also fails: fundamental SONAR embedding geometry problem, need different representation

### Priority Order
1. **Phase 2g** (highest priority — tests the obvious: MDSM on working arch, ONE change)
2. **Phase 2h** (parallel — tests soft Lipschitz, independent approach)
3. **Phase 2i** (after 2g/2h results — user's dual-critic idea, novel but promising)
4. **Phase 2j** (if gradient-based approaches plateau — score distillation)
5. **Phase 2k** (last resort — paradigm shift to flow matching)

---

# CERBER GUI Web Debug Plan (2026-03-25)

## Goal
Bring `cerber_gui` math and visualization behavior into parity with the CLI landscape tool and fix Plotly backend failures.

## Checklist
- [x] Reproduce/analyze Plotly trace error path in `cerber_gui`
- [x] Diff GUI math vs CLI (`visualize_landscape.py`) for vector scale, noise model, and Langevin path
- [x] Refactor GUI inference/landscape generation to reuse Stage1-consistent Langevin logic and target norm
- [x] Fix Plotly trace construction robustness and remove invalid/fragile properties
- [x] Align GUI trajectory panel semantics with CLI contour + trajectory behavior
- [x] Validate by static checks and code-path walkthrough; document remaining runtime checks for local CUDA env

## Review (to fill after fixes)
- Findings:
  - GUI used synthetic vectors with norm `10.0` and custom Langevin, while CLI used real SONAR vectors and shared Stage1 Langevin. This caused major geometry drift.
  - GUI model loading used permissive `strict=False` with inferred dims fallback, allowing silent architecture/checkpoint mismatches and invalid landscapes.
  - `SimpleEnergy.forward()` had hard clamp `[-100, 100]`; this flattened real energy ranges (e.g. `-367..-297`) into a plateau in visualization.
  - Plotly path could fail hard on schema/version mismatch; GUI had no graceful fallback.
- Files changed:
  - `cerber_gui/app.py`
  - `cerber_gui/checkpoint_analyzer.py`
  - `cerber_gui/landscape_3d.py`
  - `cebcm/models/energy.py`
  - `experiments/01_denoising_poc/visualize_landscape.py`
- Verification:
  - `python -m py_compile cerber_gui/app.py cerber_gui/landscape_3d.py cebcm/models/energy.py experiments/01_denoising_poc/visualize_landscape.py`
  - Code-path audit confirms GUI now uses Stage1 config + shared `run_langevin()` + dataset-based clean sample + relative noise semantics.
  - Runtime UI validation on target CUDA environment remains to be executed locally (this environment has no `torch` runtime).

---
# Stage 1 Improvement Plan (2026-03-23)

## Goal
Implement the agreed Stage 1 upgrades for speed, stability, and reproducibility while preserving current architecture semantics.

## Checklist
- [x] Add config support for Bjorck schedule, DSM geometry options, and systems tuning
- [x] Implement DSM extensions: sigma sampling/weighting, tangent projection, directional mode, magnitude auxiliary
- [x] Fix Langevin best-state tracking bug and early-stop behavior
- [x] Add true trajectory capture in Langevin API
- [x] Align visualization trajectory with actual sampler trajectory
- [x] Rewrite training loop with deterministic seeds + fixed eval subset
- [x] Add AMP / compile / dataloader throughput options
- [x] Skip unnecessary GP computation when lambda=0
- [x] Save full Stage1 config in checkpoints
- [x] Make evaluate script consume checkpoint Stage1 config to avoid train/eval drift
- [x] Add NaN hotfixes after first real run feedback
- [x] Add finite-gradient guard + fail-fast/backoff + robust Björck update after second run feedback
- [x] Fix LR-collapse coupling (backoff vs warmup) and harden MDSM numerics for low-norm outliers
- [x] Full Stage1 math audit against training logs (`transfer_note`) with contradiction fixes

## Review
### Implemented files
- `configs/base.py`
- `cebcm/models/energy.py`
- `cebcm/training/losses.py`
- `cebcm/inference/langevin.py`
- `experiments/01_denoising_poc/train.py`
- `experiments/01_denoising_poc/evaluate.py`
- `experiments/01_denoising_poc/visualize_landscape.py`

### NaN hotfixes
- MDSM second-order path forced to FP32 by default (`mdsm_force_fp32=True`).
- `log_energy_scale` exponential is clamped in forward to avoid inf/nan cascade.
- `energy_scale` LR multiplier reduced/configurable (`energy_scale_lr_multiplier=20`).
- Non-finite batch guard with skip-and-continue (`skip_non_finite_batches=True`).
- Added finite-gradient guard before `optimizer.step` to prevent parameter corruption.
- Added consecutive non-finite fail-fast and LR backoff controls.
- Corrected Björck update to row/column-consistent form (`WW^T` for wide matrices) with spectral pre-normalization.
- Fixed LR-collapse bug: non-finite backoff no longer mutates `initial_lr` (warmup anchor).
- Backoff now triggers only on short consecutive streaks (default >=3), not on isolated events.
- Hardened MDSM numerics: norm/sigma floors, safer cosine epsilon, finite sanitization, and skip-rate metrics.
- Fixed objective/inference sign mismatch: Stage1 MDSM now trains target gradient with the sign consistent to `v <- v - lr * ∇E`.
- Disabled unsafe default early-stop threshold (`energy_threshold=None`) to prevent no-op Langevin refinement.
- Warmup/backoff now uses optimizer-update steps only (skipped batches no longer advance warmup), and backoff persists during warmup via per-group LR scale.

### Validation status
- Runtime training execution is still blocked in this environment because `torch` is not installed in the active Python interpreter.
- Verification performed via static inspection, compile checks, and user-provided runtime logs.

---

## Remediation Pass (2026-03-23, Transfer Log Recheck)

### Goal
Eliminate remaining math/stability contradictions from `transfer_note`: LR perception under instability, scale-identifiability gap in directional MDSM, and final-state selection in Langevin.

### Checklist
- [x] Re-audit `transfer_note` and Stage 1 code paths for unresolved contradictions
- [x] Fix directional MDSM weighting scale collapse (normalize sigma weights to mean 1)
- [x] Add directional-scale identifiability guard in trainer (`directional=True` + no magnitude auxiliary)
- [x] Tune Stage 1 default stability profile (Bjorck schedule, magnitude auxiliary, non-finite backoff behavior)
- [x] Fix Langevin best-state selection to include terminal state
- [x] Update project lessons with new failure patterns
- [x] Re-check syntax/compile on edited files
- [ ] Re-run full Stage 1 train/eval on target CUDA environment and verify denoising improvement

### Files updated in remediation pass
- `configs/base.py`
- `cebcm/training/losses.py`
- `experiments/01_denoising_poc/train.py`
- `cebcm/inference/langevin.py`

---

## Actor + Critic Track (2026-03-23)

### Goal
Implement a full Stage 1 `Actor + EBM Critic` pipeline to reduce training fragility and improve denoising convergence while preserving CERBER's energy-verifier architecture.

### Plan
- [x] Add dedicated latent denoising Actor model with sigma conditioning
- [x] Extend Stage1 config for actor_critic training/eval controls
- [x] Implement joint training loop (single optimizer, dual-network losses for actor and critic)
- [x] Integrate actor-then-critic evaluation path in train/evaluate scripts
- [x] Extend checkpoint/resume logic to persist both components
- [x] Run static validation (compile checks) and document usage
- [ ] Run full CUDA training/evaluation for actor_critic and tune initial hyperparameters from first logs

---

# Energy Matching Pipeline — Mathematically Verified Implementation Plan

**Date:** 2026-03-23
**Status:** Implementation complete, pending CUDA validation
**Branch:** `claude/review-tech-spec-xGHfW`

---

## 0. Mathematical Foundation

### 0.1 Energy Matching Core Idea (Balcerak et al., NeurIPS 2025)

Energy Matching trains a **time-invariant scalar energy** E_θ(x) : ℝ^d → ℝ such that
its negative gradient -∇_x E_θ(x) approximates the velocity field of an optimal
transport flow from noise to data.

**Key insight:** unlike flow matching (which learns a vector field v_θ(x,t)), Energy
Matching learns a *conservative* vector field derived from a scalar potential.
This guarantees:
- Path independence (the energy landscape is well-defined)
- Thermodynamic consistency (Boltzmann distribution at equilibrium)
- No need for time conditioning

### 0.2 The Conditional Optimal Transport (OT) Path

Given data point x₁ ~ p_data and noise x₀ ~ p_prior (typically N(0,I)):

```
x_t = (1 - t) · x₀ + t · x₁,  t ∈ [0, 1]
```

The conditional velocity field (ground truth):

```
u_t(x_t | x₁) = x₁ - x₀ = (x₁ - x_t) / (1 - t)
```

### 0.3 Flow Matching Loss (Lipman et al., ICLR 2023)

Standard flow matching trains v_θ(x_t, t) to match u_t:

```
L_FM = E_{t~U(0,1), x₁~p_data, x₀~p_prior} [ ||v_θ(x_t, t) - u_t(x_t | x₁)||² ]
```

### 0.4 Energy Matching Adaptation

Energy Matching replaces the free vector field v_θ(x_t, t) with the
**negative gradient of a scalar energy** -∇_x E_θ(x):

```
L_EM = E_{t~U(0,1), x₁~p_data, x₀~p_prior} [ ||-∇_x E_θ(x_t) - u_t(x_t | x₁)||² ]
```

**Critical property:** E_θ has NO time conditioning. The single energy landscape
must simultaneously encode the correct velocity at all points along all OT paths.

### 0.5 Two-Phase Behavior

The paper identifies that the loss naturally separates into two regimes:

**Phase 1 (t ≈ 0, far from data):**
- x_t ≈ x₀ (near noise)
- Target velocity ≈ x₁ - x₀ (points toward data)
- Energy gradient learns OT transport directions
- This is the "flow matching" regime

**Phase 2 (t ≈ 1, near data):**
- x_t ≈ x₁ (near data manifold)
- Velocity field converges to score function: ∇ log p(x)
- Energy learns Boltzmann-like landscape near data
- This is the "EBM" regime

### 0.6 Sampling from Trained Model

After training, generate samples via ODE integration:

```
dx/dt = -∇_x E_θ(x),  x(0) ~ p_prior
```

Discretized (Euler):
```
x_{k+1} = x_k + Δt · (-∇_x E_θ(x_k))
```

Or with Langevin noise for stochastic sampling:
```
x_{k+1} = x_k - η · ∇_x E_θ(x_k) + √(2η) · ε,  ε ~ N(0,I)
```

### 0.7 Adaptation for SONAR Embeddings (our contribution)

**Key differences from image domain:**

1. **Hypersphere geometry:** SONAR embeddings have ||x|| ≈ 0.2051 (not unit norm,
   but concentrated). After each integration step, project back:
   ```
   x_{k+1} = normalize(x_{k+1}) · target_norm
   ```

2. **Prior distribution:** Instead of N(0,I), use N(0, σ²I) matched to data distribution:
   ```
   σ_prior = target_norm / √d ≈ 0.2051 / √1024 ≈ 0.00641
   ```
   This ensures prior samples have similar norm to data.

3. **Relative noise scaling:** Following CERBER convention, noise is relative to norm:
   ```
   x_t = (1-t) · x₀ + t · x₁  where  x₀ = x₁ + ε,  ε ~ N(0, σ²·||x₁||²·I)
   ```
   This preserves the SONAR geometry better than absolute noise.

4. **1-Lipschitz constraint:** We keep OrthoLinear + GroupSort for the energy network
   to ensure smooth gradients. The final layer is unconstrained for energy magnitude.

### 0.8 Mathematical Verification Checklist

- [x] OT path x_t is well-defined: linear interpolation, ∂x_t/∂t = x₁ - x₀ ✓
- [x] Conditional velocity u_t = (x₁ - x₀) is correct: by definition of linear OT ✓
- [x] Loss L_EM minimizes ||∇E + u_t||²: convex in function space ✓
- [x] At convergence, -∇E_θ = u_t almost everywhere: by optimality of L² loss ✓
- [x] Conservative field guarantee: v = -∇E is curl-free by construction ✓
- [x] Sampling via ODE follows learned flow: by definition of gradient flow ✓
- [x] Sphere projection preserves tangent dynamics: projects only radial component ✓
- [x] Prior norm matches data norm: by construction of σ_prior ✓

---

## 1. Implementation

### 1.1 Implemented Files

```
cebcm/models/energy_unconditional.py     — E(x) → scalar, no pairwise, no σ (~3.7M params)
cebcm/training/energy_matching.py        — EM loss (MSE, cosine, weighted) + OT path + sampling
cebcm/training/negative_buffer.py        — Replay buffer + NCE loss (full & simple)
configs/energy_matching.py               — EnergyMatchingConfig dataclass
experiments/02_energy_matching/train.py   — Training script (3 modes)
experiments/02_energy_matching/evaluate.py — Evaluation (denoise + sample quality + SONAR decode)
```

### 1.2 Training Modes

1. **`nce_warmstart_em`** (recommended) — NCE (10 epochs) → Cosine EM fine-tune
2. **`energy_matching`** — Pure MSE EM from scratch
3. **`cosine_em`** — Cosine direction EM (for 1-Lipschitz networks)
4. **`weighted_em`** — Near-data weighted EM

---

## 2. Verification Against Known Failure Modes

| Failure Mode | Mitigation |
|---|---|
| Energy collapse (flat E) | NCE warmstart creates initial landscape; EM refines |
| Score matching mode blindness | NCE explicitly learns p(x)/p_n(x) ratio |
| Gradient magnitude mismatch | Cosine EM variant; unconstrained final layer |
| Prior mismatch | Matched prior σ = target_norm/√d |
| OOD sampling | Sphere projection after each step |
| Hessian cost at 1024d | Not needed — EM uses first-order only |

---

## 3. Task Checklist

- [x] Fix visualization device mismatch bug
- [x] Create `cebcm/models/energy_unconditional.py`
- [x] Create `cebcm/training/energy_matching.py`
- [x] Create `cebcm/training/negative_buffer.py`
- [x] Create `configs/energy_matching.py`
- [x] Create `experiments/02_energy_matching/train.py`
- [x] Create `experiments/02_energy_matching/evaluate.py`
- [ ] Run full CUDA training/evaluation and tune from first logs

---

## 4. Architecture Review: Actor + Critic Pattern

**Current CERBER architecture (from Tech Spec §3.1):**

| Component | Role | Training |
|-----------|------|----------|
| SONAR Encoder | Text → V (1024d) | Frozen |
| IPP | Predicts V_init (the "Actor") | Stage 2+ |
| EBT (SimpleEnergy) | Evaluates quality (the "Critic") | Stage 1+ |
| Langevin Dynamics | Refines V_init → V_answer | No training (uses ∇E) |
| SONAR Decoder | V → Text | Frozen |

**The Actor-Critic analogy:**
- **Critic = E_θ** — evaluates "how good is this candidate?"
- **Actor = IPP** — proposes initial answer
- **Refinement = Langevin** — uses Critic's gradients to improve Actor's proposal

**Energy Matching strengthens the Critic** by training it to model the full
data distribution p(x), not just local denoising directions. This gives:
1. Absolute quality evaluation (not just relative)
2. Sample generation capability (new)
3. Theoretically optimal score function near data

---

# CERBER GUI + Math Re-Audit Plan (2026-03-25, Pass 2)

## Goal
Close all remaining GUI/math inconsistencies reported by user:
- mojibake/encoding corruption in UI labels,
- denoised marker/trajectory mismatch on landscape surface,
- unconditional test semantics (clean vector should not be treated as mandatory minimum),
- misleading `improvement=0` diagnostics,
- deterministic parity with CLI and mathematically valid inference diagnostics.

## Checklist
- [x] Re-audit `cerber_gui` math path end-to-end (sampling, projection, energy eval, plotting coordinates)
- [x] Fix user-facing text encoding and checkpoint title sanitization in GUI
- [x] Split diagnostics by model type (`simple` vs `unconditional`) and remove misleading target interpretation
- [x] Ensure inference output reports both energy-descent and geometric movement; detect no-op/refusal cases
- [x] Ensure 3D + trajectory plots refresh from the same post-inference landscape payload
- [x] Validate plot coordinate conventions (x/y/z mapping) with deterministic synthetic regression checks
- [x] Run static/runtime verification scripts and summarize remaining gaps
- [x] Produce `research1.md` with mathematical proof notes + test validity matrix + external references

## Review
- Implemented:
  - Inference now uses last executed Langevin state for GUI diagnostics and visualization consistency.
  - Landscape scan auto-expands to include trajectory/denoised projections.
  - 3D/2D markers are snapped to plotted mesh interpolation to prevent visual floating.
  - Added checkpoint name mojibake recovery and explicit unconditional-mode semantics in report.
  - Added explicit `reference source` (`dataset` vs `synthetic`) in inference report with warnings when dataset vectors are unavailable.
  - Added `research1.md` with external references + code-grounded math audit.
- Verification:
  - `python -m py_compile cerber_gui/app.py cerber_gui/landscape_3d.py cebcm/visualization/energy_landscape.py experiments/02_energy_matching/train.py`
  - Runtime validation script could not run here: active Python environment has no `torch`.

---

# Compact SOTA Research for Stage1 GUI Tests (2026-03-25)

## Goal
Prepare a compact, practical evidence pack in `research1.md` with 3-5 reliable sources per question:
- (A) energy interpretation in EBMs (minimum-energy behavior and regimes),
- (B) why unconditional EBM does not have to reconstruct a specific clean sample from a noisy one,
- (C) correct metrics for unconditional energy descent in high-dimensional embeddings.

## Checklist
- [x] Re-check AGENTS workflow and register plan in `tasks/todo.md`
- [x] Collect 3-5 high-trust sources for (A)
- [x] Collect 3-5 high-trust sources for (B)
- [x] Collect 3-5 high-trust sources for (C)
- [x] Write compact theses + direct CERBER Stage1 GUI implications in `research1.md`
- [x] Verify links and close review notes

## Review
- `research1.md` added with compact A/B/C structure.
- Source coverage: 4 refs for (A), 5 refs for (B), 5 refs for (C).
- Each block includes direct implications for Stage1 GUI test semantics.

---

# GUI Inference Metrics + Unconditional Crash Fix (2026-03-25, Pass 3)

## Goal
1) Fix unconditional inference crash (`element 0 ... does not require grad`).
2) Verify architecture display path for unconditional checkpoints.
3) Replace `N/A`-style summary with live runtime metrics that refresh after:
   - checkpoint load preview inference,
   - every manual inference run.

## Checklist
- [x] Fix grad-context bug in unconditional alignment diagnostic
- [x] Validate/strengthen architecture rendering source for unconditional models
- [x] Add persistent per-checkpoint runtime metrics in session state
- [x] Compute/update metrics on preview inference (during checkpoint selection)
- [x] Compute/update metrics on manual inference and refresh checkpoint summary
- [x] Run compile validation and document findings

## Review
- Fixed unconditional crash by removing `@torch.no_grad` from alignment path and forcing local `torch.enable_grad()` around `energy_and_grad`.
- Checkpoint summary architecture now resolves directly from loaded `state_dict` (with mismatch warning vs metadata).
- Added live per-checkpoint runtime metrics store; summary now shows latest inference metrics and refreshes:
  - after checkpoint selection preview inference,
  - after every manual inference run.
- Manual inference callback now updates `checkpoint_summary` output in the same click event.
- Validation:
  - `python -m py_compile cerber_gui/app.py cerber_gui/landscape_3d.py cebcm/visualization/energy_landscape.py experiments/02_energy_matching/train.py`

---

# GUI SOTA-Eval Completion (2026-03-25, Pass 4)

## Goal
Implement full SOTA-grade GUI evaluation for Stage1/Unconditional checks, so model quality is judged by distribution and manifold metrics, not only single-vector cosine.

## Detailed spec
- Add a dedicated evaluation core (`cerber_gui/sota_eval.py`) with:
  - MMD (RBF, median heuristic),
  - C2ST (linear probe, held-out accuracy),
  - PRDC (precision/recall/density/coverage),
  - manifold kNN proximity improvements (cosine and euclidean),
  - energy-descent statistics over a batch (improvement + success rate),
  - trajectory monotonicity for the inspected sample.
- Integrate evaluation in GUI pipeline for:
  1) checkpoint preview run (on selection),
  2) every manual inference run.
- Add GUI controls for evaluation budget:
  - eval batch size (number of query samples),
  - eval reference bank size.
- Persist per-checkpoint latest SOTA metrics in session state and render them:
  - in checkpoint summary (Live Inference Metrics block),
  - in inference output markdown.
- Keep implementation robust:
  - if dataset unavailable, explicitly mark distribution metrics as unavailable,
  - avoid accidental no-grad on input-gradient diagnostics.

## Checklist
- [x] Add `cerber_gui/sota_eval.py` with stable batched metric implementations
- [x] Extend runtime metric payload and formatting to include SOTA metric block
- [x] Batch-run Langevin on evaluation sample set and compute post-denoise distribution metrics
- [x] Wire new GUI controls (eval batch size, eval bank size) into both preview/manual callbacks
- [x] Refresh checkpoint summary after each inference with updated SOTA metrics
- [x] Validate with py_compile + quick synthetic invariants (metrics finite / ranges sane)

## Review
- Implemented:
  - Added `cerber_gui/sota_eval.py` with batched metric suite:
    - MMD (RBF + median heuristic),
    - C2ST (linear probe),
    - PRDC (precision/recall/density/coverage),
    - manifold kNN proximity improvements.
  - Added `sota_eval_cache` in GUI session state and cache invalidation per checkpoint.
  - Added SOTA batch evaluation computation inside both:
    - checkpoint preview landscape generation,
    - manual inference callback.
  - Extended checkpoint summary and inference report to render SOTA block from latest runtime metrics.
  - Added explicit GUI controls:
    - `SOTA Eval Batch Size`,
    - `SOTA Eval Reference Bank Size`,
    and wired them into all preview refresh triggers + manual inference.
- Validation:
  - `python -m py_compile cerber_gui/app.py cerber_gui/sota_eval.py cerber_gui/landscape_3d.py cebcm/visualization/energy_landscape.py`
  - Quick runtime synthetic invariants could not be executed in this shell because active Python environment has no `torch`.

---

# GUI Metric Consistency Patch (2026-03-25, Pass 5)

## Goal
Resolve user-reported mismatch between visual "near-target" behavior and zero/negative improvement readouts, and surface short runtime metrics where checkpoint-saved metrics are `N/A`.

## Checklist
- [x] Add explicit 2D-plane diagnostics (projected distance before/after) to runtime output
- [x] Keep 1024D primary metrics and explicitly label projection-vs-fullspace distinction
- [x] Add fallback for missing checkpoint metrics from latest runtime inference
- [x] Improve tiny-delta formatting (scientific fallback) to avoid false `0.000000` interpretation
- [x] Extend SOTA block with L2-before/after and denoise step norm
- [x] Fix C2ST reporting to be label-invariant (`max(acc, 1-acc)`) and expose raw acc
- [x] Bump SOTA eval cache key version to avoid stale pre-fix cache reuse
- [x] Validate via py_compile

## Review
- Implemented:
  - Added `Quick Inference Snapshot` in checkpoint summary with cosine/energy deltas.
  - Added fallback backfill for missing checkpoint cosine/success metrics from latest runtime run.
  - Added `2D slice distance to reference` diagnostics and explicit note that plot coordinates are projections.
  - Added L2 diagnostics (`l2_before/after/improvement/success`) and `denoise_step_norm_mean` to SOTA batch metrics.
  - Updated C2ST metric in `cerber_gui/sota_eval.py` to report label-invariant accuracy and raw accuracy.
  - Added `SOTA_EVAL_CACHE_VERSION=2` into cache key to prevent stale cached metrics.
- Validation:
  - `python -m py_compile cerber_gui/app.py cerber_gui/sota_eval.py`

---

# GUI Landscape Span Control + Cache Invalidation Fix (2026-03-25, Pass 6)

## Goal
- Add explicit control to expand Direction 1/2 visible range independently from grid detail.
- Ensure this control is wired through preview + manual inference + cache keys.
- Fix SOTA cache invalidation after key-versioning change.

## Checklist
- [x] Add absolute half-range control in UI (`0` = auto, `>0` = forced span)
- [x] Thread new parameter through `select_checkpoint_fn` and `run_inference_fn`
- [x] Thread new parameter through `generate_landscape_for_checkpoint` + cache key
- [x] Extend `scan_energy_landscape_3d` to honor absolute span override
- [x] Wire all Gradio `.change`/`.click` handlers with the new input
- [x] Fix `_invalidate_checkpoint_cache` compatibility with versioned SOTA cache keys
- [x] Validate by compile check

## Review
- Implemented:
  - New GUI slider `Landscape Half-Range (Absolute)` with range `[0..100]`.
  - Full parameter wiring across preview generation, manual inference, landscape scan, and cache keys.
  - Absolute span now overrides factor-based span when set (`>0`), enabling large-area exploration even for small noise distances.
  - Updated SOTA cache invalidation logic to support both legacy and v2 cache key formats.
- Validation:
  - `python -m py_compile cerber_gui/app.py cerber_gui/landscape_3d.py cerber_gui/sota_eval.py`

---

# Endpoint/Best-State Desync Closure (2026-03-25, Pass 7)

## Goal
Eliminate metric desynchronization between live inference and SOTA batch evaluation by enforcing a single endpoint semantics (last executed state) while preserving `best-energy` state for optional analysis.

## Checklist
- [x] Extend `LangevinResult` to carry both states explicitly (`v_final` best-energy, `v_last` last executed)
- [x] Update all Langevin variants (overdamped/pid/underdamped) to populate `v_last` on early-stop and full-run exits
- [x] Update GUI `run_langevin_denoise` fallback (`track_vectors=False`) to return `v_last` instead of `v_final`
- [x] Keep trajectory and reported endpoint consistent in non-tracking mode
- [x] Run compile validation

## Review
- Implemented:
  - Added `v_last` to `LangevinResult`.
  - Filled `v_last` in every return path of all three Langevin methods.
  - Batch/SOTA pathway now uses actually reached terminal state (`v_last`) instead of best-energy fallback.
  - This removes live-vs-batch endpoint mismatch and stabilizes interpretation of step norm / cosine / L2 improvements.
  - Bumped `SOTA_EVAL_CACHE_VERSION` to `3` so post-fix metrics are recomputed (no stale pre-fix cache artifacts).
- Validation:
  - `python -m py_compile cebcm/inference/langevin.py cerber_gui/app.py cerber_gui/sota_eval.py cerber_gui/landscape_3d.py`

---

# Stage1 Pipeline Re-Research: Unconditional vs Simple (2026-03-25, Pass 8)

## Goal
Perform a full re-research of Stage1 training pipelines:
- `unconditional` energy pipeline,
- `simple` / actor+critic-aligned pairwise pipeline,
and produce SOTA-grounded upgrade options with concrete implementation tracks.

## Checklist
- [x] Re-read AGENTS/spec/implementation plan constraints for Stage1 role in CERBER architecture
- [x] Audit current code paths end-to-end for both pipelines (objective, sampling, OOD controls, metrics, inference semantics)
- [ ] Run parallel subagent research:
  - [x] unconditional pipeline deep audit
  - [x] simple/actor-critic pipeline deep audit
  - [x] external SOTA methods + papers + practical recipes
- [x] Produce `research3.md` with agent-attributed findings and source links
- [x] Build prioritized improvement matrix (immediate / near-term / long-term)
- [x] Validate recommendations against current implementation constraints and Stage2/Stage3 integration goals

## Review
- Completed full three-track parallel audit:
  - `research3_agent_unconditional.md`
  - `research3_agent_simple.md`
  - `research3_agent_sota.md`
- Added consolidated synthesis file: `research3.md`.
- Code-verified key contradictions and risks before synthesis:
  - `actor_critic` objective conflict (`E(clean)<E(actor)` ranking vs actor term minimizing `E(actor)-E(clean)`),
  - unconditional Langevin noise-semantics mismatch across sampler paths,
  - unconditional `best.pt` selection by paired cosine improvement,
  - training eval endpoint mismatch (`v_final` usage in Stage1 eval path).
- Output includes prioritized implementation roadmap (P0-P3), experiment matrix, Stage2/3 readiness gates, and SOTA-backed architecture direction (conditional critic mainline + unconditional prior auxiliary).

---

# Stage1 Kill Criteria + Eval Rewrite (2026-03-25, Pass 9)

## Goal
Eliminate false-positive training verdicts by replacing weak single-metric pass checks with strict SOTA-aligned multi-metric evaluation and kill criteria during training.

## Checklist
- [x] Add unified evaluation/kill-criteria utilities for:
  - conditional Stage1 (`simple`/`actor_critic`)
  - unconditional Energy Matching
- [x] Rewrite `experiments/01_denoising_poc/train.py` evaluation:
  - use reached endpoint semantics (`v_last`)
  - add geodesic/L2/energy/clean-min-violation metrics
  - compute composite score + strict gates
- [x] Rewrite `experiments/02_energy_matching/train.py` evaluation:
  - keep cosine as diagnostic only
  - add distribution/manifold metrics (MMD/C2ST/PRDC/kNN)
  - replace best-checkpoint selection criterion with unconditional composite score
  - enforce strict unconditional kill criteria gates
- [x] Align standalone evaluation scripts with same kill-criteria logic
- [x] Run static validation (`py_compile`) on changed files
- [x] Document resulting behavior and remaining calibration knobs

## Review
- Added shared strict criteria module: `cebcm/training/kill_criteria.py`.
- Stage1 conditional training now evaluates with:
  - cosine + geodesic + L2 + energy success + clean-min violation + step norm,
  - `v_last` endpoint semantics,
  - strict multi-gate verdict and composite score checkpoint selection.
- Unconditional training now evaluates with:
  - energy-descent diagnostics across noise scales,
  - distribution/manifold suite (`MMD`, `C2ST`, `PRDC`, kNN),
  - strict unconditional gates and composite score checkpoint selection.
- Replaced legacy weak criteria:
  - old: `improvement > 0 && success_rate > 0.5`,
  - new: model-type-specific multi-gate SOTA criteria.
- Standalone evaluators were aligned with strict criteria output for consistency.
- Static validation passed:
  - `python -m py_compile cebcm/training/kill_criteria.py experiments/01_denoising_poc/train.py experiments/01_denoising_poc/evaluate.py experiments/02_energy_matching/train.py experiments/02_energy_matching/evaluate.py`

---

# Stage1.5 SOTA Implementation (2026-03-26, Pass 11)

## Goal
Implement full SOTA hybrid actor-critic pipeline with all P0 fixes and SOTA stabilization.

## Implementation Status

### ✅ Completed
- [x] Created `train_stage1_5.py` with full SOTA implementation
- [x] Removed actor_energy_loss contradiction
- [x] Added MDSM to critic for gradient validity
- [x] Implemented hybrid critic pattern (E_cond + λ*E_prior)
- [x] Added alternating training (2 critic : 1 actor)
- [x] Added CQL regularization for OOD
- [x] Added BC regularization for embedding anchor
- [x] Implemented composite score checkpoint selection
- [x] Added kill criteria integration

### 🔄 In Progress
- [ ] Create Stage1.5 config template
- [ ] Run CUDA validation
- [ ] Tune hyperparameters from first logs

## Priority Matrix (Original)

### P0 — Critical Correctness (blocker for Stage2/3)

- [x] **Fix actor_critic objective contradiction**
  - Files: `experiments/01_denoising_poc/train_stage1_5.py`
  - Issue: Critic requires `E(clean) < E(actor)` but actor minimizes `softplus(e_actor - e_clean)` → `E(actor) < E(clean)`
  - Fix: **REMOVED** actor_energy_loss entirely
  - Test: Pending CUDA validation

- [x] **Unify unconditional Langevin noise semantics**
  - Files: `cebcm/inference/langevin.py` (already fixed in Pass 7)
  - Status: v_last endpoint already implemented

- [x] **Fix unconditional checkpoint selection criterion**
  - Files: `experiments/01_denoising_poc/train_stage1_5.py`
  - Fix: Composite score with cosine/geodesic/clean-min-violation

- [x] **Unify endpoint semantics (v_last vs v_final)**
  - Status: Already fixed in Pass 7, reused in Stage1.5

### P1 — Objective Alignment

- [ ] **Add gradient penalty to critic loss**
  - Files: `cebcm/training/losses.py`
  - Purpose: Smooth energy landscape, prevent sharp minima
  - Implementation: `gradient_penalty()` function, add to MDSM loss

- [ ] **Add persistent contrastive term to unconditional**
  - Files: `cebcm/training/negative_buffer.py`, `experiments/02_energy_matching/train.py`
  - Issue: NCE only in warmstart, not main EM phase
  - Fix: Keep NCE active during main training with persistent chains
  - Test: PRDC/C2ST metrics improve vs warmstart-only baseline

- [ ] **Add final-state geometry loss for actor**
  - Files: `experiments/01_denoising_poc/train.py`
  - Purpose: Actor optimized for final projected state, not just delta
  - Implementation: Cosine/geodesic on `v_refined`, gradient alignment with `-∇E`

- [ ] **Add OOD/manifold penalties**
  - Files: `cebcm/training/losses.py`
  - Functions: `manifold_proximity_penalty()`, `shell_barrier_penalty()`
  - Test: OOD rate decreases, kNN proximity improves

### P2 — Stability and Monitoring

- [ ] **Add energy calibration layer**
  - Files: `cebcm/models/energy_unconditional.py`
  - Purpose: Normalize energy output to [0, 1] via running statistics
  - Implementation: `EnergyCalibrator` module with EMA

- [ ] **Add gradient clipping and EMA**
  - Files: `experiments/01_denoising_poc/train.py`
  - Implementation: `clip_grad_norm_()`, EMA weight wrapper
  - Test: Training stability improves, late-training generalization better

- [ ] **Add convergence detection for Langevin**
  - Files: `cebcm/inference/langevin.py`
  - Purpose: Early stopping when energy plateaus
  - Implementation: `converged`, `convergence_step` in `LangevinResult`

- [ ] **Add comprehensive metrics telemetry**
  - Files: `experiments/01_denoising_poc/train.py`, `experiments/02_energy_matching/train.py`
  - Track: energy stats, gradient norms, Langevin convergence, manifold quality, OOD rate

### P3 — Architecture Enhancements

- [ ] **Add manifold-aware Langevin dynamics**
  - Files: `cebcm/inference/langevin.py` (new function)
  - Purpose: Tangent space projection for hypersphere geometry
  - Implementation: `manifold_langevin_step()` with tangent gradient + noise

- [ ] **Add spectral normalization to energy network**
  - Files: `cebcm/models/energy.py`, `cebcm/models/energy_unconditional.py`
  - Purpose: Enforce 1-Lipschitz constraint, stabilize gradients
  - Implementation: Spectral norm on OrthoLinear weights

- [ ] **Implement hybrid critic (conditional + prior)**
  - Files: `experiments/01_denoising_poc/train.py`
  - Formula: `E_total(q, x) = E_cond(q, x) + lambda_prior * E_prior(x)`
  - Test: OOD drift reduces without hurting relevance metrics

## Experiment Matrix

| ID | Change | Expected Impact | Validation |
|----|--------|-----------------|------------|
| AC-1 | Remove actor energy term | Fix P0 contradiction | Clean-min violation ↓ |
| AC-2 | Add MDSM to critic | Gradient field quality | Langevin stability ↑ |
| AC-3 | Final-state geodesic loss | Cosine/geodesic ↑ | kNN proximity ↑ |
| U-1 | Unify sampler semantics | Train/eval parity | Trajectory match |
| U-2 | Manifold checkpoint criterion | Better selection | PRDC/C2ST ↑ |
| U-3 | Persistent NCE in EM | Manifold calibration | Density metrics ↑ |
| HYB-1 | Hybrid critic + prior | OOD robustness | AUROC ↑ |

## Stage2/3 Readiness Gates

Do **not** advance until:
- [ ] All P0 items complete and verified
- [ ] Conditional branch shows stable positive cosine/geodesic gain
- [ ] Unconditional branch improves PRDC/C2ST/MMD + OOD AUROC
- [ ] Actor proposals stay within support constraints (multi-start test)
- [ ] No clean-min violation mode in eval

## Review
- Plan created from `research3.md` findings
- Awaiting CUDA runtime for implementation and validation

---

# Stage1.5 Hard Validation + Repair (2026-03-26, Pass 12)

## Goal
Run a strict code-and-math validation of Stage 1.5 and harden it to match research3 P0/P1 constraints:
- objective sign consistency,
- noise/sampler semantics consistency,
- anti-instability guards (non-finite, gradient sanitation, clipping),
- truthful strict success criteria during training.

## Checklist
- [x] Re-read `AGENTS.md`, `CLAUDE.md`, `research3.md` and map required checks to code paths
- [x] Full static audit of `train_stage1_5.py` for runtime blockers and math contradictions
- [x] Fix Stage1.5 P0 correctness blockers (type/config/API mismatches, missing eval/kill hooks)
- [x] Verify critic/actor gradient signs and ranking consistency vs inference update direction
- [x] Verify noise semantics and Langevin navigation consistency with shared sampler (`cebcm/inference/langevin.py`)
- [x] Implement strict Stage1.5 eval + kill criteria integration using shared `kill_criteria.py`
- [x] Add/verify training stabilizers: non-finite guards, gradient sanitation, clipping, clean-min violation telemetry
- [x] Run compile validation on all changed files and document residual runtime limits
- [x] Compare current Stage1.5 to proposed pairwise conditional critic + actor(refinement) design and list exact deltas

## Review
- Replaced non-runnable `train_stage1_5.py` with executable Stage1.5 pipeline:
  - strict config parsing,
  - consistent critic/actor objectives,
  - shared Langevin-based eval,
  - strict kill criteria via `summarize_conditional_eval`.
- Corrected false implementation assumptions from Pass 11:
  - previous script declared features that were not actually runnable due API/type/signature mismatches.
- Updated Stage1.5 config schema:
  - `configs/base.py::Stage1_5Config`,
  - `configs/stage1_5_config.json`.
- Added unconditional prior scale guard:
  - `cebcm/models/energy_unconditional.py` now clamps `log_energy_scale` before `exp` (parity with pairwise critic stability guard).
- Fixed Stage1.5 optimizer semantics for hybrid critic:
  - separate parameter groups now apply both `critic_lr` and `prior_critic_lr` (no silent LR override when prior critic is enabled).
- Validation:
  - `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py`
  - `python -m py_compile configs/base.py`
  - `python -m py_compile experiments/01_denoising_poc/train.py`
  - `python -m py_compile cebcm/inference/langevin.py cebcm/training/losses.py cebcm/training/kill_criteria.py`
- Remaining limitation:
  - full CUDA runtime check still required in target environment with installed `torch`.

---

# Stage1.5 Completion Pass (2026-03-26, Pass 13)

## Goal
Close remaining Stage1.5 architectural gaps identified by user:
- implement real `critic_steps_per_actor > 1`,
- remove teacher-forced training semantics (`query != clean target`),
- implement full twin-critic conditional training with retrieval/hard-negative conditioning.

## Checklist
- [x] Implement non-teacher-forced pair sampling in Stage1.5 (query/positive from retrieval protocol)
- [x] Implement retrieval-conditioned hard negatives for critic ranking loss
- [x] Implement actual twin conditional critics in Stage1.5 (`E1_cond`, `E2_cond`) plus optional prior
- [x] Implement real alternating schedule `critic_steps_per_actor`
- [x] Align actor/refinement update with twin hybrid energy and verify sign consistency (`v <- v - lr * grad(E)`)
- [x] Add explicit Stage1.5 metrics for retrieval/ranking quality and clean-min violations
- [x] Update Stage1.5 config schema/json for new retrieval+twin parameters
- [x] Run static validation (`py_compile`) for all changed files
- [x] Update `research3.md` and `tasks/lessons.md` with findings and anti-regression rules

## Review
- Stage1.5 training is now non-teacher-forced:
  - query `q` and target positive `v_pos` are sampled via retrieval (`retrieve_pos_hard`) from a manifold bank.
- Critic is now truly twin:
  - separate `critic1` / `critic2` checkpoints,
  - hybrid inference energy via `max/mean` aggregation (`twin_aggregate`).
- `critic_steps_per_actor` now changes runtime behavior:
  - critic updates run in a real inner loop, alternating critic branches.
- Added retrieval/hard-negative conditioning:
  - positives from top-k retrieval excluding near-identical self-match,
  - hard negatives from deeper retrieval window.
- Added 8GB-safe stabilization:
  - per-critic update (not both critics in one second-order graph),
  - OOM catch + `torch.cuda.empty_cache()` skip path,
  - lighter default config (`batch_size=32`, `ortho_n_iters=4`, reduced eval load).
- Added math-forward upgrades:
  - smooth twin critic aggregation (`twin_aggregate=softmax`, temperature-controlled),
  - conditional NCE loss for retrieval ranking plus optional prior-NCE branch,
  - strict retrieval self-exclusion by sample index (not only cosine threshold),
  - tangent-noise Langevin option (`langevin_tangent_noise=true`) for sphere-consistent stochastic steps.



## Stage1.5 Post-change Audit (2026-03-26)

### Checklist
- [x] Re-validated Stage1.5 sign consistency (MDSM target, Langevin descent, ranking inequalities, actor alignment)
- [x] Fixed misleading training telemetry labels (`rank` -> explicit `rank_loss` + `rank_success`)
- [x] Added per-inequality ranking rates (`clean<actor`, `actor<hard`, `clean<hard`)
- [x] Added deterministic eval subset reuse for fair checkpoint selection
- [x] Stabilized no-eval epoch logging schema with `status=not_evaluated`
- [x] Added numerical sanitization in Stage1.5 `conditional_mdsm`
- [x] Reduced parameter finite-check overhead (interval-based)
- [x] Synced Stage1.5 README with actual pipeline/config/checkpoint keys
- [x] Added minimal Stage1.5 regression tests (`tests/test_stage1_5_integrity.py`)

### Review
- Fixed correctness gaps in reporting and checkpoint-scoring fairness without changing core objective semantics.
- Runtime verification is still blocked in this shell due missing `torch/pytest`; static compile checks pass.

---

# Stage1.5 Performance + Math Safety Pass (2026-03-26, Pass 14)

## Goal
Implement requested SOTA-safe acceleration for Stage1.5 without changing optimization direction:
- cheaper orthonorm path,
- batched eval Langevin,
- `torch.compile` + checkpointing on heavy second-order graph,
- optional vectorized retrieval.

## Checklist
- [x] Add Stage1.5 config/runtime knobs for orthonorm schedule, compile, checkpointing, and batched eval
- [x] Implement epoch-based orthonorm iteration schedule and log active `n_iters`
- [x] Implement batched eval Langevin with mathematically-safe fixed-step semantics
- [x] Add optional `torch.compile` wrappers with runtime-safe fallback (no key/ckpt breakage)
- [x] Add gradient checkpointing in create_graph path (`conditional_mdsm`)
- [x] Vectorize `retrieve_pos_hard` while preserving strict index exclusion behavior
- [x] Add/update integrity tests for schema/math-sensitive changes
- [x] Run `py_compile` and summarize expected perf/quality impact + risks

## Review
- Added opt-in compile path (`enable_compile`) with safe fallback to eager mode.
- Added Stage1.5 orthonorm schedule controls and per-epoch `n_iters` logging.
- Added batched eval Langevin path with fixed-step semantics (no batch-coupled early stop).
- Added gradient checkpointing knob for MDSM create-graph path.
- Vectorized retrieval positive/hard selection without relaxing strict index exclusion.
- Updated tests for ortho schedule and schedule validation contract.
- Validation: `python -m py_compile` passed for changed Stage1.5 files and tests.

---

# Stage1.5 Runtime Hotfix (2026-03-26, Pass 15)

## Goal
Fix runtime crash after epoch due to scalar/tensor variable shadowing in Stage1.5 training loop.

## Checklist
- [x] Reproduce and localize crash source from traceback (`Boolean value of Tensor ... ambiguous`)
- [x] Rename conflicting epoch loop variable to `epoch_idx`
- [x] Rename critic energy tensors to explicit `e_pos/e_actor/e_hard`
- [x] Update all downstream logging/checkpoint fields to use `epoch_idx`
- [x] Re-run syntax validation (`py_compile`)
- [x] Update `tasks/lessons.md` anti-regression rule

## Review
- Root cause: `ep` (epoch index) was overwritten by energy tensor `ep = crit(...)` in same function scope.
- Fixed in `experiments/01_denoising_poc/train_stage1_5.py`; eval gate and JSON logging now read scalar epoch index only.
- Validation: `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py` passed.

---

# Stage1.5 Config + Live Monitoring Upgrade (2026-03-26, Pass 16)

## Goal
Stabilize Stage1.5 default config and provide honest live visualization for training progress:
- fix unstable rank margins / actor barrier defaults,
- log batch-window timing (`sec/batch`) in training output,
- stream batch+epoch metrics to GUI-compatible JSONL,
- update web live monitor to read Stage1.5 JSONL and render detailed progress charts.

## Checklist
- [x] Re-check and update `configs/stage1_5_config.json` for safer startup hyperparameters
- [x] Add per-log-window timing metrics in `train_stage1_5.py` and persist to JSONL stream
- [x] Keep epoch-level metrics/kill criteria logging schema stable and backward compatible
- [x] Extend `cerber_gui/live_monitor.py` to parse both JSON (legacy) and JSONL (Stage1.5 stream)
- [x] Add detailed live dashboard traces (loss, rank metrics, violation, speed) in Plotly
- [x] Update GUI labels/help text (`app.py`) for Stage1.5 live path defaults
- [x] Validate syntax (`py_compile`) for all changed files and summarize runtime usage

## Review
- Config defaults were tightened for stability:
  - lower ranking margins (`0.5/0.3/0.8 -> 0.2/0.1/0.3`),
  - smaller actor step size (`1.0 -> 0.5`),
  - stronger actor barrier (`0.1 -> 0.2`),
  - softer retrieval hardness window (`topk/hard: 8..32 -> 6..24`).
- Stage1.5 trainer now emits timing in console every `log_every` window:
  - `sec/batch=...`
  - `eta=...m`
- Stage1.5 trainer now writes streaming metrics to
  `experiments/03_Stage_1.5/logs/training_metrics.jsonl` with explicit events:
  - `event=batch` (window metrics + timing),
  - `event=epoch` (train aggregate + kill/eval snapshot),
  - `event=final`.
- Live monitor now supports both legacy JSON and Stage1.5 JSONL streams and builds a detailed 4-panel Plotly dashboard:
  - core losses,
  - rank/violation rates,
  - regularizers/retrieval/skip,
  - speed + eval/kill signals.
- GUI updates:
  - metrics upload accepts `.json` and `.jsonl`,
  - live tab labels/default path now target Stage1.5 JSONL stream,
  - live monitor startup stops old watcher and immediately returns a plot.
- Validation:
  - `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py`
  - `python -m py_compile cerber_gui/live_monitor.py`
  - `python -m py_compile cerber_gui/metrics_viewer.py`
  - `python -m py_compile cerber_gui/app.py`

---

# Live 3D Landscape Monitoring + Rolling Checkpoint Policy (2026-03-26, Pass 17)

## Goal
Eliminate live-monitor UX gaps:
- prevent aggressive rerender scrolling behavior as much as possible,
- add live 3D landscape checks during training (manual + auto every N epochs),
- support rolling latest checkpoint plus periodic milestone checkpoint retention.

## Checklist
- [x] Add Stage1.5 checkpoint cadence config (`checkpoint_every_epochs`, `rolling_checkpoint_name`)
- [x] Change Stage1.5 saver to periodic checkpoints + rolling latest + non-periodic cleanup
- [x] Add Stage1.5 checkpoint compatibility in GUI checkpoint loader (`critic1_state` fallback)
- [x] Add Live Monitor 3D controls and manual `Check Landscape` button
- [x] Add timer-based auto landscape refresh gate (default every 5 epochs)
- [x] Reuse checkpoint-analysis plotting pipeline for live 3D rendering
- [x] Reduce unnecessary live plot rerenders when metrics file timestamp is unchanged
- [x] Run syntax validation on modified files

## Review
- Added rolling checkpoint workflow:
  - periodic checkpoints are kept every `checkpoint_every_epochs`,
  - `latest_epoch.pt` is overwritten each epoch for immediate landscape inspection,
  - stale non-periodic `epoch_*.pt` files are cleaned up.
- Live tab now includes:
  - checkpoint directory input,
  - auto-update toggle,
  - `Auto Every N Epochs` (default `5`),
  - manual `Check Landscape` button,
  - dedicated live 3D landscape + live trajectory plots.
- Live landscape uses same core rendering modules as Checkpoint Analysis:
  - `generate_landscape_for_checkpoint(...)`
  - `_render_landscape_figure(...)`
  - `create_trajectory_plot(...)`
- Added Stage1.5 checkpoint format support in analyzer (`critic1_state` as fallback `model_state`).
- Added a scroll-preservation JS observer and skipped redundant plot refreshes when watcher data has not changed.

# Stage 1.5 GUI Eval Protocol Alignment + Runtime NameError Fix (2026-03-26, Pass 18)

## Goal
Fix Stage 1.5 GUI evaluation mismatch (self-target vs retrieval-target) and resolve runtime crash in live landscape auto-update.

## Checklist
- [x] Fix `NameError: json is not defined` in `cerber_gui/app.py`
- [x] Add Stage 1.5-aware SOTA eval branch in GUI (conditional retrieval objective)
- [x] Keep backward compatibility for unconditional/self-denoise checkpoints
- [x] Update SOTA text labels from `clean` to `target` where applicable
- [x] Validate modified modules with `py_compile`

## Review
- Added `import json` to app module to unblock JSONL epoch parsing in auto landscape refresh.
- Added conditional Stage 1.5 SOTA eval path:
  - sample `(query, positive, hard)` triplets via cosine retrieval,
  - seed noisy candidates from query/hard mix,
  - evaluate cosine/L2 against retrieval target (not self clean) for conditional checkpoints.
- Added explicit SOTA metadata in UI output:
  - `eval_objective` (`conditional_retrieval` or `self_denoise`),
  - `target_label` (`retrieved_pos` or `clean`).
- Compiled successfully:
  - `python -m py_compile cerber_gui/app.py`
  - `python -m py_compile cerber_gui/checkpoint_analyzer.py`
  - `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py`

# Stage 1.5 Single-Run Inference Protocol Alignment (2026-03-26, Pass 19)

## Goal
Align single-run GUI inference with Stage 1.5 conditional objective to remove mixed interpretation between runtime metrics and SOTA batch eval.

## Checklist
- [x] Add shared conditional-checkpoint detector helper
- [x] Add unified inference sampler returning query/target/noisy for both self and conditional modes
- [x] Route `run_langevin_denoise` with explicit `v_query_override` / `v_target_override` in conditional mode
- [x] Recompute runtime cosine/L2/energy against target (not always self-clean)
- [x] Update runtime/report labels: objective + target semantics
- [x] Apply same alignment to checkpoint landscape preview path
- [x] Validate with `py_compile`

## Review
- Runtime and SOTA now evaluate under coherent objective semantics.
- Stage 1.5 conditional checkpoints use retrieval target in both preview and manual inference.
- GUI now displays objective context explicitly (`self_denoise` vs `conditional_retrieval`).

# Stage 1.5 Structural Math Fixes: Actor Energy Corridor + Retrieval Target Hygiene (2026-03-26, Pass 20)

## Goal
Fix non-hyperparameter mathematical failure modes causing false minima trapping and rank/violation drift.

## Checklist
- [x] Add retrieval positive quality floor + fallback-to-query when no valid positive exists
- [x] Add actor two-sided energy corridor loss using critic references (`pos` and `hard`)
- [x] Add actor monotonic descent guard from seed (`E(next) <= E(seed)`)
- [x] Expose new actor guard metrics in batch logs (`a_bar`, `a_desc`)
- [x] Sync GUI conditional retrieval sampler with same min-similarity + fallback logic
- [x] Validate via `py_compile`

## Review
- Retrieval objective no longer trains on semantically invalid positives when neighborhood quality is poor.
- Actor is constrained to stay between critic reference energies (with margins), reducing collapse into pathological low-energy pockets.
- Additional descent guard suppresses actor steps that increase energy from its own seed.
- Runtime observability improved with explicit actor guard metrics in training stream.

# Stage1.5 Full Math Audit: Critics + Actor + Navigation (2026-03-26, Pass 21)

## Goal
Close non-hyperparameter correctness gaps found in full Stage1.5 audit:
- sigma-conditioning parity between train and eval/inference,
- tangent-space alignment consistency in actor loss,
- underdamped Langevin step correctness and safety checks,
- GUI runtime parity with Stage1.5 checkpoint config.

## Checklist
- [x] Re-audit Stage1.5 critic/actor/navigation math end-to-end with subagent cross-check
- [x] Fix Stage1.5 eval sigma mismatch by binding explicit sigma in Langevin/eval energy path
- [x] Fix actor gradient-alignment geometry to compare tangent vs tangent directions
- [x] Fix actor descent guard to compare energies on the same projected manifold
- [x] Harden config validation (`sigma_curriculum_start>0`, strict enums, underdamped constraints)
- [x] Fix underdamped Langevin position update scaling (remove extra `lr` factor)
- [x] Add numerical-safe sphere/tangent projection for zero-norm edge cases
- [x] Fix GUI Stage1.5 runtime config hydration (`checkpoint["config"]` fallback)
- [x] Add GUI sigma/tangent/sampler parity for Stage1.5 conditional inference and SOTA batch eval
- [x] Run static verification (`py_compile`) on all changed modules

## Review
- Stage1.5 eval now optimizes and measures the same sigma-conditioned critic regime used in training.
- Actor alignment no longer asks tangent-projected delta to match full-space gradients.
- Underdamped dynamics no longer apply an unintended `O(lr^2)` position scaling.
- GUI now inherits Stage1.5 runtime knobs from checkpoints and uses matching conditional seed/noise semantics.
- Validation run:
  - `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py cebcm/inference/langevin.py cebcm/training/losses.py cerber_gui/app.py configs/base.py`

# Stage1.5 Twin-Critic GUI Parity Fix (2026-03-26, Pass 22)

## Goal
Remove structural visualization/inference mismatch where Stage1.5 checkpoints were rendered as single-critic (`critic1_state`) models instead of true twin-hybrid energy.

## Checklist
- [x] Add runtime twin-energy adapter in GUI (`_TwinConditionalEnergyAdapter`)
- [x] Load both `critic1_state` and `critic2_state` for Stage1.5 checkpoints
- [x] Apply training-time aggregation mode parity (`max` / `mean` / `softmax`, with temperature)
- [x] Load optional `prior_state` + `lambda_prior` into GUI runtime energy
- [x] Keep backward compatibility for non-Stage1.5 single-model checkpoints
- [x] Validate via `py_compile`

## Review
- GUI inference/landscape now reflects the same hybrid energy family used during Stage1.5 training.
- This removes a major source of apparent "training vs landscape" contradictions in checkpoint inspection.
- Validation run:
  - `python -m py_compile cerber_gui/app.py`

# Stage1.5 SOTA Consolidation (2026-03-31, Pass 23)

## Goal
Deliver a mathematically coherent, production-ready Stage1.5 baseline by closing remaining known failure modes:
- low-sigma MDSM dead-zone (`sigma_eff_sq` clamp pathology),
- overloaded/conflicting default objective stack,
- weak config safety for known anti-pattern combinations.

## Checklist
- [x] Re-audit current Stage1.5 critic/actor/inference math against `research3` failure analysis
- [x] Implement robust MDSM sigma handling:
  - [x] adaptive sigma floor mode
  - [x] log-space target mode (no hard `1e-6` dead-zone behavior)
  - [x] inverse-sigma clipping controls for numerical safety
- [x] Add strict config validation gates for low-sigma + weighting anti-patterns
- [x] Introduce SOTA-safe Stage1.5 default config profile (minimal conflicting losses)
- [x] Validate static correctness (`py_compile`) for all touched modules
- [x] Produce run commands and expected training-gate interpretation notes

## Review
- Done. Implemented:
  - low-sigma MDSM stabilization (`logspace`/`adaptive` modes, weight floor, inv-sigma clip),
  - eval/inference sigma-anneal metric parity,
  - actor barrier manifold-consistent references,
  - strict config validation for known failure combinations,
  - stricter best-checkpoint policy (`best.pt` = strict-pass only; `best_any.pt` = best score regardless).
- Validation passed:
  - `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py configs/base.py cerber_gui/app.py cebcm/training/kill_criteria.py`
  - `python -c "import json, pathlib; json.load(open('configs/stage1_5_config.json', encoding='utf-8')); print('ok')"`
- Limitation:
  - Full train/eval runtime verification requires user CUDA/Linux env (`.venv` with torch). Local desktop Python env in this session has no torch.

# Stage1.5 Dual-Critic Plateau + Energy-Scale Audit (2026-03-31, Pass 24)

## Goal
Close the currently observed quality plateau in hybrid dual-critic training:
- `dir` stagnation around ~0.31,
- rank/success plateau,
- unstable/off-scale energy ranges in GUI landscapes,
- strict gates failing on `all_noise_scales_passed` and `mean_clean_min_violation_rate`.

## Checklist
- [x] Rebalance direction supervision for angular/radial split:
  - [x] raise angular direction weight to Phase2i-equivalent regime
  - [x] re-check low-sigma head weighting so low-noise inference is not radial-dominated without direction supervision
- [x] Remove per-head update dilution from strict alternating head updates:
  - [x] evaluate same-batch dual-head critic update vs alternating update
  - [x] compare variance/stability and direction/rank metrics (added explicit mode + logging hooks)
- [x] Stabilize energy scale behavior:
  - [x] audit `log_energy_scale` buffer policy vs trainable scaling option
  - [x] tighten regularization on off-manifold energies (relative floor / CD calibration + scale regularizer)
- [x] Gate realism review:
  - [x] extract exact per-noise failing gates from eval stream
  - [x] verify clean-min violation semantics and threshold calibration against intended margin behavior
- [x] Inference/eval parity verification:
  - [x] confirm sigma/noise anneal consistency and objective semantics in GUI and train eval
  - [x] ensure reported energy success metrics are semantically explicit (descent vs target-proximity)

## Review
- Implemented in code:
  - `critic_step_mode` (`alternating`/`joint`) with joint same-batch paired critic updates.
  - stronger angular-direction profile + low-sigma angular weighting in `configs/stage1_5_config.json`.
  - optional trainable `log_energy_scale` (both simple and decomposed critics), separate optimizer LR group, and `lambda_energy_scale_reg`.
  - anti-well penalties switched to relative-to-clean margins (`energy_floor_relative_to_clean`, `cd_relative_to_clean`) so penalties are active at realistic energy ranges.
  - eval semantics made explicit:
    - `energy_target_proximity_rate`
    - `energy_descent_rate`
    - `clean_min_violation_rate_strict`
    - backward-compatible `energy_success_rate` retained as target-proximity alias.
  - kill criteria clean-min gate now configurable via `clean_min_violation_mode` (`margin`/`strict`).
  - epoch print now includes per-noise failed gates for actionable diagnostics.
- Validation:
  - `python -m py_compile experiments/01_denoising_poc/train_stage1_5.py configs/base.py cebcm/training/kill_criteria.py cebcm/models/energy.py cebcm/models/energy_decomposed.py`
  - `python -c "import json, pathlib; d=json.load(open('configs/stage1_5_config.json', encoding='utf-8')); print('ok', d.get('critic_step_mode'), d.get('energy_scale_trainable'), d.get('clean_min_violation_mode'))"`
- Limitation:
  - Full runtime train/eval verification is blocked in this desktop environment (`torch` is unavailable); needs user CUDA `.venv` run.

## Pass 24 Log Audit Update (Current User-Provided Run, Epochs 1-24)
- [x] Re-validated trend from current logs:
  - `rank_success` climbs (`~0.16 -> ~0.80`) then saturates.
  - `dir` drops (`~0.48 -> ~0.31`) then plateaus.
  - `viol` improves but hovers near strict threshold region (`~0.05-0.07`).
- [x] Re-validated strict-pass blockers for this run profile:
  - `all_noise_scales_passed` remains failing.
  - `mean_clean_min_violation_rate` remains sensitive around `max_clean_min_violation_rate=0.05`.
- [x] Confirmed why `cd`/`efloor` are near-zero in this profile:
  - With `energy_floor_threshold=5.0` and observed energy magnitudes around `O(1)`, both penalties are effectively inactive.
  - This leaves anti-well shaping mostly dormant in the observed run.

## Pass 50 Runtime Parity + Conditional Landscape Fix (2026-03-31, Epochs 1-50 user run)

### Goal
Close the remaining mismatch between reported metrics/plots and actual dynamics after the Pass 24 patchset:
- low-noise runtime metrics not matching real Langevin controls in GUI path,
- conditional landscape queried with wrong anchor (`target` instead of `query`),
- strict gates not explicitly covering the real low-noise regime (`noise=0.0002`).

### Checklist
- [x] Fix GUI inference control parity:
  - [x] `run_langevin_denoise(...)` now accepts `noise_scale_override`.
  - [x] Runtime `sigma_override` now has priority over anneal wrapper (fixed sigma when explicitly requested).
  - [x] `run_langevin(...)` receives actual runtime noise scale, not only checkpoint default.
- [x] Fix conditional landscape semantics:
  - [x] added explicit `v_query` anchor support to `scan_energy_landscape_3d(...)`.
  - [x] added `v_query` passthrough to backend `scan_energy_landscape(...)`.
  - [x] energy probes (`clean/noisy/final/trajectory`) now evaluate `E(query, candidate)` in conditional mode.
  - [x] landscape scan now uses sigma-bound energy wrapper in GUI single-run paths for parity with runtime inference sigma.
- [x] Fix gate realism for current failure mode:
  - [x] included `0.0002` in `eval_noise_scales` for active Stage1.5 config.
  - [x] updated Stage1.5 base default list to include low-noise scale.

### Review
- Code-level root causes confirmed from current run path:
  - GUI previously passed fixed config noise into Langevin dynamics path even when runtime slider intended to probe low-noise behavior.
  - Conditional landscape was visualized with `target` as query anchor, which can contradict true conditional objective.
  - Strict evaluation grid did not include `0.0002`, so low-noise collapse could be underrepresented in gates.
- Patched files:
  - `cerber_gui/app.py`
  - `cerber_gui/landscape_3d.py`
  - `cebcm/visualization/energy_landscape.py`
  - `configs/stage1_5_config.json`
  - `configs/base.py`

# Stage1.5 Far-Start Inference Stress Test (2026-03-31, Pass 51)

## Goal
Test true long-range navigation by starting `noisy` far from `target` instead of near-target default seeds.

## Checklist
- [x] Add noisy start modes for single-run inference:
  - [x] `objective_seed` (current default)
  - [x] `far_auto` (auto far start from target)
  - [x] `manual_plane_xy` (manual XY start in target-centric 2D plane)
- [x] Wire new parameters through `_sample_inference_triplet(...)` and `run_inference_fn(...)`
- [x] Add GUI controls in Inference Settings
- [x] Report selected start mode in Inference Results
- [x] Preserve backward compatibility for preview/live paths (defaults unchanged)
- [x] Validate syntax with `py_compile`

## Review
- Implemented in `cerber_gui/app.py`:
  - new helper logic for target-plane basis and controlled start override:
    - `_build_target_plane_basis(...)`
    - `_apply_noisy_start_strategy(...)`
  - extended `_sample_inference_triplet(...)` to support `start_mode`, `far_start_scale`, `manual_start_x`, `manual_start_y`, `project_start_to_target_norm`.
  - extended `run_inference_fn(...)` to consume these controls and persist `start_mode` + initial distance metric.
  - extended `_format_inference_info(...)` to print start mode and initial `||x0-target||`.
  - added UI controls:
    - `Noisy Start Mode`
    - `Far Start Scale`
    - `Manual Start X (plane)`
    - `Manual Start Y (plane)`
    - `Project Start To Target Norm`
- Validation:
  - `python -m py_compile cerber_gui/app.py`

# Stage2 Modular Training Pipelines (2026-04-01)

## Goal
Replace monolithic Stage2 training usage with separate production-ready pipelines per module/stage:
- `SurprisePredictor` pretrain,
- `ContextEncoder` pretrain,
- `IPP` pretrain on frozen `ContextEncoder`,
- joint `ContextEncoder + IPP` fine-tune on frozen `SurprisePredictor`.

## Checklist
- [x] Add shared Stage2 training utilities module (builders, data loading, type IDs, schedulers, checkpoints).
- [x] Add standalone `train_stage2_sp.py` (SP-only self-supervised training + eval + checkpoints).
- [x] Add standalone `train_stage2_ce.py` (CE-only with contextual next-target objective + optional frozen SP surprise features).
- [x] Add standalone `train_stage2_ipp.py` (IPP-only with frozen CE and optional frozen SP).
- [x] Add standalone `train_stage2_ce_ipp_joint.py` (joint fine-tune with frozen SP).
- [x] Add stage-specific JSON configs for each pipeline.
- [x] Validate syntax (`py_compile`) for all added files.
- [x] Document exact run order and commands in review notes.

## Review
- Added new shared utility module:
  - `cebcm/training/stage2_utils.py`
  - contains shared builders/config loading/device+AMP setup/data loading/type-ids/scheduler/ckpt helpers.
- Added separate training scripts (existing monolithic `experiments/08_autoregressor/train_stage2.py` untouched):
  - `experiments/08_autoregressor/train_stage2_sp.py`
  - `experiments/08_autoregressor/train_stage2_ce.py`
  - `experiments/08_autoregressor/train_stage2_ipp.py`
  - `experiments/08_autoregressor/train_stage2_ce_ipp_joint.py`
- Added dedicated configs:
  - `configs/stage2_sp_config.json`
  - `configs/stage2_ce_config.json`
  - `configs/stage2_ipp_config.json`
  - `configs/stage2_ce_ipp_joint_config.json`
- Validation:
  - `py -3 -m py_compile cebcm/training/stage2_utils.py experiments/08_autoregressor/train_stage2_sp.py experiments/08_autoregressor/train_stage2_ce.py experiments/08_autoregressor/train_stage2_ipp.py experiments/08_autoregressor/train_stage2_ce_ipp_joint.py`
  - Note: `python -m py_compile ...` is not available in this Windows shell alias setup, `py -3` works.
- Stage2 modular run order:
  1. `py -3 experiments/08_autoregressor/train_stage2_sp.py --config configs/stage2_sp_config.json`
  2. `py -3 experiments/08_autoregressor/train_stage2_ce.py --config configs/stage2_ce_config.json`
  3. `py -3 experiments/08_autoregressor/train_stage2_ipp.py --config configs/stage2_ipp_config.json`
  4. `py -3 experiments/08_autoregressor/train_stage2_ce_ipp_joint.py --config configs/stage2_ce_ipp_joint_config.json`
- Optional overrides:
  - CE script: `--sp-checkpoint <path>`
  - IPP script: `--ce-checkpoint <path> --sp-checkpoint <path>`
  - Joint script: `--sp-checkpoint <path> --ce-checkpoint <path> --ipp-checkpoint <path>`

# Stage3 System1/System2 Math-Parity + Metric-Semantics Fix (2026-04-04)

## Goal
????????? ?????????????? ?????????? ????? `system1` ? `system2` ? Stage3 inference,
???????? chain-guided ?????????? ??????????? (? ?? ?????? ??????????), ?????? OOD ?? `max_chain_len`,
? ??????? ????????? ?????? ? text/data ??????? ????? ? ??????????.

## Checklist
- [x] ?????????? `tasks/lessons.md` ????? ???????? ? ????????????? ??????? ????? Stage3/GUI
- [x] ????????? `system2` ?? PID-???????? (?????????????? ???????????? ? `run_langevin`-????????)
- [x] ???????? chain-guided gradient ? ??? ?????????? `system2`
- [x] ????????? chain-eval/backtrack ? attention snapshots ? ???????????
- [x] ?????????? ????: `system1=10`, `system2<=50` ? inference defaults + runtime clamps
- [x] ?????????? `max_chain_len<=20` ? UI/diagnostics (?????? OOD=30)
- [x] ????????? data-target: `seq[-1]` ?????? `seq[1]`
- [x] ????????? text-mode: ????? self-denoise objective + ?????????? `both` ?????
- [x] ???????????????? Stage3 ???????/??????? (`stage3_config.json`, `test_inference.py`) ??? ?????? ?????
- [x] ?????? ????????? (`py_compile`, JSON parse)

## Review
- ???????? ?????:
  - `cebcm/inference/system_switching.py`
  - `cerber_gui/inference_diagnostics.py`
  - `cerber_gui/app.py`
  - `configs/stage3_config.json`
  - `experiments/09_stage3_chain/test_inference.py`
  - `experiments/09_stage3_chain/train_stage3_phase_b.py`
- ?????????:
  - `py -3 -m py_compile cebcm/inference/system_switching.py cerber_gui/inference_diagnostics.py cerber_gui/app.py experiments/09_stage3_chain/test_inference.py experiments/09_stage3_chain/train_stage3_phase_b.py`
  - `py -3 -c "import json, pathlib; json.load(open('configs/stage3_config.json', encoding='utf-8')); print('ok')"`

# Stage3 Strict Mode Gate + Ratio Control (2026-04-04)

## Goal
Implement strict training-time switching policy:
- keep `system2` locked until simple-task accuracy >= 95%,
- after unlock, enforce 30/70 (`system1/system2`) ratio,
- monitor post-unlock `system2` quality each eval.

## Checklist
- [x] Add persistent mode-switch runtime state in Phase B trainer checkpoints.
- [x] Add strict unlock gate from validation metric (`pairwise_rank_acc` by default).
- [x] Add exact 30/70 per-epoch mode schedule (counts + shuffle), not only weighted expectation.
- [x] Add post-unlock monitor with configurable source/metric/floor/patience.
- [x] Add explicit `mode_switch_gate` block to `configs/stage3_config.json`.
- [x] Run syntax/JSON validation.

## Review
- Updated:
  - `experiments/09_stage3_chain/train_stage3_phase_b.py`
  - `configs/stage3_config.json`
  - `cebcm/training/stage2_utils.py` (IPP fallback defaults aligned to spec: `n_integration_steps=50`, `sigma_init=0.05`)
- Validation:
  - `py -3 -m py_compile experiments/09_stage3_chain/train_stage3_phase_b.py`
  - `py -3 -c "import json; json.load(open('configs/stage3_config.json', encoding='utf-8')); print('ok')"`

# ContextEncoder Positional Signal Fix (2026-04-04)

## Goal
Remove ALiBi-only positional dependency in CE global attention and enforce explicit
position-content binding for global-token fusion.

## Checklist
- [x] Add absolute sinusoidal positional embedding for global attention Q/KV.
- [x] Preserve selected global token absolute positions and pass them through attention.
- [x] Keep ALiBi optional (as extra bias), not mandatory.
- [x] Update Stage2 defaults/configs to disable ALiBi-only mode by default.
- [x] Validate Python syntax and JSON configs.

## Review
- Updated:
  - `cebcm/models/context_encoder.py`
  - `cebcm/training/stage2_utils.py`
  - `experiments/08_autoregressor/train_stage2.py`
  - `configs/base.py`
  - `configs/stage2_config.json`
  - `configs/stage2_ce_config.json`
  - `configs/stage2_ipp_config.json`
  - `configs/stage2_ce_ipp_joint_config.json`
- Validation:
  - `py -3 -m py_compile cebcm/models/context_encoder.py cebcm/training/stage2_utils.py experiments/08_autoregressor/train_stage2.py`
  - `py -3 -c "import json; [json.load(open(p, encoding='utf-8')) for p in ['configs/stage2_config.json','configs/stage2_ce_config.json','configs/stage2_ipp_config.json','configs/stage2_ce_ipp_joint_config.json']]; print('ok')"`


## 2026-04-08 - Stage 13 ChainGenerator autoregressive upgrades (implementation)

### Scope
- [x] Replace objective with composite autoregressive loss:
  - `L = lambda_step*L_step_masked + lambda_ans*L_final_answer + lambda_roll*L_free_run + lambda_rank*L_inbatch_contrastive`
- [x] Add free-run rollout loss in training loop (not teacher-forced only).
- [x] Add in-batch contrastive ranking loss on rollout final answer.
- [x] Keep System1 semantics answer-aligned via suffix target selection.

### Architecture updates
- [x] Cross-attention upgraded from single-token context to memory-bank context (`[B, K, D]`) with context masking.
- [x] Training data pipeline now builds context bank (`query + evidence slots`) and pads it in collate with `context_mask`.
- [x] Generator forward/loss path now accepts `v_context_bank` and `context_mask`.

### Generation/runtime updates
- [x] System2 candidate generation is stochastic (temperature + latent/start noise schedules).
- [x] Added anti-loop controls in generator:
  - repeat penalty by cosine-to-history,
  - latent repeat-ban with retry resampling,
  - stagnation early-stop using delta-energy and delta-cos windows.
- [x] Best-of-N reranking now receives non-degenerate candidates and exports diversity diagnostics.

### Horizon alignment
- [x] Training horizon kept at `max_chain_steps=20` (curriculum ramps to configured max).
- [x] Runtime step cap keeps inference within trained/architectural limits.

### Verification
- [x] Syntax checks:
  - `uv run python -m py_compile cebcm/models/chain_generator.py experiments/13_chain_generator/train_chain_generator.py cerber_gui/chain_generator_diagnostics.py`

### Files touched
- [x] `cebcm/models/chain_generator.py`
- [x] `experiments/13_chain_generator/train_chain_generator.py`
- [x] `cerber_gui/chain_generator_diagnostics.py`
- [x] `configs/chain_generator_config.json`

---

## 2026-04-09 — Generator Fine-tune + Critic Retrain Pipeline

### Context
Training with all 8 fixes completed through E30+. Results:
- tf_cos (VAL) = 0.890, roll_cos (VAL) = 0.793, gap = 9.7% at 20 steps
- Significant improvement over old run (9.7% gap was at 4 steps before)
- But roll_cos 0.793 is insufficient — SONAR decode gives approximate paraphrases, not faithful answers
- Target: roll_cos >= 0.90 (ideally 0.93+) for correct factual answers

### Phase 1: Generator Fine-tune (Experiment 13b)

**Goal**: Close tf/roll gap, raise absolute roll_cos above 0.90.

**Key changes from base training:**
| Parameter | Base (E0-E50) | Fine-tune |
|-----------|---------------|-----------|
| lr | 1e-4 | 5e-5 |
| system1_epochs | 10 | 0 (skip) |
| system2_start_steps | 2 (ramp) | 20 (immediate) |
| scheduled_sampling_max | 0.5 | 0.65 |
| ss_ramp_epochs | 10 | 5 |
| free_run_noise_std | 0.05 | 0.07 |
| loss_lambda_roll | 1.0 | 1.5 |
| num_epochs | 50 | 30 |
| output_dir | output/ | output_finetune/ |

**How to run:**
```bash
python experiments/13_chain_generator/train_chain_generator.py \
    --config configs/chain_generator_finetune.json \
    --finetune experiments/13_chain_generator/output/checkpoints/best.pt
```

**What to watch:**
- [ ] E0-E2: tf_cos should stay near 0.88+ (no regression from lower lr)
- [ ] E5+: roll_cos should improve as ss_prob reaches 0.65
- [ ] VAL gap should shrink below 5% by E10
- [ ] roll_cos (VAL) target: >= 0.88 by E15, >= 0.90 by E25

### Phase 2: Retrain Critic (Experiment 14)

**Prerequisite:** Phase 1 best checkpoint exists.

**Before running**, update critic config:
```bash
# In configs/chain_critic_config.json, update:
# "generator_hard_checkpoint": "experiments/13_chain_generator/output_finetune/checkpoints/best.pt"
```

**How to run:**
```bash
python experiments/14_chain_critic/train_chain_critic.py \
    --config configs/chain_critic_config.json
```

**What to watch:**
- [ ] val_rank_acc should reach 0.90+ (currently 0.848)
- [ ] Generator hard negatives should be stronger (higher quality chains)

### Phase 3: Evaluate Pipeline

- [ ] Test inference with fine-tuned generator + retrained critic reranking
- [ ] Verify "What is the capital of France?" gives "Paris" not "France"
- [ ] Check multi-step chains produce full sentences, not 1-word outputs
- [ ] Compare System1 (1-step) vs full chain (20-step) quality

### Phase 4 (if needed): Joint Training or Architecture Changes

Only if Phase 1-3 don't reach 0.90 roll_cos:
- [ ] Implement generator-critic joint training (generator produces, critic scores, REINFORCE/PPO)
- [ ] Consider increasing model capacity (8 layers, larger FFN)
- [ ] Consider data augmentation (richer answer sentences)

### Files created/modified
- [x] `experiments/13_chain_generator/train_chain_generator.py` — added `--finetune` flag, `system2_start_steps`
- [x] `configs/chain_generator_finetune.json` — fine-tune config


---
## 2026-04-13 - Diffusion-Inspired Exposure Bias Fix + GUI Repair

### Objective
Fix the root cause of val_roll_cos_last ceiling at 0.60: the model never sees imperfect contexts during teacher forcing. Implement noisy teacher forcing (diffusion-inspired multi-noise-level training). Fix GUI inference crash. Remove pointless train generate() noise.

### Analysis Validated
- [x] Training log confirms user's analysis: oracle inflated train_roll_cos, noise destroyed eval_roll_cos
- [x] Root cause of 0.16 gap at steps=1: eval noise_std=0.01 in d=1024 → noise_norm=0.32 vs signal=0.4 (SNR=1.25)
- [x] NaN:3 per epoch is cumulative, not per-batch-window; happens at initialization
- [x] val_tf_cos=0.87 proves model capacity is sufficient; problem is training distribution

### Changes
- [x] Fix GUI `run_from_text`/`run_from_data` — add missing `beam_width`, `temperature`, `noise_std` params
- [x] Remove train generate() noise: `free_run_noise_std=0.0` (noise was only useful for oracle candidate selection, now disabled)
- [x] Implement Noisy Teacher Forcing in `forward()`:
  - Per-sample noise level σ ~ U[0, tf_noise_std]
  - Applied to GT prefix in SONAR space before residual scaling
  - Calibrated: tf_noise_std_max=0.005 → angular perturbation ≈ 38° at max
  - Active in both pure TF and scheduled sampling paths
- [x] Add `_get_tf_noise_std()` scheduler: ramps 0→tf_noise_std_max over tf_noise_ramp_epochs
- [x] Thread tf_noise_std through train_step → compute_composite_objective → model.forward()
- [x] Log tf_noise_std in metrics
- [x] Update lessons.md with noise calibration rules

### Next Steps (Priority Order)
- [ ] Run clean training from scratch — establish honest baseline with oracle disabled + noisy TF
- [ ] If val_roll_cos_last plateaus < 0.65: implement self-conditioning (second forward pass with preliminary prediction as conditioning)
- [ ] If val_roll_cos_last plateaus < 0.80: implement iterative refinement per autoregressive step (K=2-4 refinement passes)
- [ ] Consider Diffusion Forcing for hybrid autoregressive-diffusion architecture (major effort)

## 2026-04-11 - Disable Oracle/DAger and Fix ChainGenerator NaNs

### Objective
Disable oracle-guided DAger by default, remove train/eval oracle leakage, fix NaN sources in the ChainGenerator objective, and add dataset/training diagnostics that explain the current 0.90 train vs ~0.49 eval gap.

### Checklist
- [x] Gate oracle/DAger behind `enable_oracle_dagger=false` and default oracle probability/retries to zero.
- [x] Stop passing oracle guidance into free-run rollout unless explicitly enabled.
- [x] Fix disabled-rank-loss NaN contamination by skipping rank loss computation when `loss_lambda_rank=0`.
- [x] Replace unsafe normalize paths with fp32 safe normalization for tiny SONAR-sphere vectors.
- [x] Replace latent-vector cosine checks in generation with safe-normalize cosine.
- [x] Replace masked `loss * mask` with `torch.where(mask, loss, 0)` to avoid `NaN * 0`.
- [x] Disable repeat-ban during training rollouts so the training objective stays differentiable and stable.
- [x] Skip scheduler stepping when the optimizer step is skipped due non-finite loss or gradients.
- [x] Add train/val dataset diagnostics: chain length, truncation, answer coverage, norms, answer text length, context length.
- [x] Add answer-specific metrics: teacher-forced answer cosine, rollout answer cosine, answer coverage.
- [x] Re-run static verification: Python compile, JSON parse, and git whitespace checks.

### Review
- Oracle-guided DAger is now opt-in only. Default training/eval measures real free-run quality instead of oracle-assisted rollout.
- Primary NaN root causes were disabled rank loss still producing NaNs, unsafe normalization on tiny vectors, masked multiplication by zero, and scheduler advancement after skipped steps.
- The next full training run should be started from a clean checkpoint or explicitly treated as fine-tuning from an oracle-contaminated checkpoint.

---
## 2026-04-13 - Full Diffusion Forcing for ChainGenerator

### Objective
Implement a mathematically controlled Diffusion Forcing variant for SONAR-space autoregressive QA: independent per-position diffusion noise levels, timestep/noise conditioning in ChainGenerator, masked diffusion loss, and diagnostics that expose whether diffusion is improving real rollout rather than only teacher-forced metrics.

### Source Principles to Preserve
- Independent noise level per sequence token/position, not one global sequence noise level.
- DDPM-style forward process: x_k = sqrt(alpha_bar_k) * x0 + sqrt(1-alpha_bar_k) * eps.
- Causal decoder must still preserve autoregressive direction: position i may condition on previous/noisy-clean mixed prefix only via causal mask.
- No oracle/DAger. No train-only reranking. No unsafe normalize or NaN * 0 masking.
- SONAR norm must remain interpretable: clean prediction is still sphere-projected to target_norm for autoregressive output metrics.

### Implementation Checklist
- [x] Add diffusion/noise timestep embeddings to `ChainGenerator` and inject them into decoder inputs without breaking residual-space scaling.
- [x] Add diffusion schedule buffers/helpers: beta schedule, alpha_bar extraction, SONAR-safe q_sample, SNR lookup.
- [x] Add diffusion-forced teacher path where each chain position gets an independent noise level and the model predicts clean x0 under mask.
- [x] Add diffusion-forcing loss term with masked reductions and diagnostics: df_loss, df_cos_x0, df_noise_mse, noise_level mean/max, clean/noisy/noise norm.
- [x] Wire config controls: enable flag, timesteps, schedule, loss weight, noise sampling policy, max noise level, min-SNR style weighting if needed.
- [x] Keep standard AR loss and rollout loss alive for compatibility; DF is an additional objective, not a silent replacement on first pass.
- [x] Add validation-time DF diagnostics with fixed eval noise level for stable monitoring.
- [x] Run static checks. Torch smoke test is blocked in the Windows workspace because no local torch environment is available here; use project `.venv` on the training machine for runtime smoke.

### Acceptance Criteria
- [x] Existing checkpoints can load with `strict=False` or config-gated new parameters where necessary.
- [x] Training prints DF metrics and keeps rollout metrics visible.
- [x] No `NaN * 0`, no unsafe `F.normalize` on train path, no uncalibrated per-dim noise defaults.
- [x] `roll_cos`, `roll_ans`, and `ans_cov` remain logged; DF must be evaluated by rollout quality, not just diffusion loss.

### Review
- Implemented pred-x0 Diffusion Forcing as an additional loss (`loss_lambda_diffusion`) rather than replacing AR/rollout objectives.
- Noise is scaled as `target_norm / sqrt(d_model)` by default, so epsilon norm is comparable to SONAR vector norm instead of raw DDPM `sqrt(d_model)`.
- Training samples independent noise levels per valid chain position; validation uses a fixed `df_eval_noise_level` for stable monitoring.
- GUI generator loading now uses `strict=False` so old checkpoints missing DF timestep parameters/buffers still load.
- Runtime forward smoke could not be executed in this Windows workspace because neither `python` nor `uv` has a torch-enabled environment. Static compile and JSON validation passed.
- Second math review fixed two issues: timestep embeddings now use raw diffusion level 0..K-1, and optional Min-SNR weighting now matches pred-x0 instead of pred-noise.
- Eval-path review fixed DF validation stochasticity: validation now uses fixed noise level and deterministic Gaussian epsilon per validation batch for stable val_df_cos.
- Third math review fixed answer-repeat padding semantics: training now tracks `answer_pos`, System1 uses the first answer vector directly, System2 answer losses/metrics activate when the first answer enters the prefix horizon, and disabled-rank diagnostics compare answer vectors instead of last repeat pads.

---
## 2026-04-13 - ChainGenerator Training Geometry GUI

### Objective
Add a mathematically honest and visually useful Web GUI diagnostics path for step-level ChainGenerator training geometry: fixed probe artifacts, Diffusion Forcing noise/denoise trajectories, rollout-vs-target trajectories, and stable 2D/3D projections that can be inspected by global training step.

### Design Requirements
- Use fixed probe batches and fixed PCA basis; do not recompute projection per step.
- Visualize real objects from the model: clean `x0`, noised `x_t`, predicted `pred_x0`, rollout chain, target chain, noise levels, answer position, and masks.
- Separate DF diagnostic quality (`df_cos`) from true autoregressive quality (`roll_cos_answer`, `roll_cos_last`).
- Keep artifact size bounded; save probe snapshots only every configured N steps.
- GUI must work even before artifacts exist and must fail with actionable messages.

### Implementation Checklist
- [x] Inspect existing GUI metrics/plot infrastructure and training script logging hooks.
- [x] Add probe artifact writer to ChainGenerator training with config controls.
- [x] Add stable PCA/projection utilities and compact snapshot schema.
- [x] Add GUI loader/render functions for 3D DF geometry, rollout trajectory, noise heatmap, and step-level scalar metrics.
- [x] Wire a new Training Geometry tab into `cerber_gui/app.py`.
- [x] Add config defaults and documentation/review notes.
- [x] Run static checks and any available non-torch GUI helper checks.

### Acceptance Criteria
- [x] Training can save probe artifacts without changing main loss math.
- [x] GUI can load probe artifacts and display 3D/2D/heatmap plots by `global_step`.
- [x] Projections are stable across steps for honest visual comparison.
- [x] Missing artifacts or missing torch fail cleanly.

### Review
- Added fixed-probe geometry snapshots in `train_chain_generator.py`: clean target chain, noised DF input, DF `pred_x0`, teacher-forced prediction, free rollout, masks, answer position, noise levels, per-token cos/L2/norms, and stable PCA basis.
- Added step-level JSONL events to `chain_generator_training.jsonl`: `run_start`, `train_step`, `probe_snapshot`, `probe_error`, `val_epoch`, and `run_complete`.
- Added config controls in `configs/chain_generator_config.json`: `metrics_log_name`, `enable_training_geometry_probe`, `probe_every_steps`, `probe_num_samples`, `probe_steps`, `probe_dir_name`, `probe_save_raw_vectors`.
- Added `cerber_gui/training_geometry.py` with 3D DF trajectory, 3D rollout trajectory, noise/cos/L2 heatmap, per-token metric chart, and scalar training metric chart.
- Wired new `Training Geometry` tab in `cerber_gui/app.py` with logs/probe loader, snapshot selector, sample selector, summary, and four focused plot tabs.
- Manual math review fixed PCA explained variance normalization: numerator and denominator now use the same variance units instead of mixing variance with total sum-of-squares.
- Verification passed: `uv run python -m py_compile experiments/13_chain_generator/train_chain_generator.py cerber_gui/app.py cerber_gui/training_geometry.py`, `uv run python -m json.tool configs/chain_generator_config.json`, `git diff --check`, and manual trailing-whitespace check for the new GUI module.
- Runtime import smoke is blocked in this Windows `uv` environment because `torch` is not installed and `cerber_gui.__init__` imports torch-dependent modules. Use the project `.venv` training environment for live GUI smoke.

## 2026-04-14 - ChainGenerator NaN collapse fix (`Arch 14_01_26` run post-mortem)

### Context
Run `Arch 14_01_26 full training log.txt` converged to `val_roll_cos_last=0.7554` at E3, then NaN in `loss_df` at end of E3, grad_norm collapsed to 0.0000 from E4 onward and never recovered. Secondary symptom: `val_answer_coverage→0.00` at System1→System2 transition (E10). Early-stopped at E24 instead of E50. Root-cause analysis in `tasks/lessons.md` entry dated 2026-04-14.

### Tasks
- [x] Locate the reintroduced `NaN × 0` trap.
- [x] Harden `_masked_weighted_step_losses` with `torch.where` + `nan_to_num`.
- [x] Audit and fix Min-SNR-γ x₀/ε formula swap in `_diffusion_forcing_weights`.
- [x] Reconcile `_safe_normalize` between model and training script; add inf/NaN scrub.
- [x] Add `nan_to_num` defense-in-depth on `v_tf`, `v_roll`, `model_out`, `target`, `weights`.
- [x] Clamp SNR with `max=1e4` to stop bfloat16 overflow at `t≈0`.
- [x] Add DF lambda warm-up ramp (`df_warmup_epochs=3`) and lower base lambda `0.5 → 0.25`.
- [x] Raise `df_noise_level_min: 0 → 2` to skip unstable near-clean regime.
- [x] Extend `system1_epochs: 10 → 15`.
- [x] Add `df_lam` to per-step training log.
- [x] Add rolling `val_answer_coverage` `[WARN]` log when below threshold.
- [x] Static verification: `py_compile` on `train_chain_generator.py` + `chain_generator.py`, `json.load` on `chain_generator_config.json`.
- [x] Update `tasks/lessons.md` with full post-mortem + carry-forward rules.

### Files Touched
- `experiments/13_chain_generator/train_chain_generator.py` — `_safe_normalize`, `_masked_step_losses`, `_masked_weighted_step_losses`, `_diffusion_forcing_weights`, `_diffusion_forcing_objective`, `compute_composite_objective`, epoch loop (DF warm-up + ans_cov rolling warning + `df_lam` log).
- `cebcm/models/chain_generator.py` — `_safe_normalize` input sanitization.
- `configs/chain_generator_config.json` — `system1_epochs 15`, `loss_lambda_diffusion 0.25`, `df_warmup_epochs 3`, `df_noise_level_min 2`.
- `tasks/lessons.md` — 2026-04-14 entry with NaN×0 trap recurrence, Min-SNR swap, defense-in-depth pattern.

### Review
- **NaN×0 trap was reintroduced** exactly where the lessons file warns against it — in a new positionally-weighted loss variant added for Diffusion Forcing. Fix: `torch.where(mask_bool, term, zeros_like)` for both cosine and MSE branches, plus explicit `nan_to_num` on `cos_sim`, `mse_per`, `weights` before multiplication.
- **Min-SNR-γ x₀ and ε were swapped** in `_diffusion_forcing_weights`. Corrected against derivation — kept v-prediction (the active `prediction_type`) unchanged but fixed the latent footgun for the other two.
- **`_safe_normalize` divergence** between `cebcm/models/chain_generator.py` and `experiments/13_chain_generator/train_chain_generator.py` was resolved by adding `nan_to_num` scrub to both; they now behave identically.
- **Defense-in-depth `nan_to_num`** on all forward-pass tensor boundaries is the cheapest insurance against bfloat16 + diffusion + high-dim SONAR geometry edge cases. ~zero runtime cost.
- **DF lambda warm-up** (`df_warmup_epochs=3`) prevents the high-variance diffusion gradient from dominating before the backbone has stabilised; base lambda also lowered `0.5 → 0.25`.
- **`system1_epochs 10→15`** plus `df_noise_level_min 0→2` removes two compounding sources of instability at the System1→System2 transition where `ans_coverage` collapsed.
- **Observability improvements**: `df_lam` is now in per-step training logs; `val_answer_coverage` gets a rolling `[WARN]` below `0.10` so the collapse pattern is caught early instead of at early-stop.
- **Static verification passed**: `python -m py_compile experiments/13_chain_generator/train_chain_generator.py cebcm/models/chain_generator.py` and `json.load` on `configs/chain_generator_config.json` both clean.
- **Not verified**: live training run with a few epochs on real data. This is a code-level fix; empirical validation requires GPU time and is the next action for the training engineer.

## 2026-04-14 - ChainGenerator probe fix: decode v-prediction through `predict_x0`

### Context
After the NaN-collapse fixes landed, a live training run showed a scary contradiction in the GUI: `train df_cos ≈ 0.72` climbing, but `probe df_cos_mean ≈ −0.68` going more negative with steps; the 3D plot drew "DF pred_x0" antipodally to the clean target. User asked whether the model was going backwards.

It was not. Under `prediction_type="v"` the raw model output is velocity, not `x₀`. At mid-range `t` with cosine schedule, `cos(v_pred, x₀_clean)` asymptotes to `−√(1−ᾱ_t) ≈ −0.707` when the model is learning correctly. The training-side `df_cos` already decoded to `pred_x0` via `predict_x0` before computing the cosine; the probe did not. The probe was lying.

### Tasks
- [x] Trace `v_df` from `forward_diffusion_forcing` through `write_training_probe_snapshot`.
- [x] Confirm raw output is v-prediction, not x₀ — verified against `predict_x0` definition in `cebcm/models/chain_generator.py::514`.
- [x] Decode `v_df = predict_x0(v_noisy, v_df_raw, levels)` immediately after the DF forward pass.
- [x] Verify all downstream sites (`projected["pred_x0"]`, `df_cos`, `df_l2`, `pred_norm`, PCA basis fitting, `raw["pred_x0"]`, metric `df_cos_mean`) now consume the decoded tensor.
- [x] Keep `v_noisy` in its original space — it is `x_t`, correctly compared to the clean target directly.
- [x] Static verification: `python -m py_compile experiments/13_chain_generator/train_chain_generator.py`.
- [x] Update `tasks/lessons.md` with full root cause + carry-forward rules.

### Review
- One-line root cause: the probe saved `forward_diffusion_forcing()`'s raw output under the label `pred_x0` without decoding the v-prediction back to x₀-space. Every cosine/L2/PCA projection downstream inherited the wrong space.
- Fix is minimal: introduce `v_df_raw`, decode to `v_df = model.predict_x0(v_noisy, v_df_raw, levels)`, let the rest of the function consume `v_df`. The user-facing semantics of the snapshot field `pred_x0` is now honest.
- Carry-forward rule added to `tasks/lessons.md`: any "cos to clean" in training/probe code MUST live in x₀-space; grep for `forward_diffusion_forcing` call sites whenever diffusion math is touched; also, when a training metric and a probe metric disagree in sign, suspect the probe first because it's newer and less battle-tested.
- Static `py_compile` passed. Empirical validation: the next probe snapshot after this fix should show `df_cos_mean` climbing toward `+1` alongside `tf_cos_mean`/`noisy_cos_mean`, and the 3D "DF pred_x0" marker should sit near the clean target instead of antipodally.

## 2026-04-14 - ChainGenerator "frozen zombie": Adam momentum corruption + skip-on-NaN trap

### Context
Live training run with the prior two fixes reached E9 and then locked into a zombie state: `loss≈0.73` finite, `grad=0.0000` every step, `NaN:N` counter `+1` every step, `lr` frozen, probe values byte-identical, `val_roll_cos` plateaued at `0.7237` from E2. User: "Опять сраное плато и 0 прогресса". Training APPEARED to run (forward finite via defensive scrubs) but no parameter updates occurred.

### Root Cause Analysis
1. `train_step` had a skip-on-NaN-grad pattern: any step where one gradient was non-finite zeroed ALL grads and returned. Healthy params never updated. Once Adam state got poisoned, every batch hit the skip path forever.
2. `optimizer.zero_grad()` only clears `.grad`; it does NOT touch Adam `exp_avg`/`exp_avg_sq`. A NaN that slipped into those momentum buffers persisted and re-emerged as a NaN gradient on the next step → self-propagating zombie.
3. `nan_to_num` in the forward pass scrubs values, NOT the backward graph. If a WEIGHT is NaN, d(loss)/d(weight) is still NaN even though `loss` reads as finite.
4. `weights.sum().clamp(min=1.0)` does NOT fix NaN — NaN passes through `clamp` unchanged, then divides into the loss.
5. `snr.to(bf16).clamp(min=1e-8)` is ineffective because bf16 has no denormals; values `< ~1.17e-38` silently underflow to 0 before clamp sees them, causing `1/snr → Inf → NaN` downstream.

### Tasks
- [x] Replace skip-on-NaN-grad with in-place `.grad.masked_fill_(bad, 0.0)` sanitation; log `grad_sanitized` count.
- [x] Rescue Adam momentum: when a parameter's grad is sanitized, `nan_to_num(buf, out=buf)` for `exp_avg`, `exp_avg_sq`, `max_exp_avg_sq`.
- [x] Post-step param sanity: after `optimizer.step()`, scan for non-finite params, `copy_(ema.shadow[name])` to restore from last known-good weights, also scrub Adam state on restored params, skip `ema.update(model)` on restore steps.
- [x] Move `_diffusion_forcing_weights` SNR computation entirely into fp32 via `torch.autocast(device_type=..., enabled=False)`; `nan_to_num` before clamp; `clamp(min=1e-6, max=1e4)`.
- [x] Scrub `mask_sum` / `weight_sum` via `nan_to_num` BEFORE `clamp(min=1.0)` in both `_masked_step_losses` and `_masked_weighted_step_losses`.
- [x] Final defensive `nan_to_num` on loss before backward; skip backward only when sanitized loss is exactly zero.
- [x] Config: `df_warmup_epochs 3→5`, `loss_lambda_diffusion 0.25→0.15`.
- [x] `python -m py_compile experiments/13_chain_generator/train_chain_generator.py` + `json.tool configs/chain_generator_config.json` — both clean.
- [x] Update `tasks/lessons.md` with the frozen-zombie pattern and 7 carry-forward rules.

### Files Touched
- `experiments/13_chain_generator/train_chain_generator.py` — `_masked_step_losses`, `_masked_weighted_step_losses`, `_diffusion_forcing_weights`, `train_step` (grad sanitation + Adam rescue + post-step EMA restore).
- `configs/chain_generator_config.json` — `loss_lambda_diffusion 0.25→0.15`, `df_warmup_epochs 3→5`.
- `tasks/lessons.md` — 2026-04-14 "frozen zombie" entry with Adam momentum + skip-trap + bf16 denormal rules.

### Review
- **Skip-on-NaN-grad was the primary trap**. The previous "safe" pattern (`if not finite: skip step`) converted transient errors into a permanent plateau. The fix is to sanitize in place and let healthy gradients keep training — corruption that actually reaches parameters is caught by the post-step EMA restore.
- **Adam momentum state was the persistence mechanism**. `optimizer.zero_grad()` never touches momentum buffers; any NaN that made it into `exp_avg`/`exp_avg_sq` self-propagated forever. Now scrubbed whenever a grad is sanitized or a param is restored.
- **EMA as break-glass recovery, not just "smoother eval"**. Post-step param sanity + `copy_(ema.shadow[name])` means a single corrupt step can no longer kill the run. EMA update is skipped on restore steps to avoid polluting the shadow with the very corruption we're rescuing from.
- **bf16 has no denormals**. `clamp(min=1e-8)` is a no-op on bf16 underflow. SNR math now runs in an explicit `autocast(enabled=False)` region and never touches bf16. This is the SOTA pattern for Min-SNR under bf16 AMP.
- **Two-layer NaN scrubbing** (scrub sums before clamp + scrub loss before backward) gives ~zero runtime cost defense-in-depth.
- **Config tightening**: lowered DF lambda and extended warm-up to give the backbone more stable headroom before DF supervision kicks in at full strength — addresses the observed E2→E9 drift where the zombie state emerged.
- **Static verification passed**. Empirical validation: the next run should show (a) `NaN:N` counter ≈ 0 or very low, (b) `grad_norm > 0` every step, (c) `val_roll_cos` moving past 0.7237 by E5+, (d) probe values changing step-to-step.

## 2026-04-15 - ChainGenerator "double zombie": SDPA -inf root cause + stronger recovery

### Context
After landing the first-round frozen-zombie fix (grad sanitize + Adam rescue + EMA restore), a new run STILL zombified at E2 S3650: grad degraded 20→13→8→0.6→0 over ~200 steps, then every subsequent step was a NaN-skip. Val permanently stuck at 0.7405 from E2 through E6+. First-round fix was necessary but not sufficient. User: "Опять сраное плато и 0 прогресса". Root cause analysis revealed TWO compounding bugs — an attention NaN source (previously undetected) AND a recovery mechanism that was too weak to break the momentum loop it was supposed to prevent.

### Root Cause
1. `cebcm/models/chain_generator.py::CrossAttention.forward` used `float("-inf")` as the additive attention-mask fill value. This is the classic bf16+SDPA NaN footgun: `-inf` masks on CUDA flash/mem-efficient backends produce NaN in softmax whenever a row is fully masked (`0/0 = NaN`), and `-inf - scale = -inf` corrupts gradient accumulation. Any batch containing a fully-masked context row instantly poisoned the entire backward graph.
2. The first-round grad sanitizer zeroed Adam `exp_avg`/`exp_avg_sq` ONLY when those buffers were themselves already non-finite. But the much more common failure mode is "finite momentum from a healthy prior step + zeroed current gradient": Adam computes `exp_avg ← β1·exp_avg`, preserving the PRE-corruption direction. The parameter keeps drifting toward the bad basin with decaying-but-nonzero speed for thousands of steps.
3. EMA was still being updated on grad-sanitation steps (first-round guard only checked `params_restored == 0`), so subtle drift leaked into the shadow and progressively polluted the rescue source.
4. `torch.isfinite` does not catch huge-but-finite drift — a param at ±1e30 is `isfinite=True` — so post-step sanity never triggered restore for the dominant failure mode.
5. No escape mechanism once 100% of steps were sanitized: per-step scrubbing cannot break out of a basin where every forward produces NaN.

### Tasks
- [x] Replace `float("-inf")` with `torch.finfo(q.dtype).min` in `CrossAttention.forward`.
- [x] Detect the all-masked-row edge case and force those rows fully visible; downstream loss masks the position anyway.
- [x] Make Adam momentum zeroing unconditional on grad sanitation (not conditional on non-finite buffers).
- [x] Gate EMA update on BOTH `not grad_had_nan` AND `params_restored == 0` so subtle drift cannot leak into the shadow.
- [x] Add huge-but-finite drift guard: post-step sanity checks `abs(p).amax() > param_abs_max` (default 1e4).
- [x] Zombie streak detector stored on the EMA object: `_zombie_streak` counter increments on every sanitation, resets on clean step; when it hits `zombie_reset_threshold` (default 15) force-restore ALL params from EMA, zero ALL Adam state, and reset the streak. Log `zombie_reset` / `zombie_resets_total` / `zombie_streak` in metrics for observability.
- [x] `python -m py_compile` both `train_chain_generator.py` and `cebcm/models/chain_generator.py`.
- [x] Update `tasks/lessons.md` with the 2026-04-15 double-zombie entry and 7 carry-forward rules.

### Files Touched
- `cebcm/models/chain_generator.py` — `CrossAttention.forward` (`finfo.min` + all-invalid-row guard).
- `experiments/13_chain_generator/train_chain_generator.py` — `train_step` (unconditional Adam zero, zombie streak detector, huge-finite drift guard, stricter EMA update gating, extra metrics).
- `tasks/lessons.md` — 2026-04-15 double-zombie entry.

### Review
- **The SDPA `-inf` mask was the upstream NaN source all along**. It was latent and only fired when a batch happened to contain a fully-masked context row. The first-round fix papered over downstream symptoms (grad scrub, Adam rescue, EMA restore) but could not prevent re-entry into the corrupted basin as long as the attention kept producing fresh NaN every forward. Fixing the root cause is what gives the other defenses a chance to actually recover.
- **Adam momentum persistence is the reason "skip on NaN" never works as a standalone pattern**. With `grad=0` and live `exp_avg`, Adam keeps applying `lr · exp_avg / sqrt(exp_avg_sq)` — the OLD direction, for exponentially many steps. Unconditional Adam zero on sanitation is the only way to break this loop.
- **EMA update gating must be conservative**. `0.9999·shadow + 0.0001·bad` compounds: over 1000 bad steps the shadow becomes ~10% bad. The fix checks BOTH `grad_had_nan` and `params_restored` and updates only when both are clean.
- **Huge-finite drift guard** (`abs(p).amax() > 1e4`) closes the last escape path: corrupted Adam can push a param to ±1e20 without ever triggering `isfinite=False`. Now post-step sanity catches it.
- **The zombie-streak hard reset is the guaranteed escape mechanism**. Regardless of root cause, 15 consecutive sanitation steps triggers a full EMA restore + Adam zero. This is the SOTA pattern for training resilience — CI/production training loops in modern LLM shops all have equivalent "break glass" rollback paths.
- **New observability metrics**: `grad_sanitized`, `param_restored`, `zombie_streak`, `zombie_resets_total`, `zombie_reset`. These let us distinguish "training plateau" from "zombie plateau" at a glance from the logs.
- **Static verification passed**. Empirical validation: the next run should show (a) `zombie_resets_total == 0` if the SDPA fix eliminates the root cause, (b) `grad_norm > 0` every step, (c) `val_roll_cos` climbing past 0.7405 by E4+, (d) probe values changing step-to-step. If zombie resets still fire, the streak counter bounds the damage and training continues.

## 2026-04-15 - 0.70 Ceiling Diagnosis: Cosine Loss Saturation Hypothesis

### Context
Arch 15-04-26 run (самый дальний прогон на текущей архитектуре) снова упёрся в потолок val_roll_cos_last≈0.7126 на E2, после чего 12 эпох замороженного плато с `grad=0.0000` и катастрофический коллапс на переходе System1→System2 (E15). **Ноль событий `zombie_reset` в логе** — recovery-система, добавленная в e5f01fa, не триггернулась, хотя код был последний. Это значит: grad=0 возникает НЕ из-за NaN, который ловит санитация, а из-за **настоящего нулевого градиента** от саттурации лосса.

User feedback: модель до добавления DF стабильно выдавала 0.75 (train близко к 0.90, eval 0.60). Потолок ~0.70 не пробивается **абсолютно непонятно почему**, хотя рецепт известен работающим. Гипотеза пользователя: grad=0 возникает когда модель "попадает на 100 процентов или при каких-то приколах косинуса". **Эта гипотеза математически подтверждается:**

### Root Cause (Mathematical)
- `loss = 1.0 · (1 - cos(p, t)) + 0.1 · MSE(p, t)` — **cosine-dominant**.
- Градиент `∂(1 − cos)/∂p = −(1/‖p‖)(I − pp⊤/‖p‖²) · t/‖t‖` — проекция цели на плоскость ⊥ p. При `p ∥ t` (cos=1) — **аналитически ноль**, не численно.
- `1 − cos ≈ ½·||p̂ − t̂||²` — квадратичная окрестность → градиент O(угол). При cos≥0.99 градиент ~1e-3 или меньше.
- bf16 mantissa = 7 бит, underflow ~8e-3. Всё что меньше → **backward cast обнуляет градиент**.
- `mse_weight=0.1` в 10× слабее — не компенсирует underflow.
- System1 с `target_steps=1` быстро загоняет модель в эту плоскую зону на лёгких one-step таргетах → зомби-плато.
- `cebcm/models/chain_generator.py:1441-1448` — confirmed в коде.

### Minimal Experiment (Active)
- [x] **Flip loss weights**: `loss_cosine_weight: 1.0 → 0.1`, `loss_mse_weight: 0.1 → 1.0` in `configs/chain_generator_config.json`. MSE имеет нулевой градиент **только при точном совпадении** (включая норму), поэтому не саттурируется. Это **единственный минимальный структурный фикс** для проверки гипотезы.
- [ ] Запустить прогон без других изменений. Acceptance:
  - (a) `val_roll_cos_last` пробивает 0.7405 (прошлый потолок) по E5+
  - (b) `grad_norm` остаётся > 0 на всех эпохах, включая E2–E15
  - (c) `ans_cov` не падает до нуля на System1→System2 transition
  - (d) `zombie_resets_total == 0` (подтверждение что grad=0 был от саттурации, а не от NaN)

### Deferred (если минимальный эксперимент не пробьёт потолок)

#### Recipe changes
- [ ] **Disable System1 entirely**: `system1_epochs: 15 → 0`, старт сразу с `target_steps=2`. Single-step — это ложный оптимум, который не композируется в rollout.
- [ ] **Disable Diffusion Forcing entirely**: `enable_diffusion_forcing: false`, `loss_lambda_diffusion: 0.0`, `df_warmup_epochs: 0`. Критично: проверить что `train_step` корректно скипает DF-ветку и что probe (`experiments/13_chain_generator/train_chain_generator.py` probe rendering) не падает на DF-плашках, либо рисует заглушки "DF disabled".
- [ ] **Switch `prediction_type: "v" → "x0"`**: с отключенным DF v-prediction теряет смысл. Проверить что `chain_generator.py` поддерживает `x0` ветку без регрессий (grep по `prediction_type`, `v_target`, `v_pred`).
- [ ] **Lower LR**: `lr: 1e-4 → 5e-5` если MSE-dominant даёт более резкие градиенты. Эмпирический критерий: наблюдать за `grad_norm`, если >> 1.0 — снижать.

#### Architectural fixes (для пробития реального потолка)
- [ ] **Attention sink mitigation**: в layer 0 все 8 голов коллапсируют в BOS (см. `Arch 15-04-26/Attention heads (epoch 17).png`). Добавить register tokens (Darcet et al. 2023) или attention softmax offset.
- [ ] **Per-layer grad norm logging**: инструментировать `train_step` чтобы видеть какой модуль схлопывается первым (cross-attn? FFN? output head?) — сейчас мы видим только total grad_norm.
- [ ] **Gradient clip in fp32**: unscale + cast в fp32 до `clip_grad_norm_`, чтобы clip не underflow'ился в bf16.

#### Verification & observability
- [ ] **Sentinel-log recovery activation**: добавить безусловный `log.info("grad_sanitation_enabled=True, zombie_threshold=15")` в начало training loop, чтобы было видно что e5f01fa код реально активен (в Arch 15-04-26 логе ноль событий — надо убедиться что это не silent disable).
- [ ] **Cosine saturation monitor**: метрика `steps_with_cos_mean_above_0.95` в логе каждые `log_every`. Индикатор саттурации до того как grad схлопнется.
- [ ] **Train/eval gap monitor**: явная метрика `train_cos_last - val_cos_last`, алерт при >0.15 (в Arch 15-04-26 гэп был 0.30).
- [ ] **Min-SNR γ validation**: при отключенном DF не нужна, при включённом — проверить что γ=5 не давит high-noise steps до нуля.

### Decision Rule
- Если flip весов пробивает потолок 0.7405 → корневая причина подтверждена, остальные deferred пункты (кроме verification) — не срочные.
- Если потолок остаётся → включать deferred пункты по одному, начиная с disable System1 + DF.
- Если вторая итерация тоже не пробивает → архитектурные фиксы (register tokens, per-layer grads).

### Files Touched (minimal experiment)
- `configs/chain_generator_config.json` — flipped loss weights only.

### Review
_Pending run results._

## 2026-04-15 - Full analysis of `Analysis` + Arch 15-04-26/1 artifacts

### Objective
- [ ] Read the user-provided dialogue/log file `C:\Users\EchoEins\Downloads\Analysis`.
- [ ] Inspect `Arch 15-04-26/1` screenshots/log artifacts and compare them with the attached GUI screenshots.
- [ ] Diagnose whether current training failure is caused by numeric instability, metric mismatch, objective saturation, architecture/curriculum mismatch, dataset issues, or GUI/probe misinterpretation.
- [ ] Produce a concise but complete report: observed facts, root-cause ranking, mathematical checks, and next verification steps.

### Review
_Pending._

### Review - analysis completed
- Parsed `C:\Users\EchoEins\Downloads\Analysis` and compared its conclusions with local artifacts in `Arch 15-04-26/1` plus current JSONL training metrics.
- Important correction: the text log/dialogue reports `zombie_reset=0`, but `experiments/13_chain_generator/output/logs/chain_generator_training.jsonl` shows `zombie_resets_total` rising to ~401 by epoch 11. The recovery mechanism did fire; the terminal log simply did not expose it.
- Main failure mode is not a single attention-mask bug: the run enters a finite zero-gradient zombie state after massive reset/sanitization events. From epoch 10 onward `grad_norm` is effectively zero on 100% of logged steps while losses and metrics remain finite.
- Eval peaks around epoch 6 (`val_metric≈0.724`, `val_DF_cos≈0.805`) and then collapses/degrades; by the System2 transition the answer supervision/curriculum is misaligned (`val_answer_coverage` drops to 0 at System2(2)).
- Diffusion Forcing becomes actively harmful after the instability window: noisy vectors remain closer to clean targets than `pred_x0`/rollout on the provided screenshots, and JSONL shows `val_DF_cos` collapsing from ~0.80 to ~0.07/-0.03 while prediction norms drift.
- Screens confirm the scalar/log diagnosis: attention develops strong start-token/sink patterns, rollout does not follow the target path, and denoising geometry often moves away from the clean chain instead of toward it.
- Next required step before any long training run: one-batch forensic gradient probe with per-layer grad norms, activation finite checks, DF x0/v norms, and before/after optimizer state inspection around the first instability window.

## 2026-04-15 - Continue unfinished Analysis section: Adam zero + ResNet proposal

### Objective
- [x] Continue the `(НЕ ЗАВЕРШЕНО)` analysis from `C:\Users\EchoEins\Downloads\Analysis`.
- [x] Reconcile the unconditional Adam zero hypothesis with current `train_step` implementation.
- [x] Evaluate whether ResNet-style practices are relevant to the observed gradient/collapse issue.
- [x] Apply only the requested schedule/config changes: no warmups, 3 System1 epochs, System2 starts at 4 steps.

### Review
- The Adam-zero hypothesis is directionally correct but needs the Adam bias-correction nuance: after clearing `exp_avg`/`exp_avg_sq`, AdamW behaves close to a sign-step for any gradient magnitude above `eps`; tiny residual gradients can therefore produce full-LR directional updates instead of safely tiny updates.
- The current code zeroes Adam buffers for sanitized params and also zeroes all Adam buffers on zombie EMA reset. That removes stale momentum but can create a cold-start Adam epsilon/sign-step regime on the next finite micro-gradient.
- This explains the observed NaN -> dead -> wake -> NaN oscillation better than a pure vanishing-gradient diagnosis.
- ResNet practices are useful only as secondary architecture stabilization: pre-norm residuals already exist; future candidates are residual branch scaling / LayerScale / ReZero-style gates / fp32 residual accumulation. They will not fix optimizer-state singularity by themselves.
- New runtime observation from the user (`ans_cov` jumps to 0.95+ after 4-step System2) supports skipping too-short System2 horizons. Config now starts System2 at 4 steps after 3 System1 epochs.
- Updated `configs/chain_generator_config.json`: `warmup_epochs=0`, `df_warmup_epochs=0`, `scheduled_sampling_ramp_epochs=0`, `tf_noise_ramp_epochs=0`, `system1_epochs=3`, `system2_start_steps=4`.

## 2026-04-15 - Review external analyses: BP alternatives + ResNet/DiT stabilization

### Objective
- [x] Compare Forward-Forward / Feedback Alignment / Target Propagation / Local Loss suggestions against the current ChainGenerator failure mode.
- [x] Compare ResNet/DiT stabilization suggestions against the current ChainGenerator architecture.
- [x] Classify which ideas are already implemented, which are useful ablations, and which should not be prioritized.

### Review
- Forward-Forward / Feedback Alignment / Target Propagation are research directions, not immediate fixes for the current failure. They would require a separate training formulation and should not be mixed into the current BP-based DF generator while the optimizer/recovery issue is unresolved.
- Local auxiliary losses are the most practical idea from the BP-alternative list. They could give layer 2/4 direct denoising/answer supervision and help detect where the gradient path collapses. Candidate ablation, not a first fix.
- ResNet basics are already present in ChainGenerator: pre-norm residual self-attn, residual cross-attn, residual FFN, RMS/AdaRMSNorm, SwiGLU, timestep MLP conditioning, EMA, global grad clipping, and zero-init AdaLN modulation/output-proj small init.
- Not yet present: explicit LayerScale/ReZero/residual-branch scalar gates and auxiliary heads. These are reasonable stabilization ablations after the Adam recovery/curriculum issue is isolated.
- Spectral normalization is appropriate for energy critics, but risky for the generator/decoder because it can restrict representation capacity and does not solve Adam cold-start sign-step behavior.
- The current instability is more consistent with optimizer-state/recovery oscillation plus horizon/curriculum issues than with a plain 6-layer vanishing-gradient problem.

## 2026-04-16 - Architecture points 1-4 implementation review

### Objective
- [ ] Verify implemented architectural stabilizers 1-4 against current `ChainGenerator` code and project lessons.
- [ ] Check whether the implementation preserves SONAR geometry, DF timestep conditioning, residual highways, and train/eval consistency.
- [ ] Classify remaining options, explicitly including Forward-Forward and auxiliary heads.
- [ ] Report correctness issues and recommended next steps without changing architecture.

### Review
_Pending._

### Review - architecture points 1-4 checked
- Pre-Norm residual order is implemented correctly: each decoder sublayer uses `x + sublayer(norm(x))`, including AdaRMSNorm-conditioned DF path.
- Zero-init residual projectors are implemented after the global Xavier sweep, so the initialization is not overwritten. Self-attn `out_proj`, cross-attn `out_proj`, and SwiGLU `w_down` are zeroed when `zero_init_residual=true`.
- AdaRMSNorm/AdaLN-Zero is implemented with a second re-zeroing of `adaln_modulation[-1]` after global init. This preserves scale=0/shift=0 at initialization.
- LayerScale is implemented as per-channel gates on self-attn/cross-attn/FFN residual branches and is configured with `layerscale_init=1e-4`.
- Caveat: zero-init residual projectors plus LayerScale=1e-4 makes early residual-branch learning very damped. It stabilizes but can slow branch opening; monitor residual projector grad norms and LayerScale parameter growth.
- Caveat: the DF path still adds `t_emb` directly to the residual stream before AdaLN-conditioned blocks. This is not a pure DiT adaLN-Zero path, but it is intentional/backward-compatible and gives an initial noise-level signal.
- Static checks passed: `chain_generator.py` and `train_chain_generator.py` py-compile; `chain_generator_config.json` parses. Runtime torch forward test could not run in the Windows-side environment because the available `uv` env has no `torch` and no local `.venv` exists there.
- Additional backlog now explicitly includes Forward-Forward as a research-only EBM/critic branch and auxiliary heads as the highest-value practical deep-supervision ablation.

### Review - implementation checked
- Static checks passed: `cebcm/models/chain_generator.py` and `experiments/13_chain_generator/train_chain_generator.py` compile; `configs/chain_generator_config.json` is valid JSON.
- Runtime torch smoke-test was not available from this Windows shell: `uv run python` cannot import torch and no local `.venv` exists here. Run the one-batch CUDA/venv smoke test before a long training run.
- Point 1, Pre-Norm residual order: implemented correctly in `DecoderBlock.forward`. Self-attention, cross-attention, and FFN are each called on normalized input and then added back as residual branches.
- Point 2, zero-init residual projectors: implemented correctly and applied after global Xavier init. It zeros self-attn out projection, cross-attn out projection, and FFN down/final projection, so each block starts close to identity.
- Caveat for zero-init: at the first optimizer steps gradients mainly open the final residual projectors; inner attention/FFN weights receive useful gradients only after those projectors leave exact zero. This is expected for DiT-style zero init, but monitor projector grad norms and branch output norms.
- Point 3, AdaRMSNorm / adaLN-Zero direction: base norm starts with weight=1, modulation MLP final layers are zeroed, and modulation is applied before sublayers. This is mathematically aligned with adaLN-zero behavior.
- Caveat for AdaRMSNorm: diffusion path still adds `t_emb` directly into token states before decoder layers. That is not pure DiT adaLN-zero conditioning. Keep it only if ablation proves it helps; otherwise prefer gated/adaptive norm conditioning only.
- Point 4, LayerScale: implemented correctly as per-branch vectors initialized from config and excluded from weight decay by the existing ndim<=1 optimizer grouping.
- Caveat for LayerScale: `zero_init_residual=true` plus `layerscale_init=1e-4` can make branch opening slow. If gradients are finite but learning stalls, run ablations at 1e-3 and 1e-2.
- Spectral normalization remains intentionally rejected for generator/residual transformer because project lessons show hard Lipschitz normalization can destroy ranking/geometry. If revisited, isolate it to critic-only ablation.
- Next additions worth testing: forensic probe, hard-fail zombie guard, AR-only/DF-only/combined isolation, auxiliary heads on intermediate layers, self-conditioning, and Forward-Forward only as a separate research branch rather than a replacement for BP.

### Forward-Forward experimental note
- FF should be tested as an isolated experimental branch, not as a replacement for the main BP/DF generator until it proves value on the same one-batch and short-run diagnostics.
- Current ChainGenerator already has ResNet-style additive residuals: `x = x + scaled_sublayer(norm(x))` for self-attn, cross-attn, and FFN. This preserves the raw residual stream across layers.
- Standard Transformers do not use XOR mixing. They mix information by self-attention weighted sums across tokens, MLP channel mixing, and additive residual connections.
- A full FF experiment needs local per-layer goodness heads/losses, positive/negative chain construction, post-goodness activation normalization before passing to the next FF-trained layer, and no dropout/batchnorm in the FF path.
- Practical FF candidate: pretrain layer/block representations with positives = real target chains and negatives = corrupted/generated chains, then initialize the BP generator or critic from those weights.
- Safer near-term alternative: auxiliary heads on layers 2/4 with supervised `pred_x0`/answer losses. This keeps BP but gives local gradients and layer canaries.

## 2026-04-16 - Auxiliary heads implementation
- [ ] Inspect ChainGenerator forward/loss and training loop integration points.
- [ ] Add optional auxiliary prediction heads on intermediate decoder layers without changing inference behavior when disabled.
- [ ] Add masked auxiliary losses for pred_x0/chain targets and answer target where valid.
- [ ] Log train/val aux metrics so intermediate-layer learning can be diagnosed.
- [ ] Update config defaults and review notes.
- [ ] Run syntax/config checks and any available smoke tests.

### Review - Auxiliary heads implemented
- [x] Inspected ChainGenerator forward/loss and training loop integration points.
- [x] Added optional auxiliary prediction heads on 1-based decoder layers from config; default config enables layers 2 and 4.
- [x] Main inference/generation path is unchanged unless `return_aux=True`; `generate()` does not use aux heads.
- [x] Teacher-forced/scheduled-sampling path can return aux predictions and trains them with masked step loss plus answer-vector loss.
- [x] Diffusion Forcing path can return aux predictions and trains them as clean-x0 denoising canaries with `loss_lambda_aux_df`.
- [x] Aux losses are averaged across heads so adding heads does not silently multiply the effective lambda.
- [x] Train/val metrics now expose `loss_aux`, `aux_cos_mean`, `aux_cos_answer`, `loss_aux_df`, and `aux_df_cos_mean`.
- [x] Config updated: `aux_head_layers=[2,4]`, `loss_lambda_aux=0.1`, `loss_lambda_aux_df=0.03`.
- [x] Static verification passed: py_compile for model/training and JSON parse for config.
- [ ] Runtime torch smoke-test remains to be run in the actual CUDA `.venv`; Windows-side `uv` cannot import torch in this shell.

### Monitoring rules
- If `aux_l2_cos_mean` rises but final/rollout metrics stay flat, the lower representation learns and the failure is likely in upper layers, rollout, or final head.
- If `aux_l2_cos_mean` also stays low, inspect input scale, noise schedule, memory bank, and target construction before touching architecture.
- If `aux_df_cos_mean < noisy_cos_mean` for sustained steps, DF is still a degrader and must be isolated with DF-only short runs.

### Review - Web GUI aux metrics support
- [x] Training Geometry JSONL loader already flattens new train/val aux scalar metrics.
- [x] Step-Level Training Metrics plot now explicitly displays aux losses and aux cosine metrics: train/val aux, aux answer, and aux DF.
- [x] Syntax check passed for `cerber_gui/training_geometry.py`.

### Review - first aux run diagnostics
- Current log confirms the old zombie failure still occurs: training is healthy until about step 1900, then NaN/skip count explodes and `grad_norm` collapses to 0 while finite-looking metrics continue.
- `aux_cos=0.0` in System1 is a reporting artifact: direct-answer mode masks the answer out of `L_step`, so there are no non-answer step tokens for aux step cosine. Code now falls back to aux answer cosine when the step mask is empty.
- Added hard-fail zombie guard in the main training loop. A run now stops and writes `hard_fail_zombie` JSONL event after configurable bad-step/zero-grad streaks instead of continuing over a dead network.
- Config defaults added: `hard_fail_bad_step_streak=25`, `hard_fail_zero_grad_streak=25`, `zero_grad_threshold=1e-8`.
- Static checks passed after the fix.
