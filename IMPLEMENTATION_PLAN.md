# CEBCM — План реализации v2

**Дата:** 21 марта 2026
**Статус:** Утверждение
**Scope:** Полный цикл разработки от PoC до production-ready модели
**Hardware:** 1x consumer GPU (RTX 3090/4090, 24GB VRAM)

---

## 0. Принятые решения (v2)

| Параметр | Решение | Обоснование |
|---|---|---|
| Autoencoder | SONAR (замороженный, 1024d) | Нулевая стоимость обучения, готовый decoder |
| Рабочая размерность | 1024d (нативный SONAR) | Projectors только при kill criterion |
| EBT | Pairwise Head → Chain Head → Joint | Инкрементальная сложность |
| SurprisePredictor | SSM (Mamba-style), тренируется параллельно с Stage 2 | Self-supervised, не зависит от пайплайна |
| IPP (Initialization Point Predictor) | MLP/Transformer, тренируется параллельно с Stage 2 | Входит в пайплайн в Stage 4, дотренировывается |
| SurprisePredictor freeze | **Навсегда замороженный** после обучения | Простая модель — другие модули подстраиваются под неё |
| ContextAggregator | Linear Attention (Mamba/GLA) + Surprise Global Tokens | O(N) + O(G²), масштабируется на 50K+ |
| Compact token | Learned embedding (отдельный token) | Чтобы модель чётко классифицировала действие сжатия |
| Decoder | Fine-tune малой LM (~1B) в конце каждого Stage | Cross-entropy: Langevin output → ground truth текст |
| Batch contrastive | 1 positive + N negatives одновременно (InfoNCE) | Модель учится **почему** каждый negative плох → System 2 |
| Curriculum | easy (0.3–0.7) → medium (0.7–0.9) → hard (0.9–0.98) | + масштабирование: short→long contexts + compact |
| Thinking modes | System 1 (fast) + System 2 (deep) с train-time sampling | Модель учится в обоих режимах |
| Inertial Navigation | cruise_ratio sampling при обучении | Модель привыкает к momentum в разных режимах |

---

## 1. Файловая структура проекта (v2)

```
CERBER/
├── CEBCM_Technical_Specification.md   # Спецификация v1.3
├── IMPLEMENTATION_PLAN.md             # Этот файл
├── README.md
│
├── configs/
│   ├── __init__.py
│   ├── base.py                        # @dataclass Config
│   ├── sonar.yaml
│   ├── ebt.yaml
│   ├── surprise.yaml                  # Параметры SurprisePredictor
│   ├── context.yaml                   # ContextAggregator + compaction
│   ├── training.yaml                  # lr, batch_size, curriculum
│   └── inference.yaml                 # Langevin, cruise_ratio, presets
│
├── cebcm/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── energy.py                  # EBT_PairwiseHead, EBT_ChainHead, EBT
│   │   ├── ipp.py                     # IPP (Initialization Point Predictor)
│   │   ├── surprise.py                # SurprisePredictor (SSM-based)
│   │   ├── context.py                 # ContextCache, ContextAggregator
│   │   ├── decoder.py                 # TrainableDecoder (fine-tuned LM)
│   │   ├── compact.py                 # CompactToken, CompactionController
│   │   ├── faiss_index.py             # FAISSIndexManager
│   │   └── sonar_wrapper.py           # SONAR encoder/decoder wrapper
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py                  # InfoNCE, margin, gradient penalty
│   │   ├── curriculum.py              # CurriculumScheduler
│   │   ├── negatives.py               # NegativeGenerator (FAISS-based)
│   │   ├── batch_contrastive.py       # Batch contrastive logic
│   │   ├── train_ebt.py               # Обучение EBT (Pairwise + Chain)
│   │   ├── train_ipp.py         # Обучение IPP (параллельный трек)
│   │   ├── train_surprise.py          # Обучение SurprisePredictor (параллельный)
│   │   ├── train_context.py           # Обучение ContextAggregator
│   │   ├── train_decoder.py           # Fine-tune Decoder LM
│   │   └── train_e2e.py               # End-to-end fine-tune (Stage 4E)
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── langevin.py                # Langevin dynamics + inertial navigation
│   │   ├── pipeline.py                # CEBCMPipeline (полный E2E)
│   │   ├── compaction.py              # External Compaction runtime
│   │   └── modes.py                   # InferencePresets (fast/balanced/deep)
│   │
│   └── data/
│       ├── __init__.py
│       ├── encode_dataset.py          # Text → SONAR vectors
│       ├── dataset.py                 # PyTorch Datasets
│       ├── dialogue_builder.py        # Синтетические multi-turn диалоги
│       └── utils.py
│
├── experiments/
│   ├── 00_sonar_validation/
│   ├── 01_denoising_poc/
│   ├── 02_qa_pairwise/
│   ├── 03_chain_of_thought/
│   ├── 04_full_pipeline/
│   └── 05_scale_test/
│
├── tests/
│   ├── test_energy.py
│   ├── test_langevin.py
│   ├── test_surprise.py
│   ├── test_context.py
│   ├── test_compaction.py
│   └── test_curriculum.py
│
├── requirements.txt
└── setup.py
```

---

## 2. Обзор стадий развития

```
Stage 0          Stage 1          Stage 2              Stage 3              Stage 4                Stage 5
SONAR Valid  →   Denoising   →   EBT Pairwise    →   Chain of Thought →   Full Pipeline      →   Projectors
                  PoC             + ‖ Surprise       + System 2            Integration            (если надо)
                                  + ‖ IPP             + ‖ IPP               (всё вместе)
                                  (параллельно)       (параллельно)
```

**Параллельные треки (не в пайплайне):**
- SurprisePredictor: тренируется с Stage 2, входит frozen в Stage 4 (остаётся frozen навсегда)
- IPP: тренируется с Stage 2, входит в Stage 4 и дотренировывается в пайплайне

---

## 3. Stage 0: Валидация SONAR пространства (3–4 дня)

**Цель:** убедиться, что SONAR-пространство пригодно для градиентной навигации.

### 3.1 Инфраструктура (день 1)

- [x] Файловая структура, `configs/base.py`, `requirements.txt`
- [x] `cebcm/models/sonar_wrapper.py`:
  ```python
  class SONARWrapper:
      def encode(self, texts: list[str], lang="eng_Latn") -> Tensor  # [N, 1024]
      def decode(self, vectors: Tensor, lang="eng_Latn") -> list[str]
      def encode_batched(self, texts, lang, batch_size=64) -> Tensor
  ```
- [x] VRAM estimation: encoder + decoder + EBT на одной GPU

### 3.2 Эксперименты (дни 2–3)

| Эксперимент | Что проверяем | Kill criterion | Результат |
|---|---|---|---|
| A: Noise robustness | decode(V + relative noise) | noise 5% relative → cos_sim < 0.85 | **PASS** (cos=0.53 при 5%, safe zone ≤1%) |
| B: Interpolation | decode(αV₁ + (1-α)V₂) | Промежуточные предложения бессмысленны | **PASS** (гладкие семантические переходы) |
| C: Distribution | norms, cosine distances | Кластеризация / коллапс пространства | **PASS** (mean cos=0.25, 0 dead dims) |
| D: Gradient flow | ∂output/∂V через decoder | Нет градиентов | **PASS** (0.27→0.97 за 100 шагов) |

### 3.3 GO/NO-GO (день 3–4)

Документируем результаты. Получаем `target_norm` из Эксперимента C.

### 3.4 Результаты (22.03.2026)

**Вердикт: GO.** Все kill criteria пройдены. SONAR-пространство пригодно для градиентной навигации.

**Параметры для Stage 1:**

| Параметр | Значение |
|---|---|
| `target_norm` | 0.2051 |
| `embedding_dim` | 1024 |
| `safe_noise_threshold` | ≤1% от нормы |
| `langevin_lr` (abs) | 0.001 |
| `langevin_lr` (rel) | 0.005 |

**Краткая выжимка:**

- **Noise:** Safe zone ≤1% relative noise (cos_sim > 0.95). Резкий обрыв качества между 1% и 5% — beam search декодера чувствителен к шуму. При ≥10% — полная потеря смысла.
- **Interpolation:** Гладкие семантические переходы. Пространство структурировано: сначала меняются ключевые слова, потом глаголы, потом контекст.
- **Distribution:** Нормы ~0.205±0.019, mean pairwise cos_sim = 0.248 (здоровый разброс). 0 мёртвых измерений из 1024.
- **Gradient flow:** Монотонная сходимость от cos_sim ~0.27 до ~0.97 за 100 шагов. Из бессмысленного текста восстанавливается семантика оригинала. Langevin dynamics будет работать.
- **VRAM:** Encoder (2.9GB) + Decoder (3.3GB) = 6.2GB из 7.8GB. EBT (~40MB) поместится. Для тренировки EBT decoder не нужен.

**Артефакты:**
- `experiments/00_sonar_validation/experiment_a_noise.json`
- `experiments/00_sonar_validation/experiment_b_interpolation.json`
- `experiments/00_sonar_validation/experiment_c_distribution.json`
- `experiments/00_sonar_validation/experiment_d_gradient.json`
- `experiments/00_sonar_validation/go_nogo_report.json`
- `experiments/00_sonar_validation/vram_estimation.json`
- `experiments/00_sonar_validation/target_norm.pt`

Полная выжимка с таблицами и примерами: `CEBCM_Technical_Specification.md`, секция 4.3.1.

---

## 4. Stage 1: Denoising PoC (2–3 дня)

**Цель:** sanity check — работает ли EBT + Langevin в SONAR-пространстве.

### 4.1 Пайплайн обучения

```
Компоненты в пайплайне:
  SONAR encoder (frozen) → [SimpleEnergy] → Langevin → SONAR decoder (frozen)
                            ^^^ТРЕНИРУЕТСЯ^^^
```

**Данные:** WikiText-103 → SONAR vectors (10K предложений)

**Training loop:**
1. Берём V_orig из датасета
2. V_noisy = V_orig + noise (noise_scale ∈ [0.1, 0.2, 0.3])
3. E_pos = SimpleEnergy(V_orig, V_orig) — энергия позитивной пары
4. E_neg = SimpleEnergy(V_orig, V_noisy) — энергия негативной пары
5. Loss = margin_contrastive(E_pos, E_neg, margin=1.0)
6. 50 эпох, batch=32, lr=1e-4

| Компонент | Статус | Параметры |
|---|---|---|
| SONAR encoder | ❄️ frozen | — |
| SimpleEnergy (MLP) | 🔥 train | ~10M params |
| SONAR decoder | ❄️ frozen | — |

### 4.2 Тестирование

```
Для 100 тестовых векторов:
  V_noisy = V_orig + noise → Langevin refine → V_denoised
  Метрики: cos_sim(V_orig, V_denoised) > cos_sim(V_orig, V_noisy)
  + decode обоих, сравнение текстов
```

**Kill criterion:** denoised не ближе к оригиналу → энергетическая функция не работает.

### 4.3 Decoder тренировка (Stage 1)

Проверяем, нужен ли fine-tune decoder:
1. Если SONAR decoder хорошо декодирует Langevin outputs → используем as is
2. Если ломается → fine-tune малой LM (~1B) на парах (V_denoised → original text)
3. Cross-entropy loss, 10-20 эпох

---

## 5. Stage 2: EBT Pairwise + Параллельные треки (2–4 недели)

**Цель:** обучить EBT различать правильные и неправильные ответы на QA, параллельно тренировать SurprisePredictor и Predictor.

### 5.1 Подготовка данных

- [ ] SQuAD v2 → SONAR: ~87K QA-пар
- [ ] WikiText → SONAR: 50K предложений (для параллельных треков)
- [ ] MultiWOZ / Ubuntu Dialogue → SONAR: multi-turn диалоги (для Stage 4)
- [ ] FAISS-индекс по V_questions для генерации негативов
- [ ] Предрассчёт негативов по уровням (easy/medium/hard)

### 5.2 Пайплайн обучения EBT Pairwise

```
Компоненты в пайплайне:
  SONAR encoder (frozen) → [EBT Pairwise] → Langevin → SONAR decoder (frozen)
                            ^^^ТРЕНИРУЕТСЯ^^^
```

| Компонент | Статус | Параметры |
|---|---|---|
| SONAR encoder | ❄️ frozen | — |
| EBT_PairwiseHead | 🔥 train | ~30M params |
| Langevin dynamics | — | lr, noise, cruise_ratio |
| SONAR decoder | ❄️ frozen | — |

#### Curriculum Learning

```
Phase 1 (Easy, 20% эпох):
  Негативы: cos_sim 0.3–0.7 (случайные из корпуса)
  Модель учится: "этот ответ вообще не из этой темы"

Phase 2 (Medium, 30% эпох):
  Негативы: cos_sim 0.7–0.9 (похожие вопросы → их ответы)
  Модель учится: "этот ответ из той же темы, но не тот"

Phase 3 (Hard, 50% эпох):
  Негативы: cos_sim 0.9–0.98 (очень похожие вопросы)
  Модель учится: "эти ответы почти одинаковы, но один правильнее"
```

#### Batch Contrastive Learning

```python
# Каждый batch содержит 1 positive + N negatives ОДНОВРЕМЕННО
def batch_contrastive_step(ebt, V_query, V_positive, V_negatives):
    """
    V_query: [batch, 1024]
    V_positive: [batch, 1024]
    V_negatives: [batch, N_neg, 1024]

    Модель видит ВСЮ пачку негативов и учится:
    1. Почему positive лучше каждого negative
    2. Почему одни negatives ближе, а другие дальше
    3. Это формирует System 2: рассуждение через сравнение
    """
    E_pos = ebt.energy(V_query, V_positive)             # [batch]
    E_neg = ebt.energy(
        V_query.unsqueeze(1).expand_as(V_negatives),
        V_negatives
    )                                                     # [batch, N_neg]

    # InfoNCE: модель сравнивает positive со ВСЕМИ negatives
    logits = torch.cat([-E_pos.unsqueeze(1), -E_neg], dim=1) / temperature
    labels = torch.zeros(batch_size, dtype=torch.long)    # positive = index 0
    loss = F.cross_entropy(logits, labels)

    # Gradient penalty для гладкости энергетического ландшафта
    loss += gradient_penalty(ebt, V_query, V_positive, lambda_gp=0.1)

    return loss
```

#### Inertial Navigation Awareness

```python
# При обучении: sampling cruise_ratio для каждого батча
cruise_ratios = [0.0, 0.0, 0.0, 0.3, 0.5, 0.7]  # bias к чистому Langevin
cruise_ratio = random.choice(cruise_ratios)

# EBT учится оценивать вектора, достигнутые разными стратегиями
# Это критично: иначе модель будет хорошо работать только с чистым Langevin
```

#### Training hyperparameters

```yaml
training:
  lr: 1e-4
  weight_decay: 0.01
  batch_size: 256
  num_negatives: 31              # 1 positive + 31 negatives в батче
  temperature: 0.07
  total_epochs: 100
  gradient_penalty_lambda: 0.1
  spectral_norm: true            # на всех Linear слоях EBT
```

### 5.3 Параллельный трек A: SurprisePredictor

```
Тренируется ОТДЕЛЬНО от пайплайна, на том же корпусе:
  WikiText + SQuAD → предложения → SONAR вектора → последовательности

Задача: next-vector prediction (self-supervised)
```

```python
class SurprisePredictor(nn.Module):
    """SSM-based, ~50-100M params, 2 слоя, state_dim=2048"""

# Training loop (полностью независимый):
for sequences in dataloader:  # [batch, seq_len, 1024]
    predictions = surprise_predictor.predict_next(sequences[:, :-1])
    targets = sequences[:, 1:]

    loss_mse = F.mse_loss(predictions, targets)
    loss_cos = (1 - F.cosine_similarity(predictions, targets, dim=-1)).mean()
    loss = loss_mse + 0.5 * loss_cos

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

| Компонент | Статус |
|---|---|
| SONAR encoder | ❄️ frozen (только для кодирования данных) |
| SurprisePredictor | 🔥 train (отдельный процесс) |

**Критерий готовности:** prediction loss сходится, surprise scores осмысленны (высокие для неожиданных предложений, низкие для банальных).

### 5.4 Параллельный трек B: IPP (Initialization Point Predictor)

```
Тренируется ОТДЕЛЬНО от пайплайна, на SQuAD QA-парах:
  Задача: V_query + context → предсказать V_answer

Это "предсказатель хорошей начальной точки" для Langevin.
```

```python
class IPP(nn.Module):
    """MLP или shallow Transformer, ~20-50M params"""

# Training loop (полностью независимый):
for V_query, V_context, V_answer_target in dataloader:
    V_init_predicted = ipp(V_query, V_context)

    loss_cos = (1 - F.cosine_similarity(V_init_predicted, V_answer_target, dim=-1)).mean()
    loss_mse = F.mse_loss(V_init_predicted, V_answer_target)
    loss = loss_cos + 0.5 * loss_mse

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

| Компонент | Статус |
|---|---|
| IPP | 🔥 train (отдельный процесс) |

**Важно:** IPP не входит в основной пайплайн до Stage 4. Он просто учится давать хорошую V_init.

### 5.5 Тестирование Stage 2

**Тест A: Ранжирование**
```
accuracy = (E_correct < E_random).mean() → ожидание: >0.9
```

**Тест B: Langevin из informed noise** (без IPP — просто noisy V_query)
```
V_init = 0.5 * V_query + 0.5 * noise
V_refined = langevin.refine(V_init, V_query)
cos_sim(V_refined, V_correct_answer) → ожидание: >0.6
SONAR decode → осмысленный текст
```

**Тест C: Ablation по cruise_ratio**
```
cruise_ratio ∈ [0.0, 0.3, 0.5, 0.7]:
  Качество vs скорость (backward passes)
  cruise_ratio=0.5 → <10% деградации при 2x ускорении
```

### 5.6 Decoder тренировка (Stage 2)

Fine-tune decoder LM на парах (V_refined → answer_text):
1. Берём Langevin outputs от Stage 2
2. Ground truth = original answer text из SQuAD
3. Cross-entropy loss, teacher forcing
4. 20-50 эпох

---

## 6. Stage 3: Chain of Thought + System 2 (3–5 недель)

**Цель:** добавить Chain Head для рассуждений, обучить System 2 thinking.

### 6.1 Фаза A: Chain Head — Scoring цепочек (1–2 недели)

```
Компоненты в пайплайне:
  SONAR (frozen) → EBT [Pairwise(FROZEN) + ChainHead(TRAIN)] → Langevin → SONAR decoder (frozen)
```

| Компонент | Статус |
|---|---|
| SONAR encoder | ❄️ frozen |
| EBT_PairwiseHead | ❄️ frozen (из Stage 2) |
| EBT_ChainHead | 🔥 train |
| Langevin | — |
| SONAR decoder | ❄️ frozen |
| SurprisePredictor | ❄️ тренируется отдельно, не в пайплайне |
| IPP | 🔥‖ тренируется отдельно, не в пайплайне |

#### Архитектура Chain Head

```python
class EBT_ChainHead(nn.Module):
    """
    Оценивает качество цепочки рассуждений V₁ → V₂ → ... → Vₙ.
    Self-attention между элементами цепочки.
    """
    def __init__(self, dim=1024, n_heads=8, n_layers=2, max_chain_len=20):
        self.position_embedding = nn.Embedding(max_chain_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim*2,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.chain_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.energy_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, chain: Tensor) -> Tensor:
        """
        chain: [batch, chain_len, 1024]
        Returns: scalar energy for each chain [batch]
        """
        positions = torch.arange(chain.size(1), device=chain.device)
        chain = chain + self.position_embedding(positions)
        encoded = self.chain_encoder(chain)
        pooled = encoded.mean(dim=1)  # или CLS-style
        return self.energy_head(pooled).squeeze(-1)
```

#### Данные для Chain Head

```python
# Positive chains: осмысленные последовательности рассуждений
# Из SQuAD: question → context_sentence₁ → context_sentence₂ → answer
# Из WikiText: последовательные предложения абзаца

# Negative chains:
# 1. Shuffled: те же вектора, но в случайном порядке
# 2. Truncated: отсутствует ключевой шаг
# 3. Corrupted: один вектор заменён на случайный
# 4. Wrong conclusion: правильные шаги → неправильный финальный ответ
```

#### Training loop Chain Head

```python
for positive_chains, negative_chains in dataloader:
    E_pos = chain_head(positive_chains)     # [batch]
    E_neg = chain_head(negative_chains)     # [batch, N_neg_chains]

    # InfoNCE по цепочкам
    logits = torch.cat([-E_pos.unsqueeze(1), -E_neg], dim=1) / temperature
    labels = torch.zeros(batch_size, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)

    # Batch contrastive: модель сравнивает негативные цепочки между собой
    # "Эта цепочка плоха потому что порядок нарушен,
    #  а эта плоха потому что вывод неверный"

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 6.2 Фаза B: Joint fine-tune + Thinking modes (1–2 недели)

```
Компоненты в пайплайне:
  SONAR (frozen) → EBT [Pairwise(TRAIN) + ChainHead(TRAIN)] → Langevin → SONAR decoder (frozen)
                    ^^^^^^^^^^^РАЗМОРАЖИВАЕМ ОБА^^^^^^^^^^^
```

| Компонент | Статус |
|---|---|
| SONAR | ❄️ frozen |
| EBT_PairwiseHead | 🔥 train (размороженный) |
| EBT_ChainHead | 🔥 train |
| Langevin | — (cruise_ratio sampling) |
| SONAR decoder | ❄️ frozen |

#### System 1 vs System 2 — Train-time switching

```python
# Каждый батч обрабатывается в случайном режиме:
thinking_mode = random.choices(
    ["system1", "system2"],
    weights=[0.3, 0.7]  # bias к System 2 (сложнее, нужно больше)
)[0]

if thinking_mode == "system1":
    # Fast shot: Pairwise scoring
    E = ebt.pairwise(V_query, V_candidate)
    langevin_config = LangevinConfig(
        max_steps=random.choice([10, 20, 50]),
        cruise_ratio=random.choice([0.5, 0.7]),
    )

elif thinking_mode == "system2":
    # Deep thinking: Chain scoring
    chain = generate_chain(V_query, V_candidate, steps=random.randint(3, 15))
    E = ebt.chain(chain)
    langevin_config = LangevinConfig(
        max_steps=random.choice([100, 200, 500]),
        cruise_ratio=random.choice([0.0, 0.1, 0.3]),
    )

# Langevin refinement с выбранным config
V_refined = langevin.refine(V_init, V_query, config=langevin_config)
```

#### Momentum (Inertial Navigation) тренировка

```python
# Модель должна привыкнуть к РАЗНЫМ cruise_ratio:
# cruise_ratio=0.0 — чистый Langevin (каждый шаг = gradient + noise)
# cruise_ratio=0.7 — агрессивная инерция (70% шагов без backward)

# При обучении: EBT видит вектора, достигнутые РАЗНЫМИ стратегиями
# Это критично для robustness при инференсе

# Training: для каждого примера генерируем 2-3 траектории с разными cruise_ratio
for cr in [0.0, 0.3, 0.5]:
    V_refined = langevin.refine(V_init, V_query,
                                 config=LangevinConfig(cruise_ratio=cr))
    # EBT должен давать сопоставимую энергию для всех V_refined,
    # если они все пришли к хорошему ответу
```

### 6.3 Тестирование Stage 3

**Тест A: Chain ranking**
```
accuracy = (E_positive_chain < E_negative_chain).mean() → >0.85
```

**Тест B: System 1 vs System 2 quality**
```
System 1 (fast): avg cos_sim с ground truth, ROUGE-L
System 2 (deep): avg cos_sim, ROUGE-L
System 2 > System 1 по качеству (но медленнее)
```

**Тест C: Momentum robustness**
```
Для cruise_ratio ∈ [0.0, 0.3, 0.5, 0.7]:
  Quality degradation < 15% при cruise_ratio=0.5
```

### 6.4 Decoder тренировка (Stage 3)

Fine-tune decoder LM на Langevin outputs от Stage 3:
1. System 1 outputs + System 2 outputs → decoder
2. Cross-entropy vs ground truth text
3. Проверяем: один decoder работает для обоих режимов?
   Если нет → два decoder или conditional decoder

---

## 7. Stage 4: Полная интеграция пайплайна (4–8 недель)

**Цель:** собрать все компоненты в единый пайплайн, обучить работать с контекстом, Global Tokens, компакцией.

### 7.0 Полный пайплайн Stage 4

```
User text
  ↓
SONAR encoder (frozen)
  ↓
V_new (1024d)
  ↓
SurprisePredictor (❄️ FROZEN, из параллельного трека)
  ↓ surprise_score + global_token_flag
Context Cache (+ metadata: surprise, turn_id)
  ↓
Полный контекст (до MAX) + все Global Tokens
  ↓
ContextAggregator (🔥 TRAIN) ← Linear Attention (SSM) + Global Tokens attention
  ↓ V_context (aggregated)
IPP (🔥 TRAIN, дотренировывается в пайплайне) → V_init
  ↓
EBT (Pairwise + Chain) (🔥 TRAIN) + Langevin
  ↓ V_answer
Decoder LM (🔥 TRAIN) → answer text
```

### 7.1 Фаза A: Интеграция новых модулей (1–2 недели)

**Что происходит:** модель впервые видит SurprisePredictor outputs и IPP V_init. IPP дотренировывается в пайплайне.

| Компонент | Статус | Комментарий |
|---|---|---|
| SONAR | ❄️ frozen | — |
| SurprisePredictor | ❄️ frozen | Навсегда frozen |
| ContextAggregator | 🔥 train | Тренируется с нуля |
| IPP | 🔥 train | Дотренировывается в пайплайне |
| EBT (Pairwise + Chain) | 🔥 train | Дообучается с Global Tokens |
| Decoder LM | ❄️ frozen | Из Stage 3 |

**Короткие контексты:** 5–20 ходов (диалоги из MultiWOZ / синтетические из SQuAD)

```python
# Тренировка Phase A:

for dialogue in short_dialogues:  # 5-20 ходов
    vectors = sonar.encode(dialogue.sentences)

    # 1. SurprisePredictor (frozen) размечает surprise scores
    surprise_scores = surprise_predictor.compute_surprise(vectors)
    global_mask = surprise_scores > theta  # top-5% → Global Tokens

    # 2. Модель ВПЕРВЫЕ видит Global Token flags
    #    ContextAggregator учится: global tokens → direct attention,
    #    обычные → через SSM state
    context = context_aggregator(
        V_query=vectors[-1],
        context_vectors=vectors[:-1],
        surprise_scores=surprise_scores[:-1],
        global_mask=global_mask[:-1]
    )

    # 3. IPP даёт V_init (дотренировывается)
    V_init = ipp(vectors[-1], context)

    # 4. EBT + Langevin refinement
    V_answer = langevin.refine(V_init, vectors[-1], ebt)

    # Loss: cos_sim(V_answer, V_target) + InfoNCE + gradient penalty
    loss = compute_loss(V_answer, V_target, ebt, V_query, negatives)
```

**Curriculum (Phase A):** Easy негативы (cos_sim 0.3–0.7) — модель привыкает к новой архитектуре.

### 7.2 Фаза B: Масштабирование контекста (1–2 недели)

| Компонент | Статус |
|---|---|
| ContextAggregator | 🔥 train |
| EBT | 🔥 train |
| Остальные | ❄️ frozen |

**Масштаб контекста:**
```
Week 1: 20 → 50 → 100 ходов
Week 2: 100 → 200 → 500 ходов
```

**Curriculum:** Easy → Medium негативы

**Данные:** Multi-turn диалоги:
- MultiWOZ: ~10K диалогов, 10-15 ходов
- Ubuntu Dialogue: ~1M диалогов, длинные
- Синтетические: конкатенация SQuAD пар в псевдо-диалоги

```python
# Mamba SSM context scaling:
# state_dim=2048, n_layers=2
# Train на 500-2K предложений, inference на 5K-10K
# SSM state обновляется рекуррентно → нет ограничения на длину
```

### 7.3 Фаза C: Compact Prompt + External Compaction (1–2 недели)

**Вводим compact token — learned embedding для сжатия контекста.**

| Компонент | Статус |
|---|---|
| ContextAggregator | 🔥 train |
| CompactToken embedding | 🔥 train |
| EBT | 🔥 train |
| Predictor + SurprisePredictor | ❄️ frozen |

```python
class CompactToken(nn.Module):
    """Learned embedding для сигнала компакции."""
    def __init__(self, dim=1024):
        super().__init__()
        self.compact_embedding = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.type_id = 2  # 0=query, 1=answer, 2=compact

    def get_token(self) -> Tensor:
        return self.compact_embedding
```

#### Compaction training loop

```python
# Когда контекст > MAX_CONTEXT:

def compaction_step(pipeline, cache, max_context=200):
    if len(cache) <= max_context:
        return  # Не нужна компакция

    # 1. Выбираем старейшие K векторов
    old_vectors = cache.get_oldest(K=50)
    old_surprises = cache.get_surprise_scores(old_vectors)

    # 2. Разделяем по surprise
    high_surprise = old_vectors[old_surprises > theta]  # Сохраняем как Global
    low_surprise = old_vectors[old_surprises <= theta]   # Суммаризуем

    # 3. Compact prompt: подаём compact_token + low_surprise вектора
    compact_input = torch.cat([
        compact_token.get_token(),  # [1, 1024] — сигнал "сожми это"
        low_surprise               # [M, 1024] — что сжимаем
    ], dim=0)

    # 4. Через пайплайн (IPP → EBT → Langevin) генерируем 2-3 summary-вектора
    summary_vectors = pipeline.generate_summary(compact_input, n_summaries=3)

    # 5. Заменяем старые вектора на summary + сохранённые globals
    cache.replace_range(old_vectors, summary_vectors, preserved_globals=high_surprise)

    # Loss: качество ответов ДО компакции ≈ ПОСЛЕ компакции
    # loss_compaction = |cos_sim(answer_before, target) - cos_sim(answer_after, target)|
```

#### Training: привыкание к compact

```
Этап C.1 (первые дни): Модель видит compact_token, но компакция не обязательна.
  Подаём compact_token в 30% батчей, остальные — обычные.
  Модель учится: compact_token → "нужно сжать контекст"

Этап C.2: Принудительная компакция на длинных контекстах.
  Контексты 200+ ходов → trigger compaction → продолжение диалога
  Loss: quality_after_compaction ≈ quality_before_compaction

Этап C.3: Итеративная компакция.
  Контексты 500+ ходов → несколько раундов compaction
  Модель учится делать 2-3 compaction подряд без деградации
```

### 7.4 Фаза D: Hard negatives + полный масштаб (1–2 недели)

| Компонент | Статус |
|---|---|
| ContextAggregator | 🔥 train |
| EBT (Pairwise + Chain) | 🔥 train |
| CompactToken | 🔥 train |
| Decoder LM | 🔥 train |
| Predictor + SurprisePredictor | ❄️ frozen |

**Полный curriculum:**
```
1. Длинные контексты: 500 → 1000 → 5000+ ходов
2. Hard negatives: cos_sim 0.9–0.98
3. Компакция: автоматическая при превышении MAX_CONTEXT
4. System 1 + System 2: switching (30/70)
5. Batch contrastive: 1 positive + 31 negatives, полный пайплайн
6. Momentum: cruise_ratio sampling [0.0, 0.3, 0.5, 0.7]
```

**Весь пайплайн проходит через все нагрузки:**
```
SurprisePredictor (frozen) → ContextAggregator → Predictor (frozen) →
→ EBT + Langevin (System 1/2) → Decoder LM → answer text

На каждом батче:
  - Random thinking_mode (System 1 / System 2)
  - Random cruise_ratio
  - Random context length (50 → 5000)
  - Compaction если context > MAX
  - Batch contrastive с hard negatives
```

### 7.5 Фаза E: End-to-end fine-tune (1–2 недели)

**Размораживаем всё кроме SONAR и SurprisePredictor:**

| Компонент | Статус | Комментарий |
|---|---|---|
| SONAR | ❄️ frozen | Никогда не трогаем |
| SurprisePredictor | ❄️ frozen | Навсегда frozen |
| ContextAggregator | 🔥 train | joint fine-tune |
| IPP | 🔥 train | Продолжает дотренировываться |
| EBT (Pairwise + Chain) | 🔥 train | joint fine-tune |
| CompactToken | 🔥 train | joint fine-tune |
| Decoder LM | 🔥 train | joint fine-tune |

**End-to-end loss:**
```python
def e2e_loss(pipeline, query, target_answer, context, negatives):
    """
    Полный loss через весь пайплайн.
    Каждый gradient flow проходит: ContextAggregator → IPP → EBT → ответ.
    """
    # Forward pass через весь пайплайн
    V_answer, metadata = pipeline.full_forward(query, context)

    # 1. Answer quality loss
    loss_answer = 1 - F.cosine_similarity(V_answer, target_answer, dim=-1).mean()

    # 2. EBT contrastive loss (batch)
    loss_contrastive = batch_contrastive_loss(
        pipeline.ebt, query, target_answer, negatives
    )

    # 3. Decoder cross-entropy loss
    decoded_logits = pipeline.decoder(V_answer)
    loss_decoder = F.cross_entropy(decoded_logits, target_text_tokens)

    # 4. Compaction quality loss (если была компакция в этом forward)
    if metadata.get("compacted"):
        loss_compact = compaction_quality_loss(
            answer_before=metadata["answer_before_compact"],
            answer_after=V_answer,
            target=target_answer
        )
    else:
        loss_compact = 0.0

    return loss_answer + loss_contrastive + loss_decoder + 0.5 * loss_compact
```

**Все режимы тренировки в Phase E:**
```
✅ Curriculum: easy → medium → hard negatives
✅ Batch contrastive: 1 positive + N negatives
✅ Short contexts (5-20 ходов) + Long contexts (500-5000)
✅ External Compaction + итеративная компакция
✅ System 1 (fast shot) + System 2 (deep thinking)
✅ Momentum: cruise_ratio ∈ [0.0 .. 0.7]
✅ Decoder: cross-entropy через весь пайплайн
✅ Global Tokens: surprise-aware attention
```

### 7.6 Тестирование Stage 4

| Тест | Метрика | Ожидание |
|---|---|---|
| QA без контекста | cos_sim + ROUGE-L | >0.7 cos_sim |
| QA с коротким контекстом (20 ходов) | cos_sim + ROUGE-L | Лучше чем без контекста |
| QA с длинным контекстом (500+ ходов) | cos_sim + ROUGE-L | Не хуже чем с коротким |
| QA после компакции | cos_sim degradation | <5% деградации |
| System 2 vs System 1 | Quality gap | System 2 > System 1 на hard примерах |
| Momentum robustness | Quality @ cruise_ratio=0.5 | <10% деградации vs cruise_ratio=0.0 |
| Global Token recall | Модель помнит high-surprise info через 100+ ходов | Качественный тест |

### 7.7 Decoder тренировка (Stage 4)

Fine-tune decoder LM на **всех** типах outputs:
1. System 1 outputs (fast, с контекстом)
2. System 2 outputs (deep, с цепочками)
3. Post-compaction outputs
4. Different cruise_ratios outputs
5. Cross-entropy vs ground truth text

---

## 8. Stage 5: Projectors (если 1024d недостаточно)

**Trigger:** kill criterion из Stage 2–4 (модель не может различать достаточно тонкие нюансы в 1024d).

**Подход:**
1. Тренируем Sparse Projector (1024d → Nd) + Deprojector (Nd → 1024d) как autoencoder
2. Reconstruction loss: minimize cos_sim(V, Deproj(Proj(V)))
3. Затем fine-tune всей системы в Nd пространстве
4. EBT, IPP, ContextAggregator работают в Nd
5. Только SONAR endpoints остаются в 1024d

**Это Stage реализуется только если предыдущие Stage показали, что 1024d мало.**

---

## 9. Сводная таблица: что замораживаем, что тренируем

| Компонент | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4A-D | Stage 4E |
|---|---|---|---|---|---|---|
| SONAR encoder | — | ❄️ | ❄️ | ❄️ | ❄️ | ❄️ |
| SONAR decoder | — | ❄️ | ❄️ | ❄️ | ❄️ | ❄️ |
| SimpleEnergy | — | 🔥 | — | — | — | — |
| EBT Pairwise | — | — | 🔥 | ❄️→🔥 | 🔥 | 🔥 |
| EBT Chain | — | — | — | 🔥 | 🔥 | 🔥 |
| SurprisePredictor | — | — | 🔥‖ | 🔥‖ | ❄️ | ❄️ |
| IPP | — | — | 🔥‖ | 🔥‖ | 🔥 | 🔥 |
| ContextAggregator | — | — | — | — | 🔥 | 🔥 |
| CompactToken | — | — | — | — | 🔥 | 🔥 |
| Decoder LM | — | ❄️ | 🔥† | 🔥† | 🔥† | 🔥 |

**Легенда:**
- ❄️ = frozen
- 🔥 = training (в пайплайне)
- 🔥‖ = training (параллельный трек, вне пайплайна)
- 🔥† = fine-tune в конце Stage
- 🔥* = размораживается только если нужно

---

## 10. Логирование и метрики

### 10.1 Wandb Integration

Все стадии логируются:
- **Training:** loss (total, contrastive, decoder, compact), accuracy, gradient norms
- **Evaluation:** cosine similarity, ROUGE-L, BLEU, energy distributions
- **Langevin:** trajectory visualization, convergence speed
- **System 2:** chain quality scores, chain length vs quality
- **Compaction:** quality before/after, compression ratio

### 10.2 Checkpointing

```python
# Каждые 10 эпох + при смене curriculum phase:
torch.save({
    "stage": stage, "phase": phase, "epoch": epoch,
    "model_states": {name: model.state_dict() for name, model in models.items()},
    "optimizer_states": {name: opt.state_dict() for name, opt in optimizers.items()},
    "config": config,
    "metrics": metrics,
    "curriculum_state": curriculum.state_dict(),
}, f"checkpoints/stage{stage}_{phase}_epoch{epoch}.pt")
```

---

## 11. Риски и Plan B

| Риск | Вероятность | Что делаем |
|---|---|---|
| SONAR decoder ломается на Langevin outputs | Средняя | Fine-tune decoder LM (заложено в каждый Stage) |
| 1024d недостаточно для QA | Средняя | Stage 5: Projectors |
| SurprisePredictor плохо скалируется | Низкая | Простая модель, SSM-based, проблем быть не должно |
| Chain Head не даёт улучшения | Средняя | Вернуться к Pairwise-only, усилить curriculum |
| Compact не сохраняет информацию | Средняя | Увеличить число summary-векторов, уменьшить compression ratio |
| End-to-end training нестабилен | Высокая | Gradient clipping, раздельные lr для компонентов, warm-up |
| Batch contrastive не помогает System 2 | Низкая | Увеличить batch size, добавить harder negatives |

---

## 12. Timeline (оценочный)

| Stage | Длительность | Блокеры |
|---|---|---|
| Stage 0: SONAR Validation | 3–4 дня | Установка SONAR |
| Stage 1: Denoising PoC | 2–3 дня | Stage 0 |
| Stage 2: EBT Pairwise | 2–4 недели | Stage 1, данные |
| Stage 2‖: SurprisePredictor + Predictor | 1–2 недели (параллельно) | Данные |
| Stage 3: Chain of Thought | 3–5 недель | Stage 2 |
| Stage 4A: Привыкание к Global Tokens | 1–2 недели | Stage 3, Stage 2‖ |
| Stage 4B: Масштабирование контекста | 1–2 недели | Stage 4A |
| Stage 4C: Compact Prompt | 1–2 недели | Stage 4B |
| Stage 4D: Hard negatives + масштаб | 1–2 недели | Stage 4C |
| Stage 4E: End-to-end fine-tune | 1–2 недели | Stage 4D |
| **Итого до первых QA результатов** | **~5–7 недель** | |
| **Итого до полного пайплайна** | **~15–22 недели** | |

---

*Этот план — живой документ. Обновляется по мере экспериментов и результатов.*
