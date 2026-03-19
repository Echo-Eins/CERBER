# CEBCM — План реализации v1 (Proof of Concept)

**Дата:** 18 марта 2026
**Статус:** Обсуждение → Утверждение
**Scope:** Milestone 1–2 (валидация SONAR + denoising + QA PoC) + скелет полного пайплайна
**Hardware:** 1x consumer GPU (RTX 3090/4090, 24GB VRAM)

---

## 0. Принятые решения

| Параметр | Решение | Обоснование |
|---|---|---|
| Autoencoder | SONAR (замороженный, 1024d) | Нулевая стоимость обучения, готовый decoder |
| Инициализация | Оба: чистый шум + informed noise | Сравним, чтобы понять реальную силу EBT |
| Обучение EBT | Curriculum learning с самого начала | Не экономим на архитектуре обучения |
| Датасет denoising | WikiText-103 → SONAR vectors | Простой корпус предложений для sanity check |
| Датасет QA | SQuAD v2 → SONAR vectors | ~100K QA-пар, быстро скачать/закодировать |
| Inertial Navigation | Гиперпараметр `cruise_ratio` + train-time sampling | Модель тренируется в тех же режимах, в которых работает |
| Кэш контекста | Multi-answer: V_query → [V_answer₁, V_answer₂, ...] | Реальный диалог ≠ 1:1 |
| Агрегация контекста | PoC: MHLA-inspired compression. Scale: Hierarchical Sparse (window + retrieval + global) | PoC: простота. Scale: O(N×200) вместо O(N²) при 50K–100K предложений |
| Attention backend | PyTorch SDPA (автоматический FlashAttention-2) + FlexAttention | Встроен в PyTorch 2.x, нулевые доп. зависимости, FA2 на Ampere GPUs |
| Линейный attention (Predictor) | GLA / Gated DeltaNet через FLA library (опционально) | Лучшее качество среди линейных моделей; не подходит для EBT (нужен bidirectional) |

---

## 1. Файловая структура проекта

```
CERBER/
├── CEBCM_Technical_Specification.md   # Спецификация (уже есть)
├── IMPLEMENTATION_PLAN.md             # Этот файл
├── README.md                          # Краткое описание + quick start
│
├── configs/                           # Конфигурации (dataclass + yaml)
│   ├── __init__.py
│   ├── base.py                        # @dataclass Config с дефолтами
│   ├── sonar.yaml                     # Параметры SONAR encoder/decoder
│   ├── ebt.yaml                       # Параметры EBT (dims, heads, layers)
│   ├── training.yaml                  # lr, batch_size, curriculum schedule
│   └── inference.yaml                 # Langevin params, cruise_ratio, presets
│
├── cebcm/                            # Основной пакет
│   ├── __init__.py
│   ├── models/                        # Все модели
│   │   ├── __init__.py
│   │   ├── energy.py                  # EBT_PairwiseHead, EBT_ChainHead, EBT
│   │   ├── predictor.py               # SimplePredictor (MLP)
│   │   ├── context.py                 # ContextCache, ContextAggregator, HierarchicalContextAggregator
│   │   ├── faiss_index.py             # FAISSIndexManager для семантического retrieval (Level 2)
│   │   └── sonar_wrapper.py           # Обёртка над SONAR encoder/decoder
│   │
│   ├── training/                      # Обучение
│   │   ├── __init__.py
│   │   ├── losses.py                  # InfoNCE, margin loss, gradient penalty
│   │   ├── curriculum.py              # CurriculumScheduler (easy→medium→hard)
│   │   ├── negatives.py               # Генерация негативов через FAISS
│   │   ├── train_ebt.py               # Скрипт обучения EBT
│   │   └── train_predictor.py         # Скрипт обучения Predictor
│   │
│   ├── inference/                     # Инференс
│   │   ├── __init__.py
│   │   ├── langevin.py                # Langevin dynamics (с inertial navigation)
│   │   ├── pipeline.py                # Полный pipeline: query → answer text
│   │   └── modes.py                   # InferenceMode presets (fast/balanced/deep/extra_deep)
│   │
│   └── data/                          # Работа с данными
│       ├── __init__.py
│       ├── encode_dataset.py          # WikiText/SQuAD → SONAR vectors
│       ├── dataset.py                 # PyTorch Dataset/DataLoader
│       └── utils.py                   # Утилиты (нормы, статистики, визуализация)
│
├── experiments/                       # Эксперименты (скрипты + результаты)
│   ├── 01_sonar_validation/           # Фаза 1: валидация пространства
│   │   ├── run_noise_test.py
│   │   ├── run_interpolation_test.py
│   │   └── run_distribution_analysis.py
│   │
│   ├── 02_denoising_poc/              # Фаза 3: denoising sanity check
│   │   ├── train_denoiser.py
│   │   └── eval_denoising.py
│   │
│   └── 03_qa_poc/                     # Фаза 4: QA с EBT
│       ├── train_qa_ebt.py
│       └── eval_qa.py
│
├── tests/                             # Юнит-тесты
│   ├── test_energy.py
│   ├── test_langevin.py
│   ├── test_curriculum.py
│   └── test_context_cache.py
│
├── requirements.txt
└── setup.py
```

---

## 2. Фазы реализации

### Фаза 1: Инфраструктура + Валидация SONAR (3–4 дня)

**Цель:** убедиться, что SONAR-пространство пригодно для градиентной навигации, и заложить скелет проекта.

#### 1.1 Инфраструктура (день 1)

- [ ] Создать файловую структуру проекта (все директории и `__init__.py`)
- [ ] Реализовать конфиг-систему (`configs/base.py` — `@dataclass` с вложенными конфигами)
- [ ] `requirements.txt` с пинами версий
- [ ] `cebcm/models/sonar_wrapper.py` — обёртка для SONAR:
  ```python
  class SONARWrapper:
      def __init__(self, device="cuda"):
          ...
      def encode(self, texts: list[str], lang="eng_Latn") -> Tensor:  # [N, 1024]
      def decode(self, vectors: Tensor, lang="eng_Latn") -> list[str]:
      def encode_batched(self, texts, lang, batch_size=64) -> Tensor:
  ```
- [ ] Проверить VRAM: encoder + decoder + EBT на одной GPU

#### 1.2 Валидация SONAR (дни 2–3)

**Эксперимент A: Устойчивость к шуму**
```
Для noise_scale в [0.01, 0.05, 0.1, 0.2, 0.5]:
  1. Закодировать 100 предложений
  2. Добавить шум
  3. Декодировать
  4. Измерить: cosine_sim(V, V_noisy), семантическое сходство decoded текстов
```
**Kill criterion:** при noise_scale > 0.05 decoded текст полностью теряет смысл → SONAR непригоден.

**Эксперимент B: Интерполяция**
```
Для 50 пар предложений, alpha в [0.0, 0.25, 0.5, 0.75, 1.0]:
  1. V_interp = (1-alpha) * V_a + alpha * V_b
  2. Декодировать V_interp
  3. Проверить: осмысленность промежуточных предложений
```

**Эксперимент C: Анализ распределения** (критично для OOD protection)
```
Для 10,000 предложений из WikiText:
  1. Закодировать в SONAR
  2. Измерить: mean(||V||), std(||V||), min/max нормы
  3. Визуализировать распределение норм (histogram)
  4. Измерить среднюю cosine similarity между случайными парами
  5. PCA/t-SNE визуализация (для понимания геометрии)
```
Эти статистики нужны для:
- `target_norm` в sphere projection при Langevin
- Порогов для OOD detection
- Калибровки `noise_scale` для informed noise

**Эксперимент D: Градиенты через SONAR decoder**
```
V = encoder("test sentence").requires_grad_(True)
output = decoder(V)  # Проверяем, что decoder дифференцируем по V
grad = torch.autograd.grad(output_loss, V)  # Проверяем наличие градиента
```
Это критично: если SONAR decoder не пропускает градиенты, нужен workaround.

> **ВАЖНО:** Для наших целей градиенты через decoder НЕ нужны напрямую.
> Langevin считает ∇_V E(V_query, V), где E — это EBT (наша MLP).
> EBT полностью дифференцируема по V. Decoder используется только
> для финальной визуализации результата. Но проверить стоит на будущее.

#### 1.3 GO/NO-GO Decision (день 3–4)

Документируем результаты. Если SONAR проходит — переходим к Фазе 2.

---

### Фаза 2: Подготовка данных (2–3 дня)

**Цель:** закодировать датасеты в SONAR-вектора, подготовить негативы для curriculum learning.

#### 2.1 WikiText для denoising (день 1)

- [ ] `cebcm/data/encode_dataset.py`:
  ```python
  def encode_wikitext(num_sentences=10000, lang="eng_Latn") -> Tensor:
      # 1. Загрузить WikiText-103 через datasets
      # 2. Разбить на предложения (sent_tokenize)
      # 3. Отфильтровать: длина 5–50 слов
      # 4. Закодировать батчами через SONAR
      # 5. Сохранить: wikitext_vectors.pt + wikitext_texts.json
  ```
- [ ] Замерить время кодирования (ожидание: ~10-20 мин на 10K при batch=64)

#### 2.2 SQuAD v2 для QA (день 1–2)

- [ ] Загрузить SQuAD v2, отфильтровать пары с ответами (~87K пар)
- [ ] Закодировать вопросы и ответы отдельно:
  ```
  V_questions: [87000, 1024]
  V_answers: [87000, 1024]
  questions_text: list[str]
  answers_text: list[str]
  ```
- [ ] Сохранить: `squad_vectors.pt`

#### 2.3 Генерация негативов для curriculum (день 2–3)

- [ ] `cebcm/training/negatives.py`:
  ```python
  class NegativeGenerator:
      def __init__(self, V_questions, V_answers):
          # Строим FAISS-индекс по V_questions
          self.index = faiss.IndexFlatIP(1024)
          self.index.add(F.normalize(V_questions, dim=-1).numpy())

      def get_negatives(self, query_idx, level="easy", num_neg=31):
          # easy:   cosine sim 0.3–0.7 (случайные из корпуса)
          # medium: cosine sim 0.7–0.9 (похожие вопросы, их ответы)
          # hard:   cosine sim 0.9–0.98 (очень похожие вопросы)
          # Возвращаем V_answers соответствующих вопросов
  ```

- [ ] **Проблема масштаба:** полная sim-матрица 87K×87K = 30GB float32.
  **Решение:** FAISS top-K поиск. Для каждого вопроса ищем 200 ближайших,
  потом фильтруем по порогам similarity.

- [ ] Предрассчитать и сохранить индексы негативов:
  ```
  negatives_easy.pt:   [87000, 10] — индексы easy-негативов
  negatives_medium.pt: [87000, 10] — индексы medium-негативов
  negatives_hard.pt:   [87000, 11] — индексы hard-негативов
  ```

- [ ] `cebcm/data/dataset.py`:
  ```python
  class EBTDataset(Dataset):
      """Возвращает (V_query, V_positive, V_negatives) с учётом curriculum phase."""
      def __init__(self, vectors_path, negatives_path, phase="easy"):
          ...
      def set_phase(self, phase: str):
          """Переключение curriculum phase."""
          ...
  ```

---

### Фаза 3: Denoising PoC — Sanity Check (2–3 дня)

**Цель:** быстро проверить, что энергетическая функция + Langevin dynamics в принципе работают в SONAR-пространстве. Не задерживаемся.

#### 3.1 SimpleEnergy для denoising

- [ ] `cebcm/models/energy.py` — начальная версия:
  ```python
  class SimpleEnergy(nn.Module):
      """Простая MLP: оценивает 'расстояние' между двумя векторами."""
      def __init__(self, dim=1024, hidden=2048):
          # Input: [V_orig; V_candidate; V_orig-V_candidate; V_orig*V_candidate] = 4096d
          # Output: scalar energy
  ```

- [ ] `cebcm/training/losses.py`:
  ```python
  def margin_contrastive_loss(E_pos, E_neg, margin=1.0):
      """L = ReLU(E_pos - E_neg + margin).mean()"""

  def infonce_loss(energies_pos, energies_neg, temperature=0.07):
      """Standard InfoNCE: -log(exp(-E_pos/τ) / Σ exp(-E_i/τ))"""
  ```

#### 3.2 Обучение denoising energy

- [ ] `experiments/02_denoising_poc/train_denoiser.py`:
  - Данные: 10K WikiText SONAR-векторов
  - Позитив: (V, V) — идентичная пара
  - Негатив: (V, V + noise) с noise_scale из [0.1, 0.2, 0.3]
  - Loss: margin contrastive
  - 50 эпох, batch=32, ~10 мин на GPU

#### 3.3 Langevin denoising loop

- [ ] `cebcm/inference/langevin.py`:
  ```python
  class LangevinDynamics:
      def __init__(self, energy_fn, config: LangevinConfig):
          self.energy_fn = energy_fn
          self.lr = config.lr                      # 0.01
          self.noise_scale = config.noise_scale    # sqrt(2*lr)
          self.max_steps = config.max_steps        # 100
          self.threshold = config.threshold        # early stopping
          self.cruise_ratio = config.cruise_ratio  # 0.0 for denoising PoC
          self.target_norm = config.target_norm    # from SONAR distribution analysis

      def refine(self, V_init, V_query, return_trajectory=False):
          """
          Langevin refinement loop.
          Returns: V_refined, metadata (энергия по шагам, trajectory если нужно)
          """
          V = V_init.clone().requires_grad_(True)
          trajectory = [V.detach().clone()] if return_trajectory else None
          energies = []

          for step in range(self.max_steps):
              E = self.energy_fn(V_query, V)
              energies.append(E.item())

              if E.item() < self.threshold:
                  break

              grad = torch.autograd.grad(E, V, create_graph=False)[0]
              V = (V - self.lr * grad + self.noise_scale * torch.randn_like(V))
              V = V.detach().requires_grad_(True)

              # OOD protection: sphere projection
              V.data = F.normalize(V.data, dim=-1) * self.target_norm

              if return_trajectory:
                  trajectory.append(V.detach().clone())

          return V.detach(), {"energies": energies, "steps": len(energies),
                              "trajectory": trajectory}
  ```

#### 3.4 Тестирование denoising

- [ ] `experiments/02_denoising_poc/eval_denoising.py`:
  ```
  Для 100 тестовых векторов:
    1. V_noisy = V_orig + noise (noise_scale=0.2)
    2. V_denoised = langevin.refine(V_noisy, V_orig)
    3. Метрики:
       - cos_sim(V_orig, V_noisy)  vs  cos_sim(V_orig, V_denoised)
       - SONAR decode обоих, сравнение текстов
    4. Дополнительно: тест с чистым шумом (V_init = randn)
       - cos_sim(V_orig, V_random) vs cos_sim(V_orig, V_denoised_from_random)
  ```

**Kill criterion:** если denoised НЕ ближе к оригиналу, чем noisy → энергетическая функция не работает.

**Ожидание:** denoising должен работать тривиально. Если нет — проблема фундаментальная.

---

### Фаза 4: EBT на QA-парах — Настоящий тест теории (1–2 недели)

**Цель:** обучить EBT различать правильные и неправильные ответы, затем проверить, может ли Langevin навигировать из шума к правильному ответу.

#### 4.1 EBT с Pairwise Head

- [ ] `cebcm/models/energy.py` — полная версия:
  ```python
  class EBT_PairwiseHead(nn.Module):
      """
      MLP: [V_q; V_c; V_q-V_c; V_q*V_c] → scalar energy.
      Spectral normalization на всех Linear слоях.
      """
      def __init__(self, dim=1024, hidden_dims=[2048, 1024]):
          # Все Linear обёрнуты в spectral_norm()
          ...

  class EBT(nn.Module):
      def __init__(self, dim=1024):
          self.pairwise = EBT_PairwiseHead(dim)
          # chain head — скелет, реализуем позже
          # self.chain = EBT_ChainHead(dim)

      def energy(self, V_query, V_candidate) -> Tensor:
          return self.pairwise(V_query, V_candidate)
  ```

#### 4.2 Curriculum Learning

- [ ] `cebcm/training/curriculum.py`:
  ```python
  class CurriculumScheduler:
      """
      Управляет переключением фаз обучения.

      phases:
        - easy:   epochs 0–19   (20%), cos_sim 0.3–0.7
        - medium: epochs 20–49  (30%), cos_sim 0.7–0.9
        - hard:   epochs 50–99  (50%), cos_sim 0.9–0.98
      """
      def __init__(self, total_epochs=100, phase_ratios=[0.2, 0.3, 0.5]):
          ...

      def get_phase(self, epoch: int) -> str:
          ...

      def get_negatives_config(self, epoch: int) -> dict:
          """Возвращает параметры для NegativeGenerator."""
          ...
  ```

#### 4.3 Обучение EBT

- [ ] `cebcm/training/train_ebt.py`:
  ```python
  # Основной training loop
  # 1. InfoNCE loss с temperature=0.07
  # 2. + gradient penalty (λ_grad * ||∇_V E||²)
  # 3. + norm penalty (λ_norm * max(0, ||E||² - margin))
  # 4. Spectral norm уже в модели
  # 5. CurriculumScheduler переключает phase каждые N эпох
  # 6. wandb logging: loss, accuracy (E_pos < E_neg), gradient norms
  # 7. Checkpointing каждые 10 эпох
  ```

- [ ] Гиперпараметры (начальные, подлежат тюнингу):
  ```yaml
  training:
    lr: 1e-4
    weight_decay: 0.01
    batch_size: 256
    num_negatives: 31
    temperature: 0.07
    total_epochs: 100
    gradient_penalty_lambda: 0.1
    norm_penalty_lambda: 0.01
    norm_penalty_margin: 10.0
  ```

#### 4.4 Тестирование QA-генерации (главный эксперимент)

- [ ] `experiments/03_qa_poc/eval_qa.py`:

  **Тест A: Ранжирование (проверяем, что EBT вообще различает ответы)**
  ```
  Для 1000 тестовых QA-пар:
    1. E_correct = EBT(V_question, V_correct_answer)
    2. E_random  = EBT(V_question, V_random_answer)
    3. Метрика: accuracy = (E_correct < E_random).mean()
    Ожидание: > 0.9 (иначе EBT не обучилась)
  ```

  **Тест B: Langevin из informed noise**
  ```
  Для 100 тестовых QA-пар:
    1. V_init = 0.5 * V_query + 0.5 * randn * target_norm_std
    2. V_refined = langevin.refine(V_init, V_query, max_steps=100)
    3. cos_sim(V_refined, V_correct_answer)
    4. Decode V_refined, сравнить с target_text (ROUGE-L, BLEU)
  ```

  **Тест C: Langevin из чистого шума**
  ```
  Для тех же 100 пар:
    1. V_init = randn(1024) * target_norm  (на сфере правильного радиуса)
    2. V_refined = langevin.refine(V_init, V_query, max_steps=500)
    3. Те же метрики
    4. Сравнить с Тестом B
  ```

  **Тест D: Ablation по cruise_ratio**
  ```
  Для 50 пар, cruise_ratio в [0.0, 0.3, 0.5, 0.7]:
    Сравнить: качество ответа vs количество backward passes
  ```

**Критерии успеха Фазы 4:**
- Тест A: accuracy > 0.9
- Тест B: avg cosine sim > 0.6, decoded текст осмысленный
- Тест C: avg cosine sim > 0.4 (ожидаем хуже, чем B — это нормально)
- Тест D: cruise_ratio=0.5 даёт < 10% деградации качества при 2x ускорении

---

## 3. Скелет для будущих компонентов (реализуется параллельно с Фазой 1)

Интерфейсы и заглушки, которые закладываем сразу, но реализуем полноценно позже:

### 3.1 Context Cache (multi-answer)

```python
# cebcm/models/context.py

@dataclass
class CacheSlot:
    query_vector: Tensor          # [1024]
    answer_vectors: list[Tensor]  # [N, 1024] — несколько ответов
    turn_ids: list[int]           # порядок в диалоге
    timestamp: float

class ContextCache:
    """
    Хранит историю диалога как последовательность (query, [answers]).
    Retrieval: top-K по cosine similarity к текущему запросу.
    Возвращает упорядоченные подпоследовательности, не мешанину.
    """
    def __init__(self, max_slots=1000):
        ...

    def add(self, V_query, V_answer, turn_id):
        """Добавляет или обновляет слот (append answer к существующему query)."""
        ...

    def retrieve(self, V_query_new, top_k=5) -> list[CacheSlot]:
        """Возвращает top-K слотов, отсортированных по turn_id (хронологически)."""
        ...

    def to_sequence(self, slots: list[CacheSlot]) -> Tensor:
        """
        Разворачивает слоты в последовательность для attention.
        [V_q1, V_a1_1, V_a1_2, V_q2, V_a2_1, ...] с type embeddings.
        """
        ...
```

### 3.2 Context Aggregator (двухфазная стратегия)

> **Обновлено (v1.2):** По результатам исследования efficient attention (март 2026). Для PoC используется простой MHLA-inspired агрегатор (§9.5 спецификации). Для масштабирования — HierarchicalContextAggregator (§9.7.3 спецификации).

```python
# cebcm/models/context.py

# === Фаза PoC: Простой агрегатор (до ~1000 контекстных векторов) ===

class ContextAggregator(nn.Module):
    """
    MHLA-inspired attention агрегация контекста для Predictor.
    Подходит для PoC (десятки–сотни обменов).
    Для масштабирования на 50K+ используется HierarchicalContextAggregator.
    """
    def __init__(self, dim=1024, n_heads=8, n_layers=2,
                 n_compress_slots=16):
        super().__init__()
        self.type_embedding = nn.Embedding(2, dim)
        self.compress_slots = nn.Parameter(
            torch.randn(1, n_compress_slots, dim)
        )
        self.kv_compressor = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim*2,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

    def forward(self, query, context_seq, type_ids):
        ...  # Реализация при полной сборке пайплайна


# === Масштабирование: Иерархический агрегатор (50K–100K предложений) ===
# См. §9.7.3 спецификации — HierarchicalContextAggregator
# Реализуется на Milestone 6 (недели 14–17).
# Стек: PyTorch SDPA + FlexAttention (sliding window) + FAISS (retrieval).
```

### 3.3 Inference Pipeline (скелет)

```python
# cebcm/inference/pipeline.py

class CEBCMPipeline:
    """End-to-end pipeline: text query → text answer."""

    def __init__(self, sonar, predictor, ebt, langevin, context_cache,
                 context_aggregator=None):
        ...

    def answer(self, query_text: str, mode="balanced") -> str:
        """
        1. Encode query через SONAR
        2. Retrieve context из cache
        3. Aggregate context (если aggregator есть)
        4. Predictor → V_init (или informed noise для PoC)
        5. Langevin refinement (с параметрами из mode)
        6. Decode V_answer через SONAR
        7. Cache (V_query, V_answer)
        8. Return text
        """
        ...
```

### 3.4 Inference Modes

```python
# cebcm/inference/modes.py

@dataclass
class InferencePreset:
    name: str
    max_langevin_steps: int
    cruise_ratio: float        # 0.0 = чистый Langevin, 0.7 = агрессивная инерция
    lr: float
    noise_scale: float
    early_stop_threshold: float

PRESETS = {
    "fast":       InferencePreset("fast",       max_langevin_steps=10,  cruise_ratio=0.7, lr=0.05, noise_scale=0.01, early_stop_threshold=0.3),
    "balanced":   InferencePreset("balanced",   max_langevin_steps=50,  cruise_ratio=0.5, lr=0.02, noise_scale=0.02, early_stop_threshold=0.2),
    "deep":       InferencePreset("deep",       max_langevin_steps=200, cruise_ratio=0.2, lr=0.01, noise_scale=0.03, early_stop_threshold=0.1),
    "extra_deep": InferencePreset("extra_deep", max_langevin_steps=500, cruise_ratio=0.0, lr=0.01, noise_scale=0.05, early_stop_threshold=0.05),
}
```

---

## 4. Критические технические решения

### 4.1 Spectral Normalization на EBT

Все Linear слои EBT оборачиваются в `torch.nn.utils.spectral_norm()`. Это:
- Ограничивает Lipschitz-константу каждого слоя до 1
- Гарантирует гладкость энергетического ландшафта
- Предотвращает взрыв градиентов при Langevin dynamics

### 4.2 Gradient Penalty

```python
def gradient_penalty(energy_fn, V_query, V_candidate, lambda_gp=0.1):
    V_candidate.requires_grad_(True)
    E = energy_fn(V_query, V_candidate)
    grad = torch.autograd.grad(E.sum(), V_candidate, create_graph=True)[0]
    penalty = (grad.norm(2, dim=-1) ** 2).mean()
    return lambda_gp * penalty
```

Штраф за слишком большой градиент → Langevin не будет "прыгать" на огромные расстояния.

### 4.3 Sphere Projection (OOD Protection)

После каждого шага Langevin:
```python
V.data = F.normalize(V.data, dim=-1) * target_norm
```

`target_norm` = среднее ||V|| по обучающим векторам (из Фазы 1, Эксперимент C).

Это не позволяет Langevin уйти в OOD-зону, где decoder SONAR выдаст мусор.

### 4.4 Inertial Navigation с train-time awareness

При обучении EBT каждый батч обрабатывается с **случайным** `cruise_ratio`:
```python
# В training loop:
cruise_ratio = random.choice([0.0, 0.0, 0.3, 0.5])  # bias к 0.0
# Прогоняем Langevin с этим cruise_ratio для генерации training trajectories
# EBT видит как "чистые", так и "инерционные" траектории
```

Это гарантирует, что EBT обучается оценивать вектора, до которых можно добраться
разными стратегиями навигации, а не только идеальным полным градиентным спуском.

> **Замечание:** для Фазы 4 (QA PoC) этот механизм реализуем в упрощённом виде —
> EBT обучается на статических парах, не на траекториях. Train-time trajectory
> sampling — для полной версии (Milestone 3+).

---

### Фаза 5: Масштабирование контекста (после PoC, 2–3 недели)

> **Добавлено (v1.2):** На основе исследования efficient attention. Реализуется после успешного завершения Фазы 4 (QA PoC).

**Цель:** масштабировать ContextAggregator до 50K–100K предложений на одной RTX 4090.

#### 5.1 Hierarchical Context Aggregator

- [ ] `cebcm/models/context.py` — `HierarchicalContextAggregator`:
  - Level 1: Sliding window attention (w=128) через FlexAttention
  - Level 2: FAISS HNSW retrieval (K=64)
  - Level 3: Global compressed slots (S=32)
  - Использует `torch.nn.functional.scaled_dot_product_attention` (автоматический FA2)

- [ ] Интеграция с `CEBCMPipeline`:
  ```python
  # В pipeline.py: автоматический выбор агрегатора
  if len(context_vectors) > 1000:
      aggregator = self.hierarchical_aggregator
  else:
      aggregator = self.simple_aggregator
  ```

#### 5.2 FAISS Index Management

- [ ] `cebcm/models/faiss_index.py`:
  ```python
  class FAISSIndexManager:
      """Управляет FAISS индексом для семантического retrieval."""
      def __init__(self, dim=1024, use_gpu=True):
          self.index = faiss.IndexHNSWFlat(dim, 32)  # HNSW с 32 связями
          if use_gpu:
              self.index = faiss.index_cpu_to_gpu(
                  faiss.StandardGpuResources(), 0, self.index
              )

      def add(self, vectors: Tensor):
          """Добавляет вектора в индекс."""

      def search(self, query: Tensor, k: int) -> tuple[Tensor, Tensor]:
          """Возвращает top-K ближайших. O(log N)."""
  ```

#### 5.3 Эксперименты с линейным attention

- [ ] Опциональный эксперимент: GLA/Gated DeltaNet как backbone Predictor:
  ```python
  # Через FLA library (pip install fla-core)
  from fla.layers import GatedLinearAttention

  class LinearPredictor(nn.Module):
      """Predictor с GLA вместо standard attention."""
      def __init__(self, dim=1024, n_layers=4, n_heads=8):
          ...
  ```
  Сравнить с Transformer-Predictor по: качество (cosine sim), скорость, memory.

#### 5.4 Бенчмаркинг

- [ ] Тест масштабирования:
  ```
  Для N в [1K, 5K, 10K, 50K, 100K]:
    Замерить: latency (ms), VRAM (MB), quality (cosine sim)
    Для каждого подхода:
      - Full attention (baseline, до OOM)
      - Hierarchical sparse (window=128, K=64, S=32)
      - Только sliding window (без retrieval)
      - Только retrieval (без window)
  ```

- [ ] Ablation по гиперпараметрам:
  ```
  window_size: [64, 128, 256]
  n_retrieved: [16, 32, 64, 128]
  n_global_slots: [8, 16, 32, 64]
  ```

---

## 5. Логирование и метрики

### 5.1 Wandb Integration

Все эксперименты логируются в Weights & Biases:
- **Training:** loss, accuracy, gradient norms, learning rate
- **Evaluation:** cosine sim, ROUGE-L, BLEU, energy distribution
- **Langevin:** trajectory visualization (энергия по шагам)

### 5.2 Checkpointing

```python
# Каждые 10 эпох:
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
    "config": config,
    "metrics": metrics,
}, f"checkpoints/ebt_epoch_{epoch}.pt")
```

---

## 6. Риски для PoC и План B

| Риск | Вероятность | Что делаем |
|------|-------------|------------|
| SONAR не пропускает градиенты через decoder | Низкая | Нам не нужно — EBT дифференцируема. Но проверим |
| SONAR decoder ломается на Langevin-точках | Средняя | Sphere projection. Если не помогает — fine-tune decoder (Фаза 3 спеки) |
| 1024d недостаточно для QA | Средняя | Начинаем с простых QA (SQuAD). Если не хватает — переход к Варианту B (свой autoencoder) |
| FAISS на 87K не находит достаточно hard negatives | Низкая | Увеличить поиск до top-500, или добавить MS MARCO |
| Curriculum не даёт преимущества над простым contrastive | Средняя | Запускаем ablation: с curriculum vs без. Держим обе версии |
| Langevin из чистого шума не сходится | Высокая | Ожидаемо хуже informed noise. Главное — informed noise работает. Чистый шум = bonus |

---

## 7. Timeline (оценочный)

| Фаза | Длительность | Блокеры |
|------|-------------|---------|
| Фаза 1: Инфраструктура + SONAR | 3–4 дня | Установка SONAR/fairseq2 |
| Фаза 2: Данные | 2–3 дня | Кодирование 87K пар (~2-3 часа GPU) |
| Фаза 3: Denoising PoC | 2–3 дня | Фазы 1-2 |
| Фаза 4: EBT QA | 7–14 дней | Фазы 1-3, время обучения |
| **Итого до первых результатов** | **~3-4 недели** | |
| Фаза 5: Масштабирование контекста | 2–3 недели | Успешная Фаза 4 |
| **Итого до production-ready контекста** | **~6-7 недель** | |

---

*Этот план — живой документ. Обновляется по мере экспериментов.*
