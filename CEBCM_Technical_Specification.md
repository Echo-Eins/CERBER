# CEBCM — Concept-Driven Energy-Based Coding Machine
## Полная техническая спецификация проекта

**Версия:** 1.3
**Дата:** 21 марта 2026
**Статус:** Концептуальное проектирование → Proof of Concept

> **Changelog v1.3 (21.03.2026):**
> - Удалены MHLA-проекторы, Hierarchical Attention, GQA — избыточны для 1024d SONAR
> - Добавлен §9.5 Surprise Mechanism (Google Titans, arXiv:2501.00663)
> - Добавлен §9.7 External Compaction (Context GC)
> - Добавлен §9.9 Linear Attention + YaRN/NTK масштабирование
> - ContextAggregator: Linear Attention (Mamba/GLA) + Surprise Global Tokens
> - Отклонённые подходы вынесены в главу 16
>
> **Changelog v1.2 (19.03.2026):**
> - Добавлен §9.7 «Масштабирование на длинные контексты (50K–100K предложений)»:
>   - HierarchicalContextAggregator (3 уровня: sliding window + FAISS retrieval + global slots)
>   - Token Merging — предварительная дедупликация предложений (2–3× сжатие)
>   - GQA (4:1 ratio) и MLA (low-rank KV compression) для оптимизации attention
>   - Полная оценка memory budget для RTX 4090 (~4 GB из 24 GB)
>   - Efficient attention стек: PyTorch SDPA, FlexAttention, FAISS, FLA
>   - Линейные альтернативы (GLA, Gated DeltaNet) для IPP
> - Позиционное кодирование: ALiBi рекомендован для длинных контекстов (сохраняет геометрию SONAR)
> - Обновлён глоссарий (+15 терминов), Milestone 6, Приложения B и C

---

## Оглавление

1. [Манифест проекта](#1-манифест-проекта)
2. [Глоссарий терминов](#2-глоссарий-терминов)
3. [Глобальная архитектура: Схема «Сэндвич»](#3-глобальная-архитектура-схема-сэндвич)
4. [Слой 1: Латентный фундамент (Autoencoder + Latent Space)](#4-слой-1-латентный-фундамент)
5. [Слой 2: Энергетический критик (EBT — Energy-Based Transformer)](#5-слой-2-энергетический-критик)
6. [Слой 3: Генеративный принтер (Decoder + Adapter)](#6-слой-3-генеративный-принтер)
7. [IPP: Инициализация ответа в латентном пространстве](#7-ipp-инициализация-ответа)
8. [Режимы инференса: Fast Shot и Deep Thinking](#8-режимы-инференса)
9. [Механизм контекста и памяти диалога](#9-механизм-контекста-и-памяти-диалога)
10. [Оптимизация навигации: за пределами градиентного спуска](#10-оптимизация-навигации)
11. [Пайплайн обучения](#11-пайплайн-обучения)
12. [Датасеты и генерация данных](#12-датасеты-и-генерация-данных)
13. [Практическая реализация: Proof of Concept](#13-практическая-реализация-poc)
14. [Риски и митигации](#14-риски-и-митигации)
15. [Roadmap проекта](#15-roadmap-проекта)

---

## 1. Манифест проекта

### 1.1 Проблема

Современные Large Language Models (LLM) работают на уровне отдельных токенов. Каждое слово генерируется последовательно, без возможности вернуться и исправить ранее принятые решения. Это приводит к фундаментальным ограничениям:

- **Потеря нити рассуждения**: ошибка в одном токене каскадно портит весь последующий текст
- **Отсутствие рефлексии**: модель не может «подумать» перед ответом в непрерывном семантическом пространстве
- **Квадратичная сложность**: attention механизм масштабируется как O(N²) по длине контекста
- **Вычислительная расточительность**: Chain of Thought в o1/DeepSeek-R1 тратит токены (деньги и время) на «раздумья» в дискретном текстовом пространстве

### 1.2 Решение: CEBCM

CEBCM (Concept-Driven Energy-Based Coding Machine) — архитектура, которая переносит интеллект с уровня слов (токенов) на уровень смыслов (векторов). Система оперирует целыми мыслями в непрерывном латентном пространстве, используя энергетическую функцию для навигации к оптимальному ответу.

**Ключевые принципы:**
- Рассуждение происходит в непрерывном семантическом пространстве, где концепты имеют геометрические отношения
- Модель не может «опечататься» или потерять согласование времён, потому что EBT оптимизирует структуру всей мысли целиком
- Deep Thinking — это итеративная минимизация энергии, «чистое раздумье» без слов
- Декодер является «принтером» — он не думает, а визуализирует готовый смысл

### 1.3 Связь с JEPA (Yann LeCun)

CEBCM реализует три столпа JEPA (Joint Embedding Predictive Architecture):

1. **Multi-vector latent representations** — аналог joint embeddings JEPA
2. **Energy function** для оценки качества представлений — аналог prediction error в JEPA
3. **Inference-time optimization** через градиентный спуск — аналог planning by energy minimization в JEPA

LeCun: *"Planning is done by optimizing the action sequence to minimize total cost. The sequence can be optimized through gradients since the cost and world model are differentiable."*

CEBCM расширяет JEPA, используя энергетическую функцию для прямой оценки качества контента, а не для оценки предсказания. EBT ближе к energy-based reward model, чем к JEPA predictor.

### 1.4 Метафора системы

| Компонент | Роль | Аналогия |
|-----------|------|----------|
| Autoencoder (SONAR) | Кодирование/декодирование | **Глаза и рот** |
| IPP (Base-LCM) | Черновик ответа | **Интуиция** |
| EBT (Energy Function) | Критик и навигатор | **Логика и совесть** |
| Decoder | Превращение вектора в текст | **Руки** |

---

## 2. Глоссарий терминов

| Термин | Определение |
|--------|-------------|
| **Latent Space** | Непрерывное многомерное пространство, в котором каждая точка соответствует некоторому смыслу. Создаётся автоэнкодером |
| **SONAR** | Мультиязычный sentence-level autoencoder от Meta. Encoder: текст → вектор 1024d. Decoder: вектор → текст на 200+ языках |
| **LCM** | Large Concept Model — модель, работающая с концептами (предложениями-смыслами) вместо токенов |
| **EBT** | Energy-Based Transformer — нейросеть, выдающая скаляр энергии для оценки качества вектора или последовательности векторов |
| **Energy Function** | E(V_query, V_candidate) → скаляр. Низкая энергия = хорошее состояние, высокая = плохое |
| **Langevin Dynamics** | Метод навигации в латентном пространстве: V_{t+1} = V_t − η∇_V E + ε, где ε — стохастический шум для избежания локальных минимумов |
| **Contrastive Learning** | Метод обучения: минимизировать расстояние для позитивных пар, максимизировать для негативных |
| **InfoNCE Loss** | Contrastive loss функция: -log(exp(sim(q,k+)/τ) / Σ exp(sim(q,ki)/τ)) |
| **Hard Negatives** | Негативные примеры с cosine similarity > 0.95 к позитивным. Самые полезные для обучения |
| **OOD (Out-of-Distribution)** | Область латентного пространства, которую декодер никогда не видел при обучении |
| **Sparse Projection** | Проекция вектора из низкой размерности в высокую с L1-регуляризацией для разреженности |
| **CoT (Chain of Thought)** | Цепочка промежуточных рассуждений перед финальным ответом |
| **KV-cache** | Буфер матриц Key и Value из attention слоёв, позволяющий избежать повторных вычислений |
| **RoPE** | Rotary Position Embedding — позиционное кодирование через вращение векторов в подпространствах |
| **GQA** | Grouped Query Attention — оптимизация, где несколько Q-головок разделяют общие K,V матрицы |
| **MoE** | Mixture of Experts — архитектура, где router направляет токен только к 2 из N экспертов (FFN-блоков) |
| **FlashAttention** | IO-aware exact attention, не материализующий N×N матрицу в HBM. Memory O(N) вместо O(N²), compute остаётся O(N²) |
| **FlexAttention** | PyTorch API для произвольных block-sparse масок attention с компилированной эффективностью (sliding window, causal, etc.) |
| **Sliding Window Attention** | Attention только к ±w ближайшим позициям. Complexity O(N×w). Captures локальную когерентность |
| **Ring Attention** | Распределение exact attention по кольцу GPU. Каждый GPU хранит N/num_GPU элементов, KV блоки циркулируют по кольцу |
| **Linear Attention** | Семейство моделей (GLA, Mamba, Gated DeltaNet), заменяющих quadratic attention на линейный recurrent state update O(N) |
| **GLA** | Gated Linear Attention — линейный attention с diagonal data-dependent гейтированием. Библиотека: FLA |
| **Gated DeltaNet** | Комбинация гейтирования и delta rule для targeted state updates. Лучшее качество среди линейных моделей (ICLR 2025) |
| **FAISS** | Facebook AI Similarity Search — библиотека для быстрого поиска ближайших соседей в высокоразмерных пространствах |
| **HNSW** | Hierarchical Navigable Small World — структура данных для approximate nearest neighbor search, O(log N) |
| **SDPA** | Scaled Dot-Product Attention — стандартная реализация в PyTorch с автоматическим выбором backend (FlashAttention, efficient, math) |
| **ALiBi** | Attention with Linear Biases — позиционное кодирование через additive bias −m\|i−j\| к attention scores. Не модифицирует вектора, хорошая экстраполяция |
| **YaRN** | Yet another RoPE extensioN — расширение RoPE для длинных контекстов. Комбинирует linear interpolation, NTK-aware scaling и temperature correction |
| **FIRE** | Functional Interpolation Relative Encoding — learned function f(log\|i−j\|) для позиционных biases, хорошая экстраполяция |
| **MLA** | Multi-head Latent Attention (DeepSeek) — сжимает KV в low-rank latent через down-projection, 8–16× экономия KV-памяти |
| **Token Merging** | Предварительная кластеризация семантически дублирующихся элементов с заменой на weighted centroids. 2–3× сжатие |
| **Surprise Score** | Мера непредсказуемости вектора: S = (1 − cos_sim(V_predicted, V_actual)) / 2. Высокий Surprise → важная информация |
| **SurprisePredictor** | SSM-based модуль, предсказывающий следующий вектор в последовательности. Prediction error = Surprise Score |
| **Global Token** | Вектор с высоким Surprise Score (S > θ), виден всем другим векторам в attention через direct attention |
| **External Compaction** | Программный механизм сжатия контекста: модель генерирует summary-заметки для старых секций диалога |
| **Titans** | Google Research architecture (arXiv:2501.00663): surprise-driven memory для test-time memorization |
| **SSM (State Space Model)** | Класс моделей (Mamba, RWKV) с рекуррентным обновлением скрытого состояния. O(N) time, O(1) memory per step |
| **IPP (Initialization Point Predictor)** | MLP/Transformer модуль, предсказывающий начальную точку (V_init) для навигации Langevin Dynamics |
| **Compact Token** | Обучаемый вектор активации (1024d) для системы External Compaction. Сигнализирует о необходимости сжатия. |
---

## 3. Глобальная архитектура: Схема «Сэндвич»

### 3.1 Обзор

Вся система делится на три независимых, но состыкованных слоя:

```
┌───────────────────────────────────────────────────────────────────────┐
│                          CEBCM Pipeline v1.3                          │
│                                                                       │
│  User Text ──► [SONAR Encoder] ──► V_new (1024d)                      │
│                                        │                              │
│                             ┌──────────┴───────────┐                  │
│                             │  Surprise Predictor  │                  │
│                             │  (SSM: O(N), O(1))   │                  │
│                             │  V̂ = predict(state)  │                  │
│                             │  S = 1-cos(V̂, V_new) │                  │
│                             └──────────┬───────────┘                  │
│                                        │                              │
│                        S > θ? ──► Global Token                        │
│                        S ≤ θ? ──► Normal Token                        │
│                                        │                              │
│                             Context Cache + metadata{S, turn_id}      │
│                                        │                              │
│                             [Context Aggregator]                      │
│                             Linear Attention (Mamba/GLA)              │
│                             + Global Tokens attention                 │
│                                        │                              │
│                              [IPP] ──► V_init (1024d)                 │
│                                        │                              │
│                              ┌───────────────────┐                    │
│                              │   EBT Critic      │                    │
│                              │  Langevin Loop:   │                    │
│                              │  V ← V - η∇E + ε  │                    │
│                              └────────┬──────────┘                    │
│                                       │                               │
│                                V_answer (1024d)                       │
│                                       │                               │
│                                [SONAR Decoder]                        │
│                                       │                               │
│                                 Output Text                           │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Context Cache: [В₁, В₂, ...] + S_scores + turn_ids             │   │
│  │ Retrieval: top-K по cosine similarity + все Global Tokens      │   │
│  │ Compaction: при размере > MAX → summary-заметки через пайплайн │   │
│  └────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
```

### 3.2 Принцип «Projector-Free по максимуму»

Ключевой архитектурный принцип: **минимизация количества проекторов**. Каждый промежуточный проектор — это отдельная модель, точность которой в композиции падает экспоненциально. Идеальный пайплайн работает целиком в одном пространстве.

**Два варианта реализации:**

**Вариант A (PoC — Proof of Concept):** Работа в нативном 1024d пространстве SONAR. Без проекторов вообще. EBT, IPP, контекст — всё в 1024d. Плюс: простота, нулевые потери на проекцию. Минус: возможно, 1024d недостаточно для тонкого различения.

**Вариант B (Target):** Sparse autoencoder, нативно работающий в высокой размерности. Без SONAR. Собственный encoder/decoder. EBT работает в нативном пространстве этого autoencoder. Проекторов нет.

**Вариант C (Промежуточный, если нужно):** SONAR + обученный sparse projector 1024d → Nd и deprojector Nd → 1024d. Наименее желательный вариант.

**Решение для PoC:** Начинаем с Варианта A (нативный SONAR 1024d). Проверяем, хватает ли размерности. Если нет — переходим к Варианту B.

### 3.3 Поток данных по этапам

| Этап | Вход | Операция | Выход | Пространство |
|------|------|----------|-------|--------------|
| 1. Encoding | Текст пользователя | SONAR encoder | V_new | 1024d |
| 2. Surprise | V_new + Predictor state | SurprisePredictor (SSM) | V_new + S_score | 1024d + скаляр |
| 3. Caching | V_new + S_score | Запись в Cache с метаданными | — | 1024d |
| 4. Context | V_query + Cache | Retrieval top-K + все Global Tokens | Context set | 1024d |
| 5. Aggregation | Context set | Linear Attention (Mamba/GLA) | V_context | 1024d |
| 6. Prediction | V_context + V_query | IPP (MLP/Transformer) | V_init | 1024d |
| 7. Refinement | V_init + V_query | EBT + Langevin Dynamics | V_answer | 1024d |
| 8. Decoding | V_answer | SONAR decoder | Текст ответа | — |

---

## 4. Слой 1: Латентный фундамент

### 4.1 Требования к латентному пространству

Для работы EBT латентное пространство должно обладать тремя свойствами:

1. **Гладкость (Smoothness):** маленький сдвиг вектора → маленькое изменение декодированного текста, не катастрофический скачок
2. **Семантическая структура:** направления в пространстве соответствуют осмысленным концептам, иначе градиент ∂E/∂V не указывает в сторону «лучшего смысла»
3. **Покрытие (Coverage):** пространство достаточно плотно заполнено, чтобы после градиентного спуска декодер не оказался в точке, которую никогда не видел

### 4.2 SONAR как фундамент для PoC

**Почему SONAR:**
- Мультиязычный encoder/decoder для 200+ языков
- Единое пространство смыслов — предложения с похожим смыслом на разных языках лежат рядом
- Sentence-level, не token-level — один вектор = одно предложение
- Encoder и decoder уже обучены и заморожены — не нужно тренировать
- Обучался на задаче перевода → пространство структурировано семантически

**Характеристики:**
- Размерность: 1024d (float32)
- Granularity: одно предложение (~10-30 токенов)
- Архитектура encoder: NLLB-1B (encoder part)
- Архитектура decoder: NLLB-1B (decoder part)
- Тренировка: MSE loss на параллельных корпусах

**Ограничения SONAR:**
- Обучался на переводе, не на генерации — декодер может некорректно обрабатывать «синтетические» вектора из Langevin dynamics
- 1024d может быть недостаточно для различения тонких семантических нюансов
- Sentence-level — не работает с sub-sentence или multi-sentence units

### 4.3 Валидация пространства: первый эксперимент

Перед построением всего пайплайна необходимо проверить, что пространство SONAR пригодно для градиентной навигации:

```python
# Эксперимент: устойчивость SONAR к сдвигам
import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

encoder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder"
)
decoder = EmbeddingToTextModelPipeline(
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_encoder"
)

# 1. Закодировать предложение
text = "The cat sat on the mat."
V = encoder.predict([text], source_lang="eng_Latn")  # [1, 1024]

# 2. Добавить шум разной интенсивности
for noise_scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
    noise = torch.randn_like(V) * noise_scale
    V_noisy = V + noise
    decoded = decoder.predict(V_noisy, target_lang="eng_Latn")
    cosine_sim = torch.cosine_similarity(V, V_noisy, dim=-1)
    print(f"noise={noise_scale:.2f}, cos_sim={cosine_sim.item():.4f}")
    print(f"  decoded: {decoded[0]}")
    print()

# 3. Проверить интерполяцию между двумя предложениями
text_a = "I love programming in Python."
text_b = "Machine learning is fascinating."
V_a = encoder.predict([text_a], source_lang="eng_Latn")
V_b = encoder.predict([text_b], source_lang="eng_Latn")

for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    V_interp = (1 - alpha) * V_a + alpha * V_b
    decoded = decoder.predict(V_interp, target_lang="eng_Latn")
    print(f"alpha={alpha:.2f}: {decoded[0]}")
```

**Критерии успеха:**
- При noise_scale ≤ 0.1 декодированный текст сохраняет смысл оригинала
- Интерполяция даёт осмысленные промежуточные предложения
- Косинусное сходство между V и V_noisy коррелирует с семантическим сходством декодированных текстов

### 4.4 Будущее: собственный autoencoder

Если SONAR окажется недостаточным, следующий шаг — обучение собственного autoencoder с нужными свойствами:

- **Архитектура:** Transformer encoder + Transformer decoder с KL-regularization (VAE-стиль) для гладкости
- **Размерность:** 2048d или выше, с L1-штрафом за разреженность (sparse VAE)
- **Обучение:** На корпусе текстов (WikiText, C4, OpenWebText), reconstruction loss + KL + sparsity penalty
- **Требование:** наличие декодера, в отличие от SPLADE/ColBERT, которые имеют только encoder

---

## 5. Слой 2: Энергетический критик (EBT)

### 5.1 Концепция

EBT — «мозг» системы. Он не генерирует текст и не предсказывает следующее слово. Он **чувствует правильность** — оценивает, насколько данный вектор является хорошим ответом на запрос.

**Формально:** E_θ(V_1, V_2, ..., V_n) → ℝ — дифференцируемая нейросеть, принимающая последовательность векторов и возвращающая скаляр (энергию).

### 5.2 Двухрежимная архитектура

EBT имеет общий backbone и два специализированных scoring head'а:

#### Режим A: Парная оценка (Pairwise Scoring)

**Когда:** на каждом шаге Langevin Dynamics (тысячи раз за запрос)

**Вход:** два вектора (V_query, V_candidate)  
**Выход:** скаляр E ∈ ℝ  
**Архитектура:** MLP поверх конкатенации  

```
Input: [V_query; V_candidate; V_query - V_candidate; V_query ⊙ V_candidate]
       ────────────────── 4096d ──────────────────
                          │
                    Linear(4096, 2048)
                         ReLU
                    Linear(2048, 1024)
                         ReLU
                    Linear(1024, 1)
                          │
                       Scalar E
```

**Attention не используется.** Это чистая feedforward оценка двух векторов. Быстро, дёшево — критично для Langevin loop.

#### Режим B: Оценка цепочки (Chain Scoring)

**Когда:** раз в 5–10 шагов при Deep Thinking, для проверки глобальной консистентности CoT

**Вход:** последовательность векторов [V_1, V_2, ..., V_n] (5–20 концептов)  
**Выход:** скаляр E_chain ∈ ℝ  
**Архитектура:** Self-attention Transformer (2–4 слоя) + CLS token  

```
Input: [CLS, V_1, V_2, ..., V_n]  с позиционным кодированием (RoPE)
                    │
          Self-Attention Layer 1
                    │
          Self-Attention Layer 2
                    │
          Вытаскиваем CLS-вектор
                    │
              Linear(1024, 512)
                  ReLU
              Linear(512, 1)
                    │
              Scalar E_chain
```

**Attention здесь нужен** — он позволяет оценить, не противоречит ли шаг 5 шагу 2. Self-attention на 5–20 векторах — мгновенная операция (~0.01ms).

> **Ревью-заметка (v1.2): Позиционное кодирование для разных компонентов**
>
> **Chain Head (5–20 элементов):** RoPE достаточен. Цепочки короткие, порядок шагов критически важен.
>
> **ContextAggregator при масштабировании (50K+ предложений):** Рекомендуется **ALiBi** (Attention with Linear Biases) как основной метод:
> - Добавляет линейный bias −m|i−j| к attention scores (m — slope, уникальный для каждой head)
> - **Критическое преимущество для SONAR:** ALiBi не модифицирует сами вектора (в отличие от RoPE, который вращает Q/K). Для pre-encoded семантических эмбеддингов это принципиально — вращение может нарушить геометрию семантического пространства SONAR
> - Отличная экстраполяция: обучение на N, работа на 50K–100K с плавной деградацией
> - Нулевые learned параметры, тривиальная реализация
> - Естественный recency bias: близкие предложения важнее дальних (что совпадает с интуицией для диалога/документа)
>
> **Альтернативы (если ALiBi недостаточно выразителен):**
> - **FIRE:** learned log-scaled bias function f(log|i−j|), сохраняет экстраполяцию при бóльшей выразительности
> - **YaRN/NTK-aware RoPE:** если backbone pretrained с RoPE, применять только к **выделенному позиционному подпространству** (64–128 dims из 1024d), не вращая остальные измерения
>
> Для PoC (до ~1000 контекстных векторов) стандартный RoPE достаточен.

### 5.3 Обучение: специализированные функции энергии

**Принцип:** Вместо одной универсальной функции энергии создаются несколько специализированных. Каждая обучается на узком, качественном датасете для своей области.

**Специализации:**
- **Text EBT:** обучена на корпусе текстов общего назначения (QA пары, диалоги, summarization)
- **Code EBT:** обучена на корпусе кода с дополнительными сигналами (lint, tests)
- **Multilingual EBT:** обучена на параллельных корпусах для перевода

**Критическое свойство:** функции энергии не зависят от структуры векторного пространства конкретного autoencoder'а. Они оценивают *отношения* между векторами. Если размерность входа совпадает — одна EBT теоретически работает с разными autoencoder'ами (с деградацией из-за разных распределений активаций).

### 5.4 Training Objective: Contrastive Loss / InfoNCE

**Цель:** научить E_θ выдавать низкую энергию для правильных пар (query, answer) и высокую для неправильных.

**Формула InfoNCE с temperature:**

```
L = -log( exp(-E(q, k+) / τ) / Σᵢ exp(-E(q, kᵢ) / τ) )
```

Где:
- q — вектор запроса (anchor)
- k+ — вектор правильного ответа (positive)
- kᵢ — все кандидаты в батче (1 positive + N negatives)
- τ — temperature (обычно 0.05–0.1)

**Батчевое обучение:**
- Размер батча: 1 positive + 31 negatives = 32
- Негативы сравниваются не только с позитивом, но и друг с другом (in-batch negatives)
- Это стандартный подход, работающий только при обучении; при инференсе не нужен

### 5.5 Curriculum Learning для негативов

**Этап 1: Easy Negatives (cosine similarity 0.3–0.7)**
- Случайные предложения из корпуса
- Модель быстро учится различать разные темы
- Длительность: ~20% от общего числа шагов

**Этап 2: Medium Negatives (cosine similarity 0.7–0.9)**
- Предложения из той же области, но с другим смыслом
- Модель учится различать нюансы внутри темы
- Длительность: ~30%

**Этап 3: Hard Negatives (cosine similarity 0.9–0.98)**
- Семантически близкие, но фактически неверные ответы
- Модель учится ловить тонкие ошибки
- Длительность: ~50%

**Фильтрация ложных негативов:**

При cosine similarity > 0.95 среди «негативов» могут оказаться парафразы правильного ответа. Решение:

```python
# Прогоняем top-10 самых "хардовых" негативов через Cross-Encoder
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

for neg in hard_negatives_top10:
    score = cross_encoder.predict([(query_text, neg_text)])
    if score > 0.9:
        # Это парафраз, перекидываем в позитивы
        positives.append(neg)
    else:
        # Настоящий hard negative
        final_negatives.append(neg)
```

### 5.6 Регуляризация пространства энергии

**Проблема:** без регуляризации энергетический ландшафт может стать «рваным», и Langevin dynamics будет вылетать в бесконечность.

**Решения:**

1. **L2-regularization:** штраф за слишком высокую норму выхода EBT
2. **Spectral Normalization:** на всех слоях EBT, ограничивает Lipschitz-константу
3. **Gradient Penalty:** штраф за слишком большой градиент ∇_V E — гарантирует гладкость
4. **Projection на сферу:** после каждого шага Langevin нормализовать вектор: V ← V / ||V|| * r, где r — средний радиус обучающих векторов

```python
# Комбинированная loss для обучения EBT
loss = infonce_loss                          # Основной contrastive loss
     + λ_spec * spectral_norm_penalty        # Гладкость весов
     + λ_grad * gradient_penalty(E, V)       # Гладкость ландшафта
     + λ_norm * max(0, ||E||² - margin)      # Ограничение величины энергии
```

---

## 6. Слой 3: Генеративный принтер (Decoder)

### 6.1 Роль декодера

Декодер — это «принтер». Он не думает, не рассуждает. Он получает готовый вектор из латентного пространства и разворачивает его в читаемый текст. Весь интеллект сосредоточен в EBT; декодер лишь визуализирует результат.

### 6.2 Для PoC: SONAR Decoder (замороженный)

Для первого прототипа используется стандартный SONAR decoder:
- Архитектура: NLLB-1B decoder
- Вход: вектор 1024d
- Выход: текст на целевом языке
- Состояние: замороженный (без обучения)

**Преимущество:** нулевая стоимость обучения. Мы проверяем только работоспособность EBT.

### 6.3 Обучение собственного декодера (после PoC)

**Двухэтапная стратегия:**

**Этап 1: Обучение «мозга» (IPP + EBT)**
- Autoencoder (SONAR) замороженный
- IPP учится предсказывать V_target из V_query
- EBT учится оценивать качество векторов
- Метрика успеха: cosine similarity между сгенерированным V и эталонным V_target

**Этап 2: Обучение декодера (после замораживания «мозга»)**
- IPP + EBT заморожены
- Decoder получает на вход вектора, реально генерируемые системой (не чистые из encoder!)
- Целевой текст известен из датасета
- Loss: cross-entropy между предсказанными и целевыми токенами

```python
# Обучение декодера
optimizer = AdamW(decoder.parameters(), lr=1e-4)

for query_text, target_text in dataset:
    # Генерируем вектор ответа через замороженный пайплайн
    V_query = sonar_encoder(query_text)    # frozen
    V_init = ipp(V_query)            # frozen
    V_answer = langevin_refine(V_init, ebt, V_query, steps=10)  # frozen

    # Обучаем декодер
    logits = decoder(V_answer, target_tokens[:-1])  # teacher forcing
    loss = cross_entropy(logits, target_tokens[1:])
    loss.backward()
    optimizer.step()
```

**Критически важно:** тренировочные вектора должны быть сгенерированы замороженной системой, а не чистыми encoder-векторами. Пропорция: 70–80% сгенерированных + 20–30% чистых из encoder (для устойчивости).

### 6.4 Архитектура декодера

Для предложений (10–30 токенов) достаточно:
- 2–4 Transformer decoder слоя
- Cross-attention к входному вектору (латентное условие)
- 100–300M параметров
- Gated cross-attention (Flamingo-стиль): tanh(α) × cross_attn_output, α=0 при инициализации

---

## 7. IPP (Initialization Point Predictor): Инициализация ответа

### 7.1 Проблема начальной точки

EBT — критик и навигатор. Langevin Dynamics — метод навигации. Но навигации нужна **начальная точка**. Откуда берётся первый черновик V_init?

### 7.2 Сравнение стратегий инициализации

| Стратегия | Обучение | Качество старта | Стоимость | Рекомендация |
|-----------|----------|-----------------|-----------|--------------|
| **Informed Noise** | Нет | Низкое | Нулевая | PoC, Этап 0 |
| **Base-LCM (MLP IPP)** | Да (MSE) | Среднее | Низкая | PoC, Этап 1 |
| **Retrieval (k-NN)** | Нет | Среднее-Высокое | Низкая | Дополнение |
| **Cluster Centroid** | Минимальное | Низкое-Среднее | Низкая | Для cold start |
| **Diffusion-LCM** | Да (сложное) | Высокое | Высокая | Target |

### 7.3 Рекомендуемый путь (от простого к сложному)

**Этап 0: Informed Noise (для валидации EBT)**

```python
# Никакого обучения — просто шум вокруг запроса
alpha = 0.5  # гиперпараметр: 0.3-0.7
V_init = alpha * V_query + (1 - alpha) * torch.randn_like(V_query) * noise_scale
```

Логика: ответ семантически связан с вопросом, поэтому стартовать из окрестности вопроса разумнее, чем из случайной точки. Если EBT работает — Langevin доведёт этот грубый старт до правильного ответа.

**Этап 1: MLP IPP (Base-LCM analog)**

```python
class SimpleIPP(nn.Module):
    def __init__(self, dim=1024, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, V_query):
        return self.net(V_query)

# Обучение:
# Loss: MSE между выходом IPP и V_target из encoder
loss = F.mse_loss(ipp(V_query), V_target)
```

**Известная проблема MSE:** при множестве валидных ответов IPP усредняет их, выдавая вектор «между» всеми вариантами — точку, не соответствующую ни одному осмысленному ответу. Это именно то, что нашла Meta в Base-LCM.

**Решение:** это нормально для нашей архитектуры! IPP выдаёт «размытый черновик», а EBT через Langevin подтягивает его к конкретному, точному ответу. Base-LCM плох сам по себе, но идеален как инициализатор для EBT.

**Этап 2: Retrieval-Augmented Initialization (дополнение)**

```python
import faiss

# Индексируем все V_query из датасета
index = faiss.IndexFlatIP(1024)  # cosine similarity через inner product
index.add(all_query_vectors)

# При инференсе: находим ближайшие вопросы
distances, indices = index.search(V_query_new, k=5)

# Берём среднее ответов на ближайшие вопросы
V_init = torch.mean(all_answer_vectors[indices], dim=0)
```

Можно комбинировать с MLP: V_init = 0.5 * ipp(V_query) + 0.5 * retrieval_mean.

---

## 8. Режимы инференса

### 8.1 Режим A: Fast Shot (Быстрый ответ)

```
      V_query ──► IPP ──► V_init
                            │
             EBT(V_query, V_init) = E₀  (одна оценка энергии)
                            │
             Выбираем лучший из N кандидатов от IPP
                 (никакого градиентного спуска)
                            │
                        V_answer ──► Decoder ──► Text
```

**Характеристики:**
- Время: один forward pass через IPP + N forward passes через EBT
- Без backward pass вообще
- Аналог «интуитивного ответа» человека
- Подходит для простых вопросов

### 8.2 Режим B: Deep Thinking (Глубокое мышление)

```
V_query ──► IPP ──► V_init = V₀
                            │
                   ┌────────┴─────────┐
                   │  Langevin Loop   │
                   │                  │
                   │  while E > θ:    │
                   │    ∇E = ∂E/∂V    │  ← backward pass
                   │    V ← V - η∇E   │
                   │    V ← V + ε     │  ← стохастический шум
                   │    V ← project(V)│  ← проекция на сферу (OOD protection)
                   │                  │
                   │  Каждые K шагов: │
                   │    E_chain(CoT)  │  ← Режим B оценки цепочки
                   │                  │
                   └────────┬─────────┘
                            │
                        V_answer ──► Decoder ──► Text
```

**Формула Langevin Dynamics:**

```
V_{t+1} = V_t − η · ∇_V E(V_query, V_t) + √(2η) · ε_t

где:
  η — learning rate (step size), обычно 0.01-0.1
  ∇_V E — градиент энергии по входному вектору (НЕ по весам модели!)
  ε_t ~ N(0, I) — гауссов шум для escape из локальных минимумов
```

**Адаптивный Deep Thinking (с RL):**

Система учится определять, когда прекратить «думать»:
- Reward: +1 если итоговый V_answer близок к эталону, −λ за каждую итерацию (штраф за медлительность)
- Результат: для простых вопросов (2+2) энергия падает мгновенно за 2–3 шага, для сложных — требует 100–500 шагов

### 8.3 Оптимизация: Inertial Navigation (Lazy Gradient)

**Проблема:** backward pass (вычисление ∇E) ≈ в 3 раза дороже forward pass (вычисление E). При 500 итерациях Deep Thinking это существенно.

**Решение:** чередование фаз с градиентом и без:

```python
def inertial_langevin(V_init, ebt, V_query, max_steps=500):
    V = V_init.clone().requires_grad_(True)
    momentum = torch.zeros_like(V)
    phase = "explore"  # "explore" или "cruise"
    gradient_history = []

    for t in range(max_steps):
        E = ebt(V_query, V)  # forward pass (дёшево)

        if phase == "explore":
            # Вычисляем градиент (дорого)
            grad = torch.autograd.grad(E, V)[0]
            gradient_history.append(grad.clone())
            momentum = 0.9 * momentum + 0.1 * grad
            V = V - lr * momentum + noise_scale * torch.randn_like(V)

            # Переключение на cruise после K шагов
            if len(gradient_history) >= 5:
                direction = torch.mean(torch.stack(gradient_history[-5:]), dim=0)
                phase = "cruise"
                gradient_history = []

        elif phase == "cruise":
            # Двигаемся по инерции (дёшево — только forward pass)
            V = V - lr * direction + noise_scale * torch.randn_like(V)

            E_new = ebt(V_query, V)
            if E_new > E:  # энергия начала расти — сбились с курса
                phase = "explore"

        # Проекция на сферу (OOD protection)
        V = F.normalize(V, dim=-1) * target_norm

        # Early stopping
        if E < threshold:
            break

    return V.detach()
```

**Экономия:** при K=5 шагов explore и 10–15 шагов cruise — сокращение backward passes примерно в 3 раза.

### 8.4 Inertial Navigation как гиперпараметр инференса

> **Ревью-заметка (v1.1):** Inertial Navigation не должна быть скрытой оптимизацией — она является полноценным гиперпараметром, влияющим на качество рассуждений.

**Принцип:** модель лучше всего работает в условиях инференса, знакомых ей по обучению. Если EBT тренировалась только на траекториях с полным градиентом, а при инференсе мы включаем cruise-фазы — энергетический ландшафт может вести себя непредсказуемо на "инерционных" участках траектории.

**Решение:** `cruise_ratio` — явный гиперпараметр, определяющий долю cruise-шагов в Langevin loop.

**Пресеты режимов инференса:**

| Пресет | max_steps | cruise_ratio | Описание |
|--------|-----------|-------------|----------|
| `fast` | 10 | 0.7 | Минимум вычислений, агрессивная инерция |
| `balanced` | 50 | 0.5 | Баланс точности и скорости |
| `deep` | 200 | 0.2 | Высокая точность, мало инерции |
| `extra_deep` | 500 | 0.0 | Чистый Langevin, без cruise |

**Train-time awareness:** при обучении EBT каждый батч обрабатывается с **случайным** `cruise_ratio`, чтобы модель видела траектории, порождённые разными стратегиями навигации:

```python
# В training loop (после базового обучения EBT на статических парах):
cruise_ratio = random.choice([0.0, 0.0, 0.3, 0.5])  # bias к полному градиенту
trajectory = langevin.refine(V_init, V_query, cruise_ratio=cruise_ratio)
# EBT обучается оценивать точки на траекториях разного типа
```

---

## 9. Механизм контекста и памяти диалога

### 9.1 Принцип: НЕ пересылать весь контекст

В обычных LLM каждый запрос содержит полную историю чата (конкатенация всех сообщений). Это расточительно и масштабируется плохо.

В CEBCM реализуется **retrieval-based контекст**: храним все вектора, при каждом запросе выбираем только релевантные.

### 9.2 Архитектура кэша

> **Ревью-заметка (v1.1):** Исходная схема предполагала 1:1 (один query — один answer). В реальном диалоге на один вопрос может быть несколько ответов (уточнения, дополнения, корректировки). Архитектура обновлена для поддержки multi-answer слотов.

```
Context Cache (в 1024d SONAR-пространстве):
┌──────────────────────────────────────────────────────────────┐
│  Slot 1: V_q1 (1024d) │ [V_a1_1, V_a1_2] (1024d each)        │  turn_id=0
│  Slot 2: V_q2 (1024d) │ [V_a2_1]          (1024d)            │  turn_id=1
│  ...                                                         │
│  Slot N: V_qN (1024d) │ [V_aN_1, ..., V_aN_M] (1024d each)   │  turn_id=N
└──────────────────────────────────────────────────────────────┘

Размер: (1 + avg_answers) × 1024 × 4 bytes ≈ 12 КБ на один обмен (при avg 2 ответа)
1000 обменов = 12 МБ (ничтожно)
```

**Ключевое свойство:** при сериализации в последовательность для attention сохраняется хронологический порядок и структура (query/answer type embeddings), чтобы attention мог восстановить, что к чему относится:

```python
@dataclass
class CacheSlot:
    query_vector: Tensor          # [1024]
    answer_vectors: list[Tensor]  # каждый [1024], может быть несколько
    turn_id: int                  # порядок в диалоге
    timestamp: float

class ContextCache:
    def __init__(self, max_slots=1000):
        self.slots: list[CacheSlot] = []

    def add(self, V_query, V_answer, turn_id):
        """Добавляет ответ. Если query уже есть — append к существующему слоту."""
        for slot in self.slots:
            if torch.cosine_similarity(slot.query_vector, V_query, dim=0) > 0.98:
                slot.answer_vectors.append(V_answer)
                return
        self.slots.append(CacheSlot(V_query, [V_answer], turn_id, time.time()))

    def to_sequence(self, slots: list[CacheSlot]) -> tuple[Tensor, Tensor]:
        """
        Разворачивает слоты в упорядоченную последовательность.
        Returns:
            vectors: [seq_len, 1024]
            type_ids: [seq_len] — 0 для query, 1 для answer
        """
        vectors, type_ids = [], []
        for slot in sorted(slots, key=lambda s: s.turn_id):
            vectors.append(slot.query_vector)
            type_ids.append(0)
            for ans in slot.answer_vectors:
                vectors.append(ans)
                type_ids.append(1)
        return torch.stack(vectors), torch.tensor(type_ids)
```

### 9.3 Где ловить вектор для кэша

**Вектор вопроса:** берётся из SONAR encoder, **до** projector (если используется). Это «чистый» вектор в нативном пространстве SONAR.

**Вектор ответа:** берётся **после** deprojector (если используется), перед SONAR decoder. Причины:

1. Все вектора в кэше должны быть в одном пространстве (SONAR 1024d)
2. Вектор после deprojector = то, что реально уходит в decoder = то, что пользователь видит
3. Если кэшировать sparse-вектор из LCM-пространства, возникнет рассинхронизация между «памятью модели» и «тем, что было сказано» — экспоненциальный drift на длинных контекстах

### 9.4 Retrieval при каждом запросе

```python
def get_context(V_query_new, cache, top_k=5):
    """
    Находит top-K наиболее релевантных прошлых обменов.
    Возвращает слоты отсортированные хронологически (по turn_id).
    """
    if len(cache.slots) == 0:
        return []

    all_queries = torch.stack([slot.query_vector for slot in cache.slots])
    similarities = F.cosine_similarity(
        V_query_new.unsqueeze(0), all_queries, dim=-1
    )
    top_indices = torch.topk(similarities, min(top_k, len(cache.slots))).indices

    selected_slots = [cache.slots[idx] for idx in top_indices]
    # Сортируем по turn_id для хронологического порядка
    selected_slots.sort(key=lambda s: s.turn_id)

    return selected_slots
```

### 9.5 Surprise Mechanism

> **Ревью-заметка (v1.3):** Механизм вдохновлён **Google Titans** (arXiv:2501.00663) — surprise-driven memory с gradient-based surprise metrics. CEBCM адаптирует эту идею для **sentence-level vectors** (SONAR 1024d), что является новым применением — Titans работает на уровне токенов.

#### 9.5.1 Концепция

Не все входящие вектора одинаково ценны. Фраза «Ладно, понял» предсказуема и малоинформативна. Фраза «WebSocket должен переподключаться с exponential backoff и сохранением сессии» — неожиданна и критически важна.

**Surprise Score** — мера того, насколько вектор отличается от предсказания модели:

```
S_t = (1 − cos_sim(V̂_t, V_t)) / 2    ∈ [0, 1]

где V̂_t = SurprisePredictor(V_1, ..., V_{t-1}) — предсказание следующего вектора
```

- **S ≈ 0** — вектор полностью предсказуем, банальная информация
- **S > θ** — вектор неожиданный, содержит важную новую информацию → **Global Token**

#### 9.5.2 Позиция в пайплайне

SurprisePredictor располагается **сразу после SONAR encoder**, до всего остального:

```
User Text → SONAR Encoder → V_new (1024d)
                                  │
                       ┌──────────┴───────────┐
                       │  Surprise Predictor  │
                       │ (сразу после encoder)│
                       │                      │
                       │  V̂ = SSM(V₁..V_{t-1})│
                       │  S = 1-cos(V̂, V_new) │
                       └──────────┬───────────┘
                                  │
                       surprise_score ∈ [0, 1]
                                  │
                       Context Cache: V_new + metadata{S, turn_id, ...}
```

#### 9.5.3 Архитектура SurprisePredictor

Требования: O(N) по времени и O(1) по памяти на шаг при инференсе → **SSM (State Space Model)**, Mamba-style linear recurrence:

```python
class SurprisePredictor(nn.Module):
    """
    Предсказывает следующий вектор в последовательности.
    Surprise = prediction error.
    
    Архитектура: Mamba-style SSM (linear recurrence).
    Complexity: O(N) time, O(1) memory per step.
    """
    def __init__(self, dim=1024, state_dim=2048, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            SSMBlock(dim=dim, state_dim=state_dim)
            for _ in range(n_layers)
        ])
        self.pred_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.hidden_state = None  # Для инкрементального инференса
    
    def predict_next(self, V_context_seq):
        """
        V_context_seq: [batch, seq_len, 1024]
        Returns: predicted next vectors [batch, seq_len, 1024]
        """
        h = V_context_seq
        for layer in self.layers:
            h = layer(h)
        return self.pred_head(h)
    
    def compute_surprise(self, V_sequence):
        """
        Для последовательности [V₁, V₂, ..., Vₙ]
        возвращает surprise для каждого V (кроме первого).
        
        surprise_i = (1 - cos_sim(V̂_i, V_i)) / 2
        """
        predictions = self.predict_next(V_sequence[:, :-1])
        actual = V_sequence[:, 1:]
        
        surprise = 1 - F.cosine_similarity(predictions, actual, dim=-1)
        surprise = surprise / 2.0  # Нормализуем в [0, 1]
        return surprise  # [batch, N-1]
    
    def step(self, V_new):
        """
        Инкрементальный шаг: принимает один новый вектор,
        возвращает surprise для него. O(1) по памяти.
        """
        V_pred = self.pred_head(self.hidden_state)
        surprise = (1 - F.cosine_similarity(
            V_pred.unsqueeze(0), V_new.unsqueeze(0), dim=-1
        )) / 2.0
        
        for layer in self.layers:
            self.hidden_state = layer.step(V_new, self.hidden_state)
        
        return surprise.item()
```

**Параметры:** ~50–100M параметров (2 SSM-слоя, state_dim=2048). Лёгкий модуль — значительно меньше основного EBT.

#### 9.5.4 Global Tokens (high-surprise вектора)

Вектора с Surprise Score выше порога θ становятся **Global Tokens** — видны **всем** другим векторам при attention:

```
Context: [V₁, V₂, V₃, V₄, V₅, ..., V₅₀₀₀₀]
Surprise: [0.1, 0.8, 0.2, 0.9, 0.1, ..., 0.3]
                 ↑           ↑
              GLOBAL       GLOBAL
              TOKEN        TOKEN

Для любого V_t при attention:
  1. V_t видит все вектора через Linear Attention (SSM state)
  2. V_t ДОПОЛНИТЕЛЬНО видит все Global Tokens через direct attention
  3. Global Tokens = вектора с surprise > θ
```

**Порог θ:** адаптивный, running percentile (top-5% по surprise). Предпочтительнее фиксированного порога, т.к. распределение surprise зависит от домена.

#### 9.5.5 Обучение SurprisePredictor

Полностью **self-supervised** — никакие labels не нужны:

```python
def train_surprise_predictor(predictor, dataset):
    """
    dataset: корпус последовательностей SONAR-векторов.
    Wikipedia абзацы / книги / диалоги → разбитые на предложения → SONAR.
    """
    optimizer = AdamW(surprise_predictor.parameters(), lr=3e-4)
    
    for sequences in dataloader:  # [batch, seq_len, 1024]
        predictions = surprise_predictor.predict_next(sequences[:, :-1])
        targets = sequences[:, 1:]
        
        # Двойной loss: MSE для нормы + cosine для направления
        loss_mse = F.mse_loss(predictions, targets)
        loss_cos = (1 - F.cosine_similarity(predictions, targets, dim=-1)).mean()
        loss = loss_mse + 0.5 * loss_cos
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Данные:** те же WikiText / SQuAD, что и для основной системы → разбить на предложения → SONAR. Это та же задача, что у Base-LCM (Meta), и она доказано работает.

#### 9.5.6 Практический пример: программирование

```
Turn 1: "Используем React для фронтенда"          → surprise=0.15 (ожидаемо)
Turn 2: "С серверным рендерингом на Next.js"       → surprise=0.20 (ожидаемо)
Turn 3: "Но WebSocket должен переподключаться      → surprise=0.85 (GLOBAL TOKEN)
         с exponential backoff и persistence"
Turn 4: "Обычный REST API для CRUD"                → surprise=0.10 (ожидаемо)
...
Turn 150: "Напиши мне сетевой модуль"
    → модель видит Turn 3 через global attention и ПОМНИТ
      про exponential backoff, хотя прошло 150 ходов
```

Без Surprise: Turn 3 давно вылетел из локального контекста, модель забыла. С Surprise: это Global Token, он виден всегда.

### 9.6 Подача контекста в IPP: Linear Attention + Surprise Global Tokens

> **Ревью-заметка (v1.3):** Заменяет MHLA-inspired агрегатор из v1.2 (перенесён в главу 16). Вместо сжатия KV через learned slots используется Linear Attention для обработки полной последовательности + direct attention к Global Tokens.

```python
class ContextAggregator(nn.Module):
    """
    Linear Attention (SSM) + Surprise Global Tokens.
    
    Архитектура:
    1. SSM (Mamba/GLA) обрабатывает ВСЮ последовательность контекста - O(N)
    2. Global Tokens (high surprise) сохраняются как явные KV-пары
    3. Query делает standard attention ТОЛЬКО к SSM-output + Global Tokens
    
    Complexity: O(N) для SSM + O(G^2) для Global Tokens, где G << N.
    """
    def __init__(self, dim=1024, n_heads=8, n_layers=2, state_dim=2048):
        super().__init__()
        self.type_embedding = nn.Embedding(2, dim)  # query=0, answer=1
        
        # Linear SSM для обработки полной последовательности
        self.ssm_layers = nn.ModuleList([
            SSMBlock(dim=dim, state_dim=state_dim)
            for _ in range(n_layers)
        ])
        self.state_projector = nn.Linear(dim, dim)
        
        # Final attention: query -> [ssm_output, global_tokens]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.final_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, V_query, context_vectors, type_ids, surprise_scores,
                surprise_threshold=None):
        """
        V_query: [batch, 1024]
        context_vectors: [batch, seq_len, 1024]
        type_ids: [batch, seq_len] — 0 query, 1 answer
        surprise_scores: [batch, seq_len]
        Returns: [batch, 1024] — агрегированный контекст для IPP
        """
        ctx = context_vectors + self.type_embedding(type_ids)
        
        # Linear SSM: обработка всей последовательности
        h = ctx
        for layer in self.ssm_layers:
            h = layer(h)
        ssm_output = self.state_projector(h[:, -1:, :])
        
        # Извлечь Global Tokens (high surprise)
        if surprise_threshold is None:
            surprise_threshold = torch.quantile(surprise_scores, 0.95, dim=-1)
        global_mask = surprise_scores > surprise_threshold.unsqueeze(-1)
        global_tokens = self._gather_globals(ctx, global_mask)
        
        # Final attention: query -> [ssm_output, global_tokens]
        full_seq = torch.cat([
            V_query.unsqueeze(1),  # [batch, 1, dim]
            ssm_output,            # [batch, 1, dim]
            global_tokens,         # [batch, G, dim]
        ], dim=1)
        
        out = self.final_attention(full_seq)
        return out[:, 0, :]
```

**Вычислительная сложность:**

| Подход | Сложность | Качество | Масштаб |
|--------|-----------|----------|---------|
| Конкатенация + MLP | O(1) | Низкое | До ~10 векторов |
| Vanilla Transformer | O(N²) | Высокое | До ~1K векторов |
| **Linear Attention + Surprise Globals (v1.3)** | O(N) + O(G²) | Высокое | 50K+ векторов |

При N=50K, G=5% от N = 2500: SSM O(50K) + Global attention O(2500²) = 6.25M → **~19× ускорение** vs O(N²) = 2.5B.

**Ключевое свойство:** ни один важный вектор не теряется — high-surprise вектора видны через direct attention, остальной контекст доступен через SSM state.

### 9.7 External Compaction (Context Garbage Collection)

> **Ревью-заметка (v1.3):** Это не attention-mechanism, а **runtime-система** управления памятью. Модель сама решает, что сжать, опираясь на весь текущий контекст и Surprise Score каждого вектора. Принципиально отличается от механического sliding window или attention-маски.

#### 9.7.1 Концепция: Context Garbage Collector

Когда контекст превышает MAX_CONTEXT (например, 1000 векторов), система запускает компакцию. Для сигнала компакции используется **Compact Token** — специальный learned embedding (1024d), обучаемый совместно с пайплайном:

```python
class CompactToken(nn.Module):
    """
    Learned embedding для сигнала компакции.
    Отдельный обучаемый вектор, не текстовый промпт.
    type_id=2 (0=query, 1=answer, 2=compact)
    """
    def __init__(self, dim=1024):
        super().__init__()
        self.compact_embedding = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.type_id = 2

    def get_token(self) -> Tensor:
        return self.compact_embedding
```

**Почему отдельный token, а не текстовый промпт:** модель однозначно классифицирует действие «нужно сжать контекст» без двусмысленности текстового описания. Compact Token обучается end-to-end вместе с остальным пайплайном.

```
Compaction trigger: context > MAX_CONTEXT (напр. 1000 vectors)
                    │
         Выбираем старейшие K векторов для compaction
                    │
         Подаём Compact Token + low-surprise вектора в пайплайн
                    │
     ┌──────────────┼───────────────┐
     │              │               │
  Low-surprise   High-surprise   Compact Token
  вектора из     вектора из      сигнализирует:
  старого        старого         «сожми контекст»
  диапазона      диапазона
     │              │               │
  СУММАРИЗУЮТСЯ  СОХРАНЯЮТСЯ     IPP → EBT → Langevin
  в 2-3 вектора- как global      генерируют
  заметки        tokens          summary-вектора
```

**Почему это лучше attention-маски (sliding window):**

- Модель **сама решает**, что важно (а не маска)
- Compact Token — обучаемый сигнал, модель учится реагировать на него
- Заметка содержит **выводы** («ошибки X были исправлены»), а не просто сжатые эмбеддинги
- Пользователь может **видеть** эти заметки и корректировать
- Масштабируется **линейно** по памяти, без квадратичности

#### 9.7.2 Surprise-aware Compaction

Surprise Score служит подсказкой для компактора:

```python
def compact_context(cache, compaction_range, pipeline):
    """
    Сжимает старую часть контекста в summary-заметки.
    
    1. Low-surprise вектора -> модель суммаризует в 2-3 заметки
    2. High-surprise вектора -> сохраняются как global tokens
       или их суть включается в summary с приоритетом
    """
    old_slots = cache.get_range(compaction_range)
    
    # Разделяем по surprise
    high_surprise = [s for s in old_slots if s.surprise_score > threshold]
    low_surprise = [s for s in old_slots if s.surprise_score <= threshold]
    
    # Модель генерирует summary для low-surprise части
    # Используется тот же IPP+EBT пайплайн,
    # но с задачей «сжатие контекста» вместо «ответ на вопрос»
    summary_vectors = pipeline.summarize(low_surprise, full_context=cache)
    
    # Заменяем старые слоты на summary + сохранённые global tokens
    cache.replace_range(
        compaction_range,
        summary_slots=summary_vectors,
        preserved_globals=high_surprise
    )
```

#### 9.7.3 Триггеры компакции

| Триггер | Порог | Действие |
|---------|-------|----------|
| Размер контекста | > MAX_CONTEXT | Компактим старейшие K векторов |
| Явный запрос | Пользователь/система | Компактим указанный диапазон |
| Смена темы | Низкая релевантность старого контекста | Компактим нерелевантный блок |

### 9.9 Математическое масштабирование: Linear Attention + YaRN/NTK

> **Ревью-заметка (v1.3):** Заменяет Hierarchical Attention (v1.2, перенесён в главу 16). Вместо «как срезать контекст» — «как максимально эффективно обращаться к каждому элементу контекста».

#### 9.9.1 Linear Attention для основного контекста

**Mamba-2 / GLA** используются как backbone ContextAggregator (см. §9.6):

- O(N) по времени, O(1) по памяти на каждый шаг
- Обрабатывает **ВСЮ** последовательность, не теряя ни один вектор
- Selective state (Mamba) — естественный фильтр: модель сама решает, что запомнить в state
- Качество: close to Transformer при 10× меньших затратах (Mamba-2 бенчмарки)

**Ограничение:** линейные модели обрабатывают слева направо. Для **EBT** (bidirectional оценка) они не подходят — EBT использует стандартный self-attention на 5–20 векторах (мгновенный). Для IPP и ContextAggregator — отличный выбор.

#### 9.9.2 Позиционное кодирование

| Компонент | PE | Обоснование |
|-----------|-------|-------------|
| Chain Head (5–20 элементов) | **RoPE** | Короткие фиксированные цепочки, порядок критичен |
| ContextAggregator (SSM) | **Implicit** (рекуррентная структура SSM) | Порядок закодирован в state updates |
| Final Attention (query → globals) | **ALiBi** | Не модифицирует вектора (сохраняет геометрию SONAR) |

**ALiBi для attention поверх SONAR-векторов:** SONAR-вектора уже предобучены с определённой геометрией. RoPE вращает Q/K подпространства, **разрушая семантические расстояния**. ALiBi добавляет только additive bias к attention scores, сохраняя вектора неизменными.

#### 9.9.3 Экстраполяция на длинные контексты

**YaRN / NTK-aware scaling** — для компонентов, использующих RoPE (Chain Head):

- Тренируем на коротких последовательностях (500–2K предложений)
- Инферим на длинных (10K–50K предложений)
- Для sentence-level vectors экстраполяция проще: «плотность информации» на позицию значительно выше

**Для SSM-based компонентов (SurprisePredictor, ContextAggregator SSM):** экстраполяция обеспечивается самой рекуррентной архитектурой — state может обновляться бесконечно.

#### 9.9.4 Валидация подхода

Подход «attention over sentence embeddings» валидирован работой **"Attention over pre-trained Sentence Embeddings for Long Document Classification"** (Abdaoui & Dutta, 2023, arXiv:2307.09084). Конкурентные результаты с fine-tuning при линейном масштабировании по длине документа.

### 9.8 Преимущества перед LLM-подходом

| Параметр | LLM | CEBCM |
|----------|-----|-------|
| Контекст 50 обменов | 3000–5000 токенов | 100 векторов (1024d) |
| Attention complexity | O(5000²) = 25M operations | O(100²) = 10K operations |
| Память | ~200 МБ KV-cache | ~0.8 МБ vector cache |
| Релевантность | Вся история, включая нерелевантное | Только top-K релевантных |
| Масштабируемость | Ограничена context window | Растёт линейно, retrieval O(log N) |

---

## 10. Оптимизация навигации

### 10.1 Momentum для Langevin Dynamics

Вместо обычного SGD используем Adam-like обновление:

```python
m_t = β₁ · m_{t-1} + (1 - β₁) · ∇E      # первый момент (направление)
v_t = β₂ · v_{t-1} + (1 - β₂) · (∇E)²   # второй момент (масштаб)

V_{t+1} = V_t - η · m_t / (√v_t + ε_adam) + noise
```

Преимущество: стабильное движение по «долинам» ландшафта, автоматическая адаптация step size для каждого измерения.

### 10.2 Oscillation Detection

```python
def detect_oscillation(history, window=5):
    """Определяет, прыгаем ли мы вокруг минимума."""
    if len(history) < window:
        return False
    recent = torch.stack(history[-window:])
    # Если текущий вектор ближе к V_{t-3}, чем к V_{t-1} → осцилляция
    dist_prev = torch.norm(recent[-1] - recent[-2])
    dist_older = torch.norm(recent[-1] - recent[-3])
    return dist_older < dist_prev * 0.8  # порог
```

При обнаружении осцилляции: уменьшить lr в 2 раза или переключиться на fine-grained mode с маленьким step size.

### 10.3 Early Stopping

```python
# Критерии остановки Langevin Loop:
# 1. Энергия ниже порога
if E < energy_threshold:
    break

# 2. Энергия перестала падать (plateau)
if abs(E_prev - E) < delta_threshold:
    plateau_counter += 1
    if plateau_counter > patience:
        break

# 3. Максимум итераций
if step >= max_steps:
    break
```

---

## 11. Пайплайн обучения

### 11.1 Общая последовательность (4 фазы)

```
Фаза 0: Валидация пространства (1-2 дня)
   └─► Убедиться, что SONAR пригоден для градиентной навигации

Фаза 1: Обучение IPP (3-5 дней)
   └─► MLP, MSE loss, пары (V_query, V_target)

Фаза 2: Обучение EBT (1-2 недели)
   └─► Contrastive Learning, Curriculum, InfoNCE loss
   └─► Два scoring head'а: pairwise + chain

Фаза 3: Обучение Decoder'а (1-2 недели, после заморозки Фаз 1-2)
   └─► Cross-entropy, teacher forcing
   └─► Вход: вектора от замороженной системы, НЕ от encoder'а
```

### 11.2 Фаза 1: Обучение IPP

```python
# === Скрипт обучения IPP ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Данные: пары (V_query, V_target) из SONAR
# Предполагается: query_vectors [N, 1024], target_vectors [N, 1024]

class IPP(nn.Module):
    def __init__(self, dim=1024, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return self.net(x)

model = IPP()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

dataset = TensorDataset(query_vectors, target_vectors)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

for epoch in range(100):
    total_loss = 0
    for V_q, V_t in loader:
        pred = model(V_q)
        loss = F.mse_loss(pred, V_t)

        # Дополнительно: cosine similarity loss для направления
        cos_loss = 1 - F.cosine_similarity(pred, V_t, dim=-1).mean()
        total_loss_step = loss + 0.5 * cos_loss

        optimizer.zero_grad()
        total_loss_step.backward()
        optimizer.step()
        total_loss += total_loss_step.item()

    scheduler.step()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}: loss={avg_loss:.4f}")

torch.save(model.state_dict(), "ipp.pt")
```

### 11.3 Фаза 2: Обучение EBT

```python
# === Скрипт обучения EBT ===

class EBT_PairwiseHead(nn.Module):
    """Режим A: оценка пары векторов."""
    def __init__(self, dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, 2048),  # [q; c; q-c; q*c]
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, V_query, V_candidate):
        diff = V_query - V_candidate
        prod = V_query * V_candidate
        x = torch.cat([V_query, V_candidate, diff, prod], dim=-1)
        return self.net(x).squeeze(-1)  # scalar energy


class EBT_ChainHead(nn.Module):
    """Режим B: оценка цепочки векторов."""
    def __init__(self, dim=1024, n_layers=2, n_heads=8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=2048,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, sequence):
        # sequence: [batch, seq_len, 1024]
        batch_size = sequence.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, sequence], dim=1)  # [batch, 1+seq_len, 1024]
        x = self.transformer(x)
        cls_out = x[:, 0, :]  # CLS token output
        return self.head(cls_out).squeeze(-1)


class EBT(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.pairwise = EBT_PairwiseHead(dim)
        self.chain = EBT_ChainHead(dim)

    def energy(self, V_query, V_candidate):
        """Режим A: быстрая оценка пары."""
        return self.pairwise(V_query, V_candidate)

    def chain_energy(self, sequence):
        """Режим B: оценка цепочки рассуждений."""
        return self.chain(sequence)


# === Обучение EBT с Curriculum Learning ===

ebt = EBT()
optimizer = torch.optim.AdamW(ebt.parameters(), lr=1e-4)

def infonce_loss(ebt, V_query, V_positive, V_negatives, temperature=0.07):
    """
    V_query: [batch, 1024]
    V_positive: [batch, 1024]
    V_negatives: [batch, num_neg, 1024]
    """
    E_pos = ebt.energy(V_query, V_positive)  # [batch]
    E_neg = torch.stack([
        ebt.energy(V_query, V_negatives[:, i])
        for i in range(V_negatives.size(1))
    ], dim=1)  # [batch, num_neg]

    # InfoNCE: позитив должен иметь НИЗКУЮ энергию
    logits = torch.cat([-E_pos.unsqueeze(1), -E_neg], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


# Curriculum: переключение сложности негативов
for phase, (neg_generator, epochs) in enumerate([
    (easy_negatives,   20),  # cosine sim 0.3-0.7
    (medium_negatives, 30),  # cosine sim 0.7-0.9
    (hard_negatives,   50),  # cosine sim 0.9-0.98
]):
    print(f"Phase {phase}: training for {epochs} epochs")
    for epoch in range(epochs):
        for V_q, V_pos in dataloader:
            V_neg = neg_generator(V_q, V_pos, num_neg=31)
            loss = infonce_loss(ebt, V_q, V_pos, V_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

torch.save(ebt.state_dict(), "ebt.pt")
```

### 11.4 Фаза 3: Обучение Decoder

Описан в разделе 6.3.

---

## 12. Датасеты и генерация данных

### 12.1 Источники данных для IPP и EBT

| Датасет | Формат | Размер | Применение |
|---------|--------|--------|------------|
| **SQuAD v2** | (question, answer) | 150K пар | QA pairs |
| **Natural Questions** | (question, long_answer) | 300K+ пар | QA pairs |
| **MS MARCO** | (query, passage) | 8.8M пар | Retrieval pairs |
| **SNLI / MultiNLI** | (premise, hypothesis, label) | 570K пар | Entailment negatives |
| **WikiText-103** | Сплошной текст | 100M tokens | Consecutive sentences |
| **OpenAssistant** | (instruction, response) | 161K пар | Instruction following |
| **Alpaca** | (instruction, output) | 52K пар | Instruction following |

### 12.2 Автоскрипт генерации датасета

```python
# === Генерация датасета пар (V_query, V_target, V_negatives) ===

import json
import torch
import numpy as np
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from datasets import load_dataset

# Загрузка SONAR
encoder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder"
)

def encode_batch(texts, lang="eng_Latn", batch_size=64):
    """Кодирует список текстов в SONAR-вектора."""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vectors = encoder.predict(batch, source_lang=lang)
        all_vectors.append(vectors)
    return torch.cat(all_vectors, dim=0)

# === Шаг 1: Загрузка и кодирование QA-пар ===

print("Loading SQuAD...")
squad = load_dataset("squad_v2", split="train")

questions = [ex["question"] for ex in squad if ex["answers"]["text"]]
answers = [ex["answers"]["text"][0] for ex in squad if ex["answers"]["text"]]

print(f"Encoding {len(questions)} questions...")
V_questions = encode_batch(questions)

print(f"Encoding {len(answers)} answers...")
V_answers = encode_batch(answers)

# === Шаг 2: Генерация негативов с разными уровнями сложности ===

def generate_negatives(V_queries, V_answers, num_neg=31):
    """
    Для каждого запроса генерирует негативы трёх уровней:
    - Easy: случайные ответы из датасета (cos_sim ~0.3-0.7)
    - Medium: ответы на похожие вопросы (cos_sim ~0.7-0.9)
    - Hard: ответы на очень похожие вопросы (cos_sim ~0.9-0.98)
    """
    N = V_queries.size(0)

    # Находим cosine similarity между всеми парами вопросов
    V_q_norm = F.normalize(V_queries, dim=-1)
    sim_matrix = V_q_norm @ V_q_norm.T  # [N, N]

    all_negatives = {
        "easy": [],    # 10 neg per sample
        "medium": [],  # 10 neg per sample
        "hard": [],    # 11 neg per sample
    }

    for i in range(N):
        sims = sim_matrix[i]
        sims[i] = -1  # исключаем самого себя

        # Easy: cosine sim 0.3-0.7
        easy_mask = (sims > 0.3) & (sims < 0.7)
        easy_indices = torch.where(easy_mask)[0]
        if len(easy_indices) >= 10:
            chosen = easy_indices[torch.randperm(len(easy_indices))[:10]]
        else:
            chosen = torch.randint(0, N, (10,))
        all_negatives["easy"].append(V_answers[chosen])

        # Medium: cosine sim 0.7-0.9
        med_mask = (sims > 0.7) & (sims < 0.9)
        med_indices = torch.where(med_mask)[0]
        if len(med_indices) >= 10:
            chosen = med_indices[torch.randperm(len(med_indices))[:10]]
        else:
            chosen = torch.randint(0, N, (10,))
        all_negatives["medium"].append(V_answers[chosen])

        # Hard: cosine sim 0.9-0.98
        hard_mask = (sims > 0.9) & (sims < 0.98)
        hard_indices = torch.where(hard_mask)[0]
        if len(hard_indices) >= 11:
            chosen = hard_indices[torch.randperm(len(hard_indices))[:11]]
        else:
            chosen = torch.randint(0, N, (11,))
        all_negatives["hard"].append(V_answers[chosen])

    return {k: torch.stack(v) for k, v in all_negatives.items()}

print("Generating negatives...")
negatives = generate_negatives(V_questions, V_answers)

# === Шаг 3: Сохранение ===

torch.save({
    "V_questions": V_questions,
    "V_answers": V_answers,
    "negatives_easy": negatives["easy"],
    "negatives_medium": negatives["medium"],
    "negatives_hard": negatives["hard"],
}, "cebcm_dataset.pt")

print(f"Dataset saved: {V_questions.size(0)} pairs")
```

### 12.3 Генерация CoT-цепочек для Chain Scoring

```python
# === Генерация латентных CoT-цепочек ===
# Используем LLM для разбиения решения на шаги, кодируем каждый шаг в SONAR

from openai import OpenAI  # или любой другой LLM API

client = OpenAI()

def generate_cot_chain(question, answer):
    """Генерирует 5-7 шагов решения через LLM."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Break the following question-answer pair into 5-7 reasoning steps.
Each step should be one clear sentence.

Question: {question}
Answer: {answer}

Output ONLY the steps, one per line, numbered 1-7."""
        }]
    )
    steps = response.choices[0].message.content.strip().split("\n")
    # Очищаем нумерацию
    steps = [s.lstrip("0123456789.) ") for s in steps if s.strip()]
    return steps

def create_cot_dataset(questions, answers, num_samples=1000):
    """Создаёт датасет CoT-цепочек в SONAR-пространстве."""
    chains_positive = []  # правильные цепочки
    chains_negative = []  # цепочки с "тупиками"

    for i in range(min(num_samples, len(questions))):
        steps = generate_cot_chain(questions[i], answers[i])
        if len(steps) < 3:
            continue

        # Кодируем все шаги
        V_steps = encode_batch(steps)
        V_q = encode_batch([questions[i]])
        V_a = encode_batch([answers[i]])

        # Позитивная цепочка: V_q → V_step1 → ... → V_stepN → V_a
        chain = torch.cat([V_q, V_steps, V_a], dim=0)
        chains_positive.append(chain)

        # Негативная цепочка: заменяем случайный шаг на случайный вектор
        chain_neg = chain.clone()
        replace_idx = torch.randint(1, len(chain) - 1, (1,)).item()
        chain_neg[replace_idx] = V_steps[torch.randint(0, len(V_steps), (1,)).item()]
        chains_negative.append(chain_neg)

    return chains_positive, chains_negative
```

---

## 13. Практическая реализация: Proof of Concept

### 13.1 Минимальный PoC: «Денойзинг в SONAR»

**Цель:** проверить, может ли EBT через градиентный спуск вернуть зашумлённый вектор ближе к оригиналу.

**Время:** 2–3 дня  
**Железо:** 1 GPU (RTX 3090/4090 или эквивалент)

```python
# === PoC Experiment 1: Denoising ===

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Подготовка данных
# Кодируем 10,000 предложений из WikiText в SONAR
texts = load_wikitext_sentences(10000)
V_all = encode_batch(texts)  # [10000, 1024]

# 2. Простая функция энергии: MLP
class SimpleEnergy(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, V_orig, V_candidate):
        diff = V_orig - V_candidate
        prod = V_orig * V_candidate
        x = torch.cat([V_orig, V_candidate, diff, prod], dim=-1)
        return self.net(x).squeeze(-1)

energy_fn = SimpleEnergy()
optimizer = torch.optim.AdamW(energy_fn.parameters(), lr=1e-4)

# 3. Обучение: позитив = (V, V), негатив = (V, V + noise)
for epoch in range(50):
    perm = torch.randperm(len(V_all))
    for i in range(0, len(V_all), 32):
        batch = V_all[perm[i:i+32]]

        # Позитив: идентичная пара → энергия = 0
        E_pos = energy_fn(batch, batch)

        # Негатив: зашумлённая версия → энергия > 0
        noise = torch.randn_like(batch) * 0.2
        E_neg = energy_fn(batch, batch + noise)

        # Contrastive: E_pos должна быть меньше E_neg
        loss = F.relu(E_pos - E_neg + 1.0).mean()  # margin = 1.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. Тест: Langevin denoising
V_orig = V_all[0:1]  # один вектор
V_noisy = V_orig + torch.randn_like(V_orig) * 0.2  # зашумлённый

V_current = V_noisy.clone().requires_grad_(True)
lr = 0.01

for step in range(100):
    E = energy_fn(V_orig, V_current)
    grad = torch.autograd.grad(E, V_current)[0]
    V_current = (V_current - lr * grad).detach().requires_grad_(True)

    cos_sim = F.cosine_similarity(V_orig, V_current).item()
    print(f"Step {step}: E={E.item():.4f}, cos_sim={cos_sim:.4f}")

# 5. Декодируем результат
decoded_orig = decoder.predict(V_orig, target_lang="eng_Latn")
decoded_noisy = decoder.predict(V_noisy, target_lang="eng_Latn")
decoded_denoised = decoder.predict(V_current.detach(), target_lang="eng_Latn")

print(f"Original:  {decoded_orig[0]}")
print(f"Noisy:     {decoded_noisy[0]}")
print(f"Denoised:  {decoded_denoised[0]}")
```

**Критерий успеха:** cos_sim между denoised и original выше, чем между noisy и original. Decoded текст denoised ближе по смыслу к оригиналу.

### 13.2 PoC Experiment 2: Генерация ответа

**Цель:** проверить, может ли EBT навигировать из V_query к V_answer.

**Время:** 1–2 недели  

```python
# === PoC Experiment 2: QA Generation ===

# Предполагается: IPP и EBT уже обучены (Фазы 1-2)

def generate_answer(V_query, ipp, ebt, decoder,
                    lr=0.01, max_steps=50, threshold=0.5):
    """Полный пайплайн генерации ответа."""

    # 1. IPP даёт начальную точку
    V_init = ipp(V_query)

    # 2. Langevin refinement
    V = V_init.clone().requires_grad_(True)
    for step in range(max_steps):
        E = ebt.energy(V_query, V)

        if E.item() < threshold:
            break

        grad = torch.autograd.grad(E, V)[0]
        V = (V - lr * grad).detach().requires_grad_(True)

        # OOD protection: нормализация
        V_data = V.data
        V_data = F.normalize(V_data, dim=-1) * target_norm
        V = V_data.requires_grad_(True)

    # 3. Декодирование
    text = decoder.predict(V.detach(), target_lang="eng_Latn")
    return text[0], V.detach()


# Тест
test_questions = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What is machine learning?",
]

for q in test_questions:
    V_q = encoder.predict([q], source_lang="eng_Latn")
    answer, V_ans = generate_answer(V_q, ipp, ebt, decoder)
    print(f"Q: {q}")
    print(f"A: {answer}")
    print()
```

### 13.3 Оценка качества

```python
# === Метрики оценки ===

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

def evaluate_cebcm(test_pairs, ipp, ebt, encoder, decoder):
    """Оценка качества генерации на тестовом наборе."""

    cosine_sims = []
    rouge_scores = []
    bleu_scores = []

    for query_text, target_text in test_pairs:
        V_q = encoder.predict([query_text], source_lang="eng_Latn")
        V_target = encoder.predict([target_text], source_lang="eng_Latn")

        generated_text, V_gen = generate_answer(V_q, ipp, ebt, decoder)

        # 1. Cosine similarity в латентном пространстве
        cos = F.cosine_similarity(V_target, V_gen).item()
        cosine_sims.append(cos)

        # 2. ROUGE-L на декодированном тексте
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge = scorer.score(target_text, generated_text)['rougeL'].fmeasure
        rouge_scores.append(rouge)

        # 3. BLEU на декодированном тексте
        bleu = sentence_bleu([target_text.split()], generated_text.split())
        bleu_scores.append(bleu)

    print(f"Avg Cosine Similarity: {np.mean(cosine_sims):.4f}")
    print(f"Avg ROUGE-L:           {np.mean(rouge_scores):.4f}")
    print(f"Avg BLEU:              {np.mean(bleu_scores):.4f}")
```

---

## 14. Риски и митигации

### 14.1 Ранжирование рисков

| # | Риск | Вероятность | Импакт | Митигация |
|---|------|-------------|--------|-----------|
| 1 | **OOD drift**: Langevin выводит вектор за пределы обитаемой зоны SONAR | Высокая | Критический | Sphere projection + L2 reg + KL к prior |
| 2 | **Ложные негативы**: Hard negatives (sim>0.95) содержат парафразы | Высокая | Высокий | Cross-Encoder фильтрация + Soft-InfoNCE |
| 3 | **MSE averaging**: IPP выдаёт «никакой» вектор между валидными ответами | Средняя | Средний | EBT refinement как design, не как костыль |
| 4 | **Decoder infidelity**: SONAR decoder ломается на синтетических векторах | Средняя | Высокий | Валидация пространства (эксп. 4.3) перед обучением |
| 5 | **Energy landscape**: функция энергии имеет «рваный» ландшафт с ложными минимумами | Средняя | Высокий | Spectral norm + gradient penalty + curriculum |
| 6 | **Latency**: Deep Thinking (100-500 итераций) слишком медленный | Низкая | Средний | Inertial navigation + early stopping + adaptive budget |

### 14.2 Kill Criteria (когда остановить проект)

Проект не имеет смысла продолжать, если:

1. **Эксперимент 4.3 (валидация SONAR)** показывает, что при noise_scale > 0.05 декодированный текст полностью теряет смысл → пространство SONAR непригодно для навигации
2. **PoC Experiment 1 (denoising)** показывает, что EBT не может вернуть зашумлённый вектор ближе к оригиналу → энергетическая функция нерабочая
3. **PoC Experiment 2** после 2 недель обучения показывает cosine similarity < 0.5 между сгенерированным и эталонным ответом → архитектура не сходится

---

## 15. Roadmap проекта

### Milestone 1: Валидация (неделя 1)
- [ ] Установить SONAR, прогнать эксперимент 4.3
- [ ] Измерить устойчивость к шуму и качество интерполяции
- [ ] Решение: GO / NO-GO

### Milestone 2: PoC Denoising (недели 2–3)
- [ ] Генерация датасета (скрипт 12.2)
- [ ] Обучить SimpleEnergy на задаче denoising
- [ ] Проверить Langevin denoising (эксперимент 13.1)
- [ ] Решение: GO / PIVOT

### Milestone 3: IPP + EBT (недели 4–7)
- [ ] Обучить IPP (Фаза 1)
- [ ] Обучить EBT с Curriculum Learning (Фаза 2)
- [ ] Генерация CoT-цепочек (скрипт 12.3)
- [ ] Обучить Chain Scoring Head
- [ ] Измерить качество (метрики 13.3)

### Milestone 4: Полный пайплайн (недели 8–10)
- [ ] Обучить Decoder (Фаза 3) или валидировать SONAR decoder
- [ ] Реализовать Context Cache + Retrieval
- [ ] Реализовать Fast Shot и Deep Thinking режимы
- [ ] End-to-end оценка на тестовых данных

### Milestone 5: Оптимизация навигации (недели 11–13)
- [ ] Inertial Navigation
- [ ] Adaptive step size
- [ ] RL для автоматического бюджета итераций
- [ ] Benchmarking: скорость vs качество

### Milestone 6: Масштабирование контекста (недели 14–17)
- [ ] Реализовать HierarchicalContextAggregator (§9.7.3)
- [ ] Sliding Window attention через FlexAttention
- [ ] Интеграция FAISS HNSW для Level 2 retrieval
- [ ] Global compression slots (Level 3)
- [ ] Тестирование на документах 10K, 50K, 100K предложений
- [ ] Ablation: sliding window size (64, 128, 256), retrieved K (32, 64, 128)
- [ ] Эксперименты с линейными альтернативами (GLA / Gated DeltaNet) для IPP
- [ ] Memory profiling на RTX 4090

---

## Приложение A: Ключевые формулы

### A.1 Langevin Dynamics

```
V_{t+1} = V_t − η · ∇_V E(V_query, V_t) + √(2η) · ε_t

ε_t ~ N(0, I)
```

### A.2 InfoNCE Loss

```
L_InfoNCE = -log( exp(−E(q, k⁺) / τ) / Σᵢ₌₁ᴺ exp(−E(q, kᵢ) / τ) )
```

### A.3 Momentum Update

```
m_t = β · m_{t-1} + (1 − β) · ∇_V E
V_{t+1} = V_t − η · m_t + √(2η) · ε_t
```

### A.4 OOD Projection

```
V ← V / ||V||₂ × r̄

r̄ = mean(||V_train||₂)  — средняя норма обучающих векторов
```

### A.5 Combined Training Loss для EBT

```
L = L_InfoNCE + λ_spec · L_spectral + λ_grad · L_gradient_penalty + λ_norm · max(0, ||E||² − margin)
```

---

## Приложение B: Зависимости и установка

```bash
# === Установка окружения ===

# Python 3.10+
conda create -n cebcm python=3.10
conda activate cebcm

# PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# SONAR
pip install sonar-space

# Данные
pip install datasets transformers sentence-transformers

# Retrieval
pip install faiss-gpu  # или faiss-cpu

# Метрики
pip install rouge-score nltk

# Efficient Attention (опционально, для экспериментов)
pip install xformers                 # Memory-efficient attention с structured sparsity
pip install fla-core                 # Flash Linear Attention: GLA, Gated DeltaNet, Mamba и др.

# Utility
pip install tqdm wandb
```

> **Заметка (v1.2):** PyTorch 2.x включает FlashAttention-2 через `torch.nn.functional.scaled_dot_product_attention` (SDPA) и FlexAttention через `torch.nn.attention.flex_attention` — дополнительных зависимостей для них не нужно. xFormers и FLA — опциональные альтернативы для продвинутых экспериментов.

---

## Приложение C: Архитектурные решения и обоснования

| Решение | Выбор | Почему НЕ альтернатива |
|---------|-------|------------------------|
| Autoencoder для PoC | SONAR | Готовый, проверенный, с декодером. ColBERT/SPLADE нет декодера |
| Размерность PoC | 1024d нативная | Без проектора, минимум потерь. Если мало — перейдём на sparse |
| Функция энергии | Specialized per-domain | Универсальная невозможна без потери точности |
| Инициализация V_init | Informed Noise → Base-LCM | Нулевая стоимость для первого теста, потом усложняем |
| Контекст | Retrieval top-K из кэша | Не пересылаем весь диалог, только релевантное |
| Attention в EBT | Только для Chain Scoring | Pairwise оценка не требует attention — экономия |
| Кэш в пространстве | SONAR 1024d (после deprojector) | Единое пространство, нет drift между «памятью» и «речью» |
| Attention для длинных контекстов | Hierarchical Sparse (window + retrieval + global slots) | Full attention O(N²) нереализуем при 50K+ предложений на 1 GPU. Ring Attention требует multi-GPU |
| Линейный attention для IPP | GLA / Gated DeltaNet (опционально) | Mamba менее экспрессивен; RetNet теряет информацию на длинных последовательностях |
| FlashAttention | PyTorch SDPA (встроенный, FA2 на Ampere) | Сторонние FA3/FA4 требуют Hopper GPU, которого нет в PoC |
| Positional encoding (Chain Head) | RoPE | Цепочки 5–20 элементов, стандартный RoPE достаточен |
| Positional encoding (Context, масштаб) | ALiBi (основной) | RoPE вращает вектора, нарушая семантическую геометрию SONAR. ALiBi сохраняет эмбеддинги нетронутыми |
| Token pre-processing | Token Merging (FAISS clustering + cosine dedup) | SONAR-пространство идеально калибровано для семантической дедупликации. 2–3× сжатие без потери информации |
| KV optimization | GQA 4:1 (PoC), MLA (scale) | MQA слишком агрессивен для EBT-критика. MLA философски выровнена с latent-space подходом |

---

*Документ является живой спецификацией. Обновляется по мере получения экспериментальных результатов.*

---

## 16. Прочие исследования (реализация отклонена)

> **Заметка (v1.3):** Следующие подходы были исследованы, но отклонены как избыточные или неоптимальные для архитектуры CEBCM с 1024d SONAR-векторами. Сохранены для справки.

### 16.1 MHLA-inspired Context Aggregator (отклонён в v1.3)

**Причина отклонения:** Сжатие 1024d → 512d (MHLA DeepSeek) теряет ~50% информации. При sentence-level vectors каждый бит ценен (в отличие от token-level с высокой redundancy). KV-кэш CEBCM на тысячи векторов ничтожно мал по сравнению с LLM — оптимизация через compression не оправдана.

Оригинальный подход: nn.MultiheadAttention для KV compression в n_compress_slots=16 векторов, затем TransformerEncoder. Заменён на Linear Attention + Surprise Global Tokens (§9.6).

### 16.2 Hierarchical Sparse Attention (отклонён в v1.3)

**Причина отклонения:** Трёхуровневая иерархия (Sliding Window + FAISS Retrieval + Global Compression Slots) — over-engineering. SONAR уже сжал предложение в один вектор — добавлять ещё уровни иерархии избыточно. Заменён на: External Compaction (§9.7) + Linear Attention (§9.9).

### 16.3 GQA / Grouped Query Attention (отклонён в v1.3)

**Причина отклонения:** KV-кэш CEBCM при 50K векторах: 50K × 1024d × 4 bytes × 2 (K+V) = 400 МБ. На GPU с 24 GB это ничтожно. GQA решает проблему, которой у нас нет, а качество attention гарантированно ухудшает.

### 16.4 Token Merging (отложен)

**Статус:** Потенциально полезен для предобработки очень длинных документов (кластеризация семантически дублирующихся предложений, 2–3× сжатие), но отложен до этапа масштабирования. FAISS kmeans + cosine dedup.
