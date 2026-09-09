# FrozenText_FM

**Sequential recommendation через Conditional Flow Matching на замороженных текстовых эмбеддингах.**

FrozenText_FM генерирует эмбеддинг следующего товара, обусловившись на историю
взаимодействий пользователя, а затем ранжирует каталог по близости к
сгенерированному вектору. В отличие от диффузионной версии (DreamRecText / DDPM),
здесь используется **Flow Matching** (velocity + ODE-сэмплинг), двунаправленный
энкодер и **FiLM**-conditioning в денойзере.

Реализация: [`src/models/flow_matcher.py`](../src/models/flow_matcher.py) (`FlowMatcher`),
компоненты — [`src/models/encoder.py`](../src/models/encoder.py) (`HistoryEncoder`),
[`src/models/denoiser.py`](../src/models/denoiser.py) (`MLPDenoiser`).
Конфиг: [`configs/fm_text.yaml`](../configs/fm_text.yaml).

---

## Идея

Классический sequential recommender предсказывает распределение над каталогом.
Здесь задача переформулирована как **условная генерация**:

1. История товаров → энкодер → conditioning-вектор `c`.
2. Из гауссова шума `z₀ ~ N(0, I)`, обусловившись на `c`, генерируется вектор
   `oracle` — приближение эмбеддинга целевого (следующего) товара.
3. Ранжирование каталога по близости `oracle` к замороженным эмбеддингам товаров.

Эмбеддинги товаров **заморожены** и являются семантическими: это эмбеддинги
текстовых описаний (TEM, `text-embedding-3-large`, 3072d), сжатые PCA + whitening
до 256d. Заморозка предотвращает коллапс (обучаемые эмбеддинги в генеративной
постановке схлопываются к нулю и вырождают retrieval).

---

## Общая схема

```
history (B, L)  ─┐
                 │  frozen item emb (PCA 256d)
mask (B, L)  ────┤        │
                 ▼        ▼
        ┌──────────────────────────┐
        │      HistoryEncoder       │   двунаправленный Transformer
        │  256 → 512 → 2×TF → 512   │   (2 слоя, 8 голов, no causal mask)
        └──────────────────────────┘
                     │ c  (B, 512)  — conditioning
                     ▼
   z₀ ~ N(0,I₂₅₆) ──►┌──────────────────────────┐
        τ ∈ [0,1] ──►│      MLPDenoiser (FiLM)   │──► v_θ  (B, 256)  velocity
                     │  z_τ(256)+t_emb → 3×FiLM  │
                     └──────────────────────────┘
                     │  Euler ODE (num_steps), CFG
                     ▼
                 oracle (B, 256)
                     │  cosine / dot / L2
                     ▼
              top-K товаров
```

---

## Компоненты

### 1. Замороженные эмбеддинги товаров (PCA + whitening)

- Вход: сырые TEM-эмбеддинги `(N_items, 3072)`.
- **PCA-проекция** `3072 → n_components (256)` + **whitening**: приводит
  ковариационную матрицу эмбеддингов к единичной (`Var(E) = I`).
  `per_component_whitening: true` — по каждой компоненте отдельно.
- Параметры (`mu`, `V`, `scale`) хранятся как buffers и попадают в чекпойнт —
  проекция при инференсе идентична обучению.
- Итоговая таблица `nn.Embedding.from_pretrained(..., freeze=True)` с нулевой
  строкой для padding-токена. Градиенты через неё не идут.

### 2. HistoryEncoder (история → conditioning `c`)

[`src/models/encoder.py`](../src/models/encoder.py):

1. lookup замороженных эмбеддингов → `(B, L, 256)`;
2. `input_proj`: `256 → d_hidden (512)` — **единственный обучаемый слой, видящий
   сырые эмбеддинги**;
3. `+ learnable positional embedding` (позиции `0..max_seq_len-1`);
4. `nn.TransformerEncoder`: `n_layers=2`, `n_heads=8`, `d_ff=2048`, **pre-norm**,
   **двунаправленное** внимание (маскируется только padding, каузальной маски
   нет — в отличие от SASRec);
5. last-item pooling: `x[:, -1]` (padding слева ⇒ последняя позиция всегда
   реальный товар);
6. `output_proj`: `512 → d_cond (512)`.

Выход: conditioning-вектор `c` размерности `d_cond = 512`.

### 3. MLPDenoiser с FiLM (предсказание velocity)

[`src/models/denoiser.py`](../src/models/denoiser.py). Предсказывает векторное поле
`v_θ(z_τ, τ, c)`:

- **Основной поток:** `[z_τ (256); t_emb]` → `input_proj` → `d_block (256)` →
  `n_blocks (3)` × `FiLMResidualBlock` → `final_norm` → `output_proj` → `256`.
- **Time embedding:** синусоидальный по непрерывному `τ ∈ [0,1]`
  (`time_emb_dim = 128`), считается **внутри** денойзера.
- **Conditioning через FiLM:** `c` → `cond_proj` → пара `(scale, shift)` для
  **каждого** блока; в блоке `h ← h + MLP(LayerNorm(h)·(1+scale) + shift)`.
  `cond_proj` инициализирован нулём ⇒ в начале обучения условие нейтрально.

> Почему FiLM, а не concat: при конкатенации `[z_τ, c, t]` input-проекция учится
> в основном на `z_τ` и почти игнорирует `c`. Для velocity-предсказания
> (у которого нет «лёгкого сигнала» из `z_τ`, в отличие от x₀-предсказания в DDPM)
> это фатально — loss не снижается. FiLM подаёт `c` отдельно и модулирует каждый
> блок, из-за чего conditioning реально используется.

### 4. Classifier-Free Guidance (CFG)

- Обучаемый unconditional-токен `phi` размерности `d_cond`.
- При обучении с вероятностью `p_uncond` условие `c` заменяется на `phi`
  (тренируется безусловный путь).
- При инференсе: `v = (1 + w_cfg)·v_cond − w_cfg·v_uncond`.

---

## Обучение (Conditional Flow Matching)

Линейный (rectified) путь между шумом и целью:

```
z_τ    = (1 − τ)·z₀ + τ·z₁,     τ ~ U(0, 1)
z₀     ~ N(0, I₂₅₆)             (шум)
z₁     = frozen_emb(target)     (эмбеддинг целевого товара, 256d)
u_true = z₁ − z₀                (истинная velocity)

L = E ‖ v_θ(z_τ, τ, c) − u_true ‖²
```

Обучается только энкодер (кроме frozen-эмбеддингов), денойзер и `phi`.

Детали тренировки ([`src/train_fm_text.py`](../src/train_fm_text.py)):
AdamW, EMA весов, mixed precision (bfloat16), linear warmup, grad clipping,
валидация full-rank каждые N эпох, сохранение лучшего по NDCG@20.

---

## Инференс

1. `c = encoder(history, mask)`.
2. **Euler ODE** от `τ=0` (шум) до `τ=1` (цель) за `num_sample_steps` шагов, с CFG:

   ```
   z ← z + dt · [ (1 + w_cfg)·v_θ(z, τ, c) − w_cfg·v_θ(z, τ, phi) ],   dt = 1/num_steps
   ```

   Результат — `oracle` (256d). Flow Matching на почти прямом пути сходится за
   малое число шагов (единицы), в отличие от DDPM (десятки-сотни).
3. **Retrieval:** top-K товаров по близости `oracle` к замороженным эмбеддингам
   (`cosine`, `dot` или `l2`).

---

## Гиперпараметры ([`configs/fm_text.yaml`](../configs/fm_text.yaml))

| Группа | Параметр | Значение |
|---|---|---|
| Данные | `n_components` (PCA) | 256 |
| | `batch_size` | 256 |
| | `max_seq_len` | 10 |
| Энкодер | `d_hidden` / `d_cond` | 512 / 512 |
| | `layers` / `heads` / `d_ff` | 2 / 8 / 2048 |
| | `dropout` | 0.1 |
| Денойзер | `d_block` / `d_ff` | 256 / 1024 |
| | `n_blocks` / `time_emb_dim` | 3 / 128 |
| Flow Matching | `p_uncond` (CFG dropout) | 0.1 |
| | `w_cfg` (инференс) | 1.0 |
| | `per_component_whitening` | true |
| | `retrieval_metric` | cosine |
| Оптимизация | `lr` / `weight_decay` | 1e-4 / 0.0 |
| | `num_epochs` | 500 |
| EMA | `decay` | 0.9 |
| Eval | `num_sample_steps` (Euler) | 50 |

Обучаемых параметров ≈ 9.2M, замороженных ≈ 3.05M (таблица эмбеддингов).

---

## Отличие от DDPM-версии (DreamRecText)

FrozenText_FM отличается от DDPM-модели **по трём осям**, а не только генератором:

| | DreamRecText (DDPM) | FrozenText_FM |
|---|---|---|
| Генеративная модель | DDPM, x₀-prediction, обратная цепочка | Flow Matching, velocity, Euler ODE |
| Энкодер | 1 каузальный SASRec-блок в d_emb | двунаправленный Transformer (2 слоя), проекции 512 |
| Conditioning в денойзере | конкатенация `[z, h, t]` | FiLM (модуляция каждого блока) |
| CFG-токен | обучаемый `none_embedding` | обучаемый `phi` |
| Пространство эмбеддингов | frozen PCA/whitened TEM | frozen PCA/whitened TEM |

Из-за нескольких одновременных отличий сравнение «DDPM vs FM» **не изолированное**:
сопоставимое качество трактуется как устойчивость подхода (замороженные
семантические эмбеддинги + генеративная постановка), а не как чистый эффект смены
генератора.

---

## Запуск

Обучение:

```bash
python src/train_fm_text.py --config configs/fm_text.yaml
```

Оценка (full-rank HR@K / NDCG@K, кривые, sweep по числу шагов сэмплинга) —
ноутбуки [`notebooks/fm_text_results.ipynb`](../notebooks/fm_text_results.ipynb) и
[`notebooks/train_eval.ipynb`](../notebooks/train_eval.ipynb).
