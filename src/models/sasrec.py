"""
SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

Архитектура:
    history (B, L) → item_emb + pos_emb → dropout
    → N × SASRecBlock (каузальный self-attention + FFN, pre-norm)
    → LayerNorm
    → h[:, -1, :]  (последняя позиция = последний реальный айтем при left-padding)
    → logits = h @ item_emb.weight[:n_items].T  (tied weights)

SASRecAlphaFuse — тот же SASRec на эмбеддингах AlphaFuse (замороженная текстовая семантика
+ обучаемые ID в её null-координатах, см. models.components.AlphaFuseItemEmbedding).

Интерфейс общий с DreamRec-моделями и evaluate():
    compute_loss(history, target, mask)  → scalar
    recommend(history, mask, top_k, ...) → (B, top_k) LongTensor
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses import LOSS_METRICS as SASREC_LOSS_METRICS, infonce_loss

SASREC_RETRIEVAL_METRICS = ('cosine', 'dot')


# --------------------------------------------------------------------------- #
# SASRec block
# --------------------------------------------------------------------------- #

class SASRecBlock(nn.Module):
    """
    Один Transformer-блок SASRec с pre-norm и каузальным self-attention.

    Pre-norm (norm_first=True) — то же соглашение, что и в encoder.py.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         (B, L, d_model)
            attn_mask: (B · n_heads, L, L) bool — True = запрещённая пара; строится
                       causal_attn_mask() и уже включает каузальность и padding.
        """
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


def causal_attn_mask(mask: torch.FloatTensor, n_heads: int) -> torch.Tensor:
    """
    Маска self-attention (B · n_heads, L, L), True = запрещено: позиция видит только не-padding
    позиции слева от себя (включая себя).

    Диагональ разрешена всегда — иначе у запроса на padding-позиции запрещены все ключи, softmax
    берётся по строке из -inf и даёт NaN. С одним блоком такой NaN сидит на padding-позициях
    (их всё равно не читают), но со второго блока он расползается через values на реальные
    позиции: вес запрещённого ключа равен нулю, а 0 · NaN = NaN. Padding-позиции при этом
    смотрят только на себя и в реальные позиции не попадают: их ключи запрещены.

    Args:
        mask:    (B, L) — 1.0 реальные позиции, 0.0 padding (padding слева).
        n_heads: число голов; MHA ждёт маску, повторённую на каждую голову.
    """
    B, L = mask.shape
    device = mask.device
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    blocked = causal[None] | (mask == 0)[:, None, :]              # (B, L, L)
    blocked &= ~torch.eye(L, dtype=torch.bool, device=device)     # диагональ всегда разрешена
    return blocked.repeat_interleave(n_heads, dim=0)              # (B · n_heads, L, L)


# --------------------------------------------------------------------------- #
# SASRec model
# --------------------------------------------------------------------------- #

class SASRec(nn.Module):
    """
    SASRec для последовательной рекомендации.

    Args:
        n_items:     число товаров (без padding). Индексы [0, n_items-1].
        padding_idx: индекс padding-токена (= n_items = 11924 для ATG).
        d_model:     размерность эмбеддингов и скрытых состояний.
        n_heads:     число голов в self-attention.
        n_layers:    число SASRecBlock.
        d_ff:        размерность FFN внутри блока.
        dropout:     вероятность dropout (attention и FFN).
        max_seq_len: максимальная длина истории.
        tie_weights: использовать ли item_embedding как output projection.
        loss:        'ce' — full-softmax cross-entropy;
                     'infonce' — контрастивный InfoNCE с neg_ratio негативами.
        train_prefixes: True — лосс на каждой позиции последовательности (каждая позиция
                     предсказывает следующий товар) за один forward, как в оригинальном SASRec;
                     False — только последняя позиция → next.
        neg_ratio:   число случайных негативов на один позитив (только для 'infonce').
        temperature: температура: логиты лосса = сходство / temperature (оба лосса).
        loss_metric: сходство h и эмбеддингов товаров в лоссе: 'cosine' — L2-нормировка обоих,
                     логиты ограничены ±1/temperature; 'dot' — скалярное произведение, нормы
                     эмбеддингов обучаются и влияют на логиты.
        retrieval_metric: ранжирование в recommend(): 'cosine' — косинус h и эмбеддингов товаров;
                     'dot' — логиты (h · эмбеддинги при tie_weights, иначе output_proj), здесь на
                     порядок влияют и нормы эмбеддингов товаров.
        items:      внешний модуль эмбеддингов товаров с интерфейсом ItemEmbedding
                     (forward(idx), table, d_emb, info) вместо собственной обучаемой таблицы;
                     требует tie_weights=True. None — своя таблица item_embedding.
        freeze_items: True — своя таблица item_embedding остаётся случайной инициализацией
                     N(0, 0.02) и не обучается (фиксированные случайные коды товаров).
    """

    def __init__(
        self,
        n_items: int = 11924,
        padding_idx: int = 11924,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.2,
        max_seq_len: int = 10,
        tie_weights: bool = True,
        loss: str = 'infonce',
        neg_ratio: int = 64,
        temperature: float = 0.07,
        loss_metric: str = 'cosine',
        retrieval_metric: str = 'cosine',
        train_prefixes: bool = False,
        items: nn.Module | None = None,
        freeze_items: bool = False,
    ):
        super().__init__()
        if freeze_items and items is not None:
            raise ValueError('freeze_items относится к своей таблице item_embedding, а не к внешним items')
        if loss_metric not in SASREC_LOSS_METRICS:
            raise ValueError(f'loss_metric должен быть одним из {SASREC_LOSS_METRICS}, '
                             f'получено {loss_metric!r}')
        if retrieval_metric not in SASREC_RETRIEVAL_METRICS:
            raise ValueError(f'retrieval_metric должен быть одним из {SASREC_RETRIEVAL_METRICS}, '
                             f'получено {retrieval_metric!r}')
        if items is not None:
            if not tie_weights:
                raise ValueError('внешние эмбеддинги товаров (items) требуют tie_weights=True')
            if items.d_emb != d_model:
                raise ValueError(f'items.d_emb={items.d_emb} не совпадает с d_model={d_model}')

        self.n_items     = n_items
        self.padding_idx = padding_idx
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.d_ff        = d_ff
        self.tie_weights = tie_weights
        self.loss        = loss
        self.neg_ratio   = neg_ratio
        self.temperature = temperature
        self.loss_metric = loss_metric
        self.retrieval_metric = retrieval_metric
        self.train_prefixes   = train_prefixes
        self.freeze_items     = freeze_items

        # Эмбеддинги товаров: внешний модуль или своя обучаемая таблица.
        # padding_idx обнуляет вектор и запрещает градиент через него.
        self.items = items
        if items is None:
            self.item_embedding = nn.Embedding(
                n_items + 1, d_model, padding_idx=padding_idx
            )
        # Learnable absolute positional embeddings (0..max_seq_len-1).
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout_emb   = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SASRecBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection используется только когда tie_weights=False.
        if not tie_weights:
            self.output_proj = nn.Linear(d_model, n_items, bias=False)

        self._init_weights()
        if freeze_items:
            self.item_embedding.weight.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        if self.items is None:
            nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
            # Явно обнуляем padding-вектор (nn.Embedding с padding_idx делает это,
            # но после normal_init нужно обнулить вручную).
            with torch.no_grad():
                self.item_embedding.weight[self.padding_idx].zero_()
        nn.init.zeros_(self.pos_embedding.weight)
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)
            for layer in block.ff:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        if not self.tie_weights:
            nn.init.xavier_uniform_(self.output_proj.weight)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def encode(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        Args:
            history: (B, L) — индексы товаров, padding слева.
            mask:    (B, L) — 1.0 реальные позиции, 0.0 padding.

        Returns:
            (B, L, d_model) — представление после каждой позиции. Attention каузальный, поэтому
            позиция i видит только товары ≤ i, то есть это представления всех префиксов истории.
        """
        L = history.shape[1]
        device = history.device

        # Embeddings
        x = self._embed(history)                                        # (B, L, d_model)
        positions = torch.arange(L, device=device).unsqueeze(0)         # (1, L)
        x = x + self.pos_embedding(positions)                           # (B, L, d_model)
        x = self.dropout_emb(x)

        # Каузальность + padding одной маской; диагональ разрешена, поэтому NaN не возникает
        attn_mask = causal_attn_mask(mask, self.n_heads)                  # (B · n_heads, L, L)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.final_norm(x)
        # Реальные позиции считаются корректно; padding-позиции смотрят только на себя, их не читают
        return x                                                          # (B, L, d_model)

    def forward(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.Tensor:
        """История → представление пользователя (B, d_model) из последней позиции."""
        return self.encode(history, mask)[:, -1, :]

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        history: torch.LongTensor,
        target: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        Loss в зависимости от self.loss:
          'ce'      — full-softmax cross-entropy по всем n_items.
          'infonce' — контрастивный InfoNCE с self.neg_ratio негативами.

        train_prefixes=False — один таргет на строку: представление последней позиции → next.
        train_prefixes=True  — позиционный лосс, как в оригинальном SASRec: за один forward
        каждая позиция i предсказывает товар i+1 (последняя — next). Даёт столько же обучающих
        сигналов, сколько разворачивание префиксов в отдельные строки, но без лишних forward.
        """
        if not self.train_prefixes:
            h = self.forward(history, mask)                              # (B, d_model)
            return self._loss(h, target)

        # Таргет позиции i — следующий товар истории, у последней позиции — next
        labels = torch.cat([history[:, 1:], target[:, None]], dim=1)     # (B, L)
        valid = mask > 0                                                 # padding слева: i реальна → i+1 тоже
        h = self.encode(history, mask)[valid]                            # (N, d_model), N = число реальных позиций
        return self._loss(h, labels[valid])

    def _loss(self, h: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        if self.loss == 'infonce':
            return infonce_loss(h, target, self._embed, self.n_items, self.neg_ratio,
                                self.loss_metric, self.temperature)
        if self.loss_metric == 'cosine':
            W = self._item_table() if self.tie_weights else self.output_proj.weight
            logits = F.normalize(h, dim=-1) @ F.normalize(W, dim=-1).T
        else:
            logits = self._logits(h)
        return F.cross_entropy(logits / self.temperature, target)

    # ------------------------------------------------------------------ #
    # Recommend
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def recommend(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
        top_k: int = 20,
        num_steps: int | None = None,   # не используется, для совместимости с evaluate()
        w_cfg: float | None = None,     # не используется, для совместимости с evaluate()
    ) -> torch.LongTensor:
        """
        Возвращает top-K товаров по убыванию score.

        Args:
            history: (B, L)
            mask:    (B, L)
            top_k:   сколько товаров вернуть

        Returns:
            (B, top_k) LongTensor — индексы товаров
        """
        h = self.forward(history, mask)                                  # (B, d_model)
        if self.retrieval_metric == 'cosine':
            W = self._item_table()
            scores = F.normalize(h, dim=-1) @ F.normalize(W, dim=-1).T
        else:
            scores = self._logits(h)                                     # (B, n_items)
        return scores.topk(top_k, dim=-1).indices                        # (B, top_k)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _embed(self, idx: torch.LongTensor) -> torch.Tensor:
        """Индексы товаров (любой формы) → эмбеддинги (..., d_model)."""
        return self.item_embedding(idx) if self.items is None else self.items(idx)

    def _item_table(self) -> torch.Tensor:
        """(n_items, d_model) — эмбеддинги всех товаров без padding."""
        return self.item_embedding.weight[:self.n_items] if self.items is None else self.items.table

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        """(B, d_model) → (B, n_items) logits."""
        if self.tie_weights:
            # Tied weights: используем эмбеддинги товаров (строки 0..n_items-1)
            return h @ self._item_table().T
        else:
            return self.output_proj(h)

    def config(self) -> dict:
        """Параметры модели для чекпойнта (model_cfg) — их читают ноутбуки анализа."""
        return {
            'model': type(self).__name__,
            'n_items': self.n_items,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': len(self.blocks),
            'd_ff': self.d_ff,
            'tie_weights': self.tie_weights,
            'loss': self.loss,
            'neg_ratio': self.neg_ratio,
            'temperature': self.temperature,
            'loss_metric': self.loss_metric,
            'retrieval_metric': self.retrieval_metric,
            'train_prefixes': self.train_prefixes,
            **({'freeze_items': True} if self.freeze_items else {}),
            **(self.items.info if self.items is not None else {}),
        }

    def param_summary(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {'total_trainable': trainable, 'total_frozen': frozen}


class SASRecAlphaFuse(SASRec):
    """
    SASRec на эмбеддингах AlphaFuse (models.components.AlphaFuseItemEmbedding): замороженная
    отбеленная семантика + обучаемые ID в её null-координатах; выход привязан к ним же.
    """

    def __init__(self, *, item_embeddings_raw, padding_idx: int, d_model: int,
                 null_dim: int | None, null_thres: float | None, fusion: str, id_init: str,
                 whitening_cache, n_heads: int, n_layers: int, d_ff: int, dropout: float,
                 max_seq_len: int, loss: str, neg_ratio: int, temperature: float,
                 retrieval_metric: str, loss_metric: str = 'cosine', train_prefixes: bool = False):
        # Импорт здесь: models.components сам импортирует SASRecBlock из этого модуля
        from models.components import AlphaFuseItemEmbedding

        items = AlphaFuseItemEmbedding(item_embeddings_raw, padding_idx, d_model, null_dim,
                                       null_thres, fusion, id_init, whitening_cache)
        super().__init__(
            n_items=items.n_items, padding_idx=padding_idx, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout, max_seq_len=max_seq_len,
            tie_weights=True, loss=loss, neg_ratio=neg_ratio, temperature=temperature,
            loss_metric=loss_metric, retrieval_metric=retrieval_metric, train_prefixes=train_prefixes, items=items,
        )


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    python -m src.models.sasrec
    """
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    N_ITEMS     = 11924
    PADDING_IDX = N_ITEMS
    B, L        = 4, 10

    model = SASRec(
        n_items=N_ITEMS,
        padding_idx=PADDING_IDX,
        d_model=256,
        n_heads=4,
        n_layers=2,
        d_ff=1024,
        dropout=0.2,
        max_seq_len=L,
        tie_weights=True,
    ).to(device)

    summary = model.param_summary()
    print(f'Trainable params: {summary["total_trainable"]:,}')

    # Батч с left-padding: первые 5 позиций — padding, последние 5 — реальные
    history = torch.full((B, L), PADDING_IDX, dtype=torch.long, device=device)
    history[:, 5:] = torch.randint(0, N_ITEMS, (B, 5), device=device)
    mask = torch.zeros(B, L, device=device)
    mask[:, 5:] = 1.0
    target = torch.randint(0, N_ITEMS, (B,), device=device)

    # Test 1: forward shape
    h = model.forward(history, mask)
    assert h.shape == (B, 256), f'Expected (B, 256), got {h.shape}'
    print(f'forward output: {h.shape}  OK')

    # Test 2: loss scalar
    loss = model.compute_loss(history, target, mask)
    assert loss.shape == (), f'Expected scalar, got {loss.shape}'
    assert not torch.isnan(loss), 'Loss is NaN'
    print(f'compute_loss: {loss.item():.4f}  OK')

    # Test 3: recommend shape
    recs = model.recommend(history, mask, top_k=20)
    assert recs.shape == (B, 20), f'Expected (B, 20), got {recs.shape}'
    print(f'recommend shape: {recs.shape}  OK')

    # Test 4: рекомендации не включают padding_idx
    assert (recs != PADDING_IDX).all(), 'Padding idx should not appear in recommendations'
    print(f'No padding in recommendations  OK')

    # Test 4b: позиционный лосс — тот же набор пар «префикс → следующий товар» за один forward
    model.train_prefixes = True
    loss_pref = model.compute_loss(history, target, mask)
    model.train_prefixes = False
    assert torch.isfinite(loss_pref)
    with torch.no_grad():                       # при истории из одного товара позиция одна → лоссы равны
        one = torch.zeros(B, L, dtype=torch.long, device=device) + PADDING_IDX
        one[:, -1] = history[:, -1]
        m_one = torch.zeros(B, L, device=device); m_one[:, -1] = 1.0
        model.eval()
        model.loss = 'ce'                       # у infonce негативы случайны, лоссы не сравнить
        a = model.compute_loss(one, target, m_one)
        model.train_prefixes = True
        b = model.compute_loss(one, target, m_one)
        model.train_prefixes = False
        model.loss = 'infonce'
        model.train()
        assert torch.allclose(a, b, atol=1e-5), (a.item(), b.item())
    print(f'Позиционный лосс: {loss_pref.item():.4f} (последняя позиция {loss.item():.4f})  OK')

    # Test 4d: loss_metric — оба сходства в обоих лоссах; dot зависит от нормы h, cosine нет
    h_rand = torch.randn(B, 256, device=device)
    for loss_name in ('infonce', 'ce'):
        model.loss = loss_name
        for metric in SASREC_LOSS_METRICS:
            model.loss_metric = metric
            with torch.no_grad():
                torch.manual_seed(1)                # одни и те же негативы infonce
                l1 = model._loss(h_rand, target)
                torch.manual_seed(1)
                l3 = model._loss(3.0 * h_rand, target)
            assert torch.isfinite(l1)
            assert torch.allclose(l1, l3, atol=1e-5) == (metric == 'cosine'), (loss_name, metric)
    model.loss, model.loss_metric = 'infonce', 'cosine'
    print('loss_metric cosine/dot в infonce и ce  OK')

    # Test 4c: при любом числе слоёв представление живое и разное у разных историй.
    # Раньше со второго блока NaN с padding-позиций расползался на реальные, h обнулялся
    # и все пользователи получали одинаковые рекомендации.
    for n_layers in (1, 2, 3):
        deep = SASRec(n_items=N_ITEMS, padding_idx=PADDING_IDX, d_model=32, n_heads=2,
                      n_layers=n_layers, d_ff=64, dropout=0.0, max_seq_len=L).to(device).eval()
        with torch.no_grad():
            h_deep = deep(history, mask)
            recs_deep = deep.recommend(history, mask, top_k=5)
        assert torch.isfinite(h_deep).all(), f'n_layers={n_layers}: NaN или inf в представлении'
        assert h_deep.norm(dim=-1).min() > 1e-6, f'n_layers={n_layers}: нулевое представление'
        assert not torch.equal(recs_deep[0], recs_deep[1]), \
            f'n_layers={n_layers}: одинаковые рекомендации у разных историй'
        # Паддинг не влияет на результат: та же история в окне большей длины
        wide = torch.full((B, L * 2), PADDING_IDX, dtype=torch.long, device=device)
        wide[:, L:] = history
        wide_mask = torch.zeros(B, L * 2, device=device); wide_mask[:, L:] = mask
        deep.pos_embedding = nn.Embedding(L * 2, 32).to(device)      # позиции считаются от начала окна
        nn.init.zeros_(deep.pos_embedding.weight)
        with torch.no_grad():
            assert torch.allclose(deep(history, mask), deep(wide, wide_mask), atol=1e-5), \
                f'n_layers={n_layers}: лишний паддинг меняет представление'
    print('Глубина 1–3 слоя: представление живое, паддинг не влияет  OK')

    # Test 5: градиент идёт через compute_loss
    loss2 = model.compute_loss(history, target, mask)
    loss2.backward()
    assert model.item_embedding.weight.grad is not None, 'Embedding должен иметь градиент'
    # Градиент на padding-строке должен быть нулевым
    pad_grad = model.item_embedding.weight.grad[PADDING_IDX]
    assert pad_grad.abs().max().item() == 0.0, 'Padding embedding grad should be zero'
    print(f'Gradient check  OK')

    # Test 6: полная последовательность vs частичная — разные предсказания
    history_full = torch.randint(0, N_ITEMS, (B, L), device=device)
    mask_full = torch.ones(B, L, device=device)
    model.eval()
    with torch.no_grad():
        h_full  = model.forward(history_full, mask_full)
        h_short = model.forward(history, mask)
    assert not torch.allclose(h_full, h_short), 'Full vs short sequences should differ'
    print(f'Full vs short sequences differ  OK')

    # Test 7: SASRecAlphaFuse — градиент только в ID, семантика заморожена, оба режима слияния
    import tempfile
    from pathlib import Path
    import numpy as np

    D, K = 32, 8
    emb_raw = np.random.default_rng(0).standard_normal((N_ITEMS, 64)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        for fusion in ('add', 'replace'):
            af = SASRecAlphaFuse(
                item_embeddings_raw=emb_raw, padding_idx=PADDING_IDX, d_model=D, null_dim=K,
                null_thres=None, fusion=fusion, id_init='normal', whitening_cache=Path(tmp) / 'wh.npz',
                n_heads=1, n_layers=1, d_ff=D, dropout=0.1, max_seq_len=L, loss='infonce',
                neg_ratio=16, temperature=0.07, retrieval_metric='cosine',
            ).to(device).train()
            af.compute_loss(history, target, mask).backward()
            id_grad = af.items.id_embedding.weight.grad
            assert af.items.semantic.weight.grad is None, 'Semantic part must be frozen'
            assert id_grad is not None and id_grad[PADDING_IDX].abs().max().item() == 0.0
            with torch.no_grad():
                sem = af.items.semantic.weight[:N_ITEMS, :D - K]
                assert torch.equal(af.items.table[:, :D - K], sem), 'Non-null part must equal semantic'
            af.eval()
            for metric in SASREC_RETRIEVAL_METRICS:
                af.retrieval_metric = metric
                recs = af.recommend(history, mask, top_k=20)
                assert recs.shape == (B, 20) and (recs != PADDING_IDX).all()
            # dot ранжирует по h · e: совпадает с логитами, а при разных нормах e отличается от cosine
            with torch.no_grad():
                h_af = af(history, mask)
                assert torch.equal(recs, (h_af @ af.items.table.T).topk(20, dim=-1).indices)
            print(f'SASRecAlphaFuse fusion={fusion}: {af.param_summary()} | {af.items.info}  OK')

    print('\nAll checks passed')
