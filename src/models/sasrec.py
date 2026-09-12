"""
SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

Архитектура:
    history (B, L) → item_emb + pos_emb → dropout
    → N × SASRecBlock (каузальный self-attention + FFN, pre-norm)
    → LayerNorm
    → h[:, -1, :]  (последняя позиция = последний реальный айтем при left-padding)
    → logits = h @ item_emb.weight[:n_items].T  (tied weights)

Интерфейс общий с DreamRec-моделями и evaluate():
    compute_loss(history, target, mask)  → scalar
    recommend(history, mask, top_k, ...) → (B, top_k) LongTensor
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, L, d_model)
            attn_mask:        (L, L) bool — True = запрещённая пара (каузальный)
            key_padding_mask: (B, L) bool — True = игнорировать ключ (padding)
        """
        h = self.norm1(x)
        h, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


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
        neg_ratio:   число случайных негативов на один позитив (только для 'infonce').
        temperature: температура InfoNCE (только для 'infonce').
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
    ):
        super().__init__()

        self.n_items     = n_items
        self.padding_idx = padding_idx
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.d_ff        = d_ff
        self.tie_weights = tie_weights
        self.loss        = loss
        self.neg_ratio   = neg_ratio
        self.temperature = temperature

        # Learnable item embeddings.
        # padding_idx обнуляет вектор и запрещает градиент через него.
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

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
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

    def forward(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        Args:
            history: (B, L) — индексы товаров, padding слева.
            mask:    (B, L) — 1.0 реальные позиции, 0.0 padding.

        Returns:
            h: (B, d_model) — представление пользователя из последней позиции.
        """
        B, L = history.shape
        device = history.device

        # Embeddings
        x = self.item_embedding(history)                                 # (B, L, d_model)
        positions = torch.arange(L, device=device).unsqueeze(0)         # (1, L)
        x = x + self.pos_embedding(positions)                           # (B, L, d_model)
        x = self.dropout_emb(x)

        # Каузальная маска: позиция i не может смотреть вперёд (j > i).
        # PyTorch MHA: True → добавляет -inf к logit → атентион обнуляется.
        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
        )                                                                 # (L, L)

        # Ключи на padding-позициях игнорируются.
        key_padding_mask = (mask == 0)                                   # (B, L)

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)

        # Padding query-позиции могут дать NaN (softmax над всеми -inf),
        # но мы читаем только последнюю позицию, которая всегда реальна.
        # nan_to_num — защита от распространения NaN через residual connection.
        x = torch.nan_to_num(x, nan=0.0)

        return x[:, -1, :]                                               # (B, d_model)

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
        """
        h = self.forward(history, mask)                                  # (B, d_model)
        if self.loss == 'infonce':
            return self._infonce_loss(h, target)
        return F.cross_entropy(self._logits(h), target)

    def _infonce_loss(
        self,
        h: torch.Tensor,
        target: torch.LongTensor,
    ) -> torch.Tensor:
        """
        InfoNCE: контрастивный loss с neg_ratio случайными негативами.

        Logit = cosine_similarity(h, e) / temperature.
        Метка 0 — позитив стоит на первом месте в конкатенированном тензоре.

        Args:
            h:      (B, d_model) — user representation
            target: (B,) — индексы позитивных айтемов
        """
        B      = h.shape[0]
        device = h.device
        W      = self.item_embedding.weight[:self.n_items]               # (N, d_model)

        # Случайные негативы: (B, neg_ratio), исключаем совпадения с позитивом
        neg = torch.randint(0, self.n_items, (B, self.neg_ratio), device=device)
        # Маскируем совпадения одним проходом (коллизии редки, двойной цикл не нужен)
        collision = (neg == target.unsqueeze(1))
        if collision.any():
            replacement = torch.randint(0, self.n_items, (B, self.neg_ratio), device=device)
            neg = torch.where(collision, replacement, neg)

        # Эмбеддинги позитива и негативов
        e_pos = W[target]                                                # (B, d_model)
        e_neg = W[neg]                                                   # (B, neg_ratio, d_model)

        # L2-нормировка
        h     = F.normalize(h,     dim=-1)                              # (B, d_model)
        e_pos = F.normalize(e_pos, dim=-1)                              # (B, d_model)
        e_neg = F.normalize(e_neg, dim=-1)                              # (B, neg_ratio, d_model)

        # Косинусные логиты
        pos_logit = (h * e_pos).sum(dim=-1, keepdim=True)               # (B, 1)
        neg_logit = torch.bmm(e_neg, h.unsqueeze(-1)).squeeze(-1)       # (B, neg_ratio)

        logits = torch.cat([pos_logit, neg_logit], dim=-1) / self.temperature  # (B, 1+neg_ratio)
        labels = torch.zeros(B, dtype=torch.long, device=device)        # позитив на индексе 0
        return F.cross_entropy(logits, labels)

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
        if self.loss == 'infonce':
            # При InfoNCE обучении пространство нормировано — используем косинус
            W = self.item_embedding.weight[:self.n_items]
            scores = F.normalize(h, dim=-1) @ F.normalize(W, dim=-1).T
        else:
            scores = self._logits(h)                                     # (B, n_items)
        return scores.topk(top_k, dim=-1).indices                        # (B, top_k)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        """(B, d_model) → (B, n_items) logits."""
        if self.tie_weights:
            # Tied weights: используем item_embedding (строки 0..n_items-1)
            return h @ self.item_embedding.weight[:self.n_items].T
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
        }

    def param_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total_trainable': total, 'total_frozen': 0}


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

    print('\nAll checks passed')
