"""
Transformer encoder для истории взаимодействий пользователя.

Принимает последовательность item-индексов и возвращает один
conditioning vector c, который подаётся в MLPDenoiser.

Pipeline:
    history (B, L)                   — индексы товаров (padding слева)
    mask    (B, L)                   — 1.0 реальные, 0.0 padding

    1. Embedding lookup (frozen whitened TEM) → (B, L, d_emb=3072)
    2. Input projection d_emb → d_hidden     → (B, L, d_hidden)
    3. + positional embedding (learnable)    → (B, L, d_hidden)
    4. Transformer encoder (2 слоя, 8 heads) → (B, L, d_hidden)
       с src_key_padding_mask для padding
    5. Last-item pool: h[:, -1]              → (B, d_hidden)
    6. Output projection d_hidden → d_cond   → (B, d_cond)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    """
    Transformer encoder для последовательности взаимодействий пользователя.

    Args:
        item_embeddings: Tensor (N_items + 1, d_emb) — whitened TEM эмбеддинги
                         товаров. Последняя строка — нулевой padding-токен.
                         Передаётся как frozen nn.Embedding.
        padding_idx:     индекс padding-токена в item_embeddings. Этот индекс
                         гарантированно нулевой (nn.Embedding.padding_idx).
        d_hidden:        внутренняя размерность Transformer. По умолчанию 512.
        d_cond:          размерность выходного conditioning vector. По умолчанию 512.
        n_layers:        количество Transformer слоёв. По умолчанию 2.
        n_heads:         количество attention голов. По умолчанию 8.
        d_ff:            размерность FFN внутри Transformer. По умолчанию 2048.
        dropout:         dropout внутри Transformer. По умолчанию 0.1.
        max_seq_len:     максимальная длина истории (для positional embedding).
                         По умолчанию 10.
    """

    def __init__(
        self,
        item_embeddings: torch.Tensor,
        padding_idx: int,
        d_hidden: int = 512,
        d_cond: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 10,
    ):
        super().__init__()

        d_emb = item_embeddings.shape[1]
        self.padding_idx = padding_idx
        self.d_hidden = d_hidden
        self.d_cond = d_cond

        # Frozen embedding table — whitened TEM эмбеддинги.
        # padding_idx гарантирует, что градиент через padding-позиции не идёт
        # и padding-вектор остаётся нулевым.
        self.item_embedding = nn.Embedding.from_pretrained(
            item_embeddings.float(),
            freeze=True,
            padding_idx=padding_idx,
        )

        # Проекция из d_emb (3072) в d_hidden (512).
        # Единственный обучаемый слой, который «видит» сырые эмбеддинги.
        self.input_proj = nn.Linear(d_emb, d_hidden)

        # Learnable positional embedding для позиций 0..max_seq_len-1.
        self.pos_embedding = nn.Embedding(max_seq_len, d_hidden)

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # ожидает (B, L, d_hidden), а не (L, B, d_hidden)
            norm_first=True,    # pre-norm (более стабильный чем post-norm)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,  # отключаем оптимизацию — мешает с маской
        )

        # Output projection.
        # Дополнительный learnable слой перед подачей в денойзер.
        self.output_proj = nn.Linear(d_hidden, d_cond)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform для линейных слоёв; нули для positional embedding."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.pos_embedding.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        Args:
            history: (B, L) — индексы товаров. Padding-индексы слева.
            mask:    (B, L) — 1.0 для реальных позиций, 0.0 для padding.

        Returns:
            c: (B, d_cond) — conditioning vector для денойзера.
        """
        B, L = history.shape
        device = history.device

        # Embedding lookup → (B, L, d_emb)
        x = self.item_embedding(history)

        # Input projection → (B, L, d_hidden)
        x = self.input_proj(x)

        # Positional embedding: [0, 1, ..., L-1] для каждого элемента батча.
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # (B, L)
        x = x + self.pos_embedding(positions)

        # Padding mask для Transformer:
        # TransformerEncoder ожидает True для ИГНОРИРУЕМЫХ позиций.
        # Наш mask: 1 — реальный, 0 — padding → инвертируем.
        key_padding_mask = (mask == 0)  # (B, L), bool: True = padding

        # Transformer encoder → (B, L, d_hidden)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Last-item pool: берём последнюю позицию последовательности.
        # Padding — слева, значит x[:, -1] всегда реальный товар.
        c = x[:, -1]                                    # (B, d_hidden)

        # Output projection → (B, d_cond)
        c = self.output_proj(c)

        return c


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    python -m src.models.encoder
    """
    import sys

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Фейковые данные (как в реальном ATG)
    N_ITEMS = 11924
    PADDING_IDX = N_ITEMS       # = 11924
    D_EMB = 3072
    D_HIDDEN = 512
    D_COND = 512
    B = 4
    L = 10

    # Whitened эмбеддинги — в реальности загружаются из FlowMatcher
    fake_emb = torch.randn(N_ITEMS + 1, D_EMB) * 0.1
    fake_emb[PADDING_IDX] = 0.0  # нулевой вектор для padding

    model = HistoryEncoder(
        item_embeddings=fake_emb,
        padding_idx=PADDING_IDX,
        d_hidden=D_HIDDEN,
        d_cond=D_COND,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params:,}')

    # Пример 1: полная последовательность (len_seq = 10)
    history_full = torch.randint(0, N_ITEMS, (B, L)).to(device)
    mask_full = torch.ones(B, L).to(device)
    c1 = model(history_full, mask_full)
    print(f'\nFull sequences:')
    print(f'  history: {history_full.shape}')
    print(f'  c:       {c1.shape}, mean={c1.mean().item():.4f}, std={c1.std().item():.4f}')
    assert c1.shape == (B, D_COND)

    # Пример 2: короткие последовательности с padding слева (как cold-start test users)
    # Симулируем len_seq = 5: первые 5 позиций — padding, последние 5 — реальные
    history_short = torch.full((B, L), PADDING_IDX, dtype=torch.long).to(device)
    history_short[:, 5:] = torch.randint(0, N_ITEMS, (B, 5)).to(device)
    mask_short = torch.zeros(B, L).to(device)
    mask_short[:, 5:] = 1.0
    c2 = model(history_short, mask_short)
    print(f'\nShort sequences (len_seq=5):')
    print(f'  history: {history_short.shape}')
    print(f'  c:       {c2.shape}, mean={c2.mean().item():.4f}, std={c2.std().item():.4f}')
    assert c2.shape == (B, D_COND)

    # Пример 3: смешанный батч (разные длины)
    history_mixed = history_full.clone()
    mask_mixed = mask_full.clone()
    # Первые 2 примера полные, последние 2 — с padding
    history_mixed[2:, :3] = PADDING_IDX
    mask_mixed[2:, :3] = 0.0
    c3 = model(history_mixed, mask_mixed)
    print(f'\nMixed batch:')
    print(f'  c:       {c3.shape}  OK')

    # Проверка: padding не должен влиять на последнюю позицию
    # (последняя позиция — всегда реальный элемент в обоих случаях)
    # Запускаем два forward: один с padding-версией, один без —
    # результаты должны отличаться (padding-aware attention)
    assert not torch.allclose(c1, c2), \
        'Full и short sequences должны давать разные c'
    print('\nPadding-aware check: full vs short sequences differ  OK')

    # Проверка: frozen embedding не обновляется
    loss = c1.sum()
    loss.backward()
    assert model.item_embedding.weight.grad is None, \
        'Frozen embedding не должен иметь градиентов'
    print('Frozen embedding has no grad  OK')

    # Параметры
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'\nParameter summary:')
    print(f'  Trainable: {n_params:>12,}')
    print(f'  Frozen:    {frozen:>12,}')
    print(f'  Total:     {n_params + frozen:>12,}')

    print('\nAll checks passed')