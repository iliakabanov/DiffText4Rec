"""
Строительные блоки генеративных рекомендеров (см. models/dreamrec.py).

    ItemEmbedding   — эмбеддинги товаров: обучаемые с нуля или замороженные текстовые.
    AlphaFuseItemEmbedding — замороженные текстовые + обучаемые ID в их null-координатах.
    HistoryEncoder  — история взаимодействий → conditioning-вектор h (один SASRec-блок).
    ConcatDenoiser  — сеть concat([x, h, t_emb]) → Linear(3d, d) с токеном «без условия» для CFG.

Время во всех моделях — доля пути от шума к данным τ ∈ [0, 1]; на вход sinusoidal
embedding подаётся целое round(τ · TIME_STEPS).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.sasrec import SASRecBlock
from whitening import apply_whitening, load_or_compute_whitening

TIME_STEPS = 1000


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """
    Sinusoidal embedding времени (как positional encoding в Transformer).

    Частоты лежат в [1/max_period, 1], поэтому t ожидается в масштабе целых шагов
    (сотни–тысячи). Если подать τ ∈ [0, 1] напрямую, эмбеддинги разных τ почти
    совпадают (cos ≈ 0.97 для τ=0 и τ=1), и сеть не видит время.

    Args:
        t:          (B,) — время в шагах.
        dim:        размерность выхода.
        max_period: минимальная частота = 1 / max_period.

    Returns:
        (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 != 0:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# --------------------------------------------------------------------------- #
# Эмбеддинги товаров
# --------------------------------------------------------------------------- #

class ItemEmbedding(nn.Module):
    """
    Таблица эмбеддингов товаров (N_items + 1, d_emb); последняя строка — нулевой padding.

    Создаётся через learned() — обучаемые с нуля (DreamRec) — или text() — замороженные
    текстовые эмбеддинги, опционально после PCA + whitening (DreamRecText, DreamRecFMText).
    """

    def __init__(self, weight: torch.Tensor, padding_idx: int, trainable: bool, info: dict):
        super().__init__()
        if padding_idx != weight.shape[0] - 1:
            raise ValueError(f'padding_idx={padding_idx} должен быть последней строкой ({weight.shape[0] - 1})')
        self.n_items = weight.shape[0] - 1
        self.d_emb   = weight.shape[1]
        self.info    = info
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=not trainable, padding_idx=padding_idx)

    @classmethod
    def learned(cls, n_items: int, d_emb: int, padding_idx: int) -> ItemEmbedding:
        """Обучаемые эмбеддинги, инициализация N(0, 1)."""
        weight = torch.randn(n_items + 1, d_emb)
        weight[padding_idx] = 0.0
        return cls(weight, padding_idx, trainable=True, info={'item_embedding': 'learned'})

    @classmethod
    def text(
        cls,
        emb_raw: np.ndarray,
        padding_idx: int,
        use_pca: bool,
        n_components: int,
        per_component_whitening: bool,
        whitening_cache: str | Path,
    ) -> ItemEmbedding:
        """
        Замороженные текстовые эмбеддинги (N_items, D_raw).

        use_pca=True  — PCA до n_components + whitening (per_component_whitening:
                        Var = I по каждой компоненте, иначе один глобальный std);
        use_pca=False — сырые эмбеддинги как есть.
        """
        if use_pca:
            mu, V, scale = load_or_compute_whitening(
                emb_raw, n_components, whitening_cache, per_component_whitening,
            )
            emb = apply_whitening(emb_raw.astype(np.float32), mu, V, scale)
        else:
            emb = emb_raw.astype(np.float32)
        weight = torch.from_numpy(np.vstack([emb, np.zeros((1, emb.shape[1]), dtype=np.float32)]))
        info = {'item_embedding': 'text', 'use_pca': use_pca,
                'per_component_whitening': per_component_whitening if use_pca else None}
        return cls(weight, padding_idx, trainable=False, info=info)

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        return self.embedding(idx)

    @property
    def table(self) -> torch.Tensor:
        """(N_items, d_emb) — эмбеддинги всех товаров без padding (для ретривала)."""
        return self.embedding.weight[:self.n_items]


ALPHAFUSE_FUSIONS = ('add', 'replace')
ALPHAFUSE_ID_INITS = ('normal', 'zeros')


class AlphaFuseItemEmbedding(nn.Module):
    """
    Эмбеддинги товаров AlphaFuse (Hu et al., 2025): ID-эмбеддинги учатся в «нуль-пространстве»
    текстовых эмбеддингов. Интерфейс как у ItemEmbedding: forward(idx), table, n_items, d_emb, info.

    Семантика — замороженные текстовые эмбеддинги после PCA до d_emb и whitening по компонентам
    (Var = I). Компоненты упорядочены по убыванию дисперсии, последние nullity из них — null-часть.
    Обучаемый ID-вектор (nullity) вливается в неё:

        fusion='add'      e = [s_1..s_{d−k},  s_{d−k+1..d} + id]    — Eq. 10 статьи, SASRec; дефолт кода (cover=False)
        fusion='replace'  e = [s_1..s_{d−k},  id]                   — статья для DreamRec (разд. 4.1.5): null-координаты
                                                                      семантики отброшены, Var(e) = I сохраняется

    nullity задаётся ровно одним из:
        null_dim   — числом;
        null_thres — порогом на собственные значения ковариации сырых эмбеддингов (λ ≤ null_thres)
                     среди d_emb оставленных компонент. Порог AlphaFuse 0.25 считается для
                     эмбеддингов ×40, у нас ему соответствует 0.25 / 40² = 1.5625e-4.
    """

    def __init__(
        self,
        emb_raw: np.ndarray,
        padding_idx: int,
        d_emb: int,
        null_dim: int | None,
        null_thres: float | None,
        fusion: str,
        id_init: str,
        whitening_cache: str | Path,
    ):
        super().__init__()
        n_items = emb_raw.shape[0]
        if padding_idx != n_items:
            raise ValueError(f'padding_idx={padding_idx} должен быть последней строкой ({n_items})')
        if (null_dim is None) == (null_thres is None):
            raise ValueError('задайте ровно одно из null_dim и null_thres')
        if fusion not in ALPHAFUSE_FUSIONS:
            raise ValueError(f'fusion должен быть одним из {ALPHAFUSE_FUSIONS}, получено {fusion!r}')
        if id_init not in ALPHAFUSE_ID_INITS:
            raise ValueError(f'id_init должен быть одним из {ALPHAFUSE_ID_INITS}, получено {id_init!r}')

        mu, V, scale = load_or_compute_whitening(emb_raw, d_emb, whitening_cache, per_component=True)
        eigvals = scale ** 2                    # std компоненты² = собственное значение ковариации
        nullity = null_dim if null_dim is not None else int((eigvals <= null_thres).sum())
        if not 0 < nullity < d_emb:
            raise ValueError(f'nullity={nullity} должен быть в (0, d_emb={d_emb})')

        semantic = apply_whitening(emb_raw.astype(np.float32), mu, V, scale)
        if fusion == 'replace':
            semantic = semantic[:, :d_emb - nullity]
        semantic = np.vstack([semantic, np.zeros((1, semantic.shape[1]), dtype=np.float32)])

        self.n_items = n_items
        self.d_emb   = d_emb
        self.nullity = nullity
        self.fusion  = fusion
        self.info    = {'item_embedding': 'alphafuse', 'fusion': fusion, 'nullity': nullity,
                        'null_dim': null_dim, 'null_thres': null_thres, 'id_init': id_init}

        self.semantic = nn.Embedding.from_pretrained(torch.from_numpy(semantic), freeze=True,
                                                     padding_idx=padding_idx)
        self.id_embedding = nn.Embedding(n_items + 1, nullity, padding_idx=padding_idx)
        if id_init == 'normal':
            nn.init.normal_(self.id_embedding.weight, 0.0, 1.0)
        else:
            nn.init.zeros_(self.id_embedding.weight)
        with torch.no_grad():
            self.id_embedding.weight[padding_idx].zero_()

    def _fuse(self, semantic: torch.Tensor, id_emb: torch.Tensor) -> torch.Tensor:
        if self.fusion == 'replace':
            return torch.cat([semantic, id_emb], dim=-1)
        k = self.nullity
        return torch.cat([semantic[..., :-k], semantic[..., -k:] + id_emb], dim=-1)

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        return self._fuse(self.semantic(idx), self.id_embedding(idx))

    @property
    def table(self) -> torch.Tensor:
        """(N_items, d_emb) — слитые эмбеддинги всех товаров без padding (для ретривала)."""
        n = self.n_items
        return self._fuse(self.semantic.weight[:n], self.id_embedding.weight[:n])


# --------------------------------------------------------------------------- #
# Энкодер истории
# --------------------------------------------------------------------------- #

class HistoryEncoder(nn.Module):
    """
    История → conditioning-вектор h ∈ d_emb: эмбеддинги товаров + позиционные,
    один каузальный SASRec-блок (pre-norm), LayerNorm, выход последней позиции
    (padding слева, поэтому последняя позиция — всегда последний реальный товар).
    """

    def __init__(self, d_emb: int, n_heads: int, d_ff: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_emb)
        self.dropout       = nn.Dropout(dropout)
        self.block         = SASRecBlock(d_emb, n_heads, d_ff, dropout)
        self.norm          = nn.LayerNorm(d_emb)
        nn.init.zeros_(self.pos_embedding.weight)

    def forward(self, seq_emb: torch.Tensor, mask: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            seq_emb: (B, L, d_emb) — эмбеддинги товаров истории.
            mask:    (B, L) — 1 для реальных товаров, 0 для padding.

        Returns:
            h: (B, d_emb)
        """
        L = seq_emb.shape[1]
        x = self.dropout(seq_emb + self.pos_embedding(torch.arange(L, device=seq_emb.device)))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq_emb.device), diagonal=1)
        x = self.norm(self.block(x, attn_mask=causal, key_padding_mask=(mask == 0)))
        return torch.nan_to_num(x, nan=0.0)[:, -1]   # пустая история даёт NaN в attention


# --------------------------------------------------------------------------- #
# Денойзер
# --------------------------------------------------------------------------- #

class ConcatDenoiser(nn.Module):
    """
    Сеть f(x, h, τ) = Linear(3d, d)(concat([x, h, t_emb])), t_emb = MLP(sinusoidal(round(τ·TIME_STEPS))).
    Что она предсказывает — чистый эмбеддинг (DDPM) или скорость (Flow Matching) — задаёт процесс.

    null_token — обучаемый conditioning «без условия» для classifier-free guidance: на обучении
    подставляется вместо h с вероятностью p_uncond (drop_condition), на инференсе даёт
    безусловное предсказание для guided().
    """

    def __init__(self, d_emb: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(d_emb, d_emb * 2),
            nn.GELU(),
            nn.Linear(d_emb * 2, d_emb),
        )
        self.linear     = nn.Linear(d_emb * 3, d_emb)
        self.null_token = nn.Parameter(torch.randn(d_emb))      # N(0, 1)
        for layer in (self.time_mlp[0], self.time_mlp[2], self.linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def time_embedding(self, tau: torch.Tensor) -> torch.Tensor:
        """τ (B,) ∈ [0, 1] → (B, d_emb): sinusoidal от целого шага round(τ · TIME_STEPS) + MLP."""
        steps = torch.round(tau * TIME_STEPS)
        return self.time_mlp(sinusoidal_embedding(steps, self.linear.out_features))

    def forward(self, x: torch.Tensor, h: torch.Tensor | None, tau: torch.Tensor) -> torch.Tensor:
        """h=None — безусловное предсказание (null_token вместо истории)."""
        if h is None:
            h = self.null_token.expand(x.shape[0], -1)
        return self.linear(torch.cat([x, h, self.time_embedding(tau)], dim=-1))

    def guided(self, x: torch.Tensor, h: torch.Tensor, tau: torch.Tensor, w: float) -> torch.Tensor:
        """Classifier-free guidance: (1 + w)·f(x, h) − w·f(x, ∅); при w = 0 — просто f(x, h)."""
        out = self(x, h, tau)
        if w != 0.0:
            out = (1.0 + w) * out - w * self(x, None, tau)
        return out

    def drop_condition(self, h: torch.Tensor, p: float) -> torch.Tensor:
        """На обучении с вероятностью p заменяет h на null_token (CFG dropout)."""
        if not self.training or p == 0.0:
            return h
        drop = torch.rand(h.shape[0], device=h.device) < p
        return torch.where(drop.unsqueeze(-1), self.null_token.expand_as(h), h)
