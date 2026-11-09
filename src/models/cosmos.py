"""
DreamRecCosmos — DreamRec с денойзером по мотивам COSMOS (Meshchaninov et al., 2025).

От DreamRec отличается только денойзер: история попадает в него не одним вектором h в concat,
а токенами энкодера, к которым токен цели обращается через cross-attention:

    x = x + time_i(emb_t)
    x = x + cross_attn(q=RMSNorm(x), k=ctx, v=ctx)
    x = x + SwiGLU(RMSNorm(x))

Self-attention нет (токен цели один, внимание к самому себе вырождается в линейный слой),
глобального h нет, CFG нет. Выход последнего блока — x̂_0.

Эмбеддинги товаров задаёт builders (model.items.source в yaml): обучаемые, текстовые или таблица
обученного SASRec. Энкодер истории, процесс DDPM и лоссы — те же, что у DreamRec.

Совместима с evaluate() через recommend(history, mask, top_k, num_steps, w_cfg).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import TIME_STEPS, HistoryEncoder, ItemEmbedding, sinusoidal_embedding
from models.dreamrec import GenerativeRecommender
from models.processes import DDPM


# --------------------------------------------------------------------------- #
# Условие денойзера
# --------------------------------------------------------------------------- #

class HistoryContext(NamedTuple):
    """
    История как условие. Q — запросов на пользователя: L при обучении на префиксах, иначе 1.

    Строки цели идут плоско (N, d), N = rows.sum(); в сетку (B, Q, d) они раскладываются только на
    время внимания. Процессу (DDPM) всё равно: он работает с плоскими строками и условие не читает.
    """
    seq:       torch.Tensor   # (B, L, d)     выход HistoryEncoder.encode()
    rows:      torch.Tensor   # (B, Q) bool   какие запросы реальные
    attn_mask: torch.Tensor   # (B, 1, Q, L)  float, -inf — запрещено
    distance:  torch.Tensor   # (Q, L)        p − j: сколько шагов от ключа j до запроса на позиции p


def _float_mask(blocked: torch.Tensor) -> torch.Tensor:
    """(B, Q, L) bool → (B, 1, Q, L) float: прибавляется к логитам внимания."""
    return torch.zeros_like(blocked, dtype=torch.float32).masked_fill(blocked, float('-inf'))[:, None]


def prefix_context(seq: torch.Tensor, mask: torch.FloatTensor) -> HistoryContext:
    """
    Обучение на префиксах: каждая реальная позиция p — запрос, видит не-padding ключи j ≤ p.
    Диагональ разрешена всегда: у запроса на padding-позиции иначе запрещены все ключи и softmax
    даёт NaN (эти запросы не читают, см. causal_attn_mask). Энкодер каузальный, поэтому ключи j ≤ p
    не содержат будущих товаров.
    """
    L, device = seq.shape[1], seq.device
    positions = torch.arange(L, device=device)
    causal    = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    blocked   = causal[None] | (mask == 0)[:, None, :]
    blocked   = blocked & ~torch.eye(L, dtype=torch.bool, device=device)
    distance  = (positions[:, None] - positions[None, :]).clamp(min=0)
    return HistoryContext(seq, mask > 0, _float_mask(blocked), distance)


def last_context(seq: torch.Tensor, mask: torch.FloatTensor) -> HistoryContext:
    """Один запрос на пользователя с последней позиции (padding слева: последний товар всегда там)."""
    B, L, device = seq.shape[0], seq.shape[1], seq.device
    rows = torch.ones(B, 1, dtype=torch.bool, device=device)
    distance = (L - 1 - torch.arange(L, device=device))[None]
    return HistoryContext(seq, rows, _float_mask((mask == 0)[:, None, :]), distance)


# --------------------------------------------------------------------------- #
# Денойзер
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    """x / √mean(x²) · γ: убирает масштаб вектора, γ — обучаемый вес на координату. Среднее не вычитает."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.scale = d ** -0.5
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) * self.scale).clamp(min=self.eps) * self.gamma


class HistoryAttention(nn.Module):
    """
    Cross-attention токенов цели к токенам истории.

    rel_bias — обучаемый сдвиг логитов по расстоянию до ключа, своя кривая на голову. Позиционные
    эмбеддинги энкодера абсолютные: по ним запрос на позиции p не знает, какой ключ для него самый
    свежий, а на инференсе p всегда последняя.
    """

    def __init__(self, d: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f'd={d} должно делиться на denoiser_heads={n_heads}')
        self.n_heads = n_heads
        self.dropout = dropout
        self.query = nn.Linear(d, d, bias=False)
        self.key   = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.out   = nn.Linear(d, d, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(n_heads, max_seq_len))

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, T, d) → (B, n_heads, T, d / n_heads)"""
        return t.view(t.shape[0], t.shape[1], self.n_heads, -1).transpose(1, 2)

    def forward(self, x: torch.Tensor, ctx: HistoryContext) -> torch.Tensor:
        """x (N, d) → (N, d)."""
        B, Q = ctx.rows.shape
        grid = x.new_zeros(B, Q, x.shape[-1])
        grid[ctx.rows] = x
        q = self._split_heads(self.query(grid))
        k = self._split_heads(self.key(ctx.seq))
        v = self._split_heads(self.value(ctx.seq))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx.attn_mask + self.rel_bias[:, ctx.distance],
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, Q, x.shape[-1])[ctx.rows])


class SwiGLU(nn.Module):
    """Gated MLP: down(silu(gate(x)) · up(x))."""

    def __init__(self, d: int, mult: int):
        super().__init__()
        self.gate = nn.Linear(d, d * mult, bias=False)
        self.up   = nn.Linear(d, d * mult, bias=False)
        self.down = nn.Linear(d * mult, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DenoiserBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, max_seq_len: int, mlp_mult: int, dropout: float):
        super().__init__()
        self.time  = nn.Sequential(nn.SiLU(), nn.Linear(d, d))
        self.norm1 = RMSNorm(d)
        self.attn  = HistoryAttention(d, n_heads, max_seq_len, dropout)
        self.norm2 = RMSNorm(d)
        self.mlp   = SwiGLU(d, mlp_mult)

    def forward(self, x: torch.Tensor, ctx: HistoryContext, emb_t: torch.Tensor) -> torch.Tensor:
        x = x + self.time(emb_t)
        x = x + self.attn(self.norm1(x), ctx)
        x = x + self.mlp(self.norm2(x))
        return x


class CosmosDenoiser(nn.Module):
    """
    Сеть f(x, ctx, τ): (N, d) → (N, d), предсказывает x_0. Интерфейс для процессов — как у ConcatDenoiser.
    Блоков нужно не меньше двух: в первом запрос состоит только из x_t и времени.
    """

    def __init__(self, d: int, n_heads: int, n_blocks: int, max_seq_len: int, mlp_mult: int, dropout: float):
        super().__init__()
        if n_blocks < 2:
            raise ValueError(f'n_blocks должно быть ≥ 2, иначе запрос ни разу не учтёт историю; получено {n_blocks}')
        self.d = d
        self.n_heads  = n_heads
        self.mlp_mult = mlp_mult
        self.dropout  = dropout
        self.time_mlp = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.blocks = nn.ModuleList([DenoiserBlock(d, n_heads, max_seq_len, mlp_mult, dropout)
                                     for _ in range(n_blocks)])

    def time_embedding(self, tau: torch.Tensor) -> torch.Tensor:
        """τ (N,) ∈ [0, 1] → (N, d): sinusoidal от целого шага round(τ · TIME_STEPS) + MLP."""
        return self.time_mlp(sinusoidal_embedding(torch.round(tau * TIME_STEPS), self.d))

    def forward(self, x: torch.Tensor, ctx: HistoryContext, tau: torch.Tensor) -> torch.Tensor:
        emb_t = self.time_embedding(tau)
        for block in self.blocks:
            x = block(x, ctx, emb_t)
        return x

    def guided(self, x: torch.Tensor, ctx: HistoryContext, tau: torch.Tensor, w: float) -> torch.Tensor:
        if w != 0.0:
            raise ValueError(f'В CosmosDenoiser нет CFG: w_cfg должен быть 0, получено {w}')
        return self(x, ctx, tau)

    def noise_like(self, ctx: HistoryContext) -> torch.Tensor:
        """Стартовый шум сэмплинга: (N, d), по строке на реальный запрос."""
        return torch.randn(int(ctx.rows.sum()), self.d, device=ctx.seq.device)


# --------------------------------------------------------------------------- #
# Модель
# --------------------------------------------------------------------------- #

class DreamRecCosmos(GenerativeRecommender):
    """DreamRec с CosmosDenoiser. Эмбеддинги товаров приходят готовым модулем, процесс — DDPM."""

    def __init__(self, *, items: ItemEmbedding, n_heads: int, n_layers: int, d_ff: int, dropout: float,
                 max_seq_len: int, n_blocks: int, denoiser_heads: int, mlp_mult: int, denoiser_dropout: float,
                 T: int, beta_schedule: str, retrieval_metric: str, train_prefixes: bool = False, **loss_kwargs):
        super().__init__(
            items=items,
            encoder=HistoryEncoder(items.d_emb, n_heads, d_ff, dropout, max_seq_len, n_layers),
            denoiser=CosmosDenoiser(items.d_emb, denoiser_heads, n_blocks, max_seq_len, mlp_mult, denoiser_dropout),
            process=DDPM(T, beta_schedule),
            p_uncond=0.0, w_cfg=0.0, retrieval_metric=retrieval_metric,
            train_prefixes=train_prefixes, **loss_kwargs,
        )

    def encode_history(self, history: torch.LongTensor, mask: torch.FloatTensor) -> HistoryContext:
        """(B, L) индексы товаров, (B, L) маска → условие с одним запросом на пользователя."""
        return last_context(self.encoder.encode(self.items(history), mask), mask)

    def compute_loss(self, history: torch.LongTensor, target: torch.LongTensor,
                     mask: torch.FloatTensor) -> torch.Tensor:
        """Как в GenerativeRecommender, только условие — токены истории, а не вектор h."""
        if not self.train_prefixes:
            return self._loss(self.encode_history(history, mask), target)
        seq = self.encoder.encode(self.items(history), mask)
        return self._loss(prefix_context(seq, mask), self._prefix_labels(history, target, mask))

    def config(self) -> dict:
        return {
            **super().config(),
            'n_blocks': len(self.denoiser.blocks),
            'denoiser_heads': self.denoiser.n_heads,
            'mlp_mult': self.denoiser.mlp_mult,
            'denoiser_dropout': self.denoiser.dropout,
        }


# --------------------------------------------------------------------------- #
# Quick sanity check:  cd src && python -m models.cosmos
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    from models.dreamrec import RETRIEVAL_METRICS

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_ITEMS, B, L, D, H = 100, 4, 8, 32, 4

    mask = torch.zeros(B, L, device=device)
    for b, n in enumerate((8, 5, 1, 3)):
        mask[b, L - n:] = 1.0
    N = int(mask.sum())

    # Денойзер отдельно: условие строится по произвольным токенам истории
    denoiser = CosmosDenoiser(D, H, 2, L, mlp_mult=1, dropout=0.0).to(device).eval()
    seq = torch.randn(B, L, D, device=device)
    x, tau = torch.randn(N, D, device=device), torch.rand(N, device=device)
    ctx = prefix_context(seq, mask)
    with torch.no_grad():
        out = denoiser(x, ctx, tau)
        assert out.shape == (N, D) and torch.isfinite(out).all()

        # Каузальность: ключи правее запроса на него не влияют, а ключи не правее — влияют
        b, p = 0, 3
        row = p                               # у первого пользователя нет паддинга: строка = позиция
        changed = seq.clone()
        changed[b, p + 1:] += 10.0
        out2 = denoiser(x, prefix_context(changed, mask), tau)
        assert torch.allclose(out[row], out2[row], atol=1e-4)
        assert not torch.allclose(out[row + 1], out2[row + 1], atol=1e-2)

        # Значения на паддинге не влияют на реальные строки
        changed = seq.clone()
        changed[mask == 0] = 100.0
        assert torch.allclose(out, denoiser(x, prefix_context(changed, mask), tau), atol=1e-4)

        # Инференс — то же, что последняя строка каждого пользователя при обучении
        ends = mask.sum(1).cumsum(0).long() - 1
        out_last = denoiser(x[ends], last_context(seq, mask), tau[ends])
        assert torch.allclose(out[ends], out_last, atol=1e-4)

        # Выход зависит от x_t
        other = denoiser(torch.randn_like(x[ends]), last_context(seq, mask), tau[ends])
        assert not torch.allclose(out_last, other, atol=1e-2)
    print('денойзер: формы, каузальность, паддинг, инференс = обучение  OK')

    # Модель целиком
    history = torch.full((B, L), N_ITEMS, dtype=torch.long, device=device)
    history[mask > 0] = torch.randint(0, N_ITEMS, (N,), device=device)
    target = torch.randint(0, N_ITEMS, (B,), device=device)

    def build(**kwargs) -> DreamRecCosmos:
        return DreamRecCosmos(items=ItemEmbedding.learned(N_ITEMS, D, N_ITEMS), n_heads=2, n_layers=1, d_ff=32,
                              dropout=0.1, max_seq_len=L, n_blocks=2, denoiser_heads=H, mlp_mult=1,
                              denoiser_dropout=0.0, T=50, beta_schedule='linear', retrieval_metric='cosine',
                              **kwargs).to(device)

    for train_prefixes in (True, False):
        model = build(train_prefixes=train_prefixes).train()
        loss = model.compute_loss(history, target, mask)
        loss.backward()
        assert torch.isfinite(loss)
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert model.items.embedding.weight.grad[N_ITEMS].abs().max() == 0.0
        assert model.denoiser.blocks[0].attn.rel_bias.grad.abs().max() > 0.0

        model.eval()
        for metric in RETRIEVAL_METRICS:
            model.retrieval_metric = metric
            recs = model.recommend(history, mask, top_k=5, num_steps=7, w_cfg=0.0)
            assert recs.shape == (B, 5) and (recs < N_ITEMS).all()
        print(f'модель, train_prefixes={train_prefixes}: loss {loss.item():.4f}, градиенты конечны, recommend  OK')

    # infonce: тот же путь через predict_clean
    model = build(train_prefixes=True, loss='infonce').train()
    loss = model.compute_loss(history, target, mask)
    loss.backward()
    assert torch.isfinite(loss) and model.denoiser.blocks[1].mlp.down.weight.grad is not None
    print(f'infonce: loss {loss.item():.4f}  OK')

    # Autocast, как в train.py
    if device == 'cuda':
        model = build(train_prefixes=True).train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model.compute_loss(history, target, mask)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print('autocast bfloat16  OK')

    # Защита от неверных настроек
    for bad in (dict(n_blocks=1), dict(denoiser_heads=5)):
        try:
            DreamRecCosmos(items=ItemEmbedding.learned(N_ITEMS, D, N_ITEMS), n_heads=2, n_layers=1, d_ff=32, dropout=0.1,
                           max_seq_len=L, mlp_mult=1, denoiser_dropout=0.0, T=50, beta_schedule='linear',
                           retrieval_metric='cosine', **{'n_blocks': 2, 'denoiser_heads': H, **bad})
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    print('проверка настроек  OK')

    print(f'config: {build(train_prefixes=True).config()}')
    print('\nAll checks passed')
