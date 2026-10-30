"""
Контрастивный лосс с сэмплированными негативами — общий для SASRec и генеративных моделей.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

LOSS_METRICS = ('cosine', 'dot')


def infonce_loss(
    query: torch.Tensor,
    target: torch.LongTensor,
    embed: Callable[[torch.LongTensor], torch.Tensor],
    n_items: int,
    neg_ratio: int,
    metric: str,
    temperature: float,
) -> torch.Tensor:
    """
    InfoNCE: позитив против neg_ratio случайных негативов (равномерно по каталогу).

    Logit = sim(query, e) / temperature, sim — косинус ('cosine') или скалярное произведение ('dot').
    Метка 0 — позитив стоит на первом месте в конкатенированном тензоре.

    Args:
        query:  (B, d) — представление, которое сравнивается с товарами.
        target: (B,) — индексы позитивных товаров.
        embed:  индексы товаров (любой формы) → эмбеддинги (..., d).
    """
    B      = query.shape[0]
    device = query.device

    # Случайные негативы: (B, neg_ratio), исключаем совпадения с позитивом
    neg = torch.randint(0, n_items, (B, neg_ratio), device=device)
    # Маскируем совпадения одним проходом (коллизии редки, двойной цикл не нужен)
    collision = (neg == target.unsqueeze(1))
    if collision.any():
        replacement = torch.randint(0, n_items, (B, neg_ratio), device=device)
        neg = torch.where(collision, replacement, neg)

    e_pos = embed(target)                                            # (B, d)
    e_neg = embed(neg)                                               # (B, neg_ratio, d)

    # L2-нормировка: скалярное произведение становится косинусом
    if metric == 'cosine':
        query = F.normalize(query, dim=-1)
        e_pos = F.normalize(e_pos, dim=-1)
        e_neg = F.normalize(e_neg, dim=-1)

    pos_logit = (query * e_pos).sum(dim=-1, keepdim=True)            # (B, 1)
    neg_logit = torch.bmm(e_neg, query.unsqueeze(-1)).squeeze(-1)    # (B, neg_ratio)

    logits = torch.cat([pos_logit, neg_logit], dim=-1) / temperature  # (B, 1+neg_ratio)
    labels = torch.zeros(B, dtype=torch.long, device=device)         # позитив на индексе 0
    return F.cross_entropy(logits, labels)
