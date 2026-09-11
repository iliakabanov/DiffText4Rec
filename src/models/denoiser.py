"""
Sinusoidal embedding времени для диффузионных и flow-matching моделей.

Используется в DreamRec, DreamRecText и DreamRecFMText: выход подаётся в time_mlp.
"""

from __future__ import annotations

import math

import torch


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """
    Sinusoidal embedding скалярного времени (как positional encoding в Transformer).

    Частоты лежат в [1/max_period, 1], поэтому t ожидается в масштабе шагов диффузии
    (сотни–тысячи): целые t ∈ [0, T) подходят как есть, а непрерывное τ ∈ [0, 1] нужно
    сначала домножить (DreamRecFMText подаёт τ · 1000). Если подать τ ∈ [0, 1] напрямую,
    эмбеддинги разных τ почти совпадают (cos ≈ 0.97 для τ=0 и τ=1), и сеть не видит время.

    Args:
        t:          (B,) или (B, 1) — время.
        dim:        размерность выходного вектора.
        max_period: минимальная частота = 1 / max_period.

    Returns:
        (B, dim)
    """
    if t.dim() == 2:
        t = t.squeeze(-1)

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
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    python src/models/denoiser.py
    """
    import torch.nn.functional as F

    t = torch.tensor([0.0, 1.0, 10.0, 500.0, 999.0])
    emb = sinusoidal_embedding(t, 128)
    assert emb.shape == (5, 128)
    assert sinusoidal_embedding(t.unsqueeze(-1), 128).shape == (5, 128)
    assert sinusoidal_embedding(t, 127).shape == (5, 127)
    print('shapes  OK')

    e = F.normalize(emb, dim=-1)
    cos_steps = (e[0] @ e[4]).item()
    tau = F.normalize(sinusoidal_embedding(torch.tensor([0.0, 1.0]), 128), dim=-1)
    cos_tau = (tau[0] @ tau[1]).item()
    print(f'cos(t=0, t=999)             = {cos_steps:.3f}  (шаги диффузии различимы)')
    print(f'cos(τ=0, τ=1) без масштаба  = {cos_tau:.3f}  (почти совпадают)')
    assert cos_steps < 0.5 and cos_tau > 0.9

    print('\nAll checks passed')
