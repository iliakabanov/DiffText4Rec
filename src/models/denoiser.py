"""
MLP-денойзер для conditional flow matching.

Принимает зашумлённый эмбеддинг z_τ, время τ и conditioning vector c,
возвращает предсказанную velocity v = z_1 - z_0.

Pipeline:
    z_τ  (B, d_emb)   — зашумлённый эмбеддинг целевого товара
    τ    (B,)         — время в [0, 1]
    c    (B, d_cond)  — conditioning из HistoryEncoder

    1. t_emb = sinusoidal_embedding(τ, time_emb_dim)   → (B, time_emb_dim)
    2. inp   = concat([z_τ, t_emb, c], dim=-1)         → (B, d_emb + time_emb_dim + d_cond)
    3. h     = input_proj(inp)                         → (B, d_block)
    4. for block in blocks: h = block(h)               → (B, d_block)
    5. h     = final_norm(h)                           → (B, d_block)
    6. v     = output_proj(h)                          → (B, d_emb)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Time embedding
# --------------------------------------------------------------------------- #

def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """
    Sinusoidal embedding для непрерывного времени τ ∈ [0, 1].

    Аналог positional embedding из "Attention Is All You Need",
    адаптированный для диффузионных моделей (DDPM, DiT).

    Args:
        t:          (B,) или (B, 1) — значения времени в [0, 1].
        dim:        размерность выходного вектора (должна быть чётной).
        max_period: контролирует диапазон частот. Большие значения → более
                    низкие частоты. Стандартное значение 10_000 из DDPM.

    Returns:
        (B, dim) — embedding времени.

    Формула:
        freqs_k = exp(-log(max_period) * k / (dim/2)),  k = 0..dim/2-1
        emb = [cos(t * freqs), sin(t * freqs)]
    """
    if t.dim() == 2:
        t = t.squeeze(-1)                               # (B, 1) → (B,)

    half = dim // 2
    # Логарифмически равномерно распределённые частоты
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )                                                   # (half,)

    args = t[:, None].float() * freqs[None, :]         # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)

    # Если dim нечётный — паддим одним нулём
    if dim % 2 != 0:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

    return emb


# --------------------------------------------------------------------------- #
# Residual MLP block
# --------------------------------------------------------------------------- #

class ResidualMLPBlock(nn.Module):
    """
    Pre-norm residual блок:
        h_out = h + MLP(LayerNorm(h))

    MLP внутри:
        Linear(d_block, d_ff) → SiLU → Linear(d_ff, d_block)

    Pre-norm (нормализация перед MLP, а не после residual) — стандарт
    современных трансформеров. Более стабилен, чем post-norm, особенно
    в начале обучения.

    SiLU (Swish) вместо ReLU — стандарт диффузионных моделей. Гладкая
    нелинейность, обычно даёт чуть лучшие результаты.

    Args:
        d_block:  размерность входа/выхода.
        d_ff:     размерность скрытого слоя MLP.
        dropout:  вероятность dropout (0.0 по умолчанию — для baseline не нужен).
    """

    def __init__(self, d_block: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_block)
        self.ff = nn.Sequential(
            nn.Linear(d_block, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_ff, d_block),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # Xavier uniform для linear слоёв — стандартная инициализация.
        for layer in self.ff:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, d_block)
        Returns:
            (B, d_block)
        """
        return h + self.ff(self.norm(h))


# --------------------------------------------------------------------------- #
# MLP Denoiser
# --------------------------------------------------------------------------- #

class MLPDenoiser(nn.Module):
    """
    MLP-денойзер для conditional flow matching.

    Предсказывает velocity v = z_1 - z_0 из зашумлённого состояния z_τ,
    времени τ и conditioning vector c из HistoryEncoder.

    Conditioning осуществляется через concat: [z_τ; t_emb; c] подаётся
    на вход input_proj. Это проще и достаточно для baseline.

    Args:
        d_emb:        размерность эмбеддинга товара. По умолчанию 3072
                      (text-embedding-3-large, ATG).
        d_cond:       размерность conditioning vector из encoder. По умолчанию 512.
        d_block:      внутренняя ширина денойзера. По умолчанию 1024.
        d_ff:         ширина MLP внутри каждого residual блока. По умолчанию 2048.
        n_blocks:     количество residual блоков. По умолчанию 6.
        time_emb_dim: размерность sinusoidal time embedding. По умолчанию 256.
        dropout:      вероятность dropout в residual блоках. По умолчанию 0.0.
    """

    def __init__(
        self,
        d_emb: int = 3072,
        d_cond: int = 512,
        d_block: int = 1024,
        d_ff: int = 2048,
        n_blocks: int = 6,
        time_emb_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # Размерность concat-входа: z_τ + t_emb + c
        d_input = d_emb + time_emb_dim + d_cond

        # Input projection: concat → d_block
        self.input_proj = nn.Linear(d_input, d_block)

        # Residual MLP блоки
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(d_block, d_ff, dropout)
            for _ in range(n_blocks)
        ])

        # Final norm перед output projection
        self.final_norm = nn.LayerNorm(d_block)

        # Output projection: d_block → d_emb (velocity того же размера, что и z_τ)
        self.output_proj = nn.Linear(d_block, d_emb)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, d_emb)  — зашумлённый эмбеддинг z_τ.
            t: (B,)        — время τ в [0, 1].
            c: (B, d_cond) — conditioning из HistoryEncoder.

        Returns:
            v: (B, d_emb) — предсказанная velocity.
        """
        # Time embedding — необучаемое sinusoidal представление τ
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)  # (B, time_emb_dim)

        # Concat всех входов
        inp = torch.cat([x, t_emb, c], dim=-1)              # (B, d_emb + time_emb_dim + d_cond)

        # Input projection → скрытое пространство
        h = self.input_proj(inp)                             # (B, d_block)

        # Residual блоки
        for block in self.blocks:
            h = block(h)                                     # (B, d_block)

        # Final norm + output projection → velocity
        v = self.output_proj(self.final_norm(h))             # (B, d_emb)

        return v


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    python -m src.models.denoiser
    """
    import sys

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    D_EMB = 3072
    D_COND = 512
    D_BLOCK = 1024
    D_FF = 2048
    N_BLOCKS = 6
    TIME_EMB_DIM = 256
    B = 4

    model = MLPDenoiser(
        d_emb=D_EMB,
        d_cond=D_COND,
        d_block=D_BLOCK,
        d_ff=D_FF,
        n_blocks=N_BLOCKS,
        time_emb_dim=TIME_EMB_DIM,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params:,}')

    # Фейковые входы
    x = torch.randn(B, D_EMB, device=device)
    t = torch.rand(B, device=device)
    c = torch.randn(B, D_COND, device=device)

    # Test 1: базовый forward pass
    print('\nTest 1: forward pass')
    v = model(x, t, c)
    assert v.shape == (B, D_EMB), f'Expected ({B}, {D_EMB}), got {v.shape}'
    print(f'  v: {v.shape}, mean={v.mean().item():.4f}, std={v.std().item():.4f}')

    # Test 2: backward pass
    print('\nTest 2: backward pass')
    loss = ((v - x) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f'No grad for {name}'
    print(f'  loss={loss.item():.4f}, all params have grads  OK')

    # Test 3: t в разных форматах
    print('\nTest 3: t shape variants')
    t_2d = t.unsqueeze(-1)                   # (B, 1)
    v2 = model(x, t_2d, c)
    assert v2.shape == (B, D_EMB)
    print(f'  t as (B, 1): OK')

    # Test 4: τ = 0 и τ = 1 дают разные результаты
    print('\nTest 4: different τ give different outputs')
    t_zero = torch.zeros(B, device=device)
    t_one = torch.ones(B, device=device)
    with torch.no_grad():
        v_zero = model(x, t_zero, c)
        v_one = model(x, t_one, c)
    assert not torch.allclose(v_zero, v_one), 'τ=0 и τ=1 должны давать разные результаты'
    print(f'  τ=0 vs τ=1: differ  OK')

    # Test 5: разные c при одинаковых x, t дают разные результаты
    print('\nTest 5: conditioning matters')
    c2 = torch.randn(B, D_COND, device=device)
    with torch.no_grad():
        v_c1 = model(x, t, c)
        v_c2 = model(x, t, c2)
    assert not torch.allclose(v_c1, v_c2), 'Разные c должны давать разные v'
    print(f'  different c: differ  OK')

    # Test 6: deterministic (нет dropout, нет случайности)
    print('\nTest 6: deterministic output')
    model.eval()
    with torch.no_grad():
        v_a = model(x, t, c)
        v_b = model(x, t, c)
    assert torch.allclose(v_a, v_b), 'При eval mode результат должен быть детерминированным'
    print(f'  eval mode deterministic  OK')

    # Test 7: sinusoidal_embedding свойства
    print('\nTest 7: sinusoidal_embedding')
    t_test = torch.linspace(0, 1, 100)
    emb = sinusoidal_embedding(t_test, dim=256)
    assert emb.shape == (100, 256)
    # Значения ограничены [-1, 1] (cos/sin)
    assert emb.abs().max() <= 1.0 + 1e-6, 'sinusoidal values must be in [-1, 1]'
    # Два разных τ дают разные эмбеддинги
    assert not torch.allclose(emb[0], emb[50])
    print(f'  shape OK, values in [-1, 1], distinct embeddings  OK')

    # Параметры по слоям
    print(f'\nParameter breakdown:')
    print(f'  input_proj:  {model.input_proj.weight.numel() + model.input_proj.bias.numel():>12,}')
    block_params = sum(p.numel() for b in model.blocks for p in b.parameters())
    print(f'  blocks ({N_BLOCKS}x): {block_params:>12,}')
    print(f'  final_norm:  {sum(p.numel() for p in model.final_norm.parameters()):>12,}')
    print(f'  output_proj: {model.output_proj.weight.numel() + model.output_proj.bias.numel():>12,}')
    print(f'  total:       {n_params:>12,}')

    print('\nAll checks passed')