"""
MLP-денойзер с FiLM conditioning для conditional flow matching.

Архитектура:
    - Основной поток: [z_τ; t_emb] → input_proj → residual блоки → output
    - Conditioning поток: c → cond_proj → (scale, shift) для каждого блока

Почему FiLM вместо concat:
    При concat([z_τ, t_emb, c]) с d_emb=3072, d_t=256, d_cond=512
    вектор z_τ занимает 3072/3840 = 80% входа. Input projection обучается
    в основном на z_τ, игнорируя c. Gradient flow через c слабый — деноизер
    не учится использовать conditioning.

    FiLM (Feature-wise Linear Modulation) подаёт c отдельно: conditioning
    модулирует scale и shift скрытого состояния в каждом блоке. Это даёт
    conditioning прямой доступ к каждому слою независимо от размерности z_τ.

FiLM в каждом residual блоке:
    h_norm = LayerNorm(h)
    h_mod  = h_norm * (1 + scale) + shift
    h_out  = h + MLP(h_mod)

Где scale, shift = chunk(Linear(c), 2) — по d_block каждый.
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

    Args:
        t:   (B,) или (B, 1) — значения времени в [0, 1].
        dim: размерность выходного вектора (чётная).

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
# FiLM Residual block
# --------------------------------------------------------------------------- #

class FiLMResidualBlock(nn.Module):
    """
    Pre-norm residual блок с FiLM conditioning:
        h_norm = LayerNorm(h)
        h_mod  = h_norm * (1 + scale) + shift
        h_out  = h + MLP(h_mod)

    scale и shift приходят извне (из cond_proj в MLPDenoiser) —
    разные для каждого блока, одни и те же для всего батча через broadcast.

    Args:
        d_block: размерность входа/выхода.
        d_ff:    размерность скрытого слоя MLP.
        dropout: вероятность dropout (0.0 по умолчанию).
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
        for layer in self.ff:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        h: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h:     (B, d_block) — скрытое состояние.
            scale: (B, d_block) — FiLM scale из conditioning.
            shift: (B, d_block) — FiLM shift из conditioning.

        Returns:
            (B, d_block)
        """
        h_norm = self.norm(h)
        h_mod  = h_norm * (1.0 + scale) + shift
        return h + self.ff(h_mod)


# --------------------------------------------------------------------------- #
# MLP Denoiser с FiLM
# --------------------------------------------------------------------------- #

class MLPDenoiser(nn.Module):
    """
    MLP-денойзер с FiLM conditioning для flow matching.

    Основной поток (z_τ, t_emb):
        [z_τ; t_emb] → input_proj → N × FiLMResidualBlock → final_norm → output_proj

    Conditioning поток (c):
        c → cond_proj → [(scale_1, shift_1), ..., (scale_N, shift_N)]
        — отдельная пара (scale, shift) для каждого из N блоков.

    Args:
        d_emb:        размерность эмбеддинга товара (3072 для text-embedding-3-large).
        d_cond:       размерность conditioning vector из encoder'а (512).
        d_block:      внутренняя ширина денойзера (1024).
        d_ff:         ширина MLP внутри residual блока (2048).
        n_blocks:     количество residual блоков (6).
        time_emb_dim: размерность sinusoidal time embedding (256).
        dropout:      вероятность dropout (0.0).
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
        self.n_blocks = n_blocks

        # Основной поток: z_τ + t_emb → d_block
        # c не включается — подаётся отдельно через FiLM
        d_input = d_emb + time_emb_dim
        self.input_proj = nn.Linear(d_input, d_block)

        # Conditioning проекция: c → (scale, shift) для КАЖДОГО блока
        # Выход: n_blocks * d_block * 2 (scale и shift для каждого блока)
        self.cond_proj = nn.Linear(d_cond, n_blocks * d_block * 2)

        # Residual блоки
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(d_block, d_ff, dropout)
            for _ in range(n_blocks)
        ])

        self.final_norm  = nn.LayerNorm(d_block)
        self.output_proj = nn.Linear(d_block, d_emb)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
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
        B = x.shape[0]

        # Time embedding
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)     # (B, time_emb_dim)

        # Основной поток: только z_τ и t_emb
        h = self.input_proj(torch.cat([x, t_emb], dim=-1))     # (B, d_block)

        # FiLM параметры из conditioning
        # cond_proj выдаёт (B, n_blocks * d_block * 2)
        film_params = self.cond_proj(c)                         # (B, n_blocks * d_block * 2)
        # Разбиваем на отдельные (scale, shift) для каждого блока
        # film_params[:, i * d_block * 2 : (i+1) * d_block * 2] → scale_i, shift_i
        d_block = h.shape[-1]
        film_params = film_params.view(B, self.n_blocks, d_block * 2)  # (B, n_blocks, d_block*2)

        # Последовательно применяем блоки с FiLM
        for i, block in enumerate(self.blocks):
            scale_shift = film_params[:, i, :]                  # (B, d_block * 2)
            scale, shift = scale_shift.chunk(2, dim=-1)         # (B, d_block) каждый
            h = block(h, scale, shift)

        v = self.output_proj(self.final_norm(h))                # (B, d_emb)
        return v


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    python -m src.models.denoiser
    """
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

    x = torch.randn(B, D_EMB, device=device)
    t = torch.rand(B, device=device)
    c = torch.randn(B, D_COND, device=device)

    # Test 1: forward pass
    print('\nTest 1: forward pass')
    v = model(x, t, c)
    assert v.shape == (B, D_EMB)
    print(f'  v: {v.shape}, mean={v.mean().item():.4f}, std={v.std().item():.4f}  OK')

    # Test 2: backward pass
    print('\nTest 2: backward pass')
    loss = ((v - x) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f'No grad for {name}'
    print(f'  loss={loss.item():.4f}, all params have grads  OK')

    # Test 3: cond_proj нулевой в начале — scale=0, shift=0
    print('\nTest 3: zero init of cond_proj')
    model_fresh = MLPDenoiser(d_emb=D_EMB, d_cond=D_COND).to(device)
    assert model_fresh.cond_proj.weight.abs().max().item() == 0.0
    assert model_fresh.cond_proj.bias.abs().max().item() == 0.0
    print('  cond_proj initialized to zeros  OK')

    # Test 4: FiLM conditioning — zero init, then learns
    print('\nTest 4: FiLM conditioning behaviour')
    model_t = MLPDenoiser(d_emb=D_EMB, d_cond=D_COND, d_block=D_BLOCK).to(device)
    c2 = torch.randn(B, D_COND, device=device)
    model_t.eval()
    with torch.no_grad():
        v1i = model_t(x, t, c)
        v2i = model_t(x, t, c2)
    assert torch.allclose(v1i, v2i), 'zero init: output should be same for different c'
    print('  zero init -> same output for different c  OK')
    model_t.train()
    opt_t = torch.optim.SGD(model_t.parameters(), lr=0.1)
    for _ in range(3):
        loss_t = ((model_t(x, t, c) - x) ** 2).mean()
        loss_t.backward(); opt_t.step(); opt_t.zero_grad()
    model_t.eval()
    with torch.no_grad():
        v1t = model_t(x, t, c)
        v2t = model_t(x, t, c2)
    assert not torch.allclose(v1t, v2t), 'after training c should matter'
    print(f'  after training ||v(c1)-v(c2)||={( v1t-v2t).norm(dim=-1).mean():.3f}  OK')

    # Test 5: t=(B,1) тоже работает
    print('\nTest 5: t shape (B,1)')
    with torch.no_grad():
        v3 = model(x, t.unsqueeze(-1), c)
    assert v3.shape == (B, D_EMB)
    print('  OK')

    # Test 6: детерминированность в eval mode
    print('\nTest 6: deterministic in eval')
    model.eval()
    with torch.no_grad():
        va = model(x, t, c)
        vb = model(x, t, c)
    assert torch.allclose(va, vb)
    print('  OK')
    print(f'\nParameter breakdown:')
    print(f'  input_proj:  {model.input_proj.weight.numel() + model.input_proj.bias.numel():>12,}')
    print(f'  cond_proj:   {model.cond_proj.weight.numel() + model.cond_proj.bias.numel():>12,}')
    block_p = sum(p.numel() for b in model.blocks for p in b.parameters())
    print(f'  blocks({N_BLOCKS}x):  {block_p:>12,}')
    print(f'  final_norm:  {sum(p.numel() for p in model.final_norm.parameters()):>12,}')
    print(f'  output_proj: {model.output_proj.weight.numel() + model.output_proj.bias.numel():>12,}')
    print(f'  total:       {n_params:>12,}')

    print('\nAll checks passed')