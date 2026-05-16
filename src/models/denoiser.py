"""
MLP-денойзер для conditional flow matching.

Архитектура:
    1. Sinusoidal time embedding для τ.
    2. Concat входа: [x_τ; t_emb; c] → проекция → d_block.
    3. N residual MLP блоков.
    4. Output projection → D_emb.

Используется в FlowMatcher для предсказания velocity v = z_1 - z_0.
"""


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """
    Sinusoidal embedding для непрерывного времени.
    
    Args:
        t:   (B,) — значения в [0, 1].
        dim: размер выходного эмбеддинга (чётный).
    
    Returns:
        (B, dim) — необучаемое представление времени.
    """


class ResidualMLPBlock(nn.Module):
    """
    Pre-norm residual блок:
        h_out = h + Linear(SiLU(Linear(LayerNorm(h))))
    """
    
    def __init__(self, d_block: int, d_ff: int, dropout: float = 0.0):
        ...
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        ...


class MLPDenoiser(nn.Module):
    """
    Денойзер для conditional flow matching.
    
    Forward:
        1. t_emb = timestep_embedding(t, time_emb_dim)        # (B, time_emb_dim)
        2. inp = concat([x, t_emb, c], dim=-1)                # (B, d_emb + time_emb_dim + d_cond)
        3. h = InputProjection(inp)                           # (B, d_block)
        4. for block in blocks: h = block(h)
        5. h = LayerNorm(h)
        6. v = OutputProjection(h)                            # (B, d_emb)
    """
    
    def __init__(
        self,
        d_emb: int = 1536,         # размерность эмбеддинга товара
        d_cond: int = 512,         # размерность conditioning vector
        d_block: int = 1024,       # внутренняя ширина денойзера
        d_ff: int = 2048,          # ширина MLP в residual блоке
        n_blocks: int = 6,
        time_emb_dim: int = 256,   # размер sinusoidal time embedding
        dropout: float = 0.0,
    ):
        """
        Атрибуты:
            self.time_emb_dim: int                                  # сохраняем для forward
            self.input_proj:   Linear(d_emb + time_emb_dim + d_cond, d_block)
            self.blocks:       ModuleList из n_blocks ResidualMLPBlock
            self.final_norm:   LayerNorm(d_block)
            self.output_proj:  Linear(d_block, d_emb)
        """
    
    def forward(
        self,
        x: torch.Tensor,           # (B, d_emb)
        t: torch.Tensor,           # (B,)
        c: torch.Tensor,           # (B, d_cond)
    ) -> torch.Tensor:
        """
        Returns:
            v: (B, d_emb) — predicted velocity.
        """