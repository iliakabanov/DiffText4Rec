"""
DreamRec: Unleashing Large Language Models for Sequential Recommendation
(Wang et al., 2023) — реализация для ATG датасета.

Ключевые особенности vs FlowMatcher/DDPM:
    - Item embeddings обучаются с нуля (не PCA/TEM)
    - x_start parameterization: денойзер предсказывает чистый эмбеддинг z_0
    - Простой MLP-денойзер: concat([x_t, h, t_emb]) → Linear → GELU → Linear
    - Стохастический DDPM reverse process (не DDIM)
    - CFG: none_embedding вместо phi-вектора

Совместим с evaluate() через recommend(history, mask, top_k, num_steps, w_cfg).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sasrec import SASRecBlock
from models.denoiser import sinusoidal_embedding


# --------------------------------------------------------------------------- #
# Noise schedules
# --------------------------------------------------------------------------- #

def _exp_schedule(T: int, beta_min: float = 0.1, beta_max: float = 10.0) -> torch.Tensor:
    """Экспоненциальный schedule (дефолт DreamRec)."""
    t = torch.arange(1, T + 1, dtype=torch.float64)
    betas = 1.0 - torch.exp(-beta_min / T - t / T ** 2 * (beta_max - beta_min))
    return betas.clamp(0.0, 0.999).float()


def _linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def _cosine_schedule(T: int) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_cumprod = (f / f[0]).clamp(min=1e-8)
    betas = (1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(0.0, 0.999)
    return betas.float()


def _make_schedule(name: str, T: int) -> torch.Tensor:
    if name == 'exp':
        return _exp_schedule(T)
    if name == 'linear':
        return _linear_schedule(T)
    if name == 'cosine':
        return _cosine_schedule(T)
    raise ValueError(f'Unknown beta_schedule: {name!r}')


# --------------------------------------------------------------------------- #
# DreamRec
# --------------------------------------------------------------------------- #

class DreamRec(nn.Module):
    """
    DreamRec для sequential recommendation на ATG датасете.

    Args:
        n_items:       число товаров (без padding).
        padding_idx:   padding token index (= n_items).
        d_model:       размерность всех эмбеддингов и скрытых состояний.
        n_heads:       число голов self-attention в истории-энкодере.
        d_ff:          ширина FFN в истории-энкодере.
        dropout:       dropout в истории-энкодере и positional emb.
        max_seq_len:   максимальная длина истории.
        T:             число шагов диффузии.
        beta_schedule: 'exp' (дефолт), 'linear', 'cosine'.
        p_uncond:      вероятность заменить h на none_embedding при обучении.
        w_cfg:         сила Classifier-Free Guidance при инференсе.
    """

    def __init__(
        self,
        n_items: int = 11924,
        padding_idx: int = 11924,
        d_model: int = 64,
        n_heads: int = 1,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 10,
        T: int = 200,
        beta_schedule: str = 'exp',
        p_uncond: float = 0.1,
        w_cfg: float = 2.0,
        retrieval_metric: str = 'l2',   # 'l2', 'dot', 'cosine'
    ):
        super().__init__()

        self.n_items          = n_items
        self.padding_idx      = padding_idx
        self.d_model          = d_model
        self.T                = T
        self.p_uncond         = p_uncond
        self.w_cfg            = w_cfg
        self.retrieval_metric = retrieval_metric

        # ------------------------------------------------------------------ #
        # Embeddings
        # ------------------------------------------------------------------ #
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=padding_idx)
        self.pos_embedding  = nn.Embedding(max_seq_len, d_model)
        # Unconditional token — используется вместо h при CFG.
        # padding_idx=0 делает вектор ВСЕГДА нулевым (frozen, не обучается),
        # что соответствует оригинальному DreamRec. Денойзер явно обучается
        # работать с нулевым conditioning при CFG dropout.
        self.none_embedding = nn.Embedding(1, d_model, padding_idx=0)
        self.dropout_emb    = nn.Dropout(dropout)

        # ------------------------------------------------------------------ #
        # History encoder (1 SASRec блок, pre-norm, каузальный)
        # ------------------------------------------------------------------ #
        self.encoder_block = SASRecBlock(d_model, n_heads, d_ff, dropout)
        self.encoder_norm  = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------ #
        # Timestep MLP: sinusoidal(d) → Linear(d,2d) → GELU → Linear(2d,d)
        # ------------------------------------------------------------------ #
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # ------------------------------------------------------------------ #
        # Denoiser MLP (mlp2): concat([x, h, t]) → Linear(3d,2d) → GELU → Linear(2d,d)
        # ------------------------------------------------------------------ #
        self.denoiser = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
        )

        # ------------------------------------------------------------------ #
        # Noise schedule buffers
        # ------------------------------------------------------------------ #
        betas = _make_schedule(beta_schedule, T)
        alphas            = 1.0 - betas
        alphas_cumprod    = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer('betas',                         betas)
        self.register_buffer('alphas_cumprod',                alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod',           alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt())

        # Posterior mean coefficients для reverse process (x_start parameterization)
        # mean = coef1 * x_start + coef2 * x_t
        coef1 = alphas_cumprod_prev.sqrt() * betas / (1.0 - alphas_cumprod)
        coef2 = alphas.sqrt() * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', coef1)
        self.register_buffer('posterior_mean_coef2', coef2)

        # Posterior variance: clamp чтобы избежать log(0) на первом шаге
        post_var = (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * betas
        self.register_buffer('posterior_variance', post_var.clamp(min=1e-20))

        self._init_weights()

    # ---------------------------------------------------------------------- #
    # Init
    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_embedding.weight, 0.0, 1.0)
        with torch.no_grad():
            self.item_embedding.weight[self.padding_idx].zero_()
        nn.init.zeros_(self.pos_embedding.weight)
        # none_embedding: не инициализируем — padding_idx=0 держит его нулевым
        for m in [self.time_mlp, self.denoiser]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ---------------------------------------------------------------------- #
    # History encoding
    # ---------------------------------------------------------------------- #

    def encode_history(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
        apply_cfg_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Кодирует историю в conditioning vector h.

        Args:
            history:            (B, L)
            mask:               (B, L), 1=реальный, 0=padding
            apply_cfg_dropout:  при True с вероятностью p_uncond заменяет h
                                на none_embedding (только при обучении)

        Returns:
            h: (B, d_model)
        """
        B, L   = history.shape
        device = history.device

        x = self.item_embedding(history)                                 # (B, L, d)
        x = x + self.pos_embedding(torch.arange(L, device=device))      # broadcast по B
        x = self.dropout_emb(x)

        causal_mask      = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        key_padding_mask = (mask == 0)

        x = self.encoder_block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.encoder_norm(x)
        x = torch.nan_to_num(x, nan=0.0)
        h = x[:, -1, :]                                                  # (B, d)

        if apply_cfg_dropout and self.training:
            none_h    = self.none_embedding(torch.zeros(1, dtype=torch.long, device=device))
            none_h    = none_h.expand(B, -1)                             # (B, d)
            drop_mask = torch.rand(B, device=device) < self.p_uncond
            h = torch.where(drop_mask.unsqueeze(-1), none_h, h)

        return h

    # ---------------------------------------------------------------------- #
    # Denoiser
    # ---------------------------------------------------------------------- #

    def _time_embed(self, t: torch.LongTensor) -> torch.Tensor:
        """t: (B,) LongTensor → (B, d_model) via sinusoidal + MLP."""
        t_norm = t.float()                                  # (B,) в [0, 1)
        t_sin  = sinusoidal_embedding(t_norm, self.d_model)              # (B, d)
        return self.time_mlp(t_sin)                                      # (B, d)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        t: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Conditional denoiser: предсказывает x_start из (x_t, h, t).

        Args:
            x: (B, d) — зашумлённый эмбеддинг x_t
            h: (B, d) — conditioning (history encoding)
            t: (B,)   — timestep

        Returns:
            x_start_pred: (B, d)
        """
        t_emb = self._time_embed(t)                                      # (B, d)
        return self.denoiser(torch.cat([x, h, t_emb], dim=-1))           # (B, d)

    def forward_uncon(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
    ) -> torch.Tensor:
        """Unconditional denoiser: использует none_embedding вместо h."""
        B      = x.shape[0]
        device = x.device
        none_h = self.none_embedding(torch.zeros(1, dtype=torch.long, device=device))
        none_h = none_h.expand(B, -1)
        return self.forward(x, none_h, t)

    # ---------------------------------------------------------------------- #
    # Training
    # ---------------------------------------------------------------------- #

    def compute_loss(
        self,
        history: torch.LongTensor,
        target: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        DreamRec loss: MSE между предсказанным и реальным x_start.

        L = E_{t,ε} || x_0 - f_θ(√ᾱ_t·x_0 + √(1-ᾱ_t)·ε, h, t) ||²
        """
        B      = history.shape[0]
        device = history.device

        z_0  = self.item_embedding(target)                               # (B, d) clean emb
        t    = torch.randint(0, self.T, (B,), device=device)            # (B,)
        eps  = torch.randn_like(z_0)

        # Forward diffusion
        sqrt_ab  = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_1ab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        x_t = sqrt_ab * z_0 + sqrt_1ab * eps                            # (B, d)

        # Conditioning (с CFG dropout)
        h = self.encode_history(history, mask, apply_cfg_dropout=True)  # (B, d)

        # Предсказываем x_start
        x_pred = self.forward(x_t, h, t)                                 # (B, d)

        return F.mse_loss(x_pred, z_0)

    # ---------------------------------------------------------------------- #
    # Inference
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """
        Один шаг DDPM reverse process с CFG.

        Args:
            x:     (B, d) — текущее состояние x_t
            h:     (B, d) — conditioning
            t_idx: int — timestep index (0..T-1)

        Returns:
            x_{t-1}: (B, d)
        """
        B      = x.shape[0]
        device = x.device
        t      = torch.full((B,), t_idx, dtype=torch.long, device=device)

        # CFG: x_start = (1+w)*cond - w*uncond
        # Без клэмпа: оригинальный DreamRec не ограничивает x_start,
        # а clamp(-5,5) зажимал бы 16% значений при std(x_start)≈3.6
        x_start_cond   = self.forward(x, h, t)
        x_start_uncond = self.forward_uncon(x, t)
        x_start = (1.0 + self.w_cfg) * x_start_cond - self.w_cfg * x_start_uncond

        # Posterior mean: E[x_{t-1} | x_t, x_0]
        coef1     = self.posterior_mean_coef1[t_idx]
        coef2     = self.posterior_mean_coef2[t_idx]
        model_mean = coef1 * x_start + coef2 * x                        # (B, d)

        if t_idx == 0:
            return model_mean

        # Добавляем стохастический шум (DDPM, не DDIM)
        post_var = self.posterior_variance[t_idx]
        noise    = torch.randn_like(x)
        return model_mean + post_var.sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        h: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """
        DDPM reverse process от x_T до x_0.

        Args:
            h:         (B, d) — conditioning.
            num_steps: None или >= T — полная цепочка по всем T шагам (оригинал).
                       < T — respaced ancestral DDPM: тот же стохастический процесс,
                       но по равномерно прорежённой сетке из num_steps таймстепов
                       (posterior-коэффициенты пересчитываются под подвыборку).

        Returns:
            x_0: (B, d) — predicted clean embedding.
        """
        if num_steps is None or num_steps >= self.T:
            x = torch.randn_like(h)
            for t_idx in reversed(range(self.T)):
                x = self.p_sample(x, h, t_idx)
            return x
        return self._sample_respaced(h, num_steps)

    @torch.no_grad()
    def _sample_respaced(self, h: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Respaced ancestral DDPM по num_steps таймстепам (num_steps < T)."""
        device = h.device
        B      = h.shape[0]

        # Равномерная подвыборка таймстепов из [0, T-1]
        S = torch.linspace(0, self.T - 1, num_steps, device=device).round().long()
        S = torch.unique(S)                                   # по возрастанию, без дублей

        # Posterior-коэффициенты для прорежённой сетки
        ac      = self.alphas_cumprod[S]                      # (k,) ᾱ на выбранных шагах
        ac_prev = torch.cat([ac.new_ones(1), ac[:-1]])        # ᾱ предыдущего выбранного шага
        betas   = 1.0 - ac / ac_prev
        alphas  = 1.0 - betas
        coef1    = ac_prev.sqrt() * betas / (1.0 - ac)
        coef2    = alphas.sqrt() * (1.0 - ac_prev) / (1.0 - ac)
        post_var = ((1.0 - ac_prev) / (1.0 - ac) * betas).clamp(min=1e-20)

        x = torch.randn_like(h)
        for i in reversed(range(len(S))):
            t = torch.full((B,), int(S[i].item()), dtype=torch.long, device=device)

            # CFG на предсказании x_start (как в p_sample)
            x_start = self.forward(x, h, t)
            if self.w_cfg != 0.0:
                x_start_uncond = self.forward_uncon(x, t)
                x_start = (1.0 + self.w_cfg) * x_start - self.w_cfg * x_start_uncond

            model_mean = coef1[i] * x_start + coef2[i] * x
            if i == 0:
                x = model_mean
            else:
                x = model_mean + post_var[i].sqrt() * torch.randn_like(x)
        return x

    @torch.no_grad()
    def recommend(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
        top_k: int = 20,
        num_steps: int | None = None,   # игнорируется, для совместимости с evaluate()
        w_cfg: float | None = None,     # игнорируется (используется self.w_cfg)
    ) -> torch.LongTensor:
        """Полный pipeline: history → sample → top-K retrieval."""
        h      = self.encode_history(history, mask, apply_cfg_dropout=False)
        oracle = self.sample(h, num_steps=num_steps)                     # (B, d)
        W      = self.item_embedding.weight[:self.n_items]               # (n_items, d)

        if self.retrieval_metric == 'cosine':
            scores = F.normalize(oracle, dim=-1) @ F.normalize(W, dim=-1).T
            return scores.topk(top_k, dim=-1).indices
        elif self.retrieval_metric == 'dot':
            scores = oracle @ W.T
            return scores.topk(top_k, dim=-1).indices
        else:  # 'l2'
            l2_sq = (oracle.pow(2).sum(-1, keepdim=True)
                     - 2.0 * oracle @ W.T
                     + W.pow(2).sum(-1).unsqueeze(0))
            return l2_sq.topk(top_k, dim=-1, largest=False).indices

    def param_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total_trainable': total, 'total_frozen': 0}


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    N_ITEMS, PADDING, B, L = 11924, 11924, 4, 10

    model = DreamRec(
        n_items=N_ITEMS, padding_idx=PADDING,
        d_model=64, T=200,
    ).to(device)

    print(f'Trainable params: {model.param_summary()["total_trainable"]:,}')

    history = torch.full((B, L), PADDING, dtype=torch.long, device=device)
    history[:, 5:] = torch.randint(0, N_ITEMS, (B, 5), device=device)
    mask   = torch.zeros(B, L, device=device); mask[:, 5:] = 1.0
    target = torch.randint(0, N_ITEMS, (B,), device=device)

    # Test 1: loss
    model.train()
    loss = model.compute_loss(history, target, mask)
    assert loss.shape == () and not torch.isnan(loss), f'Bad loss: {loss}'
    loss.backward()
    assert model.item_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad[PADDING].abs().max() == 0.0
    print(f'compute_loss: {loss.item():.4f}  (grad OK)  OK')

    # Test 2: recommend
    model.eval()
    recs = model.recommend(history, mask, top_k=20)
    assert recs.shape == (B, 20)
    assert (recs >= 0).all() and (recs < N_ITEMS).all()
    print(f'recommend: {recs.shape}  OK')

    # Test 3: noise schedule monotone
    ab = model.alphas_cumprod
    assert (ab[1:] <= ab[:-1]).all(), 'alphas_cumprod должен убывать'
    print(f'noise schedule: ab[0]={ab[0]:.4f}, ab[-1]={ab[-1]:.6f}  OK')

    # Test 4: no NaN in sample
    model.eval()
    with torch.no_grad():
        h = model.encode_history(history, mask)
        oracle = model.sample(h)
    assert not torch.isnan(oracle).any(), 'NaN in oracle'
    print(f'sample: {oracle.shape}, no NaN  OK')

    print('\nAll checks passed')
