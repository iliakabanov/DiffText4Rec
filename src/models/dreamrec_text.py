"""
DreamRecText: DreamRec с замороженными текстовыми эмбеддингами (TEM).

Отличие от базового DreamRec:
    - item_embedding: FROZEN (не обучается)
      * use_pca=True:  PCA-проекция TEM (3072d → n_components=256d)
      * use_pca=False: сырые TEM эмбеддинги (3072d)
    - d_emb = n_components (256) или d_raw (3072) — зависит от use_pca

Гипотеза: замороженные семантические эмбеддинги предотвращают коллапс
и дают диффузионному процессу смысловое пространство для генерации.

Совместим с evaluate() через recommend(history, mask, top_k, num_steps, w_cfg).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sasrec import SASRecBlock
from models.denoiser import sinusoidal_embedding
from whitening import apply_whitening, compute_whitening, load_whitening, save_whitening
from models.dreamrec import _exp_schedule, _linear_schedule, _cosine_schedule, _make_schedule


class DreamRecText(nn.Module):
    """
    DreamRec с замороженными TEM-эмбеддингами.

    Args:
        item_embeddings_raw:  (N_items, D_raw) — сырые TEM эмбеддинги.
        padding_idx:          индекс padding-токена (= N_items).
        use_pca:              True = PCA 3072→n_components, False = raw 3072d.
        n_components:         размерность PCA (используется если use_pca=True).
        whitening_cache_path: путь к кэшу PCA-параметров.
        n_heads:              число голов self-attention.
        d_ff:                 ширина FFN в энкодере.
        dropout:              dropout в энкодере и positional emb.
        max_seq_len:          максимальная длина истории.
        T:                    число шагов диффузии.
        beta_schedule:        'exp', 'linear', 'cosine'.
        p_uncond:             вероятность CFG dropout при обучении.
        w_cfg:                сила CFG при инференсе.
        retrieval_metric:     'l2', 'dot', 'cosine'.
    """

    def __init__(
        self,
        item_embeddings_raw: np.ndarray,
        padding_idx: int = 11924,
        use_pca: bool = True,
        n_components: int = 256,
        per_component_whitening: bool = False,
        whitening_cache_path: str | Path = './data/processed/whitening_cache.npz',
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 10,
        T: int = 1000,
        beta_schedule: str = 'exp',
        p_uncond: float = 0.1,
        w_cfg: float = 2.0,
        retrieval_metric: str = 'l2',
    ):
        super().__init__()

        self.n_items                  = item_embeddings_raw.shape[0]
        self.padding_idx              = padding_idx
        self.use_pca                  = use_pca
        self.per_component_whitening  = per_component_whitening
        self.T                        = T
        self.p_uncond                 = p_uncond
        self.w_cfg                    = w_cfg
        self.retrieval_metric         = retrieval_metric

        # ------------------------------------------------------------------ #
        # Item embeddings — FROZEN
        # ------------------------------------------------------------------ #
        if use_pca:
            # PCA-проекция (как в FlowMatcher)
            mu, V, scale = self._load_or_compute_whitening(
                item_embeddings_raw, n_components, Path(whitening_cache_path),
                per_component=per_component_whitening,
            )
            self.register_buffer('whitening_mu',    torch.from_numpy(mu).float())
            self.register_buffer('whitening_V',     torch.from_numpy(V).float())
            self.register_buffer('whitening_scale', torch.from_numpy(scale).float())
            emb = apply_whitening(item_embeddings_raw.astype(np.float32), mu, V, scale)
            self.d_emb = n_components
        else:
            # Сырые TEM эмбеддинги
            emb = item_embeddings_raw.astype(np.float32)
            self.d_emb = item_embeddings_raw.shape[1]   # 3072

        emb_with_pad = np.vstack([emb, np.zeros((1, self.d_emb), dtype=np.float32)])
        self.item_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(emb_with_pad),
            freeze=True,
            padding_idx=padding_idx,
        )

        d = self.d_emb  # alias

        # ------------------------------------------------------------------ #
        # Остальные компоненты — те же что в DreamRec, но с d_emb вместо d_model
        # ------------------------------------------------------------------ #
        self.pos_embedding  = nn.Embedding(max_seq_len, d)
        self.none_embedding = nn.Embedding(1, d)    # обучаемый unconditional token
        self.dropout_emb    = nn.Dropout(dropout)

        self.encoder_block = SASRecBlock(d, n_heads, d_ff, dropout)
        self.encoder_norm  = nn.LayerNorm(d)

        self.time_mlp = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )

        self.denoiser = nn.Sequential(
            nn.Linear(d * 3, d)
        )

        # ------------------------------------------------------------------ #
        # Noise schedule buffers
        # ------------------------------------------------------------------ #
        betas = _make_schedule(beta_schedule, T)
        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer('betas',                         betas)
        self.register_buffer('alphas_cumprod',                alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod',           alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1.0 - alphas_cumprod).sqrt())

        coef1    = alphas_cumprod_prev.sqrt() * betas / (1.0 - alphas_cumprod)
        coef2    = alphas.sqrt() * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        post_var = (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * betas

        self.register_buffer('posterior_mean_coef1', coef1)
        self.register_buffer('posterior_mean_coef2', coef2)
        self.register_buffer('posterior_variance',   post_var.clamp(min=1e-20))

        self._init_weights()

    # ---------------------------------------------------------------------- #
    # PCA helpers (копия из FlowMatcher)
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _load_or_compute_whitening(emb_raw, n_components, cache_path, per_component=False):
        expected_scale_shape = (n_components,) if per_component else (1,)
        if cache_path.exists():
            data = np.load(cache_path)
            if ('V' in data and data['V'].shape[1] == n_components
                    and data['scale'].shape == expected_scale_shape):
                print(f'Loading whitening from cache: {cache_path}')
                return data['mu'], data['V'], data['scale']
            cached_k = data['V'].shape[1] if 'V' in data else '?'
            cached_scale = data['scale'].shape if 'scale' in data else '?'
            print(f'Cache mismatch (n_components={cached_k}, scale={cached_scale}), '
                  f'need ({n_components}, scale={expected_scale_shape}). Recomputing...')
        print(f'Computing PCA {emb_raw.shape[1]}d -> {n_components}d '
              f'(per_component={per_component})...')
        mu, V, scale = compute_whitening(
            emb_raw.astype(np.float64), n_components=n_components,
            per_component=per_component, verbose=True,
        )
        save_whitening(cache_path, mu, V, scale)
        print(f'Cached to: {cache_path}')
        return mu, V, scale

    # ---------------------------------------------------------------------- #
    # Init
    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        nn.init.zeros_(self.pos_embedding.weight)
        nn.init.normal_(self.none_embedding.weight, 0.0, 1.0)
        for m in [self.time_mlp, self.denoiser]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ---------------------------------------------------------------------- #
    # History encoding — идентично DreamRec
    # ---------------------------------------------------------------------- #

    def encode_history(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
        apply_cfg_dropout: bool = False,
    ) -> torch.Tensor:
        B, L   = history.shape
        device = history.device

        x = self.item_embedding(history)                                 # (B, L, d_emb)
        x = x + self.pos_embedding(torch.arange(L, device=device))
        x = self.dropout_emb(x)

        causal_mask      = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        key_padding_mask = (mask == 0)

        x = self.encoder_block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.encoder_norm(x)
        x = torch.nan_to_num(x, nan=0.0)
        h = x[:, -1, :]                                                  # (B, d_emb)

        if apply_cfg_dropout and self.training:
            none_h    = self.none_embedding(torch.zeros(1, dtype=torch.long, device=device))
            none_h    = none_h.expand(B, -1)
            drop_mask = torch.rand(B, device=device) < self.p_uncond
            h = torch.where(drop_mask.unsqueeze(-1), none_h, h)

        return h

    # ---------------------------------------------------------------------- #
    # Denoiser — идентично DreamRec
    # ---------------------------------------------------------------------- #

    def _time_embed(self, t: torch.LongTensor) -> torch.Tensor:
        t_norm = t.float()
        t_sin  = sinusoidal_embedding(t_norm, self.d_emb)
        return self.time_mlp(t_sin)

    def forward(self, x: torch.Tensor, h: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        t_emb = self._time_embed(t)
        return self.denoiser(torch.cat([x, h, t_emb], dim=-1))

    def forward_uncon(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        B      = x.shape[0]
        device = x.device
        none_h = self.none_embedding(torch.zeros(1, dtype=torch.long, device=device))
        none_h = none_h.expand(B, -1)
        return self.forward(x, none_h, t)

    # ---------------------------------------------------------------------- #
    # Training — идентично DreamRec (z_0 из frozen emb)
    # ---------------------------------------------------------------------- #

    def compute_loss(
        self,
        history: torch.LongTensor,
        target: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.Tensor:
        B      = history.shape[0]
        device = history.device

        z_0  = self.item_embedding(target)                               # (B, d_emb) — frozen
        t    = torch.randint(0, self.T, (B,), device=device)
        eps  = torch.randn_like(z_0)

        sqrt_ab  = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_1ab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        x_t = sqrt_ab * z_0 + sqrt_1ab * eps

        h = self.encode_history(history, mask, apply_cfg_dropout=True)
        x_pred = self.forward(x_t, h, t)

        return F.mse_loss(x_pred, z_0)

    # ---------------------------------------------------------------------- #
    # Inference — идентично DreamRec
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, h: torch.Tensor, t_idx: int) -> torch.Tensor:
        B      = x.shape[0]
        device = x.device
        t      = torch.full((B,), t_idx, dtype=torch.long, device=device)

        x_start_cond   = self.forward(x, h, t)
        x_start_uncond = self.forward_uncon(x, t)
        x_start = (1.0 + self.w_cfg) * x_start_cond - self.w_cfg * x_start_uncond

        coef1      = self.posterior_mean_coef1[t_idx]
        coef2      = self.posterior_mean_coef2[t_idx]
        model_mean = coef1 * x_start + coef2 * x

        if t_idx == 0:
            return model_mean

        post_var = self.posterior_variance[t_idx]
        noise    = torch.randn_like(x)
        return model_mean + post_var.sqrt() * noise

    @torch.no_grad()
    def sample(self, h: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        """
        DDPM reverse process от x_T до x_0.

        num_steps: None или >= T — полная цепочка по всем T шагам (оригинал);
                   < T — respaced ancestral DDPM: тот же стохастический процесс,
                   но по равномерно прорежённой сетке из num_steps таймстепов.
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

        S = torch.linspace(0, self.T - 1, num_steps, device=device).round().long()
        S = torch.unique(S)                                   # по возрастанию, без дублей

        ac      = self.alphas_cumprod[S]                      # (k,) ᾱ на выбранных шагах
        ac_prev = torch.cat([ac.new_ones(1), ac[:-1]])
        betas   = 1.0 - ac / ac_prev
        alphas  = 1.0 - betas
        coef1    = ac_prev.sqrt() * betas / (1.0 - ac)
        coef2    = alphas.sqrt() * (1.0 - ac_prev) / (1.0 - ac)
        post_var = ((1.0 - ac_prev) / (1.0 - ac) * betas).clamp(min=1e-20)

        x = torch.randn_like(h)
        for i in reversed(range(len(S))):
            t = torch.full((B,), int(S[i].item()), dtype=torch.long, device=device)

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
        num_steps: int | None = None,
        w_cfg: float | None = None,
    ) -> torch.LongTensor:
        h      = self.encode_history(history, mask, apply_cfg_dropout=False)
        oracle = self.sample(h, num_steps=num_steps)
        W      = self.item_embedding.weight[:self.n_items]

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
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {'total_trainable': trainable, 'total_frozen': frozen}


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import os, tempfile
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    N_ITEMS, PADDING, B, L, D_RAW = 100, 100, 4, 10, 32

    rng = np.random.default_rng(42)
    emb_raw = rng.standard_normal((N_ITEMS, D_RAW)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        for use_pca in [True, False]:
            print(f'\n--- use_pca={use_pca} ---')
            model = DreamRecText(
                item_embeddings_raw=emb_raw,
                padding_idx=PADDING,
                use_pca=use_pca,
                n_components=8,
                whitening_cache_path=Path(tmp) / 'wh.npz',
                T=10,
                n_heads=2,
                d_ff=32,
            ).to(device)
            summary = model.param_summary()
            print(f'  d_emb={model.d_emb}, trainable={summary["total_trainable"]:,}, frozen={summary["total_frozen"]:,}')

            history = torch.full((B, L), PADDING, dtype=torch.long, device=device)
            history[:, 5:] = torch.randint(0, N_ITEMS, (B, 5), device=device)
            mask    = torch.zeros(B, L, device=device); mask[:, 5:] = 1.0
            target  = torch.randint(0, N_ITEMS, (B,), device=device)

            model.train()
            loss = model.compute_loss(history, target, mask)
            assert not torch.isnan(loss), f'NaN loss!'
            loss.backward()
            assert model.item_embedding.weight.grad is None, 'Frozen emb should have no grad'
            print(f'  compute_loss: {loss.item():.4f}  (frozen emb OK)')

            model.eval()
            recs = model.recommend(history, mask, top_k=5)
            assert recs.shape == (B, 5) and (recs < N_ITEMS).all()
            print(f'  recommend: {recs.shape}  OK')

    print('\nAll checks passed')
