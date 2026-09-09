"""
DreamRecFMText: Conditional Flow Matching для sequential recommendation
с замороженными текстовыми эмбеддингами (TEM).

Что сохранено «в духе DreamRec»:
    - item_embedding: FROZEN (не обучается)
      * use_pca=True:  PCA-проекция TEM (3072d → n_components=256d)
      * use_pca=False: сырые TEM эмбеддинги (3072d)
    - CFG: через обучаемый none_embedding (unconditional токен).

Генеративный механизм — Flow Matching (velocity):
    z_τ = (1 - τ) z_0 + τ z_1,   τ ∈ [0, 1]
    z_0 ~ N(0, I),   z_1 = frozen emb(target)
    цель — предсказать velocity u = z_1 - z_0,  L = MSE(v_pred, z_1 - z_0)
    сэмплинг — Euler ODE от τ=0 до τ=1 за num_steps шагов.
    денойзер — FiLM (MLPDenoiser): conditioning c модулирует каждый блок.

Компоненты (те же, что в fm_text):
    - encoder:  HistoryEncoder (Transformer, история → conditioning c ∈ d_cond).
    - denoiser: MLPDenoiser (FiLM).
Отличие от fm_text — CFG-механизм: здесь обучаемый none_embedding вместо phi.

Почему FiLM-денойзер, а не concat (как в DDPM-версии DreamRec):
    Velocity-предсказание требует, чтобы сеть опиралась ТОЛЬКО на c — в u = z_1 - z_0
    нет "лёгкого сигнала" из z_τ (в отличие от x_0-предсказания в DDPM). При concat
    conditioning input projection учится в основном на z_τ и почти игнорирует c
    (см. models/denoiser.py), из-за чего velocity-loss застревает на E‖u‖². FiLM
    подаёт c отдельно и модулирует каждый residual блок.

Совместим с evaluate() через recommend(history, mask, top_k, num_steps, w_cfg).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import HistoryEncoder
from models.denoiser import MLPDenoiser
from whitening import apply_whitening, compute_whitening, load_whitening, save_whitening


class DreamRecFMText(nn.Module):
    """
    Conditional Flow Matching с замороженными TEM-эмбеддингами и HistoryEncoder.

    Args:
        item_embeddings_raw:  (N_items, D_raw) — сырые TEM эмбеддинги.
        padding_idx:          индекс padding-токена (= N_items).
        use_pca:              True = PCA 3072→n_components, False = raw 3072d.
        n_components:         размерность PCA (используется если use_pca=True).
        whitening_cache_path: путь к кэшу PCA-параметров.
        d_hidden:             скрытая размерность encoder'а.
        d_cond:               размерность conditioning vector.
        encoder_layers:       число Transformer слоёв encoder'а.
        encoder_heads:        число attention голов encoder'а.
        encoder_d_ff:         ширина FFN в encoder'е.
        encoder_dropout:      dropout в encoder'е.
        max_seq_len:          максимальная длина истории.
        denoiser_d_block:     внутренняя ширина денойзера.
        denoiser_d_ff:        ширина FFN в денойзере.
        denoiser_n_blocks:    число residual блоков денойзера.
        denoiser_time_emb_dim: размерность time embedding.
        denoiser_dropout:     dropout в денойзере.
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
        # Encoder
        d_hidden: int = 512,
        d_cond: int = 512,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        encoder_d_ff: int = 2048,
        encoder_dropout: float = 0.1,
        max_seq_len: int = 10,
        # Denoiser
        denoiser_d_block: int = 256,
        denoiser_d_ff: int = 1024,
        denoiser_n_blocks: int = 3,
        denoiser_time_emb_dim: int = 128,
        denoiser_dropout: float = 0.1,
        # CFG
        p_uncond: float = 0.1,
        w_cfg: float = 2.0,
        retrieval_metric: str = 'l2',
    ):
        super().__init__()

        self.n_items                  = item_embeddings_raw.shape[0]
        self.padding_idx              = padding_idx
        self.use_pca                  = use_pca
        self.per_component_whitening  = per_component_whitening
        self.p_uncond                 = p_uncond
        self.w_cfg                    = w_cfg
        self.retrieval_metric         = retrieval_metric
        self.d_cond                   = d_cond

        # ------------------------------------------------------------------ #
        # Item embeddings — FROZEN
        # ------------------------------------------------------------------ #
        if use_pca:
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
            emb = item_embeddings_raw.astype(np.float32)
            self.d_emb = item_embeddings_raw.shape[1]   # 3072

        emb_with_pad = np.vstack([emb, np.zeros((1, self.d_emb), dtype=np.float32)])

        # ------------------------------------------------------------------ #
        # Encoder: история → conditioning c (frozen item_embedding внутри)
        # ------------------------------------------------------------------ #
        self.encoder = HistoryEncoder(
            item_embeddings=torch.from_numpy(emb_with_pad),
            padding_idx=padding_idx,
            d_hidden=d_hidden,
            d_cond=d_cond,
            n_layers=encoder_layers,
            n_heads=encoder_heads,
            d_ff=encoder_d_ff,
            dropout=encoder_dropout,
            max_seq_len=max_seq_len,
        )

        # ------------------------------------------------------------------ #
        # FiLM-денойзер: работает в d_emb-пространстве, conditioning c ∈ d_cond
        # ------------------------------------------------------------------ #
        self.denoiser = MLPDenoiser(
            d_emb=self.d_emb,
            d_cond=d_cond,
            d_block=denoiser_d_block,
            d_ff=denoiser_d_ff,
            n_blocks=denoiser_n_blocks,
            time_emb_dim=denoiser_time_emb_dim,
            dropout=denoiser_dropout,
        )

        # Обучаемый unconditional токен для CFG (в d_cond-пространстве).
        self.none_embedding = nn.Embedding(1, d_cond)
        nn.init.normal_(self.none_embedding.weight, 0.0, 1.0)

    # ---------------------------------------------------------------------- #
    # PCA helpers
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
    # History encoding + CFG dropout
    # ---------------------------------------------------------------------- #

    def encode_history(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
        apply_cfg_dropout: bool = False,
    ) -> torch.Tensor:
        """История → conditioning c (B, d_cond). При обучении — CFG dropout."""
        c = self.encoder(history, mask)                                  # (B, d_cond)

        if apply_cfg_dropout and self.training:
            B         = history.shape[0]
            device    = history.device
            none_c    = self.none_embedding(torch.zeros(1, dtype=torch.long, device=device))
            none_c    = none_c.expand(B, -1)
            drop_mask = torch.rand(B, device=device) < self.p_uncond
            c = torch.where(drop_mask.unsqueeze(-1), none_c, c)

        return c

    # ---------------------------------------------------------------------- #
    # Denoiser / velocity net — FiLM, τ непрерывное в [0, 1]
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor, c: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Предсказание velocity v(z_τ, τ, c). c — conditioning для FiLM."""
        return self.denoiser(x, tau, c)

    def forward_uncon(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Unconditional velocity через обучаемый none_embedding (для CFG)."""
        B      = x.shape[0]
        device = x.device
        none_c = self.none_embedding(torch.zeros(1, dtype=torch.long, device=device))
        none_c = none_c.expand(B, -1)
        return self.forward(x, none_c, tau)

    # ---------------------------------------------------------------------- #
    # Training — Conditional Flow Matching
    # ---------------------------------------------------------------------- #

    def compute_loss(
        self,
        history: torch.LongTensor,
        target: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.Tensor:
        B      = history.shape[0]
        device = history.device

        z_1 = self.encoder.item_embedding(target)                        # (B, d_emb) — frozen
        z_0 = torch.randn_like(z_1)                                      # (B, d_emb)

        tau   = torch.rand(B, device=device)                            # (B,) ~ U(0, 1)
        tau_e = tau.unsqueeze(1)

        z_tau  = (1.0 - tau_e) * z_0 + tau_e * z_1                       # (B, d_emb)
        u_true = z_1 - z_0                                              # (B, d_emb)

        c      = self.encode_history(history, mask, apply_cfg_dropout=True)
        v_pred = self.forward(z_tau, c, tau)

        return F.mse_loss(v_pred, u_true)

    # ---------------------------------------------------------------------- #
    # Inference — Euler ODE
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        num_steps: int = 50,
        w_cfg: float | None = None,
    ) -> torch.Tensor:
        """
        Euler ODE от τ=0 (шум) до τ=1 (цель) с CFG.

        Args:
            c:         (B, d_cond) — conditioning из encode_history.
            num_steps: число шагов Euler.
            w_cfg:     сила CFG. None → self.w_cfg.

        Returns:
            (B, d_emb) — oracle embedding в пространстве товаров.
        """
        B      = c.shape[0]
        device = c.device
        w      = self.w_cfg if w_cfg is None else w_cfg

        z  = torch.randn(B, self.d_emb, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau      = torch.full((B,), i * dt, device=device)
            v_cond   = self.forward(z, c, tau)
            v_uncond = self.forward_uncon(z, tau)
            v = (1.0 + w) * v_cond - w * v_uncond
            z = z + dt * v

        return z

    @torch.no_grad()
    def recommend(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
        top_k: int = 20,
        num_steps: int = 50,
        w_cfg: float | None = None,
    ) -> torch.LongTensor:
        c      = self.encode_history(history, mask, apply_cfg_dropout=False)
        oracle = self.sample(c, num_steps=num_steps, w_cfg=w_cfg)
        W      = self.encoder.item_embedding.weight[:self.n_items]

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
            model = DreamRecFMText(
                item_embeddings_raw=emb_raw,
                padding_idx=PADDING,
                use_pca=use_pca,
                n_components=8,
                whitening_cache_path=Path(tmp) / 'wh.npz',
                d_hidden=32,
                d_cond=32,
                encoder_layers=1,
                encoder_heads=4,
                encoder_d_ff=64,
                denoiser_d_block=32,
                denoiser_d_ff=64,
                denoiser_n_blocks=2,
                denoiser_time_emb_dim=16,
                w_cfg=1.0,
            ).to(device)
            summary = model.param_summary()
            print(f'  d_emb={model.d_emb}, d_cond={model.d_cond}, '
                  f'trainable={summary["total_trainable"]:,}, frozen={summary["total_frozen"]:,}')

            history = torch.full((B, L), PADDING, dtype=torch.long, device=device)
            history[:, 5:] = torch.randint(0, N_ITEMS, (B, 5), device=device)
            mask    = torch.zeros(B, L, device=device); mask[:, 5:] = 1.0
            target  = torch.randint(0, N_ITEMS, (B,), device=device)

            model.train()
            loss = model.compute_loss(history, target, mask)
            assert not torch.isnan(loss), 'NaN loss!'
            loss.backward()
            assert model.encoder.item_embedding.weight.grad is None, 'Frozen emb should have no grad'
            print(f'  compute_loss: {loss.item():.4f}  (frozen emb OK)')

            model.eval()
            recs = model.recommend(history, mask, top_k=5, num_steps=5)
            assert recs.shape == (B, 5) and (recs < N_ITEMS).all()
            print(f'  recommend: {recs.shape}  OK')

    print('\nAll checks passed')
