"""
DreamRecFMText: архитектура DreamRecText, генеративный процесс — Flow Matching.

Совпадает с DreamRecText (models/dreamrec_text.py):
    - item_embedding: FROZEN (не обучается)
      * use_pca=True:  PCA-проекция TEM (3072d → n_components)
      * use_pca=False: сырые TEM эмбеддинги (3072d)
    - энкодер: pos_embedding + SASRecBlock (каузальный) + LayerNorm, h ∈ d_emb
    - CFG: обучаемый none_embedding ∈ d_emb
    - conditioning: concat([z_τ, h, t_emb]) → денойзер
    - time_mlp: sinusoidal → Linear(d, 2d) → GELU → Linear(2d, d)

Отличается только генеративный процесс:
    DDPM (DreamRecText)                    Flow Matching (здесь)
    x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε            z_τ = (1-τ)·z_0 + τ·z_1
    t ∈ {0..T-1}, расписание β             τ ~ U(0, 1), расписания нет
    сеть предсказывает x_0                 сеть предсказывает velocity u = z_1 - z_0
    стохастический ancestral DDPM          детерминированный Euler ODE, τ: 0 → 1

    Обозначения: здесь z_0 — шум, z_1 — эмбеддинг товара. В DreamRecText
    наоборот: x_0 — эмбеддинг товара.

Денойзер — Linear(3d, d), как в DreamRecText. Известное ограничение: истинное
velocity-поле требует τ-зависимого коэффициента при z_τ, а линейный слой с
аддитивным t_emb его не выражает. Сеть не может погасить стартовый шум z_0, и он
частично остаётся в oracle — поэтому при той же архитектуре FM уступает DDPM,
где шум убирает сам сэмплер (posterior-коэффициенты).

Время: τ ∈ [0, 1] масштабируется в τ·TIME_SCALE перед sinusoidal embedding —
тот же диапазон [0, 1000), что raw t в DreamRecText. Без масштаба частоты
(max_period=10000) почти не различают τ: cos-sim эмбеддингов τ=0 и τ=1 ≈ 0.97.

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

from models.sasrec import SASRecBlock
from models.time_embedding import sinusoidal_embedding
from whitening import apply_whitening, compute_whitening, save_whitening


# τ ∈ [0, 1] → τ·TIME_SCALE ∈ [0, 1000): как raw t при T=1000 в DreamRecText.
TIME_SCALE = 1000.0


class DreamRecFMText(nn.Module):
    """
    DreamRecText с Flow Matching вместо DDPM.

    Args:
        item_embeddings_raw:     (N_items, D_raw) — сырые TEM эмбеддинги.
        padding_idx:             индекс padding-токена (= N_items).
        use_pca:                 True = PCA 3072→n_components, False = raw 3072d.
        n_components:            размерность PCA (используется если use_pca=True).
        per_component_whitening: True = Var(E)=I точно; False = глобальный std.
        whitening_cache_path:    путь к кэшу PCA-параметров.
        n_heads:                 число голов self-attention.
        d_ff:                    ширина FFN в энкодере.
        dropout:                 dropout в энкодере и positional emb.
        max_seq_len:             максимальная длина истории.
        p_uncond:                вероятность CFG dropout при обучении.
        w_cfg:                   сила CFG при инференсе.
        retrieval_metric:        'l2', 'dot', 'cosine'.
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
        self.item_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(emb_with_pad),
            freeze=True,
            padding_idx=padding_idx,
        )

        d = self.d_emb  # alias

        # ------------------------------------------------------------------ #
        # Энкодер, CFG-токен и time_mlp — те же, что в DreamRecText
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

        # ------------------------------------------------------------------ #
        # Денойзер: concat([z_τ, h, t_emb]) → velocity, Linear(3d, d) как в DreamRecText
        # ------------------------------------------------------------------ #
        self.denoiser = nn.Sequential(
            nn.Linear(d * 3, d),
        )

        self._init_weights()

    # ---------------------------------------------------------------------- #
    # PCA helpers (копия из DreamRecText)
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
    # Init — идентично DreamRecText
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
    # History encoding — идентично DreamRecText
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
    # Velocity net — concat, как денойзер DreamRecText
    # ---------------------------------------------------------------------- #

    def _time_embed(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: (B,) в [0, 1] → (B, d_emb) via sinusoidal(τ·TIME_SCALE) + MLP."""
        t_sin = sinusoidal_embedding(tau * TIME_SCALE, self.d_emb)
        return self.time_mlp(t_sin)

    def forward(self, x: torch.Tensor, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Предсказание velocity v(z_τ, h, τ)."""
        t_emb = self._time_embed(tau)
        return self.denoiser(torch.cat([x, h, t_emb], dim=-1))

    def forward_uncon(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        B      = x.shape[0]
        device = x.device
        none_h = self.none_embedding(torch.zeros(1, dtype=torch.long, device=device))
        none_h = none_h.expand(B, -1)
        return self.forward(x, none_h, tau)

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

        z_1 = self.item_embedding(target)                                # (B, d_emb) — frozen
        z_0 = torch.randn_like(z_1)                                      # (B, d_emb) — шум

        tau   = torch.rand(B, device=device)                             # (B,) ~ U(0, 1)
        tau_e = tau.unsqueeze(1)

        z_tau  = (1.0 - tau_e) * z_0 + tau_e * z_1
        u_true = z_1 - z_0

        h      = self.encode_history(history, mask, apply_cfg_dropout=True)
        v_pred = self.forward(z_tau, h, tau)

        return F.mse_loss(v_pred, u_true)

    # ---------------------------------------------------------------------- #
    # Inference — Euler ODE
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def sample(
        self,
        h: torch.Tensor,
        num_steps: int = 50,
        w_cfg: float | None = None,
    ) -> torch.Tensor:
        """
        Euler ODE от τ=0 (шум) до τ=1 (эмбеддинг товара) с CFG.

        Args:
            h:         (B, d_emb) — conditioning из encode_history.
            num_steps: число шагов Euler.
            w_cfg:     сила CFG. None → self.w_cfg.

        Returns:
            (B, d_emb) — oracle embedding в пространстве товаров.
        """
        B      = h.shape[0]
        device = h.device
        w      = self.w_cfg if w_cfg is None else w_cfg

        z  = torch.randn_like(h)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau = torch.full((B,), i * dt, device=device)
            v   = self.forward(z, h, tau)
            if w != 0.0:
                v = (1.0 + w) * v - w * self.forward_uncon(z, tau)
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
        h      = self.encode_history(history, mask, apply_cfg_dropout=False)
        oracle = self.sample(h, num_steps=num_steps, w_cfg=w_cfg)
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

    # Время должно различаться: τ=0 и τ=1 не должны давать почти один вектор.
    e = F.normalize(sinusoidal_embedding(torch.tensor([0.0, 1.0]) * TIME_SCALE, 64), dim=-1)
    cos01 = (e[0] @ e[1]).item()
    assert cos01 < 0.5, f'time embedding degenerate: cos(τ=0, τ=1) = {cos01:.3f}'
    print(f'time embedding: cos(τ=0, τ=1) = {cos01:.3f}  OK')

    with tempfile.TemporaryDirectory() as tmp:
        for use_pca in [True, False]:
            print(f'\n--- use_pca={use_pca} ---')
            model = DreamRecFMText(
                item_embeddings_raw=emb_raw,
                padding_idx=PADDING,
                use_pca=use_pca,
                n_components=8,
                whitening_cache_path=Path(tmp) / 'wh.npz',
                n_heads=2,
                d_ff=32,
                w_cfg=1.0,
            ).to(device)
            summary = model.param_summary()
            print(f'  d_emb={model.d_emb}, trainable={summary["total_trainable"]:,}, '
                  f'frozen={summary["total_frozen"]:,}')

            history = torch.full((B, L), PADDING, dtype=torch.long, device=device)
            history[:, 5:] = torch.randint(0, N_ITEMS, (B, 5), device=device)
            mask    = torch.zeros(B, L, device=device); mask[:, 5:] = 1.0
            target  = torch.randint(0, N_ITEMS, (B,), device=device)

            model.train()
            loss = model.compute_loss(history, target, mask)
            assert not torch.isnan(loss), 'NaN loss!'
            loss.backward()
            assert model.item_embedding.weight.grad is None, 'Frozen emb should have no grad'
            assert model.encoder_block.parameters().__next__().grad is not None, \
                'Encoder should receive grad'
            print(f'  compute_loss: {loss.item():.4f}  (frozen emb OK, encoder grad OK)')

            model.eval()
            recs = model.recommend(history, mask, top_k=5, num_steps=5)
            assert recs.shape == (B, 5) and (recs < N_ITEMS).all()
            print(f'  recommend: {recs.shape}  OK')

    print('\nAll checks passed')
