"""
DDPM (Denoising Diffusion Probabilistic Models) для sequential recommendation.

Архитектура идентична FlowMatcher (encoder + MLPDenoiser + PCA-проекция),
меняется только механизм генерации:

    Flow Matching:  v_θ(z_τ, τ, c) ≈ z_1 - z_0           (velocity)
    DDPM:           ε_θ(z_t, t/T, c) ≈ ε                  (шум)

Прямой процесс (forward diffusion):
    q(z_t | z_0) = N(z_t; √ᾱ_t · z_0, (1 - ᾱ_t) · I)
    z_t = √ᾱ_t · z_0 + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)

Шедуль: косинусный (Nichol & Dhariwal 2021), T=1000.

Сэмплирование: DDIM (Song et al. 2020), детерминировано, num_steps шагов.
    x̂_0 = (z_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
    z_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1 - ᾱ_{t-1}) · ε_θ
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

from whitening import apply_whitening, compute_whitening, load_whitening, save_whitening
from models.encoder import HistoryEncoder
from models.denoiser import MLPDenoiser


# --------------------------------------------------------------------------- #
# Noise schedule
# --------------------------------------------------------------------------- #

def _cosine_schedule(T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Косинусный noise schedule (Nichol & Dhariwal, 2021).

    Returns:
        betas:           (T,) — β_t
        alphas_cumprod:  (T,) — ᾱ_t = Π_{s=1}^{t} (1 - β_s)
    """
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]                                    # (T+1,)
    betas = (1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(0.0, 0.999)
    return betas, alphas_cumprod[1:].float()                     # (T,), (T,)


# --------------------------------------------------------------------------- #
# DDPM model
# --------------------------------------------------------------------------- #

class DDPMModel(nn.Module):
    """
    Conditional DDPM с PCA-проекцией и DDIM-сэмплированием.

    Все параметры идентичны FlowMatcher, добавлен только T (шаги диффузии).

    Args:
        item_embeddings_raw:  (N_items, D_raw) — сырые TEM эмбеддинги.
        padding_idx:          индекс padding-токена (= N_items).
        n_components:         размерность PCA-пространства (по умолчанию 256).
        whitening_cache_path: путь к кэшу PCA-параметров.
        T:                    число шагов диффузии (по умолчанию 1000).
        d_hidden:             скрытая размерность encoder'а.
        d_cond:               размерность conditioning vector.
        encoder_layers:       число Transformer слоёв.
        encoder_heads:        число attention голов.
        encoder_d_ff:         ширина FFN в encoder'е.
        encoder_dropout:      dropout в encoder'е.
        max_seq_len:          максимальная длина истории.
        denoiser_d_block:     внутренняя ширина денойзера.
        denoiser_d_ff:        ширина FFN в денойзере.
        denoiser_n_blocks:    число residual блоков.
        denoiser_time_emb_dim: размерность time embedding.
        denoiser_dropout:     dropout в денойзере.
        p_uncond:             вероятность CFG dropout при обучении.
        retrieval_metric:     'dot' или 'cosine'.
    """

    def __init__(
        self,
        item_embeddings_raw: np.ndarray,
        padding_idx: int,
        n_components: int = 256,
        whitening_cache_path: str | Path = './data/processed/whitening_cache.npz',
        T: int = 1000,
        # Encoder
        d_hidden: int = 512,
        d_cond: int = 512,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        encoder_d_ff: int = 2048,
        encoder_dropout: float = 0.1,
        max_seq_len: int = 10,
        # Denoiser
        denoiser_d_block: int = 512,
        denoiser_d_ff: int = 1024,
        denoiser_n_blocks: int = 6,
        denoiser_time_emb_dim: int = 128,
        denoiser_dropout: float = 0.0,
        # CFG
        p_uncond: float = 0.1,
        # Retrieval
        retrieval_metric: str = 'dot',
    ):
        super().__init__()

        self.padding_idx      = padding_idx
        self.n_items          = item_embeddings_raw.shape[0]
        self.n_components     = n_components
        self.d_emb            = n_components
        self.T                = T
        self.p_uncond         = p_uncond
        self.d_cond           = d_cond
        self.retrieval_metric = retrieval_metric

        # ------------------------------------------------------------------ #
        # PCA
        # ------------------------------------------------------------------ #
        mu, V, scale = self._load_or_compute_whitening(
            item_embeddings_raw, n_components, Path(whitening_cache_path),
        )
        self.register_buffer('whitening_mu',    torch.from_numpy(mu).float())
        self.register_buffer('whitening_V',     torch.from_numpy(V).float())
        self.register_buffer('whitening_scale', torch.from_numpy(scale).float())

        emb_pca = apply_whitening(item_embeddings_raw.astype(np.float32), mu, V, scale)
        emb_with_pad = np.vstack([
            emb_pca,
            np.zeros((1, n_components), dtype=np.float32),
        ])

        # ------------------------------------------------------------------ #
        # Encoder
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
        # Denoiser (предсказывает шум ε, та же архитектура что в FM)
        # ------------------------------------------------------------------ #
        self.denoiser = MLPDenoiser(
            d_emb=n_components,
            d_cond=d_cond,
            d_block=denoiser_d_block,
            d_ff=denoiser_d_ff,
            n_blocks=denoiser_n_blocks,
            time_emb_dim=denoiser_time_emb_dim,
            dropout=denoiser_dropout,
        )

        # Unconditional token для CFG
        self.phi = nn.Parameter(torch.zeros(d_cond))

        # ------------------------------------------------------------------ #
        # Noise schedule (buffers → попадают в state_dict)
        # ------------------------------------------------------------------ #
        betas, alphas_cumprod = _cosine_schedule(T)
        self.register_buffer('betas',                          betas)
        self.register_buffer('alphas_cumprod',                 alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod',            alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod',  (1.0 - alphas_cumprod).sqrt())

    # ---------------------------------------------------------------------- #
    # PCA helpers (идентично FlowMatcher)
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _load_or_compute_whitening(emb_raw, n_components, cache_path):
        if cache_path.exists():
            data = np.load(cache_path)
            if 'V' in data and data['V'].shape[1] == n_components:
                print(f'Loading whitening from cache: {cache_path}')
                return data['mu'], data['V'], data['scale']
            cached_k = data['V'].shape[1] if 'V' in data else '?'
            print(f'Cache has n_components={cached_k}, need {n_components}. Recomputing...')
        print(f'Computing PCA {emb_raw.shape[1]}d -> {n_components}d...')
        mu, V, scale = compute_whitening(emb_raw.astype(np.float64), n_components=n_components, verbose=True)
        save_whitening(cache_path, mu, V, scale)
        print(f'Cached to: {cache_path}')
        return mu, V, scale

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
        DDPM loss: MSE между предсказанным и реальным шумом.

        L = E_{t, z_0, ε} || ε - ε_θ(√ᾱ_t·z_0 + √(1-ᾱ_t)·ε, t/T, c) ||²
        """
        B      = history.shape[0]
        device = history.device

        # Чистый эмбеддинг целевого айтема z_0 в PCA-пространстве
        z_0 = self.encoder.item_embedding(target)               # (B, n_components)

        # Случайный timestep t ~ Uniform{0..T-2}
        # T-1 исключаем: alphas_cumprod[T-1]=0 → sqrt=0, бесполезный шаг
        t = torch.randint(0, self.T - 1, (B,), device=device)   # (B,)

        # Реальный шум
        eps = torch.randn_like(z_0)                             # (B, n_components)

        # Прямой процесс: z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε
        sqrt_ab  = self.sqrt_alphas_cumprod[t].unsqueeze(1)          # (B, 1)
        sqrt_1ab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1) # (B, 1)
        z_t = sqrt_ab * z_0 + sqrt_1ab * eps                   # (B, n_components)

        # Conditioning
        c = self.encoder(history, mask)                         # (B, d_cond)

        # CFG dropout
        drop_mask = torch.rand(B, device=device) < self.p_uncond
        phi = self.phi.unsqueeze(0).expand(B, -1)
        c = torch.where(drop_mask.unsqueeze(-1), phi, c)

        # Предсказание шума; время нормируем в [0, 1) для sinusoidal embedding
        t_norm   = t.float() / self.T                           # (B,)
        eps_pred = self.denoiser(z_t, t_norm, c)                # (B, n_components)

        return F.mse_loss(eps_pred, eps)

    # ---------------------------------------------------------------------- #
    # Inference
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def sample(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
        num_steps: int = 50,
        w_cfg: float = 1.0,
    ) -> torch.Tensor:
        """
        DDIM sampling с CFG.

        Args:
            history:   (B, L)
            mask:      (B, L)
            num_steps: число DDIM шагов (меньше T, обычно 50).
            w_cfg:     сила CFG. 0.0 = только conditional.

        Returns:
            (B, n_components) — предсказанный чистый эмбеддинг.
        """
        B      = history.shape[0]
        device = history.device

        c   = self.encoder(history, mask)               # (B, d_cond)
        phi = self.phi.unsqueeze(0).expand(B, -1)       # (B, d_cond)

        # Стартовый шум z_T ~ N(0, I)
        z = torch.randn(B, self.d_emb, device=device)   # (B, n_components)

        # Сетка timesteps от T-2 до 0.
        # T-1 исключён: alphas_cumprod[T-1]=0 (косинусный шедуль достигает нуля
        # ровно на последнем шаге), что даёт деление на sqrt(0) и полное
        # насыщение x0_pred = ±clamp на первом же шаге.
        timesteps = torch.linspace(self.T - 2, 0, num_steps).long().to(device)

        for i, t_cur in enumerate(timesteps):
            t_batch = t_cur.expand(B)                   # (B,)
            t_norm  = t_batch.float() / self.T          # (B,)

            # Предсказание шума с CFG
            eps_cond   = self.denoiser(z, t_norm, c)
            eps_uncond = self.denoiser(z, t_norm, phi)
            eps = eps_cond + w_cfg * (eps_cond - eps_uncond)

            # ᾱ_t; clamp предотвращает деление на ноль при ab_t ≈ 0
            ab_t     = self.alphas_cumprod[t_cur]        # scalar
            sqrt_ab  = ab_t.sqrt().clamp(min=1e-8)
            sqrt_1ab = (1.0 - ab_t).sqrt()

            # Предсказываем чистый z_0
            x0_pred = (z - sqrt_1ab * eps) / sqrt_ab
            x0_pred = x0_pred.clamp(-5.0, 5.0)

            # Финальный шаг — возвращаем x0_pred напрямую
            if i == len(timesteps) - 1:
                z = x0_pred
                break

            # ᾱ_{t-1}
            t_prev  = timesteps[i + 1]
            ab_prev = self.alphas_cumprod[t_prev]        # scalar

            # DDIM шаг с согласованным направлением:
            # Перевычисляем epsilon-направление из (зажатого) x0_pred,
            # чтобы оно соответствовало x0_pred, а не исходному (незажатому).
            # eps_dir = (z_t - sqrt(ab_t) * x0_pred) / sqrt(1 - ab_t)
            eps_dir = (z - sqrt_ab * x0_pred) / sqrt_1ab.clamp(min=1e-8)

            # z_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1 - ᾱ_{t-1}) · eps_dir
            z = ab_prev.sqrt() * x0_pred + (1.0 - ab_prev).sqrt() * eps_dir

        return z                                         # (B, n_components)

    @torch.no_grad()
    def retrieve(
        self,
        oracle_embeddings: torch.Tensor,
        top_k: int = 20,
    ) -> torch.Tensor:
        """Top-K retrieval в PCA-пространстве (идентично FlowMatcher)."""
        item_emb = self.encoder.item_embedding.weight[:self.n_items]
        if self.retrieval_metric == 'cosine':
            oracle_norm = F.normalize(oracle_embeddings, dim=-1)
            item_norm   = F.normalize(item_emb, dim=-1)
            scores = oracle_norm @ item_norm.T
        else:
            scores = oracle_embeddings @ item_emb.T
        return scores.topk(top_k, dim=-1).indices

    @torch.no_grad()
    def recommend(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
        top_k: int = 20,
        num_steps: int = 50,
        w_cfg: float = 1.0,
    ) -> torch.Tensor:
        """Полный pipeline: history → DDIM → top-K."""
        oracle = self.sample(history, mask, num_steps=num_steps, w_cfg=w_cfg)
        return self.retrieve(oracle, top_k=top_k)

    def param_summary(self) -> dict:
        def count(module, trainable_only=True):
            return sum(p.numel() for p in module.parameters()
                       if (not trainable_only or p.requires_grad))
        return {
            'encoder_trainable':  count(self.encoder),
            'encoder_frozen':     count(self.encoder, False) - count(self.encoder),
            'denoiser_trainable': count(self.denoiser),
            'phi':                self.phi.numel(),
            'total_trainable':    count(self),
            'total_frozen':       count(self, False) - count(self),
        }


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import tempfile
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    N_ITEMS = 200
    PADDING = N_ITEMS
    D_RAW   = 64
    N_COMP  = 16
    B, L    = 4, 10
    T_TEST  = 100

    rng = np.random.default_rng(42)
    emb_raw = rng.standard_normal((N_ITEMS, D_RAW)).astype(np.float32)
    emb_raw /= np.linalg.norm(emb_raw, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmp:
        model = DDPMModel(
            item_embeddings_raw=emb_raw,
            padding_idx=PADDING,
            n_components=N_COMP,
            whitening_cache_path=Path(tmp) / 'wh.npz',
            T=T_TEST,
            d_hidden=32, d_cond=32,
            encoder_layers=1, encoder_heads=4, encoder_d_ff=64,
            denoiser_d_block=64, denoiser_d_ff=128,
            denoiser_n_blocks=2, denoiser_time_emb_dim=16,
        ).to(device)

        summary = model.param_summary()
        print(f'Trainable: {summary["total_trainable"]:,}, Frozen: {summary["total_frozen"]:,}')

        history = torch.full((B, L), PADDING, dtype=torch.long, device=device)
        history[:, 5:] = torch.randint(0, N_ITEMS, (B, 5), device=device)
        mask   = torch.zeros(B, L, device=device); mask[:, 5:] = 1.0
        target = torch.randint(0, N_ITEMS, (B,), device=device)

        # Test 1: loss
        model.train()
        loss = model.compute_loss(history, target, mask)
        assert loss.shape == () and not torch.isnan(loss), f'Bad loss: {loss}'
        loss.backward()
        assert model.encoder.item_embedding.weight.grad is None, 'Frozen emb should have no grad'
        print(f'compute_loss: {loss.item():.4f}  (frozen emb OK)  OK')

        # Test 2: sample
        model.eval()
        oracle = model.sample(history, mask, num_steps=5, w_cfg=1.0)
        assert oracle.shape == (B, N_COMP), f'Expected ({B},{N_COMP}), got {oracle.shape}'
        assert not torch.isnan(oracle).any(), 'NaN in oracle'
        print(f'sample: {oracle.shape}  OK')

        # Test 3: recommend
        recs = model.recommend(history, mask, top_k=5, num_steps=5)
        assert recs.shape == (B, 5)
        assert (recs >= 0).all() and (recs < N_ITEMS).all()
        assert (recs != PADDING).all()
        print(f'recommend: {recs.shape}, valid ids  OK')

        # Test 4: noise schedule проверка
        ab = model.alphas_cumprod
        assert ab[0] < 1.0 and ab[-1] > 0.0, 'Schedule должен идти от ~1 к ~0'
        assert (ab[1:] <= ab[:-1]).all(), 'ᾱ_t должен убывать'
        print(f'noise schedule: ᾱ_0={ab[0]:.4f}, ᾱ_T={ab[-1]:.6f}, monotone  OK')

        # Test 5: state_dict содержит schedule
        sd = model.state_dict()
        for key in ['whitening_mu', 'whitening_V', 'alphas_cumprod', 'sqrt_alphas_cumprod']:
            assert key in sd, f'{key} not in state_dict'
        print(f'state_dict keys: whitening + schedule present  OK')

    print('\nAll checks passed')
