"""
Conditional Flow Matching для sequential recommendation с PCA-проекцией.

Архитектура (LDM-style):
    TEM embeddings (3072d)
        → PCA проекция (n_components=256d)     ← сжатие пространства
        → Flow Matching в 256d                 ← основное обучение
        → Retrieval в 256d                     ← поиск ближайших товаров

Зачем PCA:
    Flow matching в 3072d требует огромного количества данных. На 15k
    sequences модель не успевает выучить распределение. PCA сжимает
    пространство в ~12 раз (3072 → 256), делая задачу feasible.

    Это LDM-аналог для рекомендаций: как Latent Diffusion Models делают
    диффузию в сжатом латентном пространстве VAE, мы делаем FM в PCA-пространстве
    семантических эмбеддингов.

Flow matching:
    z_τ = (1 - τ) z_0 + τ z_1,   τ ∈ [0, 1]
    z_0 ~ N(0, I_{256}),   z_1 = PCA(TEM(item))
    L = E || v_θ(z_τ, τ, c) - (z_1 - z_0) ||²
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from whitening import (
    apply_whitening,
    compute_whitening,
    load_whitening,
    save_whitening,
)
from models.encoder import HistoryEncoder
from models.denoiser import MLPDenoiser


class FlowMatcher(nn.Module):
    """
    Conditional Flow Matching с PCA-проекцией.

    Args:
        item_embeddings_raw:  (N_items, D_raw) — сырые TEM эмбеддинги.
        padding_idx:          индекс padding-токена (= N_items).
        n_components:         размерность PCA-пространства для FM (по умолчанию 256).
        whitening_cache_path: путь к кэшу PCA-параметров.
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
    """

    def __init__(
        self,
        item_embeddings_raw: np.ndarray,
        padding_idx: int,
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
        denoiser_d_block: int = 512,
        denoiser_d_ff: int = 1024,
        denoiser_n_blocks: int = 6,
        denoiser_time_emb_dim: int = 128,
        denoiser_dropout: float = 0.0,
        # CFG
        p_uncond: float = 0.1,
        # Retrieval
        retrieval_metric: str = 'dot',   # 'dot' или 'cosine'
    ):
        super().__init__()

        self.padding_idx             = padding_idx
        self.n_items                 = item_embeddings_raw.shape[0]
        self.d_raw                   = item_embeddings_raw.shape[1]
        self.d_emb                   = n_components
        self.n_components            = n_components
        self.per_component_whitening = per_component_whitening
        self.p_uncond                = p_uncond
        self.d_cond                  = d_cond
        self.retrieval_metric        = retrieval_metric

        # ------------------------------------------------------------------ #
        # PCA + нормализация
        # ------------------------------------------------------------------ #
        mu, V, scale = self._load_or_compute_whitening(
            item_embeddings_raw,
            n_components,
            Path(whitening_cache_path),
            per_component=per_component_whitening,
        )

        # Buffers: попадают в state_dict при сохранении модели
        self.register_buffer('whitening_mu',    torch.from_numpy(mu).float())
        self.register_buffer('whitening_V',     torch.from_numpy(V).float())
        self.register_buffer('whitening_scale', torch.from_numpy(scale).float())

        # Применяем PCA к сырым эмбеддингам
        emb_pca = apply_whitening(
            item_embeddings_raw.astype(np.float32), mu, V, scale,
        )                                               # (N_items, n_components)

        # Нулевая строка для padding token
        emb_with_pad = np.vstack([
            emb_pca,
            np.zeros((1, n_components), dtype=np.float32),
        ])                                              # (N_items + 1, n_components)

        # ------------------------------------------------------------------ #
        # Encoder
        # История тоже работает в PCA-пространстве: item lookup возвращает
        # PCA-эмбеддинги, encoder обрабатывает их и выдаёт conditioning vector.
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
        # Denoiser работает в n_components-мерном пространстве
        # ------------------------------------------------------------------ #
        self.denoiser = MLPDenoiser(
            d_emb=n_components,           # 256, не 3072!
            d_cond=d_cond,
            d_block=denoiser_d_block,
            d_ff=denoiser_d_ff,
            n_blocks=denoiser_n_blocks,
            time_emb_dim=denoiser_time_emb_dim,
            dropout=denoiser_dropout,
        )

        # Unconditional token для CFG
        self.phi = nn.Parameter(torch.zeros(d_cond))

    # ---------------------------------------------------------------------- #
    # PCA helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _load_or_compute_whitening(
        emb_raw: np.ndarray,
        n_components: int,
        cache_path: Path,
        per_component: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Загружает PCA из кэша или вычисляет SVD."""
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
            emb_raw.astype(np.float64),
            n_components=n_components,
            per_component=per_component,
            verbose=True,
        )
        save_whitening(cache_path, mu, V, scale)
        print(f'Cached to: {cache_path}')
        return mu, V, scale

    def project_new_items(self, emb_raw: np.ndarray) -> torch.Tensor:
        """
        Проецирует новые сырые эмбеддинги в PCA-пространство.
        Использует параметры из state_dict — те же что при обучении.

        Args:
            emb_raw: (N, D_raw) numpy array.
        Returns:
            (N, n_components) FloatTensor.
        """
        device = self.whitening_mu.device
        mu = self.whitening_mu.cpu().numpy()
        V  = self.whitening_V.cpu().numpy()
        scale = self.whitening_scale.cpu().numpy()
        emb_pca = apply_whitening(emb_raw.astype(np.float32), mu, V, scale)
        return torch.from_numpy(emb_pca).to(device)

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
        FM loss на одном батче.

        Все операции в PCA-пространстве (n_components-мерном).
        """
        B = history.shape[0]
        device = history.device

        # z_1: PCA-эмбеддинг целевого товара (уже в PCA-пространстве)
        z_1 = self.encoder.item_embedding(target)       # (B, n_components)

        # z_0: гауссов шум в n_components-мерном пространстве
        z_0 = torch.randn_like(z_1)                     # (B, n_components)

        # τ ~ U(0, 1)
        tau = torch.rand(B, device=device)
        tau_e = tau.view(B, 1)

        # Линейная интерполяция
        z_tau = (1.0 - tau_e) * z_0 + tau_e * z_1      # (B, n_components)

        # Истинная velocity
        u_true = z_1 - z_0                              # (B, n_components)

        # Conditioning из encoder'а
        c = self.encoder(history, mask)                  # (B, d_cond)

        # CFG dropout: p_uncond вероятность заменить c на Φ
        drop_mask = torch.rand(B, device=device) < self.p_uncond
        phi = self.phi.unsqueeze(0).expand(B, -1)
        c = torch.where(drop_mask.unsqueeze(-1), phi, c)

        # Предсказание velocity денойзером
        v_pred = self.denoiser(z_tau, tau, c)            # (B, n_components)

        return ((v_pred - u_true) ** 2).mean()

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
        Euler ODE с CFG. Возвращает oracle embedding в PCA-пространстве.

        Args:
            history:   (B, L)
            mask:      (B, L)
            num_steps: число шагов Euler.
            w_cfg:     сила CFG. 0.0 = только conditional, >0 = усиление.

        Returns:
            (B, n_components) — oracle embedding в PCA-пространстве.
        """
        B = history.shape[0]
        device = history.device

        c   = self.encoder(history, mask)                # (B, d_cond)
        phi = self.phi.unsqueeze(0).expand(B, -1)        # (B, d_cond)

        # Стартовая точка: шум в n_components-мерном пространстве
        z = torch.randn(B, self.d_emb, device=device)

        # Euler ODE от τ=0 до τ=1: шум → цель
        # Сетка: [0.0, dt, 2dt, ..., 1-dt]
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau_val = i * dt
            tau = torch.full((B,), tau_val, device=device)

            v_cond   = self.denoiser(z, tau, c)
            v_uncond = self.denoiser(z, tau, phi)
            v = (1.0 + w_cfg) * v_cond - w_cfg * v_uncond

            z = z + dt * v

        return z

    @torch.no_grad()
    def retrieve(
        self,
        oracle_embeddings: torch.Tensor,
        top_k: int = 20,
    ) -> torch.Tensor:
        """
        Top-K retrieval в PCA-пространстве по dot product.

        Args:
            oracle_embeddings: (B, n_components)
            top_k:             сколько товаров вернуть.

        Returns:
            (B, top_k) — индексы товаров.
        """
        # Embedding table без padding строки — в PCA-пространстве
        item_emb = self.encoder.item_embedding.weight[:self.n_items]  # (N, n_components)
        if self.retrieval_metric == 'cosine':
            import torch.nn.functional as F
            oracle_norm = F.normalize(oracle_embeddings, dim=-1)
            item_norm   = F.normalize(item_emb, dim=-1)
            scores = oracle_norm @ item_norm.T
        else:
            scores = oracle_embeddings @ item_emb.T                    # (B, N)
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
        """Полный pipeline: history → oracle → top-K."""
        oracle = self.sample(history, mask, num_steps=num_steps, w_cfg=w_cfg)
        return self.retrieve(oracle, top_k=top_k)

    def param_summary(self) -> dict:
        def count(module, trainable_only=True):
            return sum(
                p.numel() for p in module.parameters()
                if (not trainable_only or p.requires_grad)
            )
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

    N_ITEMS    = 200
    PADDING    = N_ITEMS
    D_RAW      = 64      # маленький для скорости
    N_COMP     = 16
    B, L       = 4, 10

    rng = np.random.default_rng(42)
    emb_raw = rng.standard_normal((N_ITEMS, D_RAW)).astype(np.float32)
    emb_raw /= np.linalg.norm(emb_raw, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmp:
        model = FlowMatcher(
            item_embeddings_raw=emb_raw,
            padding_idx=PADDING,
            n_components=N_COMP,
            whitening_cache_path=Path(tmp) / 'wh.npz',
            d_hidden=32, d_cond=32,
            encoder_layers=1, encoder_heads=4, encoder_d_ff=64,
            denoiser_d_block=64, denoiser_d_ff=128,
            denoiser_n_blocks=2, denoiser_time_emb_dim=16,
        ).to(device)

        summary = model.param_summary()
        print(f'Params: trainable={summary["total_trainable"]:,}, '
              f'frozen={summary["total_frozen"]:,}')
        print(f'd_raw={D_RAW} → d_emb={N_COMP} (PCA)')

        history = torch.full((B, L), PADDING, dtype=torch.long)
        history[:, 5:] = torch.randint(0, N_ITEMS, (B, 5))
        mask   = torch.zeros(B, L); mask[:, 5:] = 1.0
        target = torch.randint(0, N_ITEMS, (B,))
        history, mask, target = history.to(device), mask.to(device), target.to(device)

        # Test 1: loss
        print('\nTest 1: compute_loss')
        model.train()
        loss = model.compute_loss(history, target, mask)
        assert loss.shape == () and loss.item() > 0
        loss.backward()
        assert model.encoder.item_embedding.weight.grad is None
        print(f'  loss={loss.item():.4f}, frozen emb has no grad  OK')

        # Test 2: sample в n_components пространстве
        print('\nTest 2: sample')
        model.eval()
        oracle = model.sample(history, mask, num_steps=5, w_cfg=1.0)
        assert oracle.shape == (B, N_COMP), f'Expected ({B},{N_COMP}), got {oracle.shape}'
        print(f'  oracle: {oracle.shape}  OK')

        # Test 3: retrieve
        print('\nTest 3: retrieve')
        recs = model.retrieve(oracle, top_k=5)
        assert recs.shape == (B, 5)
        assert (recs >= 0).all() and (recs < N_ITEMS).all()
        assert (recs != PADDING).all()
        print(f'  recs: {recs.shape}, valid ids  OK')

        # Test 4: state_dict содержит PCA параметры
        print('\nTest 4: state_dict')
        sd = model.state_dict()
        assert 'whitening_mu'    in sd
        assert 'whitening_V'     in sd
        assert 'whitening_scale' in sd
        assert sd['whitening_mu'].shape    == (D_RAW,)
        assert sd['whitening_V'].shape     == (D_RAW, N_COMP)
        assert sd['whitening_scale'].shape == (1,)
        print(f'  mu: {sd["whitening_mu"].shape}')
        print(f'  V:  {sd["whitening_V"].shape}')
        print(f'  scale: {sd["whitening_scale"].shape}')
        print('  OK')

        # Test 5: кэш n_components-aware — при смене n_components пересчитывается
        print('\nTest 5: cache invalidation on n_components change')
        cache_path = Path(tmp) / 'wh.npz'
        model2 = FlowMatcher(
            item_embeddings_raw=emb_raw,
            padding_idx=PADDING,
            n_components=N_COMP * 2,           # другой n_components
            whitening_cache_path=cache_path,
            d_hidden=32, d_cond=32,
            encoder_layers=1, encoder_heads=4, encoder_d_ff=64,
            denoiser_d_block=64, denoiser_d_ff=128,
            denoiser_n_blocks=2, denoiser_time_emb_dim=16,
        )
        assert model2.d_emb == N_COMP * 2
        print(f'  Recomputed for n_components={N_COMP*2}  OK')

    print('\nAll checks passed')