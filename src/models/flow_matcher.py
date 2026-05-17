"""
Conditional Flow Matching для sequential recommendation.

Связывает HistoryEncoder и MLPDenoiser в единую модель.
Реализует:
    - Инициализацию whitening (с дисковым кэшем).
    - compute_loss: FM loss с classifier-free guidance dropout.
    - sample: Euler ODE с CFG.
    - retrieve: dot product retrieval по каталогу товаров.

Теория
------
Rectified flow (Liu et al. 2022) задаёт прямую траекторию от шума к данным:
    z_τ = (1 - τ) z_0 + τ z_1,   τ ∈ [0, 1]
    z_0 ~ N(0, I),   z_1 = whitened TEM embedding целевого товара

Velocity вдоль траектории — константа:
    u(τ) = z_1 - z_0

Модель учится предсказывать эту velocity:
    v_θ(z_τ, τ, c) ≈ z_1 - z_0

Лосс — MSE между предсказанной и истинной velocity:
    L = E_{τ, z_0, (history, z_1)} || v_θ(z_τ, τ, c) - (z_1 - z_0) ||²

Classifier-Free Guidance (Ho & Salimans 2022):
    Во время обучения с вероятностью p_uncond заменяем c на обучаемый Φ.
    Во время inference:
        ṽ = (1 + w) v_θ(z_τ, τ, c) - w v_θ(z_τ, τ, Φ)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.whitening import (
    apply_whitening,
    compute_whitening,
    load_whitening,
    save_whitening,
    verify_whitening,
)
from src.models.encoder import HistoryEncoder
from src.models.denoiser import MLPDenoiser, sinusoidal_embedding


class FlowMatcher(nn.Module):
    """
    Conditional Flow Matching для sequential recommendation.

    Args:
        item_embeddings_raw: (N_items, d_emb) — сырые TEM эмбеддинги товаров
                             в float32 или float64. Padding-слот не включён —
                             будет добавлен автоматически.
        padding_idx:         индекс padding-токена (= N_items).
        whitening_cache_path: путь к .npz кэшу whitening параметров.
                              Если файл существует — загружается мгновенно.
                              Если нет — считается SVD и кэш сохраняется.
        d_hidden:            скрытая размерность encoder'а.
        d_cond:              размерность conditioning vector.
        encoder_layers:      число слоёв Transformer в encoder'е.
        encoder_heads:       число голов attention в encoder'е.
        encoder_d_ff:        размерность FFN в encoder'е.
        encoder_dropout:     dropout в encoder'е.
        max_seq_len:         максимальная длина истории.
        denoiser_d_block:    внутренняя ширина денойзера.
        denoiser_d_ff:       ширина FFN в residual блоках денойзера.
        denoiser_n_blocks:   количество residual блоков.
        denoiser_time_emb_dim: размерность sinusoidal time embedding.
        denoiser_dropout:    dropout в денойзере.
        p_uncond:            вероятность CFG dropout при обучении.
    """

    def __init__(
        self,
        item_embeddings_raw: np.ndarray,
        padding_idx: int,
        whitening_cache_path: str | Path = '../data/processed/whitening_cache.npz',
        # Encoder
        d_hidden: int = 512,
        d_cond: int = 512,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        encoder_d_ff: int = 2048,
        encoder_dropout: float = 0.1,
        max_seq_len: int = 10,
        # Denoiser
        denoiser_d_block: int = 1024,
        denoiser_d_ff: int = 2048,
        denoiser_n_blocks: int = 6,
        denoiser_time_emb_dim: int = 256,
        denoiser_dropout: float = 0.0,
        # CFG
        p_uncond: float = 0.1,
    ):
        super().__init__()

        self.padding_idx = padding_idx
        self.n_items = item_embeddings_raw.shape[0]   # без padding
        self.d_emb = item_embeddings_raw.shape[1]
        self.p_uncond = p_uncond

        # ------------------------------------------------------------------ #
        # Whitening
        # ------------------------------------------------------------------ #
        mu, W = self._load_or_compute_whitening(
            item_embeddings_raw,
            Path(whitening_cache_path),
        )

        # Регистрируем как buffers — попадут в state_dict при save
        self.register_buffer('whitening_mu', torch.from_numpy(mu).float())
        self.register_buffer('whitening_W',  torch.from_numpy(W).float())

        # Применяем whitening к сырым эмбеддингам
        emb_whitened = apply_whitening(
            item_embeddings_raw.astype(np.float32), mu, W,
        )                                               # (N_items, d_emb), float32

        # Добавляем нулевую строку для padding token
        emb_with_pad = np.vstack([
            emb_whitened,
            np.zeros((1, self.d_emb), dtype=np.float32),
        ])                                              # (N_items + 1, d_emb)

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
        # Denoiser
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

        # ------------------------------------------------------------------ #
        # Unconditional token Φ для CFG
        # ------------------------------------------------------------------ #
        # Обучаемый вектор, заменяет c при unconditional forward pass.
        # Инициализируется нулями — нейтральный старт.
        self.phi = nn.Parameter(torch.zeros(d_cond))

        self.d_cond = d_cond
        self.denoiser_time_emb_dim = denoiser_time_emb_dim

    # ---------------------------------------------------------------------- #
    # Whitening helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _load_or_compute_whitening(
        emb_raw: np.ndarray,
        cache_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Загружает whitening параметры из кэша или вычисляет их через SVD.

        Кэш — техническая оптимизация (экономит 10-20s при повторном создании
        модели с нуля). Source of truth — state_dict (register_buffer).
        """
        if cache_path.exists():
            print(f'Loading whitening from cache: {cache_path}')
            mu, W = load_whitening(cache_path)
            return mu, W

        print('Computing whitening (first time, may take ~10-20s)...')
        mu, W = compute_whitening(emb_raw.astype(np.float64))
        save_whitening(cache_path, mu, W)
        print(f'Whitening cached to: {cache_path}')
        return mu, W

    def apply_whitening_to_new_items(self, emb_raw: np.ndarray) -> torch.Tensor:
        """
        Применяет whitening к новым эмбеддингам (например, новые товары
        или query-эмбеддинги при inference).

        Использует параметры из register_buffer — те же, что при обучении.

        Args:
            emb_raw: (N, d_emb) numpy array.
        Returns:
            (N, d_emb) FloatTensor на том же device что и модель.
        """
        device = self.whitening_mu.device
        emb = torch.from_numpy(emb_raw.astype(np.float32)).to(device)
        mu = self.whitening_mu
        W = self.whitening_W
        return (emb - mu) @ W

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
        Вычисляет FM loss для одного батча.

        Args:
            history: (B, L) — индексы товаров в истории (с padding слева).
            target:  (B,)   — индексы целевых товаров.
            mask:    (B, L) — 1.0 для реальных позиций, 0.0 для padding.

        Returns:
            loss: скалярный тензор — MSE между predicted и true velocity.

        Flow matching steps:
            1. z_1 = whitened embedding целевого товара.
            2. z_0 ~ N(0, I).
            3. τ ~ U(0, 1).
            4. z_τ = (1 - τ) z_0 + τ z_1.
            5. u_true = z_1 - z_0  (true velocity).
            6. c = encoder(history, mask).
            7. CFG dropout: с вероятностью p_uncond заменяем c на Φ.
            8. v_pred = denoiser(z_τ, τ, c).
            9. loss = MSE(v_pred, u_true).
        """
        B = history.shape[0]
        device = history.device

        # z_1: whitened embedding целевого товара
        # encoder.item_embedding уже хранит whitened эмбеддинги
        z_1 = self.encoder.item_embedding(target)      # (B, d_emb)

        # z_0: стандартный гауссов шум
        z_0 = torch.randn_like(z_1)                    # (B, d_emb)

        # τ ~ Uniform(0, 1)
        tau = torch.rand(B, device=device)              # (B,)

        # z_τ = (1 - τ) z_0 + τ z_1  — линейная интерполяция
        tau_expand = tau.view(B, 1)                     # (B, 1) для broadcast
        z_tau = (1.0 - tau_expand) * z_0 + tau_expand * z_1  # (B, d_emb)

        # Истинная velocity — константа вдоль прямой траектории
        u_true = z_1 - z_0                             # (B, d_emb)

        # Conditioning: encoder history → c
        c = self.encoder(history, mask)                 # (B, d_cond)

        # CFG dropout: с вероятностью p_uncond заменяем c на Φ
        # Маска: True → заменить на Φ
        drop_mask = torch.rand(B, device=device) < self.p_uncond  # (B,), bool
        phi = self.phi.unsqueeze(0).expand(B, -1)       # (B, d_cond)
        c = torch.where(drop_mask.unsqueeze(-1), phi, c)  # (B, d_cond)

        # Predicted velocity
        v_pred = self.denoiser(z_tau, tau, c)           # (B, d_emb)

        # MSE loss
        loss = ((v_pred - u_true) ** 2).mean()

        return loss

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
        Генерирует oracle embedding через Euler ODE с CFG.

        Args:
            history:   (B, L) — история пользователя.
            mask:      (B, L) — маска для padding.
            num_steps: число шагов Euler ODE.
            w_cfg:     сила classifier-free guidance.
                       w=0 → только conditional,
                       w>0 → усиливаем conditional относительно unconditional.

        Returns:
            z_0_pred: (B, d_emb) — сгенерированный oracle embedding
                      в whitened пространстве.

        CFG formula:
            ṽ = (1 + w) v_θ(z_τ, τ, c) - w v_θ(z_τ, τ, Φ)

        Euler ODE:
            z_{τ - dt} = z_τ - dt * ṽ(z_τ, τ)
            от τ=1 до τ=0.
        """
        B = history.shape[0]
        device = history.device

        # Encoding истории (один раз — не меняется во время ODE)
        c = self.encoder(history, mask)                 # (B, d_cond)

        # Unconditional token — расширяем для батча
        phi = self.phi.unsqueeze(0).expand(B, -1)       # (B, d_cond)

        # Стартовая точка: чистый гауссов шум z_1 ~ N(0, I)
        z = torch.randn(B, self.d_emb, device=device)   # (B, d_emb)

        # Euler ODE от τ=1 до τ=0
        dt = 1.0 / num_steps
        for i in range(num_steps):
            # Текущее время: от 1 до 0
            tau_val = 1.0 - i * dt
            tau = torch.full((B,), tau_val, device=device)  # (B,)

            # Conditional velocity
            v_cond = self.denoiser(z, tau, c)            # (B, d_emb)

            # Unconditional velocity (с Φ вместо c)
            v_uncond = self.denoiser(z, tau, phi)        # (B, d_emb)

            # CFG: усиливаем conditional относительно unconditional
            v = (1.0 + w_cfg) * v_cond - w_cfg * v_uncond  # (B, d_emb)

            # Euler шаг: движемся от τ к τ - dt
            z = z - dt * v                               # (B, d_emb)

        return z                                         # oracle embedding

    @torch.no_grad()
    def retrieve(
        self,
        oracle_embeddings: torch.Tensor,
        top_k: int = 20,
    ) -> torch.Tensor:
        """
        Находит top-K ближайших товаров к oracle embedding.

        Retrieval через dot product в whitened пространстве.
        Padding token (индекс = n_items) исключается.

        Args:
            oracle_embeddings: (B, d_emb) — сгенерированные oracle embeddings.
            top_k:             сколько товаров возвращать.

        Returns:
            (B, top_k) — индексы top-K товаров для каждого пользователя.
        """
        # Embedding table без padding строки: первые n_items строк
        item_emb = self.encoder.item_embedding.weight[:self.n_items]  # (N_items, d_emb)

        # Dot product: (B, d_emb) × (d_emb, N_items) → (B, N_items)
        scores = oracle_embeddings @ item_emb.T

        # Top-K по убыванию
        top_k_indices = scores.topk(top_k, dim=-1).indices  # (B, top_k)

        return top_k_indices

    # ---------------------------------------------------------------------- #
    # Convenience
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def recommend(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
        top_k: int = 20,
        num_steps: int = 50,
        w_cfg: float = 1.0,
    ) -> torch.Tensor:
        """
        Полный inference pipeline: history → oracle → top-K товаров.

        Args:
            history: (B, L)
            mask:    (B, L)
            top_k:   сколько товаров возвращать.
            num_steps, w_cfg: параметры sample().

        Returns:
            (B, top_k) — индексы рекомендованных товаров.
        """
        oracle = self.sample(history, mask, num_steps=num_steps, w_cfg=w_cfg)
        return self.retrieve(oracle, top_k=top_k)

    def param_summary(self) -> dict:
        """Возвращает словарь с количеством параметров по компонентам."""
        def count(module, trainable_only=True):
            return sum(
                p.numel() for p in module.parameters()
                if (not trainable_only or p.requires_grad)
            )

        return {
            'encoder_trainable':   count(self.encoder),
            'encoder_frozen':      count(self.encoder, trainable_only=False) - count(self.encoder),
            'denoiser_trainable':  count(self.denoiser),
            'phi':                 self.phi.numel(),
            'total_trainable':     count(self),
            'total_frozen':        count(self, trainable_only=False) - count(self),
        }


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    python -m src.models.flow_matcher
    """
    import tempfile

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Фейковые данные (маленький каталог для скорости)
    N_ITEMS = 200
    PADDING_IDX = N_ITEMS
    D_EMB = 64             # маленькая размерность для теста
    B = 4
    L = 10

    rng = np.random.default_rng(42)
    emb_raw = rng.standard_normal((N_ITEMS, D_EMB)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / 'wh_cache.npz'

        # Строим модель
        model = FlowMatcher(
            item_embeddings_raw=emb_raw,
            padding_idx=PADDING_IDX,
            whitening_cache_path=cache_path,
            d_hidden=64,
            d_cond=64,
            encoder_layers=1,
            encoder_heads=4,
            encoder_d_ff=128,
            denoiser_d_block=128,
            denoiser_d_ff=256,
            denoiser_n_blocks=2,
            denoiser_time_emb_dim=32,
        ).to(device)

        summary = model.param_summary()
        print(f'\nParameter summary:')
        for k, v in summary.items():
            print(f'  {k:25s}: {v:>10,}')

        # Фейковый батч
        history = torch.full((B, L), PADDING_IDX, dtype=torch.long)
        history[:, 5:] = torch.randint(0, N_ITEMS, (B, 5))
        mask = torch.zeros(B, L)
        mask[:, 5:] = 1.0
        target = torch.randint(0, N_ITEMS, (B,))

        history = history.to(device)
        mask = mask.to(device)
        target = target.to(device)

        # Test 1: compute_loss
        print('\nTest 1: compute_loss')
        model.train()
        loss = model.compute_loss(history, target, mask)
        assert loss.shape == (), f'Loss should be scalar, got {loss.shape}'
        assert loss.item() > 0
        print(f'  loss = {loss.item():.4f}  OK')

        # Test 2: backward
        print('\nTest 2: backward through loss')
        loss.backward()
        # Проверяем, что обучаемые параметры имеют градиенты
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f'No grad for {name}'
        # Проверяем, что frozen embedding НЕ имеет градиентов
        assert model.encoder.item_embedding.weight.grad is None
        print(f'  all trainable params have grads  OK')
        print(f'  frozen embedding has no grad  OK')

        # Test 3: sample
        print('\nTest 3: sample')
        model.eval()
        oracle = model.sample(history, mask, num_steps=5, w_cfg=1.0)
        assert oracle.shape == (B, D_EMB), f'Expected ({B}, {D_EMB}), got {oracle.shape}'
        print(f'  oracle: {oracle.shape}, mean={oracle.mean().item():.4f}  OK')

        # Test 4: retrieve
        print('\nTest 4: retrieve')
        top_k = 10
        recs = model.retrieve(oracle, top_k=top_k)
        assert recs.shape == (B, top_k), f'Expected ({B}, {top_k}), got {recs.shape}'
        # Padding token не должен попасть в рекомендации
        assert (recs != PADDING_IDX).all(), 'Padding token should not appear in recommendations'
        # Индексы в пределах каталога
        assert (recs >= 0).all() and (recs < N_ITEMS).all()
        print(f'  recs: {recs.shape}, no padding, valid ids  OK')

        # Test 5: recommend (полный pipeline)
        print('\nTest 5: recommend (full pipeline)')
        recs2 = model.recommend(history, mask, top_k=top_k, num_steps=5)
        assert recs2.shape == (B, top_k)
        print(f'  recommend: {recs2.shape}  OK')

        # Test 6: CFG w=0 vs w>0 дают разные результаты
        print('\nTest 6: CFG strength matters')
        oracle_w0 = model.sample(history, mask, num_steps=5, w_cfg=0.0)
        oracle_w2 = model.sample(history, mask, num_steps=5, w_cfg=2.0)
        assert not torch.allclose(oracle_w0, oracle_w2)
        print(f'  w=0 vs w=2: differ  OK')

        # Test 7: кэш whitening сохраняется и загружается
        print('\nTest 7: whitening cache')
        assert cache_path.exists(), 'Cache should have been saved'
        # Строим ещё одну модель — должна загрузить из кэша
        model2 = FlowMatcher(
            item_embeddings_raw=emb_raw,
            padding_idx=PADDING_IDX,
            whitening_cache_path=cache_path,
            d_hidden=64, d_cond=64,
            encoder_layers=1, encoder_heads=4, encoder_d_ff=128,
            denoiser_d_block=128, denoiser_d_ff=256,
            denoiser_n_blocks=2, denoiser_time_emb_dim=32,
        ).to(device)
        torch.testing.assert_close(model.whitening_mu, model2.whitening_mu.float())
        torch.testing.assert_close(model.whitening_W,  model2.whitening_W.float())
        print(f'  cache loaded, whitening params match  OK')

        # Test 8: state_dict содержит whitening buffers
        print('\nTest 8: state_dict contains whitening')
        sd = model.state_dict()
        assert 'whitening_mu' in sd
        assert 'whitening_W' in sd
        assert sd['whitening_mu'].shape == (D_EMB,)
        assert sd['whitening_W'].shape  == (D_EMB, D_EMB)
        print(f'  whitening_mu: {sd["whitening_mu"].shape}  OK')
        print(f'  whitening_W:  {sd["whitening_W"].shape}   OK')

        print('\nAll checks passed')