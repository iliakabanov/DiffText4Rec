"""
Генеративные рекомендеры в стиле DreamRec (Wang et al., 2023): история пользователя →
conditioning h → генеративный процесс восстанавливает из шума эмбеддинг следующего
товара (oracle) → top-K ближайших товаров.

Модель собрана из частей (models/components.py, models/processes.py):

    items     — эмбеддинги товаров           ItemEmbedding
    encoder   — история → h                  HistoryEncoder
    denoiser  — сеть f(x, h, τ) + CFG        ConcatDenoiser
    process   — зашумление, loss, сэмплинг   DDPM | FlowMatching

Модели различаются только частями:

    DreamRec           обучаемые ID-эмбеддинги                  + DDPM
    DreamRecText       замороженные текстовые (TEM)             + DDPM
    DreamRecFMText     замороженные текстовые (TEM)             + Flow Matching
    DreamRecAlphaFuse  TEM + ID в null-координатах (AlphaFuse)  + DDPM

Совместимы с evaluate() через recommend(history, mask, top_k, num_steps, w_cfg).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import AlphaFuseItemEmbedding, ConcatDenoiser, HistoryEncoder, ItemEmbedding
from models.processes import DDPM, FlowMatching

RETRIEVAL_METRICS = ('cosine', 'dot', 'l2')


def retrieve(oracle: torch.Tensor, items: torch.Tensor, metric: str, top_k: int) -> torch.LongTensor:
    """Top-K товаров, ближайших к oracle-эмбеддингу (B, d) среди items (N, d)."""
    if metric == 'cosine':
        scores = F.normalize(oracle, dim=-1) @ F.normalize(items, dim=-1).T
    elif metric == 'dot':
        scores = oracle @ items.T
    else:   # l2: ‖o − w‖² = ‖o‖² − 2·o·w + ‖w‖², ‖o‖² не влияет на порядок
        scores = 2.0 * oracle @ items.T - items.pow(2).sum(-1)
    return scores.topk(top_k, dim=-1).indices


class GenerativeRecommender(nn.Module):
    """
    Общая модель: items + encoder + denoiser + process.

    Args:
        p_uncond:         вероятность CFG dropout (h → null_token) на обучении.
        w_cfg:            сила classifier-free guidance на инференсе (0 — без guidance).
        retrieval_metric: 'cosine', 'dot' или 'l2'.
    """

    def __init__(
        self,
        items: ItemEmbedding,
        encoder: HistoryEncoder,
        denoiser: ConcatDenoiser,
        process: DDPM | FlowMatching,
        p_uncond: float,
        w_cfg: float,
        retrieval_metric: str,
    ):
        super().__init__()
        if retrieval_metric not in RETRIEVAL_METRICS:
            raise ValueError(f'retrieval_metric должен быть одним из {RETRIEVAL_METRICS}, получено {retrieval_metric!r}')
        self.items    = items
        self.encoder  = encoder
        self.denoiser = denoiser
        self.process  = process
        self.p_uncond = p_uncond
        self.w_cfg    = w_cfg
        self.retrieval_metric = retrieval_metric

    @property
    def d_emb(self) -> int:
        return self.items.d_emb

    @property
    def n_items(self) -> int:
        return self.items.n_items

    def encode_history(self, history: torch.LongTensor, mask: torch.FloatTensor) -> torch.Tensor:
        """(B, L) индексы товаров, (B, L) маска → h (B, d_emb)."""
        return self.encoder(self.items(history), mask)

    def compute_loss(self, history: torch.LongTensor, target: torch.LongTensor,
                     mask: torch.FloatTensor) -> torch.Tensor:
        h = self.denoiser.drop_condition(self.encode_history(history, mask), self.p_uncond)
        return self.process.loss(self.denoiser, self.items(target), h)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, num_steps: int | None = None,
               w_cfg: float | None = None) -> torch.Tensor:
        """Oracle-эмбеддинг из шума; w_cfg=None — self.w_cfg."""
        w = self.w_cfg if w_cfg is None else w_cfg
        return self.process.sample(self.denoiser, h, num_steps, w)

    @torch.no_grad()
    def recommend(self, history: torch.LongTensor, mask: torch.FloatTensor, top_k: int = 20,
                  num_steps: int | None = None, w_cfg: float | None = None) -> torch.LongTensor:
        """История → oracle → top_k индексов товаров (B, top_k)."""
        oracle = self.sample(self.encode_history(history, mask), num_steps, w_cfg)
        return retrieve(oracle, self.items.table, self.retrieval_metric, top_k)

    def config(self) -> dict:
        """Параметры модели для чекпойнта (model_cfg) — их читают ноутбуки анализа."""
        return {
            'model': type(self).__name__,
            'n_items': self.n_items,
            'd_emb': self.d_emb,
            **self.items.info,
            **self.process.config(),
            'p_uncond': self.p_uncond,
            'w_cfg': self.w_cfg,
            'retrieval_metric': self.retrieval_metric,
        }

    def param_summary(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {'total_trainable': trainable, 'total_frozen': frozen}


# --------------------------------------------------------------------------- #
# Модели
# --------------------------------------------------------------------------- #

class DreamRec(GenerativeRecommender):
    """DreamRec: ID-эмбеддинги товаров обучаются с нуля, процесс — DDPM."""

    def __init__(self, *, n_items: int, padding_idx: int, d_emb: int, n_heads: int, d_ff: int,
                 dropout: float, max_seq_len: int, T: int, beta_schedule: str,
                 p_uncond: float, w_cfg: float, retrieval_metric: str):
        super().__init__(
            items=ItemEmbedding.learned(n_items, d_emb, padding_idx),
            encoder=HistoryEncoder(d_emb, n_heads, d_ff, dropout, max_seq_len),
            denoiser=ConcatDenoiser(d_emb),
            process=DDPM(T, beta_schedule),
            p_uncond=p_uncond, w_cfg=w_cfg, retrieval_metric=retrieval_metric,
        )


class DreamRecText(GenerativeRecommender):
    """DreamRec на замороженных текстовых эмбеддингах товаров (TEM), процесс — DDPM."""

    def __init__(self, *, item_embeddings_raw: np.ndarray, padding_idx: int, use_pca: bool,
                 n_components: int, per_component_whitening: bool, whitening_cache: str | Path,
                 n_heads: int, d_ff: int, dropout: float, max_seq_len: int,
                 T: int, beta_schedule: str, p_uncond: float, w_cfg: float, retrieval_metric: str):
        items = ItemEmbedding.text(item_embeddings_raw, padding_idx, use_pca, n_components,
                                   per_component_whitening, whitening_cache)
        super().__init__(
            items=items,
            encoder=HistoryEncoder(items.d_emb, n_heads, d_ff, dropout, max_seq_len),
            denoiser=ConcatDenoiser(items.d_emb),
            process=DDPM(T, beta_schedule),
            p_uncond=p_uncond, w_cfg=w_cfg, retrieval_metric=retrieval_metric,
        )


class DreamRecFMText(GenerativeRecommender):
    """DreamRecText с Flow Matching вместо DDPM."""

    def __init__(self, *, item_embeddings_raw: np.ndarray, padding_idx: int, use_pca: bool,
                 n_components: int, per_component_whitening: bool, whitening_cache: str | Path,
                 n_heads: int, d_ff: int, dropout: float, max_seq_len: int, prediction: str,
                 p_uncond: float, w_cfg: float, retrieval_metric: str):
        items = ItemEmbedding.text(item_embeddings_raw, padding_idx, use_pca, n_components,
                                   per_component_whitening, whitening_cache)
        super().__init__(
            items=items,
            encoder=HistoryEncoder(items.d_emb, n_heads, d_ff, dropout, max_seq_len),
            denoiser=ConcatDenoiser(items.d_emb),
            process=FlowMatching(prediction),
            p_uncond=p_uncond, w_cfg=w_cfg, retrieval_metric=retrieval_metric,
        )


class DreamRecAlphaFuse(GenerativeRecommender):
    """DreamRec на эмбеддингах AlphaFuse: семантика TEM заморожена, ID учатся в её null-координатах; DDPM."""

    def __init__(self, *, item_embeddings_raw: np.ndarray, padding_idx: int, d_emb: int,
                 null_dim: int | None, null_thres: float | None, fusion: str, id_init: str,
                 whitening_cache: str | Path, n_heads: int, d_ff: int, dropout: float,
                 max_seq_len: int, T: int, beta_schedule: str, p_uncond: float, w_cfg: float,
                 retrieval_metric: str):
        super().__init__(
            items=AlphaFuseItemEmbedding(item_embeddings_raw, padding_idx, d_emb, null_dim,
                                         null_thres, fusion, id_init, whitening_cache),
            encoder=HistoryEncoder(d_emb, n_heads, d_ff, dropout, max_seq_len),
            denoiser=ConcatDenoiser(d_emb),
            process=DDPM(T, beta_schedule),
            p_uncond=p_uncond, w_cfg=w_cfg, retrieval_metric=retrieval_metric,
        )


# --------------------------------------------------------------------------- #
# Quick sanity check:  cd src && python -m models.dreamrec
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import os
    import tempfile
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    from models.components import TIME_STEPS, sinusoidal_embedding

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_ITEMS, B, L, D = 100, 4, 10, 16
    common = dict(padding_idx=N_ITEMS, n_heads=2, d_ff=32, dropout=0.1, max_seq_len=L,
                  p_uncond=0.5, w_cfg=1.0, retrieval_metric='cosine')
    emb_raw = np.random.default_rng(0).standard_normal((N_ITEMS, 32)).astype(np.float32)

    # Время: целые шаги различимы, у DDPM τ(t=0) = 1 (данные), τ(t=T−1) ≈ 0 (шум)
    e = F.normalize(sinusoidal_embedding(torch.tensor([0.0, float(TIME_STEPS)]), 64), dim=-1)
    assert (e[0] @ e[1]).item() < 0.5
    ddpm = DDPM(1000, 'exp')
    assert ddpm.tau(torch.tensor([0])).item() == 1.0 and ddpm.tau(torch.tensor([999])).item() < 0.01
    print('time embedding / τ: OK')

    with tempfile.TemporaryDirectory() as tmp:
        models = {
            'DreamRec': DreamRec(n_items=N_ITEMS, d_emb=D, T=50, beta_schedule='exp', **common),
            'DreamRecText (pca)': DreamRecText(item_embeddings_raw=emb_raw, use_pca=True, n_components=D,
                                               per_component_whitening=True,
                                               whitening_cache=Path(tmp) / 'wh.npz', T=50,
                                               beta_schedule='exp', **common),
            'DreamRecText (raw)': DreamRecText(item_embeddings_raw=emb_raw, use_pca=False, n_components=D,
                                               per_component_whitening=True,
                                               whitening_cache=Path(tmp) / 'wh.npz', T=50,
                                               beta_schedule='linear', **common),
            **{f'DreamRecFMText ({pred})': DreamRecFMText(
                item_embeddings_raw=emb_raw, use_pca=True, n_components=D, per_component_whitening=False,
                whitening_cache=Path(tmp) / 'wh.npz', prediction=pred, **common)
               for pred in ('velocity', 'x1')},
            **{f'DreamRecAlphaFuse ({fusion})': DreamRecAlphaFuse(
                item_embeddings_raw=emb_raw, d_emb=D, null_dim=6, null_thres=None, fusion=fusion,
                id_init='zeros', whitening_cache=Path(tmp) / 'wh.npz', T=50, beta_schedule='linear',
                **common) for fusion in ('add', 'replace')},
        }
        history = torch.full((B, L), N_ITEMS, dtype=torch.long, device=device)
        history[:, 5:] = torch.randint(0, N_ITEMS, (B, 5), device=device)
        mask   = torch.zeros(B, L, device=device); mask[:, 5:] = 1.0
        target = torch.randint(0, N_ITEMS, (B,), device=device)

        for name, model in models.items():
            model = model.to(device).train()
            loss = model.compute_loss(history, target, mask)
            loss.backward()
            assert not torch.isnan(loss)
            if name.startswith('DreamRecAlphaFuse'):
                grad_items = model.items.id_embedding.weight.grad
                assert model.items.semantic.weight.grad is None
                assert grad_items is not None and grad_items[N_ITEMS].abs().max() == 0.0
                k = model.items.nullity
                with torch.no_grad():
                    assert torch.equal(model.items.table[:, :D - k], model.items.semantic.weight[:N_ITEMS, :D - k])
            else:
                grad_items = model.items.embedding.weight.grad
                if name == 'DreamRec':
                    assert grad_items is not None and grad_items[N_ITEMS].abs().max() == 0.0
                else:
                    assert grad_items is None
            assert model.encoder.block.parameters().__next__().grad is not None
            model.eval()
            for metric in RETRIEVAL_METRICS:
                model.retrieval_metric = metric
                recs = model.recommend(history, mask, top_k=5, num_steps=7)
                assert recs.shape == (B, 5) and (recs < N_ITEMS).all()
            print(f'{name:20s} d_emb={model.d_emb:3d} loss={loss.item():.4f} | '
                  f'params {model.param_summary()} | config {model.config()}  OK')

    print('\nAll checks passed')
