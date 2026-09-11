"""
Обучение DreamRecFMText — архитектура DreamRecText, генеративный процесс Flow Matching.

use_pca: true  → PCA-проекция 3072d → n_components + whitening
use_pca: false → сырые TEM 3072d

Запуск:
    python src/train_dreamrec_fm_text.py --config configs/dreamrec_fm_text.yaml
"""

from __future__ import annotations

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dataclasses import dataclass

from config import TextDataConfig, TrainConfig
from data import load_item_embeddings
from models.dreamrec_fm_text import DreamRecFMText
from training import run


@dataclass
class DreamRecFMTextModelConfig:
    use_pca:                 bool
    n_components:            int      # игнорируется при use_pca: false
    per_component_whitening: bool     # true — Var(E)=I по каждой компоненте; false — глобальный std
    n_heads:                 int
    d_ff:                    int
    dropout:                 float
    p_uncond:                float
    w_cfg:                   float
    retrieval_metric:        str      # l2, dot, cosine


@dataclass
class DreamRecFMTextConfig(TrainConfig):
    data:  TextDataConfig
    model: DreamRecFMTextModelConfig


def build_model(cfg: DreamRecFMTextConfig) -> DreamRecFMText:
    m = cfg.model
    emb_raw = load_item_embeddings(cfg.data.emb_path)
    print(f'Item embeddings: {emb_raw.shape}')
    return DreamRecFMText(
        item_embeddings_raw=emb_raw,
        padding_idx=cfg.data.padding_idx,
        use_pca=m.use_pca,
        n_components=m.n_components,
        per_component_whitening=m.per_component_whitening,
        whitening_cache_path=cfg.data.whitening_cache,
        n_heads=m.n_heads,
        d_ff=m.d_ff,
        dropout=m.dropout,
        max_seq_len=cfg.data.max_seq_len,
        p_uncond=m.p_uncond,
        w_cfg=m.w_cfg,
        retrieval_metric=m.retrieval_metric,
    )


def checkpoint_cfg(model: DreamRecFMText) -> dict:
    """Параметры модели, которые ноутбуки читают из чекпойнта."""
    return {
        'n_items':                 model.n_items,
        'd_emb':                   model.d_emb,
        'use_pca':                 model.use_pca,
        'per_component_whitening': model.per_component_whitening,
        'w_cfg':                   model.w_cfg,
        'p_uncond':                model.p_uncond,
        'retrieval_metric':        model.retrieval_metric,
    }


if __name__ == '__main__':
    run(DreamRecFMTextConfig, build_model, 'dreamrec_fm_text.yaml', checkpoint_cfg=checkpoint_cfg)
