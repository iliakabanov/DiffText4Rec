"""
Обучение DreamRecText — DreamRec на замороженных текстовых эмбеддингах товаров (TEM).

use_pca: true  → PCA-проекция 3072d → n_components + whitening
use_pca: false → сырые TEM 3072d

Запуск:
    python src/train_dreamrec_text.py --config configs/dreamrec_text.yaml
"""

from __future__ import annotations

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dataclasses import dataclass

from config import TextDataConfig, TrainConfig
from data import load_item_embeddings
from models.dreamrec import DreamRecText
from training import run


@dataclass
class DreamRecTextModelConfig:
    use_pca:                 bool
    n_components:            int      # игнорируется при use_pca: false
    per_component_whitening: bool     # true — Var(E)=I по каждой компоненте; false — глобальный std
    n_heads:                 int
    d_ff:                    int
    dropout:                 float
    T:                       int
    beta_schedule:           str      # exp, linear, cosine
    p_uncond:                float
    w_cfg:                   float
    retrieval_metric:        str      # cosine, dot, l2


@dataclass
class DreamRecTextConfig(TrainConfig):
    data:  TextDataConfig
    model: DreamRecTextModelConfig


def build_model(cfg: DreamRecTextConfig) -> DreamRecText:
    m = cfg.model
    emb_raw = load_item_embeddings(cfg.data.emb_path)
    print(f'Item embeddings: {emb_raw.shape}')
    return DreamRecText(
        item_embeddings_raw=emb_raw,
        padding_idx=cfg.data.padding_idx,
        use_pca=m.use_pca,
        n_components=m.n_components,
        per_component_whitening=m.per_component_whitening,
        whitening_cache=cfg.data.whitening_cache,
        n_heads=m.n_heads,
        d_ff=m.d_ff,
        dropout=m.dropout,
        max_seq_len=cfg.data.max_seq_len,
        T=m.T,
        beta_schedule=m.beta_schedule,
        p_uncond=m.p_uncond,
        w_cfg=m.w_cfg,
        retrieval_metric=m.retrieval_metric,
    )


if __name__ == '__main__':
    run(DreamRecTextConfig, build_model, 'dreamrec_text.yaml')
