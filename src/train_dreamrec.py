"""
Обучение DreamRec — эмбеддинги товаров (ID) обучаются с нуля.

Запуск:
    python src/train_dreamrec.py --config configs/dreamrec.yaml
"""

from __future__ import annotations

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dataclasses import dataclass

from config import TrainConfig
from models.dreamrec import DreamRec
from training import run


@dataclass
class DreamRecModelConfig:
    d_model:          int
    n_heads:          int
    d_ff:             int
    dropout:          float
    T:                int
    beta_schedule:    str      # exp, linear, cosine
    p_uncond:         float
    w_cfg:            float
    retrieval_metric: str      # l2, dot, cosine


@dataclass
class DreamRecConfig(TrainConfig):
    model: DreamRecModelConfig


def build_model(cfg: DreamRecConfig) -> DreamRec:
    m = cfg.model
    return DreamRec(
        n_items=cfg.data.n_items,
        padding_idx=cfg.data.padding_idx,
        d_model=m.d_model,
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


def checkpoint_cfg(model: DreamRec) -> dict:
    """Параметры модели, которые ноутбуки читают из чекпойнта."""
    return {
        'n_items':          model.n_items,
        'd_model':          model.d_model,
        'T':                model.T,
        'w_cfg':            model.w_cfg,
        'p_uncond':         model.p_uncond,
        'retrieval_metric': model.retrieval_metric,
    }


if __name__ == '__main__':
    run(DreamRecConfig, build_model, 'dreamrec.yaml', checkpoint_cfg=checkpoint_cfg)
