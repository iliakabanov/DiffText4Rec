"""
Обучение SASRec — baseline без генеративной части.

Запуск:
    python src/train_sasrec.py --config configs/sasrec.yaml
"""

from __future__ import annotations

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dataclasses import dataclass

import torch

from config import TrainConfig
from evaluate import evaluate
from models.sasrec import SASRec
from training import run


@dataclass
class SASRecModelConfig:
    d_model:     int
    n_heads:     int
    n_layers:    int
    d_ff:        int
    dropout:     float
    tie_weights: bool
    loss:        str
    neg_ratio:   int
    temperature: float


@dataclass
class SASRecConfig(TrainConfig):
    model: SASRecModelConfig


def build_model(cfg: SASRecConfig) -> SASRec:
    m = cfg.model
    return SASRec(
        n_items=cfg.data.n_items,
        padding_idx=cfg.data.padding_idx,
        d_model=m.d_model,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        d_ff=m.d_ff,
        dropout=m.dropout,
        max_seq_len=cfg.data.max_seq_len,
        tie_weights=m.tie_weights,
        loss=m.loss,
        neg_ratio=m.neg_ratio,
        temperature=m.temperature,
    )


def train_ndcg20(model: SASRec, loaders: dict, device: torch.device) -> float:
    """NDCG@20 на train — пишется в log_val.csv рядом с val, чтобы видеть переобучение."""
    return evaluate(model=model, dataloader=loaders['train'], top_k_values=(20,),
                    num_sample_steps=1, w_cfg=0.0, device=device, show_progress=False)['NDCG@20']


if __name__ == '__main__':
    run(SASRecConfig, build_model, 'sasrec.yaml', extra_val_metrics={'train_ndcg20': train_ndcg20})
