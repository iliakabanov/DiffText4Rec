"""
Реестр моделей: параметры секции model в yaml и сборка модели по конфигу.

Тип модели задаётся в yaml ключом model_type верхнего уровня, по нему из MODELS
берётся класс конфига и функция сборки. Реестр общий для обучения (training.py)
и для оценки чекпойнтов (evaluate.py), так что конфиг самодостаточен: по одному
yaml и обучение, и оценка собирают одну и ту же модель.

Добавление модели: dataclass с её параметрами, класс конфига, build_* и строка в MODELS.
"""

from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # до импорта torch

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch.nn as nn
import yaml

from config import TextDataConfig, TrainConfig, load_config
from data import load_item_embeddings
from models.dreamrec import DreamRec, DreamRecAlphaFuse, DreamRecFMText, DreamRecText
from models.sasrec import SASRec, SASRecAlphaFuse


# --------------------------------------------------------------------------- #
# SASRec — baseline без генеративной части
# --------------------------------------------------------------------------- #

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
    retrieval_metric: str         # cosine, dot


@dataclass
class SASRecConfig(TrainConfig):
    model: SASRecModelConfig


def build_sasrec(cfg: SASRecConfig) -> SASRec:
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
        retrieval_metric=m.retrieval_metric,
    )


# --------------------------------------------------------------------------- #
# SASRecAlphaFuse — SASRec на эмбеддингах AlphaFuse
# --------------------------------------------------------------------------- #

@dataclass
class SASRecAlphaFuseModelConfig:
    d_model:     int              # = n_components PCA семантики
    null_dim:    int | None       # ровно одно из null_dim / null_thres
    null_thres:  float | None     # порог на собственные значения ковариации сырых TEM
    fusion:      str              # add — ID прибавляется к null-координатам; replace — заменяет их
    id_init:     str              # normal, zeros
    n_heads:     int
    n_layers:    int
    d_ff:        int
    dropout:     float
    loss:        str
    neg_ratio:   int
    temperature: float
    retrieval_metric: str         # cosine, dot


@dataclass
class SASRecAlphaFuseConfig(TrainConfig):
    data:  TextDataConfig
    model: SASRecAlphaFuseModelConfig


def build_sasrec_alphafuse(cfg: SASRecAlphaFuseConfig) -> SASRecAlphaFuse:
    m = cfg.model
    emb_raw = load_item_embeddings(cfg.data.emb_path)
    print(f'Item embeddings: {emb_raw.shape}')
    return SASRecAlphaFuse(
        item_embeddings_raw=emb_raw,
        padding_idx=cfg.data.padding_idx,
        d_model=m.d_model,
        null_dim=m.null_dim,
        null_thres=m.null_thres,
        fusion=m.fusion,
        id_init=m.id_init,
        whitening_cache=cfg.data.whitening_cache,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        d_ff=m.d_ff,
        dropout=m.dropout,
        max_seq_len=cfg.data.max_seq_len,
        loss=m.loss,
        neg_ratio=m.neg_ratio,
        temperature=m.temperature,
        retrieval_metric=m.retrieval_metric,
    )


# --------------------------------------------------------------------------- #
# DreamRec —эмбеддинги товаров (ID) обучаются с нуля, процесс DDPM
# --------------------------------------------------------------------------- #

@dataclass
class DreamRecModelConfig:
    d_emb:            int
    n_heads:          int
    d_ff:             int
    dropout:          float
    T:                int
    beta_schedule:    str      # exp, linear, cosine
    p_uncond:         float
    w_cfg:            float
    retrieval_metric: str      # cosine, dot, l2


@dataclass
class DreamRecConfig(TrainConfig):
    model: DreamRecModelConfig


def build_dreamrec(cfg: DreamRecConfig) -> DreamRec:
    m = cfg.model
    return DreamRec(
        n_items=cfg.data.n_items,
        padding_idx=cfg.data.padding_idx,
        d_emb=m.d_emb,
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


# --------------------------------------------------------------------------- #
# DreamRecText — DreamRec на замороженных текстовых эмбеддингах (TEM)
# --------------------------------------------------------------------------- #

@dataclass
class DreamRecTextModelConfig:
    use_pca:                 bool     # false — сырые TEM 3072d
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


def build_dreamrec_text(cfg: DreamRecTextConfig) -> DreamRecText:
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


# --------------------------------------------------------------------------- #
# DreamRecFMText — архитектура DreamRecText, генеративный процесс Flow Matching
# --------------------------------------------------------------------------- #

@dataclass
class DreamRecFMTextModelConfig:
    use_pca:                 bool     # false — сырые TEM 3072d
    n_components:            int      # игнорируется при use_pca: false
    per_component_whitening: bool     # true — Var(E)=I по каждой компоненте; false — глобальный std
    n_heads:                 int
    d_ff:                    int
    dropout:                 float
    p_uncond:                float
    w_cfg:                   float
    retrieval_metric:        str      # cosine, dot, l2


@dataclass
class DreamRecFMTextConfig(TrainConfig):
    data:  TextDataConfig
    model: DreamRecFMTextModelConfig


def build_dreamrec_fm_text(cfg: DreamRecFMTextConfig) -> DreamRecFMText:
    m = cfg.model
    emb_raw = load_item_embeddings(cfg.data.emb_path)
    print(f'Item embeddings: {emb_raw.shape}')
    return DreamRecFMText(
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
        p_uncond=m.p_uncond,
        w_cfg=m.w_cfg,
        retrieval_metric=m.retrieval_metric,
    )


# --------------------------------------------------------------------------- #
# DreamRecAlphaFuse — DreamRec на эмбеддингах AlphaFuse, процесс DDPM
# --------------------------------------------------------------------------- #

@dataclass
class DreamRecAlphaFuseModelConfig:
    d_emb:            int              # = n_components PCA семантики
    null_dim:         int | None       # ровно одно из null_dim / null_thres
    null_thres:       float | None     # порог на собственные значения ковариации сырых TEM
    fusion:           str              # add — ID прибавляется к null-координатам; replace — заменяет их
    id_init:          str              # normal, zeros
    n_heads:          int
    d_ff:             int
    dropout:          float
    T:                int
    beta_schedule:    str              # exp, linear, cosine
    p_uncond:         float
    w_cfg:            float
    retrieval_metric: str              # cosine, dot, l2


@dataclass
class DreamRecAlphaFuseConfig(TrainConfig):
    data:  TextDataConfig
    model: DreamRecAlphaFuseModelConfig


def build_dreamrec_alphafuse(cfg: DreamRecAlphaFuseConfig) -> DreamRecAlphaFuse:
    m = cfg.model
    emb_raw = load_item_embeddings(cfg.data.emb_path)
    print(f'Item embeddings: {emb_raw.shape}')
    return DreamRecAlphaFuse(
        item_embeddings_raw=emb_raw,
        padding_idx=cfg.data.padding_idx,
        d_emb=m.d_emb,
        null_dim=m.null_dim,
        null_thres=m.null_thres,
        fusion=m.fusion,
        id_init=m.id_init,
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


# --------------------------------------------------------------------------- #
# Реестр
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelSpec:
    """Класс конфига этого типа модели и сборка модели по нему."""
    config_cls: type[TrainConfig]
    build:      Callable[[TrainConfig], nn.Module]


MODELS: dict[str, ModelSpec] = {
    'sasrec':             ModelSpec(SASRecConfig,            build_sasrec),
    'sasrec_alphafuse':   ModelSpec(SASRecAlphaFuseConfig,   build_sasrec_alphafuse),
    'dreamrec':           ModelSpec(DreamRecConfig,          build_dreamrec),
    'dreamrec_text':      ModelSpec(DreamRecTextConfig,      build_dreamrec_text),
    'dreamrec_fm_text':   ModelSpec(DreamRecFMTextConfig,    build_dreamrec_fm_text),
    'dreamrec_alphafuse': ModelSpec(DreamRecAlphaFuseConfig, build_dreamrec_alphafuse),
}


def load_model_config(config_path: str | Path) -> tuple[ModelSpec, TrainConfig]:
    """Читает model_type из yaml и загружает конфиг классом соответствующей модели."""
    with open(config_path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or 'model_type' not in raw:
        raise ValueError(f'{config_path}: нет ключа model_type; допустимые: {list(MODELS)}')

    model_type = raw['model_type']
    if model_type not in MODELS:
        raise ValueError(f'{config_path}: неизвестный model_type {model_type!r}; '
                         f'допустимые: {list(MODELS)}')

    spec = MODELS[model_type]
    return spec, load_config(config_path, spec.config_cls)
