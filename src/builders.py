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

import numpy as np
import torch.nn as nn
import yaml

from config import TextDataConfig, TrainConfig, load_config
from data import load_item_embeddings, load_sasrec_item_embeddings, new_item_mask, shuffle_item_embeddings
from models.components import ItemEmbedding
from models.cosmos import DreamRecCosmos
from models.dreamrec import DreamRec, DreamRecAlphaFuse, DreamRecFMText, DreamRecText
from models.sasrec import SASRec, SASRecAlphaFuse


def _text_embeddings(data: TextDataConfig) -> np.ndarray:
    """Сырые текстовые эмбеддинги товаров; при shuffle_embeddings_seed — с перемешанными строками."""
    emb = load_item_embeddings(data.emb_path)
    print(f'Item embeddings: {emb.shape}')
    if data.shuffle_embeddings_seed is not None:
        emb = shuffle_item_embeddings(emb, data.shuffle_embeddings_seed)
        print(f'Item embeddings перемешаны (seed {data.shuffle_embeddings_seed}): семантика оторвана от товаров')
    return emb


def _tag_shuffle(model: nn.Module, data: TextDataConfig) -> nn.Module:
    """Помечает перемешивание в model_cfg чекпойнта (через items.info), чтобы прогоны различались."""
    if data.shuffle_embeddings_seed is not None:
        model.items.info['shuffle_embeddings_seed'] = data.shuffle_embeddings_seed
    return model


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
    train_prefixes: bool = False  # true — лосс на каждой позиции (оригинальный SASRec) за один forward
    loss_metric: str = 'cosine'   # сходство в лоссе: cosine, dot
    freeze_items: bool = False    # true — эмбеддинги товаров остаются случайными N(0, 0.02), не обучаются
    target_window: int = 1        # W: сколько следующих товаров правильны; > 1 требует train_prefixes и loss: ce
    train_items_only: bool = False   # true — в softmax лосса только товары из train (см. SASRec.loss_items)


@dataclass
class SASRecConfig(TrainConfig):
    model: SASRecModelConfig


def _loss_items(cfg: TrainConfig):
    """Маска товаров из train для softmax лосса SASRec при model.train_items_only, иначе None."""
    if not cfg.model.train_items_only:
        return None
    mask = ~new_item_mask(cfg.data.data_dir, cfg.data.n_items)
    print(f'Softmax лосса — только товары из train: {int(mask.sum()):,} из {cfg.data.n_items:,}')
    return mask


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
        loss_metric=m.loss_metric,
        retrieval_metric=m.retrieval_metric,
        train_prefixes=m.train_prefixes,
        freeze_items=m.freeze_items,
        target_window=m.target_window,
        loss_items=_loss_items(cfg),
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
    train_prefixes: bool = False  # true — лосс на каждой позиции (оригинальный SASRec) за один forward
    loss_metric: str = 'cosine'   # сходство в лоссе: cosine, dot


@dataclass
class SASRecAlphaFuseConfig(TrainConfig):
    data:  TextDataConfig
    model: SASRecAlphaFuseModelConfig


def build_sasrec_alphafuse(cfg: SASRecAlphaFuseConfig) -> SASRecAlphaFuse:
    m = cfg.model
    emb_raw = _text_embeddings(cfg.data)
    return _tag_shuffle(SASRecAlphaFuse(
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
        loss_metric=m.loss_metric,
        retrieval_metric=m.retrieval_metric,
        train_prefixes=m.train_prefixes,
    ), cfg.data)


def _loss_kwargs(m) -> dict:
    """Параметры лосса генеративных моделей (models.dreamrec.GenerativeRecommender)."""
    return dict(loss=m.loss, loss_metric=m.loss_metric, neg_ratio=m.neg_ratio,
                temperature=m.temperature)


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
    train_prefixes:   bool = False   # true — лосс на каждом префиксе истории за один проход
    loss:             str = 'mse'    # mse — регрессия процесса; infonce — x̂_0 против таргета и негативов
    loss_metric:      str = 'dot'    # сходство в infonce: dot, cosine
    neg_ratio:        int = 64       # негативов на позитив (infonce)
    temperature:      float = 1.0    # логиты infonce = сходство / temperature
    n_layers:         int = 1        # число SASRec-блоков в энкодере истории
    freeze_items:     bool = False   # true — эмбеддинги товаров остаются случайными N(0, 1), не обучаются


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
        n_layers=m.n_layers,
        d_ff=m.d_ff,
        dropout=m.dropout,
        max_seq_len=cfg.data.max_seq_len,
        T=m.T,
        beta_schedule=m.beta_schedule,
        p_uncond=m.p_uncond,
        w_cfg=m.w_cfg,
        retrieval_metric=m.retrieval_metric,
        train_prefixes=m.train_prefixes,
        freeze_items=m.freeze_items,
        **_loss_kwargs(m),
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
    train_prefixes:          bool = False   # true — лосс на каждом префиксе истории за один проход
    loss:                    str = 'mse'    # mse — регрессия процесса; infonce — x̂_0 против таргета и негативов
    loss_metric:             str = 'dot'    # сходство в infonce: dot, cosine
    neg_ratio:               int = 64       # негативов на позитив (infonce)
    temperature:             float = 1.0    # логиты infonce = сходство / temperature
    n_layers:                int = 1        # число SASRec-блоков в энкодере истории


@dataclass
class DreamRecTextConfig(TrainConfig):
    data:  TextDataConfig
    model: DreamRecTextModelConfig


def build_dreamrec_text(cfg: DreamRecTextConfig) -> DreamRecText:
    m = cfg.model
    emb_raw = _text_embeddings(cfg.data)
    return _tag_shuffle(DreamRecText(
        item_embeddings_raw=emb_raw,
        padding_idx=cfg.data.padding_idx,
        use_pca=m.use_pca,
        n_components=m.n_components,
        per_component_whitening=m.per_component_whitening,
        whitening_cache=cfg.data.whitening_cache,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        d_ff=m.d_ff,
        dropout=m.dropout,
        max_seq_len=cfg.data.max_seq_len,
        T=m.T,
        beta_schedule=m.beta_schedule,
        p_uncond=m.p_uncond,
        w_cfg=m.w_cfg,
        retrieval_metric=m.retrieval_metric,
        train_prefixes=m.train_prefixes,
        **_loss_kwargs(m),
    ), cfg.data)


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
    prediction:              str      # velocity — сеть предсказывает z_1 − z_0; x1 — сам z_1
    p_uncond:                float
    w_cfg:                   float
    retrieval_metric:        str      # cosine, dot, l2
    train_prefixes:          bool = False   # true — лосс на каждом префиксе истории за один проход
    loss:                    str = 'mse'    # mse — регрессия процесса; infonce — x̂_0 против таргета и негативов
    loss_metric:             str = 'dot'    # сходство в infonce: dot, cosine
    neg_ratio:               int = 64       # негативов на позитив (infonce)
    temperature:             float = 1.0    # логиты infonce = сходство / temperature
    n_layers:                int = 1        # число SASRec-блоков в энкодере истории


@dataclass
class DreamRecFMTextConfig(TrainConfig):
    data:  TextDataConfig
    model: DreamRecFMTextModelConfig


def build_dreamrec_fm_text(cfg: DreamRecFMTextConfig) -> DreamRecFMText:
    m = cfg.model
    emb_raw = _text_embeddings(cfg.data)
    return _tag_shuffle(DreamRecFMText(
        item_embeddings_raw=emb_raw,
        padding_idx=cfg.data.padding_idx,
        use_pca=m.use_pca,
        n_components=m.n_components,
        per_component_whitening=m.per_component_whitening,
        whitening_cache=cfg.data.whitening_cache,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        d_ff=m.d_ff,
        dropout=m.dropout,
        max_seq_len=cfg.data.max_seq_len,
        prediction=m.prediction,
        p_uncond=m.p_uncond,
        w_cfg=m.w_cfg,
        retrieval_metric=m.retrieval_metric,
        train_prefixes=m.train_prefixes,
        **_loss_kwargs(m),
    ), cfg.data)


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
    train_prefixes:   bool = False     # true — лосс на каждом префиксе истории за один проход
    loss:             str = 'mse'      # mse — регрессия процесса; infonce — x̂_0 против таргета и негативов
    loss_metric:      str = 'dot'      # сходство в infonce: dot, cosine
    neg_ratio:        int = 64         # негативов на позитив (infonce)
    temperature:      float = 1.0      # логиты infonce = сходство / temperature
    n_layers:         int = 1          # число SASRec-блоков в энкодере истории


@dataclass
class DreamRecAlphaFuseConfig(TrainConfig):
    data:  TextDataConfig
    model: DreamRecAlphaFuseModelConfig


def build_dreamrec_alphafuse(cfg: DreamRecAlphaFuseConfig) -> DreamRecAlphaFuse:
    m = cfg.model
    emb_raw = _text_embeddings(cfg.data)
    return _tag_shuffle(DreamRecAlphaFuse(
        item_embeddings_raw=emb_raw,
        padding_idx=cfg.data.padding_idx,
        d_emb=m.d_emb,
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
        T=m.T,
        beta_schedule=m.beta_schedule,
        p_uncond=m.p_uncond,
        w_cfg=m.w_cfg,
        retrieval_metric=m.retrieval_metric,
        train_prefixes=m.train_prefixes,
        **_loss_kwargs(m),
    ), cfg.data)


# --------------------------------------------------------------------------- #
# DreamRecCosmos — DreamRec с денойзером по мотивам COSMOS, источник эмбеддингов задаётся в конфиге
# --------------------------------------------------------------------------- #

@dataclass
class CosmosItemsConfig:
    source:            str               # learned — обучаемые с нуля; text — замороженные из emb_path;
                                         # sasrec — замороженная таблица из чекпойнта SASRec
    d_emb:             int | None = None   # только learned
    emb_path:          str | None = None   # только text
    sasrec_checkpoint: str | None = None   # только sasrec
    # только text и sasrec, смысл — как у DreamRecText
    use_pca:                 bool | None = None
    n_components:            int | None = None
    per_component_whitening: bool | None = None
    whitening_cache:         str | None = None   # кэш не проверяет эмбеддинги — свой на каждый источник


@dataclass
class CosmosModelConfig:
    items:            CosmosItemsConfig
    # энкодер истории
    n_heads:          int
    n_layers:         int
    d_ff:             int
    dropout:          float
    # денойзер
    n_blocks:         int              # ≥ 2
    denoiser_heads:   int
    mlp_mult:         int              # скрытый слой SwiGLU = d · mlp_mult
    denoiser_dropout: float            # dropout весов внимания
    # процесс DDPM и ретривал
    T:                int
    beta_schedule:    str              # exp, linear, cosine
    retrieval_metric: str              # cosine, dot, l2
    train_prefixes:   bool = False     # true — лосс на каждом префиксе истории за один проход
    target_window:    int = 1          # W: цель позиции — случайный из следующих W товаров; > 1 требует train_prefixes
    loss:             str = 'mse'      # mse — регрессия процесса; infonce — x̂_0 против таргета и негативов
    loss_metric:      str = 'dot'      # сходство в infonce: dot, cosine
    neg_ratio:        int = 64         # негативов на позитив (infonce)
    temperature:      float = 1.0      # логиты infonce = сходство / temperature


@dataclass
class CosmosConfig(TrainConfig):
    model: CosmosModelConfig


_PCA_FIELDS = {'use_pca', 'n_components', 'per_component_whitening', 'whitening_cache'}
_COSMOS_ITEM_FIELDS = {
    'learned': {'d_emb'},
    'text':    {'emb_path'} | _PCA_FIELDS,
    'sasrec':  {'sasrec_checkpoint'} | _PCA_FIELDS,
}


def _cosmos_items(cfg: CosmosConfig) -> ItemEmbedding:
    """Эмбеддинги товаров по model.items: в yaml заданы все поля своего источника и только они."""
    i, d = cfg.model.items, cfg.data
    if i.source not in _COSMOS_ITEM_FIELDS:
        raise ValueError(f'model.items.source должен быть одним из {list(_COSMOS_ITEM_FIELDS)}, получено {i.source!r}')
    needed   = _COSMOS_ITEM_FIELDS[i.source]
    optional = set().union(*_COSMOS_ITEM_FIELDS.values())
    given    = {name for name in optional if getattr(i, name) is not None}
    if given != needed:
        raise ValueError(f'model.items при source={i.source}: нет полей {sorted(needed - given)}, '
                         f'лишние поля {sorted(given - needed)}')

    if i.source == 'learned':
        return ItemEmbedding.learned(d.n_items, i.d_emb, d.padding_idx)

    emb = (load_item_embeddings(i.emb_path) if i.source == 'text'
           else load_sasrec_item_embeddings(i.sasrec_checkpoint, d.n_items))
    print(f'Item embeddings ({i.source}): {emb.shape}')
    if emb.shape[0] != d.n_items:
        raise ValueError(f'В эмбеддингах {emb.shape[0]} строк, а в конфиге n_items={d.n_items}')
    items = ItemEmbedding.text(emb, d.padding_idx, i.use_pca, i.n_components,
                               i.per_component_whitening, i.whitening_cache)
    items.info['item_embedding'] = i.source
    return items


def build_cosmos(cfg: CosmosConfig) -> DreamRecCosmos:
    m = cfg.model
    return DreamRecCosmos(
        items=_cosmos_items(cfg),
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        d_ff=m.d_ff,
        dropout=m.dropout,
        max_seq_len=cfg.data.max_seq_len,
        n_blocks=m.n_blocks,
        denoiser_heads=m.denoiser_heads,
        mlp_mult=m.mlp_mult,
        denoiser_dropout=m.denoiser_dropout,
        T=m.T,
        beta_schedule=m.beta_schedule,
        retrieval_metric=m.retrieval_metric,
        train_prefixes=m.train_prefixes,
        target_window=m.target_window,
        **_loss_kwargs(m),
    )


# --------------------------------------------------------------------------- #
# SASRecText — SASRec на замороженной таблице товаров, той же, что у DreamRecCosmos
# --------------------------------------------------------------------------- #

@dataclass
class SASRecTextModelConfig:
    items:       CosmosItemsConfig  # как у dreamrec_cosmos, кроме source: learned — для него есть обычный sasrec
    n_heads:     int
    n_layers:    int
    d_ff:        int
    dropout:     float
    loss:        str
    neg_ratio:   int
    temperature: float
    retrieval_metric: str         # cosine, dot
    train_prefixes: bool = False  # true — лосс на каждой позиции за один forward
    loss_metric: str = 'cosine'   # сходство в лоссе: cosine, dot
    target_window: int = 1        # W: сколько следующих товаров правильны; > 1 требует train_prefixes и loss: ce
    train_items_only: bool = False   # true — в softmax лосса только товары из train (см. SASRec.loss_items)


@dataclass
class SASRecTextConfig(TrainConfig):
    model: SASRecTextModelConfig


def build_sasrec_text(cfg: SASRecTextConfig) -> SASRec:
    """d_model — размерность таблицы товаров (n_components), выход привязан к ней же (tie_weights)."""
    m = cfg.model
    if m.items.source == 'learned':
        raise ValueError("sasrec_text: model.items.source=learned — это обычный sasrec (model_type: sasrec)")
    items = _cosmos_items(cfg)
    return SASRec(
        n_items=cfg.data.n_items,
        padding_idx=cfg.data.padding_idx,
        d_model=items.d_emb,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        d_ff=m.d_ff,
        dropout=m.dropout,
        max_seq_len=cfg.data.max_seq_len,
        tie_weights=True,
        loss=m.loss,
        neg_ratio=m.neg_ratio,
        temperature=m.temperature,
        loss_metric=m.loss_metric,
        retrieval_metric=m.retrieval_metric,
        train_prefixes=m.train_prefixes,
        items=items,
        target_window=m.target_window,
        loss_items=_loss_items(cfg),
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
    'sasrec_text':        ModelSpec(SASRecTextConfig,        build_sasrec_text),
    'dreamrec':           ModelSpec(DreamRecConfig,          build_dreamrec),
    'dreamrec_text':      ModelSpec(DreamRecTextConfig,      build_dreamrec_text),
    'dreamrec_fm_text':   ModelSpec(DreamRecFMTextConfig,    build_dreamrec_fm_text),
    'dreamrec_alphafuse': ModelSpec(DreamRecAlphaFuseConfig, build_dreamrec_alphafuse),
    'dreamrec_cosmos':    ModelSpec(CosmosConfig,            build_cosmos),
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
