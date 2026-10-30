"""
Загрузка yaml-конфигов в dataclass'ы.

Dataclass'ы повторяют секции yaml (seed, model_type, data, model, optimizer, eval, output), поэтому
в коде параметры читаются как cfg.optimizer.lr, cfg.data.batch_size, cfg.model.T.

Загрузка строгая: лишний ключ, отсутствующий ключ или значение не того типа — сразу
ошибка с именем файла и путём до ключа. Дефолт есть только у eval.num_sample_steps
(он нужен лишь генеративным моделям), так что yaml — единственный источник значений.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


# --------------------------------------------------------------------------- #
# Общие секции конфига
# --------------------------------------------------------------------------- #

@dataclass
class DataConfig:
    data_dir:    str
    padding_idx: int
    n_items:     int
    batch_size:  int
    num_workers: int
    max_seq_len: int


@dataclass
class TextDataConfig(DataConfig):
    """Данные моделей на текстовых эмбеддингах (DreamRecText, DreamRecFMText)."""
    emb_path:        str
    whitening_cache: str
    shuffle_embeddings_seed: int | None = None   # число — перемешать строки эмбеддингов (абляция семантики)


@dataclass
class OptimizerConfig:
    name:          str              # adam — weight_decay как L2 в градиенте; adamw — развязанный decay
    lr:            float
    weight_decay:  float
    grad_clip:     float | None     # null — без клиппинга
    num_epochs:    int
    warmup_epochs: int
    ema_decay:     float | None     # null — без EMA; иначе оценка и best.pt по усреднённым весам


@dataclass
class EvalConfig:
    every_epochs:     int
    top_k:            tuple[int, ...]
    num_sample_steps: int | None = None   # шагов сэмплинга; SASRec не использует


@dataclass
class OutputConfig:
    dir:      str
    run_name: str


@dataclass
class TrainConfig:
    """
    Базовый конфиг обучения. В builders.MODELS каждый тип модели уточняет тип model
    (и data у моделей на текстовых эмбеддингах).
    """
    seed:       int
    model_type: str            # ключ в builders.MODELS: какую модель собирать по этому конфигу
    data:       DataConfig
    model:      Any
    optimizer:  OptimizerConfig
    eval:       EvalConfig
    output:     OutputConfig


# --------------------------------------------------------------------------- #
# Загрузка
# --------------------------------------------------------------------------- #

C = TypeVar('C')


def load_config(path: str | Path, cls: type[C]) -> C:
    """Читает yaml и строит из него dataclass cls со строгой проверкой ключей и типов."""
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return _build(cls, raw, str(path), '')


def _build(cls: type, raw: Any, path: str, prefix: str) -> Any:
    section = prefix.rstrip('.') or 'корень'
    if not isinstance(raw, dict):
        raise TypeError(f'{path}: [{section}] ожидалась секция-словарь, получено {raw!r}')

    fields  = {f.name: f for f in dataclasses.fields(cls)}
    hints   = typing.get_type_hints(cls)
    unknown = sorted(set(raw) - set(fields))
    missing = sorted(name for name, f in fields.items() if name not in raw
                     and f.default is dataclasses.MISSING
                     and f.default_factory is dataclasses.MISSING)
    problems = ([f'лишние ключи {unknown}'] if unknown else []) + \
               ([f'нет ключей {missing}'] if missing else [])
    if problems:
        raise ValueError(f'{path}: [{section}] ' + '; '.join(problems))

    return cls(**{name: _convert(hints[name], value, path, prefix + name)
                  for name, value in raw.items()})


def _convert(tp: Any, value: Any, path: str, key: str) -> Any:
    origin = typing.get_origin(tp)

    if origin in (typing.Union, types.UnionType):              # X | None
        if value is None and type(None) in typing.get_args(tp):
            return None
        inner = [a for a in typing.get_args(tp) if a is not type(None)]
        return _convert(inner[0], value, path, key)

    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path, key + '.')

    if origin is tuple:                                         # tuple[int, ...]
        if not isinstance(value, list):
            raise TypeError(f'{path}: {key} должен быть списком, получено {value!r}')
        elem = typing.get_args(tp)[0]
        return tuple(_convert(elem, v, path, key) for v in value)

    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        # PyYAML читает 1e-4 (без точки в мантиссе) как строку — поэтому float() и от str
        if not isinstance(value, bool):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    elif tp is str:
        if isinstance(value, str):
            return value
    elif tp is Any:
        return value
    else:
        raise TypeError(f'{path}: {key}: неподдерживаемый тип поля {tp!r}')

    raise TypeError(f'{path}: {key} должен иметь тип {tp.__name__}, получено {value!r}')
