"""
Dataset и DataLoader для последовательных рекомендаций.

Сплиты приходят в формате pandas DataFrame, сохранённого через pickle,
с тремя колонками:
    - seq:     list[int], индексы товаров истории по времени;
    - len_seq: int, число реальных взаимодействий в seq;
    - next:    int, индекс целевого товара (то, что надо предсказать).

Поддерживаются оба формата seq:
    - без паддинга, переменной длины (scripts/preprocess.py): len_seq = len(seq);
    - фиксированной длины с паддингом слева (data/ATG): реальные товары — последние len_seq.
В обоих случаях историей считаются последние len_seq элементов seq.

Окно модели задаёт seq_len: история обрезается до последних seq_len товаров и дополняется
слева padding_idx. Так данные не зависят от выбранной длины контекста, а модель видит
привычную ей раскладку: последний товар всегда в последней позиции.

Эмбеддинги товаров не загружаются здесь — они держатся в модели как
frozen nn.Embedding и индексируются по id из батча. Это эффективнее,
чем тащить тензоры эмбеддингов через DataLoader.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class ATGDataset(Dataset):
    """
    Один split (train / val / test).

    Args:
        path:        путь к .df файлу (pickle с pandas DataFrame).
        seq_len:     окно истории (max_seq_len модели): столько позиций в history.
                     Истории длиннее обрезаются с начала, короче — дополняются padding_idx слева.
        padding_idx: id паддинга (обычно = число товаров).
        seq_col:     имя колонки с историей (default 'seq').
        len_col:     имя колонки с реальной длиной (default 'len_seq').
        target_col:  имя колонки с таргетом (default 'next').

    Каждый __getitem__ возвращает dict:
        {
            'history': LongTensor[seq_len]  — индексы товаров (padding слева),
            'target':  LongTensor[]         — индекс целевого товара,
            'mask':    FloatTensor[seq_len] — 1.0 для реальных позиций, 0.0 для padding.
        }

    Padding слева, поэтому реальные взаимодействия — последние min(len_seq, seq_len) позиций,
    а последний товар истории всегда стоит в последней позиции.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        padding_idx: int,
        seq_col: str = 'seq',
        len_col: str = 'len_seq',
        target_col: str = 'next',
    ):
        df = pd.read_pickle(path)
        self._validate(df, seq_col, len_col, target_col)

        # Плотные numpy-массивы: доставать значения из pandas в __getitem__ намного дороже
        self.len_seq = np.minimum(df[len_col].to_numpy(dtype=np.int64), seq_len)
        self.target = df[target_col].to_numpy(dtype=np.int64)
        self.history = np.full((len(df), seq_len), padding_idx, dtype=np.int64)
        for i, (seq, n) in enumerate(zip(df[seq_col], self.len_seq)):
            self.history[i, seq_len - n:] = np.asarray(seq, dtype=np.int64)[-n:]

        self.seq_len = seq_len
        self.padding_idx = padding_idx
        self.n = len(df)

        # mask[i, j] = 1, если позиция реальная: j >= seq_len - len_seq[i]
        positions = np.arange(seq_len)[None, :]                  # (1, seq_len)
        threshold = (seq_len - self.len_seq)[:, None]            # (N, 1)
        self.mask = (positions >= threshold).astype(np.float32)  # (N, seq_len)

    @staticmethod
    def _validate(
        df: pd.DataFrame,
        seq_col: str,
        len_col: str,
        target_col: str,
    ) -> None:
        """Проверяет схему датафрейма перед использованием."""
        required = {seq_col, len_col, target_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f'DataFrame is missing required columns: {sorted(missing)}. '
                f'Found columns: {sorted(df.columns)}'
            )

        if len(df) == 0:
            raise ValueError('DataFrame is empty.')

        # len_seq — число реальных товаров; их берут с конца seq, поэтому seq не короче len_seq
        lens = df[len_col].to_numpy()
        if lens.min() < 1:
            raise ValueError(f'{len_col} must be >= 1, found {lens.min()}.')
        sizes = df[seq_col].iloc[:1000].map(len).to_numpy()
        if (sizes < lens[:len(sizes)]).any():
            raise ValueError(f'{seq_col} короче {len_col}: истории длиной {sizes.min()} '
                             f'при len_seq до {lens[:len(sizes)].max()}.')

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            'history': torch.from_numpy(self.history[idx]),    # int64 (seq_len,)
            'target':  torch.tensor(self.target[idx], dtype=torch.long),  # int64 scalar
            'mask':    torch.from_numpy(self.mask[idx]),       # float32 (seq_len,)
        }


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #

def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 256,
    num_workers: int = 4,
    seq_len: int = 10,
    padding_idx: int | None = None,
    splits: Iterable[str] = ('train', 'val', 'test'),
    pin_memory: bool = True,
    train_filename: str = 'train_data.df',
    val_filename: str = 'val_data.df',
    test_filename: str = 'test_data.df',
    seed: int | None = None,
) -> dict[str, DataLoader]:
    """
    Создаёт DataLoader'ы для всех или подмножества сплитов.

    Args:
        data_dir:    директория с файлами {train,val,test}_data.df.
        batch_size:  размер батча. Train shuffle, val/test без shuffle.
        num_workers: число воркеров для DataLoader.
        seq_len:     окно истории (max_seq_len модели).
        padding_idx: id паддинга; None — взять item_num из data_statis.df в data_dir.
        splits:      какие сплиты загружать.
        pin_memory:  ускоряет перенос на GPU.
        seed:        сид для шаффла train-сплита. Если None — шаффл берёт
                     глобальный RNG, и порядок зависит от всего, что успело
                     потратить случайность до создания лоадера. С явным сидом
                     поток шаффла отвязан от dropout и сэмплинга негативов,
                     поэтому одинаков между запусками.

    Returns:
        dict вида {'train': DataLoader, 'val': DataLoader, 'test': DataLoader},
        содержащий только запрошенные сплиты.
    """
    data_dir = Path(data_dir)
    files = {
        'train': data_dir / train_filename,
        'val':   data_dir / val_filename,
        'test':  data_dir / test_filename,
    }

    if padding_idx is None:
        statis = data_dir / 'data_statis.df'
        if not statis.exists():
            raise ValueError(f'padding_idx не задан, а {statis} не найден — укажите padding_idx явно')
        padding_idx = int(pd.read_pickle(statis)['item_num'].iloc[0])

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        if split not in files:
            raise ValueError(f'Unknown split {split!r}; expected one of {list(files)}.')
        path = files[split]
        if not path.exists():
            raise FileNotFoundError(f'Split file not found: {path}')

        dataset = ATGDataset(path, seq_len=seq_len, padding_idx=padding_idx)
        is_train = (split == 'train')

        # Свой генератор только для train: он же задаёт сиды воркерам.
        # val/test идут без шаффла, случайность там не нужна.
        generator = None
        if is_train and seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,   # для train выравнивает батчи; для val/test полная оценка
            persistent_workers=(num_workers > 0),
            generator=generator,
        )

    return loaders


def load_item_embeddings(path: str | Path) -> np.ndarray:
    """Текстовые эмбеддинги товаров из pickle → float32-массив (N_items, D)."""
    with open(path, 'rb') as f:
        emb = pickle.load(f)
    if not isinstance(emb, np.ndarray):
        raise ValueError(f'{path}: ожидался numpy-массив, получен {type(emb).__name__}')
    return emb.astype(np.float32)


def load_sasrec_item_embeddings(path: str | Path, n_items: int) -> np.ndarray:
    """Таблица товаров из чекпойнта SASRec (best.pt / latest.pt) → float32-массив (N_items, d_model)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['model_cfg']
    if cfg.get('model') != 'SASRec':
        raise ValueError(f'{path}: ожидался SASRec с собственной таблицей, получен {cfg.get("model")!r}')
    if cfg['n_items'] != n_items:
        raise ValueError(f'{path}: в чекпойнте {cfg["n_items"]} товаров, в конфиге {n_items}')
    return ckpt['model_state']['item_embedding.weight'][:n_items].float().numpy().astype(np.float32)


def shuffle_item_embeddings(emb: np.ndarray, seed: int) -> np.ndarray:
    """
    Строки эмбеддингов в случайной перестановке: товар i получает вектор товара perm[i].

    Абляция семантики: набор векторов (распределение, нормы, спектр) не меняется, рвётся
    только соответствие «товар ↔ его текст». Среднее и ковариация от порядка строк не зависят,
    поэтому PCA/whitening и их кэш те же. Свой генератор — перестановка зависит только от seed,
    а не от глобального RNG, и совпадает на обучении и оценке.
    """
    return emb[np.random.default_rng(seed).permutation(len(emb))]


# --------------------------------------------------------------------------- #
# Quick sanity-check entrypoint
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    Быстрая проверка (из корня проекта):
        python src/data.py [путь к каталогу со сплитами]
    Проверяет оба формата seq: переменной длины и фиксированной с паддингом слева.
    """
    import sys
    import tempfile

    # Синтетика: одни и те же истории в двух форматах должны дать одинаковые батчи
    PAD, SEQ_LEN = 100, 5
    rows = [{'seq': [1, 2, 3], 'len_seq': 3, 'next': 4},            # короче окна — паддинг слева
            {'seq': [1, 2, 3, 4, 5, 6, 7], 'len_seq': 7, 'next': 8}]  # длиннее окна — обрезка
    padded = [{'seq': [PAD] * 4 + r['seq'], 'len_seq': r['len_seq'], 'next': r['next']}
              if r['len_seq'] == 3 else r for r in rows]
    with tempfile.TemporaryDirectory() as tmp:
        out = {}
        for name, data in (('var', rows), ('pad', padded)):
            path = Path(tmp) / f'{name}.df'
            pd.DataFrame(data).to_pickle(path)
            ds = ATGDataset(path, seq_len=SEQ_LEN, padding_idx=PAD)
            out[name] = [ds[i] for i in range(len(ds))]
        assert out['var'][0]['history'].tolist() == [PAD, PAD, 1, 2, 3]
        assert out['var'][0]['mask'].tolist() == [0, 0, 1, 1, 1]
        assert out['var'][1]['history'].tolist() == [3, 4, 5, 6, 7], 'ожидались последние seq_len товаров'
        assert out['var'][1]['mask'].sum().item() == SEQ_LEN
        for a, b in zip(out['var'], out['pad']):
            assert torch.equal(a['history'], b['history']) and torch.equal(a['mask'], b['mask'])
        print('Оба формата seq дают одинаковые батчи  OK')

    # Перемешивание эмбеддингов: перестановка детерминирована, whitening-статистики те же,
    # а эмбеддинг товара i после whitening — исходный эмбеддинг товара perm[i]
    from whitening import apply_whitening, compute_whitening
    emb = np.random.default_rng(0).standard_normal((300, 24)).astype(np.float32)
    shuf = shuffle_item_embeddings(emb, seed=7)
    perm = np.random.default_rng(7).permutation(len(emb))
    assert np.array_equal(shuf, emb[perm]) and np.array_equal(shuf, shuffle_item_embeddings(emb, seed=7))
    assert not np.array_equal(shuf, emb)
    wh, wh_shuf = (compute_whitening(e.astype(np.float64), n_components=16, per_component=True)
                   for e in (emb, shuf))
    for a, b in zip(wh, wh_shuf):
        assert np.allclose(np.abs(a), np.abs(b), atol=1e-5)          # знак собственных векторов произволен
    out = apply_whitening(emb, *wh)
    assert np.allclose(apply_whitening(shuf, *wh), out[perm], atol=1e-5)
    print('Перемешивание эмбеддингов: whitening не меняется, строки переставлены  OK')

    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/processed/toys')
    if not data_dir.exists():
        print(f'Каталог со сплитами не найден: {data_dir}', file=sys.stderr)
        sys.exit(0)

    statis = pd.read_pickle(data_dir / 'data_statis.df')
    loaders = build_dataloaders(data_dir, batch_size=4, num_workers=0,
                                seq_len=int(statis['seq_size'].iloc[0]))
    for name, loader in loaders.items():
        batch = next(iter(loader))
        ds = loader.dataset
        print(f'\n=== {name} === примеров {len(ds):,}, длина истории: '
              f'min {ds.len_seq.min()}, медиана {int(np.median(ds.len_seq))}, max {ds.len_seq.max()}')
        for k, v in batch.items():
            print(f'  {k:7s}: shape={tuple(v.shape)}, dtype={v.dtype}')
        print(f'  пример: history={batch["history"][0].tolist()}, '
              f'target={batch["target"][0].item()}, mask_sum={batch["mask"][0].sum().item():.0f}')