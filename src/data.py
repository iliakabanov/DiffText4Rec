"""
Dataset и DataLoader для ATG (Amazon Toys & Games).

Сплиты приходят в формате pandas DataFrame, сохранённого через pickle,
с тремя колонками:
    - seq:     list[int] длины 10, индексы товаров (padding слева).
    - len_seq: int, число реальных взаимодействий в seq (от 1 до 10).
    - next:    int, индекс целевого товара (то, что надо предсказать).

Эмбеддинги товаров не загружаются здесь — они держатся в модели как
frozen nn.Embedding и индексируются по id из батча. Это эффективнее,
чем тащить тензоры эмбеддингов через DataLoader.
"""

from __future__ import annotations

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
    Один split (train / val / test) данных ATG.

    Args:
        path:        путь к .df файлу (pickle с pandas DataFrame).
        seq_len:     фиксированная длина истории. У ATG = 10. Если в данных
                     попадётся последовательность другой длины — упадём явно.
        seq_col:     имя колонки с историей (default 'seq').
        len_col:     имя колонки с реальной длиной (default 'len_seq').
        target_col:  имя колонки с таргетом (default 'next').

    Каждый __getitem__ возвращает dict:
        {
            'history': LongTensor[seq_len]  — индексы товаров (с padding слева),
            'target':  LongTensor[]         — индекс целевого товара,
            'mask':    FloatTensor[seq_len] — 1.0 для реальных позиций, 0.0 для padding.
        }

    Mask строится из len_seq. Padding слева, поэтому реальные взаимодействия —
    это последние `len_seq` позиций, а первые `seq_len - len_seq` — padding.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int = 10,
        seq_col: str = 'seq',
        len_col: str = 'len_seq',
        target_col: str = 'next',
    ):
        df = pd.read_pickle(path)
        self._validate(df, seq_col, len_col, target_col, seq_len)

        # Конвертируем в плотные numpy-массивы — это намного быстрее, чем
        # доставать значения из pandas в __getitem__.
        # df[seq_col].tolist() даёт List[List[int]], np.asarray превращает в (N, seq_len).
        self.history = np.asarray(df[seq_col].tolist(), dtype=np.int64)
        self.len_seq = df[len_col].to_numpy(dtype=np.int64)
        self.target = df[target_col].to_numpy(dtype=np.int64)

        self.seq_len = seq_len
        self.n = len(df)

        # Mask для каждой позиции: 1 если позиция реальная, 0 если padding.
        # Padding слева → реальные позиции это последние len_seq.
        # Эквивалентно: mask[i, j] = 1 если j >= seq_len - len_seq[i].
        positions = np.arange(seq_len)[None, :]               # (1, seq_len)
        threshold = (seq_len - self.len_seq)[:, None]          # (N, 1)
        self.mask = (positions >= threshold).astype(np.float32)  # (N, seq_len)

    @staticmethod
    def _validate(
        df: pd.DataFrame,
        seq_col: str,
        len_col: str,
        target_col: str,
        seq_len: int,
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

        # Проверим длину последовательностей на сэмпле первых 100 строк.
        sample = df[seq_col].iloc[:100]
        bad = [len(s) for s in sample if len(s) != seq_len]
        if bad:
            raise ValueError(
                f'Expected all sequences to have length {seq_len}, '
                f'but found rows with lengths {sorted(set(bad))}.'
            )

        # Проверим, что len_seq в разумных пределах.
        lens = df[len_col]
        if (lens < 1).any() or (lens > seq_len).any():
            raise ValueError(
                f'{len_col} must be in [1, {seq_len}], '
                f'found range [{lens.min()}, {lens.max()}].'
            )

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
    splits: Iterable[str] = ('train', 'val', 'test'),
    pin_memory: bool = True,
    train_filename: str = 'train_data.df',
    val_filename: str = 'val_data.df',
    test_filename: str = 'test_data.df',
) -> dict[str, DataLoader]:
    """
    Создаёт DataLoader'ы для всех или подмножества сплитов.

    Args:
        data_dir:    директория с файлами {train,val,test}_data.df.
        batch_size:  размер батча. Train shuffle, val/test без shuffle.
        num_workers: число воркеров для DataLoader.
        seq_len:     ожидаемая длина истории.
        splits:      какие сплиты загружать.
        pin_memory:  ускоряет перенос на GPU.

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

    loaders: dict[str, DataLoader] = {}
    for split in splits:
        if split not in files:
            raise ValueError(f'Unknown split {split!r}; expected one of {list(files)}.')
        path = files[split]
        if not path.exists():
            raise FileNotFoundError(f'Split file not found: {path}')

        dataset = ATGDataset(path, seq_len=seq_len)
        is_train = (split == 'train')
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,   # для train выравнивает батчи; для val/test полная оценка
            persistent_workers=(num_workers > 0),
        )

    return loaders


# --------------------------------------------------------------------------- #
# Quick sanity-check entrypoint
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    Прогоняем быстрый sanity check:
        python -m src.data
    Ожидает данные в ./data/atg/.
    """
    import sys

    data_dir = '../data/ATG'
    if not data_dir.exists():
        print(f'Data dir not found: {data_dir}', file=sys.stderr)
        sys.exit(1)

    loaders = build_dataloaders(data_dir, batch_size=4, num_workers=0)
    for name, loader in loaders.items():
        print(f'\n=== {name} ===')
        print(f'  dataset size: {len(loader.dataset):,}')
        batch = next(iter(loader))
        for k, v in batch.items():
            print(f'  {k:7s}: shape={tuple(v.shape)}, dtype={v.dtype}')
        # Первый пример из батча целиком
        print(f'  example: history={batch["history"][0].tolist()}, '
              f'target={batch["target"][0].item()}, '
              f'mask_sum={batch["mask"][0].sum().item():.0f}')