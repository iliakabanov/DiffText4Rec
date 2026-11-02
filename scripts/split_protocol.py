"""
Сплит по протоколу отбора кандидатов (split_protocol.md): пользователи делятся 8:1:1, сплиты не пересекаются.

Вход — sequences.df из scripts/preprocess.py, фильтрация остаётся той, что там. Сплит пишется в отдельную
папку: файлы leave-one-out и обученные на них чекпойнты не затрагиваются.

    train: строка на пользователя, вся последовательность: seq = items[:-1], next = items[-1].
           Позиции истории — все t = 1, ..., n − 1; цель next-item или окно W считаются при обучении
           из самой последовательности, поэтому W не требует нового сплита;
    val, test: из последовательности удаляются товары вне словаря train, пользователи с n < MIN_EVAL_LEN
           исключаются; targets — последние EVAL_TARGET_LEN товаров без повторов, в порядке первого появления
           (множество F_u), seq — все товары до них; next = targets[0] нужен только формату ATGDataset.

Id товаров те же, что в items.parquet: строки матриц эмбеддингов подходят без пересчёта. Товары вне словаря
остаются в каталоге, но не встречаются в val и test. Исключать ли товары входа из выдачи — решение оценки,
поэтому targets хранит и повторы из seq.

    python scripts/split_protocol.py
    python scripts/split_protocol.py --seed 1 --out data/processed/toys_k15i5_protocol_seed1
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from preprocess import OUT_DIR, SPLIT_COLUMNS, SPLIT_FRACTIONS, SPLIT_SEED   # noqa: E402

DEFAULT_OUT       = OUT_DIR.parent / f'{OUT_DIR.name}_protocol'
EVAL_TARGET_LEN   = 5
MIN_EVAL_LEN      = EVAL_TARGET_LEN + 1      # хотя бы один товар на вход
EVAL_COLUMNS      = [*SPLIT_COLUMNS, 'targets']

log = logging.getLogger('split_protocol')


def split_users(n_users: int, seed: int) -> dict[str, np.ndarray]:
    """Номера строк sequences.df по сплитам: случайное разбиение в долях SPLIT_FRACTIONS, внутри — по возрастанию."""
    perm = np.random.default_rng(seed).permutation(n_users)
    bounds = np.cumsum([0] + [round(f * n_users) for f in SPLIT_FRACTIONS.values()])
    bounds[-1] = n_users
    return {name: np.sort(perm[lo:hi]) for name, lo, hi in zip(SPLIT_FRACTIONS, bounds[:-1], bounds[1:])}


def eval_rows(users: list[str], seqs: list[list[int]], vocab: set[int]) -> pd.DataFrame:
    rows = []
    for user, items in zip(users, seqs):
        known = [i for i in items if i in vocab]
        if len(known) < MIN_EVAL_LEN:
            continue
        c = len(known) - EVAL_TARGET_LEN
        targets = list(dict.fromkeys(known[c:]))
        rows.append((user, known[:c], c, targets[0], targets))
    return pd.DataFrame(rows, columns=EVAL_COLUMNS)


def build_splits(sequences: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    full = [list(s) for s in sequences['seq']]
    users = sequences['user_id'].tolist()
    parts = split_users(len(full), seed)

    train_seqs = [full[i] for i in parts['train']]
    if min(map(len, train_seqs)) < 2:
        raise ValueError('в train нужны последовательности хотя бы из 2 товаров: история и цель')
    train = pd.DataFrame([(users[i], full[i][:-1], len(full[i]) - 1, full[i][-1]) for i in parts['train']],
                         columns=SPLIT_COLUMNS)
    vocab = {i for s in train_seqs for i in s}

    splits = {'train': train}
    for name in ('val', 'test'):
        splits[name] = eval_rows([users[i] for i in parts[name]], [full[i] for i in parts[name]], vocab)
    return splits


def log_summary(sequences: pd.DataFrame, splits: dict[str, pd.DataFrame], seed: int, item_num: int) -> None:
    train = splits['train']
    vocab = {i for s in train['seq'] for i in s} | set(train['next'])
    log.info(f'Пользователей {len(sequences):,}, сид {seed}, доли {SPLIT_FRACTIONS}')
    log.info(f'Словарь train: {len(vocab):,} товаров из {item_num:,}')
    log.info(f'train: пользователей {len(train):,} | историй {train["len_seq"].sum():,} | '
             f'len_seq: медиана {train["len_seq"].median():g}, max {train["len_seq"].max()}')
    source = dict(zip(sequences['user_id'], sequences['len_seq']))
    for name in ('val', 'test'):
        s = splits[name]
        sizes = s['targets'].map(len)
        repeats = np.mean([len(set(t) & set(q)) / len(t) for q, t in zip(s['seq'], s['targets'])])
        kept = sum(len(q) + len(t) for q, t in zip(s['seq'], s['targets']))
        total = sum(source[u] for u in s['user_id'])
        log.info(f'{name}: пользователей {len(s):,} | len_seq: медиана {s["len_seq"].median():g}, max {s["len_seq"].max()} | '
                 f'|F_u|: медиана {sizes.median():g}, ≤2 у {(sizes <= 2).mean():.1%}, q90 {sizes.quantile(.9):g}, max {sizes.max()} | '
                 f'доля F_u во входе {repeats:.2%} | без повторов и товаров вне словаря осталось {kept / total:.2%} взаимодействий')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--sequences-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out / 'split.log', mode='w', encoding='utf-8')])

    sequences = pd.read_pickle(args.sequences_dir / 'sequences.df')
    item_num = int(pd.read_pickle(args.sequences_dir / 'data_statis.df')['item_num'].iloc[0])
    splits = build_splits(sequences, args.seed)
    log_summary(sequences, splits, args.seed, item_num)

    for name, split in splits.items():
        split.to_pickle(args.out / f'{name}_data.df')
    seq_size = max(int(s['len_seq'].max()) for s in splits.values())
    pd.DataFrame({'seq_size': [seq_size], 'item_num': [item_num]}).to_pickle(args.out / 'data_statis.df')
    log.info(f'Сохранено в {args.out}: train_data.df, val_data.df, test_data.df, data_statis.df (seq_size {seq_size})')


if __name__ == '__main__':
    main()
