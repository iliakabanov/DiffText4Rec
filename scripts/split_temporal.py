"""
Временной сплит на две части: train — всё до CUTOFF, val — после него. Для проверки идеи с новыми товарами;
потом его сменит сплит на три части (train / val / test).

Вход — interactions.parquet и items.parquet из scripts/preprocess.py, фильтрация остаётся той, что там.

    train: строка на пользователя, его взаимодействия до CUTOFF: seq = items[:-1], next = items[-1]
           (как в split_protocol: позиции истории — все t = 1, ..., n − 1). Нужно хотя бы 2 товара до CUTOFF;
    val:   пользователи с историей до CUTOFF и хотя бы одним товаром после. seq — вся история до CUTOFF,
           targets — первые EVAL_TARGET_LEN разных товаров после CUTOFF в порядке времени (F_u),
           next = targets[0] нужен только формату ATGDataset.

Id товаров — строки items.parquet, как в остальных сплитах: эмбеддинги подходят без пересчёта. Товары, которых
нет в train, из val не удаляются: на них и проверяется идея. Файла test_data.df нет, train.py это допускает.

    python scripts/split_temporal.py
    python scripts/split_temporal.py --cutoff 2022-10-01 --out data/processed/toys_temporal_2022-10
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

from preprocess import OUT_DIR, SPLIT_COLUMNS   # noqa: E402

CUTOFF          = '2023-01-01'
EVAL_TARGET_LEN = 5
DEFAULT_OUT     = OUT_DIR.parent / f'{OUT_DIR.name}_temporal'
EVAL_COLUMNS    = [*SPLIT_COLUMNS, 'targets']

log = logging.getLogger('split_temporal')


def load_events(data_dir: Path) -> pd.DataFrame:
    """user_id, item (id товара = строка items.parquet), timestamp (мс) — по пользователям и времени."""
    events = pd.read_parquet(data_dir / 'interactions.parquet')
    asins = pd.read_parquet(data_dir / 'items.parquet', columns=['parent_asin'])['parent_asin']
    item_id = pd.Series(np.arange(len(asins)), index=asins.to_numpy())
    events = events.sort_values(['user_id', 'timestamp'], kind='stable', ignore_index=True)
    return pd.DataFrame({'user_id': events['user_id'].astype(str).to_numpy(),
                         'item': item_id.reindex(events['parent_asin'].astype(str)).to_numpy(),
                         'timestamp': events['timestamp'].to_numpy()})


def build_splits(events: pd.DataFrame, cutoff: str) -> dict[str, pd.DataFrame]:
    cut = int(pd.Timestamp(cutoff).timestamp() * 1000)
    train_rows, val_rows = [], []
    for user, group in events.groupby('user_id', sort=True):
        items, ts = group['item'].tolist(), group['timestamp'].to_numpy()
        n_before = int((ts < cut).sum())                 # события отсортированы по времени
        before, after = items[:n_before], items[n_before:]
        if n_before >= 2:
            train_rows.append((user, before[:-1], n_before - 1, before[-1]))
        if n_before >= 1 and after:
            targets = list(dict.fromkeys(after))[:EVAL_TARGET_LEN]
            val_rows.append((user, before, n_before, targets[0], targets))
    return {'train': pd.DataFrame(train_rows, columns=SPLIT_COLUMNS),
            'val':   pd.DataFrame(val_rows, columns=EVAL_COLUMNS)}


def log_summary(events: pd.DataFrame, splits: dict[str, pd.DataFrame], cutoff: str, item_num: int) -> None:
    train, val = splits['train'], splits['val']
    vocab = {i for s in train['seq'] for i in s} | set(train['next'])
    after_users = events.loc[events['timestamp'] >= int(pd.Timestamp(cutoff).timestamp() * 1000), 'user_id'].nunique()
    sizes = val['targets'].map(len)
    new_share = val['targets'].map(lambda t: np.mean([i not in vocab for i in t]))
    new_items = {i for t in val['targets'] for i in t if i not in vocab}
    in_input = np.mean([len(set(t) & set(q)) / len(t) for q, t in zip(val['seq'], val['targets'])])
    log.info(f'Отсечение {cutoff}; взаимодействий {len(events):,}, пользователей {events["user_id"].nunique():,}')
    log.info(f'train: пользователей {len(train):,} | историй {train["len_seq"].sum():,} | len_seq: медиана '
             f'{train["len_seq"].median():g}, max {train["len_seq"].max()} | товаров {len(vocab):,} из {item_num:,}')
    log.info(f'val: пользователей {len(val):,} (после отсечения активны {after_users:,}, без истории до него '
             f'{after_users - len(val):,}) | len_seq: медиана {val["len_seq"].median():g}, max {val["len_seq"].max()}')
    log.info(f'  |F_u|: медиана {sizes.median():g}, = {EVAL_TARGET_LEN} у {(sizes == EVAL_TARGET_LEN).mean():.1%}, '
             f'= 1 у {(sizes == 1).mean():.1%} | доля F_u во входе {in_input:.2%}')
    log.info(f'  новые товары (нет в train): {new_share.mean():.1%} ответов в среднем по людям, хотя бы один у '
             f'{(new_share > 0).mean():.1%}, только новые у {(new_share == 1).mean():.1%}; разных новых {len(new_items):,}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cutoff', default=CUTOFF)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--data-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.out / 'split.log', mode='w', encoding='utf-8')])

    events = load_events(args.data_dir)
    item_num = int(pd.read_pickle(args.data_dir / 'data_statis.df')['item_num'].iloc[0])
    splits = build_splits(events, args.cutoff)
    log_summary(events, splits, args.cutoff, item_num)

    for name, split in splits.items():
        split.to_pickle(args.out / f'{name}_data.df')
    seq_size = max(int(s['len_seq'].max()) for s in splits.values())
    pd.DataFrame({'seq_size': [seq_size], 'item_num': [item_num]}).to_pickle(args.out / 'data_statis.df')
    log.info(f'Сохранено в {args.out}: train_data.df, val_data.df, data_statis.df (seq_size {seq_size})')


if __name__ == '__main__':
    main()
