"""
Предобработка Amazon Reviews 2023 — Toys & Games:

    1. колонки user_id, parent_asin, timestamp (+ rating до фильтра по нему);
    2. дедупликация по DEDUP_SUBSET;
    3. фильтр по рейтингу: rating > RATING_THRESHOLD;
    4. товары с ≥ MIN_ITEM_INTERACTIONS взаимодействиями — популярность считается по всем данным
       после шага 3, до фильтра пользователей;
    5. пользователи с ≥ MIN_USER_INTERACTIONS взаимодействиями (после шага 4);
    6. у каждого пользователя — последние MAX_SEQ_LEN взаимодействий по времени;
    7. текстовые описания оставшихся товаров (src/item_text.py, параметры — TEXT);
    8. последовательности пользователей и сплит train / val / test в формате датасетов
       AlphaFuse / iDreamRec (data/ATG), способ — SPLIT.

Шаги 4–5 — по одному проходу, без итераций k-core: популярность — свойство товара, и мерится
она по всем покупателям, а не только по тем, кто остался в датасете. Поэтому у пользователей
гарантированно ≥ MIN_USER_INTERACTIONS взаимодействий, а у товара в итоговых данных их может быть
меньше MIN_ITEM_INTERACTIONS (часть покупателей — пользователи с короткой историей, убранные на шаге 5).

После каждого шага в лог пишется число строк (доля от предыдущего шага и от исходных),
пользователей и товаров.

Выход (OUT_DIR):
    interactions.parquet — user_id, parent_asin, timestamp;
    items.parquet        — parent_asin, text, text_raw; товары по убыванию числа взаимодействий.
                           Номер строки — id товара в сплитах и строка матрицы эмбеддингов
                           (scripts/embed_items.py);
    sequences.df         — полные последовательности без сплита (для анализа): строка — пользователь,
                           user_id, seq (длина MAX_SEQ_LEN, паддинг слева id = item_num), len_seq;
    {train,val,test}_data.df — сплит в формате data/ATG (+ столбец user_id): строка — один обучающий
                           или тестовый пример: seq (история, длина seq_size = MAX_SEQ_LEN − 1,
                           паддинг слева), len_seq (число настоящих товаров истории), next (таргет);
    data_statis.df       — seq_size (= MAX_SEQ_LEN − 1), item_num (= число товаров = id паддинга);
    preprocess.log.

Сплит (SPLIT):
    leave_one_out — последнее взаимодействие пользователя в test, предпоследнее в val, остальные
                    в train; пользователи во всех сплитах одни и те же. При TRAIN_PREFIXES в train
                    идут все префиксы истории (как в оригинальном DreamRec), иначе один пример на
                    пользователя — последний доступный для обучения таргет;
    users         — как в data/ATG: у пользователя один пример (вся история → последний товар),
                    сами пользователи делятся в долях SPLIT_FRACTIONS, сплиты не пересекаются
                    (протокол холодного старта по пользователям).

Источник — parquet-кэш из notebooks/toys_raw_structure.ipynb, если он построен по всему файлу;
иначе сырые JSONL читаются потоково (~15 минут).

    python scripts/preprocess.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from item_text import META_COLUMNS, TextConfig, build_item_texts, popularity   # noqa: E402

# --------------------------------------------------------------------------- #
# Параметры
# --------------------------------------------------------------------------- #

RAW_REVIEWS = ROOT / 'data/Toys/Toys_and_Games.jsonl'
RAW_META    = ROOT / 'data/Toys/meta_Toys_and_Games.jsonl'
CACHE_DIR   = ROOT / 'data/processed/toys_raw'           # кэш ноутбука toys_raw_structure
OUT_DIR     = ROOT / 'data/processed/toys'

# ('user_id', 'parent_asin', 'timestamp') — убрать только повторы одной и той же записи;
# ('user_id', 'parent_asin') — ещё и повторные отзывы на товар в другое время (остаётся самый ранний)
DEDUP_SUBSET          = ('user_id', 'parent_asin', 'timestamp')
RATING_THRESHOLD      = 3.0     # остаются отзывы с rating > RATING_THRESHOLD
MIN_ITEM_INTERACTIONS = 5       # товары с ≥ MIN_ITEM_INTERACTIONS взаимодействиями (по всем данным)
MIN_USER_INTERACTIONS = 5       # пользователи с ≥ MIN_USER_INTERACTIONS взаимодействиями
MAX_SEQ_LEN           = 12      # у пользователя остаются последние MAX_SEQ_LEN взаимодействий

SPLIT                 = 'leave_one_out'   # leave_one_out или users (см. docstring)
TRAIN_PREFIXES        = True    # leave_one_out: все префиксы истории в train, а не один пример на пользователя
SPLIT_FRACTIONS       = {'train': 0.8, 'val': 0.1, 'test': 0.1}   # только для SPLIT='users'
SPLIT_SEED            = 42

TEXT = TextConfig(
    feature_sep=' | ',
    max_features=5,
    max_features_chars=800,
    strip_symbols=True,
    strip_prefixes=True,
    fix_caps_headers=False,
)

REVIEW_COLUMNS = ['user_id', 'parent_asin', 'timestamp', 'rating']

log = logging.getLogger('preprocess')


# --------------------------------------------------------------------------- #
# Загрузка
# --------------------------------------------------------------------------- #

def _full_cache(name: str) -> Path | None:
    """Путь к кэшу <name>.parquet, если он построен по всему файлу (LIMIT = None)."""
    path, info = CACHE_DIR / f'{name}.parquet', CACHE_DIR / f'{name}_info.json'
    if path.exists() and info.exists() and json.loads(info.read_text())['limit'] is None:
        return path
    return None


def _read_jsonl(path: Path, columns: list[str]) -> pd.DataFrame:
    cols = {c: [] for c in columns}
    with open(path, 'rb') as f:
        for n, line in enumerate(f, 1):
            rec = json.loads(line)
            for c in columns:
                cols[c].append(rec.get(c))
            if n % 2_000_000 == 0:
                log.info(f'  прочитано {n:,} строк')
    return pd.DataFrame(cols)


def load_reviews() -> pd.DataFrame:
    """Колонки REVIEW_COLUMNS всех отзывов; идентификаторы — категориальные (экономит память)."""
    cache = _full_cache('reviews')
    if cache is not None:
        log.info(f'Отзывы: кэш {cache.relative_to(ROOT)}')
        return pq.read_table(cache, columns=REVIEW_COLUMNS,
                             read_dictionary=['user_id', 'parent_asin']).to_pandas()
    log.info(f'Отзывы: {RAW_REVIEWS.relative_to(ROOT)} (потоковое чтение)')
    df = _read_jsonl(RAW_REVIEWS, REVIEW_COLUMNS)
    return df.astype({'user_id': 'category', 'parent_asin': 'category'})


def load_meta(items: pd.Index) -> pd.DataFrame:
    """Колонки META_COLUMNS для товаров items в их порядке (товары без метаданных — NaN)."""
    cache = _full_cache('meta')
    if cache is not None:
        log.info(f'Метаданные: кэш {cache.relative_to(ROOT)}')
        meta = pq.read_table(cache, columns=META_COLUMNS).to_pandas()
    else:
        log.info(f'Метаданные: {RAW_META.relative_to(ROOT)} (потоковое чтение)')
        meta = _read_jsonl(RAW_META, META_COLUMNS)
    meta = meta.drop_duplicates('parent_asin').set_index('parent_asin').reindex(items)
    return meta.rename_axis('parent_asin').reset_index()


# --------------------------------------------------------------------------- #
# Шаги
# --------------------------------------------------------------------------- #

class StepLogger:
    """Пишет в лог размеры данных после шага относительно предыдущего шага и исходных."""

    def __init__(self):
        self.n_initial = None
        self.n_prev    = None

    def __call__(self, step: str, df: pd.DataFrame) -> None:
        n = len(df)
        if self.n_initial is None:
            self.n_initial = self.n_prev = n
        users = df['user_id'].nunique()
        items = df['parent_asin'].nunique()
        log.info(f'{step:<48s} строк {n:>11,} | {100 * n / self.n_prev:6.2f}% от пред. шага | '
                 f'{100 * n / self.n_initial:6.2f}% от исходных | '
                 f'пользователей {users:>9,} | товаров {items:>8,}')
        self.n_prev = n


def keep_frequent(df: pd.DataFrame, col: str, min_count: int) -> pd.DataFrame:
    """Строки, у которых значение col (категориальной) встречается в df ≥ min_count раз."""
    codes = df[col].cat.codes.to_numpy()
    return df[np.bincount(codes)[codes] >= min_count].reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    step = StepLogger()
    step('0. исходные отзывы', df)

    # Сортировка по времени (устойчивая) — дедуп оставляет самое раннее взаимодействие,
    # а обрезка берёт последние по времени
    df = df.sort_values(['user_id', 'timestamp'], kind='stable', ignore_index=True)

    df = df.drop_duplicates(list(DEDUP_SUBSET), keep='first', ignore_index=True)
    step(f'1. дедуп по {", ".join(DEDUP_SUBSET)}', df)

    df = df[df['rating'] > RATING_THRESHOLD].reset_index(drop=True)
    step(f'2. rating > {RATING_THRESHOLD:g}', df)

    df = keep_frequent(df, 'parent_asin', MIN_ITEM_INTERACTIONS)
    step(f'3. товары с ≥ {MIN_ITEM_INTERACTIONS} взаимодействиями', df)

    df = keep_frequent(df, 'user_id', MIN_USER_INTERACTIONS)
    step(f'4. пользователи с ≥ {MIN_USER_INTERACTIONS} взаимодействиями', df)

    from_end = df.groupby('user_id', observed=True).cumcount(ascending=False)
    df = df[from_end < MAX_SEQ_LEN].reset_index(drop=True)
    step(f'5. последние {MAX_SEQ_LEN} взаимодействий пользователя', df)

    df = df[['user_id', 'parent_asin', 'timestamp']]
    for c in ('user_id', 'parent_asin'):
        df[c] = df[c].cat.remove_unused_categories()

    # Порог по товарам считался по всем данным, поэтому в итоговых часть товаров реже него
    item_cnt = df['parent_asin'].value_counts()
    lens = df.groupby('user_id', observed=True).size()
    log.info(f'Длина последовательности: min {lens.min()}, медиана {lens.median():g}, '
             f'среднее {lens.mean():.2f}, max {lens.max()}; '
             f'доля пользователей с длиной {MAX_SEQ_LEN}: {(lens == MAX_SEQ_LEN).mean():.1%}')
    log.info(f'Взаимодействий на товар: медиана {item_cnt.median():g}, среднее {item_cnt.mean():.2f}; '
             f'реже {MIN_ITEM_INTERACTIONS}: {(item_cnt < MIN_ITEM_INTERACTIONS).mean():.1%} товаров, '
             f'ровно 1: {(item_cnt == 1).mean():.1%}')
    return df


def item_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Тексты товаров из df по убыванию числа взаимодействий (item_text.popularity)."""
    texts = build_item_texts(load_meta(popularity(df['parent_asin']).index), TEXT)

    n_empty = (texts['text'] == '').sum()
    lens = texts['text'].str.len()
    log.info(f'Тексты: {len(texts):,} товаров, пустых {n_empty:,}; длина text (символов): '
             f'медиана {lens.median():g}, 90% {lens.quantile(.9):g}, max {lens.max()}; '
             f'text ≠ text_raw у {(texts["text"] != texts["text_raw"]).mean():.1%}')
    return texts


def build_sequences(df: pd.DataFrame, items: pd.Series) -> pd.DataFrame:
    """
    Последовательности пользователей в формате data/ATG (без столбца next и без сплита).
    items — parent_asin в порядке items.parquet: позиция = id товара, id паддинга = len(items).
    """
    item_num = len(items)
    item_id = pd.Series(np.arange(item_num), index=items.to_numpy())
    ids = item_id.reindex(df['parent_asin'].astype(str)).to_numpy()
    if np.isnan(ids.astype(float)).any():
        raise ValueError('в interactions есть товары, которых нет в items')

    # df отсортирован по (user_id, timestamp), groupby сохраняет порядок строк внутри группы;
    # группы — по возрастанию user_id, так что строка sequences.df сопоставляется с interactions.parquet
    seqs = pd.Series(ids.astype(np.int64)).groupby(df['user_id'].astype(str).to_numpy()).agg(list)
    table = pd.DataFrame({
        'user_id': seqs.index.to_numpy(),
        'seq':     [[item_num] * (MAX_SEQ_LEN - len(s)) + s for s in seqs],
        'len_seq': seqs.map(len).to_numpy(),
    })

    seen = np.unique(np.concatenate(table['seq'].to_numpy()))
    log.info(f'Последовательности: пользователей {len(table):,} | взаимодействий {table["len_seq"].sum():,} | '
             f'товаров {(seen < item_num).sum():,} из {item_num:,} | len_seq: min {table["len_seq"].min()}, '
             f'медиана {table["len_seq"].median():g}, max {table["len_seq"].max()}')
    return table


SPLIT_COLUMNS = ['user_id', 'seq', 'len_seq', 'next']


def _example(user: str, history: list[int], target: int, seq_size: int, pad: int) -> tuple:
    """Один пример: история обрезается до последних seq_size товаров и паддится слева."""
    h = history[-seq_size:]
    return user, [pad] * (seq_size - len(h)) + h, len(h), target


def build_splits(sequences: pd.DataFrame, item_num: int) -> dict[str, pd.DataFrame]:
    """
    Сплит полных последовательностей (sequences.df) в формате data/ATG, способ — SPLIT.
    seq_size = MAX_SEQ_LEN − 1: один из MAX_SEQ_LEN товаров пользователя всегда уходит в next.
    """
    seq_size, pad = MAX_SEQ_LEN - 1, item_num
    full = [list(s[MAX_SEQ_LEN - n:]) for s, n in zip(sequences['seq'], sequences['len_seq'])]
    users = sequences['user_id'].tolist()

    if SPLIT == 'leave_one_out':
        # n = 3 не даёт train-примеров, n = 2 — пустую историю в val; MIN_USER_INTERACTIONS ≥ 4 это исключает
        short = min(map(len, full))
        if short < 4:
            raise ValueError(f'leave_one_out требует ≥ 4 взаимодействий у пользователя, найдено {short}; '
                             f'поднимите MIN_USER_INTERACTIONS')
        rows = {'train': [], 'val': [], 'test': []}
        for user, items in zip(users, full):
            n = len(items)
            rows['test'].append(_example(user, items[:n - 1], items[n - 1], seq_size, pad))
            rows['val'].append(_example(user, items[:n - 2], items[n - 2], seq_size, pad))
            targets = range(1, n - 2) if TRAIN_PREFIXES else [n - 3]
            for t in targets:
                rows['train'].append(_example(user, items[:t], items[t], seq_size, pad))
        splits = {name: pd.DataFrame(r, columns=SPLIT_COLUMNS) for name, r in rows.items()}
    elif SPLIT == 'users':
        table = pd.DataFrame([_example(u, items[:-1], items[-1], seq_size, pad)
                              for u, items in zip(users, full)], columns=SPLIT_COLUMNS)
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(table))
        bounds = np.cumsum([0] + [round(f * len(table)) for f in SPLIT_FRACTIONS.values()])
        bounds[-1] = len(table)
        splits = {name: table.iloc[np.sort(perm[lo:hi])].reset_index(drop=True)
                  for name, lo, hi in zip(SPLIT_FRACTIONS, bounds[:-1], bounds[1:])}
    else:
        raise ValueError(f"SPLIT должен быть 'leave_one_out' или 'users', получено {SPLIT!r}")

    train_items = set(np.concatenate(splits['train']['seq'].to_numpy()).tolist())
    train_items |= set(splits['train']['next'].tolist())
    train_items.discard(pad)
    for name, s in splits.items():
        log.info(f'Сплит {name:5s}: примеров {len(s):>9,} | пользователей {s["user_id"].nunique():>8,} | '
                 f'таргетов-товаров {s["next"].nunique():>8,} | len_seq: min {s["len_seq"].min()}, '
                 f'медиана {s["len_seq"].median():g}, max {s["len_seq"].max()}')
    for name in ('val', 'test'):
        cold = ~splits[name]['next'].isin(train_items)
        log.info(f'  {name}: таргет не встречается в train у {cold.mean():.1%} примеров '
                 f'({cold.sum():,}) — для ID-моделей это холодные товары')
    return splits


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(OUT_DIR / 'preprocess.log', mode='w', encoding='utf-8')],
    )
    log.info(f'DEDUP_SUBSET={DEDUP_SUBSET}, RATING_THRESHOLD={RATING_THRESHOLD}, '
             f'MIN_USER_INTERACTIONS={MIN_USER_INTERACTIONS}, MIN_ITEM_INTERACTIONS={MIN_ITEM_INTERACTIONS}, '
             f'MAX_SEQ_LEN={MAX_SEQ_LEN}, SPLIT={SPLIT!r}'
             + (f', TRAIN_PREFIXES={TRAIN_PREFIXES}' if SPLIT == 'leave_one_out'
                else f', SPLIT_FRACTIONS={SPLIT_FRACTIONS}, SPLIT_SEED={SPLIT_SEED}'))
    log.info(f'TEXT={TEXT}')

    df = preprocess(load_reviews())
    texts = item_texts(df)
    sequences = build_sequences(df, texts['parent_asin'])
    splits = build_splits(sequences, len(texts))

    out = OUT_DIR / 'interactions.parquet'
    pq.write_table(pa.Table.from_pandas(df.astype({'user_id': str, 'parent_asin': str}),
                                        preserve_index=False), out)
    log.info(f'Сохранено: {out.relative_to(ROOT)}')

    out = OUT_DIR / 'items.parquet'
    pq.write_table(pa.Table.from_pandas(texts, preserve_index=False), out)
    log.info(f'Сохранено: {out.relative_to(ROOT)}')

    sequences.to_pickle(OUT_DIR / 'sequences.df')
    for name, split in splits.items():
        split.to_pickle(OUT_DIR / f'{name}_data.df')
    seq_size = MAX_SEQ_LEN - 1
    pd.DataFrame({'seq_size': [seq_size], 'item_num': [len(texts)]}).to_pickle(OUT_DIR / 'data_statis.df')
    log.info(f'Сохранено: sequences.df, {", ".join(f"{n}_data.df" for n in splits)}, '
             f'data_statis.df (seq_size {seq_size}, item_num {len(texts):,})')


if __name__ == '__main__':
    main()
