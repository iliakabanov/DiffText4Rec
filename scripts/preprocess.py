"""
Предобработка Amazon Reviews 2023 — Toys & Games:

    1. колонки user_id, parent_asin, timestamp (+ rating до фильтра по нему);
    2. дедупликация по DEDUP_SUBSET;
    3. фильтр по рейтингу: rating > RATING_THRESHOLD;
    4–5. фильтр товаров и пользователей, способ — FILTER;
    6. у каждого пользователя — последние MAX_SEQ_LEN взаимодействий по времени (None — не обрезать);
    7. текстовые описания оставшихся товаров (src/item_text.py, параметры — TEXT);
    8. последовательности пользователей и сплит train / val / test, способ — SPLIT.

Паддинга в файлах нет: seq — список id товаров переменной длины по времени. Паддить до окна
модели — дело data.ATGDataset, id паддинга = item_num.

Фильтр (FILTER):
    global — по одному проходу: сначала товары с ≥ MIN_ITEM_INTERACTIONS взаимодействиями (считаются
             по всем данным после шага 3, до фильтра пользователей), затем пользователи с
             ≥ MIN_USER_INTERACTIONS. Популярность — свойство товара, и мерится она по всем
             покупателям, а не только по оставшимся в датасете. Поэтому у пользователей
             гарантированно ≥ MIN_USER_INTERACTIONS взаимодействий, а у товара в итоговых данных
             их может быть меньше MIN_ITEM_INTERACTIONS: часть его покупателей — пользователи с
             короткой историей, убранные фильтром пользователей;
    k_core — итеративно: удаление редких товаров снижает счётчики пользователей и наоборот, проход
             повторяется, пока ничего не меняется. В итоге пороги выполнены и по пользователям, и
             по товарам, но датасет заметно меньше: вместе с редкими уходят и популярные товары,
             чьи покупатели не прошли фильтр. Это стандарт в статьях (AlphaFuse и другие).


После каждого шага в лог пишется число строк (доля от предыдущего шага и от исходных),
пользователей и товаров, а в конце — итоговая сводка по датасету: пользователи, товары,
взаимодействия, средняя длина последовательности и разреженность.

Выход (OUT_DIR):
    interactions.parquet — user_id, parent_asin, timestamp;
    items.parquet        — parent_asin, text, text_raw; товары по убыванию числа взаимодействий.
                           Номер строки — id товара в сплитах и строка матрицы эмбеддингов
                           (scripts/embed_items.py);
    sequences.df         — полные последовательности без сплита (для анализа): строка — пользователь,
                           user_id, seq (id товаров по времени, без паддинга), len_seq;
    {train,val,test}_data.df — сплит (столбцы как в data/ATG плюс user_id): строка — один обучающий
                           или тестовый пример: seq (история, список переменной длины по времени),
                           len_seq (её длина), next (таргет);
    data_statis.df       — seq_size (самая длинная история в сплитах — справочно, для max_seq_len
                           модели), item_num (= число товаров = id паддинга);
    preprocess.log.

Сплит (SPLIT):
    leave_one_out — последнее взаимодействие пользователя в test, предпоследнее в val, остальные
                    в train; пользователи во всех сплитах одни и те же. При PREFIXES_SEPARATELY в train
                    идут все префиксы истории (как в оригинальном DreamRec), иначе (по умолчанию)
                    один пример на пользователя — последний доступный для обучения таргет;
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
MIN_ITEM_INTERACTIONS = 10       # товары с ≥ MIN_ITEM_INTERACTIONS взаимодействиями (по всем данным)
MIN_USER_INTERACTIONS = 10       # пользователи с ≥ MIN_USER_INTERACTIONS взаимодействиями
MAX_SEQ_LEN           = 50      # у пользователя остаются последние MAX_SEQ_LEN взаимодействий; None — все
FILTER                = 'k_core'   # global или k_core (см. docstring)

SPLIT                 = 'leave_one_out'   # leave_one_out или users (см. docstring)
PREFIXES_SEPARATELY   = False   # leave_one_out: каждый префикс истории — отдельная строка train,
                                # а не один пример на пользователя
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


def k_core(df: pd.DataFrame, min_user: int, min_item: int, max_iter: int = 200) -> pd.DataFrame:
    """Итеративно убирает пользователей с < min_user и товары с < min_item взаимодействиями."""
    for it in range(1, max_iter + 1):
        u = df['user_id'].cat.codes.to_numpy()
        i = df['parent_asin'].cat.codes.to_numpy()
        keep = (np.bincount(u)[u] >= min_user) & (np.bincount(i)[i] >= min_item)
        if keep.all():
            log.info(f'  k-core сошёлся за {it} итераций')
            return df
        df = df[keep].reset_index(drop=True)
        log.info(f'  итерация {it:2d}: строк {len(df):>11,} | пользователей {df["user_id"].nunique():>9,} | '
                 f'товаров {df["parent_asin"].nunique():>8,}')
    raise RuntimeError(f'k-core не сошёлся за {max_iter} итераций')


def filter_rare(df: pd.DataFrame, step: 'StepLogger') -> pd.DataFrame:
    """Шаги 4–5: фильтр товаров и пользователей способом FILTER."""
    if FILTER == 'global':
        df = keep_frequent(df, 'parent_asin', MIN_ITEM_INTERACTIONS)
        step(f'3. товары с ≥ {MIN_ITEM_INTERACTIONS} взаимодействиями', df)
        df = keep_frequent(df, 'user_id', MIN_USER_INTERACTIONS)
        step(f'4. пользователи с ≥ {MIN_USER_INTERACTIONS} взаимодействиями', df)
    elif FILTER == 'k_core':
        df = k_core(df, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)
        step(f'3–4. k-core: пользователи ≥ {MIN_USER_INTERACTIONS}, товары ≥ {MIN_ITEM_INTERACTIONS}', df)
    else:
        raise ValueError(f"FILTER должен быть 'global' или 'k_core', получено {FILTER!r}")
    return df


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

    df = filter_rare(df, step)

    if MAX_SEQ_LEN is not None:
        from_end = df.groupby('user_id', observed=True).cumcount(ascending=False)
        df = df[from_end < MAX_SEQ_LEN].reset_index(drop=True)
        step(f'5. последние {MAX_SEQ_LEN} взаимодействий пользователя', df)

    df = df[['user_id', 'parent_asin', 'timestamp']]
    for c in ('user_id', 'parent_asin'):
        df[c] = df[c].cat.remove_unused_categories()

    return df


def _quantiles(counts: pd.Series) -> str:
    q = counts.quantile([.25, .5, .75, .9])
    return (f'min {counts.min()}, q25 {q[.25]:g}, медиана {q[.5]:g}, q75 {q[.75]:g}, '
            f'q90 {q[.9]:g}, max {counts.max()}, среднее {counts.mean():.2f}')


def _top_share(counts: pd.Series, frac: float) -> float:
    """Доля взаимодействий, приходящаяся на top-frac самых популярных товаров."""
    k = max(1, int(round(frac * counts.size)))
    return counts.nlargest(k).sum() / counts.sum()


def log_dataset_summary(df: pd.DataFrame) -> None:
    """
    Итоговая сводка: размеры (как в Table 1 статей), распределения длин и популярности,
    перекос по товарам. Средние без квантилей прячут форму распределения, поэтому и то, и другое.
    """
    lens = df.groupby('user_id', observed=True).size()
    item_cnt = df['parent_asin'].value_counts()
    users, items, n = lens.size, item_cnt.size, len(df)

    log.info('--- Итог ---')
    log.info(f'Пользователей:  {users:>12,}')
    log.info(f'Товаров:        {items:>12,}')
    log.info(f'Взаимодействий: {n:>12,}')
    log.info(f'Средняя длина последовательности: {n / users:.2f}')
    log.info(f'Взаимодействий на товар в среднем: {n / items:.2f}')
    log.info(f'Разреженность: {1 - n / (users * items):.6%}')

    tail = (f'; упираются в MAX_SEQ_LEN={MAX_SEQ_LEN}: {(lens == MAX_SEQ_LEN).mean():.1%}'
            if MAX_SEQ_LEN is not None else f'; с длиной ≥ 20: {(lens >= 20).mean():.1%}')
    log.info(f'Длина последовательности: {_quantiles(lens)}{tail}')
    log.info(f'Популярность товара: {_quantiles(item_cnt)}; '
             f'реже порога {MIN_ITEM_INTERACTIONS}: {(item_cnt < MIN_ITEM_INTERACTIONS).mean():.1%}, '
             f'ровно 1: {(item_cnt == 1).mean():.1%}')
    log.info(f'Перекос: на top-1% товаров приходится {_top_share(item_cnt, 0.01):.1%} взаимодействий, '
             f'на top-20% — {_top_share(item_cnt, 0.20):.1%} '
             f'(равномерно было бы 1% и 20%)')


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
    Полные последовательности пользователей: seq — список id товаров по времени, без обрезки
    и паддинга. items — parent_asin в порядке items.parquet: позиция = id товара.
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
        'seq':     seqs.to_numpy(),
        'len_seq': seqs.map(len).to_numpy(),
    })

    seen = np.unique(np.concatenate(table['seq'].to_numpy()))
    log.info(f'Последовательности: пользователей {len(table):,} | взаимодействий {table["len_seq"].sum():,} | '
             f'товаров {seen.size:,} из {item_num:,} | len_seq: min {table["len_seq"].min()}, '
             f'медиана {table["len_seq"].median():g}, max {table["len_seq"].max()}')
    return table


SPLIT_COLUMNS = ['user_id', 'seq', 'len_seq', 'next']


def _example(user: str, history: list[int], target: int) -> tuple:
    """Один пример: история целиком (без обрезки и паддинга) и таргет."""
    return user, list(history), len(history), target


def build_splits(sequences: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Сплит полных последовательностей (sequences.df), способ — SPLIT."""
    full = [list(s) for s in sequences['seq']]
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
            rows['test'].append(_example(user, items[:n - 1], items[n - 1]))
            rows['val'].append(_example(user, items[:n - 2], items[n - 2]))
            targets = range(1, n - 2) if PREFIXES_SEPARATELY else [n - 3]
            for t in targets:
                rows['train'].append(_example(user, items[:t], items[t]))
        splits = {name: pd.DataFrame(r, columns=SPLIT_COLUMNS) for name, r in rows.items()}
    elif SPLIT == 'users':
        table = pd.DataFrame([_example(u, items[:-1], items[-1])
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
             f'MAX_SEQ_LEN={MAX_SEQ_LEN}, FILTER={FILTER!r}, SPLIT={SPLIT!r}'
             + (f', PREFIXES_SEPARATELY={PREFIXES_SEPARATELY}' if SPLIT == 'leave_one_out'
                else f', SPLIT_FRACTIONS={SPLIT_FRACTIONS}, SPLIT_SEED={SPLIT_SEED}'))
    log.info(f'TEXT={TEXT}')

    df = preprocess(load_reviews())
    texts = item_texts(df)
    sequences = build_sequences(df, texts['parent_asin'])
    splits = build_splits(sequences)

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
    # seq_size — самая длинная история в сплитах: справочно, окно модели задаётся её конфигом
    seq_size = max(int(s['len_seq'].max()) for s in splits.values())
    pd.DataFrame({'seq_size': [seq_size], 'item_num': [len(texts)]}).to_pickle(OUT_DIR / 'data_statis.df')
    log.info(f'Сохранено: sequences.df, {", ".join(f"{n}_data.df" for n in splits)}, '
             f'data_statis.df (seq_size {seq_size}, item_num {len(texts):,})')

    log_dataset_summary(df)


if __name__ == '__main__':
    main()
