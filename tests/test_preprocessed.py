"""
Проверка результата scripts/preprocess.py на соответствие сырому датасету.

Две группы тестов.

Внутренняя согласованность файлов OUT_DIR (быстро, все данные целиком):
    data_statis.df ↔ items.parquet ↔ sequences.df ↔ interactions.parquet: формат, паддинг,
    нумерация товаров, порядок, отсутствие дубликатов, совпадение последовательностей с
    interactions; эмбеддинги (если посчитаны) — по строке на товар.

Сверка с сырыми JSONL (маркер raw; один проход по ~10 ГБ, около минуты):
    ожидаемые значения считаются заново прямо по сырым строкам — без кэша ноутбука и без кода
    preprocess.py, только его пороги. Для выборки:
      * пользователей из sequences.df — seq совпадает с последними MAX_SEQ_LEN товарами их сырой
        истории (дедуп, rating > порога, только товары из items.parquet, по времени);
      * пользователей, которых нет в sequences.df, — у них действительно < MIN_USER_INTERACTIONS
        подходящих взаимодействий;
      * товаров из items.parquet — у них ≥ MIN_ITEM_INTERACTIONS положительных отзывов во всех данных;
      * товаров из items.parquet — text и text_raw совпадают с текстом из сырой записи метаданных.

Сплиты {train,val,test}_data.df проверяются на формат и на соответствие полным последовательностям
(своё для каждого SPLIT), плюс build_splits прогоняется на синтетических данных в оба режима.

    pytest tests/test_preprocessed.py              # всё
    pytest tests/test_preprocessed.py -m "not raw" # только согласованность файлов
"""
import json
import pickle
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

import preprocess as P                       # пороги и пути — те же, что при предобработке
from item_text import cleaned_text, popularity, raw_text

SEED           = 0
N_USERS        = 300        # пользователей из sequences.df
EXCLUDED_EVERY = 20_000     # пользователь вне sequences.df проверяется, если crc32(user_id) % это == 0
N_ITEMS        = 300        # товаров из items.parquet

OUT = P.OUT_DIR
SPLIT_NAMES = ('train', 'val', 'test')
FILES = ('data_statis.df', 'items.parquet', 'sequences.df', 'interactions.parquet',
         *(f'{n}_data.df' for n in SPLIT_NAMES))

pytestmark = pytest.mark.skipif(not all((OUT / f).exists() for f in FILES),
                                reason=f'нет результатов preprocess.py в {OUT}')


# --------------------------------------------------------------------------- #
# Файлы OUT_DIR
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def statis() -> dict:
    s = pd.read_pickle(OUT / 'data_statis.df')
    return {'item_num': int(s['item_num'][0]), 'seq_size': int(s['seq_size'][0])}


@pytest.fixture(scope='module')
def items() -> pd.DataFrame:
    return pd.read_parquet(OUT / 'items.parquet')


@pytest.fixture(scope='module')
def seqs() -> pd.DataFrame:
    return pd.read_pickle(OUT / 'sequences.df')


@pytest.fixture(scope='module')
def inter() -> pd.DataFrame:
    return pd.read_parquet(OUT / 'interactions.parquet')


@pytest.fixture(scope='module')
def seq_array(seqs) -> np.ndarray:
    return np.array(seqs['seq'].tolist())


@pytest.fixture(scope='module')
def splits() -> dict[str, pd.DataFrame]:
    return {n: pd.read_pickle(OUT / f'{n}_data.df') for n in SPLIT_NAMES}


@pytest.fixture(scope='module')
def full_sequences(seqs, seq_array) -> dict[str, list[int]]:
    """user_id → полная последовательность id товаров без паддинга (из sequences.df)."""
    return {u: q[len(q) - n:].tolist()
            for u, q, n in zip(seqs['user_id'], seq_array, seqs['len_seq'])}


def test_statis(statis, items):
    assert statis['item_num'] == len(items)
    assert statis['seq_size'] == P.MAX_SEQ_LEN - 1, 'один товар всегда уходит в next'


def test_items_unique(items):
    assert items['parent_asin'].is_unique


def test_items_order_is_popularity(items, inter):
    """Строка items.parquet — id товара; порядок: по убыванию популярности, ничьи по parent_asin."""
    assert popularity(inter['parent_asin']).index.tolist() == items['parent_asin'].tolist()


def test_items_texts_nonempty(items):
    assert (items['text'] == '').sum() <= 1


def test_interactions_no_duplicates(inter):
    assert not inter.duplicated(list(P.DEDUP_SUBSET)).any()


def test_interactions_items_known(inter, items):
    assert inter['parent_asin'].isin(items['parent_asin']).all()


def test_sequences_columns(seqs):
    assert list(seqs.columns) == ['user_id', 'seq', 'len_seq']


def test_sequences_shape(seq_array, seqs):
    assert seq_array.shape == (len(seqs), P.MAX_SEQ_LEN)


def test_sequences_len_bounds(seqs):
    lens = seqs['len_seq']
    assert lens.min() >= P.MIN_USER_INTERACTIONS
    assert lens.max() <= P.MAX_SEQ_LEN


def test_sequences_left_padding(seq_array, seqs, statis):
    size, lens = P.MAX_SEQ_LEN, seqs['len_seq'].to_numpy()
    pad = seq_array == statis['item_num']
    assert (pad == (np.arange(size) < (size - lens)[:, None])).all()


def test_sequences_item_ids(seq_array, statis):
    item_num = statis['item_num']
    assert seq_array.min() >= 0 and seq_array.max() <= item_num
    assert np.unique(seq_array[seq_array != item_num]).size == item_num, 'встречаются не все товары'


def test_sequences_users_sorted_unique(seqs):
    assert seqs['user_id'].is_unique
    assert seqs['user_id'].is_monotonic_increasing


def test_sequences_total_equals_interactions(seqs, inter):
    assert seqs['len_seq'].sum() == len(inter)


def test_sequences_match_interactions(seqs, seq_array, inter, items):
    """Все последовательности: interactions по (user_id, timestamp), переведённые в id товаров."""
    size = P.MAX_SEQ_LEN
    item_id = pd.Series(np.arange(len(items)), index=items['parent_asin'])
    s = inter.sort_values(['user_id', 'timestamp'], kind='stable')
    rebuilt = pd.Series(item_id[s['parent_asin']].to_numpy()).groupby(s['user_id'].to_numpy()).agg(list)
    assert rebuilt.index.tolist() == seqs['user_id'].tolist()
    bad = [u for u, r, q, n in zip(seqs['user_id'], rebuilt, seq_array, seqs['len_seq'])
           if r != q[size - n:].tolist()]
    assert not bad, f'расхождений {len(bad)}, первые: {bad[:5]}'


# --------------------------------------------------------------------------- #
# Сплиты {train,val,test}_data.df
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', SPLIT_NAMES)
def test_split_format(name, splits, statis):
    """Столбцы, длина истории, паддинг слева, допустимые id."""
    s = splits[name]
    size, item_num = statis['seq_size'], statis['item_num']
    assert list(s.columns) == P.SPLIT_COLUMNS
    arr, lens = np.array(s['seq'].tolist()), s['len_seq'].to_numpy()
    assert arr.shape == (len(s), size)
    assert lens.min() >= 1 and lens.max() <= size
    pad = arr == item_num
    assert (pad == (np.arange(size) < (size - lens)[:, None])).all(), 'паддинг не слева'
    assert arr.min() >= 0 and arr.max() <= item_num
    assert s['next'].between(0, item_num - 1).all(), 'таргет вне каталога или паддинг'
    assert s['user_id'].isin(splits['test']['user_id']).all()


@pytest.mark.skipif(P.SPLIT != 'leave_one_out', reason=f'SPLIT={P.SPLIT}')
def test_split_leave_one_out(splits, full_sequences):
    """test — последнее взаимодействие, val — предпоследнее, train — префиксы до них."""
    for name, drop in (('test', 0), ('val', 1)):
        s = splits[name]
        assert len(s) == len(full_sequences), 'в val/test должен быть каждый пользователь'
        bad = [u for u, seq, n, nxt in zip(s['user_id'], s['seq'], s['len_seq'], s['next'])
               if list(seq[len(seq) - n:]) + [nxt] != full_sequences[u][:len(full_sequences[u]) - drop]]
        assert not bad, f'{name}: расхождений {len(bad)}, первые: {bad[:3]}'

    train = splits['train']
    expected = {u: (len(items) - 3 if P.TRAIN_PREFIXES else 1) for u, items in full_sequences.items()}
    assert train['user_id'].value_counts().to_dict() == expected, 'не столько примеров на пользователя'
    bad = [u for u, seq, n, nxt in zip(train['user_id'], train['seq'], train['len_seq'], train['next'])
           if list(seq[len(seq) - n:]) + [nxt] != full_sequences[u][:n + 1]]
    assert not bad, f'train: расхождений {len(bad)}, первые: {bad[:3]}'
    # Таргеты train — строго раньше val и test того же пользователя
    last_train = train.groupby('user_id')['len_seq'].max().add(1)
    assert (last_train <= pd.Series({u: len(v) - 2 for u, v in full_sequences.items()})).all()


@pytest.mark.skipif(P.SPLIT != 'users', reason=f'SPLIT={P.SPLIT}')
def test_split_users(splits, full_sequences):
    """Один пример на пользователя, пользователи сплитов не пересекаются."""
    users = {name: set(s['user_id']) for name, s in splits.items()}
    assert sum(map(len, users.values())) == len(full_sequences)
    assert set.union(*users.values()) == set(full_sequences)
    assert not (users['train'] & users['val']) and not (users['train'] & users['test'])
    assert not (users['val'] & users['test'])
    for name, s in splits.items():
        assert s['user_id'].is_unique
        frac = len(s) / len(full_sequences)
        assert abs(frac - P.SPLIT_FRACTIONS[name]) < 0.01, f'{name}: доля {frac:.3f}'
        bad = [u for u, seq, n, nxt in zip(s['user_id'], s['seq'], s['len_seq'], s['next'])
               if list(seq[len(seq) - n:]) + [nxt] != full_sequences[u]]
        assert not bad, f'{name}: расхождений {len(bad)}'


PAD = 6      # item_num в синтетических примерах


def _synthetic(monkeypatch, split_mode: str, lens=(5, 4, 4, 4)) -> pd.DataFrame:
    """Четыре пользователя, MAX_SEQ_LEN = 5, каталог из PAD товаров."""
    monkeypatch.setattr(P, 'SPLIT', split_mode)
    monkeypatch.setattr(P, 'MAX_SEQ_LEN', 5)
    monkeypatch.setattr(P, 'TRAIN_PREFIXES', True)
    monkeypatch.setattr(P, 'SPLIT_FRACTIONS', {'train': 0.5, 'val': 0.25, 'test': 0.25})
    items = {5: [0, 1, 2, 3, 4], 4: [4, 0, 2, 1], 3: [1, 3, 4], 2: [2, 0]}
    return pd.DataFrame({
        'user_id': [f'u{k + 1}' for k in range(len(lens))],
        'seq':     [[PAD] * (5 - n) + items[n] for n in lens],
        'len_seq': list(lens),
    })


def _rows(split: pd.DataFrame) -> list[tuple]:
    return [(u, list(s), int(n), int(x)) for u, s, n, x
            in zip(split['user_id'], split['seq'], split['len_seq'], split['next'])]


def test_build_splits_synthetic_leave_one_out(monkeypatch):
    """u1 = [0,1,2,3,4]: train [0]→1 и [0,1]→2, val [0,1,2]→3, test [0,1,2,3]→4."""
    splits = P.build_splits(_synthetic(monkeypatch, 'leave_one_out', lens=(5, 4)), PAD)
    assert _rows(splits['train']) == [('u1', [PAD, PAD, PAD, 0], 1, 1),
                                      ('u1', [PAD, PAD, 0, 1], 2, 2),
                                      ('u2', [PAD, PAD, PAD, 4], 1, 0)]
    assert _rows(splits['val']) == [('u1', [PAD, 0, 1, 2], 3, 3),
                                    ('u2', [PAD, PAD, 4, 0], 2, 2)]
    assert _rows(splits['test']) == [('u1', [0, 1, 2, 3], 4, 4),
                                     ('u2', [PAD, 4, 0, 2], 3, 1)]


def test_build_splits_single_train_example(monkeypatch):
    """Без TRAIN_PREFIXES — по одному примеру на пользователя: последний доступный таргет."""
    sequences = _synthetic(monkeypatch, 'leave_one_out', lens=(5, 4))
    monkeypatch.setattr(P, 'TRAIN_PREFIXES', False)
    splits = P.build_splits(sequences, PAD)
    assert _rows(splits['train']) == [('u1', [PAD, PAD, 0, 1], 2, 2),
                                      ('u2', [PAD, PAD, PAD, 4], 1, 0)]


@pytest.mark.parametrize('lens', [(5, 3), (5, 2)])
def test_build_splits_rejects_short_sequences(monkeypatch, lens):
    """При < 4 взаимодействиях leave_one_out невозможен: нет train-примеров или пустая история."""
    sequences = _synthetic(monkeypatch, 'leave_one_out', lens=lens)
    with pytest.raises(ValueError, match='leave_one_out'):
        P.build_splits(sequences, PAD)


def test_build_splits_synthetic_users(monkeypatch):
    """Режим users: один пример на пользователя (вся история → последний товар), сплиты не пересекаются."""
    splits = P.build_splits(_synthetic(monkeypatch, 'users'), PAD)
    assert [len(s) for s in splits.values()] == [2, 1, 1]
    users = {u for s in splits.values() for u in s['user_id']}
    assert users == {'u1', 'u2', 'u3', 'u4'}
    everything = {r[0]: r for s in splits.values() for r in _rows(s)}
    assert everything['u1'] == ('u1', [0, 1, 2, 3], 4, 4)        # последний товар 4 — в next
    assert everything['u2'] == ('u2', [PAD, 4, 0, 2], 3, 1)


# --------------------------------------------------------------------------- #
# Эмбеддинги
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('path', sorted(OUT.glob('*_text*.pickle')), ids=lambda p: p.name)
def test_embeddings_row_per_item(path, statis):
    with open(path, 'rb') as f:
        emb = pickle.load(f)
    assert emb.shape[0] == statis['item_num']


# --------------------------------------------------------------------------- #
# Сверка с сырыми JSONL
# --------------------------------------------------------------------------- #

def _field(line: bytes, key: bytes) -> bytes | None:
    """Строковое значение поля верхнего уровня без json.loads (для быстрого отбора строк)."""
    i = line.rfind(b'"' + key + b'": "')
    if i < 0:
        return None
    i += len(key) + 5
    return line[i:line.index(b'"', i)]


def positive(rows: list[tuple[int, dict]]) -> list[dict]:
    """
    Сырые отзывы (номер строки, запись) → положительные после дедупа, как в preprocess.py:
    дедуп по DEDUP_SUBSET (остаётся первая запись в порядке файла), затем rating > порога.
    Порядок: по timestamp, при равенстве — по порядку в файле (устойчивая сортировка).
    """
    seen, out = set(), []
    for _, r in sorted(rows, key=lambda x: (x[1]['timestamp'], x[0])):
        key = tuple(r[k] for k in P.DEDUP_SUBSET)
        if key in seen:
            continue
        seen.add(key)
        if r['rating'] > P.RATING_THRESHOLD:
            out.append(r)
    return out


def expected_history(rows: list[tuple[int, dict]], keep_items: set[str]) -> list[str]:
    """parent_asin положительных отзывов пользователя по времени, только товары keep_items."""
    return [r['parent_asin'] for r in positive(rows) if r['parent_asin'] in keep_items]


@pytest.fixture(scope='module')
def sample(seqs, items) -> dict:
    rng = np.random.default_rng(SEED)
    return {'users': seqs.iloc[np.sort(rng.choice(len(seqs), N_USERS, replace=False))],
            'items': set(rng.choice(items['parent_asin'].to_numpy(), N_ITEMS, replace=False))}


@pytest.fixture(scope='module')
def raw_reviews(sample, seqs) -> dict:
    """Один проход по сырым отзывам: строки выбранных пользователей и отзывы выбранных товаров."""
    users_b = {u.encode() for u in sample['users']['user_id']}
    known_b = {u.encode() for u in seqs['user_id']}
    items_b = {a.encode() for a in sample['items']}
    rows = {'users': defaultdict(list), 'excluded': defaultdict(list), 'items': defaultdict(list)}
    with open(P.RAW_REVIEWS, 'rb') as f:
        for n, line in enumerate(f):
            u, a = _field(line, b'user_id'), _field(line, b'parent_asin')
            if u in users_b:
                rows['users'][u.decode()].append((n, json.loads(line)))
            elif u is not None and u not in known_b and zlib.crc32(u) % EXCLUDED_EVERY == 0:
                rows['excluded'][u.decode()].append((n, json.loads(line)))
            if a in items_b:
                rows['items'][a.decode()].append((n, json.loads(line)))
    return rows


@pytest.fixture(scope='module')
def raw_meta(sample) -> dict:
    items_b = {a.encode() for a in sample['items']}
    meta = {}
    with open(P.RAW_META, 'rb') as f:
        for line in f:
            a = _field(line, b'parent_asin')
            if a in items_b:
                meta[a.decode()] = json.loads(line)
    return meta


@pytest.mark.raw
def test_raw_user_sequences(sample, raw_reviews, items):
    """seq пользователя = последние MAX_SEQ_LEN товаров его сырой истории."""
    keep, asin = set(items['parent_asin']), items['parent_asin'].to_numpy()
    users = sample['users']
    bad = []
    for u, seq, n in zip(users['user_id'], users['seq'], users['len_seq']):
        expected = expected_history(raw_reviews['users'][u], keep)[-P.MAX_SEQ_LEN:]
        got = [asin[i] for i in seq[len(seq) - n:]]
        if got != expected:
            bad.append((u, got, expected))
    assert not bad, f'расхождений {len(bad)} из {len(users)}, первое: {bad[0]}'


@pytest.mark.raw
def test_raw_excluded_users(raw_reviews, items):
    """Пользователи вне датасета: < MIN_USER_INTERACTIONS подходящих взаимодействий."""
    keep = set(items['parent_asin'])
    counts = {u: len(expected_history(r, keep)) for u, r in raw_reviews['excluded'].items()}
    assert counts, 'в выборку не попал ни один пользователь вне датасета'
    too_many = {u: c for u, c in counts.items() if c >= P.MIN_USER_INTERACTIONS}
    assert not too_many


@pytest.mark.raw
def test_raw_item_popularity(sample, raw_reviews):
    """Товары датасета: ≥ MIN_ITEM_INTERACTIONS положительных отзывов во всех данных."""
    counts = {a: len(positive(raw_reviews['items'][a])) for a in sample['items']}
    rare = {a: c for a, c in counts.items() if c < P.MIN_ITEM_INTERACTIONS}
    assert not rare


@pytest.mark.raw
def test_raw_item_texts(sample, raw_meta, items):
    """text и text_raw = тексту, собранному из сырой записи метаданных."""
    assert set(raw_meta) == sample['items'], 'не все товары найдены в метаданных'
    by_asin = items.set_index('parent_asin')
    fields = ('title', 'store', 'features', 'categories')
    bad = [a for a, r in raw_meta.items()
           if by_asin.at[a, 'text_raw'] != raw_text(*(r.get(k) for k in fields), P.TEXT)
           or by_asin.at[a, 'text'] != cleaned_text(*(r.get(k) for k in fields), P.TEXT)]
    assert not bad, f'расхождения: {bad[:5]}'
