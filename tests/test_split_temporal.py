"""
Проверка scripts/split_temporal.py: логика на синтетике и, если сплит уже построен, файлы против interactions.

    pytest tests/test_split_temporal.py
"""
import pandas as pd
import pytest

import split_temporal as S
from data import ATGDataset

DAY = 86_400_000
CUT = pd.Timestamp('2023-01-01')
T0 = int(CUT.timestamp() * 1000)


def _events(rows: list[tuple[str, int, int]]) -> pd.DataFrame:
    """(пользователь, товар, день относительно отсечения)."""
    return pd.DataFrame({'user_id': [r[0] for r in rows], 'item': [r[1] for r in rows],
                         'timestamp': [T0 + r[2] * DAY for r in rows]})


# --------------------------------------------------------------------------- #
# Логика на синтетике
# --------------------------------------------------------------------------- #

def test_train_and_val_of_one_user():
    ev = _events([('u', 1, -3), ('u', 2, -2), ('u', 3, -1), ('u', 4, 1), ('u', 5, 2)])
    s = S.build_splits(ev, '2023-01-01')
    assert s['train'].iloc[0][['seq', 'len_seq', 'next']].tolist() == [[1, 2], 2, 3]
    assert s['val'].iloc[0][['seq', 'len_seq', 'next', 'targets']].tolist() == [[1, 2, 3], 3, 4, [4, 5]]


def test_val_takes_first_five_distinct_after_cutoff():
    ev = _events([('u', 1, -1)] + [('u', i, d) for d, i in enumerate([7, 8, 7, 9, 10, 11, 12], start=1)])
    assert S.build_splits(ev, '2023-01-01')['val'].iloc[0]['targets'] == [7, 8, 9, 10, 11]


def test_event_exactly_at_cutoff_goes_to_val():
    ev = _events([('u', 1, -1), ('u', 2, -1), ('u', 3, 0)])
    s = S.build_splits(ev, '2023-01-01')
    assert s['train'].iloc[0]['seq'] + [s['train'].iloc[0]['next']] == [1, 2]
    assert s['val'].iloc[0]['targets'] == [3]


def test_who_gets_rows():
    """Один товар до отсечения — нет train, но есть val; без истории — нет val; без будущего — нет val."""
    ev = _events([('a', 1, -1), ('a', 2, 1),
                  ('b', 3, 1), ('b', 4, 2),
                  ('c', 5, -2), ('c', 6, -1)])
    s = S.build_splits(ev, '2023-01-01')
    assert s['train']['user_id'].tolist() == ['c']
    assert s['val']['user_id'].tolist() == ['a']


def test_new_items_stay_in_val():
    ev = _events([('a', 1, -2), ('a', 2, -1), ('b', 1, -1), ('b', 99, 1)])
    s = S.build_splits(ev, '2023-01-01')
    assert s['val'].iloc[0]['targets'] == [99]


# --------------------------------------------------------------------------- #
# Построенный сплит
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def built() -> dict[str, pd.DataFrame]:
    if not all((S.DEFAULT_OUT / f'{n}_data.df').exists() for n in ('train', 'val')):
        pytest.skip(f'нет сплита в {S.DEFAULT_OUT}')
    return {n: pd.read_pickle(S.DEFAULT_OUT / f'{n}_data.df') for n in ('train', 'val')}


@pytest.fixture(scope='module')
def timeline() -> dict[str, tuple[list[int], list[int]]]:
    ev = S.load_events(S.OUT_DIR)
    return {u: (g['item'].tolist(), g['timestamp'].tolist()) for u, g in ev.groupby('user_id')}


def test_events_match_sequences():
    """Те же последовательности, что в sequences.df: та же сортировка и нумерация товаров."""
    ev = S.load_events(S.OUT_DIR)
    seqs = pd.read_pickle(S.OUT_DIR / 'sequences.df')
    rebuilt = ev.groupby('user_id')['item'].agg(list)
    assert rebuilt.index.tolist() == seqs['user_id'].tolist()
    assert all(a == list(b) for a, b in zip(rebuilt, seqs['seq']))


def test_built_matches_timeline(built, timeline):
    cut = int(pd.Timestamp(S.CUTOFF).timestamp() * 1000)
    for u, seq, nxt in zip(built['train']['user_id'], built['train']['seq'], built['train']['next']):
        items, ts = timeline[u]
        assert seq + [nxt] == [i for i, t in zip(items, ts) if t < cut]
    for u, seq, targets in zip(built['val']['user_id'], built['val']['seq'], built['val']['targets']):
        items, ts = timeline[u]
        assert seq == [i for i, t in zip(items, ts) if t < cut]
        assert targets == list(dict.fromkeys(i for i, t in zip(items, ts) if t >= cut))[:S.EVAL_TARGET_LEN]


@pytest.mark.parametrize('name', ('train', 'val'))
def test_built_loads_in_dataset(name):
    if not (S.DEFAULT_OUT / f'{name}_data.df').exists():
        pytest.skip(f'нет сплита в {S.DEFAULT_OUT}')
    statis = pd.read_pickle(S.DEFAULT_OUT / 'data_statis.df')
    ds = ATGDataset(S.DEFAULT_OUT / f'{name}_data.df', seq_len=int(statis['seq_size'].iloc[0]),
                    padding_idx=int(statis['item_num'].iloc[0]))
    assert ds[0]['history'].shape == ds[0]['mask'].shape
