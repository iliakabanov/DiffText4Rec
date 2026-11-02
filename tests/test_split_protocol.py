"""
Проверка scripts/split_protocol.py: логика на синтетике и, если сплит уже построен, файлы против sequences.df.

    pytest tests/test_split_protocol.py
"""
import pandas as pd
import pytest

import split_protocol as S
from data import ATGDataset

SPLIT_NAMES = ('train', 'val', 'test')


def _sequences(seqs: list[list[int]]) -> pd.DataFrame:
    return pd.DataFrame({'user_id': [f'u{i:03d}' for i in range(len(seqs))],
                         'seq': seqs,
                         'len_seq': [len(s) for s in seqs]})


def _eval(items: list[int], vocab: set[int] | None = None) -> pd.Series:
    return S.eval_rows(['u'], [items], set(items) if vocab is None else vocab).iloc[0]


# --------------------------------------------------------------------------- #
# Логика на синтетике
# --------------------------------------------------------------------------- #

def test_eval_example():
    """n = 10: вход — первые 5 товаров, правильные ответы — последние 5."""
    row = _eval(list(range(10)))
    assert row['seq'] == list(range(5)) and row['len_seq'] == 5
    assert row['targets'] == [5, 6, 7, 8, 9] and row['next'] == 5


def test_eval_shortest_sequence():
    row = _eval([0, 1, 2, 3, 4, 5])
    assert row['seq'] == [0] and row['targets'] == [1, 2, 3, 4, 5]


def test_eval_too_short_is_dropped():
    assert len(S.eval_rows(['u'], [[0, 1, 2, 3, 4]], {0, 1, 2, 3, 4})) == 0


def test_eval_targets_collapse_repeats_keep_order_and_seen_items():
    """Повторы внутри F_u схлопываются; товар из входа остаётся: исключать его — решение оценки."""
    row = _eval([1, 2, 3, 4, 5, 6, 7, 20, 7, 20])
    assert row['seq'] == [1, 2, 3, 4, 5]
    assert row['targets'] == [6, 7, 20]


def test_eval_removes_items_outside_vocab_before_counting():
    """Сначала удаление, затем граница: 12 товаров − 2 вне словаря = 10, вход 5."""
    items = [0, 100, 1, 2, 3, 4, 5, 6, 7, 8, 101, 9]
    row = _eval(items, vocab=set(range(10)))
    assert row['seq'] == list(range(5)) and row['targets'] == [5, 6, 7, 8, 9]


def test_eval_drops_short_after_removal():
    assert len(S.eval_rows(['u'], [[0, 1, 2, 3, 4, 100]], {0, 1, 2, 3, 4})) == 0


def test_build_splits_disjoint_and_train_is_whole():
    seqs = [[j, j + 1, j + 2, j + 3, j + 4, j + 5] for j in range(100)]
    sequences = _sequences(seqs)
    splits = S.build_splits(sequences, seed=0)
    assert len(splits['train']) == 80
    users = [set(splits[n]['user_id']) for n in SPLIT_NAMES]
    assert not (users[0] & users[1]) and not (users[0] & users[2]) and not (users[1] & users[2])
    source = dict(zip(sequences['user_id'], seqs))
    for user, seq, n, nxt in zip(*(splits['train'][c] for c in ('user_id', 'seq', 'len_seq', 'next'))):
        assert seq + [nxt] == source[user] and n == len(seq)


def test_build_splits_vocab_only_from_train():
    """Товар 999 есть только у одного пользователя: если он не в train, из val и test он удалён."""
    seqs = [[0, 1, 2, 3, 4, 5, 6] for _ in range(19)] + [[0, 1, 2, 3, 4, 5, 999]]
    for seed in range(10):
        splits = S.build_splits(_sequences(seqs), seed=seed)
        in_train = any(999 in s or n == 999 for s, n in zip(splits['train']['seq'], splits['train']['next']))
        rest = [i for name in ('val', 'test') for q, t in zip(splits[name]['seq'], splits[name]['targets']) for i in q + t]
        assert in_train or 999 not in rest


def test_build_splits_seed_is_deterministic():
    sequences = _sequences([[j, j + 1, j + 2, j + 3, j + 4] for j in range(50)])
    first, again, other = (S.build_splits(sequences, seed=s)['train']['user_id'].tolist() for s in (1, 1, 2))
    assert first == again and first != other


def test_build_splits_rejects_too_short_train_sequences():
    with pytest.raises(ValueError, match='в train'):
        S.build_splits(_sequences([[1] for _ in range(10)]), seed=0)


# --------------------------------------------------------------------------- #
# Построенный сплит
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def built() -> dict[str, pd.DataFrame]:
    if not all((S.DEFAULT_OUT / f'{n}_data.df').exists() for n in SPLIT_NAMES):
        pytest.skip(f'нет сплита в {S.DEFAULT_OUT}')
    return {n: pd.read_pickle(S.DEFAULT_OUT / f'{n}_data.df') for n in SPLIT_NAMES}


@pytest.fixture(scope='module')
def full() -> dict[str, list[int]]:
    seqs = pd.read_pickle(S.OUT_DIR / 'sequences.df')
    return dict(zip(seqs['user_id'], map(list, seqs['seq'])))


def test_built_users_partition(built, full):
    users = [set(built[n]['user_id']) for n in SPLIT_NAMES]
    assert sum(map(len, users)) == len(set.union(*users))
    assert users[0] <= set(full)
    assert abs(len(users[0]) / len(full) - S.SPLIT_FRACTIONS['train']) < 0.01


def test_built_train_matches_sequences(built, full):
    t = built['train']
    assert all(seq + [nxt] == full[u] for u, seq, nxt in zip(t['user_id'], t['seq'], t['next']))


@pytest.mark.parametrize('name', ('val', 'test'))
def test_built_eval_matches_sequences(name, built, full):
    train = built['train']
    vocab = {i for s in train['seq'] for i in s} | set(train['next'])
    s = built[name]
    for user, seq, n, nxt, targets in zip(*(s[c] for c in S.EVAL_COLUMNS)):
        known = [i for i in full[user] if i in vocab]
        assert n == len(seq) == len(known) - S.EVAL_TARGET_LEN and len(known) >= S.MIN_EVAL_LEN
        assert seq == known[:n]
        assert targets == list(dict.fromkeys(known[n:])) and nxt == targets[0]


@pytest.mark.parametrize('name', SPLIT_NAMES)
def test_built_loads_in_dataset(name):
    if not (S.DEFAULT_OUT / f'{name}_data.df').exists():
        pytest.skip(f'нет сплита в {S.DEFAULT_OUT}')
    statis = pd.read_pickle(S.DEFAULT_OUT / 'data_statis.df')
    item_num = int(statis['item_num'].iloc[0])
    dataset = ATGDataset(S.DEFAULT_OUT / f'{name}_data.df', seq_len=int(statis['seq_size'].iloc[0]), padding_idx=item_num)
    item = dataset[0]
    assert item['history'].shape == item['mask'].shape and 0 <= item['target'] < item_num
