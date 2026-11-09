"""
Обучение и оценка по протоколу: лосс SASRec по окну, метрики по множеству F_u, колонка targets в данных.

    pytest tests/test_protocol_training.py
"""
import math

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from data import ATGDataset
from evaluate import batch_targets, compute_metrics, drop_seen, hit_ratio, ndcg, set_metrics
from models.losses import future_items
from models.sasrec import SASRec

PAD = 50


def _model(window: int, **kwargs) -> SASRec:
    torch.manual_seed(0)
    defaults = dict(n_items=PAD, padding_idx=PAD, d_model=16, n_heads=2, n_layers=1, d_ff=16, dropout=0.0,
                    max_seq_len=6, loss='ce', temperature=1.0, loss_metric='dot', retrieval_metric='dot',
                    train_prefixes=True, target_window=window)
    return SASRec(**{**defaults, **kwargs}).eval()


def _batch():
    history = torch.tensor([[PAD, PAD, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10]])
    mask = (history != PAD).float()
    target = torch.tensor([11, 12])
    return history, target, mask


# --------------------------------------------------------------------------- #
# Лосс по окну
# --------------------------------------------------------------------------- #

def test_future_items_windows():
    history, target, _ = _batch()
    future = future_items(history, target, 3, PAD)
    assert future.shape == (2, 6, 3)
    assert future[0, 2].tolist() == [2, 3, 4]              # первая реальная позиция
    assert future[0, 4].tolist() == [4, 11, PAD]           # окно упирается в конец последовательности
    assert future[0, 5].tolist() == [11, PAD, PAD]
    assert future[1, 3].tolist() == [9, 10, 12]


def test_window_one_equals_cross_entropy():
    """Формула окна при одном товаре — обычный CE."""
    model = _model(2)
    h = torch.randn(7, 16)
    target = torch.randint(0, PAD, (7,))
    future = torch.stack([target, torch.full_like(target, PAD)], dim=1)
    assert torch.allclose(model._window_loss(h, future), F.cross_entropy(model._logits(h), target), atol=1e-5)


def test_window_loss_matches_definition():
    """Перебор по определению: каждый товар окна против каталога без остальных товаров окна."""
    model = _model(3)
    h = torch.randn(4, 16)
    future = torch.tensor([[1, 2, 3], [4, 5, PAD], [6, PAD, PAD], [7, 8, 9]])
    logits = model._logits(h)
    terms = []
    for i, row in enumerate(future):
        window = [x for x in row.tolist() if x != PAD]
        for p in window:
            others = [j for j in range(PAD) if j not in window]
            terms.append(-(logits[i, p] - torch.logsumexp(logits[i, [p, *others]], dim=0)))
    assert torch.allclose(model._window_loss(h, future), torch.stack(terms).mean(), atol=1e-5)


def test_compute_loss_counts_real_positions_and_backprops():
    history, target, mask = _batch()
    model = _model(5).train()
    loss = model.compute_loss(history, target, mask)
    loss.backward()
    assert torch.isfinite(loss)
    grads = model.item_embedding.weight.grad
    assert torch.isfinite(grads).all() and grads[PAD].abs().max() == 0


@pytest.mark.parametrize('kwargs', [{'loss': 'infonce'}, {'train_prefixes': False}])
def test_window_requires_ce_and_prefixes(kwargs):
    with pytest.raises(ValueError, match='target_window'):
        _model(2, **kwargs)


def test_window_one_keeps_old_path():
    """W = 1 идёт прежней веткой: те же значения, что у модели без параметра."""
    history, target, mask = _batch()
    new = _model(1).compute_loss(history, target, mask)
    torch.manual_seed(0)
    old = SASRec(n_items=PAD, padding_idx=PAD, d_model=16, n_heads=2, n_layers=1, d_ff=16, dropout=0.0, max_seq_len=6,
                 loss='ce', temperature=1.0, loss_metric='dot', retrieval_metric='dot', train_prefixes=True).eval()
    assert new.item() == old.compute_loss(history, target, mask).item()
    assert 'target_window' not in old.config()


# --------------------------------------------------------------------------- #
# Метрики по множеству
# --------------------------------------------------------------------------- #

def test_set_metrics_by_hand():
    """Выдача [1, 2, 3], ответы {2, 9}: одно попадание на втором месте из двух ответов."""
    m = set_metrics(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 9]]), 3)
    ideal = 1.0 + 1.0 / math.log2(3)
    assert m['HR@3'] == 1.0 and m['Recall@3'] == 0.5
    assert m['NDCG@3'] == pytest.approx((1.0 / math.log2(3)) / ideal)


def test_set_metrics_perfect_and_miss():
    perfect = set_metrics(torch.tensor([[2, 9, 7]]), torch.tensor([[9, 2, -1]]), 3)
    assert perfect == {'HR@3': 1.0, 'Recall@3': 1.0, 'NDCG@3': pytest.approx(1.0)}
    miss = set_metrics(torch.tensor([[1, 3, 7]]), torch.tensor([[9, 2, -1]]), 3)
    assert miss == {'HR@3': 0.0, 'Recall@3': 0.0, 'NDCG@3': 0.0}


def test_set_metrics_more_answers_than_k():
    """Ответов больше K: идеал — K ответов, поэтому полное попадание даёт NDCG 1, а Recall меньше 1."""
    m = set_metrics(torch.tensor([[1, 2]]), torch.tensor([[1, 2, 3, 4]]), 2)
    assert m['NDCG@2'] == pytest.approx(1.0) and m['Recall@2'] == 0.5


def test_set_metrics_skips_users_without_answers():
    m = set_metrics(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, -1], [-1, -1]]), 2)
    assert m['HR@2'] == 1.0 and m['Recall@2'] == 1.0


def test_one_answer_matches_old_metrics():
    torch.manual_seed(0)
    rankings = torch.stack([torch.randperm(30)[:20] for _ in range(200)])
    target = torch.randint(0, 30, (200,))
    new = set_metrics(rankings, target[:, None], 20)
    assert new['HR@20'] == pytest.approx(hit_ratio(rankings, target, 20))
    assert new['NDCG@20'] == pytest.approx(ndcg(rankings, target, 20), abs=1e-6)
    assert new['Recall@20'] == pytest.approx(new['HR@20'])


def test_compute_metrics_keeps_old_keys_for_single_target():
    rankings = torch.tensor([[1, 2, 3]])
    assert set(compute_metrics(rankings, torch.tensor([2]), (3,))) == {'HR@3', 'NDCG@3'}
    assert set(compute_metrics(rankings, torch.tensor([[2]]), (3,))) == {'HR@3', 'Recall@3', 'NDCG@3'}


def test_drop_seen_keeps_order():
    rankings = torch.tensor([[5, 1, 6, 2, 7]])
    history = torch.tensor([[PAD, 5, 6]])
    assert drop_seen(rankings, history, 3).tolist() == [[1, 2, 7]]


def test_batch_targets_excludes_seen():
    batch = {'targets': torch.tensor([[1, 2, -1], [3, 4, 5]]), 'target': torch.tensor([1, 3])}
    history = torch.tensor([[PAD, 2], [3, 5]])
    assert batch_targets(batch, history, True).tolist() == [[1, -1, -1], [-1, 4, -1]]
    assert batch_targets(batch, history, False).tolist() == batch['targets'].tolist()


def test_exclude_seen_needs_targets_column():
    with pytest.raises(ValueError, match='targets'):
        batch_targets({'target': torch.tensor([1])}, torch.tensor([[1]]), True)


# --------------------------------------------------------------------------- #
# Данные
# --------------------------------------------------------------------------- #

def test_dataset_targets_column(tmp_path):
    path = tmp_path / 'val.df'
    pd.DataFrame({'seq': [[1, 2], [3]], 'len_seq': [2, 1], 'next': [4, 5], 'targets': [[4, 6], [5]]}).to_pickle(path)
    ds = ATGDataset(path, seq_len=3, padding_idx=PAD)
    assert ds[0]['targets'].tolist() == [4, 6] and ds[1]['targets'].tolist() == [5, -1]


def test_dataset_without_targets_has_no_key(tmp_path):
    path = tmp_path / 'train.df'
    pd.DataFrame({'seq': [[1, 2]], 'len_seq': [2], 'next': [4]}).to_pickle(path)
    assert 'targets' not in ATGDataset(path, seq_len=3, padding_idx=PAD)[0]
