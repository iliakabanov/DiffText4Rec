"""
Синтетический датасет (scripts/make_synthetic.py) и маска товаров в softmax SASRec (loss_items).

    pytest tests/test_synthetic.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from make_synthetic import SyntheticConfig, generate, mode_weights   # noqa: E402
from models.sasrec import SASRec                                     # noqa: E402

SMALL = SyntheticConfig(d=16, n_triples=2, deltas_deg=(60, 120), n_background=2, items_per_cluster=40,
                        n_users=200, history_len=8, n_answers=3, kappa=0.01)


@pytest.fixture(scope='module')
def both():
    return {d: generate(SMALL, d) for d in ('between', 'random')}


# --------------------------------------------------------------------------- #
# Генератор
# --------------------------------------------------------------------------- #

def test_geometry(both):
    c = both['between']['centers']
    for k, delta in enumerate((60, 120)):
        a, b, m = c[3 * k], c[3 * k + 1], c[3 * k + 2]
        assert np.isclose(a @ b, np.cos(np.radians(delta)), atol=1e-6)
        assert np.isclose(a @ m, np.cos(np.radians(delta / 2)), atol=1e-6)   # M на биссектрисе
    r = both['random']['centers']
    off = r @ r.T - np.diag(np.diag(r @ r.T))
    for k in range(2):                                     # в контроле M ортогонален всему
        assert np.abs(off[3 * k + 2]).max() < 1e-6


def test_datasets_differ_only_in_m(both):
    b, r = both['between'], both['random']
    is_m = (b['items']['role'] == 'M').to_numpy()
    assert np.allclose(b['emb'][~is_m], r['emb'][~is_m]) and not np.allclose(b['emb'][is_m], r['emb'][is_m])
    m_clusters = set(b['items'].loc[is_m, 'cluster'])
    for seq_b, seq_r, clusters in zip(b['val']['seq'], r['val']['seq'], b['users']['clusters']):
        assert (seq_b == seq_r) or clusters[0] in m_clusters


def test_history_old_only_answers_unseen(both):
    out = both['between']
    is_new = out['items']['is_new'].to_numpy()
    for seq, targets in zip(out['val']['seq'], out['val']['targets']):
        assert not is_new[seq].any()
        assert len(set(seq)) == len(seq) and not set(seq) & set(targets) and len(set(targets)) == len(targets)


def test_users_draw_only_from_their_modes(both):
    out = both['between']
    cluster = out['items']['cluster'].to_numpy()
    for seq, targets, clusters in zip(out['val']['seq'], out['val']['targets'], out['users']['clusters']):
        assert set(cluster[seq + targets]) <= set(clusters)


def test_train_is_history_without_last(both):
    out = both['between']
    for (_, tr), (_, va) in zip(out['train'].iterrows(), out['val'].iterrows()):
        assert tr['seq'] + [tr['next']] == va['seq']


def test_uniform_kappa():
    w = mode_weights(np.eye(5, dtype=np.float32), np.ones(5, dtype=np.float32) / 5 ** 0.5, float('inf'))
    assert np.allclose(w, 0.2)


# --------------------------------------------------------------------------- #
# Маска товаров в softmax
# --------------------------------------------------------------------------- #

N = 20


def _model(mask=None) -> SASRec:
    torch.manual_seed(0)
    return SASRec(n_items=N, padding_idx=N, d_model=8, n_heads=2, n_layers=1, d_ff=8, dropout=0.0, max_seq_len=4,
                  loss='ce', temperature=1.0, loss_metric='dot', retrieval_metric='dot', train_prefixes=True,
                  loss_items=mask)


def _batch():
    history = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    return history, torch.tensor([9, 10]), torch.ones(2, 4)


def test_masked_items_get_no_gradient_and_loss_is_smaller():
    mask = torch.ones(N, dtype=torch.bool)
    mask[15:] = False
    masked, plain = _model(mask).train(), _model().train()
    loss_m, loss_p = masked.compute_loss(*_batch()), plain.compute_loss(*_batch())
    assert loss_m < loss_p                                   # меньше слагаемых в знаменателе softmax
    loss_m.backward()
    assert masked.item_embedding.weight.grad[15:N].abs().max() == 0


def test_mask_is_off_in_eval_and_not_in_state_dict():
    mask = torch.ones(N, dtype=torch.bool)
    mask[15:] = False
    masked, plain = _model(mask).eval(), _model().eval()
    assert torch.equal(masked.compute_loss(*_batch()), plain.compute_loss(*_batch()))
    assert 'loss_items' not in masked.state_dict()
    plain.load_state_dict(masked.state_dict())
    assert masked.config()['loss_items'] == 15


def test_mask_requires_ce():
    with pytest.raises(ValueError, match='loss_items'):
        SASRec(n_items=N, padding_idx=N, d_model=8, n_heads=2, n_layers=1, d_ff=8, max_seq_len=4, loss='infonce',
               loss_items=torch.ones(N, dtype=torch.bool))
