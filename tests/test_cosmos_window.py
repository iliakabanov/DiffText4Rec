"""
Окно целей в генеративных моделях (GenerativeRecommender._prefix_labels) и DreamRecCosmos на нём.

    pytest tests/test_cosmos_window.py
"""
import pytest
import torch

from models.components import ItemEmbedding
from models.cosmos import DreamRecCosmos

N_ITEMS, D = 50, 16


def _model(window: int, train_prefixes: bool = True) -> DreamRecCosmos:
    torch.manual_seed(0)
    return DreamRecCosmos(items=ItemEmbedding.learned(N_ITEMS, D, N_ITEMS), n_heads=2, n_layers=1, d_ff=16, dropout=0.0,
                          max_seq_len=6, n_blocks=2, denoiser_heads=2, mlp_mult=1, denoiser_dropout=0.0, T=50,
                          beta_schedule='linear', retrieval_metric='dot', train_prefixes=train_prefixes,
                          target_window=window)


def _batch():
    history = torch.tensor([[N_ITEMS, N_ITEMS, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10]])
    return history, torch.tensor([11, 12]), (history != N_ITEMS).float()


def test_window_one_is_next_item():
    history, target, mask = _batch()
    assert _model(1)._prefix_labels(history, target, mask).tolist() == [2, 3, 4, 11, 6, 7, 8, 9, 10, 12]


def test_window_labels_stay_in_window_and_cover_it():
    """Первые 4 строки — позиции первого пользователя: окна {2,3,4}, {3,4,11}, {4,11}, {11}."""
    history, target, mask = _batch()
    model = _model(3)
    torch.manual_seed(1)
    draws = torch.stack([model._prefix_labels(history, target, mask) for _ in range(3000)])
    assert set(draws[:, 0].tolist()) == {2, 3, 4}
    assert set(draws[:, 1].tolist()) == {3, 4, 11}
    assert set(draws[:, 2].tolist()) == {4, 11}
    assert set(draws[:, 3].tolist()) == {11}
    assert set(draws[:, 9].tolist()) == {12}
    assert (draws != N_ITEMS).all()


def test_window_labels_are_uniform():
    history, target, mask = _batch()
    model = _model(3)
    torch.manual_seed(2)
    draws = torch.stack([model._prefix_labels(history, target, mask) for _ in range(6000)])[:, 0]
    shares = [(draws == i).float().mean().item() for i in (2, 3, 4)]
    assert shares == pytest.approx([1 / 3] * 3, abs=0.03)


def test_compute_loss_backprops_with_window():
    history, target, mask = _batch()
    model = _model(5).train()
    loss = model.compute_loss(history, target, mask)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.items.embedding.weight.grad[N_ITEMS].abs().max() == 0.0


def test_window_requires_prefixes():
    with pytest.raises(ValueError, match='target_window'):
        _model(2, train_prefixes=False)


def test_config_mentions_window_only_when_used():
    assert _model(5).config()['target_window'] == 5
    assert 'target_window' not in _model(1).config()


def test_recommend_excludes_nothing_by_itself():
    history, _, mask = _batch()
    recs = _model(5).eval().recommend(history, mask, top_k=5, num_steps=3, w_cfg=0.0)
    assert recs.shape == (2, 5) and (recs < N_ITEMS).all()
