"""
Выдача из нескольких сэмплов: S мест на сэмпл, для @N считается ceil(N / S) сэмплов.

    pytest tests/test_slot_inference.py
"""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluate import evaluate, evaluate_retrievals, merge_slots, slot_rankings
from models.components import ItemEmbedding
from models.cosmos import DreamRecCosmos
from models.sasrec import SASRec

N_ITEMS, D, L = 200, 16, 6


def _model() -> DreamRecCosmos:
    torch.manual_seed(0)
    return DreamRecCosmos(items=ItemEmbedding.learned(N_ITEMS, D, N_ITEMS), n_heads=2, n_layers=1, d_ff=16, dropout=0.0,
                          max_seq_len=L, n_blocks=2, denoiser_heads=2, mlp_mult=1, denoiser_dropout=0.0, T=20,
                          beta_schedule='linear', retrieval_metric='dot', train_prefixes=True).eval()


def _loader(n_users: int = 6) -> DataLoader:
    torch.manual_seed(1)
    history = torch.randint(0, N_ITEMS, (n_users, L))
    targets = torch.randint(0, N_ITEMS, (n_users, 5))
    rows = [{'history': history[i], 'mask': torch.ones(L), 'target': targets[i, 0], 'targets': targets[i]}
            for i in range(n_users)]
    return DataLoader(rows, batch_size=4)


# --------------------------------------------------------------------------- #
# Слияние
# --------------------------------------------------------------------------- #

def test_round_robin_order():
    """Сначала первые соседи всех сэмплов, потом вторые."""
    candidates = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]])
    assert merge_slots(candidates, None, 4).tolist() == [[1, 4, 2, 5]]


def test_duplicate_goes_to_next_neighbor_of_same_sample():
    """Товар 1 уже занят первым сэмплом: второй берёт следующего своего соседа, места не теряются."""
    candidates = torch.tensor([[[1, 2, 3]], [[1, 5, 6]]])
    assert merge_slots(candidates, None, 4).tolist() == [[1, 5, 2, 6]]


def test_seen_items_are_skipped():
    candidates = torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]]])
    assert merge_slots(candidates, torch.tensor([[1, 6]]), 4).tolist() == [[2, 5, 3, 7]]


def test_each_sample_gets_slots_even_with_huge_scores():
    """Слияние по рангам: сэмпл с огромной нормой не вытесняет остальных."""
    items = torch.eye(8)
    samples = torch.stack([1000.0 * items[:1], items[4:5]])
    ranks = slot_rankings(samples, items, 'dot', 2, (4,), None)[4]
    assert ranks.shape == (1, 4) and 0 in ranks[0, :2] and 4 in ranks[0, :2]


def test_uses_first_samples_for_each_k():
    """S = 5: @20 — только первые 4 сэмпла, @50 — все 10. Нормировка: ближайший к товару — он сам."""
    torch.manual_seed(0)
    items = torch.nn.functional.normalize(torch.randn(300, 8), dim=-1)
    samples = items[:10].unsqueeze(1)
    ranks = slot_rankings(samples, items, 'dot', 5, (20, 50), None)
    assert ranks[20].shape == (1, 20) and ranks[50].shape == (1, 50)
    assert set(ranks[20][0, :4].tolist()) == set(items[:4].mm(items.T).argmax(1).tolist())
    assert ranks[50][0, :10].tolist() == items[:10].mm(items.T).argmax(1).tolist()


def test_one_sample_equals_plain_top_k():
    items = torch.randn(50, 8)
    sample = torch.randn(3, 8)
    ranks = slot_rankings(sample[None], items, 'dot', 20, (20,), None)[20]
    assert ranks.tolist() == (sample @ items.T).topk(20, dim=-1).indices.tolist()


@pytest.mark.parametrize('slots, k', [(5, 20), (5, 50), (3, 20), (7, 50), (20, 20), (50, 20)])
def test_lists_are_unique_and_full(slots, k):
    """Любое S, в том числе когда N на него не делится: ровно N разных товаров без просмотренных."""
    torch.manual_seed(0)
    items = torch.randn(100, 4)
    n_samples = -(-k // slots)
    samples = items[:n_samples][:, None].repeat(1, 3, 1)      # все сэмплы близки, совпадений много
    seen = torch.randint(0, 100, (3, L))
    ranks = slot_rankings(samples, items, 'dot', slots, (k,), seen)[k]
    assert ranks.shape == (3, k)
    for row, s in zip(ranks.tolist(), seen.tolist()):
        assert len(set(row)) == k and not set(row) & set(s)


# --------------------------------------------------------------------------- #
# Оценка целиком
# --------------------------------------------------------------------------- #

def _count_samples(model, monkeypatch) -> list[int]:
    calls = []
    original = model.sample
    monkeypatch.setattr(model, 'sample', lambda *args, **kw: calls.append(1) or original(*args, **kw))
    return calls


def test_evaluate_draws_max_k_over_s_samples(monkeypatch):
    model = _model()
    calls = _count_samples(model, monkeypatch)
    metrics = evaluate(model, _loader(4), top_k_values=(20, 50), num_sample_steps=2, w_cfg=0.0, device='cpu',
                       show_progress=False, exclude_seen=True, neighbors_per_sample=5)
    assert len(calls) == 10                                    # один батч: 10 сэмплов на всех K
    assert set(metrics) == {f'{m}@{k}' for m in ('HR', 'Recall', 'NDCG') for k in (20, 50)}


def test_evaluate_without_slots_draws_one_sample(monkeypatch):
    model = _model()
    calls = _count_samples(model, monkeypatch)
    evaluate(model, _loader(4), top_k_values=(20, 50), num_sample_steps=2, w_cfg=0.0, device='cpu',
             show_progress=False, exclude_seen=True)
    assert len(calls) == 1


def test_evaluate_retrievals_with_slots():
    results = evaluate_retrievals(_model(), _loader(), top_k_values=(20,), num_sample_steps=2, device='cpu',
                                  exclude_seen=True, neighbors_per_sample=5)
    assert set(results) == {'Dot Product', 'Cosine Sim', 'L2 Distance', 'Random'}
    assert all(set(row) == {'HR@20', 'Recall@20', 'NDCG@20'} for row in results.values())


def test_slots_require_generative_model():
    sasrec = SASRec(n_items=N_ITEMS, padding_idx=N_ITEMS, d_model=D, n_heads=2, n_layers=1, d_ff=D, max_seq_len=L, loss='ce')
    with pytest.raises(ValueError, match='neighbors_per_sample'):
        evaluate(sasrec, _loader(), top_k_values=(20,), num_sample_steps=1, w_cfg=0.0, device='cpu',
                 show_progress=False, neighbors_per_sample=5)


@pytest.mark.parametrize('slots', [0, -1])
def test_slots_must_be_positive(slots):
    with pytest.raises(ValueError, match='neighbors_per_sample'):
        evaluate(_model(), _loader(), top_k_values=(20,), num_sample_steps=1, w_cfg=0.0, device='cpu',
                 show_progress=False, neighbors_per_sample=slots)
