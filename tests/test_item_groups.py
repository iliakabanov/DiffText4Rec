"""
Метрики по старым и новым товарам (eval.item_groups) и выбор best.pt по eval.select_metric.

    pytest tests/test_item_groups.py
"""
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from data import new_item_mask
from evaluate import compute_metrics, evaluate, group_metrics, metric_names
from models.sasrec import SASRec

N_ITEMS = 10
NEW = torch.tensor([False] * 6 + [True] * 4)          # товары 6–9 новые


def test_groups_split_answers_by_hand():
    """Выдача [6, 1, 2]; ответы {1, 6, 7}: старый 1 найден, из новых {6, 7} найден один; новый в выдаче один из трёх."""
    m = group_metrics(torch.tensor([[6, 1, 2]]), torch.tensor([[1, 6, 7]]), NEW, 3)
    assert m == {'HR@3_old': 1.0, 'Recall@3_old': 1.0, 'HR@3_new': 1.0, 'Recall@3_new': 0.5,
                 'NewShare@3': pytest.approx(1 / 3)}


def test_new_share_counts_all_users():
    """Доля новых — по всем пользователям, в том числе без новых ответов, и только по первым K местам."""
    rankings = torch.tensor([[6, 7, 0], [1, 2, 8]])
    targets = torch.tensor([[0, -1], [1, -1]])
    assert group_metrics(rankings, targets, NEW, 2)['NewShare@2'] == 0.5


def test_user_without_group_answers_is_skipped():
    """У второго пользователя новых ответов нет: в HR_new он не входит, в HR_old входит."""
    rankings = torch.tensor([[6, 0], [3, 4]])
    targets = torch.tensor([[6, 5], [1, -1]])
    m = group_metrics(rankings, targets, NEW, 2)
    assert m['HR@2_new'] == 1.0 and m['Recall@2_new'] == 1.0
    assert m['HR@2_old'] == 0.0


def test_compute_metrics_with_groups_has_only_hr_and_recall():
    rankings = torch.tensor([[6, 1, 2]])
    assert set(compute_metrics(rankings, torch.tensor([[1, 7]]), (3,), NEW)) == set(metric_names((3,), True))
    assert set(compute_metrics(rankings, torch.tensor([7]), (3,), NEW)) == set(metric_names((3,), True))


def test_new_item_mask_reads_history_and_target(tmp_path):
    pd.DataFrame({'seq': [[0, 1, 2], [3]], 'len_seq': [2, 1], 'next': [4, 5]}).to_pickle(tmp_path / 'train_data.df')
    # 0 стоит в seq, но за пределами len_seq: реальная история — [1, 2]
    assert new_item_mask(tmp_path, 8).tolist() == [True, False, False, False, False, False, True, True]


def test_evaluate_returns_group_metrics():
    torch.manual_seed(0)
    model = SASRec(n_items=N_ITEMS, padding_idx=N_ITEMS, d_model=8, n_heads=2, n_layers=1, d_ff=8, max_seq_len=4,
                   loss='ce', retrieval_metric='dot').eval()
    rows = [{'history': torch.randint(0, N_ITEMS, (4,)), 'mask': torch.ones(4), 'target': torch.tensor(0),
             'targets': torch.tensor([0, 7, -1])} for _ in range(3)]
    metrics = evaluate(model, DataLoader(rows, batch_size=2), top_k_values=(5,), num_sample_steps=1, w_cfg=0.0,
                       device='cpu', show_progress=False, new_items=NEW)
    assert set(metrics) == {'HR@5_old', 'Recall@5_old', 'HR@5_new', 'Recall@5_new', 'NewShare@5'}


def test_select_metric_must_be_computed(tmp_path):
    """Опечатка в select_metric ловится до загрузки данных, а не на первой валидации."""
    from config import EvalConfig, OptimizerConfig, OutputConfig, DataConfig, TrainConfig
    from train import fit
    cfg = TrainConfig(seed=0, model_type='sasrec', model=None,
                      data=DataConfig(str(tmp_path), N_ITEMS, N_ITEMS, 2, 0, 4),
                      optimizer=OptimizerConfig('adam', 1e-3, 0.0, None, 1, 0, None),
                      eval=EvalConfig(1, (50,), item_groups=True, select_metric='NDCG@20'),
                      output=OutputConfig(str(tmp_path), 'run'))
    with pytest.raises(ValueError, match='select_metric'):
        fit(cfg, lambda c: None)
