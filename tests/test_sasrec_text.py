"""
SASRec на замороженной таблице товаров (model_type: sasrec_text).

    pytest tests/test_sasrec_text.py
"""
import pickle

import numpy as np
import pytest
import torch
import yaml

from builders import load_model_config

N_ITEMS, D_RAW, D = 60, 24, 8


@pytest.fixture
def config(tmp_path):
    emb = np.random.default_rng(0).standard_normal((N_ITEMS, D_RAW)).astype(np.float32)
    with open(tmp_path / 'emb.pickle', 'wb') as f:
        pickle.dump(emb, f)

    def write(**items):
        cfg = {
            'seed': 0, 'model_type': 'sasrec_text',
            'data': {'data_dir': str(tmp_path), 'padding_idx': N_ITEMS, 'n_items': N_ITEMS, 'batch_size': 4,
                     'num_workers': 0, 'max_seq_len': 6},
            'model': {'items': {'source': 'text', 'emb_path': str(tmp_path / 'emb.pickle'), 'use_pca': True,
                                'n_components': D, 'per_component_whitening': True,
                                'whitening_cache': str(tmp_path / 'wh.npz'), **items},
                      'n_heads': 2, 'n_layers': 1, 'd_ff': 8, 'dropout': 0.0, 'loss': 'ce', 'neg_ratio': 4,
                      'temperature': 1.0, 'retrieval_metric': 'dot', 'train_prefixes': True, 'loss_metric': 'dot'},
            'optimizer': {'name': 'adam', 'lr': 1e-3, 'weight_decay': 0.0, 'grad_clip': 1.0, 'num_epochs': 1,
                          'warmup_epochs': 0, 'ema_decay': None},
            'eval': {'every_epochs': 1, 'top_k': [5]},
            'output': {'dir': str(tmp_path), 'run_name': 'run'},
        }
        path = tmp_path / 'cfg.yaml'
        path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
        spec, loaded = load_model_config(path)
        return spec.build(loaded)
    return write


def _batch():
    history = torch.tensor([[N_ITEMS, N_ITEMS, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10]])
    return history, torch.tensor([11, 12]), (history != N_ITEMS).float()


def test_table_is_frozen_and_d_model_follows_it(config):
    model = config()
    assert model.d_model == D and model.items.table.shape == (N_ITEMS, D)
    assert not any(p.requires_grad for p in model.items.parameters())
    history, target, mask = _batch()
    model.train().compute_loss(history, target, mask).backward()
    assert all(p.grad is None for p in model.items.parameters())
    assert model.blocks[0].attn.in_proj_weight.grad is not None


def test_same_table_as_cosmos(config, tmp_path):
    """Таблица товаров совпадает с той, что собрал бы dreamrec_cosmos по тем же model.items."""
    from builders import CosmosConfig, _cosmos_items
    from config import load_config
    model = config()
    raw = yaml.safe_load((tmp_path / 'cfg.yaml').read_text(encoding='utf-8'))
    raw['model_type'] = 'dreamrec_cosmos'
    raw['model'] = {'items': raw['model']['items'], 'n_heads': 2, 'n_layers': 1, 'd_ff': 8, 'dropout': 0.0,
                    'n_blocks': 2, 'denoiser_heads': 2, 'mlp_mult': 1, 'denoiser_dropout': 0.0, 'T': 10,
                    'beta_schedule': 'linear', 'retrieval_metric': 'dot'}
    (tmp_path / 'cosmos.yaml').write_text(yaml.safe_dump(raw), encoding='utf-8')
    cosmos_items = _cosmos_items(load_config(tmp_path / 'cosmos.yaml', CosmosConfig))
    assert torch.allclose(model.items.table, cosmos_items.table)


def test_recommend_ranks_whole_catalog(config):
    model = config().eval()
    history, _, mask = _batch()
    recs = model.recommend(history, mask, top_k=N_ITEMS)
    assert recs.shape == (2, N_ITEMS) and sorted(recs[0].tolist()) == list(range(N_ITEMS))


def test_learned_source_is_rejected(config):
    with pytest.raises(ValueError, match='sasrec_text'):
        config(source='learned', emb_path=None, use_pca=None, n_components=None, per_component_whitening=None,
               whitening_cache=None, d_emb=D)
