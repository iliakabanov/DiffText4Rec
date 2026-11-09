"""
Проверки синтетического датасета (scripts/make_synthetic.py) до обучения моделей.

1. Структура: доля новых среди ответов, ответы в обеих модах, N_eff моды до и после истории.
2. Выдачи top-K по val с разбивкой по группам (двухмодальные по Δ, одномодальные):
       оракул          — знает моды, веса и личные точки: оценка товара Σ_m w_m · P(e | m) по непросмотренным;
       оракул, 1 мода  — то же, но только по моде A: лучшая выдача из одной моды и лучший один вектор;
       1 вектор, середина — косинус с норм(Σ_m w_m · p_m): один вектор между личными точками, куда его
                         тянет правдоподобие (покрыть обе моды одним пиком);
       соседи last     — косинус с последним товаром истории;
       соседи mean     — косинус со средним эмбеддингом истории.
   Метрики — как на обучении с eval.item_groups (новые = нет в train), плюс HR и Recall по всем ответам
   и состав выдачи двухмодальных: доли A, B, M своей тройки и прочих. Все выдачи — без товаров истории,
   как при eval.exclude_seen: true в configs/synthetic.

Предсказания: в synthetic_between среднее истории и вектор-середина уходят в M, в synthetic_random — на
ближние друг к другу края мод; выдача из одной моды заметно хуже оракула в обоих датасетах.

    python scripts/synthetic_checks.py data/processed/synthetic_between data/processed/synthetic_random
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from data import new_item_mask, load_item_embeddings   # noqa: E402
from evaluate import compute_metrics                    # noqa: E402
from make_synthetic import mode_weights                 # noqa: E402

K = 50


def load(data_dir: Path) -> dict:
    params = json.loads((data_dir / 'params.json').read_text(encoding='utf-8'))
    val = pd.read_pickle(data_dir / 'val_data.df')
    users = pd.read_pickle(data_dir / 'users_meta.df').set_index('user_id').loc[val['user_id']].reset_index()
    items = pd.read_pickle(data_dir / 'items_meta.df')
    emb = load_item_embeddings(data_dir / 'item_embeddings.pickle')
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)              # файл — на сфере радиуса √d, генератор считал на единичной
    new = new_item_mask(data_dir, len(emb))
    return {'params': params, 'val': val, 'users': users, 'items': items, 'emb': emb, 'new': new}


def oracle_mode_scores(ds: dict, user: pd.Series, history: list[int]) -> list[np.ndarray]:
    """
    P(e | мода m) по товарам, из которых генерировались ответы (непросмотренные; при old_only_answers — только
    старые), для каждой моды пользователя: список (N,) с нулями вне моды.
    """
    item_cluster = ds['items']['cluster'].to_numpy()
    allowed = np.ones(len(item_cluster), dtype=bool)
    allowed[history] = False
    if ds['params'].get('old_only_answers'):            # ответы генерировались только из старых товаров
        allowed &= ~ds['items']['is_new'].to_numpy()
    out = []
    for c, p in zip(user['clusters'], user['points']):
        idx = np.flatnonzero((item_cluster == c) & allowed)
        s = np.zeros(len(item_cluster))
        w = mode_weights(ds['emb'][idx], p, ds['params']['kappa'])
        s[idx] = w
        out.append(s)
    return out


def n_eff(w: np.ndarray) -> float:
    w = w[w > 0]
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


def rankings(ds: dict) -> dict[str, torch.Tensor]:
    users, val, emb = ds['users'], ds['val'], torch.from_numpy(ds['emb'])
    oracle, one_mode = [], []
    for (_, user), history in zip(users.iterrows(), val['seq']):
        scores = oracle_mode_scores(ds, user, history)
        mix = sum(wt * s for wt, s in zip(user['weights'], scores))
        oracle.append(np.argsort(-mix, kind='stable')[:K])
        one_mode.append(np.argsort(-scores[0], kind='stable')[:K])
    hist = torch.tensor(np.stack(val['seq'].to_numpy()))
    middle = torch.nn.functional.normalize(torch.tensor(np.stack(
        [(np.asarray(w)[:, None] * p).sum(0) for w, p in zip(users['weights'], users['points'])])).float(), dim=-1)
    last = emb[hist[:, -1]]
    mean = torch.nn.functional.normalize(emb[hist].mean(1), dim=-1)
    unseen_top = lambda q: (q @ emb.T).scatter_(1, hist, -torch.inf).topk(K, dim=-1).indices   # exclude_seen
    return {'оракул': torch.tensor(np.stack(oracle)),
            'оракул, 1 мода': torch.tensor(np.stack(one_mode)),
            '1 вектор, середина': unseen_top(middle),
            'соседи last': unseen_top(last),
            'соседи mean': unseen_top(mean)}


def groups(users: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {f'Δ={d}°': np.array((users['kind'] == 'bimodal') & (users['delta'] == d))
           for d in sorted(users.loc[users['kind'] == 'bimodal', 'delta'].unique())}
    out['двухмод. все'] = np.array(users['kind'] == 'bimodal')    # главный результат: по Δ потолки плоские
    out['одномод.'] = np.array(users['kind'] == 'unimodal')
    return out


def composition(ranks: torch.Tensor, ds: dict, rows: np.ndarray) -> dict[str, float]:
    """Доли A, B, M своей тройки и прочих в выдаче двухмодальных; both — в выдаче есть и A, и B."""
    cl = ds['items']['cluster'].to_numpy()[ranks[rows].numpy()]                 # (n, K)
    a = np.array([c[0] for c in ds['users'].loc[rows, 'clusters']])[:, None]
    b = np.array([c[1] for c in ds['users'].loc[rows, 'clusters']])[:, None]
    m = b + 1                                                                  # кластеры тройки идут A, B, M подряд
    in_a, in_b, in_m = cl == a, cl == b, cl == m
    return {'A': in_a.mean(), 'B': in_b.mean(), 'M': in_m.mean(), 'прочие': 1 - (in_a | in_b | in_m).mean(),
            'обе моды': (in_a.any(1) & in_b.any(1)).mean()}


def structure(ds: dict) -> pd.DataFrame:
    users, val, new = ds['users'], ds['val'], ds['new'].numpy()
    item_cluster = ds['items']['cluster'].to_numpy()
    rows = []
    for name, rows_mask in groups(users).items():
        idx = np.flatnonzero(rows_mask)
        sub = idx[:300]                                                        # N_eff — по подвыборке, это дорого
        before, after = [], []
        for i in sub:
            user = users.loc[i]
            for c, p in zip(user['clusters'], user['points']):
                members = np.flatnonzero(item_cluster == c)
                w = mode_weights(ds['emb'][members], p, ds['params']['kappa'])
                before.append(n_eff(w))
            after += [n_eff(s) for s in oracle_mode_scores(ds, user, val['seq'][i])]
        targets = np.stack(val.loc[idx, 'targets'].to_numpy())
        both = np.nan
        if name != 'одномод.':
            a = np.array([c[0] for c in users.loc[idx, 'clusters']])[:, None]
            tc = item_cluster[targets]
            both = ((tc == a).any(1) & (tc != a).any(1)).mean()
        rows.append({'группа': name, 'пользователей': len(idx), 'новых в ответах': new[targets].mean(),
                     'ответы в обеих модах': both, 'N_eff моды': np.mean(before), 'N_eff после истории': np.mean(after)})
    return pd.DataFrame(rows).set_index('группа')


def report(data_dir: Path) -> None:
    ds = load(data_dir)
    print(f'\n{"=" * 100}\n{data_dir.name}: distractor={ds["params"]["distractor"]}, товаров {len(ds["emb"]):,}, '
          f'нет в train {int(ds["new"].sum()):,} (из них новых по генератору '
          f'{int((ds["new"].numpy() & ds["items"]["is_new"].to_numpy()).sum()):,}, '
          f'старых, но не выпавших в train {int((ds["new"].numpy() & ~ds["items"]["is_new"].to_numpy()).sum()):,})')
    print('\nСтруктура данных')
    print(structure(ds).to_string(float_format=lambda x: f'{x:.3f}'))

    for method, ranks in rankings(ds).items():
        print(f'\n{method}')
        print(format_table(method_table(ranks, ds)))


def method_table(ranks: torch.Tensor, ds: dict) -> pd.DataFrame:
    """Метрики выдачи (n_users, K) по группам пользователей; состав выдачи — у двухмодальных."""
    targets = torch.tensor(np.stack(ds['val']['targets'].to_numpy()))
    table = []
    for name, rows in groups(ds['users']).items():
        r, t = ranks[rows], targets[rows]
        overall = compute_metrics(r, t, (K,))
        by_group = compute_metrics(r, t, (K,), ds['new'])
        row = {'группа': name, f'HR@{K}': overall[f'HR@{K}'], f'Recall@{K}': overall[f'Recall@{K}'],
               f'Recall@{K}_old': by_group[f'Recall@{K}_old'], f'Recall@{K}_new': by_group[f'Recall@{K}_new'],
               f'NewShare@{K}': by_group[f'NewShare@{K}']}
        if name != 'одномод.':
            row.update(composition(ranks, ds, rows))
        table.append(row)
    return pd.DataFrame(table).set_index('группа')


def format_table(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f'{x:.3f}', na_rep='—')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dirs', nargs='+', type=Path)
    args = parser.parse_args()
    for data_dir in args.data_dirs:
        report(data_dir)


if __name__ == '__main__':
    main()
