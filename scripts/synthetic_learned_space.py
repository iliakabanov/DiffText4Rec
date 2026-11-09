"""
Одномодальны ли двухмодальные пользователи в выученном пространстве ID-SASRec?

В пространстве генератора ответы двухмодального пользователя — две моды: среднее истории (один вектор) падает
между ними, а оценённые по истории точки двух мод находят ответы. Если в таблице, выученной ID-SASRec,
моды пользователя слились, один вектор (среднее истории) работает там не хуже двух точек.

Для каждого пространства (генератор и выученная таблица) — одни и те же эвристики по косинусу, кандидаты —
товары из train (строки остальных в выученной таблице не обучались), без товаров истории:
    среднее истории         — один вектор;
    точка одной моды        — норм. среднее товаров истории из моды A;
    точки двух мод          — то же для A и B, выдача по кругу без повторов (как слоты у COSMOS).
Кластеры товаров истории берутся из генератора: это проверка геометрии, а не модель.
Плюс косинусы центров кластеров тройки (A–B, A–M, B–M) и средний cos между точками двух мод пользователя.

    python scripts/synthetic_learned_space.py configs/synthetic/sasrec_between_oldtest.yaml
    python scripts/synthetic_learned_space.py configs/synthetic/sasrec_between_oldtest.yaml --run sasrec_between
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from builders import load_model_config                              # noqa: E402
from evaluate import merge_slots                                     # noqa: E402
from synthetic_checks import K, format_table, load, method_table     # noqa: E402

BATCH = 2000


def mode_points(table: torch.Tensor, hist: torch.Tensor, hist_cluster: torch.Tensor,
                mode_cluster: torch.Tensor) -> torch.Tensor:
    """Норм. среднее товаров истории из кластера mode_cluster (U,) для каждого пользователя: (U, d)."""
    mask = (hist_cluster == mode_cluster[:, None]).float()                    # (U, L)
    return F.normalize((table[hist] * mask[..., None]).sum(1), dim=-1)


def top_unseen(query: torch.Tensor, table: torch.Tensor, hist: torch.Tensor, candidates: torch.Tensor, depth: int):
    scores = query @ table.T
    scores[:, ~candidates] = -torch.inf
    return scores.scatter_(1, hist, -torch.inf).topk(depth, dim=-1).indices


def heuristics(table: torch.Tensor, ds: dict, candidates: torch.Tensor) -> dict[str, torch.Tensor]:
    """Выдачи top-K трёх эвристик в пространстве table (строки нормируются)."""
    table = F.normalize(table, dim=-1)
    cluster = torch.from_numpy(ds['items']['cluster'].to_numpy().copy())
    hist_all = torch.tensor(np.stack(ds['val']['seq'].to_numpy()))
    clusters = ds['users']['clusters'].tolist()
    a_all = torch.tensor([c[0] for c in clusters])
    b_all = torch.tensor([c[-1] for c in clusters])                          # у одномодальных B = A
    out = {'среднее истории': [], 'точка одной моды': [], 'точки двух мод': []}
    for s in range(0, len(hist_all), BATCH):
        hist, a, b = hist_all[s:s + BATCH], a_all[s:s + BATCH], b_all[s:s + BATCH]
        hc = cluster[hist]
        mean = F.normalize(table[hist].mean(1), dim=-1)
        p_a, p_b = mode_points(table, hist, hc, a), mode_points(table, hist, hc, b)
        out['среднее истории'].append(top_unseen(mean, table, hist, candidates, K))
        top_a = top_unseen(p_a, table, hist, candidates, K)
        out['точка одной моды'].append(top_a)
        both = torch.stack([top_a, top_unseen(p_b, table, hist, candidates, K)])
        two = merge_slots(both, None, K)
        out['точки двух мод'].append(torch.where((a == b)[:, None], top_a, two))   # одномодальным — одна точка
    return {k: torch.cat(v) for k, v in out.items()}


def geometry(table: torch.Tensor, ds: dict, candidates: torch.Tensor) -> dict[str, float]:
    """cos центров кластеров тройки и средний cos между точками мод двухмодальных пользователей."""
    table = F.normalize(table, dim=-1)
    cluster = ds['items']['cluster'].to_numpy().copy()
    c = torch.stack([table[torch.from_numpy((cluster == k)) & candidates].mean(0) for k in range(cluster.max() + 1)])
    c = F.normalize(c, dim=-1)
    n_triples = int(ds['items']['triple'].max()) + 1
    pair = lambda i, j: float(np.mean([float(c[3 * k + i] @ c[3 * k + j]) for k in range(n_triples)]))
    users = ds['users']
    bi = np.flatnonzero(users['kind'].to_numpy() == 'bimodal')
    hist = torch.tensor(np.stack(ds['val']['seq'].to_numpy()[bi]))
    hc = torch.from_numpy(cluster)[hist]
    a = torch.tensor([users['clusters'].iloc[i][0] for i in bi])
    b = torch.tensor([users['clusters'].iloc[i][1] for i in bi])
    p_a, p_b = mode_points(table, hist, hc, a), mode_points(table, hist, hc, b)
    mean = F.normalize(table[hist].mean(1), dim=-1)
    return {'cos центров A–B': pair(0, 1), 'A–M': pair(0, 2), 'B–M': pair(1, 2),
            'cos точек мод пользователя': float((p_a * p_b).sum(-1).mean()),
            'cos среднего с точкой моды': float(((mean * p_a).sum(-1) + (mean * p_b).sum(-1)).mean() / 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path, help='конфиг ID-SASRec (model_type: sasrec): данные и чекпойнт')
    parser.add_argument('--run', help='имя прогона в output.dir вместо output.run_name')
    parser.add_argument('--checkpoint', default='latest.pt')
    args = parser.parse_args()

    _, cfg = load_model_config(args.config)
    ckpt_path = Path(cfg.output.dir) / (args.run or cfg.output.run_name) / args.checkpoint
    state = torch.load(ckpt_path, map_location='cpu')['model_state']
    ds = load(Path(cfg.data.data_dir))
    candidates = ~ds['new']                                                    # товары из train
    spaces = {'генератор': torch.from_numpy(ds['emb']),
              'ID-SASRec': state['item_embedding.weight'][:cfg.data.n_items].float()}
    print(f'{cfg.data.data_dir}, чекпойнт {ckpt_path}')

    geo = pd.DataFrame({name: geometry(t, ds, candidates) for name, t in spaces.items()})
    print('\nГеометрия (cos, средние по тройкам и по двухмодальным пользователям)')
    print(format_table(geo))
    for name, table in spaces.items():
        for method, ranks in heuristics(table, ds, candidates).items():
            df = method_table(ranks, ds).loc[['двухмод. все', 'одномод.'],
                                             ['HR@50', 'Recall@50', 'A', 'B', 'M', 'обе моды']]
            print(f'\n{name}: {method}')
            print(format_table(df))


if __name__ == '__main__':
    main()
