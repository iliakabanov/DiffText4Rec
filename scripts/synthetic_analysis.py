"""
Анализ обученных моделей на синтетике (configs/synthetic): метрики по группам пользователей вместо
общих по всем, как в log_val.csv.

Для каждого конфига берётся чекпойнт (по умолчанию latest.pt) и считается выдача top-50 на val
без товаров истории — как на обучении с exclude_seen: true. Дальше — как в scripts/synthetic_checks.py:
HR и Recall по всем ответам, по старым и новым, NewShare и состав выдачи двухмодальных (A, B, M, прочие),
по Δ, по всем двухмодальным и по одномодальным.

У генеративных моделей дополнительно — где лежат сэмплы: ближайший центр кластера для каждого из
ceil(50 / S) сэмплов пользователя (A, B, M своей тройки или прочие), доля двухмодальных пользователей
с сэмплами в обеих модах и норма сэмплов (у товаров она 8).

    python scripts/synthetic_analysis.py configs/synthetic/sasrec_text_between.yaml configs/synthetic/cosmos_between.yaml
    python scripts/synthetic_analysis.py configs/synthetic/*.yaml --checkpoint best.pt
    python scripts/synthetic_analysis.py configs/synthetic/cosmos_between.yaml --neighbors-per-sample 50   # один сэмпл
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from builders import load_model_config                                   # noqa: E402
from data import build_dataloaders                                        # noqa: E402
from evaluate import draw_samples, drop_seen, slot_rankings               # noqa: E402
from synthetic_checks import K, format_table, groups, load, method_table  # noqa: E402
from train import EVAL_SEED, seeded                                       # noqa: E402


@torch.no_grad()
def predict(model, cfg, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Выдача (n_users, K) без товаров истории и, у генеративных моделей, сэмплы (n_samples, n_users, d)."""
    loader = build_dataloaders(cfg.data.data_dir, batch_size=512, num_workers=0, seq_len=cfg.data.max_seq_len,
                               padding_idx=cfg.data.padding_idx, splits=('val',))['val']
    generative = hasattr(model, 'sample')
    ranks, samples = [], []
    with seeded(EVAL_SEED, device):
        for batch in loader:
            history, mask = batch['history'].to(device), batch['mask'].to(device)
            if generative:
                s = draw_samples(model, history, mask, math.ceil(K / cfg.eval.neighbors_per_sample),
                                 cfg.eval.num_sample_steps, 0.0)
                ranks.append(slot_rankings(s, model.items.table, model.retrieval_metric,
                                           cfg.eval.neighbors_per_sample, (K,), history)[K])
                samples.append(s.cpu())
            else:
                top = model.recommend(history, mask, top_k=K + history.shape[1])
                ranks.append(drop_seen(top, history, K).cpu())
    return torch.cat(ranks), (torch.cat(samples, dim=1) if generative else None)


def sample_locations(samples: torch.Tensor, ds: dict, table: torch.Tensor) -> pd.DataFrame:
    """
    Ближайший центр кластера для каждого сэмпла: доли A, B, M своей тройки и прочих по группам.
    Центры — нормированные средние тёплых товаров кластера в таблице самой модели: сэмплы живут в её пространстве
    (у таблицы SASRec после PCA оно не совпадает с пространством генератора).
    """
    cluster = torch.from_numpy(ds['items']['cluster'].to_numpy().copy())
    warm, table = ~ds['new'], table.float().cpu()
    centers = torch.stack([table[(cluster == k) & warm].mean(0) for k in range(int(cluster.max()) + 1)])
    centers = torch.nn.functional.normalize(centers, dim=-1)
    nearest = (torch.nn.functional.normalize(samples, dim=-1) @ centers.T).argmax(-1).T.numpy()   # (n_users, n_samples)
    norms = samples.norm(dim=-1).T.numpy()
    users = ds['users']
    rows = []
    for name, sel in groups(users).items():
        own = [np.array(c) for c in users.loc[sel, 'clusters']]
        nr = nearest[sel]
        if name == 'одномод.':
            in_own = np.stack([nr[i] == own[i][0] for i in range(len(own))])
            rows.append({'группа': name, 'в своей моде': in_own.mean(), 'норма сэмпла': norms[sel].mean()})
            continue
        a = np.array([c[0] for c in own])[:, None]
        b = np.array([c[1] for c in own])[:, None]
        in_a, in_b, in_m = nr == a, nr == b, nr == b + 1          # кластеры тройки идут A, B, M подряд
        rows.append({'группа': name, 'A': in_a.mean(), 'B': in_b.mean(), 'M': in_m.mean(),
                     'прочие': 1 - (in_a | in_b | in_m).mean(), 'обе моды': (in_a.any(1) & in_b.any(1)).mean(),
                     'норма сэмпла': norms[sel].mean()})
    return pd.DataFrame(rows).set_index('группа')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configs', nargs='+', type=Path)
    parser.add_argument('--checkpoint', default='latest.pt', help='имя файла в checkpoints/<run_name>/')
    parser.add_argument('--run', help='имя прогона в output.dir вместо output.run_name (один конфиг)')
    parser.add_argument('--neighbors-per-sample', type=int,
                        help='S вместо eval.neighbors_per_sample из конфига; 50 — выдача из одного сэмпла (один вектор)')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets: dict[Path, dict] = {}
    for config in args.configs:
        spec, cfg = load_model_config(config)
        if args.neighbors_per_sample is not None:
            cfg.eval.neighbors_per_sample = args.neighbors_per_sample
        ckpt_path = Path(cfg.output.dir) / (args.run or cfg.output.run_name) / args.checkpoint
        if not ckpt_path.exists():
            print(f'\n{config.stem}: нет {ckpt_path} — пропускаю')
            continue
        data_dir = Path(cfg.data.data_dir)
        ds = datasets.setdefault(data_dir, load(data_dir))
        model = spec.build(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        ranks, samples = predict(model, cfg, device)
        print(f'\n{"=" * 100}\n{config.stem}: {args.checkpoint}, эпоха {ckpt["epoch"]}')
        print(format_table(method_table(ranks, ds)))
        if samples is not None:
            print(f'\nГде сэмплы ({samples.shape[0]} на пользователя): ближайший центр кластера')
            print(format_table(sample_locations(samples, ds, model.items.table)))


if __name__ == '__main__':
    main()
