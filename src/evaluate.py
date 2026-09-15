"""
Full-rank evaluation для sequential recommendation.

Метрики: HR@K (Hit Ratio) и NDCG@K (Normalized Discounted Cumulative Gain).
Оценка по всем товарам каталога (full-rank), не по sampled negatives.

Full-rank protocol — стандарт в современных работах по seq-rec
(AlphaFuse, iDreamRec, SASRec). Sampled protocol даёт завышенные
и несопоставимые числа.

Оценка чекпойнта из командной строки (из корня проекта — пути в yaml относительные):

    python src/evaluate.py --config configs/dreamrec.yaml --checkpoint checkpoints/dreamrec_1/best.pt
    python src/evaluate.py --config ... --checkpoint ... --seeds 10 --splits test
    python src/evaluate.py --self-test

Метрики считаются по протоколу тренера (batch_size, шаги сэмплинга и w_cfg — из конфига)
для сидов сэмплинга 42, 142, 242, ...; сид 42 воспроизводит val/test из лога обучения.
Печатается таблица mean ± std: строки — способ ретривала (dot / cosine / L2) и случайный
бейзлайн, столбцы — метрики. Все ретривалы считаются по одному и тому же oracle-эмбеддингу.
"""

from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Metric functions
# --------------------------------------------------------------------------- #

def hit_ratio(rankings: torch.LongTensor, targets: torch.LongTensor, k: int) -> float:
    """
    HR@K = доля пользователей, у которых target попал в top-K.

    Args:
        rankings: (B, top_k) — предсказанные индексы товаров, отсортированные
                  по убыванию score. top_k >= k.
        targets:  (B,) — истинные целевые индексы.
        k:        порог.

    Returns:
        HR@K как float в [0, 1].
    """
    # Берём только первые k позиций
    top_k = rankings[:, :k]                             # (B, k)
    # Для каждого пользователя: попал ли target в top-k?
    hits = (top_k == targets.unsqueeze(-1)).any(dim=-1) # (B,), bool
    return hits.float().mean().item()


def ndcg(rankings: torch.LongTensor, targets: torch.LongTensor, k: int) -> float:
    """
    NDCG@K = среднее по пользователям 1/log2(rank+1) если target в top-K, иначе 0.

    При одном релевантном документе ideal DCG = 1/log2(2) = 1.
    Поэтому NDCG@K = DCG@K / IDCG@K = DCG@K (нормировка уже учтена).

    Args:
        rankings: (B, top_k) — предсказанные индексы.
        targets:  (B,) — истинные индексы.
        k:        порог.

    Returns:
        NDCG@K как float в [0, 1].
    """
    top_k = rankings[:, :k]                             # (B, k)
    # Находим позицию target в top-k (0-based); если нет — -1
    matches = (top_k == targets.unsqueeze(-1))          # (B, k), bool

    # Для каждого пользователя находим rank (0-based) первого совпадения
    # Если совпадений нет — rank будет k (за пределами top-K)
    rank = matches.float().argmax(dim=-1)               # (B,)
    hit_mask = matches.any(dim=-1)                      # (B,), bool

    # NDCG = 1 / log2(rank + 2) для попаданий, 0 для промахов
    # +2 потому что rank 0-based, а формула использует 1-based: log2(rank+1)
    gains = 1.0 / torch.log2(rank.float() + 2.0)       # (B,)
    gains = gains * hit_mask.float()                    # нули для промахов

    return gains.mean().item()


# --------------------------------------------------------------------------- #
# Full-rank evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    top_k_values: tuple[int, ...] = (10, 20),
    num_sample_steps: int = 50,
    w_cfg: float = 1.0,
    device: str | torch.device = 'cuda',
    show_progress: bool = True,
) -> dict[str, float]:
    """
    Full-rank evaluation на одном split (val или test).

    Для каждого батча:
        1. model.recommend(): история → top-max(K) товаров по всему каталогу
           (у генеративных моделей — через сэмплинг oracle embedding).
        2. Считаем HR@K и NDCG@K для каждого K.

    Args:
        model:            модель с recommend(history, mask, top_k, num_steps, w_cfg)
                          (SASRec, DreamRec, DreamRecText, DreamRecFMText).
        dataloader:       DataLoader для val или test split.
        top_k_values:     кортеж значений K для метрик.
        num_sample_steps: шагов сэмплинга (DDPM reverse или Euler ODE); SASRec игнорирует.
        w_cfg:            сила classifier-free guidance.
        device:           устройство.
        show_progress:    показывать tqdm прогресс-бар.

    Returns:
        dict: {'HR@10': ..., 'NDCG@10': ..., 'HR@20': ..., 'NDCG@20': ...}
    """
    model.eval()
    max_k = max(top_k_values)

    all_rankings: list[torch.Tensor] = []
    all_targets:  list[torch.Tensor] = []

    iterator = tqdm(dataloader, desc='Eval', leave=False) if show_progress else dataloader

    for batch in iterator:
        history = batch['history'].to(device)   # (B, L)
        mask    = batch['mask'].to(device)      # (B, L)
        target  = batch['target'].to(device)    # (B,)

        # Генерируем oracle embedding и делаем retrieval
        rankings = model.recommend(
            history,
            mask,
            top_k=max_k,
            num_steps=num_sample_steps,
            w_cfg=w_cfg,
        )                                       # (B, max_k)

        all_rankings.append(rankings.cpu())
        all_targets.append(target.cpu())

    # Конкатенируем по всему split'у
    rankings_all = torch.cat(all_rankings, dim=0)   # (N_total, max_k)
    targets_all  = torch.cat(all_targets,  dim=0)   # (N_total,)

    # Считаем метрики
    metrics = {}
    for k in top_k_values:
        metrics[f'HR@{k}']   = hit_ratio(rankings_all, targets_all, k)
        metrics[f'NDCG@{k}'] = ndcg(rankings_all, targets_all, k)

    return metrics


# --------------------------------------------------------------------------- #
# Evaluation по всем способам ретривала
# --------------------------------------------------------------------------- #

# Строка таблицы → метрика ретривала в models.dreamrec.retrieve
RETRIEVAL_ROWS = {'Dot Product': 'dot', 'Cosine Sim': 'cosine', 'L2 Distance': 'l2'}


@torch.no_grad()
def evaluate_retrievals(
    model,
    dataloader: DataLoader,
    top_k_values: tuple[int, ...] = (10, 20),
    num_sample_steps: int = 50,
    w_cfg: float = 0.0,
    device: str | torch.device = 'cuda',
    show_progress: bool = False,
) -> dict[str, dict[str, float]]:
    """
    Как evaluate(), но сразу по всем способам ретривала — для одного и того же сэмплинга
    oracle, поэтому они сравнимы между собой — плюс случайный бейзлайн.

    Для SASRec, который ранжирует товары сам, без oracle-эмбеддинга, строка одна.

    Returns:
        {строка таблицы: {метрика: значение}}
    """
    from models.dreamrec import retrieve        # локально: models тянет за собой whitening

    model.eval()
    max_k = max(top_k_values)
    generative = hasattr(model, 'items') and hasattr(model, 'sample')

    rankings: dict[str, list[torch.Tensor]] = {}
    targets:  list[torch.Tensor] = []

    iterator = tqdm(dataloader, desc='Eval', leave=False) if show_progress else dataloader
    for batch in iterator:
        history = batch['history'].to(device)
        mask    = batch['mask'].to(device)
        target  = batch['target'].to(device)

        if generative:
            oracle = model.sample(model.encode_history(history, mask),
                                  num_steps=num_sample_steps, w_cfg=w_cfg)
            batch_rankings = {name: retrieve(oracle, model.items.table, metric, max_k)
                              for name, metric in RETRIEVAL_ROWS.items()}
        else:
            batch_rankings = {type(model).__name__: model.recommend(
                history, mask, top_k=max_k, num_steps=num_sample_steps, w_cfg=w_cfg)}

        batch_rankings['Random'] = torch.stack(
            [torch.randperm(model.n_items, device=device)[:max_k] for _ in range(target.shape[0])])

        for name, r in batch_rankings.items():
            rankings.setdefault(name, []).append(r.cpu())
        targets.append(target.cpu())

    targets_all = torch.cat(targets)
    return {
        name: {f'{prefix}@{k}': fn(torch.cat(rs), targets_all, k)
               for k in top_k_values for prefix, fn in (('HR', hit_ratio), ('NDCG', ndcg))}
        for name, rs in rankings.items()
    }


# --------------------------------------------------------------------------- #
# Оценка чекпойнта
# --------------------------------------------------------------------------- #

def evaluate_checkpoint(
    config_path: str,
    checkpoint_path: str,
    n_seeds: int = 5,
    splits: tuple[str, ...] = ('val', 'test'),
) -> dict:
    """
    Метрики чекпойнта по протоколу тренера для n_seeds сидов сэмплинга.

    Модель собирается по тому же конфигу, что и при обучении (builders.MODELS[model_type]).

    Returns:
        {'model', 'epoch', 'saved_metrics', 'seeds', 'retrieval_row',
         'results': {split: {строка таблицы: {метрика: [по сидам]}}}}
    """
    # Импорт здесь, а не наверху: train сам импортирует evaluate
    from builders import load_model_config
    from data import build_dataloaders
    from train import EVAL_SEED, seeded

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location='cpu', mmap=True)
    spec, cfg = load_model_config(config_path)

    model = spec.build(cfg)
    model.load_state_dict(ckpt['model_state'])      # строго: архитектура в yaml должна совпадать
    model = model.to(device).eval()

    saved_cfg = ckpt.get('model_cfg') or {}
    if saved_cfg:                                   # у старых чекпойнтов SASRec он пустой
        diff = {k: (saved_cfg.get(k), v) for k, v in model.config().items() if saved_cfg.get(k) != v}
        if diff:
            print(f'Конфиг отличается от чекпойнта (в чекпойнте → в конфиге): {diff}')

    loaders = build_dataloaders(cfg.data.data_dir, batch_size=cfg.data.batch_size, num_workers=0,
                                seq_len=cfg.data.max_seq_len, splits=splits)
    seeds = [EVAL_SEED + 100 * i for i in range(n_seeds)]
    results = {}
    for split in splits:
        runs = []
        for seed in seeds:
            with seeded(seed, device):
                runs.append(evaluate_retrievals(model, loaders[split], top_k_values=cfg.eval.top_k,
                                                num_sample_steps=cfg.eval.num_sample_steps or 1,
                                                w_cfg=getattr(cfg.model, 'w_cfg', 0.0),
                                                device=device, show_progress=False))
        results[split] = {row: {metric: [run[row][metric] for run in runs] for metric in runs[0][row]}
                          for row in runs[0]}

    # Строка таблицы, соответствующая ретривалу из конфига (с ним считались метрики при обучении);
    # у SASRec строка одна — ранжирует сама модель по своему retrieval_metric
    by_metric = {metric: row for row, metric in RETRIEVAL_ROWS.items()}
    retrieval_row = (by_metric.get(getattr(cfg.model, 'retrieval_metric', None))
                     if hasattr(model, 'sample') else None)
    return {'model': cfg.model_type, 'epoch': ckpt.get('epoch'), 'saved_metrics': ckpt.get('metrics'),
            'seeds': seeds, 'retrieval_row': retrieval_row,
            'results': results}


def _t_quantile_975(df: int) -> float:
    """Квантиль 0.975 t-распределения (для 95% доверительного интервала)."""
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        return table.get(df, 1.96)


def _mean_std(values: list[float]) -> tuple[float, float]:
    n    = len(values)
    mean = sum(values) / n
    std  = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return mean, std


def _format_table(rows: dict[str, dict[str, list[float]]]) -> str:
    """Таблица 'mean ± std': строки — способ ретривала, столбцы — метрики."""
    names   = list(rows)
    metrics = list(next(iter(rows.values())))
    cells   = {metric: [f'{_mean_std(rows[name][metric])[0]:.4f} ± {_mean_std(rows[name][metric])[1]:.4f}'
                        for name in names]
               for metric in metrics}
    try:
        import pandas as pd
        df = pd.DataFrame(cells, index=names)
        df.index.name = 'Retrieval'
        return df.to_string()
    except ImportError:
        width = max(len(n) for n in names) + 2
        head  = f'{"Retrieval":<{width}s}' + '  '.join(f'{m:>15s}' for m in metrics)
        body  = '\n'.join(f'{name:<{width}s}' + '  '.join(f'{cells[m][i]:>15s}' for m in metrics)
                          for i, name in enumerate(names))
        return f'{head}\n{body}'


def _print_report(report: dict, checkpoint_path: str) -> None:
    seeds, n = report['seeds'], len(report['seeds'])
    print(f'\n{report["model"]} | {checkpoint_path} | эпоха {report["epoch"]} | '
          f'{n} сидов сэмплинга: {seeds}')

    for split, rows in report['results'].items():
        print(f'\n=== {split.upper()} ===')
        print(_format_table(rows))
        row = report.get('retrieval_row') or next(iter(rows))
        if n > 1 and 'NDCG@20' in rows[row]:
            mean, std = _mean_std(rows[row]['NDCG@20'])
            half = _t_quantile_975(n - 1) * std / math.sqrt(n)
            print(f'95% ДИ для NDCG@20 ({row}): [{mean - half:.4f}, {mean + half:.4f}]')

    saved = report['saved_metrics'] or {}
    val   = report['results'].get('val')
    if val and 'NDCG@20' in saved:
        row = report.get('retrieval_row') or next(iter(val))
        print(f'\nПроверка: val NDCG@20 ({row}) при сиде {seeds[0]} = {val[row]["NDCG@20"][0]:.4f}, '
              f'в чекпойнте {saved["NDCG@20"]:.4f}')
    print('Разброс отражает только шум сэмплинга: различия между обучениями с разными сидами он не учитывает.')


# --------------------------------------------------------------------------- #
# Самопроверка формул метрик
# --------------------------------------------------------------------------- #

def _self_test() -> None:
    """Проверяем метрики на случайных и на идеальных предсказаниях."""
    torch.manual_seed(0)

    N_USERS = 1000
    N_ITEMS = 500
    TOP_K = 20

    # Test 1: случайные предсказания — HR и NDCG должны быть ~K/N_ITEMS
    print('Test 1: random rankings')
    rankings_random = torch.stack([
        torch.randperm(N_ITEMS)[:TOP_K] for _ in range(N_USERS)
    ])                                                  # (N_USERS, TOP_K)
    targets = torch.randint(0, N_ITEMS, (N_USERS,))

    hr10_random = hit_ratio(rankings_random, targets, k=10)
    hr20_random = hit_ratio(rankings_random, targets, k=20)
    ndcg10_random = ndcg(rankings_random, targets, k=10)
    ndcg20_random = ndcg(rankings_random, targets, k=20)

    expected_hr10 = 10 / N_ITEMS
    expected_hr20 = 20 / N_ITEMS
    print(f'  HR@10:   {hr10_random:.4f}  (expected ~{expected_hr10:.4f})')
    print(f'  HR@20:   {hr20_random:.4f}  (expected ~{expected_hr20:.4f})')
    print(f'  NDCG@10: {ndcg10_random:.4f}')
    print(f'  NDCG@20: {ndcg20_random:.4f}')
    # Допускаем 50% отклонение от ожидания (N_USERS=1000, дисперсия высокая)
    assert abs(hr10_random - expected_hr10) < expected_hr10, \
        f'HR@10 сильно отличается от ожидаемого: {hr10_random:.4f} vs {expected_hr10:.4f}'
    assert abs(hr20_random - expected_hr20) < expected_hr20, \
        f'HR@20 сильно отличается от ожидаемого: {hr20_random:.4f} vs {expected_hr20:.4f}'
    print('  random rankings sanity check  OK')

    # Test 2: идеальные предсказания — HR и NDCG должны быть 1.0
    print('\nTest 2: perfect rankings (target always at rank 0)')
    rankings_perfect = torch.zeros(N_USERS, TOP_K, dtype=torch.long)
    rankings_perfect[:, 0] = targets                    # target на первом месте

    hr10_perfect = hit_ratio(rankings_perfect, targets, k=10)
    hr20_perfect = hit_ratio(rankings_perfect, targets, k=20)
    ndcg10_perfect = ndcg(rankings_perfect, targets, k=10)
    ndcg20_perfect = ndcg(rankings_perfect, targets, k=20)

    assert hr10_perfect == 1.0, f'Perfect HR@10 should be 1.0, got {hr10_perfect}'
    assert hr20_perfect == 1.0
    assert abs(ndcg10_perfect - 1.0) < 1e-5, f'Perfect NDCG@10 should be 1.0, got {ndcg10_perfect}'
    assert abs(ndcg20_perfect - 1.0) < 1e-5
    print(f'  HR@10:   {hr10_perfect:.4f}  OK')
    print(f'  HR@20:   {hr20_perfect:.4f}  OK')
    print(f'  NDCG@10: {ndcg10_perfect:.4f}  OK')
    print(f'  NDCG@20: {ndcg20_perfect:.4f}  OK')

    # Test 3: target на последнем месте в top-K
    print('\nTest 3: target at last position in top-K')
    rankings_last = torch.zeros(N_USERS, TOP_K, dtype=torch.long)
    # Заполняем произвольными товарами, target кладём в конец
    for i in range(N_USERS):
        other = (targets[i].item() + 1) % N_ITEMS
        rankings_last[i] = torch.full((TOP_K,), other, dtype=torch.long)
        rankings_last[i, -1] = targets[i]              # target на позиции TOP_K-1

    hr10_last = hit_ratio(rankings_last, targets, k=10)
    hr20_last = hit_ratio(rankings_last, targets, k=20)
    ndcg10_last = ndcg(rankings_last, targets, k=10)
    ndcg20_last = ndcg(rankings_last, targets, k=20)

    # При K=20, target на позиции 19 (0-based) → HR@20=1.0, HR@10=0.0
    assert hr10_last == 0.0,  f'HR@10 should be 0.0 (target at pos 19), got {hr10_last}'
    assert hr20_last == 1.0,  f'HR@20 should be 1.0 (target at pos 19), got {hr20_last}'
    # NDCG@20 = 1/log2(20+1) ≈ 0.231
    expected_ndcg20_last = 1.0 / math.log2(TOP_K + 1)
    assert abs(ndcg20_last - expected_ndcg20_last) < 1e-5, \
        f'NDCG@20 should be {expected_ndcg20_last:.4f}, got {ndcg20_last:.4f}'
    print(f'  HR@10:   {hr10_last:.4f}  (expected 0.0)  OK')
    print(f'  HR@20:   {hr20_last:.4f}  (expected 1.0)  OK')
    print(f'  NDCG@10: {ndcg10_last:.4f}  (expected 0.0)  OK')
    print(f'  NDCG@20: {ndcg20_last:.4f}  (expected {expected_ndcg20_last:.4f})  OK')

    # Test 4: NDCG убывает с ростом rank
    print('\nTest 4: NDCG decreases as rank increases')
    ndcg_values = []
    for rank in range(TOP_K):
        r = torch.full((1, TOP_K), 999, dtype=torch.long)  # все не-target
        r[0, rank] = 0                                  # target на позиции rank
        t = torch.tensor([0])
        ndcg_values.append(ndcg(r, t, k=TOP_K))

    for i in range(len(ndcg_values) - 1):
        assert ndcg_values[i] > ndcg_values[i + 1], \
            f'NDCG должен убывать: pos {i} → {ndcg_values[i]:.4f}, pos {i+1} → {ndcg_values[i+1]:.4f}'
    print(f'  NDCG@{TOP_K} by rank: {[f"{v:.3f}" for v in ndcg_values[:5]]}...  OK')

    # Test 5: батч из одного пользователя с точным попаданием
    print('\nTest 5: single user, exact hit')
    r = torch.tensor([[42, 1, 2, 3]])   # target=42 на первом месте
    t = torch.tensor([42])
    assert hit_ratio(r, t, k=1) == 1.0
    assert hit_ratio(r, t, k=4) == 1.0
    assert abs(ndcg(r, t, k=1) - 1.0) < 1e-5
    print(f'  hit_ratio and ndcg correct for single user  OK')

    print('\nAll checks passed')


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Метрики чекпойнта на val/test по протоколу тренера с разбросом по сидам сэмплинга.')
    parser.add_argument('--config', help='yaml, с которым обучалась модель (configs/...)')
    parser.add_argument('--checkpoint', help='путь к best.pt или latest.pt')
    parser.add_argument('--seeds', type=int, default=5, help='число сидов сэмплинга: 42, 142, 242, ... (по умолчанию 5)')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'], choices=['val', 'test'])
    parser.add_argument('--self-test', action='store_true', help='проверить формулы HR/NDCG и выйти')
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.config or not args.checkpoint:
        parser.error('нужны --config и --checkpoint (или --self-test)')

    report = evaluate_checkpoint(args.config, args.checkpoint, n_seeds=args.seeds,
                                 splits=tuple(args.splits))
    _print_report(report, args.checkpoint)


if __name__ == '__main__':
    main()
