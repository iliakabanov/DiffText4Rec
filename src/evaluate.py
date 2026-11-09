"""
Full-rank evaluation для sequential recommendation.

Метрики: HR@K (Hit Ratio) и NDCG@K (Normalized Discounted Cumulative Gain).
Оценка по всем товарам каталога (full-rank), не по sampled negatives.

Если в сплите есть колонка targets (протокол scripts/split_protocol.py), правильных ответов у
пользователя несколько, и метрики считаются по этому множеству F_u; добавляется Recall@K.

При eval.item_groups вместо них считаются только HR@K и Recall@K, отдельно по старым ответам (товар
был в train) и по новым (не было), и доля новых товаров в выдаче NewShare@K — для временного сплита
scripts/split_temporal.py (см. group_metrics).

Генеративная модель по умолчанию строит выдачу по одному сэмплу. При eval.neighbors_per_sample = S
выдача @N собирается из ceil(N / S) сэмплов, у каждого S мест (см. merge_slots): при S = 5 для @20
это 4 сэмпла, для @50 — 10. Сэмплы общие: @20 использует первые 4 из тех 10.

Full-rank protocol — стандарт в современных работах по seq-rec
(AlphaFuse, iDreamRec, SASRec). Sampled protocol даёт завышенные
и несопоставимые числа.

Оценка прогонов из командной строки (из корня проекта — пути в yaml относительные):

    python src/evaluate.py --config configs/atg/dreamrec.yaml --n-runs 5
    python src/evaluate.py --config ... --n-runs 3 --seeds 10 --splits test
    python src/evaluate.py --config configs/toys/sasrec_w5.yaml --checkpoint checkpoints/toys/sasrec_w5/best.pt
    python src/evaluate.py --self-test

Берутся best.pt прогонов <имя конфига>_1 … <имя конфига>_N из output.dir (так их называет
scripts/run_seeds.py). Каждый чекпойнт оценивается по протоколу тренера (batch_size, шаги
сэмплинга и w_cfg — из конфига) для сидов сэмплинга 1, 2, 3, ... Печатается таблица mean ± std по всем N × seeds запускам: строки —
способ ретривала (dot / cosine / L2) и случайный бейзлайн, столбцы — метрики. Все ретривалы
считаются по одному и тому же oracle-эмбеддингу.
"""

from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import math
from pathlib import Path

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


def set_metrics(rankings: torch.LongTensor, targets: torch.LongTensor, k: int) -> dict[str, float]:
    """
    HR@K, Recall@K и NDCG@K при нескольких правильных ответах у пользователя.

        HR     — доля пользователей, у которых в top-K есть хотя бы один ответ;
        Recall — |top-K ∩ F_u| / |F_u|;
        NDCG   — DCG / IDCG, где IDCG — все min(|F_u|, K) ответов на первых местах.

    Пользователи без ответов в среднее не входят: при exclude_seen все их ответы могли оказаться во входе.

    Args:
        rankings: (B, top_k), top_k >= k.
        targets:  (B, T) — индексы ответов, свободные места заполнены -1.
    """
    top_k = rankings[:, :k]                                              # (B, k)
    n_rel = (targets >= 0).sum(dim=-1)                                   # (B,)
    keep  = n_rel > 0
    top_k, targets, n_rel = top_k[keep], targets[keep], n_rel[keep]

    hits = (top_k.unsqueeze(-1) == targets.unsqueeze(1)).any(dim=-1)     # (B, k), bool
    discount = 1.0 / torch.log2(torch.arange(k, device=top_k.device, dtype=torch.float) + 2.0)
    ideal = discount.cumsum(0)[n_rel.clamp(max=k) - 1]                   # (B,)
    return {
        f'HR@{k}':     hits.any(dim=-1).float().mean().item(),
        f'Recall@{k}': (hits.sum(dim=-1) / n_rel).mean().item(),
        f'NDCG@{k}':   ((hits.float() * discount).sum(dim=-1) / ideal).mean().item(),
    }


ITEM_GROUPS = ('old', 'new')


def group_metrics(rankings: torch.LongTensor, targets: torch.LongTensor, new_items: torch.BoolTensor,
                  k: int) -> dict[str, float]:
    """
    HR@K и Recall@K отдельно по старым ответам (товар был в train) и по новым (не было), плюс NewShare@K —
    доля новых товаров в top-K по всем пользователям.

    Выдача одна и та же, меняются только ответы: у группы остаются ответы этой группы. Пользователь без
    ответов группы в её среднее не входит, так что HR@K_new — доля попаданий среди тех, у кого новый ответ есть.
    _new-метрики читаются вместе с NewShare@K: выдача из одних новых товаров тоже поднимает HR@K_new.

    Args:
        targets:   (B, T), свободные места -1.
        new_items: (n_items,) bool — True у товаров, которых нет в train (data.new_item_mask).
    """
    real = targets >= 0
    is_new = new_items[targets.clamp(min=0)] & real
    metrics: dict[str, float] = {}
    for group, in_group in zip(ITEM_GROUPS, (real & ~is_new, is_new)):
        m = set_metrics(rankings, targets.masked_fill(~in_group, -1), k)
        metrics[f'HR@{k}_{group}'] = m[f'HR@{k}']
        metrics[f'Recall@{k}_{group}'] = m[f'Recall@{k}']
    metrics[f'NewShare@{k}'] = new_items[rankings[:, :k]].float().mean().item()
    return metrics


def metric_names(top_k_values: tuple[int, ...], item_groups: bool) -> list[str]:
    """Имена метрик, которые вернёт evaluate(): по ним проверяется eval.select_metric до обучения."""
    if item_groups:
        return [name for k in top_k_values
                for name in [f'{m}@{k}_{g}' for g in ITEM_GROUPS for m in ('HR', 'Recall')] + [f'NewShare@{k}']]
    return [f'{m}@{k}' for k in top_k_values for m in ('HR', 'Recall', 'NDCG')]


def compute_metrics(rankings: torch.LongTensor, targets: torch.LongTensor,
                    top_k_values: tuple[int, ...], new_items: torch.BoolTensor | None = None) -> dict[str, float]:
    """
    targets (N,) — один ответ на пользователя, (N, T) — множество F_u (см. set_metrics).
    new_items — вместо общих метрик только HR и Recall по старым и новым товарам (см. group_metrics).
    """
    metrics: dict[str, float] = {}
    for k in top_k_values:
        if new_items is not None:
            metrics.update(group_metrics(rankings, targets if targets.dim() == 2 else targets[:, None], new_items, k))
        elif targets.dim() == 2:
            metrics.update(set_metrics(rankings, targets, k))
        else:
            metrics[f'HR@{k}']   = hit_ratio(rankings, targets, k)
            metrics[f'NDCG@{k}'] = ndcg(rankings, targets, k)
    return metrics


def drop_seen(rankings: torch.LongTensor, history: torch.LongTensor, k: int) -> torch.LongTensor:
    """
    Убирает из выдачи товары входа и оставляет первые k, сохраняя порядок остальных.
    Выдача должна быть длиннее k на длину окна истории: тогда k непросмотренных товаров найдутся всегда.
    """
    seen = (rankings.unsqueeze(-1) == history.unsqueeze(1)).any(dim=-1)  # (B, top_k), bool
    order = (~seen).float().argsort(dim=-1, descending=True, stable=True)[:, :k]
    return rankings.gather(1, order)


def batch_targets(batch: dict, history: torch.LongTensor, exclude_seen: bool) -> torch.LongTensor:
    """Ответы батча на устройстве history: (B, T) при колонке targets, иначе (B,)."""
    if 'targets' not in batch:
        if exclude_seen:
            raise ValueError('eval.exclude_seen требует колонку targets: сплит из scripts/split_protocol.py')
        return batch['target'].to(history.device)
    targets = batch['targets'].to(history.device)
    if exclude_seen:
        seen = (targets.unsqueeze(-1) == history.unsqueeze(1)).any(dim=-1)
        targets = targets.masked_fill(seen, -1)
    return targets


# --------------------------------------------------------------------------- #
# Выдача из нескольких сэмплов
# --------------------------------------------------------------------------- #

def merge_slots(candidates: torch.LongTensor, seen: torch.LongTensor | None, k: int) -> torch.LongTensor:
    """
    Выдача из k разных товаров по кругу: первые соседи сэмплов 1, 2, ..., K, затем вторые и так далее.
    Занятый другим сэмплом или просмотренный товар пропускается, и сэмпл берёт следующего своего соседа,
    поэтому число мест у сэмплов не зависит от совпадений, а масштаб скоров сэмплов не важен.

    Args:
        candidates: (K, B, D) — соседи каждого сэмпла по убыванию близости.
        seen:       (B, L) — товары, которых не должно быть в выдаче; None — таких нет.

    Returns:
        (B, k)
    """
    n_samples, n_users, _ = candidates.shape
    cand = candidates.tolist()
    seen_rows = seen.tolist() if seen is not None else [[] for _ in range(n_users)]
    lists = []
    for u in range(n_users):
        taken, cursor, row = set(seen_rows[u]), [0] * n_samples, []
        while len(row) < k:
            s = len(row) % n_samples
            neighbors = cand[s][u]
            while neighbors[cursor[s]] in taken:
                cursor[s] += 1
            taken.add(neighbors[cursor[s]])
            row.append(neighbors[cursor[s]])
        lists.append(row)
    return torch.tensor(lists, dtype=torch.long)


def slot_rankings(samples: torch.Tensor, items: torch.Tensor, metric: str, slots: int,
                  top_k_values: tuple[int, ...], seen: torch.LongTensor | None) -> dict[int, torch.LongTensor]:
    """
    Выдача для каждого K из первых ceil(K / slots) сэмплов.

    Сэмпл занимает не больше slots мест и пропускает не больше max(K) + L товаров: столько соседей и запрашивается.

    Args:
        samples: (n_samples, B, d), n_samples = ceil(max(K) / slots).
        items:   (N, d) — таблица товаров.
        seen:    (B, L) или None, см. merge_slots.

    Returns:
        {K: (B, K)} на CPU.
    """
    from models.dreamrec import retrieve        # локально: models тянет за собой whitening

    depth = slots + max(top_k_values) + (seen.shape[1] if seen is not None else 0)
    candidates = torch.stack([retrieve(s, items, metric, min(depth, items.shape[0])) for s in samples]).cpu()
    seen = seen.cpu() if seen is not None else None
    return {k: merge_slots(candidates[:math.ceil(k / slots)], seen, k) for k in top_k_values}


def draw_samples(model, history: torch.LongTensor, mask: torch.FloatTensor, n_samples: int,
                 num_steps: int, w_cfg: float) -> torch.Tensor:
    """n_samples независимых oracle-эмбеддингов на пользователя: (n_samples, B, d)."""
    cond = model.encode_history(history, mask)
    return torch.stack([model.sample(cond, num_steps=num_steps, w_cfg=w_cfg) for _ in range(n_samples)])


def _check_slots(model, slots: int | None) -> None:
    if slots is None:
        return
    if slots < 1:
        raise ValueError(f'eval.neighbors_per_sample должен быть ≥ 1, получено {slots}')
    if not hasattr(model, 'sample'):
        raise ValueError(f'eval.neighbors_per_sample требует генеративную модель, а {type(model).__name__} не сэмплирует')


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
    exclude_seen: bool = False,
    neighbors_per_sample: int | None = None,
    new_items: torch.BoolTensor | None = None,
) -> dict[str, float]:
    """
    Full-rank evaluation на одном split (val или test).

    Для каждого батча:
        1. model.recommend(): история → top-max(K) товаров по всему каталогу
           (у генеративных моделей — через сэмплинг oracle embedding).
        2. Считаем метрики для каждого K (см. compute_metrics).

    Args:
        model:            модель с recommend(history, mask, top_k, num_steps, w_cfg)
                          (SASRec, DreamRec, DreamRecText, DreamRecFMText).
        dataloader:       DataLoader для val или test split.
        top_k_values:     кортеж значений K для метрик.
        num_sample_steps: шагов сэмплинга (DDPM reverse или Euler ODE); SASRec игнорирует.
        w_cfg:            сила classifier-free guidance.
        device:           устройство.
        show_progress:    показывать tqdm прогресс-бар.
        exclude_seen:     убрать товары входа из выдачи и из F_u.
        neighbors_per_sample: S — выдача @K из ceil(K / S) сэмплов (см. merge_slots); None — один сэмпл.
        new_items:        (n_items,) bool на CPU — метрики по старым и новым товарам (см. group_metrics).

    Returns:
        dict: {'HR@10': ..., 'NDCG@10': ..., 'HR@20': ..., 'NDCG@20': ...}; при targets ещё Recall@K;
        при new_items — {'HR@K_old', 'Recall@K_old', 'HR@K_new', 'Recall@K_new', 'NewShare@K'}
    """
    _check_slots(model, neighbors_per_sample)
    model.eval()
    max_k = max(top_k_values)

    all_rankings: dict[int, list[torch.Tensor]] = {k: [] for k in top_k_values}
    all_targets:  list[torch.Tensor] = []

    iterator = tqdm(dataloader, desc='Eval', leave=False) if show_progress else dataloader

    for batch in iterator:
        history = batch['history'].to(device)   # (B, L)
        mask    = batch['mask'].to(device)      # (B, L)

        if neighbors_per_sample is not None:
            samples = draw_samples(model, history, mask, math.ceil(max_k / neighbors_per_sample),
                                   num_sample_steps, w_cfg)
            rankings = slot_rankings(samples, model.items.table, model.retrieval_metric, neighbors_per_sample,
                                     top_k_values, history if exclude_seen else None)
        else:
            # Генерируем oracle embedding и делаем retrieval
            single = model.recommend(
                history,
                mask,
                top_k=max_k + history.shape[1] if exclude_seen else max_k,
                num_steps=num_sample_steps,
                w_cfg=w_cfg,
            )                                   # (B, top_k)
            if exclude_seen:
                single = drop_seen(single, history, max_k)
            rankings = dict.fromkeys(top_k_values, single.cpu())

        for k in top_k_values:
            all_rankings[k].append(rankings[k])
        all_targets.append(batch_targets(batch, history, exclude_seen).cpu())

    targets_all = torch.cat(all_targets, dim=0)   # (N_total,) или (N_total, T)
    metrics: dict[str, float] = {}
    for k in top_k_values:
        metrics.update(compute_metrics(torch.cat(all_rankings[k], dim=0), targets_all, (k,), new_items))
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
    exclude_seen: bool = False,
    neighbors_per_sample: int | None = None,
    new_items: torch.BoolTensor | None = None,
) -> dict[str, dict[str, float]]:
    """
    Как evaluate(), но сразу по всем способам ретривала — для одного и того же сэмплинга
    oracle, поэтому они сравнимы между собой — плюс случайный бейзлайн.

    Для SASRec, который ранжирует товары сам, без oracle-эмбеддинга, строка одна.

    Returns:
        {строка таблицы: {метрика: значение}}
    """
    from models.dreamrec import retrieve        # локально: models тянет за собой whitening

    _check_slots(model, neighbors_per_sample)
    model.eval()
    max_k = max(top_k_values)
    generative = hasattr(model, 'items') and hasattr(model, 'sample')

    rankings: dict[str, dict[int, list[torch.Tensor]]] = {}
    targets:  list[torch.Tensor] = []

    iterator = tqdm(dataloader, desc='Eval', leave=False) if show_progress else dataloader
    for batch in iterator:
        history = batch['history'].to(device)
        mask    = batch['mask'].to(device)
        top_k   = max_k + history.shape[1] if exclude_seen else max_k

        def single_list(r: torch.Tensor) -> dict[int, torch.Tensor]:
            r = (drop_seen(r, history, max_k) if exclude_seen else r).cpu()
            return dict.fromkeys(top_k_values, r)

        if neighbors_per_sample is not None:
            samples = draw_samples(model, history, mask, math.ceil(max_k / neighbors_per_sample),
                                   num_sample_steps, w_cfg)
            batch_rankings = {name: slot_rankings(samples, model.items.table, metric, neighbors_per_sample,
                                                  top_k_values, history if exclude_seen else None)
                              for name, metric in RETRIEVAL_ROWS.items()}
        elif generative:
            oracle = model.sample(model.encode_history(history, mask),
                                  num_steps=num_sample_steps, w_cfg=w_cfg)
            batch_rankings = {name: single_list(retrieve(oracle, model.items.table, metric, top_k))
                              for name, metric in RETRIEVAL_ROWS.items()}
        else:
            batch_rankings = {type(model).__name__: single_list(model.recommend(
                history, mask, top_k=top_k, num_steps=num_sample_steps, w_cfg=w_cfg))}

        batch_rankings['Random'] = single_list(torch.stack(
            [torch.randperm(model.n_items, device=device)[:top_k] for _ in range(history.shape[0])]))

        for name, by_k in batch_rankings.items():
            for k, r in by_k.items():
                rankings.setdefault(name, {}).setdefault(k, []).append(r)
        targets.append(batch_targets(batch, history, exclude_seen).cpu())

    targets_all = torch.cat(targets)
    results: dict[str, dict[str, float]] = {}
    for name, by_k in rankings.items():
        results[name] = {}
        for k in top_k_values:
            results[name].update(compute_metrics(torch.cat(by_k[k]), targets_all, (k,), new_items))
    return results


# --------------------------------------------------------------------------- #
# Оценка прогонов
# --------------------------------------------------------------------------- #

def evaluate_runs(
    config_path: str,
    n_runs: int | None = None,
    n_seeds: int = 5,
    splits: tuple[str, ...] = ('val', 'test'),
    checkpoints: list[str] | None = None,
) -> dict:
    """
    Метрики best.pt прогонов <имя конфига>_1 … <имя конфига>_{n_runs} из output.dir по протоколу
    тренера, у каждого — n_seeds сидов сэмплинга. Вместо n_runs можно передать checkpoints —
    явные пути к чекпойнтам, тогда имена прогонов не важны.

    Имена прогонов — как у scripts/run_seeds.py: run_name = <имя yaml без расширения>_<i>.
    Модель собирается один раз по конфигу (builders.MODELS[model_type]), веса чекпойнтов грузятся
    в неё по очереди — архитектура в yaml должна совпадать со всеми прогонами.

    Returns:
        {'model', 'runs': [{'name', 'epoch', 'saved_metrics'}], 'seeds', 'retrieval_row',
         'results': {split: {строка таблицы: {метрика: [n_runs × n_seeds значений, прогон за прогоном]}}}}
    """
    # Импорт здесь, а не наверху: train сам импортирует evaluate
    from builders import load_model_config
    from data import build_dataloaders, new_item_mask
    from train import seeded

    if (n_runs is None) == (checkpoints is None):
        raise ValueError('нужен ровно один из n_runs и checkpoints')
    spec, cfg = load_model_config(config_path)
    if checkpoints is not None:
        paths = [Path(p) for p in checkpoints]
    else:
        prefix = Path(config_path).stem
        paths  = [Path(cfg.output.dir) / f'{prefix}_{i}' / 'best.pt' for i in range(1, n_runs + 1)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f'нет чекпойнтов: {missing}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = spec.build(cfg).to(device)
    loaders = build_dataloaders(cfg.data.data_dir, batch_size=cfg.data.batch_size, num_workers=0,
                                seq_len=cfg.data.max_seq_len, padding_idx=cfg.data.padding_idx,
                                splits=splits)
    new_items = new_item_mask(cfg.data.data_dir, cfg.data.n_items) if cfg.eval.item_groups else None
    seeds = list(range(1, n_seeds + 1))

    runs = []
    results: dict[str, dict[str, dict[str, list[float]]]] = {split: {} for split in splits}
    for path in paths:
        name = path.parent.name
        ckpt = torch.load(path, map_location='cpu', mmap=True)
        model.load_state_dict(ckpt['model_state'])  # строго: архитектура в yaml должна совпадать
        model.eval()

        saved_cfg = ckpt.get('model_cfg') or {}
        if saved_cfg:                               # у старых чекпойнтов SASRec он пустой
            diff = {k: (saved_cfg.get(k), v) for k, v in model.config().items() if saved_cfg.get(k) != v}
            if diff:
                print(f'{name}: конфиг отличается от чекпойнта (в чекпойнте → в конфиге): {diff}')
        runs.append({'name': name, 'epoch': ckpt.get('epoch'), 'saved_metrics': ckpt.get('metrics')})
        del ckpt

        for split in splits:
            for seed in seeds:
                with seeded(seed, device):
                    run = evaluate_retrievals(model, loaders[split], top_k_values=cfg.eval.top_k,
                                              num_sample_steps=cfg.eval.num_sample_steps or 1,
                                              w_cfg=getattr(cfg.model, 'w_cfg', 0.0),
                                              device=device, show_progress=False,
                                              exclude_seen=cfg.eval.exclude_seen,
                                              neighbors_per_sample=cfg.eval.neighbors_per_sample,
                                              new_items=new_items)
                for row, metrics in run.items():
                    for metric, value in metrics.items():
                        results[split].setdefault(row, {}).setdefault(metric, []).append(value)
        print(f'{name} (эпоха {runs[-1]["epoch"]}): готово', flush=True)

    # Строка таблицы, соответствующая ретривалу из конфига (с ним считались метрики при обучении);
    # у SASRec строка одна — ранжирует сама модель по своему retrieval_metric
    by_metric = {metric: row for row, metric in RETRIEVAL_ROWS.items()}
    retrieval_row = (by_metric.get(getattr(cfg.model, 'retrieval_metric', None))
                     if hasattr(model, 'sample') else None)
    return {'model': cfg.model_type, 'runs': runs, 'seeds': seeds, 'retrieval_row': retrieval_row,
            'select_metric': cfg.eval.select_metric, 'results': results}


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


def _print_report(report: dict) -> None:
    seeds, runs, key = report['seeds'], report['runs'], report['select_metric']
    n_seeds, n_runs = len(seeds), len(runs)
    print(f'\n{report["model"]} | прогонов: {n_runs} — '
          + ', '.join(f'{r["name"]} (эпоха {r["epoch"]})' for r in runs))
    print(f'Сиды сэмплинга: {seeds} → запусков на сплит: {n_runs * n_seeds}')

    for split, rows in report['results'].items():
        print(f'\n=== {split.upper()} === mean ± std по всем запускам ({n_runs} × {n_seeds})')
        print(_format_table(rows))
        row = report.get('retrieval_row') or next(iter(rows))
        if key not in rows[row]:
            continue
        # Значения идут прогон за прогоном, по n_seeds на прогон
        values  = rows[row][key]
        per_run = [_mean_std(values[i * n_seeds:(i + 1) * n_seeds])[0] for i in range(n_runs)]
        print(f'{key} ({row}) по прогонам (среднее по сидам): '
              + ', '.join(f'{r["name"]} {m:.4f}' for r, m in zip(runs, per_run)))
        if n_runs > 1:
            # Запуски одного прогона не независимы, поэтому ДИ — по средним прогонов
            mean, std = _mean_std(per_run)
            half = _t_quantile_975(n_runs - 1) * std / math.sqrt(n_runs)
            print(f'95% ДИ для {key} ({row}) по средним прогонов: [{mean - half:.4f}, {mean + half:.4f}]')

    saved = [f'{r["name"]} {r["saved_metrics"][key]:.4f}'
             for r in runs if key in (r['saved_metrics'] or {})]
    if saved:
        print(f'\nДля справки, val {key} из чекпойнтов (сид тренера 42, ретривал из его конфига): '
              + ', '.join(saved))
    print('std учитывает и разброс между обучениями, и шум сэмплинга.')


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
        description='Метрики прогонов модели на val/test по протоколу тренера с разбросом '
                    'по прогонам и сидам сэмплинга.')
    parser.add_argument('--config', help='yaml, с которым обучалась модель (configs/<dataset>/<model>.yaml)')
    parser.add_argument('--n-runs', type=int, help='сколько прогонов взять: best.pt из <model>_1 … <model>_N')
    parser.add_argument('--checkpoint', nargs='+', help='вместо --n-runs: пути к конкретным чекпойнтам')
    parser.add_argument('--seeds', type=int, default=5, help='число сидов сэмплинга: 1, 2, 3, ... (по умолчанию 5)')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'], choices=['val', 'test'])
    parser.add_argument('--self-test', action='store_true', help='проверить формулы HR/NDCG и выйти')
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.config or bool(args.n_runs) == bool(args.checkpoint):
        parser.error('нужны --config и ровно один из --n-runs / --checkpoint (или --self-test)')

    report = evaluate_runs(args.config, n_runs=args.n_runs, n_seeds=args.seeds, splits=tuple(args.splits),
                           checkpoints=args.checkpoint)
    _print_report(report)


if __name__ == '__main__':
    main()
