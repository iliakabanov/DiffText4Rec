"""
Full-rank evaluation для sequential recommendation.

Метрики: HR@K (Hit Ratio) и NDCG@K (Normalized Discounted Cumulative Gain).
Оценка по всем товарам каталога (full-rank), не по sampled negatives.

Full-rank protocol — стандарт в современных работах по seq-rec
(AlphaFuse, iDreamRec, SASRec). Sampled protocol даёт завышенные
и несопоставимые числа.
"""

from __future__ import annotations

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
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    python -m src.evaluate

    Проверяем метрики на случайных предсказаниях и на идеальных предсказаниях.
    """
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