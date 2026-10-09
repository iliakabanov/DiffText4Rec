"""
Sanity-проверка текстовых эмбеддингов товаров: несут ли они сигнал о переходах пользователя.

Считается по test_data.df (один пример на пользователя: история → next) и матрице эмбеддингов,
строка которой соответствует строке items.parquet.

    1. Косинус «последний товар истории → next» против случайных пар товаров. Если текстовое
       пространство хоть что-то знает о переходах, первое распределение должно быть заметно правее.
    2. Повторы в тесте: как часто next — это последний товар истории, любой товар истории или
       почти-дубликат товара истории (косинус > DUPLICATE_COS). Это характеристика датасета:
       она задаёт силу тривиального бейзлайна «повтори последнее».
    3. Text-only ретривал без обучения: по эмбеддингу запроса берём top-K ближайших товаров
       каталога и смотрим, попал ли туда next. Запрос — последний товар истории или нормированное
       среднее history; кандидаты — в трёх режимах: всё, без товара-запроса, без всей истории.
       Разница между режимами показывает, сколько в метрике приходится на повторы, а сколько на
       настоящий семантический перенос. Ориентиры: случайный выбор (K / n_items) и top-K популярных.
    4. Ближайшие соседи по названиям — проверить глазами.

    python scripts/text_sanity.py
    python scripts/text_sanity.py --column text_raw --top-k 20
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import preprocess as P    # noqa: E402  — пути и параметры предобработки

MODEL         = 'qwen3-embedding-0.6b'
SEED          = 0
N_RANDOM      = 200_000   # случайных пар товаров для фона
N_EXAMPLES    = 5         # товаров для ручной проверки соседей
CHUNK         = 2048      # строк теста за один матмул
DUPLICATE_COS = 0.98      # выше этого косинуса товары считаем почти-дубликатами


def load(column: str):
    emb_path = P.OUT_DIR / f'{MODEL}_{column}.pickle'
    items = pd.read_parquet(P.OUT_DIR / 'items.parquet', columns=['parent_asin', column])
    test  = pd.read_pickle(P.OUT_DIR / 'test_data.df')
    with open(emb_path, 'rb') as f:
        emb = pickle.load(f)
    if emb.shape[0] != len(items):
        raise ValueError(f'{emb_path.name}: {emb.shape[0]:,} строк против {len(items):,} товаров')
    titles = items[column].str.split('\n').str[0].str.removeprefix('title: ')
    print(f'{emb_path.name}: {emb.shape}, товаров {len(items):,}, примеров в test {len(test):,}')
    return emb, titles, test


def cosine_check(emb: torch.Tensor, test: pd.DataFrame) -> None:
    """Косинус «последний товар истории → next» против случайных пар."""
    rng = np.random.default_rng(SEED)
    last = torch.tensor([s[-1] for s in test['seq']], device=emb.device)
    nxt  = torch.tensor(test['next'].to_numpy(), device=emb.device)
    pair = (emb[last] * emb[nxt]).sum(-1).cpu().numpy()

    i, j = (torch.tensor(rng.integers(0, emb.shape[0], N_RANDOM), device=emb.device) for _ in range(2))
    rand = (emb[i] * emb[j]).sum(-1).cpu().numpy()

    q = [5, 25, 50, 75, 95]
    print('\n1. Косинус между эмбеддингами')
    print(f'   последний → next : ' + ', '.join(f'p{p} {v:.3f}' for p, v in zip(q, np.percentile(pair, q)))
          + f', среднее {pair.mean():.3f}')
    print(f'   случайные пары   : ' + ', '.join(f'p{p} {v:.3f}' for p, v in zip(q, np.percentile(rand, q)))
          + f', среднее {rand.mean():.3f}')
    p95 = np.percentile(rand, 95)
    print(f'   доля пар «последний → next» выше p95 случайных ({p95:.3f}): {(pair > p95).mean():.1%} '
          f'(у случайных пар по определению 5%)')
    print(f'   AUC (вероятность, что настоящая пара ближе случайной): '
          f'{(pair[:, None] > rand[None, rng.integers(0, N_RANDOM, 2000)]).mean():.3f}')


def _flat_history(seqs: list, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Все товары историй одним вектором и номер строки-владельца для каждого из них."""
    lens = [len(s) for s in seqs]
    items = torch.tensor(np.concatenate([np.asarray(s) for s in seqs]), device=device)
    owner = torch.tensor(np.repeat(np.arange(len(seqs)), lens), device=device)
    return items, owner


@torch.no_grad()
def repeats_check(emb: torch.Tensor, test: pd.DataFrame) -> None:
    """Насколько часто next — повтор того, что уже было в истории."""
    seqs = list(test['seq'])
    nxt = torch.tensor(test['next'].to_numpy(), device=emb.device)
    last = torch.tensor([s[-1] for s in seqs], device=emb.device)
    items, owner = _flat_history(seqs, emb.device)

    same_last = (nxt == last).float().mean().item()
    in_hist = torch.zeros(len(test), device=emb.device)      # scatter_reduce не работает с bool на cuda
    in_hist.scatter_reduce_(0, owner, (items == nxt[owner]).float(), reduce='amax')

    cos = (emb[items] * emb[nxt[owner]]).sum(-1)
    max_cos = torch.full((len(test),), -1.0, device=emb.device)
    max_cos.scatter_reduce_(0, owner, cos, reduce='amax')

    print('\n2. Повторы в тесте (тривиальный бейзлайн «повтори последнее»)')
    print(f'   next = последний товар истории:            {same_last:.4f}')
    print(f'   next встречается где-то в истории:         {in_hist.mean().item():.4f}')
    print(f'   next почти-дубликат товара истории (cos > {DUPLICATE_COS}): '
          f'{(max_cos > DUPLICATE_COS).float().mean().item():.4f}')


@torch.no_grad()
def retrieval_check(emb: torch.Tensor, test: pd.DataFrame, top_k: int) -> None:
    """Text-only ретривал: два запроса × три набора кандидатов."""
    n_items = emb.shape[0]
    nxt = torch.tensor(test['next'].to_numpy(), device=emb.device)
    seqs = list(test['seq'])
    modes = ('всё', 'без товара-запроса', 'без истории')
    hits = {q: dict.fromkeys(modes, 0) for q in ('последний товар истории', 'среднее по истории')}

    for start in range(0, len(test), CHUNK):
        batch = seqs[start:start + CHUNK]
        target = nxt[start:start + CHUNK]
        last = torch.tensor([s[-1] for s in batch], device=emb.device)
        items, owner = _flat_history(batch, emb.device)
        mean = torch.zeros(len(batch), emb.shape[1], device=emb.device)
        mean.index_add_(0, owner, emb[items])
        queries = {'последний товар истории': emb[last],
                   'среднее по истории': torch.nn.functional.normalize(mean, dim=-1)}
        rows = torch.arange(len(batch), device=emb.device)

        for name, q in queries.items():
            scores = q @ emb.T                                  # (B, n_items)
            for mode in modes:                                  # маскируем всё больше кандидатов
                if mode == 'без товара-запроса':
                    scores[rows, last] = -np.inf
                elif mode == 'без истории':
                    scores[owner, items] = -np.inf
                top = scores.topk(top_k, dim=-1).indices
                hits[name][mode] += (top == target[:, None]).any(-1).sum().item()

    # Ориентиры: случайный выбор и самые популярные товары (по train)
    train = pd.read_pickle(P.OUT_DIR / 'train_data.df')
    pop = pd.Series(np.concatenate([np.concatenate(train['seq'].to_numpy()), train['next'].to_numpy()]))
    popular = torch.tensor(pop.value_counts().index[:top_k].to_numpy(), device=emb.device)
    pop_hit = (nxt[:, None] == popular[None, :]).any(-1).float().mean().item()

    print(f'\n3. Text-only ретривал, HR@{top_k} (без обучения); кандидаты:')
    print(f'   {"запрос":26s}' + ''.join(f'{m:>20s}' for m in modes))
    for name, by_mode in hits.items():
        print(f'   {name:26s}' + ''.join(f'{by_mode[m] / len(test):>20.4f}' for m in modes))
    print(f'   {"top-" + str(top_k) + " популярных (train)":26s}{pop_hit:>20.4f}')
    print(f'   {"случайный выбор":26s}{top_k / n_items:>20.4f}')


@torch.no_grad()
def neighbours(emb: torch.Tensor, titles: pd.Series, test: pd.DataFrame) -> None:
    """Ближайшие товары по тексту — проверить глазами."""
    rng = np.random.default_rng(SEED)
    print('\n4. Ближайшие соседи по тексту (и куда пользователь пошёл на самом деле)')
    rows = rng.choice(len(test), N_EXAMPLES, replace=False)
    for r in rows:
        seq, nxt = test['seq'].iloc[r], int(test['next'].iloc[r])
        q = int(seq[-1])
        scores = emb[q] @ emb.T
        scores[q] = -np.inf
        top = scores.topk(3).indices.tolist()
        print(f'\n   последний товар истории: {titles.iloc[q][:95]}')
        for j in top:
            print(f'      {scores[j].item():.3f}  {titles.iloc[j][:95]}')
        print(f'   фактический next (cos {(emb[q] @ emb[nxt]).item():.3f}): {titles.iloc[nxt][:95]}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Sanity-проверка текстовых эмбеддингов.')
    parser.add_argument('--column', default='text', choices=['text', 'text_raw'])
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    emb_np, titles, test = load(args.column)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emb = torch.nn.functional.normalize(torch.from_numpy(emb_np).to(device), dim=-1)

    cosine_check(emb, test)
    repeats_check(emb, test)
    retrieval_check(emb, test, args.top_k)
    neighbours(emb, titles, test)


if __name__ == '__main__':
    main()
