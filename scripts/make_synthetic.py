"""
Синтетический датасет для проверки гипотезы: одновекторная модель (SASRec на тексте) не может выдать две моды
интересов, если между ними лежат другие товары, а генеративная модель — может.

Пространство товаров — единичная сфера в ℝ^d; центры кластеров — векторы случайного ортонормированного базиса.

    Тройка k (A_k, B_k, M_k) лежит в плоскости (u_k, v_k) = (q_{2k}, q_{2k+1}):
        a_k = u_k,  b_k = cos Δ_k · u_k + sin Δ_k · v_k,
        m_k = норм(a_k + b_k)            при --distractor between (помеха на биссектрисе),
        m_k = q_{2T + B + k}             при --distractor random  (контроль: помеха ортогональна всему).
    Фоновые кластеры — следующие B векторов базиса. Разные тройки и фон ортогональны точно.

    Товар кластера с центром c: норм(c + σ·ξ), ξ ~ N(0, I/d); доля new_share товаров каждого кластера — новые,
    в train их нет. Id товаров перемешаны, кластер по id не угадать.

    Пользователь — веса над кластерами: двухмодальный (π на A_k, 1 − π на B_k) или одномодальный (1 на одном
    кластере из всех). В каждой своей моде у него личная точка p = норм(c + σ_u·ξ), и товар моды выбирается
    с вероятностью ∝ exp(cos(e, p) / κ): смесь пиков фон Мизеса — Фишера.
    Товар: сначала мода по весам, потом товар моды. История — history_len старых товаров без повторов, порядок
    случайный (последовательной структуры нет); ответы F_u — n_answers товаров среди всех непросмотренных
    (при old_only_answers — только среди старых: проверка без холодного старта на тех же данных обучения).

Формат как у scripts/split_temporal.py:
    train_data.df — seq = история[:-1], next = история[-1];  val_data.df — вся история, targets = F_u;
    data_statis.df, item_embeddings.pickle (N, d) float32 — на сфере радиуса √d (дисперсия координаты ≈ 1),
    items_meta.df (кластер, роль A/B/M/background, тройка, Δ, is_new), users_meta.df (тип, тройка, веса, точки),
    params.json — параметры генерации.
Оба варианта --distractor с одним сидом совпадают во всём, кроме положения центров M.

    python scripts/make_synthetic.py --distractor between
    python scripts/make_synthetic.py --distractor random
    python scripts/make_synthetic.py --distractor random --kappa inf    # контроль: моды равномерны
    python scripts/make_synthetic.py --distractor between --old-only-answers   # тот же train, в ответах только старые
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from preprocess import SPLIT_COLUMNS   # noqa: E402

EVAL_COLUMNS = [*SPLIT_COLUMNS, 'targets']
ROLES        = ('A', 'B', 'M', 'background')
DISTRACTORS  = ('between', 'random')


@dataclass(frozen=True)
class SyntheticConfig:
    d:                 int = 64
    n_triples:         int = 10
    deltas_deg:        tuple[int, ...] = (30, 60, 90, 120, 150)   # по n_triples / len(deltas_deg) тройки на угол
    n_background:      int = 10
    items_per_cluster: int = 200
    sigma:             float = 0.2      # разброс товаров вокруг центра: ≈ 11°
    new_share:         float = 0.3
    n_users:           int = 20_000
    bimodal_share:     float = 0.4
    pi:                float = 0.5      # вес моды A у двухмодальных
    sigma_u:           float = 0.2      # разброс личной точки вокруг центра моды
    kappa:             float = 0.002    # N_eff моды ≈ 11 товаров (см. обсуждение κ)
    history_len:       int = 20
    n_answers:         int = 5
    old_only_answers:  bool = False     # ответы только из старых товаров; случайный поток тот же — train не меняется
    seed:              int = 0

    @property
    def n_clusters(self) -> int:
        return 3 * self.n_triples + self.n_background


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def mode_weights(item_emb: np.ndarray, point: np.ndarray, kappa: float) -> np.ndarray:
    """P(товар | мода) ∝ exp(cos(e, p) / κ) по товарам кластера; эмбеддинги и точка единичные. κ = ∞ — равномерно."""
    if np.isinf(kappa):
        return np.full(len(item_emb), 1.0 / len(item_emb))
    logits = item_emb @ point / kappa
    w = np.exp(logits - logits.max())
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Кластеры и товары
# --------------------------------------------------------------------------- #

def make_clusters(cfg: SyntheticConfig, distractor: str, rng: np.random.Generator) -> tuple[np.ndarray, pd.DataFrame]:
    """Центры (C, d) и таблица кластеров: role, triple, delta (−1 у фона). Порядок: A_k, B_k, M_k по тройкам, затем фон."""
    if distractor not in DISTRACTORS:
        raise ValueError(f'distractor должен быть одним из {DISTRACTORS}, получено {distractor!r}')
    if cfg.n_triples % len(cfg.deltas_deg):
        raise ValueError(f'n_triples={cfg.n_triples} не делится на число углов {len(cfg.deltas_deg)}')
    if 3 * cfg.n_triples + cfg.n_background > cfg.d:
        raise ValueError(f'нужно {3 * cfg.n_triples + cfg.n_background} базисных векторов, а d={cfg.d}')

    Q, _ = np.linalg.qr(rng.standard_normal((cfg.d, cfg.d)))
    basis = Q.T                                                          # строки — ортонормированный базис
    T, B = cfg.n_triples, cfg.n_background
    deltas = np.repeat(cfg.deltas_deg, T // len(cfg.deltas_deg))

    centers, rows = [], []
    for k, delta in enumerate(deltas):
        u, v = basis[2 * k], basis[2 * k + 1]
        a = u
        b = np.cos(np.radians(delta)) * u + np.sin(np.radians(delta)) * v
        m = normalize(a + b) if distractor == 'between' else basis[2 * T + B + k]
        centers += [a, b, m]
        rows += [('A', k, delta), ('B', k, delta), ('M', k, delta)]
    for j in range(B):
        centers.append(basis[2 * T + j])
        rows.append(('background', -1, -1))
    meta = pd.DataFrame(rows, columns=['role', 'triple', 'delta'])
    meta.insert(0, 'cluster', np.arange(len(meta)))
    return np.stack(centers), meta


def make_items(cfg: SyntheticConfig, centers: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Эмбеддинги (N, d), кластер товара (N,), is_new (N,). Id товаров — случайная перестановка."""
    n = cfg.items_per_cluster
    emb = normalize(np.repeat(centers, n, axis=0)
                    + cfg.sigma * rng.standard_normal((len(centers) * n, cfg.d)) / np.sqrt(cfg.d))
    cluster = np.repeat(np.arange(len(centers)), n)
    n_new = round(cfg.new_share * n)
    is_new = np.concatenate([rng.permutation(n) < n_new for _ in centers])   # ровно n_new новых в каждом кластере
    order = rng.permutation(len(emb))
    return emb[order].astype(np.float32), cluster[order], is_new[order]


# --------------------------------------------------------------------------- #
# Пользователи
# --------------------------------------------------------------------------- #

def _draw(weights: np.ndarray, allowed: np.ndarray, rng: np.random.Generator) -> int:
    """Индекс по весам среди allowed; расходует ровно одно случайное число, поэтому поток не зависит от весов."""
    cum = np.cumsum(weights * allowed)
    return int(np.searchsorted(cum, rng.random() * cum[-1], side='right').clip(max=len(cum) - 1))


def make_users(cfg: SyntheticConfig, clusters: pd.DataFrame, centers: np.ndarray, emb: np.ndarray,
               item_cluster: np.ndarray, is_new: np.ndarray,
               rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train, val и users_meta. Типы сбалансированы: тройки и кластеры одномодальных — по кругу."""
    members = {c: np.flatnonzero(item_cluster == c) for c in clusters['cluster']}
    a_of = clusters[clusters['role'] == 'A'].set_index('triple')['cluster']
    b_of = clusters[clusters['role'] == 'B'].set_index('triple')['cluster']

    n_bi = round(cfg.bimodal_share * cfg.n_users)
    train_rows, val_rows, meta_rows = [], [], []
    for i, uid in enumerate(rng.permutation(cfg.n_users)):
        if i < n_bi:
            k = i % cfg.n_triples
            modes, weights, kind = [a_of[k], b_of[k]], [cfg.pi, 1 - cfg.pi], 'bimodal'
        else:
            modes, weights, kind = [(i - n_bi) % cfg.n_clusters], [1.0], 'unimodal'
        points = normalize(np.stack([centers[c] for c in modes])
                           + cfg.sigma_u * rng.standard_normal((len(modes), cfg.d)) / np.sqrt(cfg.d))
        item_w = [mode_weights(emb[members[c]], p, cfg.kappa) for c, p in zip(modes, points)]
        old = [~is_new[members[c]] for c in modes]
        free = [np.ones(len(members[c]), dtype=bool) for c in modes]

        def take(only_old: bool) -> int:
            m = int(rng.choice(len(modes), p=weights))
            j = _draw(item_w[m], free[m] & old[m] if only_old else free[m], rng)
            free[m][j] = False
            return int(members[modes[m]][j])

        history = [take(only_old=True) for _ in range(cfg.history_len)]
        answers = [take(only_old=cfg.old_only_answers) for _ in range(cfg.n_answers)]
        user = f'u{uid:05d}'
        train_rows.append((user, history[:-1], len(history) - 1, history[-1]))
        val_rows.append((user, history, len(history), answers[0], answers))
        row = clusters.loc[modes[0]]
        meta_rows.append((user, kind, int(row['triple']), int(row['delta']), modes, weights, points.astype(np.float32)))

    by_user = lambda rows, cols: pd.DataFrame(rows, columns=cols).sort_values('user_id', ignore_index=True)
    return (by_user(train_rows, SPLIT_COLUMNS), by_user(val_rows, EVAL_COLUMNS),
            by_user(meta_rows, ['user_id', 'kind', 'triple', 'delta', 'clusters', 'weights', 'points']))


# --------------------------------------------------------------------------- #
# Сборка
# --------------------------------------------------------------------------- #

def generate(cfg: SyntheticConfig, distractor: str) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    centers, clusters = make_clusters(cfg, distractor, rng)
    emb, item_cluster, is_new = make_items(cfg, centers, rng)
    train, val, users = make_users(cfg, clusters, centers, emb, item_cluster, is_new, rng)
    items = clusters.iloc[item_cluster][['cluster', 'role', 'triple', 'delta']].reset_index(drop=True)
    items.insert(0, 'item', np.arange(len(items)))
    items['is_new'] = is_new
    return {'train': train, 'val': val, 'items': items, 'users': users, 'emb': emb, 'centers': centers}


def save(out: dict[str, object], cfg: SyntheticConfig, distractor: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out['train'].to_pickle(out_dir / 'train_data.df')
    out['val'].to_pickle(out_dir / 'val_data.df')
    pd.DataFrame({'seq_size': [cfg.history_len], 'item_num': [len(out['emb'])]}).to_pickle(out_dir / 'data_statis.df')
    with open(out_dir / 'item_embeddings.pickle', 'wb') as f:
        # Сфера радиуса √d: дисперсия координаты ≈ 1, как у шума DDPM; порядок соседей тот же, что у единичной
        pickle.dump((out['emb'] * np.sqrt(cfg.d)).astype(np.float32), f)
    out['items'].to_pickle(out_dir / 'items_meta.df')
    out['users'].to_pickle(out_dir / 'users_meta.df')
    np.save(out_dir / 'cluster_centers.npy', out['centers'].astype(np.float32))
    (out_dir / 'params.json').write_text(json.dumps({**dataclasses.asdict(cfg), 'distractor': distractor}, indent=2),
                                         encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--distractor', choices=DISTRACTORS, required=True)
    parser.add_argument('--kappa', type=float, default=SyntheticConfig.kappa,
                        help='острота пика моды; inf — товар моды выбирается равномерно (контроль)')
    parser.add_argument('--old-only-answers', action='store_true',
                        help='ответы val только из старых товаров (train тот же); суффикс папки _oldtest')
    parser.add_argument('--out', type=Path,
                        help='по умолчанию data/processed/synthetic_<distractor>, при --kappa inf — с суффиксом _uniform')
    parser.add_argument('--seed', type=int, default=SyntheticConfig.seed)
    args = parser.parse_args()

    cfg = SyntheticConfig(seed=args.seed, kappa=args.kappa, old_only_answers=args.old_only_answers)
    suffix = ('_uniform' if np.isinf(cfg.kappa) else '') + ('_oldtest' if cfg.old_only_answers else '')
    out_dir = args.out or ROOT / f'data/processed/synthetic_{args.distractor}{suffix}'
    out = generate(cfg, args.distractor)
    save(out, cfg, args.distractor, out_dir)

    items, users = out['items'], out['users']
    print(f'{out_dir}: товаров {len(items):,} ({items["is_new"].sum():,} новых), кластеров {cfg.n_clusters}, '
          f'пользователей {len(users):,} ({(users["kind"] == "bimodal").sum():,} двухмодальных), d={cfg.d}')


if __name__ == '__main__':
    main()
