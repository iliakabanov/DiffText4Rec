"""
ZCA-whitening для матрицы текстовых эмбеддингов товаров.

Зачем нужно
-----------
TEM-эмбеддинги (OpenAI text-embedding-3-small) сильно анизотропны: они
заполняют узкий конус в R^d, среднее далеко от нуля, ковариация не
диагональная. Для conditional flow matching из N(0, I) это плохо — поток
вынужден проходить через "пустую" часть пространства до того, как
попасть на манифольд данных.

Whitening переводит эмбеддинги в пространство с нулевым средним и единичной
ковариацией. После него N(0, I) — это естественный prior, поток проходит
по прямой к данным, не теряя времени.

Решение — ZCA whitening:
    W = U @ diag(1/sqrt(s + eps)) @ U.T,
где U, s — из SVD(Sigma). Симметричное (W = W.T) и максимально близкое
к identity, что сохраняет семантическую структуру эмбеддингов.

Альтернатива (PCA whitening) поворачивает координаты, что портит
интерпретируемость dot-product retrieval'а.

Использование
-------------
Whitening параметры (mu, W) вычисляются один раз и хранятся как buffers
внутри FlowMatcher — они автоматически попадают в state_dict при сохранении
модели (source of truth).

Дополнительно есть save_whitening / load_whitening для кэширования на диск,
чтобы не пересчитывать SVD при каждом первом создании модели (10-20s для
матрицы 12000 x 3072). Кэш — это технический оптимизатор, а не source of truth.

    # Первый запуск:
    mu, W = compute_whitening(emb_raw)
    save_whitening(cache_path, mu, W)

    # Последующие запуски (если model state_dict ещё не существует):
    mu, W = load_whitening(cache_path)

    emb_whitened = apply_whitening(emb_raw, mu, W)
    # дальше mu, W кладутся в model как register_buffer
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------- #
# Core whitening
# --------------------------------------------------------------------------- #

def compute_whitening(
    emb: np.ndarray,
    eps: float = 1e-8,
    exclude_indices: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет параметры ZCA-whitening: среднее mu и матрицу W.

    После применения:
        e_white = (e - mu) @ W
        mean(e_white) ≈ 0,  cov(e_white) ≈ I.

    Args:
        emb:             (N, D) — матрица эмбеддингов.
        eps:             регуляризация для устойчивости (добавляется к собств. значениям
                         перед взятием обратного корня; защищает от деления на ~0
                         в малозначимых направлениях).
        exclude_indices: индексы строк, которые НЕ учитывать при вычислении статистик.
                         Типичный случай — padding-token. Если None, используются все строки.

    Returns:
        mu: (D,)   — среднее, float64.
        W:  (D, D) — whitening матрица, float64.

    Notes:
        - SVD считается через ковариационную матрицу (D, D), а не через сами данные —
          это эффективнее при N >> D и численно стабильно для PSD матрицы.
        - Возвращаемый dtype = float64. Конвертация в float32 при необходимости —
          на стороне вызывающего кода (для GPU).
    """
    if emb.ndim != 2:
        raise ValueError(f'Expected 2D matrix, got shape {emb.shape}.')

    # Подмножество для статистик
    if exclude_indices is not None:
        exclude = np.asarray(list(exclude_indices), dtype=np.int64)
        keep_mask = np.ones(emb.shape[0], dtype=bool)
        keep_mask[exclude] = False
        emb_for_stats = emb[keep_mask]
    else:
        emb_for_stats = emb

    if emb_for_stats.shape[0] < 2:
        raise ValueError(
            f'Not enough rows for whitening: have {emb_for_stats.shape[0]} '
            f'(need at least 2). Check exclude_indices.'
        )

    # Считаем в float64 для устойчивости SVD.
    emb_f64 = emb_for_stats.astype(np.float64, copy=False)

    mu = emb_f64.mean(axis=0)                              # (D,)
    centered = emb_f64 - mu                                # (N, D)
    n = centered.shape[0]
    # Ковариация без bias correction (как делают iDreamRec/AlphaFuse).
    sigma = (centered.T @ centered) / n                    # (D, D)

    # SVD ковариационной матрицы. Для PSD матрицы svd ≡ eigendecomposition
    # с неотрицательными собственными значениями, отсортированными по убыванию.
    U, s, _ = np.linalg.svd(sigma)

    # ZCA whitening: W = U diag(1/sqrt(s + eps)) U^T
    inv_sqrt = 1.0 / np.sqrt(s + eps)                      # (D,)
    W = (U * inv_sqrt) @ U.T                               # (D, D)

    return mu, W


def apply_whitening(
    emb: np.ndarray,
    mu: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """
    Применяет whitening к матрице эмбеддингов:
        e_white = (e - mu) @ W.

    Args:
        emb: (N, D) или (D,).
        mu:  (D,)
        W:   (D, D)

    Returns:
        Эмбеддинги той же формы и dtype, что и emb (сохраняется dtype входа).

    Notes:
        - Если emb во float32, операция тоже идёт во float32; ZCA достаточно
          устойчив, чтобы это не вызывало проблем.
        - Применяется ко ВСЕМ строкам, включая те, что были исключены при
          compute_whitening. Это сознательно: padding-token (если он был
          исключён) после whitening превратится из нулей в (0 - mu) @ W,
          который, скорее всего, окажется на типичной норме whitened данных.
          Чтобы padding оставался нулевым, обнуляйте его строку явно после
          whitening (это делается в encoder'е через nn.Embedding).
    """
    input_dtype = emb.dtype
    e = emb.astype(np.float32, copy=False) if input_dtype != np.float64 else emb
    mu_ = mu.astype(e.dtype, copy=False)
    W_ = W.astype(e.dtype, copy=False)

    result = (e - mu_) @ W_
    return result.astype(input_dtype, copy=False)


def inverse_whitening(
    z: np.ndarray,
    mu: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """
    Обратное преобразование: z @ W^{-1} + mu.

    Полезно для отладки и интерпретации сгенерированных эмбеддингов.
    В training/inference пайплайне не используется.

    Args:
        z:  (N, D) или (D,).
        mu: (D,)
        W:  (D, D)

    Returns:
        Эмбеддинги той же формы и dtype, что и z.
    """
    input_dtype = z.dtype
    z_f = z.astype(np.float64, copy=False)
    W_inv = np.linalg.inv(W.astype(np.float64, copy=False))
    e = z_f @ W_inv + mu.astype(np.float64)
    return e.astype(input_dtype, copy=False)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def verify_whitening(
    emb_whitened: np.ndarray,
    exclude_indices: Iterable[int] | None = None,
) -> dict[str, float]:
    """
    Sanity check: вычисляет фактические среднее и ковариацию whitened-эмбеддингов.
    Идеальный результат:
        mean_abs_max ≈ 0
        cov_diag_mean ≈ 1
        cov_offdiag_abs_mean ≈ 0

    Args:
        emb_whitened:    (N, D) — whitened эмбеддинги.
        exclude_indices: какие строки исключать (по умолчанию те же, что и при
                         compute_whitening — обычно padding).

    Returns:
        dict с метриками:
            'mean_abs_max':         max(|mean per dim|)        — должен быть ~0
            'mean_abs_mean':        mean(|mean per dim|)       — должен быть ~0
            'cov_diag_mean':        среднее по диагонали cov   — должно быть ~1
            'cov_diag_std':         std по диагонали cov       — должен быть ~0
            'cov_offdiag_abs_mean': mean(|off-diagonal|) cov   — должен быть ~0
            'cov_offdiag_abs_max':  max(|off-diagonal|) cov    — должен быть ~0
            'frobenius_dist_to_I':  ||cov - I||_F              — должен быть ~0
    """
    if exclude_indices is not None:
        exclude = np.asarray(list(exclude_indices), dtype=np.int64)
        keep_mask = np.ones(emb_whitened.shape[0], dtype=bool)
        keep_mask[exclude] = False
        emb = emb_whitened[keep_mask]
    else:
        emb = emb_whitened

    emb = emb.astype(np.float64, copy=False)
    n, d = emb.shape

    mean = emb.mean(axis=0)                                # (D,)
    centered = emb - mean
    cov = (centered.T @ centered) / n                      # (D, D)

    diag = np.diag(cov)
    # Off-diagonal: маска через identity.
    offdiag_mask = ~np.eye(d, dtype=bool)
    offdiag = cov[offdiag_mask]

    return {
        'mean_abs_max':         float(np.abs(mean).max()),
        'mean_abs_mean':        float(np.abs(mean).mean()),
        'cov_diag_mean':        float(diag.mean()),
        'cov_diag_std':         float(diag.std()),
        'cov_offdiag_abs_mean': float(np.abs(offdiag).mean()),
        'cov_offdiag_abs_max':  float(np.abs(offdiag).max()),
        'frobenius_dist_to_I':  float(np.linalg.norm(cov - np.eye(d))),
    }


# --------------------------------------------------------------------------- #
# I/O (для дискового кэша; source of truth — state_dict модели)
# --------------------------------------------------------------------------- #

def save_whitening(path: str | Path, mu: np.ndarray, W: np.ndarray) -> None:
    """Кэширует параметры whitening в .npz, чтобы не пересчитывать SVD."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mu=mu, W=W)


def load_whitening(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Загружает кэш параметров whitening из .npz."""
    data = np.load(path)
    return data['mu'], data['W']


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    Sanity check на синтетических данных:
        python -m src.whitening
    """
    rng = np.random.default_rng(42)
    N, D = 1000, 64

    # Генерируем анизотропные данные: смещённое среднее + неединичная ковариация.
    A = rng.standard_normal((D, D))
    cov_target = A @ A.T / D                                # PSD
    L = np.linalg.cholesky(cov_target)
    base = rng.standard_normal((N, D))
    emb = base @ L.T + rng.standard_normal(D) * 3.0         # смещаем
    emb = emb.astype(np.float32)

    print(f'Input emb: shape={emb.shape}, dtype={emb.dtype}')
    print(f'  Input mean abs max: {np.abs(emb.mean(axis=0)).max():.4f}')
    print(f'  Input cov diag mean: {np.cov(emb, rowvar=False).diagonal().mean():.4f}')

    mu, W = compute_whitening(emb)
    print(f'\nComputed: mu shape={mu.shape}, W shape={W.shape}')

    emb_w = apply_whitening(emb, mu, W)
    print(f'Whitened: shape={emb_w.shape}, dtype={emb_w.dtype}')

    stats = verify_whitening(emb_w)
    print(f'\nVerification:')
    for k, v in stats.items():
        print(f'  {k:25s} {v:+.6f}')

    # Round-trip
    emb_back = inverse_whitening(emb_w, mu, W)
    max_err = np.abs(emb - emb_back).max()
    print(f'\nRound-trip max error: {max_err:.6e}')
    assert max_err < 1e-4, 'Round-trip should be near-exact'
    print('Round-trip OK')