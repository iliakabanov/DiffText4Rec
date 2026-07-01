"""
PCA-проекция и нормализация эмбеддингов товаров для flow matching.

Проблема с 3072d
-----------------
TEM-эмбеддинги (OpenAI text-embedding-3-large) имеют размерность 3072.
Flow matching в 3072d требует огромного количества данных — на 15529
sequences модель не может выучить распределение в таком пространстве.

Решение — PCA-проекция в n_components=256d:
    1. Центрирование: e_c = e - mean(e).
    2. PCA: берём топ-n_components главных компонент.
    3. Проекция: e_pca = e_c @ V_top  (N, n_components).
    4. Нормализация: e_norm = e_pca / scale.

Два режима нормализации (флаг per_component в compute_whitening):
    per_component=False (по умолчанию):
        scale — скаляр (глобальное std по всем n_components измерениям).
        Var(E) после: неодинакова по компонентам (max >> min).

    per_component=True (метод авторов iDreamRec, адаптированный):
        scale — вектор (n_components,), каждый компонент делится на свой std.
        Var(E) после: = I_{n_components} точно.
        При любом alpha_t шумовой процесс DDIM: Var(E^t) = I.

Retrieval делается в PCA-пространстве — эффективен потому что PCA
максимизирует дисперсию и сохраняет ближайших соседей лучше чем
случайная проекция.

Хранение параметров
-------------------
Параметры (mu, V, scale) вычисляются один раз, хранятся как buffers
в FlowMatcher (state_dict) и кэшируются на диск.

    mu, V, scale = compute_whitening(emb_raw, n_components=256)
    emb_pca = apply_whitening(emb_raw, mu, V, scale)
    # emb_pca shape: (N, 256)

apply_whitening, inverse_whitening и I/O работают с обоими режимами
прозрачно: scale формы (1,) или (n_components,) — broadcasting справляется.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------- #
# Core: PCA + normalization
# --------------------------------------------------------------------------- #

def compute_whitening(
    emb: np.ndarray,
    n_components: int = 256,
    eps: float = 1e-8,
    per_component: bool = False,
    exclude_indices: Iterable[int] | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Вычисляет параметры PCA-проекции и нормализации.

    Преобразование:
        e_pca  = (e - mu) @ V        # проекция в n_components-мерное пространство
        e_norm = e_pca / scale       # нормализация

    Args:
        emb:           (N, D) — матрица эмбеддингов в исходном пространстве.
        n_components:  число главных компонент. По умолчанию 256.
                       Рекомендуемые значения: 64, 128, 256, 512.
        eps:           малая константа для численной стабильности scale.
        per_component: если False (по умолчанию) — scale скаляр (глобальный std),
                       Var(E) после неодинакова по компонентам.
                       если True — scale вектор (n_components,), каждый компонент
                       нормируется на свой std → Var(E) = I_{n_components} точно.
        exclude_indices: строки, не учитываемые при расчёте статистик.
        verbose:       печатать диагностику.

    Returns:
        mu:    (D,)                          — среднее в исходном пространстве.
        V:     (D, n_components)             — матрица главных компонент (PCA basis).
        scale: (1,) или (n_components,)      — масштаб (глобальный или поэлементный).

    Notes:
        - SVD вычисляется через ковариационную матрицу (D, D) — это эффективно
          когда N > D (наш случай: 11924 > 3072).
        - V ортогональная: V.T @ V = I_{n_components}.
        - apply_whitening и inverse_whitening работают с обоими форматами scale
          прозрачно через broadcasting.
        - Обратная проекция: e_orig ≈ e_pca @ V.T * scale + mu
          (приближённая, т.к. PCA отбрасывает малые компоненты).
    """
    if emb.ndim != 2:
        raise ValueError(f'Expected 2D matrix, got shape {emb.shape}.')

    D = emb.shape[1]
    if n_components > D:
        raise ValueError(f'n_components={n_components} > D={D}.')

    # Подмножество для статистик
    if exclude_indices is not None:
        exclude = np.asarray(list(exclude_indices), dtype=np.int64)
        keep_mask = np.ones(emb.shape[0], dtype=bool)
        keep_mask[exclude] = False
        emb_for_stats = emb[keep_mask]
    else:
        emb_for_stats = emb

    if emb_for_stats.shape[0] < n_components:
        raise ValueError(
            f'Not enough rows: have {emb_for_stats.shape[0]}, need at least {n_components}.'
        )

    emb_f64 = emb_for_stats.astype(np.float64, copy=False)
    N = emb_f64.shape[0]

    # Шаг 1: центрирование
    mu = emb_f64.mean(axis=0)                          # (D,)
    centered = emb_f64 - mu                            # (N, D)

    # Шаг 2: SVD ковариационной матрицы → главные компоненты
    sigma = (centered.T @ centered) / N                # (D, D)
    U, s, _ = np.linalg.svd(sigma)                    # U: (D, D), s: (D,)

    # Топ-n_components главных компонент
    V = U[:, :n_components]                            # (D, n_components)

    # Шаг 3: проекция и нормализация
    projected = centered @ V                           # (N, n_components)

    if per_component:
        # Каждый компонент делится на свой std → Var(E) = I_{n_components}
        scale = projected.std(axis=0) + eps            # (n_components,)
    else:
        # Один глобальный std на все компоненты
        scale = np.array([float(projected.std()) + eps])  # (1,)

    if verbose:
        var_explained = s[:n_components].sum() / s.sum() * 100
        print(f'  PCA: {D}d -> {n_components}d')
        print(f'  Variance explained: {var_explained:.1f}%')
        print(f'  per_component={per_component}')
        emb_norm = projected / scale
        dim_var = emb_norm.var(axis=0)
        print(f'  After projection: mean={emb_norm.mean():.6f}, '
              f'per-dim var: mean={dim_var.mean():.4f} '
              f'min={dim_var.min():.4f} max={dim_var.max():.4f}')
        print(f'  Row norm mean: {np.linalg.norm(emb_norm, axis=1).mean():.4f}')
        print(f'    (expected ~{n_components**0.5:.1f} for N(0,1) in {n_components}d)')

    return mu, V, scale


def apply_whitening(
    emb: np.ndarray,
    mu: np.ndarray,
    V: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Применяет PCA-проекцию и нормализацию:
        e_pca  = (e - mu) @ V
        e_norm = e_pca / scale

    Args:
        emb:   (N, D) или (D,) — исходные эмбеддинги.
        mu:    (D,)             — среднее.
        V:     (D, n_components) — PCA basis.
        scale: (1,) или (n_components,) — масштаб (глобальный или поэлементный).
                                          Оба формата работают через broadcasting.

    Returns:
        (N, n_components) или (n_components,) — нормализованные эмбеддинги.
        dtype: float32.
    """
    e = emb.astype(np.float64, copy=False)
    mu_ = mu.astype(np.float64, copy=False)
    V_ = V.astype(np.float64, copy=False)
    s = scale.astype(np.float64)          # (1,) или (n_components,) — broadcasting
    result = ((e - mu_) @ V_) / s
    return result.astype(np.float32)


def inverse_whitening(
    z: np.ndarray,
    mu: np.ndarray,
    V: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Приближённое обратное преобразование:
        e_approx = z * scale @ V.T + mu

    Приближённое, так как PCA отбрасывает малые компоненты.
    Используется только для интерпретации, не для retrieval.

    Args:
        z:     (N, n_components) или (n_components,)
        mu:    (D,)
        V:     (D, n_components)
        scale: (1,) или (n_components,) — оба формата работают через broadcasting.

    Returns:
        (N, D) — эмбеддинги в исходном пространстве, float32.
    """
    z_f = z.astype(np.float64, copy=False)
    V_ = V.astype(np.float64, copy=False)
    mu_ = mu.astype(np.float64, copy=False)
    s = scale.astype(np.float64)          # (1,) или (n_components,) — broadcasting
    result = (z_f * s) @ V_.T + mu_
    return result.astype(np.float32)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def verify_whitening(
    emb_whitened: np.ndarray,
    exclude_indices: Iterable[int] | None = None,
) -> dict[str, float]:
    """
    Sanity check на PCA-нормализованных эмбеддингах.

    Args:
        emb_whitened: (N, n_components) — нормализованные эмбеддинги.
        exclude_indices: строки для исключения.

    Returns:
        dict: {'mean_abs_max', 'std_mean', 'row_norm_mean', 'row_norm_std'}
    """
    if exclude_indices is not None:
        exclude = np.asarray(list(exclude_indices), dtype=np.int64)
        keep_mask = np.ones(emb_whitened.shape[0], dtype=bool)
        keep_mask[exclude] = False
        emb = emb_whitened[keep_mask]
    else:
        emb = emb_whitened

    emb = emb.astype(np.float64, copy=False)
    mean = emb.mean(axis=0)
    n_components = emb.shape[1]

    return {
        'mean_abs_max':  float(np.abs(mean).max()),
        'std_mean':      float(emb.std(axis=0).mean()),
        'row_norm_mean': float(np.linalg.norm(emb, axis=1).mean()),
        'row_norm_std':  float(np.linalg.norm(emb, axis=1).std()),
        'expected_norm': float(n_components ** 0.5),
    }


# --------------------------------------------------------------------------- #
# I/O (дисковый кэш; source of truth — state_dict модели)
# --------------------------------------------------------------------------- #

def save_whitening(
    path: str | Path,
    mu: np.ndarray,
    V: np.ndarray,
    scale: np.ndarray,
) -> None:
    """Сохраняет PCA-параметры в .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mu=mu, V=V, scale=scale)


def load_whitening(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Загружает PCA-параметры из .npz."""
    data = np.load(path)
    return data['mu'], data['V'], data['scale']


# --------------------------------------------------------------------------- #
# Quick sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    """
    python -m src.whitening
    """
    import sys
    rng = np.random.default_rng(42)

    # Симулируем L2-нормированные эмбеддинги как у OpenAI (N=500, D=128)
    N, D, K = 500, 128, 32
    emb = rng.standard_normal((N, D)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    print(f'Input: shape={emb.shape}, L2 norm={np.linalg.norm(emb, axis=1).mean():.4f}')

    # Test 1a: compute_whitening — глобальный std (per_component=False)
    print('\nTest 1a: compute_whitening  per_component=False (global std)')
    mu, V, scale = compute_whitening(emb, n_components=K, per_component=False, verbose=True)
    assert mu.shape == (D,)
    assert V.shape == (D, K)
    assert scale.shape == (1,), f'expected (1,), got {scale.shape}'
    VtV = V.T @ V
    assert np.allclose(VtV, np.eye(K), atol=1e-6), 'V not orthogonal'
    print(f'  scale shape: {scale.shape}  OK')

    # Test 1b: compute_whitening — поэлементный std (per_component=True)
    print('\nTest 1b: compute_whitening  per_component=True (per-component std)')
    mu_pc, V_pc, scale_pc = compute_whitening(emb, n_components=K, per_component=True, verbose=True)
    assert scale_pc.shape == (K,), f'expected ({K},), got {scale_pc.shape}'
    print(f'  scale shape: {scale_pc.shape}  OK')

    # Test 2a: apply_whitening — глобальный
    print('\nTest 2a: apply_whitening  (global scale)')
    emb_pca = apply_whitening(emb, mu, V, scale)
    assert emb_pca.shape == (N, K)
    assert emb_pca.dtype == np.float32
    stats = verify_whitening(emb_pca)
    print(f'  std_mean={stats["std_mean"]:.6f}  (should be ~1)')
    assert stats['mean_abs_max'] < 0.1
    assert 0.9 < stats['std_mean'] < 1.1
    print('  OK')

    # Test 2b: apply_whitening — поэлементный → Var = I_{K}
    print('\nTest 2b: apply_whitening  (per-component scale -> Var = I)')
    emb_pc = apply_whitening(emb, mu_pc, V_pc, scale_pc)
    assert emb_pc.shape == (N, K)
    dim_var = emb_pc.astype(np.float64).var(axis=0)
    print(f'  per-dim var: mean={dim_var.mean():.6f} max={dim_var.max():.6f}  (should be ~1)')
    assert np.allclose(dim_var, 1.0, atol=0.01), f'Var != I: max dev = {np.abs(dim_var - 1).max():.4f}'
    print('  OK')

    # Test 3: inverse_whitening (приближённое)
    print('\nTest 3: inverse_whitening (approximate)')
    emb_back = inverse_whitening(emb_pca, mu, V, scale)
    assert emb_back.shape == (N, D)
    # Точность зависит от variance explained — чем меньше K, тем хуже
    max_err = np.abs(emb.astype(np.float64) - emb_back.astype(np.float64)).max()
    print(f'  Max reconstruction error: {max_err:.4f}  (approximate — OK)')

    # Test 4: save/load round-trip
    print('\nTest 4: save/load')
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'subdir' / 'wh.npz'
        save_whitening(path, mu, V, scale)
        mu2, V2, scale2 = load_whitening(path)
        np.testing.assert_array_equal(mu, mu2)
        np.testing.assert_array_equal(V, V2)
        np.testing.assert_array_equal(scale, scale2)
    print('  Exact round-trip  OK')

    # Test 5: совместимость с реалистичными данными (11924 x 3072 → 256)
    print('\nTest 5: realistic scale (simulated 1000 x 512 -> 64)')
    N2, D2, K2 = 1000, 512, 64
    emb2 = rng.standard_normal((N2, D2)).astype(np.float32)
    emb2 /= np.linalg.norm(emb2, axis=1, keepdims=True)
    mu2, V2, scale2 = compute_whitening(emb2, n_components=K2, verbose=True)
    emb2_pca = apply_whitening(emb2, mu2, V2, scale2)
    stats2 = verify_whitening(emb2_pca)
    print(f'  row_norm_mean={stats2["row_norm_mean"]:.4f} (expected ~{K2**0.5:.1f})')
    assert 0.9 < stats2['std_mean'] < 1.1
    print('  OK')

    print('\nAll checks passed')