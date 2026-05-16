def compute_whitening(
    emb_matrix: np.ndarray,  # ВСЕ эмбеддинги, (N_items, D)
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Считаем μ, Σ на полной матрице эмбеддингов товаров.
    Это согласуется с iDreamRec (§4.1.3) и AlphaFuse (§3.2).
    
    В cold-start user setting все товары известны на момент обучения,
    так что утечки нет. Если бы был cold-start item setting — нужно было бы
    использовать только train-known items.
    """


def apply_whitening(
    emb_matrix: np.ndarray,
    mu: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Применяет whitening: (E - μ) @ W."""


def inverse_whitening(
    z: np.ndarray,
    mu: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """
    Обратное преобразование: z @ W^-1 + μ.
    Используется для отладки и интерпретации сгенерированных эмбеддингов.
    """


def verify_whitening(
    emb_whitened: np.ndarray,
    tol: float = 1e-3,
) -> dict:
    """
    Sanity check: считает фактические mean и cov после whitening.
    Возвращает {'mean_norm', 'cov_diag_mean', 'cov_offdiag_mean'} для проверки.
    """