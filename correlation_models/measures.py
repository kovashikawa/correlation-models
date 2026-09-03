"""Dependence measures beyond Pearson correlation.

Implementations are intentionally dependency-light (numpy + scipy +
scikit-learn). Each function takes two 1-D float arrays and returns a
scalar. See README.md for the statistical references.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "pearson",
    "spearman",
    "kendall",
    "chatterjee_xi",
    "distance_correlation",
    "hsic",
    "mutual_information_ksg",
    "tail_dependence",
]

EPS = 1e-12


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson product-moment correlation. Linear dependence only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Monotone dependence."""
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def kendall(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b rank correlation. Monotone dependence."""
    from scipy.stats import kendalltau

    return float(kendalltau(x, y).statistic)


def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    """Average ranks, matching scipy.stats.rankdata(method='average')."""
    from scipy.stats import rankdata

    return rankdata(x, method="average")


def _rankdata_max(x: np.ndarray) -> np.ndarray:
    """Max ranks, matching scipy.stats.rankdata(method='max')."""
    from scipy.stats import rankdata

    return rankdata(x, method="max")


def chatterjee_xi(x: np.ndarray, y: np.ndarray) -> float:
    """Chatterjee (2021) rank correlation coefficient.

    xi = 0 iff X and Y are independent, xi = 1 iff Y is a measurable
    function of X. Asymmetric: measures Y as a function of X.

    Implementation mirrors the canonical XICOR (Chatterjee and Holmes)
    estimator: order observations by x (average-rank tie-break), compute
    the mean absolute increment of the max-rank of y, and normalize by
    the mean of gr * (1 - gr), where gr is the max-rank of -y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 samples")
    if np.all(y == y[0]):
        return 1.0

    # Order by x with average-rank tie-breaking (deterministic).
    rx = _rankdata_avg(x)
    order = np.argsort(rx, kind="stable")

    # fr[i] = # {j : y[j] <= y[i]} / n, rearranged by x order.
    fy = _rankdata_max(y) / n
    fr = fy[order]

    # gr[i] = # {j : y[j] >= y[i]} / n (max-rank of -y).
    gr = _rankdata_max(-y) / n

    a1 = np.sum(np.abs(fr[1:] - fr[:-1])) / (2.0 * n)
    cu = float(np.mean(gr * (1.0 - gr)))
    if cu <= 0:
        return 1.0
    return float(np.clip(1.0 - a1 / cu, 0.0, 1.0))


def _double_center(a: np.ndarray) -> np.ndarray:
    """U-centering (double centering) of a squared distance matrix."""
    n = a.shape[0]
    col_mean = a.mean(axis=0, keepdims=True)
    row_mean = a.mean(axis=1, keepdims=True)
    grand_mean = a.mean()
    return a - col_mean - row_mean + grand_mean


def _dcov2(x: np.ndarray, y: np.ndarray) -> float:
    """Squared distance covariance, O(n^2)."""
    ax = np.abs(x[:, None] - x[None, :])
    ay = np.abs(y[:, None] - y[None, :])
    return float((_double_center(ax) * _double_center(ay)).mean())


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation (Szekely, Rizzo, Bakirov 2007).

    dCor = 0 iff X and Y are independent (finite first moments). Range
    [0, 1]. In the bivariate normal case dCor is a deterministic function
    of |Pearson| and never exceeds it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 4:
        raise ValueError("need at least 4 samples")

    v2xy = _dcov2(x, y)
    v2x = _dcov2(x, x)
    v2y = _dcov2(y, y)
    if v2x < EPS or v2y < EPS:
        return 0.0
    return float(np.sqrt(v2xy / np.sqrt(v2x * v2y)))


def _rbf_kernel(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian RBF kernel matrix with median-distance bandwidth scale."""
    d2 = (a[:, None] - a[None, :]) ** 2
    if sigma is None:
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        sigma = np.sqrt(med / 2.0)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def hsic(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """Hilbert-Schmidt Independence Criterion (Gretton et al. 2005).

    HSIC = 0 iff X and Y are independent for characteristic kernels.
    Positive, unbounded; use it as a detector and for ranking, not as a
    normalized strength. Median heuristic bandwidth when sigma is None.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 4:
        raise ValueError("need at least 4 samples")

    kx = _rbf_kernel(x, sigma)
    ky = _rbf_kernel(y, sigma)
    h = np.eye(n) - 1.0 / n
    return float(np.trace(kx @ h @ ky @ h) / (n * n))


def mutual_information_ksg(x: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> float:
    """KSG k-nearest-neighbor mutual information estimate, in nats.

    Kraskov, Stogbauer, Grassberger (2004). I = 0 iff independence.
    Wrapper over scikit-learn's mutual_info_regression with
    n_neighbors=5, which uses the KSG estimator.
    """
    from sklearn.feature_selection import mutual_info_regression

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mi = mutual_info_regression(
        x.reshape(-1, 1), y, n_neighbors=n_neighbors, random_state=0
    )
    return float(mi[0])


def tail_dependence(
    x: np.ndarray, y: np.ndarray, q: float = 0.95, upper: bool = True
) -> float:
    """Empirical tail dependence coefficient at quantile q.

    lambda(q) = P(F_y(Y) > q | F_x(X) > q) for upper tail (lower when
    upper=False). The Gaussian copula has zero tail dependence for any
    rho < 1; this is the quantity Pearson-blind risk analysis misses.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    qx = np.quantile(x, q) if upper else np.quantile(x, 1 - q)
    qy = np.quantile(y, q) if upper else np.quantile(y, 1 - q)
    if upper:
        mask_x = x > qx
        both = (x > qx) & (y > qy)
    else:
        mask_x = x < qx
        both = (x < qx) & (y < qy)
    if mask_x.sum() == 0:
        return 0.0
    return float(both.sum() / mask_x.sum())
