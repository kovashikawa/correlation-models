"""Self-tests for the measures module.

Usage: uv run python scripts/test_measures.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from correlation_models import measures as m


def _xi_reference(x: np.ndarray, y: np.ndarray) -> float:
    """XICOR calculateXI reference (Chatterjee and Holmes), deterministic
    average-rank tie-break on x, max-rank of y."""
    n = x.size
    rx = rankdata(x, method="average")
    fy = rankdata(y, method="max") / n
    fr = fy[np.argsort(rx, kind="stable")]
    gr = rankdata(-y, method="max") / n
    a1 = np.sum(np.abs(fr[1:] - fr[:-1])) / (2.0 * n)
    cu = float(np.mean(gr * (1.0 - gr)))
    return 1.0 - a1 / cu


def _xi_closed_form(x: np.ndarray, y: np.ndarray) -> float:
    """Chatterjee (2021), eq. 1.1: xi = 1 - 3 * sum |r_{i+1} - r_i| / (n^2 - 1),
    valid when X and Y have no ties."""
    n = x.size
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    r_sorted = ry[np.argsort(rx, kind="stable")]
    num = np.sum(np.abs(r_sorted[1:] - r_sorted[:-1]))
    return 1.0 - 3.0 * num / (n * n - 1)


def test_chatterjee_against_reference() -> None:
    rng = np.random.default_rng(0)

    # No ties: must match the XICOR reference and the paper's closed form.
    x = rng.normal(size=2_000)
    y = x + 0.5 * rng.normal(size=2_000)
    mine = m.chatterjee_xi(x, y)
    ref = _xi_reference(x, y)
    assert abs(mine - ref) < 1e-12, f"XICOR mismatch: {mine} vs {ref}"
    closed = _xi_closed_form(x, y)
    assert abs(mine - closed) < 1e-12, f"closed-form mismatch: {mine} vs {closed}"

    # Heavy ties in y (integer grid): xicor must match as well.
    x_ties = rng.normal(size=2_000)
    y_ties = np.round(x_ties + 0.5 * rng.normal(size=2_000), 0)
    mine = m.chatterjee_xi(x_ties, y_ties)
    ref = _xi_reference(x_ties, y_ties)
    assert abs(mine - ref) < 1e-12, f"tie mismatch: {mine} vs {ref}"

    # Deterministic function: xi approaches 1 (finite-sample max is
    # (n-2)/(n+1), so use a large n for a tight bound).
    xd = rng.normal(size=5_000)
    assert m.chatterjee_xi(xd, np.abs(xd)) > 0.995

    # Under independence, finite-sample xi is small and can be negative;
    # it must not be clipped to [0, 1].
    xi_ind = m.chatterjee_xi(rng.normal(size=2_000), rng.normal(size=2_000))
    assert abs(xi_ind) < 0.05, f"xi(independent) off: {xi_ind}"

    # Constant y: undefined (0/0), must be NaN, not 1.
    assert np.isnan(m.chatterjee_xi(x, np.ones_like(x)))


def test_distance_correlation_known_cases() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=2_000)

    # Perfect linear: dCor ~ 1.
    assert m.distance_correlation(x, 2 * x + 1) > 0.99

    # |X|: dCor should be clearly nonzero where Pearson is ~0.
    d = m.distance_correlation(x, np.abs(x))
    assert d > 0.5, f"dCor(|X|) too low: {d}"

    # Independent: dCor ~ 0.
    d_ind = m.distance_correlation(x, rng.normal(size=2_000))
    assert d_ind < 0.05, f"dCor(independent) too high: {d_ind}"


def test_mutual_information_zero_iff_independent() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=2_000)
    mi_ind = m.mutual_information_ksg(x, rng.normal(size=2_000))
    assert mi_ind < 0.05, f"MI(independent) too high: {mi_ind}"
    mi_dep = m.mutual_information_ksg(x, np.abs(x))
    assert mi_dep > 0.5, f"MI(|X|) too low: {mi_dep}"


def test_tail_dependence() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=50_000)
    # Y = X: perfect upper-tail co-exceedance.
    assert m.tail_dependence(x, x, q=0.95) > 0.99
    # Y = |X|: P(|X| > q_|X| | X > q_X) = P(X > q_|X|)/P(X > q_X) = 0.5.
    td_abs = m.tail_dependence(x, np.abs(x), q=0.95)
    assert abs(td_abs - 0.5) < 0.02, f"tail dep(|X|) should be ~0.5, got {td_abs}"
    # Independent pair: tail dependence ~ 0.05 at q = 0.95.
    ind = m.tail_dependence(x, rng.normal(size=50_000), q=0.95)
    assert 0.03 < ind < 0.08, f"tail dep(independent) off: {ind}"


if __name__ == "__main__":
    test_chatterjee_against_reference()
    test_distance_correlation_known_cases()
    test_mutual_information_zero_iff_independent()
    test_tail_dependence()
    print("All tests passed.")
