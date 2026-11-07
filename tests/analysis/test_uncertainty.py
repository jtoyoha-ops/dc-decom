"""Synthetic checks of the LHS, the path mapping and the PRCC (part mc)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dc_decom_analysis import uncertainty as U  # noqa: E402


def _linear(seed: int = 1, n: int = 500):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 3))
    y = 3.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=n)
    return X, y


def test_prcc_signs_and_unrelated_factor():
    X, y = _linear()
    r, p = U.prcc(X, y)
    assert r[0] > 0.9
    assert r[1] < -0.5
    assert abs(r[2]) < 0.1
    assert p[0] < 1e-6 and p[1] < 1e-6


def test_prcc_invariant_under_monotone_transform():
    X, y = _linear()
    r1, _ = U.prcc(X, y)
    r2, _ = U.prcc(X, np.exp(y))
    np.testing.assert_allclose(r1, r2, rtol=0, atol=1e-12)


def test_bootstrap_interval_contains_estimate():
    X, y = _linear(n=200)
    r, _ = U.prcc(X, y)
    lo, hi = U.bootstrap_prcc(X, y, B=100, seed=3)
    assert np.all(lo <= r) and np.all(r <= hi)


def test_lhs_one_point_per_stratum():
    n, d = 200, 5
    u = U.lhs(n, d, seed=20261107)
    assert u.shape == (n, d)
    for j in range(d):
        strata = np.sort(np.floor(u[:, j] * n).astype(int))
        np.testing.assert_array_equal(strata, np.arange(n))
    np.testing.assert_array_equal(u, U.lhs(n, d, seed=20261107))


def test_path_quantile_ends_and_monotone():
    low, val, high = {2030: 1.0, 2040: 2.0}, {2030: 3.0, 2040: 5.0}, {2030: 4.0, 2040: 9.0}
    assert U.path_quantile(0.0, low, val, high) == low
    assert U.path_quantile(0.5, low, val, high) == val
    assert U.path_quantile(1.0, low, val, high) == high
    qs = np.linspace(0, 1, 101)
    for t in val:
        v = [U.path_quantile(float(q), low, val, high)[t] for q in qs]
        assert np.all(np.diff(v) >= 0)


def test_quintile_bins_counts():
    u = (np.arange(100) + 0.5) / 100
    y = u - 0.5
    b = U.quintile_bins(u, y)
    assert list(b["n"]) == [20] * 5
    assert b["share_positive"].iloc[0] == 0.0 and b["share_positive"].iloc[4] == 1.0
