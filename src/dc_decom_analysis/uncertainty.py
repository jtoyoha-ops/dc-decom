"""Latin hypercube sampling and partial rank correlation (composite uncertainty, part mc).

lhs            n x d Latin hypercube in [0, 1) (scipy.stats.qmc, default scramble)
uniform        u -> lo + u (hi - lo)
path_quantile  q -> piecewise linear through low (q=0), value (q=0.5), high (q=1)
prcc           partial rank correlation coefficients with two-sided t p-values
bootstrap_prcc percentile 95% intervals of the PRCC over bootstrap resamples
quintile_bins  summary of an output per quintile of one u column (figure data)

Definitions: config/experiments_v35.yaml section uncertainty (defined before the runs).
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple, TypeVar

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

K = TypeVar("K")


def lhs(n: int, d: int, seed: int) -> np.ndarray:
    """n points of a d-dimensional Latin hypercube (one point per stratum 1/n per column)."""
    return qmc.LatinHypercube(d=d, seed=seed).random(n=n)


def uniform(u: float, lo: float, hi: float) -> float:
    return float(lo + u * (hi - lo))


def _pq(q: float, low: float, value: float, high: float) -> float:
    if q <= 0.5:
        return float(low + 2.0 * q * (value - low))
    return float(value + (2.0 * q - 1.0) * (high - value))


def path_quantile(q: float, low: Mapping[K, float], value: Mapping[K, float],
                  high: Mapping[K, float]) -> Dict[K, float]:
    """Per key (year or region): low at q=0, value at q=0.5, high at q=1, linear between."""
    return {k: _pq(q, float(low[k]), float(value[k]), float(high[k])) for k in value}


def _ranks(a: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(stats.rankdata, 0, a)


def _resid(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return b - A @ coef


def prcc(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """PRCC of each column of X with y and the two-sided p-values.

    Ranks with average ties; for factor j the residuals of rank(x_j) and rank(y)
    after least squares on the ranks of the other factors (with intercept) are
    correlated (Pearson). p from the t distribution with n - 2 - (k - 1) dof.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    R = _ranks(X)
    ry = stats.rankdata(y)
    out = np.empty(k)
    for j in range(k):
        A = np.column_stack([np.ones(n), np.delete(R, j, axis=1)])
        ex = _resid(A, R[:, j])
        ey = _resid(A, ry)
        den = float(np.sqrt((ex @ ex) * (ey @ ey)))
        out[j] = float(ex @ ey) / den if den > 0 else np.nan
    dof = n - 2 - (k - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = out * np.sqrt(dof / (1.0 - out ** 2))
    p = 2.0 * stats.t.sf(np.abs(t), dof)
    return out, p


def bootstrap_prcc(X: np.ndarray, y: np.ndarray, B: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Percentile 95% interval (2.5%, 97.5%) of the PRCC over B resamples of the rows."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty((B, X.shape[1]))
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        draws[b], _ = prcc(X[idx], y[idx])
    return np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def quintile_bins(u_col: np.ndarray, y: np.ndarray, lo: float = 0.18,
                  hi: float = 0.23) -> pd.DataFrame:
    """Per quintile of u (0-0.2, ..., 0.8-1): n, mean, p5, p95 of y, share y > 0 and
    share of y in [lo, hi]."""
    u = np.asarray(u_col, dtype=float)
    y = np.asarray(y, dtype=float)
    q = np.minimum((u * 5).astype(int), 4)
    rows = []
    for i in range(5):
        v = y[q == i]
        rows.append({"quintile": i + 1, "n": int(v.size),
                     "mean": float(v.mean()) if v.size else np.nan,
                     "p5": float(np.percentile(v, 5)) if v.size else np.nan,
                     "p95": float(np.percentile(v, 95)) if v.size else np.nan,
                     "share_positive": float((v > 0).mean()) if v.size else np.nan,
                     "share_in_range": float(((v >= lo) & (v <= hi)).mean()) if v.size else np.nan})
    return pd.DataFrame(rows)
