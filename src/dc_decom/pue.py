"""Dynamic PUE module.

Implements the air-cooling and liquid-cooling PUE equations (Eqs. 8 and 9 in
the paper). Coefficients are calibrated against five regional empirical data
points and verified separately in ``tests/test_pue.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PUEParameters:
    """Parameter bundle for the dynamic PUE module."""

    C_base: float = 1.14
    alpha: float = 0.0167
    beta: float = 0.0102
    T_thresh: float = 15.0
    C_pump: float = 0.055
    gamma: float = 0.0015


def pue_air(T: float | np.ndarray, p: PUEParameters | None = None) -> float | np.ndarray:
    """Air-cooling PUE as a function of ambient temperature (paper Eq. 8)."""
    p = p or PUEParameters()
    T = np.asarray(T, dtype=float)
    excess = np.maximum(T - p.T_thresh, 0.0)
    return p.C_base + p.alpha * T + p.beta * (excess ** 2)


def pue_liquid(T: float | np.ndarray, p: PUEParameters | None = None) -> float | np.ndarray:
    """Liquid-cooling PUE as a function of ambient temperature (paper Eq. 9)."""
    p = p or PUEParameters()
    T = np.asarray(T, dtype=float)
    return 1.0 + p.C_pump + p.gamma * T


def pue_for_technology(tech: str, T: float | np.ndarray,
                       p: PUEParameters | None = None) -> float | np.ndarray:
    """Return PUE for the requested cooling technology.

    ``tech`` must be one of {"air", "dlc", "immersion"}. DLC and immersion
    both use the liquid-cooling equation; immersion is offset slightly lower
    to reflect the lower pump overhead reported by MHI (Ref. [21]).
    """
    p = p or PUEParameters()
    if tech == "air":
        return pue_air(T, p)
    if tech == "dlc":
        return pue_liquid(T, p)
    if tech == "immersion":
        # MHI immersion demo achieves ~1.05; reduce pump overhead by ~0.025.
        offset = PUEParameters(C_pump=p.C_pump - 0.025)
        return pue_liquid(T, offset)
    raise ValueError(f"Unknown cooling technology: {tech!r}")


def annual_mean_pue(temperatures: Iterable[float],
                    weights: Iterable[float],
                    tech: str,
                    p: PUEParameters | None = None) -> float:
    """Weight-averaged PUE over a sequence of temperature observations."""
    T = np.asarray(list(temperatures), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    pue = pue_for_technology(tech, T, p)
    return float(np.sum(w * pue) / np.sum(w))
