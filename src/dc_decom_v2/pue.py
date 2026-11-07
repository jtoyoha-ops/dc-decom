"""PUE of the cooling technologies per time slice (Energies Eqs. 8 and 9).

air:  C_base + alpha T + beta max(T - T_thresh, 0)^2
dlc:  1 + C_pump + gamma T
imm:  1 + (C_pump - pue_immersion_offset) + gamma T

Unlike ``dc_decom_spec.pue.pue_for_technology`` (which rebuilds the immersion
parameters from the defaults, so that a changed gamma is ignored for
immersion), every coefficient here comes from the parameter set. The annual
PUE is the weighted mean over the slices (rep96) or the value at the annual
mean temperature (annual mode, one slice).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import COOLING
from .params import ParamSet
from .timeslices import TimeSlices


@dataclass(frozen=True)
class PUECoefficients:
    C_base: float
    alpha: float
    beta: float
    T_thresh: float
    C_pump: float
    gamma: float
    immersion_offset: float

    @classmethod
    def from_params(cls, ps: ParamSet) -> "PUECoefficients":
        return cls(C_base=ps.scalar("C_base"), alpha=ps.scalar("alpha"), beta=ps.scalar("beta"),
                   T_thresh=ps.scalar("T_thresh"), C_pump=ps.scalar("C_pump"),
                   gamma=ps.scalar("gamma"),
                   immersion_offset=ps.scalar("pue_immersion_offset"))


def pue(tech: str, T, c: PUECoefficients) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if tech == "air":
        return c.C_base + c.alpha * T + c.beta * np.maximum(T - c.T_thresh, 0.0) ** 2
    if tech == "dlc":
        return 1.0 + c.C_pump + c.gamma * T
    if tech == "imm":
        return 1.0 + (c.C_pump - c.immersion_offset) + c.gamma * T
    raise ValueError(f"unknown cooling technology {tech!r}")


def slice_pue(ts: TimeSlices, c: PUECoefficients) -> Dict[Tuple[str, str], np.ndarray]:
    """PUE per (region, tech) and slice."""
    return {(r, k): pue(k, ts.temperature[r], c) for r in ts.temperature for k in COOLING}


def annual_pue(ts: TimeSlices, sp: Dict[Tuple[str, str], np.ndarray]
               ) -> Dict[Tuple[str, str], float]:
    w = ts.weights
    return {key: float(np.dot(w, v) / w.sum()) for key, v in sp.items()}


def nondominated_slices(ts: TimeSlices, sp: Dict[Tuple[str, str], np.ndarray],
                        region: str) -> Tuple[int, ...]:
    """Slices whose PUE vector (air, dlc, imm) is not weakly dominated.

    Facility load and critical load are sum_k it_k x PUE_k(h) with it_k >= 0
    and the same in every slice, so a constraint 'x >= load(h) for all h' needs
    only these slices (ties keep the first slice).
    """
    m = np.column_stack([sp[(region, k)] for k in COOLING])
    keep = []
    for i in range(len(m)):
        dominated = False
        for j in range(len(m)):
            if j == i:
                continue
            if np.all(m[j] >= m[i]) and (np.any(m[j] > m[i]) or j < i):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return tuple(keep)
