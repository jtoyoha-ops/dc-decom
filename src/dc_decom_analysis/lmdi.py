"""Additive LMDI-I decomposition of grid emissions (Ang 2005; Ang & Liu 2007).

Grid emissions of region r in year t are written as the product

    E_grid(r, t) = D(t) x S(r, t) x PUE(r, t) x s(r, t) x EF(r, t)

with D the national IT energy (TWh), S the regional share of IT energy, PUE
the realized facility-to-IT energy ratio, s the grid share of facility load
(E_grid / Load) and EF the grid emission factor (kg/kWh, so TWh x kg/kWh =
Mt). CGS emissions are a separate additive term. Zero values are replaced by
1e-12 (Ang & Liu 2007) and the replaced product is decomposed, so the factor
effects sum to the change of the replaced product; the residual against the
unreplaced change is reported.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping

import pandas as pd

EPS = 1e-12
FACTORS = ("activity", "structure", "pue", "grid_share", "ef")


def logmean(a: float, b: float) -> float:
    """Logarithmic mean L(a, b) for a, b > 0."""
    if a == b:
        return a
    return (a - b) / (math.log(a) - math.log(b))


def factor_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Per (year, region) factors of the identity from a result table."""
    t = table.copy()
    D = t.groupby("year")["it_energy_TWh"].transform("sum")
    out = pd.DataFrame({
        "year": t["year"],
        "region": t["region"],
        "activity": D,
        "structure": t["it_energy_TWh"] / D,
        "pue": t["load_TWh"] / t["it_energy_TWh"],
        "grid_share": (t["E_grid_TWh"] / t["load_TWh"]).where(t["load_TWh"] > 0, 0.0),
        "ef": t["ef_grid"],
        "e_grid": t["emissions_grid_Mt"],
        "e_cgs": t["emissions_cgs_Mt"],
    })
    return out.sort_values(["year", "region"]).reset_index(drop=True)


def decompose(x0: Mapping[str, Mapping[str, float]], xT: Mapping[str, Mapping[str, float]],
              cgs0: float, cgsT: float) -> Dict[str, float]:
    """LMDI-I between state 0 and state T.

    ``x0[region][factor]`` and ``xT[region][factor]`` give the five factors.
    Returns the effect of each factor (Mt), the CGS term, their total, the
    exact change of grid + CGS emissions and the residual.
    """
    eff = {k: 0.0 for k in FACTORS}
    exact = 0.0
    for r in sorted(xT):
        a0 = {k: max(float(x0[r][k]), EPS) for k in FACTORS}
        aT = {k: max(float(xT[r][k]), EPS) for k in FACTORS}
        e0 = math.prod(a0[k] for k in FACTORS)
        eT = math.prod(aT[k] for k in FACTORS)
        w = logmean(eT, e0)
        for k in FACTORS:
            eff[k] += w * math.log(aT[k] / a0[k])
        exact += (math.prod(float(xT[r][k]) for k in FACTORS)
                  - math.prod(float(x0[r][k]) for k in FACTORS))
    out = dict(eff)
    out["cgs"] = cgsT - cgs0
    out["total"] = sum(eff.values()) + out["cgs"]
    out["exact_delta"] = exact + (cgsT - cgs0)
    out["residual"] = out["exact_delta"] - out["total"]
    return out


def _state(f: pd.DataFrame, year: int) -> Dict[str, Dict[str, float]]:
    g = f[f["year"] == year]
    return {row["region"]: {k: float(row[k]) for k in FACTORS} for _, row in g.iterrows()}


def vs_frozen_2024(table: pd.DataFrame, year: int, base_year: int = 2024) -> Dict[str, float]:
    """Year ``year`` of a run against the same activity at frozen base-year
    intensities (structure, PUE, grid share, EF and CGS emissions per unit
    of activity of ``base_year``)."""
    f = factor_frame(table)
    xT = _state(f, year)
    x0 = _state(f, base_year)
    D_T = next(iter(xT.values()))["activity"]
    D_0 = next(iter(x0.values()))["activity"]
    for r in x0:
        x0[r]["activity"] = D_T
    cgs0 = float(f.loc[f["year"] == base_year, "e_cgs"].sum()) * D_T / D_0
    cgsT = float(f.loc[f["year"] == year, "e_cgs"].sum())
    return decompose(x0, xT, cgs0, cgsT)


def between(table_base: pd.DataFrame, table_alt: pd.DataFrame, year: int) -> Dict[str, float]:
    """Year ``year`` of run ``alt`` against run ``base`` (alt - base)."""
    fb = factor_frame(table_base)
    fa = factor_frame(table_alt)
    cgs0 = float(fb.loc[fb["year"] == year, "e_cgs"].sum())
    cgsT = float(fa.loc[fa["year"] == year, "e_cgs"].sum())
    return decompose(_state(fb, year), _state(fa, year), cgs0, cgsT)


def cumulative(parts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in parts:
        for k, v in p.items():
            out[k] = out.get(k, 0.0) + v
    return out
