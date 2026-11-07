"""Normalized long tables of solve results.

Quantities are in TWh (energy), GW (IT, cooling, PV, CGS) and GWh (BESS);
costs are discounted yen (base year 2024) as in Eq. 1.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

ENERGY_COLS = ("E_grid_TWh", "E_ppa_TWh", "E_pv_TWh", "E_bess_disch_TWh", "E_cgs_TWh")


def long_table(result, scenario: str, config: str) -> pd.DataFrame:
    """One row per (year, region): quantities of ``result.table`` joined with
    the discounted cost components of ``result.components``."""
    q = pd.DataFrame(result.table)
    c = pd.DataFrame(result.components)
    df = q.merge(c, on=["year", "region"], how="left", validate="one_to_one")
    df.insert(0, "config", config)
    df.insert(0, "scenario", scenario)
    return df.sort_values(["scenario", "config", "year", "region"]).reset_index(drop=True)


def components_frame(result) -> pd.DataFrame:
    return pd.DataFrame(result.components)


def summary(result) -> Dict[str, float]:
    """Totals of one solve (yen and TJPY, TWh, Mt)."""
    t = pd.DataFrame(result.table)
    em = t.groupby("year")["emissions_Mt"].sum()
    return {
        "status": result.status,
        "total_yen": result.objective_yen,
        "total_TJPY": result.objective_yen / 1e12,
        "capex_TJPY": result.capex_total / 1e12,
        "opex_TJPY": result.opex_total / 1e12,
        "energy_TJPY": result.energy_cost / 1e12,
        "carbon_TJPY": result.carbon_cost / 1e12,
        "curtail_TJPY": result.curtail_cost / 1e12,
        "subsidy_TJPY": getattr(result, "subsidy_total", 0.0) / 1e12,
        "adder_TJPY": getattr(result, "adder_total", 0.0) / 1e12,
        "resource_cost_TJPY": (result.objective_yen + getattr(result, "subsidy_total", 0.0)) / 1e12,
        "curtail_TWh": float(t["curt_TWh"].sum()) if "curt_TWh" in t else 0.0,
        "emissions_2024_Mt": float(em.loc[em.index.min()]),
        "emissions_2040_Mt": float(em.loc[em.index.max()]),
        "emissions_cum_Mt": float(em.sum()),
    }


def concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames_l: List[pd.DataFrame] = [f for f in frames if len(f)]
    return pd.concat(frames_l, ignore_index=True) if frames_l else pd.DataFrame()
