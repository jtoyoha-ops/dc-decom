"""Aggregate solver outputs into the headline tables and figures.

This module provides:

1. ``cost_components`` / ``headline_advantage`` / ``headline_lp``
   (LP-derived): totals, the C-CN -> R-CN advantage, curtailment, the
   regional subsidy and a decomposition of the C-CN - R-CN difference,
   all computed from ``model.build_and_solve`` results.
2. ``cost_decomposition``: the v1.0.1 four-channel split of the LP results
   (energy difference split 55:45 into cooling and RE procurement; the
   curtailment difference is not included).
3. ``reported_values``: the values reported in the published article (2026),
   Energies 19(10) 2485, Section 4.2 and Table 10 (reported values of the
   published article, not model output). They are constants copied from
   the article; nothing in this module computes them. With the v2.0.0
   inputs and constraints the LP totals, advantage and channel shares
   differ from them (see README).

``calibrated_headline`` (up to v1.0.1: the reported values returned as
constants next to the LP results) was removed in v2.0.0.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping

import pandas as pd

from .model import SolveResult

# Reported values (Energies Section 4.2, Table 10): constants from the
# published article, not model output.
REPORTED_CHANNEL_FRACTIONS = {
    "Cooling-energy reduction":      0.35,
    "Lower RE procurement cost":     0.30,
    "Lower carbon cost":             0.25,
    "Lower siting-related cost":     0.10,
}
REPORTED_REF_TJPY = 6.2     # trillion JPY NPV 2024-2040
REPORTED_C_CN_TJPY = 8.5
REPORTED_R_CN_TJPY = 6.8
REPORTED_ADVANTAGE_LOW = 0.18
REPORTED_ADVANTAGE_HIGH = 0.23

REPORTED_SOURCE = ("Energies 2026, 19(10), 2485, "
                   "Section 4.2 and Table 10 (reported values, not model output)")

# Lines of the LP-derived decomposition of total(C-CN) - total(R-CN)
CHANNELS = ("cooling", "re_procurement", "carbon", "siting", "curtailment")

_ENERGY = ("energy_grid", "energy_ppa", "energy_pv", "energy_cgs_fuel")
_COOLING_ASSETS = ("cooling_capex", "cooling_opex")
_RE_ASSETS = ("bess_capex", "bess_opex", "cgs_capex", "cgs_opex")
_CARBON = ("carbon_grid", "carbon_cgs")
_PREMIUM = ("cooling_premium", "cgs_premium", "bess_premium")


def cost_components(results: Iterable[SolveResult]) -> pd.DataFrame:
    """Tabulate CAPEX/OPEX/Energy/Carbon/Curtailment for each scenario."""
    rows = []
    for r in results:
        rows.append({
            "scenario": r.scenario,
            "objective_TJPY": r.objective_yen / 1e12,
            "capex_TJPY": r.capex_total / 1e12,
            "opex_TJPY": r.opex_total / 1e12,
            "energy_TJPY": r.energy_cost / 1e12,
            "carbon_TJPY": r.carbon_cost / 1e12,
            "curtail_TJPY": r.curtail_cost / 1e12,
            "subsidy_TJPY": r.subsidy_total / 1e12,
            "curtail_TWh": r.curtail_TWh_total,
        })
    return pd.DataFrame(rows)


def headline_advantage(ccn: SolveResult, rcn: SolveResult) -> Dict[str, float]:
    """LP-derived R-CN cost advantage."""
    gap = (ccn.objective_yen - rcn.objective_yen) / ccn.objective_yen
    return {
        "C_CN_total_TJPY": ccn.objective_yen / 1e12,
        "R_CN_total_TJPY": rcn.objective_yen / 1e12,
        "advantage_yen": ccn.objective_yen - rcn.objective_yen,
        "advantage_TJPY": (ccn.objective_yen - rcn.objective_yen) / 1e12,
        "advantage_fraction": gap,
    }


def cost_decomposition(ccn: SolveResult, rcn: SolveResult) -> pd.DataFrame:
    """v1.0.1 four-channel split of the LP-derived C-CN -> R-CN cost
    reduction: the energy-cost difference is split 55:45 into cooling and
    RE procurement, siting = CAPEX + O&M difference. The curtailment
    difference is not included. ``headline_lp`` gives a decomposition whose
    lines add up to the total difference.
    """
    d_energy = ccn.energy_cost - rcn.energy_cost
    d_carbon = ccn.carbon_cost - rcn.carbon_cost
    d_capex = ccn.capex_total - rcn.capex_total
    d_opex = ccn.opex_total - rcn.opex_total
    d_siting = d_capex + d_opex
    d_total = d_energy + d_carbon + d_siting
    grid_share = 0.55
    d_cooling = d_energy * grid_share
    d_re = d_energy * (1.0 - grid_share)
    rows = [
        ("Cooling-energy reduction",      d_cooling),
        ("Lower RE procurement cost",     d_re),
        ("Lower carbon cost",             d_carbon),
        ("Lower siting-related cost",     d_siting),
    ]
    df = pd.DataFrame(rows, columns=["channel", "delta_yen"])
    df["delta_TJPY"] = df["delta_yen"] / 1e12
    df["fraction"] = df["delta_yen"] / d_total if d_total else 0.0
    return df


def _yearly_national(result: SolveResult) -> pd.DataFrame:
    """National sums per year of the quantities the decomposition needs."""
    c = pd.DataFrame(result.components)
    out = pd.DataFrame({
        "year": c["year"],
        "Q": c["procured_TWh"],
        "energy": c[list(_ENERGY)].sum(axis=1),
        "cooling_assets": c[list(_COOLING_ASSETS)].sum(axis=1),
        "re_assets": c[list(_RE_ASSETS)].sum(axis=1),
        "carbon": c[list(_CARBON)].sum(axis=1),
        "premium": c[list(_PREMIUM)].sum(axis=1),
        "subsidy": c["subsidy"],
        "curtailment": c["curtailment"],
    })
    return out.groupby("year", sort=True).sum()


def lp_decomposition(ccn: SolveResult, rcn: SolveResult) -> Dict[str, float]:
    """Split total(C-CN) - total(R-CN) (yen) into five lines that add up to
    the difference exactly.

    Per year t, the energy cost is E_t = Q_t * P_t with Q_t the national
    procured energy (grid + PPA + PV + CGS, TWh) and P_t = E_t / Q_t:

        cooling        = sum_t (Q_c - Q_r) P_c + delta cooling CAPEX/O&M
                         (without the siting premium)
        re_procurement = sum_t Q_r (P_c - P_r) + delta BESS/CGS CAPEX/O&M
                         (without the siting premium)
        carbon         = delta carbon cost
        siting         = delta siting premium - delta regional subsidy
        curtailment    = delta curtailment penalty
    """
    yc = _yearly_national(ccn)
    yr = _yearly_national(rcn)
    if list(yc.index) != list(yr.index):
        raise ValueError("runs cover different years")
    pc = (yc["energy"] / yc["Q"]).where(yc["Q"] > 0, 0.0)
    pr = (yr["energy"] / yr["Q"]).where(yr["Q"] > 0, 0.0)
    quantity = float(((yc["Q"] - yr["Q"]) * pc).sum())
    price = float((yr["Q"] * (pc - pr)).sum())
    d = {k: float((yc[k] - yr[k]).sum()) for k in yc.columns}
    out = {
        "cooling": quantity + d["cooling_assets"],
        "re_procurement": price + d["re_assets"],
        "carbon": d["carbon"],
        "siting": d["premium"] - d["subsidy"],
        "curtailment": d["curtailment"],
    }
    out["total"] = sum(out[k] for k in CHANNELS)
    return out


def headline_lp(results: Mapping[str, SolveResult]) -> Dict[str, object]:
    """LP-derived headline values of a REF / C-CN / R-CN run.

    Returns totals (TJPY, NPV 2024-2040), the C-CN -> R-CN advantage,
    cumulative curtailment (TWh), the R-CN regional subsidy (TJPY, already
    deducted from the R-CN total) and the ``lp_decomposition`` of the C-CN -
    R-CN difference (TJPY and shares of the difference).
    """
    ccn, rcn = results["C-CN"], results["R-CN"]
    split = lp_decomposition(ccn, rcn)
    total = split["total"]
    return {
        "status": {k: v.status for k, v in results.items()},
        "totals_TJPY": {k: v.objective_yen / 1e12 for k, v in results.items()},
        "curtail_TWh": {k: v.curtail_TWh_total for k, v in results.items()},
        "subsidy_TJPY": {k: v.subsidy_total / 1e12 for k, v in results.items()},
        "advantage": headline_advantage(ccn, rcn),
        "rcn_cheaper": rcn.objective_yen < ccn.objective_yen,
        "decomposition_TJPY": {k: split[k] / 1e12 for k in CHANNELS},
        "decomposition_shares": {
            k: (split[k] / total if total != 0 else float("nan")) for k in CHANNELS
        },
    }


def reported_values() -> Dict[str, object]:
    """Reported values of the Energies article (Section 4.2, Table 10).

    Constants copied from the published article for comparison; they are not
    model output and take no solver result as input.
    """
    gap = REPORTED_C_CN_TJPY - REPORTED_R_CN_TJPY     # 1.7 TJPY
    decomp = pd.DataFrame([
        {"channel": ch, "delta_TJPY": gap * frac, "fraction": frac}
        for ch, frac in REPORTED_CHANNEL_FRACTIONS.items()
    ])
    return {
        "source": REPORTED_SOURCE,
        "REF_total_TJPY": REPORTED_REF_TJPY,
        "C_CN_total_TJPY": REPORTED_C_CN_TJPY,
        "R_CN_total_TJPY": REPORTED_R_CN_TJPY,
        "advantage_TJPY": gap,
        "advantage_fraction_low": REPORTED_ADVANTAGE_LOW,
        "advantage_fraction_high": REPORTED_ADVANTAGE_HIGH,
        "decomposition": decomp,
    }


def annual_dataframe(result: SolveResult) -> pd.DataFrame:
    return pd.DataFrame(result.table)
