"""Decomposition of the difference in Eq. 1 totals between two runs.

``principled`` splits Delta = total(c) - total(r) (c = C-CN, r = R-CN in
stage 2) into five lines that add up to Delta exactly:

    per year t, energy cost E_t = Q_t * P_t, with Q_t the national procured
    energy (grid + PPA + PV + CGS, TWh) and P_t = E_t / Q_t the average
    discounted unit energy cost;
    cooling        = sum_t (Q_c - Q_r) P_c   (quantity effect)
                     + delta cooling CAPEX/O&M without the siting premium
    re_procurement = sum_t Q_r (P_c - P_r)   (price effect)
                     + delta BESS/CGS/PV CAPEX/O&M without the premium
    carbon         = delta carbon cost
    siting         = delta premium part - delta subsidy + delta non-metro adder
    curtailment    = delta curtailment penalty

``v1_method`` reproduces ``dc_decom.postprocess.cost_decomposition`` (energy
difference split 55:45 into cooling and RE procurement, siting = delta
CAPEX + O&M) for comparison.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

CHANNELS = ("cooling", "re_procurement", "carbon", "siting", "curtailment")
FOUR_CHANNELS = ("cooling", "re_procurement", "carbon", "siting")

_ENERGY = ("energy_grid", "energy_ppa", "energy_pv", "energy_cgs_fuel")
_COOLING_ASSETS = ("cooling_capex", "cooling_opex")
_RE_ASSETS = ("bess_capex", "bess_opex", "cgs_capex", "cgs_opex", "pv_capex", "pv_opex")
_CARBON = ("carbon_grid", "carbon_cgs")
_PREMIUM = ("cooling_premium", "cgs_premium", "bess_premium")


def yearly_national(components: pd.DataFrame) -> pd.DataFrame:
    """National sums per year of the quantities the decomposition needs."""
    c = components
    out = pd.DataFrame({
        "year": c["year"],
        "Q": c["procured_TWh"],
        "energy": c[list(_ENERGY)].sum(axis=1),
        "cooling_assets": c[list(_COOLING_ASSETS)].sum(axis=1),
        "re_assets": c[list(_RE_ASSETS)].sum(axis=1),
        "carbon": c[list(_CARBON)].sum(axis=1),
        "premium": c[list(_PREMIUM)].sum(axis=1),
        "subsidy": c["subsidy"],
        "adder": c["adder"],
        "curtailment": c["curtailment"],
    })
    return out.groupby("year", sort=True).sum()


def run_total(components: pd.DataFrame) -> float:
    y = yearly_national(components)
    return float((y["energy"] + y["cooling_assets"] + y["re_assets"] + y["carbon"]
                  + y["premium"] - y["subsidy"] + y["adder"] + y["curtailment"]).sum())


def principled(comp_c: pd.DataFrame, comp_r: pd.DataFrame) -> Dict[str, float]:
    """Five-line split of total(c) - total(r); also returns the sub-lines."""
    yc = yearly_national(comp_c)
    yr = yearly_national(comp_r)
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
        "siting": d["premium"] - d["subsidy"] + d["adder"],
        "curtailment": d["curtailment"],
        # sub-lines
        "quantity_effect": quantity,
        "price_effect": price,
        "d_cooling_assets": d["cooling_assets"],
        "d_re_assets": d["re_assets"],
        "d_premium": d["premium"],
        "d_subsidy": d["subsidy"],
        "d_adder": d["adder"],
        "d_energy": d["energy"],
    }
    out["total"] = sum(out[k] for k in CHANNELS)
    out["direct_total"] = run_total(comp_c) - run_total(comp_r)
    return out


def shares(split: Dict[str, float]) -> Dict[str, float]:
    """Channel / total for the five lines (NaN when the total is zero)."""
    tot = split["total"]
    return {k: (split[k] / tot if tot != 0 else float("nan")) for k in CHANNELS}


def v1_method(res_c, res_r, grid_share: float = 0.55) -> Dict[str, float]:
    """``dc_decom.postprocess.cost_decomposition`` on two SolveResults."""
    d_energy = res_c.energy_cost - res_r.energy_cost
    d_carbon = res_c.carbon_cost - res_r.carbon_cost
    d_siting = (res_c.capex_total - res_r.capex_total) + (res_c.opex_total - res_r.opex_total)
    d_total = d_energy + d_carbon + d_siting
    out = {
        "cooling": d_energy * grid_share,
        "re_procurement": d_energy * (1.0 - grid_share),
        "carbon": d_carbon,
        "siting": d_siting,
        "curtailment": 0.0,   # v1 leaves the curtailment difference out
        "total": d_total,
        "excluded_curtailment": res_c.curtail_cost - res_r.curtail_cost,
    }
    return out
