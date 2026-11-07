"""DC-DECOM LP model, spec version.

A copy of ``dc_decom.model`` (v1.0.1) with switchable fixes of the
implementation errors E1-E8 (``SpecFixes``) and readings of the points the
paper does not state (``SpecReadings``). With ``SpecFixes.none()``, the
default readings, no overrides and the v1 inputs, the LP is built with the
same variables, constraints (in the same order) and coefficients as v1.

Fixes (paper statement -> v1 code):
    E1 share_2040 by region name (v1 applies a tuple ordered Kanto, Kansai,
       Hokkaido, Tohoku, Kyushu to regions.csv order).
    E2 BESS charging E_ch in the energy balance and E_disch = eff x E_ch
       (Eq. 12; v1 discharges without charging).
    E3 Cap_CGS >= (1 + 1/N) x IT load (Eq. 16, N = 1 in Table 8; v1 uses 0.05).
    E4 2024-2025 calibration years: no DLC, immersion, PV or BESS additions
       and base-year shares (Section 3.9); R-CN shares move linearly from the
       2025 base to the 2040 target.
    E5 regional grid EF(r, t) (Eq. 10, Table 9) in the CO2 cap, carbon cost
       and reported emissions (v1 uses the national factor).
    E6 Table 9 / Section 3.2 regional inputs (checked against the inputs the
       caller passes: RegionData.source must be 'parameters_spec').
    E7 Cap_PV + E_PPA / (CF x 8.76) <= RE_max (Eq. 15; v1 bounds PV only).
    E8 5% CAPEX subsidy for non-metro DLC, immersion, CGS and BESS in R-CN
       (Table 7; the v1 flag Scenario.regional_subsidy is unused).

Decision variables, units and the annual aggregation are those of v1:
    cap_<tech>[r, t], delta_<tech>[r, t] (GW IT load / GW / GWh),
    E_<source>[r, t] and Curt[r, t] in TWh, plus E_ch[r, t] (TWh) with E2.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pulp

from .carbon import (
    carbon_price_trajectory,
    co2_cap_trajectory,
    grid_emission_factor,
)
from .config import (
    CARBON_POINTS,
    OVERRIDE_KEYS,
    SpecFixes,
    SpecReadings,
)
from .load_inputs import GlobalParameters, RegionData
from .pue import PUEParameters, pue_for_technology
from .scenarios import LEGACY_SHARE_ORDER, METRO, Scenario

CALIBRATION_YEARS = (2024, 2025)
SUBSIDY_RATE = 0.05
# 2024 shares of non-AI and AI load in the code's demand (P_base, P_AI_0).
NON_AI_BASE_TWH = 19.0


@dataclass
class SolveResult:
    scenario: str
    status: str
    objective_yen: float
    capex_total: float
    opex_total: float
    energy_cost: float
    carbon_cost: float
    curtail_cost: float
    # Annual time-series (year-indexed dict of region-indexed dict)
    annual: Dict[str, Dict[str, Dict[int, float]]]
    # Per-year, per-region snapshot table
    table: List[Dict[str, object]]
    # Spec-version additions
    subsidy_total: float = 0.0          # yen, discounted (E8 transfer)
    adder_total: float = 0.0            # yen, discounted (non-metro adder)
    curtail_TWh_total: float = 0.0
    components: List[Dict[str, object]] = field(default_factory=list)
    n_variables: int = 0
    n_constraints: int = 0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _weighted_avg_pue(tech: str, temps_df, region: str,
                     pp: PUEParameters,
                     regions: Dict[str, "RegionData"] | None = None,
                     air_poly=None) -> float:
    """Annual PUE for a region, evaluated at the regional annual mean
    temperature as in v1. ``air_poly`` = (a, b, c) replaces Eq. 8 for air
    cooling by a + b T + c T^2 (functional-form sensitivity)."""
    if regions is not None and region in regions:
        T_mean = regions[region].annual_mean_T_C
    else:
        grp = temps_df[temps_df["region"] == region]
        w = grp["weight_hours"].values
        T_mean = float(np.sum(grp["temperature_C"].values * w) / np.sum(w))
    if tech == "air" and air_poly is not None:
        a, b, c = air_poly
        return float(a + b * T_mean + c * T_mean ** 2)
    return float(pue_for_technology(tech, T_mean, pp))


def _national_it_load_TWh(p: GlobalParameters, year: int) -> float:
    """National IT load in TWh (Wood Mackenzie path of v1, Section 4.1)."""
    anchors = {
        2024: 19.0,
        2026: 27.0,
        2028: 37.0,
        2030: 47.0,
        2032: 57.0,
        2034: 61.0,
        2036: 65.0,
        2038: 68.0,
        2040: 70.0,
    }
    xs = sorted(anchors.keys())
    ys = [anchors[x] for x in xs]
    return float(np.interp(year, xs, ys))


def non_ai_TWh(p: GlobalParameters) -> float:
    """Non-AI part of the 2024 load, held flat: P_base / (P_base + P_AI) x 19 TWh."""
    return p.P_base_2024 / (p.P_base_2024 + p.P_AI_0) * NON_AI_BASE_TWH


def _bess_capex_yen_per_kWh(p: GlobalParameters, year: int) -> float:
    frac = (year - 2025) / (2035 - 2025)
    frac = max(0.0, min(1.0, frac))
    cost = p.bess_capex_2025 + (p.bess_capex_2035 - p.bess_capex_2025) * frac
    return cost * 1000.0   # yen thousand/kWh -> yen/kWh


def _apply_overrides(p: GlobalParameters, overrides: Mapping[str, object]
                     ) -> GlobalParameters:
    changes: Dict[str, Any] = {}
    if "capex" in overrides:
        capex = dict(overrides["capex"])  # type: ignore[call-overload]
        if "dlc" in capex:
            changes["cooling_dlc_capex"] = float(capex["dlc"])
        if "immersion" in capex:
            changes["cooling_immersion_capex"] = float(capex["immersion"])
    if "pue" in overrides:
        for k, v in dict(overrides["pue"]).items():  # type: ignore[call-overload]
            changes[k] = float(v)
    if "deploy_max_liquid" in overrides:
        changes["Deploy_max_liquid"] = float(overrides["deploy_max_liquid"])  # type: ignore[arg-type]
    if "discount_rate" in overrides:
        changes["discount_rate"] = float(overrides["discount_rate"])  # type: ignore[arg-type]
    if "carbon_points" in overrides:
        points: Any = overrides["carbon_points"]
        changes.update(P_carbon_2026=float(points[0]), P_carbon_2035=float(points[1]),
                       P_carbon_2040=float(points[2]))
    return replace(p, **changes) if changes else p


def _share_matrix(scenario: Scenario, regions: Dict[str, RegionData],
                  regs: List[str], years: List[int], p: GlobalParameters,
                  fixes: SpecFixes) -> np.ndarray:
    base_shares = np.array([regions[r].base_share_2024 for r in regs])
    if fixes.region_order:
        target_shares = np.array([scenario.share_2040[r] for r in regs])
    else:
        # v1: tuple ordered LEGACY_SHARE_ORDER applied positionally
        target_shares = np.array([scenario.share_2040[n] for n in LEGACY_SHARE_ORDER])
    shares = np.zeros((len(years), len(regs)))
    start = CALIBRATION_YEARS[-1] if fixes.calibration_years else p.horizon_start
    for ti, yr in enumerate(years):
        if fixes.calibration_years:
            frac = max(0.0, (yr - start) / (p.horizon_end - start))
        else:
            frac = (yr - p.horizon_start) / (p.horizon_end - p.horizon_start)
        shares[ti] = base_shares + (target_shares - base_shares) * frac
        shares[ti] /= shares[ti].sum()  # renormalise just in case
    return shares


# ----------------------------------------------------------------------
# Main builder
# ----------------------------------------------------------------------

def build_and_solve(scenario: Scenario,
                    regions: Dict[str, RegionData],
                    p: GlobalParameters,
                    temps_df,
                    fixes: SpecFixes = SpecFixes(),
                    readings: SpecReadings = SpecReadings(),
                    overrides: Optional[Mapping[str, object]] = None,
                    solver_msg: bool = False) -> SolveResult:
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(OVERRIDE_KEYS)
    if unknown:
        raise ValueError(f"unknown overrides: {sorted(unknown)}")
    sources = {reg.source for reg in regions.values()}
    expected = "parameters_spec" if fixes.table9_inputs else "parameters_csv"
    if sources != {expected}:
        raise ValueError(f"fix E6={fixes.table9_inputs} expects inputs from "
                         f"{expected}, got {sorted(sources)}")
    p = _apply_overrides(p, overrides)

    pp = PUEParameters(
        C_base=p.C_base, alpha=p.alpha, beta=p.beta,
        T_thresh=p.T_thresh, C_pump=p.C_pump, gamma=p.gamma,
    )
    years = list(range(p.horizon_start, p.horizon_end + 1))
    regs = list(regions.keys())
    air_poly = overrides.get("pue_air_poly")

    # Pre-compute per-region weighted-average PUE for each cooling tech
    pue_table = {
        (r, tech): _weighted_avg_pue(tech, temps_df, r, pp, regions, air_poly)
        for r in regs for tech in ("air", "dlc", "immersion")
    }
    ef_grid = grid_emission_factor(np.array(years), p)
    if "carbon_price" in overrides:
        path = dict(overrides["carbon_price"])  # type: ignore[call-overload]
        p_carb = np.array([float(path[t]) for t in years])
    elif readings.carbon == "mid":
        p_carb = carbon_price_trajectory(np.array(years), p)
    else:
        c26, c35, c40 = CARBON_POINTS[readings.carbon]
        p_carb = carbon_price_trajectory(
            np.array(years),
            replace(p, P_carbon_2026=c26, P_carbon_2035=c35, P_carbon_2040=c40))
    co2_cap = co2_cap_trajectory(np.array(years), p)  # Mt-CO2/yr

    # Grid EF by (region, year index): national (v1) or regional (E5)
    def ef_rt(r: str, ti: int) -> float:
        if fixes.regional_ef:
            base = regions[r].grid_ef_base
            if base is None:
                raise ValueError(f"fix E5 needs grid_ef_base for {r}")
            return float(base * ef_grid[ti] / p.EF_grid_2024)
        return float(ef_grid[ti])

    # ------------------- regional share trajectory --------------------
    shares = _share_matrix(scenario, regions, regs, years, p, fixes)

    # ------------------- LP --------------------
    prob = pulp.LpProblem(f"DC_DECOM_{scenario.name}", pulp.LpMinimize)

    techs_cool = ("air", "dlc", "immersion")
    techs_supply = ("pv", "bess", "cgs")
    energy_sources = ("grid", "ppa", "pv", "bess_disch", "cgs")

    cap = {(tech, r, t): pulp.LpVariable(f"cap_{tech}_{r}_{t}", lowBound=0)
           for tech in techs_cool + techs_supply for r in regs for t in years}
    delta = {(tech, r, t): pulp.LpVariable(f"d_{tech}_{r}_{t}", lowBound=0)
             for tech in techs_cool + techs_supply for r in regs for t in years}
    E = {(src, r, t): pulp.LpVariable(f"E_{src}_{r}_{t}", lowBound=0)
         for src in energy_sources for r in regs for t in years}
    curt = {(r, t): pulp.LpVariable(f"curt_{r}_{t}", lowBound=0)
            for r in regs for t in years}
    E_ch = ({(r, t): pulp.LpVariable(f"E_ch_{r}_{t}", lowBound=0)
             for r in regs for t in years} if fixes.bess_charge else {})

    # ------------------- capacity book-keeping --------------------
    # 2024 initial fleet: all air-cooling, distributed by 2024 shares
    p_total_2024 = p.P_base_2024 + p.P_AI_0   # GW of IT load
    init_air = {r: p_total_2024 * regions[r].base_share_2024 for r in regs}
    for r in regs:
        for t in years:
            prev = init_air[r] if t == years[0] else cap[("air", r, t - 1)]
            prob += cap[("air", r, t)] == prev + delta[("air", r, t)]
            for tech in ("dlc", "immersion", "pv", "bess", "cgs"):
                prev2 = 0 if t == years[0] else cap[(tech, r, t - 1)]
                prob += cap[(tech, r, t)] == prev2 + delta[(tech, r, t)]

    # ------------------- deployment-rate cap (Eq. 17) --------------------
    for r in regs:
        for t in years:
            prob += delta[("dlc", r, t)] + delta[("immersion", r, t)] \
                <= p.Deploy_max_liquid
            prob += delta[("pv", r, t)] <= p.Deploy_max_PV

    # ------------------- E4: calibration years (Section 3.9) ----------------
    if fixes.calibration_years:
        for r in regs:
            for t in years:
                if t in CALIBRATION_YEARS:
                    for tech in ("dlc", "immersion", "pv", "bess"):
                        prob += delta[(tech, r, t)] == 0

    # ------------------- annual IT load and total electricity --------------------
    ai_factor = overrides.get("ai_factor")
    nat_TWh = []
    for t in years:
        anchor = _national_it_load_TWh(p, t)
        if ai_factor is not None:
            base = non_ai_TWh(p)
            anchor = base + (anchor - base) * float(ai_factor[t])  # type: ignore[index]
        nat_TWh.append(anchor)
    it_total_GW = np.array([(x / 8.760) for x in nat_TWh])   # TWh -> GW * yr
    it_load_rt = {
        (r, t): float(it_total_GW[ti] * shares[ti, j])
        for ti, t in enumerate(years)
        for j, r in enumerate(regs)
    }
    if scenario.ccn_geography and readings.ccn_rule == "metro_only":
        # Table 7 reading: non-metro IT frozen at its 2025 level, growth after
        # 2025 split 60:22 between Kanto and Kansai.
        t0 = CALIBRATION_YEARS[-1]
        tot0 = sum(it_load_rt[(r, t0)] for r in regs)
        split = {"Kanto": 0.60 / 0.82, "Kansai": 0.22 / 0.82}
        for ti, t in enumerate(years):
            if t <= t0:
                continue
            growth = float(it_total_GW[ti]) - tot0
            for r in regs:
                if r in METRO:
                    it_load_rt[(r, t)] = it_load_rt[(r, t0)] + growth * split[r]
                else:
                    it_load_rt[(r, t)] = it_load_rt[(r, t0)]

    it_by_tech = {(tech, r, t): pulp.LpVariable(f"it_{tech}_{r}_{t}", lowBound=0)
                  for tech in techs_cool for r in regs for t in years}
    for r in regs:
        for t in years:
            prob += pulp.lpSum(it_by_tech[(tech, r, t)] for tech in techs_cool) \
                == it_load_rt[(r, t)]
            for tech in techs_cool:
                prob += it_by_tech[(tech, r, t)] <= cap[(tech, r, t)]

    # Facility electricity demand per region per year (GWh)
    util = readings.utilization
    fac_demand = {}
    for r in regs:
        for t in years:
            expr = pulp.lpSum(
                it_by_tech[(tech, r, t)] * pue_table[(r, tech)] * 8760.0 * util
                for tech in techs_cool
            )
            fac_demand[(r, t)] = expr

    # ------------------- energy balance (Eq. 12) --------------------
    for r in regs:
        for t in years:
            if fixes.bess_charge:
                prob += (
                    pulp.lpSum(E[(src, r, t)] for src in energy_sources)
                    + curt[(r, t)]
                    == fac_demand[(r, t)] / 1000.0 + E_ch[(r, t)]
                )
            else:
                prob += (
                    pulp.lpSum(E[(src, r, t)] for src in energy_sources)
                    + curt[(r, t)]
                    == fac_demand[(r, t)] / 1000.0
                )

    # ------------------- BESS / PV / CGS energy accounting --------------------
    cycles_per_yr = readings.bess_cycles
    cgs_cf = 1.0 if readings.cgs_cf is None else readings.cgs_cf
    for r in regs:
        for t in years:
            if cycles_per_yr is not None:
                prob += E[("bess_disch", r, t)] \
                    <= cap[("bess", r, t)] * cycles_per_yr / 1000.0 * p.bess_efficiency
            if fixes.bess_charge:
                prob += E[("bess_disch", r, t)] == p.bess_efficiency * E_ch[(r, t)]
            prob += E[("pv", r, t)] \
                <= cap[("pv", r, t)] * p.pv_cf_avg * 8.760 / 1000.0 * 1000.0
            prob += E[("cgs", r, t)] \
                <= cap[("cgs", r, t)] * cgs_cf * 8.760 / 1000.0 * 1000.0

    # ------------------- RE capacity bound (Eq. 15) --------------------
    onsite_pv_share = readings.pv_onsite_cap
    ppa_GW_per_TWh = 1.0 / (p.pv_cf_avg * 8.760)
    for r in regs:
        for t in years:
            if fixes.re_cap_eq15:
                prob += cap[("pv", r, t)] + E[("ppa", r, t)] * ppa_GW_per_TWh \
                    <= regions[r].re_max_GW
            else:
                prob += cap[("pv", r, t)] <= regions[r].re_max_GW
            if onsite_pv_share is not None:
                prob += E[("pv", r, t)] <= onsite_pv_share * (
                    fac_demand[(r, t)] / 1000.0
                )

    # ------------------- PPA market depth bound (code-only reading) ----------
    if readings.ppa_ceiling == "code":
        ppa_ceiling = {
            "Hokkaido": 0.50, "Tohoku": 0.45, "Kyushu": 0.45,
            "Kanto": 0.20, "Kansai": 0.20,
        }
        for ti, t in enumerate(years):
            ramp = (t - p.horizon_start) / (p.horizon_end - p.horizon_start)
            for r in regs:
                ceil_r = ppa_ceiling.get(r, 0.25)
                ppa_share_t = 0.05 + (ceil_r - 0.05) * ramp
                prob += E[("ppa", r, t)] <= ppa_share_t * (
                    fac_demand[(r, t)] / 1000.0
                )

    # ------------------- CGS dispatch fraction (code-only reading) ----------
    if readings.cgs_energy_cap is not None:
        for r in regs:
            for t in years:
                prob += E[("cgs", r, t)] <= readings.cgs_energy_cap * (
                    fac_demand[(r, t)] / 1000.0)

    # ------------------- Resilience (Eq. 16) --------------------
    # v1: backup >= 0.05 x IT; E3: backup >= (1 + 1/N) x IT (critical load = IT).
    reserve_factor = (1.0 + 1.0 / p.N_resilience) if fixes.n_plus_1 else 0.05
    for r in regs:
        for t in years:
            prob += cap[("cgs", r, t)] >= reserve_factor * it_load_rt[(r, t)]

    # ------------------- CO2 cap (Eq. 13-14) --------------------
    if scenario.apply_co2_cap:
        for ti, t in enumerate(years):
            if fixes.regional_ef:
                total_emissions = pulp.lpSum(
                    E[("grid", r, t)] * ef_rt(r, ti)
                    + E[("cgs", r, t)] * p.EF_fossil
                    for r in regs
                )
            else:
                total_emissions = pulp.lpSum(
                    E[("grid", r, t)] * ef_grid[ti]
                    + E[("cgs", r, t)] * p.EF_fossil
                    for r in regs
                )
            prob += total_emissions <= co2_cap[ti]

    # ------------------- Objective function (Eq. 1) --------------------
    capex_unit_yen_per_GW = {
        "air":       0.0,                                # baseline reference
        "dlc":       (p.cooling_dlc_capex - p.cooling_air_capex) * 1e6 * 1000.0,
        "immersion": (p.cooling_immersion_capex - p.cooling_air_capex) * 1e6 * 1000.0,
        "pv":        0.0,                                # included in LCOE
        "cgs":       p.cgs_capex * 1e6 * 1000.0,
    }
    om_rate = p.opex_om_rate
    siting_premium = {r: (readings.siting_premium if r in METRO else 1.0) for r in regs}
    subsidised = fixes.regional_subsidy and scenario.regional_subsidy
    sub_factor = {r: ((1.0 - SUBSIDY_RATE) if (subsidised and r not in METRO) else 1.0)
                  for r in regs}
    ppa_offset = float(overrides.get("ppa_offset", 0.0))  # type: ignore[arg-type]
    adder = float(overrides.get("nonmetro_adder_yen_per_kW", 0.0))  # type: ignore[arg-type]

    capex_terms = []
    opex_terms = []
    energy_terms = []
    carbon_terms = []
    curt_terms = []
    disc_t = {t: 1.0 / (1.0 + p.discount_rate) ** (t - p.horizon_start) for t in years}
    price_t = {t: float(p_carb[ti]) for ti, t in enumerate(years)}

    for ti, t in enumerate(years):
        disc = disc_t[t]
        for r in regs:
            premium = siting_premium[r]
            sub = sub_factor[r]
            for tech in ("dlc", "immersion", "cgs"):
                unit = capex_unit_yen_per_GW[tech] * premium
                if sub != 1.0:
                    unit = unit * sub
                capex_terms.append(delta[(tech, r, t)] * unit * disc)
            bess_unit_yen_per_GWh = _bess_capex_yen_per_kWh(p, t) * 1e6 * premium
            bess_capex_unit = bess_unit_yen_per_GWh * sub if sub != 1.0 \
                else bess_unit_yen_per_GWh
            capex_terms.append(delta[("bess", r, t)] * bess_capex_unit * disc)
            for tech in ("dlc", "immersion", "cgs"):
                unit = capex_unit_yen_per_GW[tech] * premium
                opex_terms.append(cap[(tech, r, t)] * unit * om_rate * disc)
            opex_terms.append(cap[("bess", r, t)] * bess_unit_yen_per_GWh * om_rate * disc)
            tariff = regions[r].tariff_yen_per_kWh    # yen/kWh
            ppa_price = regions[r].ppa_yen_per_kWh
            if ppa_offset != 0.0:
                ppa_price = ppa_price + ppa_offset
            energy_terms.append(E[("grid", r, t)] * tariff * 1e9 * disc)
            energy_terms.append(E[("ppa", r, t)] * ppa_price * 1e9 * disc)
            energy_terms.append(E[("pv", r, t)] * p.pv_lcoe * 1e9 * disc)
            energy_terms.append(
                E[("cgs", r, t)] * (8.0 / p.cgs_eff_elec) * 1e9 * disc
            )
            ef_g = ef_rt(r, ti)
            price = price_t[t]
            carbon_terms.append(E[("grid", r, t)] * 1e6 * ef_g * price * disc)
            carbon_terms.append(E[("cgs", r, t)] * 1e6 * p.EF_fossil * price * disc)
            curt_terms.append(curt[(r, t)] * 1e9 * p.curtail_penalty * disc)

    # Non-metro adder (sensitivity): one-time cost per kW of new IT capacity.
    adder_rt: Dict[tuple, float] = {}
    if adder != 0.0:
        for t in years[1:]:
            for r in regs:
                if r in METRO:
                    continue
                new_GW = max(0.0, it_load_rt[(r, t)] - it_load_rt[(r, t - 1)])
                adder_rt[(r, t)] = adder * new_GW * 1e6 * disc_t[t]
    adder_total = float(sum(adder_rt.values()))

    objective = (pulp.lpSum(capex_terms) + pulp.lpSum(opex_terms)
                 + pulp.lpSum(energy_terms) + pulp.lpSum(carbon_terms)
                 + pulp.lpSum(curt_terms))
    if adder_rt:
        objective = objective + adder_total
    prob += objective

    # ------------------- Solve --------------------
    solver = pulp.PULP_CBC_CMD(msg=1 if solver_msg else 0, threads=1)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    def _val(v) -> float:
        try:
            return float(pulp.value(v) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    capex_total = _val(pulp.lpSum(capex_terms))
    opex_total = _val(pulp.lpSum(opex_terms))
    energy_cost = _val(pulp.lpSum(energy_terms))
    carbon_cost_v = _val(pulp.lpSum(carbon_terms))
    curt_cost = _val(pulp.lpSum(curt_terms))
    obj = capex_total + opex_total + energy_cost + carbon_cost_v + curt_cost + adder_total

    annual: Dict[str, Dict[str, Dict[int, float]]] = {
        "capacity": {}, "energy": {}, "emissions": {}, "load": {}
    }
    table: List[Dict[str, object]] = []
    components: List[Dict[str, object]] = []
    subsidy_total = 0.0
    curt_TWh_total = 0.0
    for ti, t in enumerate(years):
        disc = disc_t[t]
        price = price_t[t]
        for r in regs:
            energy_kwh = {src: _val(E[(src, r, t)]) for src in energy_sources}
            total_E = sum(energy_kwh.values())
            ef_g = ef_rt(r, ti)
            e_ch = _val(E_ch[(r, t)]) if fixes.bess_charge else 0.0
            curt_v = _val(curt[(r, t)])
            curt_TWh_total += curt_v
            load_TWh = _val(fac_demand[(r, t)]) / 1000.0
            it_energy_TWh = it_load_rt[(r, t)] * 8.760 * util
            emissions_grid = energy_kwh["grid"] * ef_g
            emissions_cgs = energy_kwh["cgs"] * p.EF_fossil
            emissions_Mt = emissions_grid + emissions_cgs  # TWh * kg/kWh = Mt
            d = {tech: _val(delta[(tech, r, t)]) for tech in techs_cool + techs_supply}
            c = {tech: _val(cap[(tech, r, t)]) for tech in techs_cool + techs_supply}
            row: Dict[str, object] = {
                "year": t, "region": r,
                "it_load_GW": it_load_rt[(r, t)],
                "facility_load_TWh": total_E,
                **{f"E_{k}_TWh": v for k, v in energy_kwh.items()},
                **{f"cap_{tech}_GW": c[tech] for tech in techs_cool},
                "cap_pv_GW": c["pv"],
                "cap_bess_GWh": c["bess"],
                "cap_cgs_GW": c["cgs"],
                "emissions_Mt": emissions_Mt,
                # spec additions
                **{f"delta_{tech}": d[tech] for tech in techs_cool + techs_supply},
                **{f"it_{tech}_GW": _val(it_by_tech[(tech, r, t)]) for tech in techs_cool},
                "E_ch_TWh": e_ch,
                "curt_TWh": curt_v,
                "load_TWh": load_TWh,
                "it_energy_TWh": it_energy_TWh,
                "pue_air": pue_table[(r, "air")],
                "pue_dlc": pue_table[(r, "dlc")],
                "pue_immersion": pue_table[(r, "immersion")],
                "ef_grid": ef_g,
                "carbon_price": price,
                "emissions_grid_Mt": emissions_grid,
                "emissions_cgs_Mt": emissions_cgs,
            }
            table.append(row)

            # Discounted cost components (yen); gross of subsidy, premium split.
            premium = siting_premium[r]
            sub = sub_factor[r]
            comp: Dict[str, float] = {}
            for name, techs in (("cooling", ("dlc", "immersion")), ("cgs", ("cgs",))):
                base_capex = sum(d[x] * capex_unit_yen_per_GW[x] for x in techs) * disc
                base_opex = sum(c[x] * capex_unit_yen_per_GW[x] for x in techs) * om_rate * disc
                comp[f"{name}_capex"] = base_capex
                comp[f"{name}_opex"] = base_opex
                comp[f"{name}_premium"] = (base_capex + base_opex) * (premium - 1.0)
            bess_base = _bess_capex_yen_per_kWh(p, t) * 1e6
            comp["bess_capex"] = d["bess"] * bess_base * disc
            comp["bess_opex"] = c["bess"] * bess_base * om_rate * disc
            comp["bess_premium"] = (comp["bess_capex"] + comp["bess_opex"]) * (premium - 1.0)
            comp["pv_capex"] = 0.0   # v1: PV CAPEX is included in the LCOE
            comp["pv_opex"] = 0.0
            ppa_price = regions[r].ppa_yen_per_kWh + ppa_offset
            comp["energy_grid"] = energy_kwh["grid"] * regions[r].tariff_yen_per_kWh * 1e9 * disc
            comp["energy_ppa"] = energy_kwh["ppa"] * ppa_price * 1e9 * disc
            comp["energy_pv"] = energy_kwh["pv"] * p.pv_lcoe * 1e9 * disc
            comp["energy_cgs_fuel"] = energy_kwh["cgs"] * (8.0 / p.cgs_eff_elec) * 1e9 * disc
            comp["carbon_grid"] = energy_kwh["grid"] * 1e6 * ef_g * price * disc
            comp["carbon_cgs"] = energy_kwh["cgs"] * 1e6 * p.EF_fossil * price * disc
            comp["curtailment"] = curt_v * 1e9 * p.curtail_penalty * disc
            gross_capex = (comp["cooling_capex"] + comp["cgs_capex"] + comp["bess_capex"]) \
                * premium
            comp["subsidy"] = gross_capex * (1.0 - sub)
            comp["adder"] = adder_rt.get((r, t), 0.0)
            subsidy_total += comp["subsidy"]
            procured = (energy_kwh["grid"] + energy_kwh["ppa"] + energy_kwh["pv"]
                        + energy_kwh["cgs"])
            components.append({"year": t, "region": r, "procured_TWh": procured, **comp})

    return SolveResult(
        scenario=scenario.name,
        status=status,
        objective_yen=obj,
        capex_total=capex_total,
        opex_total=opex_total,
        energy_cost=energy_cost,
        carbon_cost=carbon_cost_v,
        curtail_cost=curt_cost,
        annual=annual,
        table=table,
        subsidy_total=subsidy_total,
        adder_total=adder_total,
        curtail_TWh_total=curt_TWh_total,
        components=components,
        n_variables=len(prob.variables()),
        n_constraints=len(prob.constraints),
    )


COMPONENT_COLUMNS = (
    "cooling_capex", "cooling_opex", "cooling_premium",
    "cgs_capex", "cgs_opex", "cgs_premium",
    "bess_capex", "bess_opex", "bess_premium",
    "pv_capex", "pv_opex",
    "energy_grid", "energy_ppa", "energy_pv", "energy_cgs_fuel",
    "carbon_grid", "carbon_cgs", "curtailment", "subsidy", "adder",
)


def components_total(result: SolveResult) -> float:
    """Sum of the cost components minus the subsidy (equals objective_yen)."""
    tot = 0.0
    for row in result.components:
        for k in COMPONENT_COLUMNS:
            v = float(row[k])  # type: ignore[arg-type]
            tot += -v if k == "subsidy" else v
    return tot
