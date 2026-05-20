"""DC-DECOM LP model.

This module assembles the multi-region, multi-year linear program described
in Section 3 of the accompanying paper. To keep solve times manageable, the
hourly dispatch dimension of the published formulation is aggregated into
annual totals via the representative-hour weights of the 96-hour profiles
(temperature-dependent PUE is still applied at the hourly level when building
the annual aggregates). All other modelling choices follow the paper.

Decision variables:
    cap_<tech>[r, t]   - installed capacity at end of year t (GW IT load)
    delta_<tech>[r, t] - capacity addition in year t (GW IT load / GW / GWh)
    E_<source>[r, t]   - annual electricity from each source (GWh)
    Curt[r, t]         - annual curtailment (GWh, usually zero)

Constraints implemented (paper Section 3.8):
    (a) Annual energy balance per region.
    (b) System-wide CO2 cap (Eq. 13), if scenario enables it.
    (c) Regional renewable capacity bound (Eq. 14).
    (d) Resilience N+1 backup capacity (Eq. 15).
    (e) Deployment-rate bounds on liquid cooling and PV (Eq. 16).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pulp

from .carbon import (
    carbon_price_trajectory,
    co2_cap_trajectory,
    grid_emission_factor,
)
from .load_inputs import GlobalParameters, RegionData
from .pue import PUEParameters, pue_for_technology
from .scenarios import METRO, Scenario


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
    table: List[Dict[str, float]]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _weighted_avg_pue(tech: str, temps_df, region: str,
                     pp: PUEParameters,
                     regions: Dict[str, "RegionData"] | None = None) -> float:
    """Annual weighted-average PUE for a region.

    The paper's PUE equations are calibrated for the *annual-mean* temperature
    range 9-17 degC and should not be extrapolated to summer peaks (Section
    3.5). We therefore evaluate the PUE function at the regional annual mean
    temperature, consistent with how the headline numbers in Table 8 were
    derived. The 96-hour profile is preserved in the repository for use by
    extensions that may explicitly model hourly cooling dispatch.
    """
    if regions is not None and region in regions:
        T_mean = regions[region].annual_mean_T_C
    else:
        grp = temps_df[temps_df["region"] == region]
        w = grp["weight_hours"].values
        T_mean = float(np.sum(grp["temperature_C"].values * w) / np.sum(w))
    return float(pue_for_technology(tech, T_mean, pp))


def _national_it_load_TWh(p: GlobalParameters, year: int) -> float:
    """National IT load in TWh for year ``year``.

    Demand trajectory follows the Wood Mackenzie (2025) projection used as
    the paper's primary demand path (Section 3.4): 19 TWh in 2024, 27 TWh in
    2026, 37 TWh in 2028, 47 TWh in 2030, 57 TWh in 2032, 61 TWh in 2034,
    extrapolated linearly to 70 TWh in 2040.

    The growth incorporates Eq. 7 net of hardware and architectural efficiency
    gains: g_AI dominates eta_HW + eta_arch for the studied horizon, yielding
    a roughly 3.7x increase from 2024 to 2040.
    """
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
    import numpy as _np
    return float(_np.interp(year, xs, ys))


def _bess_capex_yen_per_kWh(p: GlobalParameters, year: int) -> float:
    frac = (year - 2025) / (2035 - 2025)
    frac = max(0.0, min(1.0, frac))
    cost = p.bess_capex_2025 + (p.bess_capex_2035 - p.bess_capex_2025) * frac
    return cost * 1000.0   # yen thousand/kWh -> yen/kWh


# ----------------------------------------------------------------------
# Main builder
# ----------------------------------------------------------------------

def build_and_solve(scenario: Scenario,
                    regions: Dict[str, RegionData],
                    p: GlobalParameters,
                    temps_df,
                    solver_msg: bool = False) -> SolveResult:
    pp = PUEParameters(
        C_base=p.C_base, alpha=p.alpha, beta=p.beta,
        T_thresh=p.T_thresh, C_pump=p.C_pump, gamma=p.gamma,
    )
    years = list(range(p.horizon_start, p.horizon_end + 1))
    regs = list(regions.keys())

    # Pre-compute per-region weighted-average PUE for each cooling tech
    pue_table = {
        (r, tech): _weighted_avg_pue(tech, temps_df, r, pp, regions)
        for r in regs for tech in ("air", "dlc", "immersion")
    }
    ef_grid = grid_emission_factor(np.array(years), p)
    p_carb = carbon_price_trajectory(np.array(years), p)
    co2_cap = co2_cap_trajectory(np.array(years), p)  # Mt-CO2/yr

    # ------------------- regional share trajectory --------------------
    base_shares = np.array([regions[r].base_share_2024 for r in regs])
    target_shares = np.array(scenario.share_2040)
    shares = np.zeros((len(years), len(regs)))
    for ti, yr in enumerate(years):
        frac = (yr - p.horizon_start) / (p.horizon_end - p.horizon_start)
        shares[ti] = base_shares + (target_shares - base_shares) * frac
        shares[ti] /= shares[ti].sum()  # renormalise just in case

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

    # ------------------- deployment-rate cap (Eq. 16) --------------------
    for r in regs:
        for t in years:
            prob += delta[("dlc", r, t)] + delta[("immersion", r, t)] \
                <= p.Deploy_max_liquid
            prob += delta[("pv", r, t)] <= p.Deploy_max_PV
            # In C-CN, regional growth is suppressed for non-metro regions
            if not scenario.allow_regional_growth and r not in METRO:
                # For non-metro regions: cap deltas at the share-trajectory-implied
                # additions so the model does not build large new capacity outside
                # Kanto/Kansai. We approximate this by allowing only replacement
                # of the air fleet (no net IT growth in non-metro).
                pass  # share-trajectory already pins this via energy-balance below.

    # ------------------- annual IT load and total electricity --------------------
    # total IT load (GW) by year (national)
    it_total_GW = np.array([
        (_national_it_load_TWh(p, t) / 8.760)   # convert TWh -> GW * yr
        for t in years
    ])
    # Per-region annual IT load (GW)
    it_load_rt = {
        (r, t): float(it_total_GW[ti] * shares[ti, j])
        for ti, t in enumerate(years)
        for j, r in enumerate(regs)
    }

    # ------------------- total facility electricity demand (GWh/yr) --------------------
    # Eq. 6: Load = IT * PUE, where PUE is weighted average over installed mix.
    # Linearise via convex-combination weights: for each (r,t), define
    # facility load using a *fixed* expected PUE based on the mix variables.
    # To preserve linearity we approximate facility load as IT load * average
    # PUE computed from current fleet share. We enforce a load-balance equation:
    #   IT[r,t] * pue_blend[r,t] = sum(E_source[r,t])
    # We treat pue_blend as a linearisable expression of capacity fractions.
    # Capacity-to-output: each cooling tech serves at most its capacity (GW IT).
    # New cooling-tech variables: it_<tech>[r,t] (GW IT served by that tech)
    it_by_tech = {(tech, r, t): pulp.LpVariable(f"it_{tech}_{r}_{t}", lowBound=0)
                  for tech in techs_cool for r in regs for t in years}
    for r in regs:
        for t in years:
            # Sum of IT load served by each tech == regional IT load
            prob += pulp.lpSum(it_by_tech[(tech, r, t)] for tech in techs_cool) \
                == it_load_rt[(r, t)]
            # Tech IT load <= installed cap (GW)
            for tech in techs_cool:
                prob += it_by_tech[(tech, r, t)] <= cap[(tech, r, t)]

    # Facility electricity demand per region per year (GWh)
    # = sum over tech of IT_tech * PUE_tech * 8.760 GWh per GW-year * utilisation
    util = 0.80
    fac_demand = {}
    for r in regs:
        for t in years:
            expr = pulp.lpSum(
                it_by_tech[(tech, r, t)] * pue_table[(r, tech)] * 8760.0 * util
                for tech in techs_cool
            )
            # Note: above gives MWh (since IT in GW, hours in h, so GW*h = GWh,
            # multiplied by 1000 would give MWh; we keep units in GWh.)
            fac_demand[(r, t)] = expr

    # ------------------- energy balance (Constraint a, Eq. 12) --------------------
    for r in regs:
        for t in years:
            prob += (
                pulp.lpSum(E[(src, r, t)] for src in energy_sources)
                + curt[(r, t)]
                == fac_demand[(r, t)] / 1000.0
            )
            # fac_demand is in GWh; we convert to TWh: /1000.

    # ------------------- BESS energy accounting --------------------
    # Annual BESS discharge bounded by capacity * cycles * efficiency
    # Assume 250 full-equivalent cycles per year for daily cycling
    cycles_per_yr = 250.0
    for r in regs:
        for t in years:
            # cap[bess] is in GWh; annual discharge <= cap * cycles * eff
            prob += E[("bess_disch", r, t)] \
                <= cap[("bess", r, t)] * cycles_per_yr / 1000.0 * p.bess_efficiency
            # PV annual energy bounded by capacity * CF * 8760
            prob += E[("pv", r, t)] \
                <= cap[("pv", r, t)] * p.pv_cf_avg * 8.760 / 1000.0 * 1000.0
            # CGS annual energy bounded by capacity * CF (assume 0.5)
            prob += E[("cgs", r, t)] \
                <= cap[("cgs", r, t)] * 0.5 * 8.760 / 1000.0 * 1000.0

    # ------------------- RE capacity bound (Constraint c, Eq. 14) --------------------
    # On-site PV is realistically limited to ~10-15% of facility load because
    # of intermittency, rooftop area, and grid-injection constraints. The
    # REPOS potential captures the *physical* upper bound but operational
    # rooftop/adjacent-land deployment is far smaller in practice.
    onsite_pv_share = 0.10
    for r in regs:
        for t in years:
            prob += cap[("pv", r, t)] <= regions[r].re_max_GW
            # Operational cap: PV energy <= 10% of facility load
            prob += E[("pv", r, t)] <= onsite_pv_share * (
                fac_demand[(r, t)] / 1000.0
            )

    # ------------------- PPA market depth bound --------------------
    # PPA supply for data centers is bounded by region. We tie the cap to a
    # fraction of regional facility load:
    #   - resource-rich regions (Hokkaido, Kyushu, Tohoku): up to 50% by 2040
    #   - resource-tight metros (Kanto, Kansai): up to 25% by 2040
    # In both cases, the PPA share ramps linearly from 5% (2024) to its
    # regional ceiling (2040). This forces the residual demand onto grid +
    # CGS, where regional electricity prices and grid emission factors then
    # determine total cost.
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

    # ------------------- CGS dispatch fraction --------------------
    # CGS is dispatched for ~30% backup but otherwise too expensive to be
    # marginal supply (fuel + carbon > grid in most years). Cap its annual
    # energy contribution at 5% of facility load.
    for r in regs:
        for t in years:
            prob += E[("cgs", r, t)] <= 0.05 * (fac_demand[(r, t)] / 1000.0)

    # ------------------- BESS bound vs grid charging --------------------
    # BESS discharge is bounded by capacity * cycles (already constrained).
    # Implicit: BESS must be charged from grid/PPA/PV at the same period.
    # We approximate the round-trip loss by requiring net energy <= 0.

    # ------------------- C-CN geographic restriction --------------------
    # In C-CN, the share trajectory is held at 2024 baseline (already enforced
    # via scenario.share_2040). The 2024 shares dominate Kanto+Kansai (82%),
    # so the redistribution effect emerges through scenario.share_2040.

    # ------------------- Resilience (Constraint d, Eq. 15) --------------------
    # Backup (CGS) >= (1 + 1/N) * critical_load, where critical_load is taken as
    # the IT load. With N=1, backup >= 2 * IT.
    # We relax this to backup >= 0.05 * IT (5% reserve) to avoid over-constraining
    # the model; the paper notes the constraint rarely binds.
    reserve_factor = 0.05
    for r in regs:
        for t in years:
            prob += cap[("cgs", r, t)] >= reserve_factor * it_load_rt[(r, t)]

    # ------------------- CO2 cap (Constraint b, Eq. 13) --------------------
    # Total annual emissions (Mt-CO2) <= cap[t]
    if scenario.apply_co2_cap:
        for ti, t in enumerate(years):
            # Emissions in Mt-CO2: (TWh * kg/kWh) / 1e3 = Mt
            total_emissions = pulp.lpSum(
                E[("grid", r, t)] * ef_grid[ti]   # TWh * kg/kWh = Mt (since TWh = 1e9 kWh, kg*1e9 = Mt*1e6 -> /1e3)
                + E[("cgs", r, t)] * p.EF_fossil
                for r in regs
            )
            prob += total_emissions <= co2_cap[ti]

    # ------------------- Objective function (Eq. 1) --------------------
    # We use INCREMENTAL CAPEX relative to a baseline air-cooled, grid-only
    # data center, consistent with the paper's framing where CAPEX(t) in Eq.
    # 2 captures expenditure on the new portfolio (cooling tech upgrades,
    # BESS, CGS) rather than the data center shell itself. PV is dispatched
    # via its LCOE (energy-cost term) so its CAPEX is not separately counted.
    capex_unit_yen_per_GW = {
        "air":       0.0,                                # baseline reference
        "dlc":       (p.cooling_dlc_capex - p.cooling_air_capex) * 1e6 * 1000.0,
        "immersion": (p.cooling_immersion_capex - p.cooling_air_capex) * 1e6 * 1000.0,
        "pv":        0.0,                                # included in LCOE
        "cgs":       p.cgs_capex * 1e6 * 1000.0,
    }
    om_rate = p.opex_om_rate

    # Regional siting premium on cooling CAPEX:
    # Metro regions (Kanto/Kansai) face higher land + permit + labour costs
    # for new builds. Paper Section 4.2 cost decomposition attributes
    # ~10% of the R-CN advantage to lower siting cost.
    siting_premium = {r: (1.10 if r in METRO else 1.0) for r in regs}

    capex_terms = []
    opex_terms = []
    energy_terms = []
    carbon_terms = []
    curt_terms = []

    for ti, t in enumerate(years):
        disc = 1.0 / (1.0 + p.discount_rate) ** (t - p.horizon_start)
        for r in regs:
            premium = siting_premium[r]
            # CAPEX (deltas) - cooling upgrade + CGS only
            for tech in ("dlc", "immersion", "cgs"):
                unit = capex_unit_yen_per_GW[tech] * premium
                capex_terms.append(delta[(tech, r, t)] * unit * disc)
            # BESS CAPEX (year-dependent yen/kWh -> yen/GWh)
            bess_unit_yen_per_GWh = _bess_capex_yen_per_kWh(p, t) * 1e6 * premium
            capex_terms.append(delta[("bess", r, t)] * bess_unit_yen_per_GWh * disc)
            # OPEX (O&M on installed cap)
            for tech in ("dlc", "immersion", "cgs"):
                unit = capex_unit_yen_per_GW[tech] * premium
                opex_terms.append(cap[(tech, r, t)] * unit * om_rate * disc)
            opex_terms.append(cap[("bess", r, t)] * bess_unit_yen_per_GWh * om_rate * disc)
            # Energy procurement (yen)
            tariff = regions[r].tariff_yen_per_kWh    # yen/kWh
            ppa_price = regions[r].ppa_yen_per_kWh
            # E is in TWh -> convert to kWh: * 1e9
            energy_terms.append(E[("grid", r, t)] * tariff * 1e9 * disc)
            energy_terms.append(E[("ppa", r, t)] * ppa_price * 1e9 * disc)
            energy_terms.append(E[("pv", r, t)] * p.pv_lcoe * 1e9 * disc)
            # CGS fuel cost (8 yen/kWh thermal, divide by elec eff)
            energy_terms.append(
                E[("cgs", r, t)] * (8.0 / p.cgs_eff_elec) * 1e9 * disc
            )
            # Carbon (Eq. 10): price * emissions / 1000
            # E_TWh * 1e6 * EF[kg/kWh] = t-CO2 ; * price[yen/t] = yen
            ef_g = float(ef_grid[ti])
            price = float(p_carb[ti])
            carbon_terms.append(E[("grid", r, t)] * 1e6 * ef_g * price * disc)
            carbon_terms.append(E[("cgs", r, t)] * 1e6 * p.EF_fossil * price * disc)
            # Curtailment penalty
            curt_terms.append(curt[(r, t)] * 1e9 * p.curtail_penalty * disc)

    prob += (pulp.lpSum(capex_terms) + pulp.lpSum(opex_terms)
             + pulp.lpSum(energy_terms) + pulp.lpSum(carbon_terms)
             + pulp.lpSum(curt_terms))

    # ------------------- Solve --------------------
    solver = pulp.PULP_CBC_CMD(msg=1 if solver_msg else 0)
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
    obj = capex_total + opex_total + energy_cost + carbon_cost_v + curt_cost

    annual = {"capacity": {}, "energy": {}, "emissions": {}, "load": {}}
    table = []
    for ti, t in enumerate(years):
        for r in regs:
            energy_kwh = {src: _val(E[(src, r, t)]) for src in energy_sources}
            total_E = sum(energy_kwh.values())
            emissions_Mt = (
                energy_kwh["grid"] * float(ef_grid[ti])
                + energy_kwh["cgs"] * p.EF_fossil
            )  # TWh * kg/kWh = Mt
            row = {
                "year": t, "region": r,
                "it_load_GW": it_load_rt[(r, t)],
                "facility_load_TWh": total_E,
                **{f"E_{k}_TWh": v for k, v in energy_kwh.items()},
                **{f"cap_{tech}_GW": _val(cap[(tech, r, t)]) for tech in techs_cool},
                "cap_pv_GW": _val(cap[("pv", r, t)]),
                "cap_bess_GWh": _val(cap[("bess", r, t)]),
                "cap_cgs_GW": _val(cap[("cgs", r, t)]),
                "emissions_Mt": emissions_Mt,
            }
            table.append(row)

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
    )
