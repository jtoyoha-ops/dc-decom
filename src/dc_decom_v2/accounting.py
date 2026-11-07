"""Evaluation of a solved DC-DECOM v2 run from its flows (never from the objective).

Cost lines per region and year (yen; discounted to 2024 in the totals):
  resource  cooling_capex/opex (liquid premium over air), bess_capex/opex,
            cgs_capex/opex, gen_capex/opex, energy_grid (tariff x (G - PPA)
            with annual matching, tariff x G with hourly matching),
            energy_ppa, energy_nfc (certificates on top of the tariff),
            energy_pv (LCOE), energy_cgs_fuel, curtailment (penalty),
            reinforcement (full cost of the connected peak added in reinforced
            years), over_cost (connection overrun), shell_premium,
            comm_cost (communication / data-transfer cost: comm_rate yen/kWh
            x IT energy; 0 unless comm_cost_yen_per_kWh is set)
  transfer  carbon_grid, carbon_cgs (full carbon price x emissions), subsidy

Totals:
  social resource cost = sum of the resource lines
  Eq. 1 total          = social resource cost + carbon cost (published definition)
  operator cost        = social - (1 - theta) reinforcement
                         + pass_through x carbon cost - subsidy   (= LP objective)
  fiscal burden        = subsidy
Emissions em = EF x (G - PPA - nfc) + EF_fossil x CGS (annual matching) or
EF x (G - nfc) + EF_fossil x CGS (hourly matching), Mt-CO2.
Connection usage (conn_usage_last_max) = cumulative use from the physical
facility peak / cumulative offered capacity (``physical_connection_use``);
conn_connected_last_max is the same ratio from the connected-peak variable.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .config import CALIBRATION_YEARS, COOLING, END_YEAR, METRO

RESOURCE_COLUMNS = (
    "cooling_capex", "cooling_opex", "bess_capex", "bess_opex", "cgs_capex", "cgs_opex",
    "gen_capex", "gen_opex", "energy_grid", "energy_ppa", "energy_nfc", "energy_pv",
    "energy_cgs_fuel", "curtailment", "reinforcement", "over_cost", "shell_premium",
    "comm_cost",
)
CARBON_COLUMNS = ("carbon_grid", "carbon_cgs")
COST_COLUMNS = RESOURCE_COLUMNS + CARBON_COLUMNS + ("subsidy",)
_SLICE_ENERGY = {"L_GW": "facility_TWh", "g_GW": "grid_received_TWh", "pv_GW": "pv_TWh",
                 "cgs_GW": "cgs_TWh", "ch_GW": "bess_ch_TWh", "dis_GW": "bess_dis_TWh",
                 "curt_GW": "curt_TWh", "ppa_GW": "ppa_delivered_TWh", "spill_GW": "spill_TWh"}


def slice_energy(flows: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Annual energies (sum_h w_h x value / 1000, TWh) and peaks (GW) per region-year."""
    s = flows["slices"]
    e = s[list(_SLICE_ENERGY)].mul(s["weight_h"], axis=0) / 1000.0
    e = e.rename(columns=_SLICE_ENERGY)
    e["year"], e["region"] = s["year"], s["region"]
    out = e.groupby(["year", "region"], sort=False).sum()
    pk = s.groupby(["year", "region"], sort=False)[["L_GW", "g_GW"]].max()
    out["peak_facility_GW"] = pk["L_GW"]
    out["peak_grid_GW"] = pk["g_GW"]
    return out.reset_index()


def region_year(flows: Dict[str, pd.DataFrame], meta: Dict[str, Any]) -> pd.DataFrame:
    a = flows["annual"].merge(slice_energy(flows), on=["year", "region"], how="left",
                              validate="one_to_one")
    n = int(meta["n_tiers"])
    hourly = meta["ppa_matching"] == "hourly"
    a["ppa_TWh"] = sum(a[f"ppa_t{i + 1}_TWh"] for i in range(n))
    ppa_cost = sum(a[f"ppa_t{i + 1}_TWh"] * a[f"ppa_price_t{i + 1}"] for i in range(n))
    tariffed = a["grid_received_TWh"] if hourly else a["grid_received_TWh"] - a["ppa_TWh"]
    a["grid_net_TWh"] = tariffed - a["nfc_TWh"]
    a["it_energy_TWh"] = a["itr_GW"] * a["u_energy"] * 8.76
    a["it_share"] = a["itr_GW"] / a.groupby("year")["itr_GW"].transform("sum")
    a["em_grid_Mt"] = a["ef_grid"] * a["grid_net_TWh"]
    a["em_cgs_Mt"] = a["ef_fossil"] * a["cgs_TWh"]
    a["em_Mt"] = a["em_grid_Mt"] + a["em_cgs_Mt"]
    G9, Y6 = 1e9, 1e6
    liquid_new = a["prem_dlc"] * a["new_dlc_GW"] + a["prem_imm"] * a["new_imm_GW"]
    liquid_cap = a["prem_dlc"] * a["cap_dlc_GW"] + a["prem_imm"] * a["cap_imm_GW"]
    c = pd.DataFrame(index=a.index)
    c["cooling_capex"] = liquid_new * G9
    c["cooling_opex"] = liquid_cap * a["om_rate"] * G9
    c["bess_capex"] = a["bess_capex"] * a["new_bess_GWh"] * G9
    c["bess_opex"] = a["bess_capex"] * a["cap_bess_GWh"] * a["om_rate"] * G9
    c["cgs_capex"] = a["cgs_capex"] * a["new_cgs_GW"] * G9
    c["cgs_opex"] = a["cgs_capex"] * a["cap_cgs_GW"] * a["om_rate"] * G9
    c["gen_capex"] = a["gen_capex"] * a["new_gen_GW"] * G9
    c["gen_opex"] = a["gen_capex"] * a["cap_gen_GW"] * a["om_rate"] * G9
    c["energy_grid"] = a["tariff"] * tariffed * G9
    c["energy_ppa"] = ppa_cost * G9
    c["energy_nfc"] = a["nfc_price"] * a["nfc_TWh"] * G9
    c["energy_pv"] = a["pv_lcoe"] * a["pv_TWh"] * G9
    c["energy_cgs_fuel"] = a["cgs_fuel_e"] * a["cgs_TWh"] * G9
    c["curtailment"] = a["curt_penalty"] * a["curt_TWh"] * G9
    c["reinforcement"] = a["reinf_cost"] * a["dpeak_GW"] * a["reinforced"] * Y6
    c["over_cost"] = a["reinf_cost"] * a["over_GW"] * Y6
    nameplate = a["build_IT_GW"] / a["u_cap"]
    c["shell_premium"] = a["shell_premium"] * nameplate * Y6
    c["comm_cost"] = (a["comm_rate"] if "comm_rate" in a else 0.0) * a["it_energy_TWh"] * G9
    c["subsidy"] = a["subsidy_rate"] * nameplate * Y6
    a["conn_used_phys_cum_GW"] = physical_connection_use(a)
    c["carbon_grid"] = a["carbon_price"] * a["em_grid_Mt"] * Y6
    c["carbon_cgs"] = a["carbon_price"] * a["em_cgs_Mt"] * Y6
    for col in COST_COLUMNS:
        a[f"{col}_yen"] = c[col]
        a[f"{col}_disc"] = c[col] * a["disc"]
    a["social_disc"] = sum(a[f"{k}_disc"] for k in RESOURCE_COLUMNS)
    a["carbon_disc"] = a["carbon_grid_disc"] + a["carbon_cgs_disc"]
    a["eq1_disc"] = a["social_disc"] + a["carbon_disc"]
    a["operator_disc"] = (a["social_disc"] - (1.0 - a["theta"]) * a["reinforcement_disc"]
                          + a["pass_through"] * a["carbon_disc"] - a["subsidy_disc"])
    return a


def physical_connection_use(a: pd.DataFrame) -> pd.Series:
    """Cumulative connection use from the physical facility peak (GW): the
    running maximum of max_h L (starting from the connected peak entering the
    run) minus the 2025 peak, for the regions with a connection constraint
    (NaN elsewhere, 0 in the calibration years). conn_used_cum_GW is the same
    quantity from the connected-peak variable, which may lie above the physical
    peak where connecting early lowers the reinforcement cost."""
    out = pd.Series(np.where(a["peak_conn_GW"].notna(), 0.0, np.nan), index=a.index)
    for _, g in a[a["peak_conn_GW"].notna()].groupby("region", sort=False):
        g = g.sort_values("year")
        later = g[g["year"] > CALIBRATION_YEARS[-1]]
        if later.empty:
            continue
        base = float(later["peak_conn_GW"].iloc[0] - later["conn_used_cum_GW"].iloc[0])
        run = float(g["peak_conn_GW"].iloc[0] - g["dpeak_GW"].iloc[0])
        for idx, year, pk in zip(g.index, g["year"], g["peak_facility_GW"]):
            run = max(run, float(pk))
            if int(year) > CALIBRATION_YEARS[-1]:
                out.at[idx] = run - base
    return out


def _share(num: float, den: float) -> float:
    return num / den if den > 1e-12 else math.nan


def national(ry: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    cap = meta.get("co2_cap_Mt") or {}
    for t, g in ry.groupby("year", sort=True):
        new_all = sum(g[f"new_{k}_GW"].sum() for k in COOLING)
        new_liq = g["new_dlc_GW"].sum() + g["new_imm_GW"].sum()
        nonmetro = g[~g["region"].isin(METRO)]
        rows.append({
            "year": int(t),
            "co2_Mt": g["em_Mt"].sum(),
            "co2_cap_Mt": cap.get(str(int(t)), math.nan),
            "facility_TWh": g["facility_TWh"].sum(),
            "it_energy_TWh": g["it_energy_TWh"].sum(),
            "grid_received_TWh": g["grid_received_TWh"].sum(),
            "grid_net_TWh": g["grid_net_TWh"].sum(),
            "ppa_TWh": g["ppa_TWh"].sum(),
            "pv_TWh": g["pv_TWh"].sum(),
            "cgs_TWh": g["cgs_TWh"].sum(),
            "curt_TWh": g["curt_TWh"].sum(),
            "bess_ch_TWh": g["bess_ch_TWh"].sum(),
            "bess_dis_TWh": g["bess_dis_TWh"].sum(),
            "itr_GW": g["itr_GW"].sum(),
            "nonmetro_share": _share(nonmetro["itr_GW"].sum(), g["itr_GW"].sum()),
            "liquid_share_fleet": _share(g["it_dlc_GW"].sum() + g["it_imm_GW"].sum(),
                                         g["itr_GW"].sum()),
            "liquid_share_new": _share(new_liq, new_all),
            "new_cooling_GW": new_all,
            "new_liquid_GW": new_liq,
            "peak_facility_nonmetro_GW": nonmetro["peak_facility_GW"].sum(),
            "social_disc": g["social_disc"].sum(),
            "carbon_disc": g["carbon_disc"].sum(),
            "eq1_disc": g["eq1_disc"].sum(),
            "operator_disc": g["operator_disc"].sum(),
            "subsidy_disc": g["subsidy_disc"].sum(),
            "subsidy_yen": g["subsidy_yen"].sum(),
        })
    return pd.DataFrame(rows)


def evaluate(flows: Dict[str, pd.DataFrame], meta: Dict[str, Any]
             ) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    ry = region_year(flows, meta)
    nat = national(ry, meta)
    last = int(nat["year"].max())
    first = int(nat["year"].min())
    nl = nat[nat["year"] == last].iloc[0]
    later = nat[nat["year"] > CALIBRATION_YEARS[-1]]
    usage = ry[ry["conn_offered_cum_GW"].notna() & (ry["year"] > CALIBRATION_YEARS[-1])]
    usage = usage.assign(
        u=[_share(x, y) for x, y in zip(usage["conn_used_phys_cum_GW"],
                                        usage["conn_offered_cum_GW"])],
        uc=[_share(x, y) for x, y in zip(usage["conn_used_cum_GW"],
                                         usage["conn_offered_cum_GW"])])
    ul = usage[usage["year"] == last]["u"]
    ucl = usage[usage["year"] == last]["uc"]
    m: Dict[str, Any] = {
        "eq1_total_yen": float(ry["eq1_disc"].sum()),
        "social_cost_yen": float(ry["social_disc"].sum()),
        "operator_cost_yen": float(ry["operator_disc"].sum()),
        "fiscal_burden_yen": float(ry["subsidy_disc"].sum()),
        "fiscal_burden_undiscounted_yen": float(ry["subsidy_yen"].sum()),
        "carbon_cost_yen": float(ry["carbon_disc"].sum()),
        "carbon_payments_yen": float((ry["pass_through"] * ry["carbon_disc"]).sum()),
        "reinforcement_yen": float(ry["reinforcement_disc"].sum()),
        "over_cost_yen": float(ry["over_cost_disc"].sum()),
        "shell_premium_yen": float(ry["shell_premium_disc"].sum()),
        "comm_cost_yen": float(ry["comm_cost_disc"].sum()),
        "curtailment_yen": float(ry["curtailment_disc"].sum()),
        "objective_yen": float(meta.get("objective_yen", math.nan)),
        "first_year": first,
        "last_year": last,
        "co2_first_Mt": float(nat[nat["year"] == first]["co2_Mt"].iloc[0]),
        "co2_last_Mt": float(nl["co2_Mt"]),
        "co2_2040_Mt": float(nl["co2_Mt"]) if last == END_YEAR else math.nan,
        "co2_cum_Mt": float(nat["co2_Mt"].sum()),
        "facility_energy_cum_TWh": float(nat["facility_TWh"].sum()),
        "facility_energy_last_TWh": float(nl["facility_TWh"]),
        "it_energy_cum_TWh": float(nat["it_energy_TWh"].sum()),
        "grid_received_cum_TWh": float(nat["grid_received_TWh"].sum()),
        "grid_received_last_TWh": float(nl["grid_received_TWh"]),
        "curtail_TWh": float(nat["curt_TWh"].sum()),
        "bess_throughput_TWh": float(nat["bess_dis_TWh"].sum()),
        "bess_cap_last_GWh": float(ry[ry["year"] == last]["cap_bess_GWh"].sum()),
        "nonmetro_share_last": float(nl["nonmetro_share"]),
        "liquid_share_fleet_last": float(nl["liquid_share_fleet"]),
        "liquid_share_new_cum": _share(float(later["new_liquid_GW"].sum()),
                                       float(later["new_cooling_GW"].sum())),
        "peak_facility_nonmetro_last_GW": float(nl["peak_facility_nonmetro_GW"]),
        "conn_usage_last_max": float(ul.max()) if len(ul) and ul.notna().any() else math.nan,
        "conn_connected_last_max": (float(ucl.max()) if len(ucl) and ucl.notna().any()
                                    else math.nan),
        "uses_dev_params": bool(meta.get("uses_dev_params", False)),
    }
    return m, {"region_year": ry, "national": nat}


def analysis_frames(flows: Dict[str, pd.DataFrame], meta: Dict[str, Any],
                    basis: str = "eq1") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(table, components) in the schema of ``dc_decom_analysis`` (lmdi,
    decomposition). ``basis`` 'eq1': full carbon cost, no subsidy, full
    reinforcement (run_total = Eq. 1 total); 'operator': pass-through carbon,
    subsidy, theta share of the reinforcement (run_total = operator cost).
    Generators are reported with CGS (backup); shell premium, reinforcement,
    overrun and the communication cost go to 'adder' (siting channel). E_grid_TWh is the emitting
    grid energy (net of PPA and certificates)."""
    if basis not in ("eq1", "operator"):
        raise ValueError("basis must be 'eq1' or 'operator'")
    ry = region_year(flows, meta)
    table = pd.DataFrame({
        "year": ry["year"], "region": ry["region"],
        "it_load_GW": ry["itr_GW"],
        "it_energy_TWh": ry["it_energy_TWh"],
        "load_TWh": ry["facility_TWh"],
        "facility_load_TWh": ry["facility_TWh"],
        "E_grid_TWh": ry["grid_net_TWh"],
        "grid_received_TWh": ry["grid_received_TWh"],
        "E_ppa_TWh": ry["ppa_TWh"],
        "E_pv_TWh": ry["pv_TWh"],
        "E_bess_disch_TWh": ry["bess_dis_TWh"],
        "E_cgs_TWh": ry["cgs_TWh"],
        "curt_TWh": ry["curt_TWh"],
        "ef_grid": ry["ef_grid"],
        "emissions_grid_Mt": ry["em_grid_Mt"],
        "emissions_cgs_Mt": ry["em_cgs_Mt"],
        "emissions_Mt": ry["em_Mt"],
    })
    op = basis == "operator"
    carbon_f = ry["pass_through"] if op else 1.0
    reinf_f = ry["theta"] if op else 1.0
    procured = ry["grid_received_TWh"] + ry["pv_TWh"] + ry["cgs_TWh"]
    if meta["ppa_matching"] == "hourly":
        procured = procured + ry["ppa_delivered_TWh"]
    zero = np.zeros(len(ry))
    comp = pd.DataFrame({
        "year": ry["year"], "region": ry["region"], "procured_TWh": procured,
        "cooling_capex": ry["cooling_capex_disc"], "cooling_opex": ry["cooling_opex_disc"],
        "cooling_premium": zero,
        "cgs_capex": ry["cgs_capex_disc"] + ry["gen_capex_disc"],
        "cgs_opex": ry["cgs_opex_disc"] + ry["gen_opex_disc"],
        "cgs_premium": zero,
        "bess_capex": ry["bess_capex_disc"], "bess_opex": ry["bess_opex_disc"],
        "bess_premium": zero, "pv_capex": zero, "pv_opex": zero,
        "energy_grid": ry["energy_grid_disc"],
        "energy_ppa": ry["energy_ppa_disc"] + ry["energy_nfc_disc"],
        "energy_pv": ry["energy_pv_disc"],
        "energy_cgs_fuel": ry["energy_cgs_fuel_disc"],
        "carbon_grid": ry["carbon_grid_disc"] * carbon_f,
        "carbon_cgs": ry["carbon_cgs_disc"] * carbon_f,
        "curtailment": ry["curtailment_disc"],
        "subsidy": ry["subsidy_disc"] if op else zero,
        "adder": (ry["shell_premium_disc"] + ry["reinforcement_disc"] * reinf_f
                  + ry["over_cost_disc"] + ry["comm_cost_disc"]),
    })
    return table, comp
