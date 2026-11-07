"""(a) Base scenarios of v2 (rep96, frozen v2 parameters)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from dc_decom_analysis import decomposition

from .. import accounting, runner
from ..config import METRO
from . import common as C

FILES = ("base_metrics.csv", "base_regional.csv", "base_cooling.csv", "base_channels.csv",
         "base_summary.json", "run_log_base.csv")


def specs(cfg: Dict[str, Any]) -> List[runner.RunSpec]:
    return [runner.RunSpec(scenario=s, existing_dir=C.V2, options=C.BASE_OPTIONS,
                           store_slices=True, label=f"base:{s}")
            for s in cfg["base"]["scenarios"]]


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    cfg = C.load_config()
    sp = specs(cfg)
    res = dict(zip([s.scenario for s in sp], C.solve(sp, "base", use_cache)))
    tot = {s: C.total(r) for s, r in res.items()}

    rows = []
    for s, r in res.items():
        row: Dict[str, Any] = {"scenario": s}
        row.update(C.scenario_row(r))
        if r.status == "Optimal":
            m = r.metrics
            ry = r.tables["region_year"]
            last = ry[ry["year"] == C.YEARS[-1]]
            row.update({
                "facility_energy_cum_TWh": m["facility_energy_cum_TWh"],
                "facility_energy_2040_TWh": m["facility_energy_last_TWh"],
                "grid_received_cum_TWh": m["grid_received_cum_TWh"],
                "grid_received_2040_TWh": m["grid_received_last_TWh"],
                "peak_facility_sum_2040_GW": float(last["peak_facility_GW"].sum()),
                "peak_grid_sum_2040_GW": float(last["peak_grid_GW"].sum()),
                "peak_facility_nonmetro_2040_GW": m["peak_facility_nonmetro_last_GW"],
                "bess_cap_2040_GWh": m["bess_cap_last_GWh"],
                "ppa_cum_TWh": float(ry["ppa_TWh"].sum()),
                "nfc_cum_TWh": float(ry["nfc_TWh"].sum()),
                "nfc_cost_TJPY": float(ry["energy_nfc_disc"].sum()) / C.TJ,
                "pv_cum_TWh": float(ry["pv_TWh"].sum()),
                "liquid_share_new_cum": m["liquid_share_new_cum"],
                "conn_usage_2040_max": m["conn_usage_last_max"],
                "conn_connected_2040_max": m["conn_connected_last_max"],
                "premium_vs_REF": tot[s] / tot["REF"] - 1.0,
                "advantage_vs_C-CN": C.adv(tot["C-CN"], tot[s]),
            })
        rows.append(row)
    C.write_csv(pd.DataFrame(rows), "base_metrics.csv", out)

    reg, cool = [], []
    for s, r in res.items():
        if r.status != "Optimal":
            continue
        ry = r.tables["region_year"]
        liq = ry["it_dlc_GW"] + ry["it_imm_GW"]
        reg.append(pd.DataFrame({
            "scenario": s, "year": ry["year"], "region": ry["region"],
            "metro": ry["region"].isin(METRO), "itr_GW": ry["itr_GW"],
            "it_share": ry["it_share"], "facility_TWh": ry["facility_TWh"],
            "grid_received_TWh": ry["grid_received_TWh"], "ppa_TWh": ry["ppa_TWh"],
            "pv_TWh": ry["pv_TWh"], "nfc_TWh": ry["nfc_TWh"], "cgs_TWh": ry["cgs_TWh"],
            "curt_TWh": ry["curt_TWh"],
            "em_Mt": ry["em_Mt"],
            "liquid_share_it": (liq / ry["itr_GW"]).where(ry["itr_GW"] > 0, 0.0),
            "peak_facility_GW": ry["peak_facility_GW"],
            "conn_used_cum_GW": ry["conn_used_cum_GW"],
            "conn_used_phys_cum_GW": ry["conn_used_phys_cum_GW"],
            "conn_offered_cum_GW": ry["conn_offered_cum_GW"]}))
        nat = r.tables["national"]
        cool.append(pd.DataFrame({
            "scenario": s, "year": nat["year"], "liquid_share_fleet": nat["liquid_share_fleet"],
            "liquid_share_new": nat["liquid_share_new"], "new_cooling_GW": nat["new_cooling_GW"],
            "new_liquid_GW": nat["new_liquid_GW"], "nonmetro_share": nat["nonmetro_share"],
            "co2_Mt": nat["co2_Mt"], "co2_cap_Mt": nat["co2_cap_Mt"]}))
    C.write_csv(pd.concat(reg, ignore_index=True), "base_regional.csv", out)
    C.write_csv(pd.concat(cool, ignore_index=True), "base_cooling.csv", out)

    ch_rows = []
    summary: Dict[str, Any] = {"totals_TJPY": {s: tot[s] / C.TJ for s in tot},
                               "status": {s: r.status for s, r in res.items()},
                               "curtail_TWh": {s: res[s].metrics.get("curtail_TWh") for s in res}}
    comps = {s: accounting.analysis_frames(r.flows, r.meta, basis="eq1")[1]
             for s, r in res.items() if r.status == "Optimal"}
    for rcn in ("R-CN", "R-CN-pub"):
        if "C-CN" not in comps or rcn not in comps:
            continue
        split = decomposition.principled(comps["C-CN"], comps[rcn])
        sh = decomposition.shares(split)
        for line in decomposition.CHANNELS + ("total", "direct_total", "quantity_effect",
                                              "price_effect", "d_cooling_assets", "d_re_assets",
                                              "d_adder", "d_energy"):
            ch_rows.append({"comparison": f"C-CN_minus_{rcn}", "line": line,
                            "value_TJPY": split[line] / C.TJ,
                            "share": sh.get(line, float("nan"))})
        summary[f"channels_{rcn}"] = {k: split[k] / C.TJ for k in decomposition.CHANNELS
                                      + ("total", "direct_total")}
        summary[f"channel_shares_{rcn}"] = sh
        summary[f"advantage_{rcn}"] = C.adv(tot["C-CN"], tot[rcn])
        summary[f"channel_check_{rcn}"] = abs(split["total"] - (tot["C-CN"] - tot[rcn])) / C.TJ
    C.write_csv(pd.DataFrame(ch_rows), "base_channels.csv", out)
    C.write_json(summary, "base_summary.json", out)
    C.write_run_log("base", out)
    return C.PartResult(files=list(FILES), summary=summary)
