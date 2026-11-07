"""A/B/C rolling comparison of ch07 7.10 with the diagnostic B variants.

Rules and thresholds: config/abc_rules.yaml only (fixed before the A/B/C
runs; the run stops if the file differs from the fixed content,
``RULES_SHA256``). Every design and A0
is solved with ``dc_decom_v2.rolling`` from one shared base: the same forecast
(base paths), the same 18 actual paths, the same review years and model
options (rep96, experiments_v2 change C1: certificates). Outputs go to
outputs/abc/.

Metrics (smaller is better unless noted): social resource cost, Eq. 1 total,
operator cost, fiscal burden, CO2 2040 and cumulative, electricity (facility
energy) and grid received energy (cumulative), non-metro connection usage in
2040 (max over the regions), regret (ch07 eq:abc-regret over A, B, C) and the
share of paths whose emissions stay on or below the cap path (eq:co2cap-path).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd
import yaml  # type: ignore[import-untyped]

from .. import rolling
from ..config import METRO
from ..scenarios import default_paths
from . import common as C

OUT = C.REPO / "outputs" / "abc"
RULES = C.REPO / "config" / "abc_rules.yaml"
# sha256 of config/abc_rules.yaml as fixed before the A/B/C runs.
RULES_SHA256 = "c718b5c7333af2dc9b8f7a23c2979c5781a4d61d5e98838edd6bcc92a956248d"
FILES = ("abc_metrics.csv", "abc_summary.csv", "abc_regret.csv", "abc_switches.csv",
         "abc_diagnostics.csv", "abc_regional.csv", "abc_rule_check.csv", "abc_summary.json",
         "run_log_abc.csv")
MAIN = ("A", "B", "C")
REFERENCE = "A0"
REGRET_METRICS = ("social_cost_TJPY", "eq1_total_TJPY", "operator_cost_TJPY",
                  "fiscal_burden_TJPY", "co2_cum_Mt", "co2_2040_Mt", "electricity_cum_TWh",
                  "grid_received_cum_TWh", "conn_usage_2040_max")
SUMMARY_METRICS = REGRET_METRICS + ("conn_connected_2040_max", "reinforcement_TJPY",
                                    "over_GW_total",
                                    "nonmetro_share_2040", "liquid_share_new_cum",
                                    "std_violations")
REL_TOL = 1e-6           # 'better' / 'worse' beyond this relative difference (solver noise)
# Sensitivity of the realized regional IT rule: the main runs use
# rolling.realized_itr rule 'target'; these designs are rerun with the
# 'increment' rule and reported in abc_rule_check.csv.
RULE_CHECK = ("increment", ("A0", "A", "B"))
RULE_CHECK_COLUMNS = ("social_cost_TJPY", "operator_cost_TJPY", "reinforcement_TJPY",
                      "nonmetro_share_2040", "conn_usage_2040_max", "siting_floor_gap_min",
                      "itr_fall_max_GW")


def load_rules() -> Dict[str, Any]:
    return yaml.safe_load(RULES.read_text(encoding="utf-8"))


def rules_check() -> Dict[str, Any]:
    """The rules file must have the fixed content (sha256 ``RULES_SHA256``)."""
    sha = hashlib.sha256(RULES.read_bytes()).hexdigest()
    if sha != RULES_SHA256:
        raise RuntimeError(f"config/abc_rules.yaml differs from the fixed content "
                           f"(sha256 {sha}, expected {RULES_SHA256})")
    return {"file": "config/abc_rules.yaml", "sha256": sha, "unchanged_since_freeze": True}


def base_config() -> rolling.RollingBase:
    ps = C.params()
    return rolling.RollingBase(existing_dir=C.V2, forecast=default_paths(ps),
                               options=C.BASE_OPTIONS)


def designs(rules: Mapping[str, Any]) -> Dict[str, rolling.Design]:
    return rolling.designs_from_rules(rules, C.params())


def metrics_row(r: rolling.RollingRun) -> Dict[str, Any]:
    row: Dict[str, Any] = {"design": r.design.name, "path": r.path.name,
                           "carbon": rolling.VARIANT_LABEL[r.path.carbon],
                           "demand": rolling.VARIANT_LABEL[r.path.demand],
                           "liquid": rolling.LIQUID_LABEL[r.path.liquid],
                           "status": r.status, "note": r.note}
    if not r.ok:
        return row
    m = r.metrics
    ry = r.tables["region_year"]
    last = ry[ry["year"] == C.YEARS[-1]]
    row.update({
        "social_cost_TJPY": m["social_cost_yen"] / C.TJ,
        "eq1_total_TJPY": m["eq1_total_yen"] / C.TJ,
        "operator_cost_TJPY": m["operator_cost_yen"] / C.TJ,
        "fiscal_burden_TJPY": m["fiscal_burden_yen"] / C.TJ,
        "carbon_cost_TJPY": m["carbon_cost_yen"] / C.TJ,
        "reinforcement_TJPY": m["reinforcement_yen"] / C.TJ,
        "over_cost_TJPY": m["over_cost_yen"] / C.TJ,
        "over_GW_total": m["over_GW_total"],
        "co2_2040_Mt": m["co2_2040_Mt"],
        "co2_cum_Mt": m["co2_cum_Mt"],
        "meets_cap_path": m["meets_cap_path"],
        "cap_path_max_excess_Mt": m["cap_path_max_excess_Mt"],
        "electricity_cum_TWh": m["facility_energy_cum_TWh"],
        "electricity_2040_TWh": m["facility_energy_last_TWh"],
        "it_energy_cum_TWh": m["it_energy_cum_TWh"],
        "grid_received_cum_TWh": m["grid_received_cum_TWh"],
        "grid_received_2040_TWh": m["grid_received_last_TWh"],
        "conn_usage_2040_max": m["conn_usage_last_max"],
        "conn_connected_2040_max": m["conn_connected_last_max"],
        "peak_facility_nonmetro_2040_GW": m["peak_facility_nonmetro_last_GW"],
        "nonmetro_share_2040": m["nonmetro_share_last"],
        "liquid_share_fleet_2040": m["liquid_share_fleet_last"],
        "liquid_share_new_cum": m["liquid_share_new_cum"],
        "curtail_TWh": m["curtail_TWh"],
        "ppa_cum_TWh": float(ry["ppa_TWh"].sum()),
        "nfc_cum_TWh": float(ry["nfc_TWh"].sum()),
        "pv_cum_TWh": float(ry["pv_TWh"].sum()),
        "bess_cap_2040_GWh": float(last["cap_bess_GWh"].sum()),
        "std_violations": m["std_violations"],
        "std_fleet_pue_excess_max": m["std_fleet_pue_excess_max"],
        "std_new_pue_excess_max": m["std_new_pue_excess_max"],
        "std_liquid_floor_gap_max": m["std_liquid_floor_gap_max"],
        "siting_floor_gap_min": m["siting_floor_gap_min"],
        "itr_fall_max_GW": m["itr_fall_max_GW"],
        "n_switch_changes": m["n_switch_changes"],
        "n_policy_changes": m["n_policy_changes"],
    })
    return row


def _better(x: float, y: float) -> int:
    """-1 if x is smaller than y beyond the tolerance, +1 if larger, 0 otherwise."""
    tol = REL_TOL * max(abs(x), abs(y), 1e-12) + 1e-12
    return -1 if x < y - tol else (1 if x > y + tol else 0)


def regret_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main = df[df["design"].isin(MAIN) & (df["status"] == "Optimal")]
    for path, g in main.groupby("path", sort=False):
        for metric in REGRET_METRICS:
            vals = dict(zip(g["design"], g[metric]))
            if len(vals) < len(MAIN) or any(math.isnan(float(v)) for v in vals.values()):
                continue
            best = min(float(v) for v in vals.values())
            best_d = [d for d in MAIN if float(vals[d]) <= best + 1e-12]
            for d in MAIN:
                rows.append({"design": d, "path": path, "metric": metric,
                             "value": float(vals[d]), "best": best,
                             "best_designs": "+".join(best_d),
                             "regret": float(vals[d]) - best})
    return pd.DataFrame(rows)


def summary_frame(df: pd.DataFrame, reg: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    rows = []
    ok = df[df["status"] == "Optimal"]
    for d in order:
        g = ok[ok["design"] == d]
        for metric in SUMMARY_METRICS + ("meets_cap_path",):
            v = g[metric].astype(float)
            row = {"design": d, "metric": metric, "n_paths": int(len(v)),
                   "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                   "range": float(v.max() - v.min())}
            rg = reg[(reg["design"] == d) & (reg["metric"] == metric)] if len(reg) else reg
            row["regret_max"] = float(rg["regret"].max()) if len(rg) else math.nan
            row["regret_mean"] = float(rg["regret"].mean()) if len(rg) else math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, x: str, y: str, metrics: Sequence[str]) -> Dict[str, Any]:
    """Path-by-path comparison of design x against y (smaller is better)."""
    a = df[(df["design"] == x) & (df["status"] == "Optimal")].set_index("path")
    b = df[(df["design"] == y) & (df["status"] == "Optimal")].set_index("path")
    common = [p for p in a.index if p in b.index]
    out: Dict[str, Any] = {"n_paths": len(common)}
    for m in metrics:
        sgn = [_better(float(a.at[p, m]), float(b.at[p, m])) for p in common]
        diff = [float(a.at[p, m]) - float(b.at[p, m]) for p in common]
        out[m] = {"n_better": sgn.count(-1), "n_equal": sgn.count(0), "n_worse": sgn.count(1),
                  "mean_diff": sum(diff) / len(diff) if diff else math.nan,
                  "min_diff": min(diff) if diff else math.nan,
                  "max_diff": max(diff) if diff else math.nan,
                  "worse_paths": [p for p, s in zip(common, sgn) if s == 1]}
    both = [p for p in common
            if _better(float(a.at[p, "social_cost_TJPY"]), float(b.at[p, "social_cost_TJPY"])) < 0
            and _better(float(a.at[p, "co2_cum_Mt"]), float(b.at[p, "co2_cum_Mt"])) < 0]
    out["n_better_social_and_co2"] = len(both)
    return out


def diagnostics_frame(df: pd.DataFrame, variants: Sequence[str]) -> pd.DataFrame:
    a = df[df["design"] == "A"].set_index("path")
    cols = ("social_cost_TJPY", "co2_cum_Mt", "operator_cost_TJPY", "fiscal_burden_TJPY",
            "reinforcement_TJPY", "nonmetro_share_2040", "conn_usage_2040_max")
    rows = []
    for v in ("B",) + tuple(variants):
        g = df[df["design"] == v]
        for row in g.itertuples(index=False):
            d = row._asdict()
            out = {"variant": v, "path": d["path"], "status": d["status"]}
            if d["status"] == "Optimal":
                for c in cols:
                    out[c] = d[c]
                for c in ("social_cost_TJPY", "co2_cum_Mt", "operator_cost_TJPY"):
                    ref = float(a.at[d["path"], c])
                    out[f"d_{c}_vs_A"] = float(d[c]) - ref
                    out[f"better_{c}_than_A"] = _better(float(d[c]), ref) < 0
            rows.append(out)
    return pd.DataFrame(rows)


def regional_frame(runs: Sequence[rolling.RollingRun]) -> pd.DataFrame:
    rows = []
    for r in runs:
        if not r.ok:
            continue
        ry = r.tables["region_year"]
        last = ry[ry["year"] == C.YEARS[-1]]
        liq = last["it_dlc_GW"] + last["it_imm_GW"]
        rows.append(pd.DataFrame({
            "design": r.design.name, "path": r.path.name, "region": last["region"],
            "metro": last["region"].isin(METRO), "it_share_2040": last["it_share"],
            "itr_GW_2040": last["itr_GW"], "facility_TWh_2040": last["facility_TWh"],
            "liquid_share_it_2040": (liq / last["itr_GW"]).where(last["itr_GW"] > 0, 0.0)}))
    return pd.concat(rows, ignore_index=True)


def switches_frame(runs: Sequence[rolling.RollingRun]) -> pd.DataFrame:
    rows = []
    for r in runs:
        for s in r.switches:
            rows.append({"design": r.design.name, "path": r.path.name, **s})
    cols = ["design", "path", "review_year", "rule", "indicator", "value", "threshold",
            "triggered", "action", "setting", "before", "after", "changed", "policy_changed"]
    return pd.DataFrame(rows, columns=cols)


def settings_dict(d: rolling.Design) -> Dict[str, Any]:
    s = d.settings
    return {"adaptive": d.adaptive, "pue_rules": s.pue_rules, "phi": s.phi, "t_g": s.t_g,
            "psi": s.psi, "subsidy": s.subsidy, "floors": s.floors,
            "ppa_multiplier": s.ppa_multiplier, "theta": s.theta, "g": s.g, "pstar": s.pstar}


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    check = rules_check()
    rules = load_rules()
    ds = designs(rules)
    variants = [n for n in ds if n not in MAIN + (REFERENCE,)]
    order = [REFERENCE] + list(MAIN) + variants
    base = base_config()
    ps = C.params()
    paths = rolling.actual_paths(
        ps, carbon=rules["realization_paths"]["carbon_price"],
        demand=rules["realization_paths"]["demand_TWh"],
        liquid=rules["realization_paths"]["liquid_deploy_cap"])

    def solve(specs: List[Any], tag: str) -> List[Any]:
        return C.solve(specs, "abc", use_cache)

    calib, runs = rolling.run_designs(base, [ds[n] for n in order], paths, rules, solve=solve,
                                      keep_flows=False)
    df = pd.DataFrame([metrics_row(r) for r in runs])
    C.write_csv(df, "abc_metrics.csv", out)
    rule, rule_designs = RULE_CHECK
    _, runs_rc = rolling.run_designs(replace(base, itr_rule=rule), [ds[n] for n in rule_designs],
                                     paths, rules, solve=solve, keep_flows=False)
    dfr = pd.DataFrame([metrics_row(r) for r in runs_rc])
    rc = pd.concat([df[df["design"].isin(rule_designs)].assign(itr_rule=base.itr_rule),
                    dfr.assign(itr_rule=rule)], ignore_index=True)
    C.write_csv(rc[["itr_rule", "design", "path", "status"] + list(RULE_CHECK_COLUMNS)],
                "abc_rule_check.csv", out)
    reg = regret_frame(df)
    C.write_csv(reg, "abc_regret.csv", out)
    summ = summary_frame(df, reg, order)
    C.write_csv(summ, "abc_summary.csv", out)
    sw = switches_frame([r for r in runs if r.design.adaptive])
    C.write_csv(sw, "abc_switches.csv", out)
    diag = diagnostics_frame(df, variants)
    C.write_csv(diag, "abc_diagnostics.csv", out)
    C.write_csv(regional_frame(runs), "abc_regional.csv", out)

    cmp_metrics = REGRET_METRICS
    trig = sw[sw["rule"] != "report"] if len(sw) else sw
    summary: Dict[str, Any] = {
        "rules": check,
        "parameters": rules["parameters"],
        "base_options": rules["base_options"],
        "scenario": base.scenario,
        "review_years": list(base.review_years),
        "n_paths": len(paths),
        "paths": [p.name for p in paths],
        "designs": {n: settings_dict(ds[n]) for n in order},
        "n_runs": len(runs),
        "failed": [{"design": r.design.name, "path": r.path.name, "note": r.note}
                   for r in runs if not r.ok],
        "calibration": {"social_cost_TJPY": calib.metrics["social_cost_yen"] / C.TJ,
                        "co2_Mt": calib.metrics["co2_cum_Mt"],
                        "E_2024_Mt": calib.meta["E_2024_Mt"]},
        "tolerance_relative": REL_TOL,
        "B_vs_A": compare(df, "B", "A", cmp_metrics),
        "C_vs_B": compare(df, "C", "B", cmp_metrics),
        "C_vs_A": compare(df, "C", "A", cmp_metrics),
        "A_vs_A0": compare(df, "A", "A0", cmp_metrics),
        "B_vs_A0": compare(df, "B", "A0", cmp_metrics),
        "variants_vs_A": {v: compare(df, v, "A", ("social_cost_TJPY", "co2_cum_Mt",
                                                  "operator_cost_TJPY"))
                          for v in variants},
        "switches": {
            "n_rows": int(len(trig)),
            "n_triggered": int(trig["triggered"].sum()) if len(trig) else 0,
            "n_setting_changes": int(trig["changed"].sum()) if len(trig) else 0,
            "n_policy_changes": int(trig["policy_changed"].sum()) if len(trig) else 0,
            "by_rule": {str(k): {"triggered": int(g["triggered"].sum()),
                                 "setting_changed": int(g["changed"].sum()),
                                 "policy_changed": int(g["policy_changed"].sum())}
                        for k, g in trig.groupby("rule")} if len(trig) else {},
        },
        "C_equals_B_all_paths": bool(_identical(df, "C", "B")),
        "share_meets_cap_path": {d: float(df[df["design"] == d]["meets_cap_path"].astype(float)
                                          .mean()) for d in order},
        "siting": {"itr_rule": base.itr_rule,
                   "floor_gap_min": float(df["siting_floor_gap_min"].min()),
                   "itr_fall_max_GW": float(df["itr_fall_max_GW"].max())},
        "realization_rule_check": {
            "itr_rule": rule, "designs": list(rule_designs),
            "failed": [{"design": r.design.name, "path": r.path.name, "note": r.note}
                       for r in runs_rc if not r.ok],
            "B_vs_A": compare(dfr, "B", "A", ("social_cost_TJPY", "operator_cost_TJPY")),
            "B_vs_A0": compare(dfr, "B", "A0", ("social_cost_TJPY", "operator_cost_TJPY")),
            "mean": {d: {m: float(dfr[dfr["design"] == d][m].mean())
                         for m in RULE_CHECK_COLUMNS} for d in rule_designs},
            "floor_gap_min": float(dfr["siting_floor_gap_min"].min()),
            "itr_fall_max_GW": float(dfr["itr_fall_max_GW"].max()),
        },
    }
    base_row = df[(df["path"] == "c-base_d-base_l-base")].set_index("design")
    summary["base_path"] = {d: {m: float(base_row.at[d, m]) for m in
                                ("social_cost_TJPY", "eq1_total_TJPY", "operator_cost_TJPY",
                                 "fiscal_burden_TJPY", "co2_cum_Mt", "nonmetro_share_2040",
                                 "electricity_cum_TWh", "grid_received_cum_TWh",
                                 "conn_usage_2040_max", "reinforcement_TJPY")}
                            for d in order if d in base_row.index}
    C.write_json(summary, "abc_summary.json", out)
    C.write_run_log("abc", out)
    return C.PartResult(files=list(FILES), summary=summary)


def _identical(df: pd.DataFrame, x: str, y: str) -> bool:
    a = df[df["design"] == x].set_index("path")
    b = df[df["design"] == y].set_index("path")
    cols = [c for c in REGRET_METRICS if c in a.columns]
    return all(_better(float(a.at[p, c]), float(b.at[p, c])) == 0
               for p in a.index for c in cols)
