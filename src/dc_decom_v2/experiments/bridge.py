"""(b) Bridge from the v2 base along two assumption sets, and the conditions
analysis (config/experiments_v2.yaml).

Chain ``to_paper_only`` (assumption set S1, steps P1-P11): back to the
assumptions stated in the equations and tables of Energies, ending at the
spec paper_only run. Chain ``to_unstated`` (assumption set S2, steps T1-T9):
assumptions whose values are not stated in Energies, added to the v2 base."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..config import CALIBRATION_YEARS, METRO
from . import common as C
from .common import Setting, as_tuple

BRIDGE_FILES = ("bridge_steps.csv", "bridge_assumptions.csv", "bridge_summary.json",
                "run_log_bridge.csv")
COND_FILES = ("conditions_scan.csv", "conditions_summary.json", "run_log_conditions.csv")
ROLES = ("REF", "C-CN", "R-CN")
REGS = ("Hokkaido", "Tohoku", "Kanto", "Kansai", "Kyushu")
EF24_S2 = (0.49, 0.52, 0.44, 0.36, 0.37)
EF40_CN_S2 = (0.05, 0.07, 0.11, 0.07, 0.06)
EF40_REF_S2 = (0.09, 0.12, 0.16, 0.11, 0.10)


def _ef_path(e40: Tuple[float, ...]) -> Tuple[Tuple[str, float], ...]:
    out = {}
    for r, a, b in zip(REGS, EF24_S2, e40):
        for t in C.YEARS:
            out[f"ef_grid_path[{r}]@{t}"] = a + (b - a) * (t - 2024) / 16.0
    return as_tuple(out)


def steps_unstated() -> List[Tuple[str, Setting]]:
    s = Setting()
    out = [("B0", s)]
    s = s.with_(roles=(("REF", "REF"), ("C-CN", "C-CN-gs"), ("R-CN", "R-CN-s2")))
    out.append(("T1", s))
    s = s.with_(siting_s2=True)
    out.append(("T2", s))
    s = s.with_(liquid_fleet_max_ccn=as_tuple({r: 0.15 for r in REGS if r not in METRO}))
    out.append(("T3", s))
    s = s.with_(options=s.options.with_(co2_floor_share=0.95))
    out.append(("T4", s))
    cn, ref = _ef_path(EF40_CN_S2), _ef_path(EF40_REF_S2)
    s = s.with_(additions_by_role=(("C-CN", cn), ("R-CN", cn), ("REF", ref)))
    out.append(("T5", s))
    s = s.with_(carbon=as_tuple({t: 1500.0 + (t - 2024) / 16.0 * 10500.0 for t in C.YEARS}))
    out.append(("T6", s))
    s = s.with_overrides({"nfc_yen_per_kWh": 5.0})
    out.append(("T7", s))
    s = s.with_(options=s.options.with_(grid_constraints=False))
    out.append(("T8", s))
    s = s.with_(demand=as_tuple({t: 12.0 * 1.132 ** (t - 2024) for t in C.YEARS}))
    out.append(("T9", s))
    return out


def steps_paper_only() -> List[Tuple[str, Setting]]:
    sp = C.params(C.SPEC)
    s = Setting()
    out = [("B0", s)]
    s = s.with_(roles=(("REF", "REF"), ("C-CN", "C-CN"), ("R-CN", "R-CN-pub")))
    out.append(("P1", s))
    s = s.with_(options=s.options.with_(grid_constraints=False))
    out.append(("P2", s))
    s = s.with_(demand=as_tuple(sp.path("demand_TWh")))
    out.append(("P3", s))
    s = s.with_(carbon=as_tuple(sp.path("carbon_price")))
    out.append(("P4", s))
    ov: Dict[str, float] = {}
    for r in REGS:
        ov[f"tariff_yen_per_kWh[{r}]"] = sp.region("tariff_yen_per_kWh", r)
        ov[f"grid_ef_base[{r}]"] = sp.region("grid_ef_base", r)
    ov.update(C.path_variant(sp, "ef_national", "value"))
    s = s.with_overrides(ov)
    out.append(("P5", s))
    # P6: the spec loader gives the same paths and Table 9 values as P3-P5
    s = s.with_(existing=C.SPEC, structural=C.V2, overrides=(), demand=None, carbon=None)
    out.append(("P6", s))
    s = s.with_(options=s.options.with_(time_mode="annual"))
    out.append(("P7", s))
    ppa = {"ppa_tier1_share": 1000.0, "ppa_tier2_share": 0.0}
    ppa.update({f"ppa_volume_factor@{t}": 1.0 for t in C.YEARS})
    s = s.with_overrides(ppa)
    out.append(("P8", s))
    s = s.with_(options=s.options.with_(backup_techs=("cgs",)))
    out.append(("P9", s))
    s = s.with_(options=s.options.with_(it_energy="legacy_util"))
    out.append(("P10", s))
    s = s.with_(options=s.options.with_(certificates=False))
    out.append(("P11", s))
    return out


def solve_settings(cases: List[Tuple[str, Setting]], roles: Tuple[str, ...], part: str,
                   use_cache: bool) -> Dict[Tuple[str, str], Any]:
    specs, keys = [], []
    for cid, s in cases:
        for role in roles:
            specs.append(C.spec_for(s, role, f"{part}:{cid}:{role}"))
            keys.append((cid, role))
    return dict(zip(keys, C.solve(specs, part, use_cache)))


def metrics(cid: str, s: Setting, res: Dict[Tuple[str, str], Any],
            roles: Tuple[str, ...]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    tot: Dict[str, float] = {}
    pen: Dict[str, float] = {}
    for role in roles:
        r = res[(cid, role)]
        extra = C.siting_s2_cost(r, role) if (s.siting_s2 and r.status == "Optimal") else 0.0
        tot[role] = C.total(r, extra)
        row[f"scenario_{role}"] = s.role(role)
        row[f"status_{role}"] = r.status
        row[f"total_{role}_TJPY"] = tot[role] / C.TJ
        row[f"siting_s2_{role}_TJPY"] = extra / C.TJ
        ok = r.status == "Optimal"
        row[f"curtail_{role}_TWh"] = r.metrics["curtail_TWh"] if ok else float("nan")
        pen[role] = r.metrics["curtailment_yen"] if ok else float("nan")
        row[f"curtail_cost_{role}_TJPY"] = pen[role] / C.TJ
        row[f"co2_cum_{role}_Mt"] = r.metrics["co2_cum_Mt"] if ok else float("nan")
        row[f"nonmetro_2040_{role}"] = r.metrics["nonmetro_share_last"] if ok else float("nan")
    row["advantage"] = C.adv(tot["C-CN"], tot["R-CN"])
    row["advantage_ex_penalty"] = C.adv(tot["C-CN"] - pen["C-CN"], tot["R-CN"] - pen["R-CN"])
    if "REF" in tot:
        row["ccn_premium_vs_REF"] = tot["C-CN"] / tot["REF"] - 1.0
    _, adj = C.setting_paths(s)
    row["demand_running_max"] = adj
    return row


def run_bridge(out: Path, use_cache: bool = True) -> C.PartResult:
    cfg = C.load_config()["bridge"]
    chains = {"to_unstated": steps_unstated(), "to_paper_only": steps_paper_only()}
    rows: List[Dict[str, Any]] = []
    meta = {d["id"]: d for k in ("to_unstated", "to_paper_only") for d in cfg[k]}
    for chain, cases in chains.items():
        res = solve_settings(cases, ROLES, "bridge", use_cache)
        prev = None
        for cid, s in cases:
            row = {"chain": chain, "step": cid,
                   "name": "v2 base" if cid == "B0" else meta[cid]["name"]}
            row.update(metrics(cid, s, res, ROLES))
            row["delta_advantage"] = (row["advantage"] - prev) if prev is not None else 0.0
            prev = row["advantage"]
            rows.append(row)
    po = pd.read_csv(C.REPO / "outputs" / "stage2" / "paper_only.csv").set_index("scenario")
    refs = cfg["reference_points"]
    po_adv_ex = (po.loc["C-CN", "total_TJPY"] - po.loc["R-CN", "resource_cost_TJPY"]) \
        / po.loc["C-CN", "total_TJPY"]
    for chain, key, adv_x, note in (
            ("to_paper_only", "paper_only", refs["paper_only"]["advantage"],
             "spec paper_only (subsidy included in R-CN)"),
            ("to_paper_only", "paper_only_ex_subsidy", po_adv_ex,
             "spec paper_only with the R-CN subsidy added back")):
        tt = refs["paper_only"]["totals_TJPY"]
        rows.append({"chain": chain, "step": "reference", "name": note,
                     "total_REF_TJPY": tt["REF"], "total_C-CN_TJPY": tt["C-CN"],
                     "total_R-CN_TJPY": (po.loc["R-CN", "resource_cost_TJPY"]
                                         if key == "paper_only_ex_subsidy" else tt["R-CN"]),
                     "advantage": adv_x})
    df = pd.DataFrame(rows)
    C.write_csv(df, "bridge_steps.csv", out)
    arows = [{"chain": k, "id": d["id"], "name": d["name"], "change": d["change"],
              "energies": d["energies"], "source": d["source"]}
             for k in ("to_unstated", "to_paper_only") for d in cfg[k]]
    C.write_csv(pd.DataFrame(arows), "bridge_assumptions.csv", out)
    summ: Dict[str, Any] = {}
    for chain in chains:
        d = df[(df.chain == chain) & (df.step != "reference")]
        summ[chain] = {"start": float(d.iloc[0]["advantage"]), "end": float(d.iloc[-1]["advantage"]),
                       "steps": {r["step"]: {"advantage": r["advantage"],
                                             "delta": r["delta_advantage"]}
                                 for _, r in d.iterrows()},
                       "in_range_steps": [r["step"] for _, r in d.iterrows()
                                          if 0.18 <= r["advantage"] <= 0.23],
                       "any_curtailment": bool(((d["curtail_C-CN_TWh"] > 1e-6)
                                                | (d["curtail_R-CN_TWh"] > 1e-6)).any())}
    summ["paper_only_advantage_ex_subsidy"] = po_adv_ex
    C.write_json(summ, "bridge_summary.json", out)
    C.write_run_log("bridge", out)
    return C.PartResult(files=list(BRIDGE_FILES), summary=summ)


# ----------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------

def condition_alternatives() -> Dict[Tuple[str, str], Setting]:
    ps, sp = C.params(), C.params(C.SPEC)
    b = Setting()
    alt: Dict[Tuple[str, str], Setting] = {
        ("carbon", "low"): b.with_(carbon=as_tuple(ps.path("carbon_price", "low"))),
        ("carbon", "high"): b.with_(carbon=as_tuple(ps.path("carbon_price", "high"))),
        ("carbon", "flat20000"): b.with_(carbon=C.flat_carbon(ps, 20000.0)),
        ("hw_efficiency", "low"): b.with_(eta=0.20),
        ("hw_efficiency", "high"): b.with_(eta=0.40),
        ("tariff", "low"): b.with_overrides(C.region_variant(ps, "tariff_yen_per_kWh", "low")),
        ("tariff", "high"): b.with_overrides(C.region_variant(ps, "tariff_yen_per_kWh", "high")),
        ("grid_ef", "low"): b.with_overrides(C.region_variant(ps, "grid_ef_base", "low")),
        ("grid_ef", "table9"): b.with_overrides(
            {f"grid_ef_base[{r}]": sp.region("grid_ef_base", r) for r in REGS}),
        ("ef_national", "low"): b.with_overrides(C.path_variant(ps, "ef_national", "low")),
        ("ef_national", "high"): b.with_overrides(C.path_variant(ps, "ef_national", "high")),
        ("demand", "low"): b.with_(demand=as_tuple(ps.path("demand_TWh", "low"))),
        ("demand", "high"): b.with_(demand=as_tuple(ps.path("demand_TWh", "high"))),
        ("pv_lcoe", "low"): b.with_overrides({"pv_lcoe": ps.get("pv_lcoe").low}),
        ("pv_lcoe", "high"): b.with_overrides({"pv_lcoe": ps.get("pv_lcoe").high}),
        ("cooling_capex", "low"): b.with_overrides(
            {"cooling_dlc_capex": ps.get("cooling_dlc_capex").low,
             "cooling_immersion_capex": ps.get("cooling_immersion_capex").low}),
        ("cooling_capex", "high"): b.with_overrides(
            {"cooling_dlc_capex": ps.get("cooling_dlc_capex").high,
             "cooling_immersion_capex": ps.get("cooling_immersion_capex").high}),
        ("nfc_price", "high"): b.with_overrides({"nfc_yen_per_kWh": ps.get("nfc_yen_per_kWh").high}),
        ("subsidy", "high"): b.with_overrides(
            {"siting_subsidy_yen_per_kW": ps.get("siting_subsidy_yen_per_kW").high}),
        ("ccn_allocation", "table7"): b.with_(
            roles=(("REF", "REF"), ("C-CN", "C-CN-gs"), ("R-CN", "R-CN"))),
    }
    return alt


def merge(base: Setting, parts: List[Setting]) -> Setting:
    s = base
    for p in parts:
        if p.overrides:
            s = s.with_overrides(dict(p.overrides))
        if p.carbon is not None:
            s = s.with_(carbon=p.carbon)
        if p.demand is not None:
            s = s.with_(demand=p.demand)
        if p.eta != base.eta:
            s = s.with_(eta=p.eta)
        if p.roles != base.roles:
            s = s.with_(roles=p.roles)
    return s


def run_conditions(out: Path, use_cache: bool = True) -> C.PartResult:
    lo, hi = C.load_config()["target_range"]
    base = Setting()
    alts = condition_alternatives()
    roles = ("C-CN", "R-CN")
    cases: List[Tuple[str, Setting]] = [("base", base)]
    cases += [(f"oat:{k[0]}:{k[1]}", s) for k, s in alts.items()]
    res = solve_settings(cases, roles, "conditions", use_cache)
    rows: List[Dict[str, Any]] = []
    advs: Dict[str, float] = {}
    for cid, s in cases:
        m = metrics(cid, s, res, roles)
        advs[cid] = m["advantage"]
        rows.append({"case": cid, "s": 1.0, **m})
    a0 = advs["base"]
    items = sorted({k[0] for k in alts})
    corners: Dict[str, Setting] = {}
    choice: Dict[str, Dict[str, str]] = {"F": {}, "U": {}}
    for corner, better in (("F", lambda x, y: x > y), ("U", lambda x, y: x < y)):
        chosen = []
        for it in items:
            best, best_a = None, a0
            for (i2, lv), s in alts.items():
                a = advs[f"oat:{i2}:{lv}"]
                if i2 == it and a == a and better(a, best_a):
                    best, best_a = lv, a
            if best is not None:
                chosen.append(alts[(it, best)])
                choice[corner][it] = best
        corners[corner] = merge(base, chosen)
    res2 = solve_settings(list(corners.items()), roles, "conditions", use_cache)
    for cid, s in corners.items():
        m = metrics(cid, s, res2, roles)
        advs[cid] = m["advantage"]
        rows.append({"case": f"corner:{cid}", "s": 1.0, **m})
    grid: List[Tuple[str, Setting]] = []
    in_range = lo <= a0 <= hi
    for cid, s in corners.items():
        a1 = advs[cid]
        if not in_range and min(a0, a1) <= hi and max(a0, a1) >= lo:
            for sv in (0.25, 0.5, 0.75):
                grid.append((f"{cid}@{sv}", C.interpolate(base, s, sv)))
    if grid:
        res3 = solve_settings(grid, roles, "conditions", use_cache)
        for cid, s in grid:
            m = metrics(cid, s, res3, roles)
            advs[cid] = m["advantage"]
            rows.append({"case": f"path:{cid}", "s": float(cid.split("@")[1]), **m})
    df = pd.DataFrame(rows)
    df["in_range"] = (df["advantage"] >= lo) & (df["advantage"] <= hi)
    C.write_csv(df, "conditions_scan.csv", out)
    vals = [v for v in advs.values() if v == v]
    summ = {"target_range": [lo, hi], "base_advantage": a0, "min": min(vals), "max": max(vals),
            "reached": bool(df["in_range"].any()),
            "cases_in_range": list(df[df["in_range"]]["case"]),
            "corner_choice": choice, "corner_advantage": {k: advs[k] for k in corners},
            "any_curtailment": bool(((df["curtail_C-CN_TWh"] > 1e-6)
                                     | (df["curtail_R-CN_TWh"] > 1e-6)).any()),
            "calibration_years": list(CALIBRATION_YEARS)}
    C.write_json(summ, "conditions_summary.json", out)
    C.write_run_log("conditions", out)
    return C.PartResult(files=list(COND_FILES), summary=summ)
