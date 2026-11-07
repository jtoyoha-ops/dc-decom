"""Four interaction types by a 2x2 factorial (config/interaction_pairs.yaml).

For each pair (X, Y), condition theta and metric M (smaller is better), four
perfect-foresight solves on the base path from the A0 baseline: empty, X, Y,
X+Y. Delta_M(S) = M(empty) - M(S), I_M = Delta_M(X+Y) - Delta_M(X) - Delta_M(Y)
(ch07 eq:dpef-interaction); the type follows ch07 tab:dpef-interaction with
eps = 0.5% of |M(empty)| (robustness 0.1%, 1%). Outputs outputs/abc/
interactions.csv and interactions_table.md (ready for ch07
tab:dpef-interaction-metric).
"""
from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import yaml  # type: ignore[import-untyped]

from .. import runner
from ..config import CALIBRATION_YEARS, METRO, Paths, PolicySettings, year_map
from ..model import Result
from ..params import ParamSet
from ..scenarios import default_paths, default_policy
from . import common as C
from .abc import load_rules

CONFIG = C.REPO / "config" / "interaction_pairs.yaml"
FILES = ("interactions.csv", "interactions_table.md", "run_log_interactions.csv")
COLD = ("Hokkaido", "Tohoku")
FIRST = CALIBRATION_YEARS[-1] + 1

METRIC_JA = {
    "cooling_electricity_cum_TWh": "冷却電力（施設−IT、2024〜2040年累積、TWh）",
    "re_procurement_cost_TJPY": "再エネ調達費（PPA＋証書＋自家PV、兆円）",
    "co2_cum_Mt": "CO₂（2024〜2040年累積、Mt）",
    "co2_2040_Mt": "CO₂（2040年、Mt）",
    "peak_facility_nonmetro_2040_GW": "系統への負荷：非首都圏の施設ピークの和（2040年、GW）",
    "conn_usage_2040_max": "系統への負荷：接続容量の使用率（2040年、最大）",
    "one_minus_nonmetro_share_2040": "1−非首都圏のIT比率（2040年）",
    "one_minus_nonmetro_share_mean": "1−非首都圏のIT比率（2026〜2040年平均）",
    "fiscal_burden_TJPY": "財政負担（兆円）",
    "social_cost_TJPY": "社会的資源費用（兆円）",
}
COND_JA = {"liquid_cap": {"base": "液冷の導入上限 基準", "slow": "液冷の導入上限 遅い"},
           "connection": {"c0": "接続 c0", "c1": "接続 c1（2030年から増強）"},
           "carbon": {"low": "炭素価格 低", "base": "炭素価格 基準", "high": "炭素価格 高"},
           "certificates": {"off": "証書なし", "hourly": "24/7（時間単位の一致・証書なし）"}}
COND_SHORT = {"liquid_cap": "liquid", "connection": "conn", "carbon": "carbon",
              "certificates": "cert"}


def load_config() -> Dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _mix(ps: ParamSet, pid: str, variant: str, base: Mapping[int, float]) -> Dict[int, float]:
    v = ps.path(pid, variant)
    return {t: float(base[t] if t < FIRST else v[t]) for t in sorted(base)}


def spec_for(elements: Sequence[str], cond: Mapping[str, str], label: str,
             phi0: float) -> runner.RunSpec:
    ps = C.params()
    base = default_paths(ps)
    pol: PolicySettings = default_policy("A0", ps)
    nonmetro = [r for r in ps.region_names if r not in METRO]
    s0 = ps.scalar("siting_subsidy_yen_per_kW")
    ch: Dict[str, Any] = {}
    carbon = dict(base.carbon_price)
    liquid = dict(base.liquid_deploy_cap)
    overrides: Dict[str, float] = {}
    els = set(elements)
    if cond.get("connection") == "c1":
        els.add("reinforcement")
    if cond.get("carbon", "base") != "base":
        carbon = _mix(ps, "carbon_price", "low" if cond["carbon"] == "low" else "high",
                      base.carbon_price)
    if cond.get("liquid_cap") == "slow":
        liquid = _mix(ps, "liquid_deploy_cap", "low", base.liquid_deploy_cap)
    for e in sorted(els):
        if e == "liquid_floor":
            ch["liquid_floor_new"] = year_map(phi0, FIRST)
        elif e in ("cold_subsidy", "uniform_subsidy"):
            regs = [r for r in nonmetro if e == "uniform_subsidy" or r in COLD]
            ch["subsidy_yen_per_kW"] = {r: year_map(s0, FIRST) for r in regs}
        elif e == "carbon_high":
            carbon = _mix(ps, "carbon_price", "high", base.carbon_price)
        elif e == "reinforcement":
            tg = int(ps.scalar("reinforcement_year"))
            conn = {r: {t: (ps.region("conn_capacity_reinforced_GW_per_yr", r) if t >= tg else v)
                        for t, v in yrs.items()} for r, yrs in pol.conn_capacity.items()}
            ch["conn_capacity"] = conn
            ch["reinforced_from"] = {r: tg for r in conn}
        elif e == "ef_low":
            low = ps.path("ef_national", "low")
            overrides.update({f"ef_national@{t}": float(v) for t, v in low.items() if t >= FIRST})
        elif e == "offgrid_re":
            overrides["ppa_tier2_premium_yen_per_kWh"] = float(
                ps.get("ppa_tier2_premium_yen_per_kWh").low)
        else:
            raise ValueError(f"unknown element {e!r}")
    paths = Paths(carbon_price=carbon, demand_TWh=dict(base.demand_TWh), liquid_deploy_cap=liquid)
    options = C.BASE_OPTIONS
    if cond.get("certificates") == "off":
        options = options.with_(certificates=False)
    elif cond.get("certificates") == "hourly":
        options = options.with_(certificates=False, ppa_matching="hourly")
    return runner.RunSpec(scenario="A0", existing_dir=C.V2, options=options,
                          overrides=tuple(sorted(overrides.items())), paths=paths,
                          policy=pol.with_(**ch), store_slices=False, label=label)


def metrics(r: Result) -> Dict[str, float]:
    if r.status != "Optimal":
        return {k: math.nan for k in METRIC_JA}
    m = r.metrics
    ry = r.tables["region_year"]
    nat = r.tables["national"]
    later = nat[nat["year"] >= FIRST]
    return {
        "cooling_electricity_cum_TWh": m["facility_energy_cum_TWh"] - m["it_energy_cum_TWh"],
        "re_procurement_cost_TJPY": float((ry["energy_ppa_disc"] + ry["energy_nfc_disc"]
                                           + ry["energy_pv_disc"]).sum()) / C.TJ,
        "co2_cum_Mt": m["co2_cum_Mt"],
        "co2_2040_Mt": m["co2_2040_Mt"],
        "peak_facility_nonmetro_2040_GW": m["peak_facility_nonmetro_last_GW"],
        "conn_usage_2040_max": m["conn_usage_last_max"],
        "one_minus_nonmetro_share_2040": 1.0 - m["nonmetro_share_last"],
        "one_minus_nonmetro_share_mean": 1.0 - float(later["nonmetro_share"].mean()),
        "fiscal_burden_TJPY": m["fiscal_burden_yen"] / C.TJ,
        "social_cost_TJPY": m["social_cost_yen"] / C.TJ,
    }


def classify(m0: float, mx: float, my: float, mxy: float, rel: float,
             abs_floor: float) -> Tuple[str, bool, str, float]:
    """(type, conflict, conflict_by, eps) by ch07 tab:dpef-interaction."""
    eps = max(rel * abs(m0), abs_floor)
    dx, dy, dxy = m0 - mx, m0 - my, m0 - mxy
    i = dxy - dx - dy
    if dy <= eps and dxy - dx > eps:
        t = "前提(X→Y)"
    elif dx <= eps and dxy - dy > eps:
        t = "前提(Y→X)"
    elif i > eps:
        t = "増幅"
    elif i < -eps:
        t = "代替"
    else:
        t = "独立"
    by = [n for n, d in (("X", dx), ("Y", dy)) if d < -eps]
    return t, bool(by), ",".join(by), eps


def conditions(spec: Mapping[str, Sequence[str]]) -> List[Dict[str, str]]:
    keys = list(spec)
    return [dict(zip(keys, vals)) for vals in product(*[spec[k] for k in keys])] or [{}]


def cond_id(c: Mapping[str, str]) -> str:
    return "_".join(f"{COND_SHORT[k]}-{v}" for k, v in c.items()) or "base"


def cond_ja(c: Mapping[str, str]) -> str:
    return "，".join(COND_JA[k][v] for k, v in c.items()) or "基準"


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    cfg = load_config()
    phi0 = float(load_rules()["thresholds"]["phi_0"]["value"])
    cls = cfg["classification"]["epsilon"]
    rel, robust, floor = float(cls["relative"]), [float(x) for x in cls["robustness"]], float(
        cls["absolute_floor"])
    jobs: List[Tuple[str, Dict[str, str], str]] = []
    specs: List[runner.RunSpec] = []
    for pid, p in cfg["pairs"].items():
        if p.get("status") == "対象外":
            continue
        for c in conditions(p.get("conditions") or {}):
            for combo in ("empty", "X", "Y", "XY"):
                els = {"empty": [], "X": [p["X"]], "Y": [p["Y"]], "XY": [p["X"], p["Y"]]}[combo]
                jobs.append((pid, c, combo))
                specs.append(spec_for(els, c, f"interactions:{pid}:{cond_id(c)}:{combo}", phi0))
    res = C.solve(specs, "interactions", use_cache)
    vals: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    status: Dict[Tuple[str, str], List[str]] = {}
    for (pid, c, combo), r in zip(jobs, res):
        vals[(pid, cond_id(c), combo)] = metrics(r)
        status.setdefault((pid, cond_id(c)), []).append(r.status)
    rows = []
    for pid, p in cfg["pairs"].items():
        if p.get("status") == "対象外":
            rows.append({"pair": pid, "pair_name": p["name"], "condition": "", "metric": "",
                         "type": "対象外", "status": p.get("reason", "")})
            continue
        for c in conditions(p.get("conditions") or {}):
            cid = cond_id(c)
            st = status[(pid, cid)]
            for metric in p["metrics"]:
                m0, mx, my, mxy = (vals[(pid, cid, k)][metric] for k in ("empty", "X", "Y", "XY"))
                row: Dict[str, Any] = {"pair": pid, "pair_name": p["name"], "X": p["X"],
                                       "Y": p["Y"], "condition": cid, "condition_ja": cond_ja(c),
                                       "metric": metric, "M_empty": m0, "M_X": mx, "M_Y": my,
                                       "M_XY": mxy, "D_X": m0 - mx, "D_Y": m0 - my,
                                       "D_XY": m0 - mxy, "I": (m0 - mxy) - (m0 - mx) - (m0 - my)}
                t, conf, by, eps = classify(m0, mx, my, mxy, rel, floor)
                row.update(eps=eps, type=t, conflict=conf, conflict_by=by)
                for x in robust:
                    tr, cr, _, _ = classify(m0, mx, my, mxy, x, floor)
                    row[f"type_eps{x:g}"] = tr
                    row[f"conflict_eps{x:g}"] = cr
                row["status"] = "Optimal" if all(s == "Optimal" for s in st) else ";".join(st)
                rows.append(row)
    df = pd.DataFrame(rows)
    C.write_csv(df, "interactions.csv", out)
    (out / "interactions_table.md").write_text(table_md(df, cfg, robust), encoding="utf-8")
    C.write_run_log("interactions", out)
    summary = {"n_rows": int((df["type"] != "対象外").sum()),
               "types": df["type"].value_counts().to_dict()}
    return C.PartResult(files=list(FILES), summary=summary)


def _f(x: Any, nd: int = 4) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "—"
    return f"{xf:.{nd}g}"


def table_md(df: pd.DataFrame, cfg: Mapping[str, Any], robust: Sequence[float]) -> str:
    el = cfg["elements"]
    lines = [
        "<!-- generated by src/dc_decom_v2/experiments/interactions.py from "
        "config/interaction_pairs.yaml; do not edit -->",
        "",
        "評価指標と条件による相互作用の類型（DC-DECOM v2、rep96、基準の経路、完全予見、A0 を基準"
        "とする 2×2 の計算。ε＝|M(∅)| の 0.5%）",
        "",
        "| 組合せ | 評価指標 | 条件 | 類型 | 相反 | I_M | Δ_M(X)／Δ_M(Y)／Δ_M(X∪Y) | ε | "
        f"頑健性（ε {robust[0]:.1%}／{robust[1]:.0%}） |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in df.to_dict("records"):
        if d["type"] == "対象外":
            lines.append(f"| {d['pair_name']} | — | — | 対象外 | — | — | — | — | {d['status']} |")
            continue
        conf = f"相反（{d['conflict_by']}）" if d["conflict"] else "—"
        rob = "／".join(str(d[f"type_eps{x:g}"]) for x in robust)
        lines.append(
            f"| {d['pair_name']}（X＝{d['X']}、Y＝{d['Y']}） | {METRIC_JA[d['metric']]} | "
            f"{d['condition_ja']} | {d['type']} | {conf} | {_f(d['I'])} | "
            f"{_f(d['D_X'])}／{_f(d['D_Y'])}／{_f(d['D_XY'])} | {_f(d['eps'], 3)} | {rob} |")
    lines += ["", "要素の定義（config/interaction_pairs.yaml）："]
    lines += [f"- {k}：{v}" for k, v in el.items()]
    lines += ["", "判定：Δ_M(S)＝M(∅)−M(S)、I_M＝Δ_M(X∪Y)−Δ_M(X)−Δ_M(Y)。前提（X→Y）：Δ_M(Y)≦ε "
              "かつ Δ_M(X∪Y)−Δ_M(X)＞ε（Y→X も確認）、増幅：I_M＞ε、代替：I_M＜−ε、ほかは独立。"
              "相反は別の印（X か Y の単独で M が ε を超えて悪化）。", ""]
    return "\n".join(lines)
