"""Carbon price x certificate rule (part carbon_cert; config/experiments_v35.yaml,
section ``carbon_cert``, defined before any run).

Rules: cert_on (certificates, the base; the RunSpecs of sensitivity.sweep_cases),
cert_off (no certificates, annual matching) and hourly (no certificates,
hourly matching = 24/7). Flat carbon price from 2026, 0 to 30,000 yen/t in
steps of 1,000 (2024-2025 at the base path); C-CN, R-CN and A0 per price.

P*_2040(rule): the minimum flat price at which A0 reaches the 2040 non-metro
share of R-CN solved with the same rule and the base carbon path; the sweep
brackets the first crossing and bisection narrows it to 100 yen/t ('reached at
0' / 'not reached' within 0-30,000). P*_mean(rule): the same with the mean
non-metro share 2026-2040. P*_frozen(rule): the same with the frozen target of
outputs/stage3/pstar.json (target_share). Siting effect: the sweep prices at
which the A0 share (2040, mean) differs from its value at price 0 by more than
0.001.

carbon_cert_regions.csv (added after the results had been seen;
changes_after_registration G1 of config/experiments_v35.yaml): the 2040
regional IT capacity of A0, its share, the grid emission factor and the grid
and PPA energy per region, read from the same sweep solves. Descriptive only;
no definition or other output changes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..config import ModelOptions
from . import common as C
from .bridge import metrics, solve_settings
from .common import Setting
from .sensitivity import ROLES as SWEEP_ROLES
from .sensitivity import sweep_prices

FILES = ("carbon_cert_sweep.csv", "pstar_regimes.json", "pstar_regimes_curve.csv",
         "carbon_cert_regions.csv", "run_log_carbon_cert.csv")
PART = "carbon_cert"
SCEN = ("C-CN", "R-CN", "A0")
TOL = 100.0
STEP = 1000.0
SITING_EPS = 0.001
FIRST = 2026
TARGETS = ("2040", "mean", "frozen")


def regimes() -> Dict[str, ModelOptions]:
    on = C.BASE_OPTIONS
    return {"cert_on": on,
            "cert_off": on.with_(certificates=False),
            "hourly": on.with_(certificates=False, ppa_matching="hourly")}


def sweep_setting(opt: ModelOptions, price: float) -> Setting:
    s = Setting(roles=tuple((x, x) for x in SCEN), carbon=C.flat_carbon(C.params(), price))
    return s if opt == C.BASE_OPTIONS else s.with_(options=opt)


def mean_share(r: Any) -> float:
    if r.status != "Optimal":
        return math.nan
    nat = r.tables["national"]
    return float(nat[nat["year"] >= FIRST]["nonmetro_share"].mean())


def share_2040(r: Any) -> float:
    return float(r.metrics["nonmetro_share_last"]) if r.status == "Optimal" else math.nan


def _cum(r: Any, col: str) -> float:
    return float(r.tables["region_year"][col].sum()) if r.status == "Optimal" else math.nan


def sweep_row(regime: str, p: int, s: Setting, res: Dict[Tuple[str, str], Any]) -> Dict[str, Any]:
    cid = f"{regime}:{p}"
    m = metrics(cid, s, res, SWEEP_ROLES)
    a0 = res[(cid, "A0")]
    ok = a0.status == "Optimal"
    row: Dict[str, Any] = {
        "regime": regime, "price_yen_per_t": p, **m,
        "status_A0": a0.status, "total_A0_TJPY": C.total(a0) / C.TJ,
        "operator_A0_TJPY": a0.metrics["operator_cost_yen"] / C.TJ if ok else None,
        "nonmetro_2040_A0": a0.metrics["nonmetro_share_last"] if ok else None,
        "co2_2040_A0_Mt": a0.metrics["co2_2040_Mt"] if ok else None,
        "co2_cum_A0_Mt": a0.metrics["co2_cum_Mt"] if ok else None,
        "liquid_share_fleet_2040_A0": a0.metrics["liquid_share_fleet_last"] if ok else None,
        "nonmetro_mean_2026_2040_A0": mean_share(a0)}
    for x in SCEN:
        r = res[(cid, x)]
        row[f"nfc_cum_{x}_TWh"] = _cum(r, "nfc_TWh")
        row[f"ppa_cum_{x}_TWh"] = _cum(r, "ppa_TWh")
    return row


def region_rows(regime: str, p: int, a0: Any) -> List[Dict[str, Any]]:
    """A0 by region in the last year (descriptive; carbon_cert_regions.csv)."""
    if a0.status != "Optimal":
        return []
    ry = a0.tables["region_year"]
    last = ry[ry["year"] == ry["year"].max()]
    span = ry[ry["year"] >= FIRST]
    tot = float(last["itr_GW"].sum())
    return [{"regime": regime, "price_yen_per_t": p, "year": int(x["year"]), "region": x["region"],
             "metro": bool(x["metro"]), "itr_GW": x["itr_GW"],
             "itr_share": x["itr_GW"] / tot if tot else math.nan,
             "ef_grid_kg_per_kWh": x["ef_grid"],
             "ef_grid_mean_2026_2040": float(span[span["region"] == x["region"]]["ef_grid"].mean()),
             "grid_received_TWh": x["grid_received_TWh"], "ppa_TWh": x["ppa_TWh"],
             "ppa_delivered_TWh": x["ppa_delivered_TWh"], "grid_net_TWh": x["grid_net_TWh"],
             "em_grid_Mt": x["em_grid_Mt"]}
            for _, x in last.iterrows()]


def _a0(regime: str, opt: ModelOptions, prices: List[float], use_cache: bool) -> List[Any]:
    specs = [C.spec_for(sweep_setting(opt, p), "A0", f"{PART}:{regime}:A0:{p:g}") for p in prices]
    return C.solve(specs, PART, use_cache)


def pstar(curve: List[Tuple[float, float]], target: float) -> Dict[str, Any]:
    """First crossing of the sweep (price, share) for ``target`` (no bisection yet)."""
    reached = [p for p, sh in curve if sh >= target - 1e-9]
    if not reached:
        return {"status": "not reached", "pstar_yen_per_t": None}
    if reached[0] == 0.0:
        return {"status": "reached at 0", "pstar_yen_per_t": 0.0}
    return {"status": "found", "bracket": [reached[0] - STEP, reached[0]]}


def siting_effect(curve: List[Tuple[float, float]]) -> Dict[str, Any]:
    s0 = curve[0][1]
    moved = [p for p, sh in curve if abs(sh - s0) > SITING_EPS]
    return {"share_at_0": s0, "threshold": SITING_EPS, "n_prices": len(moved),
            "first_price": moved[0] if moved else None, "last_price": moved[-1] if moved else None,
            "prices": moved}


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    regs = regimes()
    prices = sweep_prices()
    frozen = float(json.loads((C.OUT / "pstar.json").read_text(encoding="utf-8"))["target_share"])
    cases = [(f"{rg}:{p}", sweep_setting(opt, float(p))) for rg, opt in regs.items() for p in prices]
    res = solve_settings(cases, SCEN, PART, use_cache)
    tgt_specs = [C.spec_for(Setting() if opt == C.BASE_OPTIONS else Setting(options=opt), "R-CN",
                            f"{PART}:{rg}:R-CN:base_carbon") for rg, opt in regs.items()]
    tgt = dict(zip(regs, C.solve(tgt_specs, PART, use_cache)))

    rows: List[Dict[str, Any]] = []
    regs_rows: List[Dict[str, Any]] = []
    curves: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    pts: List[Dict[str, Any]] = []
    for (cid, s) in cases:
        rg, p = cid.split(":")
        rows.append(sweep_row(rg, int(p), s, res))
        a0 = res[(cid, "A0")]
        regs_rows.extend(region_rows(rg, int(p), a0))
        cv = curves.setdefault(rg, {"2040": [], "mean": []})
        cv["2040"].append((float(p), share_2040(a0)))
        cv["mean"].append((float(p), mean_share(a0)))
        pts.append({"regime": rg, "price_yen_per_t": float(p), "nonmetro_2040_A0": share_2040(a0),
                    "nonmetro_mean_2026_2040_A0": mean_share(a0), "status": a0.status,
                    "source": "sweep"})

    summary: Dict[str, Any] = {"tolerance_yen_per_t": TOL, "frozen_target_share": frozen,
                               "frozen_target_source": "outputs/stage3/pstar.json target_share",
                               "regimes": {}}
    state: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rg in regs:
        t = tgt[rg]
        targets = {"2040": share_2040(t), "mean": mean_share(t), "frozen": frozen}
        d: Dict[str, Any] = {"status_R-CN_base_carbon": t.status,
                             "target_2040": targets["2040"], "target_mean": targets["mean"],
                             "nonmetro_2040_A0_at_0": curves[rg]["2040"][0][1],
                             "nonmetro_mean_A0_at_0": curves[rg]["mean"][0][1],
                             "siting_effect_2040": siting_effect(curves[rg]["2040"]),
                             "siting_effect_mean": siting_effect(curves[rg]["mean"])}
        for k in TARGETS:
            basis = "mean" if k == "mean" else "2040"
            r = pstar(curves[rg][basis], targets[k])
            r.update(target=targets[k], basis=basis)
            d[f"pstar_{k}"] = r
            if r["status"] == "found":
                state[(rg, k)] = r
        summary["regimes"][rg] = d

    # bisection in lockstep over the (regime, target) pairs
    while True:
        act = [(key, r) for key, r in state.items() if r["bracket"][1] - r["bracket"][0] > TOL]
        if not act:
            break
        need = sorted({(rg, 0.5 * (r["bracket"][0] + r["bracket"][1])) for (rg, _), r in act})
        got: Dict[Tuple[str, float], Any] = {}
        for rg in regs:
            ps_ = [p for g, p in need if g == rg]
            if ps_:
                got.update({(rg, p): x for p, x in zip(ps_, _a0(rg, regs[rg], ps_, use_cache))})
        for (rg, pr), a0 in sorted(got.items()):
            pts.append({"regime": rg, "price_yen_per_t": pr, "nonmetro_2040_A0": share_2040(a0),
                        "nonmetro_mean_2026_2040_A0": mean_share(a0), "status": a0.status,
                        "source": "bisection"})
        for (rg, k), r in act:
            mid = 0.5 * (r["bracket"][0] + r["bracket"][1])
            a0 = got[(rg, mid)]
            sh = mean_share(a0) if r["basis"] == "mean" else share_2040(a0)
            if sh >= r["target"] - 1e-9:
                r["bracket"][1] = mid
            else:
                r["bracket"][0] = mid
    for (rg, k), r in state.items():
        r["pstar_yen_per_t"] = r["bracket"][1]

    C.write_csv(pd.DataFrame(rows), "carbon_cert_sweep.csv", out)
    C.write_csv(pd.DataFrame(regs_rows), "carbon_cert_regions.csv", out)
    C.write_json(summary, "pstar_regimes.json", out)
    curve = pd.DataFrame(pts).drop_duplicates(["regime", "price_yen_per_t"], keep="first")
    curve = curve.sort_values(["regime", "price_yen_per_t"], kind="mergesort").reset_index(drop=True)
    C.write_csv(curve, "pstar_regimes_curve.csv", out)
    C.write_run_log(PART, out)
    return C.PartResult(files=list(FILES), summary=summary)
