"""Breakeven of the non-metro adders (part breakeven; config/experiments_v35.yaml,
section ``breakeven``, defined before any run).

One kind of adder at a time is added in the non-metro regions (Hokkaido,
Tohoku, Kyushu) of both C-CN and R-CN on the v2 base (rep96, certificates):

  construction   additions shell_premium_yen_per_kW[r] = a (yen per kW of new
                 IT nameplate, once; social resource line shell_premium)
  communication  additions comm_cost_yen_per_kWh[r] = a (yen per kWh of IT
                 energy, every year; the comm_cost hook, line comm_cost)
  grid           reinforcement structure (non-metro c1 from reinforcement_year,
                 theta 0.5, Eq. 1 counts the full cost) with g = a (yen per kW of
                 connected peak added in reinforced years; MILP)

Allocations: R-CN-fix (regional IT of the base R-CN fixed by
RunSpec.realization), R-CN-pub (published shares) and R-CN (endogenous). For the
two fixed allocations gap(a) = C-CN(a) - R-CN(a) is bracketed by doubling from
a_1 and bisected to the tolerance; the two-point closed form -gap(0) / slope
is reported as a check. The endogenous R-CN can avoid non-metro new build when
the adder is high while C-CN (base shares) cannot, so its advantage need not
vanish: it is evaluated on the pre-registered grid and bisected only if a grid
point has an advantage <= 1e-6.

The comparison levels of the config (breakeven_levels.csv) never enter a
computation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd

from . import base as B
from . import common as C
from .bridge import metrics, solve_settings
from .common import Setting, as_tuple

FILES = ("breakeven.csv", "breakeven_curve.csv", "breakeven_levels.csv",
         "run_log_breakeven.csv")
PART = "breakeven"
ROLES = ("C-CN", "R-CN")
NONMETRO = ("Hokkaido", "Tohoku", "Kyushu")
TYPES = ("construction", "communication", "grid")
ALLOCS = ("R-CN-fix", "R-CN-pub", "R-CN")
FIXED = ("R-CN-fix", "R-CN-pub")
UNIT = {"construction": "yen/kW", "communication": "yen/kWh", "grid": "yen/kW"}
A1 = {"construction": 1000.0, "communication": 0.1, "grid": 1000.0}
CAP = {"construction": 1e8, "communication": 1000.0, "grid": 1e8}
TOL = {"construction": 100.0, "communication": 0.001, "grid": 100.0}
ADDER_ID = {"construction": "shell_premium_yen_per_kW", "communication": "comm_cost_yen_per_kWh"}
# Quantity the adder multiplies (non-metro, discounted): column, unit, yen per (adder unit)
QUANTITY = {"construction": ("q_nameplate_disc_GW", "GW (discounted new IT nameplate)", 1e6),
            "communication": ("q_it_energy_disc_TWh", "TWh (discounted IT energy)", 1e9),
            "grid": ("q_reinforced_peak_disc_GW",
                     "GW (discounted connected-peak increments in reinforced years)", 1e6)}
ADV_ZERO = 1e-6
HOURS = 8760.0

Search = Generator[Tuple[str, List[float]], Dict[float, Dict[str, Any]], Dict[str, Any]]


def base_rcn_itr(use_cache: bool) -> Tuple[Tuple[str, int, float], ...]:
    """Regional IT (GW) 2024-2040 of the base R-CN (the RunSpec of part base)."""
    spec = [s for s in B.specs(C.load_config()) if s.scenario == "R-CN"]
    r = C.solve(spec, PART, use_cache)[0]
    a = r.flows["annual"]
    a = a[(a["year"] >= C.YEARS[0]) & (a["year"] <= C.YEARS[-1])]
    return tuple((str(g), int(t), float(x)) for g, t, x in zip(a["region"], a["year"], a["itr_GW"]))


def setting(typ: str, alloc: str, a: float,
            itr: Tuple[Tuple[str, int, float], ...]) -> Setting:
    s = Setting(roles=(("C-CN", "C-CN"), ("R-CN", "R-CN-pub" if alloc == "R-CN-pub" else "R-CN")))
    if alloc == "R-CN-fix":
        s = s.with_(fixed_itr_by_role=(("R-CN", itr),))
    if typ == "grid":
        return s.with_(reinforce_g=float(a))
    if a != 0.0:       # a = 0: no addition, the base RunSpec
        add = as_tuple({f"{ADDER_ID[typ]}[{r}]": a for r in NONMETRO})
        s = s.with_(additions_by_role=(("C-CN", add), ("R-CN", add)))
    return s


def quantities(r: Any) -> Dict[str, float]:
    if r.status != "Optimal":
        return {k: math.nan for k, _, _ in QUANTITY.values()}
    ry = r.tables["region_year"]
    nm = ry[ry["region"].isin(NONMETRO)]
    d = nm["disc"]
    return {"q_nameplate_disc_GW": float((d * nm["build_IT_GW"] / nm["u_cap"]).sum()),
            "q_it_energy_disc_TWh": float((d * nm["it_energy_TWh"]).sum()),
            "q_reinforced_peak_disc_GW": float((d * nm["dpeak_GW"] * nm["reinforced"]).sum())}


def evaluate(points: List[Tuple[str, str, float, str]], itr: Tuple[Tuple[str, int, float], ...],
             use_cache: bool) -> Dict[Tuple[str, str, float], Dict[str, Any]]:
    cases = [(f"{typ}:{alloc}:{a!r}", setting(typ, alloc, a, itr)) for typ, alloc, a, _ in points]
    res = solve_settings(cases, ROLES, PART, use_cache)
    out: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    for (typ, alloc, a, src), (cid, s) in zip(points, cases):
        m = metrics(cid, s, res, ROLES)
        c, r = res[(cid, "C-CN")], res[(cid, "R-CN")]
        row: Dict[str, Any] = {"type": typ, "allocation": alloc, "adder": a, "unit": UNIT[typ],
                               "source": src, **m,
                               "gap_TJPY": (C.total(c) - C.total(r)) / C.TJ}
        for role, rr in (("C-CN", c), ("R-CN", r)):
            for k, v in quantities(rr).items():
                row[f"{k}_{role}"] = v
        out[(typ, alloc, a)] = row
    return out


def fixed_search(typ: str) -> Search:
    a1, cap, tol = A1[typ], CAP[typ], TOL[typ]
    g = yield ("start", [0.0, a1])
    g0, g1 = g[0.0]["gap_TJPY"], g[a1]["gap_TJPY"]
    slope = (g1 - g0) / a1
    info: Dict[str, Any] = {"gap0_TJPY": g0, "slope_TJPY_per_unit": slope,
                            "closed_form": (-g0 / slope) if slope < 0 else math.nan}
    if math.isnan(g0) or math.isnan(g1):
        return {**info, "status": "solve failed", "lo": math.nan, "hi": math.nan}
    if g0 <= 0:
        return {**info, "status": "no advantage at zero", "lo": math.nan, "hi": math.nan}
    lo, hi, ghi = 0.0, a1, g1
    while ghi > 0 and hi < cap:
        lo, hi = hi, min(2.0 * hi, cap)
        ghi = (yield ("doubling", [hi]))[hi]["gap_TJPY"]
        if math.isnan(ghi):
            return {**info, "status": "solve failed", "lo": lo, "hi": hi}
    if ghi > 0:
        return {**info, "status": "not found", "lo": lo, "hi": hi}
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        gm = (yield ("bisection", [mid]))[mid]["gap_TJPY"]
        if math.isnan(gm):
            return {**info, "status": "solve failed", "lo": lo, "hi": hi}
        if gm > 0:
            lo = mid
        else:
            hi = mid
    return {**info, "status": "found", "lo": lo, "hi": hi}


def grid_search(typ: str, grid: List[float]) -> Search:
    tol = TOL[typ]
    g = yield ("grid", list(grid))
    adv = {a: g[a]["advantage"] for a in grid}
    info: Dict[str, Any] = {"gap0_TJPY": g[grid[0]]["gap_TJPY"], "closed_form": math.nan,
                            "slope_TJPY_per_unit": math.nan}
    ok = {a: v for a, v in adv.items() if not math.isnan(v)}
    a_min = min(ok, key=lambda a: ok[a]) if ok else math.nan
    info.update(min_advantage=ok.get(a_min, math.nan), min_advantage_adder=a_min,
                n_failed=len(adv) - len(ok))
    hit = [i for i, a in enumerate(grid) if a in ok and ok[a] <= ADV_ZERO]
    if not hit:
        return {**info, "status": "not reached in the grid", "lo": math.nan, "hi": math.nan}
    if hit[0] == 0:
        return {**info, "status": "no advantage at zero", "lo": math.nan, "hi": math.nan}
    lo, hi = grid[hit[0] - 1], grid[hit[0]]
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        am = (yield ("bisection", [mid]))[mid]["advantage"]
        if not math.isnan(am) and am <= ADV_ZERO:
            hi = mid
        else:
            lo = mid
    return {**info, "status": "found", "lo": lo, "hi": hi}


def drive(searches: Dict[Tuple[str, str], Search], itr: Tuple[Tuple[str, int, float], ...],
          use_cache: bool) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]],
                                    Dict[Tuple[str, str, float], Dict[str, Any]], Dict[str, int]]:
    """Advance every search in lockstep: one solve call per round (the cache is
    written after each round, so an interrupted run resumes)."""
    memo: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    pending: Dict[Tuple[str, str], Tuple[str, List[float]]] = {
        k: next(gen) for k, gen in searches.items()}
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    n_eval: Dict[str, int] = {f"{t}:{al}": 0 for t, al in searches}
    rounds = 0
    while pending:
        todo = [(t, al, a, src) for (t, al), (src, pts) in pending.items() for a in pts
                if (t, al, a) not in memo]
        memo.update(evaluate(todo, itr, use_cache))
        rounds += 1
        nxt: Dict[Tuple[str, str], Tuple[str, List[float]]] = {}
        for (t, al), (_, pts) in pending.items():
            n_eval[f"{t}:{al}"] += len(pts)
            try:
                nxt[(t, al)] = searches[(t, al)].send({a: memo[(t, al, a)] for a in pts})
            except StopIteration as stop:
                results[(t, al)] = stop.value
        pending = nxt
    n_eval["rounds"] = rounds
    return results, memo, n_eval


def level_rows(levels: Dict[str, Any], found: Dict[str, Optional[float]]) -> List[Dict[str, Any]]:
    """Comparison levels of the config with conversions to the breakeven units."""
    rows: List[Dict[str, Any]] = []

    def add(typ: str, d: Dict[str, Any], variant: str, value: float, unit: str,
            value_b: float, conversion: str) -> None:
        be = found.get(typ)
        rows.append({"type": typ, "id": d["id"], "variant": variant, "status": d["status"],
                     "value": value, "unit": unit, "value_breakeven_unit": value_b,
                     "breakeven_unit": UNIT[typ], "breakeven_R-CN-fix": be,
                     "level_over_breakeven_R-CN-fix": (value_b / be) if (be and value_b == value_b)
                     else math.nan,
                     "source": " ".join(str(d.get("source", d.get("searched", ""))).split()),
                     "conversion": " ".join(conversion.split()),
                     "caveat": " ".join(str(d.get("caveat", d.get("note", ""))).split())})

    for d in levels["grid"]:
        add("grid", d, "", float(d["value"]), d["unit"], float(d["value"]), d["conversion"])
    for d in levels["construction"]:
        if "values_yen_per_m2" in d:
            diff = float(d["values_yen_per_m2"]["metro_minus_nonmetro"])
            for dens in (1.0, 2.0, 3.0):
                # metro minus non-metro; the non-metro premium of the breakeven is its negative
                add("construction", d, f"it_density_{dens:g}_kW_per_m2", diff,
                    "yen/m2 (metro - nonmetro)", -diff / dens,
                    f"nonmetro premium = -(metro - nonmetro) / IT density = {-diff:g} / {dens:g} "
                    "(IT density kW/m2 仮定; 2 = centre)")
        else:
            add("construction", d, "", float(d["value"]), d["unit"], float(d["value"]),
                d.get("conversion", ""))
    for d in levels["communication"]:
        if "value" not in d:
            add("communication", d, "", math.nan, "", math.nan, "")
            continue
        for x in (1.0, 10.0, 50.0):
            yk = float(d["value"]) * 12.0 / 100.0 * x / 8.76e6
            add("communication", d, f"traffic_{x:g}_Gbps_per_MW", float(d["value"]), d["unit"],
                yk, f"{float(d['value']):g} x 12 / 100 Gbps x {x:g} Gbps/MW / 8.76e6 kWh per MW-yr "
                "(traffic intensity 仮定)")
    return rows


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    cfg = C.load_v35()["breakeven"]
    grids = {t: [float(a) for a in cfg["procedure"]["grids"][t]] for t in TYPES}
    for t in TYPES:
        if grids[t][0] != 0.0:
            raise ValueError("the endogenous grids must start at 0")
    itr = base_rcn_itr(use_cache)
    searches: Dict[Tuple[str, str], Search] = {}
    for t in TYPES:
        for al in ALLOCS:
            searches[(t, al)] = fixed_search(t) if al in FIXED else grid_search(t, grids[t])
    results, memo, n_eval = drive(searches, itr, use_cache)
    base_adv = C.adv(*[C.total(r) for r in C.solve(
        [s for s in B.specs(C.load_config()) if s.scenario in ("C-CN", "R-CN")], PART, use_cache)])

    rows: List[Dict[str, Any]] = []
    found: Dict[str, Optional[float]] = {}
    for t in TYPES:
        qcol, qunit, _ = QUANTITY[t]
        for al in ALLOCS:
            res = results[(t, al)]
            p0 = memo[(t, al, 0.0)]
            lo, hi = res["lo"], res["hi"]
            value = 0.5 * (lo + hi) if res["status"] == "found" else math.nan
            if al == "R-CN-fix":
                found[t] = value if value == value else None
            ph = memo.get((t, al, hi)) if res["status"] == "found" else None
            row: Dict[str, Any] = {
                "type": t, "allocation": al, "status": res["status"], "value": value,
                "unit": UNIT[t], "lo": lo, "hi": hi, "closed_form": res["closed_form"],
                "tolerance": TOL[t], "gap0_TJPY": res["gap0_TJPY"],
                "advantage0": p0["advantage"],
                "slope_TJPY_per_unit": res["slope_TJPY_per_unit"],
                "min_advantage": res.get("min_advantage", math.nan),
                "min_advantage_adder": res.get("min_advantage_adder", math.nan),
                "n_evaluations": n_eval[f"{t}:{al}"],
                "quantity": qcol, "quantity_unit": qunit,
                "q_C-CN_at0": p0[f"{qcol}_C-CN"], "q_R-CN_at0": p0[f"{qcol}_R-CN"],
                "q_R-CN_minus_C-CN_at0": p0[f"{qcol}_R-CN"] - p0[f"{qcol}_C-CN"],
                "q_R-CN_minus_C-CN_at_hi": (ph[f"{qcol}_R-CN"] - ph[f"{qcol}_C-CN"]) if ph
                else math.nan,
                "value_yen_per_kW_yr": value * HOURS if t == "communication" else math.nan,
                "closed_form_yen_per_kW_yr": (res["closed_form"] * HOURS if t == "communication"
                                              else math.nan),
                "base_advantage": base_adv,
                "advantage0_rel_diff_to_base": ((p0["advantage"] - base_adv) / base_adv
                                                if t != "grid" and al != "R-CN-pub"
                                                else math.nan),
            }
            rows.append(row)
    for row in rows:
        if row["allocation"] == "R-CN-fix" and row["type"] != "grid":
            if abs(row["advantage0_rel_diff_to_base"]) > 1e-6:
                raise AssertionError(f"R-CN-fix at zero differs from the base: {row}")
    spec = pd.read_csv(C.REPO / "outputs" / "stage2" / "breakeven.csv").set_index("config")
    po = spec.loc["paper_only"]
    rows.append({"type": "construction", "allocation": "spec paper_only (reference)",
                 "status": str(po["status"]), "value": float(po["breakeven_yen_per_kW"]),
                 "unit": "yen/kW", "lo": float(po["breakeven_lo_yen_per_kW"]),
                 "hi": float(po["breakeven_hi_yen_per_kW"]),
                 "closed_form": float(po["closed_form_yen_per_kW"]),
                 "tolerance": float(po["tolerance_yen_per_kW"]),
                 "gap0_TJPY": float(po["gap_at_zero_TJPY"]),
                 "advantage0": float(po["advantage_at_zero"]),
                 "quantity": "outputs/stage2/breakeven.csv (spec version, R-CN-pub shares, annual)"})
    C.write_csv(pd.DataFrame(rows), "breakeven.csv", out)
    curve = pd.DataFrame(list(memo.values()))
    curve = curve.sort_values(["type", "allocation", "adder"], kind="mergesort").reset_index(drop=True)
    C.write_csv(curve, "breakeven_curve.csv", out)
    C.write_csv(pd.DataFrame(level_rows(cfg["levels"], found)), "breakeven_levels.csv", out)
    C.write_run_log(PART, out)
    return C.PartResult(files=list(FILES), summary={"rounds": n_eval["rounds"]})

