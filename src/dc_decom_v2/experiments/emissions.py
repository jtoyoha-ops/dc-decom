"""Shapley decomposition of the emission reduction (part emissions;
config/experiments_v35.yaml, section ``emissions``, defined before any run).

Six levers, each off or on (rep96, base demand, no CO2 cap, no subsidy, all
other values frozen):
    L liquid cooling          off: PolicySettings.liquid_fleet_max = 0 in every region (2026+)
    R RE procurement          off: ppa_tier1_share = ppa_tier2_share = Deploy_max_PV = 0
    G geographic allocation   off: scenario REF (base shares); on: A0 (endogenous)
    D grid decarbonization    off: ef_national@t = 0.423 (the 2024 value) in every year
    N certificates            off: ModelOptions.certificates = False
    P carbon price            off: 0 in every year; on: the base path
All 2^6 = 64 combinations are solved. With E(S) the emissions of the run in
which the levers S are on, the game is v(S) = E(empty) - E(S) (reduction) and
dc_decom_analysis.shapley gives the exact Shapley values for the periods 2040,
2024-2040 and 2026-2040, in the world 'all' (six levers) and the world
'no_certificates' (N fixed off, five levers over 32 runs; comparable with v1,
which had no certificates). The full combination is A0 with certificates;
'cap residual' = E(full) - base R-CN, 'cap effect without certificates' =
E(full with N off) - no-certificate R-CN of sens_certificates.csv.

LMDI supplement (dc_decom_analysis.lmdi): the empty and the full combination
re-solved with store_slices=True (yearly ``between``) and the base REF and
R-CN runs of part base (``vs_frozen_2024`` and ``between``), in the form of
outputs/stage2/wedge.csv. With emissions near zero after 2026, LMDI
replaces zeros by 1e-12 and assigns most of the change to the grid-share
factor.
"""
from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from dc_decom_analysis import lmdi, shapley

from .. import accounting, runner
from ..config import Paths
from ..model import Result
from ..scenarios import default_paths, default_policy
from . import base as B
from . import common as C

FILES = ("emission_levers.csv", "emission_shapley.csv", "emission_lmdi.csv",
         "emission_summary.json", "run_log_emissions.csv")
PART = "emissions"
LEVERS = ("L", "R", "G", "D", "N", "P")
NAMES = {"L": "liquid cooling", "R": "RE procurement", "G": "geographic allocation",
         "D": "grid decarbonization", "N": "certificates", "P": "carbon price"}
PERIODS = ("2040", "cum_2024_2040", "cum_2026_2040")
METRIC = {"2040": "co2_2040_Mt", "cum_2024_2040": "co2_cum_2024_2040_Mt",
          "cum_2026_2040": "co2_cum_2026_2040_Mt"}
WORLDS = {"all": LEVERS, "no_certificates": ("L", "R", "G", "D", "P")}
EF_2024 = 0.423
EFF_TOL = 1e-9


def bits(on: Dict[str, bool]) -> str:
    return "".join("1" if on[k] else "0" for k in LEVERS)


def combinations() -> List[Dict[str, bool]]:
    return [dict(zip(LEVERS, b)) for b in product((False, True), repeat=len(LEVERS))]


def spec(on: Dict[str, bool], store_slices: bool = False) -> runner.RunSpec:
    scen = "A0" if on["G"] else "REF"
    ps = C.params()
    ov: Dict[str, float] = {}
    if not on["R"]:
        ov.update({"ppa_tier1_share": 0.0, "ppa_tier2_share": 0.0, "Deploy_max_PV": 0.0})
    if not on["D"]:
        ov.update({f"ef_national@{t}": EF_2024 for t in C.YEARS})
    ps_ov = ps.with_overrides(ov) if ov else ps
    paths: Optional[Paths] = None
    if not on["P"]:
        dp = default_paths(ps)
        paths = Paths(carbon_price={t: 0.0 for t in dp.carbon_price},
                      demand_TWh=dict(dp.demand_TWh), liquid_deploy_cap=dict(dp.liquid_deploy_cap))
    policy = None
    if not on["L"]:
        policy = default_policy(scen, ps_ov).with_(
            liquid_fleet_max={r: 0.0 for r in ps.region_names})
    options = C.BASE_OPTIONS if on["N"] else C.BASE_OPTIONS.with_(certificates=False)
    return runner.RunSpec(scenario=scen, existing_dir=C.V2, options=options,
                          overrides=tuple(sorted(ov.items())), paths=paths, policy=policy,
                          store_slices=store_slices, label=f"{PART}:{bits(on)}")


def emission_values(r: Result) -> Dict[str, float]:
    if r.status != "Optimal":
        return {METRIC[p]: math.nan for p in PERIODS} | {"eq1_total_TJPY": math.nan}
    nat = r.tables["national"]
    return {"co2_2040_Mt": float(r.metrics["co2_2040_Mt"]),
            "co2_cum_2024_2040_Mt": float(r.metrics["co2_cum_Mt"]),
            "co2_cum_2026_2040_Mt": float(nat.loc[nat["year"] >= 2026, "co2_Mt"].sum()),
            "eq1_total_TJPY": float(r.metrics["eq1_total_yen"]) / C.TJ}


def game(E: Dict[str, float], world: str) -> Dict[frozenset, float]:
    """v(S) = E(empty) - E(S) over the levers of ``world`` (N off in 'no_certificates')."""
    players = WORLDS[world]
    fixed = {k: False for k in LEVERS if k not in players}
    e0 = E[bits({k: False for k in LEVERS})]
    out: Dict[frozenset, float] = {}
    for b in product((False, True), repeat=len(players)):
        on = dict(zip(players, b)) | fixed
        out[frozenset(k for k in players if on[k])] = e0 - E[bits(on)]
    return out


def _lmdi_rows(comparison: str, parts: Sequence[Dict[str, float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for period, d in (("2040", parts[-1]), ("cumulative", lmdi.cumulative(parts))):
        for f in lmdi.FACTORS + ("cgs", "total", "exact_delta", "residual"):
            rows.append({"comparison": comparison, "period": period, "factor": f,
                         "value_Mt": d[f],
                         "share_of_total": d[f] / d["total"] if abs(d["total"]) > 1e-6
                         else math.nan})
    return rows


def _table(r: Result) -> pd.DataFrame:
    return accounting.analysis_frames(r.flows, r.meta, basis="eq1")[0]


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    combos = combinations()
    res = C.solve([spec(on) for on in combos], PART, use_cache)
    rows: List[Dict[str, Any]] = []
    E: Dict[str, Dict[str, float]] = {p: {} for p in PERIODS}
    for on, r in zip(combos, res):
        v = emission_values(r)
        b = bits(on)
        for p in PERIODS:
            E[p][b] = v[METRIC[p]]
        rows.append({"bits": b, **{k: int(on[k]) for k in LEVERS},
                     "scenario": "A0" if on["G"] else "REF", **v, "status": r.status})
    C.write_csv(pd.DataFrame(rows), "emission_levers.csv", out)

    srows: List[Dict[str, Any]] = []
    summ: Dict[str, Any] = {"levers": {k: NAMES[k] for k in LEVERS},
                            "n_runs": len(combos),
                            "n_optimal": int(sum(r.status == "Optimal" for r in res)),
                            "efficiency_tol_Mt": EFF_TOL, "worlds": {}}
    empty = bits({k: False for k in LEVERS})
    full = bits({k: True for k in LEVERS})
    full_nc = bits({k: k != "N" for k in LEVERS})
    for world, players in WORLDS.items():
        wsum: Dict[str, Any] = {}
        for p in PERIODS:
            g = game(E[p], world)
            phi = shapley.shapley(g, players)
            on1 = shapley.oat_on(g, players)
            off1 = shapley.oat_off(g, players)
            total = g[frozenset(players)]
            for k in players:
                srows.append({"world": world, "period": p, "lever": k, "name": NAMES[k],
                              "shapley_Mt": phi[k],
                              "share": phi[k] / total if total != 0 else math.nan,
                              "oat_on_Mt": on1[k], "oat_off_Mt": off1[k]})
            err = abs(sum(phi.values()) - total)
            top = full if world == "all" else full_nc
            wsum[p] = {"E_empty_Mt": E[p][empty], "E_full_Mt": E[p][top],
                       "reduction_Mt": total, "efficiency_error_Mt": err,
                       "efficiency_ok": bool(err <= EFF_TOL),
                       "shapley_Mt": phi, "share": {k: phi[k] / total if total != 0
                                                    else math.nan for k in players}}
        summ["worlds"][world] = wsum
    C.write_csv(pd.DataFrame(srows), "emission_shapley.csv", out)

    # committed outputs of parts base and certificates
    bm = pd.read_csv(C.OUT / "base_metrics.csv").set_index("scenario")
    sc = pd.read_csv(C.OUT / "sens_certificates.csv").set_index("case")
    rcn = float(bm.loc["R-CN", "co2_cum_Mt"])
    rcn_nc = float(sc.loc["no_certificates", "co2_cum_R-CN_Mt"])
    e_full = E["cum_2024_2040"][full]
    e_full_nc = E["cum_2024_2040"][full_nc]
    summ["cap_residual"] = {
        "definition": "E(full) (= A0 with certificates) - base R-CN co2_cum (base_metrics.csv)",
        "E_full_Mt": e_full, "R-CN_Mt": rcn, "value_Mt": e_full - rcn}
    summ["cap_effect_without_certificates"] = {
        "definition": "E(full with N off) (= A0 without certificates) - no-certificate R-CN "
                      "co2_cum (sens_certificates.csv)",
        "E_full_N_off_Mt": e_full_nc, "R-CN_no_certificates_Mt": rcn_nc,
        "value_Mt": e_full_nc - rcn_nc}
    summ["efficiency_all_ok"] = all(summ["worlds"][w][p]["efficiency_ok"]
                                    for w in WORLDS for p in PERIODS)

    # LMDI supplement
    on_empty = {k: False for k in LEVERS}
    on_full = {k: True for k in LEVERS}
    base_sp = {s.scenario: s for s in B.specs(C.load_config())}
    lm_specs = [spec(on_empty, store_slices=True), spec(on_full, store_slices=True),
                base_sp["REF"], base_sp["R-CN"]]
    r_empty, r_full, r_ref, r_rcn = C.solve(lm_specs, PART, use_cache)
    t_empty, t_full, t_ref, t_rcn = (_table(x) for x in (r_empty, r_full, r_ref, r_rcn))
    lrows: List[Dict[str, Any]] = []
    lrows += _lmdi_rows("full_vs_empty", [lmdi.between(t_empty, t_full, t) for t in C.YEARS])
    lrows += _lmdi_rows("R-CN_vs_frozen2024", [lmdi.vs_frozen_2024(t_rcn, t) for t in C.YEARS])
    lrows += _lmdi_rows("REF_vs_frozen2024", [lmdi.vs_frozen_2024(t_ref, t) for t in C.YEARS])
    lrows += _lmdi_rows("R-CN_vs_REF", [lmdi.between(t_ref, t_rcn, t) for t in C.YEARS])
    C.write_csv(pd.DataFrame(lrows), "emission_lmdi.csv", out)
    summ["lmdi_same_as_levers"] = {
        "empty_cum_Mt_diff": float(r_empty.metrics["co2_cum_Mt"]) - E["cum_2024_2040"][empty],
        "full_cum_Mt_diff": float(r_full.metrics["co2_cum_Mt"]) - e_full}
    C.write_json(summ, "emission_summary.json", out)
    C.write_run_log(PART, out)
    return C.PartResult(files=list(FILES), summary=summ)
