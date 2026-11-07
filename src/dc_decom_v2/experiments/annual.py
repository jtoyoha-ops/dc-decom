"""Annual vs 96-hour versions (part annual; config/experiments_v35.yaml, section
``annual``, defined before any run).

Cases: base (certificates) REF, C-CN, R-CN, R-CN-pub, A0 and no_certificates
REF, C-CN, R-CN. The rep96 side uses the RunSpecs of part base and of
sensitivity.certificate_cases (the latter with store_slices=True, which the
channel decomposition needs; slices are not part of the solve); the annual
side changes only ModelOptions.time_mode to 'annual'.

Caveat: annual mode evaluates the PUE at the JMA annual-mean temperature of
each region (one slice of 8,760 h) with the coefficients calibrated on the
hourly evaluation (data/parameters_v2/pue_calibration.json); the N+1 backup
and the peaks then follow the mean temperature too, so annual mode has no
summer peak. The rows pue_ann_air[r] / pue_ann_dlc[r] show the annual PUE
of each region in both modes (identical in every year and scenario).

Output annual_vs_rep96.csv (long): case, scenario, metric, rep96, annual, diff
(annual - rep96), rel_diff (diff / |rep96|, NaN when rep96 is 0).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from dc_decom_analysis import decomposition

from .. import accounting, runner
from . import base as B
from . import common as C
from .common import Setting
from .sensitivity import certificate_cases

FILES = ("annual_vs_rep96.csv", "run_log_annual.csv")
PART = "annual"
CASES = {"base": ("REF", "C-CN", "R-CN", "R-CN-pub", "A0"),
         "no_certificates": ("REF", "C-CN", "R-CN")}
REGS = ("Hokkaido", "Tohoku", "Kanto", "Kansai", "Kyushu")
SCALARS = (("eq1_total_TJPY", "eq1_total_yen", 1.0 / C.TJ),
           ("social_cost_TJPY", "social_cost_yen", 1.0 / C.TJ),
           ("carbon_cost_TJPY", "carbon_cost_yen", 1.0 / C.TJ),
           ("co2_cum_Mt", "co2_cum_Mt", 1.0), ("co2_2040_Mt", "co2_2040_Mt", 1.0),
           ("curtail_TWh", "curtail_TWh", 1.0),
           ("nonmetro_share_2040", "nonmetro_share_last", 1.0),
           ("facility_energy_cum_TWh", "facility_energy_cum_TWh", 1.0),
           ("liquid_share_fleet_2040", "liquid_share_fleet_last", 1.0),
           ("peak_facility_nonmetro_2040_GW", "peak_facility_nonmetro_last_GW", 1.0))


def specs() -> List[Tuple[str, str, str, runner.RunSpec]]:
    """(case, scenario, mode, spec)."""
    out: List[Tuple[str, str, str, runner.RunSpec]] = []
    base = {s.scenario: s for s in B.specs(C.load_config())}
    for sc in CASES["base"]:
        b = base[sc]
        out.append(("base", sc, "rep96", b))
        out.append(("base", sc, "annual", runner.RunSpec(
            scenario=sc, existing_dir=b.existing_dir, options=b.options.with_(time_mode="annual"),
            store_slices=True, label=f"{PART}:base:{sc}:annual")))
    nc = {cid: s for cid, _, s in certificate_cases()}["no_certificates"]
    for mode in ("rep96", "annual"):
        s: Setting = nc if mode == "rep96" else nc.with_(
            options=nc.options.with_(time_mode="annual"))
        for sc in CASES["no_certificates"]:
            out.append(("no_certificates", sc, mode,
                        C.spec_for(s, sc, f"{PART}:no_certificates:{sc}:{mode}", store_slices=True)))
    return out


def _channels(c: Any, r: Any) -> Dict[str, float]:
    comp = {k: accounting.analysis_frames(x.flows, x.meta, basis="eq1")[1] for k, x in
            (("c", c), ("r", r))}
    split = decomposition.principled(comp["c"], comp["r"])
    return {k: split[k] / C.TJ for k in decomposition.CHANNELS + ("total",)}


def values(case: str, res: Dict[str, Any]) -> List[Tuple[str, str, float]]:
    """(scenario, metric, value) of one case in one mode."""
    out: List[Tuple[str, str, float]] = []
    for sc in CASES[case]:
        r = res[sc]
        ok = r.status == "Optimal"
        for name, key, f in SCALARS:
            out.append((sc, name, float(r.metrics[key]) * f if ok else math.nan))
    tc = C.total(res["C-CN"])
    for rc in [x for x in ("R-CN", "R-CN-pub") if x in res]:
        out.append((rc, "advantage", C.adv(tc, C.total(res[rc]))))
        if res["C-CN"].status == "Optimal" and res[rc].status == "Optimal":
            for ch, v in _channels(res["C-CN"], res[rc]).items():
                out.append((f"C-CN_minus_{rc}", f"channel_{ch}_TJPY", v))
    ry = res["REF"].tables["region_year"]
    last = ry[ry["year"] == C.YEARS[-1]].set_index("region")
    for kind in ("air", "dlc"):
        for g in REGS:
            out.append(("REF", f"pue_ann_{kind}[{g}]", float(last.loc[g, f"pue_ann_{kind}"])))
    return out


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    sp = specs()
    res = C.solve([s for *_, s in sp], PART, use_cache)
    by: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (case, sc, mode, _), r in zip(sp, res):
        by.setdefault((case, mode), {})[sc] = r
    rows: List[Dict[str, Any]] = []
    for case in CASES:
        v96 = values(case, by[(case, "rep96")])
        van = values(case, by[(case, "annual")])
        for (sc, m, a), (sc2, m2, b) in zip(v96, van):
            assert (sc, m) == (sc2, m2)
            diff = b - a
            rows.append({"case": case, "scenario": sc, "metric": m, "rep96": a, "annual": b,
                         "diff": diff, "rel_diff": diff / abs(a) if a != 0 else math.nan})
    C.write_csv(pd.DataFrame(rows), "annual_vs_rep96.csv", out)
    C.write_run_log(PART, out)
    return C.PartResult(files=list(FILES))
