"""Spec version runs used by the v2 experiments (outputs/stage2/).

Usage:
    .venv/bin/python src/scripts/run_spec.py [--workers 8]

The spec version (``dc_decom_spec``) is the Energies Section 3 formulation
with the implementation errors E1-E8 fixed. Two configurations are solved:
'primary' (default readings ``SpecReadings()``) and 'paper_only'
(``PAPER_ONLY_READINGS``: every code-only bound that the paper does not state
is removed). Both readings were declared before any spec run; nothing is
tuned to the reported values of Energies.

Writes
  paper_only.csv  REF / C-CN / R-CN of 'paper_only': Eq. 1 totals and cost
                  lines, curtailment, emissions, the C-CN -> R-CN advantage
                  and the principled channel shares (reference point of the
                  bridge and of the 2x2, src/dc_decom_v2/experiments/)
  breakeven.csv   non-metro adder (yen per kW of new non-metro IT capacity)
                  at which the C-CN -> R-CN advantage vanishes, both
                  configurations (bisection, tolerance 100 yen/kW; reference
                  row of the v2 breakeven)
Output is deterministic (CBC threads=1, sorted rows, fixed float format).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from dc_decom_analysis import decomposition, tables  # noqa: E402
from dc_decom_spec import scenarios as spec_scenarios  # noqa: E402
from dc_decom_spec.config import PAPER_ONLY_READINGS, SpecFixes, SpecReadings  # noqa: E402
from dc_decom_spec.load_inputs import load_inputs  # noqa: E402
from dc_decom_spec.model import build_and_solve  # noqa: E402

OUT = REPO / "outputs" / "stage2"
SCEN = ("REF", "C-CN", "R-CN")
FLOAT_FMT = "%.10g"
FILES = ("paper_only.csv", "breakeven.csv")

PRIMARY_R = SpecReadings()
PAPER_R = PAPER_ONLY_READINGS
CONFIG_READINGS = {"primary": PRIMARY_R, "paper_only": PAPER_R}


@dataclass(frozen=True)
class Job:
    scenario: str
    fixes: SpecFixes = SpecFixes()
    readings: SpecReadings = SpecReadings()
    overrides: Tuple[Tuple[str, Any], ...] = ()
    detail: bool = True       # return the per-(year, region) table


_INPUTS: Dict[bool, Any] = {}


def _inputs(table9: bool):
    if table9 not in _INPUTS:
        _INPUTS[table9] = load_inputs(table9)
    return _INPUTS[table9]


def solve(job: Job) -> Dict[str, Any]:
    regions, p, temps = _inputs(job.fixes.table9_inputs)
    res = build_and_solve(spec_scenarios.ALL_SCENARIOS[job.scenario], regions, p, temps,
                          fixes=job.fixes, readings=job.readings,
                          overrides=dict(job.overrides))
    return {
        "status": res.status,
        "summary": tables.summary(res),
        "components": res.components,
        "table": res.table if job.detail else None,
    }


def _ov(**kw) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(kw.items()))


def _write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, float_format=FLOAT_FMT, lineterminator="\n")


def set_metrics(runs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Advantage and principled channel shares of one configuration."""
    totals = {s: runs[s]["summary"]["total_yen"] for s in SCEN}
    split = decomposition.principled(pd.DataFrame(runs["C-CN"]["components"]),
                                     pd.DataFrame(runs["R-CN"]["components"]))
    ccn, rcn = float(totals["C-CN"]), float(totals["R-CN"])
    return {"advantage": (ccn - rcn) / ccn,
            "delta_TJPY": (totals["C-CN"] - totals["R-CN"]) / 1e12,
            "shares": decomposition.shares(split)}


def metric_rows(label_cols: Dict[str, Any], runs: Mapping[str, Mapping[str, Any]],
                m: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for s in SCEN:
        sm = runs[s]["summary"]
        row = dict(label_cols)
        row.update({
            "scenario": s, "status": sm["status"],
            "total_TJPY": sm["total_TJPY"],
            "capex_TJPY": sm["capex_TJPY"], "opex_TJPY": sm["opex_TJPY"],
            "energy_TJPY": sm["energy_TJPY"], "carbon_TJPY": sm["carbon_TJPY"],
            "curtail_TJPY": sm["curtail_TJPY"], "subsidy_TJPY": sm["subsidy_TJPY"],
            "resource_cost_TJPY": sm["resource_cost_TJPY"],
            "curtail_TWh": sm["curtail_TWh"],
            "emissions_2024_Mt": sm["emissions_2024_Mt"],
            "emissions_2040_Mt": sm["emissions_2040_Mt"],
            "emissions_cum_Mt": sm["emissions_cum_Mt"],
            "advantage": m["advantage"], "delta_TJPY": m["delta_TJPY"],
            **{f"share_{k}": m["shares"][k] for k in decomposition.CHANNELS},
        })
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(8, args.workers))
    pool = ProcessPoolExecutor(max_workers=workers)

    def run(jobs: Sequence[Job]) -> List[Dict[str, Any]]:
        return list(pool.map(solve, jobs, chunksize=max(1, len(jobs) // (4 * workers))))

    # paper_only
    runs = dict(zip(SCEN, run([Job(s, SpecFixes(), PAPER_R) for s in SCEN])))
    rows = metric_rows({"config": "paper_only",
                        "readings": "ppa_ceiling=none; pv_onsite_cap=none; cgs_energy_cap=none; "
                                    "cgs_cf=none; bess_cycles=none; siting_premium=1.00"},
                       runs, set_metrics(runs))
    _write_csv(pd.DataFrame(rows), "paper_only.csv")

    def tot(r: Mapping[str, Any]) -> float:
        return r["summary"]["total_yen"]

    # Breakeven of a non-metro adder (bisection, tolerance 100 yen/kW)
    rows = []
    for cfg, rd in CONFIG_READINGS.items():
        def gap(a: float) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
            c, r = run([Job("C-CN", SpecFixes(), rd, _ov(nonmetro_adder_yen_per_kW=a), detail=False),
                        Job("R-CN", SpecFixes(), rd, _ov(nonmetro_adder_yen_per_kW=a), detail=False)])
            return tot(c) - tot(r), c, r
        g0, c0, r0 = gap(0.0)
        g1, c1, r1 = gap(1000.0)
        slope = (g1 - g0) / 1000.0   # yen per (yen/kW)
        n_iter = 0
        if g0 <= 0:
            status, a_lo, a_hi = "no_advantage_at_zero", float("nan"), float("nan")
        elif slope >= 0:
            status, a_lo, a_hi = "adder_does_not_reduce_gap", float("nan"), float("nan")
        else:
            a_lo, a_hi = 0.0, 1000.0
            g_hi = g1
            while g_hi > 0 and a_hi < 1e9:
                a_lo, a_hi = a_hi, a_hi * 2.0
                g_hi = gap(a_hi)[0]
                n_iter += 1
            while a_hi - a_lo > 100.0:
                mid = 0.5 * (a_lo + a_hi)
                if gap(mid)[0] > 0:
                    a_lo = mid
                else:
                    a_hi = mid
                n_iter += 1
            status = "found" if g_hi <= 0 else "not_found_below_1e9"
        rows.append({"config": cfg, "status": status,
                     "gap_at_zero_TJPY": g0 / 1e12, "advantage_at_zero": g0 / tot(c0),
                     "C-CN_TJPY": tot(c0) / 1e12, "R-CN_TJPY": tot(r0) / 1e12,
                     "curtail_TWh_C-CN": c0["summary"]["curtail_TWh"],
                     "curtail_TWh_R-CN": r0["summary"]["curtail_TWh"],
                     "gap_slope_TJPY_per_1000yen_per_kW": slope * 1000.0 / 1e12,
                     "breakeven_lo_yen_per_kW": a_lo, "breakeven_hi_yen_per_kW": a_hi,
                     "breakeven_yen_per_kW": 0.5 * (a_lo + a_hi),
                     "closed_form_yen_per_kW": (-g0 / slope) if slope < 0 else float("nan"),
                     "iterations": n_iter, "tolerance_yen_per_kW": 100.0})
    _write_csv(pd.DataFrame(rows), "breakeven.csv")
    pool.shutdown()
    missing = [f for f in FILES if not (OUT / f).is_file()]
    if missing:
        raise RuntimeError(f"missing outputs: {missing}")
    print(f"[run_spec] wrote {', '.join(FILES)} to {OUT} ({time.time() - t_start:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
