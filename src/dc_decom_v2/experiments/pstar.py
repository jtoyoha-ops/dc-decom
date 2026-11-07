"""P*: minimum flat carbon price at which A0 reaches the R-CN base 2040
non-metro IT share (config/experiments_v2.yaml, pstar)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .. import runner
from . import common as C
from .common import Setting
from .sensitivity import sweep_cases

FILES = ("pstar.json", "pstar_curve.csv", "run_log_pstar.csv")
TOL = 100.0


def _a0(price: float, use_cache: bool) -> Any:
    ps = C.params()
    s = Setting(roles=(("A0", "A0"),), carbon=C.flat_carbon(ps, price))
    return C.solve([C.spec_for(s, "A0", f"pstar:A0:{price:g}")], "pstar", use_cache)[0]


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    rcn = C.solve([runner.RunSpec(scenario="R-CN", existing_dir=C.V2, options=C.BASE_OPTIONS,
                                  store_slices=True, label="base:R-CN")], "pstar", use_cache)[0]
    target = float(rcn.metrics["nonmetro_share_last"])
    curve: List[Dict[str, Any]] = []
    specs = [C.spec_for(s, "A0", f"pstar:A0:{p}") for p, s in sweep_cases()]
    for (p, _), r in zip(sweep_cases(), C.solve(specs, "pstar", use_cache)):
        curve.append({"price_yen_per_t": float(p), "nonmetro_2040_A0": r.metrics["nonmetro_share_last"],
                      "status": r.status, "source": "sweep"})
    reached = [c for c in curve if c["nonmetro_2040_A0"] >= target - 1e-9]
    result: Dict[str, Any] = {"target_share": target, "tolerance_yen_per_t": TOL}
    if not reached:
        result.update(status="not reached", pstar_yen_per_t=None)
    elif reached[0]["price_yen_per_t"] == 0.0:
        result.update(status="reached at 0", pstar_yen_per_t=0.0)
    else:
        hi = reached[0]["price_yen_per_t"]
        lo = hi - 1000.0
        while hi - lo > TOL:
            mid = 0.5 * (lo + hi)
            r = _a0(mid, use_cache)
            sh = float(r.metrics["nonmetro_share_last"])
            curve.append({"price_yen_per_t": mid, "nonmetro_2040_A0": sh, "status": r.status,
                          "source": "bisection"})
            if sh >= target - 1e-9:
                hi = mid
            else:
                lo = mid
        result.update(status="found", pstar_yen_per_t=hi, bracket=[lo, hi])
    at10k = [c for c in curve if c["price_yen_per_t"] == 10000.0][0]
    result["nonmetro_2040_A0_at_10000"] = at10k["nonmetro_2040_A0"]
    result["nonmetro_2040_A0_at_0"] = curve[0]["nonmetro_2040_A0"]
    result["ch07_design_assumption_yen_per_t"] = 10000.0
    C.write_json(result, "pstar.json", out)
    df = pd.DataFrame(curve).sort_values(["price_yen_per_t", "source"]).reset_index(drop=True)
    C.write_csv(df, "pstar_curve.csv", out)
    C.write_run_log("pstar", out)
    return C.PartResult(files=list(FILES), summary=result)
