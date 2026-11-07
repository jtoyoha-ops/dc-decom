"""Training / inference load ratio (part ch03; config/experiments_v35.yaml,
section ``latency``, defined before any run).

The metro floor of the endogenous R-CN follows the latency rule of
data/parameters_v2/paths.csv (metro_floor, kanto_floor):

    lambda_B(t) = 0.82 x [(1 - a(t)) + a(t) (1 - s_train) (1 - s_inf_tol)]  (2026-2040)
    kappa(t)    = lambda_B(t) x 60 / 82
    2024-2025   base shares (0.82, 0.60)

with a(t) the ai_share path. The grid s_train {0.2, 0.4, 0.6} x s_inf_tol
{0.1, 0.3, 0.5} uses the frozen ai_share value path; two more rows use its low
and high paths at (0.4, 0.3). The base point (0.4, 0.3, value) uses the frozen
floor paths without override; the other points override metro_floor@t and
kanto_floor@t for 2026-2040. Demand is unchanged: ai_share enters the demand
only through the HW transform, which the base (eta 0.30) does not apply.
C-CN (base shares) does not depend on the floors and is solved once.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..params import ParamSet
from . import common as C
from .common import Setting, as_tuple

FILES = ("ch03_latency.csv", "run_log_ch03.csv")
PART = "ch03"
FLOOR_FIRST_YEAR = 2026
BASE_SHARE_METRO, BASE_SHARE_KANTO = 0.82, 0.60
BASE_POINT = (0.4, 0.3, "value")


def floor_paths(ai: Dict[int, float], s_train: float,
                s_inf_tol: float) -> Tuple[Dict[int, float], Dict[int, float]]:
    """(lambda_B, kappa) for every year of ``ai``."""
    lam: Dict[int, float] = {}
    for t, a in ai.items():
        lam[t] = (BASE_SHARE_METRO if t < FLOOR_FIRST_YEAR
                  else 0.82 * ((1.0 - a) + a * (1.0 - s_train) * (1.0 - s_inf_tol)))
    kap = {t: v * BASE_SHARE_KANTO / BASE_SHARE_METRO for t, v in lam.items()}
    return lam, kap


def points(ps: ParamSet) -> List[Tuple[float, float, str]]:
    cfg = C.load_v35()["latency"]
    pts = [(float(st), float(si), "value") for st in cfg["grid"]["s_train"]
           for si in cfg["grid"]["s_inf_tol"]]
    if (BASE_POINT[0], BASE_POINT[1], "value") not in pts:
        raise ValueError("the grid must contain the base point (0.4, 0.3)")
    if (ps.scalar("s_train"), ps.scalar("s_inf_tol")) != BASE_POINT[:2]:
        raise ValueError("base point differs from s_train / s_inf_tol of data/parameters_v2")
    return pts + [(BASE_POINT[0], BASE_POINT[1], "low"), (BASE_POINT[0], BASE_POINT[1], "high")]


def point_setting(ps: ParamSet, s_train: float, s_inf: float, variant: str) -> Setting:
    if (s_train, s_inf, variant) == BASE_POINT:
        return Setting()
    lam, kap = floor_paths(ps.path("ai_share", variant), s_train, s_inf)
    ov: Dict[str, float] = {}
    for t in C.YEARS:
        if t >= FLOOR_FIRST_YEAR:
            ov[f"metro_floor@{t}"] = lam[t]
            ov[f"kanto_floor@{t}"] = kap[t]
    return Setting(overrides=as_tuple(ov))


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    ps = C.params()
    pts = points(ps)
    ccn = C.solve([C.spec_for(Setting(), "C-CN", f"{PART}:C-CN")], PART, use_cache)[0]
    specs = [C.spec_for(point_setting(ps, st, si, v), "R-CN", f"{PART}:{st:g}:{si:g}:{v}:R-CN")
             for st, si, v in pts]
    res = C.solve(specs, PART, use_cache)
    tc = C.total(ccn)
    rows: List[Dict[str, Any]] = []
    for (st, si, v), r in zip(pts, res):
        lam, kap = floor_paths(ps.path("ai_share", v), st, si)
        ok = r.status == "Optimal"
        nat = r.tables["national"] if ok else None
        mean = (float(nat[nat["year"] >= FLOOR_FIRST_YEAR]["nonmetro_share"].mean())
                if nat is not None else float("nan"))
        tr = C.total(r)
        rows.append({
            "s_train": st, "s_inf_tol": si, "ai_share": v,
            "is_base": (st, si, v) == BASE_POINT,
            "lambda_2030": lam[2030], "lambda_2040": lam[C.YEARS[-1]],
            "kappa_2040": kap[C.YEARS[-1]],
            "status_C-CN": ccn.status, "status_R-CN": r.status,
            "total_C-CN_TJPY": tc / C.TJ, "total_R-CN_TJPY": tr / C.TJ,
            "advantage": C.adv(tc, tr),
            "nonmetro_2040_R-CN": r.metrics["nonmetro_share_last"] if ok else float("nan"),
            "nonmetro_mean_2026_2040_R-CN": mean,
            "co2_cum_R-CN_Mt": r.metrics["co2_cum_Mt"] if ok else float("nan"),
            "curtail_R-CN_TWh": r.metrics["curtail_TWh"] if ok else float("nan"),
        })
    C.write_csv(pd.DataFrame(rows), "ch03_latency.csv", out)
    C.write_run_log(PART, out)
    return C.PartResult(files=list(FILES))
