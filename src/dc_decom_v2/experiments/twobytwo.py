"""Structure x parameters 2x2 with the five cost channels (part twobytwo;
config/experiments_v35.yaml, section ``twobytwo``, defined before any run).

Corners (structure, parameters):
    VV  v2, v2        Setting() (= base = bridge B0)
    VP  v2, paper     Setting(existing=data/parameters_spec, structural=data/parameters_v2)
                      (bridge P3-P6 only), endogenous R-CN
    SV  spec, v2      R-CN -> R-CN-pub, grid_constraints False, time_mode annual,
                      PPA without the curve (ppa_tier1_share 1000, ppa_tier2_share 0,
                      ppa_volume_factor@t 1), backup_techs ('cgs',), it_energy
                      legacy_util, certificates False (bridge P1, P2, P7-P11 only)
    SP  spec, paper   bridge P11 (all steps)
REF, C-CN and R-CN (R-CN-pub in the S corners) are solved with
store_slices=True; the channels come from decomposition.principled on
accounting.analysis_frames(basis='eq1').
Effects: extension (VP - SP with paper parameters, VV - SV with v2
parameters), parameter update (SV - SP in the spec structure, VV - VP in the
v2 structure), interaction VV - VP - SV + SP.
Reference: spec paper_only (outputs/stage2/paper_only.csv) and SP - spec
(re-implementation difference).
Checks: VV = bridge_steps.csv B0, SP = P11 (relative 1e-9); the five channels
of each corner sum to C-CN - R-CN (1e-6 TJPY).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from dc_decom_analysis import decomposition

from .. import accounting
from ..model import Result
from . import common as C
from .bridge import steps_paper_only
from .common import Setting, as_tuple

FILES = ("decomposition_2x2.csv", "decomposition_2x2_effects.csv", "run_log_2x2.csv")
PART = "2x2"          # label prefix and run log name (registered: run_log_2x2.csv)
ROLES = ("REF", "C-CN", "R-CN")
CORNERS = ("VV", "VP", "SV", "SP")
AXES = {"VV": ("v2", "v2"), "VP": ("v2", "paper"), "SV": ("spec", "v2"), "SP": ("spec", "paper")}
EFFECTS = {"ext_paper": (("VP", 1), ("SP", -1)), "ext_v2": (("VV", 1), ("SV", -1)),
           "upd_spec": (("SV", 1), ("SP", -1)), "upd_v2": (("VV", 1), ("VP", -1)),
           "interaction": (("VV", 1), ("VP", -1), ("SV", -1), ("SP", 1))}
BRIDGE_OF = {"VV": "B0", "SP": "P11"}
REL_TOL = 1e-9
CH_TOL = 1e-6


def corners() -> Dict[str, Setting]:
    b = Setting()
    ppa = {"ppa_tier1_share": 1000.0, "ppa_tier2_share": 0.0}
    ppa.update({f"ppa_volume_factor@{t}": 1.0 for t in C.YEARS})
    sv = b.with_(roles=(("REF", "REF"), ("C-CN", "C-CN"), ("R-CN", "R-CN-pub")),
                 options=b.options.with_(grid_constraints=False, time_mode="annual",
                                         backup_techs=("cgs",), it_energy="legacy_util",
                                         certificates=False),
                 overrides=as_tuple(ppa))
    return {"VV": b, "VP": b.with_(existing=C.SPEC, structural=C.V2), "SV": sv,
            "SP": steps_paper_only()[-1][1]}


def _channels(c: Result, r: Result) -> Dict[str, float]:
    comp = {k: accounting.analysis_frames(x.flows, x.meta, basis="eq1")[1]
            for k, x in (("c", c), ("r", r))}
    split = decomposition.principled(comp["c"], comp["r"])
    return {k: split[k] / C.TJ for k in decomposition.CHANNELS + ("total",)}


def corner_values(res: Dict[str, Result]) -> Dict[str, float]:
    tot = {role: C.total(res[role]) / C.TJ for role in ROLES}
    a = C.adv(tot["C-CN"], tot["R-CN"])
    v: Dict[str, float] = {"advantage": a, "advantage_pt": 100.0 * a}
    for role in ROLES:
        v[f"total_{role}_TJPY"] = tot[role]
    v["diff_TJPY"] = tot["C-CN"] - tot["R-CN"]
    ok = res["C-CN"].status == "Optimal" and res["R-CN"].status == "Optimal"
    ch = _channels(res["C-CN"], res["R-CN"]) if ok else {}
    for k in decomposition.CHANNELS:
        v[f"channel_{k}_TJPY"] = ch.get(k, math.nan)
    v["channel_total_TJPY"] = ch.get("total", math.nan)
    for k in decomposition.CHANNELS:
        t = v["channel_total_TJPY"]
        v[f"share_{k}"] = v[f"channel_{k}_TJPY"] / t if t else math.nan
    return v


UNITS = {"advantage": "fraction", "advantage_pt": "percentage points"}


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    cs = corners()
    keys: List[Tuple[str, str]] = [(k, role) for k in CORNERS for role in ROLES]
    specs = [C.spec_for(cs[k], role, f"{PART}:{k}:{role}", store_slices=True) for k, role in keys]
    solved = dict(zip(keys, C.solve(specs, PART, use_cache)))
    res = {k: {role: solved[(k, role)] for role in ROLES} for k in CORNERS}

    rows: List[Dict[str, Any]] = []
    for k in CORNERS:
        for role in ROLES:
            r = res[k][role]
            ok = r.status == "Optimal"
            rows.append({"corner": k, "structure": AXES[k][0], "parameters": AXES[k][1],
                         "role": role, "scenario": cs[k].role(role), "status": r.status,
                         "total_TJPY": C.total(r) / C.TJ,
                         "co2_cum_Mt": r.metrics["co2_cum_Mt"] if ok else math.nan,
                         "nonmetro_share_2040": r.metrics["nonmetro_share_last"] if ok
                         else math.nan,
                         "curtail_TWh": r.metrics["curtail_TWh"] if ok else math.nan})
    C.write_csv(pd.DataFrame(rows), "decomposition_2x2.csv", out)

    val = {k: corner_values(res[k]) for k in CORNERS}
    erows: List[Dict[str, Any]] = []
    for q in val["VV"]:
        row: Dict[str, Any] = {"kind": "effect", "quantity": q,
                               "unit": UNITS.get(q, "share of C-CN - R-CN" if q.startswith("share_")
                                                 else "TJPY")}
        row.update({k: val[k][q] for k in CORNERS})
        for e, terms in EFFECTS.items():
            row[e] = sum(sgn * val[k][q] for k, sgn in terms)
        row["value"] = math.nan
        erows.append(row)

    def extra(kind: str, q: str, unit: str, value: float, ok: Any = None) -> None:
        erows.append({"kind": kind, "quantity": q, "unit": unit, "value": value,
                      **({"ok": ok} if ok is not None else {})})

    po = pd.read_csv(C.REPO / "outputs" / "stage2" / "paper_only.csv").set_index("scenario")
    tc = float(po.loc["C-CN", "total_TJPY"])
    po_adv = (tc - float(po.loc["R-CN", "total_TJPY"])) / tc
    po_adv_ex = (tc - float(po.loc["R-CN", "resource_cost_TJPY"])) / tc
    sp_adv = val["SP"]["advantage"]
    extra("reference", "spec_paper_only_advantage", "fraction", po_adv)
    extra("reference", "spec_paper_only_ex_subsidy_advantage", "fraction", po_adv_ex)
    extra("reference", "reimplementation_diff_pt (SP - spec paper_only)", "percentage points",
          100.0 * (sp_adv - po_adv))
    extra("reference", "reimplementation_diff_ex_subsidy_pt (SP - spec ex subsidy)",
          "percentage points", 100.0 * (sp_adv - po_adv_ex))
    for ch in decomposition.CHANNELS:
        s = float(po.loc["R-CN", f"share_{ch}"])
        extra("reference", f"spec_paper_only_share_{ch}", "share", s)
        extra("reference", f"reimplementation_diff_share_{ch} (SP - spec)", "share",
              val["SP"][f"share_{ch}"] - s)

    bs = pd.read_csv(C.OUT / "bridge_steps.csv")
    bs = bs[bs["chain"] == "to_paper_only"].set_index("step")
    checks: Dict[str, bool] = {}
    for k, step in BRIDGE_OF.items():
        worst = 0.0
        for col, mine in [(f"total_{role}_TJPY", val[k][f"total_{role}_TJPY"]) for role in ROLES] \
                + [("advantage", val[k]["advantage"])]:
            ref = float(bs.loc[step, col])
            worst = max(worst, abs(mine - ref) / abs(ref))
        checks[f"{k}_vs_{step}"] = worst <= REL_TOL
        extra("check", f"{k}_vs_bridge_{step}_max_rel_diff", "relative", worst, worst <= REL_TOL)
    for k in CORNERS:
        err = abs(val[k]["channel_total_TJPY"] - val[k]["diff_TJPY"])
        checks[f"channels_{k}"] = err <= CH_TOL
        extra("check", f"channel_sum_error_{k}", "TJPY", err, err <= CH_TOL)
    C.write_csv(pd.DataFrame(erows), "decomposition_2x2_effects.csv", out)
    C.write_run_log(PART, out)
    return C.PartResult(files=list(FILES), summary={"checks": checks})
