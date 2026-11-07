"""(c) Minimal sensitivity: tornado and carbon-price sweep (config/experiments_v2.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from . import common as C
from .bridge import metrics, solve_settings
from .common import Setting, as_tuple

TORNADO_FILES = ("tornado.csv", "run_log_tornado.csv")
SWEEP_FILES = ("carbon_sweep.csv", "run_log_sweep.csv")
CERT_FILES = ("sens_certificates.csv", "run_log_certificates.csv")
ROLES = ("C-CN", "R-CN")


def tornado_cases() -> List[Tuple[str, str, str, Setting]]:
    ps = C.params()
    b = Setting()
    g = ps.get("reinforcement_cost_yen_per_kW")
    out = [
        ("base", "base", "base", b),
        ("carbon_price", "low", "path low", b.with_(carbon=as_tuple(ps.path("carbon_price", "low")))),
        ("carbon_price", "high", "path high",
         b.with_(carbon=as_tuple(ps.path("carbon_price", "high")))),
        ("hw_efficiency", "low", "eta 0.20", b.with_(eta=0.20)),
        ("hw_efficiency", "high", "eta 0.40", b.with_(eta=0.40)),
        ("ppa_price", "low", "regional low",
         b.with_overrides(C.region_variant(ps, "ppa_yen_per_kWh", "low"))),
        ("ppa_price", "high", "regional high",
         b.with_overrides(C.region_variant(ps, "ppa_yen_per_kWh", "high"))),
        ("discount_rate", "low", "0.02", b.with_overrides({"discount_rate": 0.02})),
        ("discount_rate", "high", "0.04", b.with_overrides({"discount_rate": 0.04})),
        ("reinforcement_cost", "low", f"{g.low:.0f}",
         b.with_overrides({"reinforcement_cost_yen_per_kW": g.low})),
        ("reinforcement_cost", "high", f"{g.high:.0f}",
         b.with_overrides({"reinforcement_cost_yen_per_kW": g.high})),
        ("siting_cost_difference", "high", "metro shell premium 100000",
         b.with_overrides({f"shell_premium_yen_per_kW[{r}]":
                           ps.get("shell_premium_yen_per_kW", r).high for r in ("Kanto", "Kansai")})),
    ]
    for lvl, val in (("low", g.low), ("base", g.value), ("high", g.high)):
        out.append(("with_reinforcement", lvl, f"g {val:.0f}", b.with_(reinforce_g=val)))
    return out


def run_tornado(out: Path, use_cache: bool = True) -> C.PartResult:
    cases = tornado_cases()
    res = solve_settings([(f"{p}:{lv}", s) for p, lv, _, s in cases], ROLES, "tornado", use_cache)
    rows: List[Dict[str, Any]] = []
    base = None
    for p, lv, label, s in cases:
        m = metrics(f"{p}:{lv}", s, res, ROLES)
        if p == "base":
            base = m
        assert base is not None
        rows.append({"parameter": p, "level": lv, "value": label, **m,
                     "R-CN_change": m["total_R-CN_TJPY"] / base["total_R-CN_TJPY"] - 1.0,
                     "advantage_change": m["advantage"] - base["advantage"]})
    C.write_csv(pd.DataFrame(rows), "tornado.csv", out)
    C.write_run_log("tornado", out)
    return C.PartResult(files=list(TORNADO_FILES))


def sweep_prices() -> List[int]:
    sw = C.load_config()["carbon_sweep"]
    assert sw["scenarios"] == ["C-CN", "R-CN", "A0"]
    return list(range(0, 30001, 1000))


def sweep_cases() -> List[Tuple[int, Setting]]:
    ps = C.params()
    return [(p, Setting(roles=(("C-CN", "C-CN"), ("R-CN", "R-CN"), ("A0", "A0")),
                        carbon=C.flat_carbon(ps, p))) for p in sweep_prices()]


def run_sweep(out: Path, use_cache: bool = True) -> C.PartResult:
    cases = sweep_cases()
    res = solve_settings([(str(p), s) for p, s in cases], ("C-CN", "R-CN", "A0"), "sweep",
                         use_cache)
    rows = []
    for p, s in cases:
        m = metrics(str(p), s, res, ROLES)
        a0 = res[(str(p), "A0")]
        ok = a0.status == "Optimal"
        rows.append({"price_yen_per_t": p, **m,
                     "status_A0": a0.status, "total_A0_TJPY": C.total(a0) / C.TJ,
                     "operator_A0_TJPY": a0.metrics["operator_cost_yen"] / C.TJ if ok else None,
                     "nonmetro_2040_A0": a0.metrics["nonmetro_share_last"] if ok else None,
                     "co2_2040_A0_Mt": a0.metrics["co2_2040_Mt"] if ok else None,
                     "co2_cum_A0_Mt": a0.metrics["co2_cum_Mt"] if ok else None,
                     "liquid_share_fleet_2040_A0": (a0.metrics["liquid_share_fleet_last"]
                                                    if ok else None)})
    C.write_csv(pd.DataFrame(rows), "carbon_sweep.csv", out)
    C.write_run_log("sweep", out)
    return C.PartResult(files=list(SWEEP_FILES))


def certificate_cases() -> List[Tuple[str, str, Setting]]:
    ps = C.params()
    b = Setting()
    hi = ps.get("nfc_yen_per_kWh").high
    return [
        ("base", "certificates 0.4 yen/kWh (base)", b),
        ("nfc_high", f"certificates {hi:g} yen/kWh (sourced high)",
         b.with_overrides({"nfc_yen_per_kWh": hi})),
        ("no_certificates", "physical RE only (no certificates), annual matching",
         b.with_(options=b.options.with_(certificates=False))),
        ("hourly_no_certificates", "24/7: hourly matching, no certificates",
         b.with_(options=b.options.with_(certificates=False, ppa_matching="hourly"))),
    ]


def run_certificates(out: Path, use_cache: bool = True) -> C.PartResult:
    roles = ("REF", "C-CN", "R-CN")
    cases = certificate_cases()
    res = solve_settings([(c, s) for c, _, s in cases], roles, "certificates", use_cache)
    rows = []
    for cid, label, s in cases:
        m = metrics(cid, s, res, roles)
        extra: Dict[str, Any] = {}
        for role in roles:
            r = res[(cid, role)]
            ok = r.status == "Optimal"
            ry = r.tables["region_year"] if ok else None
            extra[f"nfc_cum_{role}_TWh"] = float(ry["nfc_TWh"].sum()) if ok else None
            extra[f"bess_2040_{role}_GWh"] = r.metrics["bess_cap_last_GWh"] if ok else None
            if ok:
                k = ry[(ry["region"] == "Kanto") & (ry["year"] == C.YEARS[-1])]
                extra[f"curtail_kanto_2040_{role}_TWh"] = float(k["curt_TWh"].iloc[0])
        rows.append({"case": cid, "definition": label, **m, **extra})
    C.write_csv(pd.DataFrame(rows), "sens_certificates.csv", out)
    C.write_run_log("certificates", out)
    return C.PartResult(files=list(CERT_FILES))
