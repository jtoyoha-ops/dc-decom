"""Shared helpers of the stage-3 experiments (config/experiments_v2.yaml).

A ``Setting`` describes one variant of the v2 base: parameter overrides and
additions, the carbon and demand paths, the HW efficiency eta that scales
the AI part of demand, the scenario playing the C-CN and R-CN roles, model
options and policy changes. ``interpolate`` moves the numeric parts of a
setting from the base towards an alternative (conditions analysis).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

from .. import runner
from ..config import (CALIBRATION_YEARS, END_YEAR, METRO, ModelOptions, Paths, PolicySettings,
                      Realization)
from ..model import Result
from ..params import ParamSet
from ..scenarios import default_paths, default_policy

REPO = runner.REPO
OUT = REPO / "outputs" / "stage3"
CONFIG = REPO / "config" / "experiments_v2.yaml"
LOG_DIR = runner.DEFAULT_CACHE / "logs"
V2 = "data/parameters_v2"
SPEC = "data/parameters_spec"
YEARS = list(range(2024, END_YEAR + 1))
FLOAT_FMT = "%.10g"
ETA_BASE = 0.30          # HW efficiency embedded in the base demand path (Table 7)
TJ = 1e12
# Base options after the change of 2026-11-06 (config/experiments_v2.yaml,
# changes_after_registration C1): non-fossil certificates allowed.
BASE_OPTIONS = ModelOptions(certificates=True)


def load_config() -> Dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_v35() -> Dict[str, Any]:
    """config/experiments_v35.yaml (defined before the runs)."""
    return yaml.safe_load((REPO / "config" / "experiments_v35.yaml").read_text(encoding="utf-8"))


def params(existing: str = V2, structural: Optional[str] = None) -> ParamSet:
    return runner.load_params(existing, structural)


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Setting:
    overrides: Tuple[Tuple[str, float], ...] = ()
    additions_by_role: Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...] = ()
    carbon: Optional[Tuple[Tuple[int, float], ...]] = None
    demand: Optional[Tuple[Tuple[int, float], ...]] = None   # before the HW transform
    eta: float = ETA_BASE
    roles: Tuple[Tuple[str, str], ...] = (("REF", "REF"), ("C-CN", "C-CN"), ("R-CN", "R-CN"))
    options: ModelOptions = BASE_OPTIONS
    existing: str = V2
    structural: Optional[str] = None
    liquid_fleet_max_ccn: Tuple[Tuple[str, float], ...] = ()
    reinforce_g: Optional[float] = None        # non-metro c1 from reinforcement_year at cost g
    siting_s2: bool = False                    # post-processed T2 siting cost
    # role -> ((region, year, GW), ...): regional IT fixed by RunSpec.realization (breakeven)
    fixed_itr_by_role: Tuple[Tuple[str, Tuple[Tuple[str, int, float], ...]], ...] = ()

    def role(self, r: str) -> str:
        return dict(self.roles)[r]

    def with_(self, **kw) -> "Setting":
        return replace(self, **kw)

    def with_overrides(self, ov: Dict[str, float]) -> "Setting":
        d = dict(self.overrides)
        d.update(ov)
        return replace(self, overrides=tuple(sorted(d.items())))


def as_tuple(d: Dict[Any, float]) -> Tuple[Tuple[Any, float], ...]:
    return tuple(sorted((k, float(v)) for k, v in d.items()))


def hw_demand(ps: ParamSet, D: Dict[int, float], eta: float) -> Tuple[Dict[int, float], bool]:
    """demand(t) = D(t) [(1 - a) + a ((1 + 0.30) / (1 + eta))^(t - 2024)] with the
    running maximum if the result decreases (flag True)."""
    a = ps.path("ai_share")
    out: Dict[int, float] = {}
    for t in sorted(D):
        f = (1.0 + ETA_BASE) / (1.0 + eta)
        out[t] = D[t] * ((1.0 - a[t]) + a[t] * f ** (t - YEARS[0]))
    adjusted = False
    run = -math.inf
    for t in sorted(out):
        if out[t] < run - 1e-12:
            out[t] = run
            adjusted = True
        run = max(run, out[t])
    return out, adjusted


def setting_paths(s: Setting) -> Tuple[Paths, bool]:
    ps = params(s.existing, s.structural)
    base = default_paths(ps)
    D = dict(s.demand) if s.demand is not None else dict(base.demand_TWh)
    demand, adj = (D, False) if abs(s.eta - ETA_BASE) < 1e-15 else hw_demand(ps, D, s.eta)
    carbon = dict(s.carbon) if s.carbon is not None else dict(base.carbon_price)
    return Paths(carbon_price=carbon, demand_TWh=demand,
                 liquid_deploy_cap=dict(base.liquid_deploy_cap)), adj


def setting_policy(s: Setting, role: str, scenario: str) -> Optional[PolicySettings]:
    if not s.liquid_fleet_max_ccn and s.reinforce_g is None:
        return None
    ps = params(s.existing, s.structural)
    if s.overrides:
        ps = ps.with_overrides(dict(s.overrides))
    pol = default_policy(scenario, ps)
    if s.liquid_fleet_max_ccn and role == "C-CN":
        pol = pol.with_(liquid_fleet_max=dict(s.liquid_fleet_max_ccn))
    if s.reinforce_g is not None:
        tg = int(ps.scalar("reinforcement_year"))
        conn: Dict[str, Dict[int, float]] = {}
        for r, yrs in pol.conn_capacity.items():
            c1 = ps.region("conn_capacity_reinforced_GW_per_yr", r)
            conn[r] = {t: (c1 if t >= tg else v) for t, v in yrs.items()}
        pol = pol.with_(conn_capacity=conn, reinforced_from={r: tg for r in conn},
                        reinforcement_cost_yen_per_kW=float(s.reinforce_g))
    return pol


def spec_for(s: Setting, role: str, label: str, store_slices: bool = False) -> runner.RunSpec:
    scen = s.role(role)
    paths, _ = setting_paths(s)
    adds = dict(s.additions_by_role).get(role, ())
    fixed = dict(s.fixed_itr_by_role).get(role)
    return runner.RunSpec(scenario=scen, existing_dir=s.existing, structural_dir=s.structural,
                          options=s.options, overrides=s.overrides, additions=adds,
                          paths=paths, policy=setting_policy(s, role, scen),
                          store_slices=store_slices,
                          realization=Realization(itr=fixed) if fixed else None, label=label)


def interpolate(base: Setting, alt: Setting, s: float) -> Setting:
    """Numeric parts move base + s (alt - base); everything else from ``alt``."""
    ps = params(base.existing, base.structural)
    ov_b = dict(base.overrides)
    ov_a = dict(alt.overrides)
    ov: Dict[str, float] = {}
    for k in set(ov_b) | set(ov_a):
        b = ov_b.get(k, _param_value(ps, k))
        a = ov_a.get(k, _param_value(ps, k))
        ov[k] = b + s * (a - b)
    bp, _ = setting_paths(base.with_(eta=ETA_BASE))
    ap, _ = setting_paths(alt.with_(eta=ETA_BASE))

    def mix(pb: Dict[int, float], pa: Dict[int, float]) -> Tuple[Tuple[int, float], ...]:
        return as_tuple({t: pb[t] + s * (pa[t] - pb[t]) for t in pb})

    return alt.with_(overrides=as_tuple(ov), carbon=mix(bp.carbon_price, ap.carbon_price),
                     demand=mix(bp.demand_TWh, ap.demand_TWh),
                     eta=base.eta + s * (alt.eta - base.eta))


def _param_value(ps: ParamSet, key: str) -> float:
    from ..params import _parse_key
    pid, region, year = _parse_key(key)
    return ps.get(pid, region, year).value


def region_variant(ps: ParamSet, pid: str, variant: str) -> Dict[str, float]:
    out = {}
    for r in ps.region_names:
        if ps.has_region(pid, r):
            v = getattr(ps.get(pid, r), variant)
            if not math.isnan(v):
                out[f"{pid}[{r}]"] = float(v)
    return out


def path_variant(ps: ParamSet, pid: str, variant: str) -> Dict[str, float]:
    return {f"{pid}@{t}": v for t, v in ps.path(pid, variant).items()}


def flat_carbon(ps: ParamSet, price: float) -> Tuple[Tuple[int, float], ...]:
    base = ps.path("carbon_price")
    return as_tuple({t: (base[t] if t <= CALIBRATION_YEARS[-1] else float(price)) for t in base})


# ----------------------------------------------------------------------
# Running and metrics
# ----------------------------------------------------------------------

RUN_LOG: List[Dict[str, Any]] = []


def solve(specs: Sequence[runner.RunSpec], part: str, use_cache: bool = True) -> List[Result]:
    res = runner.run_many(specs, workers=8, use_cache=use_cache,
                          log_path=LOG_DIR / f"stage3_{part}.csv")
    for s, r in zip(specs, res):
        RUN_LOG.append({"part": part, "label": s.label, "scenario": s.scenario,
                        "config_hash": r.meta["run"]["hash"][:16], "status": r.status,
                        "n_vars": r.n_vars, "n_cons": r.n_cons,
                        "build_s": round(float(r.timings.get("build_s", 0.0)), 2),
                        "solve_s": round(float(r.timings.get("solve_s", 0.0)), 2)})
    return res


def total(r: Result, extra: float = 0.0) -> float:
    if r.status != "Optimal":
        return math.nan
    return float(r.metrics["eq1_total_yen"]) + extra


def adv(c: float, r: float) -> float:
    return (c - r) / c if c and not math.isnan(c) else math.nan


def siting_s2_cost(r: Result, role: str) -> float:
    """Bridge step T2 of assumption set S2 (yen, discounted): 4.0e9 / 1.5e9 yen
    per TWh-yr of IT energy above the 2024 level, metro in C-CN, non-metro in
    R-CN."""
    if role not in ("C-CN", "R-CN"):
        return 0.0
    ry = r.tables["region_year"]
    rate = 4.0e9 if role == "C-CN" else 1.5e9
    sel = ry["region"].isin(METRO) if role == "C-CN" else ~ry["region"].isin(METRO)
    base = ry[ry["year"] == YEARS[0]].set_index("region")["it_energy_TWh"]
    inc = (ry["it_energy_TWh"] - ry["region"].map(base)).clip(lower=0.0)
    return float((rate * inc * ry["disc"])[sel].sum())


def _round(x: Any) -> Any:
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        return None if math.isnan(xf) or math.isinf(xf) else float(f"{xf:.10g}")
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, dict):
        return {str(k): _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    return x


def write_csv(df: pd.DataFrame, name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / name, index=False, float_format=FLOAT_FMT, lineterminator="\n")


def write_json(obj: Any, name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(_round(obj), indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")


def write_run_log(part: str, out: Path) -> None:
    rows = [r for r in RUN_LOG if r["part"] == part]
    write_csv(pd.DataFrame(rows), f"run_log_{part}.csv", out)


def scenario_row(r: Result) -> Dict[str, Any]:
    m = r.metrics
    if r.status != "Optimal":
        return {"status": r.status}
    return {
        "status": r.status,
        "eq1_total_TJPY": m["eq1_total_yen"] / TJ,
        "social_cost_TJPY": m["social_cost_yen"] / TJ,
        "operator_cost_TJPY": m["operator_cost_yen"] / TJ,
        "fiscal_burden_TJPY": m["fiscal_burden_yen"] / TJ,
        "carbon_cost_TJPY": m["carbon_cost_yen"] / TJ,
        "co2_2040_Mt": m["co2_2040_Mt"],
        "co2_cum_Mt": m["co2_cum_Mt"],
        "curtail_TWh": m["curtail_TWh"],
        "nonmetro_share_2040": m["nonmetro_share_last"],
        "liquid_share_fleet_2040": m["liquid_share_fleet_last"],
    }


@dataclass
class PartResult:
    files: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
