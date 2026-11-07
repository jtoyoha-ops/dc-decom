"""Rolling execution of the A/B/C designs of ch07 7.10 (plan -> realize -> observe).

For a design d and an actual path omega (``run_designs``):

S_2025   the calibration years 2024-2025 are solved once with the base paths
         and the A0 policy; the state at the end of 2025 is the same for every
         design and path (the actual paths take their variant values from 2026).
For each review year t_k in ``review_years`` with window W_k = [t_k, t_{k+1} - 1]
(the last window ends in 2040):
PLAN     solve [t_k, 2040] from the realized state S_{t_k - 1} with the forecast
         (base) paths for carbon price, demand and liquid deploy cap and the
         policy in force at t_k, which the operator assumes to persist. The
         demand forecast never falls below the demand observed in t_k - 1
         (``plan_paths``; the no-relocation rule of endogenous siting needs a
         forecast at or above the observed level).
REALIZE  solve W_k with the actual values of omega: regional IT = actual national
         demand x planned share where no region falls below its previous load,
         otherwise the previous load plus a share of the national growth (or a
         proportional decline when national demand falls; ``realized_itr``);
         BESS, PV and CGS additions = plan; generator
         additions >= plan (extra backup when the N+1 requirement rises);
         DLC / immersion additions = plan, scaled down where the plan exceeds the
         actual deploy cap (not only "<= plan": with only "<=" the window LP,
         which sees no year after the window, replaces planned liquid by air in
         the last window years whenever the payback of the liquid premium is
         longer than the rest of the window); air
         additions free (recourse; the LP has no air capex, so a tie-break weight
         of 1 yen/kW, not counted as a cost, keeps idle air builds out of the
         solution); dispatch, PPA and certificates free; the
         connection overrun 'over' allowed at the reinforcement cost. The PUE
         caps and the liquid floor bind the plan; they do not restrict the
         realized air recourse (config/abc_rules.yaml: "air cooling may be
         added to cover shortfalls"). Their realized compliance is reported.
OBSERVE  at t_{k+1}: indicators of the realized years up to t_{k+1} - 1; an
         adaptive design (C) applies the switching rules R1-R4 of
         config/abc_rules.yaml to the settings of the next stage. A and B never
         switch.
The realized windows are stitched into one 2024-2040 solution and evaluated
with ``accounting`` on the actual values (``stitch``).

Fairness: every spec is built from one ``RollingBase`` (parameters, model
options, scenario, forecast, review years); designs differ only in the
policy (``Settings`` -> ``policy_for``). ``assert_fair`` checks this for the
specs of every stage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from . import accounting, runner
from .config import (
    BASE_YEAR,
    CALIBRATION_YEARS,
    COOLING,
    END_YEAR,
    METRO,
    ModelOptions,
    Paths,
    PolicySettings,
    Realization,
    State,
    year_map,
)
from .model import Result, state_at
from .params import ParamSet
from .scenarios import default_paths, default_policy

REVIEW_YEARS = (2026, 2030, 2035)
SCENARIO = "A0"          # endogenous siting, no CO2 cap (ch07: CO2 is compared as a result)
FIXED_TECHS = ("bess", "pv", "cgs")
MIN_TECHS = ("gen",)
LIQUID_TECHS = ("dlc", "imm")
TOL = 1e-9
AIR_TIEBREAK_YEN_PER_KW = 1.0   # realization only: makes the free air recourse unique
ITR_RULES = ("target", "increment")
VARIANT_LABEL = {"low": "low", "value": "base", "high": "high"}
LIQUID_LABEL = {"value": "base", "low": "slow"}
# RunSpec fields that may differ between designs (policy) or follow from the
# design's own earlier decisions (state, realization fixings).
POLICY_FIELDS = ("policy", "label")
OUTCOME_FIELDS = ("initial_state", "realization")

Solver = Callable[[List[runner.RunSpec], str], List[Result]]


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ActualPath:
    name: str
    carbon: str          # variant of carbon_price (low | value | high)
    demand: str          # variant of demand_TWh
    liquid: str          # variant of liquid_deploy_cap (value | low)
    paths: Paths


def actual_paths(ps: ParamSet, carbon: Sequence[str] = ("low", "value", "high"),
                 demand: Sequence[str] = ("low", "value", "high"),
                 liquid: Sequence[str] = ("value", "low")) -> List[ActualPath]:
    """The realization paths: base values in the calibration years, the variant
    from 2026 (so that S_2025 is the same for every path)."""
    base = default_paths(ps)

    def mix(pid: str, variant: str, b: Mapping[int, float]) -> Dict[int, float]:
        v = ps.path(pid, variant)
        return {t: float(b[t] if t <= CALIBRATION_YEARS[-1] else v[t]) for t in sorted(b)}

    out = []
    for c in carbon:
        for d in demand:
            for liq in liquid:
                name = f"c-{VARIANT_LABEL[c]}_d-{VARIANT_LABEL[d]}_l-{LIQUID_LABEL[liq]}"
                p = Paths(carbon_price=mix("carbon_price", c, base.carbon_price),
                          demand_TWh=mix("demand_TWh", d, base.demand_TWh),
                          liquid_deploy_cap=mix("liquid_deploy_cap", liq, base.liquid_deploy_cap))
                out.append(ActualPath(name, c, d, liq, p))
    return out


def truncate(p: Paths, end: int) -> Paths:
    def cut(x: Mapping[int, float]) -> Dict[int, float]:
        return {t: float(v) for t, v in sorted(x.items()) if t <= end}
    return Paths(carbon_price=cut(p.carbon_price), demand_TWh=cut(p.demand_TWh),
                 liquid_deploy_cap=cut(p.liquid_deploy_cap))


# ----------------------------------------------------------------------
# Shared base, designs and policies
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RollingBase:
    """Everything a rolling run depends on except the policy (one object for
    every design)."""

    existing_dir: str
    forecast: Paths
    structural_dir: Optional[str] = None
    options: ModelOptions = ModelOptions(certificates=True)
    scenario: str = SCENARIO
    review_years: Tuple[int, ...] = REVIEW_YEARS
    overrides: Tuple[Tuple[str, float], ...] = ()
    itr_rule: str = "target"     # realized regional IT (``realized_itr``)

    def params(self) -> ParamSet:
        ps = runner.load_params(self.existing_dir, self.structural_dir)
        return ps.with_overrides(dict(self.overrides)) if self.overrides else ps

    def windows(self) -> List[Tuple[int, int]]:
        ys = list(self.review_years)
        if ys[0] != CALIBRATION_YEARS[-1] + 1 or ys != sorted(ys):
            raise ValueError("review years must be increasing and start right after 2025")
        return [(t, ys[i + 1] - 1 if i + 1 < len(ys) else END_YEAR) for i, t in enumerate(ys)]


@dataclass(frozen=True)
class Settings:
    """Policy knobs of a design in one stage (the C rules switch phi, t_g,
    subsidy and psi).

    pue_rules       fleet PUE cap and new-build PUE cap (pue_fleet_max, pue_new_max)
    phi             liquid share floor of new capacity from 2026 (None = none)
    t_g             reinforcement year: c1 and the reinforcement cost from t_g
                    (None = c0, no reinforcement)
    psi             PPA share required for subsidized builds (None = none)
    subsidy         'none' | 'uniform' (s0 for new non-metro IT from 2026) |
                    'rule' (s0 from t_g in years whose base-path carbon price is
                    below pstar) | 'off'
    floors          'paths' (lambda_B, kappa_B) | 'A' (lambda_A, kappa_A)
    ppa_multiplier  factor on the PPA tier volumes of non-metro regions
    theta, g        operator share and cost of the reinforcement (None = parameter)
    pstar           P* of the subsidy rule
    """

    pue_rules: bool = False
    phi: Optional[float] = None
    t_g: Optional[int] = None
    psi: Optional[float] = None
    subsidy: str = "none"
    floors: str = "paths"
    ppa_multiplier: Optional[float] = None
    theta: Optional[float] = None
    g: Optional[float] = None
    pstar: float = 0.0

    def with_(self, **kw: Any) -> "Settings":
        return replace(self, **kw)


@dataclass(frozen=True)
class Design:
    name: str
    settings: Settings
    adaptive: bool = False


def designs_from_rules(rules: Mapping[str, Any], ps: ParamSet,
                       diagnostics: bool = True) -> Dict[str, Design]:
    """A0, A, B, C (config/abc_rules.yaml 'designs') and the diagnostic B variants.
    Values come from the frozen parameters and thresholds only."""
    thr = rules["thresholds"]
    pstar = float(thr["pstar"]["value"])
    a = Settings(pue_rules=True, subsidy="uniform", floors="A")
    b = Settings(pue_rules=True, phi=float(thr["phi_0"]["value"]),
                 t_g=int(ps.scalar("reinforcement_year")),
                 psi=float(ps.scalar("ppa_floor_subsidized")), subsidy="rule", floors="paths",
                 ppa_multiplier=float(ps.scalar("ppa_tier_multiplier_induced")), pstar=pstar)
    out = {"A0": Design("A0", Settings()), "A": Design("A", a), "B": Design("B", b),
           "C": Design("C", b, adaptive=True)}
    if diagnostics:
        keys = {"operator_burden_share": "theta", "reinforcement_cost_yen_per_kW": "g",
                "pstar": "pstar", "ppa_floor_subsidized": "psi"}
        for name, changes in (rules.get("diagnostic_B_variants") or {}).items():
            kw = {keys[k]: (None if v is None else float(v))
                  for k, v in changes.items() if k in keys}
            unknown = set(changes) - set(keys) - {"note"}
            if unknown:
                raise ValueError(f"{name}: unknown variant keys {sorted(unknown)}")
            out[name] = Design(name, b.with_(**kw))
    return out


def policy_for(s: Settings, ps: ParamSet, base_carbon: Mapping[int, float],
               scenario: str = SCENARIO) -> PolicySettings:
    """PolicySettings of a stage (all years; only the years of the stage and
    later are used)."""
    pol = default_policy(scenario, ps)
    first = CALIBRATION_YEARS[-1] + 1
    years = range(first, END_YEAR + 1)
    nonmetro = [r for r in ps.region_names if r not in METRO]
    ch: Dict[str, Any] = {}
    if s.pue_rules:
        ch["pue_fleet_max"] = year_map(ps.scalar("pue_fleet_max"),
                                       int(ps.scalar("pue_fleet_max_from")))
        ch["pue_new_max"] = year_map(ps.scalar("pue_new_max"), int(ps.scalar("pue_new_max_from")))
    if s.phi is not None:
        ch["liquid_floor_new"] = year_map(float(s.phi), first)
    if s.floors == "A":
        ch["metro_floor"] = year_map(ps.scalar("metro_floor_A"), first)
        ch["kanto_floor"] = year_map(ps.scalar("kanto_floor_A"), first)
    elif s.floors != "paths":
        raise ValueError(f"unknown floors {s.floors!r}")
    if s.t_g is not None:
        conn: Dict[str, Dict[int, float]] = {}
        for r, yrs in pol.conn_capacity.items():
            c1 = ps.region("conn_capacity_reinforced_GW_per_yr", r)
            conn[r] = {t: (c1 if t >= s.t_g else v) for t, v in yrs.items()}
        ch["conn_capacity"] = conn
        ch["reinforced_from"] = {r: int(s.t_g) for r in conn}
    s0 = ps.scalar("siting_subsidy_yen_per_kW")
    if s.subsidy == "uniform":
        ch["subsidy_yen_per_kW"] = {r: {t: s0 for t in years} for r in nonmetro}
    elif s.subsidy == "rule":
        on = [t for t in years if s.t_g is not None and t >= s.t_g
              and float(base_carbon[t]) < s.pstar]
        if on:
            ch["subsidy_yen_per_kW"] = {r: {t: s0 for t in on} for r in nonmetro}
    elif s.subsidy not in ("none", "off"):
        raise ValueError(f"unknown subsidy {s.subsidy!r}")
    if s.psi is not None:
        ch["ppa_floor_subsidized"] = float(s.psi)
    if s.ppa_multiplier is not None:
        ch["ppa_tier_multiplier"] = {r: float(s.ppa_multiplier) for r in nonmetro}
    if s.theta is not None:
        ch["operator_burden_share"] = float(s.theta)
    if s.g is not None:
        ch["reinforcement_cost_yen_per_kW"] = float(s.g)
    return pol.with_(**ch)


def _from(pol: PolicySettings, t0: int) -> Any:
    """JSON form of the policy restricted to the years from t0."""
    def cut(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: cut(v) for k, v in x.items() if not (isinstance(k, int) and k < t0)}
        return x
    return runner._jsonable(cut({f.name: getattr(pol, f.name) for f in fields(pol)}))


def realize_policy(pol: PolicySettings) -> PolicySettings:
    """The plan's policy without the standards that would restrict the air recourse."""
    return pol.with_(pue_fleet_max={}, pue_new_max={}, liquid_floor_new={})


# ----------------------------------------------------------------------
# Specs
# ----------------------------------------------------------------------

def calibration_spec(base: RollingBase) -> runner.RunSpec:
    ps = base.params()
    return runner.RunSpec(scenario=base.scenario, existing_dir=base.existing_dir,
                          structural_dir=base.structural_dir, options=base.options,
                          overrides=base.overrides, paths=truncate(base.forecast,
                                                                   CALIBRATION_YEARS[-1]),
                          policy=default_policy(base.scenario, ps), start_year=BASE_YEAR,
                          end_year=CALIBRATION_YEARS[-1], store_slices=True,
                          label="rolling:calibration")


def plan_paths(base: RollingBase, actual: Paths, t_k: int) -> Paths:
    """History (actual values before t_k) + forecast (base paths) from t_k; the
    demand forecast is at least the demand observed in t_k - 1."""
    f = base.forecast
    d_obs = float(actual.demand_TWh[t_k - 1])

    def hist(a: Mapping[int, float], fc: Mapping[int, float]) -> Dict[int, float]:
        return {t: float(a[t] if t < t_k else fc[t]) for t in sorted(fc)}

    demand = {t: float(actual.demand_TWh[t] if t < t_k else max(f.demand_TWh[t], d_obs))
              for t in sorted(f.demand_TWh)}
    return Paths(carbon_price=hist(actual.carbon_price, f.carbon_price), demand_TWh=demand,
                 liquid_deploy_cap=hist(actual.liquid_deploy_cap, f.liquid_deploy_cap))


def plan_spec(base: RollingBase, path: ActualPath, t_k: int, state: State,
              policy: PolicySettings, label: str) -> runner.RunSpec:
    return runner.RunSpec(scenario=base.scenario, existing_dir=base.existing_dir,
                          structural_dir=base.structural_dir, options=base.options,
                          overrides=base.overrides, paths=plan_paths(base, path.paths, t_k),
                          policy=policy, start_year=t_k, end_year=END_YEAR,
                          initial_state=state, store_slices=False, label=label)


def planned_decisions(plan: Result, t0: int, t1: int) -> pd.DataFrame:
    """Planned regional IT shares and additions of the window [t0, t1]."""
    a = plan.flows["annual"]
    a = a[(a["year"] >= t0) & (a["year"] <= t1)].copy()
    tot = a.groupby("year")["itr_GW"].transform("sum")
    cols = ["year", "region", "itr_GW"] + [
        f"new_{k}_{'GWh' if k == 'bess' else 'GW'}" for k in COOLING + ("pv", "bess", "cgs", "gen")]
    out = a[cols].reset_index(drop=True)
    out.insert(3, "share", (a["itr_GW"] / tot).to_numpy())
    return out


def _toward_target(a: float, prev: Mapping[str, float], target: Mapping[str, float],
                   keys: Sequence[str]) -> Dict[str, float]:
    """Growth a >= 0 shared among ``keys`` in proportion to the gaps
    max(0, target - prev) (in proportion to the targets if no gap is positive)."""
    if not keys:
        return {}
    gain = {r: max(0.0, target[r] - prev[r]) for r in keys}
    tot = sum(gain.values())
    if tot <= TOL:
        gain = {r: max(0.0, target[r]) for r in keys}
        tot = sum(gain.values())
    if tot <= TOL:
        return {r: a / len(keys) for r in keys}
    return {r: a * gain[r] / tot for r in keys}


def realized_itr(planned: pd.DataFrame, actual: Paths, prev_itr: Mapping[str, float],
                 rule: str = "target", metro_floor: Optional[Mapping[int, float]] = None,
                 kanto_floor: Optional[Mapping[int, float]] = None
                 ) -> Dict[Tuple[str, int], float]:
    """Regional IT load of a realized window without relocation of existing load
    (ch04: no region falls below its previous load). Year by year, from the
    realized load R of the year before (``prev_itr`` for the first year), with
    D = actual national demand / 8.76 and the target T_r = D x planned share_r
    (the planned-share rule):
      D <= sum(R): every region scales by D / sum(R) (a national decline is
                   unavoidable; no load moves between regions, shares stay);
      D >  sum(R): the growth a = D - sum(R) is added, nothing is taken away.
    rule 'target' (used for the results): a is split between the metro and the
      non-metro regions in proportion to the group gaps max(0, T_G - R_G), but
      the metro part is at least what the metro floor lambda(t) and the Kanto
      floor kappa(t) of the stage need (feasible because the shares of R meet
      the floors of t - 1 and the floors do not rise); inside each group the
      growth goes to the regions in proportion to their gaps max(0, T_r - R_r),
      Kanto first up to its floor. When no target lies below its previous load
      the result is the target itself, so the rule differs from the
      planned-share rule only in the years in which that rule would move
      existing load.
    rule 'increment' (sensitivity): a is split in proportion to the planned
      increments (planned shares if none); floors are not enforced."""
    if rule not in ITR_RULES:
        raise ValueError(f"rule must be one of {ITR_RULES}")
    lam_of = dict(metro_floor or {})
    kap_of = dict(kanto_floor or {})
    prev = {str(r): float(x) for r, x in prev_itr.items()}
    plan_prev = dict(prev)
    out: Dict[Tuple[str, int], float] = {}
    for t in sorted(int(y) for y in planned["year"].unique()):
        g = planned[planned["year"] == t]
        share = {str(r): float(s) for r, s in zip(g["region"], g["share"])}
        plan_now = ({str(r): float(x) for r, x in zip(g["region"], g["itr_GW"])}
                    if rule == "increment" else {})
        if set(share) != set(prev):
            raise ValueError(f"planned regions {sorted(share)} differ from the state")
        D = float(actual.demand_TWh[t]) / 8.76
        S = sum(prev.values())
        a = D - S
        if a <= TOL:
            new = {r: prev[r] * D / S for r in prev}
        elif rule == "increment":
            gain = {r: max(0.0, plan_now[r] - plan_prev[r]) for r in prev}
            if sum(gain.values()) <= TOL:
                gain = dict(share)
            G = sum(gain.values())
            new = {r: prev[r] + a * gain[r] / G for r in prev}
        else:
            target = {r: D * share[r] for r in prev}
            metro = [r for r in prev if r in METRO]
            other = [r for r in prev if r not in METRO]
            gm = max(0.0, sum(target[r] - prev[r] for r in metro))
            gn = max(0.0, sum(target[r] - prev[r] for r in other))
            a_m = a * gm / (gm + gn) if gm + gn > TOL else a * sum(share[r] for r in metro)
            need_m = max(0.0, lam_of.get(t, 0.0) * D - sum(prev[r] for r in metro))
            need_k = (max(0.0, kap_of.get(t, 0.0) * D - prev["Kanto"]) if "Kanto" in prev
                      else 0.0)
            a_m = min(a, max(a_m, need_m, need_k)) if metro else 0.0
            add = _toward_target(a_m, prev, target, metro)
            if "Kanto" in add and add["Kanto"] < need_k:
                rest = [r for r in metro if r != "Kanto"]
                add = {"Kanto": need_k, **_toward_target(a_m - need_k, prev, target, rest)}
            add.update(_toward_target(a - a_m, prev, target, other))
            new = {r: prev[r] + add.get(r, 0.0) for r in prev}
        for r in prev:
            out[(r, t)] = new[r]
        prev, plan_prev = new, plan_now
    return out


def realization_from_plan(planned: pd.DataFrame, actual: Paths,
                          prev_itr: Mapping[str, float], rule: str = "target",
                          policy: Optional[PolicySettings] = None) -> Realization:
    """Regional IT from ``realized_itr`` (floors of ``policy``); BESS, PV, CGS =
    plan; generators >= plan; DLC and immersion = plan, scaled by min(1, actual
    cap / planned liquid additions) per region and year."""
    itr, fixed, lo = [], [], []
    load = realized_itr(planned, actual, prev_itr, rule,
                        metro_floor=policy.metro_floor if policy is not None else None,
                        kanto_floor=policy.kanto_floor if policy is not None else None)
    for row in planned.itertuples(index=False):
        r, t = str(row.region), int(row.year)
        itr.append((r, t, load[(r, t)]))
        d = row._asdict()
        for k in FIXED_TECHS:
            fixed.append((k, r, t, max(0.0, float(d[f"new_{k}_{'GWh' if k == 'bess' else 'GW'}"]))))
        for k in MIN_TECHS:
            lo.append((k, r, t, max(0.0, float(d[f"new_{k}_GW"]))))
        liq = {k: max(0.0, float(d[f"new_{k}_GW"])) for k in LIQUID_TECHS}
        tot = sum(liq.values())
        cap = float(actual.liquid_deploy_cap[t])
        f = cap / tot if tot > cap + TOL else 1.0
        for k in LIQUID_TECHS:
            fixed.append((k, r, t, liq[k] * f))
    return Realization(itr=tuple(itr), new_fixed=tuple(fixed), new_min=tuple(lo),
                       air_tiebreak_yen_per_kW=AIR_TIEBREAK_YEN_PER_KW)


def realize_spec(base: RollingBase, path: ActualPath, t0: int, t1: int, state: State,
                 policy: PolicySettings, planned: pd.DataFrame, label: str) -> runner.RunSpec:
    return runner.RunSpec(scenario=base.scenario, existing_dir=base.existing_dir,
                          structural_dir=base.structural_dir,
                          options=base.options.with_(connection_overrun=True),
                          overrides=base.overrides, paths=truncate(path.paths, t1),
                          policy=realize_policy(policy), start_year=t0, end_year=t1,
                          initial_state=state, store_slices=True,
                          realization=realization_from_plan(planned, path.paths, state.itr,
                                                            base.itr_rule, policy),
                          label=label)


def assert_fair(specs: Mapping[str, runner.RunSpec], outcome_ok: bool = True) -> None:
    """All fields but the policy (and, if ``outcome_ok``, the state and the
    realization fixings, which follow from the design's own decisions) must be
    equal across designs."""
    names = list(specs)
    skip = set(POLICY_FIELDS) | (set(OUTCOME_FIELDS) if outcome_ok else set())
    ref = specs[names[0]]
    for n in names[1:]:
        for f in fields(runner.RunSpec):
            if f.name in skip:
                continue
            if runner._jsonable(getattr(ref, f.name)) != runner._jsonable(getattr(specs[n], f.name)):
                raise AssertionError(f"designs {names[0]} and {n} differ in {f.name}")


# ----------------------------------------------------------------------
# Observation and switching (C)
# ----------------------------------------------------------------------

def cap_path(E_2024: float, years: Sequence[int]) -> Dict[int, float]:
    """ch05 eq:co2cap-path: E_2024 (2040 - t) / 16."""
    return {int(t): E_2024 * max(0.0, END_YEAR - t) / (END_YEAR - BASE_YEAR) for t in years}


def observe(window: Result, actual: Paths, forecast: Paths, t_next: int,
            E_2024: float) -> Dict[str, float]:
    """Indicators of the realized stage ending in t_next - 1 (nothing later)."""
    nat = window.tables["national"]
    ry = window.tables["region_year"]
    last = t_next - 1
    if int(nat["year"].max()) != last:
        raise ValueError("observation must use the stage that ends in t_next - 1")
    new_all = float(nat["new_cooling_GW"].sum())
    new_liq = float(nat["new_liquid_GW"].sum())
    nl = nat[nat["year"] == last].iloc[0]
    con = ry[(ry["year"] == last) & ry["conn_offered_cum_GW"].notna()]
    usage = [u / o for u, o in zip(con["conn_used_phys_cum_GW"], con["conn_offered_cum_GW"])
             if o > TOL]
    cap = cap_path(E_2024, [int(t) for t in nat["year"]])
    excess = [float(e) - cap[int(t)] for t, e in zip(nat["year"], nat["co2_Mt"])]
    return {
        "liquid_share_new_stage": new_liq / new_all if new_all > TOL else math.nan,
        "liquid_share_new_last": float(nl["liquid_share_new"]),
        "conn_usage_last": max(usage) if usage else math.nan,
        "carbon_price_last": float(actual.carbon_price[last]),
        "co2_cap_excess_max_Mt": max(excess),
        "nonmetro_share_last": float(nl["nonmetro_share"]),
        "demand_last_TWh": float(actual.demand_TWh[last]),
        "demand_forecast_last_TWh": float(forecast.demand_TWh[last]),
    }


def switch(design: Design, s: Settings, obs: Mapping[str, float], t_next: int,
           rules: Mapping[str, Any],
           policy_of: Optional[Callable[[Settings], PolicySettings]] = None
           ) -> Tuple[Settings, List[Dict[str, Any]]]:
    """Switching rules R1-R4 (config/abc_rules.yaml) for the stage starting in
    t_next, from the indicators ``obs`` of the stage that ended in t_next - 1.
    'changed' = the setting changed; 'policy_changed' = the policy of the years
    from t_next differs (``policy_of``), e.g. a subsidy switched off that was
    already 0 changes the setting but not the policy."""
    thr = rules["thresholds"]
    rows: List[Dict[str, Any]] = []

    def eff(a: Settings, b: Settings) -> Optional[bool]:
        if policy_of is None:
            return None
        return _from(policy_of(a), t_next) != _from(policy_of(b), t_next)

    def log(rule: str, ind: str, value: float, threshold: Any, trig: bool, action: str,
            setting: str, before: Any, after: Any) -> None:
        pc = eff(s, s.with_(**{setting: after})) if setting else False
        rows.append({"review_year": t_next, "rule": rule, "indicator": ind, "value": value,
                     "threshold": threshold, "triggered": bool(trig), "action": action,
                     "setting": setting, "before": before, "after": after,
                     "changed": before != after, "policy_changed": pc})

    new = s
    # R1 liquid share of new capacity in the observed stage
    lr = float(thr["liquid_review"][t_next])
    v = obs["liquid_share_new_stage"]
    trig = (not math.isnan(v)) and v < lr - TOL
    phi = min(1.0, (s.phi or 0.0) + 0.2) if trig else s.phi
    new = new.with_(phi=phi)
    log("R1", "liquid_share_new_stage", v, lr, trig, "phi += 0.2 (cap 1.0)" if trig else "keep",
        "phi", s.phi, phi)
    # R2 connection usage
    cu = float(thr["connection_usage"]["value"])
    v = obs["conn_usage_last"]
    trig = (not math.isnan(v)) and v >= cu - TOL
    tg = (t_next if s.t_g is None else min(s.t_g, t_next)) if trig else s.t_g
    new = new.with_(t_g=tg)
    log("R2", "conn_usage_last", v, cu, trig, "T_g := min(T_g, t_k)" if trig else "keep",
        "t_g", s.t_g, tg)
    # R3 observed carbon price vs P*
    v = obs["carbon_price_last"]
    trig = v >= s.pstar - TOL
    sub = "off" if trig else design.settings.subsidy
    new = new.with_(subsidy=sub)
    log("R3", "carbon_price_last", v, s.pstar, trig,
        "subsidy 0 for the next stage" if trig else "continue", "subsidy", s.subsidy, sub)
    # R4 emissions above the cap path in any year of the stage
    v = obs["co2_cap_excess_max_Mt"]
    trig = v > 1e-6
    psi = min(1.0, (s.psi or 0.0) + 0.2) if trig else s.psi
    new = new.with_(psi=psi)
    log("R4", "co2_cap_excess_max_Mt", v, 0.0, trig, "psi += 0.2 (cap 1.0)" if trig else "keep",
        "psi", s.psi, psi)
    # reported next to the triggers (not triggers)
    tgt = thr["target_nonmetro_share"].get(t_next)
    log("report", "nonmetro_share_last", obs["nonmetro_share_last"], tgt, False,
        "reported (not a trigger)", "", None, None)
    log("report", "demand_last_TWh", obs["demand_last_TWh"], obs["demand_forecast_last_TWh"],
        False, "reported (demand vs forecast)", "", None, None)
    return new, rows


# ----------------------------------------------------------------------
# Rolling runs
# ----------------------------------------------------------------------

@dataclass
class RollingRun:
    design: Design
    path: ActualPath
    settings: Settings
    state: State
    status: str = "Optimal"
    note: str = ""
    stage_settings: List[Settings] = field(default_factory=list)
    policies: List[PolicySettings] = field(default_factory=list)
    plan_specs: List[runner.RunSpec] = field(default_factory=list)
    plans: List[Result] = field(default_factory=list)
    planned: List[pd.DataFrame] = field(default_factory=list)
    realize_specs: List[runner.RunSpec] = field(default_factory=list)
    windows: List[Result] = field(default_factory=list)
    states: List[State] = field(default_factory=list)
    observations: List[Dict[str, float]] = field(default_factory=list)
    switches: List[Dict[str, Any]] = field(default_factory=list)
    flows: Dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "Optimal"

    def fail(self, note: str) -> None:
        self.status, self.note = "Failed", note


def default_solver(workers: int = 8, use_cache: bool = True, cache_dir: Any = None) -> Solver:
    def solve(specs: List[runner.RunSpec], tag: str) -> List[Result]:
        return runner.run_many(specs, workers=workers, use_cache=use_cache, cache_dir=cache_dir)
    return solve


def run_designs(base: RollingBase, designs: Sequence[Design], paths: Sequence[ActualPath],
                rules: Mapping[str, Any], solve: Optional[Solver] = None,
                keep_flows: bool = True) -> Tuple[Result, List[RollingRun]]:
    """Rolling runs of every design on every path (batched per stage). Returns
    the calibration result and the runs (design-major order)."""
    solve = solve or default_solver()
    ps = base.params()
    wins = base.windows()
    calib = solve([calibration_spec(base)], "calibration")[0]
    if calib.status != "Optimal":
        raise RuntimeError(f"calibration years: {calib.status}")
    s25 = state_at(calib, CALIBRATION_YEARS[-1])
    E_2024 = float(calib.meta["E_2024_Mt"])
    runs = [RollingRun(design=d, path=p, settings=d.settings, state=s25)
            for d in designs for p in paths]
    for k, (t0, t1) in enumerate(wins):
        active = [r for r in runs if r.ok]
        specs = []
        for r in active:
            pol = policy_for(r.settings, ps, base.forecast.carbon_price, base.scenario)
            r.stage_settings.append(r.settings)
            r.policies.append(pol)
            r.states.append(r.state)
            specs.append(plan_spec(base, r.path, t0, r.state, pol,
                                   f"rolling:{r.design.name}:{r.path.name}:plan{t0}"))
        _check_fair(active, specs, outcome_ok=k > 0)
        plans = solve(specs, f"plan{t0}")
        real, rspecs = [], []
        for r, sp, p in zip(active, specs, plans):
            r.plan_specs.append(sp)
            r.plans.append(p)
            if p.status != "Optimal":
                r.fail(f"plan {t0}: {p.status}")
                continue
            pdec = planned_decisions(p, t0, t1)
            r.planned.append(pdec)
            rspecs.append(realize_spec(base, r.path, t0, t1, r.state, r.policies[-1], pdec,
                                       f"rolling:{r.design.name}:{r.path.name}:real{t0}"))
            real.append(r)
        _check_fair(real, rspecs, outcome_ok=True)
        windows = solve(rspecs, f"realize{t0}")
        for r, rs, w in zip(real, rspecs, windows):
            r.realize_specs.append(rs)
            r.windows.append(w)
            if w.status != "Optimal":
                r.fail(f"realize {t0}-{t1}: {w.status}")
                continue
            r.state = state_at(w, t1)
            if k + 1 < len(wins):
                t_next = wins[k + 1][0]
                obs = observe(w, r.path.paths, base.forecast, t_next, E_2024)
                r.observations.append(obs)
                if r.design.adaptive:
                    r.settings, rows = switch(
                        r.design, r.settings, obs, t_next, rules,
                        lambda x: policy_for(x, ps, base.forecast.carbon_price, base.scenario))
                    r.switches.extend(rows)
    for r in runs:
        if r.ok:
            r.flows, r.meta, r.metrics, r.tables = stitch(calib, r.windows)
            r.metrics.update(extra_metrics(r, E_2024))
            if not keep_flows:
                r.flows = {}
                r.windows = []
                r.plans = []
    return calib, runs


def _check_fair(active: Sequence[RollingRun], specs: Sequence[runner.RunSpec],
                outcome_ok: bool) -> None:
    by_path: Dict[str, Dict[str, runner.RunSpec]] = {}
    for r, s in zip(active, specs):
        by_path.setdefault(r.path.name, {})[r.design.name] = s
    for group in by_path.values():
        if len(group) > 1:
            assert_fair(group, outcome_ok=outcome_ok)


def stitch(calib: Result, windows: Sequence[Result]
           ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], Dict[str, Any],
                      Dict[str, pd.DataFrame]]:
    """One 2024-2040 solution from the calibration years and the realized windows,
    evaluated with ``accounting`` (actual values)."""
    parts = [calib] + list(windows)
    annual = pd.concat([p.flows["annual"] for p in parts], ignore_index=True)
    slices = pd.concat([p.flows["slices"] for p in parts], ignore_index=True)
    years = [int(t) for t in annual["year"].drop_duplicates()]
    if years != list(range(BASE_YEAR, END_YEAR + 1)) or annual.duplicated(["year", "region"]).any():
        raise ValueError(f"windows do not tile 2024-2040: {years}")
    meta = dict(windows[-1].meta)
    E = float(calib.meta["E_2024_Mt"])
    meta.update(years=years, objective_yen=math.nan, siting="rolling",
                co2_cap_Mt={str(t): v for t, v in cap_path(E, years).items()})
    flows = {"annual": annual, "slices": slices}
    metrics, tables = accounting.evaluate(flows, meta)
    return flows, meta, metrics, tables


def extra_metrics(r: RollingRun, E_2024: float) -> Dict[str, Any]:
    """Cap path, overrun and realized compliance with the standards of each stage."""
    nat = r.tables["national"]
    ry = r.tables["region_year"]
    cap = cap_path(E_2024, [int(t) for t in nat["year"]])
    later = nat[nat["year"] > CALIBRATION_YEARS[-1]]
    exc = [float(e) - cap[int(t)] for t, e in zip(later["year"], later["co2_Mt"])]
    out: Dict[str, Any] = {
        "meets_cap_path": bool(max(exc) <= 1e-6),
        "cap_path_max_excess_Mt": max(exc),
        "over_GW_total": float(ry["over_GW"].sum()),
        "it_energy_cum_TWh": float(nat["it_energy_TWh"].sum()),
        "n_switch_changes": int(sum(1 for s in r.switches if s["changed"])),
        "n_policy_changes": int(sum(1 for s in r.switches if s["policy_changed"])),
    }
    fleet_ex, new_ex, phi_gap, n_viol = -math.inf, -math.inf, -math.inf, 0
    wins = [(int(rs.start_year), int(rs.end_year)) for rs in r.realize_specs]
    for (t0, t1), pol in zip(wins, r.policies):
        g = ry[(ry["year"] >= t0) & (ry["year"] <= t1)]
        for row in g.itertuples(index=False):
            d = row._asdict()
            t = int(d["year"])
            itr = float(d["itr_GW"])
            new = {k: float(d[f"new_{k}_GW"]) for k in COOLING}
            n_all = sum(new.values())
            if t in pol.pue_fleet_max and itr > TOL:
                fp = sum(float(d[f"it_{k}_GW"]) * float(d[f"pue_ann_{k}"]) for k in COOLING) / itr
                ex = fp - pol.pue_fleet_max[t]
                fleet_ex = max(fleet_ex, ex)
                n_viol += int(ex > 1e-6)
            if t in pol.pue_new_max and n_all > 1e-6:
                npue = sum(new[k] * float(d[f"pue_ann_{k}"]) for k in COOLING) / n_all
                ex = npue - pol.pue_new_max[t]
                new_ex = max(new_ex, ex)
                n_viol += int(ex > 1e-6)
            if t in pol.liquid_floor_new and n_all > 1e-6:
                gap = pol.liquid_floor_new[t] - (new["dlc"] + new["imm"]) / n_all
                phi_gap = max(phi_gap, gap)
                n_viol += int(gap > 1e-6)
    out["std_violations"] = n_viol
    out["std_fleet_pue_excess_max"] = fleet_ex if fleet_ex > -math.inf else math.nan
    out["std_new_pue_excess_max"] = new_ex if new_ex > -math.inf else math.nan
    out["std_liquid_floor_gap_max"] = phi_gap if phi_gap > -math.inf else math.nan
    # realized siting: metro / Kanto share minus the floor of the stage (< 0 = below),
    # and the largest fall of a regional IT load in a year of national growth
    loads = ry.pivot(index="year", columns="region", values="itr_GW")
    floor_gap = math.inf
    for (t0, t1), pol in zip(wins, r.policies):
        for t in range(t0, t1 + 1):
            tot = float(loads.loc[t].sum())
            metro = float(loads.loc[t, [c for c in loads.columns if c in METRO]].sum())
            if t in pol.metro_floor:
                floor_gap = min(floor_gap, metro / tot - pol.metro_floor[t])
            if t in pol.kanto_floor and "Kanto" in loads.columns:
                floor_gap = min(floor_gap, float(loads.loc[t, "Kanto"]) / tot - pol.kanto_floor[t])
    falls = [float((loads.loc[t - 1] - loads.loc[t]).max()) for t in loads.index
             if t - 1 in loads.index and loads.loc[t].sum() >= loads.loc[t - 1].sum() - TOL]
    out["siting_floor_gap_min"] = floor_gap if floor_gap < math.inf else math.nan
    out["itr_fall_max_GW"] = max(0.0, max(falls)) if falls else 0.0
    return out
