"""DC-DECOM v2 LP: one builder for the annual and the rep96 time modes.

Units: GW (power, IT, nameplate), GWh (BESS energy), TWh (annual energy),
yen (costs; the LP objective is in 1e9 yen), Mt-CO2, discount factor
1 / (1 + d)^(t - 2024).

Formulation:
  cooling   cap_k[r,t] = cap_k[r,t-1] + new_k[r,t] (k in air, dlc, imm; GW IT
            nameplate; no retirement); initial air = IT_2024 x share / u;
            sum_k it_k = itr[r,t]; it_k <= u_cap x cap_k.
  IT energy 'anchor_is_it': the demand path is IT energy, itr_total = D/8.76,
            u_cap = u, u_energy = 1. 'legacy_util' (v1): u_cap = 1,
            u_energy = u (facility energy x 0.8).
  load      L[r,h,t] = sum_k it_k x u_energy x PUE_k(r,h).
  supply    g + pv + cgs + dis + curt (+ ppa_h - spill, hourly) = L + ch per
            slice; pv <= cap_pv CF; cgs <= cap_cgs; ch, dis <= cap_bess / dur;
            soc[h] = soc[prev(h)] + eta ch - dis / eta (prev(0) = 23 of the same
            representative day); soc <= cap_bess; eta = sqrt(round trip).
  PPA       annual matching: tiers ppa_i <= V_i (TWh), sum_i ppa_i <= G and
            Eq. 15 cap_pv + sum_i ppa_i / (CF_avg 8.76) <= RE_max; emissions
            EF (G - sum ppa - nfc) + EF_fossil CGS. Hourly matching: tier
            capacity delivered with the PV profile, spill paid but unused,
            emissions EF (G - nfc) + EF_fossil CGS.
  N+1       'eq': cap_cgs + cap_gen >= (1 + 1/N) sum_k (it_k / u_cap) PUE_k(r,h)
            for every slice; 'legacy5': cap_cgs >= 0.05 itr.
  siting    base_shares | fixed_shares (by name, linear 2025 -> 2040) |
            metro_only | endogenous (national sum, metro >= lambda, Kanto >=
            kappa, no relocation) | growth_split (2025 level + split x growth
            after 2025); 2024-2025 at base shares; 'given' (rolling realization:
            regional IT load from the plan, additions fixed or bounded).
  grid      non-metro regions with connection capacity: peak >= L[h],
            peak non-decreasing (connected capacity), cumulative
            peak[t] - peak[2025] <= sum_{2026..t} (capacity + over). The
            cost g applies to the peak added in reinforced years only, so in
            the other years the objective does not fix peak; a tie-break
            weight of PEAK_TIEBREAK_YEN_PER_KW (not a cost in ``accounting``,
            excluded from objective_yen) keeps it at the running maximum of
            the facility peak there (ch04: connected maximum facility power).
            Where a reinforced year follows free years in the horizon,
            connecting ahead of load would lower the cost, so peak[t] <=
            max(peak[t-1], L[h,t]) is imposed in the free years with binaries
            (the run becomes a small MILP; ``_pin_peak_before_reinforcement``).
  CO2 cap   sum_r em[r,t] <= E_2024 (2040 - t) / 16 with E_2024 = 2024
            facility energy x national EF_2024; optional floor
            sum_r em >= f x cap after 2025 (options.co2_floor_share).
  EF        regional: grid_ef_base x ef_national(t) / ef_national(2024), or
            the entry ef_grid_path[r]@t when the parameter set has it.
  objective operator cost (see ``accounting`` for the split metrics),
            including the regional communication / data-transfer cost
            comm_cost_yen_per_kWh[r] x IT energy (itr x u_energy x 8.76 TWh);
            the rate is 0 by default and the term is added only when non-zero.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pulp

from . import accounting
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
)
from .params import ParamSet
from .pue import PUECoefficients, annual_pue, nondominated_slices, slice_pue
from .scenarios import Scenario, default_paths, default_policy
from .scenarios import get as get_scenario
from .timeslices import TimeSlices, make_timeslices

GYEN = 1e9
LE, GE, EQ = pulp.LpConstraintLE, pulp.LpConstraintGE, pulp.LpConstraintEQ
LIQUID = ("dlc", "imm")
# Objective weight (yen per kW, discounted) on every connected-peak variable.
# Tie-break only: without it the peak of a year without reinforcement cost is
# any value between the facility peak and the cumulative connection capacity.
# Not a cost in ``accounting``.
PEAK_TIEBREAK_YEN_PER_KW = 1.0


class _Lin:
    """Sparse linear expression (dict of variable -> coefficient + constant)."""

    __slots__ = ("terms", "const")

    def __init__(self) -> None:
        self.terms: Dict[pulp.LpVariable, float] = {}
        self.const = 0.0

    def add(self, x: Any, c: float = 1.0) -> "_Lin":
        if isinstance(x, pulp.LpVariable):
            self.terms[x] = self.terms.get(x, 0.0) + c
        elif isinstance(x, _Lin):
            for v, a in x.terms.items():
                self.terms[v] = self.terms.get(v, 0.0) + c * a
            self.const += c * x.const
        else:
            self.const += c * float(x)
        return self

    def value(self) -> float:
        return self.const + sum(a * (v.varValue or 0.0) for v, a in self.terms.items())


def _lin(*pairs: Tuple[Any, float]) -> _Lin:
    e = _Lin()
    for x, c in pairs:
        e.add(x, c)
    return e


def _val(x: Any) -> float:
    if isinstance(x, pulp.LpVariable):
        return float(x.varValue or 0.0)
    if isinstance(x, _Lin):
        return x.value()
    return float(x)


@dataclass
class Handle:
    """A built LP. ``v`` holds the variables / expressions by family so that a
    caller (e.g. the rolling realization) can add constraints before solving."""

    prob: pulp.LpProblem
    options: ModelOptions
    scenario: Scenario
    siting: str
    params: ParamSet
    paths: Paths
    policy: PolicySettings
    years: List[int]
    regions: List[str]
    ts: TimeSlices
    initial_state: Optional[State]
    data: Dict[str, Any]
    v: Dict[str, Dict[Any, Any]]
    build_seconds: float = 0.0


@dataclass
class Result:
    status: str
    flows: Dict[str, pd.DataFrame]
    metrics: Dict[str, Any]
    timings: Dict[str, float]
    n_vars: int
    n_cons: int
    meta: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Input preparation
# ----------------------------------------------------------------------

def _bess_capex(p: ParamSet, year: int) -> float:
    """thousand yen/kWh (= 1e9 yen/GWh), linear 2025 -> 2035 as v1."""
    frac = min(1.0, max(0.0, (year - 2025) / (2035 - 2025)))
    c25, c35 = p.scalar("bess_capex_2025"), p.scalar("bess_capex_2035")
    return c25 + (c35 - c25) * frac


def _tiers(p: ParamSet) -> List[Tuple[str, float]]:
    """[(share id, premium yen/kWh)] of the PPA supply curve; [] = one tier
    without a volume limit (only Eq. 15)."""
    out: List[Tuple[str, float]] = []
    i = 1
    while p.has(f"ppa_tier{i}_share"):
        prem = 0.0 if i == 1 else p.scalar(f"ppa_tier{i}_premium_yen_per_kWh")
        out.append((f"ppa_tier{i}_share", prem))
        i += 1
    return out


def _bound_additions(new: Dict[Tuple[str, str, int], pulp.LpVariable],
                     rz: Realization) -> None:
    """Variable bounds of a realization window (fixed, minimum, maximum additions).
    A key without a variable (e.g. generators when they are not a backup
    technology) is accepted only with the value 0."""
    for group, mode in ((rz.new_fixed, "fix"), (rz.new_min, "min"), (rz.new_max, "max")):
        for k, r, t, x in group:
            key = (str(k), str(r), int(t))
            x = max(0.0, float(x))
            if key not in new:
                if x > 1e-9:
                    raise ValueError(f"realization bounds {key}, which has no variable")
                continue
            v = new[key]
            lo = float(v.lowBound or 0.0)
            hi = v.upBound
            if mode in ("fix", "min"):
                lo = max(lo, x if hi is None else min(x, float(hi)))
            if mode in ("fix", "max"):
                hi = x if hi is None else min(float(hi), x)
            v.lowBound, v.upBound = lo, hi


def _pin_peak_before_reinforcement(prob: pulp.LpProblem, r: str, years: List[int],
                                   peak: Dict[Tuple[str, int], Any],
                                   L: Dict[Tuple[str, int, int], "_Lin"], nondom: Any,
                                   policy: PolicySettings,
                                   over: Dict[Tuple[str, int], pulp.LpVariable],
                                   base_cap: float, st: Optional[State]) -> None:
    """Connected peak = running maximum of the facility peak in the years before
    the first reinforced year of the horizon: peak[t] <= max(peak[t-1], L[h, t])
    with one binary per option. Without it a run that sees
    a reinforced year would connect ahead of load in the free years (up to the
    cumulative capacity) to avoid the cost g on the later increments, which
    contradicts ch04 (Peak = connected maximum facility power). The tie-break
    weight cannot prevent this because connecting early lowers the cost."""
    reinf = [t for t in years if policy.is_reinforced(r, t)]
    if not reinf:
        return
    free = [t for t in years if t < min(reinf) and isinstance(peak.get((r, t)), pulp.LpVariable)]
    offered = 0.0
    for t in free:
        if (r, t) in over:
            raise ValueError("connection overrun with a later reinforced year is not supported")
        offered += float(policy.conn_capacity[r].get(t, 0.0))
        big_m = base_cap + offered + 1.0
        if t - 1 in years:
            prev: Any = peak[(r, t - 1)]
        elif st is not None:
            prev = float(st.peak[r])
        else:
            continue
        options: List[Any] = [prev] + [L[(r, t, h)] for h in nondom]
        pick = [pulp.LpVariable(f"pinpeak_{r}_{t}_{i}", cat="Binary") for i in range(len(options))]
        prob.addConstraint(pulp.lpSum(pick) == 1)
        for opt, z in zip(options, pick):
            rhs = (pulp.LpAffineExpression(opt.terms, constant=opt.const) if isinstance(opt, _Lin)
                   else opt)
            prob.addConstraint(peak[(r, t)] - rhs - big_m * (1 - z) <= 0)


def _fixed_share_path(shares0: Dict[str, float], target: Dict[str, float],
                      regs: List[str], years: List[int]) -> Dict[Tuple[str, int], float]:
    missing = set(regs) - set(target)
    if missing:
        raise ValueError(f"fixed shares miss regions {sorted(missing)}")
    t0 = CALIBRATION_YEARS[-1]
    out: Dict[Tuple[str, int], float] = {}
    for t in years:
        frac = max(0.0, (t - t0) / (END_YEAR - t0))
        s = {r: shares0[r] + (target[r] - shares0[r]) * frac for r in regs}
        tot = sum(s.values())
        for r in regs:
            out[(r, t)] = s[r] / tot
    return out


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------

def build(options: ModelOptions, scenario: Union[str, Scenario], params: ParamSet,
          paths: Optional[Paths] = None, policy: Optional[PolicySettings] = None,
          start_year: int = BASE_YEAR, initial_state: Optional[State] = None,
          end_year: int = END_YEAR, realization: Optional[Realization] = None) -> Handle:
    """Build the LP. ``realization`` (rolling realization windows): the regional
    IT load is given (siting 'given', no siting constraints) and the listed
    additions are fixed or bounded."""
    t_start = time.perf_counter()
    sc = get_scenario(scenario)
    siting = "given" if realization is not None else (options.siting or sc.siting)
    if siting == "fixed_shares" and not sc.fixed_shares:
        raise ValueError(f"scenario {sc.name} has no fixed shares")
    if siting == "growth_split" and not sc.growth_split:
        raise ValueError(f"scenario {sc.name} has no growth split")
    if not BASE_YEAR <= start_year <= end_year <= END_YEAR:
        raise ValueError("need 2024 <= start_year <= end_year <= 2040")
    if start_year > BASE_YEAR:
        if initial_state is None or initial_state.year != start_year - 1:
            raise ValueError("start_year > 2024 needs initial_state of year start_year - 1")
    elif initial_state is not None:
        raise ValueError("initial_state is only used with start_year > 2024")
    paths = paths or default_paths(params)
    policy = policy or default_policy(sc, params)
    years = list(range(start_year, end_year + 1))
    regs = list(params.region_names)
    st = initial_state

    ts = make_timeslices(options.time_mode, params, regs)
    H = range(len(ts.slices))
    w = ts.weights
    sp = slice_pue(ts, PUECoefficients.from_params(params))
    pa = annual_pue(ts, sp)
    nondom = {r: nondominated_slices(ts, sp, r) for r in regs}

    u = params.scalar("utilization")
    u_cap, u_e = (u, 1.0) if options.it_energy == "anchor_is_it" else (1.0, u)
    hourly = options.ppa_matching == "hourly"

    all_years = range(BASE_YEAR, end_year + 1)
    missing = [t for t in all_years if t not in paths.demand_TWh]
    if missing:
        raise ValueError(f"demand path misses years {missing}")
    total = {t: paths.demand_TWh[t] / 8.76 for t in all_years}
    sh0 = {r: params.region("base_share_2024", r) for r in regs}
    s_tot = sum(sh0.values())
    sh0 = {r: v / s_tot for r, v in sh0.items()}
    itr_2024 = {r: total[BASE_YEAR] * sh0[r] for r in regs}

    ef_nat = params.path("ef_national")
    if options.ef_mode == "regional":
        ef = {(r, t): params.region("grid_ef_base", r) * ef_nat[t] / ef_nat[BASE_YEAR]
              for r in regs for t in years}
    else:
        ef = {(r, t): ef_nat[t] for r in regs for t in years}
    for r in regs:
        for t in years:
            x_ef = params.optional("ef_grid_path", r, t)
            if x_ef is not None:
                ef[(r, t)] = x_ef
    ef_fossil = params.scalar("EF_fossil")
    E_2024 = sum(itr_2024[r] * u_e * pa[(r, "air")] * 8.76 for r in regs) * ef_nat[BASE_YEAR]
    co2_cap = ({t: E_2024 * max(0.0, END_YEAR - t) / (END_YEAR - BASE_YEAR) for t in years}
               if sc.co2_cap else None)
    price = {t: 0.0 if sc.zero_carbon_price else float(paths.carbon_price[t]) for t in years}
    d = params.scalar("discount_rate")
    disc = {t: 1.0 / (1.0 + d) ** (t - BASE_YEAR) for t in years}

    prem = {"air": 0.0,
            "dlc": params.scalar("cooling_dlc_capex") - params.scalar("cooling_air_capex"),
            "imm": params.scalar("cooling_immersion_capex") - params.scalar("cooling_air_capex")}
    use_gen = options.resilience == "eq" and "gen" in options.backup_techs
    supply = ["pv", "bess", "cgs"] + (["gen"] if use_gen else [])
    unit_capex = {"cgs": params.scalar("cgs_capex"),
                  "gen": params.scalar("gen_capex") if use_gen else 0.0}
    om = params.scalar("opex_om_rate")
    pv_lcoe = params.scalar("pv_lcoe")
    cgs_fuel_e = params.scalar("cgs_fuel_yen_per_kWh_th") / params.scalar("cgs_eff_elec")
    penalty = params.scalar("curtail_penalty")
    eta = math.sqrt(params.scalar("bess_efficiency"))
    dur = params.scalar("bess_duration_h")
    reserve = 1.0 + 1.0 / params.scalar("N_resilience")
    deploy_pv = params.scalar("Deploy_max_PV")
    nfc_price = params.scalar("nfc_yen_per_kWh") if options.certificates else 0.0
    pv_share_max = params.scalar("pv_share_max", None)
    cgs_share_max = params.scalar("cgs_share_max", None)
    shell = {r: params.region("shell_premium_yen_per_kW", r, 0.0) for r in regs}
    comm = {r: params.region("comm_cost_yen_per_kWh", r, 0.0) for r in regs}
    tariff = {r: params.region("tariff_yen_per_kWh", r) for r in regs}
    re_max = {r: params.region("re_max_GW", r) for r in regs}
    cf_avg = {r: ts.cf_avg(r) for r in regs}
    tiers = _tiers(params)
    n_tiers = max(1, len(tiers))
    vol_factor = (params.path("ppa_volume_factor") if params.has_path("ppa_volume_factor")
                  else {t: 1.0 for t in years})
    ppa_price: Dict[Tuple[str, int, int], float] = {}
    ppa_vol: Dict[Tuple[str, int, int], float] = {}
    for r in regs:
        base_p = params.region("ppa_yen_per_kWh", r)
        re_pot = re_max[r] * cf_avg[r] * 8.76
        mult = policy.ppa_tier_multiplier.get(r, 1.0)
        for t in years:
            for i in range(n_tiers):
                if tiers:
                    sid, pr = tiers[i]
                    ppa_price[(r, i, t)] = base_p + pr
                    ppa_vol[(r, i, t)] = params.region(sid, r) * re_pot * vol_factor[t] * mult
                else:
                    ppa_price[(r, i, t)] = base_p
                    ppa_vol[(r, i, t)] = math.inf
    theta = policy.operator_burden_share
    g_cost = policy.reinforcement_cost_yen_per_kW
    pt = options.carbon_pass_through

    # demand monotonicity is needed by the no-relocation rule
    if siting in ("endogenous", "metro_only"):
        for t in years:
            if t - 1 in total and total[t] < total[t - 1] - 1e-12:
                raise ValueError(f"siting {siting} needs a non-decreasing demand path ({t})")

    prob = pulp.LpProblem(f"DC_DECOM_v2_{sc.name}_{options.time_mode}", pulp.LpMinimize)

    def var(name: str, ub: Optional[float] = None) -> pulp.LpVariable:
        return pulp.LpVariable(name, lowBound=0.0, upBound=ub)

    def con(e: _Lin, sense: int, rhs: float = 0.0) -> None:
        if not e.terms:
            lhs = e.const
            ok = {LE: lhs <= rhs + 1e-9, GE: lhs >= rhs - 1e-9,
                  EQ: abs(lhs - rhs) <= 1e-9}[sense]
            if not ok:
                raise ValueError(f"constant constraint violated: {lhs} vs {rhs}")
            return
        prob.addConstraint(pulp.LpConstraint(
            pulp.LpAffineExpression(e.terms, constant=e.const), sense=sense, rhs=rhs))

    # ------------------------------------------------------------ capacity
    cap: Dict[Tuple[str, str, int], pulp.LpVariable] = {}
    new: Dict[Tuple[str, str, int], pulp.LpVariable] = {}
    for k in COOLING + tuple(supply):
        for r in regs:
            for t in years:
                ub = None
                if t in CALIBRATION_YEARS and k in ("dlc", "imm", "pv", "bess"):
                    ub = 0.0
                elif k == "pv":
                    ub = deploy_pv
                cap[(k, r, t)] = var(f"cap_{k}_{r}_{t}")
                new[(k, r, t)] = var(f"new_{k}_{r}_{t}", ub)
    if realization is not None:
        _bound_additions(new, realization)

    def prev_cap(k: str, r: str, t: int) -> Any:
        if t - 1 in years:
            return cap[(k, r, t - 1)]
        if st is not None:
            return float(st.capacities.get(k, {}).get(r, 0.0))
        return itr_2024[r] / u_cap if k == "air" else 0.0

    for k in COOLING + tuple(supply):
        for r in regs:
            for t in years:
                con(_lin((cap[(k, r, t)], 1.0), (new[(k, r, t)], -1.0)).add(prev_cap(k, r, t), -1.0),
                    EQ)

    # -------------------------------------------------------------- siting
    itr: Dict[Tuple[str, int], Any] = {}
    if realization is not None:
        given = {(str(r), int(t)): float(x) for r, t, x in realization.itr}
        miss = [(r, t) for r in regs for t in years if (r, t) not in given]
        if miss:
            raise ValueError(f"realization misses the IT load of {miss[:3]}")
        for r in regs:
            for t in years:
                itr[(r, t)] = given[(r, t)]
    elif siting == "growth_split":
        if st is not None:
            raise ValueError("growth_split siting needs start_year 2024")
        split = dict(sc.growth_split or {})
        if abs(sum(split.values()) - 1.0) > 1e-9 or not set(split) <= set(regs):
            raise ValueError("growth_split must sum to 1 over known regions")
        t0 = CALIBRATION_YEARS[-1]
        for r in regs:
            for t in years:
                if t <= t0:
                    itr[(r, t)] = total[t] * sh0[r]
                else:
                    itr[(r, t)] = total[t0] * sh0[r] + split.get(r, 0.0) * (total[t] - total[t0])
    elif siting in ("base_shares", "fixed_shares"):
        if siting == "base_shares":
            shares = {(r, t): sh0[r] for r in regs for t in years}
        else:
            shares = _fixed_share_path(sh0, dict(sc.fixed_shares or {}), regs, years)
        for r in regs:
            for t in years:
                itr[(r, t)] = total[t] * shares[(r, t)]
    else:
        for t in years:
            for r in regs:
                if t in CALIBRATION_YEARS:
                    itr[(r, t)] = total[t] * sh0[r]
                elif siting == "metro_only" and r not in METRO:
                    itr[(r, t)] = (total[CALIBRATION_YEARS[-1]] * sh0[r] if st is None
                                   or st.year < CALIBRATION_YEARS[-1] else float(st.itr[r]))
                else:
                    itr[(r, t)] = var(f"itr_{r}_{t}")

    def prev_itr(r: str, t: int) -> Any:
        if t - 1 in years:
            return itr[(r, t - 1)]
        if st is not None:
            return float(st.itr[r])
        return None

    if siting in ("endogenous", "metro_only"):
        for t in years:
            if t in CALIBRATION_YEARS:
                continue
            con(_lin(*[(itr[(r, t)], 1.0) for r in regs]), EQ, total[t])
            lam = policy.metro_floor.get(t)
            if lam is not None and siting == "endogenous":
                con(_lin(*[(itr[(r, t)], 1.0) for r in regs if r in METRO]), GE, lam * total[t])
            kap = policy.kanto_floor.get(t)
            if kap is not None and "Kanto" in regs:
                con(_lin((itr[("Kanto", t)], 1.0)), GE, kap * total[t])
            for r in regs:
                pr = prev_itr(r, t)
                if pr is not None:
                    con(_lin((itr[(r, t)], 1.0), (pr, -1.0)), GE)

    build_it: Dict[Tuple[str, int], _Lin] = {}
    for r in regs:
        for t in years:
            pr = prev_itr(r, t)
            if pr is None:
                build_it[(r, t)] = _Lin()
            else:
                e = _lin((itr[(r, t)], 1.0), (pr, -1.0))
                if not e.terms:          # exogenous siting: max(0, delta)
                    e = _lin((max(0.0, e.const), 1.0))
                build_it[(r, t)] = e

    # --------------------------------------------------- IT by technology
    it: Dict[Tuple[str, str, int], pulp.LpVariable] = {}
    for r in regs:
        for t in years:
            for k in COOLING:
                it[(k, r, t)] = var(f"it_{k}_{r}_{t}")
                con(_lin((it[(k, r, t)], 1.0), (cap[(k, r, t)], -u_cap)), LE)
            con(_lin(*[(it[(k, r, t)], 1.0) for k in COOLING]).add(itr[(r, t)], -1.0), EQ)

    # -------------------------------------------------- liquid deployment
    for r in regs:
        for t in years:
            if t not in CALIBRATION_YEARS:
                con(_lin((new[("dlc", r, t)], 1.0), (new[("imm", r, t)], 1.0)), LE,
                    float(paths.liquid_deploy_cap[t]))

    # ------------------------------------------------------------- slices
    g: Dict[Tuple[str, int, int], pulp.LpVariable] = {}
    pv: Dict[Tuple[str, int, int], pulp.LpVariable] = {}
    cgs: Dict[Tuple[str, int, int], pulp.LpVariable] = {}
    ch: Dict[Tuple[str, int, int], pulp.LpVariable] = {}
    dis: Dict[Tuple[str, int, int], pulp.LpVariable] = {}
    soc: Dict[Tuple[str, int, int], pulp.LpVariable] = {}
    curt: Dict[Tuple[str, int, int], pulp.LpVariable] = {}
    spill: Dict[Tuple[str, int, int], pulp.LpVariable] = {}
    L: Dict[Tuple[str, int, int], _Lin] = {}
    ppa: Dict[Tuple[str, int, int], pulp.LpVariable] = {}   # TWh (annual) | GW (hourly)
    ppa_E: Dict[Tuple[str, int, int], _Lin] = {}
    for r in regs:
        for t in years:
            for i in range(n_tiers):
                vol = ppa_vol[(r, i, t)]
                if hourly:
                    ub = None if math.isinf(vol) else vol / (cf_avg[r] * 8.76)
                    ppa[(r, i, t)] = var(f"ppacap_{r}_{i}_{t}", ub)
                    ppa_E[(r, i, t)] = _lin((ppa[(r, i, t)], cf_avg[r] * 8.76))
                else:
                    ppa[(r, i, t)] = var(f"ppa_{r}_{i}_{t}", None if math.isinf(vol) else vol)
                    ppa_E[(r, i, t)] = _lin((ppa[(r, i, t)], 1.0))
            cf = ts.cf[r]
            for h in H:
                key = (r, t, h)
                g[key] = var(f"g_{r}_{t}_{h}")
                if cf[h] > 0:
                    pv[key] = var(f"pv_{r}_{t}_{h}")
                    con(_lin((pv[key], 1.0), (cap[("pv", r, t)], -cf[h])), LE)
                cgs[key] = var(f"cgs_{r}_{t}_{h}")
                ch[key] = var(f"ch_{r}_{t}_{h}")
                dis[key] = var(f"dis_{r}_{t}_{h}")
                soc[key] = var(f"soc_{r}_{t}_{h}")
                curt[key] = var(f"curt_{r}_{t}_{h}")
                L[key] = _lin(*[(it[(k, r, t)], u_e * float(sp[(r, k)][h])) for k in COOLING])
                con(_lin((cgs[key], 1.0), (cap[("cgs", r, t)], -1.0)), LE)
                con(_lin((ch[key], 1.0), (cap[("bess", r, t)], -1.0 / dur)), LE)
                con(_lin((dis[key], 1.0), (cap[("bess", r, t)], -1.0 / dur)), LE)
                con(_lin((soc[key], 1.0), (cap[("bess", r, t)], -1.0)), LE)
            for h in H:
                key = (r, t, h)
                hp = ts.slices[h].prev
                con(_lin((soc[key], 1.0), (soc[(r, t, hp)], -1.0), (ch[key], -eta),
                         (dis[key], 1.0 / eta)), EQ)
                bal = _lin((g[key], 1.0), (cgs[key], 1.0), (dis[key], 1.0), (curt[key], 1.0),
                           (ch[key], -1.0)).add(L[key], -1.0)
                if key in pv:
                    bal.add(pv[key], 1.0)
                if hourly:
                    spill[key] = var(f"spill_{r}_{t}_{h}")
                    deliv = _lin(*[(ppa[(r, i, t)], cf[h]) for i in range(n_tiers)])
                    bal.add(deliv, 1.0).add(spill[key], -1.0)
                    con(_lin((spill[key], 1.0)).add(deliv, -1.0), LE)
                con(bal, EQ)

    def annual(x: Dict[Tuple[str, int, int], pulp.LpVariable], r: str, t: int) -> _Lin:
        return _lin(*[(x[(r, t, h)], w[h] / 1000.0) for h in H if (r, t, h) in x])

    G = {(r, t): annual(g, r, t) for r in regs for t in years}
    PV_E = {(r, t): annual(pv, r, t) for r in regs for t in years}
    CGS_E = {(r, t): annual(cgs, r, t) for r in regs for t in years}
    CURT_E = {(r, t): annual(curt, r, t) for r in regs for t in years}
    L_E = {}
    for r in regs:
        for t in years:
            e = _Lin()
            for h in H:
                e.add(L[(r, t, h)], w[h] / 1000.0)
            L_E[(r, t)] = e

    # ----------------------------------------------- PPA, certificates, em
    nfc: Dict[Tuple[str, int], pulp.LpVariable] = {}
    em: Dict[Tuple[str, int], _Lin] = {}
    for r in regs:
        for t in years:
            ppa_sum = _Lin()
            for i in range(n_tiers):
                ppa_sum.add(ppa_E[(r, i, t)], 1.0)
            if hourly:
                con(_lin((cap[("pv", r, t)], 1.0), *[(ppa[(r, i, t)], 1.0)
                                                    for i in range(n_tiers)]), LE, re_max[r])
                grid_net = _Lin().add(G[(r, t)], 1.0)
            else:
                con(_Lin().add(ppa_sum, 1.0).add(G[(r, t)], -1.0), LE)
                con(_lin((cap[("pv", r, t)], 1.0)).add(ppa_sum, 1.0 / (cf_avg[r] * 8.76)), LE,
                    re_max[r])
                grid_net = _Lin().add(G[(r, t)], 1.0).add(ppa_sum, -1.0)
            if options.certificates:
                nfc[(r, t)] = var(f"nfc_{r}_{t}")
                con(_lin((nfc[(r, t)], 1.0)).add(grid_net, -1.0), LE)
                grid_net.add(nfc[(r, t)], -1.0)
            em[(r, t)] = _Lin().add(grid_net, ef[(r, t)]).add(CGS_E[(r, t)], ef_fossil)
            if pv_share_max is not None:
                con(_Lin().add(PV_E[(r, t)], 1.0).add(L_E[(r, t)], -pv_share_max), LE)
            if cgs_share_max is not None:
                con(_Lin().add(CGS_E[(r, t)], 1.0).add(L_E[(r, t)], -cgs_share_max), LE)

    # ---------------------------------------------------------- resilience
    for r in regs:
        for t in years:
            if options.resilience == "eq":
                for h in nondom[r]:
                    e = _lin(*[(cap[(k, r, t)], 1.0) for k in ("cgs", "gen")
                               if k in options.backup_techs and k in supply])
                    for k in COOLING:
                        e.add(it[(k, r, t)], -reserve * float(sp[(r, k)][h]) / u_cap)
                    con(e, GE)
            else:
                con(_lin((cap[("cgs", r, t)], 1.0)).add(itr[(r, t)], -0.05), GE)

    # ------------------------------------------------------------- CO2 cap
    if co2_cap is not None:
        for t in years:
            e = _Lin()
            for r in regs:
                e.add(em[(r, t)], 1.0)
            con(e, LE, co2_cap[t])
            f_floor = options.co2_floor_share
            if f_floor is not None and t > CALIBRATION_YEARS[-1]:
                con(_Lin().add(e, 1.0), GE, f_floor * co2_cap[t])

    # ------------------------------------------------------- grid / peaks
    constrained = ([r for r in regs if r not in METRO and r in policy.conn_capacity]
                   if options.grid_constraints else [])
    peak: Dict[Tuple[str, int], pulp.LpVariable] = {}
    dpeak: Dict[Tuple[str, int], _Lin] = {}
    over: Dict[Tuple[str, int], pulp.LpVariable] = {}
    conn_ref: Dict[str, Tuple[Any, float, float]] = {}
    first_conn = CALIBRATION_YEARS[-1] + 1
    for r in constrained:
        for t in years:
            if t in CALIBRATION_YEARS:
                # all-air fleet at base shares: the peak is data, not a choice
                # (a free calibration-year peak would loosen the cumulative
                # constraint and shrink the costed increments)
                peak[(r, t)] = max(_val(itr[(r, t)]) * u_e * float(sp[(r, "air")][h])
                                   for h in H)
            else:
                peak[(r, t)] = var(f"peak_{r}_{t}")
                for h in nondom[r]:
                    con(_lin((peak[(r, t)], 1.0)).add(L[(r, t, h)], -1.0), GE)
            if t - 1 in years:
                pprev: Any = peak[(r, t - 1)]
            elif st is not None:
                if r not in st.peak:
                    raise ValueError(f"initial_state has no peak for {r}")
                pprev = float(st.peak[r])
            else:
                pprev = None
            if pprev is not None:
                e = _lin((peak[(r, t)], 1.0), (pprev, -1.0))
                if e.terms:
                    con(e, GE)
                else:                    # calibration years: data
                    e = _lin((max(0.0, e.const), 1.0))
                dpeak[(r, t)] = e
            if options.connection_overrun and t >= first_conn:
                over[(r, t)] = var(f"over_{r}_{t}")
        if CALIBRATION_YEARS[-1] in years:
            conn_ref[r] = (peak[(r, CALIBRATION_YEARS[-1])], 0.0, 0.0)
        elif st is not None and st.year >= CALIBRATION_YEARS[-1]:
            cum = st.cumulative
            conn_ref[r] = (float(st.peak[r]), float(cum.get("conn_used", {}).get(r, 0.0)),
                           float(cum.get("conn_capacity", {}).get(r, 0.0)))
        if r in conn_ref:
            ref, used0, cap0 = conn_ref[r]
            acc = _Lin()
            for t in years:
                if t < first_conn:
                    continue
                acc.add(float(policy.conn_capacity[r].get(t, 0.0)), 1.0)
                if (r, t) in over:
                    acc.add(over[(r, t)], 1.0)
                e = _lin((peak[(r, t)], 1.0), (ref, -1.0), (used0 - cap0, 1.0)).add(acc, -1.0)
                con(e, LE)
            _pin_peak_before_reinforcement(prob, r, years, peak, L, nondom[r], policy, over,
                                           _val(ref) - used0 + cap0, st)

    # ------------------------------------------------------ policy rules
    for t in years:
        for r in regs:
            if t in policy.pue_fleet_max:
                e = _lin(*[(it[(k, r, t)], pa[(r, k)]) for k in COOLING])
                con(e.add(itr[(r, t)], -policy.pue_fleet_max[t]), LE)
            if t in policy.pue_new_max:
                con(_lin(*[(new[(k, r, t)], pa[(r, k)] - policy.pue_new_max[t])
                           for k in COOLING]), LE)
            if r in policy.liquid_fleet_max and t > CALIBRATION_YEARS[-1]:
                con(_lin(*[(it[(k, r, t)], 1.0) for k in LIQUID])
                    .add(itr[(r, t)], -policy.liquid_fleet_max[r]), LE)
            if t in policy.liquid_floor_new:
                phi = policy.liquid_floor_new[t]
                con(_lin(*[(new[(k, r, t)], (1.0 if k in LIQUID else 0.0) - phi)
                           for k in COOLING]), GE)
    sub_cum: Dict[Tuple[str, int], _Lin] = {}
    for r in regs:
        acc = _Lin()
        if st is not None:
            acc.add(float(st.cumulative.get("subsidized_build", {}).get(r, 0.0)), 1.0)
        for t in years:
            if policy.subsidy(r, t) > 0:
                acc.add(build_it[(r, t)], 1.0)
            sub_cum[(r, t)] = _Lin().add(acc, 1.0)
            psi = policy.ppa_floor_subsidized
            if psi is not None and (acc.terms or acc.const > 0):
                e = _Lin()
                for i in range(n_tiers):
                    e.add(ppa_E[(r, i, t)], 1.0)
                e.add(acc, -psi * pa[(r, "dlc")] * 8.76)
                con(e, GE)

    # ----------------------------------------------------------- objective
    obj = _Lin()
    tiebreak = _Lin()     # tie-break weights only (air recourse, connected peak)
    for t in years:
        dt = disc[t]
        bc = _bess_capex(params, t)
        for r in regs:
            for k in LIQUID:
                obj.add(new[(k, r, t)], dt * prem[k])
                obj.add(cap[(k, r, t)], dt * prem[k] * om)
            obj.add(new[("bess", r, t)], dt * bc)
            obj.add(cap[("bess", r, t)], dt * bc * om)
            for k in ("cgs", "gen"):
                if k in supply:
                    obj.add(new[(k, r, t)], dt * unit_capex[k])
                    obj.add(cap[(k, r, t)], dt * unit_capex[k] * om)
            obj.add(G[(r, t)], dt * tariff[r])
            for i in range(n_tiers):
                obj.add(ppa_E[(r, i, t)], dt * (ppa_price[(r, i, t)]
                                                - (0.0 if hourly else tariff[r])))
            if (r, t) in nfc:
                obj.add(nfc[(r, t)], dt * nfc_price)
            obj.add(PV_E[(r, t)], dt * pv_lcoe)
            obj.add(CGS_E[(r, t)], dt * cgs_fuel_e)
            obj.add(em[(r, t)], dt * pt * price[t] * 1e-3)
            obj.add(CURT_E[(r, t)], dt * penalty)
            if (r, t) in dpeak and policy.is_reinforced(r, t):
                obj.add(dpeak[(r, t)], dt * theta * g_cost * 1e-3)
            if (r, t) in over:
                obj.add(over[(r, t)], dt * g_cost * 1e-3)
            net = shell[r] - policy.subsidy(r, t)
            if net != 0.0:
                obj.add(build_it[(r, t)], dt * net * 1e-3 / u_cap)
            if comm[r] != 0.0:
                obj.add(itr[(r, t)], dt * comm[r] * u_e * 8.76)
            if realization is not None and realization.air_tiebreak_yen_per_kW > 0.0:
                tiebreak.add(new[("air", r, t)],
                             dt * realization.air_tiebreak_yen_per_kW * 1e-3)
            if isinstance(peak.get((r, t)), pulp.LpVariable):
                tiebreak.add(peak[(r, t)], dt * PEAK_TIEBREAK_YEN_PER_KW * 1e-3)
    obj.add(tiebreak, 1.0)
    prob += pulp.LpAffineExpression(obj.terms, constant=obj.const)

    data = dict(sp=sp, pa=pa, nondom=nondom, u=u, u_cap=u_cap, u_e=u_e, total=total, sh0=sh0,
                ef=ef, ef_fossil=ef_fossil, E_2024=E_2024, co2_cap=co2_cap, price=price,
                disc=disc, prem=prem, supply=supply, unit_capex=unit_capex, om=om,
                pv_lcoe=pv_lcoe, cgs_fuel_e=cgs_fuel_e, penalty=penalty, eta=eta, dur=dur,
                reserve=reserve, nfc_price=nfc_price, shell=shell, comm=comm, tariff=tariff,
                re_max=re_max, cf_avg=cf_avg, n_tiers=n_tiers, ppa_price=ppa_price,
                ppa_vol=ppa_vol, theta=theta, g_cost=g_cost, pt=pt, constrained=constrained,
                conn_ref=conn_ref, hourly=hourly, tiebreak=tiebreak)
    v: Dict[str, Dict[Any, Any]] = dict(
        cap=cap, new=new, it=it, itr=itr, build_it=build_it, g=g, pv=pv, cgs=cgs, ch=ch,
        dis=dis, soc=soc, curt=curt, spill=spill, L=L, ppa=ppa, ppa_E=ppa_E, nfc=nfc, em=em,
        G=G, L_E=L_E, peak=peak, dpeak=dpeak, over=over, sub_cum=sub_cum)
    return Handle(prob=prob, options=options, scenario=sc, siting=siting, params=params,
                  paths=paths, policy=policy, years=years, regions=regs, ts=ts,
                  initial_state=st, data=data, v=v,
                  build_seconds=time.perf_counter() - t_start)


# ----------------------------------------------------------------------
# Solve and extract
# ----------------------------------------------------------------------

def solve(handle: Handle, msg: bool = False) -> Result:
    t0 = time.perf_counter()
    handle.prob.solve(pulp.PULP_CBC_CMD(msg=1 if msg else 0, threads=1))
    solve_s = time.perf_counter() - t0
    status = pulp.LpStatus[handle.prob.status]
    t1 = time.perf_counter()
    flows = _extract(handle)
    meta = _meta(handle)
    tb = handle.data["tiebreak"].value()
    meta["objective_yen"] = (float(pulp.value(handle.prob.objective) or 0.0) - tb) * GYEN
    meta["tiebreak_yen"] = tb * GYEN
    metrics: Dict[str, Any] = {}
    tables: Dict[str, pd.DataFrame] = {}
    if status == "Optimal":
        metrics, tables = accounting.evaluate(flows, meta)
    timings = {"build_s": handle.build_seconds, "solve_s": solve_s,
               "eval_s": time.perf_counter() - t1}
    return Result(status=status, flows=flows, metrics=metrics, timings=timings,
                  n_vars=len(handle.prob.variables()), n_cons=len(handle.prob.constraints),
                  meta=meta, tables=tables)


def run(options: ModelOptions, scenario: Union[str, Scenario], params: ParamSet,
        **kwargs) -> Result:
    """build + solve."""
    return solve(build(options, scenario, params, **kwargs))


def _meta(h: Handle) -> Dict[str, Any]:
    dd = h.data
    return {
        "scenario": h.scenario.name,
        "siting": h.siting,
        "options": asdict(h.options),
        "time_mode": h.options.time_mode,
        "ppa_matching": h.options.ppa_matching,
        "years": list(h.years),
        "regions": list(h.regions),
        "n_slices": len(h.ts.slices),
        "n_tiers": dd["n_tiers"],
        "E_2024_Mt": dd["E_2024"],
        "co2_cap_Mt": (None if dd["co2_cap"] is None
                       else {str(t): float(x) for t, x in dd["co2_cap"].items()}),
        "reserve_factor": dd["reserve"],
        "eta": dd["eta"],
        "bess_duration_h": dd["dur"],
        "utilization": dd["u"],
        "theta": dd["theta"],
        "pass_through": dd["pt"],
        "constrained_regions": list(dd["constrained"]),
        "uses_dev_params": h.params.has_dev,
        "params_label": h.params.label,
        "params_fingerprint": h.params.fingerprint(),
    }


def _extract(h: Handle) -> Dict[str, pd.DataFrame]:
    dd, v = h.data, h.v
    n_tiers = dd["n_tiers"]
    rows: List[Dict[str, Any]] = []
    srows: Dict[str, List[Any]] = {c: [] for c in (
        "year", "region", "slice", "season", "hour", "weight_h", "temperature_C", "cf",
        "pue_air", "pue_dlc", "pue_imm", "L_GW", "g_GW", "pv_GW", "cgs_GW", "ch_GW",
        "dis_GW", "soc_GWh", "curt_GW", "ppa_GW", "spill_GW")}
    first_conn = CALIBRATION_YEARS[-1] + 1
    for t in h.years:
        for r in h.regions:
            row: Dict[str, Any] = {"year": t, "region": r, "metro": r in METRO,
                                   "disc": dd["disc"][t]}
            row["itr_GW"] = _val(v["itr"][(r, t)])
            row["build_IT_GW"] = _val(v["build_it"][(r, t)])
            for k in COOLING:
                row[f"it_{k}_GW"] = _val(v["it"][(k, r, t)])
            for k in COOLING + ("pv", "bess", "cgs", "gen"):
                unit = "GWh" if k == "bess" else "GW"
                ck = (k, r, t)
                row[f"cap_{k}_{unit}"] = _val(v["cap"][ck]) if ck in v["cap"] else 0.0
                row[f"new_{k}_{unit}"] = _val(v["new"][ck]) if ck in v["new"] else 0.0
            for i in range(n_tiers):
                row[f"ppa_t{i + 1}_TWh"] = _val(v["ppa_E"][(r, i, t)])
                row[f"ppa_cap_t{i + 1}_GW"] = (_val(v["ppa"][(r, i, t)]) if dd["hourly"]
                                               else math.nan)
                row[f"ppa_price_t{i + 1}"] = dd["ppa_price"][(r, i, t)]
                row[f"ppa_volume_t{i + 1}_TWh"] = dd["ppa_vol"][(r, i, t)]
            row["nfc_TWh"] = _val(v["nfc"][(r, t)]) if (r, t) in v["nfc"] else 0.0
            con = r in dd["constrained"]
            row["peak_conn_GW"] = _val(v["peak"][(r, t)]) if con else math.nan
            row["dpeak_GW"] = _val(v["dpeak"][(r, t)]) if (r, t) in v["dpeak"] else 0.0
            row["over_GW"] = _val(v["over"][(r, t)]) if (r, t) in v["over"] else 0.0
            row["conn_offered_GW"] = (float(h.policy.conn_capacity[r].get(t, 0.0))
                                      if con and t >= first_conn else 0.0)
            row["reinforced"] = 1.0 if (con and h.policy.is_reinforced(r, t)) else 0.0
            row["subsidized_build_cum_GW"] = _val(v["sub_cum"][(r, t)])
            for k in COOLING:
                row[f"pue_ann_{k}"] = dd["pa"][(r, k)]
            row.update({
                "tariff": dd["tariff"][r], "nfc_price": dd["nfc_price"], "pv_lcoe": dd["pv_lcoe"],
                "cgs_fuel_e": dd["cgs_fuel_e"], "prem_dlc": dd["prem"]["dlc"],
                "prem_imm": dd["prem"]["imm"], "bess_capex": _bess_capex(h.params, t),
                "cgs_capex": dd["unit_capex"]["cgs"], "gen_capex": dd["unit_capex"]["gen"],
                "om_rate": dd["om"], "curt_penalty": dd["penalty"],
                "carbon_price": dd["price"][t], "ef_grid": dd["ef"][(r, t)],
                "ef_fossil": dd["ef_fossil"], "reinf_cost": dd["g_cost"], "theta": dd["theta"],
                "pass_through": dd["pt"], "shell_premium": dd["shell"][r],
                "comm_rate": dd["comm"][r], "subsidy_rate": h.policy.subsidy(r, t), "u_cap": dd["u_cap"],
                "u_energy": dd["u_e"], "re_max": dd["re_max"][r], "cf_avg": dd["cf_avg"][r],
            })
            rows.append(row)
            cf = h.ts.cf[r]
            for s in h.ts.slices:
                key = (r, t, s.index)
                srows["year"].append(t)
                srows["region"].append(r)
                srows["slice"].append(s.index)
                srows["season"].append(s.season)
                srows["hour"].append(s.hour)
                srows["weight_h"].append(s.weight_h)
                srows["temperature_C"].append(h.ts.temperature[r][s.index])
                srows["cf"].append(cf[s.index])
                for k in COOLING:
                    srows[f"pue_{k}"].append(float(dd["sp"][(r, k)][s.index]))
                srows["L_GW"].append(v["L"][key].value())
                for name, fam in (("g_GW", "g"), ("pv_GW", "pv"), ("cgs_GW", "cgs"),
                                  ("ch_GW", "ch"), ("dis_GW", "dis"), ("soc_GWh", "soc"),
                                  ("curt_GW", "curt"), ("spill_GW", "spill")):
                    srows[name].append(_val(v[fam][key]) if key in v[fam] else 0.0)
                deliv = (sum(_val(v["ppa"][(r, i, t)]) * cf[s.index] for i in range(n_tiers))
                         if dd["hourly"] else 0.0)
                srows["ppa_GW"].append(deliv)
    annual = pd.DataFrame(rows)
    # cumulative connection use / offer (non-metro regions with a constraint)
    used = []
    offered = []
    for _, row in annual.iterrows():
        r, t = row["region"], int(row["year"])
        if r in dd["conn_ref"] and t >= first_conn:
            ref, used0, cap0 = dd["conn_ref"][r]
            sel = annual[(annual["region"] == r) & (annual["year"] >= first_conn)
                         & (annual["year"] <= t)]
            used.append(row["peak_conn_GW"] - _val(ref) + used0)
            offered.append(cap0 + float(sel["conn_offered_GW"].sum() + sel["over_GW"].sum()))
        else:
            used.append(0.0 if r in dd["constrained"] else math.nan)
            offered.append(0.0 if r in dd["constrained"] else math.nan)
    annual["conn_used_cum_GW"] = used
    annual["conn_offered_cum_GW"] = offered
    return {"annual": annual, "slices": pd.DataFrame(srows)}


def state_at(result: Result, year: int) -> State:
    """End-of-year state of a solved run, for a window starting in year + 1."""
    a = result.flows["annual"]
    a = a[a["year"] == year]
    if a.empty:
        raise ValueError(f"year {year} not in the result")
    caps: Dict[str, Dict[str, float]] = {}
    for k in COOLING + ("pv", "bess", "cgs", "gen"):
        col = f"cap_{k}_{'GWh' if k == 'bess' else 'GW'}"
        caps[k] = {str(r): float(x) for r, x in zip(a["region"], a[col])}
    itr = {str(r): float(x) for r, x in zip(a["region"], a["itr_GW"])}
    con = a[a["peak_conn_GW"].notna()]
    peak = {str(r): float(x) for r, x in zip(con["region"], con["peak_conn_GW"])}
    cum = {
        "conn_capacity": {str(r): float(0.0 if np.isnan(x) else x)
                          for r, x in zip(con["region"], con["conn_offered_cum_GW"])},
        "conn_used": {str(r): float(0.0 if np.isnan(x) else x)
                      for r, x in zip(con["region"], con["conn_used_cum_GW"])},
        "subsidized_build": {str(r): float(x)
                             for r, x in zip(a["region"], a["subsidized_build_cum_GW"])},
    }
    return State(year=int(year), capacities=caps, itr=itr, peak=peak, cumulative=cum)
