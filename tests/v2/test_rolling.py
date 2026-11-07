"""Rolling execution of the A/B/C designs: non-anticipativity,
fairness, realization and stitching. Frozen v2 parameters in the annual time
mode (fast; these solves are not results)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest
import yaml

from dc_decom_v2 import rolling, runner, state_at
from dc_decom_v2.config import COOLING, ModelOptions, Paths

REPO = Path(__file__).resolve().parents[2]
RULES = yaml.safe_load((REPO / "config" / "abc_rules.yaml").read_text(encoding="utf-8"))
V2 = "data/parameters_v2"
OPTS = ModelOptions(time_mode="annual", certificates=True)
BASE_PATH = "c-base_d-base_l-base"
SEL = (BASE_PATH, "c-high_d-high_l-slow", "c-low_d-low_l-slow")
UNIT = {k: ("GWh" if k == "bess" else "GW") for k in COOLING + ("pv", "bess", "cgs", "gen")}


def perturbed(p: rolling.ActualPath, t_k: int) -> rolling.ActualPath:
    """Same values before t_k, different values from t_k."""
    def bump(x: Dict[int, float], f: float, add: float = 0.0) -> Dict[int, float]:
        return {t: (v * f + add if t >= t_k else v) for t, v in x.items()}
    q = Paths(carbon_price=bump(p.paths.carbon_price, 1.0, 5000.0),
              demand_TWh=bump(p.paths.demand_TWh, 1.15),
              liquid_deploy_cap=bump(p.paths.liquid_deploy_cap, 0.5))
    return rolling.ActualPath(f"{p.name}+{t_k}", p.carbon, p.demand, p.liquid, q)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    ps = runner.load_params(V2)
    from dc_decom_v2.scenarios import default_paths
    base = rolling.RollingBase(existing_dir=V2, forecast=default_paths(ps), options=OPTS)
    designs = rolling.designs_from_rules(RULES, ps, diagnostics=False)
    paths = {p.name: p for p in rolling.actual_paths(ps)}
    solve = rolling.default_solver(workers=1, use_cache=True,
                                   cache_dir=tmp_path_factory.mktemp("rolling_cache"))
    return {"ps": ps, "base": base, "designs": designs, "paths": paths, "solve": solve}


@pytest.fixture(scope="module")
def runs(env) -> Dict[str, rolling.RollingRun]:
    d = env["designs"]
    calib, rr = rolling.run_designs(env["base"], [d["A0"], d["A"], d["B"], d["C"]],
                                    [env["paths"][n] for n in SEL], RULES, solve=env["solve"])
    out = {f"{r.design.name}|{r.path.name}": r for r in rr}
    out["_calib"] = calib  # type: ignore[assignment]
    return out


@pytest.fixture(scope="module")
def switching_design() -> rolling.Design:
    """A C-type design whose rules change the policy (the frozen C saturates)."""
    s = rolling.Settings(phi=0.0, t_g=2035, psi=0.0, subsidy="rule", pstar=1e9,
                         ppa_multiplier=1.5)
    return rolling.Design("C-test", s, adaptive=True)


@pytest.fixture(scope="module")
def perturbation_runs(env, switching_design) -> Dict[str, rolling.RollingRun]:
    p = env["paths"][BASE_PATH]
    paths = [p, perturbed(p, 2030), perturbed(p, 2035)]
    # R2 at a zero threshold so that the test design switches: the physical
    # connection usage stays below the frozen 0.9 on these paths.
    rules = copy.deepcopy(RULES)
    rules["thresholds"]["connection_usage"]["value"] = 0.0
    _, rr = rolling.run_designs(env["base"], [env["designs"]["B"], switching_design], paths,
                                rules, solve=env["solve"])
    return {f"{r.design.name}|{r.path.name}": r for r in rr}


# ------------------------------------------------------------------ (a)

def test_stage1_plans_identical_across_all_paths(env, runs):
    calib = runs["_calib"]
    s25 = state_at(calib, 2025)
    for name in ("A", "B"):
        pol = rolling.policy_for(env["designs"][name].settings, env["ps"],
                                 env["base"].forecast.carbon_price)
        hashes = {rolling.plan_spec(env["base"], p, 2026, s25, pol, "x").config_hash()
                  for p in env["paths"].values()}
        assert len(env["paths"]) == 18 and len(hashes) == 1, name
    for name in ("A", "B"):
        first = [runs[f"{name}|{p}"].planned[0] for p in SEL]
        for df in first[1:]:
            pd.testing.assert_frame_equal(first[0], df)


@pytest.mark.parametrize("t_k", [2030, 2035])
@pytest.mark.parametrize("design", ["B", "C-test"])
def test_later_values_do_not_change_earlier_decisions(perturbation_runs, design, t_k):
    a = perturbation_runs[f"{design}|{BASE_PATH}"]
    b = perturbation_runs[f"{design}|{BASE_PATH}+{t_k}"]
    assert a.ok and b.ok
    stages = [t for t in rolling.REVIEW_YEARS if t <= t_k]
    for j in range(len(stages)):
        pd.testing.assert_frame_equal(a.planned[j], b.planned[j])
        assert a.plan_specs[j].config_hash() == b.plan_specs[j].config_hash()
    ra, rb = a.flows["annual"], b.flows["annual"]
    pd.testing.assert_frame_equal(ra[ra["year"] < t_k].reset_index(drop=True),
                                  rb[rb["year"] < t_k].reset_index(drop=True))
    # the perturbation does matter afterwards
    assert not ra[ra["year"] >= t_k]["itr_GW"].reset_index(drop=True).equals(
        rb[rb["year"] >= t_k]["itr_GW"].reset_index(drop=True))


@pytest.mark.parametrize("t_k", [2030, 2035])
def test_switches_use_only_earlier_indicators(perturbation_runs, t_k):
    a = perturbation_runs[f"C-test|{BASE_PATH}"]
    b = perturbation_runs[f"C-test|{BASE_PATH}+{t_k}"]
    sa = [s for s in a.switches if s["review_year"] <= t_k]
    sb = [s for s in b.switches if s["review_year"] <= t_k]
    assert sa and sa == sb


def test_switches_change_the_next_stage(perturbation_runs):
    r = perturbation_runs[f"C-test|{BASE_PATH}"]
    changed = [s for s in r.switches if s["policy_changed"]]
    assert changed, "the test design should switch at least once"
    for s in changed:
        k = rolling.REVIEW_YEARS.index(s["review_year"])
        before = getattr(r.stage_settings[k - 1], s["setting"])
        after = getattr(r.stage_settings[k], s["setting"])
        assert before == s["before"] and after == s["after"]
    phi = {s["review_year"]: s["after"] for s in r.switches if s["rule"] == "R1"}
    for k, t in enumerate(rolling.REVIEW_YEARS[1:], start=1):
        assert r.policies[k].liquid_floor_new.get(t) == phi[t]


def test_a_and_b_never_switch(runs):
    for key, r in runs.items():
        if key.startswith(("A|", "B|", "A0|")):
            assert r.switches == []
            assert r.stage_settings == [r.design.settings] * 3


# ------------------------------------------------------------------ (b)

def test_fairness_non_policy_fields_equal(env, runs):
    for p in SEL:
        group = {n: runs[f"{n}|{p}"] for n in ("A0", "A", "B", "C")}
        for k in range(3):
            rolling.assert_fair({n: r.plan_specs[k] for n, r in group.items()},
                                outcome_ok=k > 0)
            rolling.assert_fair({n: r.realize_specs[k] for n, r in group.items()})
    s = runs[f"A|{BASE_PATH}"].plan_specs[0]
    with pytest.raises(AssertionError):
        rolling.assert_fair({"A": s, "X": runner.RunSpec(**{
            **{f: getattr(s, f) for f in s.__dataclass_fields__},
            "options": s.options.with_(certificates=False)})})


def test_frozen_designs():
    ps = runner.load_params(V2)
    d = rolling.designs_from_rules(RULES, ps)
    assert d["C"].settings == d["B"].settings and d["C"].adaptive
    assert not any(x.adaptive for n, x in d.items() if n != "C")
    assert set(d) == {"A0", "A", "B", "C", *RULES["diagnostic_B_variants"]}
    assert d["B-noPPAfloor"].settings.psi is None
    assert d["B-P10000"].settings.pstar == 10000.0


# ------------------------------------------------------------------ (c)

def test_base_path_realization_has_no_curtailment(runs):
    for n in ("A0", "A", "B", "C"):
        r = runs[f"{n}|{BASE_PATH}"]
        assert r.ok
        for w in r.windows:
            assert w.metrics["curtail_TWh"] <= 1e-9
        assert r.metrics["std_violations"] == 0


def test_realized_connected_peak_is_physical(runs):
    """The realized windows never connect ahead of the facility peak:
    the connected use equals the use from the physical peak, so the usage
    observed by R2 is physical."""
    for key, r in runs.items():
        if key == "_calib":
            continue
        ry = r.tables["region_year"]
        later = ry[ry["conn_offered_cum_GW"].notna() & (ry["year"] >= 2026)]
        assert len(later) > 0
        gap = (later["conn_used_cum_GW"] - later["conn_used_phys_cum_GW"]).abs().max()
        assert gap <= 1e-6, key
        for obs in r.observations:
            assert obs["conn_usage_last"] <= 1.0 + 1e-6


def test_overrun_reported(runs):
    for key, r in runs.items():
        if key == "_calib":
            continue
        a = r.flows["annual"]
        assert abs(r.metrics["over_GW_total"] - float(a["over_GW"].sum())) <= 1e-12
        ry = r.tables["region_year"]
        exp = float((ry["reinf_cost"] * ry["over_GW"] * 1e6 * ry["disc"]).sum())
        assert abs(r.metrics["over_cost_yen"] - exp) <= 1e-6 * max(1.0, exp)
        assert r.metrics["over_GW_total"] >= 0.0
    assert runs["A0|c-high_d-high_l-slow"].metrics["over_GW_total"] > 0.0


def test_realization_honours_the_plan(runs):
    for key in ("B|c-high_d-high_l-slow", "A|c-low_d-low_l-slow", f"A0|{BASE_PATH}"):
        r = runs[key]
        for k, (pdec, w) in enumerate(zip(r.planned, r.windows)):
            a = w.flows["annual"].set_index(["year", "region"])
            p = pdec.set_index(["year", "region"])
            for tech in ("bess", "pv", "cgs"):
                col = f"new_{tech}_{UNIT[tech]}"
                assert (a[col] - p[col].clip(lower=0)).abs().max() <= 1e-6
            assert (a["new_gen_GW"] - p["new_gen_GW"].clip(lower=0)).min() >= -1e-6
            liq = a["new_dlc_GW"] + a["new_imm_GW"]
            assert (liq - (p["new_dlc_GW"] + p["new_imm_GW"]).clip(lower=0)).max() <= 1e-6
            years = sorted({t for t, _ in a.index})
            cap = r.path.paths.liquid_deploy_cap
            assert all(liq.xs(t, level="year").max() <= cap[t] + 1e-6 for t in years)
            pol = r.policies[k]
            load = rolling.realized_itr(pdec, r.path.paths, r.states[k].itr,
                                        metro_floor=pol.metro_floor,
                                        kanto_floor=pol.kanto_floor)
            exp = pd.Series([load[(str(reg), int(t))] for t, reg in p.index], index=p.index)
            assert (a["itr_GW"] - exp).abs().max() <= 1e-9


def test_realized_load_never_relocates(runs):
    """No region falls below its previous load while national demand grows; when it
    falls, every region scales by the same factor."""
    for key, r in runs.items():
        if key == "_calib":
            continue
        itr = r.flows["annual"].pivot(index="year", columns="region", values="itr_GW")
        for t in range(2026, 2041):
            prev, now = itr.loc[t - 1], itr.loc[t]
            if now.sum() >= prev.sum() - 1e-9:
                assert (now - prev).min() >= -1e-9, (key, t)
            else:
                ratio = now / prev
                assert ratio.max() - ratio.min() <= 1e-9, (key, t)
    # the low-demand path falls in 2026 (base 2025 -> low 2026): proportional decline
    low = runs["A0|c-low_d-low_l-slow"].flows["annual"].pivot(
        index="year", columns="region", values="itr_GW")
    assert low.loc[2026].sum() < low.loc[2025].sum()


def test_realized_itr_rule():
    planned = pd.DataFrame({"year": [2030, 2030, 2031, 2031],
                            "region": ["X", "Y", "X", "Y"],
                            "share": [0.5, 0.5, 0.4, 0.6]})
    prev = {"X": 0.6, "Y": 0.4}

    def paths(d30: float, d31: float) -> Paths:
        return Paths(carbon_price={}, demand_TWh={2030: d30 * 8.76, 2031: d31 * 8.76},
                     liquid_deploy_cap={})
    # targets above the previous load: actual demand x planned share
    out = rolling.realized_itr(planned, paths(1.4, 2.0), prev)
    assert out[("X", 2030)] == pytest.approx(0.7) and out[("Y", 2030)] == pytest.approx(0.7)
    assert out[("X", 2031)] == pytest.approx(0.8) and out[("Y", 2031)] == pytest.approx(1.2)
    # X's target 0.55 < 0.6: X keeps 0.6, Y takes the growth
    out = rolling.realized_itr(planned, paths(1.1, 1.1), prev)
    assert out[("X", 2030)] == pytest.approx(0.6) and out[("Y", 2030)] == pytest.approx(0.5)
    # national decline: proportional, no load moves between regions
    out = rolling.realized_itr(planned, paths(0.8, 0.8), prev)
    assert out[("X", 2030)] == pytest.approx(0.48) and out[("Y", 2030)] == pytest.approx(0.32)
    # metro floor: the growth the floors need goes to the metro regions first
    pm = pd.DataFrame({"year": [2030] * 4, "region": ["Kanto", "Kansai", "Hokkaido", "Kyushu"],
                       "share": [0.6, 0.22, 0.10, 0.08]})
    out = rolling.realized_itr(pm, paths(1.05, 1.05),
                               {"Kanto": 0.6, "Kansai": 0.22, "Hokkaido": 0.03, "Kyushu": 0.15},
                               metro_floor={2030: 0.82}, kanto_floor={2030: 0.6})
    assert out[("Kanto", 2030)] == pytest.approx(0.63)
    assert out[("Kansai", 2030)] == pytest.approx(0.231)
    assert out[("Hokkaido", 2030)] == pytest.approx(0.039)
    assert out[("Kyushu", 2030)] == pytest.approx(0.15)
    # 'increment' (sensitivity): growth split by the planned increments
    inc = planned.assign(itr_GW=[0.7, 0.7, 0.8, 1.2])
    out = rolling.realized_itr(inc, paths(1.1, 1.1), prev, rule="increment")
    assert out[("X", 2030)] == pytest.approx(0.6 + 0.1 * 0.1 / 0.4)
    assert out[("Y", 2030)] == pytest.approx(0.4 + 0.1 * 0.3 / 0.4)


def test_realized_siting_metrics(runs):
    for key, r in runs.items():
        if key == "_calib":
            continue
        assert r.metrics["itr_fall_max_GW"] <= 1e-9, key
        assert r.metrics["siting_floor_gap_min"] >= -1e-6, key  # solver tolerance


# ------------------------------------------------------------------ (d)

def test_stitching_states_and_years(runs):
    calib = runs["_calib"]
    for key, r in runs.items():
        if key == "_calib":
            continue
        assert r.states[0] == state_at(calib, 2025)
        for k in range(len(r.windows) - 1):
            end = int(r.realize_specs[k].end_year)
            st = state_at(r.windows[k], end)
            assert r.plan_specs[k + 1].initial_state == st
            assert r.realize_specs[k + 1].initial_state == st
            nxt = r.windows[k + 1].flows["annual"]
            first = nxt[nxt["year"] == end + 1].set_index("region")
            for tech in COOLING + ("pv", "bess", "cgs", "gen"):
                col = UNIT[tech]
                for reg, prev in st.capacities[tech].items():
                    assert abs(first.at[reg, f"cap_{tech}_{col}"]
                               - prev - first.at[reg, f"new_{tech}_{col}"]) <= 1e-6
        yrs = r.flows["annual"]["year"]
        assert sorted(set(yrs)) == list(range(2024, 2041))
        parts: List[float] = [calib.metrics["social_cost_yen"]] + [
            w.metrics["social_cost_yen"] for w in r.windows]
        assert abs(sum(parts) - r.metrics["social_cost_yen"]) <= 1e-6 * r.metrics["social_cost_yen"]
