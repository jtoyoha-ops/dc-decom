"""Additional analyses (config/experiments_v35.yaml): the latency rule of part
ch03 reproduces the frozen floor paths of data/parameters_v2/paths.csv (reads
the frozen parameters; no solve)."""
from __future__ import annotations

import pytest

from dc_decom_v2 import runner
from dc_decom_v2.experiments.latency import floor_paths


@pytest.fixture(scope="module")
def v2():
    return runner.load_params("data/parameters_v2")


@pytest.mark.parametrize("variant,s_train,s_inf", [("value", 0.4, 0.3), ("low", 0.6, 0.5),
                                                   ("high", 0.2, 0.1)])
def test_lambda_formula_reproduces_paths(v2, variant, s_train, s_inf):
    lam, kap = floor_paths(v2.path("ai_share"), s_train, s_inf)
    metro = v2.path("metro_floor", variant)
    kanto = v2.path("kanto_floor", variant)
    assert set(lam) == set(metro) == set(kanto)
    for t in metro:
        assert lam[t] == pytest.approx(metro[t], abs=1e-5), (variant, t)
        assert kap[t] == pytest.approx(kanto[t], abs=1e-5), (variant, t)


# ----------------------------------------------------------------------
# abc_posthoc / b_minus_a0 (config/abc_posthoc.yaml; no solve)
# ----------------------------------------------------------------------

from dataclasses import fields  # noqa: E402

from dc_decom_v2 import rolling  # noqa: E402
from dc_decom_v2.experiments import abc, abc_posthoc  # noqa: E402

_SETTING_FIELD = {"phi_0": "phi", "reinforcement_year": "t_g", "pstar": "pstar"}


def _settings_diff(a, b):
    return {f.name for f in fields(a) if getattr(a, f.name) != getattr(b, f.name)}


def _dict_diff(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for k in set(a) | set(b):
            out |= _dict_diff(a.get(k), b.get(k), f"{prefix}{k}.")
        return out
    return set() if a == b else {prefix.rstrip(".")}


@pytest.fixture(scope="module")
def posthoc_variants():
    before = abc.RULES.read_bytes()
    abc.rules_check()
    rules = abc.load_rules()
    variants = abc_posthoc.build_variants(rules, abc_posthoc.load_config())
    return before, rules, variants


def test_variant_rules_leave_frozen_file_unchanged(posthoc_variants):
    before, _, variants = posthoc_variants
    for v in variants:
        v.rules["thresholds"]["connection_usage"]["value"] = -1.0   # mutate the copies
    assert abc.RULES.read_bytes() == before
    assert abc.rules_check()["unchanged_since_freeze"]
    assert abc.load_rules()["thresholds"]["connection_usage"]["value"] == 0.9


def test_variant_settings_differ_from_b_only_in_listed_fields():
    rules = abc.load_rules()
    variants = abc_posthoc.build_variants(rules, abc_posthoc.load_config())
    b = rolling.designs_from_rules(rules, runner.load_params("data/parameters_v2"),
                                   diagnostics=False)["B"].settings
    base = abc.base_config()
    names = [v.name for v in variants]
    assert names[0] == "frozen" and len(names) == 12
    for v in variants:
        listed = {_SETTING_FIELD[k] for k in v.changes if k in _SETTING_FIELD}
        assert _settings_diff(v.b.settings, b) == {
            f for f in listed if getattr(v.b.settings, f) != getattr(b, f)}
        for k in listed & {"phi", "t_g", "pstar"}:
            key = {v_: k_ for k_, v_ in _SETTING_FIELD.items()}[k]
            assert getattr(v.b.settings, k) == v.changes[key]
        assert v.c.settings == v.b.settings and v.c.adaptive and not v.b.adaptive
        rdiff = _dict_diff(v.rules, rules)
        assert rdiff == ({"thresholds.connection_usage.value"}
                         if "connection_usage" in v.changes
                         and v.changes["connection_usage"] != 0.9 else set())
        assert v.base.options.certificates is v.changes.get("certificates", True)
        assert v.base.options.with_(certificates=True) == base.options
        assert v.base.forecast == base.forecast and v.base.review_years == base.review_years
    frozen = variants[0]
    assert frozen.b.settings == b and frozen.rules == rules and frozen.base == base


def test_b_minus_a0_subsets():
    rules = abc.load_rules()
    a0, subs = abc_posthoc.subset_designs(rules)
    b = rolling.designs_from_rules(rules, runner.load_params("data/parameters_v2"),
                                   diagnostics=False)["B"].settings
    assert len(subs) == 16
    assert subs["1111"].settings == b
    assert _settings_diff(subs["0000"].settings, a0.settings) == {"psi", "subsidy"}
    assert subs["0000"].settings.pstar == 0.0 and b.pstar == 0.0
    for k, d in subs.items():
        off = {e for e, x in zip(abc_posthoc.ELEMENTS, k) if x == "0"}
        assert _settings_diff(d.settings, b) == {
            f for e in off for f in abc_posthoc.ELEMENT_OFF[e]}


def test_effective_policy_ignores_subsidy_maps_empty_from_review_year():
    ps = runner.load_params("data/parameters_v2")
    rules = abc.load_rules()
    base = abc.base_config()
    b = rolling.designs_from_rules(rules, ps, diagnostics=False)["B"].settings.with_(pstar=7000.0)

    def pol(s):
        return rolling.policy_for(s, ps, base.forecast.carbon_price, base.scenario)

    on, off = pol(b), pol(b.with_(subsidy="off"))
    # the B rule pays in 2030-2032 (base-path price < 7,000); nothing from 2035
    assert on.subsidy_yen_per_kW and max(max(m) for m in on.subsidy_yen_per_kW.values()) == 2032
    assert rolling._from(on, 2035) != rolling._from(off, 2035)          # raw flag: changed
    assert abc_posthoc.effective_policy(on, 2035) == abc_posthoc.effective_policy(off, 2035)
    assert abc_posthoc.effective_policy(on, 2030) != abc_posthoc.effective_policy(off, 2030)
