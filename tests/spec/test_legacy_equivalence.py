"""The spec version with all fixes E1-E8 and the default readings reproduces
the v1 package ``dc_decom`` (v2.0.0, E1-E8 fixed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dc_decom import load_inputs as v1_inputs  # noqa: E402
from dc_decom import model as v1_model  # noqa: E402
from dc_decom import scenarios as v1_scenarios  # noqa: E402
from dc_decom_spec import model as spec_model  # noqa: E402
from dc_decom_spec import scenarios as spec_scenarios  # noqa: E402
from dc_decom_spec.config import SpecFixes, SpecReadings  # noqa: E402
from dc_decom_spec.load_inputs import load_inputs  # noqa: E402

V1_TJPY = {"REF": 15.59, "C-CN": 123.17, "R-CN": 113.18}


@pytest.fixture(scope="module")
def pairs():
    p = v1_inputs.load_global_parameters()
    regions = v1_inputs.load_regions()
    temps = v1_inputs.load_temperature_96h()
    s_regions, s_p, s_temps = load_inputs(table9_inputs=True)
    out = {}
    for name in V1_TJPY:
        v1 = v1_model.build_and_solve(v1_scenarios.ALL_SCENARIOS[name], regions, p, temps)
        spec = spec_model.build_and_solve(spec_scenarios.ALL_SCENARIOS[name], s_regions, s_p,
                                          s_temps, fixes=SpecFixes(), readings=SpecReadings())
        out[name] = (v1, spec)
    return out


@pytest.mark.parametrize("name", list(V1_TJPY))
def test_objective_matches_v1(pairs, name: str) -> None:
    v1, spec = pairs[name]
    assert v1.status == spec.status == "Optimal"
    assert round(v1.objective_yen / 1e12, 2) == V1_TJPY[name]
    assert abs(spec.objective_yen / v1.objective_yen - 1.0) <= 1e-9


@pytest.mark.parametrize("name", list(V1_TJPY))
def test_energy_by_source_matches_v1(pairs, name: str) -> None:
    v1, spec = pairs[name]
    assert len(v1.table) == len(spec.table)
    for a, b in zip(v1.table, spec.table):
        assert (a["year"], a["region"]) == (b["year"], b["region"])
        for key in a:
            if key.startswith("E_"):
                assert abs(float(a[key]) - float(b[key])) <= 1e-6, (name, key, a["year"], a["region"])


@pytest.mark.parametrize("name", list(V1_TJPY))
def test_cost_terms_match_v1(pairs, name: str) -> None:
    v1, spec = pairs[name]
    for attr in ("capex_total", "opex_total", "energy_cost", "carbon_cost", "curtail_cost",
                 "subsidy_total"):
        a, b = getattr(v1, attr), getattr(spec, attr)
        assert abs(a - b) <= 1e-9 * max(1.0, abs(a))
    assert abs(spec_model.components_total(spec) / spec.objective_yen - 1.0) <= 1e-9
