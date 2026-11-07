"""Accounting identities and the hand-off to dc_decom_analysis."""
from __future__ import annotations

import pytest

from dc_decom_analysis import decomposition, lmdi
from dc_decom_v2 import accounting

KEYS = ("rcn_rep96", "rcn_annual", "grid_annual", "ccn_cert_annual", "ccn_hourly_rep96",
        "rcn_pt05_annual")


def _close(a, b, rel):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


@pytest.mark.parametrize("key", KEYS)
def test_identities(solved, key):
    res = solved(key)
    m = res.metrics
    theta, pt = res.meta["theta"], res.meta["pass_through"]
    assert _close(m["social_cost_yen"] + m["carbon_cost_yen"], m["eq1_total_yen"], 1e-12)
    op = (m["social_cost_yen"] + pt * m["carbon_cost_yen"] - m["fiscal_burden_yen"]
          - (1.0 - theta) * m["reinforcement_yen"])
    assert _close(m["operator_cost_yen"], op, 1e-12)
    assert _close(m["operator_cost_yen"], m["objective_yen"], 1e-6)


def test_subsidy_is_fiscal_burden(solved):
    m = solved("rcn_annual").metrics
    assert m["fiscal_burden_yen"] > 0.0
    assert solved("ref_annual").metrics["fiscal_burden_yen"] == 0.0


def test_pass_through_scales_operator_carbon(solved):
    res = solved("rcn_pt05_annual")
    assert res.meta["pass_through"] == 0.5
    m = res.metrics
    assert _close(m["carbon_payments_yen"], 0.5 * m["carbon_cost_yen"], 1e-12)


@pytest.mark.parametrize("key", ("rcn_rep96", "grid_annual", "ccn_hourly_rep96"))
def test_analysis_frames(solved, key):
    res = solved(key)
    table, comp = accounting.analysis_frames(res.flows, res.meta, basis="eq1")
    assert _close(decomposition.run_total(comp), res.metrics["eq1_total_yen"], 1e-12)
    _, comp_op = accounting.analysis_frames(res.flows, res.meta, basis="operator")
    assert _close(decomposition.run_total(comp_op), res.metrics["operator_cost_yen"], 1e-12)
    f = lmdi.factor_frame(table)
    assert f["structure"].groupby(f["year"]).sum().sub(1.0).abs().max() < 1e-12
    split = decomposition.principled(comp, comp)
    assert abs(split["total"]) < 1e-6
