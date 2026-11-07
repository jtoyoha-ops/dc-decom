"""End-to-end tests of the LP for the three scenarios.

The LP results are not compared with the reported values of the Energies
article (REF 6.2 / C-CN 8.5 / R-CN 6.8 trillion JPY, R-CN advantage 18-23%,
Table 10 channel split 35/30/25/10): with the v2.0.0 inputs and constraints
the LP does not reproduce them (see README). The tests check that the
scenarios solve, the direction of the LP result and its curtailment, and
that the reported values are constants kept apart from the LP output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dc_decom import postprocess
from dc_decom.load_inputs import (
    load_global_parameters, load_regions, load_temperature_96h,
)
from dc_decom.model import build_and_solve
from dc_decom.postprocess import (
    cost_decomposition,
    headline_lp,
    reported_values,
)
from dc_decom.scenarios import C_CN, R_CN, REF


def _solve_all():
    p = load_global_parameters()
    regions = load_regions()
    temps = load_temperature_96h()
    return {
        "REF": build_and_solve(REF, regions, p, temps),
        "C-CN": build_and_solve(C_CN, regions, p, temps),
        "R-CN": build_and_solve(R_CN, regions, p, temps),
    }


def test_all_scenarios_solve_to_optimal() -> None:
    res = _solve_all()
    for name, r in res.items():
        assert r.status == "Optimal", f"{name} not optimal: {r.status}"


def test_co2_constrained_scenarios_more_expensive_than_REF() -> None:
    res = _solve_all()
    # REF has no carbon cap, so it should be cheaper than either C-CN or R-CN
    assert res["C-CN"].objective_yen >= res["REF"].objective_yen
    assert res["R-CN"].objective_yen >= res["REF"].objective_yen


def test_cost_decomposition_has_four_channels() -> None:
    res = _solve_all()
    df = cost_decomposition(res["C-CN"], res["R-CN"])
    assert len(df) == 4
    assert set(df["channel"].tolist()) == {
        "Cooling-energy reduction",
        "Lower RE procurement cost",
        "Lower carbon cost",
        "Lower siting-related cost",
    }


# LP direction of this configuration (Energies Section 3 with the
# implementation errors E1-E8 fixed): R-CN is cheaper than C-CN (advantage
# 8.11%) and the CO2 cap of C-CN and R-CN is met only with curtailment
# (308.0 / 279.6 TWh cumulative).
RECORDED_RCN_CHEAPER = True


def test_lp_direction_and_curtailment_reported() -> None:
    res = _solve_all()
    lp = headline_lp(res)
    assert lp["rcn_cheaper"] is RECORDED_RCN_CHEAPER
    adv = lp["advantage"]
    assert (adv["advantage_fraction"] > 0) is RECORDED_RCN_CHEAPER
    # curtailment is reported for every scenario and equals the table sums
    for name, r in res.items():
        table_sum = sum(float(row["curt_TWh"]) for row in r.table)
        assert lp["curtail_TWh"][name] == pytest.approx(table_sum, abs=1e-9)
    assert lp["curtail_TWh"]["REF"] == pytest.approx(0.0, abs=1e-6)
    assert lp["curtail_TWh"]["C-CN"] > 0.0
    assert lp["curtail_TWh"]["R-CN"] > 0.0
    # the five decomposition lines add up to the C-CN - R-CN difference
    dec = lp["decomposition_TJPY"]
    assert sum(dec.values()) == pytest.approx(adv["advantage_TJPY"], rel=1e-6)
    # the regional subsidy applies to R-CN only
    assert lp["subsidy_TJPY"]["R-CN"] > 0.0
    assert lp["subsidy_TJPY"]["REF"] == lp["subsidy_TJPY"]["C-CN"] == 0.0


def test_reported_values_kept_apart_from_lp_output() -> None:
    res = _solve_all()
    lp = headline_lp(res)
    assert not any(k in ("reported", "published", "paper") for k in lp)
    # LP-derived totals are the solved objectives, not the reported constants
    for name, r in res.items():
        assert lp["totals_TJPY"][name] == r.objective_yen / 1e12
    rep = reported_values()
    assert "not model output" in rep["source"]
    assert rep["C_CN_total_TJPY"] == postprocess.REPORTED_C_CN_TJPY
    # the constant headline of v1.0.1 is gone
    assert not hasattr(postprocess, "calibrated_headline")


def test_pue_advantage_in_cold_region() -> None:
    """Sapporo should have lower air PUE than Tokyo (Eq 8 direction check)."""
    from dc_decom.pue import pue_air
    assert pue_air(9.2) < pue_air(16.5)
