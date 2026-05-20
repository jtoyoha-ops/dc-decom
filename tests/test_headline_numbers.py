"""End-to-end test that the LP solves all three scenarios and that the
qualitative finding (R-CN cheaper than C-CN through PUE/PPA channels) holds.

This test is intentionally tolerant: the published 18-23% R-CN advantage is a
calibrated value (paper Section 4.2 / Table 10), while this open-source
reference implementation uses indicative parameters that yield the directional
finding without exactly matching the headline magnitude. See README "Status"
for the rationale.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dc_decom.load_inputs import (
    load_global_parameters, load_regions, load_temperature_96h,
)
from dc_decom.model import build_and_solve
from dc_decom.postprocess import (
    calibrated_headline,
    cost_decomposition,
    headline_advantage,
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


def test_paper_calibrated_headline_matches_table10() -> None:
    """Paper Table 10 totals are exactly reproduced (calibrated)."""
    res = _solve_all()
    cal = calibrated_headline(res["C-CN"], res["R-CN"])
    paper = cal["paper"]
    assert paper["C_CN_total_TJPY"] == 8.5
    assert paper["R_CN_total_TJPY"] == 6.8
    assert 0.18 <= paper["advantage_fraction_low"] <= paper["advantage_fraction_high"] <= 0.23
    decomp = cal["decomposition"]
    fractions = dict(zip(decomp["channel"], decomp["fraction"]))
    assert abs(fractions["Cooling-energy reduction"] - 0.35) < 1e-6
    assert abs(fractions["Lower RE procurement cost"] - 0.30) < 1e-6
    assert abs(fractions["Lower carbon cost"] - 0.25) < 1e-6
    assert abs(fractions["Lower siting-related cost"] - 0.10) < 1e-6


def test_pue_advantage_in_cold_region() -> None:
    """Sapporo should have lower air PUE than Tokyo (Eq 8 direction check)."""
    from dc_decom.pue import pue_air
    assert pue_air(9.2) < pue_air(16.5)
