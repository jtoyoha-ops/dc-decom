"""Cumulative connection constraint of non-metro regions and reinforcement cost."""
from __future__ import annotations

import numpy as np
import pytest


def _constrained(res):
    ry = res.tables["region_year"]
    return ry[ry["region"].isin(res.meta["constrained_regions"])]


@pytest.mark.parametrize("key", ("grid_annual", "grid_rep96"))
def test_cumulative_connection_constraint(solved, key):
    res = solved(key)
    assert set(res.meta["constrained_regions"]) == {"Hokkaido", "Tohoku", "Kyushu"}
    c = _constrained(res)
    assert (c["peak_conn_GW"] >= c["peak_facility_GW"] - 1e-6).all()
    for _, g in c.groupby("region"):
        g = g.sort_values("year")
        assert (g["peak_conn_GW"].diff().iloc[1:] >= -1e-7).all()
        ref = float(g.loc[g["year"] == 2025, "peak_conn_GW"].iloc[0])
        later = g[g["year"] >= 2026]
        offered = later["conn_offered_GW"].cumsum()
        assert np.allclose(later["conn_offered_cum_GW"], offered, rtol=0, atol=1e-12)
        assert np.allclose(later["conn_used_cum_GW"], later["peak_conn_GW"] - ref, atol=1e-12)
        assert (later["conn_used_cum_GW"] <= later["conn_offered_cum_GW"] + 1e-6).all()


def test_constraint_binds_somewhere(solved):
    c = _constrained(solved("grid_annual"))
    later = c[c["year"] >= 2026]
    slack = later["conn_offered_cum_GW"] - later["conn_used_cum_GW"]
    assert (slack < 1e-6).any()


@pytest.mark.parametrize("key", ("grid_annual", "grid_rep96"))
def test_reinforcement_cost_in_social_and_theta_share_in_operator(solved, key):
    res = solved(key)
    m = res.metrics
    ry = res.tables["region_year"]
    theta = res.meta["theta"]
    assert theta == 0.4
    reinf = float((ry["reinf_cost"] * ry["dpeak_GW"] * ry["reinforced"] * 1e6 * ry["disc"]).sum())
    assert abs(m["reinforcement_yen"] - reinf) <= 1e-9 * max(1.0, reinf)
    if key == "grid_annual":
        assert reinf > 0.0
    other = sum(ry[f"{k}_disc"].sum() for k in (
        "cooling_capex", "cooling_opex", "bess_capex", "bess_opex", "cgs_capex", "cgs_opex",
        "gen_capex", "gen_opex", "energy_grid", "energy_ppa", "energy_nfc", "energy_pv",
        "energy_cgs_fuel", "curtailment", "over_cost", "shell_premium"))
    assert abs(m["social_cost_yen"] - (other + reinf)) <= 1e-9 * m["social_cost_yen"]
    op = (m["social_cost_yen"] - (1 - theta) * reinf + m["carbon_payments_yen"]
          - m["fiscal_burden_yen"])
    assert abs(m["operator_cost_yen"] - op) <= 1e-9 * m["operator_cost_yen"]
    assert abs(m["operator_cost_yen"] - m["objective_yen"]) <= 1e-6 * m["operator_cost_yen"]


@pytest.mark.parametrize("key", ("rcn_rep96", "rcn_annual"))
def test_unreinforced_peak_is_running_max_of_facility_peak(solved, key):
    """Without reinforcement cost the tie-break keeps the connected peak at the
    running maximum of the facility peak."""
    res = solved(key)
    assert set(res.meta["constrained_regions"]) == {"Hokkaido", "Tohoku", "Kyushu"}
    assert res.meta["tiebreak_yen"] > 0.0
    c = _constrained(res)
    for _, g in c.groupby("region"):
        g = g.sort_values("year")
        run = g["peak_facility_GW"].cummax()
        later = g["year"] >= 2026
        assert np.allclose(g.loc[later, "peak_conn_GW"], run[later], rtol=0, atol=1e-6)
        assert np.allclose(g.loc[later, "conn_used_phys_cum_GW"],
                           g.loc[later, "conn_used_cum_GW"], rtol=0, atol=1e-6)


@pytest.mark.parametrize("key", ("grid_annual", "grid_rep96", "rcn_rep96"))
def test_physical_use_never_exceeds_connected_use(solved, key):
    c = _constrained(solved(key))
    later = c[c["year"] >= 2026]
    assert (later["conn_used_phys_cum_GW"] <= later["conn_used_cum_GW"] + 1e-6).all()
    m = solved(key).metrics
    assert m["conn_usage_last_max"] <= m["conn_connected_last_max"] + 1e-6


def test_no_connection_ahead_of_load_before_reinforcement(solved):
    """With reinforcement from 2030 the free years before it do not connect
    ahead of the facility peak."""
    res = solved("grid_late_annual")
    assert res.status == "Optimal"
    c = _constrained(res)
    later = c[c["year"] >= 2026]
    assert (later["conn_used_cum_GW"] - later["conn_used_phys_cum_GW"]).abs().max() <= 1e-6
    assert res.metrics["reinforcement_yen"] > 0.0


def test_no_cost_before_reinforcement(solved):
    ry = solved("grid_annual").tables["region_year"]
    assert ry.loc[ry["year"] <= 2025, "reinforcement_yen"].abs().max() == 0.0
