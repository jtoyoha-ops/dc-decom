"""With all 96 temperatures at the annual mean and a flat PV CF, the rep96
model equals the annual (year-unit) model."""
from __future__ import annotations

import pytest

from dc_decom_v2 import ModelOptions, run


@pytest.fixture(scope="module")
def flat(ps):
    temps = ps.profile("temperature_96h")
    temps["temperature_C"] = [ps.region("annual_mean_T_C", r) for r in temps["region"]]
    cf = ps.profile("pv_cf_96h").merge(temps[["region", "season", "hour", "weight_hours"]],
                                       on=["region", "season", "hour"])
    mean = {r: float((g["cf"] * g["weight_hours"]).sum() / g["weight_hours"].sum())
            for r, g in cf.groupby("region")}
    cf["cf"] = [mean[r] for r in cf["region"]]
    return ps.with_profile("temperature_96h", temps).with_profile("pv_cf_96h", cf)


@pytest.mark.parametrize("scenario,matching", [("R-CN", "annual"), ("C-CN", "hourly")])
def test_rep96_equals_annual(flat, scenario, matching):
    out = {}
    for mode in ("annual", "rep96"):
        res = run(ModelOptions(time_mode=mode, ppa_matching=matching), scenario, flat)
        assert res.status == "Optimal"
        out[mode] = res
    a, b = out["annual"], out["rep96"]
    for k in ("objective_yen", "eq1_total_yen", "social_cost_yen"):
        assert abs(b.metrics[k] - a.metrics[k]) <= 1e-6 * abs(a.metrics[k]), k
    assert abs(b.meta["E_2024_Mt"] - a.meta["E_2024_Mt"]) <= 1e-12 * a.meta["E_2024_Mt"]
