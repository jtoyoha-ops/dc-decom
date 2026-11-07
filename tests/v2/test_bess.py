"""BESS state of charge, daily cycle per representative day and conservation."""
from __future__ import annotations

import numpy as np


def _slices_with_caps(res):
    a = res.flows["annual"][["year", "region", "cap_bess_GWh"]]
    return res.flows["slices"].merge(a, on=["year", "region"], validate="many_to_one")


def test_bess_used_with_hourly_matching_and_cap(solved):
    res = solved("ccn_hourly_rep96")
    assert res.metrics["bess_throughput_TWh"] > 1.0
    assert res.metrics["bess_cap_last_GWh"] > 1.0


def test_conservation_per_representative_day(solved):
    res = solved("ccn_hourly_rep96")
    rt = res.meta["eta"] ** 2
    s = res.flows["slices"]
    g = s.groupby(["year", "region", "season"])[["ch_GW", "dis_GW"]].sum()
    assert np.max(np.abs(g["dis_GW"] - rt * g["ch_GW"])) < 1e-6
    assert (g["ch_GW"] > 1e-3).any()


def test_soc_bounds_and_wraparound(solved):
    res = solved("ccn_hourly_rep96")
    eta, dur = res.meta["eta"], res.meta["bess_duration_h"]
    s = _slices_with_caps(res)
    assert (s["soc_GWh"] >= -1e-7).all()
    assert (s["soc_GWh"] <= s["cap_bess_GWh"] + 1e-6).all()
    assert (s["ch_GW"] <= s["cap_bess_GWh"] / dur + 1e-6).all()
    assert (s["dis_GW"] <= s["cap_bess_GWh"] / dur + 1e-6).all()
    worst = 0.0
    for _, day in s.groupby(["year", "region", "season"]):
        day = day.sort_values("hour")
        soc = day["soc_GWh"].to_numpy()
        prev = np.roll(soc, 1)                      # hour 0 follows hour 23
        resid = soc - prev - eta * day["ch_GW"].to_numpy() + day["dis_GW"].to_numpy() / eta
        worst = max(worst, float(np.max(np.abs(resid))))
    assert worst < 1e-6


def test_annual_mode_has_no_bess(solved):
    for key in ("rcn_annual", "ccn_hourly_annual"):
        res = solved(key)
        assert res.metrics["bess_throughput_TWh"] < 1e-9
        assert res.flows["annual"]["cap_bess_GWh"].max() < 1e-9
