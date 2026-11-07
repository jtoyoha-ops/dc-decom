"""Energy balance per slice and annual totals as weighted sums."""
from __future__ import annotations

import numpy as np
import pytest

from dc_decom_v2 import accounting

KEYS = ("rcn_rep96", "rcn_annual", "ccn_hourly_rep96", "ccn_hourly_annual")


def _with_it(res):
    a = res.flows["annual"][["year", "region", "it_air_GW", "it_dlc_GW", "it_imm_GW",
                             "u_energy"]]
    return res.flows["slices"].merge(a, on=["year", "region"], validate="many_to_one")


@pytest.mark.parametrize("key", KEYS)
def test_slice_balance(solved, key):
    s = _with_it(solved(key))
    load = s["u_energy"] * (s["it_air_GW"] * s["pue_air"] + s["it_dlc_GW"] * s["pue_dlc"]
                            + s["it_imm_GW"] * s["pue_imm"])
    assert np.max(np.abs(load - s["L_GW"])) < 1e-9
    resid = (s["g_GW"] + s["pv_GW"] + s["cgs_GW"] + s["dis_GW"] + s["curt_GW"] + s["ppa_GW"]
             - s["spill_GW"] - s["L_GW"] - s["ch_GW"])
    assert np.max(np.abs(resid)) < 1e-6


@pytest.mark.parametrize("key", KEYS)
def test_annual_totals_are_weighted_sums(solved, key):
    res = solved(key)
    s = res.flows["slices"]
    w = s.groupby(["year", "region"])["weight_h"].sum()
    assert np.allclose(w.to_numpy(), 8760.0)
    e = accounting.slice_energy(res.flows).set_index(["year", "region"])
    for col, name in (("L_GW", "facility_TWh"), ("g_GW", "grid_received_TWh"),
                      ("ch_GW", "bess_ch_TWh"), ("curt_GW", "curt_TWh")):
        manual = (s[col] * s["weight_h"] / 1000.0).groupby([s["year"], s["region"]]).sum()
        assert np.allclose(manual.loc[e.index].to_numpy(), e[name].to_numpy(), rtol=0,
                           atol=1e-9)
    nat = res.tables["national"].set_index("year")
    assert np.allclose(nat["facility_TWh"].to_numpy(),
                       e["facility_TWh"].groupby(level="year").sum().loc[nat.index].to_numpy(),
                       rtol=0, atol=1e-9)
