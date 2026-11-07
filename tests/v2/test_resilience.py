"""N+1 backup (ch04 eq:resilience): CGS + generators >= (1 + 1/N) x critical load."""
from __future__ import annotations

import numpy as np
import pytest

from dc_decom_v2 import ModelOptions, run


def _crit(res):
    a = res.flows["annual"][["year", "region", "it_air_GW", "it_dlc_GW", "it_imm_GW", "u_cap",
                             "cap_cgs_GW", "cap_gen_GW", "itr_GW"]]
    s = res.flows["slices"].merge(a, on=["year", "region"], validate="many_to_one")
    s["crit"] = (s["it_air_GW"] * s["pue_air"] + s["it_dlc_GW"] * s["pue_dlc"]
                 + s["it_imm_GW"] * s["pue_imm"]) / s["u_cap"]
    s["backup"] = s["cap_cgs_GW"] + s["cap_gen_GW"]
    return s


@pytest.mark.parametrize("key", ("rcn_rep96", "rcn_annual", "ccn_hourly_rep96"))
def test_backup_covers_every_slice(solved, ps, key):
    res = solved(key)
    reserve = 1.0 + 1.0 / ps.scalar("N_resilience")
    assert res.meta["reserve_factor"] == reserve
    s = _crit(res)
    assert (s["backup"] >= reserve * s["crit"] - 1e-6).all()
    # in 2024 nothing exists yet and backup is costly: sized by the largest slice
    g = s[s["year"] == 2024].groupby("region").agg(backup=("backup", "first"),
                                                   crit=("crit", "max"))
    assert np.max(np.abs(g["backup"] - reserve * g["crit"])) < 1e-5


def test_legacy5_rule(ps):
    res = run(ModelOptions(time_mode="annual", resilience="legacy5"), "R-CN", ps)
    a = res.flows["annual"]
    assert res.status == "Optimal"
    assert (a["cap_cgs_GW"] >= 0.05 * a["itr_GW"] - 1e-6).all()
    assert a["cap_gen_GW"].abs().max() == 0.0


def test_cgs_only_backup(ps):
    res = run(ModelOptions(time_mode="annual", backup_techs=("cgs",)), "R-CN", ps)
    s = _crit(res)
    assert s["cap_gen_GW"].abs().max() == 0.0
    assert (s["cap_cgs_GW"] >= 2.0 * s["crit"] - 1e-6).all()
