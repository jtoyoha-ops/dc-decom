"""Model hooks of the bridge (DEV parameters; not results)."""
from __future__ import annotations

from dc_decom_v2 import ModelOptions, run
from dc_decom_v2.config import METRO
from dc_decom_v2.scenarios import default_policy


def test_growth_split_allocation(ps):
    res = run(ModelOptions(time_mode="annual"), "C-CN-gs", ps)
    assert res.status == "Optimal"
    a = res.flows["annual"].set_index(["region", "year"])["itr_GW"]
    for t in (2030, 2040):
        tot_t = sum(a[(r, t)] for r in ps.region_names)
        tot_0 = sum(a[(r, 2025)] for r in ps.region_names)
        for r in ps.region_names:
            share = {"Kanto": 0.73, "Kansai": 0.27}.get(r, 0.0)
            assert abs(a[(r, t)] - (a[(r, 2025)] + share * (tot_t - tot_0))) < 1e-9


def test_liquid_fleet_max_and_co2_floor(ps):
    nonmetro = [r for r in ps.region_names if r not in METRO]
    pol = default_policy("C-CN", ps).with_(liquid_fleet_max={r: 0.15 for r in nonmetro})
    res = run(ModelOptions(time_mode="annual", co2_floor_share=0.95), "C-CN", ps, policy=pol)
    assert res.status == "Optimal"
    ry = res.tables["region_year"]
    later = ry[(ry["year"] > 2025) & ry["region"].isin(nonmetro)]
    liq = (later["it_dlc_GW"] + later["it_imm_GW"]) / later["itr_GW"]
    assert (liq <= 0.15 + 1e-7).all()
    nat = res.tables["national"]
    nat = nat[nat["year"] > 2025]
    assert (nat["co2_Mt"] >= 0.95 * nat["co2_cap_Mt"] - 1e-6).all()
    assert (nat["co2_Mt"] <= nat["co2_cap_Mt"] + 1e-6).all()


def test_ef_grid_path_addition(ps):
    key = "ef_grid_path[Kanto]@2030"
    res = run(ModelOptions(time_mode="annual"), "REF", ps.with_additions({key: 0.2}))
    ry = res.tables["region_year"].set_index(["region", "year"])
    assert abs(ry.loc[("Kanto", 2030), "ef_grid"] - 0.2) < 1e-12
