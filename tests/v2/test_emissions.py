"""Emissions recomputed from the flows; PPA and certificates within grid draw."""
from __future__ import annotations

import numpy as np
import pytest


def _recompute(res):
    s = res.flows["slices"]
    a = res.flows["annual"].set_index(["year", "region"]).sort_index()
    w = s["weight_h"] / 1000.0
    by = [s["year"], s["region"]]
    # grid draw from the balance: charging is on the demand side, so it is in G
    G_bal = ((s["L_GW"] + s["ch_GW"] - s["pv_GW"] - s["cgs_GW"] - s["dis_GW"] - s["curt_GW"]
              - s["ppa_GW"] + s["spill_GW"]) * w).groupby(by).sum().sort_index()
    G = (s["g_GW"] * w).groupby(by).sum().sort_index()
    cgs = (s["cgs_GW"] * w).groupby(by).sum().sort_index()
    n = res.meta["n_tiers"]
    ppa = sum(a[f"ppa_t{i + 1}_TWh"] for i in range(n))
    net = G - a["nfc_TWh"] if res.meta["ppa_matching"] == "hourly" else G - ppa - a["nfc_TWh"]
    em = a["ef_grid"] * net + a["ef_fossil"] * cgs
    return G_bal, G, ppa, cgs, em


@pytest.mark.parametrize("key", ("rcn_rep96", "ccn_hourly_rep96", "ccn_cert_annual",
                                 "ref_annual"))
def test_reported_emissions_equal_recomputation(solved, ps, key):
    res = solved(key)
    G_bal, G, ppa, cgs, em = _recompute(res)
    assert np.max(np.abs(G_bal - G)) < 1e-6
    ry = res.tables["region_year"].set_index(["year", "region"]).sort_index()
    assert np.max(np.abs(ry["em_Mt"] - em)) < 1e-9
    assert np.max(np.abs(ry["em_cgs_Mt"] - ps.scalar("EF_fossil") * cgs)) < 1e-12
    nat = res.tables["national"].set_index("year")["co2_Mt"]
    assert np.max(np.abs(nat - em.groupby(level="year").sum())) < 1e-9
    assert abs(res.metrics["co2_cum_Mt"] - em.sum()) < 1e-9


@pytest.mark.parametrize("key", ("rcn_rep96", "ccn_cert_annual", "rcn_annual"))
def test_ppa_and_certificates_within_grid_draw(solved, key):
    res = solved(key)
    _, G, ppa, _, _ = _recompute(res)
    a = res.flows["annual"].set_index(["year", "region"]).sort_index()
    assert (ppa <= G + 1e-6).all()
    assert (a["nfc_TWh"] <= G - ppa + 1e-6).all()


def test_certificates_are_used_and_priced_on_top_of_tariff(solved):
    res = solved("ccn_cert_annual")
    ry = res.tables["region_year"]
    assert ry["nfc_TWh"].sum() > 0.0
    tariffed = ry["grid_received_TWh"] - ry["ppa_TWh"]
    assert np.allclose(ry["energy_grid_yen"], ry["tariff"] * tariffed * 1e9, rtol=1e-12)
    assert np.allclose(ry["energy_nfc_yen"], ry["nfc_price"] * ry["nfc_TWh"] * 1e9, rtol=1e-12)


@pytest.mark.parametrize("key", ("rcn_rep96", "ccn_hourly_rep96", "ccn_cert_annual"))
def test_co2_cap_holds(solved, key):
    nat = solved(key).tables["national"]
    assert (nat["co2_Mt"] <= nat["co2_cap_Mt"] + 1e-6).all()


def test_charging_energy_is_drawn_and_emits(solved):
    """With hourly matching, G is the balance residual including charging, and
    emissions follow G: removing the charging would change the grid draw."""
    res = solved("ccn_hourly_rep96")
    s = res.flows["slices"]
    assert (s["ch_GW"] * s["weight_h"]).sum() > 0.0
    G_bal, G, _, _, em = _recompute(res)
    ry = res.tables["region_year"].set_index(["year", "region"]).sort_index()
    assert np.max(np.abs(ry["grid_received_TWh"] - G_bal)) < 1e-6
    assert np.max(np.abs(ry["em_Mt"] - em)) < 1e-9
