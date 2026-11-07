"""Siting rules: national sum, metro / Kanto floors, no relocation,
calibration years, base and fixed (by name) shares, metro-only."""
from __future__ import annotations

import numpy as np
import pytest

from dc_decom_spec.scenarios import R_CN_SHARES_2040

METRO = ("Kanto", "Kansai")


def _itr(res):
    return res.flows["annual"].pivot(index="year", columns="region", values="itr_GW")


def _base(ps):
    s = {r: ps.region("base_share_2024", r) for r in ps.region_names}
    tot = sum(s.values())
    return {r: v / tot for r, v in s.items()}


@pytest.mark.parametrize("key", ("rcn_rep96", "a0_annual", "rcn_annual"))
def test_endogenous_rules(solved, ps, key):
    res = solved(key)
    itr = _itr(res)
    D = ps.path("demand_TWh")
    total = itr.sum(axis=1)
    assert np.allclose(total.to_numpy(), [D[t] / 8.76 for t in itr.index], rtol=0, atol=1e-6)
    lam, kap = ps.path("metro_floor"), ps.path("kanto_floor")
    for t in itr.index:
        if t <= 2025:
            continue
        assert itr.loc[t, list(METRO)].sum() >= lam[t] * total[t] - 1e-6
        assert itr.loc[t, "Kanto"] >= kap[t] * total[t] - 1e-6
    assert (itr.diff().iloc[1:] >= -1e-7).all().all()
    base = _base(ps)
    for t in (2024, 2025):
        for r in itr.columns:
            assert abs(itr.loc[t, r] / total[t] - base[r]) < 1e-12
    a = res.flows["annual"]
    assert (a["build_IT_GW"] >= -1e-7).all()
    assert res.metrics["nonmetro_share_last"] > base["Hokkaido"] + base["Tohoku"] + base["Kyushu"]


def test_base_shares(solved, ps):
    itr = _itr(solved("ref_annual"))
    shares = itr.div(itr.sum(axis=1), axis=0)
    base = _base(ps)
    for r in shares.columns:
        assert np.allclose(shares[r].to_numpy(), base[r], rtol=0, atol=1e-12)


def test_fixed_shares_by_name(solved, ps):
    itr = _itr(solved("rcnpub_annual"))
    shares = itr.div(itr.sum(axis=1), axis=0)
    target = {r: v / sum(R_CN_SHARES_2040.values()) for r, v in R_CN_SHARES_2040.items()}
    for r in shares.columns:
        assert abs(shares.loc[2040, r] - target[r]) < 1e-12
        assert abs(shares.loc[2025, r] - _base(ps)[r]) < 1e-12
    assert abs(shares.loc[2040, "Hokkaido"] - 0.275) < 1e-12
    assert abs(shares.loc[2040, "Kanto"] - 0.375) < 1e-12


def test_metro_only(solved, ps):
    res = solved("ccnmetro_annual")
    itr = _itr(res)
    nonmetro = [r for r in itr.columns if r not in METRO]
    for r in nonmetro:
        assert np.allclose(itr.loc[2026:, r].to_numpy(), itr.loc[2025, r], rtol=0, atol=1e-12)
    kap = ps.path("kanto_floor")
    total = itr.sum(axis=1)
    for t in itr.index:
        if t > 2025:
            assert itr.loc[t, "Kanto"] >= kap[t] * total[t] - 1e-6
    assert (itr.diff().iloc[1:] >= -1e-7).all().all()
