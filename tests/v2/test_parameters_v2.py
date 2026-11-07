"""Checks of the frozen v2 parameter set data/parameters_v2.

No LP is solved: the tests read the CSV files, the source table and the profiles.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dc_decom_v2.params import (  # noqa: E402
    GROUPS,
    PATH_COLUMNS,
    REGION_COLUMNS,
    SCALAR_COLUMNS,
    STATUSES,
    ParamSet,
)
from dc_decom_v2.timeslices import make_timeslices  # noqa: E402

P2 = ROOT / "data" / "parameters_v2"
REGIONS = ("Hokkaido", "Tohoku", "Kanto", "Kansai", "Kyushu")
NONMETRO = ("Hokkaido", "Tohoku", "Kyushu")
YEARS = list(range(2024, 2041))
NEEDS_SOURCE = ("sourced", "derived", "paper")
ALL_REGION_IDS = ("annual_mean_T_C", "base_share_2024", "tariff_yen_per_kWh", "ppa_yen_per_kWh",
                  "re_max_GW", "grid_ef_base")
# ids read by src/dc_decom_v2 (model.py, pue.py, scenarios.py, timeslices.py)
MODEL_SCALARS = ("C_base", "alpha", "beta", "T_thresh", "C_pump", "gamma",
                 "pue_immersion_offset", "utilization", "cooling_air_capex",
                 "cooling_dlc_capex", "cooling_immersion_capex", "bess_capex_2025",
                 "bess_capex_2035", "bess_efficiency", "bess_duration_h", "pv_lcoe",
                 "Deploy_max_PV", "cgs_capex", "cgs_eff_elec", "cgs_fuel_yen_per_kWh_th",
                 "EF_fossil", "gen_capex", "opex_om_rate", "curtail_penalty", "discount_rate",
                 "N_resilience", "nfc_yen_per_kWh", "ppa_tier1_share", "ppa_tier2_share",
                 "ppa_tier2_premium_yen_per_kWh", "reinforcement_cost_yen_per_kW",
                 "operator_burden_share", "siting_subsidy_yen_per_kW")
MODEL_PATHS = ("demand_TWh", "carbon_price", "ef_national", "liquid_deploy_cap",
               "ppa_volume_factor", "metro_floor", "kanto_floor")


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(P2 / name)


@pytest.fixture(scope="module")
def tables():
    return {"parameters": _csv("parameters.csv"), "regions": _csv("regions.csv"),
            "paths": _csv("paths.csv")}


@pytest.fixture(scope="module")
def sources() -> pd.DataFrame:
    return _csv("sources.csv")


def test_schema(tables):
    cols = {"parameters": SCALAR_COLUMNS, "regions": REGION_COLUMNS, "paths": PATH_COLUMNS}
    for name, df in tables.items():
        assert set(cols[name]) <= set(df.columns), name
        assert df["status"].isin(STATUSES).all(), name
        assert df["group"].isin(GROUPS).all(), name
        assert df["value"].notna().all(), name
        assert df["unit"].notna().all(), name
    ps = ParamSet.load(P2)
    assert ps.region_names == REGIONS
    assert not ps.has_dev


def test_no_dev_rows(tables):
    for name, df in tables.items():
        assert not (df["status"] == "dev").any(), name
        assert not df["source_id"].astype(str).str.upper().eq("DEV").any(), name


def test_sources_resolvable(tables, sources):
    assert sources["source_id"].is_unique
    assert sources["url_or_document"].notna().all()
    assert sources["accessed"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    known = set(sources["source_id"])
    for name, df in tables.items():
        assert df["source_id"].notna().all(), name
        missing = set(df["source_id"]) - known
        assert not missing, f"{name}: unknown source_id {missing}"
        need = df[df["status"].isin(NEEDS_SOURCE)]
        assert not need["source_id"].eq("ASSUMPTION").any(), \
            f"{name}: sourced/derived/paper rows must cite a source"


def test_assumptions_have_ranges(tables):
    for name in ("parameters", "regions"):
        df = tables[name]
        a = df[df["status"] == "assumption"]
        assert a["low"].notna().all() and a["high"].notna().all(), name
        assert (a["low"] < a["high"]).all(), name
        assert ((a["low"] <= a["value"]) & (a["value"] <= a["high"])).all(), name
    p = tables["paths"]
    a = p[p["status"] == "assumption"]
    assert a["low"].notna().all() and a["high"].notna().all()
    assert ((a["low"] <= a["value"]) & (a["value"] <= a["high"])).all()
    for pid, g in a.groupby("id"):
        assert (g["low"] < g["high"]).any(), pid


def test_ranges_contain_values(tables):
    for name, df in tables.items():
        rng = df[df["low"].notna() & df["high"].notna()]
        assert ((rng["low"] <= rng["value"] + 1e-12)
                & (rng["value"] <= rng["high"] + 1e-12)).all(), name


def test_regions_complete(tables):
    reg = tables["regions"]
    for pid in ALL_REGION_IDS:
        got = list(reg.loc[reg["id"] == pid, "region"])
        assert got == list(REGIONS), pid
    for pid in ("conn_capacity_GW_per_yr", "conn_capacity_reinforced_GW_per_yr"):
        assert sorted(reg.loc[reg["id"] == pid, "region"]) == sorted(NONMETRO), pid
    shares = reg.loc[reg["id"] == "base_share_2024", "value"]
    assert math.isclose(shares.sum(), 1.0, abs_tol=1e-9)


def test_paths_cover_horizon(tables):
    p = tables["paths"]
    for pid, g in p.groupby("id"):
        assert sorted(g["year"]) == YEARS, pid
    for pid in ("demand_TWh", "demand_facility_TWh"):
        for col in ("value", "low", "high"):
            v = p[p["id"] == pid].sort_values("year")[col].to_numpy()
            assert (v[1:] >= v[:-1]).all(), (pid, col)  # endogenous siting needs this


def test_model_ids_present():
    ps = ParamSet.load(P2)
    for pid in MODEL_SCALARS:
        ps.scalar(pid)
    for pid in MODEL_PATHS:
        assert sorted(ps.path(pid)) == YEARS, pid
    for r in REGIONS:
        for pid in ALL_REGION_IDS:
            ps.region(pid, r)
    ts = make_timeslices("rep96", ps, list(REGIONS))
    assert len(ts.slices) == 96
    assert math.isclose(float(ts.weights.sum()), 8760.0)
    for r in REGIONS:
        assert 0.10 < ts.cf_avg(r) < 0.16, r


def test_profiles_and_calibration():
    t = _csv("temperature_96h_jma.csv")
    c = _csv("pv_cf_96h_jma.csv")
    for df in (t, c):
        assert len(df) == 96 * 5
        assert df.groupby("region")["weight_hours"].sum().eq(8760.0).all()
        assert (df["n_obs"] >= 300).all()
    cal = json.loads((P2 / "pue_calibration.json").read_text())
    assert cal["fit"]["dof"] == 2
    assert cal["all_inside_survey_ranges"] is True
    assert cal["base"]["physical_check_pue_ge_1"] is True
    ps = ParamSet.load(P2)
    for k in ("C_base", "alpha", "beta"):
        assert ps.scalar(k) == pytest.approx(cal["coefficients"][k])
    j = _csv("jma_summary.csv").set_index("region")
    for r in REGIONS:
        assert ps.region("annual_mean_T_C", r) == pytest.approx(j.loc[r, "annual_mean_T_C"])


def test_demand_conversion(tables):
    p = tables["paths"]
    s = tables["parameters"].set_index("id")
    fleet = float(s.loc["pue_air_fleet_2024", "value"])
    fac = p[p["id"] == "demand_facility_TWh"].set_index("year")
    it = p[p["id"] == "demand_TWh"].set_index("year")
    for col in ("value", "low", "high"):
        assert (it[col] * fleet - fac[col]).abs().max() < 1e-3, col
    assert fac.loc[2024, "value"] == pytest.approx(19.0)
