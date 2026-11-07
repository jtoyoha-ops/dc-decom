"""Each implementation-error fix E1-E8 does what the paper states."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dc_decom_spec import model as spec_model  # noqa: E402
from dc_decom_spec import scenarios as sc  # noqa: E402
from dc_decom_spec.config import SpecFixes, SpecReadings  # noqa: E402
from dc_decom_spec.load_inputs import load_inputs  # noqa: E402

TOL = 1e-6


@pytest.fixture(scope="module")
def primary():
    regions, p, temps = load_inputs(table9_inputs=True)
    res = {name: spec_model.build_and_solve(s, regions, p, temps, fixes=SpecFixes(),
                                            readings=SpecReadings())
           for name, s in sc.ALL_SCENARIOS.items()}
    return res, regions, p


def _rows(res, year=None):
    return [r for r in res.table if year is None or r["year"] == year]


def test_all_optimal(primary) -> None:
    res, _, _ = primary
    assert all(r.status == "Optimal" for r in res.values())


def test_e1_rcn_2040_shares_by_region_name(primary) -> None:
    res, _, _ = primary
    rows = _rows(res["R-CN"], 2040)
    total = sum(float(r["it_load_GW"]) for r in rows)
    for r in rows:
        assert abs(float(r["it_load_GW"]) / total - sc.R_CN_SHARES_2040[r["region"]]) <= 1e-9


def test_e1_off_keeps_positional_application() -> None:
    regions, p, temps = load_inputs(table9_inputs=False)
    res = spec_model.build_and_solve(sc.R_CN, regions, p, temps, fixes=SpecFixes.none())
    rows = _rows(res, 2040)
    total = sum(float(r["it_load_GW"]) for r in rows)
    hok = next(r for r in rows if r["region"] == "Hokkaido")
    assert abs(float(hok["it_load_GW"]) / total - 0.375) <= 1e-9   # Kanto's value lands on Hokkaido


def test_e2_bess_charge_and_balance(primary) -> None:
    res, _, p = primary
    for r in res.values():
        for row in r.table:
            assert abs(float(row["E_bess_disch_TWh"]) - p.bess_efficiency * float(row["E_ch_TWh"])) <= TOL
            supply = sum(float(row[k]) for k in ("E_grid_TWh", "E_ppa_TWh", "E_pv_TWh",
                                                 "E_bess_disch_TWh", "E_cgs_TWh"))
            assert abs(supply + float(row["curt_TWh"])
                       - float(row["load_TWh"]) - float(row["E_ch_TWh"])) <= TOL


def test_e3_backup_at_least_twice_it(primary) -> None:
    res, _, p = primary
    assert p.N_resilience == 1
    for r in res.values():
        for row in r.table:
            # CBC reports solution values to about 8 significant digits.
            assert float(row["cap_cgs_GW"]) >= 2.0 * float(row["it_load_GW"]) - TOL


def test_e4_no_investment_in_calibration_years(primary) -> None:
    res, regions, _ = primary
    for r in res.values():
        for year in (2024, 2025):
            rows = _rows(r, year)
            total = sum(float(x["it_load_GW"]) for x in rows)
            for row in rows:
                for tech in ("dlc", "immersion", "pv", "bess"):
                    assert abs(float(row[f"delta_{tech}"])) <= 1e-9
                assert abs(float(row["it_load_GW"]) / total
                           - regions[row["region"]].base_share_2024) <= 1e-9


def test_e5_regional_emission_factor(primary) -> None:
    res, regions, p = primary
    for r in res.values():
        for row in r.table:
            frac = (row["year"] - 2024) / 16
            ef_nat = p.EF_grid_2024 + (p.EF_grid_2040 - p.EF_grid_2024) * frac
            ef = regions[row["region"]].grid_ef_base * ef_nat / p.EF_grid_2024
            assert abs(float(row["ef_grid"]) - ef) <= 1e-12
            expected = float(row["E_grid_TWh"]) * ef + float(row["E_cgs_TWh"]) * p.EF_fossil
            assert abs(float(row["emissions_Mt"]) - expected) <= 1e-9


def test_e7_pv_plus_ppa_within_re_max(primary) -> None:
    res, regions, p = primary
    for r in res.values():
        for row in r.table:
            used = float(row["cap_pv_GW"]) + float(row["E_ppa_TWh"]) / (p.pv_cf_avg * 8.760)
            assert used <= regions[row["region"]].re_max_GW + TOL


def test_e8_subsidy_only_rcn_non_metro(primary) -> None:
    res, _, _ = primary
    assert res["REF"].subsidy_total == 0.0
    assert res["C-CN"].subsidy_total == 0.0
    assert res["R-CN"].subsidy_total > 0.0
    for row in res["R-CN"].components:
        if row["region"] in sc.METRO:
            assert row["subsidy"] == 0.0
    assert abs(sum(float(c["subsidy"]) for c in res["R-CN"].components)
               - res["R-CN"].subsidy_total) <= 1e-3


def test_components_sum_to_objective(primary) -> None:
    res, _, _ = primary
    for r in res.values():
        assert abs(spec_model.components_total(r) / r.objective_yen - 1.0) <= 1e-9
