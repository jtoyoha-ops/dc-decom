"""Smoke tests for the carbon module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dc_decom.carbon import (
    carbon_cost,
    carbon_price_trajectory,
    co2_cap_trajectory,
    emissions_diagnostic,
    grid_emission_factor,
)
from dc_decom.load_inputs import load_global_parameters


def test_carbon_price_trajectory_anchors() -> None:
    p = load_global_parameters()
    years = np.array([2024, 2026, 2030, 2035, 2040])
    prices = carbon_price_trajectory(years, p)
    # 2026 anchor
    assert abs(prices[1] - p.P_carbon_2026) < 1e-9
    # 2035 anchor
    assert abs(prices[3] - p.P_carbon_2035) < 1e-9
    # 2040 anchor
    assert abs(prices[4] - p.P_carbon_2040) < 1e-9
    # monotonic non-decreasing
    assert all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1))


def test_grid_emission_factor_declines() -> None:
    p = load_global_parameters()
    ef_2024 = float(grid_emission_factor(np.array([2024]), p)[0])
    ef_2040 = float(grid_emission_factor(np.array([2040]), p)[0])
    assert abs(ef_2024 - p.EF_grid_2024) < 1e-9
    assert abs(ef_2040 - p.EF_grid_2040) < 1e-9
    assert ef_2040 < ef_2024   # decarbonisation trajectory


def test_co2_cap_hits_zero_at_2040() -> None:
    p = load_global_parameters()
    cap = co2_cap_trajectory(np.array([2024, 2032, 2040]), p)
    assert abs(cap[0] - p.E_2024) < 1e-9
    assert abs(cap[2] - 0.0) < 1e-9
    # linearly declining
    assert cap[1] == p.E_2024 / 2.0


def test_carbon_cost_positive() -> None:
    p = load_global_parameters()
    c = carbon_cost(E_grid_kWh=1e9, E_cgs_kWh=0.0, year=2030, p=p)
    assert c > 0


def test_diagnostic_identity() -> None:
    # Eq 11: load * ef * (1 - f_re)
    assert emissions_diagnostic(1.0, 0.4, 0.5) == 0.2
