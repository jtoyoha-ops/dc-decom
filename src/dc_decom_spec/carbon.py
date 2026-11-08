"""Carbon-cost module (Eq. 10) and diagnostic decomposition (Eq. 11)."""
from __future__ import annotations

import numpy as np

from .load_inputs import GlobalParameters


def carbon_price_trajectory(years: np.ndarray, p: GlobalParameters) -> np.ndarray:
    """Piecewise-linear carbon-price path in yen/t-CO2: GX-ETS FY2026 band mid-point
    (2026), mid-point of the Table 8 setting (2035, an assumption), code value (2040)."""
    years = np.asarray(years, dtype=float)
    # Three anchor points: 2026, 2035, 2040
    xp = np.array([2026.0, 2035.0, 2040.0])
    fp = np.array([p.P_carbon_2026, p.P_carbon_2035, p.P_carbon_2040])
    # For years <= 2025, hold at 2026 floor; for years >= 2040, extrapolate flat
    return np.interp(years, xp, fp, left=fp[0], right=fp[-1])


def grid_emission_factor(years: np.ndarray, p: GlobalParameters) -> np.ndarray:
    """Linear decline from EF_grid_2024 to EF_grid_2040 (kg-CO2/kWh)."""
    years = np.asarray(years, dtype=float)
    frac = (years - p.horizon_start) / (p.horizon_end - p.horizon_start)
    frac = np.clip(frac, 0.0, 1.0)
    return p.EF_grid_2024 + (p.EF_grid_2040 - p.EF_grid_2024) * frac


def regional_grid_emission_factor(years: np.ndarray, p: GlobalParameters,
                                  ef_base_r: float) -> np.ndarray:
    """Fix E5: EF(r, t) = EF_base(r) x EF_nat(t) / EF_grid_2024 (kg-CO2/kWh).

    EF_base(r) is the Table 9 value; EF_nat(t) is the national linear path of
    ``grid_emission_factor``, so every region declines in the same proportion.
    """
    return ef_base_r * grid_emission_factor(years, p) / p.EF_grid_2024


def co2_cap_trajectory(years: np.ndarray, p: GlobalParameters) -> np.ndarray:
    """Linear decline to net-zero by 2040 (Eq. 13). Returns Mt-CO2/yr."""
    years = np.asarray(years, dtype=float)
    frac = (p.horizon_end - years) / (p.horizon_end - p.horizon_start)
    frac = np.clip(frac, 0.0, 1.0)
    return p.E_2024 * frac


def carbon_cost(E_grid_kWh: float, E_cgs_kWh: float, year: int,
                p: GlobalParameters) -> float:
    """Annual carbon cost (yen) per Eq. 10.

    E_grid_kWh / E_cgs_kWh : annual energy from grid and from CGS (kWh).
    Returns the cost in yen.
    """
    ef_grid = float(grid_emission_factor(np.array([year]), p)[0])
    price = float(carbon_price_trajectory(np.array([year]), p)[0])  # yen/t-CO2
    emissions_kg = E_grid_kWh * ef_grid + E_cgs_kWh * p.EF_fossil
    emissions_t = emissions_kg / 1000.0
    return price * emissions_t


def emissions_diagnostic(load_kWh: float, ef_grid_kg_per_kWh: float,
                         f_re: float) -> float:
    """Diagnostic multiplicative identity (Eq. 11) - NOT used by the LP.

    Returns annual CO2 emissions (kg).
    """
    return load_kWh * ef_grid_kg_per_kWh * (1.0 - f_re)
