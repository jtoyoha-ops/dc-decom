"""Load input parameters and temperature profiles for DC-DECOM."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "parameters_csv"


@dataclass(frozen=True)
class RegionData:
    name: str
    city: str
    annual_mean_T_C: float
    base_share_2024: float
    tariff_yen_per_kWh: float
    ppa_yen_per_kWh: float
    re_max_GW: float
    grid_ef_base: float       # kg-CO2/kWh, regional grid EF in 2024 (Table 9)


@dataclass(frozen=True)
class GlobalParameters:
    """All scalar (non-regional, non-temporal) inputs."""

    # PUE
    C_base: float
    alpha: float
    beta: float
    T_thresh: float
    C_pump: float
    gamma: float

    # Demand
    P_base_2024: float        # GW
    P_AI_0: float             # GW
    g_AI: float               # fraction/yr
    eta_HW_rate: float        # fraction/yr
    eta_arch_rate: float      # fraction/yr

    # Carbon
    EF_grid_2024: float       # kg-CO2/kWh
    EF_grid_2040: float       # kg-CO2/kWh
    EF_fossil: float          # kg-CO2/kWh
    P_carbon_2026: float      # yen/t-CO2
    P_carbon_2035: float      # yen/t-CO2
    P_carbon_2040: float      # yen/t-CO2
    E_2024: float             # Mt-CO2/yr

    # Constraints
    N_resilience: int
    Deploy_max_liquid: float  # GW/yr/region
    Deploy_max_PV: float      # GW/yr/region

    # Objective
    discount_rate: float
    curtail_penalty: float    # yen/kWh
    horizon_start: int
    horizon_end: int
    hours_per_year: int
    rep_hours_per_year: int

    # Technology costs (yen thousand per kW or kWh)
    cooling_air_capex: float
    cooling_dlc_capex: float
    cooling_immersion_capex: float
    bess_capex_2025: float
    bess_capex_2035: float
    bess_efficiency: float
    pv_lcoe: float            # yen/kWh
    pv_cf_avg: float
    cgs_capex: float
    cgs_eff_total: float
    cgs_eff_elec: float
    pue_air_2024_avg: float
    pue_dlc_avg: float
    pue_immersion_avg: float
    opex_om_rate: float

    extra: Dict[str, float] = field(default_factory=dict)


def load_regions(path: Path | None = None) -> Dict[str, RegionData]:
    path = path or (DATA_DIR / "regions.csv")
    df = pd.read_csv(path)
    if "grid_ef_base" not in df.columns:
        raise KeyError(f"{path}: missing column grid_ef_base "
                       "(regional grid emission factor, paper Table 9)")
    out: Dict[str, RegionData] = {}
    for _, row in df.iterrows():
        out[row["region"]] = RegionData(
            name=row["region"],
            city=row["city"],
            annual_mean_T_C=float(row["annual_mean_T_C"]),
            base_share_2024=float(row["base_share_2024"]),
            tariff_yen_per_kWh=float(row["tariff_yen_per_kWh"]),
            ppa_yen_per_kWh=float(row["ppa_yen_per_kWh"]),
            re_max_GW=float(row["re_max_GW"]),
            grid_ef_base=float(row["grid_ef_base"]),
        )
    return out


def load_global_parameters(path: Path | None = None) -> GlobalParameters:
    path = path or (DATA_DIR / "global_parameters.csv")
    df = pd.read_csv(path)
    series = pd.Series(df["value"].values, index=df["parameter"])

    def num(key: str, *, cast=float):
        if key not in series.index:
            raise KeyError(f"missing parameter: {key}")
        return cast(series[key])

    return GlobalParameters(
        C_base=num("C_base"),
        alpha=num("alpha"),
        beta=num("beta"),
        T_thresh=num("T_thresh"),
        C_pump=num("C_pump"),
        gamma=num("gamma"),
        P_base_2024=num("P_base_2024"),
        P_AI_0=num("P_AI_0"),
        g_AI=num("g_AI"),
        eta_HW_rate=num("eta_HW_rate"),
        eta_arch_rate=num("eta_arch_rate"),
        EF_grid_2024=num("EF_grid_2024"),
        EF_grid_2040=num("EF_grid_2040"),
        EF_fossil=num("EF_fossil"),
        P_carbon_2026=num("P_carbon_2026"),
        P_carbon_2035=num("P_carbon_2035"),
        P_carbon_2040=num("P_carbon_2040"),
        E_2024=num("E_2024"),
        N_resilience=num("N_resilience", cast=int),
        Deploy_max_liquid=num("Deploy_max_liquid"),
        Deploy_max_PV=num("Deploy_max_PV"),
        discount_rate=num("discount_rate"),
        curtail_penalty=num("curtail_penalty"),
        horizon_start=num("horizon_start", cast=int),
        horizon_end=num("horizon_end", cast=int),
        hours_per_year=num("hours_per_year", cast=int),
        rep_hours_per_year=num("rep_hours_per_year", cast=int),
        cooling_air_capex=num("cooling_air_capex"),
        cooling_dlc_capex=num("cooling_dlc_capex"),
        cooling_immersion_capex=num("cooling_immersion_capex"),
        bess_capex_2025=num("bess_capex_2025"),
        bess_capex_2035=num("bess_capex_2035"),
        bess_efficiency=num("bess_efficiency"),
        pv_lcoe=num("pv_lcoe"),
        pv_cf_avg=num("pv_cf_avg"),
        cgs_capex=num("cgs_capex"),
        cgs_eff_total=num("cgs_eff_total"),
        cgs_eff_elec=num("cgs_eff_elec"),
        pue_air_2024_avg=num("pue_air_2024_avg"),
        pue_dlc_avg=num("pue_dlc_avg"),
        pue_immersion_avg=num("pue_immersion_avg"),
        opex_om_rate=num("opex_om_rate"),
    )


def load_temperature_96h(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "temperature_96h.csv")
    return pd.read_csv(path)
