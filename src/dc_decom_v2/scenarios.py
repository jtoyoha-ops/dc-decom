"""Scenarios of DC-DECOM v2 and their default paths and policy settings.

REF         base-year shares, no CO2 cap
C-CN        base-year shares, CO2 cap (Energies Section 3.11 reading)
C-CN-metro  non-metro frozen at 2025, metro endogenous, CO2 cap (Table 7 reading)
R-CN        endogenous siting, CO2 cap, non-metro siting subsidy
R-CN0       endogenous siting, CO2 cap, no subsidy
R-CN-pub    R-CN 2040 shares of the v1.0.1 code fixed by region name, CO2 cap
REF-P0      REF with a zero carbon price
A0          endogenous siting, no CO2 cap, no added policy (A/B/C baseline)

Bridge scenarios (bridge step T1 of assumption set S2, assumptions whose
values are not stated in Energies; config/experiments_v2.yaml):
C-CN-gs     growth after 2025 split 73:27 between Kanto and Kansai, non-metro
            frozen at 2025, CO2 cap
R-CN-s2     2040 shares of assumption set S2 fixed by region name, CO2 cap

Default policy: connection capacity ``conn_capacity_GW_per_yr`` of the
parameter set for every year after the calibration years (non-metro regions
that have the value), no reinforcement, ``operator_burden_share``, the siting
floors from the paths ``metro_floor`` and ``kanto_floor`` and, for R-CN, the
subsidy ``siting_subsidy_yen_per_kW`` for new non-metro IT nameplate from
2026. No PUE rule, liquid floor or PPA floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Union

from dc_decom_spec.scenarios import R_CN_SHARES_2040

from .config import CALIBRATION_YEARS, END_YEAR, METRO, Paths, PolicySettings
from .params import ParamSet

# Assumption set S2 (bridge step T1): R-CN 2040 shares by region name (no source).
SHARES_S2_2040: Dict[str, float] = {
    "Hokkaido": 0.27, "Tohoku": 0.08, "Kanto": 0.35, "Kansai": 0.15, "Kyushu": 0.15,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    siting: str
    co2_cap: bool
    subsidy: bool = False
    zero_carbon_price: bool = False
    fixed_shares: Optional[Dict[str, float]] = None
    growth_split: Optional[Dict[str, float]] = None
    description: str = ""


SCENARIOS: Dict[str, Scenario] = {s.name: s for s in (
    Scenario("REF", "base_shares", False, description="base-year shares, no cap"),
    Scenario("C-CN", "base_shares", True, description="base-year shares, CO2 cap"),
    Scenario("C-CN-metro", "metro_only", True,
             description="non-metro frozen at 2025, metro endogenous, CO2 cap"),
    Scenario("R-CN", "endogenous", True, subsidy=True,
             description="endogenous siting, CO2 cap, non-metro subsidy"),
    Scenario("R-CN0", "endogenous", True, description="endogenous siting, CO2 cap"),
    Scenario("R-CN-pub", "fixed_shares", True, fixed_shares=dict(R_CN_SHARES_2040),
             description="R-CN 2040 shares of the v1.0.1 code by name, CO2 cap"),
    Scenario("REF-P0", "base_shares", False, zero_carbon_price=True,
             description="REF with zero carbon price"),
    Scenario("A0", "endogenous", False, description="endogenous siting, no cap, no policy"),
    Scenario("C-CN-gs", "growth_split", True, growth_split={"Kanto": 0.73, "Kansai": 0.27},
             description="growth after 2025 split 73:27 Kanto:Kansai, CO2 cap"),
    Scenario("R-CN-s2", "fixed_shares", True, fixed_shares=dict(SHARES_S2_2040),
             description="2040 shares of assumption set S2 by name, CO2 cap"),
)}


def get(scenario: Union[str, Scenario]) -> Scenario:
    if isinstance(scenario, Scenario):
        return scenario
    try:
        return SCENARIOS[scenario]
    except KeyError:
        raise KeyError(f"unknown scenario {scenario!r}; known: {sorted(SCENARIOS)}") from None


def default_paths(params: ParamSet, variant: str = "value") -> Paths:
    return Paths(carbon_price=params.path("carbon_price", variant),
                 demand_TWh=params.path("demand_TWh", variant),
                 liquid_deploy_cap=params.path("liquid_deploy_cap", variant))


def default_policy(scenario: Union[str, Scenario], params: ParamSet,
                   years: Optional[Iterable[int]] = None) -> PolicySettings:
    sc = get(scenario)
    yrs = list(years) if years is not None else list(range(CALIBRATION_YEARS[0], END_YEAR + 1))
    later = [t for t in yrs if t > CALIBRATION_YEARS[-1]]
    nonmetro = [r for r in params.region_names if r not in METRO]
    conn = {r: {t: params.region("conn_capacity_GW_per_yr", r) for t in later}
            for r in nonmetro if params.has_region("conn_capacity_GW_per_yr", r)}
    subsidy: Dict[str, Dict[int, float]] = {}
    if sc.subsidy:
        s = params.scalar("siting_subsidy_yen_per_kW")
        subsidy = {r: {t: s for t in later} for r in nonmetro}
    floors = {}
    for key, pid in (("metro_floor", "metro_floor"), ("kanto_floor", "kanto_floor")):
        floors[key] = params.path(pid) if params.has_path(pid) else {}
    return PolicySettings(
        conn_capacity=conn,
        reinforcement_cost_yen_per_kW=params.scalar("reinforcement_cost_yen_per_kW", 0.0),
        operator_burden_share=params.scalar("operator_burden_share", 1.0),
        subsidy_yen_per_kW=subsidy,
        metro_floor=floors["metro_floor"],
        kanto_floor=floors["kanto_floor"],
    )
