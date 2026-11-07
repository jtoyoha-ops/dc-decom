"""Options, policy settings, exogenous paths and the rolling state of DC-DECOM v2.

All classes are frozen dataclasses. Year-indexed values are plain dicts
``{year: value}``; region-and-year values are ``{region: {year: value}}``.
A year missing from a policy dict means "no such rule in that year".
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Optional, Tuple

TIME_MODES = ("annual", "rep96")
SITING_RULES = ("base_shares", "metro_only", "endogenous", "fixed_shares", "growth_split")
PPA_MATCHING = ("annual", "hourly")
EF_MODES = ("regional", "national")
RESILIENCE_RULES = ("eq", "legacy5")
BACKUP_TECHS = ("cgs", "gen")
IT_ENERGY_RULES = ("anchor_is_it", "legacy_util")

METRO = ("Kanto", "Kansai")
CALIBRATION_YEARS = (2024, 2025)
BASE_YEAR = 2024        # discounting base year and start of the CO2 cap path
END_YEAR = 2040         # net-zero year of the CO2 cap path (Eq. 13)
COOLING = ("air", "dlc", "imm")

YearMap = Dict[int, float]
RegionYearMap = Dict[str, Dict[int, float]]


@dataclass(frozen=True)
class ModelOptions:
    """Structural switches of the v2 LP.

    time_mode           'annual' (one slice of 8,760 h at the annual-mean
                        temperature) | 'rep96' (4 seasons x 24 h)
    siting              None = the scenario's rule, else one of SITING_RULES
    grid_constraints    connection constraint on non-metro facility peaks
    ppa_matching        'annual' (SHK market-based, annual kWh) | 'hourly'
    ef_mode             'regional' (Table 9 scaled) | 'national'
    resilience          'eq' (ch04 eq:resilience) | 'legacy5' (v1 5% rule)
    backup_techs        technologies counted as backup for 'eq'
    it_energy           'anchor_is_it' (demand path = IT energy, utilization
                        only sizes nameplate) | 'legacy_util' (v1: facility
                        energy x 0.8)
    certificates        allow non-fossil certificates nfc
    carbon_pass_through share of the carbon price paid by the operator
    connection_overrun  allow the costed slack 'over' in the connection
                        constraint (rolling realization only)
    co2_floor_share     None, or a share f: national emissions >= f x CO2 cap
                        after the calibration years (cap scenarios; bridge
                        step T4 of assumption set S2)
    """

    time_mode: str = "rep96"
    siting: Optional[str] = None
    grid_constraints: bool = True
    ppa_matching: str = "annual"
    ef_mode: str = "regional"
    resilience: str = "eq"
    backup_techs: Tuple[str, ...] = ("cgs", "gen")
    it_energy: str = "anchor_is_it"
    certificates: bool = False
    carbon_pass_through: float = 1.0
    connection_overrun: bool = False
    co2_floor_share: Optional[float] = None

    def __post_init__(self) -> None:
        checks = (("time_mode", TIME_MODES), ("ppa_matching", PPA_MATCHING),
                  ("ef_mode", EF_MODES), ("resilience", RESILIENCE_RULES),
                  ("it_energy", IT_ENERGY_RULES))
        for name, allowed in checks:
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.siting is not None and self.siting not in SITING_RULES:
            raise ValueError(f"siting must be None or one of {SITING_RULES}")
        if not self.backup_techs or not set(self.backup_techs) <= set(BACKUP_TECHS):
            raise ValueError(f"backup_techs must be a non-empty subset of {BACKUP_TECHS}")
        if not 0.0 <= self.carbon_pass_through <= 1.0:
            raise ValueError("carbon_pass_through must lie in [0, 1]")
        if self.co2_floor_share is not None and not 0.0 <= self.co2_floor_share <= 1.0:
            raise ValueError("co2_floor_share must be None or lie in [0, 1]")

    def with_(self, **changes) -> "ModelOptions":
        return replace(self, **changes)


@dataclass(frozen=True)
class PolicySettings:
    """Policy instruments and the siting floors.

    pue_fleet_max         {year: max fleet PUE} per region (annual PUE)
    pue_new_max           {year: max PUE of the capacity added in the year}
    liquid_floor_new      {year: phi}, liquid share of the capacity added
    conn_capacity         {region: {year: GW}} connection capacity offered in
                          the year to non-metro facility peaks; unused
                          capacity carries over (cumulative constraint)
    reinforcement_cost_yen_per_kW  cost per kW of connected peak added in
                          reinforced years (and of the overrun slack)
    reinforced_from       {region: first year in which added connection
                          capacity counts as reinforced (costed)}
    operator_burden_share theta, operator share of the reinforcement cost
    subsidy_yen_per_kW    {region: {year: yen per kW of new IT nameplate}}
    ppa_floor_subsidized  psi, PPA share required for subsidized build
    ppa_tier_multiplier   {region: factor on every PPA tier volume}
    metro_floor           {year: lambda}, (Kanto + Kansai) / national IT
    kanto_floor           {year: kappa}, Kanto / national IT
    liquid_fleet_max      {region: max liquid (DLC + immersion) share of the
                          regional IT load after the calibration years}
    """

    pue_fleet_max: YearMap = field(default_factory=dict)
    pue_new_max: YearMap = field(default_factory=dict)
    liquid_floor_new: YearMap = field(default_factory=dict)
    conn_capacity: RegionYearMap = field(default_factory=dict)
    reinforcement_cost_yen_per_kW: float = 0.0
    reinforced_from: Dict[str, int] = field(default_factory=dict)
    operator_burden_share: float = 1.0
    subsidy_yen_per_kW: RegionYearMap = field(default_factory=dict)
    ppa_floor_subsidized: Optional[float] = None
    ppa_tier_multiplier: Dict[str, float] = field(default_factory=dict)
    metro_floor: YearMap = field(default_factory=dict)
    kanto_floor: YearMap = field(default_factory=dict)
    liquid_fleet_max: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.operator_burden_share <= 1.0:
            raise ValueError("operator_burden_share must lie in [0, 1]")

    def with_(self, **changes) -> "PolicySettings":
        return replace(self, **changes)

    def subsidy(self, region: str, year: int) -> float:
        return float(self.subsidy_yen_per_kW.get(region, {}).get(year, 0.0))

    def is_reinforced(self, region: str, year: int) -> bool:
        start = self.reinforced_from.get(region)
        return start is not None and year >= start


@dataclass(frozen=True)
class Paths:
    """Exogenous paths that differ between actual-path realizations."""

    carbon_price: YearMap          # yen/t-CO2
    demand_TWh: YearMap            # national IT energy (anchor_is_it)
    liquid_deploy_cap: YearMap     # GW/yr/region, DLC + immersion additions

    def with_(self, **changes) -> "Paths":
        return replace(self, **changes)


@dataclass(frozen=True)
class State:
    """End-of-year state handed to a window that starts in ``year + 1``.

    capacities  {tech: {region: value}} for air, dlc, imm (GW IT nameplate),
                pv, cgs, gen (GW) and bess (GWh)
    itr         {region: average IT load, GW}
    peak        {region: connected peak, GW} (regions with a connection
                constraint)
    cumulative  {'conn_capacity' | 'conn_used' | 'subsidized_build':
                {region: value}}: connection capacity offered (incl. any
                overrun) and used since 2025 (GW), subsidized IT build (GW)
    """

    year: int
    capacities: Dict[str, Dict[str, float]]
    itr: Dict[str, float]
    peak: Dict[str, float] = field(default_factory=dict)
    cumulative: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Realization:
    """Fixings of a rolling realization window (``rolling``).

    itr        ((region, year, GW), ...) regional IT load of every region and
               year of the window; replaces the siting rule (actual national
               demand x planned share)
    new_fixed  ((tech, region, year, value), ...) additions fixed to the value
    new_min    ((tech, region, year, value), ...) lower bounds on additions
    new_max    ((tech, region, year, value), ...) upper bounds on additions
    air_tiebreak_yen_per_kW  objective weight on air additions (tie-break only;
               the LP has no air capex, so air additions are otherwise free and
               the realized air recourse is not unique). Not a cost in
               ``accounting``.
    Units as the capacities (GW; BESS GWh).
    """

    itr: Tuple[Tuple[str, int, float], ...]
    new_fixed: Tuple[Tuple[str, str, int, float], ...] = ()
    new_min: Tuple[Tuple[str, str, int, float], ...] = ()
    new_max: Tuple[Tuple[str, str, int, float], ...] = ()
    air_tiebreak_yen_per_kW: float = 0.0


def year_map(value: float, start: int, end: int = END_YEAR) -> YearMap:
    """``{year: value}`` for every year from ``start`` to ``end``."""
    return {int(t): float(value) for t in range(int(start), int(end) + 1)}


def region_year_map(values: Dict[str, float], start: int, end: int = END_YEAR,
                    regions: Optional[Iterable[str]] = None) -> RegionYearMap:
    keys = list(regions) if regions is not None else list(values)
    return {r: year_map(values[r], start, end) for r in keys if r in values}
