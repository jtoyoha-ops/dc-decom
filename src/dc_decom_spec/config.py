"""Switches of the spec version: implementation-error fixes and readings.

``SpecFixes`` toggles the fixes E1-E8 of implementation errors (code that
contradicts an explicit statement of the paper). ``SpecReadings`` selects how
points the paper does not state are read; its defaults reproduce the v1 code.
``OVERRIDE_KEYS`` lists the sensitivity overrides accepted by
``model.build_and_solve``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Dict, Iterable, Optional, Tuple

# Order of the fixes E1-E8.
FIX_ORDER: Tuple[str, ...] = (
    "region_order",       # E1
    "bess_charge",        # E2
    "n_plus_1",           # E3
    "calibration_years",  # E4
    "regional_ef",        # E5
    "table9_inputs",      # E6
    "re_cap_eq15",        # E7
    "regional_subsidy",   # E8
)
FIX_IDS: Dict[str, str] = {name: f"E{i + 1}" for i, name in enumerate(FIX_ORDER)}


@dataclass(frozen=True)
class SpecFixes:
    """Implementation-error fixes (True = fixed as the paper states)."""

    region_order: bool = True       # E1 share_2040 applied by region name
    bess_charge: bool = True        # E2 BESS discharge needs charging (Eq. 12)
    n_plus_1: bool = True           # E3 backup >= (1 + 1/N) x critical load (Eq. 16)
    calibration_years: bool = True  # E4 2024-2025 fixed at observed values (Section 3.9)
    regional_ef: bool = True        # E5 regional grid EF(r, t) (Eq. 10, Table 9)
    table9_inputs: bool = True      # E6 regional inputs of Table 9 / Section 3.2
    re_cap_eq15: bool = True        # E7 Cap_PV + Cap_PPA <= RE_max (Eq. 15)
    regional_subsidy: bool = True   # E8 5% regional CAPEX subsidy in R-CN (Table 7)

    @classmethod
    def none(cls) -> "SpecFixes":
        """All fixes off: the v1 behaviour."""
        return cls(**{f.name: False for f in fields(cls)})

    @classmethod
    def only(cls, names: Iterable[str]) -> "SpecFixes":
        """Only the named fixes on."""
        names = set(names)
        unknown = names - set(FIX_ORDER)
        if unknown:
            raise ValueError(f"unknown fixes: {sorted(unknown)}")
        return cls(**{f.name: (f.name in names) for f in fields(cls)})

    def ids(self) -> Tuple[str, ...]:
        return tuple(FIX_IDS[n] for n in FIX_ORDER if getattr(self, n))


PPA_CEILING_CHOICES = ("code", "none")
CCN_RULE_CHOICES = ("same_shares", "metro_only")
CARBON_CHOICES = ("mid", "low", "high")


@dataclass(frozen=True)
class SpecReadings:
    """Readings of points the paper does not state (defaults = v1 code)."""

    ppa_ceiling: str = "code"                # 'code' share ceilings | 'none'
    pv_onsite_cap: Optional[float] = 0.10    # PV energy share of load; None = no cap
    cgs_energy_cap: Optional[float] = 0.05   # CGS energy share of load; None = no cap
    cgs_cf: Optional[float] = 0.5            # CGS capacity factor; None = physical 1.0
    bess_cycles: Optional[float] = 250.0     # full cycles per year; None = no limit
    utilization: float = 0.80                # applied to average IT power
    siting_premium: float = 1.10             # metro multiplier on CAPEX/O&M
    ccn_rule: str = "same_shares"            # 'same_shares' | 'metro_only'
    carbon: str = "mid"                      # 'mid' | 'low' | 'high'

    def __post_init__(self) -> None:
        if self.ppa_ceiling not in PPA_CEILING_CHOICES:
            raise ValueError(f"ppa_ceiling must be one of {PPA_CEILING_CHOICES}")
        if self.ccn_rule not in CCN_RULE_CHOICES:
            raise ValueError(f"ccn_rule must be one of {CCN_RULE_CHOICES}")
        if self.carbon not in CARBON_CHOICES:
            raise ValueError(f"carbon must be one of {CARBON_CHOICES}")

    def with_(self, **changes) -> "SpecReadings":
        return replace(self, **changes)


# Alternative reading 'paper_only', declared before the spec runs: every
# code-only bound the paper does not state is removed.
PAPER_ONLY_READINGS = SpecReadings(
    ppa_ceiling="none",
    pv_onsite_cap=None,
    cgs_energy_cap=None,
    cgs_cf=None,
    bess_cycles=None,
    siting_premium=1.00,
)

# Carbon-price anchor points (2026, 2035, 2040) in yen/t-CO2 for the readings
# 'low' and 'high' (Table 8 ranges); 2040 = 2035 x 18/11, the ratio of the
# code's 18,000 / 11,000. 'mid' uses global_parameters.csv.
CARBON_2040_RATIO = 18000.0 / 11000.0
CARBON_POINTS: Dict[str, Tuple[float, float, float]] = {
    "low": (1700.0, 7000.0, 7000.0 * CARBON_2040_RATIO),
    "high": (4300.0, 15000.0, 15000.0 * CARBON_2040_RATIO),
}

# Sensitivity overrides accepted by build_and_solve(overrides=...).
OVERRIDE_KEYS = (
    "carbon_price",               # {year: yen/t-CO2}, replaces the whole path
    "carbon_points",              # (P2026, P2035, P2040) yen/t-CO2
    "ppa_offset",                 # yen/kWh added to every region's PPA price
    "capex",                      # {'dlc'|'immersion': thousand yen/kW}
    "pue",                        # {PUEParameters field: value}
    "pue_air_poly",               # (a, b, c): PUE_air(T) = a + b T + c T^2
    "ai_factor",                  # {year: multiplier on the AI portion of demand}
    "deploy_max_liquid",          # GW/yr/region
    "discount_rate",              # fraction
    "nonmetro_adder_yen_per_kW",  # one-time cost per kW of new non-metro IT capacity
)


def ai_factor_path(years: Iterable[int], g_ai: float = 0.30, hw_rate: float = 0.30,
                   base_year: int = 2024) -> Dict[int, float]:
    """Multiplier on the AI portion of national demand.

    ((1 + g_ai) / 1.30)^(t - 2024) x (1.30 / (1 + hw_rate))^(t - 2024): the
    code's demand path embeds g_AI = 30%/yr and eta_HW = 30%/yr (Table 7), so
    both factors equal 1 at the baseline values.
    """
    out: Dict[int, float] = {}
    for t in years:
        n = t - base_year
        out[int(t)] = ((1.0 + g_ai) / 1.30) ** n * (1.30 / (1.0 + hw_rate)) ** n
    return out
