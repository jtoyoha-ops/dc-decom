"""Scenario definitions (paper Table 7).

Each scenario is a simple data container that toggles the few choices that
differ between REF, C-CN and R-CN. The numerical parameters themselves come
from ``GlobalParameters`` so that all three scenarios remain consistent.

v2.0.0: ``share_2040`` is a mapping by region name. Up to v1.0.1 it was a
tuple ordered (Kanto, Kansai, Hokkaido, Tohoku, Kyushu) that the model
applied in the row order of regions.csv (Hokkaido, Tohoku, Kanto, Kansai,
Kyushu), so the 2040 targets went to the wrong regions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# Region "tiers" used for the geographic constraints
METRO = ("Kanto", "Kansai")
REGIONAL = ("Hokkaido", "Tohoku", "Kyushu")

# Base-year shares of IT load (paper Section 3.4, Table 9).
BASE_SHARES: Dict[str, float] = {
    "Kanto": 0.60, "Kansai": 0.22, "Hokkaido": 0.03, "Tohoku": 0.05, "Kyushu": 0.10,
}
# R-CN's geographic redistribution by 2040 - paper Section 4.5
# Approximately: Kanto 35-40%, Kansai 17%, Hokkaido 25-30%, Tohoku 7%, Kyushu 15-20%
R_CN_SHARES_2040: Dict[str, float] = {
    "Kanto": 0.375, "Kansai": 0.170, "Hokkaido": 0.275, "Tohoku": 0.070, "Kyushu": 0.110,
}


@dataclass(frozen=True)
class Scenario:
    """Bundle of scenario-specific flags / overrides."""

    name: str
    apply_co2_cap: bool          # whether Eq. 13 emission cap is enforced
    allow_regional_growth: bool  # whether new IT load can move to regional hubs
    regional_subsidy: bool       # 5% effective CAPEX discount in regional hubs
    description: str

    # The 2040 share target by region name, used to drive R-CN's spatial
    # reallocation (baseline = REF/C-CN: status quo)
    share_2040: Dict[str, float] = field(default_factory=lambda: dict(BASE_SHARES))


REF = Scenario(
    name="REF",
    apply_co2_cap=False,
    allow_regional_growth=False,
    regional_subsidy=False,
    description="Reference: current trends continue, no carbon cap.",
)

C_CN = Scenario(
    name="C-CN",
    apply_co2_cap=True,
    allow_regional_growth=False,
    regional_subsidy=False,
    description="Centralised carbon-neutral: net-zero by 2040, "
                "new capacity restricted to Kanto/Kansai.",
    # In C-CN the share is held at the 2024 baseline to model 'no relocation'
    share_2040=dict(BASE_SHARES),
)

R_CN = Scenario(
    name="R-CN",
    apply_co2_cap=True,
    allow_regional_growth=True,
    regional_subsidy=True,
    description="Regional carbon-neutral: net-zero by 2040, "
                "load shifts to Hokkaido and Kyushu.",
    share_2040=dict(R_CN_SHARES_2040),
)


ALL_SCENARIOS: Dict[str, Scenario] = {s.name: s for s in (REF, C_CN, R_CN)}
