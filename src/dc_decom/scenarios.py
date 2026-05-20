"""Scenario definitions (paper Table 7).

Each scenario is a simple data container that toggles the few choices that
differ between REF, C-CN and R-CN. The numerical parameters themselves come
from ``GlobalParameters`` so that all three scenarios remain consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Region "tiers" used for the geographic constraints
METRO = ("Kanto", "Kansai")
REGIONAL = ("Hokkaido", "Tohoku", "Kyushu")


@dataclass(frozen=True)
class Scenario:
    """Bundle of scenario-specific flags / overrides."""

    name: str
    apply_co2_cap: bool          # whether Eq. 13 emission cap is enforced
    allow_regional_growth: bool  # whether new IT load can move to regional hubs
    regional_subsidy: bool       # 5% effective CAPEX discount in regional hubs
    description: str

    # The 2040 share split target used to drive R-CN's spatial reallocation
    # (Kanto, Kansai, Hokkaido, Tohoku, Kyushu)
    share_2040: Tuple[float, float, float, float, float] = (
        0.60, 0.22, 0.03, 0.05, 0.10  # baseline = REF/C-CN: status quo
    )


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
    share_2040=(0.60, 0.22, 0.03, 0.05, 0.10),
)

R_CN = Scenario(
    name="R-CN",
    apply_co2_cap=True,
    allow_regional_growth=True,
    regional_subsidy=True,
    description="Regional carbon-neutral: net-zero by 2040, "
                "load shifts to Hokkaido and Kyushu.",
    # R-CN's geographic redistribution by 2040 - paper Section 4.5
    # Approximately: Kanto 35-40%, Kansai 17%, Hokkaido 25-30%, Tohoku 7%, Kyushu 15-20%
    share_2040=(0.375, 0.170, 0.275, 0.070, 0.110),
)


ALL_SCENARIOS: Dict[str, Scenario] = {s.name: s for s in (REF, C_CN, R_CN)}
