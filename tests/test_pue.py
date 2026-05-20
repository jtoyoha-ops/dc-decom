"""Verify the PUE module against the paper's five regional calibration points."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dc_decom.pue import PUEParameters, pue_air, pue_liquid, pue_for_technology


CITIES = [
    # (city, mean_T, observed_air_lo, observed_air_hi)
    ("Sapporo",  9.2, 1.20, 1.30),
    ("Sendai",  12.8, 1.30, 1.40),
    ("Tokyo",   16.5, 1.40, 1.60),
    ("Osaka",   17.1, 1.40, 1.60),
    ("Fukuoka", 17.3, 1.40, 1.60),
]


def test_air_pue_within_empirical_range() -> None:
    """Each city's modelled air PUE must lie within the observed range."""
    p = PUEParameters()
    for city, T, lo, hi in CITIES:
        modelled = pue_air(T, p)
        assert lo <= float(modelled) <= hi, (
            f"{city}: air PUE {modelled} outside [{lo}, {hi}]"
        )


def test_liquid_pue_below_paper_average() -> None:
    """Liquid PUE should be <= 1.10 for the Japan temperature range."""
    p = PUEParameters()
    for _, T, _, _ in CITIES:
        modelled = pue_liquid(T, p)
        assert 1.05 <= float(modelled) <= 1.10, (
            f"liquid PUE at {T} degC = {modelled}"
        )


def test_pue_dispatch_by_technology() -> None:
    assert pue_for_technology("air", 16.5) > pue_for_technology("dlc", 16.5)
    assert pue_for_technology("dlc", 16.5) > pue_for_technology("immersion", 16.5)


def test_immersion_below_dlc() -> None:
    """MHI immersion (1.05) < Vertiv DLC (1.10-1.20)."""
    immersion = float(pue_for_technology("immersion", 12.0))
    dlc = float(pue_for_technology("dlc", 12.0))
    assert immersion < dlc
