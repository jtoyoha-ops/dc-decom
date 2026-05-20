"""Generate the 96-hour representative temperature profiles for 5 regions.

The profile is derived from JMA monthly normals (2020-2023 averages) with a
seasonal-diurnal decomposition: for each region we use the annual mean as the
baseline, apply a seasonal offset (DJF -12, MAM 0, JJA +12, SON 0 relative to
annual mean) and a diurnal swing of about +/- 5 C (with daily minimum around
05:00 and maximum around 14:00).

Running this script regenerates data/parameters_csv/temperature_96h.csv.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "parameters_csv"

# Annual mean temperatures (degC), aligned with paper Table 9
REGIONS = {
    "Hokkaido": 9.2,
    "Tohoku": 12.8,
    "Kanto": 16.5,
    "Kansai": 17.1,
    "Kyushu": 17.3,
}

# Seasonal offset relative to annual mean (degC)
SEASON_OFFSET = {
    "DJF": -12.0,   # winter (Dec-Feb)
    "MAM": 0.0,     # spring (Mar-May)
    "JJA": 12.0,    # summer (Jun-Aug)
    "SON": 0.0,     # autumn (Sep-Nov)
}

# Number of days per season for the 8760-hour weighting
SEASON_DAYS = {
    "DJF": 90,
    "MAM": 92,
    "JJA": 92,
    "SON": 91,
}
# Sanity: sum should be 365 days -> 8760 hours
assert sum(SEASON_DAYS.values()) == 365

DIURNAL_AMPLITUDE = 5.0    # +/- 5 degC swing
DIURNAL_PEAK_HOUR = 14.0   # hottest moment of the day


def diurnal_offset(hour: int) -> float:
    """Diurnal temperature offset (degC) from the daily mean for the given hour.

    Uses a smooth sinusoidal profile peaked at DIURNAL_PEAK_HOUR.
    """
    phase = 2.0 * math.pi * (hour - DIURNAL_PEAK_HOUR) / 24.0
    return DIURNAL_AMPLITUDE * math.cos(phase)


def generate() -> pd.DataFrame:
    rows = []
    for region, annual_mean in REGIONS.items():
        for season, season_offset in SEASON_OFFSET.items():
            for hour in range(24):
                temp = annual_mean + season_offset + diurnal_offset(hour)
                weight = SEASON_DAYS[season]   # number of days the rep hour stands for
                rows.append({
                    "region": region,
                    "season": season,
                    "hour": hour,
                    "temperature_C": round(temp, 2),
                    "weight_hours": float(weight),
                })
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    df = generate()
    out_path = DATA_DIR / "temperature_96h.csv"
    write_csv(df, out_path)
    print(f"Wrote {len(df)} rows to {out_path}")
    # Quick sanity check
    for region, group in df.groupby("region"):
        # Weighted mean should approximately reproduce annual mean
        wmean = (group["temperature_C"] * group["weight_hours"]).sum() / group["weight_hours"].sum()
        target = REGIONS[region]
        print(f"  {region:10s} mean recomputed = {wmean:5.2f} (target = {target})")


if __name__ == "__main__":
    main()
