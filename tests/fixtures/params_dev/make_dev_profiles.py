"""Write the DEV fixture files paths.csv and pv_cf_96h.csv (test placeholders).

PV capacity factor: a half-sine between sunrise and sunset per season, scaled
so that the 8,760-hour weighted mean of each region equals a placeholder
annual CF inside Energies Table 6 (13-17%). Paths: linear placeholders.
Run: .venv/bin/python tests/fixtures/params_dev/make_dev_profiles.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SEASONS = {  # season: (days, sunrise h, sunset h, relative peak)
    "DJF": (90, 7.0, 17.0, 0.70),
    "MAM": (92, 5.5, 18.5, 0.95),
    "JJA": (92, 5.0, 19.0, 1.00),
    "SON": (91, 6.0, 17.5, 0.80),
}
ANNUAL_CF = {"Hokkaido": 0.13, "Tohoku": 0.14, "Kanto": 0.14, "Kansai": 0.15, "Kyushu": 0.16}
YEARS = range(2024, 2041)


def shape(season: str, hour: int) -> float:
    _, rise, sset, peak = SEASONS[season]
    x = (hour + 0.5 - rise) / (sset - rise)
    return peak * math.sin(math.pi * x) if 0.0 < x < 1.0 else 0.0


def main() -> None:
    rows = []
    for region, target in ANNUAL_CF.items():
        raw = {(s, h): shape(s, h) for s in SEASONS for h in range(24)}
        mean = sum(SEASONS[s][0] * v for (s, _), v in raw.items()) / 8760.0
        for (s, h), v in raw.items():
            rows.append({"region": region, "season": s, "hour": h,
                         "cf": round(v * target / mean, 6), "status": "dev"})
    pd.DataFrame(rows).to_csv(HERE / "pv_cf_96h.csv", index=False)

    prow = []
    for t in YEARS:
        frac = (t - 2024) / 16.0
        metro = 0.82 + (0.55 - 0.82) * frac
        for pid, val, unit in (("metro_floor", metro, "fraction"),
                               ("kanto_floor", metro * 60.0 / 82.0, "fraction"),
                               ("ppa_volume_factor", 0.30 + 0.70 * frac, "factor")):
            prow.append({"id": pid, "year": t, "group": "path", "value": round(val, 6),
                         "low": "", "high": "", "unit": unit, "status": "dev",
                         "source_id": "DEV"})
    pd.DataFrame(prow).to_csv(HERE / "paths.csv", index=False)


if __name__ == "__main__":
    main()
