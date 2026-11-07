"""Time slices of DC-DECOM v2.

'annual': one slice of weight 8,760 h at the regional annual-mean temperature
          (``annual_mean_T_C``) and the annual-mean PV capacity factor; this is
          the year-unit version kept for comparison.
'rep96':  4 seasons x 24 hours; the weight of an hour is the number of days
          of its season (sum 8,760 h). Each season is one representative day;
          the hour before hour 0 is hour 23 of the same day (BESS wrap-around).

PV capacity factors come from the profile ``pv_cf_96h`` of the parameter set;
without it a flat ``pv_cf_avg`` is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .params import ParamSet

SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
HOURS_PER_YEAR = 8760.0


@dataclass(frozen=True)
class Slice:
    index: int
    season: str
    hour: int
    weight_h: float
    prev: int          # index of the previous hour of the same representative day


@dataclass(frozen=True)
class TimeSlices:
    mode: str
    slices: Tuple[Slice, ...]
    temperature: Dict[str, Tuple[float, ...]]
    cf: Dict[str, Tuple[float, ...]]

    @property
    def weights(self) -> np.ndarray:
        return np.array([s.weight_h for s in self.slices])

    def cf_avg(self, region: str) -> float:
        w = self.weights
        return float(np.dot(w, self.cf[region]) / w.sum())

    def days(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for s in self.slices:
            out.setdefault(s.season, []).append(s.index)
        return out


def _season_order(seasons: Sequence[str]) -> List[str]:
    uniq = list(dict.fromkeys(seasons))
    if set(uniq) == set(SEASON_ORDER):
        return list(SEASON_ORDER)
    return uniq


def make_timeslices(mode: str, params: ParamSet, regions: Sequence[str]) -> TimeSlices:
    temps = params.profile("temperature_96h")
    cfp = params.profile("pv_cf_96h")
    temperature: Dict[str, Tuple[float, ...]] = {}
    cf: Dict[str, Tuple[float, ...]] = {}
    if mode == "annual":
        temperature = {r: (params.region("annual_mean_T_C", r),) for r in regions}
        if cfp is None:
            cf = {r: (params.scalar("pv_cf_avg"),) for r in regions}
        else:
            if temps is None:
                raise ValueError("annual CF from a profile needs temperature_96h weights")
            merged = _merge(temps, cfp)
            for r in regions:
                g = merged[merged["region"] == r]
                cf[r] = (float(np.dot(g["weight_hours"], g["cf"]) / g["weight_hours"].sum()),)
        return TimeSlices("annual", (Slice(0, "ANN", 0, HOURS_PER_YEAR, 0),), temperature, cf)
    if mode != "rep96":
        raise ValueError(f"unknown time mode {mode!r}")
    if temps is None:
        raise ValueError("rep96 needs the temperature_96h profile")
    seasons = _season_order(list(temps["season"]))
    rank = {s: i for i, s in enumerate(seasons)}
    temps = temps.assign(_s=temps["season"].map(rank)).sort_values(["region", "_s", "hour"])
    ref = temps[temps["region"] == regions[0]]
    if [int(h) for h in ref["hour"]] != list(range(24)) * len(seasons):
        raise ValueError("temperature_96h must have hours 0-23 for every season and region")
    slices: List[Slice] = []
    for i, (_, row) in enumerate(ref.iterrows()):
        hour = int(row["hour"])
        prev = i + 23 if hour == 0 else i - 1
        slices.append(Slice(i, str(row["season"]), hour, float(row["weight_hours"]), prev))
    total = sum(s.weight_h for s in slices)
    if abs(total - HOURS_PER_YEAR) > 1e-6:
        raise ValueError(f"rep96 weights sum to {total}, expected 8760")
    for r in regions:
        g = temps[temps["region"] == r]
        if list(g["season"]) != [s.season for s in slices] or \
                list(g["hour"]) != [s.hour for s in slices] or \
                not np.allclose(g["weight_hours"].to_numpy(), [s.weight_h for s in slices]):
            raise ValueError(f"temperature_96h slices of {r} differ from {regions[0]}")
        temperature[r] = tuple(float(x) for x in g["temperature_C"])
        if cfp is None:
            cf[r] = tuple([params.scalar("pv_cf_avg")] * len(slices))
        else:
            c = cfp[cfp["region"] == r].set_index(["season", "hour"])["cf"]
            cf[r] = tuple(float(c.loc[(s.season, s.hour)]) for s in slices)
    return TimeSlices("rep96", tuple(slices), temperature, cf)


def _merge(temps, cfp):
    return temps.merge(cfp, on=["region", "season", "hour"], how="inner", validate="one_to_one")
