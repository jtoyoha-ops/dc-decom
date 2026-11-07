"""Seasonal-diurnal profiles from the JMA hourly observations (DC-DECOM v2).

Input: data/raw/jma/jma_hourly_<station>_<city>_2020-2023.csv as downloaded by
data/raw/jma/fetch_jma.py (cp932; columns: timestamp, temperature, quality, homogeneity,
global solar radiation MJ/m2 over the previous hour, quality, homogeneity).

Rules:
- A slice (date, hour h) is the local-time hour [h:00, h+1:00) of that date (JST).
  Temperature: the JMA value stamped k:00 is the reading at k:00 and goes to hour k of
  its date (the row stamped 00:00 of the next day, JMA's 24:00, is hour 0 of that day).
  Solar radiation: the value stamped k:00 is the sum over (k-1, k] and goes to the hour
  that starts at k-1. MJ/m2 per hour / 3.6 = kWh/m2 per hour (= mean kW/m2).
- Only dates 2020-01-01 .. 2023-12-31 are used. Values with JMA quality 8 (normal) or 5
  (quasi-normal) are used; quality 4 or lower (insufficient, doubtful, missing) is dropped.
- Seasons DJF / MAM / JJA / SON by the month of the date; profile value = mean over all
  valid observations of (season, hour). weight_hours = days of the season in a 365-day
  year (90 / 92 / 92 / 91; sum 8,760 h, as v1).
- PV capacity factor = GHI (kWh/m2 per hour) x performance ratio PR (assumption, base 0.80,
  range 0.75-0.85), i.e. per kW at STC (1 kW/m2) on a horizontal plane.

Outputs (data/parameters_v2/): temperature_96h_jma.csv (region, season, hour,
temperature_C, weight_hours, n_obs), pv_cf_96h_jma.csv (region, season, hour, cf,
ghi_kWh_per_m2, weight_hours, n_obs), jma_summary.csv (annual means per region).
Prints the annual-mean temperature and the annual CF per region next to Energies
Table 9 temperatures and the Table 6 CF range 13-17%.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "jma"
OUT = ROOT / "parameters_v2"
STATIONS = (
    ("Hokkaido", "Sapporo", "47412"),
    ("Tohoku", "Sendai", "47590"),
    ("Kanto", "Tokyo", "47662"),
    ("Kansai", "Osaka", "47772"),
    ("Kyushu", "Fukuoka", "47807"),
)
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
SEASON_DAYS = {"DJF": 90, "MAM": 92, "JJA": 92, "SON": 91}
VALID_QUALITY = (8, 5)
PR_BASE = 0.80
FIRST, LAST = pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31")
TABLE9_T = {"Hokkaido": 9.2, "Tohoku": 12.8, "Kanto": 16.5, "Kansai": 17.1, "Kyushu": 17.3}


def read_raw(station: str, city: str) -> pd.DataFrame:
    path = RAW / f"jma_hourly_{station}_{city}_2020-2023.csv"
    text = path.read_bytes().decode("cp932")
    lines = text.splitlines()
    head = [ln for ln in lines[:6] if ln.startswith("年月日時")]
    if not head or "気温" not in head[0] or "日射量" not in head[0]:
        raise ValueError(f"{path.name}: unexpected header {lines[:6]}")
    start = next(i for i, ln in enumerate(lines) if ln[:4].isdigit())
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), header=None,
                     names=["ts", "T", "T_q", "T_h", "R", "R_q", "R_h"])
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M:%S")
    return df


def slices(df: pd.DataFrame, value: str, quality: str, shift_h: int) -> pd.DataFrame:
    ok = df[quality].isin(VALID_QUALITY) & df[value].notna()
    start = df.loc[ok, "ts"] - pd.Timedelta(hours=shift_h)
    out = pd.DataFrame({"date": start.dt.normalize(), "hour": start.dt.hour,
                        "v": df.loc[ok, value].astype(float)})
    out = out[(out["date"] >= FIRST) & (out["date"] <= LAST)]
    out["season"] = out["date"].dt.month.map(SEASON_OF_MONTH)
    return out


def profile(sl: pd.DataFrame) -> pd.DataFrame:
    g = sl.groupby(["season", "hour"])["v"].agg(["mean", "count"]).reset_index()
    g["_s"] = g["season"].map({s: i for i, s in enumerate(SEASON_ORDER)})
    g = g.sort_values(["_s", "hour"]).drop(columns="_s").reset_index(drop=True)
    if len(g) != 96:
        raise ValueError("profile does not have 96 slices")
    g["weight_hours"] = g["season"].map(SEASON_DAYS).astype(float)
    return g


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    temps, cfs, summary = [], [], []
    for region, city, station in STATIONS:
        df = read_raw(station, city)
        t = profile(slices(df, "T", "T_q", 0))
        r = profile(slices(df, "R", "R_q", 1))
        temps.append(pd.DataFrame({
            "region": region, "season": t["season"], "hour": t["hour"],
            "temperature_C": t["mean"].round(4), "weight_hours": t["weight_hours"],
            "n_obs": t["count"]}))
        ghi = r["mean"] / 3.6
        cfs.append(pd.DataFrame({
            "region": region, "season": r["season"], "hour": r["hour"],
            "cf": (ghi * PR_BASE).round(6), "ghi_kWh_per_m2": ghi.round(6),
            "weight_hours": r["weight_hours"], "n_obs": r["count"]}))
        w = t["weight_hours"].to_numpy()
        t_ann = float(np.dot(w, t["mean"]) / w.sum())
        ghi_ann = float(np.dot(r["weight_hours"], ghi))  # kWh/m2/yr
        summary.append({
            "region": region, "city": city, "station": station,
            "annual_mean_T_C": round(t_ann, 4),
            "table9_T_C": TABLE9_T[region],
            "ghi_kWh_per_m2_yr": round(ghi_ann, 2),
            "cf_annual_pr075": round(ghi_ann * 0.75 / 8760, 5),
            "cf_annual_pr080": round(ghi_ann * PR_BASE / 8760, 5),
            "cf_annual_pr085": round(ghi_ann * 0.85 / 8760, 5),
            "T_max_slice_C": round(float(t["mean"].max()), 4),
            "T_n_obs": int(t["count"].sum()), "R_n_obs": int(r["count"].sum()),
        })
    pd.concat(temps).to_csv(OUT / "temperature_96h_jma.csv", index=False)
    pd.concat(cfs).to_csv(OUT / "pv_cf_96h_jma.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "jma_summary.csv", index=False)
    print("JMA 2020-2023, 96-slice weighted annual means (Table 9 T and Table 6 CF 13-17% for "
          "comparison)")
    print(s[["region", "city", "annual_mean_T_C", "table9_T_C", "ghi_kWh_per_m2_yr",
             "cf_annual_pr075", "cf_annual_pr080", "cf_annual_pr085"]].to_string(index=False))


if __name__ == "__main__":
    main()
