"""Recalibrate the air-cooling PUE of Energies Eq. 8 for hourly evaluation (DC-DECOM v2).

PUE_air(T) = C_base + alpha T + beta H(T - T_thresh) (T - T_thresh)^2, T_thresh = 15 C fixed.
The v2 model evaluates PUE per representative hour and weights the hours, so the annual
PUE of a city is the weighted mean over the 96 slices of temperature_96h_jma.csv. That
weighted mean is linear in the coefficients: regressors per city are the weighted means of
1, T and H(T - 15)(T - 15)^2. Targets are the midpoints of the survey ranges of Energies
Table 5 / dissertation ch04 tab:pue-validation (Sapporo 1.2-1.3, Sendai 1.3-1.4, Tokyo,
Osaka, Fukuoka 1.4-1.6). 5 targets, 3 coefficients: residual degrees of freedom 2. The
targets are survey ranges, not model results (no LP is solved here).

Two fits are reported:
- 'bounds_only' (alpha, beta >= 0 only): scipy.optimize.lsq_linear, alpha >= 0,
  beta >= 0, C_base free.
- 'base' (used in data/parameters_v2/parameters.csv): the same least-squares problem with
  the definitional constraint PUE_air >= 1 in every slice of every city (facility energy
  cannot be below IT energy), solved with scipy.optimize.minimize (SLSQP). It is used
  because the bounds-only fit gives PUE < 1 in cold slices (see 'physical_check').
  Coefficients are rounded to 6 decimals; with the rounded alpha and beta, C_base is the
  least-squares value raised (if needed) to the smallest 6-decimal value that keeps every
  slice at PUE >= 1. All reported PUE values use the rounded coefficients.

For comparison the paper coefficients (1.14, 0.0167, 0.0102) are evaluated (a) at the
Table 9 annual-mean temperatures, (b) at the JMA 2020-2023 annual means and (c) hourly
weighted with the JMA profile. Peak-slice air PUE per city is reported for every set.
Liquid cooling (Eq. 9) is linear in T, so the hourly weighted mean equals the value at
the mean temperature; its coefficients are kept from Energies Table 8.

Output: data/parameters_v2/pue_calibration.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, lsq_linear, minimize

ROOT = Path(__file__).resolve().parent
P2 = ROOT / "parameters_v2"
REGIONS = ("Hokkaido", "Tohoku", "Kanto", "Kansai", "Kyushu")
CITY = {"Hokkaido": "Sapporo", "Tohoku": "Sendai", "Kanto": "Tokyo", "Kansai": "Osaka",
        "Kyushu": "Fukuoka"}
RANGES = {"Hokkaido": (1.2, 1.3), "Tohoku": (1.3, 1.4), "Kanto": (1.4, 1.6),
          "Kansai": (1.4, 1.6), "Kyushu": (1.4, 1.6)}
LIQUID_RANGE = (1.05, 1.10)
TABLE9_T = {"Hokkaido": 9.2, "Tohoku": 12.8, "Kanto": 16.5, "Kansai": 17.1, "Kyushu": 17.3}
T_THRESH = 15.0
PUE_FLOOR = 1.0
PAPER = {"C_base": 1.14, "alpha": 0.0167, "beta": 0.0102}
C_PUMP, GAMMA, IMM_OFFSET = 0.055, 0.0015, 0.025  # Energies Table 8; immersion offset code-only


def hinge2(T):
    return np.maximum(np.asarray(T, dtype=float) - T_THRESH, 0.0) ** 2


def air(T, c):
    T = np.asarray(T, dtype=float)
    return c["C_base"] + c["alpha"] * T + c["beta"] * hinge2(T)


def r6(x) -> float:
    return float(round(float(x), 6))


def r2_stats(y, yhat, k=3):
    n = len(y)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    dof = n - k
    return {"R2": r6(r2), "adj_R2": r6(1.0 - (1.0 - r2) * (n - 1) / dof), "ss_res": r6(ss_res),
            "ss_tot": r6(ss_tot), "residual_se": r6(math.sqrt(ss_res / dof)), "n": n, "k": k,
            "dof": dof}


def main() -> None:
    t = pd.read_csv(P2 / "temperature_96h_jma.csv")
    prof, rows, slices = {}, [], []
    for r in REGIONS:
        g = t[t["region"] == r]
        w = g["weight_hours"].to_numpy() / g["weight_hours"].sum()
        T = g["temperature_C"].to_numpy()
        prof[r] = (w, T)
        rows.append([1.0, float(np.dot(w, T)), float(np.dot(w, hinge2(T)))])
        slices.append(np.column_stack([np.ones_like(T), T, hinge2(T)]))
    X = np.array(rows)
    A = np.vstack(slices)
    y = np.array([(RANGES[r][0] + RANGES[r][1]) / 2 for r in REGIONS])
    T_mean = X[:, 1]

    def city_table(c):
        out = {}
        for r in REGIONS:
            w, T = prof[r]
            p = air(T, c)
            ann = float(np.dot(w, p))
            lo, hi = RANGES[r]
            out[r] = {"city": CITY[r], "T_mean_jma_C": r6(np.dot(w, T)),
                      "pue_air_hourly_weighted": r6(ann), "survey_range": [lo, hi],
                      "midpoint": r6((lo + hi) / 2),
                      "inside_range": bool(lo - 1e-9 <= ann <= hi + 1e-9),
                      "pue_air_min_slice": r6(p.min()), "pue_air_peak_slice": r6(p.max()),
                      "T_min_slice_C": r6(T.min()), "T_peak_slice_C": r6(T.max())}
        return out

    def summary(c):
        cities = city_table(c)
        pred = np.array([cities[r]["pue_air_hourly_weighted"] for r in REGIONS])
        return {"coefficients": {k: r6(v) for k, v in c.items()},
                "fit_hourly_weighted_vs_midpoints": r2_stats(y, pred),
                "per_city": cities,
                "all_inside_survey_ranges": all(v["inside_range"] for v in cities.values()),
                "min_slice_pue_all_cities": r6(min(v["pue_air_min_slice"]
                                                   for v in cities.values())),
                "physical_check_pue_ge_1": bool(min(v["pue_air_min_slice"]
                                                    for v in cities.values()) >= PUE_FLOOR - 1e-9)}

    # (1) bounds-only fit (alpha, beta >= 0 only)
    fb = lsq_linear(X, y, bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]),
                    method="bvls", tol=1e-12)
    c_bounds = {"C_base": r6(fb.x[0]), "alpha": r6(fb.x[1]), "beta": r6(fb.x[2])}

    # (2) base fit: + PUE >= 1 in every slice
    x0 = np.array([1.2, 0.01, 0.001])
    res = minimize(lambda b: float(np.sum((X @ b - y) ** 2)), x0,
                   jac=lambda b: 2.0 * X.T @ (X @ b - y), method="SLSQP",
                   bounds=[(None, None), (0.0, None), (0.0, None)],
                   constraints=[LinearConstraint(A, PUE_FLOOR, np.inf)],
                   options={"ftol": 1e-15, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"constrained PUE fit failed: {res.message}")
    alpha, beta = r6(res.x[1]), r6(res.x[2])
    resid = y - alpha * X[:, 1] - beta * X[:, 2]
    c_ls = float(resid.mean())
    c_floor = float(np.max(PUE_FLOOR - alpha * A[:, 1] - beta * A[:, 2]))
    c0 = max(c_ls, c_floor)
    c0 = math.ceil(c0 * 1e6 - 1e-6) / 1e6 if c_floor >= c_ls else round(c0, 6)
    c_base = {"C_base": r6(c0), "alpha": alpha, "beta": beta}
    floor_binding = bool(c_floor >= c_ls - 1e-9)

    paper_cmp = {
        "at_table9_annual_means": {**r2_stats(y, air(np.array([TABLE9_T[r] for r in REGIONS]),
                                                     PAPER)),
                                   "pue": {r: r6(air(TABLE9_T[r], PAPER)) for r in REGIONS}},
        "at_jma_annual_means": {**r2_stats(y, air(T_mean, PAPER)),
                                "pue": {r: r6(v) for r, v in zip(REGIONS, air(T_mean, PAPER))}},
    }
    paper_hourly = summary(PAPER)
    paper_cmp["hourly_weighted_jma"] = {
        **paper_hourly["fit_hourly_weighted_vs_midpoints"],
        "pue": {r: paper_hourly["per_city"][r]["pue_air_hourly_weighted"] for r in REGIONS}}

    liquid = {}
    for i, r in enumerate(REGIONS):
        dlc = 1.0 + C_PUMP + GAMMA * T_mean[i]
        imm = 1.0 + (C_PUMP - IMM_OFFSET) + GAMMA * T_mean[i]
        liquid[r] = {"pue_dlc": r6(dlc), "pue_immersion": r6(imm),
                     "dlc_inside_1.05-1.10": bool(LIQUID_RANGE[0] <= dlc <= LIQUID_RANGE[1]),
                     "immersion_inside_1.05-1.10":
                         bool(LIQUID_RANGE[0] <= imm <= LIQUID_RANGE[1])}

    base = summary(c_base)
    out = {
        "description": "Air-cooling PUE (Energies Eq. 8) recalibrated so that the hourly-weighted "
                       "annual PUE (JMA 2020-2023 96-slice profile) matches the survey-range "
                       "midpoints; T_thresh fixed at 15 C; alpha, beta >= 0. Base = fit with "
                       "PUE >= 1 in every slice (the bounds-only fit gives PUE < 1 in cold slices).",
        "inputs": {"temperature_profile": "data/parameters_v2/temperature_96h_jma.csv",
                   "targets": "midpoints of Energies Table 5 / ch04 tab:pue-validation ranges",
                   "survey_ranges": {r: list(RANGES[r]) for r in REGIONS},
                   "T_thresh_C": T_THRESH, "pue_floor": PUE_FLOOR,
                   "regressors": {r: {"one": 1.0, "T_mean": r6(X[i, 1]),
                                      "hinge2_mean": r6(X[i, 2])}
                                  for i, r in enumerate(REGIONS)}},
        "coefficients": base["coefficients"],
        "fit": base["fit_hourly_weighted_vs_midpoints"],
        "per_city": base["per_city"],
        "all_inside_survey_ranges": base["all_inside_survey_ranges"],
        "base": {**base, "method": "scipy.optimize.minimize SLSQP: least squares on the 5 "
                                  "midpoints, alpha >= 0, beta >= 0, PUE_air >= 1 in all "
                                  "480 slices; rounded to 6 decimals (C_base raised to keep "
                                  "the floor if binding)",
                 "floor_binding": floor_binding,
                 "unrounded": [float(v) for v in res.x]},
        "bounds_only": {**summary(c_bounds),
                        "method": "scipy.optimize.lsq_linear (bvls), alpha >= 0, beta >= 0, "
                                  "C_base free (alpha, beta >= 0 only)",
                        "active_bounds": {"alpha": bool(fb.active_mask[1] != 0),
                                          "beta": bool(fb.active_mask[2] != 0)},
                        "unrounded": [float(v) for v in fb.x]},
        "paper_coefficients": {"coefficients": PAPER, "comparison": paper_cmp,
                               "per_city_hourly": paper_hourly["per_city"]},
        "liquid": {"note": "Eq. 9 is linear in T: the hourly weighted mean equals the value at "
                           "the mean temperature; coefficients kept (C_pump 0.055, gamma 0.0015, "
                           "immersion offset 0.025).",
                   "C_pump": C_PUMP, "gamma": GAMMA, "immersion_offset": IMM_OFFSET,
                   "per_city_at_jma_mean_T": liquid},
        "limitations": [
            "Targets are survey ranges (not plant measurements); the fit is a match to the "
            "calibration targets, not an independent validation.",
            "The 96-slice profile averages each season-hour over 2020-2023, so the peak-slice "
            "PUE is lower than the PUE of the hottest actual hours.",
            "The PUE >= 1 constraint is definitional; no sourced lower bound above 1 is used.",
        ],
    }
    (P2 / "pue_calibration.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)
                                              + "\n")
    for name, s in (("base", base), ("bounds_only", out["bounds_only"])):
        f = s["fit_hourly_weighted_vs_midpoints"]
        print(f"[{name}] {s['coefficients']} R2 {f['R2']} dof {f['dof']} "
              f"min slice PUE {s['min_slice_pue_all_cities']}")
        for r in REGIONS:
            c = s["per_city"][r]
            print(f"  {r:9s} T {c['T_mean_jma_C']:6.2f} PUE {c['pue_air_hourly_weighted']:.4f} "
                  f"{c['survey_range']} inside {c['inside_range']} min {c['pue_air_min_slice']:.3f}"
                  f" peak {c['pue_air_peak_slice']:.3f}")
    for name, v in paper_cmp.items():
        print(f"paper coefficients {name}: R2 {v['R2']:.4f}")


if __name__ == "__main__":
    main()
