"""Build data/parameters_v2/{parameters,regions,paths}.csv (DC-DECOM v2).

Every value is either copied from a source, derived from sources by the arithmetic below,
taken from the Energies paper (status 'paper'), or an assumption with a low/high range.
Derived inputs are read from files in the repository:
  data/parameters_v2/pue_calibration.json  (data/calibrate_pue_v2.py)
  data/parameters_v2/jma_summary.csv       (data/preprocess_jma.py)
  data/raw/egc/*.xlsx   電力・ガス取引監視等委員会「電力取引報」monthly results, FY2025
  data/raw/moe/r08_denki_coefficient_rev4.xlsx  環境省・経産省 電気事業者別排出係数 (R6年度実績)
No LP is solved here. Sources: data/parameters_v2/sources.csv and SOURCES.md.

Usage: .venv/bin/python data/build_parameters_v2.py
Writes parameters.csv, regions.csv, paths.csv, tariff_fy2025.csv and ef_fy2024.csv.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent
P2 = ROOT / "parameters_v2"
RAW = ROOT / "raw"
REGIONS = ("Hokkaido", "Tohoku", "Kanto", "Kansai", "Kyushu")
NONMETRO = ("Hokkaido", "Tohoku", "Kyushu")
AREA_JA = {"Hokkaido": "北海道", "Tohoku": "東北", "Kanto": "東京", "Kansai": "関西",
           "Kyushu": "九州"}
INCUMBENT = {"Hokkaido": "北海道電力(株)", "Tohoku": "東北電力(株)",
             "Kanto": "東京電力エナジーパートナー(株)", "Kansai": "関西電力(株)",
             "Kyushu": "九州電力(株)"}
YEARS = list(range(2024, 2041))
# FY2025 monthly 電力取引報 files (month -> file); see data/raw/egc/README.md
EGC_FILES = {
    "2025-04": "20250717001b.xlsx", "2025-05": "20250815001b.xlsx",
    "2025-06": "20250916001b.xlsx", "2025-07": "20251022001b.xlsx",
    "2025-08": "20251120001b.xlsx", "2025-09": "202512150001b.xlsx",
    "2025-10": "20260116001b.xlsx", "2025-11": "20260220001b.xlsx",
    "2025-12": "20260316001b.xlsx", "2026-01": "20260415002b.xlsx",
    "2026-02": "20260520001b.xlsx", "2026-03": "20260615001b.xlsx",
}
BASE_SHARE = {"Hokkaido": 0.03, "Tohoku": 0.05, "Kanto": 0.60, "Kansai": 0.22, "Kyushu": 0.10}

NAN = math.nan
Row = Dict[str, object]


def r(x: float, nd: int = 6) -> float:
    return float(round(float(x), nd))


def row(pid: str, group: str, value: float, unit: str, status: str, source_id: str,
        low: float = NAN, high: float = NAN, derivation: str = "", notes: str = "",
        **extra) -> Row:
    out: Row = {"id": pid, **extra, "group": group, "value": value, "unit": unit,
                "low": low, "high": high, "status": status, "source_id": source_id,
                "derivation": derivation, "notes": notes}
    return out


# ----------------------------------------------------------------------------- inputs
def pue_inputs() -> Dict[str, object]:
    cal = json.loads((P2 / "pue_calibration.json").read_text())
    coef = cal["coefficients"]
    per_city = {k: v["pue_air_hourly_weighted"] for k, v in cal["per_city"].items()}
    fleet = sum(BASE_SHARE[k] * per_city[k] for k in REGIONS) / sum(BASE_SHARE.values())
    return {"coef": coef, "per_city": per_city, "fleet_2024": fleet, "cal": cal}


def tariffs() -> pd.DataFrame:
    rows = []
    for month, fname in EGC_FILES.items():
        path = RAW / "egc" / fname
        if not path.exists():
            continue
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        vol = list(wb["販売電力量"].iter_rows(values_only=True))
        rev = list(wb["販売額"].iter_rows(values_only=True))
        stamp = vol[0][0]
        if stamp is None or stamp.strftime("%Y-%m") != month:
            raise ValueError(f"{fname}: month {stamp} != {month}")
        # 販売電力量: columns 15.. = '販売電力量 合計' (special-high voltage first)
        hdr = [i for i, c in enumerate(vol[2]) if isinstance(c, str) and "販売電力量" in c
               and "合計" in c]
        if not hdr:
            raise ValueError(f"{fname}: no total table in 販売電力量")
        c_vol = hdr[0]
        start = next(i for i, rw in enumerate(rev) if rw[0] and "販売額" in str(rw[0])
                     and "合計" in str(rw[0]))
        for reg in REGIONS:
            v = next(rw for rw in vol[5:] if rw[0] == AREA_JA[reg])[c_vol]
            m = next(rw for rw in rev[start:] if rw[0] == AREA_JA[reg])[1]
            rows.append({"region": reg, "month": month, "file": fname,
                         "sales_MWh": float(v), "revenue_thousand_yen": float(m),
                         "yen_per_kWh": float(m) / float(v)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("no 電力取引報 files in data/raw/egc")
    return df


def emission_factors() -> pd.DataFrame:
    wb = openpyxl.load_workbook(RAW / "moe" / "r08_denki_coefficient_rev4.xlsx",
                                read_only=True, data_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True))
    out = []
    for reg in REGIONS:
        i = next(k for k, rw in enumerate(rows) if rw[1] == INCUMBENT[reg])
        resid = whole = None
        for rw in rows[i:i + 20]:
            if rw[1] not in (None, INCUMBENT[reg]):
                break
            menu = str(rw[2] or "")
            if "残差" in menu:
                resid = float(rw[3])
            if "事業者全体" in menu:
                whole = float(rw[3])
        if resid is None or whole is None:
            raise ValueError(f"EF rows of {INCUMBENT[reg]} not found")
        out.append({"region": reg, "retailer": INCUMBENT[reg],
                    "basic_ef_residual_kg_per_kWh": resid * 1000.0,
                    "basic_ef_company_reference_kg_per_kWh": whole * 1000.0})
    nat = next(rw for rw in rows if rw[1] == "全国平均係数(t-CO₂/kWh)")[2]
    alt = next(rw for rw in rows if rw[1] == "代替値")[2]
    df = pd.DataFrame(out)
    df.attrs["national_average"] = float(nat) * 1000.0
    df.attrs["substitute"] = float(alt) * 1000.0
    return df


# ----------------------------------------------------------------------------- scalars
def scalars(pue: Dict[str, object], jma: pd.DataFrame) -> List[Row]:
    c = pue["coef"]
    cal = pue["cal"]
    bo = cal["bounds_only"]["coefficients"]
    fit = cal["fit"]
    cf_mean = float(jma["cf_annual_pr080"].mean())
    rows: List[Row] = [
        # PUE (Energies Eqs. 8-9)
        row("C_base", "existing", c["C_base"], "dimensionless", "derived", "PUECAL_V2",
            derivation="data/calibrate_pue_v2.py base fit: hourly-weighted annual PUE (JMA "
                       "2020-2023 96 slices) vs survey midpoints 1.25/1.35/1.50/1.50/1.50, "
                       f"alpha,beta>=0, PUE>=1 in every slice; R2 {fit['R2']}, dof {fit['dof']}",
            notes=f"bounds-only fit (alpha, beta >= 0 only) {bo} gives PUE < 1 in cold slices; "
                  "Energies Table 8 value 1.14"),
        row("alpha", "existing", c["alpha"], "1/C", "derived", "PUECAL_V2",
            derivation="as C_base", notes="Energies Table 8 value 0.0167"),
        row("beta", "existing", c["beta"], "1/C^2", "derived", "PUECAL_V2",
            derivation="as C_base", notes="Energies Table 8 value 0.0102"),
        row("T_thresh", "existing", 15.0, "C", "paper", "ENERGIES2026",
            derivation="Table 8; Eq. 8 (NTT Facilities); fixed in the recalibration"),
        row("C_pump", "existing", 0.055, "dimensionless", "paper", "ENERGIES2026",
            derivation="Table 8; Eq. 9 (Vertiv)"),
        row("gamma", "existing", 0.0015, "1/C", "paper", "ENERGIES2026",
            derivation="Table 8; Eq. 9 (Haghshenas)",
            notes="Eq. 9 is linear in T: hourly averaging equals evaluation at the mean"),
        row("pue_immersion_offset", "existing", 0.025, "dimensionless", "assumption",
            "ASSUMPTION", 0.0, 0.04,
            derivation="v1 code constant (pue.py, MHI); immersion PUE = 1 + C_pump - offset + "
                       "gamma T = 1.045-1.057 at the JMA means, inside Table 6 1.03-1.08"),
        row("pue_air_fleet_2024", "existing", r(pue["fleet_2024"]), "dimensionless", "derived",
            "PUECAL_V2",
            derivation="sum_r base_share_2024[r] x hourly-weighted air PUE[r] (base fit); "
                       "converts the facility-energy demand anchors to IT energy"),
        # IT load
        row("utilization", "existing", 0.80, "fraction", "assumption", "ASSUMPTION", 0.60, 0.90,
            derivation="average IT power / IT nameplate; Energies Table 8 note only "
                       "('inferred from 19 TWh load & 80%'), no primary source",
            notes="with it_energy='anchor_is_it' it sizes nameplate only"),
        row("ai_share_2024", "sensitivity", 0.1669, "fraction of DC electricity", "derived",
            "IEA_EAI2025", 0.10, 0.25,
            derivation="IEA Energy and AI base case: accelerated servers +30%/yr, ~half of the "
                       "2024-2030 increase 415->945 TWh: A x (1.3^6 - 1) = 0.5 x 530 -> A = "
                       "69.2 TWh = 16.7% of 415 TWh (global); applied to Japan (assumption)",
            notes="range is an assumption"),
        row("s_train", "sensitivity", 0.40, "fraction of AI load", "sourced", "PATTERSON2022",
            0.20, 0.60, derivation="Google 2019-2021: about 2/5 of ML energy for training, 3/5 "
                                   "for inference; range assumption",
            notes="training treated as latency-tolerant"),
        row("s_inf_tol", "sensitivity", 0.30, "fraction of inference load", "assumption",
            "ASSUMPTION", 0.10, 0.50,
            derivation="latency-tolerant (batch / asynchronous) share of AI inference; no "
                       "public measurement found"),
        row("metro_floor_A", "policy", 0.82, "fraction", "derived", "ENERGIES2026",
            derivation="design A (no latency classes): Kanto + Kansai base shares 0.60 + 0.22 "
                       "(Table 9)"),
        row("kanto_floor_A", "policy", 0.60, "fraction", "derived", "ENERGIES2026",
            derivation="design A: Kanto base share 0.60 (Table 9)"),
        row("eta_HW_rate", "sensitivity", 0.30, "fraction per year", "paper", "ENERGIES2026",
            0.20, 0.40, derivation="Table 7/8 (JST Moore); Section 4.6 uses 40% as the fast "
                                   "case and 20% as the slow case",
            notes="not an LP input; the v2 demand path already embeds the source's efficiency "
                  "assumptions"),
        # cooling CAPEX (thousand yen per kW IT)
        row("cooling_air_capex", "existing", 150.0, "thousand yen/kW", "assumption",
            "ASSUMPTION", 100.0, 200.0,
            derivation="v1 code value; Table 6 gives 'Baseline' without a number"),
        row("cooling_dlc_capex", "existing", 187.5, "thousand yen/kW", "derived",
            "ENERGIES2026", 180.0, 195.0,
            derivation="air x 1.25 (midpoint of Table 6 +20-30%); range air x 1.20-1.30",
            notes="v1/spec use +30% (195)"),
        row("cooling_immersion_capex", "existing", 225.0, "thousand yen/kW", "derived",
            "ENERGIES2026", 210.0, 240.0,
            derivation="air x 1.50 (midpoint of Table 6 +40-60%); range air x 1.40-1.60"),
        # BESS, PV, CGS, generators
        row("bess_capex_2025", "existing", 60.0, "thousand yen/kWh", "paper", "ENERGIES2026",
            derivation="Table 6 (NEDO [67])"),
        row("bess_capex_2035", "existing", 30.0, "thousand yen/kWh", "paper", "ENERGIES2026",
            derivation="Table 6 (NEDO [67]); linear 2025->2035, constant after"),
        row("bess_efficiency", "existing", 0.90, "fraction", "paper", "ENERGIES2026",
            derivation="Table 6 round trip"),
        row("bess_duration_h", "structural", 4.0, "h", "assumption", "ASSUMPTION", 2.0, 6.0,
            derivation="energy / power of the BESS; no source in Energies"),
        row("pv_lcoe", "existing", 10.0, "yen/kWh", "sourced", "COSTWG2025", 6.6, 10.9,
            derivation="事業用太陽光 LCOE 2023 (政策経費なし) 10.0; low = 2040 lower bound "
                       "6.6 (政策経費なし); high = 2023 政策経費あり 10.9",
            notes="constant over 2024-2040 (the model has one PV cost); Energies used 9.9"),
        row("pv_performance_ratio", "structural", 0.80, "fraction", "assumption", "ASSUMPTION",
            0.75, 0.85, derivation="CF = GHI (horizontal) x PR (data/preprocess_jma.py)"),
        row("pv_cf_avg", "existing", r(cf_mean), "fraction", "derived", "JMA_OBSDL",
            r(float(jma["cf_annual_pr075"].mean())), r(float(jma["cf_annual_pr085"].mean())),
            derivation="mean of the 5 regional annual CF (JMA GHI x PR 0.80); used only if "
                       "no pv_cf_96h profile",
            notes="Energies Table 6: 13-17%"),
        row("Deploy_max_PV", "existing", 1.0, "GW/yr/region", "paper", "ENERGIES2026",
            derivation="Table 8; Eq. 17 (FIT historical)"),
        row("cgs_capex", "existing", 230.0, "thousand yen/kW", "paper", "ENERGIES2026",
            derivation="Table 6 (METI cogeneration WG)"),
        row("cgs_eff_elec", "existing", 0.423, "fraction", "paper", "ENERGIES2026",
            derivation="Table 6 (METI cogeneration WG)"),
        row("cgs_fuel_yen_per_kWh_th", "existing", 8.0, "yen/kWh_th", "assumption",
            "ASSUMPTION", 5.0, 11.0,
            derivation="v1 code value; about 100 yen/m3 city gas / 12.5 kWh_th/m3 (45 MJ/m3); "
                       "no sourced industrial gas price used"),
        row("EF_fossil", "existing", 0.49, "kg-CO2/kWh", "paper", "ENERGIES2026",
            derivation="Table 8 (METI cogeneration)"),
        row("gen_capex", "structural", 80.0, "thousand yen/kW", "assumption", "ASSUMPTION",
            40.0, 150.0, derivation="emergency diesel generator incl. fuel storage; no "
                                    "Japanese public cost found"),
        row("opex_om_rate", "existing", 0.02, "fraction of CAPEX per year", "assumption",
            "ASSUMPTION", 0.01, 0.04, derivation="v1 code value (Eq. 3 o_tau without a number)"),
        row("curtail_penalty", "existing", 500.0, "yen/kWh", "paper", "ENERGIES2026",
            derivation="Section 3.10 (VOLL proxy)"),
        row("discount_rate", "existing", 0.03, "fraction", "paper", "ENERGIES2026", 0.02, 0.04,
            derivation="Section 3.10 (Imagawa et al.; IEA-ETSAP); range assumption (upper "
                       "bound = the 4% social discount rate of Japanese public-works "
                       "appraisal)"),
        row("N_resilience", "existing", 1.0, "unitless", "paper", "ENERGIES2026", 1.0, 4.0,
            derivation="Table 8 / ch04 eq:resilience literal (backup >= 2 x critical load "
                       "with N = 1); sensitivity N = 2, 4 and the v1 5% rule"),
        # certificates
        row("nfc_yen_per_kWh", "structural", 0.4, "yen/kWh", "sourced", "ENEHUB_JEPX2025", 0.4,
            1.3, derivation="JEPX FY2025 2nd auction: FIT certificates 0.4-0.7, volume-"
                            "weighted 0.4 (floor); non-FIT certificates cleared at the cap "
                            "1.3 (high)"),
        # PPA supply curve
        row("ppa_tier1_share", "structural", 0.05, "fraction of RE potential energy",
            "assumption", "ASSUMPTION", 0.02, 0.10,
            derivation="tier-1 volume = share x re_max_GW x CF_avg x 8.76 x ppa_volume_factor(t)"
                       " at the regional PPA price; no public DC-accessible PPA volume found"),
        row("ppa_tier2_share", "structural", 0.10, "fraction of RE potential energy",
            "assumption", "ASSUMPTION", 0.05, 0.30,
            derivation="tier-2 volume at price + premium; tiers 1+2 (15% of the 5-region "
                       "REPOS solar potential energy, ~126 TWh at JMA CF) ~ 40% of the 7th SEP "
                       "2040 national PV generation (23-29% of 1.1-1.2 trillion kWh)"),
        row("ppa_tier2_premium_yen_per_kWh", "structural", 3.0, "yen/kWh", "assumption",
            "ASSUMPTION", 1.0, 6.0,
            derivation="premium of the second tier over the regional PPA price"),
        # policy (A/B/C)
        row("pue_fleet_max", "policy", 1.4, "dimensionless", "sourced", "ANRE_DC2026",
            derivation="省エネ法ベンチマーク: data-centre operators' PUE <= 1.4 by FY2030",
            notes="year in pue_fleet_max_from"),
        row("pue_fleet_max_from", "policy", 2030.0, "year", "sourced", "ANRE_DC2026"),
        row("pue_new_max", "policy", 1.3, "dimensionless", "sourced", "ANRE_DC2026",
            derivation="new DCs from FY2029: PUE <= 1.3 (from the fiscal year after two years "
                       "of operation)"),
        row("pue_new_max_from", "policy", 2029.0, "year", "sourced", "ANRE_DC2026"),
        row("siting_subsidy_yen_per_kW", "policy", 600000.0, "yen/kW", "derived",
            "METI_DC_CHIHO_R5", 100000.0, 780000.0,
            derivation="METI FY2023 DC地方拠点整備 subsidy, Tomakomai: up to 30 billion yen "
                       "(1/2 of eligible cost) for a 50 MW (受電容量) first phase -> 30e9 / "
                       "50,000 kW = 600,000 yen/kW; high = IT-nameplate basis at design PUE "
                       "1.3 (780,000); low = budget-limited average (assumption)",
            notes="one project with a cap; using it as a uniform per-kW rate is a design "
                  "assumption (s0 of ch07 design A, R-CN subsidy)"),
        row("reinforcement_cost_yen_per_kW", "policy", 118000.0, "yen/kW", "derived",
            "OCCTO_MP2023", 50000.0, 500000.0,
            derivation="Masterplan base scenario: Hokkaido internal 1.0 + Tohoku internal 0.65 "
                       "trillion yen for HVDC Hokkaido-Tohoku 6 GW + Tohoku-Tokyo 8 GW -> "
                       "1.65e12 / 14e6 kW = 117,857 (rounded 118,000); high = East HVDC route "
                       "~3 trillion yen / 6 GW = 500,000; low = assumption (local connection only)"),
        row("operator_burden_share", "policy", 0.5, "fraction", "assumption", "ASSUMPTION",
            0.0, 1.0, derivation="theta: operator share of the reinforcement cost; demand-side "
                                 "general/specific burden split has no public number"),
        row("reinforcement_year", "policy", 2030.0, "year", "assumption", "OCCTO_MP2023",
            2028.0, 2035.0, derivation="T_g: first year of the reinforced capacity c1; the "
                                       "Masterplan HVDC (2 GW plan under study since 2022) "
                                       "timing is not fixed"),
        row("ppa_floor_subsidized", "policy", 0.3, "fraction", "assumption", "ASSUMPTION",
            0.1, 0.5, derivation="psi: PPA share required for subsidized builds (design B/C)"),
        row("ppa_tier_multiplier_induced", "policy", 1.5, "factor", "assumption", "ASSUMPTION",
            1.2, 2.0, derivation="PPA tier-volume uplift in non-metro induced zones (design B/C)"),
        row("carbon_pass_through", "sensitivity", 1.0, "fraction", "assumption", "ASSUMPTION",
            0.0, 1.0, derivation="share of the carbon price in the operator's cost; GX-ETS "
                                 "covers direct emissions, the electricity price carries it"),
    ]
    return rows


# ----------------------------------------------------------------------------- regions
def regional(pue: Dict[str, object], jma: pd.DataFrame, tar: pd.DataFrame,
             ef: pd.DataFrame) -> List[Row]:
    spec = pd.read_csv(ROOT / "parameters_spec" / "regions.csv").set_index("region")
    j = jma.set_index("region")
    rows: List[Row] = []
    months = sorted(tar["month"].unique())
    for reg in REGIONS:
        rows.append(row("annual_mean_T_C", "existing", r(j.loc[reg, "annual_mean_T_C"], 4), "C",
                        "derived", "JMA_OBSDL", region=reg,
                        derivation="96-slice weighted mean, JMA 2020-2023 hourly (station "
                                   f"{int(j.loc[reg, 'station'])} {j.loc[reg, 'city']})",
                        notes=f"Table 9: {spec.loc[reg, 'annual_mean_T_C']}"))
    for reg in REGIONS:
        rows.append(row("base_share_2024", "existing", float(spec.loc[reg, "base_share_2024"]),
                        "fraction", "paper", "ENERGIES2026", region=reg,
                        derivation="Table 9 (MIC)"))
    for reg in REGIONS:
        t = tar[tar["region"] == reg]
        val = t["revenue_thousand_yen"].sum() / t["sales_MWh"].sum()
        rows.append(row("tariff_yen_per_kWh", "existing", r(val, 3), "yen/kWh", "sourced",
                        "EGC_TORIHIKI", r(t["yen_per_kWh"].min(), 3),
                        r(t["yen_per_kWh"].max(), 3), region=reg,
                        derivation=f"特別高圧 sales revenue / sales volume, all retailers, "
                                   f"{AREA_JA[reg]} area, {len(t)} months {months[0]}..{months[-1]}"
                                   "; low/high = monthly min/max",
                        notes=f"Table 9: {spec.loc[reg, 'tariff_yen_per_kWh']}"))
    for reg in REGIONS:
        v = float(spec.loc[reg, "ppa_yen_per_kWh"])
        rows.append(row("ppa_yen_per_kWh", "existing", v, "yen/kWh", "paper", "ENERGIES2026",
                        v - 2.0, v + 2.0, region=reg,
                        derivation="Table 9 (REI corporate PPA 2024); range +-2 yen/kWh "
                                   "(assumption)"))
    for reg in REGIONS:
        rows.append(row("re_max_GW", "existing", float(spec.loc[reg, "re_max_GW"]), "GW",
                        "paper", "ENERGIES2026", region=reg,
                        derivation="Table 9 / Section 3.2 (REPOS solar potential)"))
    e = ef.set_index("region")
    for reg in REGIONS:
        res = float(e.loc[reg, "basic_ef_residual_kg_per_kWh"])
        whole = float(e.loc[reg, "basic_ef_company_reference_kg_per_kWh"])
        rows.append(row("grid_ef_base", "existing", r(res, 4), "kg-CO2/kWh", "sourced",
                        "MOE_EF_R6", r(min(res, whole), 4), r(max(res, whole), 4), region=reg,
                        derivation=f"{INCUMBENT[reg]} 基礎排出係数 (残差 menu), R6年度実績; low "
                                   "= company-wide reference value",
                        notes=f"Table 9: {spec.loc[reg, 'grid_ef_base']}; scaled by "
                              "ef_national(t)/ef_national(2024) in the model"))
    for reg in NONMETRO:
        rows.append(row("conn_capacity_GW_per_yr", "policy", 0.2, "GW/yr", "assumption",
                        "ASSUMPTION", 0.05, 0.6, region=reg,
                        derivation="c0: facility peak that can newly connect per year without "
                                   "reinforcement; no regional public value. Context: OCCTO "
                                   "FY2025 forecast DC new-build peak +0.63 GW/yr nationally "
                                   "(2025-2034)"))
    for reg in NONMETRO:
        rows.append(row("conn_capacity_reinforced_GW_per_yr", "policy", 0.6, "GW/yr",
                        "assumption", "ASSUMPTION", 0.3, 1.2, region=reg,
                        derivation="c1 after reinforcement (from reinforcement_year)"))
    for reg in ("Kanto", "Kansai"):
        rows.append(row("shell_premium_yen_per_kW", "structural", 0.0, "yen/kW", "assumption",
                        "ASSUMPTION", 0.0, 100000.0, region=reg,
                        derivation="metro construction/land premium per kW of new IT "
                                   "nameplate; no sourced regional cost index for the "
                                   "non-metro regions (non-metro = 0 reference)"))
    return rows


# ----------------------------------------------------------------------------- paths
def lin(t0: float, v0: float, t1: float, v1: float, t: float) -> float:
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def paths(pue: Dict[str, object]) -> List[Row]:
    out: List[Row] = []
    fleet = float(pue["fleet_2024"])

    # demand (facility, TWh): Wood Mackenzie 19 (2024) -> 57-66 (2034)
    def fac(t: int, variant: str) -> float:
        if variant == "low":  # JST Vol.3 Modest 24 TWh (2030), linear continuation
            return lin(2024, 19.0, 2030, 24.0, t)
        end = {"value": 61.5, "high": 66.0}[variant]
        return lin(2024, 19.0, 2034, end, t)

    for t in YEARS:
        v, lo, hi = fac(t, "value"), fac(t, "low"), fac(t, "high")
        if t == 2024:
            st, src, der = "sourced", "WOODMAC2025", "Wood Mackenzie 19 TWh (2024)"
        elif t <= 2034:
            st, src = "derived", "WOODMAC2025"
            der = "linear 19 (2024) -> 61.5 (2034, midpoint of 57-66); high -> 66; low: JST " \
                  "Modest 24 TWh in 2030, linear"
        else:
            st, src = "assumption", "ASSUMPTION"
            der = "linear continuation of the 2024-2034 slopes (base 4.25, high 4.7, low " \
                  "0.83 TWh/yr)"
        out.append(row("demand_facility_TWh", "path", r(v, 4), "TWh (facility)", st, src,
                       r(lo, 4), r(hi, 4), year=t, derivation=der,
                       notes="Wood Mackenzie measures data-centre electricity consumption "
                             "(facility); cross-check OCCTO FY2025 forecast: existing ~19 + DC "
                             "new-build 44.0 TWh (FY2034)"))
    for t in YEARS:
        out.append(row("demand_TWh", "path", r(fac(t, "value") / fleet, 4), "TWh (IT)",
                       "derived", "PUECAL_V2", r(fac(t, "low") / fleet, 4),
                       r(fac(t, "high") / fleet, 4), year=t,
                       derivation=f"demand_facility_TWh / pue_air_fleet_2024 ({r(fleet)}); "
                                  "IT energy at the 2024 all-air fleet PUE (it_energy "
                                  "'anchor_is_it')"))

    # carbon price (yen/t-CO2)
    band = {"low": (1700.0, 7000.0), "value": (3000.0, 11000.0), "high": (4300.0, 15000.0)}

    def carbon(t: int, variant: str) -> float:
        p26, p35 = band[variant]
        if t < 2026:
            return 0.0
        p30 = p26 * 1.03 ** 4
        if t <= 2030:
            return p26 * 1.03 ** (t - 2026)
        if t <= 2035:
            return lin(2030, p30, 2035, p35, t)
        return p35 + (p35 - p30) / 5.0 * (t - 2035)

    for t in YEARS:
        if t < 2026:
            st, src, der = "sourced", "METI_ETS", "no mandatory carbon price before FY2026 " \
                                                  "(GX-ETS starts FY2026)"
        elif t == 2026:
            st, src, der = "derived", "METI_KOKUJI27_2026", "midpoint of the FY2026 band " \
                                                            "1,700-4,300 (low/high = band)"
        elif t <= 2030:
            st, src, der = "derived", "METI_KOKUJI27_2026", "FY2026 value x 1.03^(t-2026): " \
                "告示 rule for FY2027-2030 (x (1.03 + CGPI outlook)); real terms, CGPI omitted"
        elif t < 2035:
            st, src, der = "derived", "MRI2025", "linear 2030 -> 2035"
        elif t == 2035:
            st, src, der = "paper", "ENERGIES2026", "Table 8: MRI [50] 7,000-15,000 (midpoint " \
                "11,000); the MRI column text does not state the range"
        else:
            st, src, der = "assumption", "ASSUMPTION", "linear continuation of the 2030-2035 " \
                                                       "slope"
        out.append(row("carbon_price", "path", r(carbon(t, "value"), 2), "yen/t-CO2", st, src,
                       r(carbon(t, "low"), 2), r(carbon(t, "high"), 2), year=t, derivation=der))

    # national grid EF (kg/kWh): 0.423 (FY2024) -> 2040 scenarios of the 7th SEP
    e40 = {"value": 0.035, "low": 0.0, "high": 0.13}
    for t in YEARS:
        v = {k: lin(2024, 0.423, 2040, e40[k], t) for k in e40}
        if t == 2024:
            st, src, der = "sourced", "MOE_EF_R6", "全国平均係数 0.000423 t/kWh (R6年度実績)"
        elif t == 2040:
            st, src, der = "derived", "ANRE_SEP7_KANREN", "median of the 4 scenarios on the " \
                "linear 2050 net-zero path (0.04/0.03/0.00/0.04) = 0.035; low 0.00 (CCS), " \
                "high 0.13 (技術進展, -56%)"
        else:
            st, src, der = "assumption", "ASSUMPTION", "linear interpolation 2024 -> 2040"
        out.append(row("ef_national", "path", r(v["value"]), "kg-CO2/kWh", st, src,
                       r(v["low"]), r(v["high"]), year=t, derivation=der))

    # liquid deployment cap (GW/yr/region)
    for t in YEARS:
        out.append(row("liquid_deploy_cap", "path", 0.5, "GW/yr/region", "paper",
                       "ENERGIES2026", 0.25, 0.5, year=t,
                       derivation="Table 8 / Eq. 17 (SoftBank Tomakomai); low = slow variant "
                                  "0.25 (assumption)"))

    # PPA volume factor: 7th SEP PV share 9.2% (FY2022) -> 23-29% (2040), normalized to 26%
    for t in YEARS:
        f = {k: (9.2 + (s - 9.2) * (t - 2022) / 18.0) / 26.0
             for k, s in (("value", 26.0), ("low", 23.0), ("high", 29.0))}
        out.append(row("ppa_volume_factor", "path", r(f["value"]), "factor", "derived",
                       "ANRE_SEP7_KANREN", r(f["low"]), r(f["high"]), year=t,
                       derivation="PV share of generation 9.2% (FY2022) -> 26% (2040, midpoint "
                                  "of 23-29%), linear, divided by 26%"))

    # AI share and the latency floors
    def ai(t: int) -> float:
        a24, a30 = 0.1669, 0.1669 * 1.3 ** 6 / (945.0 / 415.0)
        return min(1.0, lin(2024, a24, 2030, a30, t))

    s_tr, s_inf = 0.40, 0.30
    variants = {"value": (s_tr, s_inf), "low": (0.60, 0.50), "high": (0.20, 0.10)}
    for t in YEARS:
        a = ai(t)
        st = "derived" if t <= 2030 else "assumption"
        src = "IEA_EAI2025" if t <= 2030 else "ASSUMPTION"
        out.append(row("ai_share", "path", r(a), "fraction of DC electricity", st, src,
                       r(0.6 * a), r(min(0.9, 1.5 * a)), year=t,
                       derivation="2024 16.7% -> 2030 35.3% (IEA base: accelerated x1.3^6, "
                                  "total 415 -> 945 TWh), linear; after 2030 linear "
                                  "continuation (assumption); low/high x0.6 / x1.5"))
    for t in YEARS:
        def lam(v: str) -> float:
            if t <= 2025:
                return 0.82
            st_, si_ = variants[v]
            return 0.82 * (1.0 - ai(t) * (1.0 - (1.0 - st_) * (1.0 - si_)))
        for pid, scale in (("metro_floor", 1.0), ("kanto_floor", 60.0 / 82.0)):
            out.append(row(pid, "path", r(lam("value") * scale), "fraction", "derived",
                           "PATTERSON2022", r(lam("low") * scale), r(lam("high") * scale),
                           year=t,
                           derivation="lambda_B(t) = 0.82 x [nonAI(t) + AI(t)(1 - s_train)"
                                      "(1 - s_inf_tol)], s_train 0.4, s_inf_tol 0.3; low "
                                      "(0.6, 0.5), high (0.2, 0.1); kappa = lambda x 60/82; "
                                      "2024-25 = base shares",
                           notes="lambda_A = 0.82 (metro_floor_A) is the same rule with "
                                 "latency-sensitive share 1"))
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    pue = pue_inputs()
    jma = pd.read_csv(P2 / "jma_summary.csv")
    tar = tariffs()
    ef = emission_factors()
    tar.round(6).to_csv(P2 / "tariff_fy2025.csv", index=False)
    efo = ef.copy()
    efo["national_average_kg_per_kWh"] = ef.attrs["national_average"]
    efo["substitute_kg_per_kWh"] = ef.attrs["substitute"]
    efo.round(6).to_csv(P2 / "ef_fy2024.csv", index=False)

    sc = pd.DataFrame(scalars(pue, jma))
    sc.to_csv(P2 / "parameters.csv", index=False)
    rg = pd.DataFrame(regional(pue, jma, tar, ef))
    rg = rg[["id", "region", "group", "value", "unit", "low", "high", "status", "source_id",
             "derivation", "notes"]]
    rg.to_csv(P2 / "regions.csv", index=False)
    pt = pd.DataFrame(paths(pue))
    pt = pt[["id", "year", "group", "value", "low", "high", "unit", "status", "source_id",
             "derivation", "notes"]]
    pt.to_csv(P2 / "paths.csv", index=False)
    months = sorted(tar["month"].unique())
    print(f"parameters {len(sc)}, regional {len(rg)}, path rows {len(pt)}; "
          f"tariff months {len(months)} ({months[0]}..{months[-1]}); "
          f"PUE fleet 2024 {pue['fleet_2024']:.6f}")
    print(rg[rg["id"].isin(["tariff_yen_per_kWh", "grid_ef_base", "annual_mean_T_C"])]
          [["id", "region", "value", "low", "high"]].to_string(index=False))


if __name__ == "__main__":
    main()
