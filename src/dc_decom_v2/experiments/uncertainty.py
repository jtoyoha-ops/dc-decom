"""Composite uncertainty: Latin hypercube Monte Carlo of the endogenous
R-CN advantage over C-CN (part mc).

Definitions: config/experiments_v35.yaml section ``uncertainty`` (defined
before any run). Blocks: M (factors 1-7, base structure,
headline), R (the same u matrix with the reinforcement structure, so the cost
g acts) and X (factors 1-28, adding the unsourced assumptions).
Every sample solves C-CN and R-CN; a sample that is not Optimal is kept in
mc_samples.csv, listed in mc_summary.json and excluded from the statistics.

Columns of mc_samples.csv: ``u_<factor>`` is the LHS coordinate, ``value_<factor>``
the mapped value; for the path factors (``*_q``) the value is a representative
point (see VALUE_NOTE).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from scipy import stats

from dc_decom_analysis import uncertainty as U

from ..params import ParamSet
from . import common as C
from .bridge import REGS, metrics, solve_settings
from .common import ETA_BASE, Setting, as_tuple

CONFIG_V35 = C.REPO / "config" / "experiments_v35.yaml"
FILES = ("mc_samples.csv", "mc_summary.json", "mc_prcc.csv", "mc_bins.csv", "run_log_mc.csv")
PART = "mc"
ROLES = ("C-CN", "R-CN")
METRO_REGS = ("Kanto", "Kansai")
NONMETRO = ("Hokkaido", "Tohoku", "Kyushu")
ORDER = ("M", "X", "R")
CHUNK = {"M": 100, "X": 100, "R": 50}
BOOT_B, BOOT_SEED = 1000, 20261109
OUTPUTS = ("advantage", "total_R-CN_TJPY")
FLOOR_FIRST_YEAR = 2026

# Factors 1-28 in the order of the config. Uniform scalars: (low, high);
# path factors (*_q): q ~ U(0, 1).
FACTORS: Tuple[str, ...] = (
    "carbon_q", "demand_q", "eta", "ppa_offset", "nfc", "g", "siting_diff", "tariff_q",
    "grid_ef_q", "ef_national_q", "pv_lcoe", "cooling_air_capex", "dlc_mult", "imm_mult",
    "bess_duration_h", "cgs_fuel_yen_per_kWh_th", "gen_capex", "opex_om_rate", "utilization",
    "discount_rate", "ppa_tier1_share", "ppa_tier2_share", "ppa_tier2_premium_yen_per_kWh",
    "conn_capacity_GW_per_yr", "pue_immersion_offset", "s_train", "s_inf_tol", "ai_share_q")
RANGES: Dict[str, Tuple[float, float]] = {
    "eta": (0.20, 0.40), "ppa_offset": (-2.0, 2.0), "nfc": (0.4, 1.3),
    "g": (50000.0, 500000.0), "siting_diff": (-100000.0, 100000.0),
    "pv_lcoe": (6.6, 10.9), "cooling_air_capex": (100.0, 200.0), "dlc_mult": (1.20, 1.30),
    "imm_mult": (1.40, 1.60), "bess_duration_h": (2.0, 6.0),
    "cgs_fuel_yen_per_kWh_th": (5.0, 11.0), "gen_capex": (40.0, 150.0),
    "opex_om_rate": (0.01, 0.04), "utilization": (0.6, 0.9), "discount_rate": (0.02, 0.04),
    "ppa_tier1_share": (0.02, 0.10), "ppa_tier2_share": (0.05, 0.30),
    "ppa_tier2_premium_yen_per_kWh": (1.0, 6.0), "conn_capacity_GW_per_yr": (0.05, 0.6),
    "pue_immersion_offset": (0.0, 0.04), "s_train": (0.2, 0.6), "s_inf_tol": (0.1, 0.5)}
# Scalar factors whose range is the low / high of data/parameters_v2 (checked at run time).
FROZEN_RANGE = {"nfc": "nfc_yen_per_kWh", "g": "reinforcement_cost_yen_per_kW",
                "pv_lcoe": "pv_lcoe", "cooling_air_capex": "cooling_air_capex",
                "bess_duration_h": "bess_duration_h",
                "cgs_fuel_yen_per_kWh_th": "cgs_fuel_yen_per_kWh_th", "gen_capex": "gen_capex",
                "opex_om_rate": "opex_om_rate", "utilization": "utilization",
                "discount_rate": "discount_rate", "ppa_tier1_share": "ppa_tier1_share",
                "ppa_tier2_share": "ppa_tier2_share",
                "ppa_tier2_premium_yen_per_kWh": "ppa_tier2_premium_yen_per_kWh",
                "pue_immersion_offset": "pue_immersion_offset", "s_train": "s_train",
                "s_inf_tol": "s_inf_tol"}
VALUE_NOTE = {
    "carbon_q": "carbon_price 2040 (yen/t)", "demand_q": "demand_TWh 2040 before the HW transform",
    "tariff_q": "tariff_yen_per_kWh[Kanto]", "grid_ef_q": "grid_ef_base[Kanto]",
    "ef_national_q": "ef_national 2040", "ai_share_q": "ai_share 2040",
    "dlc_mult": "multiplier (cooling_dlc_capex = air x value)",
    "imm_mult": "multiplier (cooling_immersion_capex = air x value)",
    "siting_diff": "d (> 0 metro shell premium, < 0 non-metro addition |d|)"}


def load_mc_config() -> Dict[str, Any]:
    cfg = yaml.safe_load(CONFIG_V35.read_text(encoding="utf-8"))
    mc = cfg["uncertainty"]
    names = tuple(f["name"] for f in sorted(mc["factors"], key=lambda f: f["id"]))
    if names != FACTORS:
        raise ValueError(f"factor list differs from config/experiments_v35.yaml: {names}")
    if tuple(mc["blocks"]["M"]["factors"]) != FACTORS[:7]:
        raise ValueError("block M factors differ from the config")
    return {"target_range": tuple(cfg["target_range"]), "blocks": mc["blocks"]}


def check_ranges(ps: ParamSet) -> None:
    for name, pid in FROZEN_RANGE.items():
        p = ps.get(pid)
        if not (math.isclose(p.low, RANGES[name][0]) and math.isclose(p.high, RANGES[name][1])):
            raise ValueError(f"range of {name} differs from {pid} low/high in data/parameters_v2")
    for r in NONMETRO:
        p = ps.get("conn_capacity_GW_per_yr", r)
        if (p.low, p.high) != RANGES["conn_capacity_GW_per_yr"]:
            raise ValueError("range of c0 differs from data/parameters_v2")


def u_matrix(block: str, blocks: Dict[str, Any]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """M and R share LHS(n, 7, 20261107); X uses LHS(n, 28, 20261108)."""
    b = blocks[block]
    names = FACTORS if block == "X" else FACTORS[:7]
    return U.lhs(int(b["n"]), len(names), int(b["seed"])), names


def _paths3(ps: ParamSet, pid: str) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    return ps.path(pid, "low"), ps.path(pid, "value"), ps.path(pid, "high")


def build_sample(names: Tuple[str, ...], u: np.ndarray, ps: ParamSet,
                 reinforce: bool) -> Tuple[Setting, Dict[str, float], bool]:
    """Setting of one sample, the mapped values and the running-max flag of demand."""
    f = dict(zip(names, (float(x) for x in u)))
    val: Dict[str, float] = {}
    ov: Dict[str, float] = {}
    adds: Dict[str, float] = {}

    def uni(name: str) -> float:
        lo, hi = RANGES[name]
        v = U.uniform(f[name], lo, hi)
        val[name] = v
        return v

    carbon = U.path_quantile(f["carbon_q"], *_paths3(ps, "carbon_price"))
    val["carbon_q"] = carbon[C.YEARS[-1]]
    D = U.path_quantile(f["demand_q"], *_paths3(ps, "demand_TWh"))
    val["demand_q"] = D[C.YEARS[-1]]
    eta = uni("eta")
    off = uni("ppa_offset")
    for r in REGS:
        ov[f"ppa_yen_per_kWh[{r}]"] = ps.region("ppa_yen_per_kWh", r) + off
    ov["nfc_yen_per_kWh"] = uni("nfc")
    g = uni("g")
    ov["reinforcement_cost_yen_per_kW"] = g
    d = uni("siting_diff")
    if d > 0:
        for r in METRO_REGS:
            ov[f"shell_premium_yen_per_kW[{r}]"] = d
    elif d < 0:
        for r in NONMETRO:
            adds[f"shell_premium_yen_per_kW[{r}]"] = -d
    ps_d = ps
    if "tariff_q" in f:
        q = f["tariff_q"]
        tl = {r: ps.get("tariff_yen_per_kWh", r).low for r in REGS}
        tv = {r: ps.get("tariff_yen_per_kWh", r).value for r in REGS}
        th = {r: ps.get("tariff_yen_per_kWh", r).high for r in REGS}
        tar = U.path_quantile(q, tl, tv, th)
        ov.update({f"tariff_yen_per_kWh[{r}]": v for r, v in tar.items()})
        val["tariff_q"] = tar["Kanto"]
        q = f["grid_ef_q"]
        for r in REGS:
            p = ps.get("grid_ef_base", r)
            ov[f"grid_ef_base[{r}]"] = p.low + q * (p.value - p.low)
        val["grid_ef_q"] = ov["grid_ef_base[Kanto]"]
        efn = U.path_quantile(f["ef_national_q"], *_paths3(ps, "ef_national"))
        ov.update({f"ef_national@{t}": v for t, v in efn.items()})
        val["ef_national_q"] = efn[C.YEARS[-1]]
        ov["pv_lcoe"] = uni("pv_lcoe")
        air = uni("cooling_air_capex")
        ov["cooling_air_capex"] = air
        ov["cooling_dlc_capex"] = air * uni("dlc_mult")
        ov["cooling_immersion_capex"] = air * uni("imm_mult")
        for name in ("bess_duration_h", "cgs_fuel_yen_per_kWh_th", "gen_capex", "opex_om_rate",
                     "utilization", "discount_rate", "ppa_tier1_share", "ppa_tier2_share",
                     "ppa_tier2_premium_yen_per_kWh", "pue_immersion_offset"):
            ov[name] = uni(name)
        c0 = uni("conn_capacity_GW_per_yr")
        for r in NONMETRO:
            ov[f"conn_capacity_GW_per_yr[{r}]"] = c0
        s_tr, s_inf = uni("s_train"), uni("s_inf_tol")
        ai = U.path_quantile(f["ai_share_q"], *_paths3(ps, "ai_share"))
        val["ai_share_q"] = ai[C.YEARS[-1]]
        ps_d = ps.with_overrides({f"ai_share@{t}": v for t, v in ai.items()})
        for t, a in ai.items():
            if t >= FLOOR_FIRST_YEAR:
                lam = 0.82 * ((1.0 - a) + a * (1.0 - s_tr) * (1.0 - s_inf))
                ov[f"metro_floor@{t}"] = lam
                ov[f"kanto_floor@{t}"] = lam * 60.0 / 82.0
        val["lambda_B_2040"] = ov[f"metro_floor@{C.YEARS[-1]}"]
    demand, adj = C.hw_demand(ps_d, D, eta)
    val["demand_hw_2040"] = demand[C.YEARS[-1]]
    add_t = as_tuple(adds)
    s = Setting(overrides=as_tuple(ov),
                additions_by_role=(("C-CN", add_t), ("R-CN", add_t)) if adds else (),
                carbon=as_tuple(carbon), demand=as_tuple(demand), eta=ETA_BASE,
                reinforce_g=g if reinforce else None)
    return s, val, adj


def _status(row: Dict[str, Any]) -> str:
    st = [row[f"status_{r}"] for r in ROLES]
    return "Optimal" if all(x == "Optimal" for x in st) else ";".join(
        f"{r}:{x}" for r, x in zip(ROLES, st))


def run_block(block: str, blocks: Dict[str, Any], ps: ParamSet,
              use_cache: bool) -> List[Dict[str, Any]]:
    um, names = u_matrix(block, blocks)
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    n, step = um.shape[0], CHUNK[block]
    for start in range(0, n, step):
        built = []
        for i in range(start, min(start + step, n)):
            s, val, adj = build_sample(names, um[i], ps, reinforce=(block == "R"))
            built.append((f"{block}{i:04d}", i, s, val, adj))
        res = solve_settings([(cid, s) for cid, _, s, _, _ in built], ROLES, PART, use_cache)
        for cid, i, s, val, adj in built:
            m = metrics(cid, s, res, ROLES)
            m = {k: v for k, v in m.items() if not k.startswith("siting_s2")}
            m["demand_running_max"] = adj
            row: Dict[str, Any] = {"block": block, "sample": i}
            row.update({f"u_{k}": float(x) for k, x in zip(names, um[i])})
            row.update({f"value_{k}": v for k, v in val.items()})
            row.update(m)
            row["status"] = _status(m)
            rows.append(row)
        del res
        done = min(start + step, n)
        el = time.perf_counter() - t0
        print(f"[mc] block {block}: {done}/{n} samples, {el:.0f} s, projected "
              f"{el / done * n:.0f} s", flush=True)
    return rows


def _stats(v: np.ndarray) -> Dict[str, float]:
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "p5": float(np.percentile(v, 5)), "p25": float(np.percentile(v, 25)),
            "p50": float(np.percentile(v, 50)), "p75": float(np.percentile(v, 75)),
            "p95": float(np.percentile(v, 95)), "min": float(v.min()), "max": float(v.max())}


def summarize(df: pd.DataFrame, block: str, names: Tuple[str, ...], lo: float, hi: float,
              base_adv: float) -> Dict[str, Any]:
    d = df[df.block == block]
    ok = d[d.status == "Optimal"]
    bad = d[d.status != "Optimal"]
    a = ok["advantage"].to_numpy(dtype=float)
    fails = []
    for _, r in bad.iterrows():
        rec: Dict[str, Any] = {"sample": int(r["sample"]), "status": r["status"]}
        rec.update({f"u_{k}": float(r[f"u_{k}"]) for k in names})
        rec.update({c: float(r[c]) for c in d.columns
                    if c.startswith("value_") and not pd.isna(r[c])})
        fails.append(rec)
    share_fail = len(bad) / len(d) if len(d) else math.nan
    out: Dict[str, Any] = {
        "n": int(len(d)), "n_ok": int(len(ok)), "n_failed": int(len(bad)),
        "failure_share": share_fail, "failure_share_above_1pct": bool(share_fail > 0.01),
        "failures": fails, "factors": list(names),
        "advantage": _stats(a),
        "advantage_ex_penalty": _stats(ok["advantage_ex_penalty"].to_numpy(dtype=float)),
        "total_R-CN_TJPY": _stats(ok["total_R-CN_TJPY"].to_numpy(dtype=float)),
        "total_C-CN_TJPY": _stats(ok["total_C-CN_TJPY"].to_numpy(dtype=float)),
        "share_positive": float((a > 0).mean()),
        "share_in_published_range": float(((a >= lo) & (a <= hi)).mean()),
        "share_ex_penalty_positive": float((ok["advantage_ex_penalty"] > 0).mean()),
        "share_any_curtailment": float(((ok["curtail_C-CN_TWh"] > 1e-6)
                                        | (ok["curtail_R-CN_TWh"] > 1e-6)).mean()),
        "share_demand_running_max": float(d["demand_running_max"].astype(bool).mean()),
        "base_advantage": base_adv,
        "quantile_of_base_advantage": float((a <= base_adv).mean()),
        "co2_cum_R-CN_Mt": _stats(ok["co2_cum_R-CN_Mt"].to_numpy(dtype=float)),
        "nonmetro_2040_R-CN": _stats(ok["nonmetro_2040_R-CN"].to_numpy(dtype=float)),
    }
    return out


def paired(df: pd.DataFrame) -> Dict[str, Any]:
    m = df[(df.block == "M") & (df.status == "Optimal")].set_index("sample")["advantage"]
    r = df[(df.block == "R") & (df.status == "Optimal")].set_index("sample")["advantage"]
    j = pd.concat([m.rename("M"), r.rename("R")], axis=1, join="inner")
    diff = (j["R"] - j["M"]).to_numpy(dtype=float)
    return {"n_pairs": int(len(j)), "advantage_R_minus_M": _stats(diff),
            "share_R_above_M": float((diff > 0).mean()),
            "share_R_below_M": float((diff < 0).mean())}


def prcc_rows(df: pd.DataFrame, block: str, names: Tuple[str, ...]) -> List[Dict[str, Any]]:
    ok = df[(df.block == block) & (df.status == "Optimal")]
    X = ok[[f"u_{k}" for k in names]].to_numpy(dtype=float)
    rows: List[Dict[str, Any]] = []
    for output in OUTPUTS:
        y = ok[output].to_numpy(dtype=float)
        r, p = U.prcc(X, y)
        lo, hi = U.bootstrap_prcc(X, y, BOOT_B, BOOT_SEED)
        order = np.argsort(-np.abs(r), kind="stable")
        rank = np.empty(len(names), dtype=int)
        rank[order] = np.arange(1, len(names) + 1)
        for j, k in enumerate(names):
            sp = stats.spearmanr(X[:, j], y)
            rows.append({"block": block, "output": output, "factor": k, "prcc": r[j], "p": p[j],
                         "ci_lo": lo[j], "ci_hi": hi[j], "rank": int(rank[j]),
                         "spearman": float(sp.statistic)})
    return rows


def bin_rows(df: pd.DataFrame, block: str, names: Tuple[str, ...], lo: float,
             hi: float) -> List[Dict[str, Any]]:
    ok = df[(df.block == block) & (df.status == "Optimal")]
    y = ok["advantage"].to_numpy(dtype=float)
    rows: List[Dict[str, Any]] = []
    for k in names:
        b = U.quintile_bins(ok[f"u_{k}"].to_numpy(dtype=float), y, lo, hi)
        for rec in b.to_dict(orient="records"):
            rows.append({"block": block, "factor": k, **rec})
    return rows


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    cfg = load_mc_config()
    lo, hi = cfg["target_range"]
    ps = C.params()
    check_ranges(ps)
    base_adv = float(json.loads((C.OUT / "base_summary.json").read_text())["advantage_R-CN"])
    rows: List[Dict[str, Any]] = []
    for block in ORDER:
        rows += run_block(block, cfg["blocks"], ps, use_cache)
    df = pd.DataFrame(rows)
    names_of = {b: (FACTORS if b == "X" else FACTORS[:7]) for b in ORDER}
    lead = ["block", "sample", "status"]
    ucols = [f"u_{k}" for k in FACTORS]
    vcols = [c for c in df.columns if c.startswith("value_")]
    rest = [c for c in df.columns if c not in lead + ucols + vcols]
    df = df[lead + ucols + vcols + rest]
    C.write_csv(df, "mc_samples.csv", out)
    summ: Dict[str, Any] = {
        "definition": "config/experiments_v35.yaml uncertainty",
        "target_range": [lo, hi],
        "target_range_label": "掲載論文の報告値の幅（比較のためだけ。目標ではない）",
        "value_columns": VALUE_NOTE,
        "blocks": {b: summarize(df, b, names_of[b], lo, hi, base_adv) for b in ORDER},
        "paired_R_minus_M": paired(df),
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "interval": "percentile 2.5-97.5%"},
    }
    prows: List[Dict[str, Any]] = []
    brows: List[Dict[str, Any]] = []
    for b in ORDER:
        prows += prcc_rows(df, b, names_of[b])
        brows += bin_rows(df, b, names_of[b], lo, hi)
    pr = pd.DataFrame(prows)
    for b in ORDER:
        for o in OUTPUTS:
            top = pr[(pr.block == b) & (pr.output == o)].sort_values("rank").head(3)
            summ["blocks"][b].setdefault("top3_prcc", {})[o] = [
                {"factor": r["factor"], "prcc": r["prcc"]} for _, r in top.iterrows()]
    C.write_csv(pr, "mc_prcc.csv", out)
    C.write_csv(pd.DataFrame(brows), "mc_bins.csv", out)
    C.write_json(summ, "mc_summary.json", out)
    C.write_run_log(PART, out)
    return C.PartResult(files=list(FILES), summary=summ)
