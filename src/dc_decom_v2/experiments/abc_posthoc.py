"""A/B/C post-hoc sensitivity (事後の感度) and the breakdown of B - A0.

Definitions: config/abc_posthoc.yaml (definitions H and I), written before
any of these runs. Both analyses were
defined after the frozen A/B/C results had been seen; every output carries the
label 事後の感度 and none replaces the frozen comparison (outputs/abc/abc_*).

Part abc_posthoc (definition H). The frozen rules are checked
(``abc.rules_check``) and read once; a variant changes only an
in-memory deep copy of the rules dictionary (the connection-usage threshold of
R2) and the B settings (phi_0, T_g, P*) or the base options (certificates of
R4). For each variant B' (adaptive false) and C' (adaptive true) are solved
with the same settings on the 18 realization paths of the frozen rules; the
effect is C' - B' per path. Variants with the same base and rules copy share
one ``rolling.run_designs`` call. Variant 'frozen' must reproduce B and C of
outputs/abc/abc_metrics.csv (relative 1e-9).

Part b_minus_a0 (definition I). Four policy elements of B, each on or off:
E1 PUE rules, E2 liquid floor phi_0 = 1.0, E3 reinforcement (c1 from
T_g = 2030, g, theta), E4 non-metro PPA tier multiplier 1.5. The 16
combinations are built from the B settings by switching elements off (psi 0.3,
subsidy 'rule' and P* = 0 are kept; they are inert because P* = 0 gives no
subsidy and psi binds only subsidized builds) and solved with the rolling
procedure on the 18 paths, together with A0. Exact Shapley values
(dc_decom_analysis.shapley) of the social resource cost and the operator cost
per path, and the accounting resource lines of B and A0.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import yaml  # type: ignore[import-untyped]

from dc_decom_analysis import shapley

from .. import accounting, rolling
from . import abc
from . import common as C

OUT = abc.OUT
CONFIG = C.REPO / "config" / "abc_posthoc.yaml"
LABEL = "事後の感度"
FROZEN_TOL = 1e-9          # relative, frozen variant vs abc_metrics.csv; subsets vs A0 / B
SUM_TOL = 1e-9             # TJPY, Shapley efficiency and line sums
POSTHOC_PART = "abc_posthoc"
BMA_PART = "b_minus_a0"
POSTHOC_FILES = ("posthoc_metrics.csv", "posthoc_switches.csv", "posthoc_summary.csv",
                 "posthoc_summary.json", "run_log_posthoc.csv")
BMA_FILES = ("b_minus_a0_lines.csv", "b_minus_a0_shapley.csv", "b_minus_a0_subsets.csv",
             "b_minus_a0_summary.json", "run_log_b_minus_a0.csv")
# variant change key -> Settings field (B'), rules threshold (R2), base option (R4)
SETTING_KEYS = {"phi_0": "phi", "reinforcement_year": "t_g", "pstar": "pstar"}
RULE_KEYS = {"connection_usage": ("thresholds", "connection_usage", "value")}
OPTION_KEYS = ("certificates",)
RULES = ("R1", "R2", "R3", "R4")
DIFF_METRICS = ("social_cost_TJPY", "eq1_total_TJPY", "operator_cost_TJPY",
                "fiscal_burden_TJPY", "co2_cum_Mt")
ELEMENTS = ("E1", "E2", "E3", "E4")
ELEMENT_OFF: Dict[str, Dict[str, Any]] = {
    "E1": {"pue_rules": False}, "E2": {"phi": None}, "E3": {"t_g": None},
    "E4": {"ppa_multiplier": None}}
ELEMENT_NAMES = {"E1": "PUE rules", "E2": "liquid floor phi_0 = 1.0",
                 "E3": "reinforcement (c1 from T_g = 2030, g, theta)",
                 "E4": "non-metro PPA tier multiplier 1.5"}
SHAPLEY_METRICS = ("social_cost_TJPY", "operator_cost_TJPY")
SUBSET_METRICS = ("social_cost_TJPY", "eq1_total_TJPY", "operator_cost_TJPY",
                  "fiscal_burden_TJPY", "reinforcement_TJPY", "co2_cum_Mt",
                  "nonmetro_share_2040", "liquid_share_new_cum", "ppa_cum_TWh")
OTHER_LINES = {"social_cost": "social_disc", "operator_cost": "operator_disc",
               "fiscal_burden": "subsidy_disc", "carbon_cost": "carbon_disc",
               "eq1_total": "eq1_disc"}


def load_config() -> Dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Variants (definition H)
# ----------------------------------------------------------------------

@dataclass
class Variant:
    name: str
    focus: str                       # R1 .. R4, or 'none' for the reference
    changes: Dict[str, Any]
    rules: Dict[str, Any]            # deep copy of the frozen rules (thresholds changed)
    base: rolling.RollingBase
    b: rolling.Design                # B'
    c: rolling.Design                # C'


def variant_rules(rules: Mapping[str, Any], changes: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep copy of the frozen rules with the listed thresholds changed."""
    out = copy.deepcopy(dict(rules))
    for k, path in RULE_KEYS.items():
        if k in changes:
            node = out
            for p in path[:-1]:
                node = node[p]
            node[path[-1]] = float(changes[k])
    return out


def variant_settings(b: rolling.Settings, changes: Mapping[str, Any]) -> rolling.Settings:
    kw: Dict[str, Any] = {}
    for k, f in SETTING_KEYS.items():
        if k in changes:
            kw[f] = int(changes[k]) if f == "t_g" else float(changes[k])
    return b.with_(**kw)


def variant_base(base: rolling.RollingBase, changes: Mapping[str, Any]) -> rolling.RollingBase:
    if "certificates" in changes:
        return replace(base, options=base.options.with_(certificates=bool(changes["certificates"])))
    return base


def build_variants(rules: Mapping[str, Any], cfg: Mapping[str, Any]) -> List[Variant]:
    ps = C.params()
    b = rolling.designs_from_rules(rules, ps, diagnostics=False)["B"].settings
    base = abc.base_config()
    out = []
    for name, v in cfg["posthoc"]["variants"].items():
        ch = v["change"] if isinstance(v["change"], dict) else {}
        unknown = set(ch) - set(SETTING_KEYS) - set(RULE_KEYS) - set(OPTION_KEYS)
        if unknown:
            raise ValueError(f"{name}: unknown change keys {sorted(unknown)}")
        s = variant_settings(b, ch)
        focus = name.split("-")[0] if name != "frozen" else "none"
        out.append(Variant(name=name, focus=focus, changes=dict(ch),
                           rules=variant_rules(rules, ch), base=variant_base(base, ch),
                           b=rolling.Design(f"{name}-B", s),
                           c=rolling.Design(f"{name}-C", s, adaptive=True)))
    return out


def _groups(variants: Sequence[Variant]) -> List[List[Variant]]:
    groups: List[List[Variant]] = []
    for v in variants:
        for g in groups:
            if g[0].base == v.base and g[0].rules == v.rules:
                g.append(v)
                break
        else:
            groups.append([v])
    return groups


def _changes_text(ch: Mapping[str, Any]) -> str:
    return "; ".join(f"{k}={v}" for k, v in ch.items()) if ch else "none"


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    skip = {"design", "path", "carbon", "demand", "liquid", "status", "note", "variant",
            "analysis", "meets_cap_path"}
    return [c for c in df.columns if c not in skip]


def max_rel_diff(a: pd.DataFrame, b: pd.DataFrame, cols: Sequence[str]) -> float:
    """Largest |x - y| / max(|x|, |y|, 1e-12) over the paths and columns (NaN = NaN)."""
    a = a.set_index("path")
    b = b.set_index("path")
    worst = 0.0
    for p in a.index:
        for c in cols:
            x, y = float(a.at[p, c]), float(b.at[p, c])
            if math.isnan(x) and math.isnan(y):
                continue
            if math.isnan(x) or math.isnan(y):
                return math.inf
            worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-12))
    return worst


def effective_policy(pol: Any, t0: int) -> Any:
    """JSON form of the policy from t0 (``rolling._from``) with the subsidy map
    reduced to its non-zero entries: a region whose subsidy years all lie
    before t0 (an empty map after the cut) pays the same as a region without
    an entry. rolling.switch compares the plain ``_from`` forms, so its
    'policy_changed' flag is also True when the subsidy rule only switches
    off a map that is empty from t0 (B with P* > 0 at a review after its last
    subsidy year); the frozen run (P* = 0, no map) is not affected."""
    d = rolling._from(pol, t0)
    sub = d.get("subsidy_yen_per_kW") or {}
    d["subsidy_yen_per_kW"] = {r: {t: v for t, v in m.items() if v != 0}
                               for r, m in sub.items() if any(v != 0 for v in m.values())}
    return d


def effective_changes(r: rolling.RollingRun, base: rolling.RollingBase,
                      ps: Any) -> List[bool]:
    """Per row of r.switches: does the switched setting change the policy of the
    years from the review year (``effective_policy``)?"""
    years = list(base.review_years)

    def pol(x: rolling.Settings) -> Any:
        return rolling.policy_for(x, ps, base.forecast.carbon_price, base.scenario)

    out = []
    for row in r.switches:
        if not row["setting"]:
            out.append(False)
            continue
        t = int(row["review_year"])
        s = r.stage_settings[years.index(t) - 1]
        after = s.with_(**{row["setting"]: row["after"]})
        out.append(effective_policy(pol(s), t) != effective_policy(pol(after), t))
    return out


def _rule_counts(sw: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    trig = sw[sw["rule"].isin(RULES)]
    for rule in RULES:
        g = trig[trig["rule"] == rule]
        out[f"{rule}_triggered"] = int(g["triggered"].astype(bool).sum())
        out[f"{rule}_setting_changed"] = int(g["changed"].astype(bool).sum())
        out[f"{rule}_policy_changed"] = int(g["policy_changed"].astype(bool).sum())
        out[f"{rule}_policy_changed_effective"] = int(
            g["policy_changed_effective"].astype(bool).sum())
        v = pd.to_numeric(g["value"], errors="coerce")
        out[f"{rule}_value_min"] = float(v.min()) if v.notna().any() else math.nan
        out[f"{rule}_value_max"] = float(v.max()) if v.notna().any() else math.nan
    for col, tag in (("policy_changed", ""), ("policy_changed_effective", "_effective")):
        pc = trig[trig[col].astype(bool)]
        out[f"n_policy_changed{tag}"] = int(len(pc))
        paths = list(dict.fromkeys(pc["path"]))
        out[f"n_paths_policy_changed{tag}"] = len(paths)
        out[f"paths_policy_changed{tag}"] = "; ".join(paths) or "none"
    return out


def _indicators(sw: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    trig = sw[sw["rule"].isin(RULES)]
    for rule, g in trig.groupby("rule"):
        out[str(rule)] = {}
        for year, h in g.groupby("review_year"):
            v = pd.to_numeric(h["value"], errors="coerce")
            thr = sorted({str(x) for x in h["threshold"]})
            out[str(rule)][str(int(year))] = {
                "indicator": str(h["indicator"].iloc[0]),
                "threshold": thr[0] if len(thr) == 1 else thr,
                "n": int(len(h)), "value_min": float(v.min()), "value_max": float(v.max()),
                "n_triggered": int(h["triggered"].astype(bool).sum()),
                "n_setting_changed": int(h["changed"].astype(bool).sum()),
                "n_policy_changed": int(h["policy_changed"].astype(bool).sum()),
                "n_policy_changed_effective": int(
                    h["policy_changed_effective"].astype(bool).sum()),
                "paths_policy_changed_effective": [
                    str(p) for p in h[h["policy_changed_effective"].astype(bool)]["path"]]}
    return out


def run(out: Path, use_cache: bool = True) -> C.PartResult:
    check = abc.rules_check()
    rules = abc.load_rules()
    cfg = load_config()
    variants = build_variants(rules, cfg)
    ps = C.params()
    paths = rolling.actual_paths(
        ps, carbon=rules["realization_paths"]["carbon_price"],
        demand=rules["realization_paths"]["demand_TWh"],
        liquid=rules["realization_paths"]["liquid_deploy_cap"])

    def solve(specs: List[Any], tag: str) -> List[Any]:
        return C.solve(specs, POSTHOC_PART, use_cache)

    rows: List[Dict[str, Any]] = []
    sws: List[pd.DataFrame] = []
    of = {}
    for v in variants:
        of[v.b.name] = (v.name, "B'")
        of[v.c.name] = (v.name, "C'")
    for grp in _groups(variants):
        designs = [d for v in grp for d in (v.b, v.c)]
        _, runs = rolling.run_designs(grp[0].base, designs, paths, grp[0].rules, solve=solve,
                                      keep_flows=False)
        for r in runs:
            name, role = of[r.design.name]
            rows.append({"analysis": LABEL, "variant": name,
                         **{**abc.metrics_row(r), "design": role}})
        adaptive = [r for r in runs if r.design.adaptive]
        sw = abc.switches_frame(adaptive)
        sw["policy_changed_effective"] = [x for r in adaptive
                                          for x in effective_changes(r, grp[0].base, ps)]
        sw["variant"] = [of[d][0] for d in sw["design"]]
        sw["design"] = "C'"
        sws.append(sw)
    df = pd.DataFrame(rows)
    sw_all = pd.concat(sws, ignore_index=True)
    sw_all.insert(0, "variant", sw_all.pop("variant"))
    sw_all.insert(0, "analysis", LABEL)
    C.write_csv(df, "posthoc_metrics.csv", out)
    C.write_csv(sw_all, "posthoc_switches.csv", out)

    frozen_abc = pd.read_csv(abc.OUT / "abc_metrics.csv")
    a_frozen = frozen_abc[frozen_abc["design"] == "A"].set_index("path")
    cols = _numeric_cols(frozen_abc)
    fz = df[df["variant"] == "frozen"]
    frozen_check: Dict[str, Any] = {"tolerance_relative": FROZEN_TOL,
                                    "reference": "outputs/abc/abc_metrics.csv"}
    for role, ref in (("B'", "B"), ("C'", "C")):
        d = max_rel_diff(fz[fz["design"] == role][["path"] + cols],
                         frozen_abc[frozen_abc["design"] == ref][["path"] + cols], cols)
        frozen_check[f"{ref}_max_rel_diff"] = d
    frozen_check["ok"] = all(frozen_check[f"{x}_max_rel_diff"] <= FROZEN_TOL for x in ("B", "C"))

    srows: List[Dict[str, Any]] = []
    vjson: Dict[str, Any] = {}
    for v in variants:
        g = df[df["variant"] == v.name]
        s = v.b.settings
        row: Dict[str, Any] = {
            "analysis": LABEL, "variant": v.name, "rule_focus": v.focus,
            "changes": _changes_text(v.changes),
            "certificates": bool(v.base.options.certificates),
            "connection_usage": float(v.rules["thresholds"]["connection_usage"]["value"]),
            "phi_0": s.phi, "t_g": s.t_g, "pstar": s.pstar,
            "n_paths": int(g["path"].nunique()),
            "n_failed": int((g["status"] != "Optimal").sum())}
        sw = sw_all[sw_all["variant"] == v.name]
        row.update(_rule_counts(sw))
        cmp = abc.compare(g, "C'", "B'", DIFF_METRICS)
        for m in DIFF_METRICS:
            for k in ("mean", "min", "max"):
                row[f"dCB_{m}_{k}"] = cmp[m][f"{k}_diff"]
            row[f"dCB_{m}_n_better"] = cmp[m]["n_better"]
            row[f"dCB_{m}_n_worse"] = cmp[m]["n_worse"]
        ok = g[g["status"] == "Optimal"]
        for role, tag in (("C'", "CmA"), ("B'", "BmA")):
            h = ok[ok["design"] == role].set_index("path")
            for m in ("social_cost_TJPY", "co2_cum_Mt"):
                # frozen A has certificates; no like-for-like A without certificates
                row[f"{tag}_{m}_mean"] = (math.nan if not v.base.options.certificates else float(
                    (h[m] - a_frozen.loc[h.index, m]).mean()))
        srows.append(row)
        vjson[v.name.replace(".", "_")] = {
            "name": v.name, "rule_focus": v.focus, "changes": v.changes,
            "reason": cfg["posthoc"]["variants"][v.name].get("reason"),
            "value_status": cfg["posthoc"]["variants"][v.name].get("value_status"),
            "settings": abc.settings_dict(v.b), "certificates": row["certificates"],
            "connection_usage": row["connection_usage"],
            "counts": _rule_counts(sw), "indicators": _indicators(sw),
            "C_minus_B": cmp,
            "C_minus_A_social_cost_mean": row["CmA_social_cost_TJPY_mean"],
            "C_minus_A_co2_cum_mean": row["CmA_co2_cum_Mt_mean"]}
    summ_df = pd.DataFrame(srows)
    C.write_csv(summ_df, "posthoc_summary.csv", out)
    summary: Dict[str, Any] = {
        "label": LABEL,
        "label_text": cfg["label"],
        "rules": check,
        "config": "config/abc_posthoc.yaml",
        "changes_after_registration": cfg.get("changes_after_registration") or [],
        "common": cfg["posthoc"]["common"],
        "rule_conditions": cfg["posthoc"]["rule_conditions"],
        "n_paths": len(paths), "paths": [p.name for p in paths],
        "review_years": list(abc.base_config().review_years),
        "tolerance_relative": abc.REL_TOL,
        "variant_keys": "variant name with '.' replaced by '_' (token-safe)",
        "A_reference": "frozen A of abc_metrics.csv (certificates on); C' - A is not reported "
                       "for the variant without certificates",
        "frozen_check": frozen_check,
        "n_runs": int(len(df)),
        "failed": [{"variant": r["variant"], "design": r["design"], "path": r["path"],
                    "note": r["note"]} for r in rows if r["status"] != "Optimal"],
        "policy_change_note": (
            "policy_changed = the flag of rolling.switch (as in abc_switches.csv); "
            "policy_changed_effective = the same comparison with the subsidy map reduced "
            "to its non-zero entries (effective_policy). They differ only for R3 at a "
            "review after the last subsidy year of the B rule, where switching the "
            "subsidy off changes no payment; the results use the effective flag"),
        "n_variants_with_policy_change": int((summ_df["n_policy_changed"] > 0).sum()),
        "n_variants_with_policy_change_effective": int(
            (summ_df["n_policy_changed_effective"] > 0).sum()),
        "variants": vjson,
    }
    C.write_json(summary, "posthoc_summary.json", out)
    log = [r for r in C.RUN_LOG if r["part"] == POSTHOC_PART]
    C.write_csv(pd.DataFrame(log).assign(analysis=LABEL), "run_log_posthoc.csv", out)
    if not frozen_check["ok"]:
        raise RuntimeError(f"variant 'frozen' differs from abc_metrics.csv: {frozen_check}")
    return C.PartResult(files=list(POSTHOC_FILES), summary=summary)


# ----------------------------------------------------------------------
# B - A0 (definition I)
# ----------------------------------------------------------------------

def bits(on: frozenset) -> str:
    return "".join("1" if e in on else "0" for e in ELEMENTS)


def subsets() -> List[frozenset]:
    return [frozenset(e for e, x in zip(ELEMENTS, f) if x)
            for f in product((False, True), repeat=len(ELEMENTS))]


def subset_settings(b: rolling.Settings, on: frozenset) -> rolling.Settings:
    kw: Dict[str, Any] = {}
    for e in ELEMENTS:
        if e not in on:
            kw.update(ELEMENT_OFF[e])
    return b.with_(**kw)


def subset_designs(rules: Mapping[str, Any]) -> Tuple[rolling.Design, Dict[str, rolling.Design]]:
    ds = rolling.designs_from_rules(rules, C.params(), diagnostics=False)
    b = ds["B"].settings
    return ds["A0"], {bits(s): rolling.Design(f"S{bits(s)}", subset_settings(b, s))
                      for s in subsets()}


def _lines(r: rolling.RollingRun) -> Dict[str, float]:
    ry = r.tables["region_year"]
    out = {k: float(ry[f"{k}_disc"].sum()) / C.TJ for k in accounting.RESOURCE_COLUMNS}
    out.update({k: float(ry[c].sum()) / C.TJ for k, c in OTHER_LINES.items()})
    return out


def run_b_minus_a0(out: Path, use_cache: bool = True) -> C.PartResult:
    check = abc.rules_check()
    rules = abc.load_rules()
    cfg = load_config()["b_minus_a0"]
    ps = C.params()
    a0, subs = subset_designs(rules)
    full, empty = bits(frozenset(ELEMENTS)), bits(frozenset())
    paths = rolling.actual_paths(
        ps, carbon=rules["realization_paths"]["carbon_price"],
        demand=rules["realization_paths"]["demand_TWh"],
        liquid=rules["realization_paths"]["liquid_deploy_cap"])

    def solve(specs: List[Any], tag: str) -> List[Any]:
        return C.solve(specs, BMA_PART, use_cache)

    _, runs = rolling.run_designs(abc.base_config(), [a0] + list(subs.values()), paths, rules,
                                  solve=solve, keep_flows=False)
    by = {(r.design.name, r.path.name): r for r in runs}
    failed = [{"design": r.design.name, "path": r.path.name, "note": r.note}
              for r in runs if not r.ok]
    if failed:
        raise RuntimeError(f"b_minus_a0: failed runs {failed}")
    # subsets table
    srows: List[Dict[str, Any]] = []
    met: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in runs:
        mr = abc.metrics_row(r)
        met[(r.design.name, r.path.name)] = mr
        b_ = r.design.name[1:] if r.design.name != "A0" else "A0"
        flags = {e: (int(b_[i]) if b_ != "A0" else 0) for i, e in enumerate(ELEMENTS)}
        srows.append({"analysis": LABEL, "design": r.design.name,
                      "subset": b_ if b_ != "A0" else "A0", **flags, "path": r.path.name,
                      "status": r.status, **{k: mr[k] for k in SUBSET_METRICS}})
    C.write_csv(pd.DataFrame(srows), "b_minus_a0_subsets.csv", out)

    frozen_abc = pd.read_csv(abc.OUT / "abc_metrics.csv").set_index(["design", "path"])
    checks: Dict[str, Any] = {"tolerance_relative": FROZEN_TOL, "tolerance_sum_TJPY": SUM_TOL}

    def rel(x: float, y: float) -> float:
        return abs(x - y) / max(abs(x), abs(y), 1e-12)

    pn = [p.name for p in paths]
    checks["empty_vs_A0_max_rel"] = max(
        rel(met[(f"S{empty}", p)]["social_cost_TJPY"], met[("A0", p)]["social_cost_TJPY"])
        for p in pn)
    checks["full_vs_B_max_rel"] = max(
        rel(met[(f"S{full}", p)]["social_cost_TJPY"],
            float(frozen_abc.at[("B", p), "social_cost_TJPY"])) for p in pn)
    checks["A0_vs_abc_metrics_max_rel"] = max(
        rel(met[("A0", p)]["social_cost_TJPY"],
            float(frozen_abc.at[("A0", p), "social_cost_TJPY"])) for p in pn)
    checks["full_settings_equal_B"] = (subs[full].settings ==
                                       rolling.designs_from_rules(rules, ps, False)["B"].settings)

    # Shapley values
    shrows: List[Dict[str, Any]] = []
    eff_err = {m: 0.0 for m in SHAPLEY_METRICS}
    eff_err_a0 = {m: 0.0 for m in SHAPLEY_METRICS}
    phis: Dict[str, Dict[str, List[float]]] = {m: {e: [] for e in ELEMENTS}
                                               for m in SHAPLEY_METRICS}
    tot: Dict[str, List[float]] = {m: [] for m in SHAPLEY_METRICS}
    oat: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        k: {m: {e: [] for e in ELEMENTS} for m in SHAPLEY_METRICS} for k in ("on", "off")}
    for p in pn:
        for m in SHAPLEY_METRICS:
            game = {s: float(met[(f"S{bits(s)}", p)][m]) for s in subsets()}
            phi = shapley.shapley(game, list(ELEMENTS))
            on1 = shapley.oat_on(game, list(ELEMENTS))
            off1 = shapley.oat_off(game, list(ELEMENTS))
            d_b_a0 = float(met[(f"S{full}", p)][m]) - float(met[("A0", p)][m])
            total = game[frozenset(ELEMENTS)] - game[frozenset()]
            eff_err[m] = max(eff_err[m], abs(sum(phi.values()) - total))
            eff_err_a0[m] = max(eff_err_a0[m], abs(sum(phi.values()) - d_b_a0))
            tot[m].append(d_b_a0)
            for e in ELEMENTS:
                phis[m][e].append(phi[e])
                oat["on"][m][e].append(on1[e])
                oat["off"][m][e].append(off1[e])
                shrows.append({"analysis": LABEL, "path": p, "metric": m, "element": e,
                               "name": ELEMENT_NAMES[e], "shapley_TJPY": phi[e],
                               "share": phi[e] / d_b_a0 if d_b_a0 != 0 else math.nan,
                               "oat_on_TJPY": on1[e], "oat_off_TJPY": off1[e],
                               "B_minus_A0_TJPY": d_b_a0})

    def mean(x: Sequence[float]) -> float:
        return sum(x) / len(x)

    mean_json: Dict[str, Any] = {}
    for m in SHAPLEY_METRICS:
        t = mean(tot[m])
        mean_json[m] = {"B_minus_A0_mean_TJPY": t, "min_TJPY": min(tot[m]),
                        "max_TJPY": max(tot[m]), "elements": {}}
        for e in ELEMENTS:
            ph = mean(phis[m][e])
            shrows.append({"analysis": LABEL, "path": "mean", "metric": m, "element": e,
                           "name": ELEMENT_NAMES[e], "shapley_TJPY": ph,
                           "share": ph / t if t != 0 else math.nan,
                           "oat_on_TJPY": mean(oat["on"][m][e]),
                           "oat_off_TJPY": mean(oat["off"][m][e]), "B_minus_A0_TJPY": t})
            mean_json[m]["elements"][e] = {
                "name": ELEMENT_NAMES[e], "shapley_mean_TJPY": ph,
                "share": ph / t if t != 0 else math.nan,
                "shapley_min_TJPY": min(phis[m][e]), "shapley_max_TJPY": max(phis[m][e]),
                "oat_on_mean_TJPY": mean(oat["on"][m][e]),
                "oat_off_mean_TJPY": mean(oat["off"][m][e])}
    C.write_csv(pd.DataFrame(shrows), "b_minus_a0_shapley.csv", out)
    checks["shapley_efficiency_max_err_TJPY"] = eff_err
    checks["shapley_sum_vs_B_minus_A0_max_err_TJPY"] = eff_err_a0

    # accounting lines of B (= full subset) and A0
    lrows: List[Dict[str, Any]] = []
    line_err = 0.0
    names = list(accounting.RESOURCE_COLUMNS) + list(OTHER_LINES)
    acc: Dict[str, Dict[str, List[float]]] = {k: {"B": [], "A0": []} for k in names}
    for p in pn:
        lb, la = _lines(by[(f"S{full}", p)]), _lines(by[("A0", p)])
        res_sum = sum(lb[k] - la[k] for k in accounting.RESOURCE_COLUMNS)
        d_soc = (float(met[(f"S{full}", p)]["social_cost_TJPY"])
                 - float(met[("A0", p)]["social_cost_TJPY"]))
        line_err = max(line_err, abs(res_sum - d_soc))
        for k in names:
            acc[k]["B"].append(lb[k])
            acc[k]["A0"].append(la[k])
            lrows.append({"analysis": LABEL, "path": p, "line": k,
                          "kind": "resource" if k in accounting.RESOURCE_COLUMNS else "total",
                          "B_TJPY": lb[k], "A0_TJPY": la[k], "diff_TJPY": lb[k] - la[k],
                          "share_of_social_diff": (lb[k] - la[k]) / d_soc if d_soc else math.nan})
    d_mean = mean(acc["social_cost"]["B"]) - mean(acc["social_cost"]["A0"])
    line_mean: Dict[str, Any] = {}
    for k in names:
        bm, am = mean(acc[k]["B"]), mean(acc[k]["A0"])
        lrows.append({"analysis": LABEL, "path": "mean", "line": k,
                      "kind": "resource" if k in accounting.RESOURCE_COLUMNS else "total",
                      "B_TJPY": bm, "A0_TJPY": am, "diff_TJPY": bm - am,
                      "share_of_social_diff": (bm - am) / d_mean if d_mean else math.nan})
        line_mean[k] = {"B_TJPY": bm, "A0_TJPY": am, "diff_TJPY": bm - am,
                        "share_of_social_diff": (bm - am) / d_mean if d_mean else math.nan}
    C.write_csv(pd.DataFrame(lrows), "b_minus_a0_lines.csv", out)
    checks["line_sum_vs_social_diff_max_err_TJPY"] = line_err
    checks["ok"] = bool(checks["empty_vs_A0_max_rel"] <= FROZEN_TOL
                        and checks["full_vs_B_max_rel"] <= FROZEN_TOL
                        and checks["full_settings_equal_B"]
                        and max(eff_err.values()) <= SUM_TOL
                        and max(eff_err_a0.values()) <= SUM_TOL
                        and line_err <= SUM_TOL)
    summary: Dict[str, Any] = {
        "label": LABEL, "label_text": cfg["label"], "rules": check,
        "config": "config/abc_posthoc.yaml (b_minus_a0)",
        "value_note": cfg["value_note"],
        "elements": dict(ELEMENT_NAMES),
        "excluded": cfg["excluded"],
        "n_paths": len(pn), "n_subsets": len(subs), "n_runs": len(runs),
        "B_minus_A0_social_mean_TJPY": d_mean,
        "shapley": mean_json,
        "lines_mean": line_mean,
        "checks": checks,
    }
    C.write_json(summary, "b_minus_a0_summary.json", out)
    log = [r for r in C.RUN_LOG if r["part"] == BMA_PART]
    C.write_csv(pd.DataFrame(log).assign(analysis=LABEL), "run_log_b_minus_a0.csv", out)
    if not checks["ok"]:
        raise RuntimeError(f"b_minus_a0 checks failed: {checks}")
    return C.PartResult(files=list(BMA_FILES), summary=summary)

