"""Run all three DC-DECOM scenarios and print the headline numbers.

Usage:
    python src/scripts/run_all.py

Outputs the LP-derived cost-component table, curtailment, the C-CN -> R-CN
advantage and its decomposition, followed by the reported values of the
Energies article for comparison (constants from the article, not model
output).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dc_decom.load_inputs import load_global_parameters, load_regions, load_temperature_96h
from dc_decom.model import build_and_solve
from dc_decom.postprocess import (
    cost_components,
    headline_lp,
    reported_values,
)
from dc_decom.scenarios import ALL_SCENARIOS


def main() -> int:
    p = load_global_parameters()
    regions = load_regions()
    temps = load_temperature_96h()

    results = {}
    for name, scenario in ALL_SCENARIOS.items():
        print(f"Solving {name} ...", flush=True)
        results[name] = build_and_solve(scenario, regions, p, temps,
                                         solver_msg=False)
        print(f"  status: {results[name].status}, "
              f"obj = {results[name].objective_yen/1e12:.2f} trillion yen")

    print("\n=== Cost components (trillion JPY, NPV 2024-2040) - LP output ===")
    print(cost_components(results.values()).to_string(index=False))

    lp = headline_lp(results)
    adv = lp["advantage"]
    print("\n=== R-CN advantage - LP derived ===")
    print(f"  C-CN total: {adv['C_CN_total_TJPY']:.2f} TJPY")
    print(f"  R-CN total: {adv['R_CN_total_TJPY']:.2f} TJPY")
    pct = adv["advantage_fraction"] * 100
    print(f"  advantage : {pct:+.2f}%")
    for name, twh in lp["curtail_TWh"].items():
        print(f"  curtailment {name}: {twh:.1f} TWh (cumulative 2024-2040)")
    print(f"  R-CN regional subsidy (deducted): {lp['subsidy_TJPY']['R-CN']:.3f} TJPY")

    print("\n=== Decomposition of C-CN - R-CN (LP derived) ===")
    for ch, v in lp["decomposition_TJPY"].items():
        share = lp["decomposition_shares"][ch] * 100
        print(f"  {ch:<15s} {v:+9.3f} TJPY  {share:+7.1f}%")

    rep = reported_values()
    print("\n=== Reported values (Energies 2026) - constants from the article, "
          "not model output ===")
    print(f"  source    : {rep['source']}")
    print(f"  REF total : {rep['REF_total_TJPY']:.2f} TJPY")
    print(f"  C-CN total: {rep['C_CN_total_TJPY']:.2f} TJPY")
    print(f"  R-CN total: {rep['R_CN_total_TJPY']:.2f} TJPY")
    print(f"  advantage : {rep['advantage_fraction_low']*100:.0f}-"
          f"{rep['advantage_fraction_high']*100:.0f}% "
          f"(gap = {rep['advantage_TJPY']:.2f} TJPY)")
    print("  Table 10 channel split:")
    print(rep["decomposition"].to_string(index=False))

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    for name, res in results.items():
        with open(out_dir / f"{name.replace('-', '_').lower()}.json", "w") as f:
            json.dump({
                "scenario": res.scenario,
                "status": res.status,
                "objective_yen": res.objective_yen,
                "capex_total": res.capex_total,
                "opex_total": res.opex_total,
                "energy_cost": res.energy_cost,
                "carbon_cost": res.carbon_cost,
                "curtail_cost": res.curtail_cost,
                "subsidy_total": res.subsidy_total,
                "curtail_TWh_total": res.curtail_TWh_total,
            }, f, indent=2)
    print(f"\nResults written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
