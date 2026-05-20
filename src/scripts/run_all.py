"""Run all three DC-DECOM scenarios and print the headline numbers.

Usage:
    python src/scripts/run_all.py

Outputs the cost-component table and the C-CN -> R-CN advantage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dc_decom.load_inputs import load_global_parameters, load_regions, load_temperature_96h
from dc_decom.model import build_and_solve
from dc_decom.postprocess import (
    calibrated_headline,
    cost_components,
    cost_decomposition,
    headline_advantage,
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

    adv = headline_advantage(results["C-CN"], results["R-CN"])
    print("\n=== R-CN advantage - LP derived ===")
    print(f"  C-CN total: {adv['C_CN_total_TJPY']:.2f} TJPY")
    print(f"  R-CN total: {adv['R_CN_total_TJPY']:.2f} TJPY")
    pct = adv["advantage_fraction"] * 100
    print(f"  advantage : {pct:+.2f}%")

    print("\n=== Cost decomposition (LP derived) ===")
    print(cost_decomposition(results["C-CN"], results["R-CN"]).to_string(index=False))

    cal = calibrated_headline(results["C-CN"], results["R-CN"])
    paper = cal["paper"]
    print("\n=== Paper headline (Section 4.2, Round 2 manuscript) ===")
    print(f"  C-CN total: {paper['C_CN_total_TJPY']:.2f} TJPY")
    print(f"  R-CN total: {paper['R_CN_total_TJPY']:.2f} TJPY")
    print(f"  advantage : {paper['advantage_fraction_low']*100:.0f}-"
          f"{paper['advantage_fraction_high']*100:.0f}% "
          f"(gap = {paper['advantage_TJPY']:.2f} TJPY)")

    print("\n=== Cost decomposition (Paper Table 10) ===")
    print(cal["decomposition"].to_string(index=False))

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
            }, f, indent=2)
    print(f"\nResults written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
