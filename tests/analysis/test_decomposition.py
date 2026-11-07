"""Synthetic checks of the cost-channel split and the LMDI decomposition."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dc_decom_analysis import decomposition, lmdi  # noqa: E402

COLS = ("cooling_capex", "cooling_opex", "cooling_premium", "cgs_capex", "cgs_opex",
        "cgs_premium", "bess_capex", "bess_opex", "bess_premium", "pv_capex", "pv_opex",
        "energy_grid", "energy_ppa", "energy_pv", "energy_cgs_fuel", "carbon_grid",
        "carbon_cgs", "curtailment", "subsidy", "adder")


def _components(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year in (2024, 2025, 2026):
        for region in ("Kanto", "Hokkaido"):
            row = {"year": year, "region": region, "procured_TWh": float(rng.uniform(1, 10))}
            row.update({c: float(rng.uniform(0, 1e11)) for c in COLS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_channels_sum_to_total_difference() -> None:
    c, r = _components(1), _components(2)
    split = decomposition.principled(c, r)
    total = sum(split[k] for k in decomposition.CHANNELS)
    direct = decomposition.run_total(c) - decomposition.run_total(r)
    assert abs(total - direct) <= 1e-6 * max(1.0, abs(direct))
    assert abs(split["total"] - split["direct_total"]) <= 1e-6 * max(1.0, abs(direct))
    sh = decomposition.shares(split)
    assert abs(sum(sh.values()) - 1.0) <= 1e-9


def test_lmdi_sums_to_change_with_zero_end_state() -> None:
    f = ("activity", "structure", "pue", "grid_share", "ef")
    x0 = {"Kanto": dict(zip(f, (20.0, 0.6, 1.4, 0.9, 0.44))),
          "Hokkaido": dict(zip(f, (20.0, 0.4, 1.3, 0.8, 0.49)))}
    xT = {"Kanto": dict(zip(f, (60.0, 0.5, 1.1, 0.0, 0.10))),     # zero-emission end state
          "Hokkaido": dict(zip(f, (60.0, 0.5, 1.08, 0.2, 0.11)))}
    d = lmdi.decompose(x0, xT, cgs0=0.3, cgsT=0.0)
    assert abs(sum(d[k] for k in lmdi.FACTORS) + d["cgs"] - d["total"]) <= 1e-12
    assert abs(d["total"] - d["exact_delta"]) <= 1e-6
    assert d["grid_share"] < 0


def test_lmdi_between_identical_runs_is_zero() -> None:
    table = pd.DataFrame({
        "year": [2040, 2040], "region": ["Kanto", "Kyushu"],
        "it_energy_TWh": [30.0, 10.0], "load_TWh": [33.0, 11.0],
        "E_grid_TWh": [5.0, 0.0], "ef_grid": [0.1, 0.08],
        "emissions_grid_Mt": [0.5, 0.0], "emissions_cgs_Mt": [0.1, 0.0],
    })
    d = lmdi.between(table, table, 2040)
    assert all(abs(d[k]) <= 1e-12 for k in lmdi.FACTORS + ("cgs", "total"))
