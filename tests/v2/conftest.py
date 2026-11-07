"""Shared fixtures of the v2 tests: paper parameters (data/parameters_spec) plus
the DEV structural placeholders (tests/fixtures/params_dev, status 'dev').
Solves are memoized per session; none of them is a result."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dc_decom_v2 import ModelOptions, ParamSet, Result, run  # noqa: E402
from dc_decom_v2.config import METRO  # noqa: E402
from dc_decom_v2.runner import DEV_DIR, SPEC_DIR, load_params  # noqa: E402
from dc_decom_v2.scenarios import default_policy  # noqa: E402


def tight_grid_policy(ps: ParamSet):
    """R-CN policy with a small connection capacity reinforced from 2026."""
    nonmetro = [r for r in ps.region_names if r not in METRO]
    return default_policy("R-CN", ps).with_(
        conn_capacity={r: {t: 0.15 for t in range(2026, 2041)} for r in nonmetro},
        reinforced_from={r: 2026 for r in nonmetro},
        operator_burden_share=0.4)


def late_grid_policy(ps: ParamSet):
    """R-CN policy whose reinforcement starts in 2030 (free connection before)."""
    nonmetro = [r for r in ps.region_names if r not in METRO]
    return default_policy("R-CN", ps).with_(
        conn_capacity={r: {t: (0.6 if t >= 2030 else 0.15) for t in range(2026, 2041)}
                       for r in nonmetro},
        reinforced_from={r: 2030 for r in nonmetro},
        operator_burden_share=0.4)


@pytest.fixture(scope="session")
def ps() -> ParamSet:
    return load_params(SPEC_DIR, DEV_DIR)


@pytest.fixture(scope="session")
def solved(ps) -> Callable[[str], Result]:
    configs: Dict[str, Callable[[], Result]] = {
        "rcn_rep96": lambda: run(ModelOptions(time_mode="rep96"), "R-CN", ps),
        "rcn_annual": lambda: run(ModelOptions(time_mode="annual"), "R-CN", ps),
        "ccn_hourly_rep96": lambda: run(ModelOptions(time_mode="rep96", ppa_matching="hourly"),
                                        "C-CN", ps),
        "ccn_hourly_annual": lambda: run(ModelOptions(time_mode="annual", ppa_matching="hourly"),
                                         "C-CN", ps),
        "ccn_cert_annual": lambda: run(ModelOptions(time_mode="annual", certificates=True),
                                       "C-CN", ps),
        "grid_annual": lambda: run(ModelOptions(time_mode="annual"), "R-CN", ps,
                                   policy=tight_grid_policy(ps)),
        "grid_rep96": lambda: run(ModelOptions(time_mode="rep96"), "R-CN", ps,
                                  policy=tight_grid_policy(ps)),
        "grid_late_annual": lambda: run(ModelOptions(time_mode="annual"), "R-CN", ps,
                                        policy=late_grid_policy(ps)),
        "ref_annual": lambda: run(ModelOptions(time_mode="annual"), "REF", ps),
        "rcnpub_annual": lambda: run(ModelOptions(time_mode="annual"), "R-CN-pub", ps),
        "ccnmetro_annual": lambda: run(ModelOptions(time_mode="annual"), "C-CN-metro", ps),
        "a0_annual": lambda: run(ModelOptions(time_mode="annual"), "A0", ps),
        "rcn_pt05_annual": lambda: run(ModelOptions(time_mode="annual", carbon_pass_through=0.5),
                                       "R-CN", ps),
    }
    memo: Dict[str, Result] = {}

    def get(key: str) -> Result:
        if key not in memo:
            res = configs[key]()
            assert res.status == "Optimal", (key, res.status)
            memo[key] = res
        return memo[key]

    return get
