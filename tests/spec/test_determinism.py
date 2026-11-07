"""Two solves of the same configuration give the same objective."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dc_decom_spec import model as spec_model  # noqa: E402
from dc_decom_spec import scenarios as sc  # noqa: E402
from dc_decom_spec.config import PAPER_ONLY_READINGS, SpecFixes, SpecReadings  # noqa: E402
from dc_decom_spec.load_inputs import load_inputs  # noqa: E402


def test_two_solves_identical() -> None:
    regions, p, temps = load_inputs(table9_inputs=True)
    for readings in (SpecReadings(), PAPER_ONLY_READINGS):
        a = spec_model.build_and_solve(sc.R_CN, regions, p, temps, SpecFixes(), readings)
        b = spec_model.build_and_solve(sc.R_CN, regions, p, temps, SpecFixes(), readings)
        assert a.objective_yen == b.objective_yen
        assert a.table == b.table
