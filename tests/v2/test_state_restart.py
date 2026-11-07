"""Start-year / initial-state interface for rolling execution: restarting from
the end-of-2030 state of a full-horizon optimum gives the same 2031-2040 cost
(principle of optimality: the state carries every inter-year coupling)."""
from __future__ import annotations

import pytest

from dc_decom_v2 import ModelOptions, run, state_at

from .conftest import tight_grid_policy


@pytest.mark.parametrize("key,scenario", [("grid_annual", "R-CN"), ("ccn_cert_annual", "C-CN")])
def test_restart_reproduces_tail(solved, ps, key, scenario):
    full = solved(key)
    st = state_at(full, 2030)
    policy = tight_grid_policy(ps) if key == "grid_annual" else None
    opts = ModelOptions(time_mode="annual", certificates=(key == "ccn_cert_annual"))
    tail = run(opts, scenario, ps, policy=policy, start_year=2031, initial_state=st)
    assert tail.status == "Optimal"
    nat = full.tables["national"]
    expected = float(nat.loc[nat["year"] >= 2031, "operator_disc"].sum())
    assert abs(tail.metrics["operator_cost_yen"] - expected) <= 1e-6 * expected
    assert tail.flows["annual"]["year"].min() == 2031


def test_initial_state_required(ps):
    with pytest.raises(ValueError):
        run(ModelOptions(time_mode="annual"), "REF", ps, start_year=2031)
