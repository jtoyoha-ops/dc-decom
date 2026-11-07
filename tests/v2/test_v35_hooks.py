"""Hooks of the additional analyses (config/experiments_v35.yaml): the
non-metro communication / data-transfer cost (zero by default), a non-metro
construction premium and the fixed allocation by a realization (DEV
parameters; not results)."""
from __future__ import annotations

from typing import Dict

import pytest

from dc_decom_v2 import ModelOptions, ParamSet, run
from dc_decom_v2.config import METRO, Realization

ANNUAL = ModelOptions(time_mode="annual")


def _nonmetro(ps: ParamSet):
    return [r for r in ps.region_names if r not in METRO]


def _set(ps: ParamSet, values: Dict[str, float]) -> ParamSet:
    """Override the entries that exist, add the others."""
    have = {k: v for k, v in values.items() if ps.has_region(k.split("[")[0],
                                                             k.split("[")[1].rstrip("]"))}
    new = {k: v for k, v in values.items() if k not in have}
    out = ps.with_overrides(have) if have else ps
    return out.with_additions(new) if new else out


def _comm(ps: ParamSet, rate: float) -> ParamSet:
    return _set(ps, {f"comm_cost_yen_per_kWh[{r}]": rate for r in _nonmetro(ps)})


def _ok(res):
    assert res.status == "Optimal", res.status
    return res


def test_comm_zero_is_inert(ps):
    base = _ok(run(ANNUAL, "R-CN", ps))
    zero = _ok(run(ANNUAL, "R-CN", _comm(ps, 0.0)))
    for key in ("eq1_total_yen", "social_cost_yen"):
        assert zero.metrics[key] == base.metrics[key], key
    assert zero.metrics["comm_cost_yen"] == 0.0


def test_comm_cost_fixed_allocation(ps):
    base = _ok(run(ANNUAL, "C-CN", ps))
    comm = _ok(run(ANNUAL, "C-CN", _comm(ps, 1.0)))
    ry = comm.tables["region_year"]
    nm = ry[ry["region"].isin(_nonmetro(ps))]
    expected = float((nm["disc"] * 1.0 * nm["it_energy_TWh"] * 1e9).sum())
    assert expected > 0.0
    assert comm.metrics["comm_cost_yen"] == pytest.approx(expected, rel=1e-9)
    inc = comm.metrics["social_cost_yen"] - base.metrics["social_cost_yen"]
    assert inc == pytest.approx(expected, rel=1e-9)


def test_comm_cost_operator_equals_objective(ps):
    res = _ok(run(ANNUAL, "R-CN", _comm(ps, 2.0)))
    assert res.metrics["comm_cost_yen"] > 0.0
    assert res.metrics["operator_cost_yen"] == pytest.approx(res.metrics["objective_yen"],
                                                             rel=1e-6)


def test_nonmetro_shell_premium(ps):
    base = _ok(run(ANNUAL, "C-CN", ps))
    nm_regions = _nonmetro(ps)
    res = _ok(run(ANNUAL, "C-CN",
                  _set(ps, {f"shell_premium_yen_per_kW[{r}]": 1e5 for r in nm_regions})))
    ry = res.tables["region_year"]
    nm = ry[ry["region"].isin(nm_regions)]
    added = float((nm["disc"] * 1e5 * nm["build_IT_GW"] / nm["u_cap"] * 1e6).sum())
    assert added > 0.0
    assert res.metrics["shell_premium_yen"] == pytest.approx(
        base.metrics["shell_premium_yen"] + added, rel=1e-9)


def test_rcn_fixed_by_realization(ps):
    base = _ok(run(ANNUAL, "R-CN", ps))
    a = base.flows["annual"]
    a = a[(a["year"] >= 2024) & (a["year"] <= 2040)]
    itr = tuple((str(r), int(t), float(x)) for r, t, x in zip(a["region"], a["year"],
                                                              a["itr_GW"]))
    assert len(itr) == len(ps.region_names) * 17
    fixed = _ok(run(ANNUAL, "R-CN", ps, realization=Realization(itr=itr)))
    assert fixed.metrics["eq1_total_yen"] == pytest.approx(base.metrics["eq1_total_yen"],
                                                           rel=1e-6)
