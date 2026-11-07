"""Two solves of the same configuration give identical flows and metrics."""
from __future__ import annotations

import pytest

from dc_decom_v2 import ModelOptions, run


@pytest.mark.parametrize("mode", ("annual", "rep96"))
def test_two_solves_identical(ps, mode):
    a = run(ModelOptions(time_mode=mode), "R-CN", ps)
    b = run(ModelOptions(time_mode=mode), "R-CN", ps)
    assert a.meta["objective_yen"] == b.meta["objective_yen"]
    for k in ("annual", "slices"):
        assert a.flows[k].equals(b.flows[k]), k
    assert a.metrics == b.metrics
