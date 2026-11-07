"""Config hash, JSON cache, --no-cache and the parallel runner."""
from __future__ import annotations

from dataclasses import replace

from dc_decom_v2 import ModelOptions, run
from dc_decom_v2.runner import DEV_DIR, SPEC_DIR, RunSpec, load_params, run_many


def _spec(**kw) -> RunSpec:
    base = RunSpec(scenario="REF", existing_dir=SPEC_DIR, structural_dir=DEV_DIR,
                   options=ModelOptions(time_mode="annual"), label="a")
    return replace(base, **kw)


def test_hash_stable_and_sensitive():
    s = _spec()
    assert s.config_hash() == _spec().config_hash()
    assert s.config_hash() == _spec(label="other").config_hash()
    assert s.config_hash() != _spec(scenario="C-CN").config_hash()
    assert s.config_hash() != _spec(options=ModelOptions(time_mode="rep96")).config_hash()
    assert s.config_hash() != _spec(overrides=(("discount_rate", 0.04),)).config_hash()
    assert s.config_hash() != _spec(end_year=2035).config_hash()


def test_cache_hit_and_no_cache(tmp_path):
    s = _spec()
    (r1,) = run_many([s], workers=1, cache_dir=tmp_path)
    assert r1.meta["run"]["cache_hit"] is False
    assert (tmp_path / f"{s.config_hash()}.json").exists()
    (r2,) = run_many([s], workers=1, cache_dir=tmp_path)
    assert r2.meta["run"]["cache_hit"] is True
    assert r1.metrics == r2.metrics
    for k in r1.flows:
        assert r1.flows[k].equals(r2.flows[k])
    (r3,) = run_many([s], workers=1, cache_dir=tmp_path, use_cache=False)
    assert r3.meta["run"]["cache_hit"] is False
    assert r3.metrics == r1.metrics


def test_parallel_order_and_equivalence(tmp_path):
    specs = [_spec(scenario="R-CN", label="x"), _spec(scenario="REF", label="y"),
             _spec(scenario="R-CN", label="z"), _spec(scenario="C-CN-metro", label="w")]
    res = run_many(specs, workers=3, cache_dir=tmp_path, log_path=tmp_path / "log.csv")
    assert [r.meta["scenario"] for r in res] == ["R-CN", "REF", "R-CN", "C-CN-metro"]
    assert [r.meta["run"]["label"] for r in res] == ["x", "y", "z", "w"]
    assert res[0].metrics == res[2].metrics
    assert (tmp_path / "log.csv").read_text().count("\n") == 5
    direct = run(ModelOptions(time_mode="annual"), "R-CN", load_params(SPEC_DIR, DEV_DIR))
    assert abs(direct.metrics["eq1_total_yen"] - res[0].metrics["eq1_total_yen"]) \
        <= 1e-12 * direct.metrics["eq1_total_yen"]


def test_overrides_and_no_slices(tmp_path):
    (r,) = run_many([_spec(overrides=(("discount_rate", 0.05),), store_slices=False)],
                    workers=1, cache_dir=tmp_path)
    assert "slices" not in r.flows
    assert abs(r.flows["annual"]["disc"].iloc[-1] - 1.05 ** -16) < 1e-12
