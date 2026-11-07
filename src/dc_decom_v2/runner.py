"""Parallel cached runner of DC-DECOM v2.

A ``RunSpec`` fixes everything a solve depends on. Its config hash (sha256)
covers the canonical spec (without the label), the fingerprint of the composed
parameter set and the source files of ``dc_decom_v2``, so a change of inputs
or code invalidates the cache. Results are cached as JSON in
``outputs/cache/<hash>.json`` (gitignored). ``run_many`` solves the specs that
are not cached in a ``ProcessPoolExecutor`` (at most 8 workers; CBC msg=0,
threads=1) and returns results in input order; cached and fresh results go
through the same JSON round trip, so both are identical objects.

CLI (smoke runs; not results):
    PYTHONPATH=src .venv/bin/python -m dc_decom_v2.runner --smoke [--no-cache]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import model
from .config import BASE_YEAR, END_YEAR, ModelOptions, Paths, PolicySettings, Realization, State
from .params import ParamSet

REPO = Path(__file__).resolve().parents[2]
SPEC_DIR = "data/parameters_spec"
DEV_DIR = "tests/fixtures/params_dev"
DEFAULT_CACHE = REPO / "outputs" / "cache"
MAX_WORKERS = 8
_CODE_FILES = (sorted((REPO / "src" / "dc_decom_v2").glob("*.py"))
               + [REPO / "src" / "dc_decom_spec" / "scenarios.py"])


@dataclass(frozen=True)
class RunSpec:
    scenario: str
    existing_dir: str
    structural_dir: Optional[str] = None
    options: ModelOptions = ModelOptions()
    overrides: Tuple[Tuple[str, float], ...] = ()
    additions: Tuple[Tuple[str, float], ...] = ()
    paths: Optional[Paths] = None
    policy: Optional[PolicySettings] = None
    start_year: int = BASE_YEAR
    end_year: int = END_YEAR
    initial_state: Optional[State] = None
    store_slices: bool = True
    realization: Optional[Realization] = None
    label: str = ""

    def config(self) -> Dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)
                if f.name != "label"}

    def config_hash(self) -> str:
        payload = {"config": self.config(),
                   "params": load_params(self.existing_dir, self.structural_dir).fingerprint(),
                   "code": code_fingerprint()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _jsonable(x: Any) -> Any:
    if is_dataclass(x) and not isinstance(x, type):
        return {f.name: _jsonable(getattr(x, f.name)) for f in fields(x)}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return repr(x)
        return x
    return x


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    h = hashlib.sha256()
    for p in _CODE_FILES:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _abs(d: str) -> Path:
    p = Path(d)
    return p if p.is_absolute() else REPO / p


@lru_cache(maxsize=16)
def load_params(existing_dir: str, structural_dir: Optional[str] = None) -> ParamSet:
    ex = ParamSet.load(_abs(existing_dir))
    if structural_dir is None:
        return ex
    return ParamSet.compose(ex, ParamSet.load(_abs(structural_dir)))


# ----------------------------------------------------------------------
# Result <-> JSON
# ----------------------------------------------------------------------

def result_to_dict(res: model.Result, store_slices: bool = True) -> Dict[str, Any]:
    flows = {k: df.to_dict(orient="list") for k, df in res.flows.items()
             if store_slices or k != "slices"}
    tables = {k: df.to_dict(orient="list") for k, df in res.tables.items()}
    return {"status": res.status, "metrics": res.metrics, "timings": res.timings,
            "n_vars": res.n_vars, "n_cons": res.n_cons, "meta": res.meta,
            "flows": flows, "tables": tables}


def result_from_dict(d: Dict[str, Any]) -> model.Result:
    return model.Result(
        status=d["status"],
        flows={k: pd.DataFrame(v) for k, v in d["flows"].items()},
        metrics=dict(d["metrics"]), timings=dict(d["timings"]),
        n_vars=int(d["n_vars"]), n_cons=int(d["n_cons"]), meta=dict(d["meta"]),
        tables={k: pd.DataFrame(v) for k, v in d["tables"].items()})


def execute(spec: RunSpec) -> Dict[str, Any]:
    """Solve one spec; returns the JSON-ready result dict."""
    ps = load_params(spec.existing_dir, spec.structural_dir)
    if spec.additions:
        ps = ps.with_additions(dict(spec.additions))
    if spec.overrides:
        ps = ps.with_overrides(dict(spec.overrides))
    res = model.run(spec.options, spec.scenario, ps, paths=spec.paths, policy=spec.policy,
                    start_year=spec.start_year, initial_state=spec.initial_state,
                    end_year=spec.end_year, realization=spec.realization)
    return json.loads(json.dumps(result_to_dict(res, spec.store_slices)))


def _execute_timed(spec: RunSpec) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    d = execute(spec)
    return d, time.perf_counter() - t0


def run_many(specs: Sequence[RunSpec], workers: int = 4, cache_dir: Optional[Path] = None,
             use_cache: bool = True, log_path: Optional[Path] = None) -> List[model.Result]:
    """Solve ``specs`` (cached, parallel); results in input order. Each result
    gets ``meta['run']`` = {label, hash, cache_hit, wall_s}."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    hashes = [s.config_hash() for s in specs]
    dicts: Dict[str, Dict[str, Any]] = {}
    hits: Dict[str, bool] = {}
    walls: Dict[str, float] = {}
    todo: List[int] = []
    for i, h in enumerate(hashes):
        if h in dicts or any(hashes[j] == h for j in todo):
            continue
        f = cache / f"{h}.json"
        if use_cache and f.exists():
            dicts[h] = json.loads(f.read_text())["result"]
            hits[h], walls[h] = True, 0.0
        else:
            todo.append(i)
    n_workers = max(1, min(int(workers), MAX_WORKERS, len(todo)))
    if n_workers == 1:
        outs = [_execute_timed(specs[i]) for i in todo]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_execute_timed, specs[i]) for i in todo]
            outs = [f.result() for f in futures]
    for i, (d, wall) in zip(todo, outs):
        h = hashes[i]
        dicts[h], hits[h], walls[h] = d, False, wall
        tmp = cache / f"{h}.json.tmp"
        tmp.write_text(json.dumps({"spec": specs[i].config(), "result": d}))
        os.replace(tmp, cache / f"{h}.json")
    results: List[model.Result] = []
    records: List[Dict[str, Any]] = []
    for s, h in zip(specs, hashes):
        r = result_from_dict(dicts[h])
        r.meta["run"] = {"label": s.label, "hash": h, "cache_hit": hits[h], "wall_s": walls[h]}
        results.append(r)
        records.append({"label": s.label, "scenario": s.scenario,
                        "time_mode": s.options.time_mode, "hash": h, "status": r.status,
                        "cache_hit": hits[h], "n_vars": r.n_vars, "n_cons": r.n_cons,
                        "build_s": r.timings.get("build_s"), "solve_s": r.timings.get("solve_s"),
                        "eval_s": r.timings.get("eval_s"), "wall_s": walls[h]})
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(records[0]) if records else ["label"])
            wr.writeheader()
            wr.writerows(records)
    return results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def smoke_specs(scenarios: Sequence[str], modes: Sequence[str]) -> List[RunSpec]:
    return [RunSpec(scenario=sc, existing_dir=SPEC_DIR, structural_dir=DEV_DIR,
                    options=ModelOptions(time_mode=m), store_slices=False,
                    label=f"smoke:{sc}:{m}")
            for m in modes for sc in scenarios]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true",
                    help="REF, C-CN, R-CN in annual and rep96 with paper + DEV parameters")
    ap.add_argument("--scenarios", nargs="+", default=["REF", "C-CN", "R-CN"])
    ap.add_argument("--modes", nargs="+", default=["annual", "rep96"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-cache", action="store_true", help="recompute and overwrite the cache")
    ap.add_argument("--log", type=Path, default=DEFAULT_CACHE / "logs" / "smoke.csv")
    args = ap.parse_args(argv)
    if not args.smoke:
        ap.error("only --smoke is available from the command line")
    specs = smoke_specs(args.scenarios, args.modes)
    t0 = time.perf_counter()
    results = run_many(specs, workers=args.workers, use_cache=not args.no_cache,
                       log_path=args.log)
    print(f"{'run':<22}{'status':<11}{'n_vars':>8}{'n_cons':>8}{'build_s':>9}{'solve_s':>9}"
          f"{'cache':>7}")
    ok = True
    for s, r in zip(specs, results):
        ok = ok and r.status == "Optimal"
        print(f"{s.label:<22}{r.status:<11}{r.n_vars:>8}{r.n_cons:>8}"
              f"{r.timings['build_s']:>9.2f}{r.timings['solve_s']:>9.2f}"
              f"{str(r.meta['run']['cache_hit']):>7}")
    print(f"wall {time.perf_counter() - t0:.1f} s; log {args.log}; DEV parameters (not results)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
