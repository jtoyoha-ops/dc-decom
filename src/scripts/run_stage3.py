"""DC-DECOM v2 experiment runner: v2 base, bridge, conditions, tornado, carbon sweep, P*,
A/B/C rolling comparison, the four-type factorial and the additional analyses
(config/experiments_v35.yaml: mc composite uncertainty, breakeven, ch03
training / inference ratio, annual vs rep96, carbon_cert carbon price x
certificate rule, emissions Shapley decomposition of the emission reduction,
twobytwo structure x parameters 2x2) and the A/B/C post-hoc sensitivity
(config/abc_posthoc.yaml: abc_posthoc, b_minus_a0).

Usage:
    .venv/bin/python src/scripts/run_stage3.py --part {base,certificates,bridge,
        conditions,tornado,sweep,pstar,abc,interactions,mc,breakeven,ch03,
        annual,carbon_cert,emissions,twobytwo,abc_posthoc,b_minus_a0,all}
        [--check] [--no-cache]

Definitions: config/experiments_v2.yaml, config/abc_rules.yaml,
config/interaction_pairs.yaml, config/experiments_v35.yaml and
config/abc_posthoc.yaml, each written before its runs.
Results go to outputs/stage3/ (abc, interactions, abc_posthoc and b_minus_a0:
outputs/abc/).
Solves are cached in outputs/cache/ (gitignored), so a
part can be resumed after a timeout. --check recomputes the part from the
cache into a temporary directory and compares every file with the committed
one (exit code 1 on any difference).
"""
from __future__ import annotations

import argparse
import filecmp
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from dc_decom_v2.experiments import PART_OUT, PARTS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", required=True, choices=list(PARTS) + ["all"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    parts = list(PARTS) if args.part == "all" else [args.part]
    ok = True
    for part in parts:
        t0 = time.perf_counter()
        if args.check:
            with tempfile.TemporaryDirectory() as tmp:
                res = PARTS[part](Path(tmp), not args.no_cache)
                for name in res.files:
                    ref = PART_OUT[part] / name
                    new = Path(tmp) / name
                    same = ref.exists() and filecmp.cmp(ref, new, shallow=False)
                    ok = ok and same
                    print(f"[check] {part}: {name} {'same' if same else 'DIFFERENT'}")
        else:
            res = PARTS[part](PART_OUT[part], not args.no_cache)
            print(f"[{part}] wrote {len(res.files)} files to {PART_OUT[part]}")
        print(f"[{part}] {time.perf_counter() - t0:.1f} s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
