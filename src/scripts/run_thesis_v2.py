"""One command that regenerates every committed output of DC-DECOM v2.

Usage:
    .venv/bin/python src/scripts/run_thesis_v2.py --all [--quick] [--no-cache]

--all runs, in order:
  1. src/scripts/run_spec.py                     -> outputs/stage2/ (spec version)
  2. src/scripts/run_stage3.py --part <each>     -> outputs/stage3/, outputs/abc/
  3. outputs/manifest.json (git HEAD, versions, CBC, seconds per step, sha256 of
     every output file)
--quick skips step 1 and runs step 2 with --check from the cache (no output is
rewritten): a check that every committed stage-3 file is reproduced from the
cache. --no-cache re-solves every LP of step 2 (solves are otherwise read from
outputs/cache/, which a fresh clone does not have).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from dc_decom_analysis import manifest as M  # noqa: E402

PY = sys.executable
PARTS = ("base", "certificates", "bridge", "conditions", "tornado", "sweep", "pstar", "abc",
         "interactions", "mc", "breakeven", "ch03", "annual", "carbon_cert", "emissions",
         "twobytwo", "abc_posthoc", "b_minus_a0")
OUT_DIRS = ("stage2", "stage3", "abc")


def _git_state() -> Dict[str, object]:
    """HEAD is the commit the code ran from. The run itself rewrites outputs/, so
    'dirty' is expected; files changed outside outputs/ are listed (should be none)."""
    st = M.git_state()
    status = M._git("status", "--porcelain", "--untracked-files=no")
    changed = [ln.strip().split(maxsplit=1)[1] for ln in status.splitlines() if ln.strip()]
    st["changed_outside_outputs"] = [p for p in changed if not p.startswith("outputs/")]
    st["note"] = ("head = commit of the code that ran; dirty = outputs/ rewritten by this run; "
                  "changed_outside_outputs must be empty")
    return st


def _run(cmd: List[str], seconds: Dict[str, float], key: str) -> None:
    t0 = time.perf_counter()
    print(f"[run_thesis_v2] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)
    seconds[key] = round(time.perf_counter() - t0, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    started = datetime.now(timezone.utc)
    seconds: Dict[str, float] = {}
    if not args.quick:
        _run([PY, "src/scripts/run_spec.py"], seconds, "stage2")
    for part in PARTS:
        cmd = [PY, "src/scripts/run_stage3.py", "--part", part]
        if args.quick:
            cmd.append("--check")
        if args.no_cache:
            cmd.append("--no-cache")
        _run(cmd, seconds, f"stage3:{part}")
    if args.quick:
        print("[run_thesis_v2] quick check passed (manifest not rewritten)")
        return 0
    finished = datetime.now(timezone.utc)
    files = {}
    for d in OUT_DIRS:
        for name, h in M.file_hashes(REPO / "outputs" / d).items():
            files[f"{d}/{name}"] = h
    manifest = {
        "command": " ".join([Path(PY).name] + sys.argv),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": finished.isoformat(timespec="seconds"),
        "seconds_per_step": seconds,
        "git": _git_state(),
        "environment": {"packages": M.package_versions(), "cbc": M.cbc_info()},
        "files_sha256": files,
    }
    path = REPO / "outputs" / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[run_thesis_v2] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
