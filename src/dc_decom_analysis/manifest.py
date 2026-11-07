"""Provenance manifest of generated outputs.

The manifest has three blocks: ``run`` (timestamps and the git state at run
time; differs between reruns), ``environment`` (Python, packages, CBC) and
``files`` (sha256 of every output file, so reruns can be compared).
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, Optional

REPO = Path(__file__).resolve().parents[2]
PACKAGES = ("pulp", "numpy", "pandas", "scipy", "pyyaml", "openpyxl", "matplotlib")


def _git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                         text=True, check=False)
    return out.stdout.strip()


def git_state() -> Dict[str, object]:
    return {"head": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no"))}


def cbc_info() -> Dict[str, str]:
    import pulp
    path = pulp.PULP_CBC_CMD().path
    out = subprocess.run([path, "-quit"], capture_output=True, text=True, check=False)
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    version = next((ln for ln in lines if ln.startswith("Version")), "")
    build = next((ln for ln in lines if ln.startswith("Build Date")), "")
    rel = Path(path)
    try:
        rel = rel.resolve().relative_to(REPO)
    except ValueError:
        pass
    return {"path": str(rel), "version": version, "build": build}


def package_versions(names: Iterable[str] = PACKAGES) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "not installed"
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hashes(out_dir: Path, exclude: Iterable[str] = ("manifest.json",)) -> Dict[str, str]:
    ex = set(exclude)
    return {str(p.relative_to(out_dir)): sha256(p)
            for p in sorted(out_dir.rglob("*")) if p.is_file() and p.name not in ex}


def write_manifest(out_dir: Path, started: datetime, extra: Optional[dict] = None) -> Path:
    finished = datetime.now(timezone.utc)
    manifest = {
        "run": {
            "started_utc": started.isoformat(timespec="seconds"),
            "finished_utc": finished.isoformat(timespec="seconds"),
            "seconds": round((finished - started).total_seconds(), 1),
            "git": git_state(),
            "command": " ".join([Path(sys.executable).name] + sys.argv),
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(terse=True),
            "machine": platform.machine(),
            "packages": package_versions(),
            "cbc": cbc_info(),
        },
        "files": file_hashes(out_dir),
    }
    if extra:
        manifest["extra"] = extra
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
                    encoding="utf-8")
    return path
