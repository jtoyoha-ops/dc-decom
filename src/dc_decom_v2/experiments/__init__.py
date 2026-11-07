"""Stage-3 experiments of DC-DECOM v2 (definitions: config/experiments_v2.yaml,
config/abc_rules.yaml, config/interaction_pairs.yaml, config/experiments_v35.yaml,
config/abc_posthoc.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from . import (abc, abc_posthoc, annual, base, breakeven, bridge, carbon_cert, emissions,
               interactions, latency, pstar, sensitivity, twobytwo, uncertainty)
from .common import OUT, PartResult

PARTS: Dict[str, Callable[[Path, bool], PartResult]] = {
    "base": base.run,
    "certificates": sensitivity.run_certificates,
    "bridge": bridge.run_bridge,
    "conditions": bridge.run_conditions,
    "tornado": sensitivity.run_tornado,
    "sweep": sensitivity.run_sweep,
    "pstar": pstar.run,
    "abc": abc.run,
    "interactions": interactions.run,
    "mc": uncertainty.run,
    "breakeven": breakeven.run,
    "ch03": latency.run,
    "annual": annual.run,
    "carbon_cert": carbon_cert.run,
    "emissions": emissions.run,
    "twobytwo": twobytwo.run,
    "abc_posthoc": abc_posthoc.run,
    "b_minus_a0": abc_posthoc.run_b_minus_a0,
}
# Output directory of each part (default outputs/stage3).
PART_OUT: Dict[str, Path] = {name: OUT for name in PARTS}
PART_OUT.update({"abc": abc.OUT, "interactions": abc.OUT, "abc_posthoc": abc_posthoc.OUT,
                 "b_minus_a0": abc_posthoc.OUT})
