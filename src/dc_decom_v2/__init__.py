"""DC-DECOM v2: the model of the doctoral thesis.

Annual and 96-representative-hour modes in one LP builder, BESS with state of
charge and a daily cycle, N+1 backup by CGS and emergency generators,
endogenous siting, a non-metro grid connection constraint with reinforcement
cost, a stepped PPA supply curve with annual (SHK market-based) or hourly
matching, regional or national grid emission factors, an operator objective
with separately evaluated social resource cost, Eq. 1 total, fiscal burden
and CO2, a start-year / initial-state interface for rolling execution and a
parallel cached runner. ``dc_decom`` (the v1 model of Energies 2026, E1-E8
fixed) and ``dc_decom_spec`` (spec version) are separate packages.

Modules: config, params, timeslices, pue, scenarios, model, accounting, runner,
rolling (A/B/C rolling execution).
"""
from .config import ModelOptions, Paths, PolicySettings, Realization, State
from .model import Handle, Result, build, run, solve, state_at
from .params import ParamSet
from .scenarios import SCENARIOS, Scenario

__version__ = "2.0.0"
__author__ = "Jin Toyohara"

__all__ = ["ModelOptions", "Paths", "PolicySettings", "Realization", "State", "Handle", "Result", "build",
           "run", "solve", "state_at", "ParamSet", "SCENARIOS", "Scenario"]
