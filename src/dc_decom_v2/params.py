"""Parameter sets of DC-DECOM v2.

A ``ParamSet`` is loaded from one directory in one of two layouts.

v2 layout (``data/parameters_v2/``, ``tests/fixtures/params_dev/``):
    parameters.csv  scalars:  id, group, value, unit, low, high, status, source_id
    regions.csv     regional: id, region, group, value, unit, low, high, status, source_id
    paths.csv       yearly:   id, year, group, value, low, high, unit, status, source_id
    temperature_96h*.csv      region, season, hour, temperature_C, weight_hours
    pv_cf_96h*.csv            region, season, hour, cf
  Extra columns (derivation, notes, n_obs, ...) are allowed and ignored.
  regions.csv, paths.csv and the profiles are optional.
  group  in GROUPS; status in STATUSES ('dev' = test placeholder, never a
  result). For paths, ``low`` / ``high`` hold the low and high variants.

spec layout (``data/parameters_spec/``, Energies tables): global_parameters.csv,
  regions.csv (wide) and temperature_96h.csv as read by ``dc_decom_spec``.
  Every value gets status 'paper' and group 'existing'; the paper reference is
  kept as source_id. Paths are derived as the spec model does (demand anchors,
  carbon interpolation, linear national EF, constant liquid deploy cap), and
  the code-only constants the spec model uses (utilization 0.80, immersion
  pump offset 0.025, CGS fuel 8 yen/kWh thermal) are added.

``ParamSet.compose(existing_from, structural_from)`` keeps every entry of
``structural_from`` whose id does not occur in ``existing_from`` and takes all
of ``existing_from``: runs can combine paper values for the existing
parameters with v2 (or DEV) values for the new structural ones.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

STATUSES = ("paper", "sourced", "derived", "assumption", "dev")
GROUPS = ("existing", "structural", "policy", "path", "sensitivity")
SCALAR_COLUMNS = ("id", "group", "value", "unit", "low", "high", "status", "source_id")
REGION_COLUMNS = ("id", "region", "group", "value", "unit", "low", "high", "status",
                  "source_id")
PATH_COLUMNS = ("id", "year", "group", "value", "low", "high", "unit", "status",
                "source_id")
PROFILE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "temperature_96h": ("region", "season", "hour", "temperature_C", "weight_hours"),
    "pv_cf_96h": ("region", "season", "hour", "cf"),
}
SPEC_REGION_COLUMNS = ("annual_mean_T_C", "base_share_2024", "tariff_yen_per_kWh",
                       "ppa_yen_per_kWh", "re_max_GW", "grid_ef_base")

# Demand anchors of the spec / v1 model (dc_decom_spec.model._national_it_load_TWh),
# Wood Mackenzie path of Energies Section 4.1, TWh.
SPEC_DEMAND_ANCHORS: Dict[int, float] = {
    2024: 19.0, 2026: 27.0, 2028: 37.0, 2030: 47.0, 2032: 57.0,
    2034: 61.0, 2036: 65.0, 2038: 68.0, 2040: 70.0,
}
# Code-only constants of dc_decom_spec.model / pue.py (v1 code values).
SPEC_CODE_CONSTANTS: Tuple[Tuple[str, float, str, str], ...] = (
    ("utilization", 0.80, "fraction", "code-only: SpecReadings.utilization (v1 model.py l.230)"),
    ("pue_immersion_offset", 0.025, "dimensionless",
     "code-only: pue.py immersion C_pump - 0.025 (MHI)"),
    ("cgs_fuel_yen_per_kWh_th", 8.0, "yen/kWh_th",
     "code-only: model.py CGS fuel 8 / eff_elec"),
)

Key = Tuple[str, Optional[str], Optional[int]]
_MISSING = object()


@dataclass(frozen=True)
class Param:
    id: str
    group: str
    value: float
    unit: str = ""
    low: float = math.nan
    high: float = math.nan
    status: str = "paper"
    source_id: str = ""
    region: Optional[str] = None
    year: Optional[int] = None
    origin: str = ""

    @property
    def key(self) -> Key:
        return (self.id, self.region, self.year)


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _s(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)


class ParamSet:
    """Scalars, regional values, yearly paths and 96-hour profiles."""

    def __init__(self, entries: Mapping[Key, Param], profiles: Mapping[str, pd.DataFrame],
                 region_names: Iterable[str], label: str = "") -> None:
        self._entries: Dict[Key, Param] = dict(entries)
        self._profiles: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in profiles.items()}
        self.region_names: Tuple[str, ...] = tuple(region_names)
        self.label = label
        for p in self._entries.values():
            if p.status not in STATUSES:
                raise ValueError(f"{p.id}: status {p.status!r} not in {STATUSES}")
            if p.group not in GROUPS:
                raise ValueError(f"{p.id}: group {p.group!r} not in {GROUPS}")

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, directory: str | Path) -> "ParamSet":
        d = Path(directory)
        if (d / "parameters.csv").exists():
            return cls._load_v2(d)
        if (d / "global_parameters.csv").exists():
            return cls._load_spec(d)
        raise FileNotFoundError(f"no parameters.csv or global_parameters.csv in {d}")

    @classmethod
    def _load_v2(cls, d: Path) -> "ParamSet":
        entries: Dict[Key, Param] = {}
        order: List[str] = []

        def add(p: Param) -> None:
            if p.key in entries:
                raise ValueError(f"duplicate parameter {p.key} in {d}")
            entries[p.key] = p

        df = _read(d / "parameters.csv", SCALAR_COLUMNS)
        for _, row in df.iterrows():
            add(_row_param(row, d.name))
        if (d / "regions.csv").exists():
            df = _read(d / "regions.csv", REGION_COLUMNS)
            for _, row in df.iterrows():
                p = _row_param(row, d.name, region=str(row["region"]))
                add(p)
                if p.region not in order:
                    order.append(str(p.region))
        if (d / "paths.csv").exists():
            df = _read(d / "paths.csv", PATH_COLUMNS)
            for _, row in df.iterrows():
                add(_row_param(row, d.name, year=int(row["year"])))
        profiles: Dict[str, pd.DataFrame] = {}
        for name, cols in PROFILE_COLUMNS.items():
            files = sorted(d.glob(f"{name}*.csv"))
            if len(files) > 1:
                raise ValueError(f"more than one {name}*.csv in {d}: {files}")
            if files:
                profiles[name] = _read(files[0], cols)[list(cols)]
        return cls(entries, profiles, order, label=d.name)

    @classmethod
    def _load_spec(cls, d: Path) -> "ParamSet":
        entries: Dict[Key, Param] = {}
        g = pd.read_csv(d / "global_parameters.csv")
        for _, row in g.iterrows():
            ref = _s(row.get("paper_ref")) or _s(row.get("source"))
            p = Param(id=str(row["parameter"]), group="existing", value=_f(row["value"]),
                      unit=_s(row.get("unit")), status="paper", source_id=ref, origin=d.name)
            entries[p.key] = p
        for pid, value, unit, ref in SPEC_CODE_CONSTANTS:
            p = Param(id=pid, group="existing", value=value, unit=unit, status="paper",
                      source_id=ref, origin=d.name)
            entries[p.key] = p
        reg = pd.read_csv(d / "regions.csv")
        order = [str(r) for r in reg["region"]]
        for _, row in reg.iterrows():
            for col in SPEC_REGION_COLUMNS:
                if col not in reg.columns:
                    continue
                p = Param(id=col, group="existing", value=_f(row[col]), status="paper",
                          source_id=_s(row.get("paper_ref")), region=str(row["region"]),
                          origin=d.name)
                entries[p.key] = p

        def scal(pid: str) -> float:
            return entries[(pid, None, None)].value

        years = range(int(scal("horizon_start")), int(scal("horizon_end")) + 1)
        xs = sorted(SPEC_DEMAND_ANCHORS)
        demand = np.interp(list(years), xs, [SPEC_DEMAND_ANCHORS[x] for x in xs])
        carbon = np.interp(list(years), [2026.0, 2035.0, 2040.0],
                           [scal("P_carbon_2026"), scal("P_carbon_2035"),
                            scal("P_carbon_2040")])
        y0, y1 = scal("horizon_start"), scal("horizon_end")
        ef = [scal("EF_grid_2024") + (scal("EF_grid_2040") - scal("EF_grid_2024"))
              * min(1.0, max(0.0, (t - y0) / (y1 - y0))) for t in years]
        derived = (
            ("demand_TWh", demand, "TWh", "derived: spec demand anchors (Section 4.1)"),
            ("carbon_price", carbon, "yen/t-CO2",
             "derived: P_carbon_2026/2035/2040, 2026 value before 2026"),
            ("ef_national", ef, "kg-CO2/kWh", "derived: EF_grid_2024 -> EF_grid_2040 linear"),
            ("liquid_deploy_cap", [scal("Deploy_max_liquid")] * len(years), "GW/yr/region",
             "derived: Deploy_max_liquid constant (Eq. 17)"),
        )
        for pid, values, unit, ref in derived:
            for t, v in zip(years, values):
                p = Param(id=pid, group="path", value=float(v), unit=unit, status="paper",
                          source_id=ref, year=int(t), origin=d.name)
                entries[p.key] = p
        temps = pd.read_csv(d / "temperature_96h.csv")
        profiles = {"temperature_96h": temps[list(PROFILE_COLUMNS["temperature_96h"])]}
        return cls(entries, profiles, order, label=d.name)

    # --------------------------------------------------------------- compose
    @staticmethod
    def compose(existing_from: "ParamSet", structural_from: "ParamSet") -> "ParamSet":
        ids = {k[0] for k in existing_from._entries}
        entries = {k: v for k, v in structural_from._entries.items() if k[0] not in ids}
        entries.update(existing_from._entries)
        profiles = dict(structural_from._profiles)
        profiles.update(existing_from._profiles)
        order = existing_from.region_names or structural_from.region_names
        label = f"{existing_from.label}+{structural_from.label}"
        return ParamSet(entries, profiles, order, label=label)

    # ---------------------------------------------------------------- access
    def has(self, pid: str) -> bool:
        return any(k[0] == pid for k in self._entries)

    def get(self, pid: str, region: Optional[str] = None,
            year: Optional[int] = None) -> Param:
        return self._entries[(pid, region, year)]

    def scalar(self, pid: str, default=_MISSING) -> float:
        p = self._entries.get((pid, None, None))
        if p is None:
            if default is _MISSING:
                raise KeyError(f"missing scalar parameter {pid!r} in {self.label}")
            return default
        return p.value

    def region(self, pid: str, region: str, default=_MISSING) -> float:
        """Regional value; falls back to the scalar of the same id."""
        p = self._entries.get((pid, region, None)) or self._entries.get((pid, None, None))
        if p is None:
            if default is _MISSING:
                raise KeyError(f"missing regional parameter {pid!r}[{region}] in {self.label}")
            return default
        return p.value

    def has_region(self, pid: str, region: str) -> bool:
        return (pid, region, None) in self._entries

    def path(self, pid: str, variant: str = "value") -> Dict[int, float]:
        if variant not in ("value", "low", "high"):
            raise ValueError("variant must be value, low or high")
        out = {k[2]: float(getattr(p, variant)) for k, p in self._entries.items()
               if k[0] == pid and k[2] is not None}
        if not out:
            raise KeyError(f"missing path {pid!r} in {self.label}")
        return {int(t): out[t] for t in sorted(out)}  # type: ignore[index]

    def has_path(self, pid: str) -> bool:
        return any(k[0] == pid and k[2] is not None for k in self._entries)

    def profile(self, name: str) -> Optional[pd.DataFrame]:
        df = self._profiles.get(name)
        return None if df is None else df.copy()

    def entries(self) -> List[Param]:
        return [self._entries[k] for k in sorted(self._entries, key=_sort_key)]

    @property
    def has_dev(self) -> bool:
        return any(p.status == "dev" for p in self._entries.values())

    # ------------------------------------------------------------- variants
    def with_overrides(self, overrides: Mapping[str, float]) -> "ParamSet":
        """Replace values. Keys: 'id' (scalar), 'id[region]', 'id@year'."""
        entries = dict(self._entries)
        for key, value in overrides.items():
            pid, region, year = _parse_key(key)
            k = (pid, region, year)
            if k not in entries:
                raise KeyError(f"override of unknown parameter {key!r}")
            entries[k] = replace(entries[k], value=float(value))
        return ParamSet(entries, self._profiles, self.region_names, label=self.label)

    def with_additions(self, additions: Mapping[str, float]) -> "ParamSet":
        """Add entries that do not exist yet (keys as in ``with_overrides``,
        also 'id[region]@year'); group 'sensitivity', status 'assumption'."""
        entries = dict(self._entries)
        for key, value in additions.items():
            pid, region, year = _parse_key(key)
            k = (pid, region, year)
            if k in entries:
                raise KeyError(f"addition of existing parameter {key!r}; use an override")
            entries[k] = Param(id=pid, group="sensitivity", value=float(value),
                               status="assumption", source_id="experiment addition",
                               region=region, year=year, origin="additions")
        return ParamSet(entries, self._profiles, self.region_names, label=self.label)

    def optional(self, pid: str, region: Optional[str] = None,
                 year: Optional[int] = None) -> Optional[float]:
        p = self._entries.get((pid, region, year))
        return None if p is None else p.value

    def with_profile(self, name: str, df: pd.DataFrame) -> "ParamSet":
        if name not in PROFILE_COLUMNS:
            raise ValueError(f"unknown profile {name!r}")
        profiles = dict(self._profiles)
        profiles[name] = df[list(PROFILE_COLUMNS[name])].copy()
        return ParamSet(self._entries, profiles, self.region_names, label=self.label)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for p in self.entries():
            h.update(repr((p.id, p.region, p.year, p.group, p.value, p.unit, p.low, p.high,
                           p.status, p.source_id)).encode())
        for name in sorted(self._profiles):
            h.update(name.encode())
            h.update(self._profiles[name].to_csv(index=False).encode())
        h.update(repr(self.region_names).encode())
        return h.hexdigest()


def _sort_key(k: Key):
    return (k[0], k[1] or "", -1 if k[2] is None else k[2])


def _parse_key(key: str) -> Key:
    m = re.fullmatch(r"([^\[\]@]+)(?:\[([^\]]+)\])?(?:@(\d{4}))?", key)
    if not m:
        raise ValueError(f"bad override key {key!r}")
    return (m.group(1), m.group(2), int(m.group(3)) if m.group(3) else None)


def _read(path: Path, required: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def _row_param(row, origin: str, region: Optional[str] = None,
               year: Optional[int] = None) -> Param:
    return Param(id=str(row["id"]), group=str(row["group"]), value=_f(row["value"]),
                 unit=_s(row.get("unit")), low=_f(row.get("low")), high=_f(row.get("high")),
                 status=str(row["status"]), source_id=_s(row.get("source_id")),
                 region=region, year=year, origin=origin)
