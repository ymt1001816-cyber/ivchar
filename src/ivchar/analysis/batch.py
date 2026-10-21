"""Batch characterization across dies, temperatures and process corners.

``run_batch`` measures every (corner, die, temperature) combination on the
simulated backend and extracts parameters; ``analyze`` adds statistics,
robust outlier flags and spec-based yield. ``analyze`` also works on curves
loaded from CSV (real measurements or external SPICE/TCAD runs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ivchar.device import PRESETS, make_device
from ivchar.extraction import extract
from ivchar.instruments import SimulatedSMU
from ivchar.measure import standard_sweeps

KEY_PARAMS = ["vth_cc_lin", "vth_cc_sat", "vth_gm_lin", "ss_sat", "gm_max_lin",
              "ion", "ioff", "ion_ioff", "dibl", "ig_max"]
# Leakage currents are log-normally distributed: do statistics on log10.
LOG_PARAMS = {"ioff", "ion_ioff", "ig_max"}
HIGH_SIDE_ONLY = {"ioff", "ig_max", "ss_sat", "dibl"}

# Room-temperature (25 C) wafer-sort style limits: param -> (min, max).
DEFAULT_SPECS = {
    "planar": {
        "vth_cc_sat": (0.20, 0.40), "ss_sat": (None, 90.0), "ion": (5.0e-4, None),
        "ioff": (None, 5e-9), "dibl": (None, 120.0), "ig_max": (None, 1e-11),
    },
    "finfet": {
        "vth_cc_sat": (0.19, 0.38), "ss_sat": (None, 75.0), "ion": (4.0e-4, None),
        "ioff": (None, 2e-9), "dibl": (None, 70.0), "ig_max": (None, 1e-11),
    },
}


@dataclass
class BatchConfig:
    preset: str = "planar"
    n_dies: int = 25
    temps_c: tuple[float, ...] = (-40.0, 25.0, 125.0)
    corners: tuple[str, ...] = ("TT", "FF", "SS")
    seed: int = 0
    sigma_vth: float = 0.010  # random die-to-die Vth variation (V)
    sigma_mu: float = 0.03  # relative mobility variation
    radial_vth: float = 0.015  # systematic center-to-edge Vth shift (V)
    defect_rate: float = 0.04  # fraction of dies with an injected defect
    spec: dict = field(default_factory=dict)  # empty -> DEFAULT_SPECS[preset]

    def resolved_spec(self) -> dict:
        return self.spec or DEFAULT_SPECS.get(self.preset, {})


def wafer_dies(n: int) -> pd.DataFrame:
    """``n`` die positions on a square grid, closest to the wafer center first.

    Coordinates are normalized so the outermost die sits at radius 1.
    """
    side = int(np.ceil(np.sqrt(n * 4 / np.pi))) + 2
    side += 1 - side % 2  # odd, so one die sits at the center
    coords = np.arange(side) - (side - 1) / 2
    xx, yy = np.meshgrid(coords, coords)
    pts = sorted(zip(xx.ravel(), yy.ravel()), key=lambda p: (np.hypot(*p), p[1], p[0]))[:n]
    xy = np.array(pts)
    r_max = np.hypot(xy[:, 0], xy[:, 1]).max() or 1.0
    return pd.DataFrame({
        "die": [f"D{k:03d}" for k in range(n)],
        "x": xy[:, 0].astype(int),
        "y": xy[:, 1].astype(int),
        "radius": np.hypot(xy[:, 0], xy[:, 1]) / r_max,
    })


def _die_variation(cfg: BatchConfig, rng: np.random.Generator, radius: float) -> dict:
    v = {
        "dvth": cfg.radial_vth * radius**2 + rng.normal(0, cfg.sigma_vth),
        "mu_scale": 1 + rng.normal(0, cfg.sigma_mu),
        "gate_leak_scale": 1.0,
        "injected_defect": "",
    }
    if rng.random() < cfg.defect_rate:
        kind = rng.choice(["gate_leak", "vth_shift"])
        v["injected_defect"] = str(kind)
        if kind == "gate_leak":
            v["gate_leak_scale"] = 1e4  # e.g. oxide damage / pinhole
        else:
            v["dvth"] += 0.15  # e.g. implant or charge-trapping excursion
    return v


def run_batch(cfg: BatchConfig, progress: Callable[[int, int], None] | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure all devices on the simulated backend. Returns (params, curves)."""
    base = PRESETS[cfg.preset]
    dies = wafer_dies(cfg.n_dies)
    total = len(cfg.corners) * len(dies) * len(cfg.temps_c)
    rows, curves, done = [], [], 0
    for ci, corner in enumerate(cfg.corners):
        for di, die in dies.iterrows():
            # One generator per physical die: the same die keeps its variation
            # (and any defect) at every temperature.
            rng = np.random.default_rng([cfg.seed, ci, di])
            var = _die_variation(cfg, rng, die["radius"])
            for temp in cfg.temps_c:
                dev = make_device(cfg.preset, corner, temp, var["dvth"], var["mu_scale"],
                                  var["gate_leak_scale"])
                smu = SimulatedSMU(dev, seed=int(rng.integers(2**31)))
                df = standard_sweeps(smu, base.vdd)
                meta = {"corner": corner, "die": die["die"], "x": die["x"], "y": die["y"],
                        "temp_c": float(temp)}
                params = extract(df, base.w, base.l, base.vdd)
                rows.append({**meta, **params, "injected_defect": var["injected_defect"]})
                curves.append(df.assign(**meta))
                done += 1
                if progress:
                    progress(done, total)
    params = pd.DataFrame(rows)
    curves_df = pd.concat(curves, ignore_index=True)
    params.attrs = curves_df.attrs = {"simulated": True, "config": asdict(cfg)}
    return params, curves_df


CURVE_DEFAULTS = {"corner": "NA", "die": "D000", "temp_c": 25.0, "x": 0, "y": 0,
                  "compliance": False, "ig": 0.0}


def normalize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Fill optional columns of an external curve table with defaults."""
    curves = curves.copy()
    for col, default in CURVE_DEFAULTS.items():
        if col not in curves:
            curves[col] = default
    curves["compliance"] = curves["compliance"].astype(bool)
    return curves


def extract_from_curves(curves: pd.DataFrame, w: float, l: float, vdd: float) -> pd.DataFrame:
    """Extract parameters for every device in a long-format curve table.

    Devices are identified by the columns corner/die/temp_c (missing columns are
    filled with defaults), so measured or SPICE-generated CSVs can be analyzed
    with the same pipeline as simulated batches.
    """
    curves = normalize_curves(curves)
    rows = []
    for (corner, die, temp), df in curves.groupby(["corner", "die", "temp_c"], sort=False):
        rows.append({"corner": corner, "die": die, "x": df["x"].iloc[0], "y": df["y"].iloc[0],
                     "temp_c": float(temp), **extract(df, w, l, vdd), "injected_defect": ""})
    return pd.DataFrame(rows)


def _stat_view(params: pd.DataFrame, col: str) -> pd.Series:
    x = params[col].astype(float)
    return np.log10(x.where(x > 0)) if col in LOG_PARAMS else x


def summarize(params: pd.DataFrame, cols=KEY_PARAMS, by=("corner", "temp_c")) -> pd.DataFrame:
    """Per-group count/mean/std/median/min/max. Leakage columns are in log10."""
    view = params[list(by)].copy()
    for col in cols:
        if col in params:
            name = f"log10_{col}" if col in LOG_PARAMS else col
            view[name] = _stat_view(params, col)
    return view.groupby(list(by)).agg(["count", "mean", "std", "median", "min", "max"])


def flag_outliers(params: pd.DataFrame, cols=KEY_PARAMS, by=("corner", "temp_c"),
                  threshold: float = 3.5) -> pd.DataFrame:
    """Robust z-score outliers within each group.

    z = 0.6745 * (x - median) / MAD. Median and MAD are not dragged by the
    outliers themselves, unlike mean and standard deviation. Leakage, SS and
    DIBL are flagged only on the high side: a die that leaks *less* is not a
    problem.
    """
    out = params.copy()
    reasons = [[] for _ in range(len(out))]
    keys = [out[b] for b in by]
    for col in cols:
        if col not in out:
            continue
        x = _stat_view(out, col)
        med = x.groupby(keys).transform("median")
        mad = (x - med).abs().groupby(keys).transform("median")
        z = 0.6745 * (x - med) / mad.replace(0, np.nan)
        out[f"z_{col}"] = z
        hit = z > threshold if col in HIGH_SIDE_ONLY else z.abs() > threshold
        for k in np.nonzero(hit.to_numpy())[0]:
            reasons[k].append(f"{col} (z={z.iloc[k]:+.1f})")
    out["outlier_reason"] = [", ".join(r) for r in reasons]
    out["outlier"] = out["outlier_reason"] != ""
    return out


def apply_spec(params: pd.DataFrame, spec: dict, temp_c: float = 25.0) -> pd.DataFrame:
    """Pass/fail against spec limits for the rows at ``temp_c``."""
    rows = params[np.isclose(params["temp_c"], temp_c)].copy()
    reasons = [[] for _ in range(len(rows))]
    for col, (lo, hi) in spec.items():
        x = rows[col]
        bad = x.isna()
        if lo is not None:
            bad |= x < lo
        if hi is not None:
            bad |= x > hi
        rows[f"pass_{col}"] = ~bad
        for k in np.nonzero(bad.to_numpy())[0]:
            reasons[k].append(col)
    rows["fail_reason"] = [", ".join(r) for r in reasons]
    rows["pass"] = rows["fail_reason"] == ""
    return rows


def yield_table(spec_rows: pd.DataFrame) -> pd.DataFrame:
    g = spec_rows.groupby("corner")["pass"]
    return pd.DataFrame({"dies": g.size(), "pass": g.sum(), "yield_pct": 100 * g.mean()})


@dataclass
class Analysis:
    params: pd.DataFrame  # with outlier flags
    summary: pd.DataFrame
    spec_rows: pd.DataFrame
    yields: pd.DataFrame
    spec: dict
    spec_temp_c: float


def analyze(params: pd.DataFrame, spec: dict, spec_temp_c: float = 25.0) -> Analysis:
    flagged = flag_outliers(params)
    spec_rows = apply_spec(flagged, spec, spec_temp_c) if spec else flagged.iloc[0:0]
    return Analysis(
        params=flagged,
        summary=summarize(params),
        spec_rows=spec_rows,
        yields=yield_table(spec_rows) if len(spec_rows) else pd.DataFrame(),
        spec=spec,
        spec_temp_c=spec_temp_c,
    )
