"""Automatic MOSFET parameter extraction from I-V curves (n-channel convention).

Methods
-------
Vth, constant current (CC)
    Vg at which Id = I_norm * W/L (I_norm = 100 nA by default). Simple, robust
    to series resistance, works at any Vd -> used for DIBL.
Vth, max-gm linear extrapolation (LE)
    In the linear region Id ~ k (Vg - Vth - Vd/2) Vd. Draw the tangent to
    Id(Vg) at the point of maximum gm and extrapolate to Id = 0:
    Vth = Vg* - Id*/gm_max - Vd/2.
Subthreshold swing (SS)
    Below Vth the current is diffusion-limited, Id ~ exp(Vg / (n kT/q)), so
    SS = dVg / dlog10(Id) = n (kT/q) ln10, about 60 mV/dec * n at 300 K.
    Reported as the minimum SS over a sliding window.
gm max
    Peak of dId/dVg; set by mobility, Cox and series resistance.
Ion / Ioff
    Id at Vg = Vdd and Vg = 0, both at Vd = Vdd.
DIBL
    (Vth_lin - Vth_sat) / (Vd_sat - Vd_lin), in mV/V, from CC Vth.

Points flagged as in compliance are removed before extraction: they are the
instrument's limit, not the device's response.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def local_slope(x, y, window: int = 7) -> np.ndarray:
    """dy/dx from a least-squares line over a centered window of points.

    Less noise-sensitive than a two-point finite difference, which matters for
    gm (a derivative of noisy data).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    half = max(1, window // 2)
    out = np.empty(n)
    for k in range(n):
        lo, hi = max(0, k - half), min(n, k + half + 1)
        xs, ys = x[lo:hi], y[lo:hi]
        xm = xs - xs.mean()
        denom = (xm**2).sum()
        out[k] = (xm * (ys - ys.mean())).sum() / denom if denom > 0 else np.nan
    return out


def transconductance(vg, id_, window: int = 7) -> np.ndarray:
    return local_slope(vg, id_, window)


def vth_constant_current(vg, id_, w: float, l: float, i_norm: float = 100e-9) -> float:
    """CC threshold: Vg where Id first reaches I_norm * W/L (log-interpolated)."""
    vg = np.asarray(vg, dtype=float)
    id_ = np.asarray(id_, dtype=float)
    i_crit = i_norm * w / l
    above = np.nonzero(id_ >= i_crit)[0]
    if above.size == 0 or above[0] == 0:
        return np.nan
    k = above[0]
    lo, hi = id_[k - 1], id_[k]
    if lo <= 0:
        return float(vg[k])
    frac = (np.log10(i_crit) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return float(vg[k - 1] + frac * (vg[k] - vg[k - 1]))


def vth_max_gm(vg, id_, vd: float, window: int = 7) -> tuple[float, float]:
    """Linear-extrapolation threshold at max gm. Returns (Vth, gm_max)."""
    vg = np.asarray(vg, dtype=float)
    id_ = np.asarray(id_, dtype=float)
    gm = transconductance(vg, id_, window)
    if not np.isfinite(gm).any():
        return np.nan, np.nan
    k = int(np.nanargmax(gm))
    vth = vg[k] - id_[k] / gm[k] - vd / 2
    return float(vth), float(gm[k])


def subthreshold_swing(vg, id_, i_max: float, i_min: float = 1e-12, window: int = 5) -> float:
    """Minimum SS (mV/dec) between the noise floor ``i_min`` and ``i_max``."""
    vg = np.asarray(vg, dtype=float)
    id_ = np.asarray(id_, dtype=float)
    mask = (id_ > i_min) & (id_ < i_max)
    if mask.sum() < window:
        return np.nan
    x, y = vg[mask], np.log10(id_[mask])
    best = 0.0
    for k in range(x.size - window + 1):
        slope = np.polyfit(x[k:k + window], y[k:k + window], 1)[0]  # decades per volt
        best = max(best, slope)
    return 1e3 / best if best > 0 else np.nan


def dibl(vth_lin: float, vth_sat: float, vd_lin: float, vd_sat: float) -> float:
    """Drain-induced barrier lowering in mV/V (positive = Vth drops with Vd)."""
    return 1e3 * (vth_lin - vth_sat) / (vd_sat - vd_lin)


def _value_at(x, y, x0: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or x0 < x.min() - 1e-9 or x0 > x.max() + 1e-9:
        return np.nan
    return float(np.interp(x0, x, y))


def extract(curves: pd.DataFrame, w: float, l: float, vdd: float,
            i_norm: float = 100e-9, i_min: float = 1e-12, i_crit: float | None = None) -> dict:
    """Extract the standard parameter set from one device's Id-Vg curves.

    ``curves`` uses the long format from :mod:`ivchar.measure`; the lowest Vd of
    the Id-Vg sweeps is taken as Vd_lin and the highest as Vd_sat. The CC
    threshold current is ``i_norm * W/L`` unless ``i_crit`` (A) is given, which
    suits discrete parts whose W/L is unknown (e.g. 1 mA, the datasheet
    condition for a 2N7000).
    """
    tr = curves[(curves["sweep"] == "idvg") & ~curves["compliance"].astype(bool)]
    vds = np.sort(tr["vd"].unique())
    if vds.size < 2:
        raise ValueError("need Id-Vg sweeps at two drain biases (linear and saturation)")
    vd_lin, vd_sat = float(vds[0]), float(vds[-1])
    lin = tr[tr["vd"] == vd_lin].sort_values("vg")
    sat = tr[tr["vd"] == vd_sat].sort_values("vg")
    if i_crit is None:
        i_crit = i_norm * w / l
    # vth_constant_current scales its current by W/L; pass W/L = 1 with i_crit.
    vth_lin = vth_constant_current(lin["vg"], lin["id"], 1.0, 1.0, i_crit)
    vth_sat = vth_constant_current(sat["vg"], sat["id"], 1.0, 1.0, i_crit)
    vth_gm, gm_lin = vth_max_gm(lin["vg"], lin["id"], vd_lin)
    gm_sat = float(np.nanmax(transconductance(sat["vg"], sat["id"]))) if len(sat) > 2 else np.nan
    ion = _value_at(sat["vg"], sat["id"], vdd)
    ioff = _value_at(sat["vg"], sat["id"], 0.0)

    return {
        "i_crit": i_crit,
        "vd_lin": vd_lin,
        "vd_sat": vd_sat,
        "vth_cc_lin": vth_lin,
        "vth_cc_sat": vth_sat,
        "vth_gm_lin": vth_gm,
        "ss_sat": subthreshold_swing(sat["vg"], sat["id"], i_crit, i_min),
        "ss_lin": subthreshold_swing(lin["vg"], lin["id"], i_crit, i_min),
        "gm_max_lin": gm_lin,
        "gm_max_sat": gm_sat,
        "ion": ion,
        "ioff": ioff,
        "ion_ioff": ion / ioff if ioff and ioff > 0 else np.nan,
        "ig_max": float(np.abs(sat["ig"]).max()) if "ig" in sat and len(sat) else np.nan,
        "dibl": dibl(vth_lin, vth_sat, vd_lin, vd_sat),
        "n_compliance": int(curves["compliance"].astype(bool).sum()),
    }
