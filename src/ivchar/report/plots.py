"""Matplotlib figures for the characterization report.

Colors: corners use fixed categorical slots (color follows the corner, never
its rank); temperatures are ordered, so they use one sequential blue ramp.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
CORNER_COLORS = {"TT": "#2a78d6", "FF": "#eb6834", "SS": "#1baf7a", "FS": "#eda100", "SF": "#e87ba4"}
EXTRA_COLORS = ["#6250d6", "#008300", "#e34948"]  # remaining categorical slots, for other labels
TEMP_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#104281", "#0d366b"]
CRITICAL = "#d03b3b"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "legend.frameon": False,
        "legend.fontsize": 8, "font.family": "DejaVu Sans",
    })


def _corners(df) -> list[str]:
    """Corners present in ``df``: standard corners in fixed order, then any others."""
    present = list(dict.fromkeys(df["corner"]))
    return [c for c in CORNER_COLORS if c in present] + sorted(c for c in present if c not in CORNER_COLORS)


def _color(corner, df) -> str:
    if corner in CORNER_COLORS:
        return CORNER_COLORS[corner]
    extras = sorted(c for c in set(df["corner"]) if c not in CORNER_COLORS)
    k = extras.index(corner)
    return EXTRA_COLORS[k] if k < len(EXTRA_COLORS) else TEXT_2


def _temp_colors(temps):
    temps = sorted(temps)
    if len(temps) == 1:
        return {temps[0]: TEMP_RAMP[1]}
    idx = np.linspace(0, len(TEMP_RAMP) - 1, len(temps)).round().astype(int)
    return {t: TEMP_RAMP[i] for t, i in zip(temps, idx)}


def _sim_note(fig, simulated: bool):
    if simulated:
        fig.text(0.995, 0.01, "SIMULATED DATA", ha="right", va="bottom", fontsize=7, color=TEXT_2)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _median_die(params: pd.DataFrame, corner: str, temp: float) -> str:
    rows = params[(params.corner == corner) & np.isclose(params.temp_c, temp) & ~params.get("outlier", False)]
    if rows.empty:
        rows = params[(params.corner == corner) & np.isclose(params.temp_c, temp)]
    k = (rows["vth_cc_sat"] - rows["vth_cc_sat"].median()).abs().idxmin()
    return rows.loc[k, "die"]


def _curve(curves, corner, die, temp, sweep):
    """One device's sweep, without points taken in compliance (instrument limit, not data)."""
    return curves[(curves.corner == corner) & (curves.die == die) & np.isclose(curves.temp_c, temp)
                  & (curves.sweep == sweep) & ~curves["compliance"].astype(bool)]


def _current_unit(curves) -> tuple[float, str]:
    """Scale and label for linear current axes: mA for power devices, µA otherwise."""
    peak = curves.loc[~curves["compliance"].astype(bool), "id"].abs().max()
    return (1e3, "mA") if peak > 2e-3 else (1e6, "µA")


def _legend(ax, **kw):
    """A legend only when there is more than one series."""
    if len(ax.get_legend_handles_labels()[1]) > 1:
        ax.legend(**kw)


def _compliance_note(ax, curves):
    if curves["compliance"].astype(bool).any():
        ax.text(0.02, 0.98, "points in compliance omitted", transform=ax.transAxes, ha="left",
                va="top", fontsize=7, color=TEXT_2)


def plot_idvg_corners(curves, params, temp, out: Path, simulated=True) -> Path:
    """Transfer curves of the median die per corner, log and linear panels."""
    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(9, 3.6))
    scale, unit = _current_unit(curves)
    vd_sat = curves.loc[curves.sweep == "idvg", "vd"].max()
    for corner in _corners(params):
        die = _median_die(params, corner, temp)
        d = _curve(curves, corner, die, temp, "idvg")
        d = d[np.isclose(d.vd, vd_sat)].sort_values("vg")
        ax_log.semilogy(d.vg, np.clip(d.id, 1e-15, None), color=_color(corner, params), label=corner)
        ax_lin.plot(d.vg, d.id * scale, color=_color(corner, params), label=corner)
    ax_log.set(title=f"Id-Vg, Vd = {vd_sat:g} V, {temp:g} °C (log)", xlabel="Vg (V)", ylabel="Id (A)")
    ax_lin.set(title="Id-Vg (linear)", xlabel="Vg (V)", ylabel=f"Id ({unit})")
    _legend(ax_lin, title="corner", loc="upper left", bbox_to_anchor=(0, 0.93))
    _compliance_note(ax_lin, curves)
    _sim_note(fig, simulated)
    return _save(fig, out)


def plot_idvg_temperature(curves, params, corner, out: Path, simulated=True) -> Path:
    """One die across temperature: Ioff rises, SS degrades, Ion drops."""
    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(9, 3.6))
    scale, unit = _current_unit(curves)
    temps = sorted(params.temp_c.unique())
    colors = _temp_colors(temps)
    die = _median_die(params, corner, 25.0 if 25.0 in temps else temps[len(temps) // 2])
    vd_sat = curves.loc[curves.sweep == "idvg", "vd"].max()
    for t in temps:
        d = _curve(curves, corner, die, t, "idvg")
        d = d[np.isclose(d.vd, vd_sat)].sort_values("vg")
        ax_log.semilogy(d.vg, np.clip(d.id, 1e-15, None), color=colors[t], label=f"{t:g} °C")
        ax_lin.plot(d.vg, d.id * scale, color=colors[t], label=f"{t:g} °C")
    ax_log.set(title=f"Id-Vg vs temperature, {corner} {die} (log)", xlabel="Vg (V)", ylabel="Id (A)")
    ax_lin.set(title="Id-Vg vs temperature (linear)", xlabel="Vg (V)", ylabel=f"Id ({unit})")
    _legend(ax_lin, loc="upper left", bbox_to_anchor=(0, 0.93))
    _compliance_note(ax_lin, curves)
    _sim_note(fig, simulated)
    return _save(fig, out)


def plot_idvd(curves, params, corner, temp, out: Path, simulated=True) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    scale, unit = _current_unit(curves)
    die = _median_die(params, corner, temp)
    d = _curve(curves, corner, die, temp, "idvd")
    vgs = sorted(d.vg.unique())
    ramp = _temp_colors(vgs)  # ordered family -> same sequential ramp
    for vg in vgs:
        s = d[np.isclose(d.vg, vg)].sort_values("vd")
        ax.plot(s.vd, s.id * scale, color=ramp[vg])
        ax.annotate(f"Vg={vg:g} V", (s.vd.iloc[-1], s.id.iloc[-1] * scale), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7, color=TEXT_2)
    ax.set(title=f"Id-Vd, {corner} {die}, {temp:g} °C", xlabel="Vd (V)", ylabel=f"Id ({unit})")
    _compliance_note(ax, curves)
    ax.set_xlim(right=d.vd.max() * 1.25)
    _sim_note(fig, simulated)
    return _save(fig, out)


def plot_param_distribution(params, col, label, temp, out: Path, scale=1.0, log=False,
                            simulated=True) -> Path:
    """Strip + box per corner; flagged outliers ringed in the critical color."""
    rows = params[np.isclose(params.temp_c, temp)]
    corners = _corners(rows)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    rng = np.random.default_rng(0)
    for k, corner in enumerate(corners):
        r = rows[rows.corner == corner]
        y = r[col].to_numpy(dtype=float) * scale
        ax.boxplot(y[np.isfinite(y)], positions=[k], widths=0.5, showfliers=False,
                   medianprops={"color": TEXT}, boxprops={"color": TEXT_2},
                   whiskerprops={"color": TEXT_2}, capprops={"color": TEXT_2})
        x = k + rng.uniform(-0.15, 0.15, len(y))
        ax.scatter(x, y, s=22, color=_color(corner, params), edgecolor=SURFACE, linewidth=1.2, zorder=3)
        if "outlier" in r:
            o = r["outlier"].to_numpy()
            ax.scatter(x[o], y[o], s=90, facecolor="none", edgecolor=CRITICAL, linewidth=1.5, zorder=4)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(corners)), corners)
    ax.set(title=f"{label} by corner, {temp:g} °C", ylabel=label)
    ax.grid(axis="x", visible=False)
    _sim_note(fig, simulated)
    return _save(fig, out)


def plot_param_vs_temp(params, col, label, out: Path, scale=1.0, simulated=True) -> Path:
    clean = params[~params.get("outlier", False)]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for corner in _corners(clean):
        g = clean[clean.corner == corner].groupby("temp_c")[col]
        m, s = g.median() * scale, g.std() * scale
        ax.errorbar(m.index, m.values, yerr=s.values, color=_color(corner, params), marker="o",
                    markersize=5, markeredgecolor=SURFACE, capsize=3, label=corner)
    ax.set(title=f"{label} vs temperature (median ± 1σ)", xlabel="Temperature (°C)", ylabel=label)
    _legend(ax, title="corner")
    _sim_note(fig, simulated)
    return _save(fig, out)


def plot_ion_ioff(params, temp, out: Path, simulated=True) -> Path:
    rows = params[np.isclose(params.temp_c, temp)]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for corner in _corners(rows):
        r = rows[rows.corner == corner]
        ax.scatter(r.ioff, r.ion * 1e6, s=22, color=_color(corner, params), edgecolor=SURFACE,
                   linewidth=1.2, label=corner, zorder=3)
    if "outlier" in rows:
        o = rows[rows.outlier]
        ax.scatter(o.ioff, o.ion * 1e6, s=90, facecolor="none", edgecolor=CRITICAL, linewidth=1.5,
                   label="flagged outlier", zorder=4)
    ax.set_xscale("log")
    ax.set(title=f"Ion vs Ioff, {temp:g} °C", xlabel="Ioff (A)", ylabel="Ion (µA)")
    ax.legend()
    _sim_note(fig, simulated)
    return _save(fig, out)


def plot_wafer_map(params, col, label, corner, temp, out: Path, scale=1.0, simulated=True) -> Path:
    rows = params[(params.corner == corner) & np.isclose(params.temp_c, temp)]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
    v = rows[col] * scale
    clean = v[~rows.get("outlier", pd.Series(False, index=rows.index))]
    sc = ax.scatter(rows.x, rows.y, c=v, cmap=cmap, s=420, marker="s", edgecolor=SURFACE, linewidth=2,
                    vmin=clean.min(), vmax=clean.max())
    if "outlier" in rows:
        o = rows[rows.outlier]
        ax.scatter(o.x, o.y, s=420, marker="s", facecolor="none", edgecolor=CRITICAL, linewidth=2)
        for _, r in o.iterrows():
            ax.annotate("!", (r.x, r.y), ha="center", va="center", fontsize=10, color=SURFACE,
                        fontweight="bold")
    r_max = np.hypot(rows.x, rows.y).max() + 0.9
    ax.add_patch(plt.Circle((0, 0), r_max, fill=False, color=GRID, linewidth=1.5))
    ax.set_aspect("equal")
    ax.set_xlim(-r_max - 0.3, r_max + 0.3)
    ax.set_ylim(-r_max - 0.3, r_max + 0.3)
    ax.axis("off")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(label, color=TEXT_2, fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title(f"Wafer map: {label}, {corner}, {temp:g} °C")
    _sim_note(fig, simulated)
    return _save(fig, out)


def make_all(curves, params, out_dir: Path, simulated=True) -> dict[str, Path]:
    """Render the standard figure set; returns {key: png path}."""
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    temps = sorted(params.temp_c.unique())
    t_ref = 25.0 if any(np.isclose(temps, 25.0)) else temps[len(temps) // 2]
    corner_ref = "TT" if "TT" in set(params.corner) else params.corner.iloc[0]
    figs = {
        "idvg_corners": plot_idvg_corners(curves, params, t_ref, out_dir / "idvg_corners.png", simulated),
        "idvd": plot_idvd(curves, params, corner_ref, t_ref, out_dir / "idvd.png", simulated),
        "vth_dist": plot_param_distribution(params, "vth_cc_sat", "Vth,sat CC (mV)", t_ref,
                                            out_dir / "vth_distribution.png", 1e3, simulated=simulated),
        "ion_ioff": plot_ion_ioff(params, t_ref, out_dir / "ion_ioff.png", simulated),
        "wafer_vth": plot_wafer_map(params, "vth_cc_sat", "Vth,sat (mV)", corner_ref, t_ref,
                                    out_dir / "wafer_map_vth.png", 1e3, simulated),
    }
    if len(temps) > 1:
        figs["idvg_temp"] = plot_idvg_temperature(curves, params, corner_ref, out_dir / "idvg_temperature.png",
                                                  simulated)
        figs["vth_temp"] = plot_param_vs_temp(params, "vth_cc_sat", "Vth,sat CC (mV)",
                                              out_dir / "vth_vs_temperature.png", 1e3, simulated)
        figs["ss_temp"] = plot_param_vs_temp(params, "ss_sat", "SS (mV/dec)",
                                             out_dir / "ss_vs_temperature.png", 1.0, simulated)
    return figs
