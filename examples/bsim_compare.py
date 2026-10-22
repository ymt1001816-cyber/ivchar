"""Compare ivchar's extraction against Cadence gpdk045 BSIM models (SIMULATED).

    python examples/bsim_compare.py <dir with <corner>_T<temp>/psf/ folders> [--out docs/bsim]

The folders come from running ``ivchar spice netlist`` output with Spectre
(see README). For each corner and temperature this imports the curves,
runs the normal extraction, and compares it with the BSIM model's own
threshold voltage (operating-point ``vth``) at the same drain bias.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ivchar.analysis import extract_from_curves
from ivchar.spice import import_results

W, L, VDD, VD_LIN = 1e-6, 45e-9, 1.0, 0.05


def load(root: Path) -> pd.DataFrame:
    frames = []
    for d in sorted(root.iterdir()):
        m = re.fullmatch(r"(\w+)_T(-?\d+)", d.name)
        if not m or not (d / "psf").is_dir():
            continue
        corner, temp = m.group(1).upper(), float(m.group(2))
        frames.append(import_results(d / "psf", VD_LIN, VDD, die="g45n1svt", corner=corner, temp=temp))
    return pd.concat(frames, ignore_index=True)


def bsim_reference(curves: pd.DataFrame) -> pd.DataFrame:
    """Model vth at Vd_lin and Vdd (taken at the CC-threshold gate bias region; BSIM vth
    depends on Vds/Vbs, not Vgs) and the DIBL implied by it."""
    tr = curves[curves.sweep == "idvg"]
    rows = []
    for (corner, temp), g in tr.groupby(["corner", "temp_c"]):
        v_lin = g[np.isclose(g.vd, VD_LIN)].bsim_vth.median()
        v_sat = g[np.isclose(g.vd, VDD)].bsim_vth.median()
        rows.append({"corner": corner, "temp_c": temp, "bsim_vth_lin": v_lin, "bsim_vth_sat": v_sat,
                     "bsim_dibl": 1e3 * (v_lin - v_sat) / (VDD - VD_LIN)})
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, path: Path) -> None:
    from ivchar.report import plots

    plots._style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    x = table.bsim_vth_lin * 1e3
    lo, hi = min(x.min(), table.vth_cc_lin.min() * 1e3) - 20, max(x.max(), table.vth_gm_lin.max() * 1e3) + 20
    ax.plot([lo, hi], [lo, hi], color=plots.TEXT_2, linewidth=1, label="y = x")
    ax.scatter(x, table.vth_gm_lin * 1e3, s=36, color=plots.CORNER_COLORS["TT"], edgecolor=plots.SURFACE,
               linewidth=1.5, label="max-gm LE", zorder=3)
    ax.scatter(x, table.vth_cc_lin * 1e3, s=36, color=plots.CORNER_COLORS["FF"], edgecolor=plots.SURFACE,
               linewidth=1.5, label="constant current", zorder=3)
    for corner in ("SS", "TT", "FF"):
        r = table[(table.corner == corner) & (table.temp_c == 25)]
        ax.annotate(corner, (r.bsim_vth_lin.iloc[0] * 1e3, r.vth_gm_lin.iloc[0] * 1e3), xytext=(-6, 8),
                    textcoords="offset points", fontsize=8, color=plots.TEXT_2)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="BSIM model vth at Vd = 50 mV (mV)",
           ylabel="Extracted Vth (mV)", title="ivchar vs gpdk045 BSIM, 3 corners × 3 temps")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.text(0.995, 0.01, "SIMULATED (Spectre, gpdk045)", ha="right", va="bottom", fontsize=7, color=plots.TEXT_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/bsim"))
    args = ap.parse_args()

    curves = load(args.root)
    ext = extract_from_curves(curves, W, L, VDD)
    ref = bsim_reference(curves)
    cmp_ = ext.merge(ref, on=["corner", "temp_c"])
    cmp_["cc_minus_bsim_mV"] = 1e3 * (cmp_.vth_cc_lin - cmp_.bsim_vth_lin)
    cmp_["le_minus_bsim_mV"] = 1e3 * (cmp_.vth_gm_lin - cmp_.bsim_vth_lin)
    cols = ["corner", "temp_c", "bsim_vth_lin", "vth_cc_lin", "vth_gm_lin", "cc_minus_bsim_mV",
            "le_minus_bsim_mV", "bsim_dibl", "dibl", "ss_sat", "ion", "ioff"]
    order = {"SS": 0, "TT": 1, "FF": 2}
    table = cmp_[cols].sort_values(["corner", "temp_c"], key=lambda s: s.map(order) if s.name == "corner" else s)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "bsim_comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(table.to_string(index=False))

    plot(table, args.out / "bsim_vth.png")
    for corner in ("SS", "FF"):
        for col in ("bsim_vth_lin", "vth_cc_lin", "vth_gm_lin"):
            a = table[(table.corner == corner) & (table.temp_c == 25)][col].iloc[0]
            b = table[(table.corner == "TT") & (table.temp_c == 25)][col].iloc[0]
            print(f"{corner} - TT at 25 C, {col:13s} {1e3 * (a - b):+6.1f} mV")

    tt = table[table.corner == "TT"].set_index("temp_c")
    t0, t1 = tt.index.min(), tt.index.max()
    for col in ["bsim_vth_lin", "vth_cc_lin", "vth_gm_lin"]:
        print(f"dVth/dT {col:13s} {1e3 * (tt.loc[t1, col] - tt.loc[t0, col]) / (t1 - t0):6.2f} mV/K")


if __name__ == "__main__":
    main()
