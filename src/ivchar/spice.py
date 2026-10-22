"""Reference I-V curves from an industry compact model (BSIM) via Cadence Spectre.

Workflow (the simulator runs on a machine with Spectre and a PDK; ivchar
itself does not need to be installed there):

1. ``ivchar spice netlist``  writes a Spectre netlist with the same sweeps as
   :func:`ivchar.measure.standard_sweeps`, plus a ``run.sh``.
2. Copy the folder to the EDA machine and run ``sh run.sh`` (Spectre writes
   PSF-ASCII results, one file per analysis).
3. ``ivchar spice import`` converts the results into ivchar's curve CSV, and
   ``ivchar analyze`` extracts parameters from them.

The netlist also saves the model's own threshold voltage (BSIM ``vth``
operating-point parameter) so extracted Vth can be compared with it.
Data produced this way is SIMULATED (BSIM model), not measured.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import numpy as np
import pandas as pd

NETLIST_HEADER = """// ivchar reference sweeps: {model}, W={w:g} L={l:g}, T={temp:g} C (SIMULATED, BSIM)
simulator lang=spectre
include "{model_file}"{section}
parameters vgs=0 vds={vd_lin:g}
VG (g 0) vsource dc=vgs
VD (d 0) vsource dc=vds
M1 (d g 0 0) {model} w={w:g} l={l:g}
opts options temp={temp:g} tnom=27 rawfmt=psfascii
save M1:vth M1:ids VD:p VG:p
"""


def _arange(start, stop, step):
    return np.round(np.arange(start, stop + step / 2, step), 6)


def spectre_netlist(model_file: str, model: str, w: float, l: float, vdd: float,
                    section: str | None = "tt", temp: float = 25.0, vd_lin: float = 0.05,
                    step: float = 0.01, vg_out=None) -> str:
    """Spectre netlist with Id-Vg at Vd_lin and Vdd, and an Id-Vd family."""
    if vg_out is None:
        vg_out = np.round(np.linspace(0.4 * vdd, vdd, 4), 6)
    lines = [NETLIST_HEADER.format(model=model, w=w, l=l, temp=temp, model_file=model_file,
                                   section=f" section={section}" if section else "", vd_lin=vd_lin)]
    for k, vd in enumerate([vd_lin, vdd]):
        lines.append(f"setvd{k} alter param=vds value={vd:g}")
        lines.append(f"idvg_vd{k} dc param=vgs start=0 stop={vdd:g} step={step:g}")
    for k, vg in enumerate(vg_out):
        lines.append(f"setvg{k} alter param=vgs value={vg:g}")
        lines.append(f"idvd_vg{k} dc param=vds start=0 stop={vdd:g} step={2 * step:g}")
    return "\n".join(lines) + "\n"


def write_netlist_dir(out_dir: Path, **kwargs) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ivchar_ref.scs").write_text(spectre_netlist(**kwargs))
    (out_dir / "run.sh").write_text(
        "#!/bin/sh\n# Run on the EDA machine (needs spectre on PATH and a license).\n"
        "cd \"$(dirname \"$0\")\" && spectre ivchar_ref.scs -raw psf +log spectre.log\n")
    return out_dir


# ---------------------------------------------------------------------------
# PSF-ASCII reader (DC sweeps)
# ---------------------------------------------------------------------------

_SECTIONS = ("HEADER", "TYPE", "SWEEP", "TRACE", "VALUE", "END")


def parse_psfascii(text: str) -> dict[str, np.ndarray]:
    """Parse a PSF-ASCII DC sweep into {signal name: array}, sweep variable included.

    Only the subset Spectre writes for swept DC analyses is handled: in the
    VALUE section each point is the sweep value followed by one line per trace,
    all as ``"name" number`` pairs.
    """
    section = None
    sweep_name = None
    traces: list[str] = []
    values: dict[str, list[float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in _SECTIONS:
            section = line
            continue
        if section == "SWEEP" and sweep_name is None:
            sweep_name = shlex.split(line)[0]
        elif section == "TRACE":
            tok = shlex.split(line)
            if tok and not line.startswith(("GROUP", "PROP")):
                traces.append(tok[0])
        elif section == "VALUE":
            m = re.match(r'^"([^"]+)"\s+(\S+)', line)
            if m:
                values.setdefault(m.group(1), []).append(float(m.group(2)))
    if sweep_name is None:
        raise ValueError("not a PSF-ASCII sweep: no SWEEP section")
    return {k: np.asarray(v) for k, v in values.items()}


def import_results(raw_dir: Path, vd_lin: float, vdd: float, vg_out=None,
                   die: str = "BSIM", corner: str = "TT", temp: float = 25.0) -> pd.DataFrame:
    """Convert Spectre's per-analysis PSF files into ivchar's long-format curves."""
    if vg_out is None:
        vg_out = np.round(np.linspace(0.4 * vdd, vdd, 4), 6)
    rows = []
    for k, vd in enumerate([vd_lin, vdd]):
        d = parse_psfascii(_find(raw_dir, f"idvg_vd{k}"))
        rows.append(_frame("idvg", d["vgs"], vd, d))
    for k, vg in enumerate(vg_out):
        d = parse_psfascii(_find(raw_dir, f"idvd_vg{k}"))
        rows.append(_frame("idvd", vg, d["vds"], d))
    df = pd.concat(rows, ignore_index=True)
    df["die"], df["corner"], df["temp_c"] = die, corner, temp
    df.attrs["simulated"] = True
    return df


def _find(raw_dir: Path, analysis: str) -> str:
    hits = sorted(Path(raw_dir).glob(f"{analysis}.dc*"))
    if not hits:
        raise FileNotFoundError(f"no PSF result for analysis {analysis!r} in {raw_dir}")
    return hits[0].read_text()


def _frame(sweep, vg, vd, d: dict) -> pd.DataFrame:
    n = len(d.get("vgs", d.get("vds")))
    # Spectre reports a source's current flowing *into* its + terminal, so the
    # current the drain source delivers to the device is -VD:p. That terminal
    # current (channel + junction leakage) is what an SMU would measure; the
    # model's channel current M1:ids is kept alongside for reference.
    id_ = -d["VD:p"]
    return pd.DataFrame({
        "sweep": sweep,
        "vg": np.broadcast_to(vg, n).astype(float),
        "vd": np.broadcast_to(vd, n).astype(float),
        "id": id_,
        "ig": -d["VG:p"] if "VG:p" in d else 0.0,
        "isrc": np.nan,
        "compliance": False,
        "bsim_vth": d.get("M1:vth", np.full(n, np.nan)),
        "bsim_ids": d.get("M1:ids", np.full(n, np.nan)),
    })
