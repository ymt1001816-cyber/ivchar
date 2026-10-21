"""Standard MOSFET I-V measurement routines built on the SMU interface.

Both routines return a long-format DataFrame (one row per bias point) with
columns: sweep, vg, vd, id, ig, isrc, compliance. Instrument metadata,
including the ``simulated`` flag, is stored in ``df.attrs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ivchar.instruments.base import SMU


@dataclass(frozen=True)
class Terminals:
    """SMU channel wired to each device terminal."""

    gate: int = 1
    drain: int = 2
    source: int = 3
    body: int = 4


@dataclass(frozen=True)
class Compliance:
    """Current limits (A). The gate limit is small to protect the thin oxide."""

    gate: float = 1e-6
    drain: float = 20e-3
    source: float = 20e-3
    body: float = 1e-3


COLUMNS = ["sweep", "vg", "vd", "id", "ig", "isrc", "compliance"]


def _ground(smu: SMU, t: Terminals, c: Compliance) -> None:
    smu.force_voltage(t.source, 0.0, c.source)
    smu.force_voltage(t.body, 0.0, c.body)


def _frame(rows: list[pd.DataFrame], smu: SMU) -> pd.DataFrame:
    df = pd.concat(rows, ignore_index=True)[COLUMNS]
    df.attrs.update({"instrument": smu.name, "simulated": smu.simulated})
    return df


def id_vg(smu: SMU, vg: Sequence[float], vd_values: Sequence[float],
          terminals: Terminals = Terminals(), compliance: Compliance = Compliance(),
          settle_time: float = 0.0) -> pd.DataFrame:
    """Transfer curves: sweep Vg at each fixed Vd (source and body grounded)."""
    t, c = terminals, compliance
    rows = []
    _ground(smu, t, c)
    for vd in vd_values:
        smu.force_voltage(t.drain, vd, c.drain)
        res = smu.voltage_sweep(t.gate, vg, c.gate, measure_channels=[t.drain, t.gate, t.source],
                                settle_time=settle_time)
        rows.append(pd.DataFrame({
            "sweep": "idvg", "vg": res.forced, "vd": float(vd),
            "id": res.currents[t.drain], "ig": res.currents[t.gate], "isrc": res.currents[t.source],
            "compliance": res.any_compliance,
        }))
    smu.all_outputs_off()
    return _frame(rows, smu)


def id_vd(smu: SMU, vd: Sequence[float], vg_values: Sequence[float],
          terminals: Terminals = Terminals(), compliance: Compliance = Compliance(),
          settle_time: float = 0.0) -> pd.DataFrame:
    """Output curves: sweep Vd at each fixed Vg (source and body grounded)."""
    t, c = terminals, compliance
    rows = []
    _ground(smu, t, c)
    for vg in vg_values:
        smu.force_voltage(t.gate, vg, c.gate)
        res = smu.voltage_sweep(t.drain, vd, c.drain, measure_channels=[t.drain, t.gate, t.source],
                                settle_time=settle_time)
        rows.append(pd.DataFrame({
            "sweep": "idvd", "vg": float(vg), "vd": res.forced,
            "id": res.currents[t.drain], "ig": res.currents[t.gate], "isrc": res.currents[t.source],
            "compliance": res.any_compliance,
        }))
    smu.all_outputs_off()
    return _frame(rows, smu)


def standard_sweeps(smu: SMU, vdd: float, vd_lin: float = 0.05, step: float = 0.01,
                    vg_out: Sequence[float] | None = None, **kwargs) -> pd.DataFrame:
    """The usual characterization set: Id-Vg at Vd_lin and Vdd, plus Id-Vd family."""
    vg = np.round(np.arange(0.0, vdd + step / 2, step), 6)
    vd = np.round(np.arange(0.0, vdd + step / 2, 2 * step), 6)
    if vg_out is None:
        vg_out = np.round(np.linspace(0.4 * vdd, vdd, 4), 6)
    transfer = id_vg(smu, vg, [vd_lin, vdd], **kwargs)
    output = id_vd(smu, vd, vg_out, **kwargs)
    df = pd.concat([transfer, output], ignore_index=True)
    df.attrs.update(transfer.attrs)
    return df
