"""Extraction methods on analytic curves with known answers, then on the simulator."""

import numpy as np
import pandas as pd
import pytest

from ivchar.device import make_device, thermal_voltage
from ivchar.extraction import (
    dibl, extract, local_slope, subthreshold_swing, vth_constant_current, vth_max_gm,
)
from ivchar.instruments import SimulatedSMU
from ivchar.measure import standard_sweeps


def test_local_slope_exact_on_line():
    x = np.linspace(0, 1, 21)
    np.testing.assert_allclose(local_slope(x, 3 * x + 1), 3.0)


def test_constant_current_vth_on_exponential():
    vg = np.linspace(0, 1, 101)
    ss = 0.075  # V/dec
    id_ = 1e-7 * 10 ** ((vg - 0.4) / ss)  # crosses 100 nA * (W/L=1) at exactly 0.4 V
    assert vth_constant_current(vg, id_, w=1, l=1) == pytest.approx(0.4, abs=1e-9)


def test_constant_current_vth_not_reached():
    assert np.isnan(vth_constant_current([0, 1], [1e-12, 1e-11], 1, 1))


def test_max_gm_vth_on_linear_region_model():
    # Id = k (Vg - Vt - Vd/2) Vd above Vt, zero below: extrapolation gives Vt.
    vt, vd, k = 0.35, 0.05, 2e-3
    vg = np.linspace(0, 1, 101)
    id_ = np.where(vg > vt + vd / 2, k * (vg - vt - vd / 2) * vd, 0.0)
    vth, gm = vth_max_gm(vg, id_, vd)
    assert vth == pytest.approx(vt, abs=2e-3)
    assert gm == pytest.approx(k * vd, rel=1e-6)


def test_subthreshold_swing_on_exponential():
    vg = np.linspace(0, 0.6, 61)
    id_ = 1e-12 * 10 ** (vg / 0.085)
    assert subthreshold_swing(vg, id_, i_max=1e-6) == pytest.approx(85.0, rel=1e-6)


def test_dibl_sign_and_units():
    assert dibl(0.40, 0.32, 0.05, 1.05) == pytest.approx(80.0)


def _quiet_curves(preset="planar", temp=26.85, **kw):
    dev = make_device(preset, temp_c=temp, **kw)
    smu = SimulatedSMU(dev, noise_floor=0.0, gain_error=0.0)
    return dev, standard_sweeps(smu, dev.params.vdd, step=0.005)


@pytest.mark.parametrize("preset", ["planar", "finfet"])
def test_extract_recovers_model_parameters(preset):
    dev, curves = _quiet_curves(preset)
    p = dev.params
    r = extract(curves, p.w, p.l, p.vdd)
    assert r["ss_sat"] == pytest.approx(p.n * thermal_voltage(300) * np.log(10) * 1e3, rel=0.03)
    # Extracted DIBL also picks up CLM and the low-Vd (1 - e^-Vd/Ut) term,
    # so it reads somewhat above the model's eta.
    assert p.eta * 1e3 < r["dibl"] < 1.4 * p.eta * 1e3
    assert r["vth_gm_lin"] > r["vth_cc_lin"]  # LE sits above CC by construction here
    assert r["ion_ioff"] > 1e4
    assert r["n_compliance"] == 0


def test_extract_tracks_injected_vth_shift():
    _, base = _quiet_curves()
    dev, shifted = _quiet_curves(dvth=0.05)
    p = dev.params
    a, b = extract(base, p.w, p.l, p.vdd), extract(shifted, p.w, p.l, p.vdd)
    assert b["vth_cc_sat"] - a["vth_cc_sat"] == pytest.approx(0.05, abs=2e-3)


def test_extract_excludes_compliance_points():
    dev, curves = _quiet_curves()
    p = dev.params
    curves = curves.copy()
    hot = (curves.sweep == "idvg") & (curves.vd == curves.vd.max()) & (curves.vg > 0.9)
    curves.loc[hot, "compliance"] = True
    curves.loc[hot, "id"] = 1e-4  # clamped values must not be used as Ion
    r = extract(curves, p.w, p.l, p.vdd)
    assert np.isnan(r["ion"])  # Vg = Vdd point was in compliance -> not reported
    assert r["n_compliance"] == int(hot.sum())


def test_extract_needs_two_drain_biases():
    df = pd.DataFrame({"sweep": "idvg", "vg": [0, 1], "vd": 0.05, "id": [1e-12, 1e-4],
                       "ig": 0.0, "compliance": False})
    with pytest.raises(ValueError):
        extract(df, 1e-6, 1e-7, 1.0)
