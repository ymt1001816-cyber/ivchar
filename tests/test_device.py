"""Physics sanity checks on the compact model."""

import numpy as np
import pytest

from ivchar.device import PRESETS, make_device, thermal_voltage


@pytest.mark.parametrize("preset", list(PRESETS))
def test_subthreshold_slope_matches_n_kT_ln10(preset):
    dev = make_device(preset, temp_c=26.85)  # 300 K
    p = dev.params
    vg = np.array([0.0, 0.05])
    i = dev.channel_current(vg, p.vdd)
    ss = 1e3 * (vg[1] - vg[0]) / np.log10(i[1] / i[0])
    assert ss == pytest.approx(p.n * thermal_voltage(300) * np.log(10) * 1e3, rel=0.03)


def test_swapping_source_and_drain_reverses_current():
    dev = make_device()
    fwd = dev.channel_current(vg=0.8, vd=0.3, vs=0.0)
    rev = dev.channel_current(vg=0.8, vd=0.0, vs=0.3)
    assert fwd > 0 and rev == pytest.approx(-fwd, rel=1e-9)
    assert dev.channel_current(0.8, 0.0) == pytest.approx(0.0, abs=1e-18)


def test_kcl_terminal_currents_sum_to_zero():
    dev = make_device(gate_leak_scale=1e3)
    i = dev.terminal_currents(vg=0.7, vd=0.9, vs=0.0, vb=0.0)
    assert sum(i.values()) == pytest.approx(0.0, abs=1e-15)
    assert i["drain"] > 0 > i["source"]


def test_temperature_raises_ioff_and_lowers_ion():
    p = PRESETS["planar"]
    cold, hot = make_device(temp_c=-40), make_device(temp_c=125)
    assert hot.channel_current(0.0, p.vdd) > 100 * cold.channel_current(0.0, p.vdd)
    assert hot.channel_current(p.vdd, p.vdd) < cold.channel_current(p.vdd, p.vdd)


def test_corners_order_drive_current():
    p = PRESETS["planar"]
    ion = {c: make_device(corner=c).channel_current(p.vdd, p.vdd) for c in ("SS", "TT", "FF")}
    assert ion["SS"] < ion["TT"] < ion["FF"]


def test_unknown_corner_and_preset_rejected():
    with pytest.raises(ValueError):
        make_device(corner="XX")
    with pytest.raises(ValueError):
        make_device(preset="gaafet")
