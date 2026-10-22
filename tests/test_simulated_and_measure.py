"""Simulated SMU backend and the Id-Vg / Id-Vd measurement routines."""

import numpy as np
import pytest

from ivchar.device import make_device
from ivchar.instruments import SimulatedSMU
from ivchar.measure import Compliance, id_vd, id_vg, standard_sweeps


def quiet_smu(**kw):
    return SimulatedSMU(make_device(**kw), noise_floor=0.0, gain_error=0.0)


def test_vectorized_sweep_matches_point_by_point():
    smu = quiet_smu()
    smu.force_voltage(2, 0.5, 1e-2)
    vg = np.linspace(0, 1, 11)
    fast = smu.voltage_sweep(1, vg, 1e-6, measure_channels=[2])
    slow = super(SimulatedSMU, smu).voltage_sweep(1, vg, 1e-6, measure_channels=[2])
    np.testing.assert_allclose(fast.currents[2], slow.currents[2], rtol=1e-12)


def test_metadata_marks_data_simulated():
    df = id_vg(quiet_smu(), [0.0, 0.5], [0.05])
    assert df.attrs["simulated"] is True
    assert df.attrs["instrument"] == "simulated-smu"


def test_id_vg_shape_and_kcl():
    df = id_vg(quiet_smu(), np.linspace(0, 1, 21), [0.05, 1.0])
    assert len(df) == 42 and set(df.vd) == {0.05, 1.0}
    # source current mirrors drain current (gate leakage is tiny)
    np.testing.assert_allclose(df.isrc, -df.id, rtol=1e-3, atol=1e-12)


def test_id_vd_saturates():
    df = id_vd(quiet_smu(), np.linspace(0, 1, 51), [1.0])
    i = df.sort_values("vd").id.to_numpy()
    assert abs(i[0]) < 1e-12  # only gate leakage flows at Vd = 0
    slope_lin = (i[5] - i[0]) / 0.1
    slope_sat = (i[-1] - i[-6]) / 0.1
    assert slope_sat < 0.2 * slope_lin  # output curve flattens


def test_drain_compliance_is_flagged():
    df = id_vg(quiet_smu(), np.linspace(0, 1, 11), [1.0], compliance=Compliance(drain=1e-4))
    hit = df[df.compliance]
    assert len(hit) > 0
    np.testing.assert_allclose(hit.id, 1e-4)


def test_noise_is_reproducible_with_seed():
    a = standard_sweeps(SimulatedSMU(make_device(), seed=7), 1.0)
    b = standard_sweeps(SimulatedSMU(make_device(), seed=7), 1.0)
    np.testing.assert_array_equal(a.id, b.id)


def test_force_current_solves_drain_voltage():
    smu = quiet_smu()
    smu.force_voltage(1, 1.0, 1e-6)
    smu.force_voltage(3, 0.0, 1e-2)
    smu.force_voltage(4, 0.0, 1e-2)
    smu.force_current(2, 100e-6, 2.0)
    v = smu.measure_voltage(2)
    assert not v.in_compliance
    assert make_device().terminal_currents(1.0, v.value)["drain"] == pytest.approx(100e-6, rel=1e-6)


def test_force_current_hits_voltage_compliance():
    smu = quiet_smu()
    smu.force_voltage(1, 0.0, 1e-6)  # device off: 1 mA cannot flow
    smu.force_current(2, 1e-3, 1.5)
    v = smu.measure_voltage(2)
    assert v.in_compliance and v.value == pytest.approx(1.5)


def test_grounded_source_and_custom_recipe():
    """Discrete-part setup: only gate and drain on SMUs, separate Vg/Vd ranges."""
    from ivchar.measure import Terminals

    dev = make_device("discrete")
    smu = SimulatedSMU(dev, terminals={"gate": 1, "drain": 2}, noise_floor=0.0, gain_error=0.0)
    df = standard_sweeps(smu, vdd=3.0, vd_lin=0.1, step=0.05, vg_max=4.0, vd_sat=2.0, vd_max=3.0,
                         vg_out=[2.5, 3.0], terminals=Terminals(gate=1, drain=2, source=None, body=None),
                         compliance=Compliance(gate=1e-5, drain=0.1))
    tr = df[df.sweep == "idvg"]
    assert tr.vg.max() == pytest.approx(4.0) and set(tr.vd) == {0.1, 2.0}
    assert df[df.sweep == "idvd"].vd.max() == pytest.approx(3.0)
    assert df.isrc.isna().all()  # no SMU on the source
    assert df.compliance.any()  # 4 V gate drive exceeds the 100 mA drain limit
