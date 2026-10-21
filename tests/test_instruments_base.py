"""Tests for the SMU abstraction, using a fake instrument with a known load."""

import numpy as np
import pytest

from ivchar.instruments import SMU, Reading, SourceMode


class ResistorSMU(SMU):
    """Fake SMU where every channel drives a resistor to ground.

    Models compliance the way a real SMU does: in voltage mode, if V/R would
    exceed the current limit, the output clamps at the limit and the forced
    voltage collapses to I_limit * R.
    """

    def __init__(self, resistance=1e3, n_channels=2):
        super().__init__("fake-resistor")
        self.r = resistance
        self._channels = list(range(1, n_channels + 1))
        self.state = {}
        self.connected = False

    def connect(self):
        self.connected = True

    def close(self):
        self.connected = False

    @property
    def channels(self):
        return self._channels

    def force_voltage(self, channel, voltage, compliance):
        self._check_channel(channel)
        self._check_compliance(compliance)
        self.state[channel] = (SourceMode.VOLTAGE, voltage, compliance)

    def force_current(self, channel, current, compliance):
        self._check_channel(channel)
        self._check_compliance(compliance)
        self.state[channel] = (SourceMode.CURRENT, current, compliance)

    def _solve(self, channel):
        """Return (voltage, current, in_compliance) at the channel."""
        mode, level, limit = self.state.get(channel, (SourceMode.VOLTAGE, 0.0, 1.0))
        if mode is SourceMode.VOLTAGE:
            i = level / self.r
            if abs(i) > limit:
                i = np.sign(i) * limit
                return i * self.r, i, True
            return level, i, False
        v = level * self.r
        if abs(v) > limit:
            v = np.sign(v) * limit
            return v, v / self.r, True
        return v, level, False

    def measure_current(self, channel):
        _, i, comp = self._solve(channel)
        return Reading(i, comp)

    def measure_voltage(self, channel):
        v, _, comp = self._solve(channel)
        return Reading(v, comp)

    def output_off(self, channel):
        self.state.pop(channel, None)


def test_smu_is_abstract():
    with pytest.raises(TypeError):
        SMU("bare")


def test_voltage_sweep_follows_ohms_law():
    smu = ResistorSMU(resistance=1e3)
    v = np.linspace(0, 0.5, 6)
    res = smu.voltage_sweep(1, v, compliance=1e-3)
    np.testing.assert_allclose(res.currents[1], v / 1e3)
    assert not res.any_compliance.any()


def test_compliance_clamps_and_flags_in_all_four_quadrants():
    smu = ResistorSMU(resistance=1e3)
    v = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    res = smu.voltage_sweep(1, v, compliance=1e-3)
    # |V/R| > 1 mA at +/-2 V: current clamps at +/-1 mA and the point is flagged.
    np.testing.assert_allclose(res.currents[1], [-1e-3, -0.5e-3, 0.0, 0.5e-3, 1e-3])
    np.testing.assert_array_equal(res.compliance[1], [True, False, False, False, True])


def test_voltage_collapses_in_compliance():
    smu = ResistorSMU(resistance=1e3)
    smu.force_voltage(1, 5.0, compliance=1e-3)
    reading = smu.measure_voltage(1)
    assert reading.in_compliance
    assert reading.value == pytest.approx(1.0)  # 1 mA * 1 kOhm, not the 5 V requested


def test_force_current_mode():
    smu = ResistorSMU(resistance=1e3)
    smu.force_current(1, 2e-3, compliance=10.0)
    assert smu.measure_voltage(1) == Reading(2.0, False)


def test_sweep_holds_other_channel_bias():
    smu = ResistorSMU(resistance=1e3)
    smu.force_voltage(2, 0.1, compliance=1e-3)  # e.g. drain held at Vd = 0.1 V
    res = smu.voltage_sweep(1, [0.0, 0.2, 0.4], compliance=1e-3, measure_channels=[1, 2])
    np.testing.assert_allclose(res.currents[2], 0.1e-3)


def test_metadata_carries_simulated_flag():
    smu = ResistorSMU()
    res = smu.voltage_sweep(1, [0.0, 0.1], compliance=1e-3)
    assert res.metadata["instrument"] == "fake-resistor"
    assert res.metadata["simulated"] is False


def test_context_manager_turns_outputs_off():
    smu = ResistorSMU()
    with smu:
        assert smu.connected
        smu.force_voltage(1, 0.3, compliance=1e-3)
    assert not smu.connected
    assert smu.state == {}


def test_context_manager_turns_outputs_off_after_error():
    smu = ResistorSMU()
    with pytest.raises(RuntimeError):
        with smu:
            smu.force_voltage(1, 0.3, compliance=1e-3)
            raise RuntimeError("measurement crashed")
    assert smu.state == {}


@pytest.mark.parametrize("bad", [0.0, -1e-3])
def test_rejects_non_positive_compliance(bad):
    with pytest.raises(ValueError):
        ResistorSMU().force_voltage(1, 0.1, compliance=bad)


def test_rejects_unknown_channel():
    with pytest.raises(ValueError, match="no channel 9"):
        ResistorSMU().voltage_sweep(9, [0.0], compliance=1e-3)
