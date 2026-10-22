"""BTI model, measure-stress-measure flow, hysteresis and lifetime projection."""

import numpy as np
import pandas as pd
import pytest

from ivchar.cli import main
from ivchar.device import make_device
from ivchar.instruments import SimulatedSMU
from ivchar.reliability import (
    BTIModel, StudyConfig, fit_power_law, hysteresis, measure_stress_measure, project_lifetime,
    run_study, summarize_study,
)

T125 = 398.15


def test_fresh_model_has_no_shift_and_unbiased_gate_does_nothing():
    m = BTIModel()
    assert m.delta_vth == 0 and m.delta_n == 0
    m.advance(1e4, vgs=0.0, temp_k=T125)
    assert m.delta_vth == 0


def test_stress_grows_and_recovery_relaxes_only_the_recoverable_part():
    m = BTIModel()
    shifts = []
    for _ in range(4):
        m.advance(100.0, vgs=1.8, temp_k=T125)
        shifts.append(m.delta_vth)
    assert all(b > a for a, b in zip(shifts, shifts[1:]))
    end, perm = m.delta_vth, m.permanent
    m.advance(1e4, vgs=0.0, temp_k=T125)
    assert perm <= m.delta_vth < end  # partial recovery, permanent part stays
    assert m.delta_n > 0  # interface traps degrade the slope


def test_higher_voltage_and_temperature_degrade_faster():
    def shift(vgs, temp_k):
        m = BTIModel()
        m.advance(1000.0, vgs, temp_k)
        return m.delta_vth
    assert shift(2.0, T125) > shift(1.6, T125)
    assert shift(1.8, T125) > shift(1.8, 300.0)


def test_power_law_fit_exact():
    t = np.logspace(0, 4, 9)
    a, n = fit_power_law(t, 0.01 * t**0.2)
    assert a == pytest.approx(0.01) and n == pytest.approx(0.2)


def test_project_lifetime_on_synthetic_data():
    rows = []
    for vg, a in [(1.6, 1e-3), (1.8, 1e-2), (2.0, 1e-1)]:  # TTF drops 4 decades per 0.2 V
        for t in np.logspace(0, 4, 5):
            rows.append({"stress_vg": vg, "time_s": t, "dvth": a * t**0.25})
    lt = project_lifetime(pd.DataFrame(rows), use_vg=1.0, criterion=0.03)
    ttf = lt["fits"].set_index("stress_vg").ttf_s
    assert ttf[1.6] == pytest.approx((0.03 / 1e-3) ** 4)
    assert lt["log10_ttf_slope_per_V"] == pytest.approx(-20.0)  # 4 decades per 0.2 V
    assert lt["ttf_use_s"] == pytest.approx(ttf[1.6] * 1e12, rel=1e-6)


def _bti_smu(point_time=1e-3):
    dev = make_device(temp_c=125)
    dev.aging = BTIModel()
    return SimulatedSMU(dev, noise_floor=0.0, gain_error=0.0, point_time=point_time)


def test_msm_spot_sensing_recovers_model_shift():
    smu = _bti_smu()
    r = measure_stress_measure(smu, 1.8, [10.0, 100.0], vdd=1.0, i_crit=1e-6, delay=1e-6)
    s = r.data[r.data.phase == "stress"]
    assert (s.dvth.diff().dropna() > 0).all()
    # with negligible delay, spot sensing reads the model's own shift closely
    assert s.dvth.iloc[-1] == pytest.approx(smu.device.aging.delta_vth, rel=0.1)


def test_measurement_delay_underestimates_degradation():
    fast = measure_stress_measure(_bti_smu(), 1.8, [100.0], vdd=1.0, i_crit=1e-6, delay=1e-3).data
    slow = measure_stress_measure(_bti_smu(), 1.8, [100.0], vdd=1.0, i_crit=1e-6, delay=10.0).data
    assert slow.dvth.iloc[-1] < 0.9 * fast.dvth.iloc[-1]


def test_msm_rejects_unknown_mode():
    with pytest.raises(ValueError):
        measure_stress_measure(_bti_smu(), 1.8, [1.0], vdd=1.0, i_crit=1e-6, mode="pulsed")


def test_hysteresis_positive_for_electron_trapping():
    _, dv = hysteresis(_bti_smu(1e-2), 1.8, 1e-6)
    assert dv > 0


def test_devices_without_aging_unchanged_by_clock():
    smu = SimulatedSMU(make_device(), noise_floor=0.0, gain_error=0.0, point_time=1.0)
    smu.force_voltage(1, 1.0, 1e-6)
    smu.force_voltage(2, 0.05, 1e-2)
    i0 = smu.measure_current(2).value
    smu.hold(1e6)
    assert smu.measure_current(2).value == pytest.approx(i0)
    assert smu.clock > 1e6


def test_study_summary_and_cli(tmp_path):
    s = summarize_study(run_study(StudyConfig(stress_times=(1.0, 10.0, 100.0), recovery_times=(1.0, 100.0),
                                              hysteresis_point_times=(1e-3,))))
    years = s["ttf_years_by_method"]
    assert set(years) == {"spot, 1 ms delay", "spot, 1 s delay", "full sweep, 1 ms delay"}
    assert 0 < s["recovered_fraction_1000s"] < 1
    out = tmp_path / "rel"
    assert main(["reliability", "--out", str(out)]) == 0
    html = (out / "reliability_report.html").read_text()
    assert "SIMULATED DATA" in html and (out / "msm.csv").exists()
