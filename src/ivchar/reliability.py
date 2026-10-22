"""Bias temperature instability (BTI): degradation model and test routines.

Physics (simulator side)
------------------------
:class:`BTIModel` follows the defect-centric picture of BTI. Gate-oxide
defects with a broad, log-uniform spread of capture and emission times
trap channel carriers while the gate is biased into inversion
(stress) and release them when the bias is removed (recovery). This
*recoverable* part adds up to a roughly logarithmic, power-law-like
Vth shift and relaxes over many decades of time. A smaller *permanent* part
(interface-state generation) grows as a power law of the equivalent stress
time, does not recover, and also degrades the subthreshold slope.

Gate overdrive does two things. It raises the fraction of defects whose
level is pulled below the channel Fermi level (more defects become
chargeable), f_eq = ((Vgs - Vg_on) / (Vg_full - Vg_on))^m, and it speeds up
capture, together with temperature:
    tau_c = tau_c0 * exp(-gamma * (Vgs - Vg_ref)) * exp(Ea/k * (1/T - 1/T_ref))

Test side
---------
:func:`measure_stress_measure` runs the industry measure-stress-measure (MSM)
sequence on any :class:`~ivchar.instruments.base.SMU`, with either fast spot
sensing (Id at one gate bias, converted to ΔVth on the fresh Id-Vg curve) or
full Id-Vg sweeps. :func:`hysteresis` runs forward/reverse sweeps and
:func:`project_lifetime` fits power laws and extrapolates time-to-failure
to the use voltage.

All numbers produced on the simulator are SIMULATED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ivchar.device import K_B, Q_E
from ivchar.extraction import subthreshold_swing, vth_constant_current
from ivchar.instruments.base import SMU
from ivchar.measure import Compliance, Terminals

K_EV = K_B / Q_E  # Boltzmann constant in eV/K


@dataclass
class BTIModel:
    """Recoverable trap ensemble plus a permanent power-law component (NMOS, PBTI)."""

    vg_ref: float = 1.0  # gate bias at which tau_c0 applies (V); usually the use voltage
    t_ref_k: float = 398.15  # 125 C
    vg_on: float = 0.45  # gate bias above which defects capture (channel inverted)
    vg_full: float = 2.2  # gate bias at which every defect is chargeable (V)
    occupancy_m: float = 2.5  # power of the accessible-defect fraction vs overdrive
    n_traps: int = 80
    tau_c_range: tuple[float, float] = (1e-5, 1e9)  # capture times at reference (s)
    emission_spread: tuple[float, float] = (0.0, 2.0)  # log10(tau_e / tau_c)
    dvth_rec_max: float = 0.10  # V, all recoverable defects charged
    gamma: float = 6.0  # 1/V, voltage acceleration of capture
    ea_capture: float = 0.15  # eV
    ea_emission: float = 0.10  # eV
    perm_a: float = 6.0e-4  # V, permanent shift after 1 s equivalent stress at reference
    perm_n: float = 0.25  # power-law exponent of the permanent part
    n_per_volt: float = 2.0  # increase of slope factor n per volt of permanent shift
    seed: int = 0
    _tau_c: np.ndarray = field(init=False, repr=False)
    _tau_e: np.ndarray = field(init=False, repr=False)
    occupancy: np.ndarray = field(init=False, repr=False)
    t_equiv: float = field(init=False, default=0.0)  # equivalent stress time at reference (s)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        lo, hi = np.log10(self.tau_c_range)
        self._tau_c = 10 ** np.linspace(lo, hi, self.n_traps)
        self._tau_e = self._tau_c * 10 ** rng.uniform(*self.emission_spread, self.n_traps)
        self.occupancy = np.zeros(self.n_traps)

    def _arrhenius(self, ea, temp_k):
        return np.exp(ea / K_EV * (1 / temp_k - 1 / self.t_ref_k))

    def acceleration(self, vgs: float, temp_k: float) -> float:
        """Stress acceleration relative to (vg_ref, t_ref_k)."""
        return float(np.exp(self.gamma * (vgs - self.vg_ref)) / self._arrhenius(self.ea_capture, temp_k))

    def accessible_fraction(self, vgs: float) -> float:
        x = (vgs - self.vg_on) / (self.vg_full - self.vg_on)
        return float(np.clip(x, 0.0, 1.0) ** self.occupancy_m)

    def advance(self, dt: float, vgs: float, temp_k: float) -> None:
        """Evolve the defect state for ``dt`` seconds at constant gate bias (exact update)."""
        if dt <= 0:
            return
        f_eq = self.accessible_fraction(vgs)
        acc = self.acceleration(vgs, temp_k)
        # Each defect relaxes toward the equilibrium occupancy at this bias:
        # through capture if it is below it, through emission if above.
        capturing = self.occupancy < f_eq
        tau = np.where(capturing, self._tau_c / acc, self._tau_e * self._arrhenius(self.ea_emission, temp_k))
        self.occupancy = f_eq + (self.occupancy - f_eq) * np.exp(-dt / tau)
        if vgs > self.vg_on:
            self.t_equiv += dt * acc

    @property
    def permanent(self) -> float:
        return self.perm_a * self.t_equiv**self.perm_n if self.t_equiv > 0 else 0.0

    @property
    def delta_vth(self) -> float:
        return float(self.dvth_rec_max * self.occupancy.mean() + self.permanent)

    @property
    def delta_n(self) -> float:
        return self.n_per_volt * self.permanent


# ---------------------------------------------------------------------------
# Test routines (work on any SMU; time passes through smu.hold)
# ---------------------------------------------------------------------------


def _ground(smu, t: Terminals, c: Compliance):
    if t.source is not None:
        smu.force_voltage(t.source, 0.0, c.source)
    if t.body is not None:
        smu.force_voltage(t.body, 0.0, c.body)


def _idvg(smu, vg, vd, t, c):
    smu.force_voltage(t.drain, vd, c.drain)
    res = smu.voltage_sweep(t.gate, vg, c.gate, measure_channels=[t.drain])
    return res.forced, res.currents[t.drain]


@dataclass
class MSMResult:
    data: pd.DataFrame  # phase, time_s, dvth, vth, ss, delay_s, mode, stress_vg
    fresh_vth: float
    vg_sense: float


def measure_stress_measure(
    smu: SMU, stress_vg: float, stress_times: Sequence[float], *, vdd: float,
    i_crit: float, vd_sense: float = 0.05, delay: float = 1e-3, mode: str = "spot",
    recovery_times: Sequence[float] = (), step: float = 0.01,
    terminals: Terminals = Terminals(), compliance: Compliance = Compliance(),
) -> MSMResult:
    """Measure-stress-measure sequence at constant gate stress (drain grounded).

    ``stress_times`` are cumulative stress times (s). After each stress interval
    the gate drops to the sense bias, waits ``delay`` seconds (the time a real
    tester needs to switch and settle, during which the device already
    recovers), then senses:

    * ``mode="spot"``: one Id reading at ``vg_sense`` (the fresh CC threshold),
      converted to ΔVth by interpolating the fresh Id-Vg curve (fast, little
      recovery during the measurement);
    * ``mode="sweep"``: a full Id-Vg sweep, Vth by constant current (slower:
      the device partly recovers, and is partly re-stressed, while sweeping).

    After the last stress, ``recovery_times`` (cumulative, s) are sensed with
    the gate at 0 V in between.
    """
    if mode not in ("spot", "sweep"):
        raise ValueError("mode must be 'spot' or 'sweep'")
    t, c = terminals, compliance
    vg = np.round(np.arange(0.0, vdd + step / 2, step), 6)
    _ground(smu, t, c)
    vg_f, id_f = _idvg(smu, vg, vd_sense, t, c)
    fresh_vth = vth_constant_current(vg_f, id_f, 1.0, 1.0, i_crit)
    fresh_ss = subthreshold_swing(vg_f, id_f, i_crit)
    log_id_f = np.log10(np.clip(id_f, 1e-18, None))

    def sense():
        smu.force_voltage(t.gate, vg_sense if mode == "spot" else 0.0, c.gate)
        smu.force_voltage(t.drain, vd_sense, c.drain)
        smu.hold(delay)
        if mode == "spot":
            i = smu.measure_current(t.drain).value
            # Vg the fresh device would need for this current, minus the bias applied.
            vg_equiv = np.interp(np.log10(max(i, 1e-18)), log_id_f, vg_f)
            return vg_sense - vg_equiv + 0.0, np.nan
        vgs, ids = _idvg(smu, vg, vd_sense, t, c)
        vth = vth_constant_current(vgs, ids, 1.0, 1.0, i_crit)
        return vth - fresh_vth, subthreshold_swing(vgs, ids, i_crit)

    vg_sense = fresh_vth
    rows = [{"phase": "fresh", "time_s": 0.0, "dvth": 0.0, "ss": fresh_ss}]
    elapsed = 0.0
    for target in sorted(stress_times):
        smu.force_voltage(t.drain, 0.0, c.drain)
        smu.force_voltage(t.gate, stress_vg, c.gate)
        smu.hold(target - elapsed)
        elapsed = target
        dvth, ss = sense()
        rows.append({"phase": "stress", "time_s": target, "dvth": dvth, "ss": ss})
    rec_elapsed = 0.0
    for target in sorted(recovery_times):
        smu.force_voltage(t.gate, 0.0, c.gate)
        smu.hold(target - rec_elapsed)
        rec_elapsed = target
        dvth, ss = sense()
        rows.append({"phase": "recovery", "time_s": target, "dvth": dvth, "ss": ss})
    smu.all_outputs_off()

    df = pd.DataFrame(rows)
    df["vth"] = fresh_vth + df["dvth"]
    df["delay_s"], df["mode"], df["stress_vg"] = delay, mode, stress_vg
    return MSMResult(df, fresh_vth, vg_sense)


def hysteresis(smu: SMU, vg_max: float, i_crit: float, vd: float = 0.05, step: float = 0.01,
               terminals: Terminals = Terminals(), compliance: Compliance = Compliance()
               ) -> tuple[pd.DataFrame, float]:
    """Forward (0 -> vg_max) then reverse (vg_max -> 0) Id-Vg sweep.

    Returns the curves and the hysteresis ΔVth = Vth(reverse) - Vth(forward),
    positive when charge trapped during the up-sweep raises Vth.
    """
    t, c = terminals, compliance
    up = np.round(np.arange(0.0, vg_max + step / 2, step), 6)
    _ground(smu, t, c)
    vf, idf = _idvg(smu, up, vd, t, c)
    vr, idr = _idvg(smu, up[::-1], vd, t, c)
    smu.all_outputs_off()
    vth_f = vth_constant_current(vf, idf, 1.0, 1.0, i_crit)
    vth_r = vth_constant_current(vr[::-1], idr[::-1], 1.0, 1.0, i_crit)
    df = pd.concat([pd.DataFrame({"direction": "forward", "vg": vf, "id": idf}),
                    pd.DataFrame({"direction": "reverse", "vg": vr, "id": idr})], ignore_index=True)
    return df, vth_r - vth_f


# ---------------------------------------------------------------------------
# Lifetime projection
# ---------------------------------------------------------------------------


def fit_power_law(time_s, dvth) -> tuple[float, float]:
    """Fit ΔVth = A * t^n on log-log axes. Returns (A, n)."""
    time_s, dvth = np.asarray(time_s, float), np.asarray(dvth, float)
    ok = (time_s > 0) & (dvth > 0)
    n, log_a = np.polyfit(np.log10(time_s[ok]), np.log10(dvth[ok]), 1)
    return 10**log_a, n


def project_lifetime(stress: pd.DataFrame, use_vg: float, criterion: float = 0.03) -> dict:
    """Time-to-failure at each stress voltage and its extrapolation to ``use_vg``.

    ``stress`` has columns stress_vg, time_s, dvth (stress phase only). Each
    voltage gets a power-law fit; time to reach ``criterion`` (V) is then fitted
    as log10(TTF) linear in Vg (exponential voltage acceleration) and
    extrapolated to the use voltage.
    """
    rows = []
    for vg, g in stress.groupby("stress_vg"):
        a, n = fit_power_law(g.time_s, g.dvth)
        rows.append({"stress_vg": vg, "A": a, "n": n, "ttf_s": (criterion / a) ** (1 / n)})
    fits = pd.DataFrame(rows)
    slope, intercept = np.polyfit(fits.stress_vg, np.log10(fits.ttf_s), 1)
    ttf_use = 10 ** (intercept + slope * use_vg)
    return {"fits": fits, "log10_ttf_slope_per_V": slope, "ttf_use_s": ttf_use,
            "ttf_use_years": ttf_use / 3.156e7, "criterion_V": criterion, "use_vg": use_vg}


# ---------------------------------------------------------------------------
# Simulated study: several stress voltages x sensing methods, plus hysteresis
# ---------------------------------------------------------------------------

METHODS = {  # label -> (mode, delay s)
    "spot, 1 ms delay": ("spot", 1e-3),
    "spot, 1 s delay": ("spot", 1.0),
    "full sweep, 1 ms delay": ("sweep", 1e-3),
}


@dataclass
class StudyConfig:
    preset: str = "planar"
    temp_c: float = 125.0
    stress_vg: tuple[float, ...] = (1.6, 1.8, 2.0)
    use_vg: float = 1.0
    stress_times: tuple[float, ...] = tuple(np.round(np.logspace(0, 4, 9), 3))
    recovery_times: tuple[float, ...] = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0)
    criterion: float = 0.03  # V, failure when ΔVth reaches this
    point_time: float = 1e-3  # s per bias step (instrument speed)
    hysteresis_vg: float = 1.8
    hysteresis_point_times: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0)
    i_norm: float = 100e-9  # CC criterion per W/L
    seed: int = 0


def run_study(cfg: StudyConfig) -> dict:
    """Run the MSM study on fresh simulated devices (one device per test)."""
    from ivchar.device import PRESETS, make_device
    from ivchar.instruments.simulated import SimulatedSMU

    p = PRESETS[cfg.preset]
    i_crit = cfg.i_norm * p.w / p.l

    def fresh_smu(point_time):
        dev = make_device(cfg.preset, temp_c=cfg.temp_c)
        dev.aging = BTIModel(vg_ref=cfg.use_vg, seed=cfg.seed)
        return SimulatedSMU(dev, noise_floor=0.0, gain_error=0.0, point_time=point_time)

    msm = []
    for label, (mode, delay) in METHODS.items():
        for vg in cfg.stress_vg:
            r = measure_stress_measure(fresh_smu(cfg.point_time), vg, cfg.stress_times, vdd=p.vdd,
                                       i_crit=i_crit, delay=delay, mode=mode,
                                       recovery_times=cfg.recovery_times)
            msm.append(r.data.assign(method=label))
    msm = pd.concat(msm, ignore_index=True)

    lifetimes = {}
    for label in METHODS:
        s = msm[(msm.method == label) & (msm.phase == "stress")]
        lifetimes[label] = project_lifetime(s, cfg.use_vg, cfg.criterion)

    hyst_curves, hyst = [], []
    for pt in cfg.hysteresis_point_times:
        curves, dv = hysteresis(fresh_smu(pt), cfg.hysteresis_vg, i_crit, step=0.01)
        hyst_curves.append(curves.assign(point_time=pt))
        hyst.append({"point_time_s": pt, "sweep_time_s": 2 * pt * (cfg.hysteresis_vg / 0.01 + 1),
                     "dvth_hysteresis": dv})
    return {"config": cfg, "msm": msm, "lifetimes": lifetimes,
            "hysteresis": pd.DataFrame(hyst), "hysteresis_curves": pd.concat(hyst_curves, ignore_index=True),
            "simulated": True}


def summarize_study(study: dict) -> dict:
    """Headline numbers used by the report and README."""
    cfg, msm = study["config"], study["msm"]
    fast = msm[(msm.method == "spot, 1 ms delay")]
    vg_mid = sorted(cfg.stress_vg)[len(cfg.stress_vg) // 2]
    out = {"vg_mid": vg_mid}
    st = fast[(fast.phase == "stress") & (fast.stress_vg == vg_mid)]
    out["dvth_end_fast"] = st.dvth.iloc[-1]
    end = {}
    for label in METHODS:
        s = msm[(msm.method == label) & (msm.phase == "stress") & (msm.stress_vg == vg_mid)]
        end[label] = (s.dvth.iloc[0], s.dvth.iloc[-1])
    out["dvth_first_last_by_method"] = end
    rec = fast[(fast.phase == "recovery") & (fast.stress_vg == vg_mid)]
    out["recovered_fraction_1000s"] = 1 - rec.dvth.iloc[-1] / st.dvth.iloc[-1]
    out["n_by_method"] = {k: v["fits"].set_index("stress_vg").n.to_dict() for k, v in study["lifetimes"].items()}
    out["ttf_years_by_method"] = {k: v["ttf_use_years"] for k, v in study["lifetimes"].items()}
    sw = msm[(msm.method == "full sweep, 1 ms delay") & (msm.stress_vg == vg_mid) & (msm.phase != "recovery")]
    out["ss_fresh_end"] = (sw.ss.iloc[0], sw.ss.iloc[-1])
    out["hysteresis"] = study["hysteresis"].set_index("point_time_s").dvth_hysteresis.to_dict()
    return out
