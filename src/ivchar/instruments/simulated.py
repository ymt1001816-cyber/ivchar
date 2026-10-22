"""Simulated 4-channel SMU wired to a compact-model MOSFET.

Lets the whole pipeline run without hardware. Every sweep it produces carries
``metadata["simulated"] = True`` so reports label the data as SIMULATED.

Measurement non-idealities modelled:
  * Gaussian noise: sqrt(floor^2 + (gain_error * I)^2), floor ~ 10 fA
  * compliance: a terminal current beyond the limit is clamped and flagged
    (simplification: the other terminal voltages are not re-solved)

Time: the SMU keeps a virtual clock. Each bias step takes ``point_time``
seconds and :meth:`hold` advances the clock without real waiting, so a device
with an aging model (BTI) degrades under stress and recovers between
measurements exactly as the bias history dictates.
"""

from __future__ import annotations

import numpy as np

from ivchar.device import MOSFET, make_device
from ivchar.instruments.base import SMU, Reading, SourceMode, SweepResult

DEFAULT_TERMINALS = {"gate": 1, "drain": 2, "source": 3, "body": 4}


class SimulatedSMU(SMU):
    simulated = True

    def __init__(
        self,
        device: MOSFET | None = None,
        terminals: dict[str, int] | None = None,
        noise_floor: float = 10e-15,
        gain_error: float = 2e-3,
        seed: int | None = 0,
        point_time: float = 0.0,
    ):
        super().__init__("simulated-smu")
        self.device = device or make_device()
        self.terminals = dict(terminals or DEFAULT_TERMINALS)
        self._terminal_of = {ch: name for name, ch in self.terminals.items()}
        self.noise_floor = noise_floor
        self.gain_error = gain_error
        self.rng = np.random.default_rng(seed)
        self.point_time = point_time  # seconds each bias step lasts (virtual clock)
        self.clock = 0.0
        self._state: dict[int, tuple[SourceMode, float, float]] = {}

    def connect(self) -> None:
        self._state.clear()

    def close(self) -> None:
        pass

    @property
    def channels(self):
        return sorted(self._terminal_of)

    def force_voltage(self, channel, voltage, compliance):
        self._check_channel(channel)
        self._check_compliance(compliance)
        self._state[channel] = (SourceMode.VOLTAGE, float(voltage), compliance)
        self._advance(self.point_time)

    def force_current(self, channel, current, compliance):
        self._check_channel(channel)
        self._check_compliance(compliance)
        self._state[channel] = (SourceMode.CURRENT, float(current), compliance)
        self._advance(self.point_time)

    def output_off(self, channel):
        self._check_channel(channel)
        self._state.pop(channel, None)

    def hold(self, seconds: float) -> None:
        self._advance(seconds)

    def _advance(self, dt: float) -> None:
        """Advance the virtual clock; evolve the device's aging state at present bias."""
        if dt <= 0:
            return
        self.clock += dt
        aging = self.device.aging
        if aging is not None:
            v = self._voltages()
            aging.advance(dt, v.get("gate", 0.0) - v.get("source", 0.0), self.device.temp_k)

    # --- device solution --------------------------------------------------

    def _voltages(self) -> dict[str, float]:
        """Terminal voltages, resolving any current-forced channel by bisection."""
        v = {name: 0.0 for name in self.terminals}
        current_forced = []
        for ch, (mode, level, limit) in self._state.items():
            name = self._terminal_of[ch]
            if mode is SourceMode.VOLTAGE:
                v[name] = level
            else:
                current_forced.append((name, level, limit))
        for name, target, vlimit in current_forced:
            v[name] = self._solve_voltage(v, name, target, vlimit)
        return v

    def _solve_voltage(self, v, name, target, vlimit):
        """Find the terminal voltage that draws ``target`` current (clamped to +/-vlimit)."""

        def f(x):
            return self._currents({**v, name: x})[name] - target

        lo, hi = -vlimit, vlimit
        if f(lo) > 0 or f(hi) < 0:  # target not reachable inside the voltage limit
            return hi if f(hi) < 0 else lo
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _currents(self, v) -> dict[str, float]:
        i = self.device.terminal_currents(v.get("gate", 0.0), v.get("drain", 0.0),
                                          v.get("source", 0.0), v.get("body", 0.0))
        return {k: float(x) for k, x in i.items()}

    def _noisy(self, value: float) -> float:
        sigma = np.hypot(self.noise_floor, self.gain_error * value)
        return value + self.rng.normal(0.0, sigma)

    # --- measurements -----------------------------------------------------

    def measure_current(self, channel) -> Reading:
        self._check_channel(channel)
        name = self._terminal_of[channel]
        mode, level, limit = self._state.get(channel, (SourceMode.VOLTAGE, 0.0, np.inf))
        if mode is SourceMode.CURRENT:
            return Reading(level, self._voltage_limited(channel))
        i = self._currents(self._voltages())[name]
        if abs(i) > limit:
            return Reading(float(np.sign(i) * limit), True)
        return Reading(self._noisy(i), False)

    def measure_voltage(self, channel) -> Reading:
        self._check_channel(channel)
        name = self._terminal_of[channel]
        return Reading(self._voltages()[name], self._voltage_limited(channel))

    def voltage_sweep(self, channel, voltages, compliance, measure_channels=None, settle_time=0.0):
        """Vectorized sweep: one model evaluation for the whole staircase.

        Same physics as stepping point by point, the way a hardware sweep on a
        real analyzer is faster than software-timed spot measurements.
        """
        if self.device.aging is not None or any(mode is SourceMode.CURRENT for mode, _, _ in self._state.values()):
            # Bias history matters (aging) or a current-forced node must be solved: step point by point.
            return super().voltage_sweep(channel, voltages, compliance, measure_channels, settle_time)
        self._check_channel(channel)
        self._check_compliance(compliance)
        measure_channels = list(measure_channels or [channel])
        for ch in measure_channels:
            self._check_channel(ch)

        forced = np.asarray(voltages, dtype=float)
        v = {name: 0.0 for name in self.terminals}
        for ch, (_, level, _) in self._state.items():
            v[self._terminal_of[ch]] = level
        v[self._terminal_of[channel]] = forced
        i = self.device.terminal_currents(v["gate"], v["drain"], v.get("source", 0.0), v.get("body", 0.0))

        currents, flags = {}, {}
        for ch in measure_channels:
            limit = compliance if ch == channel else self._state.get(ch, (None, 0.0, np.inf))[2]
            ich = np.broadcast_to(i[self._terminal_of[ch]], forced.shape).astype(float)
            noisy = ich + self.rng.normal(0.0, np.hypot(self.noise_floor, self.gain_error * ich))
            over = np.abs(ich) > limit
            currents[ch] = np.where(over, np.sign(ich) * limit, noisy)
            flags[ch] = over
        if forced.size:
            self.force_voltage(channel, forced[-1], compliance)

        return SweepResult(
            swept_channel=channel, forced=forced, currents=currents, compliance=flags,
            metadata={"instrument": self.name, "simulated": True, "compliance_A": compliance,
                      "settle_time_s": settle_time},
        )

    def _voltage_limited(self, channel) -> bool:
        mode, _, limit = self._state.get(channel, (SourceMode.VOLTAGE, 0.0, np.inf))
        if mode is not SourceMode.CURRENT:
            return False
        return abs(self._voltages()[self._terminal_of[channel]]) >= limit * (1 - 1e-9)
