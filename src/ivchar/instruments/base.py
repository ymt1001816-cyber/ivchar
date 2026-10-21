"""Instrument abstraction layer for source-measure units (SMUs).

Every backend (real hardware over PyVISA, or the simulated device) implements
:class:`SMU`. Measurement and analysis code only talks to this interface, so the
same Id-Vg / Id-Vd routines run unchanged on a Keysight B1500A, a generic SMU or
the simulator.

Sign convention (Keysight/Keithley): positive current flows *out of* the SMU's
force terminal into the device under test. For an NMOS biased at Vd > 0, the
drain SMU reads Id > 0 and the source SMU reads about -Id.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class SourceMode(str, Enum):
    VOLTAGE = "voltage"  # force V, measure I, current limited by compliance
    CURRENT = "current"  # force I, measure V, voltage limited by compliance


@dataclass(frozen=True)
class Reading:
    """One measured value plus whether the channel was in compliance.

    A reading taken in compliance is the instrument's limit, not the device's
    response, so extraction code must not treat it as valid device data.
    """

    value: float
    in_compliance: bool = False


@dataclass
class SweepResult:
    """Result of a staircase sweep on one channel."""

    swept_channel: int
    forced: np.ndarray  # values forced on the swept channel (V)
    currents: dict[int, np.ndarray]  # measured current per channel (A)
    compliance: dict[int, np.ndarray]  # True where that channel was in compliance
    metadata: dict = field(default_factory=dict)

    @property
    def any_compliance(self) -> np.ndarray:
        """True at each step where any measured channel hit compliance."""
        masks = list(self.compliance.values())
        return np.logical_or.reduce(masks) if masks else np.zeros_like(self.forced, dtype=bool)


class SMU(ABC):
    """Abstract multi-channel source-measure unit.

    Subclasses implement the primitive operations (force, measure, output off);
    sweeps are built on top of them here, and drivers with a native hardware
    sweep may override :meth:`voltage_sweep`.
    """

    #: True for backends that do not measure real silicon. Reports use this flag
    #: to label data as SIMULATED.
    simulated: bool = False

    def __init__(self, name: str):
        self.name = name

    # --- connection -------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Open the connection and put the instrument in a known state."""

    @abstractmethod
    def close(self) -> None:
        """Release the connection."""

    def __enter__(self) -> SMU:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        # Always leave the device unbiased, even if the measurement crashed.
        try:
            self.all_outputs_off()
        finally:
            self.close()

    # --- primitives -------------------------------------------------------

    @property
    @abstractmethod
    def channels(self) -> Sequence[int]:
        """Channel numbers available on this instrument."""

    @abstractmethod
    def force_voltage(self, channel: int, voltage: float, compliance: float) -> None:
        """Force ``voltage`` (V) on ``channel``, limiting current to +/-``compliance`` (A)."""

    @abstractmethod
    def force_current(self, channel: int, current: float, compliance: float) -> None:
        """Force ``current`` (A) on ``channel``, limiting voltage to +/-``compliance`` (V)."""

    @abstractmethod
    def measure_current(self, channel: int) -> Reading:
        """Measure the current (A) flowing out of ``channel``."""

    @abstractmethod
    def measure_voltage(self, channel: int) -> Reading:
        """Measure the voltage (V) at ``channel``."""

    @abstractmethod
    def output_off(self, channel: int) -> None:
        """Disable the output of ``channel`` (0 V, output relay open)."""

    def all_outputs_off(self) -> None:
        for ch in self.channels:
            self.output_off(ch)

    # --- sweeps -----------------------------------------------------------

    def voltage_sweep(
        self,
        channel: int,
        voltages: Sequence[float],
        compliance: float,
        measure_channels: Sequence[int] | None = None,
        settle_time: float = 0.0,
    ) -> SweepResult:
        """Staircase voltage sweep.

        Steps ``channel`` through ``voltages`` and, at each step, measures the
        current on ``measure_channels`` (default: the swept channel). Other
        channels keep whatever bias was set before the call, which is how a
        Id-Vg sweep holds Vd constant while stepping Vg.
        """
        self._check_channel(channel)
        measure_channels = list(measure_channels or [channel])
        for ch in measure_channels:
            self._check_channel(ch)

        forced = np.asarray(voltages, dtype=float)
        currents = {ch: np.empty(forced.size) for ch in measure_channels}
        flags = {ch: np.zeros(forced.size, dtype=bool) for ch in measure_channels}

        for k, v in enumerate(forced):
            self.force_voltage(channel, v, compliance)
            if settle_time > 0:
                time.sleep(settle_time)
            for ch in measure_channels:
                reading = self.measure_current(ch)
                currents[ch][k] = reading.value
                flags[ch][k] = reading.in_compliance

        return SweepResult(
            swept_channel=channel,
            forced=forced,
            currents=currents,
            compliance=flags,
            metadata={
                "instrument": self.name,
                "simulated": self.simulated,
                "compliance_A": compliance,
                "settle_time_s": settle_time,
            },
        )

    # --- helpers ----------------------------------------------------------

    def _check_channel(self, channel: int) -> None:
        if channel not in self.channels:
            raise ValueError(f"{self.name}: no channel {channel}; available: {list(self.channels)}")

    @staticmethod
    def _check_compliance(compliance: float) -> None:
        if compliance <= 0:
            raise ValueError(f"compliance must be > 0, got {compliance}")
