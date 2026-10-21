"""PyVISA/SCPI drivers for real instruments.

* :class:`B1500A` - Keysight B1500A semiconductor parameter analyzer, using its
  FLEX command set (``CN``, ``DV``, ``DI``, ``TI``, ``TV``, ``CL`` ...).
* :class:`GenericSCPISMU` - a single-channel SCPI SMU (Keithley 2400-style
  ``:SOUR``/``:SENS``/``:READ?`` subsystem).
* :class:`MultiSMU` - combines several instruments into one multi-channel SMU,
  e.g. four single-channel units wired to G/D/S/B.

The drivers are unit-tested against recorded command strings with a fake VISA
resource; they have not been validated on physical hardware in this project.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from ivchar.instruments.base import SMU, Reading


class VisaResource(Protocol):
    """The subset of ``pyvisa.resources.MessageBasedResource`` the drivers use."""

    timeout: int

    def write(self, message: str) -> object: ...
    def query(self, message: str) -> str: ...
    def close(self) -> None: ...


def open_resource(address: str, timeout_ms: int = 10_000) -> VisaResource:
    try:
        import pyvisa
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Hardware backends need PyVISA: pip install 'ivchar[visa]'") from exc
    res = pyvisa.ResourceManager().open_resource(address)
    res.timeout = timeout_ms
    res.read_termination = "\n"
    res.write_termination = "\n"
    return res


class VisaSMU(SMU):
    """Common connection handling for VISA instruments."""

    def __init__(self, name: str, address: str | None = None, resource: VisaResource | None = None):
        super().__init__(name)
        if address is None and resource is None:
            raise ValueError("give either a VISA address or an open resource")
        self.address = address
        self.resource = resource

    def connect(self) -> None:
        if self.resource is None:
            self.resource = open_resource(self.address)
        self._initialize()

    def close(self) -> None:
        if self.resource is not None:
            self.resource.close()

    def _initialize(self) -> None:
        """Put the instrument in a known state after connecting."""

    def write(self, cmd: str) -> None:
        self.resource.write(cmd)

    def query(self, cmd: str) -> str:
        return self.resource.query(cmd).strip()


# ---------------------------------------------------------------------------
# Keysight B1500A (FLEX commands)
# ---------------------------------------------------------------------------

# FMT 1 ASCII data: <status><channel letter><data type><value>, e.g. "NAI+1.234567E-09".
_B1500_DATA = re.compile(r"^([A-Z])([A-Z@])([A-Z])([+-]?\d+(?:\.\d*)?(?:E[+-]?\d+)?)$")
# Status letters that mean the reading was limited by compliance on this channel
# ('C') or another channel ('T').
_B1500_COMPLIANCE = {"C", "T"}


class B1500A(VisaSMU):
    """Keysight B1500A with SMU modules in the given slots/channels.

    ``integration`` selects the ADC integration: ``("auto", 1)``,
    ``("manual", n)`` or ``("plc", n)`` (power-line cycles). Longer integration
    lowers noise on small currents at the cost of sweep time.
    """

    _AIT_MODE = {"auto": 0, "manual": 1, "plc": 2}

    def __init__(self, address: str | None = None, channels: Sequence[int] = (1, 2, 3, 4),
                 integration: tuple[str, int] = ("plc", 1), resource: VisaResource | None = None):
        super().__init__("keysight-b1500a", address, resource)
        self._channels = list(channels)
        self.integration = integration

    @property
    def channels(self):
        return self._channels

    def _initialize(self) -> None:
        self.write("*RST")
        self.write("FMT 1,0")  # ASCII with header, no source data in the output
        mode, coeff = self.integration
        self.write(f"AIT 1,{self._AIT_MODE[mode]},{coeff}")  # high-resolution ADC
        self.write("CN " + ",".join(str(ch) for ch in self._channels))
        for ch in self._channels:
            self.write(f"AAD {ch},1")
        self._check_errors()

    def _check_errors(self) -> None:
        err = self.query("ERRX?")
        if not err.startswith("0"):
            raise RuntimeError(f"B1500A error: {err}")

    def force_voltage(self, channel, voltage, compliance):
        self._check_channel(channel)
        self._check_compliance(compliance)
        self.write(f"DV {channel},0,{voltage:.6E},{compliance:.6E}")  # range 0 = auto

    def force_current(self, channel, current, compliance):
        self._check_channel(channel)
        self._check_compliance(compliance)
        self.write(f"DI {channel},0,{current:.6E},{compliance:.6E}")

    def measure_current(self, channel) -> Reading:
        self._check_channel(channel)
        return self.parse(self.query(f"TI {channel},0"))

    def measure_voltage(self, channel) -> Reading:
        self._check_channel(channel)
        return self.parse(self.query(f"TV {channel},0"))

    def output_off(self, channel):
        self._check_channel(channel)
        self.write(f"DZ {channel}")

    def close(self) -> None:
        if self.resource is not None:
            self.write("CL")
        super().close()

    @staticmethod
    def parse(response: str) -> Reading:
        """Parse one FMT 1 data element, e.g. ``"CBI+1.000000E-03"``."""
        m = _B1500_DATA.match(response.strip())
        if not m:
            raise ValueError(f"unexpected B1500A response: {response!r}")
        status, _channel, _kind, value = m.groups()
        return Reading(float(value), status in _B1500_COMPLIANCE)


# ---------------------------------------------------------------------------
# Generic single-channel SCPI SMU (Keithley 2400-style)
# ---------------------------------------------------------------------------


class GenericSCPISMU(VisaSMU):
    """Single-channel SMU exposed as channel 1."""

    def __init__(self, address: str | None = None, resource: VisaResource | None = None,
                 name: str = "generic-scpi-smu", nplc: float = 1.0):
        super().__init__(name, address, resource)
        self.nplc = nplc
        self._mode = None

    @property
    def channels(self):
        return [1]

    def _initialize(self) -> None:
        self.write("*RST")
        self.write(f":SENS:CURR:NPLC {self.nplc}")
        self.write(f":SENS:VOLT:NPLC {self.nplc}")

    def force_voltage(self, channel, voltage, compliance):
        self._check_channel(channel)
        self._check_compliance(compliance)
        if self._mode != "VOLT":
            self.write(":SOUR:FUNC VOLT")
            self.write(':SENS:FUNC "CURR"')
            self._mode = "VOLT"
        self.write(f":SENS:CURR:PROT {compliance:.6E}")
        self.write(f":SOUR:VOLT:LEV {voltage:.6E}")
        self.write(":OUTP ON")

    def force_current(self, channel, current, compliance):
        self._check_channel(channel)
        self._check_compliance(compliance)
        if self._mode != "CURR":
            self.write(":SOUR:FUNC CURR")
            self.write(':SENS:FUNC "VOLT"')
            self._mode = "CURR"
        self.write(f":SENS:VOLT:PROT {compliance:.6E}")
        self.write(f":SOUR:CURR:LEV {current:.6E}")
        self.write(":OUTP ON")

    def _read(self, element: str, trip_query: str) -> Reading:
        self.write(f":FORM:ELEM {element}")
        value = float(self.query(":READ?"))
        tripped = self.query(trip_query) == "1"
        return Reading(value, tripped)

    def measure_current(self, channel) -> Reading:
        self._check_channel(channel)
        return self._read("CURR", ":SENS:CURR:PROT:TRIP?")

    def measure_voltage(self, channel) -> Reading:
        self._check_channel(channel)
        return self._read("VOLT", ":SENS:VOLT:PROT:TRIP?")

    def output_off(self, channel):
        self._check_channel(channel)
        self.write(":OUTP OFF")


# ---------------------------------------------------------------------------
# Several instruments as one
# ---------------------------------------------------------------------------


class MultiSMU(SMU):
    """Map logical channels to (instrument, instrument channel) pairs."""

    def __init__(self, mapping: dict[int, tuple[SMU, int]], name: str = "multi-smu"):
        super().__init__(name)
        self.mapping = dict(mapping)
        self.simulated = all(smu.simulated for smu, _ in self.mapping.values())

    def _units(self):
        seen = []
        for smu, _ in self.mapping.values():
            if all(smu is not s for s in seen):
                seen.append(smu)
        return seen

    @property
    def channels(self):
        return sorted(self.mapping)

    def connect(self):
        for smu in self._units():
            smu.connect()

    def close(self):
        for smu in self._units():
            smu.close()

    def _route(self, channel):
        self._check_channel(channel)
        return self.mapping[channel]

    def force_voltage(self, channel, voltage, compliance):
        smu, ch = self._route(channel)
        smu.force_voltage(ch, voltage, compliance)

    def force_current(self, channel, current, compliance):
        smu, ch = self._route(channel)
        smu.force_current(ch, current, compliance)

    def measure_current(self, channel):
        smu, ch = self._route(channel)
        return smu.measure_current(ch)

    def measure_voltage(self, channel):
        smu, ch = self._route(channel)
        return smu.measure_voltage(ch)

    def output_off(self, channel):
        smu, ch = self._route(channel)
        smu.output_off(ch)
