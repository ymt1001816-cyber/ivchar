"""Hardware drivers against a fake VISA resource (no instrument needed)."""

import pytest

from ivchar.instruments import B1500A, GenericSCPISMU, MultiSMU, Reading


class FakeResource:
    def __init__(self, replies=None):
        self.timeout = 0
        self.writes = []
        self.queries = []
        self.replies = dict(replies or {})
        self.closed = False

    def write(self, msg):
        self.writes.append(msg)

    def query(self, msg):
        self.queries.append(msg)
        for prefix, reply in self.replies.items():
            if msg.startswith(prefix):
                return reply
        raise AssertionError(f"unexpected query {msg!r}")

    def close(self):
        self.closed = True


def test_b1500_init_and_force_commands():
    res = FakeResource({"ERRX?": '0,"No Error."'})
    smu = B1500A(resource=res, channels=(1, 2, 3, 4), integration=("plc", 2))
    smu.connect()
    assert res.writes[:3] == ["*RST", "FMT 1,0", "AIT 1,2,2"]
    assert "CN 1,2,3,4" in res.writes
    smu.force_voltage(2, 1.0, 0.01)
    assert res.writes[-1] == "DV 2,0,1.000000E+00,1.000000E-02"
    smu.force_current(2, 1e-6, 2.0)
    assert res.writes[-1] == "DI 2,0,1.000000E-06,2.000000E+00"


def test_b1500_raises_on_instrument_error():
    res = FakeResource({"ERRX?": '120,"Incorrect parameter"'})
    with pytest.raises(RuntimeError, match="Incorrect parameter"):
        B1500A(resource=res).connect()


@pytest.mark.parametrize("raw, expected", [
    ("NAI+1.234500E-09", Reading(1.2345e-9, False)),
    ("CBI+1.000000E-02", Reading(1e-2, True)),     # this channel in compliance
    ("TAI-3.300000E-12", Reading(-3.3e-12, True)),  # another channel in compliance
    ("NBV+5.000000E-01", Reading(0.5, False)),
])
def test_b1500_parse(raw, expected):
    assert B1500A.parse(raw) == expected


def test_b1500_parse_rejects_garbage():
    with pytest.raises(ValueError):
        B1500A.parse("hello")


def test_b1500_measure_and_close():
    res = FakeResource({"ERRX?": "0", "TI 2": "NBI+2.500000E-04"})
    with B1500A(resource=res) as smu:
        assert smu.measure_current(2) == Reading(2.5e-4, False)
    assert "CL" in res.writes and res.closed
    assert [w for w in res.writes if w.startswith("DZ")] == ["DZ 1", "DZ 2", "DZ 3", "DZ 4"]


def test_generic_scpi_voltage_source_and_trip():
    res = FakeResource({":READ?": "1.0E-03", ":SENS:CURR:PROT:TRIP?": "1"})
    smu = GenericSCPISMU(resource=res)
    smu.connect()
    smu.force_voltage(1, 0.5, 1e-3)
    assert ":SOUR:FUNC VOLT" in res.writes and ":SENS:CURR:PROT 1.000000E-03" in res.writes
    assert smu.measure_current(1) == Reading(1e-3, True)
    with pytest.raises(ValueError):
        smu.force_voltage(2, 0.5, 1e-3)


def test_multi_smu_routes_channels():
    a, b = FakeResource({":READ?": "1e-6", ":SENS:CURR:PROT:TRIP?": "0"}), FakeResource(
        {":READ?": "2e-6", ":SENS:CURR:PROT:TRIP?": "0"})
    gate, drain = GenericSCPISMU(resource=a, name="gate"), GenericSCPISMU(resource=b, name="drain")
    multi = MultiSMU({1: (gate, 1), 2: (drain, 1)})
    assert multi.channels == [1, 2] and multi.simulated is False
    assert multi.measure_current(2).value == 2e-6
    assert multi.measure_current(1).value == 1e-6


def test_visa_smu_needs_address_or_resource():
    with pytest.raises(ValueError):
        B1500A()
