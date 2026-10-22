"""Physics-based compact model of an n-channel MOSFET / FinFET.

This is the device behind the simulated backend. It is an EKV-style charge
model: a single expression that is exponential in subthreshold, quadratic in
strong inversion and continuous in between, so every extraction method sees a
realistic curve shape.

    Id = Ispec * [F(xf) - F(xr)] * (1 + lambda*|Vds|)
    F(x) = ln(1 + exp(x/2))^2
    xf = (Vp - Vsb) / Ut,   xr = (Vp - Vdb) / Ut,   Vp = (Vgb - Vth_eff) / n
    Ispec = 2 * n * mu_eff * Cox * (W/L) * Ut^2

Effects included:
  * subthreshold slope factor n            -> SS = n * (kT/q) * ln(10)
  * DIBL: Vth_eff = Vth - eta * |Vds|
  * mobility degradation: mu_eff = mu / (1 + theta * Vov)
  * channel-length modulation (lambda)
  * temperature: Ut = kT/q, Vth(T) = Vth(300 K) - kappa*(T - 300), mu ~ T^-1.5
  * gate leakage and drain-body junction leakage (sets the Ioff floor)

All data produced from this model is SIMULATED, not measured silicon.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

K_B = 1.380649e-23  # J/K
Q_E = 1.602176634e-19  # C
EPS_OX = 3.9 * 8.854e-12  # F/m, SiO2
T_REF = 300.0  # K


def thermal_voltage(temp_k: float) -> float:
    return K_B * temp_k / Q_E


@dataclass(frozen=True)
class MOSFETParams:
    """Compact-model parameters at T_REF (300 K). SI units."""

    name: str
    w: float  # effective width (m); for a FinFET, nfin * (2*Hfin + Wfin)
    l: float  # channel length (m)
    tox: float  # equivalent oxide thickness (m)
    vth0: float  # threshold voltage at Vds -> 0 (V)
    n: float  # subthreshold slope factor
    mu0: float  # low-field mobility (m^2/V/s)
    theta: float  # mobility degradation (1/V)
    eta: float  # DIBL coefficient (V/V)
    lam: float  # channel-length modulation (1/V)
    kappa: float  # Vth temperature coefficient (V/K)
    g_gate: float  # gate leakage conductance (S)
    i_junction: float  # drain-body junction saturation current (A)
    vdd: float  # nominal supply (V)

    @property
    def cox(self) -> float:
        return EPS_OX / self.tox


# Representative (not foundry) parameter sets.
PRESETS = {
    "planar": MOSFETParams(
        name="planar", w=1e-6, l=100e-9, tox=1.6e-9, vth0=0.45, n=1.35, mu0=0.030,
        theta=0.9, eta=0.08, lam=0.10, kappa=0.9e-3, g_gate=1e-13, i_junction=2e-15, vdd=1.0,
    ),
    "finfet": MOSFETParams(
        # 4 fins, Hfin 42 nm, Wfin 8 nm -> W_eff = 4 * (2*42 + 8) nm = 368 nm
        name="finfet", w=368e-9, l=20e-9, tox=1.0e-9, vth0=0.35, n=1.10, mu0=0.012,
        theta=1.2, eta=0.035, lam=0.06, kappa=0.7e-3, g_gate=5e-14, i_junction=5e-16, vdd=0.8,
    ),
    # Roughly a 2N7000-class small-signal discrete MOSFET (Vth ~2 V, long
    # channel, thick oxide). Illustrative, not fitted to any datasheet; used to
    # dry-run the discrete-device measurement recipe on the simulator.
    "discrete": MOSFETParams(
        name="discrete", w=0.08, l=10e-6, tox=50e-9, vth0=2.1, n=1.6, mu0=0.05,
        theta=0.15, eta=0.004, lam=0.01, kappa=3.5e-3, g_gate=1e-12, i_junction=1e-11, vdd=5.0,
    ),
}


# Process corners for the n-channel device: (Vth shift in V, mobility scale).
# FS = fast NMOS / slow PMOS, SF = slow NMOS / fast PMOS; only the NMOS side
# is modelled here.
CORNERS = {
    "TT": (0.000, 1.00),
    "FF": (-0.035, 1.08),
    "SS": (+0.035, 0.92),
    "FS": (-0.035, 1.08),
    "SF": (+0.035, 0.92),
}


def apply_corner(p: MOSFETParams, corner: str) -> MOSFETParams:
    try:
        dvth, mu_scale = CORNERS[corner]
    except KeyError:
        raise ValueError(f"unknown corner {corner!r}; choose from {list(CORNERS)}") from None
    return replace(p, vth0=p.vth0 + dvth, mu0=p.mu0 * mu_scale)


def _softplus(x):
    """ln(1 + e^x), numerically stable for large |x|."""
    return np.logaddexp(0.0, x)


@dataclass
class MOSFET:
    """A device instance at a given temperature."""

    params: MOSFETParams
    temp_c: float = 25.0

    @property
    def temp_k(self) -> float:
        return self.temp_c + 273.15

    def channel_current(self, vg, vd, vs=0.0, vb=0.0):
        """Drain-to-source channel current (A). Accepts scalars or arrays."""
        p = self.params
        t = self.temp_k
        ut = thermal_voltage(t)
        vg, vd, vs, vb = (np.asarray(x, dtype=float) for x in (vg, vd, vs, vb))

        vds = np.abs(vd - vs)
        vth = p.vth0 - p.kappa * (t - T_REF) - p.eta * vds
        # Smooth overdrive: ~0 in subthreshold, Vgs - Vth in strong inversion.
        # Referenced to the lower of S/D so the device stays symmetric.
        vov = 2 * p.n * ut * _softplus((vg - np.minimum(vs, vd) - vth) / (2 * p.n * ut))
        mu = p.mu0 * (t / T_REF) ** -1.5 / (1 + p.theta * vov)
        ispec = 2 * p.n * mu * p.cox * (p.w / p.l) * ut**2

        vp = (vg - vb - vth) / p.n
        xf = (vp - (vs - vb)) / ut
        xr = (vp - (vd - vb)) / ut
        return ispec * (_softplus(xf / 2) ** 2 - _softplus(xr / 2) ** 2) * (1 + p.lam * vds)

    def terminal_currents(self, vg, vd, vs=0.0, vb=0.0):
        """Currents flowing *into* each terminal (A), SMU sign convention.

        Returns a dict with keys gate/drain/source/body. They sum to zero (KCL).
        """
        p = self.params
        t = self.temp_k
        ut = thermal_voltage(t)
        vg, vd, vs, vb = (np.asarray(x, dtype=float) for x in (vg, vd, vs, vb))

        ich = self.channel_current(vg, vd, vs, vb)
        # Gate leakage roughly doubles every ~25 K; junction leakage follows ni^2,
        # i.e. much stronger temperature dependence (activation ~ Eg).
        g_gate = p.g_gate * 2 ** ((t - T_REF) / 25)
        i_j0 = p.i_junction * (t / T_REF) ** 3 * np.exp(-(1.12 / thermal_voltage(t)) + 1.12 / thermal_voltage(T_REF))
        i_gd = g_gate * (vg - vd)
        i_gs = g_gate * (vg - vs)
        i_db = i_j0 * (1 - np.exp(-(vd - vb) / ut))  # reverse-biased diode for Vdb > 0

        return {
            "gate": i_gd + i_gs,
            "drain": ich - i_gd + i_db,
            "source": -ich - i_gs,
            "body": -i_db,
        }


def make_device(preset: str = "planar", corner: str = "TT", temp_c: float = 25.0,
                dvth: float = 0.0, mu_scale: float = 1.0, gate_leak_scale: float = 1.0) -> MOSFET:
    """Build a device from a preset, a corner and optional per-die variation."""
    try:
        p = PRESETS[preset]
    except KeyError:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}") from None
    p = apply_corner(p, corner)
    p = replace(p, vth0=p.vth0 + dvth, mu0=p.mu0 * mu_scale, g_gate=p.g_gate * gate_leak_scale)
    return MOSFET(p, temp_c)
