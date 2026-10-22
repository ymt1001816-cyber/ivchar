"""Reproduce the numbers in the README "Findings" section (SIMULATED data).

    python examples/findings.py

Runs the default simulated batch (planar NMOS, 25 dies, -40/25/125 C, TT/FF/SS,
seed 0) and prints the method comparisons and temperature behavior.
"""

import numpy as np

from ivchar.analysis import BatchConfig, analyze, run_batch
from ivchar.device import PRESETS, thermal_voltage

K_OVER_Q = 8.617333e-5  # V/K


def main():
    cfg = BatchConfig()
    params, curves = run_batch(cfg)
    a = analyze(params, cfg.resolved_spec())
    p = a.params[~a.params.outlier]
    model = PRESETS[cfg.preset]
    temps = sorted(p.temp_c.unique())
    t_lo, t_hi = temps[0], temps[-1]
    tt = p[p.corner == "TT"].groupby("temp_c").median(numeric_only=True)

    print("## 1. Vth: constant-current vs max-gm extrapolation (TT, median die, Vd = 50 mV)")
    for t in temps:
        d = tt.loc[t]
        print(f"  {t:6.0f} C  CC {d.vth_cc_lin * 1e3:6.1f} mV   LE {d.vth_gm_lin * 1e3:6.1f} mV   "
              f"LE - CC {1e3 * (d.vth_gm_lin - d.vth_cc_lin):5.1f} mV")

    print("\n## 2. Vth temperature coefficient by method (TT)")
    dt = t_hi - t_lo
    for col, label in [("vth_cc_lin", "CC (lin)"), ("vth_cc_sat", "CC (sat)"), ("vth_gm_lin", "max-gm LE")]:
        tc = 1e3 * (tt.loc[t_hi, col] - tt.loc[t_lo, col]) / dt
        print(f"  {label:10s} {tc:6.2f} mV/K")
    print(f"  model kappa {-model.kappa * 1e3:6.2f} mV/K (the physical Vth shift built into the simulator)")

    print("\n## 3. Subthreshold swing vs temperature (TT)")
    for t in temps:
        theory = model.n * thermal_voltage(t + 273.15) * np.log(10) * 1e3
        print(f"  {t:6.0f} C  SS {tt.loc[t, 'ss_sat']:6.1f} mV/dec   n*kT/q*ln10 {theory:6.1f} mV/dec")
    slope = (tt.loc[t_hi, "ss_sat"] - tt.loc[t_lo, "ss_sat"]) / dt
    print(f"  slope {slope * 1e3:.0f} uV/dec/K vs n*(k/q)*ln10 = {model.n * K_OVER_Q * np.log(10) * 1e6:.0f} uV/dec/K")

    print("\n## 4. Zero-temperature-coefficient (ZTC) point (TT median die, Vd = Vdd)")
    die = p[(p.corner == "TT") & (p.temp_c == 25.0)].set_index("die").vth_cc_sat
    die = (die - die.median()).abs().idxmin()
    sat = curves[(curves.corner == "TT") & (curves.die == die) & (curves.sweep == "idvg")
                 & np.isclose(curves.vd, model.vdd)]
    lo = sat[sat.temp_c == t_lo].sort_values("vg")
    hi = sat[sat.temp_c == t_hi].sort_values("vg")
    diff = hi.id.to_numpy() - lo.id.to_numpy()
    vg = lo.vg.to_numpy()
    k = np.nonzero((diff[:-1] > 0) & (diff[1:] <= 0))[0][0]
    vg_ztc = vg[k] + (vg[k + 1] - vg[k]) * diff[k] / (diff[k] - diff[k + 1])
    i_ztc = np.interp(vg_ztc, vg, lo.id)
    print(f"  die {die}: Vg_ZTC = {vg_ztc:.3f} V, Id = {i_ztc * 1e6:.0f} uA")
    for v in (0.5, round(vg_ztc, 2), model.vdd):
        i1, i2 = np.interp(v, vg, lo.id), np.interp(v, vg, hi.id)
        print(f"  Vg = {v:4.2f} V: Id changes {100 * (i2 / i1 - 1):+7.1f} % from {t_lo:.0f} to {t_hi:.0f} C")

    print("\n## 5. Outlier detection vs injected defects")
    flagged = a.params[a.params.outlier][["corner", "die"]].drop_duplicates()
    injected = a.params[a.params.injected_defect != ""][["corner", "die", "injected_defect"]].drop_duplicates()
    caught = injected.merge(flagged)
    print(f"  injected {len(injected)}, caught {len(caught)}, extra flagged dies {len(flagged) - len(caught)}")


if __name__ == "__main__":
    main()
