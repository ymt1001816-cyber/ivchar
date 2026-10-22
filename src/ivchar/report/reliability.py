"""Figures and HTML report for a BTI / hysteresis study."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from jinja2 import Template

from ivchar import __version__
from ivchar.reliability import METHODS, fit_power_law, summarize_study
from ivchar.report import plots
from ivchar.report.html import CSS, _b64, _table

plt = plots.plt
METHOD_COLORS = dict(zip(METHODS, [plots.CORNER_COLORS["TT"], plots.CORNER_COLORS["FF"],
                                   plots.CORNER_COLORS["SS"]]))
FAST = "spot, 1 ms delay"


def _vg_colors(vgs):
    return plots._temp_colors(sorted(vgs))  # ordered stress levels -> sequential ramp


def plot_dvth_vs_time(study, out: Path) -> Path:
    msm = study["msm"]
    s = msm[(msm.method == FAST) & (msm.phase == "stress")]
    colors = _vg_colors(s.stress_vg.unique())
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for vg, g in s.groupby("stress_vg"):
        a, n = fit_power_law(g.time_s, g.dvth)
        t = np.logspace(np.log10(g.time_s.min()), np.log10(g.time_s.max()), 50)
        ax.loglog(t, 1e3 * a * t**n, color=colors[vg], linewidth=1)
        ax.loglog(g.time_s, 1e3 * g.dvth, "o", color=colors[vg], markersize=5,
                  markeredgecolor=plots.SURFACE, label=f"Vg = {vg:g} V (n = {n:.2f})")
    ax.set(title=f"ΔVth vs stress time ({FAST})", xlabel="Stress time (s)", ylabel="ΔVth (mV)")
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend()
    plots._sim_note(fig, True)
    return plots._save(fig, out)


def plot_delay_effect(study, vg: float, out: Path) -> Path:
    msm = study["msm"]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for label in METHODS:
        g = msm[(msm.method == label) & (msm.phase == "stress") & np.isclose(msm.stress_vg, vg)]
        ax.semilogx(g.time_s, 1e3 * g.dvth, "-o", color=METHOD_COLORS[label], markersize=5,
                    markeredgecolor=plots.SURFACE, label=label)
    ax.set(title=f"Three ways to measure, {vg:g} V stress", xlabel="Stress time (s)",
           ylabel="Measured ΔVth (mV)")
    ax.legend()
    plots._sim_note(fig, True)
    return plots._save(fig, out)


def plot_recovery(study, out: Path) -> Path:
    msm = study["msm"]
    r = msm[(msm.method == FAST) & (msm.phase == "recovery")]
    s_end = msm[(msm.method == FAST) & (msm.phase == "stress")].groupby("stress_vg").dvth.last()
    colors = _vg_colors(r.stress_vg.unique())
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for vg, g in r.groupby("stress_vg"):
        ax.semilogx(g.time_s, 100 * g.dvth / s_end[vg], "-o", color=colors[vg], markersize=5,
                    markeredgecolor=plots.SURFACE, label=f"after {vg:g} V stress")
    ax.set(title="Recovery after stress (gate at 0 V)", xlabel="Recovery time (s)",
           ylabel="Remaining ΔVth (% of end of stress)", ylim=(0, 105))
    ax.legend()
    plots._sim_note(fig, True)
    return plots._save(fig, out)


def plot_lifetime(study, out: Path) -> Path:
    cfg = study["config"]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    vgs = np.linspace(cfg.use_vg, max(cfg.stress_vg), 50)
    for label, lt in study["lifetimes"].items():
        f = lt["fits"]
        intercept = np.log10(lt["ttf_use_s"]) - lt["log10_ttf_slope_per_V"] * cfg.use_vg
        ax.semilogy(vgs, 10 ** (intercept + lt["log10_ttf_slope_per_V"] * vgs), color=METHOD_COLORS[label],
                    linewidth=1)
        ax.semilogy(f.stress_vg, f.ttf_s, "o", color=METHOD_COLORS[label], markersize=5,
                    markeredgecolor=plots.SURFACE, label=f"{label}: {lt['ttf_use_years']:.0f} y")
    ax.axhline(10 * 3.156e7, color=plots.TEXT_2, linewidth=1)
    ax.annotate("10 years", (vgs[-1], 10 * 3.156e7), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=7, color=plots.TEXT_2)
    ax.axvline(cfg.use_vg, color=plots.GRID, linewidth=1)
    ax.set(title=f"Time to ΔVth = {1e3 * cfg.criterion:g} mV, extrapolated to {cfg.use_vg:g} V",
           xlabel="Gate stress voltage (V)", ylabel="Time to failure (s)")
    ax.legend(fontsize=7)
    plots._sim_note(fig, True)
    return plots._save(fig, out)


def plot_hysteresis(study, out: Path) -> Path:
    curves = study["hysteresis_curves"]
    pt = curves.point_time.max()
    c = curves[curves.point_time == pt]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for direction, color in (("forward", plots.CORNER_COLORS["TT"]), ("reverse", plots.CORNER_COLORS["FF"])):
        d = c[c.direction == direction]
        ax.semilogy(d.vg, np.clip(d.id, 1e-15, None), color=color, label=direction)
    h = study["hysteresis"].set_index("point_time_s").dvth_hysteresis
    ax.set(title=f"Hysteresis, {pt:g} s per point: ΔVth = {1e3 * h[pt]:.1f} mV",
           xlabel="Vg (V)", ylabel="Id (A)")
    ax.legend(loc="lower right")
    plots._sim_note(fig, True)
    return plots._save(fig, out)


def make_figures(study, out_dir: Path) -> dict[str, Path]:
    plots._style()
    out_dir.mkdir(parents=True, exist_ok=True)
    vg_mid = sorted(study["config"].stress_vg)[len(study["config"].stress_vg) // 2]
    return {
        "dvth_time": plot_dvth_vs_time(study, out_dir / "bti_dvth_vs_time.png"),
        "delay": plot_delay_effect(study, vg_mid, out_dir / "bti_measurement_delay.png"),
        "recovery": plot_recovery(study, out_dir / "bti_recovery.png"),
        "lifetime": plot_lifetime(study, out_dir / "bti_lifetime.png"),
        "hysteresis": plot_hysteresis(study, out_dir / "hysteresis.png"),
    }


TEMPLATE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTI Reliability Report</title><style>{{ css }}</style></head>
<body><main>
<h1>BTI Reliability Report</h1>
<div class="meta">{{ device }} · {{ temp }} °C · generated {{ generated }} · ivchar {{ version }}</div>
<div class="banner">⚠ SIMULATED DATA. Degradation comes from ivchar's defect-based BTI model on the
compact-model simulator, not from stressed silicon. It shows what the test flow measures, not device results.</div>
<div class="kpis">{% for k in kpis %}<div class="kpi"><b>{{ k.value }}</b><span>{{ k.label }}</span></div>{% endfor %}</div>
<h2>Lifetime projection (criterion ΔVth = {{ crit }} mV at Vg = {{ use_vg }} V)</h2>
<div class="table-wrap">{{ lifetime_html }}</div>
<h2>Figures</h2>
<div class="grid">{% for f in figures %}<figure><img src="data:image/png;base64,{{ f }}" alt=""></figure>{% endfor %}</div>
<h2>Hysteresis vs sweep speed</h2>
<div class="table-wrap">{{ hyst_html }}</div>
<h2>Method</h2>
<ul class="meta">
<li>Measure-stress-measure: constant gate stress with source, drain and body grounded; stress is interrupted at
log-spaced times to sense ΔVth at Vd = 50 mV.</li>
<li>Spot sensing: one Id reading at the fresh CC threshold, converted to ΔVth on the fresh Id-Vg curve.
Full sweep: CC Vth from a complete Id-Vg sweep.</li>
<li>Lifetime: power-law fit ΔVth = A·tⁿ per stress voltage, time to the criterion, log(TTF) linear in Vg
extrapolated to the use voltage.</li>
</ul>
</main></body></html>
""")


def render(study, figures: dict[str, Path], out_path: Path) -> Path:
    cfg = study["config"]
    s = summarize_study(study)
    rows = []
    for label, lt in study["lifetimes"].items():
        f = lt["fits"]
        rows.append({"method": label, "n (power-law exponent)": ", ".join(f"{x:.3f}" for x in f.n),
                     "TTF at stress (s)": ", ".join(f"{x:.3g}" for x in f.ttf_s),
                     "projected lifetime (years)": lt["ttf_use_years"]})
    import pandas as pd

    first, last = s["dvth_first_last_by_method"][FAST]
    kpis = [
        {"value": f"{1e3 * last:.0f} mV", "label": f"ΔVth after {max(cfg.stress_times):g} s at {s['vg_mid']:g} V"},
        {"value": f"{100 * s['recovered_fraction_1000s']:.0f}%", "label": "recovered after 1000 s"},
        {"value": " / ".join(f"{y:.0f}" for y in s["ttf_years_by_method"].values()),
         "label": "projected years: " + " / ".join(METHODS)},
    ]
    hyst = study["hysteresis"].assign(dvth_hysteresis_mV=lambda d: 1e3 * d.dvth_hysteresis).drop(
        columns="dvth_hysteresis")
    html = TEMPLATE.render(
        css=CSS, device=f"{cfg.preset} NMOS (simulated), PBTI", temp=f"{cfg.temp_c:g}",
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), version=__version__, kpis=kpis,
        crit=f"{1e3 * cfg.criterion:g}", use_vg=f"{cfg.use_vg:g}",
        lifetime_html=_table(pd.DataFrame(rows)), hyst_html=_table(hyst),
        figures=[_b64(p) for p in figures.values()],
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path
