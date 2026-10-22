"""Command-line entry point for ivchar.

    ivchar run       simulated batch (dies x temperatures x corners) -> CSVs, figures, HTML report
    ivchar measure   one device on the simulator or a real instrument -> curves CSV
    ivchar analyze   extraction + report from curve CSVs (measured, SPICE or TCAD data)
    ivchar spice     Spectre/BSIM reference sweeps: write a netlist, import the results
    ivchar reliability  simulated BTI study: stress-measure-stress, recovery, lifetime, hysteresis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ivchar import __version__


def _write_outputs(analysis, curves, out: Path, simulated: bool, device: str, use_llm: bool) -> Path:
    from ivchar.report import make_all, render

    out.mkdir(parents=True, exist_ok=True)
    analysis.params.to_csv(out / "params.csv", index=False)
    analysis.summary.to_csv(out / "summary.csv")
    if len(analysis.spec_rows):
        analysis.spec_rows.to_csv(out / "spec_results.csv", index=False)
    figures = make_all(curves, analysis.params, out / "figures", simulated)

    ai_summary = ai_model = None
    if use_llm:
        from ivchar.report.llm import MODEL, summarize_anomalies

        try:
            ai_summary, ai_model = summarize_anomalies(analysis, simulated), MODEL
        except Exception as exc:  # the report is still useful without the summary
            print(f"warning: AI summary skipped: {exc}", file=sys.stderr)
    return render(analysis, figures, out / "report.html", simulated, device, ai_summary, ai_model)


def cmd_run(args) -> int:
    from ivchar.analysis import BatchConfig, analyze, run_batch

    cfg = BatchConfig(preset=args.device, n_dies=args.dies, temps_c=tuple(args.temps),
                      corners=tuple(args.corners), seed=args.seed, defect_rate=args.defect_rate)

    def progress(done, total):
        if done == total or done % 25 == 0:
            print(f"\r  measured {done}/{total} devices", end="", file=sys.stderr, flush=True)

    print(f"Simulated batch: {cfg.preset}, {cfg.n_dies} dies x {len(cfg.temps_c)} temps x "
          f"{len(cfg.corners)} corners (SIMULATED DATA)", file=sys.stderr)
    params, curves = run_batch(cfg, progress)
    print(file=sys.stderr)
    analysis = analyze(params, cfg.resolved_spec())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out / "curves.csv", index=False)
    (out / "config.json").write_text(json.dumps({"simulated": True, **params.attrs["config"]}, indent=2))
    report = _write_outputs(analysis, curves, out, True, f"{cfg.preset} NMOS (simulated)", args.llm)
    _print_summary(analysis, report)
    return 0


def cmd_analyze(args) -> int:
    from ivchar.analysis import DEFAULT_SPECS, analyze, extract_from_curves, normalize_curves

    curves = normalize_curves(pd.concat([pd.read_csv(f) for f in args.curves], ignore_index=True))
    params = extract_from_curves(curves, args.w, args.l, args.vdd, i_crit=args.i_crit)
    spec = DEFAULT_SPECS.get(args.spec, {}) if args.spec else {}
    analysis = analyze(params, spec)
    simulated = args.simulated
    report = _write_outputs(analysis, curves, Path(args.out), simulated,
                            args.label or Path(args.curves[0]).stem, args.llm)
    _print_summary(analysis, report)
    return 0


def cmd_measure(args) -> int:
    from ivchar.device import PRESETS, make_device
    from ivchar.extraction import extract
    from ivchar.instruments import B1500A, GenericSCPISMU, MultiSMU, SimulatedSMU
    from ivchar.measure import Compliance, Terminals, standard_sweeps

    p = PRESETS[args.device]
    terminals = Terminals()
    if args.backend == "sim":
        dev = make_device(args.device, args.corner, args.temp)
        smu = SimulatedSMU(dev, seed=args.seed)
    elif args.backend == "b1500":
        if not args.resource:
            print("error: --resource is required for the b1500 backend", file=sys.stderr)
            return 2
        smu = B1500A(args.resource)
    else:  # two single-channel SCPI SMUs on gate and drain; source/body grounded in the fixture
        if not (args.gate_resource and args.drain_resource):
            print("error: --gate-resource and --drain-resource are required for the scpi backend",
                  file=sys.stderr)
            return 2
        smu = MultiSMU({1: (GenericSCPISMU(args.gate_resource, name="gate-smu", nplc=args.nplc), 1),
                        2: (GenericSCPISMU(args.drain_resource, name="drain-smu", nplc=args.nplc), 1)})
        terminals = Terminals(gate=1, drain=2, source=None, body=None)

    w, l = args.w or p.w, args.l or p.l
    vdd = args.vdd or p.vdd
    compliance = Compliance(gate=args.ig_max, drain=args.id_max, source=args.id_max)
    recipe = dict(vd_lin=args.vd_lin, step=args.step, vg_max=args.vg_max, vd_sat=args.vd_sat,
                  vd_max=args.vd_max, vg_out=args.vg_out, terminals=terminals,
                  compliance=compliance, settle_time=args.settle)
    with smu:
        curves = standard_sweeps(smu, vdd, **recipe)
    curves["temp_c"] = args.temp
    curves["die"] = args.label or "D000"
    curves.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(curves)} points, {'SIMULATED' if smu.simulated else smu.name})")
    for k, v in extract(curves, w, l, vdd, i_crit=args.i_crit).items():
        print(f"  {k:12s} {v:.4g}")
    return 0


def cmd_spice_netlist(args) -> int:
    from ivchar.spice import write_netlist_dir

    out = write_netlist_dir(Path(args.out), model_file=args.model_file, model=args.model, w=args.w,
                            l=args.l, vdd=args.vdd, section=args.section or None, temp=args.temp,
                            vd_lin=args.vd_lin, step=args.step)
    print(f"wrote {out}/ivchar_ref.scs and run.sh; run 'sh run.sh' where Spectre is available")
    return 0


def cmd_spice_import(args) -> int:
    from ivchar.spice import import_results

    df = import_results(Path(args.raw_dir), args.vd_lin, args.vdd, die=args.label, temp=args.temp)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} points, SIMULATED with BSIM)")
    return 0


def cmd_reliability(args) -> int:
    from ivchar.reliability import StudyConfig, run_study, summarize_study
    from ivchar.report.reliability import make_figures, render

    cfg = StudyConfig(preset=args.device, temp_c=args.temp, stress_vg=tuple(args.stress_vg),
                      use_vg=args.use_vg, criterion=args.criterion, point_time=args.point_time)
    print(f"Simulated BTI study: {cfg.preset}, {cfg.temp_c:g} C, stress Vg {list(cfg.stress_vg)} "
          "(SIMULATED DATA)", file=sys.stderr)
    study = run_study(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    study["msm"].to_csv(out / "msm.csv", index=False)
    study["hysteresis"].to_csv(out / "hysteresis.csv", index=False)
    figures = make_figures(study, out / "figures")
    report = render(study, figures, out / "reliability_report.html")
    s = summarize_study(study)
    print(f"ΔVth after {max(cfg.stress_times):g} s at {s['vg_mid']:g} V: "
          f"{1e3 * s['dvth_end_fast']:.1f} mV; {100 * s['recovered_fraction_1000s']:.0f}% recovers in 1000 s")
    for label, years in s["ttf_years_by_method"].items():
        print(f"  projected lifetime ({label}): {years:.3g} years")
    print(f"Report: {report.resolve()}")
    return 0


def _print_summary(analysis, report: Path) -> None:
    if len(analysis.yields):
        print("\nYield at %g C:" % analysis.spec_temp_c)
        print(analysis.yields.round(1).to_string())
    flagged = analysis.params[analysis.params.outlier]
    print(f"\nFlagged outlier rows: {len(flagged)} "
          f"({flagged.drop_duplicates(['corner', 'die']).shape[0]} dies)")
    print(f"Report: {report.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ivchar", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"ivchar {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="simulated batch characterization + report")
    run.add_argument("--device", choices=["planar", "finfet"], default="planar")
    run.add_argument("--dies", type=int, default=25)
    run.add_argument("--temps", type=float, nargs="+", default=[-40.0, 25.0, 125.0])
    run.add_argument("--corners", nargs="+", default=["TT", "FF", "SS"],
                     choices=["TT", "FF", "SS", "FS", "SF"])
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--defect-rate", type=float, default=0.04)
    run.add_argument("--out", default="output")
    run.add_argument("--llm", action="store_true", help="add an AI anomaly summary (needs ivchar[llm])")
    run.set_defaults(func=cmd_run)

    ana = sub.add_parser("analyze", help="extract + report from a curves CSV")
    ana.add_argument("curves", nargs="+",
                     help="one or more CSVs with columns sweep,vg,vd,id[,ig,compliance,corner,die,temp_c,x,y]")
    ana.add_argument("--w", type=float, required=True, help="device width (m)")
    ana.add_argument("--l", type=float, required=True, help="channel length (m)")
    ana.add_argument("--vdd", type=float, required=True, help="Vg for Ion (V)")
    ana.add_argument("--i-crit", type=float,
                     help="fixed CC threshold current (A); default 100 nA * W/L")
    ana.add_argument("--spec", choices=["planar", "finfet"], help="apply a built-in spec")
    ana.add_argument("--simulated", action=argparse.BooleanOptionalAction, default=True,
                     help="label the report as simulated (default); --no-simulated for measured data")
    ana.add_argument("--label", help="device description for the report header")
    ana.add_argument("--out", default="output_analyze")
    ana.add_argument("--llm", action="store_true")
    ana.set_defaults(func=cmd_analyze)

    mea = sub.add_parser("measure", help="standard sweeps on one device")
    mea.add_argument("--backend", choices=["sim", "b1500", "scpi"], default="sim",
                     help="sim, Keysight B1500A, or two single-channel SCPI SMUs (gate + drain)")
    mea.add_argument("--resource", help="B1500A VISA address, e.g. GPIB0::17::INSTR")
    mea.add_argument("--gate-resource", help="VISA address of the gate SMU (scpi backend)")
    mea.add_argument("--drain-resource", help="VISA address of the drain SMU (scpi backend)")
    mea.add_argument("--nplc", type=float, default=1.0, help="integration time in power-line cycles")
    mea.add_argument("--device", choices=["planar", "finfet", "discrete"], default="planar",
                     help="simulator preset; also the default W, L and Vdd")
    mea.add_argument("--corner", default="TT")
    mea.add_argument("--temp", type=float, default=25.0, help="device temperature (C), recorded in the CSV")
    mea.add_argument("--seed", type=int, default=0)
    mea.add_argument("--label", help="device ID recorded in the CSV")
    mea.add_argument("--w", type=float)
    mea.add_argument("--l", type=float)
    mea.add_argument("--vdd", type=float, help="Vg for Ion (V)")
    mea.add_argument("--vg-max", type=float, help="Id-Vg gate sweep end (V); default Vdd")
    mea.add_argument("--vd-lin", type=float, default=0.05, help="linear-region drain bias (V)")
    mea.add_argument("--vd-sat", type=float, help="saturation drain bias for Id-Vg (V); default Vdd")
    mea.add_argument("--vd-max", type=float, help="Id-Vd drain sweep end (V); default Vdd")
    mea.add_argument("--vg-out", type=float, nargs="+", help="gate steps for the Id-Vd family (V)")
    mea.add_argument("--step", type=float, default=0.01, help="Vg step (V); Vd step is twice this")
    mea.add_argument("--id-max", type=float, default=20e-3, help="drain compliance (A)")
    mea.add_argument("--ig-max", type=float, default=1e-6, help="gate compliance (A)")
    mea.add_argument("--settle", type=float, default=0.0, help="delay after each step (s)")
    mea.add_argument("--i-crit", type=float, help="fixed CC threshold current (A)")
    mea.add_argument("--out", default="curves.csv")
    mea.set_defaults(func=cmd_measure)

    rel = sub.add_parser("reliability", help="simulated BTI / hysteresis study + report")
    rel.add_argument("--device", choices=["planar", "finfet"], default="planar")
    rel.add_argument("--temp", type=float, default=125.0)
    rel.add_argument("--stress-vg", type=float, nargs="+", default=[1.6, 1.8, 2.0])
    rel.add_argument("--use-vg", type=float, default=1.0, help="use voltage for lifetime projection (V)")
    rel.add_argument("--criterion", type=float, default=0.03, help="failure criterion ΔVth (V)")
    rel.add_argument("--point-time", type=float, default=1e-3, help="seconds per bias step")
    rel.add_argument("--out", default="output_reliability")
    rel.set_defaults(func=cmd_reliability)

    sp = sub.add_parser("spice", help="Spectre/BSIM reference sweeps")
    sps = sp.add_subparsers(dest="spice_command", required=True)
    net = sps.add_parser("netlist", help="write a Spectre netlist + run.sh")
    net.add_argument("--model-file", required=True, help="PDK model include file (.scs)")
    net.add_argument("--section", default="tt", help="model library section ('' for none)")
    net.add_argument("--model", required=True, help="model name, e.g. g45n1svt")
    net.add_argument("--w", type=float, required=True)
    net.add_argument("--l", type=float, required=True)
    net.add_argument("--vdd", type=float, required=True)
    net.add_argument("--vd-lin", type=float, default=0.05)
    net.add_argument("--step", type=float, default=0.01)
    net.add_argument("--temp", type=float, default=25.0)
    net.add_argument("--out", default="spice_ref")
    net.set_defaults(func=cmd_spice_netlist)
    imp = sps.add_parser("import", help="convert Spectre PSF-ASCII results to a curves CSV")
    imp.add_argument("raw_dir", help="Spectre raw directory (contains idvg_vd0.dc ...)")
    imp.add_argument("--vdd", type=float, required=True)
    imp.add_argument("--vd-lin", type=float, default=0.05)
    imp.add_argument("--temp", type=float, default=25.0)
    imp.add_argument("--label", default="BSIM")
    imp.add_argument("--out", default="bsim_curves.csv")
    imp.set_defaults(func=cmd_spice_import)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
