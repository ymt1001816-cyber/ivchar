"""Command-line entry point for ivchar.

    ivchar run       simulated batch (dies x temperatures x corners) -> CSVs, figures, HTML report
    ivchar measure   one device on the simulator or a real instrument -> curves CSV
    ivchar analyze   extraction + report from a curves CSV (measured, SPICE or TCAD data)
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

    curves = normalize_curves(pd.read_csv(args.curves))
    params = extract_from_curves(curves, args.w, args.l, args.vdd)
    spec = DEFAULT_SPECS.get(args.spec, {}) if args.spec else {}
    analysis = analyze(params, spec)
    simulated = args.simulated
    report = _write_outputs(analysis, curves, Path(args.out), simulated,
                            args.label or Path(args.curves).stem, args.llm)
    _print_summary(analysis, report)
    return 0


def cmd_measure(args) -> int:
    from ivchar.device import PRESETS, make_device
    from ivchar.extraction import extract
    from ivchar.instruments import B1500A, SimulatedSMU
    from ivchar.measure import standard_sweeps

    if args.backend == "sim":
        dev = make_device(args.device, args.corner, args.temp)
        smu = SimulatedSMU(dev, seed=args.seed)
        w, l, vdd = dev.params.w, dev.params.l, dev.params.vdd
    else:
        if not args.resource:
            print("error: --resource is required for the b1500 backend", file=sys.stderr)
            return 2
        smu = B1500A(args.resource)
        p = PRESETS[args.device]
        w, l = args.w or p.w, args.l or p.l
        vdd = args.vdd or p.vdd

    with smu:
        curves = standard_sweeps(smu, vdd)
    curves.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(curves)} points, {'SIMULATED' if smu.simulated else smu.name})")
    for k, v in extract(curves, w, l, vdd).items():
        print(f"  {k:12s} {v:.4g}")
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
    ana.add_argument("curves", help="CSV with columns sweep,vg,vd,id[,ig,compliance,corner,die,temp_c,x,y]")
    ana.add_argument("--w", type=float, required=True, help="device width (m)")
    ana.add_argument("--l", type=float, required=True, help="channel length (m)")
    ana.add_argument("--vdd", type=float, required=True)
    ana.add_argument("--spec", choices=["planar", "finfet"], help="apply a built-in spec")
    ana.add_argument("--simulated", action=argparse.BooleanOptionalAction, default=True,
                     help="label the report as simulated (default); --no-simulated for measured data")
    ana.add_argument("--label", help="device description for the report header")
    ana.add_argument("--out", default="output_analyze")
    ana.add_argument("--llm", action="store_true")
    ana.set_defaults(func=cmd_analyze)

    mea = sub.add_parser("measure", help="standard sweeps on one device")
    mea.add_argument("--backend", choices=["sim", "b1500"], default="sim")
    mea.add_argument("--resource", help="VISA address, e.g. GPIB0::17::INSTR")
    mea.add_argument("--device", choices=["planar", "finfet"], default="planar")
    mea.add_argument("--corner", default="TT")
    mea.add_argument("--temp", type=float, default=25.0)
    mea.add_argument("--seed", type=int, default=0)
    mea.add_argument("--w", type=float)
    mea.add_argument("--l", type=float)
    mea.add_argument("--vdd", type=float)
    mea.add_argument("--out", default="curves.csv")
    mea.set_defaults(func=cmd_measure)
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
