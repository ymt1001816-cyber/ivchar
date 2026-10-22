"""Batch statistics, report rendering, AI summary plumbing and the CLI."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from ivchar.analysis import BatchConfig, analyze, apply_spec, flag_outliers, run_batch, wafer_dies
from ivchar.cli import main


@pytest.fixture(scope="module")
def batch():
    cfg = BatchConfig(n_dies=12, temps_c=(25.0, 125.0), corners=("TT", "FF"), defect_rate=0.25, seed=3)
    params, curves = run_batch(cfg)
    return cfg, params, curves


def test_wafer_dies_centered_and_normalized():
    d = wafer_dies(21)
    assert len(d) == 21 and d.radius.max() == pytest.approx(1.0)
    assert (d.iloc[0][["x", "y"]] == 0).all()


def test_batch_shape_and_labels(batch):
    cfg, params, curves = batch
    assert len(params) == 12 * 2 * 2
    assert params.attrs["simulated"] is True
    # same die keeps its defect at every temperature
    per_die = params.groupby(["corner", "die"]).injected_defect.nunique()
    assert (per_die == 1).all()


def test_injected_defects_are_flagged(batch):
    _, params, _ = batch
    flagged = flag_outliers(params)
    defect_dies = params.loc[params.injected_defect != "", ["corner", "die"]].drop_duplicates()
    assert len(defect_dies) > 0
    hits = flagged[flagged.outlier][["corner", "die"]].drop_duplicates()
    assert len(defect_dies.merge(hits)) == len(defect_dies)


def test_outlier_flags_on_synthetic_group():
    df = pd.DataFrame({"corner": "TT", "temp_c": 25.0, "die": [f"D{i}" for i in range(20)],
                       "vth_cc_sat": np.r_[np.linspace(0.29, 0.31, 19), 0.45]})
    out = flag_outliers(df, cols=["vth_cc_sat"])
    assert out.outlier.tolist() == [False] * 19 + [True]
    assert "vth_cc_sat" in out.outlier_reason.iloc[-1]


def test_high_side_only_for_leakage():
    df = pd.DataFrame({"corner": "TT", "temp_c": 25.0, "die": [f"D{i}" for i in range(10)],
                       "ioff": [1e-10] * 9 + [1e-14]})  # one die leaks far LESS: not a defect
    assert not flag_outliers(df, cols=["ioff"]).outlier.any()


def test_apply_spec_reasons():
    df = pd.DataFrame({"corner": "TT", "temp_c": [25.0, 25.0, 125.0], "die": ["A", "B", "A"],
                       "vth_cc_sat": [0.30, 0.50, 0.50], "ioff": [1e-10, 1e-8, 1e-8]})
    rows = apply_spec(df, {"vth_cc_sat": (0.2, 0.4), "ioff": (None, 5e-9)})
    assert len(rows) == 2  # only the 25 C rows
    assert rows["pass"].tolist() == [True, False]
    assert rows.fail_reason.iloc[1] == "vth_cc_sat, ioff"


def test_report_renders_with_simulated_banner(batch, tmp_path):
    from ivchar.report import make_all, render

    cfg, params, curves = batch
    analysis = analyze(params, cfg.resolved_spec())
    figs = make_all(curves, analysis.params, tmp_path / "fig")
    html = render(analysis, figs, tmp_path / "r.html", True, "test").read_text()
    assert "SIMULATED DATA" in html and "data:image/png;base64," in html
    assert "Injected defects" in html


def test_llm_summary_with_fake_client(batch, monkeypatch):
    from ivchar.report import llm

    cfg, params, _ = batch
    analysis = analyze(params, cfg.resolved_spec())
    sent = {}

    class FakeMessages:
        def create(self, **kw):
            sent.update(kw)
            block = types.SimpleNamespace(type="text", text="- SIMULATED data: all good")
            return types.SimpleNamespace(stop_reason="end_turn", content=[block])

    fake = types.SimpleNamespace(Anthropic=lambda: types.SimpleNamespace(
        beta=types.SimpleNamespace(messages=FakeMessages())))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    text = llm.summarize_anomalies(analysis, simulated=True)
    assert text.startswith("- SIMULATED")
    assert sent["model"] == "claude-opus-5-5"
    assert "SIMULATED" in sent["messages"][0]["content"]


def test_llm_refusal_raises(batch, monkeypatch):
    from ivchar.report import llm

    cfg, params, _ = batch
    analysis = analyze(params, cfg.resolved_spec())
    resp = types.SimpleNamespace(stop_reason="refusal", content=[])
    fake = types.SimpleNamespace(Anthropic=lambda: types.SimpleNamespace(beta=types.SimpleNamespace(
        messages=types.SimpleNamespace(create=lambda **kw: resp))))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    with pytest.raises(RuntimeError, match="declined"):
        llm.summarize_anomalies(analysis, simulated=True)


def test_cli_run_end_to_end(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["run", "--dies", "6", "--temps", "25", "125", "--corners", "TT", "--out", str(out)]) == 0
    for name in ["report.html", "params.csv", "curves.csv", "summary.csv", "config.json"]:
        assert (out / name).exists()
    assert '"simulated": true' in (out / "config.json").read_text()
    assert "Report:" in capsys.readouterr().out


def test_cli_measure_and_analyze_roundtrip(tmp_path):
    csv = tmp_path / "dev.csv"
    assert main(["measure", "--out", str(csv)]) == 0
    out = tmp_path / "ana"
    assert main(["analyze", str(csv), "--w", "1e-6", "--l", "1e-7", "--vdd", "1.0",
                 "--spec", "planar", "--out", str(out)]) == 0
    params = pd.read_csv(out / "params.csv")
    assert len(params) == 1 and 0.2 < params.vth_cc_sat.iloc[0] < 0.4


def test_cli_measure_discrete_recipe(tmp_path):
    csv = tmp_path / "d.csv"
    assert main(["measure", "--device", "discrete", "--vdd", "3", "--vg-max", "4", "--vd-lin", "0.1",
                 "--vd-sat", "2", "--vd-max", "3", "--vg-out", "2.5", "3", "--step", "0.05",
                 "--id-max", "0.1", "--ig-max", "1e-5", "--i-crit", "1e-3", "--label", "Q1",
                 "--out", str(csv)]) == 0
    df = pd.read_csv(csv)
    assert set(df.die) == {"Q1"} and df.compliance.any()


def test_cli_scpi_backend_requires_both_resources(capsys):
    assert main(["measure", "--backend", "scpi", "--gate-resource", "GPIB0::24::INSTR"]) == 2
    assert "--drain-resource" in capsys.readouterr().err
