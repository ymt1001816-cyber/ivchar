"""Spectre netlist generation and PSF-ASCII import (synthetic result files)."""

import numpy as np
import pytest

from ivchar.spice import import_results, parse_psfascii, spectre_netlist, write_netlist_dir


def psf(sweep, sweep_vals, traces):
    lines = ['HEADER', '"PSFversion" "1.00"', 'TYPE', '"sweep" FLOAT DOUBLE PROP( "key" "sweep" )',
             'SWEEP', f'"{sweep}" "sweep"', 'TRACE']
    lines += [f'"{name}" "I"' for name in traces]
    lines.append('VALUE')
    for k, v in enumerate(sweep_vals):
        lines.append(f'"{sweep}" {v:.15e}')
        lines += [f'"{name}" {vals[k]:.15e}' for name, vals in traces.items()]
    lines.append('END')
    return "\n".join(lines)


def test_netlist_has_all_analyses():
    text = spectre_netlist(model_file="/pdk/gpdk045.scs", model="g45n1svt", w=1e-6, l=45e-9, vdd=1.1)
    assert 'include "/pdk/gpdk045.scs" section=tt' in text
    assert "M1 (d g 0 0) g45n1svt w=1e-06 l=4.5e-08" in text
    assert text.count(" dc param=vgs") == 2 and text.count(" dc param=vds") == 4
    assert "save M1:vth" in text and "rawfmt=psfascii" in text


def test_parse_psfascii_roundtrip():
    vg = np.linspace(0, 1, 5)
    d = parse_psfascii(psf("vgs", vg, {"VD:p": -vg * 1e-4, "M1:vth": np.full(5, 0.42)}))
    np.testing.assert_allclose(d["vgs"], vg)
    np.testing.assert_allclose(d["VD:p"], -vg * 1e-4)


def test_parse_rejects_non_sweep():
    with pytest.raises(ValueError):
        parse_psfascii("HEADER\nEND\n")


def test_import_results_builds_curves(tmp_path):
    vdd, vd_lin = 1.0, 0.05
    vg = np.round(np.arange(0, vdd + 0.005, 0.01), 6)
    vd = np.round(np.arange(0, vdd + 0.01, 0.02), 6)
    for k in range(2):
        id_ = 1e-12 * 10 ** (vg / 0.08)
        (tmp_path / f"idvg_vd{k}.dc").write_text(psf("vgs", vg, {"VD:p": -id_, "VG:p": 0 * vg,
                                                                "M1:vth": np.full(vg.size, 0.4)}))
    for k in range(4):
        (tmp_path / f"idvd_vg{k}.dc").write_text(psf("vds", vd, {"VD:p": -vd * 1e-4, "VG:p": 0 * vd}))
    df = import_results(tmp_path, vd_lin, vdd)
    assert set(df.sweep) == {"idvg", "idvd"} and df.attrs["simulated"]
    tr = df[(df.sweep == "idvg") & (df.vd == vdd)]
    assert (tr.id > 0).all() and np.allclose(tr.bsim_vth, 0.4)


def test_write_netlist_dir(tmp_path):
    out = write_netlist_dir(tmp_path / "ref", model_file="m.scs", model="nch", w=1e-6, l=1e-7, vdd=1.0,
                            section=None)
    assert (out / "run.sh").read_text().count("spectre ivchar_ref.scs") == 1
    assert "section=" not in (out / "ivchar_ref.scs").read_text()
