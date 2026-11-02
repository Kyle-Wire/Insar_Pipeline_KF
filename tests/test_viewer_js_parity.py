"""19 Oct 2026, results viewer plan phase 1: the browser's arithmetic
(viewer_assets/viewer_core.js, run here under node) is viewer_bundle's -
which tests/test_viewer_window_math.py holds to results.analyse - on the
same bundle: every point's window figures (a tested window, a look, the
whole record), the polygon average the page shows, a point's chart series,
the load check, and the CSV safety of typed names (areas.formula_safe /
formula_safe_text / one_line, SEC-05)."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from insar_pipeline import areas
from insar_pipeline import results_viewer as rv
from insar_pipeline import viewer_bundle as vb
from insar_pipeline.tests.test_viewer_window_math import make_proj
from insar_pipeline.tests import test_fix_L9b as L9b

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed (install/envs/insar.yml pins nodejs)")
RUNNER = vb.Path(vb.__file__).resolve().parent / "viewer_assets" / "tests" / "parity_runner.js"
HOSTILE = ["=cmd|' /C calc'!A0", "North,=HYPERLINK(\"x\")", "+1", "-1.5", "@SUM(A1)", "Field =2",
           "  \"=x", "Plain name", "a;-b", "\tTab", "-", "1e-3"]


# (value, decimals, limit): the balloon's signed figures and the figures held
# below a threshold (client_kmz._beside_limit) - no exact binary ties, where
# Python's and the browser's rounding may differ by design
FMT = [(-0.001, 2, None), (0.0049, 2, None), (-0.0051, 2, None), (1.2345, 2, None), (-12.71, 1, None),
       (0.4999, 0, None), (-0.51, 0, None), (0.996, 2, 1.0), (0.994, 2, 1.0), (1.9599, 2, 1.96),
       (1.9551, 2, 1.96), (0.6999, 1, 0.7), (0.68, 1, 0.7), (2.5, 2, 1.0)]


def _nan(a):
    return np.array([np.nan if v is None else v for v in a], float)


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    return make_proj(tmp_path_factory.mktemp("vjs"))


@pytest.fixture(scope="module")
def run(proj, tmp_path_factory):
    d = tmp_path_factory.mktemp("js")
    header, arrays = vb.bundle(proj.cfg, proj.fd, proj.rep)
    (d / "b.bin").write_bytes(vb.encode(header, arrays))
    dates = proj.fd["meta"]["dates"]
    T = len(dates)
    w1 = vb.window_index(dates, "2018-01-01", "2020-12-31")
    w2 = vb.window_index(dates, "2022-03-01", "2022-09-30")
    table = {0: 3.1, 1: 2.7, 2: 2.2, 4: 1.9}
    s_, w, n, e = (float(x) for x in L9b._box_spec((slice(20, 70), slice(30, 100))).split(","))
    ring = [[w, s_], [e - 0.01, s_ + 0.002], [e, n], [w + 0.013, n - 0.004], [w, s_]]
    req = {"windows": [{"i0": w1[0], "i1": w1[1], "table": table}, {"i0": w1[0], "i1": w1[1]},
                       {"i0": w2[0], "i1": w2[1]}, {"i0": 0, "i1": T - 1}],
           "polygons": [{"i0": 0, "i1": T - 1, "rings": [ring]}, {"i0": w1[0], "i1": w1[1], "rings": [ring]}],
           "series": [{"j": j, "i0": w1[0], "i1": w1[1], "vertical": v} for j in (0, 777, 5000) for v in (True, False)],
           "safe": HOSTILE, "fmt": FMT}
    (d / "req.json").write_text(json.dumps(req))
    p = subprocess.run([NODE, str(RUNNER), str(d / "b.bin"), str(d / "req.json")], capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout), req, table, ring


KEYS = ("los", "vertical", "snr", "signal_to_threshold", "sd_vertical", "change_los", "change_los_ends")


def test_every_point_reads_the_same_in_the_browser(proj, run):
    out, req, table, _ = run
    for got, w in zip(out["windows"], req["windows"]):
        want = vb.window_figures(proj.fd, w["i0"], w["i1"], table=w.get("table"))
        assert got["kind"] == want["kind"]
        assert {int(k): v for k, v in got["table"].items()} == want["table"]
        for k in KEYS:
            np.testing.assert_allclose(_nan(got[k]), want[k], rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=k)
        if want["kind"] == "look":
            assert {int(k): v for k, v in got["spread"].items()} == want["spread"]
            assert {int(k): v for k, v in got["n_sample"].items()} == want["n_sample"]


def test_the_polygon_average_is_the_viewers(proj, run):      
    out, req, _, ring = run
    idx = rv._inside(proj.fd, [ring])
    whole, win = out["polygons"]
    assert whole["idx"] == idx.tolist() and win["idx"] == idx.tolist()
    st = rv._poly_stats(proj.fd, idx)
    for k in ("n", "n_too_fast", "vertical_in_yr", "vertical_in_yr_p10", "vertical_in_yr_p90", "relative_in_yr",
              "los_in_yr", "los_change_in", "median_snr", "percent_standing_out", "mean_coherence"):
        assert whole["stats"][k] == pytest.approx(st[k], rel=1e-9, abs=1e-9), k
    # over a window: the same rule on the window's figures
    wf = vb.window_figures(proj.fd, req["polygons"][1]["i0"], req["polygons"][1]["i1"])
    ok = idx[~proj.fd["fast"][idx]]
    assert win["stats"]["vertical_in_yr"] == pytest.approx(float(np.nanmean(wf["vertical"][ok])) / 25.4, rel=1e-9)


def test_a_points_chart_series_is_the_same(proj, run):     
    out, req, _, _ = run
    for got, s in zip(out["series"], req["series"]):
        want = vb.point_series(proj.fd, s["j"], s["i0"], s["i1"], s["vertical"])
        np.testing.assert_allclose(_nan(got["y"]), want, rtol=0, atol=1e-6)


def test_the_load_check_passes(run):
    out = run[0]
    assert out["check"]["rate"] <= 0.01 and out["check"]["change"] <= 0.01


def test_typed_names_are_made_safe_as_python_does(run):
    out = run[0]
    for t in HOSTILE:
        assert out["safe"][t]["cell"] == areas.formula_safe(t), t
        assert out["safe"][t]["text"] == areas.formula_safe_text(t), t
        assert out["safe"][t]["line"] == areas.one_line(t), t


def test_numbers_read_as_the_kmz_balloon_writes_them(run):
    from insar_pipeline import client_kmz as ck
    from insar_pipeline import report_insights as ri
    for got, (v, nd, lim) in zip(run[0]["fmt"], FMT):
        assert got["signed"] == ri.signed(v, nd), (v, nd)
        if lim is not None:
            assert got["beside"] == ck._beside_limit(v, lim, nd), (v, lim)
