"""19 Oct 2026, results viewer plan phase 1: the date-window figures of the
viewer (viewer_bundle.window_figures, the reference the browser's
viewer_core.js is tested against) are results.analyse's.

The project is test_fix_L9b's with five coherence classes - four measured,
one too small to measure and borrowing - and the report's second farmland
pass (its feature is cut out of the farmland). Outcome checks, point by
point against results.analyse over the same window:

- the relative and vertical rate to 1e-3 mm/yr;
- the change between the window ends (analyse's 30-day means) to 1e-3 mm;
- given analyse's null table (a tested window): the signal-to-threshold
  and the stands-out flag exactly;
- without it (a look): the noise table from the farmland sample, which here
  holds every farmland cell, so it equals analyse's;
- the whole record: the stored figures, and the own-cell series reproduce
  them (the check the viewer makes on load)."""
import types

import h5py
import numpy as np
import pytest

from insar_pipeline import client_kmz as ck
from insar_pipeline import client_report as cr
from insar_pipeline import field_points as fp
from insar_pipeline import results as rs
from insar_pipeline import viewer_bundle as vb
from insar_pipeline.panel_history import _Context
from insar_pipeline.tests import test_fix_L9b as L9b

MM_IN = 25.4
WIDE = (slice(20, 112), slice(50, 116))          # over every class, the feature and the borrowed patch
BORROWED = (slice(100, 110), slice(100, 110))


def _five_classes(cfg):
    p = cfg.mintpy_dir / "geo" / "geo_temporalCoherence.h5"
    with h5py.File(p, "r+") as f:
        tc = np.full(f["temporalCoherence"].shape, 0.95, np.float32)
        tc[0:30] = 0.5
        tc[30:60] = 0.65
        tc[60:90] = 0.75
        tc[BORROWED] = 0.85                        # 100 cells: class 3 cannot be measured, it borrows
        f["temporalCoherence"][...] = tc


def make_proj(root):
    """The five-class project, its report and its version-2 viewer table
    (test_viewer_js_parity.py builds its own from this)."""
    cfg = L9b.make_project(root)
    _five_classes(cfg)
    cr.build(cfg)
    rep = rs.report_record(cfg)
    vpath = ck.build(cfg, out_dir=root / "preview", viewer_only=True)
    fd = fp.load(vpath)
    return types.SimpleNamespace(cfg=cfg, rep=rep, fd=fd, root=root)


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    return make_proj(tmp_path_factory.mktemp("vwin"))


def _match(fd, pts):
    key = {(round(float(a), 7), round(float(b), 7)): j for j, (a, b) in enumerate(zip(fd["lat"], fd["lon"]))
           if fd["group"][j] == 0}
    jj = np.array([key.get((round(float(a), 7), round(float(b), 7)), -1) for a, b in zip(pts["lat"], pts["lon"])])
    assert (jj >= 0).all(), "every analysed cell is a field point on this 100 m grid"
    return jj


def test_the_table_is_version_2_and_reproduces_the_whole_record(proj):
    m = proj.fd["meta"]
    assert m["table_version"] == 2
    assert m["class_measured"] == [0, 1, 2, 4]
    assert m["class_lender"]["3"] == 2                       # 2 and 4 are equally near: the lower lends
    assert m["whole_record_check_mm_yr"] <= 0.01
    assert m["whole_record_change_check_mm"] <= 0.01
    assert m["far_partial_nan_cells"] == 0
    assert proj.rep.get("farmland_second_pass"), "the report's second farmland pass is exercised"
    fd = proj.fd
    assert np.isfinite(fd["cell_los"][np.isin(fd["kcls"], [0, 1, 2, 3, 4])]).all()


@pytest.fixture(scope="module")
def window(proj):
    ctx = _Context(proj.cfg)
    res = rs.analyse(proj.cfg, L9b._box_spec(WIDE), start="2018-01-01", end="2020-12-31", ctx=ctx, rep=proj.rep)
    i0, i1 = vb.window_index(proj.fd["meta"]["dates"], "2018-01-01", "2020-12-31")
    return types.SimpleNamespace(res=res, i0=i0, i1=i1, jj=_match(proj.fd, res.points))


def test_a_window_reads_results_analyse_point_by_point(proj, window):
    res, jj = window.res, window.jj
    wf = vb.window_figures(proj.fd, window.i0, window.i1, table=res.cell_null_by_class)
    assert wf["kind"] == "tested" and wf["n_images"] == res.n_dates
    pts = res.points
    v_an = np.asarray(pts["vertical_velocity_in_yr"], float) * MM_IN
    np.testing.assert_allclose(wf["vertical"][jj], v_an, rtol=0, atol=1e-3)
    e_an = np.asarray(pts["vertical_change_between_window_ends_in"], float) * MM_IN
    bg = proj.fd["meta"]["bg_v"] * wf["years"] if proj.fd["meta"].get("anchored") else 0.0
    np.testing.assert_allclose(wf["change_los_ends"][jj] / proj.fd["up"][jj] + bg, e_an, rtol=0, atol=1e-3)
    # analyse sums in float32 (its streaming least squares): ~1e-5 apart
    stt = np.asarray(pts["signal_to_threshold"], float)
    np.testing.assert_allclose(wf["signal_to_threshold"][jj], stt, rtol=0, atol=1e-4)
    edge = np.abs(stt - 1.0) < 1e-4
    assert (np.asarray(pts["stands_out"], bool) == (wf["signal_to_threshold"][jj] >= 1))[~edge].all()
    # every class is in the area, the borrowed one included
    assert set(np.unique(proj.fd["kcls"][jj])) == {0, 1, 2, 3, 4}


def test_a_look_measures_the_same_noise_from_the_farmland_sample(proj, window):
    wf = vb.window_figures(proj.fd, window.i0, window.i1)
    assert wf["kind"] == "look"
    an = window.res.cell_null_by_class
    assert window.res.null_source == "window"
    assert set(wf["table"]) == set(an)
    for k, v in an.items():
        assert wf["table"][k] == pytest.approx(v, abs=0.0015)     # both rounded to 3 decimals


def test_the_whole_record_is_the_stored_figures(proj):
    fd = proj.fd
    T = len(fd["meta"]["dates"])
    wf = vb.window_figures(fd, 0, T - 1)
    assert wf["kind"] == "report"
    np.testing.assert_array_equal(wf["los"], fd["los_rel"])
    np.testing.assert_allclose(wf["vertical"], fd["rate"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(wf["snr"], fd["snr"], rtol=1e-9, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(wf["change_los"], fd["change_los"])
    # and the own-cell series give them back (what the viewer checks on load)
    b = fp.slopes(np.asarray(fd["meta"]["t"]), fd["cell_los"])
    np.testing.assert_allclose(b, fd["los_rel"], rtol=0, atol=0.01, equal_nan=True)


def test_window_rule_is_results_rule(proj):
    t = np.asarray(proj.fd["meta"]["t"])
    dates = proj.fd["meta"]["dates"]
    i0, i1 = vb.window_index(dates, "2019-01-01", "2019-02-15")           # 4 images at 12 days
    assert "at least 6" in vb.window_problem(t, i0, i1)
    i0, i1 = vb.window_index(dates, "2019-01-01", "2019-03-10")           # 6 images, 2.3 months
    assert "3 months" in vb.window_problem(t, i0, i1)
    with pytest.raises(ValueError, match="at least"):
        rs.analyse(proj.cfg, "Test Field", start="2019-01-01", end="2019-03-10")
    i0, i1 = vb.window_index(dates, "2019-01-01", "2019-04-15")
    assert vb.window_problem(t, i0, i1) is None


def test_the_bundle_round_trips(proj):
    header, arrays = vb.bundle(proj.cfg, proj.fd, proj.rep)
    data = vb.encode(header, arrays)
    h, a = vb.decode(data)
    assert h["meta"]["windows"] is True and h["meta"]["window_note"] is None
    assert h["meta"]["stamp"] and "ground movement report" in h["meta"]["stamp"][-1]
    assert h["meta"]["final"] is False                                   # never signed off here
    np.testing.assert_array_equal(a["cell_los"], proj.fd["cell_los"])
    np.testing.assert_array_equal(a["lat"], proj.fd["lat"])
    assert a["far"].dtype == np.uint8 and a["kcls"].dtype == np.int8
    assert "project_dir" not in h["meta"]                                 # no local path goes to a page
