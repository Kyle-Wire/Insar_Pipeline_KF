"""19 Oct 2026, results viewer plan phase 0: a coherence class with too few
farmland cells to measure borrows the nearest measured class's centre for
its RATE (significance.class_means), but field_points gave it no FRAME, so
its cells had no line-of-sight change and no series. One project's delivered
line_of_sight_cumulative_displacement raster missed 132 cells of coherence
0.90-0.98 that its velocity raster had (class 4 borrowing class 3).

Outcome checks: every good cell of a borrowing class has a change and a
series, equal to what the lender's frame gives; a cell below the first class
has neither (as it has no relative rate); the delivered raster has a value
wherever the velocity raster has one."""
import types

import h5py
import numpy as np
import pytest

from insar_pipeline import client_report as cr
from insar_pipeline import field_points as fp
from insar_pipeline import significance as sg
from insar_pipeline.tests import test_fix_L9b as L9b


def _ctx(tmp_path, kcls, far, cube, class_means, tilt=None):
    p = tmp_path / "ts.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("timeseries", data=cube)
    t = np.linspace(0.0, 2.0, cube.shape[0])
    return types.SimpleNamespace(ts_file=p, kcls=kcls, far=far, class_means=class_means,
                                 class_tilt=tilt or {}, t=t, good=np.ones(kcls.shape, bool))


def test_a_borrowing_class_takes_its_lenders_frame(tmp_path):
    ny, nx, T = 6, 8, 5
    kcls = np.full((ny, nx), 3)
    kcls[:, 6:] = 4                                     # class 4: too few farmland cells, borrows 3
    kcls[0, 0] = -1                                     # below the first class
    far = np.zeros((ny, nx), bool)
    far[:, :4] = True                                   # the farmland: class 3 only
    rng = np.random.default_rng(1)
    cube = rng.normal(0, 0.003, (T, ny, nx)).astype(np.float32)
    ctx = _ctx(tmp_path, kcls, far, cube, {3: 0.0, "_borrowed": [0, 1, 2, 4]}, tilt={3: 0.5})
    ch = fp.relative_change(ctx)
    v0, v1 = cube[0] * 1000.0, cube[-1] * 1000.0
    f3 = far & (kcls == 3)
    lender = (v1[f3].mean() + 0.5 * 2.0) - v0[f3].mean()
    want = (v1 - v0) - lender
    sel = kcls == 4
    assert np.isfinite(ch[sel]).all()
    np.testing.assert_allclose(ch[sel], want[sel], rtol=0, atol=1e-6)
    np.testing.assert_allclose(ch[kcls == 3], want[kcls == 3], rtol=0, atol=1e-6)
    assert np.isnan(ch[0, 0])                           # no relative rate there, so no change either


def test_the_nearest_measured_class_lends_ties_to_the_lower(tmp_path):
    kcls = np.array([[0, 1, 2, 3, 4]])
    far = np.ones_like(kcls, bool)
    ctx = _ctx(tmp_path, kcls, far, np.zeros((3, 1, 5), np.float32), {1: 0.0, 3: 0.0})
    setup = fp._class_setup(ctx)
    assert setup[3] == {0: 1, 1: 1, 2: 1, 3: 3, 4: 3}
    # the same lender as the rate's centre: class 1 and 3 measured, 0/2/4 borrow
    cm = sg.class_means(np.array([10.0, 30.0]), np.array([0.65, 0.85]), np.ones(2, bool), min_n=1)
    assert cm["_borrowed"] == [0, 2, 4]
    assert {i: cm[i] for i in range(5)} == {i: {1: 10.0, 3: 30.0}[setup[3][i]] for i in range(5)}


def test_no_measured_class_takes_the_pooled_farmland(tmp_path):
    kcls = np.array([[2, 2, 4, 4]])
    far = np.array([[True, True, True, False]])
    cube = np.zeros((3, 1, 4), np.float32)
    cube[-1] = np.array([[1, 3, 5, 9]], np.float32) / 1000.0
    ctx = _ctx(tmp_path, kcls, far, cube, {})
    ch = fp.relative_change(ctx)
    np.testing.assert_allclose(ch, np.array([[1, 3, 5, 9]]) - 3.0)


@pytest.fixture(scope="module")
def borrowed_project(tmp_path_factory):
    """The L9b project with a patch of class-3 coherence inside the farmland,
    too small (< 200 farmland cells) for class 3 to be measured."""
    root = tmp_path_factory.mktemp("p0")
    cfg = L9b.make_project(root)
    p = cfg.mintpy_dir / "geo" / "geo_temporalCoherence.h5"
    with h5py.File(p, "r+") as f:
        tc = f["temporalCoherence"][:]
        tc[100:110, 100:110] = 0.85                      # 100 cells of class 3, in the farmland
        f["temporalCoherence"][...] = tc
    cr.build(cfg)
    return cfg


def test_the_delivered_cumulative_raster_has_every_cell_the_velocity_raster_has(borrowed_project):
    import rasterio
    D = borrowed_project.project_dir / "deliverables"
    with rasterio.open(D / "line_of_sight_velocity_in_per_yr.tif") as r:
        vel = r.read(1)
    (cum_p,) = sorted(D.glob("line_of_sight_cumulative_displacement_in_*.tif"))
    with rasterio.open(cum_p) as r:
        cum = r.read(1)
    patch = (slice(100, 110), slice(100, 110))
    assert np.isfinite(vel[patch]).all()                 # the borrowed class has a rate ...
    assert np.isfinite(cum[patch]).all()                 # ... and now its change
    assert not (np.isfinite(vel) & ~np.isfinite(cum)).any()


def test_results_gives_a_borrowing_class_its_change_between_window_ends(borrowed_project):
    """results.analyse had the same gap: a borrowing class's cells had a rate
    but no change between the window ends."""
    from insar_pipeline import results as rs
    from insar_pipeline.panel_history import _Context
    spec = L9b._box_spec((slice(98, 112), slice(98, 112)))           # 's,w,n,e', latitude first
    res = rs.analyse(borrowed_project, spec, start="2018-01-01", end="2020-12-31",
                     ctx=_Context(borrowed_project))
    pts = res.points
    tc = np.asarray(pts["temporal_coherence"], float)
    ch = np.asarray(pts["vertical_change_between_window_ends_in"], float)
    v = np.asarray(pts["vertical_velocity_in_yr"], float)
    cls3 = np.isclose(tc, 0.85)
    assert cls3.sum() == 100
    assert np.isfinite(v[cls3]).all() and np.isfinite(ch[cls3]).all()
