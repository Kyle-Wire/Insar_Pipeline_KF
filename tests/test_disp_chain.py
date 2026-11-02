"""DISP-S1 record chain: resampling, gap-filling, grid and the progress feed."""

import numpy as np
import pytest

from insar_pipeline import disp_chain as dc
from insar_pipeline import progress


def test_gap_fill_interpolates_in_time_and_holds_ends():
    t = np.array([0.0, 1.0, 2.0, 4.0])
    ts = np.full((4, 1, 3), np.nan, dtype="float32")
    ts[:, 0, 0] = [0.0, np.nan, 2.0, 4.0]        # interior gap, uneven spacing
    ts[:, 0, 1] = [np.nan, 1.0, 2.0, np.nan]     # both ends missing
    ts[:, 0, 2] = [np.nan, np.nan, np.nan, 3.0]  # below the minimum count
    counts = dc.gap_fill(ts, t, min_count=2, max_end_hold=None)      # holding with no limit
    assert counts.tolist() == [[3, 2, 1]]
    assert ts[:, 0, 0].tolist() == pytest.approx([0.0, 1.0, 2.0, 4.0])
    assert ts[:, 0, 1].tolist() == pytest.approx([1.0, 1.0, 2.0, 2.0])
    assert np.isnan(ts[:, 0, 2]).all()


def test_gap_fill_by_default_drops_a_pixel_held_for_most_of_the_record():
    t = np.array([0.0, 1.0, 2.0, 4.0])
    ts = np.full((4, 1, 2), np.nan, dtype="float32")
    ts[:, 0, 0] = [0.0, np.nan, 2.0, 4.0]        # interior gap only: filled
    ts[:, 0, 1] = [np.nan, 1.0, 2.0, np.nan]     # ends held for 3 of 4 years: dropped
    st = {}
    dc.gap_fill(ts, t, min_count=2, stats=st)
    assert ts[:, 0, 0].tolist() == pytest.approx([0.0, 1.0, 2.0, 4.0])
    assert np.isnan(ts[:, 0, 1]).all() and st["dropped_end_hold"] == 1


def test_gap_fill_uses_time_not_index():
    t = np.array([0.0, 0.1, 1.0])
    ts = np.array([0.0, np.nan, 10.0], dtype="float32").reshape(3, 1, 1)
    dc.gap_fill(ts, t, min_count=2)
    assert float(ts[1, 0, 0]) == pytest.approx(1.0)


def test_target_grid_coarsens_record_grid(tmp_path):
    import h5py
    f = tmp_path / "geo_velocity.h5"
    with h5py.File(f, "w") as h:
        h.create_dataset("velocity", data=np.zeros((11, 9), "float32"))
        h.attrs.update({"Y_FIRST": "38.5", "X_FIRST": "-87.9", "Y_STEP": "-0.000135",
                        "X_STEP": "0.000172", "LENGTH": "11", "WIDTH": "9"})

    class C:
        disp_target_grid_file = str(f)
        disp_target_grid_factor = 2
    g = dc.target_grid(C)
    assert (g["LENGTH"], g["WIDTH"]) == (5, 4)
    assert g["Y_STEP"] == pytest.approx(-0.00027) and g["X_STEP"] == pytest.approx(0.000344)
    lat, lon = dc.cell_centres(g)
    # a coarse cell's centre is the shared corner of the 2x2 fine cells it covers
    assert lat[0] == pytest.approx(38.5 - 0.000135 * 1.0)
    assert lon[0] == pytest.approx(-87.9 + 0.000172 * 1.0)


def test_resample_index_hits_the_right_utm_pixel():
    from pyproj import Transformer
    g = {"Y_FIRST": 38.30, "X_FIRST": -87.70, "Y_STEP": -0.0003, "X_STEP": 0.0004, "LENGTH": 20, "WIDTH": 20}
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
    x0, y0 = tr.transform(-87.705, 38.305)
    xs = np.round(x0) + 15.0 + 30.0 * np.arange(60)            # pixel centres, 30 m
    ys = np.round(y0) - 15.0 - 30.0 * np.arange(60)
    row, col, ok = dc.resample_index(g, xs, ys, 32616)
    assert ok.all()
    lat, lon = dc.cell_centres(g)
    X, Y = tr.transform(lon[7], lat[5])
    assert abs(xs[col[5, 7]] - X) <= 15.0 and abs(ys[row[5, 7]] - Y) <= 15.0
    src = np.arange(3600, dtype="float32").reshape(60, 60)
    out = dc.take(src, (row, col, ok))
    assert out[5, 7] == src[row[5, 7], col[5, 7]]


def test_resample_index_flags_cells_off_the_source_grid():
    g = {"Y_FIRST": 38.30, "X_FIRST": -87.70, "Y_STEP": -0.0003, "X_STEP": 0.0004, "LENGTH": 4, "WIDTH": 4}
    xs = 1000.0 + 30.0 * np.arange(5)            # nowhere near the grid
    ys = 1000.0 - 30.0 * np.arange(5)
    _, _, ok = dc.resample_index(g, xs, ys, 32616)
    assert not ok.any()
    out = dc.take(np.ones((5, 5), "float32"), dc.resample_index(g, xs, ys, 32616))
    assert np.isnan(out).all()


def test_progress_latest_and_eta(tmp_path):
    progress.emit(tmp_path, "prefetch", "start", done=0, total=10)
    progress.emit(tmp_path, "prefetch", "running", done=4, total=10)
    progress.emit(tmp_path, "mintpy", "start")
    with open(progress.path_for(tmp_path), "a") as fh:
        fh.write('{"torn": \n')                    # a torn line is skipped
    L = progress.latest(tmp_path)
    assert list(L) == ["prefetch", "mintpy"]
    assert L["prefetch"]["done"] == 4 and L["prefetch"]["started"] <= L["prefetch"]["t"]
    ev = dict(L["prefetch"], started=L["prefetch"]["t"] - 40.0)
    assert progress.eta_seconds(ev) == pytest.approx(60.0)
    with pytest.raises(ValueError):
        progress.emit(tmp_path, "x", "nonsense")


def test_progress_restart_clears_old_counts(tmp_path):
    progress.emit(tmp_path, "s", "running", done=9, total=10)
    progress.emit(tmp_path, "s", "start")
    assert "done" not in progress.latest(tmp_path)["s"]


