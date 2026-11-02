"""Lane L10 (5 Oct 2026 review): the DISP-S1 route and the independent checks.

Every test drives PRODUCTION code - assemble_stack with its granule reads
monkeypatched, disp_chain.run with MintPy faked by small numpy steps - and
fails on the behaviour the review found. No network: _open_remote raises.
"""
import json
import types
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest

from insar_pipeline import disp_backend as db
from insar_pipeline import disp_chain as dc
from insar_pipeline.config import PipelineConfig

WL = 0.05546576
CYC = WL / 2


# ======================================================================= helpers
def sim_granules(n=60, seg=12, ny=40, nx=40, rate=-0.010, noise=0.002, seed=1, frame="09999",
                 hhmmss="234758", start=datetime(2020, 1, 1)):
    """Granules as OPERA ships them: each segment relative to its reference,
    the reference of segment s>0 being the LAST secondary of segment s-1."""
    rng = np.random.default_rng(seed)
    sec = [start + timedelta(days=12 * (i + 1)) for i in range(n)]
    tyr = np.array([(d - start).days / 365.25 for d in sec])
    truth = (rate * tyr)[:, None, None] * np.ones((1, ny, nx)) + rng.normal(0, noise, (n, ny, nx))
    segs = np.arange(n) // seg
    ref_idx = {0: None}
    ref_idx.update({s: s * seg - 1 for s in range(1, int(segs.max()) + 1)})
    refdate = {0: start}
    refdate.update({s: sec[s * seg - 1] for s in range(1, int(segs.max()) + 1)})
    raw = np.stack([truth[i] - (truth[ref_idx[int(segs[i])]] if ref_idx[int(segs[i])] is not None else 0)
                    for i in range(n)]).astype("float32")
    names = [f"OPERA_L3_DISP-S1_IW_F{frame}_VV_{refdate[int(segs[i])]:%Y%m%d}T{hhmmss}Z_"
             f"{sec[i]:%Y%m%d}T{hhmmss}Z_v1.0_20250101T000000Z.nc" for i in range(n)]
    anchors = [s * seg - 1 for s in range(1, int(segs.max()) + 1)]
    return types.SimpleNamespace(raw=raw, truth=truth, names=names, anchors=anchors, sec=sec, tyr=tyr,
                                 start=start, seg=seg)


def results_of(names, track=19, direction="ASCENDING", geometry=None):
    return [types.SimpleNamespace(properties={"fileName": nm, "pathNumber": track,
                                              "flightDirection": direction},
                                  geometry=geometry, umm={}) for nm in names]


def no_network(*a, **k):
    raise OSError("network disabled in tests")


def patch_reads(monkeypatch, sim, extra=None, missing=(), window=(0.0, 1.0, 0.0, 1.0, 32616),
                corrections=None):
    arrays = {}
    for i, nm in enumerate(sim.names):
        a = {"displacement": sim.raw[i]}
        if extra:
            a.update(extra(i))
        arrays[nm] = a
    monkeypatch.setattr(db, "_data_url", lambda p: "https://example.invalid/g.nc")
    monkeypatch.setattr(db, "_geometry_from_granule", lambda u: (37.5, None))
    monkeypatch.setattr(db, "_open_remote", no_network)

    def read(cfg, url, fname, win, layers, attempts=4):
        if fname in missing:
            return None, win, None, None
        return arrays[fname], window, None, None
    monkeypatch.setattr(db, "_read_granule", read)
    if corrections is not None:
        monkeypatch.setattr(db, "_read_corrections", corrections)
    return arrays


def make_cfg(tmp_path, **kw):
    cfg = PipelineConfig(project_dir=tmp_path / "proj", bbox=[-87.52, 38.29, -87.47, 38.33],
                         start_date="2020-01-01", end_date="2022-12-31")
    base = dict(backend="disp", track=19, orbit_direction="ascending", disp_apply_corrections=False,
                disp_mask_connected_components=False, disp_flag_unwrap_jumps=True,
                disp_repair_boundary_steps=False, disp_repair_boundary_datum=False, disp_decimate=1)
    base.update(kw)
    for k, v in base.items():
        setattr(cfg, k, v)
    return cfg


# ================================================================ disp-2: the anchor epoch
def test_anchor_error_through_assemble_stack_is_found_at_the_anchor(tmp_path, monkeypatch):
    """+1 cycle in the ANCHOR granule for 5% of pixels (the value the carry
    copies). Old code: suspect_pixels 0, excess -0.38%, rate +57.5 vs -10."""
    sim = sim_granules()
    rng = np.random.default_rng(1)
    bad = rng.random(sim.raw.shape[1:]) < 0.05
    for a in sim.anchors:
        sim.raw[a][bad] += CYC
    patch_reads(monkeypatch, sim)
    A = db.assemble_stack(make_cfg(tmp_path), results_of(sim.names))
    assert A["anchor_idx"] == sim.anchors == [11, 23, 35, 47]
    assert A["boundary_idx"] == [a + 1 for a in sim.anchors]
    b = A["boundary_steps"]
    assert b["suspect_pixels"] >= 300 and b["excess"] > 0.01
    # the per-pixel layer marks the affected pixels, and only them
    hit = A["anchor_suspect"] > 0
    assert (hit & bad).sum() >= 0.95 * bad.sum() and (hit & ~bad).sum() <= 2
    assert any(p["code"] == "anchor_whole_cycle" for p in A["stitch_problems"])


def test_reverting_spike_at_the_first_epoch_of_a_segment_is_not_repaired_into_a_step(tmp_path, monkeypatch):
    """The opt-in repair used to test the step into the NEW segment's first
    epoch and turned a reverting spike there into a permanent -1 cycle."""
    sim = sim_granules(noise=0.001)
    rng = np.random.default_rng(3)
    bad = rng.random(sim.raw.shape[1:]) < 0.05
    b = sim.anchors[0] + 1
    sim.raw[b][bad] += CYC                       # one granule's own error: reverts next epoch
    patch_reads(monkeypatch, sim)
    cfg = make_cfg(tmp_path, disp_repair_boundary_steps=True, disp_flag_unwrap_jumps=False)
    A = db.assemble_stack(cfg, results_of(sim.names))
    err = A["ts"][-1] - (sim.truth[-1] - 0)
    assert np.allclose(np.median(err[bad]) / CYC, 0.0, atol=0.05), "the repair baked in a reverting spike"


def test_repair_at_the_anchor_removes_the_carried_cycle(tmp_path, monkeypatch):
    sim = sim_granules(noise=0.001)
    rng = np.random.default_rng(4)
    bad = rng.random(sim.raw.shape[1:]) < 0.10
    for a in sim.anchors:
        sim.raw[a][bad] += CYC
    patch_reads(monkeypatch, sim)
    cfg = make_cfg(tmp_path, disp_repair_boundary_steps=True)
    A = db.assemble_stack(cfg, results_of(sim.names))
    err = A["ts"][-1] - sim.truth[-1]
    assert abs(np.median(err[bad]) / CYC) < 0.05
    assert A["boundary_steps"]["after_repair"]["suspect_pixels"] == 0


def test_check_and_repair_skip_the_reversal_of_a_spike_before_the_anchor():
    from insar_pipeline.disp_backend import _check_boundary_steps, _repair_boundary_steps
    n = 40
    ts = (np.arange(n) * -0.0003)[:, None, None] * np.ones((1, 6, 6))
    ts = ts + np.random.default_rng(5).normal(0, 0.0005, ts.shape)
    ts[19, 2, 2] += CYC                          # spike at a-1 = 19, reverts INTO the anchor 20
    info = _check_boundary_steps(ts.copy(), [20], WL)
    assert info["suspect_pixels"] == 0 and info["reversal_excluded"] == 1
    before = ts.copy()
    _repair_boundary_steps(ts, [20], WL)
    assert np.allclose(ts, before)


# ================================================================ disp-4: measured, refused
def _corr_reader(fail_dates):
    def read(cfg, url, fname, win, shape, attempts=4):
        if db._dates(fname)[1] in fail_dates:
            return {}
        return {name: np.zeros(shape, "float32") for name in db.CORRECTION_LAYERS}
    return read


def test_partly_corrected_stack_is_refused_not_reported(tmp_path, monkeypatch):
    sim = sim_granules(n=36)
    fail = {f"{sim.sec[5]:%Y%m%d}", f"{sim.sec[30]:%Y%m%d}"}
    patch_reads(monkeypatch, sim, corrections=_corr_reader(fail))
    cfg = make_cfg(tmp_path, disp_apply_corrections=True)
    A = db.assemble_stack(cfg, results_of(sim.names))
    assert A["corrections"]["status"] == "partial" and A["corrections"]["epochs_corrected"] == 34
    assert sorted(A["corrections"]["uncorrected_dates"]) == sorted(fail)
    with pytest.raises(RuntimeError, match="34 of 36"):
        db.check_stitch(A, cfg)


def test_fully_corrected_stack_passes(tmp_path, monkeypatch):
    sim = sim_granules(n=36)
    patch_reads(monkeypatch, sim, corrections=_corr_reader(set()))
    cfg = make_cfg(tmp_path, disp_apply_corrections=True)
    A = db.assemble_stack(cfg, results_of(sim.names))
    assert A["corrections"]["status"] == "all"
    assert not [p for p in db.check_stitch(A, cfg) if p["hard"]]


def test_lost_anchor_granule_is_refused_unless_accepted(tmp_path, monkeypatch):
    sim = sim_granules(n=36)
    lost = sim.names[sim.anchors[0]]
    patch_reads(monkeypatch, sim, missing={lost})
    cfg = make_cfg(tmp_path)
    A = db.assemble_stack(cfg, results_of(sim.names))
    d = f"{sim.sec[sim.anchors[0]]:%Y%m%d}"
    assert A["lost_anchors"] == [d]
    with pytest.raises(RuntimeError, match="anchor granule"):
        db.check_stitch(A, cfg)
    cfg.allow_missing_dates = (d,)
    probs = db.check_stitch(A, cfg)
    assert any(p["code"] == "anchor_lost" and not p["hard"] for p in probs)


def test_coverage_ratchet_is_refused_but_a_late_seasonal_dip_is_not(tmp_path, monkeypatch):
    sim = sim_granules(n=60)
    # pixels genuinely lost for good, progressively: the carry cannot save them
    n_px = sim.raw.shape[1] * sim.raw.shape[2]
    die_at = np.random.default_rng(2).integers(5, 50, n_px).reshape(sim.raw.shape[1:])
    for i in range(60):
        sim.raw[i][die_at <= i] = np.nan
    patch_reads(monkeypatch, sim)
    cfg = make_cfg(tmp_path)
    A = db.assemble_stack(cfg, results_of(sim.names))
    assert A["coverage_ratchet"]["fired"]
    with pytest.raises(RuntimeError, match="observed at all"):
        db.check_stitch(A, cfg)

    sim2 = sim_granules(n=60)
    dip = np.random.default_rng(3).random((5,) + sim2.raw.shape[1:]) < 0.8
    for k, i in enumerate(range(55, 60)):       # snowy last month: 80% missing, pixels alive
        sim2.raw[i][dip[k]] = np.nan
    patch_reads(monkeypatch, sim2)
    A2 = db.assemble_stack(cfg, results_of(sim2.names))
    assert A2["per_epoch_valid"][-5:].mean() < 0.5 * A2["per_epoch_valid"][:5].mean()
    assert not A2["coverage_ratchet"]["fired"]


# ================================================================ disp-11: versions, references
def test_dedupe_keeps_newest_production_and_refuses_mixed_majors():
    a = "OPERA_L3_DISP-S1_IW_F04861_VV_20200101T234758Z_20200113T234758Z_v1.0_20250101T000000Z.nc"
    b = "OPERA_L3_DISP-S1_IW_F04861_VV_20200101T234758Z_20200113T234758Z_v1.0_20250301T000000Z.nc"
    c = "OPERA_L3_DISP-S1_IW_F04861_VV_20200101T234758Z_20200113T234758Z_v1.1_20250201T000000Z.nc"
    d = "OPERA_L3_DISP-S1_IW_F04861_VV_20200101T234758Z_20200125T234758Z_v1.0_20250101T000000Z.nc"
    kept = [p.properties["fileName"] for p in db.dedupe(results_of([a, b, d]))]
    assert sorted(kept) == sorted([b, d])
    kept = [p.properties["fileName"] for p in db.dedupe(results_of([a, b, c, d]))]
    assert c in kept and a not in kept and b not in kept and d in kept
    e = d.replace("_v1.0_", "_v2.0_")
    with pytest.raises(ValueError, match="major"):
        db.dedupe(results_of([a, e]))


def test_reference_date_that_is_not_a_prior_secondary_is_refused(tmp_path, monkeypatch):
    sim = sim_granules(n=36)
    k = sim.anchors[1] + 1
    for i in range(k, len(sim.names)):           # segment 2 claims a reference date nobody observed
        sim.names[i] = sim.names[i].replace(f"_{sim.sec[sim.anchors[1]]:%Y%m%d}T", "_20200105T", 1)
    patch_reads(monkeypatch, sim)
    with pytest.raises(RuntimeError, match="not the secondary date"):
        db.assemble_stack(make_cfg(tmp_path), results_of(sim.names))


def test_overlapping_segment_is_refused(tmp_path, monkeypatch):
    sim = sim_granules(n=36)
    k = sim.anchors[1] + 1
    earlier = f"{sim.sec[sim.anchors[1] - 3]:%Y%m%d}"
    for i in range(k, len(sim.names)):
        sim.names[i] = sim.names[i].replace(f"_{sim.sec[sim.anchors[1]]:%Y%m%d}T", f"_{earlier}T", 1)
    patch_reads(monkeypatch, sim)
    with pytest.raises(RuntimeError, match="overlap"):
        db.assemble_stack(make_cfg(tmp_path), results_of(sim.names))


def test_cached_coordinates_and_identification_need_no_earthdata(tmp_path, monkeypatch):
    sim = sim_granules(n=24)
    cfg = make_cfg(tmp_path)
    win = (500000.0, 501190.0, 4200000.0, 4201190.0, 32616)
    xs = 500007.0 + 30.0 * np.arange(40)
    ys = 4201183.0 - 30.0 * np.arange(40)
    c = db._granule_cache_path(cfg, sim.names[0])
    c.parent.mkdir(parents=True)
    np.savez(c, __window=np.array(win), __x=xs, __y=ys, displacement=sim.raw[0])
    patch_reads(monkeypatch, sim, window=win)
    calls = []
    monkeypatch.setattr(db, "_geometry_from_granule", lambda u: (calls.append(u), (38.3, "23:47:58"))[1])
    A = db.assemble_stack(cfg, results_of(sim.names))
    assert (A["x0"], A["y0"]) == (xs[0], ys[0]) and A["grid_origin"].startswith("first pixel centre")
    assert A["incidence_deg"] == 38.3 and len(calls) == 1
    db.assemble_stack(cfg, results_of(sim.names))        # second run: identification from the sidecar
    assert len(calls) == 1
    assert dc.source_coords(cfg, results_of(sim.names), (40, 40))[0][0] == xs[0]


# ================================================================ gap-geolocation-8
def test_legacy_route_origin_is_the_first_pixel_edge(tmp_path, monkeypatch):
    sim = sim_granules(n=24, ny=30, nx=30)
    cfg = make_cfg(tmp_path, disp_use_static_layers=False, disp_tropo_correction=False,
                   disp_seasonal_model=False, reference_lalo="", disp_use_recommended_mask=False)
    win = (500000.0, 500900.0, 4200000.0, 4200900.0, 32616)
    xs = 500012.0 + 30.0 * np.arange(30)                 # first centre 12 m inside the AOI corner
    ys = 4200889.0 - 30.0 * np.arange(30)
    c = db._granule_cache_path(cfg, sim.names[0])
    c.parent.mkdir(parents=True)
    np.savez(c, __window=np.array(win), __x=xs, __y=ys, displacement=sim.raw[0])
    patch_reads(monkeypatch, sim, window=win)
    out = db.build_stack(cfg, results_of(sim.names))
    with h5py.File(out / "velocity.h5", "r") as h:
        assert float(h.attrs["X_FIRST"]) == pytest.approx(xs[0] - 15.0)
        assert float(h.attrs["Y_FIRST"]) == pytest.approx(ys[0] + 15.0)
    # the old corner-based origin would have been the window corner
    assert float(win[0]) != pytest.approx(xs[0] - 15.0)


# ================================================================ disp-5: the ERA5 hour
def test_base_attrs_takes_the_granule_time_for_either_pass():
    class C:
        disp_target_grid_file = ""
        orbit_direction = "descending"
        track = 77
    g = {"Y_FIRST": 38.3, "X_FIRST": -87.5, "Y_STEP": -0.0003, "X_STEP": 0.0003, "LENGTH": 4, "WIDTH": 4}
    desc = dc.base_attrs(C, g, {"acq_time": "11:58:30", "frame_id": 1})
    assert desc["CENTER_LINE_UTC"] == str(11 * 3600 + 58 * 60 + 30)          # was 85680 (23:48)
    assert desc["ORBIT_DIRECTION"] == "DESCENDING"
    C.orbit_direction = "ascending"
    asc = dc.base_attrs(C, g, {"acq_seconds": 85678, "frame_id": 1})
    assert asc["CENTER_LINE_UTC"] == "85678"
    with pytest.raises(RuntimeError, match="hour is unknown"):
        dc.base_attrs(C, g, {"acq_time": None, "frame_id": 1})


def test_acquisition_time_comes_from_the_granule_names(tmp_path, monkeypatch):
    sim = sim_granules(n=24, hhmmss="115830")
    patch_reads(monkeypatch, sim)
    A = db.assemble_stack(make_cfg(tmp_path), results_of(sim.names))
    assert A["acq_time"] == "11:58:30" and A["acq_seconds"] == 43110


def test_granule_without_a_crs_is_refused_not_put_in_zone_16():
    ds = types.SimpleNamespace(attrs={}, variables={})
    with pytest.raises(RuntimeError, match="spatial_epsg"):
        db._granule_epsg(ds)
    from pyproj import CRS
    ds2 = types.SimpleNamespace(attrs={"crs_wkt": CRS.from_epsg(32617).to_wkt()}, variables={})
    assert db._granule_epsg(ds2) == 32617


# ================================================================ disp-6: frame choice
def _fp(lon0, lat0, lon1, lat1):
    return {"type": "Polygon", "coordinates": [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]}


def _frames():
    box = (-87.45, 38.85, -87.15, 39.10)
    a = [f"OPERA_L3_DISP-S1_IW_F04862_VV_20200101T234758Z_2020{m:02d}15T234758Z_v1.0_20250101T000000Z.nc"
         for m in range(1, 7)]
    b = [f"OPERA_L3_DISP-S1_IW_F32248_VV_20200101T235500Z_2020{m:02d}20T235500Z_v1.0_20250101T000000Z.nc"
         for m in range(1, 10)]
    res = results_of(a, track=19, geometry=_fp(-88.0, 38.5, -86.5, 39.6)) + \
        results_of(b, track=121, geometry=_fp(-87.30, 38.5, -86.0, 39.6))
    return box, res


def test_frame_is_chosen_by_track_and_overlap_not_by_count(tmp_path):
    box, res = _frames()
    cfg = make_cfg(tmp_path, bbox=list(box), track=19, disp_frame_id="")
    frame, info, rows = db.choose_frame(res, cfg)
    assert frame == "04862" and info["overlap_fraction"] == pytest.approx(1.0)   # 6 granules beat 9
    kept = db._select(cfg, res)
    assert {db._frame_of(p.properties["fileName"]) for p in kept} == {"04862"}
    cfg.track = 121
    assert db.choose_frame(res, cfg)[0] == "32248"
    cfg.track, cfg.disp_frame_id = 19, "32248"
    with pytest.raises(ValueError, match="track 121"):
        db.choose_frame(res, cfg)
    cfg.disp_frame_id, cfg.orbit_direction = "", "descending"
    with pytest.raises(ValueError, match="no DISP-S1 frame"):
        db.choose_frame(res, cfg)


def test_partial_frame_coverage_is_warned(tmp_path, caplog):
    box, res = _frames()
    cfg = make_cfg(tmp_path, bbox=list(box), track=121, disp_frame_id="")
    with caplog.at_level("WARNING"):
        frame, info, _ = db.choose_frame(res, cfg)
    assert info["overlap_fraction"] < db.MIN_FRAME_OVERLAP
    assert "covers only" in caplog.text


def test_granule_off_the_configured_track_is_refused(tmp_path):
    box, res = _frames()
    res[0].properties["pathNumber"] = 20
    cfg = make_cfg(tmp_path, bbox=list(box), track=19, disp_frame_id="04862")
    with pytest.raises(ValueError):
        db._select(cfg, res)


# ================================================================ disp-9: gap-fill ends
def test_gap_fill_drops_pixels_whose_held_ends_would_shrink_the_rate():
    t = np.arange(0, 10, 12 / 365.25)
    n = t.size
    ts = np.repeat((-0.010 * t)[:, None], 3, 1).reshape(n, 1, 3).astype("float32")
    ts[: int(0.30 * n), 0, 0] = np.nan          # 30% leading gap: -7.8 mm/yr if held
    ts[: int(0.05 * n), 0, 1] = np.nan          # 5%: kept, bias < 1%
    st = {}
    counts = dc.gap_fill(ts, t, min_count=3, stats=st)
    assert st["dropped_end_hold"] == 1 and st["dropped_end_hold_mask"][0].tolist() == [True, False, False]
    assert np.isnan(ts[:, 0, 0]).all()
    assert counts[0, 0] == n - int(0.30 * n)     # validEpochFraction still reports the real count
    rate = np.polyfit(t, ts[:, 0, 1], 1)[0] * 1000
    assert abs(rate + 10.0) < 0.1


# ================================================================ disp-7: tests that run
def test_rebase_comparison_skips_visibly(monkeypatch):
    from insar_pipeline.tests import test_rebase_vs_opera_utils as t
    monkeypatch.setattr(t, "OPERA", None)
    with pytest.raises(pytest.skip.Exception):
        t.test_agrees_exactly_with_opera_utils_when_nothing_is_missing()


def test_rebase_tests_drive_the_production_carry():
    """A pixel never valid before a boundary is NaN for the rest of the
    record in production (carry = last_abs); the old copied `_ours` kept it."""
    from insar_pipeline.tests import test_rebase_vs_opera_utils as t
    raw, refs, _ = t._scenario(nan_at=[(i, (3, 3)) for i in range(t.SEG_LEN)])
    ours = t._ours(raw.copy(), refs)
    assert np.isnan(ours[t.SEG_LEN:, 3, 3]).all()


def test_carry_tests_drive_the_production_carry(monkeypatch):
    from insar_pipeline.tests import test_disp_carry as t
    called = []
    real = db.assemble_stack
    monkeypatch.setattr(db, "assemble_stack", lambda *a, **k: (called.append(1), real(*a, **k))[1])
    t.test_carrying_the_last_valid_value_keeps_coverage_flat(monkeypatch)
    assert called


# ================================================================ the chain end to end
def _fake_mintpy(calls, add_noise_in_tropo=False):
    def run(argv, cwd, name, timeout=0):
        calls.append(list(argv))
        prog = argv[0]

        def copy(src, dst, noise=0.0):
            with h5py.File(cwd / src, "r") as a, h5py.File(cwd / dst, "w") as b:
                d = a["timeseries"][:]
                if noise:
                    d = d + np.random.default_rng(9).normal(0, noise, d.shape).astype("float32")
                b.create_dataset("timeseries", data=d)
                for k in ("date", "bperp"):
                    b.create_dataset(k, data=a[k][:])
                for k, v in a.attrs.items():
                    b.attrs[k] = v
        if prog == "tropo_pyaps3.py":
            copy(argv[argv.index("-f") + 1], argv[argv.index("-o") + 1], 0.004 if add_noise_in_tropo else 0.0)
        elif prog in ("remove_ramp.py", "dem_error.py"):
            copy(argv[1], argv[argv.index("-o") + 1])
        elif prog == "timeseries2velocity.py":
            with h5py.File(cwd / argv[1], "r") as a:
                d = a["timeseries"][:]
                dates = [x.decode() for x in a["date"][:]]
                attrs = dict(a.attrs)
            t = np.array([(datetime.strptime(x, "%Y%m%d") - datetime.strptime(dates[0], "%Y%m%d")).days
                          for x in dates]) / 365.25
            flat = d.reshape(d.shape[0], -1)
            ok = np.isfinite(flat).all(axis=0)
            v = np.full(flat.shape[1], np.nan, "float32")
            v[ok] = np.polyfit(t, flat[:, ok], 1)[0]
            with h5py.File(cwd / argv[argv.index("-o") + 1], "w") as b:
                b.create_dataset("velocity", data=v.reshape(d.shape[1:]))
                b.create_dataset("velocityStd", data=np.full(d.shape[1:], 0.001, "float32"))
                for k, val in attrs.items():
                    b.attrs[k] = val
                b.attrs["FILE_TYPE"] = "velocity"
    return run


def chain_env(tmp_path, monkeypatch, n=40, seg=10, hhmmss="115830", **cfgkw):
    """A DISP project over a ~4 km box: synthetic granules on the real UTM
    grid of that box, one cached granule (window + coordinates), MintPy faked."""
    from pyproj import Transformer
    cfg = make_cfg(tmp_path, bbox=[-87.52, 38.29, -87.475, 38.325], orbit_direction="descending", track=77,
                   mintpy_deramp="no", mintpy_tropo_method="pyaps", disp_baseline_dir="",
                   disp_target_grid_file="", disp_geometry_file="", min_temporal_coherence=0.4,
                   reference_lalo="38.3075,-87.4975", disp_min_valid_fraction=0.5, disp_frame_id="")
    for k, v in cfgkw.items():
        setattr(cfg, k, v)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
    xs_, ys_ = [], []
    for lo in (cfg.bbox[0], cfg.bbox[2]):
        for la in (cfg.bbox[1], cfg.bbox[3]):
            x, y = tr.transform(lo, la); xs_.append(x); ys_.append(y)
    win = (min(xs_), max(xs_), min(ys_), max(ys_), 32616)
    xs = np.arange(np.floor(win[0] / 30) * 30 + 45, win[1], 30.0)
    ys = np.arange(np.ceil(win[3] / 30) * 30 - 45, win[2], -30.0)
    ny, nx = ys.size, xs.size
    sim = sim_granules(n=n, seg=seg, ny=ny, nx=nx, noise=0.001, hhmmss=hhmmss)
    rng = np.random.default_rng(7)
    edge_dead = np.zeros((ny, nx), bool)
    edge_dead[:, :3] = True                       # pixels no longer measured in the last 30% of the record

    def extra(i):
        out = {"temporal_coherence": np.full((ny, nx), 0.8, "float32"),
               "water_mask": np.ones((ny, nx), "float32"),
               "timeseries_inversion_residuals": rng.normal(0, 1.5, (ny, nx)).astype("float32"),
               "connected_component_labels": (rng.random((ny, nx)) > 0.1).astype("float32")}
        return out
    for i in range(n - int(0.3 * n), n):
        sim.raw[i][edge_dead] = np.nan
    patch_reads(monkeypatch, sim, extra=extra, window=win)
    c = db._granule_cache_path(cfg, sim.names[0])
    c.parent.mkdir(parents=True, exist_ok=True)
    np.savez(c, __window=np.array(win, dtype="float64"), __x=xs, __y=ys, displacement=sim.raw[0])
    res = results_of(sim.names, track=77, direction="DESCENDING")
    monkeypatch.setattr(db, "search", lambda cfg_: res)
    monkeypatch.setattr(db, "prefetch", lambda cfg_, r: {"read": len(r), "failed": [], "uncorrected": []})
    grids = []

    def static_load(cfg_, grid, frame=None, want_dem=False):
        grids.append(dict(grid))
        sh = (grid["ny"], grid["nx"])
        return {"incidence_deg": np.full(sh, 38.0, "float32"), "height": np.full(sh, 120.0, "float32"),
                "los_east": np.full(sh, 0.6, "float32"), "los_north": np.full(sh, -0.1, "float32")}
    from insar_pipeline import static_layers
    monkeypatch.setattr(static_layers, "load", static_load)
    calls = []
    monkeypatch.setattr(dc, "_mintpy", _fake_mintpy(calls))
    return types.SimpleNamespace(cfg=cfg, sim=sim, xs=xs, ys=ys, calls=calls, grids=grids, edge_dead=edge_dead)


def test_chain_end_to_end_records_what_was_measured(tmp_path, monkeypatch):
    E = chain_env(tmp_path, monkeypatch)
    M = dc.run(E.cfg)
    rec = json.loads((M / "disp_chain.json").read_text())
    assert rec["status"] == "complete"
    # disp-5: the granule hour, not 23:48, reaches MintPy's ERA5 step
    with h5py.File(M / "timeseries.h5", "r") as h:
        assert h.attrs["CENTER_LINE_UTC"] == "43110" and h.attrs["ORBIT_DIRECTION"] == "DESCENDING"
    assert {"scatter_before_mm", "scatter_after_mm", "helped"} <= set(rec["troposphere"])
    # gap-geolocation-8: the static-layer grid is the real first pixel centre
    assert E.grids and E.grids[0]["x0"] == E.xs[0] and E.grids[0]["y0"] == E.ys[0]
    # disp-9: pixels held flat for 30% of the record are left out and counted
    assert rec["gap_fill"]["pixels_dropped_end_hold"] > 0
    # disp-4: corrections.json holds what was measured
    corr = json.loads((M / "corrections.json").read_text())
    assert corr["disp_corrections"]["status"] == "off" and corr["solid_earth_tides"] is False
    assert corr["ionosphere"] is None
    # disp-10: quality layers delivered as GeoTIFFs a reviewer can open
    import rasterio
    q = Path(E.cfg.project_dir) / "deliverables" / "technical" / "disp_quality"
    for name in ("valid_epoch_fraction", "network_misclosure_fraction", "anchor_suspect_count",
                 "anchor_conncomp0_share"):
        with rasterio.open(q / f"{name}.tif") as src:
            assert src.count == 1 and src.crs is not None and np.isfinite(src.read(1)).any()
    assert set(rec["quality_layers"]) >= {"networkMisclosureFraction", "anchorSuspect", "validEpochFraction"}
    assert rec["misclosure_vs_rate"] is not None
    # disp-3: no DEM-error step ran, so no residual chose the reference date
    assert not any(c[0] == "timeseries_rms.py" for c in E.calls)
    # disp-11: no remote read anywhere (coordinates from the cache)
    assert rec["stitch"]["grid_origin"].startswith("first pixel centre")


def test_chain_flags_a_tropo_correction_that_made_it_worse(tmp_path, monkeypatch, caplog):
    E = chain_env(tmp_path, monkeypatch)
    monkeypatch.setattr(dc, "_mintpy", _fake_mintpy(E.calls, add_noise_in_tropo=True))
    with caplog.at_level("ERROR"):
        M = dc.run(E.cfg)
    rec = json.loads((M / "disp_chain.json").read_text())
    assert rec["troposphere"]["helped"] is False and "WORSE" in caplog.text


def test_chain_ignores_a_residual_it_did_not_write(tmp_path, monkeypatch):
    E = chain_env(tmp_path, monkeypatch)
    M = E.cfg.mintpy_dir
    M.mkdir(parents=True)
    with h5py.File(M / "timeseriesResidual.h5", "w") as h:          # left by another run
        h.create_dataset("timeseries", data=np.zeros((2, 2, 2), "float32"))
    dc.run(E.cfg)
    assert not any(c[0] == "timeseries_rms.py" for c in E.calls)
    ref = [c for c in E.calls if c[0] == "reference_date.py"][0]
    assert ref[-1] == f"{E.sim.sec[0]:%Y%m%d}"


# ================================================================ disp-3 (L10 part): the guard
@pytest.mark.parametrize("what", ["ifgram", "isce_velocity", "run_products"])
def test_chain_refuses_a_mintpy_folder_that_is_not_disp(tmp_path, monkeypatch, what):
    cfg = make_cfg(tmp_path)
    M = cfg.mintpy_dir
    (M / "geo").mkdir(parents=True)
    (M / "inputs").mkdir()
    sentinel = M / "geo" / "geo_timeseries_ERA5_ramp_demErr.h5"
    sentinel.write_bytes(b"official")
    if what == "ifgram":
        with h5py.File(M / "inputs" / "ifgramStack.h5", "w") as h:
            h.attrs["x"] = 1
    elif what == "isce_velocity":
        with h5py.File(M / "geo" / "geo_velocity.h5", "w") as h:
            h.attrs["PROCESSOR"] = "isce"
    else:
        (M / "run_products.json").write_text("{}")
    mt = sentinel.stat().st_mtime_ns
    monkeypatch.setattr(db, "search", lambda c: (_ for _ in ()).throw(AssertionError("searched")))
    with pytest.raises(RuntimeError, match="Refusing to write DISP-S1 products"):
        dc.run(cfg)
    with pytest.raises(RuntimeError, match="Refusing"):
        db.run(cfg)
    assert sentinel.read_bytes() == b"official" and sentinel.stat().st_mtime_ns == mt
    assert not (M / "timeseries.h5").exists()


def test_chain_accepts_its_own_previous_products(tmp_path):
    cfg = make_cfg(tmp_path)
    (cfg.mintpy_dir / "geo").mkdir(parents=True)
    with h5py.File(cfg.mintpy_dir / "geo" / "geo_velocity.h5", "w") as h:
        h.attrs["PROCESSOR"] = "opera_disp"
    db.refuse_foreign_mintpy(cfg)          # no raise


# ================================================================ crosscheck / injection fixtures
def _write_h5(path, data: dict, attrs: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        for k, v in data.items():
            h.create_dataset(k, data=v)
        for k, v in attrs.items():
            h.attrs[k] = v


def _grid_attrs(y0, x0, dy, dx, ny, nx, extra=None):
    a = {"Y_FIRST": repr(y0), "X_FIRST": repr(x0), "Y_STEP": repr(dy), "X_STEP": repr(dx),
         "LENGTH": str(ny), "WIDTH": str(nx), "Y_UNIT": "degrees", "X_UNIT": "degrees"}
    a.update(extra or {})
    return a


def _bowl(ny, nx, dy, dx, r0, c0, sigma_m, peak):
    lat_m, lon_m = abs(dy) * 110_950, abs(dx) * 111_320 * np.cos(np.radians(38.3))
    rr, cc = np.mgrid[0:ny, 0:nx]
    d2 = ((rr - r0) * lat_m) ** 2 + ((cc - c0) * lon_m) ** 2
    return peak * np.exp(-d2 / (2 * sigma_m ** 2))


def official_project(tmp_path, ny=160, nx=160, n_dates=40, noise_mm=1.0, bowl=None, mine_box=None,
                     deramp=True, seed=0, dates_end=None, name="official"):
    """A finished phase-linked-style project on a 15 m geo grid (velocity,
    coherence, masks, geometry, a geocoded time series with dates/bperp, the
    deramp mask MintPy used) - everything the checks read, nothing else."""
    P = tmp_path / name
    M = P / "mintpy"
    y0, x0, dy, dx = 38.33, -87.52, -0.000135, 0.000172
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    base = datetime(2017, 1, 3)
    dates = [(base + timedelta(days=12 * i)).strftime("%Y%m%d") for i in range(n_dates)]
    if dates_end:
        dates = [d for d in dates if d <= dates_end]
    t = np.array([(datetime.strptime(d, "%Y%m%d") - base).days / 365.25 for d in dates])
    # farmland-like noise: spatially correlated (~200 m) plus a white part,
    # so a region's mean does not average it away as white noise would
    corr = gaussian_filter(rng.normal(0, 1, (ny, nx)), 13)
    vel = (noise_mm * corr / corr.std() + rng.normal(0, 0.3 * noise_mm, (ny, nx))) / 1000.0
    if bowl is not None:
        vel = vel + _bowl(ny, nx, dy, dx, *bowl) / 1000.0
    tc = rng.uniform(0.55, 0.95, (ny, nx)).astype("float32")
    ref_r, ref_c = 10, 10
    lat_ref, lon_ref = y0 + (ref_r + 0.5) * dy, x0 + (ref_c + 0.5) * dx
    A = _grid_attrs(y0, x0, dy, dx, ny, nx, {"PROCESSOR": "isce", "REF_Y": str(ref_r), "REF_X": str(ref_c),
                                             "REF_LAT": str(lat_ref), "REF_LON": str(lon_ref),
                                             "REF_DATE": dates[0], "CENTER_LINE_UTC": "85663.0",
                                             "mintpy.timeFunc.polynomial": "1",
                                             "mintpy.timeFunc.excludeDate": "[]", "mintpy.timeFunc.periodic": "[]"})
    _write_h5(M / "geo" / "geo_velocity.h5", {"velocity": vel.astype("float32"),
                                               "velocityStd": np.full((ny, nx), 0.001, "float32")},
              {**A, "FILE_TYPE": "velocity"})
    _write_h5(M / "geo" / "geo_temporalCoherence.h5", {"temporalCoherence": tc}, {**A, "FILE_TYPE": "temporalCoherence"})
    _write_h5(M / "geo" / "geo_maskTempCoh.h5", {"mask": tc >= 0.4}, {**A, "FILE_TYPE": "mask"})
    _write_h5(M / "geo" / "geo_geometryRadar.h5", {"incidenceAngle": np.full((ny, nx), 38.0, "float32"),
                                                    "height": np.full((ny, nx), 120.0, "float32")},
              {**A, "FILE_TYPE": "geometry"})
    bperp = (rng.normal(0, 60, len(dates)) + 30 * t).astype("float32")
    cube = (vel[None] * t[:, None, None]).astype("float32")
    gj = ""
    if mine_box is not None:
        r0, r1, c0, c1 = mine_box
        w, e = x0 + c0 * dx, x0 + c1 * dx
        n_, s_ = y0 + r0 * dy, y0 + r1 * dy
        gj = P / "context" / "mines.geojson"
        gj.parent.mkdir(parents=True, exist_ok=True)
        gj.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                      "geometry": {"type": "Polygon", "coordinates": [[[w, s_], [e, s_], [e, n_], [w, n_], [w, s_]]]}}]}))
    ts_attrs = {**A, "FILE_TYPE": "timeseries", "mintpy.topographicResidual.polyOrder": "2",
                "mintpy.topographicResidual.excludeDate": "[]", "mintpy.topographicResidual.stepDate": "[]"}
    if deramp:
        dmask = tc >= 0.4
        if mine_box is not None:
            r0, r1, c0, c1 = mine_box
            dmask[max(0, r0 - 20):r1 + 20, max(0, c0 - 20):c1 + 20] = False     # mine + buffer out of the fit
        _write_h5(M / "maskDeramp.h5", {"mask": dmask}, {**A, "FILE_TYPE": "mask"})
        ts_attrs.update({"mintpy.deramp": "linear", "mintpy.deramp.maskFile": str(M / "maskDeramp.h5")})
    _write_h5(M / "geo" / "geo_timeseries_ERA5_ramp_demErr.h5",
              {"timeseries": cube, "date": np.array(dates, dtype="S8"), "bperp": bperp}, ts_attrs)
    lat1, lon1 = y0 + ny * dy, x0 + nx * dx
    cfg = PipelineConfig(project_dir=P, bbox=[x0 + 2 * dx, lat1 - 2 * dy, lon1 - 2 * dx, y0 + 2 * dy],
                         start_date="2017-01-01", end_date="2019-12-31")
    for k, v in dict(backend="miaplpy", track=19, orbit_direction="ascending", deramp_exclude_geojson=str(gj),
                     acceptance_far_field_buffer_m=300.0, min_temporal_coherence=0.4, display_units="in",
                     reference_lalo=f"{lat_ref:.6f},{lon_ref:.6f}").items():
        setattr(cfg, k, v)
    return types.SimpleNamespace(cfg=cfg, vel=vel, tc=tc, attrs=A, dates=dates, t=t, bperp=bperp,
                                 dy=dy, dx=dx, y0=y0, x0=x0, ny=ny, nx=nx)


# ================================================================ methodology-11: injection-recovery
def test_injection_recovery_matches_the_operators_and_the_rule(tmp_path):
    from insar_pipeline import injection_test as it
    O = official_project(tmp_path, mine_box=(55, 105, 55, 105), noise_mm=2.0)
    p = it.run(O.cfg, out_dir=tmp_path / "out", n_farmland=3, n_mine=2, sizes_km=(0.6,),
               rates_mm_yr=(2.0, 20.0), types=("trough", "settlement"), seed=11)
    rec = json.loads(p.read_text())
    assert rec["status"] == "complete" and rec["seed"] == 11 and rec["sizes_km"] == [0.6]
    assert rec["rates_mm_yr"] == [2.0, 20.0] and "unwrapping" in rec["not_covered"]
    assert rec["deramp"]["modelled"] and rec["deramp"]["coordinates"] == "geo"

    # the temporal gain equals an independent DEM-error + velocity fit
    t, bp = O.t, O.bperp.astype(float)
    G = np.c_[bp, np.ones_like(t), t, t ** 2]
    f = t - t[0]
    f_dem = f - bp * np.linalg.lstsq(G, f, rcond=None)[0][0]
    assert rec["temporal_gain"]["trough"] == pytest.approx(np.polyfit(t, f_dem, 1)[0], abs=1e-3)
    g = it.time_function("settlement", t)
    assert np.polyfit(t, g, 1)[0] == pytest.approx(1.0)              # the stated rate is the straight-line rate
    g_dem = g - bp * np.linalg.lstsq(G, g, rcond=None)[0][0]
    assert rec["temporal_gain"]["settlement"] == pytest.approx(np.polyfit(t, g_dem, 1)[0], abs=1e-3)

    # each case's chain attenuation equals the analytic plane fit over the mask
    with h5py.File(O.cfg.mintpy_dir / "maskDeramp.h5", "r") as h:
        dm = h["mask"][:]
    rows, cols = np.mgrid[0:O.ny, 0:O.nx]
    lat = O.y0 + (rows + 0.5) * O.dy
    lon = O.x0 + (cols + 0.5) * O.dx
    Gm = np.c_[rows[dm], cols[dm], np.ones(int(dm.sum()))].astype(float)
    up = np.cos(np.radians(38.0))
    for c in rec["cases"]:
        if c["type"] != "trough" or not c.get("testable"):
            continue
        s = rec["sites"][c["site"]]
        shape = it.gaussian(lat, lon, s["lat"], s["lon"], c["width_km"]) * up
        co = np.linalg.lstsq(Gm, shape[dm], rcond=None)[0]
        plane = co[0] * rows + co[1] * cols + co[2]
        plane -= co[0] * 10 + co[1] * 10 + co[2]                   # re-referenced at REF_Y/REF_X
        region = (shape >= 0.5 * up) & (O.tc >= 0.4)
        expect = (shape - plane)[region].mean() / shape[region].mean() * rec["temporal_gain"]["trough"]
        assert c["chain_attenuation"] == pytest.approx(expect, rel=0.05)
        if c["kind"] == "mine":                                     # inside the exclusion: ~untouched
            assert 0.9 <= c["chain_attenuation"] <= 1.1

    by = {(c["kind"], c["type"], c["rate_mm_yr"]): c for c in rec["cases"]}
    big = [c for c in rec["cases"] if c["rate_mm_yr"] == 20.0 and c["kind"] == "mine" and c["type"] == "trough"]
    assert big and all(c["detected"] for c in big)
    assert all(0.85 <= c["recovered_over_planted"] <= 1.15 for c in big)
    small = [c for c in rec["cases"] if c["rate_mm_yr"] == 2.0 and c.get("testable")]
    assert small and not any(c["detected"] for c in small)
    # E4 (trust review 7 Oct 2026): technical only - no client sentence; the
    # record says which rule 'found' uses and summarises beside the control
    assert by and rec["summary"] and rec["sentence"] is None and rec["aliasing_ceiling"]["quarter_wavelength_mm"] == 13.9
    assert rec["technical_summary"].startswith("Technical record") and "report-area rule" in rec["decision_rule"]


def test_injection_cli_writes_to_out_dir(tmp_path):
    import yaml
    from insar_pipeline import injection_test as it
    O = official_project(tmp_path, mine_box=(55, 105, 55, 105))
    raw = {"project_dir": str(O.cfg.project_dir), "bbox": list(O.cfg.bbox), "start_date": "2017-01-01",
           "end_date": "2019-12-31", "backend": "miaplpy", "stack_virtual_merge": "False",
           "track": 19, "orbit_direction": "ascending",
           "deramp_exclude_geojson": O.cfg.deramp_exclude_geojson, "acceptance_far_field_buffer_m": 300.0,
           "reference_lalo": O.cfg.reference_lalo}
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(raw))
    out = tmp_path / "cli_out"
    assert it.main([str(cp), "--out-dir", str(out), "--sites", "1", "--mine-sites", "0",
                    "--sizes", "0.6,5", "--rates", "20"]) == 0
    rec = json.loads((out / "injection_recovery.json").read_text())
    assert rec["status"] == "complete" and not (O.cfg.project_dir / "deliverables" / "technical").exists()
    assert rec["sizes_skipped_km"] == [5.0] and {c["width_km"] for c in rec["cases"]} == {0.6}


def test_injection_failure_is_written_as_failed(tmp_path):
    from insar_pipeline import injection_test as it
    O = official_project(tmp_path)
    (O.cfg.mintpy_dir / "geo" / "geo_geometryRadar.h5").unlink()     # no incidence anywhere
    with pytest.raises(RuntimeError, match="vertical-to-LOS"):
        it.run(O.cfg, out_dir=tmp_path / "o", n_farmland=1, n_mine=0, sizes_km=(0.6,), rates_mm_yr=(20.0,))
    assert json.loads((tmp_path / "o" / "injection_recovery.json").read_text())["status"] == "failed"


# ================================================================ methodology-11 / disp-8: the cross-check
def _disp_products(O, sib, seed=5, noise_mm=1.0, flip=False, dates=None):
    """What disp_chain.run leaves in the sibling: the record's grid at factor 2."""
    from insar_pipeline import crosscheck as cc
    ny, nx = O.ny // 2, O.nx // 2
    v = O.vel[: ny * 2, : nx * 2].reshape(ny, 2, nx, 2).mean(axis=(1, 3))
    if flip:
        v = -v
    v = v + np.random.default_rng(seed).normal(0, noise_mm / 1000, v.shape)
    A = _grid_attrs(O.y0, O.x0, 2 * O.dy, 2 * O.dx, ny, nx, {"PROCESSOR": "opera_disp", "REF_Y": "5", "REF_X": "5"})
    G = Path(sib.mintpy_dir) / "geo"
    tc = O.tc[::2, ::2][:ny, :nx]
    _write_h5(G / "geo_velocity.h5", {"velocity": v.astype("float32")}, {**A, "FILE_TYPE": "velocity"})
    _write_h5(G / "geo_temporalCoherence.h5", {"temporalCoherence": tc}, A)
    _write_h5(G / "geo_maskTempCoh.h5", {"mask": tc >= 0.4}, A)
    _write_h5(G / "geo_geometryRadar.h5", {"incidenceAngle": np.full((ny, nx), 38.0, "float32")}, A)
    dates = dates or O.dates
    t = np.array([(datetime.strptime(d, "%Y%m%d") - datetime.strptime(dates[0], "%Y%m%d")).days / 365.25
                  for d in dates])
    _write_h5(G / "geo_timeseries_ERA5.h5", {"timeseries": (v[None] * t[:, None, None]).astype("float32"),
                                             "date": np.array(dates, dtype="S8")}, A)
    (Path(sib.mintpy_dir) / "disp_chain.json").write_text(json.dumps(
        {"status": "complete", "grid": {"source": str(cc.product(O.cfg, "velocity.h5"))}}))


def _xcheck_env(tmp_path, monkeypatch, flip=False, disp_dates=None, **kw):
    from insar_pipeline import crosscheck as cc
    # 8 km square: the 1 km comparison needs at least 50 one-km blocks
    O = official_project(tmp_path, ny=540, nx=540, bowl=(270, 270, 250.0, -40.0), noise_mm=1.0, **kw)
    lat_c = O.y0 + 270.5 * O.dy
    lon_c = O.x0 + 270.5 * O.dx
    box = f"{lat_c - 0.004},{lon_c - 0.005},{lat_c + 0.004},{lon_c + 0.005}"
    O.cfg.report_areas = ("Full Satellite Imaging:full", f"Bowl:{box}")
    rep = O.cfg.project_dir / "deliverables" / "client_report.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({"features": [{"lat": lat_c, "lon": lon_c, "vertical_relative_mm_yr": -45.0,
                                             "bbox": [lon_c - 0.004, lat_c - 0.003, lon_c + 0.004, lat_c + 0.003]}]}))
    names = [f"OPERA_L3_DISP-S1_IW_F04861_VV_20170103T234758Z_{d}T234758Z_v1.0_20250101T000000Z.nc"
             for d in (disp_dates or O.dates)]
    seen = {}
    monkeypatch.setattr(cc, "_search_disp", lambda sib: (seen.setdefault("sib", sib), results_of(names))[1])
    monkeypatch.setattr(cc, "_run_disp_chain", lambda sib: _disp_products(O, sib, flip=flip, dates=disp_dates))
    monkeypatch.setattr(cc, "_disk_ok", lambda sib, n: (True, "test"))
    return O, seen


def test_crosscheck_writes_per_area_and_feature_agreement(tmp_path, monkeypatch):
    from insar_pipeline import crosscheck as cc
    O, seen = _xcheck_env(tmp_path, monkeypatch)
    p = cc.run(O.cfg)
    assert p == O.cfg.project_dir / "deliverables" / "technical" / "cross_check.json"
    rec = json.loads(p.read_text())
    assert rec["available"] and rec["status"] == "complete" and rec["gates_verdict"] is False
    sib = seen["sib"]
    assert sib.backend == "disp" and sib.disp_record_chain and sib.disp_target_grid_factor == 2
    assert sib.disp_target_grid_file == str(cc.product(O.cfg, "velocity.h5"))
    assert Path(sib.project_dir).name == "official_disp_check" and sib.track == 19
    assert rec["one_km_means"]["correlation"] > 0.8 and rec["indicative_pass"] is True
    bowl = next(a for a in rec["areas"] if a["name"] == "Bowl")
    assert bowl["agree"] is True and bowl["official_distinguishable"] and bowl["disp_distinguishable"]
    assert rec["features"][0]["confirmed"] is True
    assert "OPERA DISP-S1" in rec["sentence"]


def test_crosscheck_disagreement_is_reported_not_hidden(tmp_path, monkeypatch):
    from insar_pipeline import crosscheck as cc
    O, _ = _xcheck_env(tmp_path, monkeypatch, flip=True)
    rec = json.loads(cc.run(O.cfg).read_text())
    bowl = next(a for a in rec["areas"] if a["name"] == "Bowl")
    assert bowl["agree"] is False and rec["indicative_pass"] is False
    assert rec["features"][0]["confirmed"] is False


def test_crosscheck_compares_over_the_common_window(tmp_path, monkeypatch):
    from insar_pipeline import crosscheck as cc
    O0 = official_project(tmp_path / "probe", ny=4, nx=4)
    disp_dates = [d for d in O0.dates if d <= O0.dates[25]]          # the DISP archive ends earlier
    O, _ = _xcheck_env(tmp_path, monkeypatch, disp_dates=disp_dates)
    rec = json.loads(cc.run(O.cfg).read_text())
    assert rec["common_window"] == [O.dates[0], disp_dates[-1]]
    assert rec["official_refitted_to_common_window"] is True
    assert rec["disp_refitted_to_common_window"] is False


def test_crosscheck_without_disp_coverage_says_so(tmp_path, monkeypatch):
    from insar_pipeline import crosscheck as cc
    O = official_project(tmp_path)

    def none(sib):
        # a missing frame is a result; any other ValueError is a failed check (#53)
        raise db.NoFrameOnTrack("no DISP-S1 frame over this AOI is on track 19 ascending")
    monkeypatch.setattr(cc, "_search_disp", none)
    monkeypatch.setattr(cc, "_run_disp_chain", lambda sib: (_ for _ in ()).throw(AssertionError("ran")))
    rec = json.loads(cc.run(O.cfg).read_text())
    assert rec["available"] is False and "track 19" in rec["reason"]
    assert rec["sentence"].startswith("No independent satellite product")
    monkeypatch.setattr(cc, "_search_disp", lambda sib: [])
    rec = json.loads(cc.run(O.cfg).read_text())
    assert rec["available"] is False and "North America" in rec["reason"]


def test_crosscheck_failure_is_written_and_raised(tmp_path, monkeypatch):
    from insar_pipeline import crosscheck as cc
    O, _ = _xcheck_env(tmp_path, monkeypatch)
    monkeypatch.setattr(cc, "_run_disp_chain", lambda sib: (_ for _ in ()).throw(RuntimeError("stitch refused")))
    with pytest.raises(RuntimeError):
        cc.run(O.cfg)
    rec = json.loads((O.cfg.project_dir / "deliverables" / "technical" / "cross_check.json").read_text())
    assert rec["status"] == "failed" and "stitch refused" in rec["reason"]


def test_compare_products_uses_the_shared_core_and_each_products_geometry():
    import importlib.util
    from insar_pipeline import crosscheck as cc
    spec = importlib.util.spec_from_file_location(
        "compare_products", Path(__file__).resolve().parents[1] / "scripts" / "compare_products.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.agree is cc.agree and mod.onto is cc.onto and not hasattr(mod, "LOS_UP")
    assert mod.rate_over_window is cc.rate_over_window


def test_rate_over_window_streams_the_cube(tmp_path):
    from insar_pipeline import crosscheck as cc
    O = official_project(tmp_path, ny=20, nx=20)
    f = O.cfg.mintpy_dir / "geo" / "geo_timeseries_ERA5_ramp_demErr.h5"
    r = cc.rate_over_window(f, O.dates[5], O.dates[20])
    assert np.allclose(r, O.vel, atol=1e-6)


def test_static_layers_take_the_newest_production_of_the_frame(tmp_path, monkeypatch):
    """disp-11: sorting by name and taking the first picked the OLDEST
    production whenever the static layers had been published twice."""
    import asf_search
    from insar_pipeline import static_layers as sl

    def prod(version, when):
        stem = f"OPERA_L3_DISP-S1-STATIC_F04861_20140403_S1A_v{version}_{when}"
        return types.SimpleNamespace(properties={
            "fileName": stem + ".h5", "productVersion": version, "processingDate": when,
            "frameNumber": 4861, "url": f"https://x/{stem}_line_of_sight_enu.tif",
            "additionalUrls": [f"https://x/{stem}_dem.tif", f"https://x/{stem}_layover_shadow_mask.tif"]})
    res = [prod("1.0", "2025-01-01T00:00:00Z"), prod("1.0", "2025-06-01T00:00:00Z")]
    monkeypatch.setattr(asf_search, "search", lambda **k: list(res))
    out = sl.find(make_cfg(tmp_path), frame=4861)
    assert "2025-06-01" in out["los_enu"] and "2025-06-01" in out["dem"]
