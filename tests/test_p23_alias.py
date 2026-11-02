"""PLAN phase 2, F2 / F1 / F3 (review 8 Oct 2026, group 'alias'): the
pre-unwrap aliasing diagnostic, the detectability preflight and the
fast-network options (aliasing.py, hooks in miaplpy_backend.py,
stack_backend.py and plan_run.py).

What is checked, on a synthetic phase-linked stack (6 dates, 2 nearest
connections, 3 x 5 looks of ~40 m, a southern-hemisphere lat/lon grid): a
cone that sinks 30 cycles between the 3rd and 4th dates (about one cycle per
look cell: aliased on the looked grid), a slow cone elsewhere in time, and a
patch that decorrelates on the last date while the ground around it stays
coherent:

  - the pure measures: the full-resolution gradient reads above half a cycle
    per look cell where the multilooked interferogram cannot; noise is not a
    gradient; residues; a coherence hole is local, a scene-wide low is not;
  - pre_unwrap: level 2 over the cone in the three pairs spanning the event
    and only there, level 1 (hole) on the patch in the pairs with the last
    date; per-pair, per-interval and per-epoch summaries; nothing measured
    twice; off for a client config (nothing written); never raises;
  - export_products: deliverables/quality_aliasing_risk.tif (uint8, 0/1/2,
    255 outside the crop) and aliasing_risk.json (method, thresholds,
    lower-bound text, pairs, intervals, epochs), read by the harness's own
    validation.read_flag_layer as a grading flag; risk_between for one
    interval; the old flag relabelled a lower bound only where it is not;
    an earlier run's flag set aside when this run writes none;
  - F1: the quarter wavelength from the product (C-band vs L-band changes the
    verdict), the nominal value only before any product and labelled so,
    polygon properties, the half-width from the polygon, the plan note in
    plan_stack / the re-link plan / plan_run, nothing for a client config;
  - F3: nearest pairs only inside fast_zone_geojson (longer pairs set to
    MintPy's no-data there, originals kept), changed, removed and re-unwrapped
    cases, the MintPy loader hook; the re-link of another project's stack
    (links only, nothing written in the source, refusals, its plan, the SLC
    stack deletion that must not follow the link);
  - the hooks: pre_unwrap before snaphu and the fast zones after it, on both
    routes; config keys and their checks.

aliasing.py did not exist before 8 Oct 2026, and the hooks, the re-link and
the plan notes are new, so every test here fails on the code before it. No
network, nothing under ~/insar_projects.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from insar_pipeline import aliasing as A
from insar_pipeline import miaplpy_backend as mb
from insar_pipeline import stack_backend as sb
from insar_pipeline.config import PipelineConfig

AL, RL = 3, 5                      # azimuth, range looks
LM, WM = 110, 110                  # multilooked frame
R0, R1, C0, C1 = 5, 105, 5, 105    # unwrap crop (multilooked rows/cols)
CELL = 40.0                        # metres per look cell
LAT0, LON0 = -34.20, 150.65
KY = 110_574.0
KX = 111_320.0 * math.cos(math.radians(LAT0))
DATES = ["20200101", "20200113", "20200125", "20200206", "20200218", "20200301"]
PEAK_CYC = [0.0, 1.0, 2.0, 32.0, 33.0, 34.0]     # cone depth in cycles: 30 cycles between dates 3 and 4
CONE_RC, CONE_R = (50, 50), 1200.0               # crop cell, metres
PATCH_RC, PATCH_R = (18, 82), 300.0              # decorrelates on the last date
FAST = {"20200113_20200206", "20200125_20200206", "20200125_20200218"}     # pairs spanning the event


def network(dates, k=2):
    return [f"{dates[i]}_{dates[j]}" for i in range(len(dates)) for j in range(i + 1, min(len(dates), i + k + 1))]


def crop_latlon(r, c):
    """lat, lon of crop cell centres (r, c may be arrays)."""
    return LAT0 - (np.asarray(r) + R0 + 0.5) * CELL / KY, LON0 + (np.asarray(c) + C0 + 0.5) * CELL / KX


def lonlat_square(rc, half_m):
    lat, lon = crop_latlon(*rc)
    dlat, dlon = half_m / KY, half_m / KX
    ring = [[lon - dlon, lat - dlat], [lon + dlon, lat - dlat], [lon + dlon, lat + dlat], [lon - dlon, lat + dlat],
            [lon - dlon, lat - dlat]]
    return {"type": "Feature", "properties": {"name": "panel"},
            "geometry": {"type": "Polygon", "coordinates": [[[float(x), float(y)] for x, y in ring]]}}


def write_geojson(path, *features):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": list(features)}))
    return path


def make_cfg(root: Path, **over) -> PipelineConfig:
    kw = dict(project_dir=root, bbox=[150.6, -34.3, 150.8, -34.1], start_date="2020-01-01",
              end_date="2020-03-31", backend="miaplpy", stack_reference_date="20200101", track=147,
              stack_swaths="1 2", stack_polarization="vv", stack_range_looks=RL, stack_azimuth_looks=AL,
              miaplpy_network_connections=2, aliasing_diagnostic="on")
    kw.update(over)
    return PipelineConfig(**kw)


def date_phases(seed=7):
    """Full-resolution linked phase of the crop, one array per date (wrapped)."""
    H, W = (R1 - R0) * AL, (C1 - C0) * RL
    ii, jj = np.mgrid[0:H, 0:W]
    y, x = (ii + 0.5) / AL * CELL, (jj + 0.5) / RL * CELL
    d = np.hypot(y - (CONE_RC[0] + 0.5) * CELL, x - (CONE_RC[1] + 0.5) * CELL)
    shape = np.clip(1 - d / CONE_R, 0, None)
    rng = np.random.default_rng(seed)
    patch = np.hypot(y - (PATCH_RC[0] + 0.5) * CELL, x - (PATCH_RC[1] + 0.5) * CELL) < PATCH_R
    out = []
    for k, p in enumerate(PEAK_CYC):
        ph = 2 * np.pi * p * shape
        if k == len(PEAK_CYC) - 1:
            ph = np.where(patch, rng.uniform(-np.pi, np.pi, ph.shape), ph)
        out.append(np.angle(np.exp(1j * ph)).astype(np.float32))
    return np.stack(out)


def make_project(tmp_path, name="proj", **over):
    """A phase-linked project as miaplpy_backend leaves it just before
    unwrapping: crop window, merged geometry, SLC metadata, the linked phase
    series, the exported interferograms and the linked temporal coherence."""
    root = tmp_path / name
    S = root / "stack"
    G = S / "merged" / "geom_reference"
    G.mkdir(parents=True)
    (S / "crop_window.json").write_text(json.dumps({"r0": R0, "r1": R1, "c0": C0, "c1": C1,
                                                    "length": LM, "width": WM}))
    rr, cc = np.mgrid[0:LM, 0:WM]
    (LAT0 - (rr + 0.5) * CELL / KY).astype(np.float64).tofile(G / "lat.rdr")
    (LON0 + (cc + 0.5) * CELL / KX).astype(np.float64).tofile(G / "lon.rdr")
    for d in DATES[:1]:
        x = S / "merged" / "SLC" / d / f"{d}.slc.full.xml"
        x.parent.mkdir(parents=True)
        x.write_text(f'<imageFile><property name="width"><value>{WM * RL}</value></property>'
                     f'<property name="length"><value>{LM * AL}</value></property></imageFile>')
    ph = date_phases()
    inv = root / "miaplpy" / "inverted"
    inv.mkdir(parents=True)
    with h5py.File(inv / "phase_series.h5", "w") as f:
        f.create_dataset("phase", data=ph)
        f.create_dataset("date", data=np.array([d.encode() for d in DATES]))
    tc = np.full((LM, WM), np.nan, np.float32)
    tc[R0:R1, C0:C1] = 0.9
    np.savez_compressed(S / "merged" / "linked_temporal_coherence.npz", layer_order=np.array("test"),
                        average=tc, full=tc)
    idx = {d: i for i, d in enumerate(DATES)}
    pairs = network(DATES)
    Lc, Wc = R1 - R0, C1 - C0
    for p in pairs:
        a, b = p.split("_")
        z = np.exp(1j * (ph[idx[a]] - ph[idx[b]])).reshape(Lc, AL, Wc, RL).mean(axis=(1, 3))
        ifg = np.zeros((LM, WM), np.complex64)
        cor = np.zeros((LM, WM), np.float32)
        ifg[R0:R1, C0:C1] = z
        cor[R0:R1, C0:C1] = np.abs(z)
        pdir = S / "merged" / "interferograms" / p
        pdir.mkdir(parents=True)
        ifg.tofile(pdir / "filt_fine.int")
        cor.tofile(pdir / "filt_fine.cor")
    zone = write_geojson(tmp_path / f"{name}_zone.geojson", lonlat_square(CONE_RC, 1300.0))
    cfg = make_cfg(root, aliasing_zone_geojson=str(zone), **over)
    return cfg, pairs


def crop_disc(rc, r_m):
    r, c = np.mgrid[0:R1 - R0, 0:C1 - C0]
    return np.hypot((r - rc[0]) * CELL, (c - rc[1]) * CELL) < r_m


def write_velocity(cfg, step_m=20.0, wavelength="0.05546576"):
    """A geocoded product grid (20 m, lat/lon) covering the whole frame."""
    geo = cfg.mintpy_dir / "geo"
    geo.mkdir(parents=True, exist_ok=True)
    ny, nx = int(LM * CELL / step_m), int(WM * CELL / step_m)
    attrs = {"X_FIRST": str(LON0), "Y_FIRST": str(LAT0), "X_STEP": str(step_m / KX), "Y_STEP": str(-step_m / KY),
             "WIDTH": str(nx), "LENGTH": str(ny), "REF_DATE": DATES[0], "START_DATE": DATES[0],
             "END_DATE": DATES[-1], "UNIT": "m/year", "FILE_TYPE": "velocity"}
    if wavelength:
        attrs["WAVELENGTH"] = wavelength
    with h5py.File(geo / "geo_velocity.h5", "w") as f:
        f.create_dataset("velocity", data=np.zeros((ny, nx), np.float32))
        f.attrs.update(attrs)
    return (ny, nx), attrs


def product_rc(attrs, lat, lon):
    return (int((float(lat) - float(attrs["Y_FIRST"])) / float(attrs["Y_STEP"])),
            int((float(lon) - float(attrs["X_FIRST"])) / float(attrs["X_STEP"])))


# ======================================================================
# F2: the measures
# ======================================================================
def test_full_res_gradient_reads_beyond_half_a_cycle_where_the_looked_ifg_cannot():
    g = 0.8                                            # cycles per look cell along range
    h, w = 8, 10
    jj = np.arange(w * RL)[None, :].repeat(h * AL, 0)
    dphi = 2 * np.pi * g * jj / RL
    full = A.full_res_gradient(dphi, AL, RL)
    assert np.nanmedian(full["cyc_rg"][:, :-1]) == pytest.approx(g, abs=0.01)
    assert np.nanmedian(full["cons_rg"][:, :-1]) > 0.99
    assert A.gradient_exceeds(full)[:, :-1].all()
    # the same ramp multilooked: neighbouring cells differ by 0.8 cycle, which
    # wraps to 0.2 - the looked interferogram shows a slow, clean gradient
    looked = np.exp(1j * dphi).reshape(h, AL, w, RL).mean(axis=(1, 3))
    lg = A.looked_gradient(looked)
    assert np.nanmedian(lg["cyc_rg"][2:-2, 2:-2]) == pytest.approx(1 - g, abs=0.02)
    assert not A.gradient_exceeds(lg)[2:-2, 2:-2].any()


def test_noise_is_not_a_gradient_and_few_looks_raise_the_floor():
    rng = np.random.default_rng(1)
    noise = rng.uniform(-np.pi, np.pi, (60 * AL, 60 * 11))
    g = A.full_res_gradient(noise, AL, 11)
    assert np.nanmedian(g["cons_rg"]) < 0.25                       # ~sqrt(pi / 4N), N = 33
    assert A.gradient_exceeds(g).mean() < 0.002
    assert A.min_consistency(33) == A.GRADIENT_MIN_CONSISTENCY == 0.7
    assert A.min_consistency(15) == 0.7
    assert A.min_consistency(6) > 1.0                               # 3 x 2 looks: never a gradient flag


def test_the_block_estimator_ignores_smooth_noise_and_keeps_a_flank():
    """The phase-linking window makes noise smooth over ~15 pixels: over one
    look cell it can read as a consistent gradient (1-2% of still farmland
    per pair at Site B); over 3 x 3 look cells it does not. A flank five
    look cells wide still reads above half a cycle."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(5)
    h, w = 120, 120
    z = rng.normal(size=(h * AL, w * 11)) + 1j * rng.normal(size=(h * AL, w * 11))
    z = gaussian_filter(z.real, (5, 5)) + 1j * gaussian_filter(z.imag, (5, 5))
    smooth = np.angle(z)
    one = A.gradient_exceeds(A.full_res_gradient(smooth, AL, 11, block=1))
    three = A.gradient_exceeds(A.full_res_gradient(smooth, AL, 11, block=A.GRADIENT_BLOCK),
                               A.min_consistency(A.GRADIENT_BLOCK ** 2 * AL * 11))
    assert one.mean() > 0.002 and three.mean() < one.mean() / 10
    jj = np.arange(w * 11)[None, :].repeat(h * AL, 0)
    flank = (jj // 11 >= 50) & (jj // 11 < 55)                       # five look cells at 0.7 cycle each
    ramp = 2 * np.pi * 0.7 * np.cumsum(flank, axis=1) / 11
    g = A.full_res_gradient(ramp, AL, 11, block=A.GRADIENT_BLOCK)
    assert A.gradient_exceeds(g)[5:-5, 51:54].all()
    assert not A.gradient_exceeds(g)[5:-5, :48].any()


def test_residues_count_a_vortex():
    yy, xx = np.mgrid[0:20, 0:20]
    ph = np.angle(np.exp(1j * np.arctan2(yy - 9.5, xx - 9.5)))
    res = A.residues(ph)
    assert np.count_nonzero(res) == 1 and abs(int(res.sum())) == 1
    assert np.count_nonzero(A.residues(np.zeros((10, 10)))) == 0


def test_a_coherence_hole_is_local_and_a_scene_wide_low_is_not():
    coh = np.full((100, 100), 0.8, np.float32)
    coh[45:55, 45:55] = 0.1
    holes = A.coherence_holes(coh, None, 40.0, 40.0)
    assert holes[45:55, 45:55].all() and holes.sum() == 100
    low = np.full((100, 100), 0.2, np.float32)
    low[45:55, 45:55] = 0.1
    assert not A.coherence_holes(low, None, 40.0, 40.0).any()
    # too few reliable cells around: no judgement
    rel = np.zeros((100, 100), bool)
    rel[45:55, 45:55] = True
    assert not A.coherence_holes(coh, rel, 40.0, 40.0).any()


# ======================================================================
# F2: pre_unwrap on the synthetic stack
# ======================================================================
def test_pre_unwrap_flags_the_event_pairs_and_the_hole(tmp_path):
    cfg, pairs = make_project(tmp_path)
    s = A.pre_unwrap(cfg, pairs, route="miaplpy")
    assert "error" not in s, s
    names, L, stats = A.load_levels(cfg)
    assert names == sorted(pairs) and L.shape == (len(pairs), R1 - R0, C1 - C0)
    cone = crop_disc(CONE_RC, CONE_R - 150) & ~crop_disc(CONE_RC, 150)      # not the apex, not the rim
    patch = crop_disc(PATCH_RC, PATCH_R - 60)
    for p, lv in zip(names, L):
        if p in FAST:
            assert (lv[cone] == A.LEVEL_LIKELY).mean() > 0.95, p
        else:
            assert (lv[cone] == A.LEVEL_LIKELY).mean() == 0, p
        if p.endswith(DATES[-1]):
            assert (lv[patch] >= A.LEVEL_RISK).mean() > 0.9, p
            assert (lv[patch] == A.LEVEL_LIKELY).mean() < 0.05, p          # noise is not a gradient
        far = ~crop_disc(CONE_RC, CONE_R + 200) & ~crop_disc(PATCH_RC, PATCH_R + 200)
        assert (lv[far] > 0).mean() < 0.01, p
    by = {r["pair"]: r for r in stats}
    fast = by["20200125_20200206"]
    assert fast["gradient"]["rg"]["max_cycles_per_look_cell_zones"] > 0.9
    assert fast["zones"]["likely_share"] > 0.5 and fast["connection"] == 1 and fast["days"] == 12
    assert by["20200101_20200113"]["zones"]["likely_share"] == 0
    assert set(fast["residue_density"]) == {"crop", "zones", "ring"}
    summ = json.loads((A.record_dir(cfg) / "summary.json").read_text())
    assert summ["gradient_measured_from"].startswith("full-resolution linked phase")
    assert summ["gradient_min_consistency_used"] == 0.7
    iv = {(i["from"], i["to"]): i for i in summ["intervals"]}
    assert iv[("20200125", "20200206")]["flagged_share_zones"] > 0.3
    assert iv[("20200101", "20200113")]["flagged_share_zones"] == 0
    ep = {e["date"]: e for e in summ["epochs"]}
    assert ep["20200206"]["likely_share"] > 0 and ep["20200101"]["flagged_share"] == 0
    assert summ["overall"]["pairs"] == len(pairs) and summ["overall"]["likely_share_zones"] > 0.3


def test_a_chunked_series_is_read_a_chunk_at_a_time_with_the_same_result(tmp_path):
    cfg, pairs = make_project(tmp_path)
    A.pre_unwrap(cfg, pairs, route="miaplpy")
    _, plain, _ = A.load_levels(cfg)
    cfg2, _ = make_project(tmp_path, name="chunked")
    ps = cfg2.project_dir / "miaplpy" / "inverted" / "phase_series.h5"
    ph = date_phases()
    ps.unlink()
    with h5py.File(ps, "w") as f:                       # MiaplPy chunks over 5 dates (Site B: (5, 82, 289))
        f.create_dataset("phase", data=ph, chunks=(5, 60, 100))
        f.create_dataset("date", data=np.array([d.encode() for d in DATES]))
    lp = A._LinkedPhase(cfg2, 2)
    assert lp.block == 5
    lp.date(DATES[1])
    assert set(lp.cache) == set(DATES[:5])
    lp.close()
    A.pre_unwrap(cfg2, pairs, route="miaplpy")
    _, chunked, _ = A.load_levels(cfg2)
    assert (chunked == plain).all()


def test_pre_unwrap_measures_once_and_again_when_a_pair_changes(tmp_path):
    cfg, pairs = make_project(tmp_path)
    A.pre_unwrap(cfg, pairs, route="miaplpy")
    f = A.record_dir(cfg) / "pairs" / f"{pairs[0]}.npz"
    t0 = f.stat().st_mtime_ns
    time.sleep(0.01)
    A.pre_unwrap(cfg, pairs, route="miaplpy")
    assert f.stat().st_mtime_ns == t0
    (cfg.stack_dir / "merged" / "interferograms" / pairs[0] / "filt_fine.cor").touch()
    time.sleep(0.01)
    A.pre_unwrap(cfg, pairs, route="miaplpy")
    assert f.stat().st_mtime_ns != t0


def test_classic_route_reports_the_gradient_but_flags_only_holes(tmp_path):
    cfg, pairs = make_project(tmp_path, backend="stack")
    s = A.pre_unwrap(cfg, pairs, route="stack")
    assert "only coherence holes flag" in s["gradient_measured_from"]
    _, L, _ = A.load_levels(cfg)
    assert not (L == A.LEVEL_LIKELY).any()
    assert (L > 0).any()                               # the fast cone decorrelates the looked ifg: a hole


def test_off_for_a_client_config_and_never_raises(tmp_path):
    cfg, pairs = make_project(tmp_path, aliasing_diagnostic="auto")
    assert cfg.purpose == "client" and not A.enabled(cfg)
    assert "skipped" in A.pre_unwrap(cfg, pairs, route="miaplpy")
    assert not A.record_dir(cfg).exists()
    assert A.plan_note(cfg, DATES) is None
    assert A.enabled(make_cfg(tmp_path / "x", aliasing_diagnostic="auto", purpose="experiment"))
    assert A.enabled(make_cfg(tmp_path / "x", aliasing_diagnostic="auto", site_profile="mine_longwall"))
    assert A.enabled(make_cfg(tmp_path / "x", aliasing_diagnostic=True))
    assert not A.enabled(make_cfg(tmp_path / "x", aliasing_diagnostic=False, purpose="experiment"))
    # a broken file: logged and recorded, the run goes on
    cfg.aliasing_diagnostic = "on"
    (cfg.stack_dir / "merged" / "interferograms" / pairs[0] / "filt_fine.cor").write_bytes(b"short")
    r = A.pre_unwrap(cfg, pairs, route="miaplpy")
    assert "error" in r
    assert "error" in json.loads((A.record_dir(cfg) / "summary.json").read_text())


# ======================================================================
# F2: products
# ======================================================================
def test_export_writes_the_flag_the_harness_grades_with(tmp_path):
    cfg, pairs = make_project(tmp_path)
    A.pre_unwrap(cfg, pairs, route="miaplpy")
    shape, attrs = write_velocity(cfg)
    r = A.export_products(cfg)
    assert "tif" in r, r
    import rasterio
    D = cfg.project_dir / "deliverables"
    with rasterio.open(D / "quality_aliasing_risk.tif") as src:
        a = src.read(1)
        tags = src.tags()
        assert src.nodata == 255 and a.dtype == np.uint8 and a.shape == shape
        assert "lower bound" in (src.descriptions[0] or "").lower()
    assert set(np.unique(a)) <= {0, 1, 2, 255}
    lat, lon = crop_latlon(CONE_RC[0] + 10, CONE_RC[1])
    assert a[product_rc(attrs, lat, lon)] == 2
    lat, lon = crop_latlon(*PATCH_RC)
    assert a[product_rc(attrs, lat, lon)] == 1
    lat, lon = crop_latlon(90, 15)
    assert a[product_rc(attrs, lat, lon)] == 0
    assert a[0, 0] == 255                              # outside the radar crop
    assert float(tags["QUARTER_WAVELENGTH_MM"]) == pytest.approx(13.87, abs=0.01)
    assert "WAVELENGTH" in tags["WAVELENGTH_SOURCE"]
    js = json.loads((D / "aliasing_risk.json").read_text())
    assert js["values"]["0"].startswith("no sign of aliasing found") and js["values"]["2"].startswith("likely")
    assert "LOWER BOUND" in js["lower_bound"]
    assert js["thresholds"]["gradient_limit_cycles_per_look_cell"] == 0.5 and js["thresholds"]["hole_floor"] == 0.4
    assert len(js["pairs"]) == len(pairs) and js["intervals"] and js["epochs"]
    assert js["record_check"] == {"checked": True, "complete": True, "unwrapped_not_measured": [],
                                  "rewritten_since_measured": [], "fast_zones_as_measured": True}
    assert js["harness"]["flagged_values"] == [1, 2]
    # the harness's own reader takes it as a grading flag
    from insar_pipeline import validation
    layer, what, grades = validation.read_flag_layer(cfg, shape)
    assert grades and "F2" in what and np.nanmax(layer) == 2 and np.isnan(layer[0, 0])


def test_risk_between_follows_the_pairs_that_span_the_interval(tmp_path):
    cfg, pairs = make_project(tmp_path)
    A.pre_unwrap(cfg, pairs, route="miaplpy")
    shape, attrs = write_velocity(cfg)
    rc = product_rc(attrs, *crop_latlon(CONE_RC[0] + 10, CONE_RC[1]))
    early = A.risk_between(cfg, "2020-01-01", "2020-01-13")
    event = A.risk_between(cfg, "20200125", "20200206")
    assert early.shape == shape and early[rc] == 0 and event[rc] == 2
    assert A.risk_between(cfg, "2020-02-18", "2020-03-01")[product_rc(attrs, *crop_latlon(*PATCH_RC))] == 1


def test_review_risk_between_matches_the_exported_record_and_either_date_order(tmp_path):
    """Reviewer, 8 Oct 2026: risk_between read every pair file in aliasing/pairs/,
    so a pair left there by an earlier network (or a partial measurement)
    flagged an interval although quality_aliasing_risk.tif (the summary's
    pairs) did not; and a reversed (d1, d0) selected only pairs holding the
    whole interval."""
    cfg, pairs = make_project(tmp_path)
    A.pre_unwrap(cfg, pairs, route="miaplpy")
    shape, attrs = write_velocity(cfg)
    rc = product_rc(attrs, *crop_latlon(CONE_RC[0] + 10, CONE_RC[1]))
    assert A.risk_between(cfg, "20200206", "20200125")[rc] == 2                 # reversed: same interval
    assert (A.risk_between(cfg, "20200206", "20200125") == A.risk_between(cfg, "20200125", "20200206")).all()
    # a stray pair file (not in the summary) holding level 2 everywhere
    stray = A._pair_file(cfg, "20200101_20200301")
    lv = np.full((R1 - R0, C1 - C0), A.LEVEL_LIKELY, np.uint8)
    np.savez_compressed(stray, level=lv, key=np.array("{}"), stats=np.array(json.dumps({"pair": "20200101_20200301"})))
    A._LEVELS.clear()
    early = A.risk_between(cfg, "2020-01-01", "2020-01-13")
    assert early[rc] == 0
    r = A.export_products(cfg)
    import rasterio
    with rasterio.open(r["tif"]) as src:
        assert src.read(1)[rc] == 2                         # the tif: the event pairs, not the stray file
    lat, lon = crop_latlon(90, 15)
    assert A.risk_between(cfg, "2020-01-01", "2020-03-01")[product_rc(attrs, lat, lon)] == 0


def test_review_the_record_says_which_coherence_the_hole_test_read_and_where_the_gradient_was_judged(tmp_path):
    cfg, pairs = make_project(tmp_path)
    s = A.pre_unwrap(cfg, pairs, route="miaplpy")
    assert s["pair_coherence"].startswith("phasor coherence of the LINKED phase")
    fast = {r["pair"]: r for r in s["pairs"]}["20200125_20200206"]
    assert fast["gradient"]["rg"]["share_consistent_zones"] > 0.5
    (cfg.stack_dir / "merged" / "pair_coherence.json").write_text(json.dumps({"kind": "sample", "pairs": pairs}))
    assert A.pair_coherence_kind(cfg, "miaplpy").startswith("sample coherence")
    assert A.pair_coherence_kind(cfg, "stack").startswith("topsStack")
    write_velocity(cfg)
    A.export_products(cfg)
    js = json.loads((cfg.project_dir / "deliverables" / "aliasing_risk.json").read_text())
    assert js["method"]["pair_coherence"].startswith("phasor") and js["hole_rule_note"]["pair_coherence_read"]
    assert "not judged" in js["method"]["gradient_judged_where"]


def test_review_f1_states_the_mean_tilt_basis_and_the_disp_interval_caveat(tmp_path):
    rec = A.detectability(make_cfg(tmp_path / "s", backend="miaplpy", expected_max_rate_mm_yr=1000.0,
                                   expected_trough_halfwidth_m=450.0))
    lines = "\n".join(A.detectability_lines(rec))
    assert "MEAN tilt" in lines and "steepest flank" in lines and "DISP-S1" not in lines
    d = A.detectability(make_cfg(tmp_path / "d", backend="disp", expected_max_rate_mm_yr=1000.0,
                                 expected_trough_halfwidth_m=450.0))
    assert "mini-stack reference" in d["intervals_note"]
    assert "mini-stack reference" in "\n".join(A.detectability_lines(d))
    # the note shows without an expected rate too
    d0 = A.detectability(make_cfg(tmp_path / "d0", backend="disp"))
    assert "mini-stack reference" in "\n".join(A.detectability_lines(d0))


def test_review_a_change_of_fast_zones_at_mintpy_onward_remeasures_the_f2_record(tmp_path):
    """Reviewer, 8 Oct 2026: F2 drops the long pairs inside the fast zones; zones
    added, changed or removed before a 'MintPy onward' restart left the record
    describing other pairs than MintPy then read."""
    cfg, pairs = make_project(tmp_path)
    add_unwrapped(cfg, pairs)
    fz = write_geojson(tmp_path / "fast.geojson", lonlat_square(CONE_RC, 600.0))
    A.pre_unwrap(cfg, pairs, route="miaplpy")                    # measured without zones
    long_fast = "20200125_20200218"                               # rank 2, spans the event
    inside = A.polygon_mask(fz, *crop_latlon(*np.mgrid[0:R1 - R0, 0:C1 - C0]))
    lv0 = dict(zip(*A.load_levels(cfg)[:2]))[long_fast]
    assert (lv0[inside] == A.LEVEL_LIKELY).mean() > 0.5
    cfg.fast_zone_geojson = str(fz)                               # zones added, MintPy onward
    out = A.ensure_fast_zones(cfg)
    assert "error" not in out["f2_record"] and not out["f2_record"].get("unchanged")
    lv1 = dict(zip(*A.load_levels(cfg)[:2]))[long_fast]
    assert (lv1[inside] == 0).all()                               # not used there now: cannot flag there
    assert A.record_check(cfg, sorted(pairs))["fast_zones_as_measured"] is True
    assert A.ensure_fast_zones(cfg)["f2_record"] == {"unchanged": True}
    cfg.fast_zone_geojson = ""                                    # zones removed again
    A.ensure_fast_zones(cfg)
    lv2 = dict(zip(*A.load_levels(cfg)[:2]))[long_fast]
    assert (lv2[inside] == A.LEVEL_LIKELY).mean() > 0.5
    # where it cannot be measured the same way, the record is marked stale
    cfg.fast_zone_geojson = str(fz)
    (cfg.project_dir / "miaplpy" / "inverted" / "phase_series.h5").unlink()
    out = A.ensure_fast_zones(cfg)
    assert "stale" in out["f2_record"]
    assert json.loads((A.record_dir(cfg) / "summary.json").read_text())["fast_zones_stale"]
    assert A.record_check(cfg, sorted(pairs))["complete"] is False
    # a client config: nothing
    assert A.refresh_for_fast_zones(make_cfg(tmp_path / "c", aliasing_diagnostic="off"))["skipped"]


def test_old_flag_relabelled_only_where_it_is_not_and_stale_flag_set_aside(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg, pairs = make_project(tmp_path)
    A.pre_unwrap(cfg, pairs, route="miaplpy")
    shape, attrs = write_velocity(cfg)
    D = cfg.project_dir / "deliverables"
    D.mkdir()
    old = D / A.OLD_FLAG
    dl._write_geotiff(old, np.zeros(shape, np.uint8), attrs, nodata=255, dtype="uint8",
                      description="1 where the rate differs ...", units="flag")
    A.export_products(cfg)
    import rasterio
    with rasterio.open(old) as src:
        assert src.descriptions[0].startswith("LOWER BOUND") and "LOWER_BOUND" in src.tags()
    t = old.stat().st_mtime_ns
    time.sleep(0.01)
    r = A.export_products(cfg)                          # labelled now: left alone
    assert old.stat().st_mtime_ns == t and r["old_flag"]["already_labelled"]
    # the diagnostic switched off: this run writes no flag, the earlier one moves aside
    cfg.aliasing_diagnostic = "off"
    r = A.export_products(cfg)
    assert "skipped" in r and r.get("set_aside")
    assert not (D / "quality_aliasing_risk.tif").exists() and not (D / "aliasing_risk.json").exists()
    assert list((D / "superseded").glob("aliasing_*/quality_aliasing_risk.tif"))


def test_record_check_names_pairs_unwrapped_but_never_measured(tmp_path):
    cfg, pairs = make_project(tmp_path)
    A.pre_unwrap(cfg, pairs[:-1], route="miaplpy")
    for p in pairs:
        (cfg.stack_dir / "merged" / "interferograms" / p / "filt_fine.unw.conncomp").write_bytes(b"")
    chk = A.record_check(cfg, A.load_levels(cfg)[0])
    assert chk["unwrapped_not_measured"] == [pairs[-1]] and not chk["complete"]


# ======================================================================
# F1: detectability
# ======================================================================
def test_detectability_takes_the_wavelength_from_the_product(tmp_path):
    cfg = make_cfg(tmp_path / "p", backend="stack", expected_max_rate_mm_yr=3000.0, expected_trough_halfwidth_m=200.0)
    c = A.detectability(cfg, DATES, attrs={"WAVELENGTH": "0.05546576", "CENTER_INCIDENCE_ANGLE": "39"})
    l_ = A.detectability(cfg, DATES, attrs={"WAVELENGTH": "0.2384", "CENTER_INCIDENCE_ANGLE": "39"})
    assert c["quarter_wavelength_mm"] == pytest.approx(13.87, abs=0.01)
    assert l_["quarter_wavelength_mm"] == pytest.approx(59.6, abs=0.01)
    vc = c["zones"][0]["intervals"]["nearest pairs (median)"]
    vl = l_["zones"][0]["intervals"]["nearest pairs (median)"]
    assert vc["cell_to_cell_mm"] == vl["cell_to_cell_mm"]
    assert vc["verdict"] == "too fast: aliased" and vl["verdict"] == "measurable"
    assert c["zones"][0]["intervals"]["3-connection pairs (median)"]["days"] == 36
    # before any product: the sensor's nominal value, and it says so
    n = A.detectability(cfg, None)
    assert "nominal" in n["wavelength_source"] and n["intervals_source"].startswith("nominal")
    assert A.wavelength(cfg, None, allow_nominal=False)[0] is None
    nis = make_cfg(tmp_path / "n", backend="nisar")
    assert A.wavelength(nis)[0] == pytest.approx(0.2439, abs=1e-3)
    # a product's own wavelength wins over anything else
    assert A.wavelength(cfg, {"WAVELENGTH": 0.2384})[0] == 0.2384
    assert A.wavelength(cfg, {"WAVELENGTH": "5.0"})[1].startswith("nominal")      # not a radar wavelength


def test_detectability_from_the_zone_polygons(tmp_path):
    f1 = lonlat_square((50, 50), 0)
    lat, lon = crop_latlon(50, 50)
    hw, hl = 200.0 / KX, 1000.0 / KY                      # 400 m wide (east-west), 2000 m long
    f1["geometry"]["coordinates"] = [[[lon - hw, lat - hl], [lon + hw, lat - hl], [lon + hw, lat + hl],
                                      [lon - hw, lat + hl], [lon - hw, lat - hl]]]
    f2 = json.loads(json.dumps(f1))
    f2["properties"] = {"name": "LW902", "max_rate_mm_yr": 6000, "halfwidth_m": 150}
    gj = write_geojson(tmp_path / "z.geojson", f1, f2)
    cfg = make_cfg(tmp_path / "p", aliasing_zone_geojson=str(gj), expected_max_rate_mm_yr=1000.0)
    rec = A.detectability(cfg, DATES, attrs={"WAVELENGTH": "0.05546576"})
    z = {r["zone"]: r for r in rec["zones"]}
    assert z["panel"]["halfwidth_m"] == pytest.approx(200.0, rel=0.02)
    assert z["panel"]["halfwidth_from"] == "half the polygon's narrow side" and z["panel"]["rate_mm_yr"] == 1000.0
    assert z["LW902"]["rate_mm_yr"] == 6000 and z["LW902"]["halfwidth_m"] == 150
    assert z["LW902"]["support_change_mm_nearest"] is not None             # miaplpy: the window's support
    text = "\n".join(A.detectability_lines(rec))
    assert "LW902" in text and "quarter wavelength" in text and "Fastest followable" in text
    # a trough narrower than a look cell: the cap, not a faster figure
    cfg2 = make_cfg(tmp_path / "q", expected_max_rate_mm_yr=1000.0, expected_trough_halfwidth_m=10.0)
    r2 = A.detectability(cfg2, DATES, attrs={"WAVELENGTH": "0.05546576"})["zones"][0]
    r3 = A.detectability(make_cfg(tmp_path / "r", expected_max_rate_mm_yr=1000.0, expected_trough_halfwidth_m=40.0),
                         DATES, attrs={"WAVELENGTH": "0.05546576"})["zones"][0]
    assert r2["intervals"]["nearest pairs (median)"]["cell_to_cell_mm"] == pytest.approx(
        1000.0 * 12 / 365.25 * math.cos(math.radians(39)), abs=0.01)
    assert r2["fastest_followable_mm_yr_nearest"] <= r3["fastest_followable_mm_yr_nearest"] * 1.2


@pytest.fixture
def planner(monkeypatch):
    """plan_stack without the network (as tests/test_fix_L6.py)."""
    from insar_pipeline import data_acquisition as da
    from insar_pipeline import settings
    import datetime as dt
    d0 = dt.date(2020, 1, 1)
    monkeypatch.setattr(da, "search_scenes", lambda cfg: [SimpleNamespace(properties={
        "startTime": (d0 + dt.timedelta(days=12 * i)).isoformat() + "T23:47:41"}) for i in range(40)])
    monkeypatch.setattr(sb, "check_weather_credentials", lambda cfg: [])
    monkeypatch.setattr(settings, "disk_budget_gb", lambda p: {"budget": 400.0, "source": "test"}, raising=False)
    monkeypatch.setattr(settings, "machine_resources", lambda: {"mem_gb": 24, "swap_gb": 0, "cores": 24},
                        raising=False)
    monkeypatch.setattr(sb.shutil, "disk_usage", lambda p: SimpleNamespace(total=1e12, used=2e11, free=8e11))


def test_plan_shows_the_preflight_only_when_asked(tmp_path, planner):
    def cfg_(**over):
        c = make_cfg(tmp_path / "plan", bbox=[-87.55995, 38.24325, -87.33112, 38.42952], stack_swaths="2",
                     stack_use_burst2safe=True, stack_virtual_merge="False", **over)
        c.slc_dir.mkdir(parents=True, exist_ok=True)
        c.stack_dir.mkdir(parents=True, exist_ok=True)
        (c.slc_dir / "dropped_short_dates.json").write_text(json.dumps({"modal_burst_count": 3, "dropped": []}))
        return c
    client = sb.plan_stack(cfg_(aliasing_diagnostic="auto"))
    assert not any("Fast motion (F1)" in n for n in client.notes)
    exp = sb.plan_stack(cfg_(aliasing_diagnostic="auto", expected_max_rate_mm_yr=2000.0,
                             expected_trough_halfwidth_m=250.0))
    note = next(n for n in exp.notes if "Fast motion (F1)" in n)
    assert "nearest pairs (median) 12 d" in note and "mm/yr over a 250 m half-width" in note
    assert "Fast motion (F1)" in sb.format_plan(exp)


def test_plan_run_prints_the_preflight_for_routes_it_does_not_size(tmp_path, capsys):
    from insar_pipeline import plan_run
    plan_run._detectability(make_cfg(tmp_path / "n", backend="nisar", expected_max_rate_mm_yr=2000.0,
                                     expected_trough_halfwidth_m=250.0))
    out = capsys.readouterr().out
    assert "Fast motion (F1)" in out and "nominal NISAR L-band" in out
    plan_run._detectability(make_cfg(tmp_path / "c", backend="nisar", aliasing_diagnostic="auto"))
    assert capsys.readouterr().out == ""


# ======================================================================
# F3: fast zones
# ======================================================================
def add_unwrapped(cfg, pairs):
    for i, p in enumerate(sorted(pairs)):
        pdir = cfg.stack_dir / "merged" / "interferograms" / p
        unw = np.zeros((LM, 2, WM), np.float32)
        unw[:, 0, :] = 1.0
        unw[:, 1, :] = 0.5 + i
        unw.tofile(pdir / "filt_fine.unw")
        np.ones((LM, WM), np.uint8).tofile(pdir / "filt_fine.unw.conncomp")


def read_unw(cfg, p):
    pdir = cfg.stack_dir / "merged" / "interferograms" / p
    return (np.fromfile(pdir / "filt_fine.unw", np.float32).reshape(LM, 2, WM)[:, 1, :],
            np.fromfile(pdir / "filt_fine.unw.conncomp", np.uint8).reshape(LM, WM))


def test_fast_zones_keep_only_nearest_pairs_inside_and_put_originals_back(tmp_path):
    cfg, pairs = make_project(tmp_path)
    add_unwrapped(cfg, pairs)
    before = {p: [a.copy() for a in read_unw(cfg, p)] for p in pairs}
    fz = write_geojson(tmp_path / "fast.geojson", lonlat_square(CONE_RC, 400.0))
    cfg.fast_zone_geojson = str(fz)
    _, rank = A.pair_dates(pairs)
    rec = A.apply_fast_zones(cfg, pairs)
    inside = np.zeros((LM, WM), bool)
    inside[R0:R1, C0:C1] = A.polygon_mask(fz, *crop_latlon(*np.mgrid[0:R1 - R0, 0:C1 - C0]))
    assert 300 < inside.sum() < 500 and rec["zone_cells"] == inside.sum()
    for p in pairs:
        ph, cc = read_unw(cfg, p)
        if rank[p] > 1:
            assert (ph[inside] == 0).all() and (cc[inside] == 0).all(), p
            assert (ph[~inside] == before[p][0][~inside]).all()
            assert (cfg.stack_dir / "merged" / "interferograms" / p / A.FAST_SIDECAR).exists()
        else:
            assert (ph == before[p][0]).all() and (cc == before[p][1]).all(), p
    assert rec["outcome"] == {"applied": 4, "none": 5}
    assert A.apply_fast_zones(cfg, pairs)["outcome"] == {"kept": 4, "none": 5}
    # two connections inside the zones now: everything back
    cfg.fast_zone_connections = 2
    assert A.apply_fast_zones(cfg, pairs)["outcome"] == {"restored": 4, "none": 5}
    for p in pairs:
        assert all((x == y).all() for x, y in zip(read_unw(cfg, p), before[p]))
    # applied again, then the setting removed: the MintPy loader hook restores it
    cfg.fast_zone_connections = 1
    A.apply_fast_zones(cfg, pairs)
    cfg.fast_zone_geojson = ""
    lines = sb.mintpy_loader_lines(cfg)
    assert "filt_fine.unw" in lines
    for p in pairs:
        assert all((x == y).all() for x, y in zip(read_unw(cfg, p), before[p]))
        assert not (cfg.stack_dir / "merged" / "interferograms" / p / A.FAST_SIDECAR).exists()
    assert json.loads((A.record_dir(cfg) / "fast_zones.json").read_text())["fast_zone_geojson"] is None


def test_fast_zones_never_restore_over_a_new_unwrap(tmp_path):
    cfg, pairs = make_project(tmp_path)
    add_unwrapped(cfg, pairs)
    cfg.fast_zone_geojson = str(write_geojson(tmp_path / "fast.geojson", lonlat_square(CONE_RC, 400.0)))
    A.apply_fast_zones(cfg, pairs)
    p = "20200101_20200125"                              # a rank-2 pair, now re-unwrapped
    pdir = cfg.stack_dir / "merged" / "interferograms" / p
    fresh = np.full((LM, 2, WM), 7.0, np.float32)
    time.sleep(0.01)
    fresh.tofile(pdir / "filt_fine.unw")
    cfg.fast_zone_geojson = ""
    out = A.ensure_fast_zones(cfg)
    assert out["outcome"].get("dropped (files rewritten since)") == 1
    assert (read_unw(cfg, p)[0] == 7.0).all()


def test_mintpy_loader_hook_applies_zones_on_a_mintpy_onward_restart(tmp_path):
    cfg, pairs = make_project(tmp_path)
    add_unwrapped(cfg, pairs)
    cfg.fast_zone_geojson = str(write_geojson(tmp_path / "fast.geojson", lonlat_square(CONE_RC, 400.0)))
    sb.mintpy_loader_lines(cfg)
    assert sum((cfg.stack_dir / "merged" / "interferograms" / p / A.FAST_SIDECAR).exists() for p in pairs) == 4
    # and a project without zones is not touched (no record written)
    cfg2, pairs2 = make_project(tmp_path, name="plain")
    add_unwrapped(cfg2, pairs2)
    assert A.ensure_fast_zones(cfg2) == {"applied": False}
    sb.mintpy_loader_lines(cfg2)
    assert not (A.record_dir(cfg2) / "fast_zones.json").exists()


# ======================================================================
# the hooks
# ======================================================================
def test_unwrap_exported_runs_the_diagnostic_before_snaphu_and_the_zones_after(tmp_path, monkeypatch):
    cfg, pairs = make_project(tmp_path)
    calls = []
    monkeypatch.setattr(A, "pre_unwrap", lambda c, p, route=None: calls.append(("pre_unwrap", route, len(p))))
    monkeypatch.setattr(A, "apply_fast_zones", lambda c, p: calls.append(("fast_zones", len(p))))
    monkeypatch.setattr(sb, "execute_run_file", lambda c, rf, fail_fast=False, after=None:
                        calls.append(("snaphu", rf.name)) or {})
    monkeypatch.setattr(mb, "_unwrap_marker_ok", lambda c, cmd, p: True)
    mb.unwrap_exported(cfg, pairs)
    assert calls == [("pre_unwrap", "miaplpy", len(pairs)), ("snaphu", "run_16_unwrap"), ("fast_zones", len(pairs))]


def test_classic_route_hook_reads_the_pairs_of_the_unwrap_run_file(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path / "p", backend="stack")
    rf = tmp_path / "run_16_unwrap"
    rf.write_text("python -m insar_pipeline.unwrap_crop --config /x/configs/config_igram_unw_20200101_20200113 "
                  "--window /x/crop_window.json\n"
                  "python -m insar_pipeline.unwrap_crop --config /x/configs/config_igram_unw_20200113_20200125 "
                  "--window /x/crop_window.json\n")
    seen = []
    monkeypatch.setattr(A, "pre_unwrap", lambda c, p, route=None: seen.append(("pre", route, list(p))))
    monkeypatch.setattr(A, "apply_fast_zones", lambda c, p: seen.append(("fast", list(p))))
    pairs = sb._aliasing_hook(cfg, rf, "before")
    sb._aliasing_hook(cfg, rf, "after", pairs)
    assert seen == [("pre", "stack", ["20200101_20200113", "20200113_20200125"]),
                    ("fast", ["20200101_20200113", "20200113_20200125"])]


def test_settings_load_and_bad_values_are_refused(tmp_path):
    from insar_pipeline.config import validate_config
    cfg = make_cfg(tmp_path / "p", aliasing_diagnostic="sometimes", expected_max_rate_mm_yr=-1,
                   fast_zone_connections=0)
    probs = A.check_settings(cfg)
    assert any("aliasing_diagnostic" in p for p in probs)
    assert any("expected_max_rate_mm_yr" in p for p in probs)
    assert any("fast_zone_connections" in p for p in probs)
    assert any("aliasing_diagnostic" in p for p in validate_config(cfg))
    assert A.check_settings(make_cfg(tmp_path / "p")) == []
    hy = make_cfg(tmp_path / "p", backend="hyp3", fast_zone_geojson="/z.geojson")
    assert any("topsStack routes" in p for p in A.check_settings(hy))
    cl = make_cfg(tmp_path / "p", fast_zone_geojson="/z.geojson", mintpy_closure_correction=True)
    assert any("closure_correction" in p for p in A.check_settings(cl))
    pc = make_cfg(tmp_path / "p", fast_zone_geojson="/z.geojson", mintpy_unwrap_error_method="phase_closure")
    assert any("phase-closure" in p for p in A.check_settings(pc))
    rel = make_cfg(tmp_path / "p", miaplpy_relink_from="exp_a1")
    assert any("relative path" in p for p in A.check_settings(rel))
    same = make_cfg(tmp_path / "p", miaplpy_relink_from=str(tmp_path / "p"))
    assert any("names this project itself" in p for p in A.check_settings(same))
    st = make_cfg(tmp_path / "p", backend="stack", miaplpy_relink_from=str(tmp_path / "src"))
    assert any("backend must be miaplpy" in p for p in A.check_settings(st))
    # defaults: off for client projects, nothing set
    d = PipelineConfig(project_dir=tmp_path / "d", bbox=[0, 0, 1, 1], start_date="2020-01-01", end_date="2020-12-31")
    assert (d.aliasing_diagnostic, d.expected_max_rate_mm_yr, d.fast_zone_geojson, d.miaplpy_relink_from) == \
        ("auto", 0.0, "", "")
    assert not A.enabled(d)


# ======================================================================
# F3 / A1b: re-linking another project's stack
# ======================================================================
def make_source(tmp_path, looks=(RL, AL), n=4):
    """The parts of a finished phase-linked project a re-link reads."""
    S = tmp_path / "exp_a1"
    st = S / "stack"
    G = st / "merged" / "geom_reference"
    G.mkdir(parents=True)
    w, l_ = 60, 40
    for name, (W, L) in (("lat.rdr.full.xml", (w * looks[0], l_ * looks[1])), ("lat.rdr.xml", (w, l_))):
        (G / name).write_text(f'<imageFile><property name="width"><value>{W}</value></property>'
                              f'<property name="length"><value>{L}</value></property></imageFile>')
    (G / "lat.rdr").write_bytes(b"\0" * 8)
    (st / "crop_window.json").write_text(json.dumps({"r0": 2, "r1": 38, "c0": 3, "c1": 57,
                                                     "length": l_, "width": w}))
    (st / "generation.json").write_text(json.dumps({"input_dates": DATES[:n]}))
    (st / "reference").mkdir()
    (st / "reference" / "IW1.xml").write_text('<property name="radarwavelength"><value>0.05546576</value></property>')
    (st / "baselines").mkdir()
    for d in DATES[:n]:
        (st / "merged" / "SLC" / d).mkdir(parents=True)
        (st / "merged" / "SLC" / d / f"{d}.slc.full.xml").write_text(
            f'<imageFile><property name="width"><value>{w * looks[0]}</value></property>'
            f'<property name="length"><value>{l_ * looks[1]}</value></property></imageFile>')
    (st / "configs").mkdir()
    status = st / ".status" / "run_07_merge_reference_secondary_slc"
    status.mkdir(parents=True)
    for d in DATES[:n]:
        (st / "configs" / f"config_merge_{d}").write_text("x")
        (status / f"{d}.done").write_text(json.dumps({"cmd": f"SentinelWrapper.py -c config_merge_{d}",
                                                      "returncode": 0}))
    inp = S / "miaplpy" / "inputs"
    inp.mkdir(parents=True)
    rng = np.random.default_rng(3)
    with h5py.File(inp / "slcStack.h5", "w") as f:
        f.create_dataset("slc", data=(rng.normal(size=(n, 40, 50)) + 1j * rng.normal(size=(n, 40, 50))
                                      ).astype(np.complex64))
        f.create_dataset("date", data=np.array([d.encode() for d in DATES[:n]]))
    with h5py.File(inp / "geometryRadar.h5", "w") as f:
        f.create_dataset("height", data=np.zeros((40, 50), np.float32))
    return S


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()}


def test_relink_links_the_source_and_writes_nothing_there(tmp_path):
    S = make_source(tmp_path)
    before = snapshot(S)
    cfg = make_cfg(tmp_path / "exp_a1b", miaplpy_relink_from=str(S), miaplpy_range_window=9,
                   miaplpy_azimuth_window=5)
    assert mb.relink_problems(cfg) == []
    rec = mb.prepare_relink(cfg)
    P = cfg.project_dir
    assert (P / "stack" / "merged" / "geom_reference").is_symlink()
    assert (P / "miaplpy" / "inputs" / "slcStack.h5").is_symlink()
    assert (P / "stack" / "configs" / "config_merge_20200101").is_symlink()
    assert (P / "stack" / "crop_window.json").is_file() and not (P / "stack" / "crop_window.json").is_symlink()
    assert rec["windows"] == [9, 5] and rec["source_looks"] == [RL, AL] and rec["source_slc_stack"]["probe"]
    assert mb._slc_stack_verified(cfg)                            # load_data is not re-run
    assert mb.aligned_subset(cfg) == (2 * AL, 38 * AL, 3 * RL, 57 * RL)
    from insar_pipeline import runs
    cs = runs.coregistration_status(cfg.stack_dir)
    assert cs["missing"] == [] and len(cs["expected"]) == 4
    assert A.wavelength(cfg)[0] == pytest.approx(0.05546576)    # read through the link
    again = mb.prepare_relink(cfg)
    assert again["linked_or_copied_now"] == [] and again["links_already_there"] > 0
    assert snapshot(S) == before
    # MiaplPy opens the stack in append mode while it links: the mtime moves, the data do not
    with h5py.File(S / "miaplpy" / "inputs" / "slcStack.h5", "a") as f:
        f["slc"][0, 0, 0]
    chk = mb.check_relink_source_unchanged(cfg)
    assert chk["unchanged"] and chk["probe_compared"]
    with h5py.File(S / "miaplpy" / "inputs" / "slcStack.h5", "a") as f:
        f["slc"][0, 20, 25] = 99 + 0j                            # someone wrote data
    assert not mb.check_relink_source_unchanged(cfg)["unchanged"]


def test_relink_refusals(tmp_path, monkeypatch):
    S = make_source(tmp_path)
    P = tmp_path / "exp_a1b"
    cfg = make_cfg(P, miaplpy_relink_from=str(S), stack_range_looks=7)
    assert any("looks" in p for p in mb.relink_problems(cfg))
    cfg = make_cfg(P, miaplpy_relink_from=str(tmp_path / "nothere"))
    assert mb.relink_problems(cfg) == [f"miaplpy_relink_from {tmp_path / 'nothere'} is not a folder"]
    cfg = make_cfg(P, miaplpy_relink_from=str(S))
    (S / "stack" / "crop_window.json").rename(S / "stack" / "cw.json")
    assert any("crop_window.json" in p for p in mb.relink_problems(cfg))
    (S / "stack" / "cw.json").rename(S / "stack" / "crop_window.json")
    from insar_pipeline import runs
    monkeypatch.setattr(runs, "lock_holder", lambda d: {"pid": 4242, "kind": "full"})
    assert any("running now" in p for p in mb.relink_problems(cfg))
    monkeypatch.undo()
    # a real file of this project where a link belongs: refused, nothing replaced
    own = P / "stack" / "merged" / "geom_reference"
    own.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="real file or folder"):
        mb.prepare_relink(cfg)
    assert own.is_dir() and not own.is_symlink()
    # this project's own SLC stack: a project loads its own or re-links one
    own.rmdir()
    (P / "miaplpy" / "inputs").mkdir(parents=True)
    (P / "miaplpy" / "inputs" / "slcStack.h5").write_bytes(b"x")
    assert any("own SLC stack" in p for p in mb.relink_problems(cfg))
    # a full start downloads and coregisters nothing
    (P / "miaplpy" / "inputs" / "slcStack.h5").unlink()
    with pytest.raises(SystemExit, match="Phase linking onward"):
        sb.stage_inputs(cfg)


def test_relink_plan_and_the_slc_stack_deletion_that_must_not_follow_the_link(tmp_path):
    S = make_source(tmp_path)
    cfg = make_cfg(tmp_path / "exp_a1b", miaplpy_relink_from=str(S), expected_max_rate_mm_yr=2000.0,
                   expected_trough_halfwidth_m=250.0, miaplpy_delete_slc_stack_after_concatenate=True)
    plan = sb.plan_stack(cfg)                                  # no scene search, no download
    assert plan.n_dates == 4 and plan.dates == DATES[:4]
    assert any("Re-linking exp_a1's stack writes" in n for n in plan.notes)
    assert any("Fast motion (F1)" in n and "radarwavelength" in n for n in plan.notes)
    mb.prepare_relink(cfg)
    link = cfg.project_dir / "miaplpy" / "inputs" / "slcStack.h5"
    assert mb.delete_slc_stack_after_concatenate(cfg) == 0.0
    assert link.is_symlink() and (S / "miaplpy" / "inputs" / "slcStack.h5").exists()
