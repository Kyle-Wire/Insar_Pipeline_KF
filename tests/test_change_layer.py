"""Local change layer (Route B, 22 Oct 2026; review fixes the same day):
partners, filtering, steps without unwrapping, the sample-count gate, the
vendor's while-observed bookkeeping, the t-test, pairs formed from merged SLCs
and the DEM-error term - on synthetic data with known answers, at made-up
coordinates. (Named test_change_layer, not test_local_*: that pattern is
git-ignored for tests that read client data.)"""
from pathlib import Path

import numpy as np
from scipy import stats

from insar_pipeline import local_change as lc


def _probe(W, y, x, R, boundary_col):
    """(partners across the boundary, partners on the probe's side) at interior pixel (y, x)."""
    across = side = 0
    for o, (dy, dx) in enumerate(lc.offsets(R)):
        if (dy, dx) == (0, 0):
            continue
        if x + R + dx >= boundary_col:
            across += int(W[o, y, x])
        else:
            side += int(W[o, y, x])
    return across, side


def test_partners_need_both_a_similar_mean_and_a_similar_history():
    rng = np.random.default_rng(0)
    D, h, w, R = 48, 21, 21, 3
    t = np.arange(D)
    y, x = 7, 5                                               # interior col 8; window cols 5..11 crosses col 10
    # correlation case: same mean, opposite seasonal history across col 10
    A = np.empty((D, h, w))
    A[:, :, :10] = (1 + 0.3 * np.sin(t / 3))[:, None, None]
    A[:, :, 10:] = (1 - 0.3 * np.sin(t / 3))[:, None, None]
    A *= np.exp(0.03 * rng.standard_normal(A.shape))
    across, side = _probe(lc.amplitude_partners(A, R, corr=0.35, dmu=0.15), y, x, R, 10)
    assert across == 0 and side > 5
    # mean case: same history, three times brighter across col 10
    B = np.empty((D, h, w))
    B[:, :, :10] = (1 + 0.3 * np.sin(t / 3))[:, None, None]
    B[:, :, 10:] = 3 * (1 + 0.3 * np.sin(t / 3))[:, None, None]
    B *= np.exp(0.03 * rng.standard_normal(B.shape))
    across, side = _probe(lc.amplitude_partners(B, R, corr=0.35, dmu=0.15), y, x, R, 10)
    assert across == 0 and side > 5


def test_filter_coherence_and_direction():
    rng = np.random.default_rng(1)
    R, h, w = 2, 12, 12
    sh = (h + 2 * R, w + 2 * R)
    s1 = rng.standard_normal(sh) + 1j * rng.standard_normal(sh)
    n2 = rng.standard_normal(sh) + 1j * rng.standard_normal(sh)
    s2 = (s1 + 0.8 * n2) * np.exp(-1j * 0.7)                  # partly coherent pair
    z = (s1 * np.conj(s2)).astype(np.complex64)
    p1, p2 = np.abs(s1) ** 2, np.abs(s2) ** 2
    W = np.ones((len(lc.offsets(R)), h, w), np.uint8)
    f, c = lc.partner_filter(z, p1, p2, W, R)
    win = (slice(3, 3 + 2 * R + 1), slice(4, 4 + 2 * R + 1))   # output pixel (3, 4) covers input rows 3..7, cols 4..8
    expect = abs(z[win].sum()) / np.sqrt(p1[win].sum() * p2[win].sum())
    assert np.isclose(c[3, 4], expect, rtol=1e-4) and 0.3 < c[3, 4] < 0.95
    # direction: only the centre and the right-hand neighbour are partners
    Wd = np.zeros_like(W); Wd[lc.offsets(R).index((0, 0))] = 1; Wd[lc.offsets(R).index((0, 1))] = 1
    fd, _ = lc.partner_filter(z, p1, p2, Wd, R)
    assert np.isclose(fd[3, 4], (z[3 + R, 4 + R] + z[3 + R, 5 + R]) / 2, rtol=1e-5)


def test_step_is_read_against_the_surroundings_and_weather_is_removed():
    h, w = 256, 256
    yy, xx = np.mgrid[0:h, 0:w]
    ramp = 0.02 * xx + 0.01 * yy                              # orbit/atmosphere-like ramp (radians)
    local = np.zeros((h, w)); local[120:136, 120:136] = 0.8   # a small patch moving 0.8 rad
    z = np.exp(1j * (ramp + local)).astype(np.complex64)
    coh = np.ones((h, w), np.float32)
    s = lc.step_mm(z, coh, 30, 30)
    patch = np.median(s[124:132, 124:132]); far = np.median(s[70:90, 70:90])   # away from edges
    assert abs(patch - far - 0.8 * lc.RAD2MM) < 0.15 * abs(0.8 * lc.RAD2MM)
    assert abs(far) < 0.5
    wmm = 3.0 * np.ones((h, w)); wmm[:, 128:] = -3.0
    zw = np.exp(1j * (wmm / lc.RAD2MM)).astype(np.complex64)
    assert np.abs(lc.step_mm(zw, coh, 30, 30, weather_mm=wmm)).max() < 1e-3


def test_gate_rejects_noise_at_pixels_with_few_partners():
    rng = np.random.default_rng(2)
    rlks, trials = 4, 4000
    for k in (0, 3, 8, 40):
        n = rlks * (k + 1)
        a = rng.standard_normal((trials, n)) + 1j * rng.standard_normal((trials, n))
        b = rng.standard_normal((trials, n)) + 1j * rng.standard_normal((trials, n))
        coh = np.abs((a * np.conj(b)).sum(1)) / np.sqrt((np.abs(a) ** 2).sum(1) * (np.abs(b) ** 2).sum(1))
        acc = lc.step_gate(coh, np.full(trials, k), thr=0.3, rlks=rlks, min_partners=0, eff_ratio=1.0)
        assert acc.mean() < 0.08, (k, acc.mean())             # about alpha (0.05), not 50-75%
    assert not lc.step_gate(np.array([0.99]), np.array([2]), 0.3, 4, min_partners=4)[0]


def test_rejected_steps_add_nothing_and_rate_is_per_counted_time():
    steps = np.array([[-1.0], [-5.0], [-1.0], [-1.0]])
    acc = np.array([[True], [False], [True], [True]])
    g = lc.gated_rate(steps, acc, np.array([12, 12, 12, 12]))
    assert g["displacement_mm"][0] == -3.0 and g["n_steps"][0] == 3
    assert np.isclose(g["years"][0], 36 / 365.25)
    assert np.isclose(g["rate_mm_yr"][0], -3.0 / (36 / 365.25)) and np.isclose(g["accepted_share"][0], 0.75)


def test_t_test_holds_its_false_positive_rate_at_any_step_count():
    rng = np.random.default_rng(3)
    for n in (2, 4, 12, 60):
        steps = rng.normal(0.0, 2.0, (n, 20000))
        acc = np.ones_like(steps, bool)
        days = np.full(n, 12.0)
        g = lc.gated_rate(steps, acc, days)
        t = g["rate_mm_yr"] / lc.rate_noise(steps, acc, days)
        sig = lc.significant(t, g["n_steps"], min_snr=3.0, min_steps=2)
        assert sig.mean() < 3 * 2 * stats.norm.sf(3.0) + 0.002, (n, sig.mean())   # near 0.27%, not 20%
    assert not lc.significant(np.array([50.0]), np.array([5]), min_steps=10)[0]


def test_consecutive_pairs_only(tmp_path):
    for p in ("20240101_20240113", "20240113_20240125", "20240101_20240125", "20240125_20240206"):
        (tmp_path / p).mkdir(); (tmp_path / p / "fine.int").write_bytes(b"")
    assert lc.consecutive_pairs(tmp_path) == ["20240101_20240113", "20240113_20240125", "20240125_20240206"]
    assert lc.consecutive_pairs(tmp_path, "2024-01-13", "2024-01-25") == ["20240113_20240125"]


def test_sample_refuses_points_outside_the_window(tmp_path):
    import h5py
    from types import SimpleNamespace
    la, lo = np.mgrid[45.10:45.11:0.0001, -100.05:-100.04:0.0001]
    with h5py.File(tmp_path / "steps.h5", "w") as h:
        h["step_mm"] = np.full((3,) + la.shape, -1.0, np.float32)
        h["coherence"] = np.full((3,) + la.shape, 0.9, np.float16)
        h["latitude"] = la.astype(np.float32); h["longitude"] = lo.astype(np.float32)
        h["partners"] = np.full(la.shape, 60, np.uint8)
        h["pair"] = np.array(["20240101_20240113", "20240113_20240125", "20240125_20240206"], dtype="S17")
        h["days"] = np.array([12.0, 12.0, 12.0])
        h.attrs.update({"complete": True, "rlks": 4})
    cfg = SimpleNamespace(local_change_threshold=0.3, local_change_min_partners=4)
    g = lc.sample(cfg, tmp_path / "steps.h5", [45.105, 45.20], [-100.045, -100.045])
    assert g["near"].tolist() == [True, False] and np.isfinite(g["rate_mm_yr"][0]) and np.isnan(g["rate_mm_yr"][1])
    g2 = lc.sample(cfg, tmp_path / "steps.h5", [45.105], [-100.045], start="20240113", end="20240206")
    assert g2["pairs"] == ["20240113_20240125", "20240125_20240206"]


# ----------------------------------------------------------------- pairs from merged SLCs (22 Oct 2026)
def _vrt(path, h, w, dt):
    Path(str(path) + ".vrt").write_text(f'<VRTDataset rasterXSize="{w}" rasterYSize="{h}">'
                                        f'<VRTRasterBand dataType="{dt}" band="1"/></VRTDataset>')


def _project(root, slcs, dates, rlks, with_fine_int):
    from types import SimpleNamespace
    M = root / "stack" / "merged"
    h, W = slcs[0].shape
    for d, s in zip(dates, slcs):
        (M / "SLC" / d).mkdir(parents=True)
        f = M / "SLC" / d / f"{d}.slc.full"
        s.astype(np.complex64).tofile(f); _vrt(f, h, W, "CFloat32")
    (M / "geom_reference").mkdir(parents=True)
    for name, base in (("lat", 45.1), ("lon", -100.05)):
        f = M / "geom_reference" / f"{name}.rdr"
        a = base + 1e-4 * (np.mgrid[0:h, 0:W // rlks][0 if name == "lat" else 1])
        a.astype(np.float64).tofile(f); _vrt(f, h, W // rlks, "Float64")
    if with_fine_int:
        for i in range(len(dates) - 1):
            d = M / "interferograms" / f"{dates[i]}_{dates[i + 1]}"
            d.mkdir(parents=True)
            z = lc.looks_mean(slcs[i] * np.conj(slcs[i + 1]), rlks).astype(np.complex64)   # what topsStack writes
            z.tofile(d / "fine.int"); _vrt(d / "fine.int", h, W // rlks, "CFloat32")
    return SimpleNamespace(project_dir=root, stack_dir=root / "stack", start_date="2024-01-01", end_date="2024-12-31",
                           stack_range_looks=rlks, stack_azimuth_looks=1, local_change_window=2,
                           local_change_partner_corr=0.35, local_change_partner_dmu=0.15, local_change_sigma_km=4.0)


def test_pairs_formed_from_merged_slcs_equal_the_stack_routes(tmp_path, monkeypatch):
    """A project merged the phase-linked way has no fine.int; the layer forms the
    pairs from the merged SLCs and must get exactly what the stack route's
    fine.int (burst a x conj(b), merged with a range MEAN) gives."""
    from insar_pipeline import unwrap_crop
    rng = np.random.default_rng(5)
    rlks, h, W = 4, 24, 4 * 20
    dates = ["20240101", "20240113", "20240125", "20240206"]
    base = rng.standard_normal((h, W)) + 1j * rng.standard_normal((h, W))
    slcs = [(base + 0.6 * (rng.standard_normal((h, W)) + 1j * rng.standard_normal((h, W)))) * np.exp(1j * 0.3 * k)
            for k in range(len(dates))]
    monkeypatch.setattr(unwrap_crop, "compute_crop_window", lambda cfg, write=False: {"r0": 0, "r1": h, "c0": 0, "c1": W // rlks})
    a = _project(tmp_path / "stack_route", slcs, dates, rlks, with_fine_int=True)
    b = _project(tmp_path / "merge_route", slcs, dates, rlks, with_fine_int=False)
    ra = lc.run(a, steps=("partners", "filter"), workers=1)
    rb = lc.run(b, steps=("partners", "filter"), workers=1)
    assert ra["filter"]["source"] == "fine_int" and rb["filter"]["source"] == "slc"
    assert ra["pairs"] == rb["pairs"] == ["20240101_20240113", "20240113_20240125", "20240125_20240206"]
    for p in ra["pairs"]:
        for ext, dt in ((".c8", np.complex64), (".coh.f4", np.float32)):
            x = np.fromfile(tmp_path / "stack_route" / "local_change" / "pairs" / f"{p}{ext}", dt)
            y = np.fromfile(tmp_path / "merge_route" / "local_change" / "pairs" / f"{p}{ext}", dt)
            assert x.size and np.allclose(x, y, rtol=1e-5, atol=1e-6), (p, ext)


def test_pairs_from_slcs_respect_the_date_range():
    assert lc.date_pairs(["20240101", "20240113", "20240125", "20240206"], "2024-01-13", "2024-02-06") == \
        ["20240113_20240125", "20240125_20240206"]
    assert lc.date_pairs(["20240101"]) == []


# ----------------------------------------------------------------- DEM-error term (22 Oct 2026)
def _geo(path, name, value, attrs):
    import h5py
    with h5py.File(path, "w") as f:
        f[name] = np.full((20, 20), value, np.float32)
        f.attrs.update(attrs)


def test_dem_error_residual_has_the_size_and_sign_mintpy_fits(tmp_path):
    import h5py
    attrs = {"Y_FIRST": "45.11", "X_FIRST": "-100.05", "Y_STEP": "-0.0005", "X_STEP": "0.0005"}
    _geo(tmp_path / "geo_demErr.h5", "dem", 25.0, attrs)
    with h5py.File(tmp_path / "geo_geometryRadar.h5", "w") as f:
        f["slantRangeDistance"] = np.full((20, 20), 850e3, np.float32)
        f["incidenceAngle"] = np.full((20, 20), 35.0, np.float32)
        f.attrs.update(attrs)
    LAT = np.array([[45.105, 45.0]]); LON = np.array([[-100.045, -100.045]])      # second point off the grid
    k = lc.topo_factor(tmp_path / "geo_demErr.h5", LAT, LON)
    assert np.isclose(k[0, 0], 25.0 / (850e3 * np.sin(np.radians(35.0))), rtol=1e-5) and k[0, 1] == 0.0
    mm = lc.topo_step_mm(k, 10.0, 96.5)                       # 86.5 m of baseline change, as at Site D
    assert np.isclose(mm[0, 0], 1000 * 86.5 * k[0, 0]) and 4.0 < mm[0, 0] < 5.0
    # a patch with that residual reads as movement unless the term is removed
    h = w = 128
    topo = np.zeros((h, w)); topo[56:72, 56:72] = mm[0, 0]
    z = np.exp(1j * (topo / lc.RAD2MM)).astype(np.complex64)
    coh = np.ones((h, w), np.float32)
    assert abs(np.median(lc.step_mm(z, coh, 30, 30)[60:68, 60:68]) - mm[0, 0]) < 0.5
    assert np.abs(lc.step_mm(z, coh, 30, 30, weather_mm=topo)).max() < 1e-3


def test_a_set_dem_error_path_must_exist(tmp_path):
    import pytest
    from types import SimpleNamespace
    cfg = SimpleNamespace(local_change_dem_error=str(tmp_path / "nope.h5"), stack_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        lc.dem_error_source(cfg)
    assert lc.dem_error_source(SimpleNamespace(local_change_dem_error="", stack_dir=tmp_path)) is None


def test_variant_tags_stay_inside_the_layer_folder(tmp_path):
    import pytest
    from types import SimpleNamespace
    cfg = SimpleNamespace(project_dir=tmp_path)
    assert lc.variant_dir(cfg, "sigma8km") == tmp_path / "local_change" / "variants" / "sigma8km"
    with pytest.raises(ValueError):
        lc.variant_dir(cfg, "../x")


# ----------------------------------------------------------------- per-period partners and persistent scatterers (22 Oct 2026)
def test_partner_periods_cover_the_record_without_a_runt_period():
    dates = [f"{d:%Y%m%d}" for d in np.arange(np.datetime64("2016-01-05"), np.datetime64("2026-06-20"), 12)
             .astype("datetime64[D]").tolist()]
    per = lc.partner_periods(dates, 730)
    assert [d for p in per for d in p] == dates                       # every date, in order, once
    assert len(per) == 5 and all(len(p) >= 45 for p in per)            # the last ~6 months joined the period before
    assert lc.partner_periods(dates, 0) == [dates]
    pp = lc.pair_periods([f"{a}_{b}" for a, b in zip(dates[:-1], dates[1:])], per)
    assert pp[0] == 0 and pp[-1] == len(per) - 1 and all(b - a in (0, 1) for a, b in zip(pp[:-1], pp[1:]))


def test_amplitude_dispersion_separates_a_stable_scatterer_from_speckle():
    rng = np.random.default_rng(7)
    R, D = 1, 60
    power = rng.exponential(1.0, (D, 5, 5, 4)).mean(3)                 # 4-look speckle, a new draw every date
    A = np.sqrt(power)
    A[:, 2, 2] = 10 * (1 + 0.03 * rng.standard_normal(D))               # one bright, stable scatterer
    da = lc.amplitude_dispersion(A, R).astype(float)
    assert da[1, 1] < 0.05 and 0.2 < np.median(np.delete(da.ravel(), 4)) < 0.3


def test_a_persistent_scatterer_counts_without_partners_but_not_without_coherence():
    coh = np.array([0.97, 0.97, 0.60])
    n = np.array([0, 0, 0])
    assert lc.step_gate(coh, n, 0.3, 4, min_partners=4).tolist() == [False, False, False]
    ps = np.array([True, False, True])
    assert lc.step_gate(coh, n, 0.3, 4, min_partners=4, ps=ps).tolist() == [True, False, False]   # floor 0.92 still holds


def test_per_period_partners_follow_ground_that_changes(tmp_path, monkeypatch):
    """Two field halves whose brightness histories match in the first year and are
    opposite in the second (a crop rotation): chosen over the whole record they are
    not partners anywhere near the join; chosen per period they are, in period 1."""
    from insar_pipeline import unwrap_crop
    rng = np.random.default_rng(3)
    rlks, h, W, n = 4, 20, 4 * 16, 60
    dates = [f"{d:%Y%m%d}" for d in np.arange(np.datetime64("2020-01-05"), np.datetime64("2020-01-05") + 12 * n, 12)
             .astype("datetime64[D]").tolist()]
    t = np.arange(n)
    slcs = []
    for k in range(n):
        left = 1 + 0.5 * np.sin(t[k] / 2)
        right = left if k < n // 2 else 1 - 0.5 * np.sin(t[k] / 2) + 1
        amp = np.where(np.arange(W)[None, :] < W // 2, left, right) * np.ones((h, 1))
        slcs.append(amp * np.exp(1j * rng.uniform(0, 0.2, (h, W))) * (1 + 0.02 * rng.standard_normal((h, W))))
    monkeypatch.setattr(unwrap_crop, "compute_crop_window", lambda cfg, write=False: {"r0": 0, "r1": h, "c0": 0, "c1": W // rlks})
    cfg = _project(tmp_path / "p", slcs, dates, rlks, with_fine_int=False)
    cfg.start_date, cfg.end_date = "2020-01-01", "2022-12-31"
    cfg.local_change_window = 2
    cfg.local_change_partner_window_days = 12 * n / 2                  # two periods, split at the rotation
    rec = lc.run(cfg, steps=("partners",), workers=1)
    assert [p["dates"] for p in rec["partners"]["periods"]] == [n // 2, n // 2]
    R, wb = 2, W // rlks - 4
    w1 = np.fromfile(tmp_path / "p" / "local_change" / "partners_R2_p0.u8", np.uint8).reshape(len(lc.offsets(R)), h - 4, wb)
    w2 = np.fromfile(tmp_path / "p" / "local_change" / "partners_R2_p1.u8", np.uint8).reshape(len(lc.offsets(R)), h - 4, wb)
    across = lc.offsets(R).index((0, 1))
    col = wb // 2 - 1                                                  # interior column whose right neighbour is the other half
    assert w1[across, 5, col] == 1 and w2[across, 5, col] == 0


def test_steps_and_samples_use_each_steps_own_period(tmp_path):
    import h5py
    from types import SimpleNamespace
    la, lo = np.mgrid[45.10:45.11:0.001, -100.05:-100.04:0.001]
    shape = la.shape
    with h5py.File(tmp_path / "steps.h5", "w") as f:
        f["step_mm"] = np.full((4,) + shape, -1.0, np.float32)
        f["coherence"] = np.full((4,) + shape, 0.95, np.float16)
        f["latitude"] = la.astype(np.float32); f["longitude"] = lo.astype(np.float32)
        npart = np.zeros((2,) + shape, np.uint8); npart[0] = 50                 # partners in period 0 only
        f["partners"] = npart
        disp = np.full((2,) + shape, 0.3, np.float16); disp[1, 3, 3] = 0.05     # one persistent scatterer in period 1
        f["dispersion"] = disp
        f["pair_period"] = np.array([0, 0, 1, 1], np.int16)
        f["pair"] = np.array(["20240101_20240113", "20240113_20240125", "20240125_20240206", "20240206_20240218"], dtype="S17")
        f["days"] = np.full(4, 12.0)
        f.attrs.update({"complete": True, "rlks": 4})
    cfg = SimpleNamespace(local_change_threshold=0.3, local_change_min_partners=4, local_change_ps_dispersion=0.15)
    g = lc.sample(cfg, tmp_path / "steps.h5", [float(la[3, 3]), float(la[5, 5])], [float(lo[3, 3]), float(lo[5, 5])])
    assert g["n_steps"].tolist() == [4, 2]           # the scatterer keeps its period-1 steps, the other loses them


# ----------------------------------------------------------------- map gridding (review lc-3, 22 Oct 2026)
def _cells(tmp_path, majority, name):
    """6 x 6 map cells of 15 m with 2 x 2 radar pixels in each, except the last
    column, where no radar pixel falls. Values (NaN = not significant):
    cell (1, 1) 3 of 4 pixels (-20, -22, -24); (2, 4) 4 of 4 (-30); (4, 1) 2 of 4
    (-10, -12); (4, 3) 1 of 4 (-8)."""
    import rasterio
    from types import SimpleNamespace
    w, n, dlat = -100.05, 45.11, 15.0 / 111320.0
    vals = {(1, 1): [-20, -22, -24, np.nan], (2, 4): [-30] * 4, (4, 1): [-10, -12, np.nan, np.nan],
            (4, 3): [-8, np.nan, np.nan, np.nan]}
    rows, cols, val = [], [], []
    for cy in range(6):
        for cx in range(5):                                            # column 5: no radar pixel
            for k, (oy, ox) in enumerate(((0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75))):
                rows.append(cy + oy); cols.append(cx + ox); val.append(vals.get((cy, cx), [np.nan] * 4)[k])
    lat = n - np.array(rows) * dlat
    dlon = 15.0 / (111320.0 * np.cos(np.radians(float(np.nanmedian(lat)))))
    lon = w + np.array(cols) * dlon
    cfg = SimpleNamespace(bbox=(w, n - 5.5 * dlat, w + 5.5 * dlon, n), local_change_sigma_km=4.0, local_change_threshold=0.3)
    p = lc._grid_tif(tmp_path / f"{name}.tif", lat, lon, np.array(val, np.float32), cfg, name, majority=majority)
    with rasterio.open(p) as r:
        return r.read(1)


def test_significance_map_shows_only_cells_most_of_whose_pixels_are_significant(tmp_path):
    """NaN in the significance map means "not significant", not "no radar pixel":
    a cell holding one significant pixel among others must not show, and cells
    whose pixels are all non-significant must not be filled from a neighbour
    (one significant pixel became a 3 x 3 patch)."""
    a = _cells(tmp_path, True, "local_change_rate_significant_mm_yr")
    shown = {tuple(map(int, ij)): float(a[tuple(ij)]) for ij in np.argwhere(np.isfinite(a))}
    assert shown == {(1, 1): -22.0, (2, 4): -30.0, (2, 5): -30.0}      # (2, 5) has no radar pixel: filled from (2, 4)


def test_other_maps_average_what_is_there_and_fill_only_cells_without_radar(tmp_path):
    a = _cells(tmp_path, False, "local_change_rate_mm_yr")
    shown = {tuple(map(int, ij)): float(a[tuple(ij)]) for ij in np.argwhere(np.isfinite(a))}
    assert shown == {(1, 1): -22.0, (2, 4): -30.0, (2, 5): -30.0, (4, 1): -11.0, (4, 3): -8.0}


def test_summarize_masks_the_significance_map_by_majority_and_counts_the_share_on_radar_pixels(tmp_path):
    """End to end through summarize() (review lc-3, 22 Oct 2026): 5 x 6 map cells of
    15 m, 2 x 2 radar pixels in each, 20 counted steps per pixel. Cell (1, 1) holds
    one sinking pixel (-1 mm per 12 days, -30.4 mm/yr) among three still ones; cell
    (3, 2) holds four. The significance map shows (3, 2) only; the rate map averages
    (1, 1) to -7.6; share_significant counts radar pixels: 5 of 120."""
    import h5py
    import rasterio
    from types import SimpleNamespace
    w, n, dlat = -100.05, 45.11, 15.0 / 111320.0
    H, W, N = 12, 10, 20
    lat = np.repeat(n - (np.arange(H) * 0.5 + 0.25) * dlat, W).reshape(H, W)
    dlon = 15.0 / (111320.0 * np.cos(np.radians(float(np.nanmedian(lat)))))
    lon = np.tile(w + (np.arange(W) * 0.5 + 0.25) * dlon, (H, 1))
    flip = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)[:, None, None]
    step = np.broadcast_to(flip, (N, H, W)).astype(np.float32).copy()          # +-1 mm: rate 0, t 0
    sink = np.zeros((H, W), bool); sink[2, 2] = True; sink[6:8, 4:6] = True     # cell (1, 1): 1 of 4; (3, 2): 4 of 4
    step[:, sink] = (-1.0 + 0.01 * flip[:, :, 0])
    days = np.full(N, 12.0)
    d = [str(np.datetime64("2024-01-01") + 12 * k).replace("-", "") for k in range(N + 1)]
    names = [f"{a}_{b}" for a, b in zip(d[:-1], d[1:])]
    with h5py.File(tmp_path / "steps.h5", "w") as f:
        f["step_mm"] = step
        f["coherence"] = np.full((N, H, W), 0.95, np.float16)
        f["latitude"] = lat.astype(np.float32); f["longitude"] = lon.astype(np.float32)
        f["partners"] = np.full((1, H, W), 50, np.uint8)
        f["dispersion"] = np.full((1, H, W), 0.3, np.float16)
        f["pair_period"] = np.zeros(N, np.int16)
        f["pair"] = np.array(names, dtype="S17")
        f["days"] = days
        f.attrs.update({"complete": True, "rlks": 4})
    cfg = SimpleNamespace(bbox=(w, n - 5.5 * dlat, w + 4.5 * dlon, n), local_change_sigma_km=4.0,
                          local_change_threshold=0.3, local_change_min_partners=4, local_change_ps_dispersion=0.15,
                          local_change_min_snr=3.0, local_change_min_steps=10, stack_range_looks=4)
    s = lc.summarize(cfg, tmp_path / "steps.h5")
    with rasterio.open(s["files"]["local_change_rate_significant_mm_yr"]) as r:
        sig = r.read(1)
    with rasterio.open(s["files"]["local_change_rate_mm_yr"]) as r:
        rate = r.read(1)
    assert sig.shape == rate.shape == (6, 5)
    want = -365.25 / 12.0
    shown = {tuple(map(int, ij)): float(sig[tuple(ij)]) for ij in np.argwhere(np.isfinite(sig))}
    assert list(shown) == [(3, 2)] and np.isclose(shown[(3, 2)], want, rtol=1e-4)
    assert np.isfinite(rate).all() and np.isclose(rate[1, 1], want / 4, rtol=1e-4) and np.isclose(rate[0, 0], 0, atol=1e-6)
    assert s["eligible_pixels"] == H * W and np.isclose(s["share_significant"], 5 / (H * W))
