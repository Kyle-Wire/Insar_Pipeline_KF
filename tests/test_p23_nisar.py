"""
Phase 3, group nisar (review_2026-10-08/PLAN.md M1, M2, F4; 8 Oct 2026).

  - M1: wavelength, look side and revisit come from the product
    (stack_qc.product_wavelength_m / half_wavelength_mm / product_look_side /
    revisit_days), used by stack_qc's whole-cycle test, injection_test's
    aliasing ceiling and figures' wrapped-phase panel. A Sentinel-1 product
    gives exactly what the code before the change gave (checked against the
    copy in backups/pre_phase23_20261008/).
  - M2: the NISAR GUNW route (nisar_backend): PROVISIONAL only, one track and
    direction, calibration frames out, the four kinds of duplicate, chain gaps,
    windowed reads onto one lat/lon lattice, chronological pairs, the LOS
    vector re-encoded so MintPy 1.6.4's prep_nisar writes MintPy's azimuth
    (found on the real verification crop: it was 90 degrees off), one
    wavelength per stack, the ionosphere sign rule, the cleanup entry, and the
    whole route offline on synthetic GUNW files.
  - F4: lband.py's C-band vs L-band cycle-consistency test.
  - The nine NISAR experiment configs load and match their protocols.

Every behaviour tested here is new: on the code in
backups/pre_phase23_20261008/ these tests fail (missing functions, BETA
searched, frame-keyed groups, prep_nisar's azimuth, C-band cycle at L-band).
No network access: synthetic GUNW files are written to tmp_path.
"""
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from insar_pipeline import lband
from insar_pipeline import nisar_backend as nb
from insar_pipeline import stack_qc

PKG = Path(__file__).resolve().parents[1]
BACKUP = PKG / "backups" / "pre_phase23_20261008"
REVIEW = PKG / "review_2026-10-08"

AOI = (-119.72, 36.18, -119.68, 36.22)          # inside the Corcoran box, UTM 11N
EPSG = 32611
S1_WL = 0.05546576
L40_WL = 299792458.0 / 1.239e9                  # the 40 MHz mode's centre frequency (real ASC-34 granule)
L20_WL = 299792458.0 / 1.229e9


# ============================================================== helpers
def _pre(module: str):
    """The module as it was before phase 2/3, imported from the backup copy
    as its own package (so relative imports resolve inside the backup)."""
    name = "pre_p23_pkg"
    from client_files import need
    need(BACKUP / "__init__.py", "the pre-fix code snapshots stay on the developer's machine")
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BACKUP / "__init__.py",
                                                      submodule_search_locations=[str(BACKUP)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(f"{name}.{module}")


def _utm_centre():
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
    return tr.transform((AOI[0] + AOI[2]) / 2, (AOI[1] + AOI[3]) / 2)


def make_gunw(path, ref="20260908", sec="20260920", *, n=200, cfreq=1.239e9, east=0.641, north=0.182,
              phase=None, ion=None, coherence=0.7, index_phase=False, chunks=(32, 32), pol="HH",
              random_layers=False, mask=11):
    """A small GUNW-layout HDF5 (the datasets crop_gunw and prep_nisar read),
    on an 80 m UTM 11N grid of n x n centred on AOI. phase/ion: callables of
    (X, Y) in metres from the centre, in radians. index_phase: unwrapped phase
    = col + 1000 row (to check the resampling against the source grid).
    random_layers: coherence and ionosphere layers of random values, which do
    not compress, so the data chunks outweigh the metadata as in a real
    2 GB granule (for the windowed-read test)."""
    xc, yc = _utm_centre()
    x = xc + (np.arange(n) - n / 2 + 0.5) * 80.0
    y = yc - (np.arange(n) - n / 2 + 0.5) * 80.0
    X, Y = np.meshgrid(x - xc, y - yc)
    R, C = np.indices((n, n))
    if index_phase:
        unw = (C + 1000.0 * R).astype("float32")
    else:
        unw = (phase(X, Y) if phase else 1e-4 * X + 2e-4 * Y).astype("float32")
    ionv = (ion(X, Y) if ion else 0.2 * np.sin(X / 3000.0)).astype("float32")
    rng = np.random.default_rng(7)
    cohv = rng.uniform(0.3, 0.9, (n, n)).astype("float32") if random_layers else np.full((n, n), coherence, "float32")
    if random_layers:
        ionv = rng.normal(0, 0.2, (n, n)).astype("float32")
    U = nb.UNW + pol + "/"
    with h5py.File(path, "w") as f:
        def ds(p, a, **kw):
            d = f.create_dataset(p, data=a, chunks=chunks if np.ndim(a) == 2 else None,
                                 compression="gzip" if np.ndim(a) == 2 else None)
            for k, v in kw.items():
                d.attrs[k] = v
            return d
        ds(U + "unwrappedPhase", unw, _FillValue=np.float32(np.nan), units=b"radians")
        ds(U + "coherenceMagnitude", cohv, _FillValue=np.float32(np.nan))
        cc = np.ones((n, n), "uint16")
        cc[:10, :10] = 0
        ds(U + "connectedComponents", cc, _FillValue=np.uint16(65535))
        ds(U + "ionospherePhaseScreen", ionv, _FillValue=np.float32(np.nan))
        ds(U + "ionospherePhaseScreenUncertainty", np.full((n, n), 0.05, "float32"), _FillValue=np.float32(np.nan))
        ds(nb.UNW + "mask", np.full((n, n), mask, "uint32"), _FillValue=np.uint32(255))
        for g in (U, nb.UNW, nb.RADARGRID):
            f.create_dataset(g + "projection", data=np.uint32(EPSG)).attrs["epsg_code"] = EPSG
        f.create_dataset(U + "xCoordinates", data=x)
        f.create_dataset(U + "yCoordinates", data=y)
        f.create_dataset(U + "xCoordinateSpacing", data=80.0)
        f.create_dataset(U + "yCoordinateSpacing", data=-80.0)
        f.create_dataset(nb.GRIDS + "centerFrequency", data=float(cfreq))
        f.create_dataset(nb.GRIDS + "listOfPolarizations", data=np.array([pol.encode()], dtype="S2"))
        # radar-grid cube: 500 m, 4 heights, covering the grid + 1 km
        cx = np.arange(x[0] - 1000, x[-1] + 1000.1, 500.0)
        cy = np.arange(y[0] + 1000, y[-1] - 1000.1, -500.0)
        h = np.array([-500.0, 0.0, 500.0, 1000.0])
        shp = (len(h), len(cy), len(cx))
        f.create_dataset(nb.RADARGRID + "xCoordinates", data=cx)
        f.create_dataset(nb.RADARGRID + "yCoordinates", data=cy)
        f.create_dataset(nb.RADARGRID + "heightAboveEllipsoid", data=h)
        CX = np.broadcast_to(cx[None, None, :], shp)
        cube = {"referenceSlantRange": 9.6e5 + (CX - cx[0]) * 0.7, "secondarySlantRange": 9.6e5 + (CX - cx[0]) * 0.7,
                "incidenceAngle": np.full(shp, 41.8), "losUnitVectorX": np.full(shp, east),
                "losUnitVectorY": np.full(shp, north), "wetTroposphericPhaseScreen": np.full(shp, 0.3),
                "hydrostaticTroposphericPhaseScreen": np.full(shp, 1.2),
                "slantRangeSolidEarthTidesPhase": np.full(shp, 0.05), "perpendicularBaseline": np.full(shp, 120.0),
                "parallelBaseline": np.full(shp, 30.0)}
        for k, v in cube.items():
            f.create_dataset(nb.RADARGRID + k, data=np.asarray(v, "float32"), chunks=(1, 16, 16), compression="gzip")
        rt = f"{ref[:4]}-{ref[4:6]}-{ref[6:]}T13:08:56.000000"
        st = f"{sec[:4]}-{sec[4:6]}-{sec[6:]}T13:08:56.000000"
        for k, v in {"referenceZeroDopplerStartTime": rt, "referenceZeroDopplerEndTime": rt.replace("08:56", "09:31"),
                     "secondaryZeroDopplerStartTime": st, "secondaryZeroDopplerEndTime": st.replace("08:56", "09:31"),
                     "lookDirection": "Left", "orbitPassDirection": "Ascending", "missionId": "NISAR",
                     "granuleId": Path(path).stem}.items():
            f.create_dataset(nb.IDENT + k, data=np.bytes_(v))
        f.create_dataset(nb.PARAMS + "numberOfAzimuthLooks", data=np.uint32(16))
        f.create_dataset(nb.PARAMS + "numberOfRangeLooks", data=np.uint32(13))
    return Path(path)


def flat_dem(path, lattice, height=80.0):
    import rasterio
    from rasterio.transform import from_origin
    w, s, e, n = lattice.bbox()
    with rasterio.open(path, "w", driver="GTiff", height=200, width=200, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(w - 0.05, n + 0.05, (e - w + 0.1) / 200,
                                                              (n - s + 0.1) / 200)) as d:
        d.write(np.full((200, 200), height, "float32"), 1)
    return Path(path)


def gname(trk=34, d="A", frame=20, mode="4000", ref="20260908", sec="20260920", cov="F", counter=1):
    return (f"NISAR_L2_PR_GUNW_030_{trk:03d}_{d}_{frame:03d}_031_{mode}_SH_{ref}T130856_{ref}T130931_"
            f"{sec}T130856_{sec}T130931_P05023_N_{cov}_J_{counter:03d}")


def rec(name, coverage=1.0, processing="2026-09-22T19:54:14Z", maturity="PROVISIONAL", url=None):
    return {**nb.parse_name(name), "name": name, "coverage_aoi": coverage, "processing_date": processing,
            "maturity": maturity, "url": url}


# ============================================================== M1
def test_m1_wavelength_from_product_and_sentinel1_unchanged():
    assert stack_qc.product_wavelength_m({"WAVELENGTH": "0.05546576"}) == S1_WL
    assert stack_qc.product_wavelength_m({"WAVELENGTH": 0.0555}) == S1_WL          # within 0.5%: S1 exactly
    assert stack_qc.product_wavelength_m({"WAVELENGTH": L40_WL}) == pytest.approx(L40_WL)
    assert stack_qc.product_wavelength_m({"PLATFORM": "NISAR"}) == stack_qc.NISAR_L_WAVELENGTH_M
    assert stack_qc.product_wavelength_m({}) == S1_WL
    assert stack_qc.half_wavelength_mm({"WAVELENGTH": S1_WL}) == stack_qc.C_BAND_HALF_WAVELENGTH_MM == 27.73
    assert stack_qc.half_wavelength_mm({"WAVELENGTH": L40_WL}) == pytest.approx(1000 * L40_WL / 2)
    assert stack_qc.sensor_band(L40_WL) == "L" and stack_qc.sensor_band(S1_WL) == "C"
    assert stack_qc.default_step_threshold_mm({"WAVELENGTH": S1_WL}) == 20.0
    assert stack_qc.default_step_threshold_mm({"WAVELENGTH": L40_WL}) == pytest.approx(20.0 * 1000 * L40_WL / 2
                                                                                     / 27.73, abs=0.01)


def test_m1_look_side_and_revisit():
    assert stack_qc.product_look_side({"ANTENNA_SIDE": "-1"}) == "right"
    assert stack_qc.product_look_side({"ANTENNA_SIDE": 1}) == "left"
    assert stack_qc.product_look_side({"PLATFORM": "NISAR"}) == "left"
    assert stack_qc.product_look_side({"PLATFORM": "Sen"}) == "right"
    assert stack_qc.product_look_side({}) is None
    assert stack_qc.revisit_days(["20260101", "20260113", "20260125", "20260218"]) == 12.0
    assert stack_qc.revisit_days([], {"PLATFORM": "NISAR"}) == 12.0


def _ts_file(path, wavelength, steps_mm, n=40):
    """A geographic MintPy-like series: every cell takes the same steps."""
    disp = np.concatenate([[0.0], np.cumsum(steps_mm)]) / 1000.0
    ts = np.repeat(disp[:, None, None], n * n, axis=1).reshape(len(disp), n, n).astype("float32")
    ts += np.random.default_rng(1).normal(0, 0.0005, ts.shape).astype("float32")
    with h5py.File(path, "w") as f:
        f.create_dataset("timeseries", data=ts)
        f.create_dataset("date", data=np.array([f"2026{m:02d}01".encode() for m in range(1, len(disp) + 1)]))
        f.attrs.update({"X_FIRST": -119.8, "Y_FIRST": 36.3, "X_STEP": 0.001, "Y_STEP": -0.001, "LENGTH": n,
                        "WIDTH": n, "REF_Y": 0, "REF_X": 0, "WAVELENGTH": wavelength,
                        "PLATFORM": "NISAR" if wavelength > 0.1 else "Sen"})
    return path


def test_m1_whole_cycle_test_uses_the_product_cycle(tmp_path):
    cyc_l = 1000 * L40_WL / 2
    f = _ts_file(tmp_path / "l.h5", L40_WL, [cyc_l, -cyc_l, 2 * cyc_l, 5.0, cyc_l])
    out = stack_qc.step_rms_vs_distance(f, bins_km=(50,))
    assert out["half_wavelength_mm"] == pytest.approx(cyc_l, abs=0.01)
    assert out["step_threshold_mm"] == stack_qc.default_step_threshold_mm({"WAVELENGTH": L40_WL})
    # four of the five steps are whole L-band cycles: the test sees them as such
    assert out["bins"][0]["frac_near_whole_cycle"] > 0.95
    old = _pre("stack_qc").step_rms_vs_distance(f, bins_km=(50,))
    assert old["bins"][0]["frac_near_whole_cycle"] < 0.9          # the C-band cycle mislabels them


def test_m1_sentinel1_step_test_identical_to_before(tmp_path):
    f = _ts_file(tmp_path / "s1.h5", S1_WL, [27.73, -3.0, 40.0, 55.46, 1.0, -27.0])
    new = stack_qc.step_rms_vs_distance(f, bins_km=(1, 50))
    old = _pre("stack_qc").step_rms_vs_distance(f, bins_km=(1, 50))
    assert new == old


def test_m1_injection_aliasing_ceiling_by_sensor():
    from insar_pipeline import injection_test as it
    attrs = {"X_FIRST": -119.8, "Y_FIRST": 36.3, "X_STEP": 0.00072, "Y_STEP": -0.00072, "LENGTH": 100, "WIDTH": 100}
    dates = ["20260101", "20260113", "20260125"]
    cfg = SimpleNamespace()
    s1 = it._aliasing(cfg, {**attrs, "WAVELENGTH": S1_WL}, dates, 0.78)
    assert s1 == _pre("injection_test")._aliasing(cfg, {**attrs, "WAVELENGTH": S1_WL}, dates, 0.78)
    assert "band" not in s1 and s1["quarter_wavelength_mm"] == 13.9
    lb = it._aliasing(cfg, {**attrs, "WAVELENGTH": L40_WL}, dates, 0.78)
    assert lb["band"] == "L" and lb["quarter_wavelength_mm"] == pytest.approx(1000 * L40_WL / 4, abs=0.05)
    assert lb["max_los_rate_gradient_mm_yr_per_m"] == pytest.approx(
        s1["max_los_rate_gradient_mm_yr_per_m"] * L40_WL / S1_WL, rel=0.01)


def test_m1_figures_read_the_product_wavelength(tmp_path):
    from insar_pipeline import figures
    cfg = SimpleNamespace(mintpy_dir=tmp_path)
    assert figures._product_wavelength_m(cfg) == S1_WL                 # nothing states one
    with h5py.File(tmp_path / "velocity.h5", "w") as f:
        f.create_dataset("velocity", data=np.zeros((2, 2), "float32"))
        f.attrs["WAVELENGTH"] = L40_WL
        f.attrs["PLATFORM"] = "NISAR"
    assert figures._product_wavelength_m(cfg) == pytest.approx(L40_WL)


def test_m1_gnss_tie_constant_unchanged():
    from insar_pipeline import gnss_tie
    assert gnss_tie.NOMINAL_LOS_UP == _pre("gnss_tie").NOMINAL_LOS_UP == math.cos(math.radians(38.5))
    assert gnss_tie.NOMINAL_INCIDENCE_DEG["nisar"] != gnss_tie.NOMINAL_INCIDENCE_DEG["sentinel-1"]


# ============================================================== M2: search, groups, duplicates, chain
def test_m2_provisional_only():
    assert nb.GUNW_COLLECTIONS == ["NISAR_L2_GUNW_PROVISIONAL_V1"]
    assert nb.allowed_collections(SimpleNamespace()) == ["NISAR_L2_GUNW_PROVISIONAL_V1"]
    assert "NISAR_L2_GUNW_BETA_V1" in nb.allowed_collections(SimpleNamespace(nisar_allow_beta=True))


def test_m2_name_fields_and_window_filter():
    f = nb.parse_name(gname(trk=76, frame=21, mode="2000", cov="P", counter=2) + ".h5")
    assert (f["track"], f["direction"], f["frame"], f["bandwidth_mhz"], f["full_frame"], f["counter"]) == \
        (76, "A", 21, 20, False, 2)
    assert nb.parse_name("S1A_IW_SLC__1SDV_20200101") is None
    from shapely.geometry import box
    res = [{"properties": {"fileName": gname(ref=r, sec=s) + ".h5", "url": "u", "collectionName": c},
            "geometry": box(-119.73, 36.17, -119.67, 36.23).__geo_interface__}
           for r, s, c in (("20251020", "20251101", "NISAR_L2_GUNW_BETA_V1"), ("20251101", "20251113", ""),
                           ("20260920", "20261002", ""))]
    out = nb.records_from_results(res, box(*AOI).wkt, "2025-11-01", "2026-09-30")
    assert [(r["ref_date"], r["sec_date"]) for r in out] == [("20251101", "20251113")]
    assert out[0]["coverage_aoi"] == pytest.approx(1.0)
    out = nb.records_from_results(res[:1], box(*AOI).wkt)
    assert out[0]["maturity"] == "BETA"


def test_m2_groups_by_track_not_frame():
    """Site A: T076A and T148A share frame 021; the old code merged them."""
    rs = [rec(gname(trk=76, frame=21, ref="20260625", sec="20260707")),
          rec(gname(trk=148, frame=21, ref="20260630", sec="20260712")),
          rec(gname(trk=156, d="D", frame=69, ref="20260701", sec="20260713"))]
    g = nb.groups(rs)
    assert set(g) == {(76, "A"), (148, "A"), (156, "D")}
    chosen = nb.select_group(SimpleNamespace(track=148, orbit_direction="ascending"), rs)
    assert [r["track"] for r in chosen] == [148]
    with pytest.raises(ValueError, match="track"):
        nb.select_group(SimpleNamespace(track=None, orbit_direction="ascending"), rs)
    with pytest.raises(ValueError, match="No PROVISIONAL GUNW on track 34"):
        nb.select_group(SimpleNamespace(track=34, orbit_direction="ascending"), rs)


def test_m2_beta_never_mixed_and_calibration_frames_dropped():
    rs = [rec(gname(ref="20251101", sec="20251113"), maturity="BETA"), rec(gname(ref="20260620", sec="20260702"))]
    chosen = nb.select_group(SimpleNamespace(track=34, orbit_direction="ascending"), rs)
    assert [r["maturity"] for r in chosen] == ["PROVISIONAL"]
    cal = [rec(gname(trk=161, frame=174, ref="20260620", sec="20260702")),
           rec(gname(trk=161, frame=174, ref="20260805", sec="20260817"))]
    kept, dropped = nb.exclude_calibration_frames(cal)
    assert [r["ref_date"] for r in kept] == ["20260805"] and len(dropped) == 1


def test_m2_four_kinds_of_duplicate():
    p = dict(ref="20260908", sec="20260920")
    rs = [
        # pair 1: full vs partial frame (Site A T148A)
        rec(gname(cov="F", **p)), rec(gname(cov="P", **p), processing="2026-09-30T00:00:00Z"),
        # pair 2: product counter 001 vs 002 (Bailey T018A)
        rec(gname(ref="20260920", sec="20261002", counter=1), processing="2026-10-04T00:00:00Z"),
        rec(gname(ref="20260920", sec="20261002", counter=2), processing="2026-10-04T14:00:00Z"),
        # pair 3: 20 vs 40 MHz partials, both covering (Appin T167A F160)
        rec(gname(ref="20260827", sec="20260908", mode="2000", cov="P")),
        rec(gname(ref="20260827", sec="20260908", mode="4000", cov="P")),
        # pair 4: adjacent frames (Katy T062A F016/F017); F021 covers less
        rec(gname(ref="20260815", sec="20260827", frame=20)),
        rec(gname(ref="20260815", sec="20260827", frame=21), coverage=0.5),
    ]
    dd = nb.deduplicate(list(reversed(rs)))          # the answer must not depend on the order ASF returns
    best = {f"{x['pair'][0]}_{x['pair'][1]}": x["candidates"][0] for x in dd["pairs"]}
    assert best["20260908_20260920"]["full_frame"] is True
    assert best["20260920_20261002"]["counter"] == 2
    assert best["20260827_20260908"]["bandwidth_mhz"] == dd["stack_mode_mhz"] == 40
    assert best["20260815_20260827"]["frame"] == 20 == dd["dominant_frame"]
    kinds = {d["pair"]: d["dropped"][0]["kind"] for d in dd["decisions"]}
    assert kinds == {"20260908_20260920": "full vs partial frame",
                     "20260920_20261002": "same pair reprocessed (product counter)",
                     "20260827_20260908": "same pair in another range-bandwidth mode",
                     "20260815_20260827": "same pair in an adjacent frame"}
    assert dd == nb.deduplicate(rs)


def test_m2_one_mode_per_stack_even_when_the_other_is_full():
    """Appin T167A: 20 MHz has every pair, 40 MHz misses one, so 20 MHz wins
    for ALL pairs (one mode per stack), not just the one 40 MHz lacks."""
    pairs = [("20260625", "20260707"), ("20260707", "20260719"), ("20260719", "20260812")]
    rs = [rec(gname(trk=167, frame=160, mode="2000", cov="P", ref=a, sec=b)) for a, b in pairs]
    rs += [rec(gname(trk=167, frame=160, mode="4000", cov="P", ref=a, sec=b)) for a, b in pairs[1:]]
    dd = nb.deduplicate(rs)
    assert dd["stack_mode_mhz"] == 20
    assert {x["candidates"][0]["bandwidth_mhz"] for x in dd["pairs"]} == {20}
    assert dd["mode_mixed"] == []


def test_m2_chain_gaps_and_longest_piece():
    pairs = [("20251101", "20251113"), ("20251113", "20251125"), ("20251125", "20251207"),
             ("20260101", "20260113"), ("20260113", "20260125")]
    segs = nb.chain_segments(pairs)
    assert [len(s) for s in segs] == [3, 2]
    rep = nb.chain_report(pairs)
    assert rep["pieces"] == [4, 3] and rep["gaps"] == [{"from": "20251207", "to": "20260101"}]
    assert rep["loops"] == 0
    assert nb.chain_report([("20260716", "20260821")])["non_12_day_pairs"] == ["20260716_20260821 (36 d)"]


def test_m2_real_metadata_choices():
    """The 8 Oct ASF metadata of the experiment boxes (no data): one mode,
    one frame and the gaps the configs describe."""
    p = REVIEW / "p23" / "nisar" / "meta_probe_20261008.json"
    if not p.exists():
        pytest.skip("metadata probe not present")
    d = json.loads(p.read_text())

    def group(box, trk, dr):
        rs = [rec(r["fileName"][:-3], coverage=r["coverage"], processing=r["processingDate"]) for r in d[box]]
        return nb.deduplicate([r for r in rs if r["track"] == trk and r["direction"] == dr])
    a = group("A5_appin", 167, "A")
    assert (a["stack_mode_mhz"], a["dominant_frame"], len(a["pairs"])) == (20, 160, 7)
    h = group("B2c_houston_katy", 62, "A")
    assert h["dominant_frame"] == 16 and all(x["candidates"][0]["frame"] == 16 for x in h["pairs"])
    e = group("B5_elnido_madera", 42, "D")
    assert nb.chain_report([x["pair"] for x in e["pairs"]])["gaps"] == [{"from": "20260617", "to": "20260629"}]


# ============================================================== M2: reading
def test_m2_windowed_read_and_nearest_neighbour_resampling(tmp_path):
    src = make_gunw(tmp_path / "big.h5", n=400, index_phase=True, chunks=(32, 32), random_layers=True)
    lat = nb.make_lattice(AOI, 80.0, 0.02)
    out = tmp_path / "crop.h5"
    r = nb.crop_gunw(src, lat, out, height_range=(-50, 300), aoi_bbox=AOI)
    assert r["covered"] and r["epsg_source"] == EPSG and r["polarization"] == "HH"
    # only the window was read: the 32 km source grid holds ~25x the lattice's area
    assert r["bytes_read"] < 0.35 * src.stat().st_size
    # every lattice cell holds its NEAREST source pixel (decoded from col + 1000 row)
    from pyproj import Transformer
    with h5py.File(out) as f, h5py.File(src) as s:
        v = f[nb.UNW + "HH/unwrappedPhase"][()]
        lon, la = f[nb.UNW + "HH/xCoordinates"][()], f[nb.UNW + "HH/yCoordinates"][()]
        x, y = s[nb.UNW + "HH/xCoordinates"][()], s[nb.UNW + "HH/yCoordinates"][()]
    LON, LAT = np.meshgrid(lon, la)
    X, Y = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True).transform(LON, LAT)
    row = (v // 1000).astype(int)
    col = (v - 1000 * row).astype(int)
    assert np.all(np.abs(x[col] - X) <= 40.0 + 1e-6) and np.all(np.abs(y[row] - Y) <= 40.0 + 1e-6)
    with h5py.File(out) as f:
        assert int(f[nb.UNW + "HH/projection"][()]) == 4326
        assert f.attrs["crop_version"] == nb.CROP_VERSION


def test_m2_chronological_order_and_reversed_pair(tmp_path):
    lat = nb.make_lattice(AOI, 80.0, 0.02)
    fwd = nb.crop_gunw(make_gunw(tmp_path / "f.h5", "20260908", "20260920"), lat, tmp_path / "cf.h5",
                       height_range=(-50, 300), aoi_bbox=AOI)
    rev = nb.crop_gunw(make_gunw(tmp_path / "r.h5", "20260920", "20260908"), lat, tmp_path / "cr.h5",
                       height_range=(-50, 300), aoi_bbox=AOI)
    assert (fwd["reference"], fwd["secondary"], fwd["reversed_by_pipeline"]) == ("20260908", "20260920", False)
    assert (rev["reference"], rev["secondary"], rev["reversed_by_pipeline"]) == ("20260908", "20260920", True)
    with h5py.File(tmp_path / "cf.h5") as a, h5py.File(tmp_path / "cr.h5") as b:
        for name in ("unwrappedPhase", "ionospherePhaseScreen"):
            np.testing.assert_array_equal(b[nb.UNW + "HH/" + name][()], -a[nb.UNW + "HH/" + name][()])
        np.testing.assert_allclose(b[nb.RADARGRID + "perpendicularBaseline"][()],
                                   -a[nb.RADARGRID + "perpendicularBaseline"][()])
        assert b[nb.IDENT + "referenceZeroDopplerStartTime"][()].decode().startswith("2026-09-08")


def test_m2_los_vector_reaches_mintpy_in_its_own_convention(tmp_path):
    """The defect found on the real ASC-34 crop: MintPy 1.6.4's prep_nisar
    computes azimuthAngle = atan2(-losy, -losx), 90 degrees off MintPy's
    convention, so the pipeline would have read east 0.18 / north -0.64 for a
    product saying east 0.64 / north 0.18. Through the crop it must come out
    right, for an ascending and a descending (left-looking) geometry."""
    from mintpy.prep_nisar import load_nisar
    from argparse import Namespace
    lat = nb.make_lattice(AOI, 80.0, 0.02)
    dem = flat_dem(tmp_path / "dem.tif", lat)
    sin_inc = math.sin(math.radians(41.8))         # make_gunw's incidence: the vector must agree with it
    for tag, (e, n) in {"asc": (0.641, 0.182), "desc": (-0.612, 0.205)}.items():
        e, n = (sin_inc * e / math.hypot(e, n), sin_inc * n / math.hypot(e, n))
        crop = tmp_path / f"crop_{tag}.h5"
        nb.crop_gunw(make_gunw(tmp_path / f"{tag}.h5", east=e, north=n), lat, crop, height_range=(-50, 300),
                     aoi_bbox=AOI)
        with h5py.File(crop) as f:
            np.testing.assert_allclose(np.nanmedian(f[nb.RADARGRID + "losUnitVectorEast"][()]), e, atol=1e-6)
            np.testing.assert_allclose(np.nanmedian(f[nb.RADARGRID + "losUnitVectorNorth"][()]), n, atol=1e-6)
            assert f.attrs["los_vector_encoding"] in [k for k, _ in nb._LOS_ENCODINGS]
        mp = tmp_path / f"mintpy_{tag}"
        (mp / "inputs").mkdir(parents=True)
        load_nisar(Namespace(input_glob=str(crop), dem_file=str(dem), mask_file=None, frequency="A",
                             out_dir=str(mp), update_mode=False, polarization="HH",
                             subset_lat=[AOI[1], AOI[3]], subset_lon=[AOI[0], AOI[2]]))
        with h5py.File(mp / "inputs" / "geometryGeo.h5") as f:
            az = np.radians(np.nanmedian(f["azimuthAngle"][()]))
            inc = np.radians(np.nanmedian(f["incidenceAngle"][()]))
        # deliverables.geometry_field's formulas
        assert -np.sin(inc) * np.sin(az) == pytest.approx(e, abs=2e-3)
        assert np.sin(inc) * np.cos(az) == pytest.approx(n, abs=2e-3)
        assert np.degrees(az) == pytest.approx(float(nb.mintpy_azimuth_deg(e, n)), abs=0.05)


def test_m2_los_encoding_is_found_by_behaviour():
    import inspect
    from mintpy import prep_nisar as pn
    name, enc = nb.los_encoding_for_prep_nisar()
    assert name in [k for k, _ in nb._LOS_ENCODINGS]
    if "np.arctan2(-losy, -losx)" in inspect.getsource(pn.interpolate_geometry):     # MintPy 1.6.4
        assert name == "minus_north_east"
        x, y = enc(0.641, 0.182)
        assert np.degrees(np.arctan2(-y, -x)) == pytest.approx(float(nb.mintpy_azimuth_deg(0.641, 0.182)))
    # the real ASC-34 crop: east 0.640998, north 0.182203 -> -74.13 deg (prep_nisar 1.6.4 wrote -164.13)
    assert float(nb.mintpy_azimuth_deg(0.640998, 0.182203)) == pytest.approx(-74.13, abs=0.01)


def test_m2_crop_through_mintpy_keeps_dates_wavelength_and_lat_lon(tmp_path):
    from mintpy.prep_nisar import load_nisar
    from argparse import Namespace
    lat = nb.make_lattice(AOI, 80.0, 0.02)
    dem = flat_dem(tmp_path / "dem.tif", lat)
    crop = tmp_path / "c.h5"
    nb.crop_gunw(make_gunw(tmp_path / "g.h5"), lat, crop, height_range=(-50, 300), aoi_bbox=AOI)
    mp = tmp_path / "mintpy"
    (mp / "inputs").mkdir(parents=True)
    load_nisar(Namespace(input_glob=str(crop), dem_file=str(dem), mask_file=None, frequency="A", out_dir=str(mp),
                         update_mode=False, polarization="HH", subset_lat=[AOI[1], AOI[3]],
                         subset_lon=[AOI[0], AOI[2]]))
    with h5py.File(mp / "inputs" / "ifgramStack.h5") as f:
        assert [x.decode() for x in f["date"][0]] == ["20260908", "20260920"]
        assert float(f.attrs["WAVELENGTH"]) == pytest.approx(L40_WL)
        assert f.attrs["X_UNIT"] == "degree" and int(f.attrs["EPSG"]) == 4326
    assert (mp / "inputs" / "ionStack.h5").exists()


def test_m2_one_wavelength_per_stack(tmp_path):
    lat = nb.make_lattice(AOI, 80.0, 0.02)
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    ra = nb.crop_gunw(make_gunw(tmp_path / "ga.h5", "20260827", "20260908", cfreq=1.239e9), lat, a,
                      height_range=(-50, 300), aoi_bbox=AOI)
    rb = nb.crop_gunw(make_gunw(tmp_path / "gb.h5", "20260908", "20260920", cfreq=1.229e9), lat, b,
                      height_range=(-50, 300), aoi_bbox=AOI)
    target = nb.stack_wavelength([ra, ra, rb])
    assert target == pytest.approx(L40_WL)
    with h5py.File(b) as f:
        before = f[nb.UNW + "HH/unwrappedPhase"][()].copy()
        tropo0 = f[nb.RADARGRID + "wetTroposphericPhaseScreen"][()].copy()
    changed = nb.harmonise_wavelength([a, b], target)
    assert [Path(c["file"]).name for c in changed] == ["b.h5"]
    with h5py.File(b) as f:
        k = L20_WL / L40_WL
        np.testing.assert_allclose(f[nb.UNW + "HH/unwrappedPhase"][()], before * k, rtol=1e-6)
        np.testing.assert_allclose(f[nb.RADARGRID + "wetTroposphericPhaseScreen"][()], tropo0 * k, rtol=1e-6)
        assert 299792458.0 / float(f[nb.GRIDS + "centerFrequency"][()]) == pytest.approx(L40_WL)
        # the range change of the pair is unchanged: -lambda/4pi x phase
        np.testing.assert_allclose(-L40_WL / (4 * np.pi) * f[nb.UNW + "HH/unwrappedPhase"][()],
                                   -L20_WL / (4 * np.pi) * before, rtol=1e-6)
    assert nb.harmonise_wavelength([a, b], target) == []           # idempotent


def test_m2_ionosphere_sign_rules():
    rng = np.random.default_rng(3)
    ion = rng.normal(0, 1, (60, 60))
    unw = ion + 0.3 * rng.normal(0, 1, (60, 60))
    ok = np.ones((60, 60), bool)
    s = nb.ionosphere_sign_stats(unw, ion, ok)
    assert s["informative"] and s["subtract_better"] and s["corr"] > 0.9
    assert not nb.ionosphere_sign_stats(unw, -ion, ok)["subtract_better"]
    inf_sub = {"informative": True, "subtract_better": True}
    inf_add = {"informative": True, "subtract_better": False}
    assert nb.summarise_ionosphere_sign([inf_add] * 3 + [inf_sub] * 2)["verdict"].startswith("doubtful")
    assert nb.summarise_ionosphere_sign([inf_add] * 4 + [inf_sub])["verdict"] == "opposite sign"
    assert nb.summarise_ionosphere_sign([inf_add] * 4)["verdict"].startswith("doubtful")   # < 5 pairs: no stop
    assert nb.summarise_ionosphere_sign([inf_sub] * 5)["verdict"].startswith("subtract")
    assert nb.summarise_ionosphere_sign([inf_add] * 2)["verdict"].startswith("not testable")


def test_m2_loader_lines_and_no_phase_closure_on_a_chain():
    cfg = SimpleNamespace(mintpy_unwrap_error_method="bridging+phase_closure")
    assert nb.unwrap_error_method(cfg)[0] == "bridging"
    assert nb.unwrap_error_method(SimpleNamespace(mintpy_unwrap_error_method="phase_closure"))[0] == "no"
    txt = nb.mintpy_loader_lines(cfg, Path("/x/stack"), Path("/x/dem.tif"), True)
    assert "mintpy.load.processor              = nisar" in txt
    assert "mintpy.load.demFile                = /x/dem.tif" in txt
    assert "mintpy.ionosphericDelay.method     = split_spectrum" in txt
    assert "mintpy.network.coherenceBased      = no" in txt
    assert "phase_closure" not in txt


# ============================================================== M2: the whole route, offline
def _cfg(tmp_path, **kw):
    return SimpleNamespace(project_dir=tmp_path / "proj", mintpy_dir=tmp_path / "proj" / "mintpy",
                           dem_dir=tmp_path / "proj" / "DEM", dem_path=None, bbox=list(AOI),
                           aoi_wkt=__import__("shapely.geometry", fromlist=["box"]).box(*AOI).wkt,
                           start_date="2025-11-01", end_date="2026-09-30", track=34, orbit_direction="ascending",
                           min_temporal_coherence=0.4, mintpy_unwrap_error_method="bridging",
                           mintpy_tropo_method="pyaps", skip_existing=True, subset_to_aoi=True,
                           ionosphere_correction="no", **kw)


def test_m2_run_offline(tmp_path, monkeypatch):
    """search -> one track -> duplicates -> crops -> screen -> longest chain ->
    stack links, record and MintPy loader, on synthetic local granules."""
    pairs = [("20260816", "20260828"), ("20260828", "20260909"), ("20260909", "20260921"),
             ("20251201", "20251213")]          # the last is cut off from the chain
    rng = np.random.default_rng(0)
    recs = []
    for i, (a, b) in enumerate(pairs):
        ionf = (lambda X, Y, k=i: 0.4 * np.sin(X / 2500.0 + k) + 0.3 * np.cos(Y / 1800.0))
        g = make_gunw(tmp_path / f"g{i}.h5", a, b, ion=ionf,
                      phase=lambda X, Y, f=ionf: f(X, Y) + 0.05 * rng.normal(size=X.shape))
        recs.append(rec(gname(ref=a, sec=b), url=str(g)))
    # a duplicate of pair 0: a partial frame that must lose
    recs.append(rec(gname(ref=pairs[0][0], sec=pairs[0][1], cov="P"), url=str(tmp_path / "g0.h5")))
    # another track: must be ignored
    recs.append(rec(gname(trk=114, d="D", frame=70, ref="20260820", sec="20260901"), url="none"))
    monkeypatch.setattr(nb, "search_gunw", lambda cfg: recs)
    monkeypatch.setattr(nb, "stage_dem", lambda cfg, lat: flat_dem(tmp_path / "dem.tif", lat))
    cfg = _cfg(tmp_path)
    stack_dir, loader = nb.run(cfg)
    links = sorted(p.name for p in stack_dir.glob("*.h5"))
    assert links == [nb._crop_name(p, 34, "A") for p in pairs[:3]]
    R = json.loads((cfg.project_dir / "nisar_gunw" / "nisar_stack.json").read_text())
    assert R["chain_dropped_pairs"] == ["20251201_20251213"]
    assert R["duplicates"]["decisions"][0]["dropped"][0]["kind"] == "full vs partial frame"
    assert R["chain"]["pieces"] == [4] and R["chain"]["phase_closure"].startswith("not applicable")
    assert R["sensor"]["wavelength_m"] == pytest.approx(L40_WL) and R["sensor"]["look_direction"] == ["left"]
    assert R["sensor"]["los_vector_encoding"] == nb.los_encoding_for_prep_nisar()[0]
    assert R["ionosphere"]["sign_check"]["subtract_better"] == 3
    assert "split_spectrum" in loader and str(stack_dir) in loader
    assert R["coherence_gate"]["share_reliable"] == 1.0
    assert (cfg.project_dir / "nisar_gunw" / "coherence_gate.tif").exists()
    # a second run reuses the crops (nothing read again)
    stack_dir2, _ = nb.run(cfg)
    R2 = json.loads((cfg.project_dir / "nisar_gunw" / "nisar_stack.json").read_text())
    assert R2["fetch"]["bytes_read_total"] == 0


def test_product_mask_decoded_as_prep_nisar_decodes_it():
    """Real values from Site E T019A (26 Oct 2026): 11 and 12 valid (bit 24,
    the ionosphere-filter flag, set on most), 124 water, 10 / 1 the
    2026-09-07 acquisition with no valid sample (subswath digit 0), 255 fill."""
    vals = np.array([11, 16777228, 16777340, 10, 1, 255, 111, 12], "uint32")
    assert nb.product_mask_keep(vals, 255).tolist() == [True, True, False, False, False, False, False, True]


def test_a_pair_the_product_mask_rejects_is_screened_even_from_a_reused_crop(tmp_path, monkeypatch):
    """Site E (26 Oct 2026): two pairs whose mask rejected the whole box
    passed the old screen at 100% valid; prep_nisar then masked every cell of
    the stack (it keeps only cells valid in EVERY pair) and MintPy stopped
    with NaN coherence in all pairs. Crops kept from that run carry the old
    valid share in their attributes: a re-run screens them again from their
    layers, without reading them again."""
    pairs = [("20260721", "20260802"), ("20260802", "20260814"), ("20260814", "20260826"),
             ("20260826", "20260907"), ("20260907", "20260919"), ("20260919", "20261001")]
    recs = []
    for i, (a, b) in enumerate(pairs):
        g = make_gunw(tmp_path / f"g{i}.h5", a, b, mask=11)
        recs.append(rec(gname(ref=a, sec=b), url=str(g)))
    monkeypatch.setattr(nb, "search_gunw", lambda cfg: recs)
    monkeypatch.setattr(nb, "stage_dem", lambda cfg, lat: flat_dem(tmp_path / "dem.tif", lat))
    cfg = _cfg(tmp_path)
    cfg.start_date, cfg.end_date = "2026-07-01", "2026-10-15"
    stack_dir, _ = nb.run(cfg)
    assert len(list(stack_dir.glob("*.h5"))) == 6
    # the 2026-09-07 acquisition turns out empty: rewrite the two crops' masks in place
    crops = cfg.project_dir / "nisar_gunw" / "crops"
    for pair, v in ((pairs[3], 10), (pairs[4], 1)):
        with h5py.File(crops / nb._crop_name(pair, 34, "A"), "r+") as f:
            f[nb.UNW + "mask"][...] = v
            assert float(f.attrs["valid_share_aoi"]) > 0.9           # what the old screen recorded
    stack_dir, _ = nb.run(cfg)
    R = json.loads((cfg.project_dir / "nisar_gunw" / "nisar_stack.json").read_text())
    assert R["fetch"]["bytes_read_total"] == 0                       # reused, not read again
    assert sorted(p.name for p in stack_dir.glob("*.h5")) == [nb._crop_name(p, 34, "A") for p in pairs[:3]]
    assert R["chain"]["pieces"] == [4] and R["chain_dropped_pairs"] == ["20260919_20261001"]


def test_m2_run_refuses_gim_and_wrong_sign(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.ionosphere_correction = "gim"
    with pytest.raises(ValueError, match="twice|IGS"):
        nb.run(cfg)
    cfg.ionosphere_correction = "no"
    dates = ["20260705", "20260717", "20260729", "20260810", "20260822", "20260903"]
    pairs = list(zip(dates, dates[1:]))
    recs = []
    for i, (a, b) in enumerate(pairs):
        ionf = (lambda X, Y, k=i: 0.4 * np.sin(X / 2500.0 + k))
        g = make_gunw(tmp_path / f"w{i}.h5", a, b, ion=ionf, phase=lambda X, Y, f=ionf: -f(X, Y))
        recs.append(rec(gname(ref=a, sec=b), url=str(g)))
    monkeypatch.setattr(nb, "search_gunw", lambda c: recs)
    monkeypatch.setattr(nb, "stage_dem", lambda c, lat: flat_dem(tmp_path / "dem.tif", lat))
    with pytest.raises(RuntimeError, match="NEGATIVELY"):
        nb.run(cfg)


# ============================================================== M2: cleanup entry
def test_m2_cleanup_entry(tmp_path):
    cfg = SimpleNamespace(project_dir=tmp_path / "p", mintpy_dir=tmp_path / "p" / "mintpy")
    P = cfg.project_dir / "nisar_gunw"
    (P / "crops").mkdir(parents=True)
    (P / "stack").mkdir()
    (P / "whole").mkdir()
    (P / "whole" / "G.h5").write_bytes(b"x" * 100)
    for n in ("NISAR_GUNW_CROP_T034A_1_2.h5", "NISAR_GUNW_CROP_T034A_2_3.h5", "NISAR_GUNW_CROP_T034A_9_10.h5"):
        (P / "crops" / n).write_bytes(b"y" * 10)
    for n in ("NISAR_GUNW_CROP_T034A_1_2.h5", "NISAR_GUNW_CROP_T034A_2_3.h5"):
        (P / "stack" / n).symlink_to(P / "crops" / n)
    (P / "coherence_gate.tif").write_bytes(b"g")
    t1 = nb.cleanup_targets(cfg)
    lv1 = sorted(Path(p).name for lv, p, _ in t1 if lv == 1)
    assert lv1 == ["G.h5", "NISAR_GUNW_CROP_T034A_9_10.h5"]
    rec2 = nb.cleanup(cfg, 2)
    assert rec2["level_2_refused"] and all(r["level"] == 1 for r in rec2["items"])
    (cfg.mintpy_dir / "inputs").mkdir(parents=True)
    (cfg.mintpy_dir / "inputs" / "ifgramStack.h5").write_bytes(b"s")
    dry = nb.cleanup(cfg, 2)
    assert dry["deleted"] == [] and {Path(r["path"]).name for r in dry["items"]} == {"G.h5", "crops", "stack"}
    assert nb.cleanup(cfg, 2, delete=True, confirm=lambda _: "no")["cancelled"]
    done = nb.cleanup(cfg, 2, delete=True, confirm=lambda _: "yes")
    assert len(done["deleted"]) == 3
    assert sorted(p.name for p in P.iterdir()) == ["coherence_gate.tif", "whole"]
    assert (cfg.mintpy_dir / "inputs" / "ifgramStack.h5").exists()


def test_review_crop_records_the_ionosphere_estimator(tmp_path):
    """Independent review, 8 Oct 2026: the PROVISIONAL products' screen comes
    from ISCE3's main_diff_ms_band estimator (not the split_main_band one the
    first sign check read); each crop now says which, so the sign evidence can
    be matched to the product."""
    src = make_gunw(tmp_path / "g.h5")
    alg = ("Range split-spectrum with sub-band interferograms from the main band and the difference of the "
           "main and side band of the input RSLCs")
    with h5py.File(src, "a") as f:
        f.create_dataset(nb.IONO_ALGORITHM, data=np.bytes_(alg))
    lat = nb.make_lattice(AOI, 80.0, 0.02)
    r = nb.crop_gunw(src, lat, tmp_path / "c.h5", height_range=(-50, 300), aoi_bbox=AOI)
    assert r["ionosphere_algorithm"] == alg
    assert nb.read_crop_record(tmp_path / "c.h5")["ionosphere_algorithm"] == alg
    r2 = nb.crop_gunw(make_gunw(tmp_path / "h.h5"), lat, tmp_path / "d.h5", height_range=(-50, 300), aoi_bbox=AOI)
    assert r2["ionosphere_algorithm"] is None


def test_review_isce3_estimators_put_the_screen_inside_the_main_band_phase():
    """The sign argument, re-derived numerically from ISCE3's own solvers
    (main_band_estimation.py, copy in review_2026-10-08/p23/nisar/
    isce3_sign_review/): plant phi_main = nd + disp with a side band at f1 and
    the solvers must return disp, so subtracting the screen is the correction."""
    f0, f1 = 1.239e9, 1.2925e9                      # 40 MHz main band, 5 MHz side band (approximate centres)
    nd = np.array([[0.7, -1.3]])
    disp = np.array([[0.2, 0.05]])
    phi_main = nd + disp
    phi_side = (f1 / f0) * nd + (f0 / f1) * disp
    # estimate_iono_main_diff: rows [1, 1] and [(f1-f0)/f1, (f0-f1)/f0], output (disp, nd)
    A = np.array([[1.0, 1.0], [(f1 - f0) / f1, (f0 - f1) / f0]])
    out = np.linalg.pinv(A) @ np.vstack([phi_main.ravel(), (phi_main - phi_side).ravel()])
    np.testing.assert_allclose(out[0], disp.ravel(), atol=1e-9)
    # estimate_iono_main_side: rows [1, 1] and [f1/f0, f0/f1], output (nd, disp)
    B = np.array([[1.0, 1.0], [f1 / f0, f0 / f1]])
    out = np.linalg.pinv(B) @ np.vstack([phi_main.ravel(), phi_side.ravel()])
    np.testing.assert_allclose(out[1], disp.ravel(), atol=1e-9)
    from client_files import need
    src = need(REVIEW / "p23" / "nisar" / "isce3_sign_review" / "main_band_estimation.py.txt",
               "review records stay on the developer's machine").read_text()
    assert "coeff_mat[1, 0] = (f1 - f0) / f1" in src and "dispersive = output[0].reshape" in src


# ============================================================== F4: lband
def test_f4_classes_and_chance_level():
    cyc = 27.73
    d = np.array([0.0, 10.0, cyc, -2 * cyc + 3.0, 20.0, 1.5 * cyc, np.nan])
    r = lband.classify_differences(d, cyc)
    assert r["classes"].tolist() == [1, 1, 2, 2, 3, 3, 0]
    assert r["k_counts"] == {"-2": 1, "1": 1}
    assert r["chance_share"] == pytest.approx(2 * 5.0 / cyc, abs=1e-3)
    assert r["tol_mm"] == 5.0 == lband.TOL_MM                      # protocol_B1 against_sentinel1



def test_review_f4_noise_ceiling_bounds_the_null():
    """Independent review: the 36% chance level is for differences spread over
    many cycles. Pure noise (no aliasing) of any width gives at most ~0.37
    among disagreeing cells, so a tiny excess with a huge z is not evidence;
    the record now says where that line is."""
    cyc = stack_qc.C_BAND_HALF_WAVELENGTH_MM
    ceil = lband.noise_null_ceiling(cyc)
    assert 2 * 5.0 / cyc < ceil <= 0.375
    rng = np.random.default_rng(3)
    for sd in (8.0, 20.0, 60.0):
        r = lband.classify_differences(rng.normal(0, sd, 200000), cyc)
        assert r["whole_cycle_share_of_disagreeing"] <= ceil + 0.005
    # and planted aliasing clears it by far
    d = np.concatenate([rng.normal(0, 8.0, 50000), cyc + rng.normal(0, 2.0, 20000)])
    r = lband.classify_differences(d, cyc)
    assert r["whole_cycle_share_of_disagreeing"] > ceil + 0.3

def test_f4_intervals_and_end_offsets():
    c = ["20260601", "20260613", "20260625", "20260707", "20260719"]
    l_ = ["20260603", "20260615", "20260627", "20260709"]
    iv = lband.match_intervals(c, l_, 6)
    assert [x["kind"] for x in iv] == ["consecutive"] * 3 + ["whole window"]
    assert iv[0]["end_offsets_days"] == [-2, -2]
    assert iv[-1]["dates_l"] == ["20260603", "20260709"]


def test_f4_planted_c_band_aliasing_is_found():
    """Truth: a bowl sinking 60 mm per 12 days. L-band sees it whole; C-band
    misses one cycle (27.7 mm) in its centre in every interval."""
    gl = lband.Grid(-119.8, 36.3, 0.0008, -0.0008, 40, 40)
    gc = lband.Grid(-119.8, 36.3, 0.0002, -0.0002, 160, 160)
    yy, xx = np.indices((40, 40))
    bowl = -60.0 * np.exp(-(((xx - 20) ** 2 + (yy - 20) ** 2) / 40.0))
    l_up, c_up = 0.75, 0.80
    dates_l = ["20260601", "20260613", "20260625", "20260707"]
    dates_c = ["20260602", "20260614", "20260626", "20260708"]
    cube_l = np.stack([k * bowl * l_up for k in range(4)])
    centre = np.zeros((40, 40), bool)
    centre[17:24, 17:24] = True
    cube_c = []
    for k in range(4):
        v = k * bowl * c_up + np.where(centre, k * 27.73, 0.0)    # aliased: one cycle short per interval
        cube_c.append(np.kron(v, np.ones((4, 4))))
    c = {"dates": dates_c, "cube": np.stack(cube_c), "grid": gc, "wavelength_m": S1_WL, "los_up": c_up}
    l_ = {"dates": dates_l, "cube": cube_l, "grid": gl, "wavelength_m": L40_WL, "los_up": l_up}
    r = lband.cycle_consistency(c, l_)
    assert r["cycle_mm"] == 27.73 and r["summary"]["intervals"] == 3
    cnt = r["count_map"]
    assert (cnt[centre] == 3).all() and (cnt[~centre] == 0).all()        # consecutive intervals only
    w = r["summary"]["whole_window"]
    assert w["k_counts"] == {"3": int(centre.sum())}


def test_f4_resample_and_nisar_gate_on_its_own_lattice(tmp_path):
    gc = lband.Grid(0.0, 1.0, 0.25, -0.25, 8, 8)
    gl = lband.Grid(0.0, 1.0, 0.5, -0.5, 4, 4)
    src = np.arange(64, dtype=float).reshape(8, 8)
    out = lband.resample_to(src, gc, gl)
    assert out[0, 0] == np.median([0, 1, 8, 9])
    back = lband.resample_to(out, gl, gc)
    assert back[1, 1] == out[0, 0]
    # the coherence gate: lattice with a margin, series grid inside it
    import rasterio
    from rasterio.transform import from_origin
    lat = nb.make_lattice(AOI, 80.0, 0.02)
    gate = np.zeros((lat.ny, lat.nx), "float32")
    gate[: lat.ny // 2] = 1.0
    P = tmp_path / "nisar_gunw"
    P.mkdir()
    with rasterio.open(P / "coherence_gate.tif", "w", driver="GTiff", height=lat.ny, width=lat.nx, count=2,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(lat.west, lat.north, lat.dlon, lat.dlat)) as d:
        d.write(gate, 1)
        d.write(gate, 2)
    k = int(round(0.02 / lat.dlat))
    g = lband.Grid(lat.west + k * lat.dlon, lat.north - k * lat.dlat, lat.dlon, -lat.dlat, 30, 40)
    m = lband._reliable(SimpleNamespace(project_dir=tmp_path, mintpy_dir=tmp_path / "mintpy"), (40, 30), g)
    assert m is not None and m.shape == (40, 30)
    assert m[0].all() and not m[-1].any()


def _fake_run(root, wavelength, dates, cube_mm, grid, inc_deg=None, az_deg=None):
    M = root / "mintpy"
    (M / "inputs").mkdir(parents=True)
    with h5py.File(M / "timeseries.h5", "w") as f:
        f.create_dataset("timeseries", data=(np.asarray(cube_mm) / 1000.0).astype("float32"))
        f.create_dataset("date", data=np.array([d.encode() for d in dates]))
        f.attrs.update({"X_FIRST": grid.x_first, "Y_FIRST": grid.y_first, "X_STEP": grid.dx, "Y_STEP": grid.dy,
                        "LENGTH": grid.ny, "WIDTH": grid.nx, "WAVELENGTH": wavelength,
                        "PLATFORM": "NISAR" if wavelength > 0.1 else "Sen"})
    if inc_deg is not None:
        with h5py.File(M / "inputs" / "geometryGeo.h5", "w") as f:
            f.create_dataset("incidenceAngle", data=np.full((grid.ny, grid.nx), inc_deg, "float32"))
            f.create_dataset("azimuthAngle", data=np.full((grid.ny, grid.nx), az_deg, "float32"))
    return SimpleNamespace(project_dir=root, mintpy_dir=M)


def test_f4_run_reads_two_projects_and_writes_the_record(tmp_path, monkeypatch):
    from insar_pipeline import deliverables
    import rasterio
    gl = lband.Grid(-119.8, 36.3, 0.0008, -0.0008, 20, 20)
    gc = lband.Grid(-119.8, 36.3, 0.0004, -0.0004, 40, 40)
    dl_, dc_ = ["20260601", "20260613", "20260625"], ["20260603", "20260615", "20260627"]
    patch = np.zeros((20, 20), bool)
    patch[5:9, 5:9] = True
    l_up = math.cos(math.radians(41.8))
    cube_l = np.stack([k * -30.0 * l_up * np.ones((20, 20)) for k in range(3)])
    cube_c = np.stack([np.kron(k * (-30.0 * 0.78 + np.where(patch, 27.73, 0.0)), np.ones((2, 2))) for k in range(3)])
    cfg_c = _fake_run(tmp_path / "c", S1_WL, dc_, cube_c, gc)
    cfg_l = _fake_run(tmp_path / "l", L40_WL, dl_, cube_l, gl, inc_deg=41.8, az_deg=-74.13)
    monkeypatch.setattr(deliverables, "timeseries_file", lambda cfg: cfg.mintpy_dir / "timeseries.h5")
    monkeypatch.setattr(deliverables, "geometry_field",
                        lambda cfg, name: (np.full((40, 40), 0.78) if name == "los_up" else np.full((40, 40), -0.6))
                        if cfg is cfg_c else None)
    rec = lband.run(cfg_c, cfg_l, tmp_path / "out")
    assert rec["l_band"]["look_side"] == "left" and rec["c_band"]["look_side"] == "right"
    assert rec["cycle_mm"] == 27.73 and rec["summary"]["intervals"] == 2
    assert rec["by_interval"][0]["end_offsets_days"] == [2, 2]
    with rasterio.open(tmp_path / "out" / "lband_whole_cycle_count.tif") as src:
        cnt = src.read(1)
    assert (cnt[patch] == 2).all() and (cnt[~patch] == 0).all()
    assert json.loads((tmp_path / "out" / "lband_cycle_consistency.json").read_text())["tol_mm"] == 5.0


# ============================================================== configs