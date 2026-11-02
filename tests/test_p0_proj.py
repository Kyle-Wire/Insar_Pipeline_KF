"""Phase 0, P0.3 (review_2026-10-08/PLAN.md section 3): projection fallbacks
for anywhere.

Experiment A (Appin LW902, NSW) is in UTM 56S and experiment B (Corcoran,
CA) in UTM 11N - which is latitude band 11S. Until 8 Oct 2026 a projected
product without an EPSG attribute fell back to UTM 16N (Indiana):

  * stack_qc._reference_rc built 326xx from UTM_ZONE (the 'S' dropped) and
    defaulted to zone 16;
  * stack_backend._polygon_mask and _fit_surrounds_aoi defaulted to "16N" and
    read any trailing 'S' as the southern hemisphere, so a latitude-band
    "11S" put Corcoran 10,000 km south.

Now the zone and hemisphere come from the product (EPSG, else UTM_ZONE)
placed against the point being mapped (stack_qc.product_epsg); there is no
default zone. The Indiana cases are unchanged (regression tests below).

disp_backend had no zone fallback left (_granule_epsg refuses a granule that
states no CRS); the end-to-end test drives the production reads, stitch and
rate fit on a synthetic southern-hemisphere F39360-like frame in EPSG:32756
and checks that the AOI window, the reference point and the product's
georeferencing land on the right ground.

Synthetic inputs only; no network, nothing under insar_projects.
"""
import json
import logging
import sys
import types
from datetime import datetime, timedelta

import h5py
import numpy as np
import pytest
from pyproj import Transformer

from insar_pipeline import disp_backend as db
from insar_pipeline import stack_backend as sb
from insar_pipeline import stack_qc
from insar_pipeline.config import PipelineConfig

CA19 = (150.691241, -34.178878)             # lon, lat: GA CEMP GNSS over Appin LW902 (UTM 56S)
APPIN_BOX = (150.55, -34.30, 150.78, -34.14)
CORCORAN = (-119.67, 36.21)                 # UTM 11N, latitude band S
SITE_A = (-87.69, 38.27)                # UTM 16N


def _xy(epsg, lon, lat):
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)


def _grid_around(epsg, lon, lat, r=12, c=17, ny=40, nx=50, step=30.0, **extra):
    """MintPy attrs of a north-up projected grid with (lon, lat) a quarter
    cell inside cell (r, c) (clear of the rounding boundary)."""
    x, y = _xy(epsg, lon, lat)
    attrs = {"X_FIRST": str(x - (c + 0.25) * step), "Y_FIRST": str(y + (r + 0.25) * step),
             "X_STEP": str(step), "Y_STEP": str(-step), "LENGTH": str(ny), "WIDTH": str(nx),
             "X_UNIT": "meters", "Y_UNIT": "meters"}
    attrs.update({k: str(v) for k, v in extra.items()})
    return attrs, (ny, nx)


def _square(tmp_path, lon, lat, half_deg=0.0012, name="poly.geojson"):
    ring = [[lon - half_deg, lat - half_deg], [lon + half_deg, lat - half_deg],
            [lon + half_deg, lat + half_deg], [lon - half_deg, lat + half_deg],
            [lon - half_deg, lat - half_deg]]
    p = tmp_path / name
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
    return p


# ============================================================ stack_qc.product_epsg
def test_product_epsg_reads_the_product_first():
    assert stack_qc.product_epsg({"EPSG": "32756"}) == 32756
    assert stack_qc.product_epsg({"EPSG": "32756.0"}, lonlat=SITE_A) == 32756   # metadata beats the hint
    assert stack_qc.product_epsg({"EPSG": b"32611"}) == 32611


def test_product_epsg_without_a_location_uses_the_mintpy_letter_and_has_no_default():
    assert stack_qc.product_epsg({"UTM_ZONE": "56S"}) == 32756
    assert stack_qc.product_epsg({"UTM_ZONE": "16N"}) == 32616
    assert stack_qc.product_epsg({"UTM_ZONE": "56H"}) == 32756          # latitude band H: south
    with pytest.raises(ValueError):
        stack_qc.product_epsg({})                                       # no Indiana default
    with pytest.raises(ValueError):
        stack_qc.product_epsg({"UTM_ZONE": "56"})                       # no hemisphere, nothing to decide it
    # a location alone (no grid origin) gives its own zone and hemisphere
    assert stack_qc.product_epsg({}, lonlat=CA19) == 32756
    assert stack_qc.product_epsg({"UTM_ZONE": "56"}, lonlat=CA19) == 32756
    assert stack_qc.product_epsg({}, lonlat=CORCORAN) == 32611


def test_product_epsg_takes_the_hemisphere_from_the_grid_not_the_letter(caplog):
    """'11S' over Corcoran is the latitude band (36 N), not the southern hemisphere."""
    attrs, shape = _grid_around(32611, *CORCORAN, UTM_ZONE="11S")
    with caplog.at_level(logging.WARNING, logger="insar_pipeline.stack_qc"):
        assert stack_qc.product_epsg(attrs, lonlat=CORCORAN, shape=shape) == 32611
    assert "EPSG:32611" in caplog.text
    attrs, shape = _grid_around(32756, *CA19, UTM_ZONE="56S")
    assert stack_qc.product_epsg(attrs, lonlat=CA19, shape=shape) == 32756
    attrs, shape = _grid_around(32756, *CA19)                          # no metadata at all
    assert stack_qc.product_epsg(attrs, lonlat=CA19, shape=shape) == 32756


def test_product_epsg_ignores_a_utm_zone_that_is_not_one(caplog):
    """A 'None' UTM_ZONE (MintPy writes 'None' for absent values) states no
    zone: its last letter 'E' is not a latitude band C-M (south)."""
    with caplog.at_level(logging.WARNING, logger="insar_pipeline.stack_qc"):
        assert stack_qc.product_epsg({"UTM_ZONE": "None"}, lonlat=CORCORAN) == 32611
    assert "not a UTM zone" in caplog.text
    assert stack_qc.product_epsg({"UTM_ZONE": "nan"}, lonlat=CA19) == 32756
    attrs, shape = _grid_around(32611, *CORCORAN, UTM_ZONE="None")
    assert stack_qc.product_epsg(attrs, lonlat=CORCORAN, shape=shape) == 32611
    with pytest.raises(ValueError):
        stack_qc.product_epsg({"UTM_ZONE": "None"})                     # still no default
    with pytest.raises(ValueError):
        stack_qc.product_epsg({"UTM_ZONE": "99N"})                      # zone 99 does not exist


def test_product_epsg_tries_the_neighbouring_zone():
    """150.1 E is zone 56 by longitude, but a frame centred west of 150 E is
    projected in zone 55; the grid decides."""
    pt = (150.1, -34.2)
    attrs, shape = _grid_around(32755, *pt)
    assert stack_qc.product_epsg(attrs, lonlat=pt, shape=shape) == 32755
    attrs, shape = _grid_around(32756, *pt)
    assert stack_qc.product_epsg(attrs, lonlat=pt, shape=shape) == 32756


# ============================================================ stack_qc._reference_rc
@pytest.mark.parametrize("extra", [{}, {"UTM_ZONE": "56S"}])
def test_reference_rc_finds_appin_on_a_southern_grid(extra):
    """Old code: zone 16 (no metadata) or 32656 (the 'S' dropped) put CA19
    off the grid, and the QA fell back to the scene centre."""
    attrs, shape = _grid_around(32756, *CA19, r=12, c=17, **extra)
    r, c, src = stack_qc._reference_rc(attrs, shape, ref_lalo=f"{CA19[1]},{CA19[0]}")
    assert (r, c, src) == (12, 17, "ref_lalo")


def test_reference_rc_corcoran_and_neighbouring_zone():
    attrs, shape = _grid_around(32611, *CORCORAN, r=5, c=30, UTM_ZONE="11S")
    assert stack_qc._reference_rc(attrs, shape, f"{CORCORAN[1]},{CORCORAN[0]}") == (5, 30, "ref_lalo")
    pt = (150.1, -34.2)
    attrs, shape = _grid_around(32755, *pt, r=20, c=3)
    assert stack_qc._reference_rc(attrs, shape, f"{pt[1]},{pt[0]}") == (20, 3, "ref_lalo")


@pytest.mark.parametrize("extra", [{"EPSG": "32616"}, {"UTM_ZONE": "16N"}, {}])
def test_reference_rc_indiana_unchanged(extra):
    """Regression: Site A resolves to 32616 as before, whatever the metadata."""
    attrs, shape = _grid_around(32616, *SITE_A, r=7, c=9, **extra)
    assert stack_qc._reference_rc(attrs, shape, f"{SITE_A[1]},{SITE_A[0]}") == (7, 9, "ref_lalo")


# ============================================================ stack_backend._polygon_mask
def test_polygon_mask_on_a_southern_grid_without_epsg(tmp_path):
    """Old code: no EPSG -> 16N -> an empty mask over Appin."""
    attrs, shape = _grid_around(32756, *CA19, r=20, c=25)
    gj = _square(tmp_path, *CA19)
    got = sb._polygon_mask(gj, shape, attrs)
    want = sb._polygon_mask(gj, shape, {**attrs, "EPSG": "32756"})
    assert want.sum() > 20 and want[20, 25]
    assert np.array_equal(got, want)


def test_polygon_mask_reads_a_band_letter_11S_as_north(tmp_path):
    """Old code: '11S' -> 32711, the mask 10,000 km away and empty."""
    attrs, shape = _grid_around(32611, *CORCORAN, r=20, c=25, UTM_ZONE="11S")
    gj = _square(tmp_path, *CORCORAN)
    got = sb._polygon_mask(gj, shape, attrs)
    want = sb._polygon_mask(gj, shape, {**attrs, "EPSG": "32611"})
    assert want.sum() > 20 and want[20, 25]
    assert np.array_equal(got, want)


def test_polygon_mask_indiana_unchanged_and_empty_geojson(tmp_path):
    for extra in ({"EPSG": "32616"}, {"UTM_ZONE": "16N"}, {}):
        attrs, shape = _grid_around(32616, *SITE_A, r=20, c=25, **extra)
        m = sb._polygon_mask(_square(tmp_path, *SITE_A), shape, attrs)
        assert m[20, 25] and m.sum() > 20
    # nothing to place and nothing stated: an empty mask, not a guess and not a crash
    empty = tmp_path / "empty.geojson"
    empty.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    attrs, shape = _grid_around(32756, *CA19)
    assert not sb._polygon_mask(empty, shape, attrs).any()


# ============================================================ stack_backend._fit_surrounds_aoi
def test_fit_surrounds_aoi_on_a_southern_grid_without_epsg():
    """A ring of fit pixels round the Appin box must count in all four
    quadrants. Old code projected the grid as 16N: every pixel 'north-west'
    of the box centre, one quadrant, and the deramp judged lopsided."""
    w, s, e, n = APPIN_BOX
    xs, ys = zip(*(_xy(32756, lo, la) for lo in (w, e) for la in (s, n)))
    step = 300.0
    x0, y0 = min(xs), max(ys)
    nx, ny = int((max(xs) - x0) / step) + 1, int((y0 - min(ys)) / step) + 1
    attrs = {"X_FIRST": str(x0), "Y_FIRST": str(y0), "X_STEP": str(step), "Y_STEP": str(-step)}
    fit = np.ones((ny, nx), bool)
    fit[ny // 4: 3 * ny // 4, nx // 4: 3 * nx // 4] = False          # the AOI interior excluded
    cfg = types.SimpleNamespace(bbox=list(APPIN_BOX))
    res = sb._fit_surrounds_aoi(cfg, fit, attrs)
    assert res["quadrants"] == 4 and res["surrounds"], res
    assert res == sb._fit_surrounds_aoi(cfg, fit, {**attrs, "EPSG": "32756"})


# ============================================================ disp_backend, southern frame
def _fake_frame(tmp_path, n=24, bowl_rate=-0.10, start=datetime(2018, 5, 1)):
    """An F39360-like descending frame in EPSG:32756 (6 x 6 km at 30 m
    around CA19), served through _open_remote as OPERA's netCDF would be
    (x ascending, y descending, spatial_ref.spatial_epsg). A bowl sinking at
    `bowl_rate` m/yr is centred on CA19; the rest is still."""
    import xarray as xr
    cx, cy = _xy(32756, *CA19)
    cx, cy = 30.0 * round(cx / 30.0), 30.0 * round(cy / 30.0)           # CA19 on a pixel centre
    x = cx + 30.0 * np.arange(-100, 100)
    y = cy - 30.0 * np.arange(-100, 100)
    X, Y = np.meshgrid(x, y)
    bowl = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 150.0 ** 2))
    sec = [start + timedelta(days=12 * (i + 1)) for i in range(n)]
    tyr = np.array([(d - start).days / 365.25 for d in sec])
    names = [f"OPERA_L3_DISP-S1_IW_F39360_VV_{start:%Y%m%d}T191555Z_{d:%Y%m%d}T191555Z_"
             f"v1.0_20250729T120024Z.nc" for d in sec]
    frames = {}
    for i, nm in enumerate(names):
        disp = (bowl_rate * tyr[i] * bowl).astype("float32")
        frames[nm] = xr.Dataset(
            {"displacement": (("y", "x"), disp),
             "spatial_ref": ((), 0, {"spatial_epsg": 32756})},
            coords={"x": x, "y": y})
    return names, frames, (cx, cy)


def test_disp_backend_runs_a_southern_hemisphere_frame(tmp_path, monkeypatch):
    names, frames, (cx, cy) = _fake_frame(tmp_path)
    monkeypatch.setattr(db, "_data_url", lambda p: "https://example.invalid/" + p.properties["fileName"])
    monkeypatch.setattr(db, "_open_remote", lambda url, group=None: frames[url.rsplit("/", 1)[-1]])
    monkeypatch.setattr(db, "_geometry_from_granule", lambda u: (38.0, "19:15:55"))
    stable = (150.700, -34.170)                                          # ~1.2 km from the bowl
    cfg = PipelineConfig(project_dir=tmp_path / "proj", bbox=[150.675, -34.19, 150.705, -34.165],
                         start_date="2018-05-01", end_date="2019-06-01")
    for k, v in dict(backend="disp", track=147, orbit_direction="descending", disp_apply_corrections=False,
                     disp_mask_connected_components=False, disp_flag_unwrap_jumps=False,
                     disp_repair_boundary_steps=False, disp_repair_boundary_datum=False, disp_decimate=1,
                     disp_use_static_layers=False, disp_tropo_correction=False, disp_seasonal_model=False,
                     disp_use_recommended_mask=False, reference_lalo=f"{stable[1]},{stable[0]}").items():
        setattr(cfg, k, v)
    results = [types.SimpleNamespace(properties={"fileName": nm, "pathNumber": 147,
                                                 "flightDirection": "DESCENDING"}, geometry=None, umm={})
               for nm in names]

    # the AOI window is cut from the granule's own CRS (production _read_granule)
    arrays, win, _, _ = db._read_granule(cfg, "https://example.invalid/" + names[0], names[0], None,
                                         db.granule_layers())
    assert arrays is not None and win[-1] == 32756
    assert win[2] > 6e6 and win[3] > 6e6                                 # southern false northing
    assert arrays["displacement"].shape[0] > 50 and arrays["displacement"].shape[1] > 50

    out = db.build_stack(cfg, results)
    with h5py.File(out / "velocity.h5", "r") as h:
        vel = h["velocity"][:]
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h.attrs.items()}
    assert attrs["EPSG"] == "32756" and attrs["FRAME_ID"] == "39360"

    # the reference pixel is the stable point, on the product's own grid
    sx, sy = _xy(32756, *stable)
    x0c = float(attrs["X_FIRST"]) + 15.0
    y0c = float(attrs["Y_FIRST"]) - 15.0
    assert (int(attrs["REF_Y"]), int(attrs["REF_X"])) == (round((sy - y0c) / -30.0), round((sx - x0c) / 30.0))

    # the bowl sits on CA19: find CA19 on the product the way the QA does, with
    # and without the EPSG attribute (the fallback must agree with the product)
    bare = {k: v for k, v in attrs.items() if k not in ("EPSG", "REF_Y", "REF_X")}
    r1, c1, _ = stack_qc._reference_rc({k: v for k, v in attrs.items() if k not in ("REF_Y", "REF_X")},
                                       vel.shape, f"{CA19[1]},{CA19[0]}")
    r2, c2, src = stack_qc._reference_rc(bare, vel.shape, f"{CA19[1]},{CA19[0]}")
    assert (r1, c1) == (r2, c2) and src == "ref_lalo"
    rmin, cmin = np.unravel_index(np.nanargmin(vel), vel.shape)
    assert abs(rmin - r1) <= 1 and abs(cmin - c1) <= 1
    assert vel[rmin, cmin] == pytest.approx(-0.10, abs=0.01)
    # and the product's georeferencing puts that cell back on CA19
    from insar_pipeline import deliverables as dl
    lon, lat = dl._rc_to_lonlat(attrs, int(rmin), int(cmin))
    ex, ey = _xy(32756, float(lon), float(lat))
    assert np.hypot(ex - cx, ey - cy) < 1.0


def test_disp_choose_frame_picks_f39360_on_the_configured_track():
    """Frame choice reads the archive's own track, direction and footprint;
    nothing about it is North American."""
    w, s, e, n = APPIN_BOX
    frame_poly = {"type": "Polygon", "coordinates": [[[149.9, -34.9], [151.3, -34.9], [151.3, -33.4],
                                                      [149.9, -33.4], [149.9, -34.9]]]}
    other = {"type": "Polygon", "coordinates": [[[150.6, -34.25], [151.6, -34.25], [151.6, -33.2],
                                                 [150.6, -33.2], [150.6, -34.25]]]}
    mk = lambda f, trk, d, g, day: types.SimpleNamespace(       # noqa: E731
        properties={"fileName": f"OPERA_L3_DISP-S1_IW_F{f}_VV_20180501T191555Z_201805{day:02d}T191555Z_"
                                f"v1.0_20250729T120024Z.nc", "pathNumber": trk, "flightDirection": d},
        geometry=g, umm={})
    results = [mk("39360", 147, "DESCENDING", frame_poly, d) for d in (13, 25)] + \
              [mk("31111", 9, "ASCENDING", other, d) for d in (13, 25, 26)]
    cfg = types.SimpleNamespace(bbox=[w, s, e, n], orbit_direction="descending", track=147, disp_frame_id="")
    frame, info, rows = db.choose_frame(results, cfg)
    assert frame == "39360" and info["overlap_fraction"] == pytest.approx(1.0)
    cfg.disp_frame_id = "F39360"
    assert db.choose_frame(results, cfg)[0] == "39360"


def test_disp_no_granule_message_does_not_claim_north_america_only(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "asf_search", types.SimpleNamespace(search=lambda **k: []))
    cfg = PipelineConfig(project_dir=tmp_path / "proj", bbox=list(APPIN_BOX),
                         start_date="2017-11-01", end_date="2022-12-31")
    with caplog.at_level(logging.WARNING, logger="insar_pipeline.disp_backend"):
        assert db.search(cfg) == []
    assert "F39360" in caplog.text and "Coverage is North America" not in caplog.text
