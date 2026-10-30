"""gis_products.py (Stage 3b, the classic/local ISCE route) - readiness review TST-03, 16 Oct 2026.

These are the first tests of make_wrapped_phase, make_los_displacement and
make_gis_products (before this only test_fix_L7a's zero-and-sign check
touched the module). Every input is built the way topsApp leaves it in
merged/: a raw little-endian binary plus ISCE2's .xml sidecar (the property
and component names ISCE2 2.6.5's own renderHdr writes), opened by rasterio
through GDAL's ISCE driver exactly as in production - complex64 BIP for
filt_topophase.flat.geo, float32 BIL with band 1 amplitude and band 2
unwrapped phase for filt_topophase.unw.geo. Each test reads the OUTCOME
back from disk: values, file names, dtype, nodata, CRS and transform.

One defect found and fixed here: make_gis_products wrapped both rasters in
one try, so an unreadable wrapped interferogram stopped the displacement
raster from being written at all (test_one_unreadable_input_does_not_cost_the_other_raster
fails on f3a0a0a).

The grid is synthetic (lon -100.0, lat 40.0); it is no real site.
"""
import logging

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from insar_pipeline import gis_products as gp

NY, NX = 6, 8
X0, Y0, D = -100.0, 40.0, 0.001
GRID = Affine(D, 0.0, X0, 0.0, -D, Y0)
L_BAND = 0.2384                       # an L-band wavelength (m), for the scaling check

# ISCE2's .xml sidecar, cut down to what GDAL's ISCE driver and ISCE2's own
# readers use; names and layout as isce2 2.6.5 writes them.
_ISCE_XML = """<imageFile>
    <property name="access_mode"><value>read</value></property>
    <property name="byte_order"><value>l</value></property>
    <component name="coordinate1">
        <factorymodule>isceobj.Image</factorymodule>
        <factoryname>createCoordinate</factoryname>
        <property name="delta"><value>{dx}</value></property>
        <property name="size"><value>{width}</value></property>
        <property name="startingvalue"><value>{x0}</value></property>
    </component>
    <component name="coordinate2">
        <factorymodule>isceobj.Image</factorymodule>
        <factoryname>createCoordinate</factoryname>
        <property name="delta"><value>{dy}</value></property>
        <property name="size"><value>{length}</value></property>
        <property name="startingvalue"><value>{y0}</value></property>
    </component>
    <property name="data_type"><value>{data_type}</value></property>
    <property name="file_name"><value>{name}</value></property>
    <property name="image_type"><value>{image_type}</value></property>
    <property name="length"><value>{length}</value></property>
    <property name="number_bands"><value>{bands}</value></property>
    <property name="scheme"><value>{scheme}</value></property>
    <property name="width"><value>{width}</value></property>
</imageFile>
"""


def _sidecar(path, shape, *, data_type, image_type, bands, scheme):
    length, width = shape
    path.with_name(path.name + ".xml").write_text(_ISCE_XML.format(
        dx=D, dy=-D, x0=X0, y0=Y0, width=width, length=length, data_type=data_type,
        name=path.name, image_type=image_type, bands=bands, scheme=scheme))


def write_flat(merged, cpx):
    """filt_topophase.flat.geo: one complex64 band, ISCE's 0+0j outside the footprint."""
    merged.mkdir(parents=True, exist_ok=True)
    path = merged / "filt_topophase.flat.geo"
    np.asarray(cpx, dtype="<c8").tofile(path)
    _sidecar(path, cpx.shape, data_type="CFLOAT", image_type="cpx", bands=1, scheme="BIP")
    return path


def write_unw(merged, amp, phase):
    """filt_topophase.unw.geo: float32 BIL, line by line amplitude then unwrapped phase."""
    merged.mkdir(parents=True, exist_ok=True)
    path = merged / "filt_topophase.unw.geo"
    np.stack([amp, phase], axis=1).astype("<f4").tofile(path)        # (line, band, column)
    _sidecar(path, amp.shape, data_type="FLOAT", image_type="unw", bands=2, scheme="BIL")
    return path


def read(path):
    with rasterio.open(path) as r:
        return r.read(1)


def wrap(x):
    return (np.asarray(x, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def assert_gis_ready_on_input_grid(out, src):
    """The output is what QGIS needs (single-band Float32 GeoTIFF, nodata NaN)
    and sits on exactly the input's grid: same size, transform and CRS."""
    with rasterio.open(src) as s, rasterio.open(out) as o:
        assert o.driver == "GTiff"
        assert o.count == 1 and o.dtypes == ("float32",)
        assert o.nodata is not None and np.isnan(o.nodata)
        assert (o.width, o.height) == (s.width, s.height) == (NX, NY)
        assert o.transform == s.transform
        assert o.transform.almost_equals(GRID)
        assert o.crs == s.crs and o.crs.to_epsg() == 4326


# ------------------------------------------------------------------ the fixture itself

def test_synthetic_inputs_read_back_as_topsapp_outputs(tmp_path):
    """Guards every test below: the hand-built inputs open through GDAL's ISCE
    driver (as topsApp's do), on the synthetic grid in WGS84, with the BIL
    band order the module relies on (band 1 amplitude, band 2 phase)."""
    amp = np.full((NY, NX), 7.0, np.float32)
    ph = np.arange(NY * NX, dtype=np.float32).reshape(NY, NX)
    unw = write_unw(tmp_path, amp, ph)
    flat = write_flat(tmp_path, np.exp(1j * ph))
    with rasterio.open(unw) as r:
        assert r.driver == "ISCE" and r.count == 2 and r.dtypes == ("float32", "float32")
        assert r.transform.almost_equals(GRID) and r.crs.to_epsg() == 4326
        assert np.array_equal(r.read(1), amp) and np.array_equal(r.read(2), ph)
    with rasterio.open(flat) as r:
        assert r.driver == "ISCE" and r.count == 1 and r.dtypes == ("complex64",)
        assert r.transform.almost_equals(GRID)


# ------------------------------------------------------------------ make_wrapped_phase

def test_wrapped_phase_wraps_a_ramp_and_ignores_amplitude(tmp_path):
    """A phase ramp spanning almost three cycles comes out wrapped into
    [-pi, pi] (np.angle; both ends are the same phase), cell for cell equal
    to the ramp modulo 2*pi, whatever the amplitude (0.01 to 100). A genuine
    zero phase stays 0.0 and ISCE's 0+0j footprint fill becomes NaN."""
    ramp = np.linspace(-9.0, 9.0, NY * NX).reshape(NY, NX)
    # precondition: no cell within 1 mrad of an odd multiple of pi, where either end is right
    assert np.min(np.abs(np.abs(wrap(ramp)) - np.pi)) > 1e-3
    amp = np.geomspace(0.01, 100.0, NY * NX).reshape(NY, NX)
    cpx = amp * np.exp(1j * ramp)
    cpx[0, :] = 0                                   # outside the geocoded footprint
    cpx[3, 4] = 5.0 + 0.0j                          # zero phase, real signal
    src = write_flat(tmp_path, cpx)

    out = gp.make_wrapped_phase(tmp_path)

    assert out == tmp_path / "wrapped_phase.tif" and out.exists()
    assert_gis_ready_on_input_grid(out, src)
    ph = read(out)
    assert np.isnan(ph[0]).all()                    # no data stays nodata
    assert ph[3, 4] == 0.0                          # a zero phase is a measurement
    body = np.ones((NY, NX), bool)
    body[0] = False
    body[3, 4] = False
    assert np.isfinite(ph[1:]).all()
    assert np.all(np.abs(ph[1:]) <= np.float32(np.pi))
    assert np.allclose(ph[body], wrap(ramp)[body], atol=1e-5)
    # the ramp really did wrap: the signal cells span over two cycles, the output under one
    assert np.ptp(ramp[body]) > 4 * np.pi and np.nanmax(ph) - np.nanmin(ph) <= 2 * np.pi + 1e-5


def test_wrapped_phase_all_fill_and_nan_cells_are_nodata(tmp_path):
    """A pair whose wrapped product holds no signal at all (every cell 0+0j)
    still gets a raster, entirely NaN - never zeros, which would read as a
    flat, motionless interferogram. NaN complex cells are NaN too."""
    write_flat(tmp_path, np.zeros((NY, NX), np.complex64))
    out = gp.make_wrapped_phase(tmp_path)
    assert out is not None and np.isnan(read(out)).all()

    cpx = np.exp(1j * np.full((NY, NX), 0.5))
    cpx[1, 1] = complex(np.nan, np.nan)
    cpx[1, 2] = complex(np.nan, 0.0)
    write_flat(tmp_path, cpx)
    ph = read(gp.make_wrapped_phase(tmp_path))          # a rerun replaces the old raster
    assert np.isnan(ph[1, 1]) and np.isnan(ph[1, 2])
    assert np.isfinite(ph).sum() == NY * NX - 2
    assert np.allclose(ph[np.isfinite(ph)], 0.5, atol=1e-6)


def test_wrapped_phase_missing_input_is_skipped(tmp_path, caplog):
    """No filt_topophase.flat.geo: None, a warning naming the file, and no
    output written."""
    with caplog.at_level(logging.WARNING, logger="insar_pipeline.gis_products"):
        assert gp.make_wrapped_phase(tmp_path) is None
    assert "No wrapped interferogram" in caplog.text and "filt_topophase.flat.geo" in caplog.text
    assert not (tmp_path / "wrapped_phase.tif").exists()


# ------------------------------------------------------------------ make_los_displacement

def _unw_case():
    """Unwrapped phase over +-4 fringes with anchor cells of known displacement."""
    ph = np.linspace(-8 * np.pi, 8 * np.pi, NY * NX).reshape(NY, NX).astype(np.float32)
    ph[2, 1] = 2 * np.pi                    # one fringe of range increase
    ph[2, 2] = -2 * np.pi                   # one fringe of range decrease
    ph[2, 3] = 0.0                          # the reference: no change
    ph[2, 4] = 10 * np.pi                   # five fringes: must not be wrapped again
    amp = np.geomspace(0.05, 500.0, NY * NX).reshape(NY, NX).astype(np.float32)
    return amp, ph


def test_los_displacement_metres_sign_and_scale(tmp_path):
    """d = -phase * wavelength / (4 pi), in metres: one 2*pi fringe is half a
    wavelength (2.775 cm at C-band), range increase (positive phase, away
    from the satellite) is negative, range decrease positive, a zero phase
    is 0.0 (not nodata), and five fringes stay five fringes - the unwrapped
    phase is converted, never re-wrapped. Amplitude (band 1) does not enter
    the value."""
    amp, ph = _unw_case()
    src = write_unw(tmp_path, amp, ph)

    out = gp.make_los_displacement(tmp_path)

    assert out == tmp_path / "los_displacement_m.tif" and out.exists()
    assert_gis_ready_on_input_grid(out, src)
    d = read(out)
    half = gp.S1_WAVELENGTH / 2
    assert d[2, 1] == pytest.approx(-half, rel=1e-6)
    assert d[2, 2] == pytest.approx(+half, rel=1e-6)
    assert d[2, 3] == 0.0
    assert d[2, 4] == pytest.approx(-5 * half, rel=1e-6)
    assert d[2, 4] == pytest.approx(-0.13875, rel=1e-4)                 # metres, not cm or mm
    expected = -ph.astype(float) * gp.S1_WAVELENGTH / (4 * np.pi)
    assert np.allclose(d, expected, rtol=1e-6, atol=1e-9)
    assert d.max() == pytest.approx(4 * half, rel=1e-6)                # -8 pi -> +4 half-wavelengths


def test_los_displacement_wavelength_scales_and_default_is_c_band(tmp_path):
    """The displacement scales with the wavelength passed in (an L-band
    product would otherwise read 4.3 times too small), and the default is
    Sentinel-1 C-band: c / 5.405 GHz to within 0.1% (S1_WAVELENGTH is that
    value rounded to 0.0555 m)."""
    amp, ph = _unw_case()
    write_unw(tmp_path, amp, ph)
    c_band = read(gp.make_los_displacement(tmp_path))
    explicit = read(gp.make_los_displacement(tmp_path, wavelength=gp.S1_WAVELENGTH))
    l_band = read(gp.make_los_displacement(tmp_path, wavelength=L_BAND))
    assert np.array_equal(c_band, explicit)
    nz = ph != 0
    assert np.allclose(l_band[nz] / c_band[nz], L_BAND / gp.S1_WAVELENGTH, rtol=1e-5)
    assert l_band[2, 1] == pytest.approx(-L_BAND / 2, rel=1e-6)
    assert gp.S1_WAVELENGTH == pytest.approx(299_792_458.0 / 5.405e9, rel=1e-3)


def test_los_displacement_masks_no_amplitude_and_non_finite_phase(tmp_path):
    """NaN where there is no measurement: zero amplitude (outside the
    footprint, even if the phase band holds a finite value), and a NaN or
    infinite phase. Every other cell keeps its value - including zeros."""
    amp = np.ones((NY, NX), np.float32)
    ph = np.full((NY, NX), np.pi, np.float32)
    amp[0, :] = 0.0                          # footprint fill, phase finite
    ph[0, :3] = 0.0                          # ... and the usual 0 there too
    ph[3, 3] = np.nan
    ph[3, 4] = np.inf
    ph[3, 5] = -np.inf
    ph[4, 4] = 0.0                           # genuine zero
    write_unw(tmp_path, amp, ph)
    d = read(gp.make_los_displacement(tmp_path))
    assert np.isnan(d[0]).all()
    assert np.isnan(d[3, 3:6]).all()
    assert d[4, 4] == 0.0
    valid = np.isfinite(d)
    assert valid.sum() == NY * NX - NX - 3
    assert np.allclose(d[valid & (ph != 0)], -gp.S1_WAVELENGTH / 4, rtol=1e-6)


def test_los_displacement_all_nodata_is_all_nan(tmp_path):
    """An unwrapped product with no valid cell (amplitude 0 everywhere, as
    ISCE fills it) still gets a raster, entirely NaN: never a field of zeros
    that would read as 'no movement'."""
    write_unw(tmp_path, np.zeros((NY, NX), np.float32), np.zeros((NY, NX), np.float32))
    out = gp.make_los_displacement(tmp_path)
    assert out is not None and np.isnan(read(out)).all()


def test_los_displacement_missing_input_is_skipped(tmp_path, caplog):
    """No filt_topophase.unw.geo: None, a warning naming the file, and no
    output written."""
    with caplog.at_level(logging.WARNING, logger="insar_pipeline.gis_products"):
        assert gp.make_los_displacement(tmp_path) is None
    assert "No unwrapped interferogram" in caplog.text and "filt_topophase.unw.geo" in caplog.text
    assert not (tmp_path / "los_displacement_m.tif").exists()


# ------------------------------------------------------------------ make_gis_products

def _consistent_pair(pair_dir):
    """A pair whose wrapped interferogram is its unwrapped phase modulo 2*pi
    (as topsApp's are), first line outside the footprint in both."""
    unw = np.linspace(-3 * np.pi, 5 * np.pi, NY * NX).reshape(NY, NX)
    amp = np.full((NY, NX), 3.0)
    amp[0] = 0.0
    cpx = amp * np.exp(1j * unw)
    unw[0] = 0.0
    merged = pair_dir / "merged"
    return write_flat(merged, cpx), write_unw(merged, amp, unw), unw


def test_gis_products_writes_both_rasters_that_agree(tmp_path):
    """One call writes wrapped_phase.tif and los_displacement_m.tif into
    <pair>/merged, both on the input grid with the same footprint, and the
    two agree: converting the displacement back to phase and wrapping it
    gives the wrapped raster - so the two cannot carry opposite signs."""
    pair = tmp_path / "20200101_20200113"
    flat, unw_path, unw = _consistent_pair(pair)

    assert gp.make_gis_products(pair) is None

    wrapped_tif = pair / "merged" / "wrapped_phase.tif"
    los_tif = pair / "merged" / "los_displacement_m.tif"
    assert_gis_ready_on_input_grid(wrapped_tif, flat)
    assert_gis_ready_on_input_grid(los_tif, unw_path)
    w, d = read(wrapped_tif), read(los_tif)
    assert np.array_equal(np.isnan(w), np.isnan(d))
    assert np.isnan(w[0]).all() and np.isfinite(w[1:]).all()
    back = -4 * np.pi * d[1:].astype(float) / gp.S1_WAVELENGTH
    assert np.allclose(back, unw[1:], atol=1e-4)
    assert np.all(np.abs(np.angle(np.exp(1j * (wrap(back) - w[1:])))) < 1e-4)


def test_gis_products_with_no_inputs_writes_nothing_and_does_not_raise(tmp_path, caplog):
    """A pair directory with no merged/ (topsApp never got there): both
    rasters are skipped with a warning each, nothing is created, and the
    stack carries on (no exception)."""
    pair = tmp_path / "20200101_20200113"
    pair.mkdir()
    with caplog.at_level(logging.WARNING, logger="insar_pipeline.gis_products"):
        gp.make_gis_products(pair)
    assert "No wrapped interferogram" in caplog.text and "No unwrapped interferogram" in caplog.text
    assert list(pair.iterdir()) == []


def test_gis_products_missing_wrapped_still_writes_displacement(tmp_path):
    """Only the unwrapped product exists (the wrapped one was not in the
    geocode list, or was cleaned up): the displacement raster is written."""
    pair = tmp_path / "20200101_20200113"
    amp, ph = _unw_case()
    write_unw(pair / "merged", amp, ph)
    gp.make_gis_products(pair)
    assert not (pair / "merged" / "wrapped_phase.tif").exists()
    assert np.isfinite(read(pair / "merged" / "los_displacement_m.tif")).all()


@pytest.mark.parametrize("broken", ["filt_topophase.flat.geo", "filt_topophase.unw.geo"])
def test_one_unreadable_input_does_not_cost_the_other_raster(tmp_path, caplog, broken):
    """An input GDAL cannot open - the raw binary without the .xml sidecar
    topsApp writes last, as when it is killed in the geocode step - is
    logged, not raised, and the OTHER raster is still written. Until 16 Oct
    2026 both conversions shared one try, so an unreadable wrapped product
    stopped the displacement raster too (fails on f3a0a0a for the .flat.geo
    case)."""
    pair = tmp_path / "20200101_20200113"
    _consistent_pair(pair)
    (pair / "merged" / (broken + ".xml")).unlink()
    with caplog.at_level(logging.WARNING, logger="insar_pipeline.gis_products"):
        gp.make_gis_products(pair)                      # must not raise
    assert "Could not write GIS products for 20200101_20200113" in caplog.text
    wrapped_tif = pair / "merged" / "wrapped_phase.tif"
    los_tif = pair / "merged" / "los_displacement_m.tif"
    if broken == "filt_topophase.flat.geo":
        assert not wrapped_tif.exists()
        assert los_tif.exists() and np.isfinite(read(los_tif)[1:]).all()
    else:
        assert not los_tif.exists()
        assert wrapped_tif.exists() and np.isfinite(read(wrapped_tif)[1:]).all()
