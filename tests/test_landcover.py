"""Land cover labels (15 Oct 2026): wooded cells out of the comparison ground,
a wooded share per feature/area, no change without a map."""
import numpy as np
import pytest

from insar_pipeline import landcover, significance as sg

GEO = {"X_FIRST": -87.60, "Y_FIRST": 38.50, "X_STEP": 0.001, "Y_STEP": -0.001, "LENGTH": 100, "WIDTH": 100}


class _Cfg:
    deramp_exclude_geojson = ""
    acceptance_far_field_buffer_m = 2000.0
    landcover_raster = ""

    def __init__(self, **kw):
        self.bbox = (GEO["X_FIRST"] + 0.010, GEO["Y_FIRST"] - 0.090, GEO["X_FIRST"] + 0.090, GEO["Y_FIRST"] - 0.010)
        self.__dict__.update(kw)


def _write_wc(path, classes):
    """A WorldCover-coded raster exactly on the GEO grid."""
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", width=classes.shape[1], height=classes.shape[0], count=1,
                       dtype="uint8", crs="EPSG:4326", nodata=0,
                       transform=from_origin(GEO["X_FIRST"], GEO["Y_FIRST"], GEO["X_STEP"], -GEO["Y_STEP"])) as d:
        d.write(classes.astype(np.uint8), 1)
        d.update_tags(scheme="worldcover")


@pytest.fixture(autouse=True)
def _clear_cache():
    landcover._GRID_CACHE.clear()
    yield
    landcover._GRID_CACHE.clear()


def test_tiles_named_by_lower_left_corner():
    assert landcover._tiles(-90.55, 41.04, -90.22, 41.29) == ["N39W093"]
    assert landcover._tiles(-90.5, 41.5, -89.5, 42.5) == ["N39W093", "N39W090", "N42W093", "N42W090"]
    assert landcover._tiles(150.4, -34.3, 150.8, -34.0) == ["S36E150"]


def test_no_map_changes_nothing(tmp_path):
    good = np.ones((100, 100), bool)
    base, meta = sg.far_field_mask(_Cfg(), good, GEO)
    same, meta2 = sg.far_field_mask(_Cfg(project_dir=str(tmp_path)), good, GEO)
    assert (base == same).all() and meta["wooded"] is None and meta2["wooded"] is None


def test_trees_leave_the_comparison_ground(tmp_path):
    cls = np.full((100, 100), 40)                  # cropland
    cls[20:40, 20:40] = 10                         # a wood
    _write_wc(tmp_path / "lc.tif", cls)
    good = np.ones((100, 100), bool)
    base, _ = sg.far_field_mask(_Cfg(), good, GEO)
    far, meta = sg.far_field_mask(_Cfg(landcover_raster=str(tmp_path / "lc.tif")), good, GEO)
    assert meta["wooded"]["applied"] and meta["wooded"]["wooded_cells"] == 400
    assert not far[20:40, 20:40].any() and (far | (cls == 10)).sum() == base.sum()
    off, m_off = sg.far_field_mask(_Cfg(landcover_raster=str(tmp_path / "lc.tif"), landcover_label_wooded=False),
                                   good, GEO)
    assert (off == base).all() and m_off["wooded"] is None


def test_a_wooded_region_keeps_its_trees(tmp_path):
    cls = np.full((100, 100), 10)
    cls[:, :20] = 40                               # mostly forest: removing it would empty the ground
    _write_wc(tmp_path / "lc.tif", cls)
    good = np.ones((100, 100), bool)
    base, _ = sg.far_field_mask(_Cfg(), good, GEO)
    far, meta = sg.far_field_mask(_Cfg(landcover_raster=str(tmp_path / "lc.tif")), good, GEO)
    assert meta["wooded"]["applied"] is False and "kept" in meta["wooded"]["reason"]
    assert (far == base).all()


def test_share_and_sentence(tmp_path):
    cls = np.full((100, 100), 40)
    cls[0:50, 0:30] = 10
    _write_wc(tmp_path / "lc.tif", cls)
    cfg = _Cfg(landcover_raster=str(tmp_path / "lc.tif"))
    m = np.zeros((100, 100), bool); m[0:50, 0:50] = True
    lc = landcover.share_of(cfg, GEO, m)
    assert lc["wooded_percent"] == 60.0 and lc["main_cover"] == "trees" and lc["cover_percent"]["cropland"] == 40.0
    assert "canopy" in landcover.wooded_sentence(lc)
    m2 = np.zeros((100, 100), bool); m2[60:90, 60:90] = True
    lc2 = landcover.share_of(cfg, GEO, m2)
    assert lc2["wooded_percent"] == 0.0 and "mostly cropland (100%)" in landcover.wooded_sentence(lc2)
    assert landcover.wooded_sentence(None) == ""


def test_nlcd_codes_recognised(tmp_path):
    cls = np.full((100, 100), 82)                  # NLCD cultivated crops
    cls[:10, :10] = 41                             # deciduous forest
    import rasterio
    from rasterio.transform import from_origin
    p = tmp_path / "nlcd.tif"
    with rasterio.open(p, "w", driver="GTiff", width=100, height=100, count=1, dtype="uint8", crs="EPSG:4326",
                       nodata=0, transform=from_origin(GEO["X_FIRST"], GEO["Y_FIRST"], 0.001, 0.001)) as d:
        d.write(cls.astype(np.uint8), 1)
    wood = landcover.wooded_on_grid(_Cfg(landcover_raster=str(p)), GEO, (100, 100))
    assert wood.sum() == 100 and wood[:10, :10].all()


def test_tree_layer_stripes():
    from insar_pipeline import client_kmz
    wood = np.zeros((8, 8), bool); wood[:4] = True
    rgba = client_kmz.tree_rgba(wood)
    assert (rgba[4:, :, 3] == 0).all() and 0 < (rgba[:4, :, 3] > 0).sum() < 32
    xml = client_kmz.tree_folder([(b"", (0, 0, 1, 1))])
    assert "layers/trees_0.png" in xml and "<visibility>0</visibility>" in xml


# ------------------------------------------------------------------ phase-linking skip
def _points():
    lat = GEO["Y_FIRST"] + (np.arange(100)[:, None] + 0.5) * GEO["Y_STEP"] + np.zeros((100, 100))
    lon = GEO["X_FIRST"] + (np.arange(100)[None, :] + 0.5) * GEO["X_STEP"] + np.zeros((100, 100))
    return lat, lon


def test_skip_water_keeps_the_shore_and_woods_by_default(tmp_path):
    cls = np.full((100, 100), 40)
    cls[10:40, 10:40] = 80                          # a lake
    cls[60:95, 60:95] = 10                          # a wood
    cls[50:52, :] = 10                              # a creek-line strip of trees
    _write_wc(tmp_path / "lc.tif", cls)
    cfg = _Cfg(landcover_raster=str(tmp_path / "lc.tif"))
    lat, lon = _points()
    skip, rec = landcover.skip_on_points(cfg, lat, lon)            # default "water"
    assert rec["mode"] == "water" and skip[25, 25] and not skip[10, 10] and not skip[12, 25]   # shore kept
    assert not skip[60:95, 60:95].any() and not skip[50:52].any()
    skip2, rec2 = landcover.skip_on_points(cfg, lat, lon, "water_and_dense_woods")
    assert skip2[77, 77] and not skip2[61, 77] and not skip2[50:52].any()                      # wood edge, strip kept
    assert rec2["dense_woods_percent"] > 0 and rec2["water_percent"] == rec["water_percent"]
    assert landcover.skip_on_points(cfg, lat, lon, "none") is None
    assert landcover.skip_on_points(_Cfg(), lat, lon) is None                                  # no map


def test_wetland_never_skipped(tmp_path):
    cls = np.full((100, 100), 90)                   # herbaceous wetland: dry part of the year
    _write_wc(tmp_path / "lc.tif", cls)
    lat, lon = _points()
    skip, _ = landcover.skip_on_points(_Cfg(landcover_raster=str(tmp_path / "lc.tif")), lat, lon,
                                       "water_and_dense_woods")
    assert not skip.any()


def test_skip_mask_only_before_linking(tmp_path, monkeypatch):
    import h5py
    from insar_pipeline import miaplpy_backend as mb
    cls = np.full((100, 100), 40); cls[10:40, 10:40] = 80
    _write_wc(tmp_path / "lc.tif", cls)
    work = tmp_path / "miaplpy"; (work / "inputs").mkdir(parents=True)
    lat, lon = _points()
    with h5py.File(work / "inputs" / "geometryRadar.h5", "w") as g:
        g["latitude"], g["longitude"] = lat, lon
    cfg = _Cfg(landcover_raster=str(tmp_path / "lc.tif"), project_dir=tmp_path, miaplpy_skip="water")
    monkeypatch.setattr(mb, "work_dir", lambda c: work)
    rec = mb.write_skip_mask(cfg)
    assert rec["mode"] == "water" and rec["file"] and mb.skip_mask_path(cfg).exists()
    keep = mb.load_skip_keep(cfg, (100, 100))
    assert not keep[25, 25] and keep[5, 5]
    assert mb.load_skip_keep(cfg, (50, 50)) is None                # another grid: not used
    # linking started: the mask it began with stays, whatever the config says now
    flag = work / "inverted" / "PATCHES" / "PATCH_0000"; flag.mkdir(parents=True); (flag / "flag.npy").write_bytes(b"")
    cfg.miaplpy_skip = "none"
    assert mb.write_skip_mask(cfg)["mode"] == "as linked" and mb.skip_mask_path(cfg).exists()
    # not linked yet and switched off: no mask
    import shutil; shutil.rmtree(work / "inverted")
    assert mb.write_skip_mask(cfg) == {"mode": "none"} and not mb.skip_mask_path(cfg).exists()


def test_template_names_the_skip_mask():
    import inspect
    from insar_pipeline import miaplpy_backend as mb
    assert "miaplpy.inversion.mask" in inspect.getsource(mb.write_template)


def test_products_drop_wooded_cells(tmp_path):
    """mask_wooded (21 Oct 2026): woods leave the delivered products' mask;
    off, or without a map, nothing changes."""
    cls = np.full((100, 100), 40)
    cls[20:40, 20:40] = 10
    _write_wc(tmp_path / "lc.tif", cls)
    good = np.ones((100, 100), bool); good[0, 0] = False
    out, meta = landcover.drop_wooded(_Cfg(landcover_raster=str(tmp_path / "lc.tif")), good, GEO)
    assert meta["applied"] and meta["wooded_cells_removed"] == 400
    assert not out[20:40, 20:40].any() and out.sum() == good.sum() - 400
    off, m_off = landcover.drop_wooded(_Cfg(landcover_raster=str(tmp_path / "lc.tif"), mask_wooded=False), good, GEO)
    assert (off == good).all() and not m_off["applied"]
    none, m_none = landcover.drop_wooded(_Cfg(), good, GEO)
    assert (none == good).all() and m_none["reason"] == "no land-cover map"


def test_masked_woods_worded_as_left_out(tmp_path):
    cls = np.full((100, 100), 40)
    cls[0:50, 0:30] = 10
    _write_wc(tmp_path / "lc.tif", cls)
    m = np.zeros((100, 100), bool); m[0:50, 0:50] = True
    on = landcover.share_of(_Cfg(landcover_raster=str(tmp_path / "lc.tif")), GEO, m)
    assert on["wooded_masked"] and "left out" in landcover.wooded_sentence(on)
    off = landcover.share_of(_Cfg(landcover_raster=str(tmp_path / "lc.tif"), mask_wooded=False), GEO, m)
    assert "less certain" in landcover.wooded_sentence(off)
