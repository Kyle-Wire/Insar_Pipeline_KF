"""Combined view (combine.py, 28 Oct 2026): an ascending and a descending
project of the same ground into vertical and east-west movement.

Synthetic sources only (combine.Source built directly, no MintPy run):
  * pure east-west movement reads as no vertical movement and possible
    east-west movement, though each direction alone reads it as vertical;
  * pure sinking reads as vertical movement;
  * the decision table row by row, with the one-direction note;
  * east-west in the client record carries no threshold, limit or
    uncertainty (indicative until the re-test passes);
  * an area covered from one direction only keeps that direction's reading;
  * different report areas or record windows are refused;
  * a combined result goes out of date when a source's files change;
  * run() writes only into its own folder: the sources are never changed;
  * the project list and the source reports' pointer find it.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from insar_pipeline import combine

DX, DY = 0.000173, -0.000135
NX = NY = 160
X0, Y0 = -104.90, 38.98
ASC = (-0.60, -0.12, 0.79)            # LOS unit vector toward the satellite (east, north, up)
DESC = (0.58, -0.13, 0.80)


def _attrs(x0=X0, y0=Y0, nx=NX, ny=NY):
    return {"X_FIRST": repr(x0), "Y_FIRST": repr(y0), "X_STEP": repr(DX), "Y_STEP": repr(DY),
            "WIDTH": str(nx), "LENGTH": str(ny), "X_UNIT": "degrees", "Y_UNIT": "degrees", "EPSG": "4326"}


def _box(r0, r1, c0, c1, x0=X0, y0=Y0):
    """An area spec (S,W,N,E) over rows r0:r1, cols c0:c1 of the base grid."""
    n, s = y0 + r0 * DY, y0 + r1 * DY
    w, e = x0 + c0 * DX, x0 + c1 * DX
    return f"{s + 1e-7},{w + 1e-7},{n - 1e-7},{e - 1e-7}"


AREA = ("Patch", _box(70, 80, 70, 80))
AREAS = [("Full Satellite Imaging", "full"), AREA]


def _source(label, vec, up_mm, east_mm, *, seed, rep_areas=None, dates=("20160801", "20261009"),
            x0=X0, nx=NX, areas=AREAS, project_dir="/tmp/x"):
    """A direction whose relative LOS field is noise + the planted Up/East."""
    rng = np.random.default_rng(seed)
    shape = (NY, nx)
    e, n, u = (np.full(shape, c, np.float32) for c in vec)
    rel = (rng.normal(0, 1.0, shape) + e * east_mm + u * up_mm).astype(np.float32)
    good = np.ones(shape, bool)
    far = np.ones(shape, bool)
    cfg = SimpleNamespace(project_dir=project_dir, control_ground_label="farmland", track=1,
                          orbit_direction=label)
    rep = {"dates": list(dates), "anchored": False, "areas": rep_areas or []}
    return combine.Source(label=label, cfg=cfg, rep=rep, attrs=_attrs(x0=x0, nx=nx), rel=rel,
                          tc=np.full(shape, 0.8, np.float32), good=good, far_wide=far,
                          los_e=e, los_n=n, los_u=u, areas=list(areas))


def _patch(mm):
    f = np.zeros((NY, NX), np.float32)
    f[70:80, 70:80] = mm
    return f


BBOX = [X0, Y0 + NY * DY, X0 + NX * DX, Y0]


def _area(res, name="Patch"):
    return next(a for a in res["areas"] if a["name"] == name)


# ------------------------------------------------------------------ the combination

def test_pure_east_west_is_not_read_as_vertical():
    singles = [{"name": "Patch", "detected": True, "excess_vertical_mm_yr": -8.0, "reading": "Movement detected"},
               {"name": "Patch", "detected": True, "excess_vertical_mm_yr": 8.0, "reading": "Movement detected"}]
    a = _source("ascending", ASC, 0, _patch(10.0), seed=1, rep_areas=[singles[0]])
    d = _source("descending", DESC, 0, _patch(10.0), seed=2, rep_areas=[singles[1]])
    res = combine.analyse([a, d], AREAS, BBOX)
    r = _area(res)
    assert r["combined"] and r["vertical"]["testable"]
    assert not r["vertical"]["detected"]
    assert r["east_west"]["stands_out_indicative"] is True
    assert r["east_west"]["movement_mm_yr"] == pytest.approx(10.0, abs=1.0)
    assert r["reading"] == "No vertical movement detected; " + combine.EAST_PHRASE
    # each direction alone read it as vertical, with opposite signs: east-west explains both
    assert r["note"].startswith("Seen from the ascending and descending direction alone")
    assert "explained by east-west movement" in r["note"]


def test_pure_sinking_is_vertical_movement():
    a = _source("ascending", ASC, _patch(-12.0), 0, seed=3)
    d = _source("descending", DESC, _patch(-12.0), 0, seed=4)
    r = _area(combine.analyse([a, d], AREAS, BBOX))
    assert r["vertical"]["detected"] and r["reading"] == "Vertical movement (sinking)"
    assert r["vertical"]["relative_vertical_mm_yr"] == pytest.approx(-12.0, abs=1.0)
    assert r["east_west"]["movement_mm_yr"] == pytest.approx(0.0, abs=1.5)


def test_east_west_client_record_has_no_threshold_or_limit():
    a = _source("ascending", ASC, 0, _patch(10.0), seed=5)
    d = _source("descending", DESC, 0, _patch(10.0), seed=6)
    res = combine.analyse([a, d], AREAS, BBOX)
    ew = _area(res)["east_west"]
    assert set(ew) == {"movement_mm_yr", "movement_in_yr", "stands_out_indicative", "status", "sign"}
    assert ew["status"] == "indicative"
    # the full test is kept for the technical file
    tech = next(x for x in res["east_rows"] if x["name"] == "Patch")["east"]
    assert tech["threshold_95_mm_yr"] > 0 and "detection_limit_sink_mm_yr" in tech


def test_area_seen_from_one_direction_keeps_its_reading():
    # the descending grid ends at column 100; an area beyond it is ascending-only
    far_area = ("East edge", _box(70, 80, 130, 140))
    areas = AREAS + [far_area]
    singles = [{"name": "East edge", "detected": False, "reading": "Within normal ground variation"}]
    a = _source("ascending", ASC, 0, 0, seed=7, areas=areas, rep_areas=singles)
    d = _source("descending", DESC, 0, 0, seed=8, areas=areas, nx=100)
    r = _area(combine.analyse([a, d], areas, BBOX), "East edge")
    assert not r["combined"] and r["reading"] == "Measured from one direction only"
    assert "ascending" in r["sentence"] and "Within normal ground variation" in r["sentence"]


@pytest.mark.parametrize("up,east,reading", [
    ({"testable": True, "distinguishable_from_farmland": True, "excess_mm_yr": -6.0},
     {"testable": True, "distinguishable_from_farmland": False, "excess_mm_yr": 0.1}, "Vertical movement (sinking)"),
    ({"testable": True, "distinguishable_from_farmland": True, "excess_mm_yr": 5.0},
     {"testable": True, "distinguishable_from_farmland": True, "excess_mm_yr": 3.0}, "Vertical movement (rising)"),
    ({"testable": True, "distinguishable_from_farmland": False, "excess_mm_yr": -1.0},
     {"testable": True, "distinguishable_from_farmland": True, "excess_mm_yr": 3.0},
     "No vertical movement detected; " + combine.EAST_PHRASE),
    ({"testable": True, "distinguishable_from_farmland": False, "excess_mm_yr": -1.0},
     {"testable": True, "distinguishable_from_farmland": False, "excess_mm_yr": 0.3}, "No movement detected"),
    ({"testable": False, "not_testable_reason": "Too little ordinary farmland around this area."},
     {"testable": False}, "Not testable"),
])
def test_decision_table(up, east, reading):
    d = combine.decision(up, east, [{"direction": "ascending"}, {"direction": "descending"}], [-0.6, 0.6], -1.0)
    assert d["reading"] == reading
    if reading == "Not testable":
        assert "farmland" not in combine._words(up["not_testable_reason"])


def test_one_direction_reading_not_explained_by_east_west():
    up = {"testable": True, "distinguishable_from_farmland": False, "excess_mm_yr": -0.5}
    east = {"testable": True, "distinguishable_from_farmland": False, "excess_mm_yr": -2.0}   # westward
    singles = [{"direction": "ascending", "detected": True, "excess_vertical_mm_yr": -6.0},
               {"direction": "descending", "detected": False, "excess_vertical_mm_yr": 0.0}]
    # westward movement reads as rising from the ascending side, not sinking
    d = combine.decision(up, east, singles, [-0.6, 0.6], None)
    assert d["note"] == ("Seen from the ascending direction alone; not confirmed by the two directions together.")


def test_different_areas_or_windows_are_refused():
    a = _source("ascending", ASC, 0, 0, seed=9)
    d = _source("descending", DESC, 0, 0, seed=10, areas=[AREAS[0], ("Patch", _box(10, 20, 10, 20))])
    with pytest.raises(combine.CombinedConfigError, match="different outline"):
        combine.check_areas(a, d)
    d2 = _source("descending", DESC, 0, 0, seed=10, dates=("20170101", "20261009"))
    with pytest.raises(combine.CombinedConfigError, match="different years"):
        combine.check_window(a, d2)
    d3 = _source("descending", DESC, 0, 0, seed=10, dates=("20160725", "20261008"))
    assert combine.check_window(a, d3)["common"] == ["20160801", "20261008"]


# ------------------------------------------------------------------ projects on disk

def _pipeline_cfg(tmp, name, direction):
    proj = tmp / name
    (proj / "mintpy" / "geo").mkdir(parents=True)
    (proj / "deliverables").mkdir()
    for f in combine.SOURCE_FILES:
        (proj / "mintpy" / "geo" / f"geo_{f}").write_bytes(f"{name} {f}".encode())
    (proj / "deliverables" / "client_report.json").write_text(json.dumps({"dates": ["20160801", "20261009"]}))
    p = tmp / "configs" / f"{name}.yaml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(f"project_dir: {proj}\nbbox: [{BBOX[0]}, {BBOX[1]}, {BBOX[2]}, {BBOX[3]}]\n"
                 f"start_date: 2016-08-01\nend_date: 2026-10-09\norbit_direction: {direction}\ntrack: 1\n")
    return p


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.setenv("INSAR_PROJECTS", str(tmp_path))
    a = _pipeline_cfg(tmp_path, "site", "ascending")
    d = _pipeline_cfg(tmp_path, "site_desc", "descending")
    c = combine.write_config(a, d)
    return tmp_path, a, d, c


def _tree_hash(folder: Path) -> dict:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_write_config_and_load(pair):
    tmp, a, d, c = pair
    assert c == tmp / "configs" / "site_combined.yaml"
    assert combine.is_combined_config(c) and not combine.is_combined_config(a)
    cc = combine.load(c)
    assert cc.project_dir == tmp / "site_combined" and cc.ascending == a.resolve()
    with pytest.raises(FileExistsError):
        combine.write_config(a, d)
    with pytest.raises(combine.CombinedConfigError, match="ascending"):
        combine.write_config(d, a, out=tmp / "x.yaml")


def test_run_writes_only_its_own_folder_and_goes_stale(pair, monkeypatch):
    tmp, a, d, c = pair
    before = {n: _tree_hash(tmp / n) for n in ("site", "site_desc")}

    def fake_load_source(cfg, label):
        vec = ASC if label == "ascending" else DESC
        s = _source(label, vec, 0, _patch(10.0), seed=11 if label == "ascending" else 12,
                    project_dir=str(cfg.project_dir))
        s.cfg = cfg
        return s

    monkeypatch.setattr(combine, "load_source", fake_load_source)
    assert combine.stale_reason(c) == "not run yet"
    out = combine.run(c)
    assert out == tmp / "site_combined" / "deliverables" / "combined_report.json"
    rec = json.loads(out.read_text())
    assert [s["direction"] for s in rec["sources"]] == ["ascending", "descending"]
    assert rec["comparison_ground"]["words"].startswith("the surrounding ground")
    names = {Path(p).name for p in rec["rasters"]}
    assert {"vertical_movement_combined_in_per_yr.tif", "east_west_movement_INDICATIVE_in_per_yr.tif"} <= names
    assert (tmp / "site_combined" / "deliverables" / "technical" / combine.EAST_TEST).exists()
    # the sources are never changed
    assert {n: _tree_hash(tmp / n) for n in ("site", "site_desc")} == before
    assert combine.stale_reason(c) is None
    # a rebuilt source makes the combined result out of date
    (tmp / "site_desc" / "mintpy" / "geo" / "geo_velocity.h5").write_bytes(b"rebuilt")
    why = combine.stale_reason(c)
    assert why.startswith("the descending project has changed since") and "velocity.h5" in why


def test_project_list_and_source_pointer(pair, monkeypatch):
    tmp, a, d, c = pair
    from insar_pipeline import runs
    from insar_pipeline.config import load_config
    monkeypatch.setattr(runs, "known_configs", lambda: [a.resolve(), d.resolve(), c.resolve()])
    p = runs.project_summary(c)
    assert (p.backend, p.state, p.error) == ("combined", "idle", "")
    ptr = combine.pointer_for(load_config(str(a)))
    assert ptr["direction"] == "ascending" and ptr["status"] == "not run"
    assert "also measured from the descending pass" in ptr["sentence"]
    other = _pipeline_cfg(tmp, "elsewhere", "ascending")
    assert combine.pointer_for(load_config(str(other))) is None


def _write_wc(path, classes):
    """A WorldCover-coded raster exactly on the base grid."""
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", width=classes.shape[1], height=classes.shape[0], count=1,
                       dtype="uint8", crs="EPSG:4326", nodata=0, transform=from_origin(X0, Y0, DX, -DY)) as d:
        d.write(classes.astype(np.uint8), 1)
    return str(path)


def test_ground_words_follow_the_single_project_rule(tmp_path):
    """The combined records name the comparison ground as the single-project
    report does (landcover.comparison_ground, 28 Oct 2026): farmland only
    where cropland dominates; a label someone set wins; no map, plain ground."""
    from insar_pipeline import landcover
    landcover._GRID_CACHE.clear()
    far = np.ones((NY, NX), bool)
    town = np.full((NY, NX), 30)
    town[:, : int(0.55 * NX)] = 50                         # the first combined site's mix: built-up and grassland
    crops = np.full((NY, NX), 40)
    crops[:, :20] = 30
    nora = SimpleNamespace(landcover_raster=_write_wc(tmp_path / "town.tif", town), control_ground_label="farmland")
    words, g = combine.ground_label(nora, _attrs(), far)
    assert words == "the surrounding ground (mostly built-up and grassland)"
    assert g["source"] == "measured" and set(g["cover_percent"]) == {"built-up", "grassland"}
    fields = SimpleNamespace(landcover_raster=_write_wc(tmp_path / "crops.tif", crops), control_ground_label="")
    assert combine.ground_label(fields, _attrs(), far)[0] == "the surrounding farmland"
    own = SimpleNamespace(landcover_raster=nora.landcover_raster, control_ground_label="open rangeland")
    words, g = combine.ground_label(own, _attrs(), far)
    assert words == "the surrounding open rangeland" and g["source"] == "config"
    bare = SimpleNamespace(landcover_raster="", control_ground_label="farmland")
    assert combine.ground_label(bare, _attrs(), far) == (
        "the surrounding ground", {"label": "ground", "source": "unmeasured", "cover_percent": None, "scheme": None})
    landcover._GRID_CACHE.clear()
