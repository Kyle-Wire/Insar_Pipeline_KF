"""7 Oct 2026 trust review follow-up, group records (quality_assessment.py,
crosscheck.py, deliverables.py, gnss_tie.py, release.py): decisions F7(b) and
F7(d). Each test fails on the code in backups/pre_followup_20261007/.

F7(b)  gnss_tie records whether NGL's steps file was read (steps_file_read,
       top level and per station); when the stations a report rests on were
       not screened for equipment changes and earthquakes - the file could
       not be read, or the tie predates the screening - release.py lists a
       soft [read] item, 'gnss_steps_not_screened'. FINAL / DRAFT does not
       change.
F7(d)  the QA, the DISP cross-check, the products' per-cell null and the
       GNSS inside-station check build their farmland with E28's exclusion
       (significance.far_field_mask(exclude=...) with the report areas other
       than the whole study area, as the client report does), so their
       technical records read against the report's farmland.
"""
import json
import types

import numpy as np
import pytest

import test_fix_L1 as L1T                                        # the synthetic GNSS tie project
import test_fix_L7b as L7b                                       # the release gate's synthetic project
import test_fix_L8 as L8T                                        # the QA's synthetic geocoded project
from insar_pipeline import significance as sg

# A report area on ordinary farmland of the L8 grid (away from the mine
# square and its 500 m buffer, and from the reference pixel), sinking 20 mm/yr
# LOS: the way Site B's project area sat in its own farmland.
SITE = "Site:38.432,-87.775,38.466,-87.715"


def _site_mask(shape, attrs):
    s_, w, n, e = (float(v) for v in SITE.split(":", 1)[1].split(","))
    return sg.aoi_mask((w, s_, e, n), attrs, shape)


def _l8_attrs():
    return {k: float(v) for k, v in L8T._attrs().items() if k in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP")}


def _moving_site_project(tmp_path):
    attrs = _l8_attrs()
    site = _site_mask((L8T.NY, L8T.NX), attrs)
    vel = np.random.default_rng(5).normal(0, 1.0, (L8T.NY, L8T.NX))
    vel[site] -= 20.0
    cfg = L8T.build(tmp_path, vel_mm=vel, report_areas=("Full Satellite Imaging:full", SITE))
    return cfg, site, attrs


@pytest.fixture(autouse=True)
def _no_osm_and_fresh_cache(monkeypatch):
    from insar_pipeline import deliverables as dl
    from insar_pipeline import stack_qc

    def boom(*a, **k):
        raise AssertionError("a test tried to fetch OSM water")
    monkeypatch.setattr(stack_qc, "fetch_osm_water", boom, raising=False)
    dl.clear_cache()
    yield
    dl.clear_cache()


# ---------------------------------------------------------------- F7(d): the one cut
def test_report_area_cut_is_the_reports_own_outlines(tmp_path):
    """The union of the non-full report areas' whole outlines, drawn as
    client_report._mask draws an area's footprint (box and geojson), with
    their names; the whole study area is never cut."""
    from insar_pipeline import client_report as cr
    from insar_pipeline import quality_assessment as qa
    gj = tmp_path / "zone.geojson"
    gj.write_text(json.dumps(L8T._fc(L8T._square(-87.79, 38.37, -87.76, 38.39))))
    cfg = types.SimpleNamespace(bbox=[-87.80, 38.362, -87.635, 38.47], report_areas=(
        "Full Satellite Imaging:full", SITE, f"Zone:{gj}"))
    attrs, shape = _l8_attrs(), (L8T.NY, L8T.NX)
    cut, names = qa.report_area_cut(cfg, attrs, shape)
    ctx = types.SimpleNamespace(sg=sg, good=np.ones(shape, bool), attrs=attrs, cfg=cfg)
    want = cr._mask(ctx, SITE.split(":", 1)[1])[2] | cr._mask(ctx, str(gj))[2]
    assert names == ["Site", "Zone"] and cut.dtype == bool and np.array_equal(cut, want)
    assert 0 < cut.sum() < cut.size
    only_full = types.SimpleNamespace(bbox=cfg.bbox, report_areas=("Full Satellite Imaging:full",))
    assert qa.report_area_cut(only_full, attrs, shape) == (None, [])


def test_qa_farmland_leaves_the_report_areas_out(tmp_path):
    """Before: the QA's far field held the sinking area, so the farmland it
    measured everything against read about -2.8 mm/yr where ordinary ground
    reads 0, and the area was compared with itself."""
    from insar_pipeline import quality_assessment as qa
    cfg, site, attrs = _moving_site_project(tmp_path)
    r = qa.evaluate(cfg)
    assert abs(r["far_field"]["mean_rate_mm_yr"]) < 0.5
    reg = r["regions"]
    assert reg["report_areas_cut"] == ["Site"]
    assert reg["report_area_cells_removed"]["far_field"] > 0 and reg["report_area_cells_removed"]["far_field_wide"] > 0
    assert "report areas other than the whole study area" in reg["far_field_definition"]
    # the same count far_field_mask(exclude=...) gives with the report's areas
    import h5py
    with h5py.File(cfg.mintpy_dir / "geo" / "geo_velocity.h5", "r") as f:
        vel = f["velocity"][:]
    with h5py.File(cfg.mintpy_dir / "geo" / "geo_maskTempCoh.h5", "r") as f:
        good = f["mask"][:].astype(bool) & np.isfinite(vel)
    full = {k: v for k, v in L8T._attrs().items()}
    far, _ = sg.far_field_mask(cfg, good, full, exclude=site)
    assert reg["far_field_px"] == int(far.sum())
    assert not (far & sg.euclidean_buffer(site, full, 450.0)).any()


def test_report_areas_covering_the_farmland_are_not_cut_and_hold_the_report(tmp_path):
    """An area as large as the study area would leave no comparison ground
    (the QA crashed on the empty farmland when it was cut regardless): it is
    not cut, the record says why, and the far field reads thin, which holds
    the report - nothing is compared with nothing."""
    from insar_pipeline import quality_assessment as qa
    cfg = L8T.build(tmp_path, report_areas=("Full Satellite Imaging:full", "Huge:38.363,-87.799,38.469,-87.636"))
    r = qa.evaluate(cfg)
    reg = r["regions"]
    assert reg["report_areas_cut"] == [] and reg["report_areas_not_cut"] == ["Huge"]
    # the QA's prose says its measured ground: "ground" without a land-cover map (28 Oct 2026)
    assert "cover nearly all the ground" in reg["report_areas_not_cut_reason"]
    assert r["far_field"]["thin"] is True and "Huge" in r["far_field"]["thin_reason"]
    assert any("too little ordinary ground" in h for h in r["send_rule"]["hold_reasons"])


def test_disp_crosscheck_fields_leave_the_report_areas_out(tmp_path):
    """crosscheck.load_fields (both products' class centres and nulls): the
    far field and the wide far field without the area plus 500 m, so the
    class centre is ordinary ground's and the area reads its full -20."""
    from insar_pipeline import crosscheck as cc
    cfg, site, _ = _moving_site_project(tmp_path)
    F = cc.load_fields(cfg)
    grown = sg.euclidean_buffer(site, F["attrs"], 450.0)
    assert not (F["far"] & grown).any() and not (F["far_wide"] & grown).any()
    cg = F["comparison_ground"]
    assert F["far"].any() and cg["report_areas_cut"] == ["Site"] and cg["applied"] is True
    assert cg["cells_removed"]["far_field"] > 0 and cg["cells_removed"]["far_field_wide"] > 0
    k = sg.coherence_class(np.array([0.75]))[0]
    assert abs(F["class_means"][int(k)]) < 0.5
    assert np.nanmean(F["r_cm"][site & F["good"]]) == pytest.approx(-20.0, abs=0.6)


def test_products_cell_null_is_measured_without_the_report_areas(tmp_path):
    """deliverables._empirical_context: the per-cell null the report quotes
    and the rasters divide by is measured on farmland without the area, and
    uncertainty_model.json says so; the area's relative rate is its own
    -20, not -20 less its share of the farmland."""
    from insar_pipeline import deliverables as dl
    cfg, site, _ = _moving_site_project(tmp_path)
    far, rel, emp, model, null = dl._empirical_context(cfg)
    attrs = {k: float(v) for k, v in L8T._attrs().items() if k in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP")}
    assert not (far & sg.euclidean_buffer(site, attrs, 450.0)).any() and far.any()
    good = dl.load_mask(cfg)
    assert np.nanmean(rel[site & good]) == pytest.approx(-20.0, abs=0.6)
    cg = model["comparison_ground"]
    assert cg["report_areas_cut"] == ["Site"] and cg["applied"] is True and cg["cells_removed"]["far_field"] > 0
    assert cg["far_cells"] == int(far.sum()) and cg["buffer_m"] == 500.0
    on_disk = json.loads((cfg.project_dir / "deliverables" / "uncertainty_model.json").read_text())
    assert on_disk["comparison_ground"]["report_areas_cut"] == ["Site"]
    assert "report areas" in on_disk["comparison_ground"]["definition"]


@pytest.fixture
def tie_project(tmp_path, monkeypatch):
    return L1T.project.__wrapped__(tmp_path, monkeypatch)


def test_gnss_inside_check_farmland_leaves_the_report_areas_out(tie_project, monkeypatch):
    """The check at a station inside the area: its farmland (the full-record
    diagnostic and the same-window field) without the report area plus
    500 m, recorded as comparison_ground."""
    from insar_pipeline import gnss_tie as gt
    cfg = tie_project.cfg
    cfg.report_areas = ("Full Satellite Imaging:full", "Site:39.02,-87.34,39.04,-87.32")
    monkeypatch.setitem(L1T.SITES, "HHHH", (39.0, -87.30, 2015.5, 2018.3, -1.0))
    monkeypatch.setattr(gt, "discover", lambda cfg, root, radius=None: ["AAAA", "HHHH"])
    seen = []
    real = gt._series_check

    def spy(cfg, dates, g, r, c, los_up_map, good, tc=None, far=None, regional=None):
        seen.append(far.copy())
        return real(cfg, dates, g, r, c, los_up_map, good, tc=tc, far=far, regional=regional)
    monkeypatch.setattr(gt, "_series_check", spy)
    out = gt.run(cfg)
    chk = next(c for c in out["inside_checks"] if c["site"] == "HHHH")
    attrs = {"X_FIRST": -87.36, "Y_FIRST": 39.06, "X_STEP": 0.003, "Y_STEP": -0.003}
    site = sg.aoi_mask((-87.34, 39.02, -87.32, 39.04), attrs, (40, 40))
    assert len(seen) == 1 and seen[0].any() and not (seen[0] & sg.euclidean_buffer(site, attrs, 450.0)).any()
    assert chk["comparison_ground"]["report_areas_cut"] == ["Site"]
    assert chk["comparison_ground"]["cells_removed"]["far_field"] > 0


# ---------------------------------------------------------------- F7(b): was the steps file read
def test_tie_records_whether_the_steps_file_was_read(tie_project):
    """An empty steps_not_straddled used to read the same whether no step
    fell in the record or the list was never read."""
    from insar_pipeline import gnss_tie as gt
    out = gt.run(tie_project.cfg)                            # no ngl_steps.txt in the cache
    assert out["steps_file_read"] is False and out["steps"]["screened"] is False
    assert out["stations"] and all(s["screening"]["steps_file_read"] is False for s in out["stations"])
    (tie_project.cache / gt.STEPS_FILE).write_text("INVI  24JUN06  1  Antenna_and_Radome_Type_Changed\n" * 40)
    out = gt.run(tie_project.cfg)
    assert out["steps_file_read"] is True
    assert all(s["screening"]["steps_file_read"] is True for s in out["stations"])
    assert gt.unscreened_reason(out) is None


ANCHORED = {"anchored": True, "background": {"vertical_mm_yr": -1.5, "vertical_uncertainty_mm_yr": 1.5}}


@pytest.mark.parametrize("tie,fires,words", [
    (dict(ANCHORED, steps={"screened": False}, steps_file_read=False), True, "could not be read"),
    (dict(ANCHORED, steps={"screened": False}), True, "could not be read"),     # before steps_file_read
    (dict(ANCHORED), True, "before the stations' records were checked"),       # before the screening
    (dict(ANCHORED, steps={"screened": True}, steps_file_read=True), False, None),
    ({"anchored": False, "background": None, "steps": {"screened": False}, "steps_file_read": False,
      "inside_checks": [{"site": "LAMC", "consistent": None}]}, False, None),   # nothing rests on them
    ({"anchored": False, "background": None, "steps_file_read": False,
      "inside_checks": [{"site": "LAMC", "consistent": True}]}, True, "station inside the area"),
    (None, False, None),
])
def test_unscreened_reason(tie, fires, words):
    from insar_pipeline import gnss_tie as gt
    why = gt.unscreened_reason(tie)
    assert bool(why) is fires
    if fires:
        assert words in why and "Re-run the GNSS tie" in why and ".." not in why
        assert "steps" not in why.lower()                    # plain words, not the file's name


@pytest.fixture
def _identity_set(monkeypatch):
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})


def test_release_lists_unscreened_gnss_stations_as_a_soft_item(tmp_path, _identity_set):
    """F7(b): a soft [read] item when the GNSS stations the report is
    anchored to were not screened; a signed report stays FINAL."""
    from insar_pipeline import release
    rep = L7b._rep(gnss=dict(ANCHORED, steps={"screened": False, "note": "NGL steps file not available"},
                             steps_file_read=False))
    cfg = L7b._cfg(tmp_path)
    L7b._project(tmp_path, rep, L7b._qa())
    L7b._sign_all(tmp_path, rep)
    b = release.blockers(cfg)
    hits = [x for x in b if x["code"] == "gnss_steps_not_screened"]
    assert len(hits) == 1 and hits[0]["hard"] is False
    assert "equipment changes and earthquakes" in hits[0]["text"]
    assert release.is_final(cfg) is True
    # screened: no item; a report without a GNSS block: no item
    ok = L7b._rep(gnss=dict(ANCHORED, steps={"screened": True}, steps_file_read=True))
    assert "gnss_steps_not_screened" not in {x["code"] for x in release.blockers(cfg, rep=ok)}
    assert "gnss_steps_not_screened" not in {x["code"] for x in release.blockers(cfg, rep=L7b._rep())}


def test_thin_farmland_the_cut_does_not_touch_is_not_blamed_on_the_report_areas():
    """Reviewer fix: the 'cover nearly all the farmland' exception is the
    cut's own doing only. Farmland already under CUT_MIN_CELLS without the
    cut (here 150 cells, far from the area) is the QA's ordinary thin far
    field: the areas are cut (nothing to cut), and no reason blames them.
    Before, any thin farmland with a report area configured was recorded as
    the areas covering it, and they were left uncut."""
    from insar_pipeline import quality_assessment as qa
    attrs, shape = _l8_attrs(), (L8T.NY, L8T.NX)
    cfg = types.SimpleNamespace(bbox=[-87.80, 38.362, -87.635, 38.47], deramp_exclude_geojson="",
                                acceptance_far_field_buffer_m=2000.0,
                                report_areas=("Full Satellite Imaging:full", SITE))
    site = _site_mask(shape, attrs)
    good = np.zeros(shape, bool)
    rows, cols = np.nonzero(~sg.euclidean_buffer(site, attrs, 2000.0) & sg.aoi_mask(cfg.bbox, attrs, shape))
    good[rows[:150], cols[:150]] = True
    far, far_wide, cut, meta = qa.report_farmland_masks(cfg, good, attrs)
    cg = meta["comparison_ground"]
    assert int(far.sum()) == 150 < qa.CUT_MIN_CELLS
    assert cg["cells_removed"]["far_field"] == 0 and cg["applied"] is True and cg["report_areas_cut"] == ["Site"]
    assert "reason" not in cg and cut is not None
    # the cut that does empty the farmland still falls back, and says why
    good[site] = True
    good[~site] = False
    good[rows[:150], cols[:150]] = True
    big = types.SimpleNamespace(**dict(vars(cfg), report_areas=(
        "Full Satellite Imaging:full", "Huge:38.363,-87.799,38.469,-87.636")))
    far, _, cut, meta = qa.report_farmland_masks(big, good, attrs, wide=False)
    cg = meta["comparison_ground"]
    assert cut is None and cg["applied"] is False and cg["report_areas_not_cut"] == ["Huge"]
    assert "cover nearly all the farmland" in cg["reason"] and cg["buffer_m"] is None
