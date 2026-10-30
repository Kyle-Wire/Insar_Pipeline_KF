"""Trust review 7 Oct 2026, group inj: the injection test as a technical
record (E4, findings #52-58) and the credits (E27, finding #47). Synthetic
projects only, no network.

  E4  'found' is the report-area rule and sinking only (#52, #57)
  E4  a zero-rate control beside every detection rate, with 95% ranges (#57, #58)
  E4  sites and nulls on the report's farmland: second pass, report areas out (#53, E28)
  E4  no client sentence; the record states its rule (#52, C5)
  E4  wide or poorly recovered patches are flagged, not representative (#54)
  E27 the Copernicus DEM licence notice and liability sentence, MiaplPy volume
      171, MIDAS, snaphu's three papers, the ASF line, no internal path (#47)
"""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import test_fix_L10 as L10                     # noqa: E402  (the official-project fixture)
from insar_pipeline import credits
from insar_pipeline import injection_test as it
from insar_pipeline import significance as sg


def _project(tmp_path, n=260, **kw):
    """L10's official project, with the azimuth the report's LOS map reads
    (deliverables.geometry_field needs it for a topsStack geometry)."""
    O = L10.official_project(tmp_path, ny=n, nx=n, mine_box=(150, 200, 150, 200), noise_mm=2.0, **kw)
    with h5py.File(O.cfg.mintpy_dir / "geo" / "geo_geometryRadar.h5", "a") as f:
        f["azimuthAngle"] = np.full((O.ny, O.nx), -100.0, "float32")
    return O


def _light_fields(seed=3, n=200):
    """A farmland-like field and the keys _decide reads (no files)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    corr = gaussian_filter(rng.normal(0, 1, (n, n)), 6)
    vel = 2.0 * corr / corr.std() + rng.normal(0, 0.6, (n, n))
    tc = rng.uniform(0.55, 0.95, (n, n))
    far = np.ones((n, n), bool)
    aoi = np.ones((n, n), bool)
    place = sg.placement_field(vel, tc, far, far, aoi)
    return {"vel": vel, "tc": tc, "far": far, "far_wide": far, "aoi": aoi, "placement": place,
            "good": far, "attrs": {}}


# ================================================================ E4: 'found' is sinking only
def test_found_is_sign_aware_and_counts_the_wrong_sign():
    F = _light_fields()
    region = np.zeros(F["vel"].shape, bool)
    region[90:110, 90:110] = True
    meta = {"planted_region_los_mm_yr": -15.0}
    up = np.zeros(F["vel"].shape)
    up[region] = 15.0                                   # a patch planted as SINKING that reads as uplift
    c = it._decide(F, up, region, dict(meta))
    assert c["detected"] is False                       # not 'found' (it was until 7 Oct 2026)...
    assert c["distinguishable"] is True and c["wrong_sign"] is True     # ...though the two-sided test fires
    c2 = it._decide(F, -up, region, dict(meta))
    assert c2["detected"] is True and c2["wrong_sign"] is False and c2["excess_mm_yr"] < 0
    assert it.found({"testable": True, "distinguishable_from_farmland": True, "excess_mm_yr": 4.0}) is False
    assert it.found({"testable": False, "distinguishable_from_farmland": None}) is None


# ================================================================ E4: control, ranges, no client sentence
def test_zero_rate_control_beside_every_rate_and_no_client_sentence(tmp_path):
    O = _project(tmp_path)
    rec = json.loads(it.run(O.cfg, out_dir=tmp_path / "out", n_farmland=4, n_mine=2, sizes_km=(0.6,),
                            rates_mm_yr=(0.0, 20.0), types=("trough",), seed=11).read_text())
    assert rec["status"] == "complete"
    # the control: every site and width, nothing planted, kept apart from the planted cases
    assert rec["rates_mm_yr"] == [20.0] and rec["rates_dropped"]["rates_mm_yr"] == [0.0]
    ctl = rec["control_cases"]
    assert len(ctl) == len(rec["sites"]) and {c["type"] for c in ctl} == {"control"}
    assert all(c["rate_mm_yr"] == 0.0 and c["planted_region_los_mm_yr"] == 0.0 for c in ctl)
    assert all(c["rate_mm_yr"] > 0 for c in rec["cases"])
    # the control reads the report's own field with nothing planted (the same rule)
    c0 = next(c for c in ctl if c.get("testable"))
    st = rec["sites"][c0["site"]]
    assert c0["kind"] == st["kind"] and c0["region_cells"] > 0
    # every planted row sits beside its control and carries its 95% range
    rows = rec["summary"]
    ctrl_rows = [r for r in rows if r["type"] == "control"]
    assert {r["kind"] for r in ctrl_rows} == {"farmland", "mine"}
    assert "false alarms" in next(r for r in ctrl_rows if r["kind"] == "farmland")["meaning"]
    assert "may really be moving" in next(r for r in ctrl_rows if r["kind"] == "mine")["meaning"]
    for r in rows:
        assert r["detection_ci95"] == it.clopper_pearson(r["detected"], r["testable"])
        if r["type"] != "control":
            c = next(x for x in ctrl_rows if x["kind"] == r["kind"] and x["width_km"] == r["width_km"])
            assert r["control_rate"] == c["detection_rate"] and r["control_ci95"] == c["detection_ci95"]
            assert r["median_planted_area_mean_vertical_mm_yr"] < 0
    # a technical record only: no client sentence, the rule written out
    assert rec["sentence"] is None
    assert "report-area rule" in rec["decision_rule"] and "sinking" in rec["decision_rule"]
    assert "feature search" in rec["decision_rule"] and "feature search" in rec["not_covered"]
    assert rec["technical_summary"].startswith("Technical record, not for the client page")
    assert "nothing planted" in rec["technical_summary"] and "0.6 km" in rec["technical_summary"]


def test_summary_ranges_and_the_not_representative_flag():
    assert it.clopper_pearson(16, 20) == [0.563, 0.943]
    assert it.clopper_pearson(20, 20) == [0.832, 1.0]
    assert it.clopper_pearson(0, 20)[0] == 0.0 and it.clopper_pearson(0, 0) is None
    base = {"kind": "farmland", "width_km": 2.0, "testable": True, "planted_region_vertical_mm_yr": -14.4}
    cases = [dict(base, type="trough", rate_mm_yr=20.0, detected=i < 16, distinguishable=i < 16,
                  recovered_over_planted=0.9, chain_attenuation=0.97) for i in range(20)]
    cases += [dict(base, type="control", rate_mm_yr=0.0, detected=i == 0, distinguishable=i < 2,
                   wrong_sign=i == 1, planted_region_vertical_mm_yr=0.0) for i in range(20)]
    rows = it.summarise(cases, [2.0], [20.0], ["trough"])
    ctl = next(r for r in rows if r["type"] == "control")
    tr = next(r for r in rows if r["type"] == "trough")
    assert ctl["detected"] == 1 and ctl["detection_rate"] == 0.05 and ctl["wrong_sign"] == 1
    assert ctl["distinguishable_either_sign"] == 2
    assert tr["detection_ci95"] == [0.563, 0.943]
    assert tr["control_rate"] == 0.05 and tr["control_ci95"] == it.clopper_pearson(1, 20)
    # recovered at 0.9 of planted: the planting moved the zero/deramp - flagged
    assert tr["representative"] is False and "not representative of a mine area" in tr["note"]
    assert "[not representative]" in it.technical_summary(rows)


def test_not_representative_is_the_farmland_pull_only():
    """Reviewer 7 Oct 2026: the flag says the planting on FARMLAND moved the
    zero or the deramp, so it is judged on recovered/planted with the type's
    temporal gain divided out (the DEM-error and velocity-fit loss a mine
    suffers too), and a row planted on a mine is never flagged."""
    def cases(kind, ratio):
        base = {"kind": kind, "type": "trough", "width_km": 1.0, "rate_mm_yr": 20.0, "testable": True,
                "planted_region_vertical_mm_yr": -14.4}
        return [dict(base, detected=True, distinguishable=True, recovered_over_planted=ratio,
                     chain_attenuation=ratio) for _ in range(5)]
    rows = it.summarise(cases("mine", 0.9) + cases("farmland", 0.93), [1.0], [20.0], ["trough"],
                        gains={"trough": 0.93})
    mine = next(r for r in rows if r["kind"] == "mine")
    farm = next(r for r in rows if r["kind"] == "farmland")
    assert mine["representative"] is True and "note" not in mine          # planted on a mine: what a mine reads
    assert farm["representative"] is True                                  # all of the 0.93 is the time fit
    farm2 = it.summarise(cases("farmland", 0.9), [1.0], [20.0], ["trough"], gains={"trough": 1.0})[0]
    assert farm2["representative"] is False and "time-fit loss" in farm2["note"]


def test_decision_rule_states_the_area_stays_in_its_own_comparison_ground():
    # E28 takes a report area out of its own comparison ground; the drawn area is not (said, with the measure)
    assert "own comparison ground" in it.DECISION_RULE and "E28" in it.DECISION_RULE


def test_widths_beyond_the_default_are_flagged(tmp_path, monkeypatch):
    O = _project(tmp_path)
    monkeypatch.setattr(it, "SIZES_KM", (0.3,))       # 0.6 km is then beyond the default widths
    rec = json.loads(it.run(O.cfg, out_dir=tmp_path / "o", n_farmland=2, n_mine=0, sizes_km=(0.6,),
                            rates_mm_yr=(20.0,), types=("trough",), seed=5).read_text())
    assert rec["sizes_beyond_default_km"]["widths_km"] == [0.6]
    assert "not representative of a mine-sized area" in rec["sizes_beyond_default_km"]["note"]


# ================================================================ E4: the report's farmland (second pass, E28)
def _fake_search(cut_mask):
    """client_report._feature_search stand-in: one candidate that beats chance."""
    def search(ctx, los_up_map, poly, aoi, standout):
        assert standout is None                     # no QA stand-out: the QA has not run yet
        f = {"_mask": cut_mask & ctx.good, "stands_out": True, "sign": -1}
        return {"field": None}, [f, {"_mask": np.zeros_like(cut_mask), "stands_out": False}], {}, True
    return search


def test_sites_and_nulls_use_the_reports_second_pass_farmland(tmp_path, monkeypatch):
    from insar_pipeline import client_report as cr
    O = _project(tmp_path)
    cut = np.zeros((O.ny, O.nx), bool)
    cut[40:70, 150:190] = True                         # an unmapped feature the report finds in the farmland
    monkeypatch.setattr(cr, "_feature_search", _fake_search(cut))
    # a report area other than the whole study area is not comparison ground either
    rows, cols = (100, 120), (40, 60)
    lat = [round(O.y0 + (r + 0.25) * O.dy, 6) for r in rows]
    lon = [round(O.x0 + (c + 0.25) * O.dx, 6) for c in cols]
    O.cfg.report_areas = ["Full Satellite Imaging:full",
                          f"Box:{min(lat)},{min(lon)},{max(lat)},{max(lon)}"]
    F, info = it.report_farmland(O.cfg)
    grown = sg.euclidean_buffer(cut, F["attrs"], cr.FARMLAND_SECOND_PASS_M)
    assert not (F["far"] & grown).any() and not (F["far_wide"] & grown).any()
    box = sg.aoi_mask((min(lon), min(lat), max(lon), max(lat)), F["attrs"], cut.shape)
    assert box.sum() > 50 and not (F["far"] & box).any() and not (F["far_wide"] & box).any()
    # the report's rule for a report area (contract C2): far_field_mask(exclude=...), grown by its buffer
    box_grown = sg.euclidean_buffer(box, F["attrs"], sg.EXCLUDE_BUFFER_M)
    assert not (F["far"] & box_grown).any() and not (F["far_wide"] & box_grown).any()
    far_c2, _ = sg.far_field_mask(O.cfg, F["good"], F["attrs"], exclude=box)
    assert np.array_equal(F["far"], far_c2 & ~grown)            # the report areas out, then the second pass
    assert info["report_area_buffer_m"] == sg.EXCLUDE_BUFFER_M
    assert info["features_cut"] == 1 and info["report_areas_excluded"] == ["Box"]
    assert info["far_cells"] < info["far_cells_without_report_areas"] < info["far_cells_first_pass"]
    assert cr.runs_mask(info["cut_cells"], cut.shape).sum() == int((cut & F["good"]).sum())
    # the null field is the report's placement field on THAT farmland
    expect = sg.placement_field(F["vel"], F["tc"], F["far"], F["far_wide"], F["aoi"])
    assert np.allclose(F["placement"], expect, equal_nan=True)

    rec = json.loads(it.run(O.cfg, out_dir=tmp_path / "o", n_farmland=6, n_mine=1, sizes_km=(0.3,),
                            rates_mm_yr=(20.0,), types=("trough",), seed=3).read_text())
    assert rec["farmland"]["features_cut"] == 1 and rec["farmland"]["report_areas_excluded"] == ["Box"]
    for s in rec["sites"]:
        if s["kind"] == "farmland":                    # never on the found feature, its buffer or the area
            assert F["far"][s["row"], s["col"]] and not grown[s["row"], s["col"]] and not box[s["row"], s["col"]]


# ================================================================ E27: credits
def _cfg(tmp_path, **kw):
    base = dict(project_dir=tmp_path, backend="miaplpy", start_date="2016-01-05", end_date="2026-06-17",
                mintpy_tropo_method="pyaps", dem_path=None, dem_dir=tmp_path / "dem",
                mine_layer_sources=(), mintpy_dir=tmp_path / "mintpy")
    base.update(kw)
    return SimpleNamespace(**base)


REP = {"gnss": {"anchored": True}, "dates": ["20160105", "20260617"]}
NOTICE_6B = ("produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
             "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved")
LIABILITY_6C = ("The organisations in charge of the Copernicus programme by law or by delegation do not incur "
                "any liability for any use of the Copernicus WorldDEM-30.")


def test_dem_credit_carries_the_licence_notice_and_liability(tmp_path):
    items = credits.credits(_cfg(tmp_path), REP, prov={})
    dem = next(c for c in items if c["source"] == "Copernicus DEM GLO-30")
    assert NOTICE_6B in dem["text"] and LIABILITY_6C in dem["text"]
    line = credits.short_line(_cfg(tmp_path), REP, {})
    assert NOTICE_6B in line and "© Airbus 2014-2018" not in line     # the footer carries 6(b) in full
    block = credits.text_block(_cfg(tmp_path), REP, {})
    assert NOTICE_6B in block and LIABILITY_6C in block
    # a supplied DEM is not given the Copernicus notice
    t = " ".join(c["text"] for c in credits.credits(_cfg(tmp_path, dem_path=tmp_path / "lidar.tif"), REP, prov={}))
    assert "WorldDEM" not in t


def test_citations_are_corrected_and_complete(tmp_path):
    prov = {"software": {"miaplpy": {"version": "0.2.1+overlapdatum"}, "snaphu": "snaphu v2.0.6"}}
    items = credits.credits(_cfg(tmp_path), REP, prov=prov)
    by = {c["source"]: c["text"] for c in items}
    mp = by["MiaplPy"]
    assert "Computers & Geosciences 171, 105291" in mp and " 173" not in mp
    assert "Non-linear phase linking using joined distributed and persistent scatterers" in mp
    assert "modified by us" in mp
    assert not any("patches/" in c["text"] for c in items)              # no internal path in client text
    ngl = by["Nevada Geodetic Laboratory GNSS"]
    assert "MIDAS" in ngl and "doi:10.1002/2015JB012552" in ngl and "121, 2054-2068" in ngl
    sn = by["snaphu"]
    assert "17(3), 401-414" in sn and "18(2), 338-351" in sn and "40(8), 1709-1719" in sn and "v2.0.6" in sn
    assert "retrieved from the Alaska Satellite Facility DAAC" in by["ASF DAAC"] and "2016-2026" in by["ASF DAAC"]
    # DISP products are OPERA's, distributed by ASF: no separate SLC retrieval line
    disp = credits.credits(_cfg(tmp_path, backend="disp"), REP, prov={})
    assert not any(c["source"] == "ASF DAAC" for c in disp)


@pytest.mark.parametrize("backend", ["miaplpy", "stack", "hyp3", "disp"])
def test_no_internal_path_in_any_route_credit(tmp_path, backend):
    prov = {"software": {"miaplpy": {"version": "0.2.1+overlapdatum"}}}
    items = credits.credits(_cfg(tmp_path, backend=backend), REP, prov=prov)
    assert items and not any("patches/" in c["text"] or "/home/" in c["text"] for c in items)
