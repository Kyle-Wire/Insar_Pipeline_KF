"""Review 7 Oct 2026, group I_disp: injection sign, cross-check order and
masks, screening guards. Synthetic projects only, no network.

  #47 injection_test plants SINKING (it planted uplift and called it sinking)
  #48 the cross-check and injection test run before the QA that copies them;
      the client report can redo the feature rows for its own features
  #49 a feature is measured by both products over the same cells (outline)
  #50 'Products only' on a DISP config refuses an official folder
  #51 the screening set-aside moves DRAFT/SCREENING KMZs and the area PDF
  #52 the cross-check sentence says 'matches' only above the pass bar
  #53 only 'no frame on this track' is 'no independent product'
  #54 a series that does not match the velocity is refused, not swapped
"""
import json

import numpy as np
import pytest

import test_fix_L10 as L10                     # noqa: E402  (the official-project and DISP fixtures)
from insar_pipeline import disp_backend as db
from insar_pipeline import notify, pipeline, progress, runs


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("INSAR_NOTIFY", "off")
    monkeypatch.setattr(notify, "LOGFILE", tmp_path / "notifications.log")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    yield
    from insar_pipeline import procs
    procs.clear_stopping()


def _cfg(tmp_path, backend="miaplpy", **attrs):
    from insar_pipeline.config import load_config
    f = tmp_path / "c.yaml"
    f.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    cfg = load_config(str(f))
    cfg.backend = backend
    for k, v in attrs.items():
        setattr(cfg, k, v)
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _stub_products(monkeypatch, calls):
    def mk(name):
        def f(cfg):
            calls.append(name)
            progress.emit(cfg.project_dir, name, "done")
            return True
        return f

    def export(cfg):
        calls.append("deliverables")
        return cfg.project_dir / "deliverables"
    monkeypatch.setattr(pipeline, "export_all", export)
    for name in ("quality_assessment", "figures", "client_report", "crosscheck", "injection_test"):
        monkeypatch.setattr(pipeline, f"run_{name}", mk(name))


# ================================================================ #47 injection sign
def test_injection_plants_sinking_and_keeps_the_ratios_positive(tmp_path):
    from insar_pipeline import injection_test as it
    O = L10.official_project(tmp_path, mine_box=(55, 105, 55, 105), noise_mm=2.0)
    rec = json.loads(it.run(O.cfg, out_dir=tmp_path / "out", n_farmland=3, n_mine=2, sizes_km=(0.6,),
                            rates_mm_yr=(20.0,), types=("trough", "settlement"), seed=11).read_text())
    assert rec["status"] == "complete" and rec["cases"]
    for c in rec["cases"]:
        assert c["rate_mm_yr"] > 0                                   # the size the sentence names
        assert c["planted_region_los_mm_yr"] < 0 and c["planted_region_vertical_mm_yr"] < 0
        if c.get("testable"):
            assert c["recovered_over_planted"] > 0 and c["chain_attenuation"] > 0
    troughs = [c for c in rec["cases"] if c["type"] == "trough" and c.get("testable")]
    assert troughs and all(c["recovered_los_mm_yr"] < 0 for c in troughs)
    found = [c for c in rec["cases"] if c.get("detected")]
    assert found and all(c["excess_mm_yr"] < 0 for c in found)     # found as sinking, not uplift
    # E4 (trust review 7 Oct 2026): no client sentence; 'found' is sinking
    # only, as the record's decision rule says
    assert rec["sentence"] is None and "sinking" in rec["decision_rule"] and "sinking" in rec["rate_definition"].lower()


# ================================================================ #48 stage order + feature recheck
def test_crosscheck_and_injection_run_before_the_quality_assessment(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, cross_check_disp=True, injection_test=True)
    calls = []
    _stub_products(monkeypatch, calls)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    assert calls == ["deliverables", "crosscheck", "injection_test", "quality_assessment", "figures",
                     "client_report"]


def test_optional_stage_frees_the_deliverables_cache_first(tmp_path, monkeypatch):
    from insar_pipeline import crosscheck as cc, deliverables
    cfg = _cfg(tmp_path)
    deliverables._CACHE["x"] = np.zeros(3)
    seen = {}
    monkeypatch.setattr(cc, "run", lambda c: seen.setdefault("cache", dict(deliverables._CACHE)))
    assert pipeline.run_crosscheck(cfg)
    assert seen["cache"] == {}


def test_recheck_features_scores_the_reports_own_features(tmp_path, monkeypatch):
    """The check ran against a previous report with 3 features; the report
    being built lists 2, so the sentence must say 'of 2'."""
    from insar_pipeline import crosscheck as cc
    O, _ = L10._xcheck_env(tmp_path, monkeypatch)
    rep = O.cfg.project_dir / "deliverables" / "client_report.json"
    f0 = json.loads(rep.read_text())["features"][0]
    old = [f0, dict(f0, lat=f0["lat"] + 0.01), dict(f0, lat=f0["lat"] - 0.01)]
    for f in old[1:]:
        f["bbox"] = [f["bbox"][0], f["lat"] - 0.003, f["bbox"][2], f["lat"] + 0.003]
    rep.write_text(json.dumps({"features": old}))
    rec = json.loads(cc.run(O.cfg).read_text())
    assert "of 3 reported features" in rec["sentence"] and len(rec["features"]) == 3
    assert rec["features_from"]["report"].endswith("client_report.json")
    new = cc.recheck_features(O.cfg, rec, old[:2])
    assert len(new["features"]) == 2 and "of 2 reported features" in new["sentence"]
    assert new["features"][0]["confirmed"] is True
    assert new["one_km_means"] == rec["one_km_means"] and new["areas"] == rec["areas"]
    # anything but a completed comparison is left alone
    na = {"status": "no independent product", "available": False, "sentence": "No independent ..."}
    assert cc.recheck_features(O.cfg, na, old) is na


# ================================================================ #49 same cells for both products
def test_feature_is_measured_over_its_outline_by_both_products(tmp_path):
    from insar_pipeline import crosscheck as cc
    O = L10.official_project(tmp_path, ny=200, nx=200, bowl=(100, 100, 150.0, -40.0), noise_mm=1.0)
    lat_c, lon_c = O.y0 + 100.5 * O.dy, O.x0 + 100.5 * O.dx
    a, b = 0.0015, 0.002                       # outline: about 330 x 340 m around the centre
    ring = [[lon_c - b, lat_c - a], [lon_c + b, lat_c - a], [lon_c + b, lat_c + a], [lon_c - b, lat_c + a],
            [lon_c - b, lat_c - a]]
    bbox = [lon_c - 0.006, lat_c - 0.0045, lon_c + 0.006, lat_c + 0.0045]     # a much larger box
    rep = O.cfg.project_dir / "deliverables" / "client_report.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({"features": [{"lat": lat_c, "lon": lon_c, "vertical_relative_mm_yr": -45.0,
                                             "ring": ring, "bbox": bbox}]}))
    out = cc.compare(O.cfg, O.cfg)             # the official product against itself
    f = out["features"][0]
    assert f["cells_from"] == "outline" and f["report_vertical_relative_mm_yr"] == -45.0
    assert f["disp_relative_los_mm_yr"] == pytest.approx(f["official_relative_los_mm_yr"], abs=0.01)
    assert f["disp_relative_vertical_mm_yr"] == pytest.approx(f["official_relative_vertical_mm_yr"], abs=0.01)
    # the box would have diluted it: its mean is clearly weaker than the outline's
    Dd = cc.load_fields(O.cfg)
    rel = np.where(Dd["good"], Dd["r_cm"], np.nan)
    box = cc._bbox_mask(bbox, Dd["attrs"], rel.shape) & np.isfinite(rel)
    assert abs(np.nanmean(rel[box])) < 0.8 * abs(f["official_relative_los_mm_yr"])
    assert f["confirmed"] is True


# ================================================================ #50 refusal before 'Products only'
def test_products_only_on_a_disp_config_refuses_an_official_folder(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, backend="disp", disp_record_chain=True)
    M = cfg.mintpy_dir
    M.mkdir(parents=True, exist_ok=True)
    (M / "run_products.json").write_text("{}")                     # the phase-linked run's record
    D = cfg.project_dir / "deliverables"
    D.mkdir()
    (D / "client_report.html").write_text("official report")
    calls = []
    _stub_products(monkeypatch, calls)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) != 0
    assert calls == []
    assert (D / "client_report.html").read_text() == "official report"
    assert not (D / pipeline.SCREENING_NOTE).exists() and not (D / "screening_not_for_clients").exists()
    r = json.loads((cfg.project_dir / runs.RESULTFILE).read_text())
    assert r["status"] == "failed" and r["stage"] == "checks"


# ================================================================ #51 every client file set aside
def test_screening_set_aside_moves_draft_kmzs_and_the_pdf(tmp_path):
    cfg = _cfg(tmp_path, backend="disp")
    D = cfg.project_dir / "deliverables"
    D.mkdir()
    names = ["client_report.html", "client_report.json", "client_area_summaries.pdf",
             "Bear_SCREENING_ground_movement_DRAFT.kmz", "Bear_ground_movement_DRAFT.kmz",
             "Bear_ground_movement.kmz"]
    for n in names:
        (D / n).write_text(n)
    (D / "summary.json").write_text("{}")                           # technical: stays
    moved = pipeline.set_aside_client_files(cfg)
    assert sorted(p.name for p in moved) == sorted(names)
    assert not any((D / n).exists() for n in names) and (D / "summary.json").exists()
    assert all(p.parent.parent.name == "screening_not_for_clients" for p in moved)


# ================================================================ #52 the sentence
def test_crosscheck_sentence_says_matches_only_above_the_bar():
    from insar_pipeline import crosscheck as cc
    weak = cc.sentence({"one_km_means": {"correlation": 0.2}, "common_window": ["20160101", "20260101"]})
    assert "matches" not in weak and "only weakly" in weak and "0.20" in weak
    neg = cc.sentence({"one_km_means": {"correlation": -0.3}, "common_window": ["20160101", "20260101"]})
    assert "matches" not in neg and "agrees" not in neg and "does not follow" in neg and "-0.30" in neg
    good = cc.sentence({"one_km_means": {"correlation": 0.8}, "common_window": ["20160101", "20260101"]})
    assert "matches the record's pattern" in good


# ================================================================ #53 'no product' vs a failed search
def test_only_no_frame_on_track_is_no_independent_product(tmp_path, monkeypatch):
    from insar_pipeline import crosscheck as cc
    O = L10.official_project(tmp_path)
    monkeypatch.setattr(cc, "_run_disp_chain", lambda sib: (_ for _ in ()).throw(AssertionError("ran")))
    f = O.cfg.project_dir / "deliverables" / "technical" / "cross_check.json"

    def mixed(sib):
        raise ValueError("DISP-S1 granules over this AOI mix major product versions (v1.0: 3, v2.0: 2)")
    monkeypatch.setattr(cc, "_search_disp", mixed)
    with pytest.raises(ValueError, match="mix major"):
        cc.run(O.cfg)
    rec = json.loads(f.read_text())
    assert rec["status"] == "failed" and "mix major" in rec["reason"] and "sentence" not in rec

    def none(sib):
        raise db.NoFrameOnTrack("no DISP-S1 frame over this AOI is on track 19 ascending")
    monkeypatch.setattr(cc, "_search_disp", none)
    rec = json.loads(cc.run(O.cfg).read_text())
    assert rec["status"] == "no independent product" and "track 19" in rec["reason"]


def test_choose_frame_raises_no_frame_on_track_only_for_a_missing_frame(tmp_path):
    box, res = L10._frames()
    cfg = L10.make_cfg(tmp_path, bbox=list(box), track=19, disp_frame_id="", orbit_direction="descending")
    with pytest.raises(db.NoFrameOnTrack):
        db.choose_frame(res, cfg)
    with pytest.raises(db.NoFrameOnTrack):
        db.choose_frame([], cfg)
    cfg.orbit_direction = "ascending"
    res[0].properties["pathNumber"] = 20                           # a granule off the track: a failure
    cfg.disp_frame_id = "04862"
    with pytest.raises(ValueError) as e:
        db._select(cfg, res)
    assert not isinstance(e.value, db.NoFrameOnTrack)
    assert issubclass(db.NoFrameOnTrack, ValueError)                # old `except ValueError` callers still work


# ================================================================ #54 a mismatched series is refused
def test_mismatched_time_series_is_refused_not_swapped(tmp_path, monkeypatch):
    from insar_pipeline import crosscheck as cc, deliverables, injection_test as it
    O = L10.official_project(tmp_path)

    def refuse(cfg):
        raise ValueError("The time series x does not match the velocity product: its REF_Y is 3")
    monkeypatch.setattr(deliverables, "timeseries_file", refuse)
    with pytest.raises(ValueError, match="does not match the velocity"):
        cc.timeseries_file(O.cfg)
    with pytest.raises(ValueError, match="does not match the velocity"):
        it.run(O.cfg, out_dir=tmp_path / "o", n_farmland=1, n_mine=0, sizes_km=(0.6,), rates_mm_yr=(20.0,))
    assert json.loads((tmp_path / "o" / "injection_recovery.json").read_text())["status"] == "failed"
    # nothing to choose from in deliverables is still the logged fallback by name
    monkeypatch.setattr(deliverables, "timeseries_file",
                        lambda cfg: (_ for _ in ()).throw(FileNotFoundError("No timeseries file")))
    assert cc.timeseries_file(O.cfg).name == "geo_timeseries_ERA5_ramp_demErr.h5"
