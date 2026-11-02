"""
INT1 (integration after wave 1, 5 Oct 2026 review): the requests the wave-1
lanes filed against each other's files, each checked on its OUTCOME with
synthetic data - what is on disk, in the records, or raised - so the code
before this lane fails and the code after it passes. No network.

Run: conda run -n insar python -m pytest tests/test_fix_INT1.py -q
"""
import dataclasses
import json
import os
import re
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from insar_pipeline.config import PipelineConfig, load_config   # noqa: E402

PKG = Path(__file__).resolve().parents[1]


def _w(path, data: dict, attrs: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)
        for k, v in attrs.items():
            f.attrs[k] = v


def _cfg(tmp_path, **over):
    kw = dict(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.52, 38.47], start_date="2018-06-01",
              end_date="2019-06-30", backend="miaplpy")
    kw.update(over)
    cfg = PipelineConfig(**kw)
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _now():
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


# ================================================================== config
def test_era5_allow_partial_is_a_field_read_by_the_coverage_check(tmp_path):
    """timeseries-2 (L2 -> config.py): the opt-in exists, defaults to the safe
    stop, loads from YAML, and corrections reads it as a field."""
    assert {f.name: f.default for f in dataclasses.fields(PipelineConfig)}["era5_allow_partial"] is False
    y = tmp_path / "c.yaml"
    y.write_text("project_dir: %s\nbbox: [-87.8, 38.2, -87.52, 38.47]\nstart_date: '2018-06-01'\n"
                 "end_date: '2019-06-30'\nbackend: hyp3\nera5_allow_partial: true\n" % (tmp_path / "p"))
    assert load_config(str(y)).era5_allow_partial is True
    src = (PKG / "corrections.py").read_text()
    assert "cfg.era5_allow_partial" in src and 'getattr(cfg, "era5_allow_partial"' not in src


def test_every_config_field_is_read_outside_config_py():
    """CONTRACTS: no setting a coworker can edit that nothing reads."""
    files = [p for p in PKG.rglob("*.py") if not ({"backups", "tests", "__pycache__"} & set(p.parts))
             and not p.relative_to(PKG).parts[0].startswith("review_") and p.name != "config.py"]
    text = "\n".join(p.read_text() for p in files)
    unread = [f.name for f in dataclasses.fields(PipelineConfig)
              if not re.search(r"(\.|[\"'])" + re.escape(f.name) + r"\b", text)]
    assert unread == []


def test_load_config_records_the_source_yaml_outside_the_fields(tmp_path):
    """techout-4 (L7b -> config.py): provenance needs the exact YAML; the
    attribute is not a setting (never in the snapshot)."""
    from insar_pipeline import provenance
    y = tmp_path / "c.yaml"
    y.write_text("project_dir: %s\nbbox: [-87.8, 38.2, -87.52, 38.47]\nstart_date: '2018-06-01'\n"
                 "end_date: '2019-06-30'\nbackend: hyp3\n" % (tmp_path / "p"))
    cfg = load_config(str(y))
    assert cfg._source_path == y.resolve()
    assert "_source_path" not in provenance.config_snapshot(cfg)[0]
    assert provenance._source_config(cfg)["sha256"] is not None


def test_max_plausible_comment_names_the_gradient_limit_not_metres_per_year():
    """gap-surface_mine_site-8 (L8 -> config.py, areas.py)."""
    src = (PKG / "config.py").read_text()
    i = src.index("max_plausible_velocity_mm_yr: float")
    block = src[i - 900:i]
    assert "DIFFERENCE" in block and "422" in block and "metres per\n    # year and a low threshold" not in block
    assert "mining can be fast" not in (PKG / "areas.py").read_text()


# ================================================================== pipeline
def test_qa_stage_fetches_water_and_attaches_its_verdict_to_the_summary(tmp_path, monkeypatch):
    """gap-geolocation-7 + empirical-4 (L8/L7a -> pipeline.py)."""
    from insar_pipeline import deliverables, pipeline, progress, quality_assessment as qa
    cfg = _cfg(tmp_path)
    seen = {}
    monkeypatch.setattr(qa, "evaluate", lambda c, fetch_water=False: seen.setdefault("fw", fetch_water) and
                        {"verdict": "REVIEW", "send_rule": {"value": "send with notes"}} or
                        {"verdict": "REVIEW", "send_rule": {"value": "send with notes"}})
    monkeypatch.setattr(qa, "write", lambda c, r: tmp_path / "qa.json")
    monkeypatch.setattr(qa, "format_report", lambda r: "")
    monkeypatch.setattr(deliverables, "refresh_summary", lambda c: seen.setdefault("refreshed", True))
    msgs = []
    monkeypatch.setattr(progress, "emit", lambda *a, **k: msgs.append(k.get("msg")))
    assert pipeline.run_quality_assessment(cfg)
    assert seen["fw"] is True and seen.get("refreshed") is True
    assert any(m and "send with notes" in m for m in msgs)


def test_report_stage_refreshes_the_summary_even_when_the_tie_fails_and_records_stages(tmp_path, monkeypatch):
    """empirical-4 + techout-4 (L7a/L7b -> pipeline.py)."""
    from insar_pipeline import client_kmz, client_report, deliverables, gnss_tie, pipeline, progress, provenance, release
    cfg = _cfg(tmp_path)
    calls = []
    monkeypatch.setattr(progress, "emit", lambda *a, **k: None)
    monkeypatch.setattr(deliverables, "refresh_summary", lambda c: calls.append("refresh"))

    def tie_fails(c):
        calls.append("tie")
        raise RuntimeError("no station")
    monkeypatch.setattr(gnss_tie, "run", tie_fails)
    assert pipeline.run_client_report(cfg) is False
    assert calls == ["tie", "refresh"]                      # the tie's outcome reached summary.json
    calls.clear()
    monkeypatch.setattr(gnss_tie, "run", lambda c: calls.append("tie"))
    monkeypatch.setattr(client_report, "build", lambda c: calls.append("report") or "r.html")
    monkeypatch.setattr(client_kmz, "build", lambda c: calls.append("kmz") or "k.kmz")
    monkeypatch.setattr(provenance, "record_stages", lambda c: calls.append("stages") or tmp_path / "p.json")
    monkeypatch.setattr(release, "status", lambda c: {"final": False, "blockers": [
        {"code": "not_signed_off", "text": "Not yet reviewed", "hard": True}]})
    assert pipeline.run_client_report(cfg) is True
    assert calls == ["tie", "refresh", "report", "kmz", "stages"]


def test_official_runner_refuses_a_folder_holding_a_screening(tmp_path):
    """disp-3 part 4 (L4b -> pipeline.py): DISP first, official second."""
    from insar_pipeline import pipeline
    cfg = _cfg(tmp_path, stack_virtual_merge="False", stack_cleanup_coreg_after_filter=False)
    (cfg.mintpy_dir / "disp_chain.json").write_text("{}")
    with pytest.raises(SystemExit, match="DISP-S1 screening"):
        pipeline.run_miaplpy(cfg, types.SimpleNamespace(plan_only=True, skip_download=True))
    (cfg.mintpy_dir / "inputs").mkdir()
    (cfg.mintpy_dir / "inputs" / "ifgramStack.h5").write_bytes(b"x")
    pipeline.refuse_screening_folder(cfg)                    # an official stack beside it: fine


# ================================================================== provenance / release / credits
def test_provenance_records_rejected_reference_dem_epoch_deramp_geometry_and_acquisitions(tmp_path):
    """phaselinked-10, gap-surface_mine_site-6/-5, stackrunner-1 (-> provenance.py)."""
    from insar_pipeline import provenance
    cfg = _cfg(tmp_path)
    cfg.reference_yaml_rejected = "38.30,-87.60"
    _w(cfg.mintpy_dir / "geo" / "geo_velocity.h5", {"velocity": np.zeros((4, 4), np.float32)}, {"UNIT": "m/year"})
    _w(cfg.mintpy_dir / "maskDeramp.h5", {"mask": np.ones((4, 4), bool)},
       {"DERAMP_BUFFER_M": 2000.0, "DERAMP_BUFFER_KIND": "euclidean", "DERAMP_QUADRANTS": 4,
        "DERAMP_QUADRANT_SHARE": json.dumps({"NE": 0.3, "NW": 0.2, "SE": 0.3, "SW": 0.2})})
    cfg.slc_dir.mkdir(parents=True)
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps(
        {"catalogue_dates": ["20180601", "20180613", "20180625"], "processed": ["20180601", "20180625"],
         "dropped": {"20180613": "catalogue_short"}, "failed": {}, "accepted_failures": []}))
    (cfg.mintpy_dir / "run_products.json").write_text(json.dumps(
        {"run_stamp": "20261005_120000", "superseded_dir": "x", "removed_ramp": {"gradient_mm_yr_per_10km": 2.5},
         "second_gate": {"applied": False}, "era5": {"dates_corrected": 2, "dates_total": 2}}))
    rec = provenance.collect(cfg)
    assert rec["reference_yaml_rejected"]["yaml"] == "38.30,-87.60"
    assert "2011-2015" in rec["dem"]["epoch"]
    out = rec["outcome"]
    assert out["deramp_mask_buffer_kind"] == "euclidean" and out["deramp_mask_quadrant_share"]["NE"] == 0.3
    assert rec["acquisitions"]["catalogue_dates"] == 3 and rec["acquisitions"]["left_out"] == {"20180613": "catalogue_short"}
    assert rec["mintpy_run"]["run_stamp"] == "20261005_120000" and rec["mintpy_run"]["removed_ramp"]


def test_record_stages_rereads_the_summary_and_the_optional_stage_records(tmp_path, monkeypatch):
    """techout-10 + empirical-14 + methodology-11 (-> provenance.py): the
    stages come from deliverables/technical/, where L10 writes them."""
    from insar_pipeline import provenance
    cfg = _cfg(tmp_path)
    D = cfg.project_dir / "deliverables"
    (D / "technical").mkdir(parents=True)
    (D / "provenance.json").write_text(json.dumps({"summary": {"data_quality": "Good", "velocity_median": 1.0}}))
    (D / "summary.json").write_text(json.dumps({"data_quality": "REVIEW", "data_quality_source": "QA"}))
    (cfg.project_dir / "quality_assessment.json").write_text(json.dumps(
        {"verdict": "REVIEW", "checks": {}, "dates": {"available": 3, "processed": 2, "dropped": {"20180613": "x"}}}))
    (D / "technical" / "injection_recovery.json").write_text(json.dumps(
        {"status": "complete", "seed": 20261005, "sizes_km": [0.6], "rates_mm_yr": [20]}))
    (D / "technical" / "cross_check.json").write_text(json.dumps(
        {"status": "complete", "sibling_project": "/x_disp_check", "disp": {"frame": "F04861"}, "indicative_pass": True}))
    path = provenance.record_stages(cfg)
    rec = json.loads(path.read_text())
    assert rec["summary"]["data_quality"] == "REVIEW"
    st = rec["stages"]
    assert st["quality_assessment"]["dates"]["dropped"] == {"20180613": "x"}
    assert st["injection_test"]["seed"] == 20261005 and st["cross_check"]["disp"]["frame"] == "F04861"


def test_release_and_credits_read_the_cross_check_where_it_is_written(tmp_path):
    """disp-8 consumer paths (L7b files read L10's record)."""
    from insar_pipeline import credits, release
    cfg = _cfg(tmp_path, cross_check_disp=True)
    D = cfg.project_dir / "deliverables" / "technical"
    D.mkdir(parents=True)
    (D / "cross_check.json").write_text(json.dumps({"status": "no independent product", "available": False}))
    assert not credits.opera_used(cfg)
    rec = {"status": "complete", "available": True, "indicative_pass": True, "generated": _now(),
           "areas": [{"name": "Site", "agree": False}]}
    (D / "cross_check.json").write_text(json.dumps(rec))
    assert credits.opera_used(cfg)
    # 7 Oct 2026: a record older than the velocity product, or left after the
    # option was turned off, is not this run's and is neither credited nor gating
    assert not credits.opera_used(_cfg(tmp_path))
    (D / "cross_check.json").write_text(json.dumps(dict(rec, generated="2020-01-01T00:00:00")))
    (cfg.mintpy_dir / "velocity.h5").write_bytes(b"")
    assert not credits.opera_used(cfg)
    assert "cross_check_disagrees" not in [b["code"] for b in release.blockers(cfg, {"areas": [], "features": []}, {"checks": {}})]
    (D / "cross_check.json").write_text(json.dumps(dict(rec, generated=_now())))
    rep = {"areas": [], "features": []}
    codes = [b["code"] for b in release.blockers(cfg, rep, {"checks": {}})]
    assert "cross_check_disagrees" in codes


def test_osm_water_used_by_the_qa_is_credited(tmp_path):
    """empirical-14 request (-> credits.py): ODbL attribution."""
    from insar_pipeline import credits
    cfg = _cfg(tmp_path)
    assert not credits.osm_used(cfg)
    (cfg.project_dir / "quality_assessment.json").write_text(json.dumps(
        {"geolocation": {"water_map": str(cfg.project_dir / "context" / "osm_water.geojson")}}))
    assert any(c["source"] == "OpenStreetMap" and "ODbL" in c["text"] for c in credits.credits(cfg))


# ================================================================== cleanup
def _done_project(tmp_path):
    cfg = _cfg(tmp_path)
    M, D = cfg.mintpy_dir, cfg.project_dir / "deliverables"
    (M / "geo").mkdir(parents=True, exist_ok=True)
    D.mkdir(parents=True, exist_ok=True)
    return cfg, M, D


def test_cleanup_lists_each_retired_mintpy_run_and_deletes_it_only_on_a_finished_project(tmp_path, monkeypatch):
    """ops-1 (L2 -> scripts/cleanup.py)."""
    from insar_pipeline.scripts import cleanup
    cfg, M, D = _done_project(tmp_path)
    run = M / "superseded" / "run_20261001_101010"
    (run / "geo").mkdir(parents=True)
    (run / "geo" / "geo_velocity.h5").write_bytes(b"x" * 10)
    (run / "RETIRED.json").write_text(json.dumps({"stamp": "20261001_101010", "files": ["geo/geo_velocity.h5"],
                                                  "why": "MintPy re-run"}))
    (M / "geo" / "geo_velocity.h5").write_bytes(b"current")
    t = cleanup.targets(cfg, 1)
    hit = [(p, why) for _, p, why in t if p == run]
    assert hit and "1 files" in hit[0][1] and "accepted" in hit[0][1]
    assert not any(p == M / "superseded" for _, p, _ in t)                   # listed once, by run
    assert not any(M / "geo" in (p, *p.parents) for _, p, _ in t)            # never the current geo/
    assert cleanup.needs_finished(cfg, 1, run) and not cleanup.needs_finished(cfg, 1, M / "pic")


def test_cleanup_never_targets_releases_reviews_or_the_point_products(tmp_path):
    """techout-5 (L7b) + techout-7 (L7a) -> scripts/cleanup.py."""
    from insar_pipeline.scripts import cleanup
    cfg, M, D = _done_project(tmp_path)
    for rel in ("releases/20261005_1200/client/1_report.html", "reviews/review_x.json", "config_snapshot.yaml",
                "points_attributes.gpkg", "points_timeseries.parquet", "measurement_points_egms.gpkg"):
        (D / rel).parent.mkdir(parents=True, exist_ok=True)
        (D / rel).write_bytes(b"x")
    items = cleanup.targets(cfg, 3)
    paths = [p for _, p, _ in items]
    for keep in ("releases", "reviews", "config_snapshot.yaml", "points_attributes.gpkg", "points_timeseries.parquet"):
        assert not any(D / keep in (p, *p.parents) for p in paths), keep
    old = [lv for lv, p, _ in items if p.name == "measurement_points_egms.gpkg"]
    assert old == [1]                       # the broken-geometry GPKG is superseded once the new one exists


def test_cleanup_keeps_the_deramp_pair_until_the_qa_has_measured_it(tmp_path):
    """methodology-14 (L8 -> scripts/cleanup.py)."""
    import time
    from insar_pipeline.scripts import cleanup
    cfg, M, D = _done_project(tmp_path)
    for n in ("timeseries.h5", "timeseries_ERA5.h5", "timeseries_ERA5_ramp.h5", "timeseries_ERA5_ramp_demErr.h5"):
        (M / n).write_bytes(b"x")
    names = {p.name for _, p, _ in cleanup.targets(cfg, 1)}
    assert "timeseries.h5" in names and not {"timeseries_ERA5.h5", "timeseries_ERA5_ramp.h5"} & names
    qa = cfg.project_dir / "quality_assessment.json"
    qa.write_text("{}")                                    # a newer QA that did not measure it (pre-5-Oct)
    t = time.time() + 5
    os.utime(qa, (t, t))
    names = {p.name for _, p, _ in cleanup.targets(cfg, 1)}
    assert not {"timeseries_ERA5.h5", "timeseries_ERA5_ramp.h5"} & names
    qa.write_text(json.dumps({"ramp_sensitivity": {"before_deramp": "measured: timeseries_ERA5.h5 minus ..."}}))
    os.utime(qa, (t, t))
    names = {p.name for _, p, _ in cleanup.targets(cfg, 1)}
    assert {"timeseries_ERA5.h5", "timeseries_ERA5_ramp.h5"} <= names


# ================================================================== quality assessment
import test_fix_L8 as L8                                         # noqa: E402  (the QA project builder)


@pytest.fixture
def no_osm(monkeypatch):
    from insar_pipeline import stack_qc

    def boom(*a, **k):
        raise AssertionError("the QA tried to fetch OSM water in a test")
    monkeypatch.setattr(stack_qc, "fetch_osm_water", boom, raising=False)


def test_qa_nulls_ignore_an_offset_on_the_ring_outside_the_box(tmp_path, no_osm):
    """stats-4 / gap-geolocation-5 (L1 -> quality_assessment.py): the mine and
    report-area nulls are placed on significance.placement_field."""
    from insar_pipeline import quality_assessment as qa
    lon, lat = L8._lonlat()
    meds, sds = [], []
    for off in (0.0, 3.0):
        rng = np.random.default_rng(5)
        vel = rng.normal(0, 1.0, (L8.NY, L8.NX))
        outside = (lat < 38.40)                           # rows south of the bbox: the outside ring
        vel[outside] += off
        cfg = L8.build(tmp_path / f"o{off}", vel_mm=vel, bbox=[-87.80, 38.40, -87.635, 38.47])
        r = qa.evaluate(cfg)
        sig = r["findings"]["workings_relative_to_far_field"]["significance"]
        meds.append(sig["null_median_mm_yr"]); sds.append(sig["farmland_spread_1sigma_mm_yr"])
    # on the plain class-matched field the +3 ring widened the spread 0.05 -> 1.41
    assert abs(meds[1] - meds[0]) < 0.3 and sds[1] < sds[0] + 0.2


def test_untestable_report_area_and_borrowed_classes_reach_the_send_rule(tmp_path, no_osm):
    """gap-geolocation-5 (L1 -> quality_assessment.py)."""
    from insar_pipeline import quality_assessment as qa
    tc = np.full((L8.NY, L8.NX), 0.75)
    tc[:, :2] = 0.45                                       # a sliver of a low class: 180 cells, too few far cells
    cfg = L8.build(tmp_path, tc=tc, report_areas=("Full Satellite Imaging:full",
                                                  "Huge:38.363,-87.799,38.469,-87.636",
                                                  "Edge:38.40,-87.80,38.44,-87.791"))
    r = qa.evaluate(cfg)
    at = r["area_testability"]
    assert at["Huge"]["testable"] is False and at["Huge"]["flag"]
    assert "0.4-0.6" in r["far_field"]["borrowed_classes"] and at["Edge"]["borrowed_classes"] == ["0.4-0.6"]
    sr = r["send_rule"]
    assert any("'Huge' cannot be compared" in h for h in sr["hold_reasons"])
    assert any("'Edge'" in n and "neighbouring class" in n for n in sr["notes"])


def test_disturbed_layer_effect_and_second_gate_counts_are_recorded(tmp_path, no_osm):
    """gap-surface_mine_site-1 (L2 -> quality_assessment.py)."""
    from insar_pipeline import quality_assessment as qa
    um = np.full((L8.NY, L8.NX), 0.8)
    um[:, :20] = 0.45
    cfg = L8.build(tmp_path, um=um)
    dis = np.zeros((L8.NY, L8.NX), np.int16)
    dis[:30, :] = 35
    dis[0, 0] = -1
    a = L8._attrs()
    L8._h5(cfg.mintpy_dir / "geo" / "geo_disturbed.h5", {"disturbed": dis},
           dict(a, pair_noise_level="0.92", pair_coherence_kind="phasor"))
    r = qa.evaluate(cfg)
    d = r["checks"]["coverage"]["disturbed"]
    assert d["available"] and d["pair_coherence_kind"] == "phasor"
    assert d["removed_percent_of_measured_aoi"]["10"] > 0 and d["removed_percent_of_measured_aoi"]["60"] == 0
    pf = r["checks"]["coverage"]["second_gate"]["points_failing"]
    assert pf["0.5"] > 0 and pf["0.7"] >= pf["0.5"] >= pf["0.4"]


def test_resolution_check_measures_the_smoothing_of_the_values(tmp_path, no_osm):
    """gap-geolocation-5 (L1 -> quality_assessment.py): lag-1 autocorrelation."""
    from scipy.ndimage import uniform_filter
    from insar_pipeline import quality_assessment as qa
    rng = np.random.default_rng(2)
    rough = rng.normal(0, 1.0, (L8.NY, L8.NX))
    smooth = uniform_filter(rough, 5) * 5
    r1 = qa.evaluate(L8.build(tmp_path / "a", vel_mm=rough))["resolution_check"]
    r2 = qa.evaluate(L8.build(tmp_path / "b", vel_mm=smooth))["resolution_check"]
    assert r1["available"] and r1["lag1_correlation"]["east_west"] < 0.3
    assert r2["lag1_correlation"]["east_west"] > r1["lag1_correlation"]["east_west"] + 0.3
    assert r2["half_correlation_distance_m"]["east_west"] > r1["half_correlation_distance_m"]["east_west"]


def test_reference_inside_the_mine_buffer_is_flagged(tmp_path, no_osm):
    """tests-2 (L6b -> quality_assessment.py)."""
    from insar_pipeline import quality_assessment as qa
    cfg = L8.build(tmp_path, acceptance_far_field_buffer_m=2000.0)
    assert qa.evaluate(cfg)["reference_distance"]["flag"] is False      # the corner, ~8.8 km away
    a = L8._attrs(REF_LAT="38.415", REF_LON="-87.655", REF_Y="46", REF_X="96")
    for name in ("geo_velocity.h5", "geo_timeseries_ERA5_ramp_demErr.h5"):
        with h5py.File(cfg.mintpy_dir / "geo" / name, "r+") as f:
            f.attrs.update(a)
    r = qa.evaluate(cfg)["reference_distance"]
    assert r["flag"] is True and r["distance_to_nearest_mapped_mine_km"] < 2.0


def test_removed_ramp_bigger_than_half_the_detection_limit_is_noted(tmp_path, no_osm):
    """gap-surface_mine_site-1 / timeseries-6 (L2 -> quality_assessment.py)."""
    from insar_pipeline import quality_assessment as qa
    cfg = L8.build(tmp_path, mintpy_deramp="linear")
    (cfg.mintpy_dir / "deramp_effect.json").write_text(json.dumps(
        {"applied": True, "gradient_mm_yr_per_10km": 40.0, "removed_rate_p2_mm_yr": -9, "removed_rate_p98_mm_yr": 9}))
    r = qa.evaluate(cfg)
    rs = r["ramp_sensitivity"]
    assert rs["removed_ramp"]["gradient_mm_yr_per_10km"] == 40.0 and rs["removed_gradient_flag"] is True
    assert any("tilt the deramp removed" in n for n in r["send_rule"]["notes"])


def test_date_accounting_reads_the_download_record_as_it_is_written(tmp_path):
    """stackrunner-2 / empirical-14 (L6a -> quality_assessment.py): the keys
    data_acquisition writes ('dropped' {date: code}, accepted_failures)."""
    from insar_pipeline import quality_assessment as qa
    cfg = _cfg(tmp_path)
    cfg.slc_dir.mkdir(parents=True)
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps(
        {"catalogue_dates": ["20180601", "20180613", "20180625", "20180707"],
         "dropped": {"20180613": "catalogue_short"}, "failed": {"20180707": "timeout"},
         "accepted_failures": ["20180707"], "accepted_missing": [], "processed": ["20180601", "20180625"]}))
    r = qa.date_accounting(cfg, ["20180601", "20180625"])
    assert r["available"] == 4 and "only part of the area's bursts" in r["dropped"]["20180613"]
    assert "accepted as missing" in r["dropped"]["20180707"] and r["unexplained"] == []


def test_stitch_check_reads_the_chain_record_as_it_is_written():
    """usability-6 / disp-4 (L10 -> quality_assessment.py)."""
    from insar_pipeline import quality_assessment as qa
    dc = {"stitch": {"epochs": 100, "lost_anchor_granules": ["20200101"], "corrections_applied": True,
                     "epochs_fully_corrected": 100, "corrections": {"status": "all"},
                     "coverage_ratchet": {"fired": False}, "boundary_steps": {},
                     "problems": [{"code": "skipped_granules", "text": "2 ordinary granules skipped", "hard": False}]},
          "anchor_suspect_cells": 300, "gap_fill": {"pixels_complete": 9000, "pixels_gap_filled": 1000,
                                                    "pixels_dropped_end_hold": 12},
          "troposphere": {"helped": False, "scatter_before_mm": 5.0, "scatter_after_mm": 6.0},
          "misclosure_vs_rate": {"spread_ratio_high_to_low": 1.4}}
    r = qa.stitch_check(dc)
    assert r["lost_anchors"] == ["20200101"] and r["pass"] is False
    assert any("anchor" in p for p in r["problems"])                     # 3% of cells > 1%
    assert any("ERA5" in n for n in r["notes"]) and any("held flat" in n for n in r["notes"])
    assert r["misclosure_vs_rate"]["spread_ratio_high_to_low"] == 1.4


def test_injection_and_cross_check_records_are_shown_never_gating(tmp_path):
    """methodology-11 (L10 -> quality_assessment.py)."""
    from insar_pipeline import quality_assessment as qa
    cfg = _cfg(tmp_path, cross_check_disp=True, injection_test=True)
    assert qa._injection_record(cfg) is None and qa._independent_check(cfg) is None
    T = cfg.project_dir / "deliverables" / "technical"
    T.mkdir(parents=True)
    (T / "injection_recovery.json").write_text(json.dumps({"status": "complete", "sentence": "s", "seed": 7,
                                                           "generated": _now(),
                                                           "summary": [{"width_km": 1.0, "detection_rate": 0.5}]}))
    (T / "cross_check.json").write_text(json.dumps({"status": "complete", "available": True, "indicative_pass": False,
                                                    "generated": _now(),
                                                    "one_km_means": {"correlation": 0.7},
                                                    "areas": [{"name": "Site", "agree": True}]}))
    inj, cc = qa._injection_record(cfg), qa._independent_check(cfg)
    assert inj["seed"] == 7 and inj["gates_verdict"] is False
    assert cc["one_km_correlation"] == 0.7 and cc["areas_agree"] == {"Site": True} and cc["gates_verdict"] is False
    assert "injection_recovery" not in qa.send_rule({"checks": {}, "injection_recovery": inj})["reasons"]


# ================================================================== deliverables
import test_fix_L7a as L7a                                       # noqa: E402  (the deliverables project builder)


@pytest.fixture
def fresh_cache():
    from insar_pipeline import deliverables as dl
    dl.clear_cache()
    yield
    dl.clear_cache()


def test_the_cube_is_streamed_and_reads_the_same_values(tmp_path, monkeypatch, fresh_cache):
    """generality-12 (L6a2 -> deliverables.py): no export holds the whole
    cube; every value equals the in-memory computation."""
    from insar_pipeline import deliverables as dl
    cfg, truth = L7a.make_project(tmp_path)
    ts = truth["ts"]
    ts[3:7, 5:9, 10:14] = np.nan                                     # gaps: first/last valid epochs differ
    with h5py.File(dl.timeseries_file(cfg), "r+") as f:
        f["timeseries"][...] = ts
    dates, cube, _ = dl._dates_and_series(cfg)
    assert isinstance(cube, dl._Cube) and cube.shape == ts.shape
    with pytest.raises(TypeError):
        np.asarray(cube)                                           # never built whole by accident
    for a, b in zip(dl._anchors(cube), dl._anchors(ts)):
        assert np.array_equal(a, b, equal_nan=True)
    assert np.allclose(dl._valid_fraction(cube), np.isfinite(ts).mean(axis=0))
    rng = np.random.default_rng(0)
    r, c = rng.integers(0, L7a.NY, 500), rng.integers(0, L7a.NX, 500)
    monkeypatch.setattr(dl._Cube, "BAND_BYTES", 2 * L7a.NX * len(dates) * 4)   # several bands
    assert np.array_equal(cube[:, r, c], ts[:, r, c], equal_nan=True)
    # the largest single read any cube-reading export makes is a band, not the cube
    reads = []
    real = h5py.Dataset.__getitem__

    def spy(self, key):
        out = real(self, key)
        if self.name == "/timeseries":
            reads.append(np.asarray(out).size)
        return out
    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    D = cfg.project_dir / "deliverables"
    dl.export_cumulative_displacement(cfg, D)
    dl.plot_selected_points(cfg, D, n_points=20)
    dl.export_measurement_points(cfg, D)
    assert reads and max(reads) < ts.size


def test_extra_layers_are_delivered_with_their_meaning(tmp_path, fresh_cache):
    """phaselinked-6 / gap-surface_mine_site-6 (L2) and gap-surface_mine_site-8 /
    gap-geolocation-4 (L8) -> deliverables.py."""
    import rasterio
    from insar_pipeline import deliverables as dl
    cfg, truth = L7a.make_project(tmp_path)
    geo = cfg.mintpy_dir / "geo"
    a = truth["attrs"]
    shape = (L7a.NY, L7a.NX)
    _w(geo / "geo_temporalCoherence_mintpy.h5", {"temporalCoherence": np.full(shape, 0.8, np.float32)},
       dict(a, temporalCoherenceSource="MintPy network-inversion temporal coherence (unwrapping consistency)"))
    dem = np.zeros(shape, np.float32); dem[10, 10] = 30.0
    _w(geo / "geo_demErr.h5", {"dem": dem}, a)
    dis = np.full(shape, 12, np.int16); dis[0, :] = -1
    _w(geo / "geo_disturbed.h5", {"disturbed": dis}, dict(a, pair_noise_level="0.92", pair_coherence_kind="phasor"))
    with h5py.File(geo / "geo_velocity.h5", "r+") as f:               # one cliff no unwrapping can follow
        v = f["velocity"][:]; v[20, 30] = 2.0; f["velocity"][...] = v
    with h5py.File(geo / "geo_temporalCoherence.h5", "r+") as f:
        f.attrs["temporalCoherenceSource"] = "MiaplPy linked coherence"
    D = cfg.project_dir / "deliverables"
    out = {p.name for p in dl.export_extra_layers(cfg, D)}
    assert out == {"quality_unwrapping_consistency.tif", "dem_difference_m.tif",
                   "quality_position_uncertainty_m.tif", "disturbed_ground_runs.tif", "quality_too_fast_to_measure.tif"}
    with rasterio.open(D / "disturbed_ground_runs.tif") as src:
        assert src.dtypes[0] == "int16" and src.nodata == -1 and src.read(1)[0, 0] == -1
    with rasterio.open(D / "quality_position_uncertainty_m.tif") as src:
        assert src.read(1)[10, 10] == pytest.approx(30.0 / np.tan(np.radians(38.5)), rel=1e-4)
    with rasterio.open(D / "quality_too_fast_to_measure.tif") as src:
        flag = src.read(1)
        assert flag[20, 30] == 1 and (flag == 1).sum() < 20
    with rasterio.open(D / "quality_unwrapping_consistency.tif") as src:
        assert "unwrapping consistency" in src.tags()["COHERENCE_SOURCE"]
    dl.export_quality(cfg, D)
    with rasterio.open(D / "quality_temporal_coherence.tif") as src:
        assert src.tags()["COHERENCE_SOURCE"] == "MiaplPy linked coherence"
    assert (PKG / "deliverables.py").read_text().count("metres per year is normal") == 0


def test_package_classifies_the_new_files_and_skips_failed_writes(tmp_path):
    """techout-7 (L7a -> package.py)."""
    from insar_pipeline import package
    cfg = _cfg(tmp_path)
    D = cfg.project_dir / "deliverables"
    D.mkdir(parents=True)
    for n in ("vertical_movement_in_per_yr.tif", "quality_unwrapping_consistency.tif", "disturbed_ground_runs.tif",
              "quality_too_fast_to_measure.tif", "dem_difference_m.tif", "points_attributes.gpkg",
              "points_timeseries.parquet", "points_README.txt", "points_attributes.gpkg.INVALID",
              "points_timeseries.partial.parquet", "los_velocity_cell_noise_1sigma_in_per_year.tif"):
        (D / n).write_bytes(b"x")
    copies, not_copied, unclassified = package._plan(cfg, {}, draft=True)
    dest = {Path(s).name: d for s, d, _, _ in copies}
    assert dest["quality_unwrapping_consistency.tif"].startswith("data/")
    assert dest["disturbed_ground_runs.tif"].startswith("data/")
    assert dest["dem_difference_m.tif"] == "technical/rasters/dem_difference_m.tif"
    assert dest["points_attributes.gpkg"] == "technical/points/points_attributes.gpkg"
    assert "points_attributes.gpkg.INVALID" not in dest and "points_timeseries.partial.parquet" not in dest
    assert not unclassified
    meaning = {Path(s).name: m for s, _, _, m in copies}
    assert "not a map of disturbed ground on its own" in meaning["disturbed_ground_runs.tif"]


# ================================================================== stack backend / acquisition
from insar_pipeline import stack_backend as sb                   # noqa: E402
import test_stack_chain as TSC                                    # noqa: E402


def test_workers_are_clamped_to_the_memory_available_now(tmp_path, monkeypatch, caplog):
    """generality-13 (L4a -> stack_backend.py)."""
    from insar_pipeline import settings
    cfg = TSC.make_cfg(tmp_path, stack_num_proc=12, stack_num_proc_unwrap=24)
    # Kyle's 24 cores, so the core clamp stays out of a test of the memory clamp
    # (a 4-core GitHub runner otherwise caps the unwrap at 4; 30 Oct 2026)
    monkeypatch.setattr(sb.os, "cpu_count", lambda: 24)
    monkeypatch.setattr(settings, "machine_resources", lambda: {"mem_available_gb": 20.0})
    assert sb._workers_for(cfg, "run_12_merge_reference_secondary_slc") == 12      # Kyle's machine: unchanged
    assert sb._workers_for(cfg, "run_16_unwrap") == 24
    monkeypatch.setattr(settings, "machine_resources", lambda: {"mem_available_gb": 4.0})
    assert sb._workers_for(cfg, "run_12_merge_reference_secondary_slc") == 2       # 0.8 x 4 / 1.3
    assert sb._workers_for(cfg, "run_16_unwrap") == 5


def test_any_failed_unwrap_is_quarantined_and_a_kill_is_named(tmp_path, monkeypatch):
    """stackrunner-8 (L6b) + generality-13 (L4a) -> stack_backend.py."""
    cfg = TSC.make_cfg(tmp_path, skip_existing=False)
    pair = cfg.stack_dir / "merged" / "interferograms" / "20180611_20180623"
    pair.mkdir(parents=True)
    (pair / "filt_fine.unw").write_bytes(b"old settings")
    (pair / "filt_fine.unw.conncomp").write_bytes(b"old")
    conf = cfg.stack_dir / "configs" / "config_igram_unw_20180611_20180623"
    conf.write_text(f"[Common]\nunw : {pair / 'filt_fine.unw'}\n")

    class P:
        def __init__(self, rc):
            self.pid, self.returncode, self.args = 999999, rc, "x"

        def communicate(self, timeout=None):
            return "", "Killed"
    monkeypatch.setattr(sb._procs, "popen_group", lambda *a, **k: P(-9))
    monkeypatch.setattr(sb._procs, "unregister", lambda pid: None)
    rec = sb._run_one(cfg, "run_16_unwrap", 0, f"SentinelWrapper.py -c {conf}")
    assert rec["reason"] == "killed (most likely out of memory)"
    q = cfg.stack_dir / "quarantine_unwrap_timeout" / "20180611_20180623"
    assert rec["quarantined"] and (q / "filt_fine.unw").exists() and not (pair / "filt_fine.unw").exists()
    # nothing left to move: the REASON is still written (it used to raise)
    shutil_q = q.parent / "20180611_20180705"
    assert sb.quarantine_unwrap(cfg, f"SentinelWrapper.py -c {cfg.stack_dir / 'configs' / 'none'}", "x") is None
    conf2 = cfg.stack_dir / "configs" / "config_igram_unw_20180611_20180705"
    conf2.write_text(f"[Common]\nunw : {cfg.stack_dir / 'merged' / 'interferograms' / '20180611_20180705' / 'filt_fine.unw'}\n")
    (cfg.stack_dir / "merged" / "interferograms" / "20180611_20180705").mkdir()
    assert sb.quarantine_unwrap(cfg, f"SentinelWrapper.py -c {conf2}", "no output") == shutil_q
    assert (shutil_q / "REASON.txt").exists()


def test_phase_linked_merge_is_fail_fast(tmp_path, monkeypatch):
    """generality-13 (L4a -> stack_backend.py): every merged SLC is required.
    A regression guard: the merge already counts as coregistration chain
    (_passed_coreg), so it was fail-fast before; this keeps it so."""
    cfg = TSC.make_cfg(tmp_path, backend="miaplpy", stack_virtual_merge="False",
                       stack_cleanup_coreg_after_filter=False, stack_crop_unwrap=False)
    (cfg.stack_dir / "run_files" / "run_12_merge_reference_secondary_slc").write_text("x\n")
    seen = {}
    monkeypatch.setattr(sb, "execute_run_file", lambda c, f, fail_fast, after=None: seen.setdefault(
        f.name, fail_fast) and {"run_file": f.name} or {"run_file": f.name})
    for name in ("_check_merged_dem", "cleanup_safes", "cleanup_burst_geometry", "_record_timing"):
        monkeypatch.setattr(sb, name, lambda *a, **k: None)
    sb.execute(cfg)
    assert seen["run_12_merge_reference_secondary_slc"] is True


def test_an_off_site_fallback_polygon_is_refused(tmp_path, monkeypatch):
    """timeseries-2 / -6 (L2 -> stack_backend.build_deramp_mask)."""
    # a site far from the package's own (Gibson) polygons, so no fallback lies on it
    cfg = TSC.make_cfg(tmp_path, mintpy_deramp="linear", deramp_exclude_geojson="", bbox=[-100.2, 40.0, -100.0, 40.2])
    ctx = cfg.project_dir / "deliverables" / "context"
    ctx.mkdir(parents=True)
    ring = [[-90.0, 30.0], [-89.9, 30.0], [-89.9, 30.1], [-90.0, 30.1], [-90.0, 30.0]]
    (ctx / "mine_active_far.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                                                    "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
    _w(cfg.mintpy_dir / "maskTempCoh.h5", {"mask": np.ones((10, 10), bool)},
       {"X_FIRST": -100.2, "Y_FIRST": 40.2, "X_STEP": 0.02, "Y_STEP": -0.02, "LENGTH": 10, "WIDTH": 10})
    assert sb._geojson_overlaps_bbox(ctx / "mine_active_far.geojson", cfg.bbox) is False
    with pytest.raises(FileNotFoundError, match="outside the processing box"):
        sb.build_deramp_mask(cfg)


def test_era5_lag_is_named_in_the_plan_and_products_ram_follows_cells(tmp_path):
    """timeseries-2 (L2) + generality-12 (L6a2) -> stack_backend.py."""
    import datetime as dt
    cfg = TSC.make_cfg(tmp_path, end_date="2026-10-02")
    note = sb.era5_lag_note(cfg, today=dt.date(2026, 10, 5))
    assert note and "2026-09-29" in note and "era5_allow_partial" in note
    assert sb.era5_lag_note(TSC.make_cfg(tmp_path / "b", end_date="2026-09-20"), today=dt.date(2026, 10, 5)) is None
    cfg.era5_allow_partial = True
    assert sb.era5_lag_note(cfg, today=dt.date(2026, 10, 5)) is None
    assert sb.products_ram_gb(3_490_000, 270) < 6 and sb.products_ram_gb(14_000_000, 270) > 12


def test_bursts_that_miss_part_of_a_report_area_stop_the_download(tmp_path):
    """generality-11 (L4b -> data_acquisition.py): checked before anything is fetched."""
    from insar_pipeline import data_acquisition as da
    poly = tmp_path / "east.geojson"
    poly.write_text(json.dumps({"type": "Polygon", "coordinates": [[[-87.56, 38.3], [-87.53, 38.3], [-87.53, 38.35],
                                                                    [-87.56, 38.35], [-87.56, 38.3]]]}))
    cfg = TSC.make_cfg(tmp_path, report_areas=("Full Satellite Imaging:full", f"East:{poly}"))
    full = {"a": [[-87.9, 38.1], [-87.4, 38.1], [-87.4, 38.6], [-87.9, 38.6], [-87.9, 38.1]]}
    ok = da.burst_coverage_check(cfg, full)
    assert ok["shares"]["processing box"] == 1.0 and ok["shares"]["East"] == 1.0
    west = {"a": [[-87.9, 38.1], [-87.57, 38.1], [-87.57, 38.6], [-87.9, 38.6], [-87.9, 38.1]]}
    with pytest.raises(RuntimeError, match="East"):
        da.burst_coverage_check(cfg, west)


# ================================================================== phase linking / unwrapping / significance
def test_phase_linking_refuses_a_stack_missing_merged_dates_or_shorter_than_a_ministack(tmp_path):
    """stackrunner-1 (L6a) + phaselinked-11 (L4a) -> miaplpy_backend.py."""
    from insar_pipeline import miaplpy_backend as mb
    cfg = _cfg(tmp_path, miaplpy_ministack_size=3)
    for d in ("20230104", "20230116", "20230128", "20230209"):
        f = cfg.stack_dir / "merged" / "SLC" / d / f"{d}.slc.full"
        f.parent.mkdir(parents=True); f.write_bytes(b"x")
    work = mb.work_dir(cfg); (work / "inputs").mkdir(parents=True)
    _w(work / "inputs" / "slcStack.h5", {"slc": np.ones((3, 4, 4), np.complex64),
                                          "date": np.array([b"20230104", b"20230116", b"20230128"])}, {})
    with pytest.raises(RuntimeError, match="20230209"):
        mb.check_stack_dates(cfg)
    (cfg.stack_dir / "merged" / "SLC" / "20230209" / "20230209.slc.full").unlink()
    assert mb.check_stack_dates(cfg)["missing_from_stack"] == []
    cfg.miaplpy_ministack_size = 10
    with pytest.raises(RuntimeError, match="fewer than one mini-stack"):
        mb.check_stack_dates(cfg)


def test_unknown_unwrap_engine_raises_instead_of_falling_to_isce(tmp_path):
    """generality-6 (L4a -> unwrap_crop.py)."""
    from insar_pipeline import unwrap_crop
    assert unwrap_crop.engine_lines(types.SimpleNamespace(stack_unw_engine="isce")) == {}
    with pytest.raises(ValueError, match="snaphu"):
        unwrap_crop.engine_lines(types.SimpleNamespace(stack_unw_engine="snaphu_2"))


def test_a_configured_mine_layer_that_is_missing_raises(tmp_path):
    """generality-9 (L4b -> significance.panel_mask)."""
    from insar_pipeline import significance as sg
    a = {"X_FIRST": -87.8, "Y_FIRST": 38.47, "X_STEP": 0.01, "Y_STEP": -0.01}
    assert not sg.panel_mask("", (5, 5), a).any()
    with pytest.raises(FileNotFoundError, match="treated as ordinary ground"):
        sg.panel_mask(str(tmp_path / "gone.geojson"), (5, 5), a)


# ================================================================== GNSS tie / builder helpers
import test_fix_L1 as L1T                                        # noqa: E402  (the L1 tie fixture)


@pytest.fixture
def project(tmp_path, monkeypatch):
    return L1T.project.__wrapped__(tmp_path, monkeypatch)


def test_tie_records_when_each_station_series_was_downloaded(project, monkeypatch, tmp_path):
    """techout-4 (L7b -> gnss_tie.py): the cache is never refreshed, so its date is recorded."""
    import time
    from insar_pipeline import gnss_tie as gt
    f = tmp_path / "AAAA.tenv3"
    f.write_text("x" * 2000)
    t = time.mktime((2024, 3, 1, 12, 0, 0, 0, 0, -1))
    os.utime(f, (t, t))
    monkeypatch.setattr(gt, "fetch_tenv3", lambda site, root, timeout=180: f)
    monkeypatch.setattr(gt, "read_tenv3", lambda p: L1T._tenv3("AAAA"))
    out = gt.run(project.cfg)
    st = [s for s in out["stations"] if s["site"] == "AAAA"][0]
    assert st["data_file"] == str(f) and st["data_file_modified"].startswith("2024-03-01")


def test_builder_gnss_preview_says_which_stations_the_tie_will_use(project, monkeypatch):
    """generality-3 (L1 -> areas.py/map_builder.py): not 'within 50 km ... anchored to them'."""
    from insar_pipeline import areas, gnss_tie as gt, settings
    monkeypatch.setattr(settings, "projects_root", lambda: project.cache.parent)
    a = areas.box(38.95, -87.35, 39.05, -87.25, "site")
    g = areas.gnss_preview(a, "2016-01-01", "2026-06-30")
    pv = gt.preview_stations(39.0, -87.3, "20160101", "20260630", cache_root=project.cache)
    assert g["used"] == [s["site"] for s in pv if s["will_use"]] and "AAAA" in g["used"]
    assert g["not_used"] and all(s.get("lat") is not None for s in g["stations"])
    assert g["warning"] and "Only 1 station" in g["warning"]
    html = (PKG / "project_builder.html").read_text()
    assert "The report is anchored to them" not in html and "Used for the GNSS anchor" in html


def test_far_field_check_warns_when_the_box_is_mostly_mine(tmp_path):
    """generality-2 (L1 -> areas.py): suggest pad_km before a 1-2 day run."""
    from insar_pipeline import areas
    aoi = areas.box(38.30, -87.70, 38.34, -87.64, "Site")
    mine = areas.box(38.28, -87.72, 38.36, -87.62, "mine")
    ok = areas.far_field_check(aoi.padded(10), None, [("Full", "full"), ("Site", aoi)])
    assert ok["warning"] is None and ok["ratio"] > 10
    bad = areas.far_field_check(aoi.padded(1), mine, [("Full", "full"), ("Site", aoi)])
    assert bad["ratio"] < 10 and "pad_km" in bad["warning"]


def test_frame_for_track_picks_overlap_on_the_chosen_track():
    """disp-6 (L10 -> areas.py / project_builder.html)."""
    from insar_pipeline import areas
    detail = [{"frame": "32248", "track": 121, "direction": "ascending", "granules": 300, "overlap_fraction": 1.0},
              {"frame": "04862", "track": 19, "direction": "ascending", "granules": 224, "overlap_fraction": 0.97},
              {"frame": "04861", "track": 19, "direction": "ascending", "granules": 254, "overlap_fraction": 0.6}]
    assert areas.frame_for_track(detail, 19, "ascending")["frame"] == "04862"
    assert areas.frame_for_track(detail, 19, "descending") is None
    js = (PKG / "project_builder.html").read_text()
    assert "disp_frame: frames[0]" not in js and "function dispFrame()" in js


def _write_project(tmp_path, monkeypatch, route="miaplpy", **kw):
    from insar_pipeline import areas
    monkeypatch.setenv("INSAR_PROJECTS", str(tmp_path / "p"))
    aoi = areas.box(38.30, -87.70, 38.34, -87.64, "site")
    return areas.write_project("My Test Site", aoi, route, "2020-01-01", "2024-12-31", track=19,
                               direction="ascending", disp_frame="04861", swaths=[2],
                               report_areas=[("Full Satellite Imaging", "full"), ("Site", aoi)],
                               projects_root=tmp_path / "p", configs_dir=tmp_path / "c", **kw)


def test_new_phase_linked_projects_carry_no_dead_or_misleading_settings(tmp_path, monkeypatch):
    """generality-17 (L3b) + gap-surface_mine_site-8 (L8) -> areas.write_project."""
    import yaml
    raw = yaml.safe_load(_write_project(tmp_path, monkeypatch).read_text())
    assert "max_perp_baseline" not in raw and "max_temp_baseline" not in raw
    assert "max_plausible_velocity_mm_yr" not in raw and "disp_client_report" not in raw


def test_dated_mine_layer_reaches_panel_history_and_the_mine_mask(tmp_path, monkeypatch):
    """generality-10 (L9c -> areas.py / map_builder.py): the years are kept."""
    from insar_pipeline import areas, panel_history as ph
    ring = [[-87.69, 38.31], [-87.67, 38.31], [-87.67, 38.33], [-87.69, 38.33], [-87.69, 38.31]]
    far = [[-80.0, 30.0], [-79.9, 30.0], [-79.9, 30.1], [-80.0, 30.1], [-80.0, 30.0]]
    up = tmp_path / "pits.geojson"
    up.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"PIT": "North pit", "YR_MINED": "2019-05-01"},
         "geometry": {"type": "Polygon", "coordinates": [ring]}},
        {"type": "Feature", "properties": {"PIT": "Far pit", "YR_MINED": 2018},
         "geometry": {"type": "Polygon", "coordinates": [far]}}]}))
    f = areas.attribute_fields(up)
    assert f["year_guess"] == "YR_MINED" and f["name_guess"] == "PIT"
    with pytest.raises(ValueError, match="no attribute 'YEAR'"):
        areas.dated_features(up, "YEAR", "PIT")
    fc = areas.dated_features(up, "YR_MINED", "PIT")
    cfg_path = _write_project(tmp_path, monkeypatch, dated=fc)
    cfg = load_config(str(cfg_path))
    assert cfg.mined_year_field == "mined_year" and cfg.mined_name_field == "mine_name"
    kept = json.loads(Path(cfg.mined_year_geojson).read_text())["features"]
    assert [k["properties"]["mine_name"] for k in kept] == ["North pit"]          # cut to the processing box
    feats, why = ph._load_dated(cfg)
    assert not why and feats[0]["properties"] == {"mine": "North pit", "year": 2019}
    assert cfg.deramp_exclude_geojson and any("Dated mined areas" in s for s in cfg.mine_layer_sources)


def test_builder_check_and_create_carry_gnss_ram_and_dated_layers(tmp_path, monkeypatch):
    """generality-3 / generality-12 / generality-10 / timeseries-2 (-> map_builder.py)."""
    from insar_pipeline import areas, map_builder as mb, runs
    monkeypatch.setenv("INSAR_PROJECTS", str(tmp_path / "p"))
    monkeypatch.setenv("INSAR_SETTINGS", str(tmp_path / "s.json"))
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    monkeypatch.setattr(mb, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(areas, "archive_check", lambda a, s, e, box=None: {
        "tracks": [{"direction": "ASCENDING", "track": 19, "dates": 270, "by_year": {}}], "disp_frames": {},
        "disp_frame_detail": [], "box_seen": box})
    monkeypatch.setattr(areas, "gnss_preview", lambda a, s, e, r=None: {
        "stations": [{"site": "INLN", "lat": 38.5, "lon": -87.4, "distance_km": 20.0, "will_use": True},
                     {"site": "FAR1", "lat": 38.9, "lon": -87.0, "distance_km": 48.0, "will_use": False}],
        "used": ["INLN"], "not_used": ["FAR1"], "warning": "Only 1 station(s) can anchor the report",
        "source": "test"})
    monkeypatch.setattr(areas, "burst_coverage", lambda *a, **k: {"swaths": [2], "bursts": 3, "covered": 1.0,
                                                                   "targets": {"area": 1.0}})
    aoi = areas.box(38.30, -87.70, 38.34, -87.64, "site")
    r = mb.api_check({"area": {"role": "area", "geojson": aoi.geojson()}, "pad_km": 5})
    assert [g["will_use"] for g in r["gnss"]] == [True, False] and r["gnss_warning"]
    assert r["ram"]["est_ram_gb"] > 1.5 and "GB" in r["ram"]["text"]
    assert mb.api_info({})["default_end"] < mb.api_info({})["today"]
    ring = [[-87.69, 38.31], [-87.67, 38.31], [-87.67, 38.33], [-87.69, 38.33], [-87.69, 38.31]]
    dated = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"Y": 2017, "N": "Pit A"},
                                                        "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    d = mb.api_create({"name": "Test Site", "route": "miaplpy", "start": "2020-01-01", "end": "2024-12-31",
                       "track": 19, "direction": "ascending", "pad_km": 5,
                       "layers": [{"role": "area", "name": "", "geojson": aoi.geojson()},
                                  {"role": "mine", "kind": "dated", "name": "pits", "geojson": dated,
                                   "year_field": "Y", "name_field": "N"}]})
    cfg = load_config(d["config"])
    assert d["dated_polygons"] == 1 and cfg.mined_year_geojson and d["far_field"]["ratio"] is not None
    saved = json.loads((Path(cfg.project_dir) / "context" / "builder_layers.json").read_text())["layers"]
    dl_ = [L for L in saved if L.get("kind") == "dated"][0]
    assert dl_["year_field"] == "mined_year" and dl_["geojson"]["features"][0]["properties"]["mine_name"] == "Pit A"


# ================================================================== operations
def test_cli_runs_the_optional_checks_under_the_project_lock(tmp_path, monkeypatch, capsys):
    """methodology-11 (L10 -> cli.py) + usability-13 (version)."""
    from insar_pipeline import cli, injection_test, runs
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(cli, "load_cfg", lambda p: cfg)
    seen = {}
    monkeypatch.setattr(injection_test, "main", lambda argv: seen.setdefault("held", runs.lock_holder(cfg.project_dir)) and 0)
    assert cli.tool_main("injection-test", ["c.yaml", "--sites", "2"]) == 0
    assert seen["held"]["kind"] == "injection_test"
    assert cli.tool_main("cross-check", []) == 2
    with runs.project_lock(cfg.project_dir, kind="full"):
        import subprocess
        p = subprocess.run([sys.executable, "-c",
                            "import sys; from insar_pipeline import cli, config\n"
                            f"cli.load_cfg = lambda p: config.PipelineConfig(project_dir=__import__('pathlib').Path({str(cfg.project_dir)!r}),"
                            " bbox=[-87.8, 38.2, -87.52, 38.47], start_date='2018-06-01', end_date='2019-06-30')\n"
                            "sys.exit(cli.tool_main('injection-test', ['c.yaml']))"],
                           cwd=str(PKG.parent), capture_output=True, text=True, timeout=60)
        assert p.returncode == 3, p.stderr
    cli.main(["--version"])
    assert "insar_pipeline" in capsys.readouterr().out


def test_advice_names_the_disp_refusals_in_plain_words():
    """disp-4 (L10 -> advice.py)."""
    from insar_pipeline import advice
    a = advice.explain(RuntimeError("DISP-S1 stitching is not usable:\n  - corrections partial"), "processing")
    assert a["category"] == "disp_stitch" and "allow_missing_dates" in a["next_action"]
    b = advice.explain(RuntimeError("Refusing to write DISP-S1 products into /x/mintpy: holds ifgramStack.h5"), "x")
    assert b["category"] == "refused" and "_disp_screening" in b["next_action"]


def test_disp_correction_intent_is_not_recorded_as_done(tmp_path):
    """disp-4 (L10 -> corrections.py)."""
    from insar_pipeline import corrections
    cfg = _cfg(tmp_path, backend="disp", disp_apply_corrections=False)
    d = corrections.decide(cfg)
    assert d["solid_earth_tides"] is False and d["ionosphere"] is None and "NOT" in d["why"]
    cfg.disp_apply_corrections = True
    assert corrections.decide(cfg)["requested_only"] is True and "always" not in corrections.decide(cfg)["why"]
    rec = json.loads(corrections.record(cfg, extra={"disp_corrections": {"status": "partial"},
                                                    "solid_earth_tides": False}).read_text())
    assert "requested_only" not in rec and rec["solid_earth_tides"] is False


def test_doctor_lists_projects_whose_patch_check_needs_their_own_overlap(tmp_path):
    """phaselinked-2 (L6b -> doctor.py)."""
    from insar_pipeline import doctor
    (tmp_path / "a.yaml").write_text("backend: miaplpy\nmiaplpy_time_lag: 1\n")
    (tmp_path / "b.yaml").write_text("backend: miaplpy\n")
    (tmp_path / "c.yaml").write_text("backend: disp\nmiaplpy_time_lag: 0\n")
    assert doctor.miaplpy_settings_in(tmp_path) == {(10, 1): ["a.yaml"]}


def test_resume_reason_names_the_coregistered_dates(tmp_path, monkeypatch):
    """ops-2 (L6a -> runs.py)."""
    from insar_pipeline import runs
    cfg = _cfg(tmp_path)
    S = cfg.stack_dir
    (S / "configs").mkdir(parents=True)
    for d in ("20180611", "20180623", "20180705"):
        (S / "configs" / f"config_merge_{d}").write_text("x")
    (S / "coreg_secondarys" / "20180611").mkdir(parents=True)
    monkeypatch.setattr(sb, "coregistered_dates", lambda c, include_reference=True: {"20180611", "20180623"})
    kind, why = runs.suggest_kind(cfg.project_dir, cfg)
    assert kind == "full" and "2 dates are coregistered" in why and "allow_missing_dates" in why


def test_panel_history_rate_shares_the_zero_of_the_significance(tmp_path):
    """stats-3 (L1 -> panel_history.py): the printed rate uses the same class
    centres as the significance value (mean by default since 6 Oct 2026)."""
    import test_fix_L9c as L9c
    from insar_pipeline import panel_history as ph
    rate = np.zeros((L9c.NY, L9c.NX))
    rate[2:12, 2:20] = -0.03                            # an unmapped sinking patch in the far field (m/yr)
    cfg = L9c.make_project(tmp_path, rate=rate)
    ctx = ph._Context(cfg)
    m = np.zeros((L9c.NY, L9c.NX), bool); m[40:50, 40:50] = True
    series, far, _ = ctx.series({"x": m})
    s = ctx.summarise(series["x"], m, far)
    assert s["rate_mm_yr"] == pytest.approx(s["significance"]["value_mm_yr"], abs=0.3)
    assert "start_mm" in s and "total_at_average_rate_mm" in s
    # mean class centres by default (6 Oct 2026): the series needs no tilt to
    # match the map; with robust centres it would need one
    assert all(abs(v) < 1e-9 for v in ctx.class_tilt.values())


def test_coherence_caption_quotes_the_measured_noise_level():
    """gap-surface_mine_site-1 (L2 -> figures.py)."""
    from insar_pipeline import figures
    t = figures._noise_sentence({"noise_level": {"linked_noise_level_from_run": 0.387}}, 0.4)
    assert "0.387" in t and "+0.013" in t and "not reliable" in t
    # 7 Oct 2026: with the QA's measured floor (typical + 3 spreads) the
    # threshold is judged against the floor, and the typical level is the median
    t = figures._noise_sentence({"noise_level": {"median": 0.3856, "floor": 0.4359,
                                                 "linked_noise_level_from_run": 0.384}}, 0.4)
    assert "about 0.386" in t and "up to about 0.436" in t and "-0.036" in t and "not reliable" in t
    assert figures._noise_sentence({}, 0.4) == ""


def test_run_summary_page_gives_each_left_out_date_its_recorded_reason():
    """stackrunner-2 (L6a -> scripts/run_summary_page.py)."""
    src = (PKG / "scripts" / "run_summary_page.py").read_text()
    assert "only 2 of their 3 bursts" not in src and "assembly_status" in src
    from insar_pipeline.scripts import run_summary_page as rsp
    assert "part of the area's bursts" in rsp._DROP_WHY["catalogue_short"]
