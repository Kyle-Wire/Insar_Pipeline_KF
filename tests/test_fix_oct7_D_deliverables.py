"""7 Oct 2026 fixes, group D_deliverables (deliverables.py): findings 14, 15,
16, 18 and 21 of the 5 Oct review, and the deliverables side of the
mintpy/run_incomplete.json contract. Each test checks the OUTCOME on a
synthetic project (what is on disk, in summary.json, in the raster tags, or
raised) and failed on the code before these fixes.

Run: python -m pytest insar_pipeline/tests/test_fix_oct7_D_deliverables.py -q -p no:cacheprovider
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L7a as L7a                                       # noqa: E402  (the deliverables project builder)


@pytest.fixture(autouse=True)
def _clear_cache():
    from insar_pipeline import deliverables as dl
    dl.clear_cache()
    yield
    dl.clear_cache()


def _newer_than_velocity(cfg, path: Path):
    from insar_pipeline import deliverables as dl
    t = dl._product(cfg, "velocity.h5").stat().st_mtime + 5
    os.utime(path, (t, t))


# ------------------------------------------------------------------ 14. earlier runs' extra layers
def test_extra_layers_not_written_this_run_are_moved_aside(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    for n in ("dem_difference_m.tif", "quality_unwrapping_consistency.tif", "quality_position_uncertainty_m.tif"):
        (D / n).write_bytes(b"previous run")
    (D / "dem_difference_m.tif.aux.xml").write_text("<PAMDataset/>")
    # no geo_demErr.h5 / geo_temporalCoherence_mintpy.h5 this run (a failed geocode_extra_layers)
    out = {p.name for p in dl.export_extra_layers(cfg, D)}
    assert out == {"quality_too_fast_to_measure.tif"}
    for n in ("dem_difference_m.tif", "dem_difference_m.tif.aux.xml", "quality_unwrapping_consistency.tif",
              "quality_position_uncertainty_m.tif"):
        assert not (D / n).exists()
        moved = list((D / "superseded").glob(f"layers_*/{n}"))
        assert len(moved) == 1 and moved[0].read_bytes() != b""          # moved, never deleted
    assert (D / "quality_too_fast_to_measure.tif").exists()             # this run's own stays


def test_too_fast_layer_left_from_an_earlier_run_is_moved_aside_when_its_block_fails(tmp_path, monkeypatch):
    from insar_pipeline import deliverables as dl, stack_qc
    cfg, _ = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    (D / "quality_too_fast_to_measure.tif").write_bytes(b"previous run")

    def boom(*a, **k):
        raise ValueError("no dates")
    monkeypatch.setattr(stack_qc, "gradient_limit_mm_yr", boom)
    assert dl.export_extra_layers(cfg, D) == []
    assert not (D / "quality_too_fast_to_measure.tif").exists()
    assert list((D / "superseded").glob("layers_*/quality_too_fast_to_measure.tif"))


# ------------------------------------------------------------------ 15. summary 'gnss' follows the current tie
WINDOW = [L7a.DATES[0], L7a.DATES[-1]]
STALE_TIE = {"record_window": WINDOW, "stations_used": ["ABCD"],
             "background": {"vertical_mm_yr": -1.2, "sigma_mm_yr": 0.73}}
GOOD_TIE = {"record_window": WINDOW, "stations_used": ["ABCD"], "uncertainty_floor_mm_yr": 0.9,
            "background": {"vertical_mm_yr": -1.2, "sigma_mm_yr": 1.01, "uncertainty_basis": "regional floor"}}
NO_STATION_TIE = {"record_window": WINDOW, "background": None, "reason": "No GNSS station within 40 km"}


def _write_tie(cfg, tie: dict):
    p = cfg.project_dir / "deliverables" / "gnss_tie.json"
    p.write_text(json.dumps(tie))
    _newer_than_velocity(cfg, p)


def test_summary_gnss_line_is_rewritten_when_the_tie_is_rerun(tmp_path):
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path)
    stats = {"gnss": "skipped (gnss_validation is false)"}
    _write_tie(cfg, STALE_TIE)
    dl._apply_qa_and_gnss(cfg, stats)
    assert stats["gnss"].startswith("not tied to GNSS") and "re-run the GNSS tie" in stats["gnss"]
    _write_tie(cfg, GOOD_TIE)                                  # the tie re-run with the regional floor
    dl._apply_qa_and_gnss(cfg, stats)
    assert stats["gnss_tie"]["anchored"] is True
    assert stats["gnss"] == "tied to GNSS (gnss_tie.json)"
    _write_tie(cfg, NO_STATION_TIE)                            # and the reverse
    dl._apply_qa_and_gnss(cfg, stats)
    assert stats["gnss_tie"]["anchored"] is False
    assert stats["gnss"] == "not tied to GNSS: No GNSS station within 40 km"
    # a gnss_validation verdict is its own record, left alone
    verdict = {"gnss": "One station inside the AOI. Adding +1.00 mm/yr to every line-of-sight velocity ..."}
    dl._apply_qa_and_gnss(cfg, verdict)
    assert verdict["gnss"].startswith("One station inside the AOI")


# ------------------------------------------------------------------ 16. failed point products
def test_failed_attribute_gpkg_moves_the_old_one_aside_and_fails(tmp_path, monkeypatch):
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    (D / "points_attributes.gpkg").write_bytes(b"previous run")
    monkeypatch.setattr(dl, "_write_points_gpkg", lambda *a, **k: None)      # failed its read-back check
    with pytest.raises(RuntimeError, match="points_attributes.gpkg"):
        dl.export_measurement_points(cfg, D)
    assert not (D / "points_attributes.gpkg").exists()
    moved = list((D / "superseded").glob("points_*/points_attributes.gpkg"))
    assert len(moved) == 1 and moved[0].read_bytes() == b"previous run"
    assert (D / "points_timeseries.parquet").exists()                       # this run's, written


def test_failed_timeseries_parquet_moves_the_old_one_aside_and_fails(tmp_path, monkeypatch):
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    (D / "points_timeseries.parquet").write_bytes(b"previous run")

    def boom(*a, **k):
        raise OSError("No space left on device")
    monkeypatch.setattr(dl, "_write_points_timeseries", boom)
    with pytest.raises(RuntimeError, match="points_timeseries.parquet"):
        dl.export_measurement_points(cfg, D)
    assert not (D / "points_timeseries.parquet").exists()
    assert list((D / "superseded").glob("points_*/points_timeseries.parquet"))
    assert (D / "points_attributes.gpkg").exists()


def _stub_exports(monkeypatch, dl, **over):
    for name in ("export_velocity", "export_velocity_uncertainty", "export_empirical_uncertainty",
                 "export_cell_noise", "export_signal_to_noise", "export_vertical_velocity",
                 "export_quality", "export_extra_layers", "export_cumulative_displacement",
                 "plot_displacement_series", "plot_selected_points", "export_kmz_points",
                 "export_measurement_points", "export_coverage_curve"):
        monkeypatch.setattr(dl, name, over.get(name, lambda *a, **k: None))
    monkeypatch.setattr(dl, "summary_stats", lambda cfg: {})
    monkeypatch.setattr(dl.provenance, "write", lambda *a, **k: None)


def test_export_all_finishes_the_record_then_fails_on_a_failed_point_product(tmp_path, monkeypatch):
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path, gnss_validation=False)

    def failed(*a, **k):
        raise dl.PointProductError("Point product(s) not written: points_attributes.gpkg")
    _stub_exports(monkeypatch, dl, export_measurement_points=failed)
    with pytest.raises(RuntimeError, match="points_attributes.gpkg"):
        dl.export_all(cfg)
    assert (cfg.project_dir / "deliverables" / "summary.json").exists()   # written before the stage fails


def test_export_all_still_stops_at_once_on_any_other_point_error(tmp_path, monkeypatch):
    """Only a failed point-product WRITE is held to the end; anything else
    raised there (an unreadable velocity, a bad mask) stops export_all as before."""
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path, gnss_validation=False)

    def broken(*a, **k):
        raise RuntimeError("mask unreadable")
    _stub_exports(monkeypatch, dl, export_measurement_points=broken)
    with pytest.raises(RuntimeError, match="mask unreadable"):
        dl.export_all(cfg)
    assert not (cfg.project_dir / "deliverables" / "summary.json").exists()


def test_no_accepted_point_moves_the_earlier_point_products_aside(tmp_path, monkeypatch):
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path)
    D = cfg.project_dir / "deliverables"
    for n in ("points_attributes.gpkg", "points_timeseries.parquet", "points_README.txt"):
        (D / n).write_bytes(b"previous run")
    monkeypatch.setattr(dl, "accepted_points", lambda cfg, vel: np.zeros(vel.shape, bool))
    assert dl.export_measurement_points(cfg, D) is None                    # nothing to export is not a failure
    for n in ("points_attributes.gpkg", "points_timeseries.parquet", "points_README.txt"):
        assert not (D / n).exists(), n
        assert list((D / "superseded").glob(f"points_*/{n}")), n


# ------------------------------------------------------------------ contract 1: mintpy/run_incomplete.json
def test_export_all_refuses_while_a_mintpy_run_is_incomplete(tmp_path, monkeypatch):
    from insar_pipeline import deliverables as dl
    cfg, _ = L7a.make_project(tmp_path, gnss_validation=False)
    _stub_exports(monkeypatch, dl)
    marker = cfg.project_dir / "mintpy" / "run_incomplete.json"
    marker.write_text(json.dumps({"stamp": "20261007_101500", "started": "2026-10-07T10:15"}))
    with pytest.raises(RuntimeError, match="did not finish") as e:
        dl.export_all(cfg)
    assert "2026-10-07T10:15" in str(e.value) and "Re-run the time series" in str(e.value)
    assert not (cfg.project_dir / "deliverables" / "summary.json").exists()
    marker.unlink()                                            # the run finished: export goes ahead
    dl.export_all(cfg)
    assert (cfg.project_dir / "deliverables" / "summary.json").exists()


# ------------------------------------------------------------------ 18. sign tags on rasters that are not motion
def test_non_motion_rasters_do_not_carry_the_line_of_sight_sign_note(tmp_path):
    import rasterio
    from insar_pipeline import deliverables as dl
    cfg, truth = L7a.make_project(tmp_path)
    geo = cfg.mintpy_dir / "geo"
    a = truth["attrs"]
    shape = (L7a.NY, L7a.NX)
    L7a._w(geo / "geo_temporalCoherence_mintpy.h5", {"temporalCoherence": np.full(shape, 0.8, np.float32)}, a)
    dem = np.zeros(shape, np.float32)
    dem[10, 10] = 30.0
    L7a._w(geo / "geo_demErr.h5", {"dem": dem}, a)
    L7a._w(geo / "geo_disturbed.h5", {"disturbed": np.full(shape, 12, np.int16)}, a)
    D = cfg.project_dir / "deliverables"
    written = dl.export_extra_layers(cfg, D) + [dl.export_quality(cfg, D)]
    assert len(written) == 6
    for p in written:
        with rasterio.open(p) as r:
            sign = r.tags()["SIGN_CONVENTION"]
        assert "satellite" not in sign, p.name
    with rasterio.open(D / "dem_difference_m.tif") as r:
        assert r.tags()["SIGN_CONVENTION"].startswith("Positive = ground above the elevation model")
    for f in (dl.export_velocity_uncertainty, dl.export_empirical_uncertainty, dl.export_cell_noise):
        p = f(cfg, D)
        assert p is not None, f.__name__
        with rasterio.open(p) as r:
            assert r.tags()["SIGN_CONVENTION"].startswith("unsigned"), p.name
    with rasterio.open(dl.export_velocity(cfg, D)) as r:                   # the motion raster keeps its note
        assert r.tags()["SIGN_CONVENTION"] == dl.SIGN_NOTE


# ------------------------------------------------------------------ 21. summary send rule sees the GNSS tie
def test_summary_send_rule_is_re_evaluated_with_the_gnss_tie(tmp_path):
    from insar_pipeline import deliverables as dl, quality_assessment as qa
    cfg, _ = L7a.make_project(tmp_path)
    P = cfg.project_dir
    Q = {"verdict": "REVIEW", "passed": 5, "flagged": 1,
         "checks": {"rate_precision": {"pass": False}}, "findings": {},
         "geolocation": {"flag": False}}
    Q["send_rule"] = qa.send_rule(Q)
    assert Q["send_rule"]["value"] == "send with notes"
    (P / "quality_assessment.json").write_text(json.dumps(Q))
    _newer_than_velocity(cfg, P / "quality_assessment.json")
    (P / "deliverables" / "summary.json").write_text(json.dumps({"gnss": "skipped (gnss_validation is false)"}))
    # before the tie: the QA's own rule
    S0 = dl.refresh_summary(cfg)
    assert S0["send_rule"] == "send with notes"
    # the tie found no station: the report and the release list hold the report, and so does summary.json
    _write_tie(cfg, NO_STATION_TIE)
    S = dl.refresh_summary(cfg)
    assert S["send_rule"] == qa.send_rule(Q, gnss=NO_STATION_TIE)["value"] == "hold for specialist"
    assert S.get("send_rule_gnss_considered") is True and S0.get("send_rule_gnss_considered") is False
    assert json.loads((P / "deliverables" / "summary.json").read_text())["send_rule"] == "hold for specialist"
    # an anchored tie adds no hold
    _write_tie(cfg, GOOD_TIE)
    assert dl.refresh_summary(cfg)["send_rule"] == "send with notes"
