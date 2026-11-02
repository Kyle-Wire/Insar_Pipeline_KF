"""Lane L7b (5 Oct 2026 review): the audit trail and the release gate.

release.py (the send gate and the sign-off), package.py (the dated client
package), version.py (which code made a result), credits.py (the attribution
the data sources require) and provenance.py on the official phase-linked
route. Every test builds a small synthetic project in tmp_path; none touches
a real project or the network."""
import json
import os
import re
import stat
import time
from pathlib import Path

import numpy as np
import pytest

from insar_pipeline import credits as cr
from insar_pipeline import package, provenance, release, version
from insar_pipeline.config import PipelineConfig

PKG = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def _identity_set(monkeypatch):
    """A firm and a preparer are set: since 6 Oct 2026 release.blockers holds
    a report that names neither (test_fix_final), and these tests are about
    the sign-off and the package, not the identity."""
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})


def _cfg(tmp_path: Path, backend: str = "miaplpy", **kw) -> PipelineConfig:
    cfg = PipelineConfig(project_dir=tmp_path, bbox=(-87.6, 38.2, -87.3, 38.5), start_date="2016-01-01",
                         end_date="2026-06-30", backend=backend, track=19, orbit_direction="ascending", **kw)
    cfg.project_title = "Testsite"
    cfg.project_location = "Somewhere County, Indiana"
    return cfg


def _rep(area_name="Area A1", measured=92.0, testable=True, n_features=1, **extra) -> dict:
    rep = {"project": "/home/x/insar_projects/testsite", "generated": "2026-10-05T10:00", "title": "Testsite",
           "location": "Somewhere County, Indiana", "dates": ["20160105", "20260617"], "n_dates": 270,
           "track": 19, "orbit": "ASCENDING", "backend": "miaplpy", "corrections": ["weather delays (ERA5 atmospheric model)"],
           "anchored": True, "gnss_stations_used": ["INES"],
           "areas": [{"name": "Full Satellite Imaging", "spec": "full", "kind": "full", "vertical_mm_yr": -1.85},
                     {"name": area_name, "spec": "/home/x/insar_projects/testsite/context/project_zone.geojson",
                      "kind": "area", "vertical_mm_yr": 0.6, "detected": False, "z": 1.38,
                      "testable": testable, "measured_percent": measured, "range95_mm_yr": [-3.0, 4.2],
                      "reading": "within normal ground variation"}],
           "features": [{"sign": -1, "area_km2": 2.0, "lat": 38.304, "lon": -87.384, "vertical_relative_mm_yr": -39.7}
                        for _ in range(n_features)],
           "observations": [{"title": f"Feature {i + 1}: sinking ground, about 2.0 km² near 38.304 N, 87.384 W",
                             "text": "...", "kind": "feature"} for i in range(n_features)]}
    rep.update(extra)
    return rep


def _qa(connected=True, flagged=("rate_precision",)) -> dict:
    checks = {"network": {"connected": connected, "pass": connected},
              "coverage": {"pass": True, "note": "share of cells"}}
    for k in flagged:
        checks[k] = {"pass": False, "note": f"{k} is beyond practice."}
    return {"checks": checks, "verdict": "REVIEW", "passed": 2, "flagged": len(flagged)}


def _geotiff(path: Path, value: float):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=16, width=16, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(-87.6, 38.5, 0.0002, 0.0002)) as ds:
        ds.write(np.full((1, 16, 16), value, np.float32))


def _project(tmp_path: Path, rep: dict, qa: dict) -> Path:
    D = tmp_path / "deliverables"
    (D / "figures").mkdir(parents=True)
    (D / "client_report.json").write_text(json.dumps(rep))
    (D / "client_report.html").write_text("<html><body>Testsite report</body></html>")
    (tmp_path / "quality_assessment.json").write_text(json.dumps(qa))
    for f in ("client_summary.png", "client_map.png", "client_series.png", "summary_table.png", "velocity_relative.png"):
        (D / "figures" / f).write_bytes(b"\x89PNG fake " + f.encode())
    import zipfile
    from insar_pipeline import release
    # a client KMZ carries the numbers it was built from (package only takes a matching one)
    with zipfile.ZipFile(D / "Testsite_ground_movement.kmz", "w") as z:
        z.writestr("doc.kml", "<kml><Document><ExtendedData><Data name='numbers_hash'><value>"
                   f"{release.numbers_hash(rep)}</value></Data></ExtendedData></Document></kml>")
    _geotiff(D / "vertical_movement_in_per_yr.tif", 0.0)
    _geotiff(D / "rate_signal_to_noise.tif", 0.5)
    _geotiff(D / "los_velocity_in_per_year.tif", -0.31)
    _geotiff(D / "vertical_velocity_in_per_yr_ASSUMES_VERTICAL.tif", -0.4)
    (D / "measurement_points_egms.csv").write_text("pid,lat,lon\n")
    (D / "new_thing.dat").write_text("unknown product")
    (D / "gnss_tie.json").write_text(json.dumps({"record_window": rep.get("dates"),
                                                 "background": {"vertical_mm_yr": -1.9,
                                                                "uncertainty_basis": "regional floor"},
                                                 "uncertainty_floor_mm_yr": 0.9, "stations": [], "source": "NGL"}))
    return D


def _sign_all(tmp_path, rep):
    return release.sign_off(tmp_path, "K. Reviewer", "imagery checked", [f["n"] for f in release.features(rep)], rep)


# ------------------------------------------------------------------ methodology-10: the release gate
def test_release_gate_blocks_file_stem_names_unchecked_features_and_unsigned(tmp_path):
    cfg = _cfg(tmp_path)
    rep = _rep(area_name="project_zone")
    _project(tmp_path, rep, _qa())
    b = release.blockers(cfg)
    hard = [x for x in b if x["hard"]]
    codes = {x["code"] for x in hard}
    assert {"area_name", "feature_unchecked", "not_signed_off"} <= codes
    assert any("project_zone" in x["text"] for x in hard)
    assert any(x["text"].startswith("Feature 1") for x in hard)
    # The flagged QA check is on the reviewer's list, not a stop
    soft = [x for x in b if not x["hard"]]
    assert any("rate precision" in x["text"] for x in soft)
    assert not any("REVIEW" == x["text"] for x in b)
    assert release.is_final(cfg) is False


def test_release_gate_hard_rules(tmp_path):
    cfg = _cfg(tmp_path)
    _project(tmp_path, _rep(measured=31.0, testable=False), _qa(connected=False))
    b = release.blockers(cfg)
    codes = {x["code"] for x in b if x["hard"]}
    assert {"network_not_connected", "area_under_measured"} <= codes
    # D15 (7 Oct 2026): an area the report itself calls "Not testable" is a
    # reviewer item - read and accepted under [read] - not a stop
    assert "area_not_testable" not in codes
    assert "area_not_testable" in {x["code"] for x in b if not x["hard"]}
    # A legacy report (before 'testable', 5 Oct 2026) with no comparison does
    # not say "Not testable": its figure cannot be read, so it stays a stop
    legacy = _rep()
    for k in ("testable", "detected", "z"):
        legacy["areas"][1].pop(k, None)
    lb = release.blockers(cfg, legacy, _qa())
    assert "area_not_compared" in {x["code"] for x in lb if x["hard"]}
    assert "area_not_testable" not in {x["code"] for x in lb}
    # A screening-route report never goes out
    rep = _rep(route="screening")
    assert "screening_route" in {x["code"] for x in release.blockers(_cfg(tmp_path, backend="disp"), rep, _qa())}
    # A legacy report (no measured share) cannot pass silently
    old = _rep()
    del old["areas"][1]["measured_percent"]
    assert "area_measured_unknown" in {x["code"] for x in release.blockers(cfg, old, _qa())}


def test_sign_off_binds_to_the_numbers_and_a_changed_number_voids_it(tmp_path):
    cfg = _cfg(tmp_path)
    rep = _rep(n_features=2)
    _project(tmp_path, rep, _qa())
    with pytest.raises(ValueError, match="Feature 2"):
        release.sign_off(tmp_path, "K. Reviewer", acknowledged_features=[1], rep=rep)
    with pytest.raises(ValueError):
        release.sign_off(tmp_path, "  ", acknowledged_features=[1, 2], rep=rep)
    assert release.review_record(tmp_path) is None
    _sign_all(tmp_path, rep)
    assert release.is_final(cfg) is True
    assert release.status(cfg)["reviewed_by"] == "K. Reviewer"
    # A rebuild with identical numbers but a new timestamp, path and release block keeps the binding
    same = dict(rep, generated="2026-10-06T09:00", project="/elsewhere/testsite",
                release={"final": True, "blockers": []})
    assert release.numbers_hash(same) == release.numbers_hash(rep)
    # A changed number voids it
    moved = json.loads(json.dumps(rep))
    moved["areas"][1]["vertical_mm_yr"] = 0.7
    (tmp_path / "deliverables" / "client_report.json").write_text(json.dumps(moved))
    assert release.is_final(cfg) is False
    assert "signoff_stale" in {x["code"] for x in release.blockers(cfg)}
    # A second sign-off keeps the first in the trail
    _sign_all(tmp_path, moved)
    assert release.is_final(cfg) is True
    assert len(list((tmp_path / "deliverables" / "reviews").glob("review_*.json"))) == 1


def test_stale_inputs_block(tmp_path):
    cfg = _cfg(tmp_path)
    rep = _rep()
    _project(tmp_path, rep, _qa())
    vel = tmp_path / "mintpy" / "geo" / "geo_velocity.h5"
    vel.parent.mkdir(parents=True)
    vel.write_bytes(b"")                                   # written now, after the report's 2026-10-05T10:00
    os.utime(tmp_path / "quality_assessment.json", (time.time() - 3600, time.time() - 3600))
    texts = [x["text"] for x in release.blockers(cfg) if x["code"] == "stale_inputs"]
    assert any("newer than the report" in t for t in texts)
    assert any("quality assessment is older" in t for t in texts)


# ------------------------------------------------------------------ clientout-8 / usability-5: the package
def test_package_refuses_unless_final_and_builds_a_draft_on_request(tmp_path):
    cfg = _cfg(tmp_path)
    _project(tmp_path, _rep(area_name="project_zone"), _qa())
    with pytest.raises(package.PackageError, match="project_zone"):
        package.build(cfg)
    assert not any((tmp_path / "deliverables" / "releases").glob("2*"))
    out = package.build(cfg, force_draft=True)
    assert out.name.endswith("_DRAFT") and out.parent.name == "releases"
    readme = (out / "README.txt").read_text()
    assert "DRAFT" in readme and "project_zone" in readme
    assert list((out / "client").glob("1_Testsite_ground_movement_report_DRAFT.html"))


def test_package_layout_keeps_raw_products_out_of_client_and_data(tmp_path):
    cfg = _cfg(tmp_path)
    rep = _rep()
    _project(tmp_path, rep, _qa())
    _sign_all(tmp_path, rep)
    out = package.build(cfg)
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    client = sorted(n for n in names if n.startswith("client/"))
    assert client[0] == "client/1_Testsite_ground_movement_report.html"
    assert "client/2_areas_summary.png" in names and "client/5_Testsite_ground_movement_Google_Earth.kmz" in names
    assert "data/vertical_movement_in_per_yr.tif" in names and "data/rate_signal_to_noise.tif" in names
    bad = re.compile(r"los_|ASSUMES_VERTICAL|summary_table|velocity_relative|measurement_points")
    assert not [n for n in names if n.split("/")[0] in ("client", "data") and bad.search(n)]
    assert "technical/rasters/los_velocity_in_per_year.tif" in names
    assert "technical/figures/summary_table.png" in names
    assert "technical/new_thing.dat" in names
    assert not any("measurement_points_egms" in n for n in names)
    readme = (out / "README.txt").read_text()
    assert "technical/new_thing.dat" in readme.split("Unclassified")[1]
    assert "measurement_points_egms.csv" in readme.split("Left in deliverables/")[1]
    # 'Do NOT send', with the median measured from the raw raster itself
    assert "Do NOT send" in readme and "-0.31 in/yr in los_velocity_in_per_year.tif" in readme
    assert "Area A1" in readme and "2016-01-05 to 2026-06-17" in readme


def test_package_raises_when_the_client_raster_is_missing(tmp_path):
    cfg = _cfg(tmp_path)
    rep = _rep()
    D = _project(tmp_path, rep, _qa())
    _sign_all(tmp_path, rep)
    (D / "vertical_movement_in_per_yr.tif").unlink()
    with pytest.raises(package.PackageError, match="vertical_movement_in_per_yr"):
        package.build(cfg)
    assert not list((D / "releases").glob("*")) if (D / "releases").exists() else True


def test_package_refuses_a_signed_report_still_rendered_as_draft(tmp_path):
    cfg = _cfg(tmp_path)
    rep = _rep(release={"final": False, "blockers": [{"code": "not_signed_off", "text": "x", "hard": True}]})
    D = _project(tmp_path, rep, _qa())
    _sign_all(tmp_path, rep)
    assert release.is_final(cfg)                    # the release block is not part of the numbers
    with pytest.raises(package.PackageError, match="rebuild the client report"):
        package.build(cfg)
    (D / "client_report.json").write_text(json.dumps(dict(rep, release={"final": True, "blockers": []})))
    assert package.build(cfg).exists()


# ------------------------------------------------------------------ techout-5: a release is a frozen copy
def test_release_is_a_checksummed_read_only_copy_that_survives_a_rebuild(tmp_path):
    cfg = _cfg(tmp_path)
    rep = _rep()
    D = _project(tmp_path, rep, _qa())
    _sign_all(tmp_path, rep)
    out = package.build(cfg)
    man = json.loads((out / "MANIFEST.json").read_text())
    assert man["final"] is True and man["numbers_hash"] == release.numbers_hash(rep)
    assert man["version"]["code_hash"] == version.code_hash()
    assert man["provenance"]["file"] == "technical/provenance.json"
    import hashlib
    listed = {e["path"]: e["sha256"] for e in man["files"]}
    for rel, sha in listed.items():
        p = out / rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha
        assert p.stat().st_nlink == 1 and not (p.stat().st_mode & stat.S_IWUSR)
    src, dst = D / "client_report.html", out / "client" / "1_Testsite_ground_movement_report.html"
    assert src.stat().st_ino != dst.stat().st_ino
    # A rebuild rewrites deliverables/ in place; the release does not change
    src.write_text("<html>rebuilt with other numbers</html>")
    _geotiff(D / "vertical_movement_in_per_yr.tif", 9.9)
    for rel, sha in listed.items():
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == sha
    # and a second release goes beside it, never over it
    (D / "client_report.html").write_text("<html>Testsite report</html>")
    out2 = package.build(cfg)
    assert out2 != out and out.exists()
    assert (out / "technical" / "provenance.json").exists() and (out / "technical" / "config_snapshot.yaml").exists()


# ------------------------------------------------------------------ usability-13: version in outputs
def test_code_hash_identifies_the_code_and_describe_reports_it(tmp_path):
    tree = tmp_path / "pkg"
    (tree / "tests").mkdir(parents=True)
    (tree / "a.py").write_text("x = 1\n")
    (tree / "tests" / "test_a.py").write_text("y = 1\n")
    h1 = version.code_hash(tree)
    (tree / "tests" / "test_a.py").write_text("y = 2\n")           # tests are not the code that runs
    assert version.code_hash(tree) == h1
    (tree / "a.py").write_text("x = 2\n")
    assert version.code_hash(tree) != h1 and len(h1) == 12
    d = version.describe()
    assert d["version"] == version.__version__ != "unknown" and d["code_hash"] == version.code_hash()
    assert version.code_hash() in version.label()


def test_provenance_and_summary_page_carry_the_version(tmp_path):
    rec = provenance.collect(_cfg(tmp_path))
    assert rec["software"]["insar_pipeline"] == version.__version__
    assert rec["software"]["code_hash"] == version.code_hash()
    from insar_pipeline.scripts import run_summary_page as rsp
    res = {"project": str(tmp_path), "identity": {"title": "Testsite", "location": "Somewhere County"},
           "version": version.describe(), "backend": "miaplpy", "provenance": rec, "config": {"looks": [11, 3]}}
    page = rsp.build(res)
    assert version.__version__ in page and version.code_hash() in page


# ------------------------------------------------------------------ the phase-linked provenance fixture
def _phase_linked_project(tmp_path: Path) -> PipelineConfig:
    import h5py
    cfg = _cfg(tmp_path)
    M = tmp_path / "mintpy"
    (M / "inputs").mkdir(parents=True)
    (M / "geo").mkdir()
    with h5py.File(M / "inputs" / "ifgramStack.h5", "w") as f:
        f["date"] = np.array([[b"20200101", b"20200113"], [b"20200101", b"20200125"], [b"20200113", b"20200125"]])
        f["dropIfgram"] = np.array([1, 1, 0], bool)
        f["unwrapPhase"] = np.zeros((3, 2, 2), np.float32)
        f["bperp"] = np.array([10.0, -20.0, 5.0], np.float32)
    with h5py.File(M / "geo" / "geo_velocity.h5", "w") as f:
        f["velocity"] = np.zeros((2, 2), np.float32)
        f.attrs["REF_DATE"] = "20200101"
    (M / "unwrap_error_applied.json").write_text(json.dumps(
        {"requested": "bridging+phase_closure", "applied": "bridging", "dataset": "unwrapPhase_bridging"}))
    S = tmp_path / "stack"
    (S / "merged" / "interferograms" / "20200101_20200113").mkdir(parents=True)
    (S / "SAFE_files.txt").write_text("\n".join(f"/x/S1A_IW_SLC__1SDV_{d}T000000_{d}T000030_x.SAFE"
                                                for d in ("20200101", "20200113", "20200125")))
    (S / "merged" / "linked_datum_check.json").write_text(json.dumps({"ratio_across_over_within": 0.894}))
    (S / "merged" / "interferograms" / "20200101_20200113" / "unwrap_crop.json").write_text(json.dumps(
        {"engine": "snaphu2", "unwrapper": "snaphu v2.0.7", "cost": "DEFO", "defomax": "4.0", "max_components": 64}))
    (tmp_path / "slc" / "S1A_IW_SLC__1SDV_20200125T000000_x.SAFE").mkdir(parents=True)
    W = tmp_path / "miaplpy"
    W.mkdir()
    (W / "miaplpyApp.cfg").write_text("miaplpy.inversion.ministackSize = 10   # images\n"
                                      "miaplpy.inversion.phaseLinkingMethod = sequential_EMI\n"
                                      "miaplpy.load.slcFile = /x\n")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "miaplpy_20260929_101426.log").write_text(
        "2026-09-29 10:14:27,496 INFO MiaplPy patch verified (0.2.1+overlapdatum): step error across "
        "mini-stack boundaries 1.29x inside\n")
    (tmp_path / "dem").mkdir()
    (tmp_path / "dem" / "dem_extent.json").write_text(json.dumps({"source": "glo_30", "bbox_wsen": [0, 0, 1, 1]}))
    return cfg


# ------------------------------------------------------------------ empirical-10 / techout-3
def test_phase_linked_provenance_records_the_real_network(tmp_path):
    cfg = _phase_linked_project(tmp_path)
    rec = provenance.collect(cfg)
    inp = rec["inputs"]
    assert inp["n_dates_processed"] == 3 and inp["dates_processed"] == ["20200101", "20200113", "20200125"]
    assert inp["n_pairs"] == 3 == rec["outcome"]["network"]["interferograms_loaded"]
    assert inp["network_source"] == "mintpy/inputs/ifgramStack.h5"
    assert inp["n_scenes_used"] == 3 and inp["n_scene_archives_on_disk"] == 1
    assert rec["outcome"]["unwrap_error_method_applied"].startswith("bridging")
    assert "unwrap_error_applied.json" in rec["outcome"]["unwrap_error_method_applied"]
    assert rec["stack_settings"]["unwrap_error_method"] == "bridging"
    assert rec["miaplpy_settings"]["ministack_size"] == 10
    assert rec["miaplpy_settings"]["as_run"]["miaplpy.inversion.ministackSize"] == "10"
    assert rec["records"]["unwrap_error_applied"]["applied"] == "bridging"


@pytest.mark.parametrize("backend", ["stack", "miaplpy", "local", "hyp3", "disp", "nisar"])
def test_every_backend_records_the_network_when_ifgramstack_exists(tmp_path, backend):
    cfg = _phase_linked_project(tmp_path)
    cfg.backend = backend
    rec = provenance.collect(cfg)
    assert rec["inputs"]["n_dates_processed"] == 3 and rec["inputs"]["n_pairs"] == 3


def test_written_provenance_has_no_null_blocks(tmp_path):
    cfg = _phase_linked_project(tmp_path)
    p = provenance.write(cfg, tmp_path / "deliverables")
    rec = json.loads(p.read_text())
    assert not [k for k, v in rec.items() if v is None and k not in ("track", "reference_pixel")]
    assert "disp_settings" not in rec and "miaplpy_settings" in rec


def test_stack_outcome_after_cleanup_is_not_a_zero_pair_network(tmp_path):
    cfg = _phase_linked_project(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "stack" / "merged" / "interferograms")
    so = provenance._stack_outcome(cfg)
    assert "network_connected" not in so and "cleanup" in so["note"]


# ------------------------------------------------------------------ techout-4: reproducible record
def test_provenance_snapshot_identity_and_later_stages(tmp_path):
    cfg = _phase_linked_project(tmp_path)
    src = tmp_path / "testsite.yaml"
    src.write_text("project_dir: x\n")
    (tmp_path / "run.json").write_text(json.dumps({"config": str(src), "kind": "miaplpy", "started": 1790691266.3}))
    p = provenance.write(cfg, tmp_path / "deliverables")
    rec = json.loads(p.read_text())
    snap = (tmp_path / "deliverables" / "config_snapshot.yaml").read_text()
    body = "".join(snap.splitlines(keepends=True)[2:])
    import hashlib
    assert hashlib.sha256(body.encode()).hexdigest() == rec["config"]["snapshot_sha256"]
    assert "miaplpy_ministack_size: 10" in body
    assert rec["config"]["source"]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert rec["run"]["operator"] and rec["run"]["run_kind"] == "miaplpy"
    assert "stages" not in rec
    rep = _rep()
    _project_files_only(tmp_path, rep)
    provenance.record_stages(cfg)
    st = json.loads(p.read_text())["stages"]
    assert st["quality_assessment"]["verdict"] == "REVIEW" and "rate_precision" in st["quality_assessment"]["flagged"]
    assert st["gnss_tie"]["anchored"] is True
    assert st["client_report"]["numbers_hash"] == release.numbers_hash(rep)
    assert st["client_report"]["review"] == "not signed off"


def _project_files_only(tmp_path, rep):
    D = tmp_path / "deliverables"
    D.mkdir(exist_ok=True)
    (D / "client_report.json").write_text(json.dumps(rep))
    (tmp_path / "quality_assessment.json").write_text(json.dumps(_qa()))
    (D / "gnss_tie.json").write_text(json.dumps({"record_window": rep["dates"],
                                                 "background": {"vertical_mm_yr": -1.9,
                                                                "uncertainty_basis": "regional floor"},
                                                 "uncertainty_floor_mm_yr": 0.9, "stations": [], "source": "NGL"}))


# ------------------------------------------------------------------ phaselinked-4 / gap-licensing-10
def test_phase_linked_provenance_records_miaplpy_build_patch_check_and_unwrapper(tmp_path, monkeypatch):
    cfg = _phase_linked_project(tmp_path)
    env = tmp_path / "fake_env"
    (env / "lib" / "python3.11" / "site-packages" / "miaplpy-0.2.1+overlapdatum.dist-info").mkdir(parents=True)
    (env / "conda-meta").mkdir()
    (env / "conda-meta" / "miaplpy-0.2.1-py_0.json").write_text("{}")
    monkeypatch.setattr(version, "_env_prefix", lambda e: env)
    rec = provenance.collect(cfg)
    mo = rec["miaplpy_outcome"]
    assert mo["patch_check"]["ratio"] == 1.29 and mo["patch_check"]["miaplpy_version"] == "0.2.1+overlapdatum"
    assert mo["linked_datum_check"]["ratio_across_over_within"] == 0.894
    assert mo["unwrap"]["settings"][0]["engine"] == "snaphu2"
    sw = rec["software"]
    assert sw["miaplpy"]["version"] == "0.2.1+overlapdatum" and sw["miaplpy"]["patched_build"] is True
    assert sw["snaphu"] == "snaphu v2.0.7" and "unwrap_crop.json" in sw["snaphu_source"]
    assert sw["env_miaplpy"]["packages"] == 1
    assert rec["dem"]["source"] == "Copernicus DEM GLO-30"


# ------------------------------------------------------------------ gap-licensing-1: credits
def test_credits_follow_what_the_run_used(tmp_path):
    cfg = _cfg(tmp_path)
    rep = _rep()
    t = " ".join(c["text"] for c in cr.credits(cfg, rep, prov={}))
    assert "Contains modified Copernicus Sentinel data 2016-2026" in t
    assert "Copernicus Climate Change Service" in t and "neither the european commission nor ecmwf" in t.lower()
    assert "CC BY 4.0" in t and "© DLR e.V." in t and "Blewitt" in t
    assert "MiaplPy" in t and "MintPy" in t and "OPERA" not in t
    # ERA5 credited from the record of what was applied, not from the config
    cfg2 = _cfg(tmp_path)
    (tmp_path / "mintpy").mkdir(exist_ok=True)
    (tmp_path / "mintpy" / "run_products.json").write_text(json.dumps({"corrections_applied": ["ramp", "demErr"]}))
    t2 = " ".join(c["text"] for c in cr.credits(cfg2, dict(rep, anchored=False, gnss_stations_used=[]), prov={}))
    assert "Climate Change Service" not in t2 and "Blewitt" not in t2
    # a user DEM is not credited as GLO-30; DISP credits OPERA
    t3 = " ".join(c["text"] for c in cr.credits(_cfg(tmp_path, dem_path=tmp_path / "lidar.tif"), rep, prov={}))
    assert "DLR" not in t3 and "lidar.tif" in t3
    assert "OPERA" in " ".join(c["text"] for c in cr.credits(_cfg(tmp_path, backend="disp"), rep, prov={}))
    html = cr.text_block(cfg, rep, prov={}, html=True)
    assert html.startswith('<ul class="credits">') and "&lt;" not in cr.text_block(cfg, rep, {})
    assert "Copernicus Sentinel" in cr.short_line(cfg, rep, {})


def test_package_readme_carries_the_credits(tmp_path):
    cfg = _cfg(tmp_path)
    rep = _rep()
    _project(tmp_path, rep, _qa())
    out = package.build(cfg, force_draft=True)
    readme = (out / "README.txt").read_text()
    assert "Contains modified Copernicus Sentinel data 2016-2026" in readme and "ECMWF" in readme


# ------------------------------------------------------------------ run_summary_page: no site text
def test_summary_page_has_no_site_literals_and_takes_identity():
    src = (PKG / "scripts" / "run_summary_page.py").read_text()
    assert not re.search(r"Site A|Gibson|11 × 3|CMIS|24 workers|Mar–May 2018", src)
    from insar_pipeline.scripts import run_summary_page as rsp
    res = {"project": "/x/site_d", "identity": {"title": "Site D", "location": "Sullivan County, Indiana"},
           "backend": "miaplpy", "config": {"looks": [11, 3], "orbit_direction": "descending"},
           "provenance": {"orbit_direction": "descending", "track": 165},
           "network_gaps": {"min_gap_days": 36, "gaps": [{"from": "20180319", "to": "20180518", "days": 60,
                                                          "bridging_pairs": 6, "longest_bridge_days": 96,
                                                          "max_bridge_bperp_m": 76.0}]},
           "network_connectivity": {"kept_pairs": 10, "dates": 5, "excluded_dates": []}}
    page = rsp.build(res)
    assert "<title>Site D Ground Motion Record" in page and "Sullivan County, Indiana" in page
    assert "satellite to the east on this descending track" in page
    assert "2018-03-19 → 2018-05-18 (60 days)" in page


def test_package_leaves_out_a_kmz_built_from_older_numbers(tmp_path):
    """A KMZ whose embedded numbers_hash is not the signed report's states other
    figures: a final package must not carry it (the stale file stays in deliverables/)."""
    import zipfile
    from insar_pipeline import package
    rep = _rep()
    D = _project(tmp_path, rep, _qa())
    with zipfile.ZipFile(D / "Testsite_ground_movement.kmz", "w") as z:
        z.writestr("doc.kml", "<kml><Data name='numbers_hash'><value>0123abcd</value></Data></kml>")
    copies, not_copied, _ = package._plan(_cfg(tmp_path), rep, draft=False)
    assert not any(d.endswith(".kmz") for _, d, _, _ in copies)
    assert any(str(s).endswith(".kmz") and "earlier version" in why for s, _, why in not_copied)
