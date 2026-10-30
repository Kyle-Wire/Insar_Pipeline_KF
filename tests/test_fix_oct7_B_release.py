"""7 Oct 2026 fixes, group B_release: the send gate, the client package, the
client KMZ and provenance (release.py, package.py, client_kmz.py,
provenance.py). Each test failed on the code in backups/pre_fixes_20261007/
unless its docstring says otherwise. Synthetic projects in tmp_path only;
no real project and no network."""
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import test_fix_L7b as L7b                                     # the gate's and the package's synthetic project
from insar_pipeline import package, provenance, release

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _identity_set(monkeypatch):
    from insar_pipeline import settings
    monkeypatch.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})


def _signed_project(tmp_path, **rep_extra):
    cfg = L7b._cfg(tmp_path)
    rep = L7b._rep(**rep_extra)
    D = L7b._project(tmp_path, rep, L7b._qa())
    L7b._sign_all(tmp_path, rep)
    return cfg, rep, D


def _write_report(D: Path, rep: dict):
    (D / "client_report.json").write_text(json.dumps(rep))


def _age(p: Path, seconds: float):
    t = time.time() - seconds
    os.utime(p, (t, t))


# ------------------------------------------------------------------ #1 / #41 + contract 4: the GNSS tie's timestamps
def test_a_re_stamped_gnss_tie_does_not_void_the_sign_off(tmp_path):
    """The sign-off rebuild re-runs the tie (gnss_tie._write stamps
    'written' = now) and rep['gnss'] is the tie copied whole: with the stamp
    in the numbers hash no report could ever become FINAL."""
    gnss = {"anchored": True, "written": "2026-10-06T21:59:03", "background": {"vertical_mm_yr": -1.9},
            "stations": [{"site": "INES", "data_file_modified": "2026-09-27T14:08:45",
                          "vertical_over_record_mm_yr": {"rate_mm_yr": -1.8}}]}
    cfg, rep, D = _signed_project(tmp_path, gnss=gnss)
    assert release.is_final(cfg)
    h = release.numbers_hash(rep)
    rebuilt = json.loads(json.dumps(rep))
    rebuilt["gnss"]["written"] = "2026-10-07T09:12:44"                     # the tie re-run, same numbers
    rebuilt["gnss"]["stations"][0]["data_file_modified"] = "2026-10-07T09:12:30"   # the station file re-fetched
    _write_report(D, rebuilt)
    assert release.numbers_hash(rebuilt) == h
    st = release.status(cfg)
    codes = {b["code"] for b in st["blockers"] if b["hard"]}
    assert st["final"] is True and not codes & {"signoff_stale", "feature_unchecked"}, st["blockers"]
    # a changed tie number still voids it
    rebuilt["gnss"]["background"]["vertical_mm_yr"] = -2.4
    _write_report(D, rebuilt)
    assert "signoff_stale" in {b["code"] for b in release.blockers(cfg)}


def test_sign_then_tie_rewrite_then_rebuild_stays_final(tmp_path):
    """The dashboard's order: sign_off -> gnss_tie.run (writes the file
    again) -> client_report copies the file into rep['gnss']."""
    from insar_pipeline import gnss_tie
    cfg, rep, D = _signed_project(tmp_path)
    tie = json.loads((D / "gnss_tie.json").read_text())
    first = gnss_tie._write(cfg, tie)
    _write_report(D, dict(rep, gnss={**first, "anchored": True}))
    L7b._sign_all(tmp_path, json.loads((D / "client_report.json").read_text()))
    time.sleep(1.1)                                                       # 'written' has second resolution
    second = gnss_tie._write(cfg, tie)                                    # the rebuild re-runs the tie
    assert second["written"] != first["written"]
    _write_report(D, dict(rep, gnss={**second, "anchored": True}))
    assert release.is_final(cfg), [b["text"] for b in release.hard_blockers(cfg)]


# ------------------------------------------------------------------ contract 3: the station inside the area
def test_an_inside_station_that_disagrees_is_a_hard_blocker(tmp_path):
    """client_report stores 'consistent' on each gnss.inside_checks entry
    (the page's station_check rule); the gate reads it. Passed before these
    fixes too: the gate side was already there, this pins the contract."""
    cfg, rep, D = _signed_project(tmp_path)
    for ok, blocked in ((False, True), (True, False)):
        r = dict(rep, gnss={"anchored": True, "inside_checks": [{"site": "INES", "consistent": ok}]})
        hard = [b for b in release.blockers(cfg, r) if b["hard"]]
        assert any(b["code"] == "gnss_inside_inconsistent" and "INES" in b["text"] for b in hard) is blocked
    # an entry the check could not read carries no verdict and does not block
    r = dict(rep, gnss={"anchored": True, "inside_checks": [{"site": "INES"}]})
    assert not any(b["code"] == "gnss_inside_inconsistent" for b in release.blockers(cfg, r))


# ------------------------------------------------------------------ contract 1: a time-series run part-way
def test_an_unfinished_time_series_run_blocks_the_release(tmp_path):
    cfg, rep, D = _signed_project(tmp_path)
    assert release.is_final(cfg)
    marker = Path(cfg.mintpy_dir) / "run_incomplete.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"stamp": "20261007_0900", "started": "2026-10-07T09:00:12"}))
    hard = release.hard_blockers(cfg)
    b = next(x for x in hard if x["code"] == "mintpy_incomplete")
    assert "did not finish" in b["text"] and "2026-10-07 09:00" in b["text"]
    assert release.is_final(cfg) is False
    with pytest.raises(package.PackageError):
        package.build(cfg)
    marker.unlink()
    assert release.is_final(cfg)


# ------------------------------------------------------------------ #11: a tie the report could not use
def test_a_tie_older_than_the_velocity_product_is_stale(tmp_path):
    """The velocity product re-run and the report rebuilt without the tie:
    gnss_tie.check refuses the file, the report is relative-only, and the
    gate said nothing."""
    cfg = L7b._cfg(tmp_path)
    rep = L7b._rep(anchored=False, gnss={"anchored": False}, generated=time.strftime("%Y-%m-%dT%H:%M"))
    D = L7b._project(tmp_path, rep, L7b._qa())
    vel = Path(cfg.mintpy_dir) / "geo" / "geo_velocity.h5"
    vel.parent.mkdir(parents=True)
    vel.write_bytes(b"")
    _age(vel, 600)
    _age(D / "gnss_tie.json", 1200)                       # the tie from before the velocity re-run
    st = release._stale(cfg, rep)
    assert any("re-run the GNSS tie" in s and "velocity product" in s for s in st), st
    assert "stale_inputs" in {b["code"] for b in release.hard_blockers(cfg, rep)}
    # the dashboard asks with the project folder alone
    assert any("re-run the GNSS tie" in s for s in release._stale(tmp_path, rep))
    # a tie written after the product is this record's: nothing to re-run
    _age(D / "gnss_tie.json", 60)
    assert not any("GNSS tie" in s for s in release._stale(cfg, rep))
    # a tie for another record window, report not anchored
    (D / "gnss_tie.json").write_text(json.dumps(dict(json.loads((D / "gnss_tie.json").read_text()),
                                                     record_window=["20170101", "20260617"])))
    assert sum("GNSS tie" in s for s in release._stale(cfg, rep)) == 1


# ------------------------------------------------------------------ #13: the signal-to-noise raster
def _snr(path: Path, **tags):
    L7b._geotiff(path, 0.5)
    if tags:
        import rasterio
        with rasterio.open(path, "r+") as ds:
            ds.update_tags(**tags)


def test_a_first_pass_signal_to_noise_raster_beside_the_report_is_stale(tmp_path):
    cfg, rep, D = _signed_project(tmp_path)
    snr = D / "rate_signal_to_noise.tif"
    # the export's first pass, the report stage never got to rewrite it
    _snr(snr, QUANTITY="rate signal-to-noise", FARMLAND="first pass (before the client report; it rewrites "
                                                        "this file from its own farmland)")
    _age(snr, 3600)
    assert any("rate_signal_to_noise" in s for s in release._stale(cfg, rep))
    assert release.is_final(cfg) is False
    # a report still being built (client_report.compute's release block) rewrites the raster next:
    # its in-memory record is not the report on disk, and is not held for it
    building = dict(rep, generated="2026-10-07T10:00")
    assert not any("rate_signal_to_noise" in s for s in release._stale(cfg, building))
    # tagged by the export of 5-6 Oct code (no FARMLAND, no WRITTEN_BY): Site B's file today
    _snr(snr, QUANTITY="rate signal-to-noise")
    _age(snr, 3600)
    assert any("rate_signal_to_noise" in s for s in release._stale(cfg, rep))
    # the report's own file, written just before client_report.json
    _snr(snr, QUANTITY="rate signal-to-noise", WRITTEN_BY="client_report", FARMLAND="the report's farmland")
    _age(snr, 3600)
    assert not any("rate_signal_to_noise" in s for s in release._stale(cfg, rep))
    assert release.is_final(cfg)
    # the report's file, then an export rewrote it after the report (the report stage then crashed)
    _age(D / "client_report.json", 7200)
    _snr(snr, QUANTITY="rate signal-to-noise", WRITTEN_BY="client_report")
    assert any("rate_signal_to_noise" in s for s in release._stale(cfg, rep))


# ------------------------------------------------------------------ #12: one resolution statement
def test_posting_note_reads_the_product_attributes(tmp_path):
    """provenance.json said 'about 95 m east-west' (Sentinel-1 defaults) in
    stack_settings.posting_note and 'about 90 m' in resolution.text."""
    import h5py
    from insar_pipeline.resolution import effective_resolution
    cfg = L7b._cfg(tmp_path)
    geo = Path(cfg.mintpy_dir) / "geo"
    geo.mkdir(parents=True)
    with h5py.File(geo / "geo_velocity.h5", "w") as f:
        f.create_dataset("velocity", data=np.zeros((8, 8), np.float32))
        f.attrs.update({"X_FIRST": "-87.6", "Y_FIRST": "38.5", "X_STEP": "0.000277", "Y_STEP": "-0.000277",
                        "LENGTH": "8", "WIDTH": "8", "UNIT": "m/year", "FILE_TYPE": "velocity",
                        "RANGE_PIXEL_SIZE": "3.5", "RLOOKS": "1", "AZIMUTH_PIXEL_SIZE": "15.0", "ALOOKS": "1",
                        "CENTER_INCIDENCE_ANGLE": "30.0", "HEIGHT": "693000", "PROCESSOR": "isce"})
    note = provenance._posting_note(cfg)
    assert note == provenance._resolution(cfg)["text"]
    assert note != effective_resolution(cfg)["text"]           # the defaults would have said otherwise


# ------------------------------------------------------------------ #42: one client KMZ in deliverables/
def test_draft_package_leaves_out_a_kmz_without_the_numbers(tmp_path):
    """Site B: the 29 Sep Site B_ground_movement.kmz (no numbers_hash)
    went into a draft package's client/ under a name without DRAFT."""
    cfg = L7b._cfg(tmp_path)
    rep = L7b._rep()
    D = L7b._project(tmp_path, rep, L7b._qa())
    with zipfile.ZipFile(D / "Testsite_ground_movement.kmz", "w") as z:
        z.writestr("doc.kml", "<kml><Document><name>old</name></Document></kml>")
    copies, not_copied, _ = package._plan(cfg, rep, draft=True)
    assert not any(d.endswith(".kmz") for _, d, _, _ in copies)
    assert any(s.name == "Testsite_ground_movement.kmz" and "earlier version" in why for s, _, why in not_copied)


@pytest.fixture(scope="module")
def l9b_built(tmp_path_factory):
    """L9b's synthetic project with a client report, then a 'Products only'
    run's export (a first-pass signal-to-noise raster beside the existing
    report) and the report rebuilt."""
    import test_fix_L9b as L9b
    from insar_pipeline import client_report as cr, settings
    cfg = L9b.make_project(tmp_path_factory.mktemp("oct7b"))
    D = Path(cfg.project_dir) / "deliverables"
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "identity", lambda: {"firm": "Test Firm", "preparer": "A. Engineer", "contact": ""})
    try:
        cr.build(cfg)
        _snr(D / "rate_signal_to_noise.tif", QUANTITY="rate signal-to-noise",
             FARMLAND="first pass (before the client report; it rewrites this file)")
        between = release._stale(cfg, None)            # the files on disk before the rebuild
        cr.build(cfg)
    finally:
        mp.undo()
    return SimpleNamespace(cfg=cfg, between=between)


def test_a_report_build_is_not_held_for_the_raster_it_rewrites(l9b_built):
    """The signal-to-noise check (#13) on a real rebuild: the report's own
    release block is computed before it rewrites the export's first-pass
    raster, so the check must not fire inside the build, and the files the
    build leaves pass it."""
    assert any("rate_signal_to_noise" in s for s in l9b_built.between)    # on disk: not the report's
    cfg = l9b_built.cfg
    D = Path(cfg.project_dir) / "deliverables"
    rep = json.loads((D / "client_report.json").read_text())
    assert not any("rate_signal_to_noise" in b["text"].lower() for b in rep["release"]["blockers"])
    import rasterio
    with rasterio.open(D / "rate_signal_to_noise.tif") as ds:
        assert ds.tags().get("WRITTEN_BY") == "client_report"
    assert not any("rate_signal_to_noise" in s for s in release._stale(cfg, rep))


def test_the_kmz_build_moves_the_other_release_state_aside(l9b_built, tmp_path, monkeypatch):
    """client_kmz.build wrote only kmz_name(final) and left the other file."""
    from insar_pipeline import client_kmz as ck
    cfg = l9b_built.cfg
    D = Path(cfg.project_dir) / "deliverables"
    (D / "Testsite_ground_movement.kmz").write_bytes(b"an earlier final-named KMZ")
    (D / "geo_timeseries_ERA5_ramp_demErr.kmz").write_bytes(b"MintPy's own")
    out = ck.build(cfg)
    assert out.name == "Testsite_ground_movement_DRAFT.kmz"
    left = sorted(p.name for p in D.glob("*ground_movement*.kmz"))
    assert left == ["Testsite_ground_movement_DRAFT.kmz"]
    moved = list((D / "superseded").rglob("Testsite_ground_movement.kmz"))
    assert len(moved) == 1 and moved[0].read_bytes() == b"an earlier final-named KMZ"
    assert (D / "geo_timeseries_ERA5_ramp_demErr.kmz").exists()
    # signed: the final-named KMZ is written and the draft goes aside
    monkeypatch.setattr(ck, "release_state", lambda c, r: (True, []))
    out = ck.build(cfg)
    assert sorted(p.name for p in D.glob("*ground_movement*.kmz")) == ["Testsite_ground_movement.kmz"]
    assert list((D / "superseded").rglob("Testsite_ground_movement_DRAFT.kmz"))
    # a preview never touches deliverables/
    monkeypatch.undo()
    ck.build(cfg, out_dir=tmp_path / "preview")
    assert sorted(p.name for p in D.glob("*ground_movement*.kmz")) == ["Testsite_ground_movement.kmz"]


# ------------------------------------------------------------------ #44: the draft KMZ in a final package
def test_final_package_never_files_the_draft_kmz_and_needs_a_client_kmz(tmp_path):
    cfg, rep, D = _signed_project(tmp_path)
    (D / "Testsite_ground_movement.kmz").rename(D / "Testsite_ground_movement_DRAFT.kmz")
    copies, not_copied, unclassified = package._plan(cfg, rep, draft=False)
    assert not any(d.endswith(".kmz") for _, d, _, _ in copies), [d for _, d, _, _ in copies if d.endswith(".kmz")]
    assert any(s.name == "Testsite_ground_movement_DRAFT.kmz" for s, _, _ in not_copied)
    assert not any(u.endswith(".kmz") for u in unclassified)
    with pytest.raises(package.PackageError, match="no client KMZ"):
        package.build(cfg)
    assert not [p for p in (D / "releases").iterdir()] if (D / "releases").exists() else True


# ------------------------------------------------------------------ #43: the area-summaries PDF
def test_final_package_leaves_out_a_pdf_from_an_earlier_build(tmp_path):
    cfg, rep, D = _signed_project(tmp_path)
    pdf = D / "client_area_summaries.pdf"
    pdf.write_bytes(b"%PDF-1.4 ... /Subject (Client report area summaries) /Keywords () >>")
    _age(pdf, 3600)                                     # area_pdf raised on the last build: the old one stayed
    copies, not_copied, _ = package._plan(cfg, rep, draft=False)
    assert not any(d.endswith(".pdf") for _, d, _, _ in copies)
    assert any(s == pdf and "older than the report" in why for s, _, why in not_copied)
    # written after the report but still marked DRAFT
    pdf.write_bytes(b"%PDF-1.4 ... /Subject (DRAFT - not for release) /Keywords (DRAFT) >>")
    copies, not_copied, _ = package._plan(cfg, rep, draft=False)
    assert not any(d.endswith(".pdf") for _, d, _, _ in copies)
    assert any(s == pdf and "DRAFT" in why for s, _, why in not_copied)
    # this build's final PDF goes in
    pdf.write_bytes(b"%PDF-1.4 ... /Subject (Client report area summaries) /Keywords () >>")
    copies, _, _ = package._plan(cfg, rep, draft=False)
    assert any(d == "client/5_Testsite_area_summaries.pdf" for _, d, _, _ in copies)
    out = package.build(cfg)
    assert (out / "client" / "5_Testsite_area_summaries.pdf").exists()


# ------------------------------------------------------------------ #46: workbooks exported before signing
def _results_export(D: Path, rep_hash: str | None):
    from openpyxl import Workbook
    rd = D / "results" / "area_a1_20160105_20260617"
    rd.mkdir(parents=True)
    wb = Workbook()
    wb.active.title = "Summary"
    wb.create_sheet("Notes")
    wb.save(rd / "area_a1_2016-01-05_2026-06-17.xlsx")
    (rd / "chart.png").write_bytes(b"\x89PNG chart")
    summ = {"area_summary": {"area": "Area A1"}, "points_sheet": True}
    if rep_hash is not None:
        summ["numbers_hash"] = rep_hash
    (rd / "summary.json").write_text(json.dumps(summ))
    for p in rd.iterdir():
        _age(p, 3600)                                   # exported, then the sign-off rebuild rewrote the report
    return rd


def test_workbook_carrying_the_report_hash_survives_the_sign_off_rebuild(tmp_path):
    cfg, rep, D = _signed_project(tmp_path)
    _results_export(D, release.numbers_hash(rep))
    copies, not_copied, _ = package._plan(cfg, rep, draft=False)
    dests = [d for _, d, _, _ in copies]
    assert any(d.endswith("Area_A1_points_yearly_rates_and_statistics.xlsx") and d.startswith("client/")
               for d in dests), dests
    assert any(d.endswith("Area_A1_yearly_rates_chart.png") for d in dests)


def test_workbook_with_another_hash_or_none_and_older_stays_out(tmp_path):
    cfg, rep, D = _signed_project(tmp_path)
    rd = _results_export(D, "0123abcd")
    copies, not_copied, _ = package._plan(cfg, rep, draft=False)
    assert not any(d.endswith(".xlsx") for _, d, _, _ in copies)
    assert any(s.suffix == ".xlsx" and "earlier version" in why for s, _, why in not_copied)
    # an export from before the hash was recorded: judged by its age, as before
    summ = json.loads((rd / "summary.json").read_text())
    summ.pop("numbers_hash")
    (rd / "summary.json").write_text(json.dumps(summ))
    copies, not_copied, _ = package._plan(cfg, rep, draft=False)
    assert any(s.suffix == ".xlsx" and "before the report was last rebuilt" in why for s, _, why in not_copied)


# ------------------------------------------------------------------ #45 + contract 2: the package lock
def test_package_refuses_while_a_run_holds_the_project(tmp_path):
    cfg, rep, D = _signed_project(tmp_path)
    code = ("import sys, time\nfrom insar_pipeline import runs\n"
            "with runs.project_lock(sys.argv[1], kind='products'):\n"
            "    print('held', flush=True)\n    time.sleep(60)\n")
    env = {**os.environ, "PYTHONPATH": str(PKG.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    p = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdout=subprocess.PIPE, text=True, env=env)
    try:
        assert p.stdout.readline().strip() == "held"
        with pytest.raises(package.PackageError, match="a run is using this project.*products run"):
            package.build(cfg, force_draft=True)
        assert not (D / "releases").exists()
    finally:
        p.kill()
        p.wait()
    # the run over: the package builds, and holds the lock only while it does
    out = package.build(cfg)
    assert (out / "MANIFEST.json").exists()
    from insar_pipeline import runs
    assert runs.lock_holder(tmp_path) is None
