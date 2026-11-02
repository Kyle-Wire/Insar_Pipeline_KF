"""Integration round 7 Oct 2026, group I3 (dashboard.py, doctor.py):
the release-status cache watches every file release.blockers reads, the
send rule shown is the report's (it saw the GNSS tie), the MiaplPy overlap
advice says what to set, and an Earthdata / ASF outage is a warning, not a
login failure with a ~/.netrc fix. No network, no real runs."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from insar_pipeline import dashboard as dash, doctor, release, settings
from insar_pipeline import data_acquisition as da

AREA = {"name": "Area A2", "spec": "38.35114,-87.66331,38.35349,-87.65863", "kind": "area",
        "vertical_mm_yr": -5.67, "vertical_uncertainty_1sigma_mm_yr": 7.53, "detected": False, "z": -0.61,
        "testable": True, "measured_percent": 90.0}


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """A built project: client report, QA, an older velocity product, and an
    empty settings file location (nobody named as preparer yet)."""
    monkeypatch.setenv("INSAR_SETTINGS", str(tmp_path / "settings.json"))
    dash._RELEASE_CACHE.clear()
    P = tmp_path / "demo"
    (P / "deliverables").mkdir(parents=True)
    (P / "mintpy").mkdir()
    vel = P / "mintpy" / "velocity.h5"
    vel.write_bytes(b"")
    os.utime(vel, (time.time() - 3600, time.time() - 3600))
    rep = {"generated": time.strftime("%Y-%m-%d %H:%M"), "verdict": "REVIEW", "dates": ["20170101", "20200101"],
           "areas": [AREA]}
    (P / "deliverables" / "client_report.json").write_text(json.dumps(rep))
    (P / "quality_assessment.json").write_text(json.dumps({"checks": {}, "send_rule": {"value": "send"}}))
    yield P
    dash._RELEASE_CACHE.clear()


def _codes(st):
    return {b["code"] for b in (st or {}).get("blockers", [])}


# ---------------------------------------------------------------- (a) the cache key
def test_release_status_sees_the_run_incomplete_marker_the_snr_raster_and_the_preparer(proj):
    """These three change the gate's answer with no change to any file the
    cache watched, so the dashboard kept showing the old blockers."""
    first = dash.release_status(proj)
    assert "mintpy_incomplete" not in _codes(first) and "identity_missing" in _codes(first)
    assert not any("signal-to-noise" in b["text"] for b in first["blockers"])

    (proj / "mintpy" / "run_incomplete.json").write_text(json.dumps({"stamp": "x", "started": "2026-10-07T09:00"}))
    assert "mintpy_incomplete" in _codes(dash.release_status(proj))

    (proj / "deliverables" / "rate_signal_to_noise.tif").write_bytes(b"")    # not the report's own raster
    assert any("signal-to-noise" in b["text"] for b in dash.release_status(proj)["blockers"])

    settings.save(firm="Acme Surveys", preparer="K. Wire")
    assert "identity_missing" not in _codes(dash.release_status(proj))


def test_release_status_watches_the_project_mintpy_marker_when_the_config_points_elsewhere(proj):
    """release._mintpy_incomplete looks in cfg.mintpy_dir AND <project>/mintpy."""
    cfg = SimpleNamespace(project_dir=proj, mintpy_dir=proj / "mintpy_other")
    assert "mintpy_incomplete" not in _codes(dash.release_status(cfg))
    (proj / "mintpy" / "run_incomplete.json").write_text("{}")
    assert "mintpy_incomplete" in _codes(dash.release_status(cfg))


def test_a_folder_and_a_config_are_cached_apart(proj, monkeypatch):
    """The cross-check counts only when the config turns it on, so a status
    worked out from the folder alone must not be served to the run screen."""
    calls = []
    real = release.status
    monkeypatch.setattr(release, "status", lambda c, *a, **k: calls.append(c) or real(c, *a, **k))
    dash.release_status(proj)
    dash.release_status(proj)
    assert len(calls) == 1                                                # cached
    dash.release_status(SimpleNamespace(project_dir=proj, mintpy_dir=proj / "mintpy"))
    assert len(calls) == 2


def test_run_screen_status_follows_a_config_edit(proj, tmp_path):
    """The run screen reloads its config after an edit (e); turning
    cross_check_disp on changes the gate's answer with no project file
    changing, so the config's own file is part of the cache key (reviewer)."""
    tech = proj / "deliverables" / "technical"
    tech.mkdir()
    (tech / "cross_check.json").write_text(json.dumps({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                       "status": "complete", "agree": False}))
    y = tmp_path / "demo.yaml"
    y.write_text("cross_check_disp: false\n")
    os.utime(y, (time.time() - 60, time.time() - 60))
    off = SimpleNamespace(project_dir=proj, mintpy_dir=proj / "mintpy", cross_check_disp=False, _source_path=y)
    assert "cross_check_disagrees" not in _codes(dash.release_status(off))
    y.write_text("cross_check_disp: true\n")                                 # edited, then reloaded
    on = SimpleNamespace(project_dir=proj, mintpy_dir=proj / "mintpy", cross_check_disp=True, _source_path=y)
    assert "cross_check_disagrees" in _codes(dash.release_status(on))


# ---------------------------------------------------------------- (b) the send rule shown
def test_send_rule_is_the_reports_then_the_summarys_then_the_qas(proj):
    D = proj / "deliverables"
    rep = json.loads((D / "client_report.json").read_text())
    (D / "summary.json").write_text(json.dumps({"send_rule": "send with notes",         # stored as the value
                                                "send_rule_gnss_considered": True}))
    (D / "client_report.json").write_text(json.dumps({**rep, "send_rule": {"value": "hold for specialist",
                                                                          "reasons": ["GNSS"]}}))
    assert dash.qa_send_rule(proj) == "hold for specialist"
    # the projects table shows the report's rule, not the QA's 'send'
    row = dash.project_row(SimpleNamespace(project_dir=str(proj), verdict="REVIEW", error=None))
    assert row["verdict"].plain.startswith("hold for specialist")
    # a report with no rule of its own (built before it existed): summary.json's
    (D / "client_report.json").write_text(json.dumps({**rep, "send_rule": {"value": None, "reasons": ["old QA"]}}))
    assert dash.qa_send_rule(proj) == "send with notes"
    # summary.json's rule not re-evaluated with the tie is a copy of the QA's,
    # possibly an older one: the QA's own decides (reviewer, 7 Oct)
    (D / "summary.json").write_text(json.dumps({"send_rule": "send with notes", "send_rule_gnss_considered": False}))
    assert dash.qa_send_rule(proj) == "send"
    (D / "summary.json").write_text(json.dumps({"send_rule": "send with notes"}))      # written before the flag
    assert dash.qa_send_rule(proj) == "send"
    (D / "summary.json").write_text(json.dumps({"other": 1}))
    assert dash.qa_send_rule(proj) == "send"
    (proj / "quality_assessment.json").write_text(json.dumps({"checks": {}}))
    assert dash.qa_send_rule(proj) is None


# ---------------------------------------------------------------- (c) the overlap advice
def test_patch_check_advice_sets_the_tested_default_rather_than_raising_the_lag():
    fix = doctor.MIAPLPY_TIME_LAG_FIX
    assert fix.startswith("set miaplpy_time_lag to 3, the tested default, in those configs")
    assert "fails at 0 and from about 7 with mini-stack 10" in fix and "raise" not in fix
    src = Path(doctor.__file__).read_text()
    assert "raise miaplpy_time_lag" not in src and "MIAPLPY_TIME_LAG_FIX)" in src


# ---------------------------------------------------------------- (d) an outage is not a bad login
@pytest.fixture
def netrc(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARTHDATA_USERNAME", "kyle")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "right")
    f = tmp_path / ".netrc"
    f.write_text("machine urs.earthdata.nasa.gov login kyle password right\n")
    return f


def _asf_raises(monkeypatch, text):
    class ASFAuthenticationError(Exception):
        pass

    class S:
        def auth_with_creds(self, u, p):
            raise ASFAuthenticationError(text)
    monkeypatch.setattr(da.asf, "ASFSession", S)


OUTAGE = ("Failed to log in with provided credentials. Original Exception: 503 Server Error: "
          "Service Unavailable for url: https://urs.earthdata.nasa.gov/api/users/find_or_create_token")


def test_doctor_shows_an_outage_as_a_warning_with_no_netrc_fix(netrc, monkeypatch):
    _asf_raises(monkeypatch, OUTAGE)                  # through data_acquisition's own coded check
    ok, msg, fix, warn = doctor.earthdata_checks(netrc)[-1]
    assert (ok, warn) == (False, True)
    assert "the Earthdata / ASF servers did not answer; try again later" in msg and "503" in msg
    assert "netrc" not in fix
    # wrong credentials stay a failure with the ~/.netrc fix
    _asf_raises(monkeypatch, "Failed to log in with provided credentials. Original Exception: 401 Client Error")
    ok, msg, fix, warn = doctor.earthdata_checks(netrc)[-1]
    assert (ok, warn) == (False, False) and "~/.netrc" in fix and "rejected" in msg
    # ASF access not authorised: a failure, but the fix is ASF's terms, not the password
    _asf_raises(monkeypatch, "Failed to set asf-urs cookies. Original exception: Failed to find `asf-urs` cookie")
    ok, msg, fix, warn = doctor.earthdata_checks(netrc)[-1]
    assert (ok, warn) == (False, False) and "search.asf.alaska.edu" in fix and "netrc" not in fix
    monkeypatch.setattr(da, "_earthdata_login_status", lambda: ("ok", "fine"))
    assert doctor.earthdata_checks(netrc)[-1][0] is True
    # no login in ~/.netrc: that line fails and nothing is tried
    netrc.write_text("machine other.host login a password b\n")
    out = doctor.earthdata_checks(netrc)
    assert len(out) == 1 and out[0][0] is False


def test_preflight_warns_on_an_outage_and_fails_on_a_bad_login(tmp_path, monkeypatch):
    P = tmp_path / "demo"
    cfg = SimpleNamespace(project_dir=P, backend="disp", start_date="2017-01-01", end_date="2020-01-01",
                          track=165, mintpy_tropo_method="no")
    monkeypatch.setattr(settings, "disk_budget_gb", lambda p: {"budget": 400.0, "source": "test"})
    monkeypatch.setattr(dash.runs, "live_runs", lambda exclude=None: [])

    def earthdata(res):
        return [(lv, t) for lv, t in res if "Earthdata" in t]
    monkeypatch.setattr(da, "_earthdata_login_status",
                        lambda: ("unreachable", "NASA Earthdata or ASF answered with a server error (503)"))
    lines = earthdata(dash.preflight_checks(cfg, "x.yaml", "full"))
    assert len(lines) == 1 and lines[0][0] == "warn"
    assert "the Earthdata / ASF servers did not answer; try again later" in lines[0][1]
    assert "netrc" not in lines[0][1]
    monkeypatch.setattr(da, "_earthdata_login_status",
                        lambda: ("rejected", "NASA Earthdata rejected the login for 'kyle' (... ~/.netrc)"))
    lines = earthdata(dash.preflight_checks(cfg, "x.yaml", "full"))
    assert len(lines) == 1 and lines[0][0] == "fail" and "rejected" in lines[0][1]
