"""7 Oct 2026 fixes, group F_acquisition (data_acquisition.py). Each test here
failed on the code in backups/pre_fixes_20261007/ unless its docstring says
otherwise. All synthetic: no network, no ISCE.

#28 a download pass refused by the plan check rewrote dropped_short_dates.json
    without the render-gap dates it was refusing, and left assembly_status.json
    with no 'processed'; dates accepted with allow_missing_dates went only to
    'accepted_not_in_plan', which QA and provenance never read;
#29 one CMR hiccup dropped a short date for good, without stopping the run;
#30 an Earthdata/ASF server outage (5xx) was reported as a wrong password or a
    missing ASF authorisation.
"""
import json

import pytest

import test_fix_L6 as L6                                        # download fixtures
from insar_pipeline import data_acquisition as da
from insar_pipeline import procs
from insar_pipeline import provenance
from insar_pipeline import quality_assessment as qa

RENDER_GAP, SHORT = "20250517", "20160516"
PLANNED = ["20200101", "20200113", "20200125"]
DATES = [SHORT, *PLANNED, RENDER_GAP]


def _entries(complete=(RENDER_GAP,)):
    """catalogue_burst_entries stand-in: every modal burst for the dates in
    `complete`, one burst for the rest (really short in the archive)."""
    def entries(cfg_, date, pol, swaths):
        n = 3 if date in complete else 1
        return {f"0388{86 + i}_IW2": {"title": L6.granules(date)[i], "swath": "IW2", "pol": "VV",
                                      "slc": "S1A_X", "index": i + 1, "tiff": "t", "xml": "x"}
                for i in range(n)}
    return entries


def _no_waiting(monkeypatch):
    monkeypatch.setattr(da.time, "sleep", lambda s: None)
    monkeypatch.setattr(da, "_earthdata_login_status", lambda: ("ok", "fine"))
    monkeypatch.setattr(procs, "install_teardown", lambda: None)


def _finished_phase_linked(tmp_path, **over):
    """Site B's shape in miniature: the plan holds three dates, two of them
    coregistered and cleaned after the phase-linked merge, the reference with
    its SAFE; a legacy dropped_short_dates.json lists both short dates."""
    cfg = L6.make_cfg(tmp_path, start_date="2016-01-01", end_date="2026-06-30", **over)
    (cfg.stack_dir / "run_files").mkdir()
    (cfg.stack_dir / "generation.json").write_text(json.dumps({"argv": ["x"], "input_dates": PLANNED}))
    for d in PLANNED[1:]:
        L6.write_cleaned_pl(cfg, d)
    L6.fake_safe(cfg.slc_dir, PLANNED[0])
    (cfg.slc_dir / "dropped_short_dates.json").write_text(json.dumps(
        {"modal_burst_count": 3, "dropped": [SHORT, RENDER_GAP]}))
    return cfg


# ---------------------------------------------------------------- #28 records kept on a refused run
def test_refused_run_keeps_every_reason_and_what_is_processed(tmp_path, monkeypatch):
    cfg = _finished_phase_linked(tmp_path)
    _no_waiting(monkeypatch)
    monkeypatch.setattr(da, "search_bursts_complete",
                        L6.fake_search(DATES, short={SHORT: 1, RENDER_GAP: 1}))
    monkeypatch.setattr(da, "catalogue_burst_entries", _entries())
    rec = L6.Recorder()
    monkeypatch.setattr(procs, "run_group", rec)
    before = qa.date_accounting(cfg, PLANNED)
    assert before["text"] == "3 of 5 images used; 2 left out (see dropped)"

    with pytest.raises(RuntimeError, match=RENDER_GAP):
        da.download_bursts_as_safes(cfg)
    assert rec.calls == []                                   # refused before downloading
    ds = json.loads((cfg.slc_dir / "dropped_short_dates.json").read_text())
    assert ds["dropped"] == [SHORT, RENDER_GAP]              # the refusal erased neither
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert st["dropped"] == {SHORT: "catalogue_short", RENDER_GAP: "not_in_plan"}
    assert st["dropped_detail"][RENDER_GAP]["reason"] == "not_in_plan"
    assert st["processed"] == PLANNED                        # the reference SAFE + two coregistered
    after = qa.date_accounting(cfg, PLANNED)
    assert sorted(after["dropped"]) == [SHORT, RENDER_GAP]
    assert after["text"] == before["text"] and not after["unexplained"]
    assert provenance._acquisitions(cfg)["processed"] == 3


def test_accepted_not_in_plan_dates_are_recorded_as_left_out(tmp_path, monkeypatch):
    """The refusal's advice (allow_missing_dates) records the date as left
    out, as the message says, instead of in a key nobody reads."""
    cfg = _finished_phase_linked(tmp_path, allow_missing_dates=(RENDER_GAP,))
    _no_waiting(monkeypatch)
    monkeypatch.setattr(da, "search_bursts_complete",
                        L6.fake_search(DATES, short={SHORT: 1, RENDER_GAP: 1}))
    monkeypatch.setattr(da, "catalogue_burst_entries", _entries())
    monkeypatch.setattr(da, "recover_from_extractor",
                        lambda *a, **k: pytest.fail("an accepted date is not fetched"))
    rec = L6.Recorder()
    monkeypatch.setattr(procs, "run_group", rec)
    out = da.download_bursts_as_safes(cfg)
    assert [p.name[17:25] for p in out] == [PLANNED[0]] and rec.calls == []
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert st["dropped"][RENDER_GAP] == "not_in_plan"
    assert st["accepted_not_in_plan"] == [RENDER_GAP]
    assert json.loads((cfg.slc_dir / "dropped_short_dates.json").read_text())["dropped"] == [SHORT, RENDER_GAP]
    acc = qa.date_accounting(cfg, PLANNED)
    assert acc["text"].endswith("2 left out (see dropped)")
    assert acc["dropped"][RENDER_GAP].endswith("(accepted as missing by allow_missing_dates)")


def test_refused_login_record_says_what_is_processed(tmp_path, monkeypatch):
    """The other early stop (a refused login) also leaves 'processed', so
    provenance does not read 0 processed from it."""
    cfg = L6.make_cfg(tmp_path)
    _no_waiting(monkeypatch)
    L6.fake_safe(cfg.slc_dir, "20200101")
    monkeypatch.setattr(da, "search_bursts_complete", L6.fake_search(["20200101", "20200113"]))
    monkeypatch.setattr(da, "_earthdata_login_status", lambda: ("rejected", "NASA Earthdata rejected the login"))
    with pytest.raises(RuntimeError, match="rejected"):
        da.download_bursts_as_safes(cfg)
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert st["processed"] == ["20200101"]


# ---------------------------------------------------------------- #29 catalogue hiccups
def test_catalogue_check_is_retried_before_giving_up(tmp_path, monkeypatch):
    import requests
    cfg = L6.make_cfg(tmp_path)
    calls, sleeps = [], []
    full = _entries(complete=("20200113",))

    def flaky(cfg_, date, pol, swaths):
        calls.append(date)
        if len(calls) <= 2:
            raise requests.ConnectionError("read timed out")
        return full(cfg_, date, pol, swaths)
    monkeypatch.setattr(da, "catalogue_burst_entries", flaky)
    monkeypatch.setattr(da.time, "sleep", sleeps.append)
    keys = {f"0388{86 + i}_IW2" for i in range(3)}
    out = da.classify_short_dates(cfg, {"20200113": L6.granules("20200113", 1)}, ["20200113"],
                                  keys, "VV", {"IW2"})
    assert out["20200113"]["reason"] == "render_gap" and len(calls) == 3
    assert sleeps == list(da.CATALOGUE_BACKOFF_S)


def test_unreachable_catalogue_stops_the_run_unless_accepted(tmp_path, monkeypatch):
    import requests
    cfg = L6.make_cfg(tmp_path)
    _no_waiting(monkeypatch)
    dates = ["20200101", "20200113", "20200125", "20200206"]
    gap = "20200125"
    monkeypatch.setattr(da, "search_bursts_complete", L6.fake_search(dates, short={gap: 1}))
    tries = []

    def down(cfg_, date, pol, swaths):
        tries.append(date)
        raise requests.ConnectionError("503 from CMR")
    monkeypatch.setattr(da, "catalogue_burst_entries", down)
    monkeypatch.setattr(procs, "run_group", L6.Recorder())
    with pytest.raises(RuntimeError, match=gap):
        da.download_bursts_as_safes(cfg)
    assert tries == [gap] * da.CATALOGUE_ATTEMPTS
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert gap in st["failed"] and "catalogue unreachable" in st["failed"][gap]
    assert gap not in st["dropped"]
    # the next run checks it again; listed in allow_missing_dates, the run goes on
    cfg.allow_missing_dates = (gap,)
    out = da.download_bursts_as_safes(cfg)
    assert sorted(p.name[17:25] for p in out) == ["20200101", "20200113", "20200206"]
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert st["accepted_failures"] == [gap]
    acc = qa.date_accounting(cfg, ["20200101", "20200113", "20200206"])
    assert acc["dropped"][gap].endswith("(accepted as missing by allow_missing_dates)")
    assert acc["text"] == "3 of 4 images used; 1 left out (see dropped)"


def test_catalogue_hiccup_on_a_coregistered_date_does_not_stop_the_run(tmp_path, monkeypatch):
    """Review of #29: a short date already in the stack needs no catalogue
    answer (nothing is fetched for it), so three failed checks on a resume
    neither stop the run nor record a date the stack holds as failed."""
    import requests
    cfg = L6.make_cfg(tmp_path, start_date="2016-01-01", end_date="2026-06-30")
    (cfg.stack_dir / "run_files").mkdir()
    (cfg.stack_dir / "generation.json").write_text(json.dumps(
        {"argv": ["x"], "input_dates": [*PLANNED, RENDER_GAP]}))
    for d in [*PLANNED[1:], RENDER_GAP]:
        L6.write_cleaned_pl(cfg, d)
    L6.fake_safe(cfg.slc_dir, PLANNED[0])
    _no_waiting(monkeypatch)
    monkeypatch.setattr(da, "search_bursts_complete",
                        L6.fake_search([*PLANNED, RENDER_GAP], short={RENDER_GAP: 1}))

    def down(cfg_, date, pol, swaths):
        raise requests.ConnectionError("CMR read timed out")
    monkeypatch.setattr(da, "catalogue_burst_entries", down)
    rec = L6.Recorder()
    monkeypatch.setattr(procs, "run_group", rec)
    out = da.download_bursts_as_safes(cfg)
    assert [p.name[17:25] for p in out] == [PLANNED[0]] and rec.calls == []
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert st["failed"] == {} and RENDER_GAP not in st["dropped"]
    assert RENDER_GAP in st["skipped_coregistered"]
    assert st["processed"] == sorted([*PLANNED, RENDER_GAP])
    assert json.loads((cfg.slc_dir / "dropped_short_dates.json").read_text())["dropped"] == []


# ---------------------------------------------------------------- #30 server outage is not a bad login
def test_server_error_is_an_outage_not_a_wrong_password(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARTHDATA_USERNAME", "kyle")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "right")

    class ASFAuthenticationError(Exception):
        pass

    def session(exc):
        class S:
            def auth_with_creds(self, u, p):
                raise exc
        return S
    # the texts asf_search 13.1 builds around requests' HTTPError
    for text in ("Failed to log in with provided credentials. Original Exception: 503 Server Error: "
                 "Service Unavailable for url: https://urs.earthdata.nasa.gov/api/users/find_or_create_token",
                 "Failed to set asf-urs cookies. Original exception: 502 Server Error: Bad Gateway for url: "
                 "https://urs.earthdata.nasa.gov/oauth/authorize"):
        monkeypatch.setattr(da.asf, "ASFSession", session(ASFAuthenticationError(text)))
        code, msg = da._earthdata_login_status()
        assert code == "unreachable" and "not a login problem" in msg
        assert "wrong username" not in msg and "authorised ASF" not in msg
    # the real refusals keep their codes
    monkeypatch.setattr(da.asf, "ASFSession", session(ASFAuthenticationError(
        "Failed to log in with provided credentials. Original Exception: 401 Client Error: Unauthorized")))
    assert da._earthdata_login_status()[0] == "rejected"
    monkeypatch.setattr(da.asf, "ASFSession", session(ASFAuthenticationError(
        "Failed to set asf-urs cookies. Original exception: Failed to find `asf-urs` cookie in response")))
    assert da._earthdata_login_status()[0] == "not_authorised"
