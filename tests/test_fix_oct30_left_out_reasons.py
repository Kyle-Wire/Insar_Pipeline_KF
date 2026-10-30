"""30 Oct 2026: the reasons the report gives for left-out dates. The first
city site's methods sentence gave two false or ungrammatical ones.

  a catalogue-defect date accepted automatically (record_auto_accepted) was
  never in the stack's plan, so the next download relabelled it 'not_in_plan'
  and the report said it "was found in the archive only after this project's
  processing had started"; it keeps the defect reason now, from the last
  download record or the config's auto-accept note
  a date left out of the stack on purpose (stack_exclude_dates) was also
  relabelled 'not_in_plan' in the download record
  a date the weather model has no data for read "in the interferogram stack
  but not in the time series (all its pairs dropped)"; it has its own reason

The client sentence itself (every reason a clause after "because") is
tested in test_fix_oct7_int_I1_client_report.py.

Each test failed on the code before these fixes, except the one whose docstring
says it is unchanged. All synthetic: no network.
Run: conda run -n insar python -m pytest tests/test_fix_oct30_left_out_reasons.py -q
"""
import json

import pytest

import test_fix_L6 as L6                                        # download fixtures
import test_fix_oct7_F_acquisition as F
from insar_pipeline import data_acquisition as da
from insar_pipeline import procs
from insar_pipeline import quality_assessment as qa

DEFECT = "burst annotations empty (catalogue returned the date short)"
LATE = "found in the archive only after"


def _with_config_file(cfg, tmp_path):
    """The config the run was started from, as load_config leaves it."""
    src = tmp_path / "site.yaml"
    src.write_text(f"project_dir: {cfg.project_dir}\nallow_missing_dates: []\n")
    cfg._source_path = src
    return cfg


def _second_download(cfg, monkeypatch):
    """The download pass after the one that failed: coregistration has
    started and the defect date is not in the plan."""
    F._no_waiting(monkeypatch)
    monkeypatch.setattr(da, "search_bursts_complete",
                        L6.fake_search(F.DATES, short={F.SHORT: 1, F.RENDER_GAP: 1}))
    monkeypatch.setattr(da, "catalogue_burst_entries", F._entries())
    monkeypatch.setattr(da, "recover_from_extractor",
                        lambda *a, **k: pytest.fail("an accepted date is not fetched"))
    monkeypatch.setattr(procs, "run_group", L6.Recorder())
    da.download_bursts_as_safes(cfg)
    return json.loads((cfg.slc_dir / "assembly_status.json").read_text())


def _defect_kept(cfg, st):
    assert F.RENDER_GAP not in st["dropped"]
    assert st["failed"][F.RENDER_GAP] == da.DEFECT_PREFIX + DEFECT
    assert F.RENDER_GAP in st["accepted_failures"] and F.RENDER_GAP not in st["accepted_not_in_plan"]
    why = qa.date_accounting(cfg, F.PLANNED)["dropped"][F.RENDER_GAP]
    assert LATE not in why
    assert why.startswith(qa._CATALOGUE_DEFECT.format(DEFECT))


# ------------------------------------------------------------------ the defect reason survives a later run
def test_a_defect_date_keeps_its_reason_from_the_config_note(tmp_path, monkeypatch):
    """The first city site's case: the note record_auto_accepted wrote is the
    only record left of why the date is missing."""
    cfg = _with_config_file(F._finished_phase_linked(tmp_path), tmp_path)
    da.record_auto_accepted(cfg, {F.RENDER_GAP: DEFECT})
    assert da.auto_accepted_in_config(cfg) == {F.RENDER_GAP: DEFECT}
    _defect_kept(cfg, _second_download(cfg, monkeypatch))


def test_a_defect_date_keeps_its_reason_from_the_last_download_record(tmp_path, monkeypatch):
    """No config note (accepted by hand): the last assembly_status.json's
    accepted failure is carried forward, and so survives the run after too."""
    cfg = F._finished_phase_linked(tmp_path, allow_missing_dates=(F.RENDER_GAP,))
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps(
        {"failed": {F.RENDER_GAP: da.DEFECT_PREFIX + DEFECT}, "accepted_failures": [F.RENDER_GAP]}))
    _defect_kept(cfg, _second_download(cfg, monkeypatch))
    _defect_kept(cfg, _second_download(cfg, monkeypatch))


def test_a_record_already_relabelled_reads_the_config_note(tmp_path):
    """A download record written before the fix (the site's slc/ as it is):
    the QA takes the defect reason from the config note."""
    cfg = _with_config_file(L6.make_cfg(tmp_path), tmp_path)
    D = "20200125"
    da.record_auto_accepted(cfg, {D: DEFECT})
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps(
        {"catalogue_dates": ["20200101", "20200113", D], "dropped": {D: "not_in_plan"},
         "accepted_not_in_plan": [D], "failed": {}, "render_gap": [D]}))
    r = qa.date_accounting(cfg, ["20200101", "20200113"])
    assert r["dropped"][D].startswith(qa._CATALOGUE_DEFECT.format(DEFECT)) and LATE not in r["dropped"][D]
    assert not r["unexplained"]


def test_a_date_the_plan_does_not_take_is_still_called_late(tmp_path, monkeypatch):
    """Nothing earlier says why: an accepted date beyond the plan is still
    'not_in_plan' (test_fix_oct7_F_acquisition's case, unchanged)."""
    cfg = _with_config_file(F._finished_phase_linked(tmp_path, allow_missing_dates=(F.RENDER_GAP,)), tmp_path)
    st = _second_download(cfg, monkeypatch)
    assert st["dropped"][F.RENDER_GAP] == "not_in_plan" and F.RENDER_GAP not in st["failed"]


def test_a_date_left_out_on_purpose_is_not_called_late(tmp_path, monkeypatch):
    """stack_exclude_dates: recorded as excluded, not as a date the plan
    cannot take (the site's 20250429 was relabelled 'not_in_plan')."""
    extra = "20200206"
    cfg = F._finished_phase_linked(tmp_path, stack_exclude_dates=(extra,))
    F._no_waiting(monkeypatch)
    monkeypatch.setattr(da, "search_bursts_complete", L6.fake_search(F.PLANNED + [extra]))
    monkeypatch.setattr(procs, "run_group", L6.Recorder())
    da.download_bursts_as_safes(cfg)
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert extra not in st["dropped"] and st["excluded"] == [extra]
    assert qa.date_accounting(cfg, F.PLANNED)["dropped"][extra].startswith(qa._EXCLUDED_FROM_STACK)


# ------------------------------------------------------------------ no weather data
def test_a_date_without_weather_data_says_so(tmp_path):
    """Its pairs are in ifgramStack.h5 and it is not in the series: the
    reason is the weather check's, not 'all its pairs dropped'."""
    h5py = pytest.importorskip("h5py")
    D = "20200125"
    cfg = L6.make_cfg(tmp_path, mintpy_tropo_method="raider", weather_model="hrrr")
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    stack.parent.mkdir(parents=True)
    with h5py.File(stack, "w") as f:
        f["date"] = [[b"20200101", b"20200113"], [b"20200113", D.encode()], [b"20200101", D.encode()]]
    u = cfg.tropo_dir / "HRRR" / "unavailable_dates.json"
    u.parent.mkdir(parents=True)
    u.write_text(json.dumps({"model": "HRRR", "dates": {D: "NOAA's HRRR archive has no file for the hours "
                                                           "around the pass"}}))
    why = qa.date_accounting(cfg, ["20200101", "20200113"])["dropped"][D]
    assert why == qa._NO_WEATHER.format("HRRR", "NOAA's HRRR archive has no file for the hours around the pass")
    assert "interferogram stack" not in why
