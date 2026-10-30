"""
stack_exclude_dates (29 Oct 2026): a date left out of the Sentinel-1 stack on
purpose. The first city site: one date coregistered, but its bottom
burst-overlap image came out all zeros, so every ESD pair with it failed and
the misregistration network check stopped the run. allow_missing_dates could
not help - the date was not missing, it was in the plan.

  a new plan gets the dates as stackSentinel -x
  a plan already under way is reused, the dates' commands skipped
  the dates' part-made coregistered images are moved, not deleted
  the misregistration check then sees a connected network
  the reference date cannot be excluded
  the dates are accepted missing and not downloaded

Run: conda run -n insar python -m pytest tests/test_stack_exclude_dates.py -q
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import config as cf                   # noqa: E402
from insar_pipeline import data_acquisition as da         # noqa: E402
from insar_pipeline import procs                          # noqa: E402
from insar_pipeline import stack_backend as sb            # noqa: E402
import test_fix_L6 as l6                                  # noqa: E402

quiet = l6.quiet
DATES = ["20200101", "20200113", "20200125", "20200206", "20200218"]
BAD = "20200125"


def _misreg_run_file(cfg, dates):
    """run_07 as stackSentinel writes it: each date with the next two."""
    lines = [f"SentinelWrapper.py -c {cfg.stack_dir}/configs/config_misreg_{a}_{b}"
             for i, a in enumerate(dates) for b in dates[i + 1:i + 3]]
    rf = cfg.stack_dir / "run_files"
    rf.mkdir(parents=True, exist_ok=True)
    (rf / "run_07_pairs_misreg").write_text("\n".join(lines) + "\n")
    return rf / "run_07_pairs_misreg"


def test_a_new_plan_gets_them_as_stacksentinel_x(tmp_path):
    cfg = l6.make_cfg(tmp_path, stack_exclude_dates=("20200206", BAD))
    argv = sb._stack_argv(cfg, Path("/dem"))
    assert argv[argv.index("-x") + 1] == "20200125,20200206"
    assert "-x" not in sb._stack_argv(l6.make_cfg(tmp_path / "b"), Path("/dem"))


def test_a_plan_under_way_is_reused_and_the_dates_commands_skipped(tmp_path, monkeypatch):
    cfg = l6.make_cfg(tmp_path)
    for d in DATES:
        l6.fake_safe(cfg.slc_dir, d)
    calls = []
    monkeypatch.setattr(sb.subprocess, "run", l6._fake_stacksentinel(cfg, calls))
    sb._generate_run_files(cfg, Path("/dem"))
    f = _misreg_run_file(cfg, DATES)
    for d in DATES[1:]:
        l6.write_resampled(cfg, d)                               # coregistration under way
    cfg.stack_exclude_dates = (BAD,)
    argv, files = sb._generate_run_files(cfg, Path("/dem"))
    assert len(calls) == 1                                       # stackSentinel not asked to re-plan
    assert (cfg.stack_dir / "run_files").exists() and not list(cfg.stack_dir.glob("run_files.superseded_*"))
    rec = json.loads((cfg.stack_dir / "generation.json").read_text())
    assert rec["argv"] == argv and "-x" in argv and rec["exclude_dates_applied_to_plan"]
    # the same arguments next time: an ordinary reuse
    sb._generate_run_files(cfg, Path("/dem"))
    assert len(calls) == 1
    kept = sb.plan_commands(cfg, f)
    assert len(sb.parse_run_file(f)) == 7 and len(kept) == 3
    assert not any(BAD in c for c in kept)
    assert BAD not in sb.planned_dates(cfg)
    assert sb.dates_missing_from_plan(cfg) == []


def test_the_dates_part_made_images_are_moved_not_deleted(tmp_path):
    cfg = l6.make_cfg(tmp_path, stack_exclude_dates=(BAD,))
    l6.write_resampled(cfg, "20200113")
    part = cfg.stack_dir / "coreg_secondarys" / BAD / "overlap" / "IW2"
    part.mkdir(parents=True)
    (part / "burst_bot_01_02.slc").write_bytes(b"\0" * 64)
    moved = sb.set_aside_excluded(cfg)
    assert moved == [f"coreg_secondarys/{BAD}"]
    assert not (cfg.stack_dir / "coreg_secondarys" / BAD).exists()
    assert (cfg.stack_dir / "excluded_dates" / "coreg_secondarys" / BAD / "overlap" / "IW2"
            / "burst_bot_01_02.slc").exists()
    assert (cfg.stack_dir / "coreg_secondarys" / "20200113").exists()  # the others stay
    assert sb.set_aside_excluded(cfg) == []                              # nothing left to move


def test_the_misreg_check_then_sees_a_connected_network(tmp_path, monkeypatch):
    """Every pair with the bad date failed: refused before, connected without it."""
    cfg = l6.make_cfg(tmp_path)
    f = _misreg_run_file(cfg, DATES)

    def result(c):
        return {"cmd": c, "returncode": 1 if BAD in c else 0, "seconds": 1.0, "peak_rss_gb": None,
                "skipped": False}
    monkeypatch.setattr(sb, "_misreg_pair_estimated", lambda cmd: True)
    with pytest.raises(RuntimeError, match=BAD):
        sb.check_misreg_network(cfg, f.name, [result(c) for c in sb.plan_commands(cfg, f)])
    cfg.stack_exclude_dates = (BAD,)
    out = sb.check_misreg_network(cfg, f.name, [result(c) for c in sb.plan_commands(cfg, f)])
    assert out == {"misreg_network_connected": True}


def test_the_reference_date_cannot_be_excluded(tmp_path):
    cfg = l6.make_cfg(tmp_path, stack_exclude_dates=("20200101",))
    assert any("reference date" in p for p in cf.validate_config(cfg))
    (cfg.stack_dir / "reference").mkdir()
    (cfg.stack_dir / "reference" / "IW2.xml").write_text("<product/>")
    with pytest.raises(RuntimeError, match="reference date"):
        sb.set_aside_excluded(cfg)
    assert any("YYYYMMDD" in p for p in cf.validate_config(l6.make_cfg(tmp_path / "b", stack_exclude_dates=("2020-1-25",))))


def test_they_are_accepted_missing_and_not_downloaded(tmp_path, monkeypatch, quiet):
    cfg = l6.make_cfg(tmp_path, stack_exclude_dates=(BAD,))
    assert da.accepted_missing_dates(cfg) == ({BAD}, False)
    monkeypatch.setattr(da, "search_bursts_complete", l6.fake_search(DATES[:4]))
    rec = l6.Recorder()
    monkeypatch.setattr(procs, "run_group", rec)
    da.download_bursts_as_safes(cfg)
    assert sorted(c[0] for c in rec.calls) == ["20200101", "20200113", "20200206"]
    assert json.loads((cfg.slc_dir / "assembly_status.json").read_text())["excluded"] == [BAD]


def test_the_misreg_inversion_is_not_asked_for_their_correction(tmp_path):
    """run_08 checks a <date>.txt for every date of the misreg configs: an
    excluded date has configs from the old plan but no pairs, so none."""
    cfg = l6.make_cfg(tmp_path)
    (cfg.stack_dir / "configs").mkdir()
    for a, b in (("20200113", BAD), (BAD, "20200206"), ("20200113", "20200206")):
        (cfg.stack_dir / "configs" / f"config_misreg_{a}_{b}").write_text("x\n")
    cmd = f"invertMisreg.py -i {cfg.stack_dir}/misreg/azimuth/pairs/ -o {cfg.stack_dir}/misreg/azimuth/dates/"
    want = {p.name for p, _ in sb._configless_outputs(cfg, cmd)}
    assert want == {"20200113.txt", f"{BAD}.txt", "20200206.txt"}
    cfg.stack_exclude_dates = (BAD,)
    assert {p.name for p, _ in sb._configless_outputs(cfg, cmd)} == {"20200113.txt", "20200206.txt"}


def test_the_qa_names_them_left_out_on_purpose_not_unexplained(tmp_path):
    """The excluded date's SAFE stays in slc/: without a reason the QA's
    date accounting called it unexplained and held the report."""
    from insar_pipeline import client_report as cr, quality_assessment as qa
    cfg = l6.make_cfg(tmp_path)
    for d in DATES:
        l6.fake_safe(cfg.slc_dir, d)
    proc = [d for d in DATES if d != BAD]
    assert qa.date_accounting(cfg, proc)["unexplained"] == [BAD]
    cfg.stack_exclude_dates = (BAD,)
    r = qa.date_accounting(cfg, proc)
    assert r["unexplained"] == [] and not r["hold"] and r["dropped"][BAD] == qa._EXCLUDED_FROM_STACK
    assert "stack_exclude" not in r["dropped"][BAD]
    assert cr._plural_reason(qa._EXCLUDED_FROM_STACK).startswith("they could not be aligned with the other dates")


def test_on_a_stack_under_way_they_are_not_called_beyond_the_plan(tmp_path, monkeypatch, quiet):
    """The first city site: once excluded, the date dropped out of the planned
    dates and the download record called it 'found only after processing
    started' (not_in_plan), and the QA added 'accepted by allow_missing_dates'."""
    from insar_pipeline import quality_assessment as qa
    cfg = l6.make_cfg(tmp_path, stack_exclude_dates=(BAD,))
    l6.write_cleaned_pl(cfg, "20200113")
    (cfg.stack_dir / "run_files").mkdir()
    (cfg.stack_dir / "generation.json").write_text(json.dumps({"argv": ["x"], "input_dates": DATES[:4]}))
    l6.fake_safe(cfg.slc_dir, "20200101")
    monkeypatch.setattr(da, "search_bursts_complete", l6.fake_search(DATES[:4]))
    monkeypatch.setattr(procs, "run_group", l6.Recorder())
    da.download_bursts_as_safes(cfg)
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert BAD not in st["dropped"] and BAD not in st.get("accepted_not_in_plan", []) and st["excluded"] == [BAD]
    st["accepted_missing"] = [BAD]                       # as an earlier run's allow_missing_dates left it
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps(st))
    r = qa.date_accounting(cfg, ["20200101", "20200113", "20200206"])
    assert r["dropped"][BAD] == qa._EXCLUDED_FROM_STACK
