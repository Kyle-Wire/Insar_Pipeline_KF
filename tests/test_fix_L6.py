"""
Lane L6a1 (review 5 Oct 2026): acquisition accounting and the first half of
the stack runner. Each test fails on the pre-fix code and passes now; all
synthetic, no network, no ISCE.

  stackrunner-1 / ops-7 / tests-3   every date accounted for: retries, slc/assembly_status.json,
                                    loud failure unless allow_missing_dates; the plan's input
                                    dates in generation.json, regenerate or refuse on change
  stackrunner-2                     short dates checked against the plain-JSON catalogue
  stackrunner-4 / ops-2             .cleaned dates are not re-downloaded; no re-plan of a
                                    cleaned stack; geom_reference.cleaned marker
  phaselinked-9 / stackrunner-9     SAFE cleanup waits for the COMPLETE resample
  stackrunner-10                    run_04/08/11 output predicates, misreg connectivity, real merge size
  stackrunner-12                    Earthdata login actually tried
  stackrunner-13                    existing SAFEs verified, partials quarantined, timeout
  stackrunner-15                    DEM/AOI check before cleanup, AOI outside the frame
  ops-3                             download/orbit progress events

Run: conda run -n insar python -m pytest tests/test_fix_L6.py -q
"""
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import data_acquisition as da         # noqa: E402
from insar_pipeline import procs                          # noqa: E402
from insar_pipeline import progress                       # noqa: E402
from insar_pipeline import stack_backend as sb            # noqa: E402
from insar_pipeline.config import PipelineConfig          # noqa: E402


# ------------------------------------------------------------------ helpers

def make_cfg(tmp_path, **over) -> PipelineConfig:
    kw = dict(project_dir=tmp_path / "proj", bbox=[-87.8, 38.2, -87.52, 38.47],
              start_date="2020-01-01", end_date="2020-12-31", backend="miaplpy",
              stack_reference_date="20200101", track=19, stack_swaths="2",
              stack_polarization="vv")
    kw.update(over)
    cfg = PipelineConfig(**kw)
    cfg.slc_dir.mkdir(parents=True, exist_ok=True)
    cfg.stack_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def safe_name(d: str) -> str:
    return f"S1A_IW_SLC__1SSV_{d}T234807_{d}T234813_065016_0831B4_CAF2.SAFE"


def fake_safe(parent: Path, d: str, bursts: int = 3, lines_per_burst: int = 10,
              samples: int = 12, truncate: bool = False, manifest: bool = True) -> Path:
    """A burst2safe-shaped SAFE: manifest, IW2 VV annotation, measurement tiff."""
    s = Path(parent) / safe_name(d)
    (s / "annotation").mkdir(parents=True, exist_ok=True)
    (s / "measurement").mkdir(parents=True, exist_ok=True)
    if manifest:
        (s / "manifest.safe").write_text("<xfdu/>")
    lines = bursts * lines_per_burst
    stem = f"s1a-iw2-slc-vv-{d.lower()}t234807-{d.lower()}t234816-065016-0831b4-001"
    (s / "annotation" / f"{stem}.xml").write_text(
        f"<product><imageInformation><numberOfLines>{lines}</numberOfLines>"
        f"<numberOfSamples>{samples}</numberOfSamples></imageInformation>"
        f'<swathTiming><burstList count="{bursts}"></burstList></swathTiming></product>')
    n = lines * samples * 4 + 64
    (s / "measurement" / f"{stem}.tiff").write_bytes(b"\0" * (n - 200 if truncate else n))
    return s


def granules(d: str, n: int = 3) -> list:
    return [f"S1_0388{86 + i}_IW2_{d}T234808_VV_1B94-BURST" for i in range(n)]


def fake_search(dates, short=None):
    """search_bursts_complete stand-in: {date: granules} with optional short dates."""
    short = short or {}

    def search(cfg, swaths, pol, want, attempts=4, pause_s=20.0, footprints=None):
        by = {d: granules(d, short.get(d, 3)) for d in dates}
        return by, [-87.9, -87.0, -87.0, -87.9], [38.1, 38.1, 38.6, 38.6]
    return search


class Recorder:
    """procs.run_group stand-in: burst2safe writes a SAFE into --output-dir,
    or fails per a per-date script."""

    def __init__(self, script=None, text_fail="Traceback: connection reset"):
        self.calls = []
        self.script = script or {}
        self.text_fail = text_fail

    def __call__(self, cmd, timeout=None, **kw):
        out = Path(cmd[cmd.index("--output-dir") + 1])
        d = cmd[-1].split("_")[3][:8]
        self.calls.append((d, timeout))
        plan = self.script.get(d, [])
        n_before = sum(1 for c in self.calls if c[0] == d) - 1
        rc = plan[n_before] if n_before < len(plan) else 0
        if rc == "partial":
            fake_safe(out, d, truncate=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if rc:
            return subprocess.CompletedProcess(cmd, rc, "", self.text_fail)
        fake_safe(out, d)
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def quiet(monkeypatch):
    """No sleeping, no real login, no signal handlers from worker threads."""
    monkeypatch.setattr(da.time, "sleep", lambda s: None)
    monkeypatch.setattr(da, "_earthdata_login_status", lambda: ("ok", "fine"), raising=False)
    monkeypatch.setattr(procs, "install_teardown", lambda: None)
    monkeypatch.setattr(da, "classify_short_dates",
                        lambda cfg, by, short, keys, pol, sw: {d: {"reason": "catalogue_short",
                                                                   "bursts_found": 1,
                                                                   "bursts_needed": 3}
                                                               for d in short},
                        raising=False)


def write_cleaned_pl(cfg, d):
    """The phase-linked merge's layout for one date (Site B, 269 of these)."""
    coreg = cfg.stack_dir / "coreg_secondarys"
    coreg.mkdir(parents=True, exist_ok=True)
    (coreg / f"{d}.cleaned").write_text("merged real SLC verified; bursts removed by miaplpy_backend\n")
    m = cfg.stack_dir / "merged" / "SLC" / d
    m.mkdir(parents=True, exist_ok=True)
    (m / "DELETED_AFTER_MIAPLPY_LOAD").write_text("x\n")


def write_resampled(cfg, d, bursts=3, written=None, xml=True):
    """coreg_secondarys/<d>/ as resamp_withCarrier leaves it: burst SLCs, and
    IW2.xml (written last) listing numberofbursts."""
    dd = cfg.stack_dir / "coreg_secondarys" / d
    (dd / "IW2").mkdir(parents=True, exist_ok=True)
    for i in range(1, (bursts if written is None else written) + 1):
        (dd / "IW2" / f"burst_{i:02d}.slc").write_bytes(b"\0" * 64)
    if xml:
        (dd / "IW2.xml").write_text(f'<product><property name="numberofbursts">'
                                    f'<value>{bursts}</value></property></product>')
    return dd


# ============================================ stackrunner-1 / ops-7 / tests-3

def _fake_stacksentinel(cfg, calls, stdout=""):
    def run(argv, cwd=None, env=None, capture_output=True, text=True):
        calls.append(list(argv))
        if stdout:
            return subprocess.CompletedProcess(argv, 1, stdout, "")
        rf = cfg.stack_dir / "run_files"
        rf.mkdir(parents=True, exist_ok=True)
        safes = sorted(cfg.slc_dir.glob("S1*_IW_SLC*.SAFE"))
        (rf / "run_01_unpack_topo_reference").write_text(
            "".join(f"# {s.name}\n" for s in safes) + "SentinelWrapper.py -c x\n")
        (cfg.stack_dir / "SAFE_files.txt").write_text("".join(f"{s}\n" for s in safes))
        return subprocess.CompletedProcess(argv, 0, "ok", "")
    return run


def test_new_safe_before_coreg_regenerates_and_records_input_dates(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    for d in ("20200101", "20200113", "20200125"):
        fake_safe(cfg.slc_dir, d)
    calls = []
    monkeypatch.setattr(sb.subprocess, "run", _fake_stacksentinel(cfg, calls))
    sb._generate_run_files(cfg, Path("/dem"))
    rec = json.loads((cfg.stack_dir / "generation.json").read_text())
    assert rec["input_dates"] == ["20200101", "20200113", "20200125"]
    # same argv, nothing new: the plan is reused
    sb._generate_run_files(cfg, Path("/dem"))
    assert len(calls) == 1
    # a date that failed to download arrives on a resume: old code reused the plan
    fake_safe(cfg.slc_dir, "20200206")
    sb._generate_run_files(cfg, Path("/dem"))
    assert len(calls) == 2
    assert list(cfg.stack_dir.glob("run_files.superseded_*"))
    assert "20200206" in json.loads((cfg.stack_dir / "generation.json").read_text())["input_dates"]


def test_new_safe_after_coreg_started_refuses_naming_the_date(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    for d in ("20200101", "20200113", "20200125"):
        fake_safe(cfg.slc_dir, d)
    calls = []
    monkeypatch.setattr(sb.subprocess, "run", _fake_stacksentinel(cfg, calls))
    sb._generate_run_files(cfg, Path("/dem"))
    write_cleaned_pl(cfg, "20200113")                       # coregistration has started
    fake_safe(cfg.slc_dir, "20200119")                      # 20161206-style late arrival
    with pytest.raises(RuntimeError, match="20200119"):
        sb._generate_run_files(cfg, Path("/dem"))
    assert len(calls) == 1                                  # stackSentinel never called
    assert (cfg.stack_dir / "run_files").exists()           # plan untouched
    assert not list(cfg.stack_dir.glob("run_files.superseded_*"))
    assert json.loads((cfg.stack_dir / "dates_not_in_plan.json").read_text())["dates_not_in_plan"] == ["20200119"]
    # SAFE cleanup removing planned dates' archives must NOT trigger anything
    import shutil
    shutil.rmtree(cfg.slc_dir / safe_name("20200113"))
    cfg.allow_missing_dates = ("20200119",)                 # the way out the message names
    argv, files = sb._generate_run_files(cfg, Path("/dem"))
    assert len(calls) == 1 and files


def test_legacy_record_checks_against_safe_files_txt_and_backfills(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    for d in ("20200101", "20200113"):
        fake_safe(cfg.slc_dir, d)
    (cfg.stack_dir / "run_files").mkdir()
    (cfg.stack_dir / "run_files" / "run_01_unpack_topo_reference").write_text("x\n")
    argv = sb._stack_argv(cfg, Path("/dem"))
    (cfg.stack_dir / "generation.json").write_text(json.dumps({"argv": argv, "generated": "x"}))
    (cfg.stack_dir / "SAFE_files.txt").write_text(
        "".join(f"{cfg.slc_dir / safe_name(d)}\n" for d in ("20200101", "20200113")))
    monkeypatch.setattr(sb.subprocess, "run", lambda *a, **k: pytest.fail("must reuse"))
    sb._generate_run_files(cfg, Path("/dem"))
    assert json.loads((cfg.stack_dir / "generation.json").read_text())["input_dates"] == \
        ["20200101", "20200113"]


def test_failed_date_is_retried_then_named_and_recorded(tmp_path, monkeypatch, quiet):
    cfg = make_cfg(tmp_path)
    dates = ["20200101", "20200113", "20200125"]
    monkeypatch.setattr(da, "search_bursts_complete", fake_search(dates))
    rec = Recorder({"20200113": [1, 1, 0], "20200125": [1, 1, 1]})
    monkeypatch.setattr(procs, "run_group", rec)
    with pytest.raises(RuntimeError, match="20200125"):
        da.download_bursts_as_safes(cfg)
    assert [c[0] for c in rec.calls].count("20200113") == 3      # retried, then succeeded
    assert all(t == da.ASSEMBLE_TIMEOUT_S for _, t in rec.calls)  # never unbounded
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert list(st["failed"]) == ["20200125"]
    assert "20200113" in st["assembled"]
    assert not list(cfg.slc_dir.glob("*20200125*.SAFE"))
    # accepted as missing: the run continues and the acceptance is recorded
    cfg.allow_missing_dates = ["20200125"]
    rec2 = Recorder({"20200125": [1, 1, 1]})
    monkeypatch.setattr(procs, "run_group", rec2)
    out = da.download_bursts_as_safes(cfg)
    assert sorted(p.name[17:25] for p in out) == ["20200101", "20200113"]
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert st["accepted_failures"] == ["20200125"] and st["accepted_missing"] == ["20200125"]


def test_accepted_missing_dates_parsing(tmp_path):
    cfg = make_cfg(tmp_path)
    assert da.accepted_missing_dates(cfg) == (set(), False)
    cfg.allow_missing_dates = ("2016-12-06", "20170101")
    assert da.accepted_missing_dates(cfg) == ({"20161206", "20170101"}, False)
    cfg.allow_missing_dates = ("all",)
    assert da.accepted_missing_dates(cfg)[1] is True


# ================================================================ ops-3

def test_download_and_orbits_emit_progress(tmp_path, monkeypatch, quiet):
    cfg = make_cfg(tmp_path)
    dates = ["20200101", "20200113"]
    monkeypatch.setattr(da, "search_bursts_complete", fake_search(dates))
    monkeypatch.setattr(procs, "run_group", Recorder())
    out = da.download_bursts_as_safes(cfg)
    ev = [e for e in progress.read(cfg.project_dir) if e["stage"] == "download"]
    assert ev[0]["status"] == "start" and ev[0]["total"] == 2
    assert ev[-1]["status"] == "done" and ev[-1]["done"] == 2 and ev[-1]["total"] == 2
    fake_eof = types.ModuleType("eof.download")
    def _eof(**kw):          # a file that covers the scene, as the real one would
        d = Path(kw["sentinel_file"]).name[17:25]
        f = Path(kw["save_dir"]) / f"S1A_OPER_AUX_POEORB_OPOD_X_V{d}T000000_{d}T235959.EOF"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
        return [f]
    fake_eof.download_eofs = _eof
    monkeypatch.setitem(sys.modules, "eof", types.ModuleType("eof"))
    monkeypatch.setitem(sys.modules, "eof.download", fake_eof)
    da.download_orbits(cfg, out, "u", "p")
    ev = [e for e in progress.read(cfg.project_dir) if e["stage"] == "orbits"]
    assert ev[-1]["status"] == "done" and ev[-1]["done"] == 2


# ===================================================== stackrunner-13

def test_existing_partial_safe_is_quarantined_and_reassembled(tmp_path, monkeypatch, quiet):
    cfg = make_cfg(tmp_path)
    fake_safe(cfg.slc_dir, "20200113", truncate=True)       # an interrupted assembly
    fake_safe(cfg.slc_dir, "20200101")                      # a good one
    monkeypatch.setattr(da, "search_bursts_complete", fake_search(["20200101", "20200113"]))
    rec = Recorder()
    monkeypatch.setattr(procs, "run_group", rec)
    out = da.download_bursts_as_safes(cfg)
    assert [c[0] for c in rec.calls] == ["20200113"]        # only the bad one re-fetched
    assert (cfg.slc_dir / "quarantine" / safe_name("20200113")).exists()
    assert not da.verify_safe(cfg.slc_dir / safe_name("20200113"), {"IW2": 3}, "VV")
    assert len(out) == 2


def test_partial_assembly_never_lands_in_slc(tmp_path, monkeypatch, quiet):
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(da, "search_bursts_complete", fake_search(["20200101"]))
    monkeypatch.setattr(procs, "run_group", Recorder({"20200101": ["partial", "partial", "partial"]}))
    with pytest.raises(RuntimeError, match="20200101"):
        da.download_bursts_as_safes(cfg)
    assert not list(cfg.slc_dir.glob("*.SAFE"))             # stackSentinel would have globbed it
    assert not (cfg.slc_dir / ".partial").exists()


def test_verify_safe_checks_bursts_tiff_and_manifest(tmp_path):
    assert da.verify_safe(fake_safe(tmp_path / "a", "20200101"), {"IW2": 3}, "VV") == ""
    assert "bursts" in da.verify_safe(fake_safe(tmp_path / "b", "20200101", bursts=2), {"IW2": 3}, "VV")
    assert "bytes" in da.verify_safe(fake_safe(tmp_path / "c", "20200101", truncate=True), {"IW2": 3}, "VV")
    assert "manifest" in da.verify_safe(fake_safe(tmp_path / "d", "20200101", manifest=False), {"IW2": 3}, "VV")
    assert "IW1" in da.verify_safe(fake_safe(tmp_path / "e", "20200101"), {"IW1": 3}, "VV")


# ===================================================== stackrunner-12

def test_earthdata_login_is_tried_not_just_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    ok, msg = da.check_earthdata_login()
    assert not ok and "netrc" in msg
    monkeypatch.setenv("EARTHDATA_USERNAME", "kyle")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "typo")

    class ASFAuthenticationError(Exception):
        pass

    def session(exc):
        class S:
            def auth_with_creds(self, u, p):
                if exc:
                    raise exc
                return self
        return S
    monkeypatch.setattr(da.asf, "ASFSession", session(ASFAuthenticationError(
        "Failed to log in with provided credentials. Original Exception: 401")))
    ok, msg = da.check_earthdata_login()
    assert not ok and "rejected" in msg
    monkeypatch.setattr(da.asf, "ASFSession", session(ASFAuthenticationError(
        "Failed to set asf-urs cookies.")))
    ok, msg = da.check_earthdata_login()
    assert not ok and "authorised ASF data access" in msg
    monkeypatch.setattr(da.asf, "ASFSession", session(ConnectionError("no route")))
    assert da._earthdata_login_status()[0] == "unreachable"
    monkeypatch.setattr(da.asf, "ASFSession", session(None))
    assert da.check_earthdata_login()[0] is True


def test_refused_login_mid_download_stops_with_plain_message(tmp_path, monkeypatch, quiet):
    cfg = make_cfg(tmp_path, download_workers=1)
    monkeypatch.setattr(da, "search_bursts_complete", fake_search(["20200101", "20200113", "20200125"]))
    rec = Recorder({"20200101": [1]}, text_fail="HTTPError: 401 Client Error: Unauthorized")
    monkeypatch.setattr(procs, "run_group", rec)
    with pytest.raises(RuntimeError, match="login"):
        da.download_bursts_as_safes(cfg)
    assert len(rec.calls) == 1                               # no retries, no other dates

    monkeypatch.setattr(da, "_earthdata_login_status", lambda: ("rejected", "NASA Earthdata rejected the login"))
    rec2 = Recorder()
    monkeypatch.setattr(procs, "run_group", rec2)
    with pytest.raises(RuntimeError, match="rejected"):
        da.download_bursts_as_safes(cfg)
    assert rec2.calls == []                                  # checked before any download


# ========================================== stackrunner-4 / ops-2

def test_download_skips_dates_cleaned_after_the_merge(tmp_path, monkeypatch, quiet):
    """Site B's layout: coreg_secondarys holds only <date>.cleaned files."""
    cfg = make_cfg(tmp_path)
    dates = ["20200101", "20200113", "20200125", "20200206"]
    for d in dates[1:3]:
        write_cleaned_pl(cfg, d)
    # a marker whose merged SLC was removed by hand: not evidence of anything
    (cfg.stack_dir / "coreg_secondarys" / "20200206.cleaned").write_text("x\n")
    fake_safe(cfg.slc_dir, "20200101")                      # the reference keeps its SAFE
    monkeypatch.setattr(da, "search_bursts_complete", fake_search(dates))
    rec = Recorder()
    monkeypatch.setattr(procs, "run_group", rec)
    da.download_bursts_as_safes(cfg)
    assert [c[0] for c in rec.calls] == ["20200206"]
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert st["skipped_coregistered"] == ["20200113", "20200125"]
    # parity: the SAFE cleanup's notion of done is the same predicate
    assert sb.coregistered_dates(cfg, include_reference=False) == {"20200113", "20200125"}


def test_download_refuses_dates_a_cleaned_stack_cannot_take(tmp_path, monkeypatch, quiet):
    cfg = make_cfg(tmp_path)
    write_cleaned_pl(cfg, "20200113")
    (cfg.stack_dir / "run_files").mkdir()
    (cfg.stack_dir / "generation.json").write_text(json.dumps(
        {"argv": ["x"], "input_dates": ["20200101", "20200113"]}))
    monkeypatch.setattr(da, "search_bursts_complete", fake_search(["20200101", "20200113", "20200125"]))
    rec = Recorder()
    monkeypatch.setattr(procs, "run_group", rec)
    with pytest.raises(RuntimeError, match="20200125"):
        da.download_bursts_as_safes(cfg)
    assert rec.calls == []                                   # refused BEFORE downloading


def test_replan_of_a_cleaned_stack_is_refused_before_anything_moves(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    fake_safe(cfg.slc_dir, "20200101")
    (cfg.stack_dir / "run_files").mkdir()
    (cfg.stack_dir / "run_files" / "run_01_unpack_topo_reference").write_text("x\n")
    old = sb._stack_argv(cfg, Path("/dem"))
    (cfg.stack_dir / "generation.json").write_text(json.dumps(
        {"argv": old, "input_dates": ["20200101", "20200113"]}))
    write_cleaned_pl(cfg, "20200113")
    cfg.end_date = "2021-06-30"                              # a "monitoring update"
    monkeypatch.setattr(sb.subprocess, "run", lambda *a, **k: pytest.fail("must refuse first"))
    with pytest.raises(RuntimeError, match="--stop_date 2020-12-31 -> 2021-06-30"):
        sb._generate_run_files(cfg, Path("/dem"))
    assert (cfg.stack_dir / "run_files").exists()
    assert not list(cfg.stack_dir.glob("*.superseded_*"))


def test_original_safe_files_failure_restores_the_plan(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, backend="stack")
    fake_safe(cfg.slc_dir, "20200101")
    (cfg.stack_dir / "run_files").mkdir()
    (cfg.stack_dir / "run_files" / "run_01_unpack_topo_reference").write_text("x\n")
    (cfg.stack_dir / "generation.json").write_text(json.dumps({"argv": ["old"], "input_dates": ["20200101"]}))

    def run(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, "Exception: The original SAFE files for "
                                           "latest 6 coregistered SLCs is needed", "")
    monkeypatch.setattr(sb.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="original archives"):
        sb._generate_run_files(cfg, Path("/dem"))
    assert (cfg.stack_dir / "run_files" / "run_01_unpack_topo_reference").exists()


def test_burst_geometry_cleanup_leaves_marker_run01_accepts_it(tmp_path):
    cfg = make_cfg(tmp_path, stack_cleanup_burst_geometry=True)
    s = cfg.stack_dir
    (s / "geom_reference" / "IW2").mkdir(parents=True)
    (s / "geom_reference" / "IW2" / "hgt_01.rdr").write_bytes(b"\0" * 64)
    (s / "merged" / "geom_reference").mkdir(parents=True)
    (s / "reference" / "IW2").mkdir(parents=True)
    (s / "reference" / "IW2.xml").write_text("x")
    conf = s / "config_reference"
    conf.write_text(f"[Function-1]\nSentinel1_TOPS : \noutdir : {s / 'reference'}\n"
                    f"[Function-2]\ntopo : \ngeom_referenceDir : {s / 'geom_reference'}\n")
    cmd = f"SentinelWrapper.py -c {conf}"
    assert sb.verify_outputs(cfg, "run_01_unpack_topo_reference", cmd) == ""
    import shutil
    shutil.rmtree(s / "geom_reference")                      # removed without a marker
    assert "geom_reference" in sb.verify_outputs(cfg, "run_01_unpack_topo_reference", cmd)
    (s / "geom_reference" / "IW2").mkdir(parents=True)
    (s / "geom_reference" / "IW2" / "hgt_01.rdr").write_bytes(b"\0" * 64)
    (s / "merged" / "geom_reference" / "hgt.rdr").write_bytes(b"\0" * 8)
    sb.cleanup_burst_geometry(cfg)
    assert (s / "geom_reference.cleaned").exists() and not (s / "geom_reference").exists()
    assert sb.verify_outputs(cfg, "run_01_unpack_topo_reference", cmd) == ""


# ================================== phaselinked-9 / stackrunner-9

def test_safe_kept_until_the_resample_is_complete(tmp_path):
    cfg = make_cfg(tmp_path, stack_delete_safes_after_coreg=True, stack_keep_recent_safes=0)
    for d in ("20200113", "20200125", "20200206"):
        fake_safe(cfg.slc_dir, d)
    write_resampled(cfg, "20200113")                         # complete
    write_resampled(cfg, "20200125", written=1, xml=False)   # in flight: burst 1 of 3
    write_resampled(cfg, "20200206", written=2)              # failed part-way, xml from a stale run
    sb.cleanup_safes(cfg)
    assert not (cfg.slc_dir / safe_name("20200113")).exists()
    assert (cfg.slc_dir / safe_name("20200125")).exists()
    assert (cfg.slc_dir / safe_name("20200206")).exists()
    # and the resample outcome check agrees: one burst is not "done"
    conf = cfg.stack_dir / "config_fullBurst_resample_20200125"
    conf.write_text(f"[Function-1]\nresamp_withCarrier : \ncoregdir : "
                    f"{cfg.stack_dir / 'coreg_secondarys' / '20200125'}\noverlap : False\n")
    assert sb.verify_outputs(cfg, "run_10_fullBurst_resample", f"SentinelWrapper.py -c {conf}") != ""
    write_resampled(cfg, "20200125")
    assert sb.verify_outputs(cfg, "run_10_fullBurst_resample", f"SentinelWrapper.py -c {conf}") == ""


# ===================================================== stackrunner-10

def test_configless_coregistration_commands_have_output_checks(tmp_path):
    cfg = make_cfg(tmp_path)
    s = cfg.stack_dir
    (s / "reference").mkdir()
    (s / "reference" / "IW2.xml").write_text("x")
    (s / "configs").mkdir()
    for p in ("20200101_20200113", "20200101_20200125"):
        (s / "configs" / f"config_misreg_{p}").write_text("x")
    run04 = f"subsetReference.py -m {s / 'reference'} -g {s / 'geom_reference'}"
    run08 = f"invertMisreg.py -i {s}/misreg/azimuth/pairs/ -o {s}/misreg/azimuth/dates/"
    run11 = f"extractCommonValidRegion.py -m {s / 'reference'} -s {s / 'coreg_secondarys'}"
    for name, cmd in (("run_04_extract_burst_overlaps", run04), ("run_08_timeseries_misreg", run08),
                      ("run_11_extract_stack_valid_region", run11)):
        assert sb.expected_outputs(cfg, name, cmd), name          # used to be []
        assert sb.verify_outputs(cfg, name, cmd) != "", name
    for n in ("IW2_top.xml", "IW2_bottom.xml"):
        (s / "reference" / "overlap").mkdir(exist_ok=True)
        (s / "reference" / "overlap" / n).write_text("x")
    (s / "stack").mkdir()
    (s / "stack" / "IW2.xml").write_text("x")
    dd = s / "misreg" / "azimuth" / "dates"
    dd.mkdir(parents=True)
    for d in ("20200101", "20200113"):                          # 20200125 has no correction
        (dd / f"{d}.txt").write_text("0.001\n")
    assert sb.verify_outputs(cfg, "run_04_extract_burst_overlaps", run04) == ""
    assert sb.verify_outputs(cfg, "run_11_extract_stack_valid_region", run11) == ""
    assert "20200125" in sb.verify_outputs(cfg, "run_08_timeseries_misreg", run08)
    (dd / "20200125.txt").write_text("0.002\n")
    assert sb.verify_outputs(cfg, "run_08_timeseries_misreg", run08) == ""


def _misreg_results(cfg, pairs, failed=(), nan=()):
    s = cfg.stack_dir
    (s / "configs").mkdir(exist_ok=True)
    res = []
    for p in pairs:
        out = s / "misreg" / "azimuth" / "pairs" / p / f"{p}.txt"
        conf = s / "configs" / f"config_misreg_{p}"
        conf.write_text(f"[Function-3]\nestimateAzimuthMisreg : \nout_azimuth : {out}\n")
        if p not in failed:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"median : {'nan' if p in nan else '0.001'}\nmean : 0.0\n")
        res.append({"cmd": f"SentinelWrapper.py -c {conf}", "returncode": 1 if p in failed else 0,
                    "seconds": 1.0, "peak_rss_gb": None, "skipped": False})
    return res


def test_misreg_network_checked_not_claimed(tmp_path):
    cfg = make_cfg(tmp_path)
    pairs = ["20200101_20200113", "20200101_20200125", "20200113_20200125",
             "20200113_20200206", "20200125_20200206", "20200206_20200218", "20200125_20200218"]
    # 2 of 7 fail (29% > 25% used to raise the ESD-threshold message; here the
    # network is the question) - 20200218 keeps 20200125_20200218 -> connected
    ok = _misreg_results(cfg, pairs, failed=("20200206_20200218",))
    sb._finish_run_file(cfg, "run_07_pairs_misreg", [r["cmd"] for r in ok], ok, 0.0, False)
    assert json.loads((cfg.stack_dir / "misreg_network.json").read_text())["connected"] is True
    # every pair of 20200218 lost (one failed, one NaN median): orphaned -> loud
    bad = _misreg_results(cfg, pairs, failed=("20200206_20200218",), nan=("20200125_20200218",))
    with pytest.raises(RuntimeError, match="20200218"):
        sb._finish_run_file(cfg, "run_07_pairs_misreg", [r["cmd"] for r in bad], bad, 0.0, False)
    assert json.loads((cfg.stack_dir / "misreg_network.json").read_text())["orphaned_dates"] == ["20200218"]


def test_real_merge_checks_size_not_just_the_vrt(tmp_path):
    cfg = make_cfg(tmp_path, stack_virtual_merge="False")
    d = cfg.stack_dir / "merged" / "SLC" / "20200113"
    d.mkdir(parents=True)
    conf = cfg.stack_dir / "config_merge_20200113"
    conf.write_text(f"[Function-1]\nmergeBursts : \noutfile : {d / '20200113.slc'}\n"
                    "use_virtual_files : False\nmultilook : False\n")
    cmd = f"SentinelWrapper.py -c {conf}"
    full = d / "20200113.slc.full"
    (d / "20200113.slc.full.vrt").write_text("<VRT/>")
    full.with_name(full.name + ".xml").write_text(
        '<imageFile><property name="width"><value>8</value></property>'
        '<property name="length"><value>6</value></property>'
        '<property name="data_type"><value>CFLOAT</value></property></imageFile>')
    full.write_bytes(b"\0" * (8 * 6 * 8 - 8))                 # truncated by a full disk
    assert "truncated" in sb.verify_outputs(cfg, "run_12_merge_reference_secondary_slc", cmd)
    full.write_bytes(b"\0" * (8 * 6 * 8))
    assert sb.verify_outputs(cfg, "run_12_merge_reference_secondary_slc", cmd) == ""
    full.unlink()
    assert sb.verify_outputs(cfg, "run_12_merge_reference_secondary_slc", cmd) != ""
    (d / "DELETED_AFTER_MIAPLPY_LOAD").write_text("x")       # consumed into slcStack.h5
    assert sb.verify_outputs(cfg, "run_12_merge_reference_secondary_slc", cmd) == ""


# ===================================================== stackrunner-15

def _write_tif(path: Path, arr):
    from osgeo import gdal
    gdal.UseExceptions()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.GetDriverByName("GTiff").Create(str(path), arr.shape[1], arr.shape[0], 1,
                                              gdal.GDT_Float64)
    ds.GetRasterBand(1).WriteArray(arr)
    ds = None


def test_burst_dem_check_catches_aoi_outside_the_frame(tmp_path):
    cfg = make_cfg(tmp_path)                                 # AOI -87.8..-87.52, 38.2..38.47
    lat, lon = np.meshgrid(np.linspace(38.1, 38.6, 60), np.linspace(-87.9, -87.0, 200), indexing="ij")
    g = cfg.stack_dir / "geom_reference" / "IW2"
    _write_tif(g / "lat_01.rdr", lat)
    _write_tif(g / "lon_01.rdr", lon)
    _write_tif(g / "hgt_01.rdr", np.full(lat.shape, 120.0))
    assert sb._check_burst_dem(cfg, 2, 2) == pytest.approx(1.0)
    # the frame stops at -87.66: half the AOI lies in a sub-swath not processed;
    # the old frame-pixels-only test saw no missing height and passed
    _write_tif(g / "lon_01.rdr", np.linspace(-87.9, -87.66, 200)[None, :].repeat(60, 0))
    with pytest.raises(RuntimeError, match="inside the radar"):
        sb._check_burst_dem(cfg, 2, 2)
    _write_tif(g / "lon_01.rdr", lon)
    _write_tif(g / "hgt_01.rdr", np.full(lat.shape, -500.0))
    with pytest.raises(RuntimeError, match="no DEM height"):
        sb._check_burst_dem(cfg, 2, 2)


def test_dem_checks_run_before_the_cleanups(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, stack_crop_unwrap=False)
    rf = cfg.stack_dir / "run_files"
    rf.mkdir()
    for n in ("run_01_unpack_topo_reference", "run_02_unpack_secondary_slc",
              "run_11_extract_stack_valid_region", "run_12_merge_reference_secondary_slc"):
        (rf / n).write_text("x\n")
    order = []
    monkeypatch.setattr(sb, "_install_teardown", lambda: None)
    monkeypatch.setattr(sb, "_record_timing", lambda c, s: None)
    monkeypatch.setattr(sb, "execute_run_file", lambda c, f, fail_fast, after=None:
                        order.append(f.name[:6]) or {"run_file": f.name})
    for fn in ("cleanup_safes", "cleanup_burst_geometry", "_check_merged_dem"):
        monkeypatch.setattr(sb, fn, lambda c, _fn=fn: order.append(_fn))

    def burst_check(c):
        order.append("_check_burst_dem")
        raise RuntimeError("AOI outside the frame")
    monkeypatch.setattr(sb, "_check_burst_dem", burst_check)
    with pytest.raises(RuntimeError):
        sb.execute(cfg)
    assert order == ["run_01", "_check_burst_dem"]           # no secondary touched
    order.clear()
    monkeypatch.setattr(sb, "_check_burst_dem", lambda c: order.append("_check_burst_dem"))
    sb.execute(cfg)
    m = order.index("run_12")
    assert order[m + 1:m + 4] == ["_check_merged_dem", "cleanup_safes", "cleanup_burst_geometry"]
    assert order.index("cleanup_burst_geometry") > order.index("_check_merged_dem")


# ===================================================== stackrunner-2

def test_burst_search_stops_on_a_deterministic_gap(tmp_path, monkeypatch):
    """Same short set on every attempt: the union cannot help, so it stops at 2."""
    class R:
        def __init__(self, fid, date):
            self.properties = {"fileID": fid, "flightDirection": "ASCENDING",
                               "burst": {"subswath": "IW2"},
                               "startTime": f"{date[:4]}-{date[4:6]}-{date[6:]}T23:47:00Z"}
            self.geometry = {"type": "Polygon", "coordinates": [[[-87.9, 38.1], [-87.0, 38.1]]]}
    res = [R(g, "20260617") for g in granules("20260617")] + [R(granules("20260605")[2], "20260605")]
    calls = {"n": 0}

    class FakeASF:
        class DATASET:
            SLC_BURST = "SLC-BURST"

        @staticmethod
        def search(**kw):
            calls["n"] += 1
            return res
    monkeypatch.setitem(sys.modules, "asf_search", FakeASF)
    monkeypatch.setattr(da.time, "sleep", lambda s: None)
    by_date, _, _ = da.search_bursts_complete(make_cfg(tmp_path), {"IW2"}, "VV", "ASC")
    assert calls["n"] == 2 and len(by_date["20260605"]) == 1


def _cmr_entry(burst, date, pol="VV", slc="S1A_IW_SLC__1SDV_20260605T234802_20260605T234830_064841_082BA0_1B94",
               index=2):
    base = f"https://sentinel1-burst.asf.alaska.edu/{slc}/IW2/{pol}/{index}"
    return {"title": f"S1_{burst}_IW2_{date}T234808_{pol}_1B94-BURST",
            "links": [{"href": base + ".tiff"}, {"href": base + ".xml"},
                      {"href": base + ".zip"}, {"href": "https://search.asf.alaska.edu/"}]}


def test_catalogue_burst_entries_parses_plain_json(tmp_path, monkeypatch):
    feed = {"feed": {"entry": [_cmr_entry(b, "20260605", pol, index=i)
                               for i, b in enumerate(("038886", "038887", "038888"), 1)
                               for pol in ("VV", "VH")]
                     + [_cmr_entry("038886", "20260604")]}}
    seen = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return feed

    def get(url, params=None, timeout=None):
        seen.update(params)
        return Resp()
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=get))
    ents = da.catalogue_burst_entries(make_cfg(tmp_path), "20260605", "VV", {"IW2"})
    assert sorted(ents) == ["038886_IW2", "038887_IW2", "038888_IW2"]
    assert ents["038887_IW2"]["index"] == 2 and ents["038887_IW2"]["tiff"].endswith("/IW2/VV/2.tiff")
    assert seen["short_name"] == "SENTINEL-1_BURSTS" and seen["temporal"].startswith("2026-06-05T00")


def test_render_gap_dates_recovered_short_dates_recorded(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, start_date="2016-01-01", end_date="2026-06-30",
                   stack_reference_date="20260617")
    dates = ["20160516", "20260605", "20260617", "20260629"]   # a 2-2 tie: modal is the larger count

    def entries(cfg_, date, pol, swaths):
        n = 3 if date == "20260605" else 1                      # 20160516 really is short
        return {f"0388{86 + i}_IW2": {"title": granules(date)[i], "swath": "IW2", "pol": "VV",
                                      "slc": "S1A_X", "index": i + 1, "tiff": "t", "xml": "x"}
                for i in range(n)}
    monkeypatch.setattr(da.time, "sleep", lambda s: None)
    monkeypatch.setattr(da, "_earthdata_login_status", lambda: ("ok", "fine"))
    monkeypatch.setattr(procs, "install_teardown", lambda: None)
    monkeypatch.setattr(da, "search_bursts_complete",
                        fake_search(dates, short={"20160516": 1, "20260605": 1}))
    monkeypatch.setattr(da, "catalogue_burst_entries", entries)
    recovered = []

    def recover(cfg_, date, ents, out_dir, session=None):
        recovered.append((date, sorted(ents)))
        fake_safe(out_dir, date)
        return 0, ""
    monkeypatch.setattr(da, "recover_from_extractor", recover)
    rec = Recorder()
    monkeypatch.setattr(procs, "run_group", rec)
    out = da.download_bursts_as_safes(cfg)
    assert recovered == [("20260605", ["038886_IW2", "038887_IW2", "038888_IW2"])]
    assert sorted(c[0] for c in rec.calls) == ["20260617", "20260629"]
    assert sorted(p.name[17:25] for p in out) == ["20260605", "20260617", "20260629"]
    st = json.loads((cfg.slc_dir / "assembly_status.json").read_text())
    assert st["catalogue_dates"] == dates
    assert st["dropped"] == {"20160516": "catalogue_short"}
    assert st["recovered"] == ["20260605"]
    assert st["processed"] == ["20260605", "20260617", "20260629"]
    assert not list(cfg.slc_dir.glob("*.zip"))
    assert json.loads((cfg.slc_dir / "dropped_short_dates.json").read_text())["dropped"] == ["20160516"]


def test_recover_from_extractor_builds_the_local2safe_tree(tmp_path, monkeypatch):
    """The burst extractor answers 202 while it cuts a burst; then 200."""
    monkeypatch.setattr(da.time, "sleep", lambda s: None)
    answers = {}

    class Resp:
        def __init__(self, code):
            self.status_code = code

        def iter_content(self, n):
            yield b"x" * 10

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Session:
        def get(self, url, stream=True, timeout=None):
            answers[url] = answers.get(url, 0) + 1
            return Resp(202 if answers[url] == 1 else 200)
    ran = {}

    def run_group(cmd, timeout=None, **kw):
        ran["cmd"], ran["timeout"] = cmd, timeout
        fake_safe(Path(cmd[-1]), "20260605")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(procs, "run_group", run_group)
    ents = {f"0388{86 + i}_IW2": {"title": granules("20260605")[i], "swath": "IW2", "pol": "VV",
                                  "slc": "S1A_SLC", "index": i + 2,
                                  "tiff": f"https://e/{i}.tiff", "xml": f"https://e/{i}.xml"}
            for i in range(3)}
    cfg = make_cfg(tmp_path)
    rc, _ = da.recover_from_extractor(cfg, "20260605", ents, tmp_path / "w", Session())
    assert rc == 0 and all(v == 2 for v in answers.values()) and len(answers) == 6
    tree = json.loads((tmp_path / "w" / "slc_tree.json").read_text())
    assert sorted(tree["S1A_SLC"]["IW2"]["VV"]) == ["2", "3", "4"]
    assert set(tree["S1A_SLC"]["IW2"]["VV"]["2"]) == {"DATA", "METADATA"}
    assert "local2safe" in " ".join(ran["cmd"]) and ran["timeout"] == da.ASSEMBLE_TIMEOUT_S
    assert da.verify_safe(tmp_path / "w" / safe_name("20260605"), {"IW2": 3}, "VV") == ""


def test_no_fresh_plan_over_a_cleaned_stack(tmp_path, monkeypatch):
    """run_files/ deleted by hand on a merged phase-linked stack: stackSentinel
    would plan a new stack from the 7 SAFEs left in slc/ (Site B)."""
    cfg = make_cfg(tmp_path)
    fake_safe(cfg.slc_dir, "20200101")
    fake_safe(cfg.slc_dir, "20200125")
    write_cleaned_pl(cfg, "20200113")
    monkeypatch.setattr(sb.subprocess, "run", lambda *a, **k: pytest.fail("must not plan"))
    with pytest.raises(RuntimeError, match="only the archives still in slc"):
        sb._generate_run_files(cfg, Path("/dem"))


# ======================================================================
# Lane L6a2: ERA5 licence and prefetch, planner, deramp mask, progress
#
#   gap-licensing-2 / ops-8 / stackrunner-11 / usability-15
#                                   ERA5 licences read from the CDS catalogue, fail vs warn
#   stackrunner-5 / phaselinked-7 / generality-12
#                                   planner: real disk budget, burst count, crop-scaled
#                                   phase-linked terms, RAM estimate
#   ops-13                          ERA5 prefetch under MintPy's own file names
#   gap-surface_mine_site-5         Euclidean deramp buffer; refuse a one-sided fit
#   ops-15 (stack side)             skipped count in the stack progress events
# ======================================================================
import io                                                   # noqa: E402
import urllib.error                                         # noqa: E402
import urllib.request                                       # noqa: E402
from types import SimpleNamespace                           # noqa: E402

CDS = "https://cds.example/api"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def cds_server(monkeypatch, tmp_path, accepted=None, required=None, profile_exc=None, catalogue_exc=None):
    """A ~/.cdsapirc under a temporary HOME and urlopen serving the two CDS
    endpoints check_weather_credentials reads."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".cdsapirc").write_text(f"url: {CDS}\nkey: abcdef-0123\n")
    monkeypatch.setenv("HOME", str(home))
    calls = []

    def urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        calls.append(url)
        if url.endswith("/profiles/v1/account/licences"):
            if profile_exc:
                raise profile_exc
            return _Resp(json.dumps({"licences": accepted or []}).encode())
        if "/catalogue/v1/collections/" in url:
            if catalogue_exc:
                raise catalogue_exc
            return _Resp(json.dumps({"id": "reanalysis-era5-pressure-levels",
                                     "links": (required or []) + [{"rel": "self", "href": "x"}]}).encode())
        raise AssertionError(url)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


PYAPS = SimpleNamespace(mintpy_tropo_method="pyaps")
CCBY = {"rel": "license", "id": "cc-by", "rev": 1, "title": "CC-BY licence",
        "href": "https://spdx.org/licenses/CC-BY-4.0"}


def test_era5_licence_read_from_the_catalogue_kyles_account_passes(tmp_path, monkeypatch):
    """Kyle's account on 5 Oct: 'cc-by' accepted, the old id not; the
    catalogue requires cc-by rev 1. The old check returned 'NOT accepted ...
    403' and preflight disabled Start."""
    calls = cds_server(monkeypatch, tmp_path, required=[CCBY],
                       accepted=[{"id": "cc-by", "revision": 1}, {"id": "terms-of-use-cds", "revision": 11}])
    assert sb.check_weather_credentials(PYAPS) == []
    assert any("/catalogue/v1/collections/reanalysis-era5-pressure-levels" in c for c in calls)


def test_era5_licence_missing_is_a_failure_naming_it(tmp_path, monkeypatch):
    cds_server(monkeypatch, tmp_path, required=[CCBY], accepted=[{"id": "terms-of-use-cds", "revision": 11}])
    probs = sb.check_weather_credentials(PYAPS)
    assert len(probs) == 1 and probs[0]["level"] == "fail"
    assert "cc-by" in probs[0]["text"] and "licence-to-use-copernicus-products" not in probs[0]["text"]


def test_era5_licence_old_revision_is_a_failure(tmp_path, monkeypatch):
    cds_server(monkeypatch, tmp_path, required=[dict(CCBY, rev=2)], accepted=[{"id": "cc-by", "revision": 1}])
    probs = sb.check_weather_credentials(PYAPS)
    assert [p["level"] for p in probs] == ["fail"] and "revision 2" in probs[0]["text"]


def test_era5_licence_unknown_requirement_warns_never_blocks(tmp_path, monkeypatch):
    down = urllib.error.URLError("catalogue down")
    cds_server(monkeypatch, tmp_path, catalogue_exc=down, accepted=[{"id": "cc-by", "revision": 1}])
    assert sb.check_weather_credentials(PYAPS) == []
    cds_server(monkeypatch, tmp_path, catalogue_exc=down, accepted=[{"id": "terms-of-use-cds"}])
    probs = sb.check_weather_credentials(PYAPS)
    assert [p["level"] for p in probs] == ["warn"]


def test_era5_token_rejected_fails_unreachable_warns(tmp_path, monkeypatch):
    cds_server(monkeypatch, tmp_path, required=[CCBY],
               profile_exc=urllib.error.HTTPError(CDS, 401, "Unauthorized", {}, None))
    probs = sb.check_weather_credentials(PYAPS)
    assert [p["level"] for p in probs] == ["fail"] and "rejected" in probs[0]["text"]
    cds_server(monkeypatch, tmp_path, required=[CCBY], profile_exc=urllib.error.URLError("no route"))
    assert [p["level"] for p in sb.check_weather_credentials(PYAPS)] == ["warn"]


def test_weather_problem_still_reads_as_the_old_string():
    """dashboard.py slices probs[0][:160]; cli prints f'! {p}'; doctor reads
    level/text from a dict - all three keep working."""
    p = sb.WeatherProblem("fail", "The CDS token works but ...")
    assert p[:7] == "The CDS" and f"! {p}" == "! The CDS token works but ..." and str(p).startswith("The")
    assert isinstance(p, dict) and p.get("level") == "fail" and p["text"].endswith("...")


# ------------------------------------------------------------------ planner

def _fake_dates(n, start="2016-01-05"):
    import datetime as dt
    d0 = dt.date.fromisoformat(start)
    return [types.SimpleNamespace(properties={"startTime": (d0 + dt.timedelta(days=12 * i)).isoformat()
                                              + "T23:47:41"}) for i in range(n)]


@pytest.fixture
def planner(monkeypatch):
    """plan_stack without the network: n dates, a fixed budget and RAM."""
    from insar_pipeline import settings
    st = {"n": 270, "budget": 422.0, "mem": 23.47}
    monkeypatch.setattr(da, "search_scenes", lambda cfg: _fake_dates(st["n"]))
    monkeypatch.setattr(sb, "check_weather_credentials", lambda cfg: [])
    monkeypatch.setattr(settings, "disk_budget_gb",
                        lambda p: {"budget": st["budget"], "source": "WSL: vhdx headroom + C: free (test)"},
                        raising=False)
    monkeypatch.setattr(settings, "machine_resources",
                        lambda: {"mem_gb": st["mem"], "swap_gb": 0, "cores": 24}, raising=False)
    monkeypatch.setattr(sb.shutil, "disk_usage",
                        lambda p: SimpleNamespace(total=1e12, used=2.6e11, free=7.43e11))
    return st


def site_b_cfg(tmp_path, **over):
    kw = dict(bbox=[-87.55995, 38.24325, -87.33112, 38.42952], start_date="2016-01-01",
              end_date="2026-06-30", stack_use_burst2safe=True, stack_swaths="2",
              stack_virtual_merge="False", stack_cleanup_coreg_after_filter=False,
              stack_delete_safes_after_coreg=True, stack_cleanup_offsets=True,
              stack_cleanup_misreg_inputs=True)
    kw.update(over)
    cfg = make_cfg(tmp_path, **kw)
    (cfg.slc_dir / "dropped_short_dates.json").write_text(json.dumps({"modal_burst_count": 3, "dropped": []}))
    return cfg


def test_plan_compares_with_the_wsl_budget_not_ext4_free(tmp_path, planner):
    """ext4 said 743 GB free; the real budget was ~422 GB (5 Oct 2026)."""
    planner["budget"] = 200.0
    plan = sb.plan_stack(site_b_cfg(tmp_path))
    assert plan.free_gb == 200.0 and "WSL" in plan.budget_source
    assert any("EXCEEDS THE DISK BUDGET" in n for n in plan.notes)
    assert "disk budget" in sb.format_plan(plan) and "free on disk" not in sb.format_plan(plan)


def test_plan_resume_needs_only_what_the_project_lacks(tmp_path, planner, monkeypatch):
    planner["budget"] = 100.0
    monkeypatch.setattr(sb, "_allocated_gb", lambda p: 205.0)
    plan = sb.plan_stack(site_b_cfg(tmp_path))
    assert plan.project_gb == 205.0 and not any("EXCEEDS" in n for n in plan.notes)


def test_plan_bursts_per_date_from_the_record_scales_the_estimate(tmp_path, planner):
    cfg = site_b_cfg(tmp_path)
    p3 = sb.plan_stack(cfg)
    (cfg.slc_dir / "assembly_status.json").write_text(json.dumps({"modal_burst_count": 6}))
    p6 = sb.plan_stack(cfg)
    assert (p3.bursts_per_date, p6.bursts_per_date) == (3, 6) and "assembly_status" in p6.bursts_source
    # whole-frame products double; the phase-linking crop is the same ground,
    # so its share of a 6-burst frame halves and its terms stay put
    assert p6.est_slc_gb == pytest.approx(2 * p3.est_slc_gb, rel=0.01)
    assert p6.est_merged_gb == pytest.approx(2 * p3.est_merged_gb, rel=0.01)
    assert p6.est_coreg_gb > 1.9 * p3.est_coreg_gb - 3
    assert p6.crop_fraction == pytest.approx(p3.crop_fraction / 2, rel=0.01)
    assert 1.7 * p3.est_peak_gb < p6.est_peak_gb < 2 * p3.est_peak_gb


def test_plan_bursts_per_date_from_one_catalogue_query(tmp_path, monkeypatch):
    """No local record: one burst query over the first ~5 weeks, not the
    full retried search; 4 bursts per date here, not the assumed 3."""
    cfg = make_cfg(tmp_path, orbit_direction="ascending")
    seen = {}

    def search(**kw):
        seen.update(kw)
        out = []
        for d in ("2020-01-01", "2020-01-13", "2020-01-25"):
            for i in range(4 if d != "2020-01-13" else 3):
                out.append(types.SimpleNamespace(properties={
                    "startTime": f"{d}T23:47:41", "flightDirection": "ASCENDING",
                    "burst": {"subswath": "IW2"}}))
        out.append(types.SimpleNamespace(properties={"startTime": "2020-01-01T23:47:41",
                                                     "flightDirection": "ASCENDING",
                                                     "burst": {"subswath": "IW3"}}))
        return out
    fake = types.SimpleNamespace(search=search, DATASET=types.SimpleNamespace(SLC_BURST="BURST"))
    monkeypatch.setitem(sys.modules, "asf_search", fake)
    n, src = sb._bursts_per_date(cfg, 1)
    assert n == 4 and "catalogue" in src and seen["end"] == "2020-02-06"


def test_phase_linked_terms_scale_with_the_crop(tmp_path, planner):
    """Site B measured: slcStack 49, PATCHES 52.7, phase_series 50 GB at
    crop 0.2331 over 270 dates; the old constants planned 77 / 94 / 85."""
    cfg = site_b_cfg(tmp_path)
    (cfg.stack_dir / "crop_window.json").write_text(json.dumps({"crop_fraction": 0.2331}))
    plan = sb.plan_stack(cfg)
    assert plan.crop_fraction == 0.2331 and "measured" in plan.crop_source
    assert plan.est_merged_gb == pytest.approx(226.8, abs=1)
    # the peak is loading the SLC stack beside the merged SLCs: 226.8 + 49.1 + 3
    assert plan.est_peak_gb == pytest.approx(278.9, abs=1.5)
    note = next(n for n in plan.notes if n.startswith("Phase-linked route"))
    assert "SLC stack (~49 GB)" in note and "patches ~53 GB" in note
    (cfg.stack_dir / "crop_window.json").write_text(json.dumps({"crop_fraction": 0.6}))
    planner["n"] = 275
    big = sb.plan_stack(cfg)
    # f = 0.6 at 275 dates: phase linking + concatenation ~ 3 + 275 * 0.6 * (0.78 + 0.84 + 0.79)
    assert big.est_peak_gb >= 3 + 275 * 0.6 * (0.78 + 0.84 + 0.79) - 1


def test_crop_estimated_from_the_bbox_before_the_geometry(tmp_path):
    cfg = site_b_cfg(tmp_path)
    f, src = sb._crop_fraction(cfg, 3)
    assert "estimated" in src and f == pytest.approx(0.2331, abs=0.03)     # Site B measured 0.2331


def test_plan_estimates_memory_and_warns_for_a_large_area(tmp_path, planner):
    # INT1: the deliverables stream the cube (deliverables._Cube), so the
    # peak follows the cells, not cells x dates: ~5.6 GB for Site B (the
    # whole-cube estimate was 10.6), and a 0.6 x 0.5 deg box (~13 M cells)
    # needs more than half of the 23.5 GB machine
    small = sb.plan_stack(site_b_cfg(tmp_path / "a"))
    assert 4 < small.est_ram_gb < 8 and not any(n.startswith("MEMORY") for n in small.notes)
    assert "products RAM" in sb.format_plan(small)
    big = sb.plan_stack(site_b_cfg(tmp_path / "b", bbox=[-87.9, 38.0, -87.3, 38.5]))
    assert big.est_ram_gb > 0.5 * 23.47 and big.est_ram_gb > 2 * small.est_ram_gb
    assert any(n.startswith("MEMORY") for n in big.notes)
    assert sb.products_ram_gb(3_490_000, 540) - sb.products_ram_gb(3_490_000, 270) < 0.3
    assert any(n.startswith("MEMORY") and "WILL NOT FIT" in n for n in
               [sb.plan_stack(site_b_cfg(tmp_path / "c", bbox=[-88.3, 37.8, -87.1, 38.7])).notes][0])


# ------------------------------------------------------------------ ERA5 prefetch

def test_era5_names_and_request_are_mintpys_own(tmp_path, monkeypatch):
    """The prefetch is useless unless MintPy finds the files: tile, hour,
    name and request compared with MintPy's and pyaps3's own code."""
    tp = pytest.importorskip("mintpy.tropo_pyaps3")
    autoget = pytest.importorskip("pyaps3.autoget")
    meta = {"LENGTH": "1882", "WIDTH": "1854", "X_FIRST": "-87.60547845605468",
            "Y_FIRST": "38.46293142211914", "X_STEP": "0.000172", "Y_STEP": "-0.000135"}
    lat0, lat1, lon0, lon1 = tp.get_bounding_box(meta)
    assert sb.era5_snwe(lat0, lat1, lon0, lon1) == tp.get_snwe(meta) == (30, 50, -90, -80)
    for ext in ((-1.5, 2.0, 179.0, 179.9), (-41.2, -40.1, -71.0, -70.0), (9.9, 10.2, 0.5, 2.0)):
        m = {"LENGTH": "2", "WIDTH": "2", "Y_FIRST": str(ext[1]), "X_FIRST": str(ext[2]),
             "Y_STEP": str(ext[0] - ext[1]), "X_STEP": str(ext[3] - ext[2])}
        la0, la1, lo0, lo1 = tp.get_bounding_box(m)
        assert sb.era5_snwe(la0, la1, lo0, lo1) == tp.get_snwe(m)
    assert sb.era5_hour(85662.983946) == (tp.closest_weather_model_hour(85662.983946), False)
    assert sb.era5_hour(45000)[1] is True                     # 12:30 exactly: 12 or 13, not settled
    assert sb.era5_hour(84600) == ("23", False)               # 23:30: ERA5 hour 24 does not exist
    # what pyaps3 itself would request and name, captured from its own loop
    got = {}

    class Client:
        def __init__(self, *a, **k):
            pass

        def retrieve(self, ds, req, target):
            got.update(ds=ds, req=req, target=target)
    monkeypatch.setattr(autoget.cdsapi, "Client", Client)
    monkeypatch.setattr(autoget.os.path, "isfile", lambda p: True)
    autoget.ECMWFdload(["20160105"], "23", str(tmp_path) + "/", model="ERA5", snwe=(30, 50, -90, -80))
    assert Path(got["target"]).name == sb.era5_grib_name((30, 50, -90, -80), "20160105", "23") \
        == "ERA5_N30_N50_W90_W80_20160105_23.grb"
    assert got["ds"] == sb.ERA5_DATASET
    assert got["req"] == sb.era5_request("20160105", (30, 50, -90, -80), "23")


def prefetch_cfg(tmp_path, dates=("20200101", "20200113", "20200125", "20200206"), **over):
    kw = dict(bbox=[-87.55995, 38.24325, -87.33112, 38.42952], mintpy_tropo_method="pyaps")
    kw.update(over)
    cfg = make_cfg(tmp_path, **kw)
    for d in dates:
        fake_safe(cfg.slc_dir, d)
    return cfg


def test_prefetch_fetches_the_missing_files_under_mintpys_names(tmp_path, monkeypatch):
    cfg = prefetch_cfg(tmp_path)
    monkeypatch.setattr(sb, "check_weather_credentials", lambda c: [])
    era5 = cfg.project_dir / "mintpy" / "ERA5"
    era5.mkdir(parents=True)
    (era5 / "ERA5_N30_N50_W90_W80_20200101_23.grb").write_bytes(b"G" * 100)
    fetched = []

    def fetch(date, path, snwe, hour):
        fetched.append(date)
        Path(path).write_bytes(b"G" * 100)
        return "ok"
    st = sb.start_era5_prefetch(cfg, fetch=fetch)
    assert st["status"] == "done" and st["snwe"] == [30, 50, -90, -80] and st["hour"] == "23"
    assert fetched == ["20200206", "20200125", "20200113"]          # newest first, the cached one skipped
    assert sorted(p.name for p in era5.glob("*.grb")) == [
        f"ERA5_N30_N50_W90_W80_{d}_23.grb" for d in ("20200101", "20200113", "20200125", "20200206")]
    rec = json.loads((cfg.mintpy_dir / sb.ERA5_PREFETCH_STATUS).read_text())
    assert rec["fetched"] == 3 and rec["present_before"] == 1 and rec["failed"] == {}
    ev = [e for e in progress.read(cfg.project_dir) if e["stage"] == "era5_prefetch"]
    assert ev[-1]["status"] == "done" and ev[-1]["done"] == ev[-1]["total"] == 4


def test_prefetch_stops_at_once_when_access_is_refused(tmp_path, monkeypatch):
    cfg = prefetch_cfg(tmp_path)
    monkeypatch.setattr(sb, "check_weather_credentials", lambda c: [])
    calls = []

    def refused(date, path, snwe, hour):
        calls.append(date)
        return "FAILED 403 Client Error: required licences not accepted"
    st = sb.start_era5_prefetch(cfg, workers=1, fetch=refused)
    assert st["status"] == "failed" and "refused" in st["reason"] and len(calls) == 1
    assert [e for e in progress.read(cfg.project_dir) if e["stage"] == "era5_prefetch"][-1]["status"] == "failed"
    # a licence the check already knows is missing: nothing is requested at all
    monkeypatch.setattr(sb, "check_weather_credentials",
                        lambda c: [sb.WeatherProblem("fail", "licence 'cc-by' not accepted")])
    calls.clear()
    st = sb.start_era5_prefetch(cfg, fetch=refused)
    assert st["status"] == "failed" and "cc-by" in st["reason"] and calls == []


def test_prefetch_skips_when_the_tile_is_not_settled(tmp_path, monkeypatch):
    """A box 0.05 deg from the -88 line (-90 minus 2 deg of buffer floors to
    -100 or not depending on the exact geometry): fetch nothing rather than
    hours of files under a name MintPy might not use."""
    cfg = prefetch_cfg(tmp_path, bbox=[-87.97, 38.24, -87.70, 38.43])
    monkeypatch.setattr(sb, "check_weather_credentials", lambda c: [])
    st = sb.start_era5_prefetch(cfg, fetch=lambda *a: pytest.fail("must not fetch"))
    assert st["status"] == "skipped" and "tile" in st["reason"]
    assert sb.start_era5_prefetch(make_cfg(tmp_path / "x"))["status"] == "skipped"   # tropo 'no'


def test_fetch_era5_grib_never_leaves_a_partial_file(tmp_path):
    class Boom:
        def retrieve(self, ds, req, target):
            Path(target).write_bytes(b"half")
            raise RuntimeError("connection reset")

    class Fine:
        def retrieve(self, ds, req, target):
            Path(target).write_bytes(b"G" * 10)
    dst = tmp_path / "ERA5_N30_N50_W90_W80_20200101_23.grb"
    assert sb.fetch_era5_grib("20200101", dst, (30, 50, -90, -80), "23", client=Boom()).startswith("FAILED")
    assert list(tmp_path.iterdir()) == []
    assert sb.fetch_era5_grib("20200101", dst, (30, 50, -90, -80), "23", client=Fine()) == "ok"
    assert dst.read_bytes() == b"G" * 10


# ------------------------------------------------------------------ deramp mask

def _square(w, s, e, n):
    return [[w, s], [e, s], [e, n], [w, n], [w, s]]


def _geo_mask_case(tmp_path, mine, bbox=(-87.80, 38.20, -87.52, 38.47), n=270, **over):
    import h5py
    cfg = make_cfg(tmp_path, bbox=list(bbox), deramp_exclude_geojson=str(tmp_path / "mine.geojson"), **over)
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    w, s, e, nn = bbox
    attrs = {"X_FIRST": str(w), "Y_FIRST": str(nn), "X_STEP": str((e - w) / n), "Y_STEP": str(-(nn - s) / n),
             "LENGTH": str(n), "WIDTH": str(n), "FILE_TYPE": "mask"}
    with h5py.File(cfg.mintpy_dir / "maskTempCoh.h5", "w") as f:
        f.create_dataset("mask", data=np.ones((n, n), bool))
        f.attrs.update(attrs)
    (tmp_path / "mine.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [mine]}}]}))
    yy, xx = np.mgrid[0:n, 0:n]
    lon = w + (xx + 0.5) * float(attrs["X_STEP"])
    lat = nn + (yy + 0.5) * float(attrs["Y_STEP"])
    return cfg, lat, lon


def _min_ground_distance(kept, poly, lat, lon):
    from scipy.spatial import cKDTree
    c = np.cos(np.radians(lat.mean()))
    xy = lambda m: np.c_[lon[m] * 111_320 * c, lat[m] * 110_900]    # noqa: E731
    return cKDTree(xy(poly)).query(xy(kept))[0].min()


def test_deramp_buffer_is_a_true_distance(tmp_path):
    """binary_dilation's cross element left fit pixels 1.43 km from the mines
    (Site B) under a '2 km' buffer."""
    import h5py
    cfg, lat, lon = _geo_mask_case(tmp_path, _square(-87.70, 38.30, -87.62, 38.37))
    path = sb.build_deramp_mask(cfg, buffer_m=2000.0)
    with h5py.File(path) as f:
        kept = f["mask"][:].astype(bool)
        assert f.attrs["DERAMP_BUFFER_KIND"] == "euclidean" and f.attrs["DERAMP_QUADRANTS"] == 4
    poly = (lon > -87.70) & (lon < -87.62) & (lat > 38.30) & (lat < 38.37)
    d = _min_ground_distance(kept, poly, lat, lon)
    px = 0.28 / 270 * 111_320 * np.cos(np.radians(38.3))
    assert 2000 - px < d < 2000 + 2 * px


def test_deramp_buffer_defaults_to_the_config(tmp_path):
    import h5py
    cfg, lat, lon = _geo_mask_case(tmp_path, _square(-87.70, 38.30, -87.62, 38.37))
    cfg.deramp_buffer_m = 500.0
    with h5py.File(sb.build_deramp_mask(cfg)) as f:
        assert f.attrs["DERAMP_BUFFER_M"] == 500.0


def test_deramp_radar_buffer_measured_on_the_ground(tmp_path):
    """Radar grid ~13 deg off north with 40 x 42 m pixels: the buffer is
    measured from the pixels' own lat/lon."""
    import h5py
    cfg = make_cfg(tmp_path, bbox=[-87.80, 38.20, -87.52, 38.47],
                   deramp_exclude_geojson=str(tmp_path / "mine.geojson"))
    cfg.mintpy_dir.mkdir(parents=True, exist_ok=True)
    (cfg.mintpy_dir / "inputs").mkdir()
    ny, nx = 160, 150
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    th = np.radians(13.0)
    north_m = (ny / 2 - yy) * 41.8 * np.cos(th) + (xx - nx / 2) * 40.5 * np.sin(th)
    east_m = (xx - nx / 2) * 40.5 * np.cos(th) - (ny / 2 - yy) * 41.8 * np.sin(th)
    lat = 38.335 + north_m / 110_900
    lon = -87.66 + east_m / (111_320 * np.cos(np.radians(38.335)))
    with h5py.File(cfg.mintpy_dir / "inputs" / "geometryRadar.h5", "w") as g:
        g.create_dataset("latitude", data=lat)
        g.create_dataset("longitude", data=lon)
    with h5py.File(cfg.mintpy_dir / "maskTempCoh.h5", "w") as f:
        f.create_dataset("mask", data=np.ones((ny, nx), bool))
    (tmp_path / "mine.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Polygon", "coordinates": [_square(-87.68, 38.32, -87.64, 38.35)]}}]}))
    with h5py.File(sb.build_deramp_mask(cfg, buffer_m=1000.0)) as f:
        kept = f["mask"][:].astype(bool)
    poly = (lon > -87.68) & (lon < -87.64) & (lat > 38.32) & (lat < 38.35)
    assert 1000 <= _min_ground_distance(kept, poly, lat, lon) < 1060


def test_deramp_refuses_a_fit_that_does_not_surround_the_area(tmp_path):
    """Mines plus buffer cover the east of the box: the plane would be fitted
    on a western strip and subtracted across the mines. It used to be a log
    line; now nothing is written and the run stops with the remedies."""
    cfg, lat, lon = _geo_mask_case(tmp_path, _square(-87.70, 38.15, -87.40, 38.55))
    with pytest.raises(RuntimeError, match="does not surround") as e:
        sb.build_deramp_mask(cfg, buffer_m=2000.0)
    assert "deramp_buffer_m" in str(e.value) and "mintpy_deramp: no" in str(e.value)
    assert not (cfg.mintpy_dir / "maskDeramp.h5").exists()


# ------------------------------------------------------------------ progress (ops-15)

def test_stack_progress_counts_verified_commands_separately(tmp_path):
    cfg = make_cfg(tmp_path)
    tick = sb._progress_counter(cfg, "run_15_unwrap", 5)
    for _ in range(3):
        tick({"returncode": 0, "skipped": True})
    tick({"returncode": 0, "skipped": False})
    ev = progress.read(cfg.project_dir)[-1]
    assert ev["done"] == 4 and ev["skipped"] == 3 and "3 verified from the last run" in ev["msg"]
    tick({"returncode": 1})
    ev = progress.read(cfg.project_dir)[-1]
    assert ev["status"] == "failed" and ev["skipped"] == 3 and "1 failed" in ev["msg"]


def test_phase_linking_disk_rechecked_on_the_measured_crop(tmp_path, monkeypatch):
    """The bbox estimate said fine; the measured crop (0.6 of the frame) does
    not fit: refused before load_data, not 20 h in."""
    from insar_pipeline import settings
    cfg = site_b_cfg(tmp_path)
    monkeypatch.setattr(sb, "planned_dates", lambda c, rec=None: {f"d{i:03d}" for i in range(275)})
    monkeypatch.setattr(settings, "disk_budget_gb", lambda p: {"budget": 300.0, "source": "test"}, raising=False)
    (cfg.stack_dir / "crop_window.json").write_text(json.dumps({"crop_fraction": 0.2331}))
    assert sb.check_phase_linking_disk(cfg)["ok"]
    (cfg.stack_dir / "crop_window.json").write_text(json.dumps({"crop_fraction": 0.6}))
    with pytest.raises(RuntimeError, match="Phase linking still needs ~401 GB"):
        sb.check_phase_linking_disk(cfg)


def test_phase_linking_disk_credits_the_merged_slcs_deleted_after_loading(tmp_path, monkeypatch):
    """13 Oct 2026 (Site C): with miaplpy_delete_merged_slc_after_load the
    merged secondary SLCs go right after load_data, so the peak is the SLC
    stack beside them or everything written after they are gone - not the
    sum. Without the deletion the sum stands."""
    import dataclasses
    from insar_pipeline import settings
    cfg = site_b_cfg(tmp_path)
    monkeypatch.setattr(sb, "planned_dates", lambda c, rec=None: {f"d{i:03d}" for i in range(275)})
    monkeypatch.setattr(settings, "disk_budget_gb", lambda p: {"budget": 300.0, "source": "test"}, raising=False)
    (cfg.stack_dir / "crop_window.json").write_text(json.dumps({"crop_fraction": 0.6}))
    merged = cfg.stack_dir / "merged" / "SLC"
    merged.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sb, "_allocated_gb", lambda p: 230.0 if Path(p) == merged else 0.0)
    on = dataclasses.replace(cfg, miaplpy_delete_merged_slc_after_load=True)
    d = sb.check_phase_linking_disk(on)
    assert d["ok"] and d["merged_freed_gb"] > 200 and d["need_gb"] < 401 - 200
    off = dataclasses.replace(cfg, miaplpy_delete_merged_slc_after_load=False)
    with pytest.raises(RuntimeError, match="Phase linking still needs ~401 GB"):
        sb.check_phase_linking_disk(off)
