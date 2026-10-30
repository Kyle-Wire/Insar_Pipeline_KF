"""Readiness review of 16 Oct 2026, the runs batch: what a run and a report
build leave out, and which build made what.

ERR-01  client_report's optional blocks: a failure is recorded in
        client_report.json ('omitted'), named in the page's technical part
        ('Left out of this build') and read by the release gate (one soft
        reading item; hard for the single-point figures and the send rule,
        whose absence makes a printed figure or the gate's own input wrong).
        A failed geocode of the extra quality layers is a note in
        run_result.json instead of a discarded record.
ERR-02  a run's WARNING-and-above log records are counted, grouped by
        message template, into run_result.json, and the dashboard shows one
        line for them.
ERR-04  every earlier run_result.json is kept in run_result_history.jsonl;
        a run's log starts with the build label.
OUT-02  provenance.json keeps every build that wrote it ('builds'), so a
        products-only rerun no longer erases the processing run's build.
OUT-05  one registry line per run under the projects root (no paths, user or
        host names).

Each test here failed on the code at f3a0a0a. Synthetic projects in tmp_path
only: runs.REGISTRY, the run registry and the projects root are pointed at
tmp_path, so nothing under ~/insar_projects or any dotfile is touched.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L5 as L5                                          # noqa: E402  (the client-report project builder)
import test_fix_L7b as L7b                                        # noqa: E402  (the provenance config)

from insar_pipeline import client_report as cr                   # noqa: E402
from insar_pipeline import pipeline, progress, release, runs, version   # noqa: E402
from insar_pipeline.config import load_config                    # noqa: E402


def _boom(msg, kind=RuntimeError):
    def f(*a, **k):
        raise kind(msg)
    return f


# ------------------------------------------------------------------ ERR-01: the report's optional blocks
def test_a_failed_block_is_recorded_named_on_the_page_and_read_by_the_gate(tmp_path, monkeypatch):
    """The version line (made in compute) and the credits section (made while
    the page is drawn) fail: both are in client_report.json's 'omitted'
    list, named under 'How confident are these numbers?' - not in the
    client's summary at the top - and the release block lists them as one
    soft item. Before: a WARNING in the log, nothing anywhere else."""
    from insar_pipeline import credits
    L5._no_network(monkeypatch)
    cfg = L5.make_project(tmp_path)
    monkeypatch.setattr(version, "describe", _boom("version store unreadable"))
    monkeypatch.setattr(credits, "text_block", _boom("no credits file", ValueError))
    out = cr.build(cfg, out_dir=tmp_path / "preview")
    rep = json.loads((tmp_path / "preview" / "client_report.json").read_text())
    om = {o["block"]: o for o in rep["omitted"]}
    assert set(om) == {"version", "credits_section"}
    assert om["version"]["error"] == "RuntimeError: version store unreadable"
    assert om["credits_section"]["what"] == cr.OPTIONAL_BLOCKS["credits_section"]
    t = L5._text(out.read_text())
    i0 = t.index("How confident are these numbers?")
    sec = t[i0:t.index("Methodology", i0)]
    assert "Left out of this build" in sec
    assert "The software version line." in sec and "The data sources and credits section." in sec
    assert "Left out of this build" not in t[:i0]
    items = [b for b in rep["release"]["blockers"] if b["code"] in release.OMITTED_CODES]
    assert [(b["code"], b["hard"]) for b in items] == [("report_parts_omitted", False)]
    assert "version store unreadable" in items[0]["text"] and "no credits file" in items[0]["text"]
    # the gate reading the record gives the same item
    again = [b for b in release.blockers(cfg, rep) if b["code"] == "report_parts_omitted"]
    assert again and again[0]["text"] == items[0]["text"]
    # a sign-off of this record survives its rebuild: the block the page left
    # out (after compute's release block hashed the record) is not hashed
    assert release.numbers_hash(rep) == release.numbers_hash(cr.compute(cfg)[0])
    page_blocks = {"figure_credits", "footer_version", "footer_identity", "credits_section"}
    assert page_blocks <= release._OMITTED_NOT_HASHED and not page_blocks & set(release.OMITTED_HARD)


def test_a_failed_send_rule_or_point_figures_holds_the_report(tmp_path, monkeypatch):
    """The send rule re-evaluated with the GNSS tie, and the single-point
    figures, are hard when they fail: without the first the report keeps the
    QA's rule that never saw the tie (a failed tie would not hold it), and
    without the second 'Reading the numbers' quotes the split-half
    repeatability, smaller than the rasters' uncertainty (E18)."""
    from insar_pipeline import quality_assessment as qa
    cfg = L5.make_project(tmp_path)
    q = json.loads((cfg.project_dir / "quality_assessment.json").read_text())
    q["send_rule"] = {"value": "send", "reasons": []}
    (cfg.project_dir / "quality_assessment.json").write_text(json.dumps(q))
    monkeypatch.setattr(qa, "send_rule", _boom("tie record malformed", KeyError))
    monkeypatch.setattr(cr, "_point_figures", _boom("uncertainty model unreadable", ValueError))
    rep = cr.compute(cfg)[0]
    assert {o["block"] for o in rep["omitted"]} == {"send_rule", "point_figures"}
    assert rep["send_rule"] == {"value": "send", "reasons": []}              # the QA's own, kept
    hard = [b for b in rep["release"]["blockers"] if b["code"] == "report_part_failed"]
    assert len(hard) == 2 and all(b["hard"] and b["page_text"] == release.PAGE_PART_MISSING for b in hard)
    assert any("tie record malformed" in b["text"] and "GNSS tie" in b["text"] for b in hard)
    assert any("uncertainty model unreadable" in b["text"] for b in hard)
    assert not any(b["code"] == "report_parts_omitted" for b in rep["release"]["blockers"])
    assert rep["release"]["final"] is False


def test_omitted_items_and_the_numbers_hash():
    """The gate's items from the list; the numbers hash counts which blocks
    were left out, not their error text, and an empty list not at all (a
    report that left nothing out hashes as before the list existed)."""
    assert release.omitted_items({}) == [] and release.omitted_items({"omitted": []}) == []
    it = release.omitted_items({"omitted": [{"block": "gradient_limit", "what": "the fastest movement",
                                             "error": "KeyError: 'WAVELENGTH'"}]})
    assert [(i["code"], i["hard"]) for i in it] == [("report_parts_omitted", False)]
    assert "the fastest movement (KeyError: 'WAVELENGTH')" in it[0]["text"]
    base = {"areas": [{"name": "A", "vertical_mm_yr": 1.0}]}
    assert release.numbers_hash(base) == release.numbers_hash({**base, "omitted": []})
    h1 = release.numbers_hash({**base, "omitted": [{"block": "standouts", "error": "OSError: /tmp/a"}]})
    h2 = release.numbers_hash({**base, "omitted": [{"block": "standouts", "error": "OSError: /tmp/b"}]})
    assert h1 == h2 != release.numbers_hash(base)
    # the credits, like the credits themselves (_VOLATILE_KEYS), are not the numbers
    assert release.numbers_hash({**base, "omitted": [{"block": "credits", "error": "x"}]}) == \
        release.numbers_hash(base)


# ------------------------------------------------------------------ runs: a products run with stubbed stages
def _cfg(tmp_path, backend="miaplpy"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    y = tmp_path / "c.yaml"
    y.write_text(f"project_dir: {tmp_path / 'proj'}\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    cfg = load_config(str(y))
    cfg.backend = backend
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _stubs(monkeypatch, figures=None, layers=None):
    def done(name):
        def f(cfg):
            progress.emit(cfg.project_dir, name, "done")
            return True
        return f
    monkeypatch.setattr(pipeline, "export_all", lambda cfg: cfg.project_dir / "deliverables")
    monkeypatch.setattr(pipeline, "geocode_missing_layers", lambda cfg: layers or {})
    monkeypatch.setattr(pipeline, "run_quality_assessment", done("quality_assessment"))
    monkeypatch.setattr(pipeline, "run_figures", figures or done("figures"))
    monkeypatch.setattr(pipeline, "run_client_report", done("client_report"))


def test_a_failed_geocode_of_the_extra_layers_is_a_note_in_run_result(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _stubs(monkeypatch, layers={"present": [], "made": [], "missing_after": ["geo_demErr.h5"], "skipped": None,
                                "error": "OSError: [Errno 28] No space left on device"})
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    r = runs.result(cfg.project_dir)
    assert r["status"] == "done"
    assert any(n.startswith("Geocoding the extra quality layers failed (OSError: [Errno 28] No space left on device)")
               and "geo_demErr.h5" in n for n in r["notes"])
    assert "Geocoding the extra quality layers failed" in r["plain_cause"]
    # a route that does not make the layers has nothing to say
    assert pipeline.extra_layers_note(_cfg(tmp_path / "d", "disp"), {"missing_after": ["geo_demErr.h5"]}) is None


def test_warnings_are_counted_into_run_result_and_shown_in_the_dashboard(tmp_path, monkeypatch):
    from insar_pipeline import dashboard
    lg = logging.getLogger("insar_pipeline.oct16_runs_test")

    def figures(cfg):
        for i in range(5):
            lg.warning("Tile %d could not be drawn", i)
        lg.warning("Basemap offline")
        lg.error("A figure failed: %s", "boom")
        lg.info("not counted")
        return True
    cfg = _cfg(tmp_path)
    _stubs(monkeypatch, figures=figures)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    r = runs.result(cfg.project_dir)
    w = r["warnings"]
    top = {m: n for m, n in w["top"]}
    assert top["WARNING insar_pipeline.oct16_runs_test: Tile 0 could not be drawn"] == 5      # one template
    assert top["WARNING insar_pipeline.oct16_runs_test: Basemap offline"] == 1
    assert top["ERROR insar_pipeline.oct16_runs_test: A figure failed: boom"] == 1
    assert w["count"] >= 7 and w["errors"] >= 1 and w["distinct"] >= 3 and len(w["top"]) <= 10
    assert dashboard.warnings_line(r) == (f"{w['count']} warnings ({w['errors']} error{'' if w['errors'] == 1 else 's'})"
                                          f", {w['distinct']} distinct - see the log")
    assert dashboard.warnings_line({"status": "done"}) is None
    assert dashboard.warnings_line({"warnings": {"count": 0, "distinct": 0}}) is None
    # the counter goes with the run
    assert pipeline._RUN_WARNINGS is None
    assert not any(isinstance(h, pipeline.WarningCounter) for h in logging.getLogger().handlers)


# ------------------------------------------------------------------ ERR-04: history and the log label
def test_each_earlier_run_result_is_kept_in_the_history(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _stubs(monkeypatch, figures=lambda c: False)                  # a stage fails: status failed
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 1
    _stubs(monkeypatch)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    P = cfg.project_dir
    assert [h["status"] for h in runs.result_history(P)] == ["failed"]
    first = runs.result_history(P)[0]
    assert first["failed_stages"] == ["figures"] and first["plain_cause"].startswith("The run finished, but")
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    assert [h["status"] for h in runs.result_history(P)] == ["failed", "done"]
    assert runs.result(P)["status"] == "done" and len((P / runs.HISTORYFILE).read_text().splitlines()) == 2


def test_a_run_log_starts_with_the_build_label(tmp_path, monkeypatch):
    cfgp = tmp_path / "x.yaml"
    cfgp.write_text("")
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")            # not ~/.insar_pipeline_cli.json
    monkeypatch.setitem(runs.KINDS, "quick", ("test", ["-c", "print('child output')"]))
    rec = runs.start(str(cfgp), tmp_path, "quick")
    t0 = time.time()
    while runs.alive(rec) and time.time() - t0 < 20:
        time.sleep(0.05)
    lines = Path(rec["log"]).read_text().splitlines()
    assert lines[0].endswith(" " + version.label()) and lines[1] == "child output"
    assert runs.build_label() == version.label()

    # the report rebuild's log too (dashboard.rebuild_report; its child is stubbed)
    from insar_pipeline import dashboard

    class Done:
        returncode = 0

    def fake_run(argv, stdout=None, **k):
        stdout.write("rebuilt\n")
        return Done()
    monkeypatch.setattr(dashboard.subprocess, "run", fake_run)
    ok, log = dashboard.rebuild_report(str(cfgp), tmp_path / "p2")
    assert ok and Path(log).read_text().splitlines()[0].endswith(version.label())


# ------------------------------------------------------------------ OUT-02: the builds in provenance.json
def test_provenance_keeps_the_build_of_the_processing_run(tmp_path, monkeypatch):
    """A record written before 'builds' gives its own build as the first
    entry; a products-only rerun with other code is appended, not
    swapped in; the same run and code again adds nothing; a report rebuilt
    outside the run with other code adds a 'report' entry."""
    from insar_pipeline import provenance
    cfg = L7b._cfg(tmp_path)
    D = tmp_path / "deliverables"
    D.mkdir()
    (D / "provenance.json").write_text(json.dumps({
        "generated_utc": "2026-09-22T14:16:00+00:00",
        "software": {"insar_pipeline": "2026.9.20", "code_hash": "0123456789ab"},
        "run": {"run_kind": "full", "run_started": "2026-09-20T08:00:00"}}))
    (tmp_path / "run.json").write_text(json.dumps({"pid": os.getpid(), "kind": "products", "started": 1790000000.0}))
    monkeypatch.setattr(version, "code_hash", lambda root=None: "bbbbbbbbbbbb")
    provenance.write(cfg, D)
    b = json.loads((D / "provenance.json").read_text())["builds"]
    assert [(x["run_kind"], x["code_hash"], x["step"]) for x in b] == [
        ("full", "0123456789ab", "export"), ("products", "bbbbbbbbbbbb", "export")]
    assert b[0]["started"] == "2026-09-20T08:00:00" and b[1]["version"] == version.__version__
    provenance.write(cfg, D)                                     # the same run, the same code
    assert len(json.loads((D / "provenance.json").read_text())["builds"]) == 2
    (tmp_path / "run.json").write_text(json.dumps({"pid": os.getpid() + 1, "kind": "products",
                                                   "started": 1790000000.0}))
    monkeypatch.setattr(version, "code_hash", lambda root=None: "cccccccccccc")
    provenance.record_stages(cfg)
    b = json.loads((D / "provenance.json").read_text())["builds"]
    assert [(x["run_kind"], x["code_hash"], x["step"]) for x in b][-1] == (None, "cccccccccccc", "report")
    assert len(b) == 3
    # the package's fresh record carries the history (package.build uses collect)
    assert [x["code_hash"] for x in provenance.collect(cfg)["builds"]] == ["0123456789ab", "bbbbbbbbbbbb",
                                                                         "cccccccccccc"]
    long = [{"run_kind": "k", "started": str(i), "version": "v", "code_hash": "h"} for i in range(60)]
    kept = provenance.add_build(long, {"run_kind": "k", "started": "new", "version": "v", "code_hash": "h"})
    assert len(kept) == provenance.MAX_BUILDS and kept[0]["started"] == "0" and kept[-1]["started"] == "new"


# ------------------------------------------------------------------ OUT-05: the run registry
def test_every_run_has_one_line_in_the_registry_under_the_projects_root(tmp_path, monkeypatch):
    monkeypatch.delenv(runs.RUN_REGISTRY_ENV, raising=False)
    monkeypatch.setenv("INSAR_PROJECTS", str(tmp_path / "root"))
    cfg = _cfg(tmp_path)
    _stubs(monkeypatch, figures=lambda c: False)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 1
    _stubs(monkeypatch)
    assert pipeline.run_from(cfg, "products", watchdog=False, notify=False) == 0
    f = tmp_path / "root" / runs.RUN_REGISTRY
    assert runs.run_registry_path() == f
    lines = [json.loads(x) for x in f.read_text().splitlines()]
    assert [(x["project"], x["kind"], x["status"]) for x in lines] == [("proj", "products", "failed"),
                                                                       ("proj", "products", "done")]
    assert all(x["version"] == version.__version__ and x["code_hash"] == version.code_hash() for x in lines)
    assert all(set(x) == {"project", "kind", "started", "finished", "status", "stage", "exit_code", "version",
                          "code_hash"} for x in lines)
    assert all(x["started"] <= x["finished"] for x in lines)
    assert str(tmp_path) not in f.read_text()                   # a folder name, never a path
    assert len(runs.run_registry(code_hash=version.code_hash(), project="proj")) == 2
    assert runs.run_registry(code_hash="not-a-build") == []
    # the run's own end record names its build too
    r = runs.result(cfg.project_dir)
    assert (r["version"], r["code_hash"]) == (version.__version__, version.code_hash())


@pytest.fixture(autouse=True)
def _restore_counter():
    yield
    pipeline._RUN_WARNINGS = None
