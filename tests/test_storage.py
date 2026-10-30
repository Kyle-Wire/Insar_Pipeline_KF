"""Storage (13 Oct 2026): the overview of every project and the two ways to
free disk - keep results only (scripts/cleanup.py levels 1-3, its guards)
and delete a project after a slim archive. Synthetic projects in tmp_path;
INSAR_PROJECTS, the settings file and the remembered-configs list point
there, and runs.known_configs sees only the configs made here."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from insar_pipeline import procs, runs, storage

PARENT = str(Path(__file__).resolve().parents[2])
BIG = 1000                      # storage.BIG_FILE_BYTES in these tests


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    (root / "configs").mkdir(parents=True)
    monkeypatch.setenv("INSAR_NOTIFY", "off")
    monkeypatch.setenv("INSAR_PROJECTS", str(root))
    monkeypatch.setenv("INSAR_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(runs, "REGISTRY", tmp_path / "reg.json")
    monkeypatch.delenv(runs.LOCK_FD_ENV, raising=False)
    monkeypatch.setattr(storage, "BIG_FILE_BYTES", BIG)
    extra: list = []
    monkeypatch.setattr(runs, "known_configs",
                        lambda: sorted((root / "configs").glob("*.yaml")) + [Path(c) for c in extra])
    # the shared folders storage must never touch
    _touch(root / "gnss" / "unr_llh.out", b"g" * 100)
    _touch(root / "_scratch" / "mintpy" / "x.h5")
    _touch(root / "configs" / "notes.txt")
    return {"root": root, "tmp": tmp_path, "extra": extra}


def _touch(p: Path, data: bytes = b"x" * 64, t: float | None = None) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if t is not None:
        os.utime(p, (t, t))
    return p


def _yaml(f: Path, project_dir: Path) -> Path:
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"project_dir: {project_dir}\nbackend: miaplpy\ntrack: 165\nmintpy_deramp: 'no'\n"
                 "stack_virtual_merge: 'False'\nbbox: [-87.7, 38.2, -87.6, 38.3]\n"
                 "start_date: '2020-01-01'\nend_date: '2021-01-01'\n")
    return f


def _finished(home, name: str, release: bool = False, cfg_file: Path | None = None) -> tuple[Path, Path]:
    """A finished phase-linked project (products newer than the velocity),
    processing inputs of every level, a big raster and a big client report."""
    P = home["root"] / name if cfg_file is None or "elsewhere" not in str(cfg_file) else \
        home["tmp"] / "elsewhere" / name
    cf = _yaml(cfg_file or home["root"] / "configs" / f"{name}.yaml", P)
    M, D = P / "mintpy", P / "deliverables"
    t = time.time() - 10_000
    for i, n in enumerate(("timeseries", "timeseries_ramp", "timeseries_ramp_demErr")):
        _touch(M / f"{n}.h5", t=t + i)
    _touch(M / "geo" / "geo_timeseries_ramp_demErr.h5", t=t + 20)
    _touch(M / "velocity.h5", t=t + 20)
    _touch(M / "geo" / "geo_velocity.h5", t=t + 21)
    _touch(M / "pic" / "a.png", b"p" * 500)
    _touch(P / "quality_assessment.json", b"{}", t=t + 100)
    _touch(D / "client_report.json", b"{}", t=t + 200)
    _touch(D / "client_report.html", b"h" * (BIG + 500))                 # big, but the report: archived
    _touch(D / f"{name}_ground_movement_DRAFT.kmz", b"k" * (BIG + 1))    # the client KMZ: archived
    _touch(D / "vertical_movement_in_per_yr.tif", b"r" * 300)
    _touch(D / "points_timeseries.parquet", b"q" * (BIG * 3))           # big: left out of the archive
    _touch(D / "measurement_points_egms.csv", b"c" * 2000)              # level 3
    _touch(D / "config_snapshot.yaml", b"a: 1\n")
    _touch(P / "validation" / "summary.json", b"{}")
    _touch(P / "validation" / "gnss_cache" / "STAT.tenv3", b"s" * 50)
    _touch(P / "logs" / "full_20261001_0100.log", b"log\n")
    _touch(P / "slc" / "S1A_X.SAFE" / "manifest.safe", b"s" * 5000)
    _touch(P / "slc" / "dropped_short_dates.json", b"{}")
    _touch(P / "miaplpy" / "inputs" / "slcStack.h5", b"m" * 4000)
    _touch(P / "stack" / "merged" / "interferograms" / "20200101_20200113" / "filt_fine.unw", b"u" * 3000)
    _touch(P / "orbits" / "o.EOF")
    (P / runs.RESULTFILE).write_text(json.dumps({"kind": "full", "status": "done", "route": "official",
                                                "backend": "miaplpy", "started": t, "finished": t + 300,
                                                "config": str(cf)}))
    os.utime(P / runs.RESULTFILE, (t + 300, t + 300))
    if release:
        _touch(D / "releases" / "20261005_1200" / "client" / "01_report.html")
        _touch(D / "releases" / "20261005_1200" / "data" / "big_points.csv", b"b" * (BIG * 2))
    return P, cf


def _running(P: Path) -> None:
    (P / runs.RUNFILE).write_text(json.dumps({"pid": os.getpid(), "start_ticks": procs.start_ticks(os.getpid()),
                                              "kind": "miaplpy", "started": time.time()}))
    assert runs.state(P) == "running"


def _tree(d: Path) -> dict:
    return {str(f.relative_to(d)): f.stat().st_size for f in d.rglob("*") if f.is_file()}


# ---------------------------------------------------------------- inventory
def test_inventory_finds_projects_sizes_and_what_keep_results_frees(home):
    from insar_pipeline.scripts import cleanup
    from insar_pipeline.config import load_config
    A, _ = _finished(home, "alpha", release=True)
    B, _ = _finished(home, "beta")
    G = home["root"] / "gamma"                                    # a project folder without a known config
    _touch(G / "mintpy" / "velocity.h5", b"v" * 10)
    _touch(home["root"] / "not_a_project" / "notes.txt")          # no run.json, mintpy/, deliverables/, slc/
    E, ecf = _finished(home, "delta", cfg_file=home["tmp"] / "elsewhere_cfgs" / "delta.yaml")
    home["extra"].append(ecf)

    inv = storage.inventory()
    by = {p.name: p for p in inv["projects"]}
    assert set(by) == {"alpha", "beta", "gamma", "delta"}         # not configs, gnss, _scratch, not_a_project
    assert by["delta"].path == str(E)
    a = by["alpha"]
    assert a.size == storage.du(A) == sum(f.stat().st_size for f in A.rglob("*") if f.is_file())
    assert sum(a.folders.values()) == a.size and a.folders["slc"] == 5000 + 2
    assert a.state == "done" and not a.running and a.last_kind == "full" and a.last_status == "done"
    assert a.route == "official" and a.backend == "miaplpy" and a.purpose == "client"
    assert a.release == "DRAFT" and a.delivered == ["releases/20261005_1200"]
    assert by["beta"].delivered == []
    want = sum(storage.du(p) for _, p, _ in cleanup.targets(load_config(str(a.config)), 3))
    assert a.reclaimable == want and want >= 5000 + 4000 + 3000 + 2000
    g = by["gamma"]
    assert g.config == "" and g.reclaimable is None and "no config" in g.reclaimable_why
    assert "budget" in inv["budget"]
    text = storage.overview_text(inv)
    assert "alpha" in text and "yes: releases/20261005_1200" in text and "n/a (no config)" in text


def test_inventory_says_why_keep_results_is_not_available(home):
    A, _ = _finished(home, "alpha")
    (A / "deliverables" / "client_report.json").unlink()
    B, _ = _finished(home, "beta")
    _running(B)
    by = {p.name: p for p in storage.inventory()["projects"]}
    assert by["alpha"].reclaimable is None and "client_report.json is missing" in by["alpha"].reclaimable_why
    assert by["beta"].running and by["beta"].reclaimable is None and by["beta"].reclaimable_why == "a run is going"
    assert "run going" in storage.reclaim_text(by["beta"])


# ---------------------------------------------------------------- keep results only
def test_keep_results_is_a_dry_run_then_delegates_to_cleanup(home, monkeypatch):
    from insar_pipeline.scripts import cleanup
    A, _ = _finished(home, "alpha", release=True)
    before = _tree(A)
    plan = storage.keep_results_only("alpha")
    assert not plan["refused"] and not plan["applied"] and _tree(A) == before
    paths = {it["path"] for it in plan["items"]}
    assert str(A / "slc") in paths and str(A / "miaplpy" / "inputs") in paths
    assert plan["bytes"] == sum(it["bytes"] for it in plan["items"])

    calls = []
    real = cleanup.delete
    monkeypatch.setattr(cleanup, "delete", lambda cfg, items: calls.append(items) or real(cfg, items))
    plan = storage.keep_results_only("alpha", apply=True)
    assert plan["applied"] and not plan["failed"] and plan["freed"] == plan["bytes"]
    assert len(calls) == 1 and {str(p) for _, p, _ in calls[0]} == paths       # cleanup's own targets
    assert not (A / "slc").exists() and not (A / "miaplpy" / "inputs").exists()
    assert not (A / "stack" / "merged").exists()
    for kept in ("logs/full_20261001_0100.log", "deliverables/client_report.json",
                 "deliverables/releases/20261005_1200/data/big_points.csv", "mintpy/geo/geo_velocity.h5",
                 "deliverables/points_timeseries.parquet", "quality_assessment.json", "run_result.json"):
        assert (A / kept).exists(), kept
    assert (A / "logs" / "records" / "slc" / "dropped_short_dates.json").exists()   # cleanup.save_records


def test_keep_results_refuses_while_running_and_on_an_unfinished_project(home):
    A, _ = _finished(home, "alpha")
    _running(A)
    before = _tree(A)
    plan = storage.keep_results_only("alpha", apply=True)
    assert plan["refused"] and "run is going" in plan["refused"][0] and not plan["applied"]
    assert _tree(A) == before
    B, _ = _finished(home, "beta")
    (B / "quality_assessment.json").unlink()
    before = _tree(B)
    plan = storage.keep_results_only("beta", apply=True)
    assert any("not finished" in r for r in plan["refused"]) and _tree(B) == before


def test_keep_results_refuses_when_another_process_holds_the_lock(home):
    A, _ = _finished(home, "alpha")
    p = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import time
        from insar_pipeline import runs
        with runs.project_lock({str(A)!r}, kind="full"):
            print("held", flush=True); time.sleep(30)
    """)], stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": PARENT}, start_new_session=True)
    try:
        assert p.stdout.readline().strip() == "held"
        before = _tree(A)
        assert storage.keep_results_only("alpha", apply=True)["refused"]
        d = storage.delete_project("alpha", apply=True)
        assert d["refused"] and A.is_dir()
        assert _tree(A) == before
    finally:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


# ---------------------------------------------------------------- delete
def test_delete_dry_run_plans_the_archive_and_changes_nothing(home):
    A, _ = _finished(home, "alpha", release=True)
    before = _tree(A)
    plan = storage.delete_project("alpha")
    assert not plan["applied"] and _tree(A) == before and not (home["root"] / "_archive").exists()
    assert plan["bytes"] == storage.du(A)
    assert plan["needs_name"] and "releases/20261005_1200" in plan["why_name"]
    skipped = dict(plan["skipped"])
    assert "deliverables/points_timeseries.parquet" in skipped
    assert "deliverables/client_report.html" not in skipped                       # the report goes whatever its size
    assert "deliverables/releases/20261005_1200/data/big_points.csv" not in skipped
    assert Path(plan["archive_dir"]).name == f"alpha_{time.strftime('%Y%m%d')}"
    assert "points_timeseries.parquet" in storage.delete_plan_text(plan)


def test_delete_archives_the_right_files_and_removes_only_that_project(home):
    A, acf = _finished(home, "alpha")
    B, bcf = _finished(home, "beta")
    runs._remember(acf)
    runs._remember(bcf)
    beta_before, gnss_before = _tree(B), _tree(home["root"] / "gnss")
    configs_before = _tree(home["root"] / "configs")
    seen = []
    plan = storage.delete_project("alpha", apply=True, progress=lambda d, t, s: seen.append(s))
    assert not plan["refused"], plan["refused"]
    assert plan["applied"] and plan["archived"] and plan["deleted"] and plan["forgot"]
    assert not A.exists() and seen
    X = Path(plan["archive_dir"])
    got = _tree(X)
    for f in ("deliverables/client_report.json", "deliverables/client_report.html",
              "deliverables/alpha_ground_movement_DRAFT.kmz", "deliverables/vertical_movement_in_per_yr.tif",
              "deliverables/config_snapshot.yaml", "validation/summary.json", "logs/full_20261001_0100.log",
              "run_result.json", "quality_assessment.json", "config/alpha.yaml", "ARCHIVE.json", "README.txt"):
        assert f in got, f
    assert "deliverables/points_timeseries.parquet" not in got
    assert not any(k.startswith(("slc", "miaplpy", "stack", "mintpy", "orbits", "validation/gnss_cache"))
                   for k in got)
    man = json.loads((X / "ARCHIVE.json").read_text())
    assert {s["path"] for s in man["left_out_over_50MB"]} == {"deliverables/points_timeseries.parquet",
                                                             "deliverables/measurement_points_egms.csv"}
    # nothing else moved
    assert _tree(B) == beta_before and _tree(home["root"] / "gnss") == gnss_before
    assert _tree(home["root"] / "configs") == configs_before and acf.exists()      # the yaml is kept
    assert (home["root"] / "_scratch" / "mintpy" / "x.h5").exists()
    reg = json.loads(runs.REGISTRY.read_text())
    assert str(acf) not in reg["configs"] and str(bcf) in reg["configs"]
    assert {p.name for p in storage.inventory()["projects"]} == {"beta"}


def test_delete_refuses_while_running(home):
    A, _ = _finished(home, "alpha")
    _running(A)
    before = _tree(A)
    plan = storage.delete_project("alpha", apply=True)
    assert plan["refused"] and "run is going" in plan["refused"][0]
    assert _tree(A) == before and not (home["root"] / "_archive").exists()


@pytest.mark.parametrize("why", ["release", "final_folder", "final_report"])
def test_a_delivered_or_final_project_needs_its_name(home, why):
    A, _ = _finished(home, "alpha", release=(why == "release"))
    if why == "final_folder":
        _touch(A / "deliverables" / "final_20260928" / "README.txt")
    if why == "final_report":
        (A / "deliverables" / "client_report.json").write_text(json.dumps({"release": {"final": True}}))
    assert storage.delete_project("alpha")["needs_name"]
    for name in (None, "", "Alpha", "beta"):
        plan = storage.delete_project("alpha", apply=True, confirm_name=name)
        assert any("Type the project's name" in r for r in plan["refused"]) and A.is_dir()
    assert not (home["root"] / "_archive").exists()
    plan = storage.delete_project("alpha", apply=True, confirm_name="alpha")
    assert plan["deleted"] and not A.exists()
    if why == "release":
        assert (Path(plan["archive_dir"]) / "deliverables" / "releases" / "20261005_1200" / "data"
                / "big_points.csv").exists()                                   # what went to the client, whole


def test_delete_without_archive_and_a_config_inside_the_folder(home):
    P = home["root"] / "inner"
    cf = _yaml(P / "inner.yaml", P)
    _touch(P / "mintpy" / "velocity.h5")
    home["extra"].append(cf)
    runs._remember(cf)
    plan = storage.delete_project("inner")
    assert any("inside the folder" in n for n in plan["notes"])
    plan = storage.delete_project("inner", archive=False, apply=True)
    assert plan["deleted"] and not P.exists() and not (home["root"] / "_archive").exists()
    assert plan["forgot"] and str(cf) not in json.loads(runs.REGISTRY.read_text())["configs"]


def test_delete_never_takes_a_folder_holding_other_projects(home, monkeypatch):
    A, _ = _finished(home, "alpha")
    cf = _yaml(home["tmp"] / "bad.yaml", home["root"])                # a config whose project is the projects folder
    home["extra"].append(cf)
    assert all(p.path != str(home["root"]) for p in storage.inventory()["projects"])
    with pytest.raises(KeyError):
        storage.delete_project(str(home["root"]), apply=True)
    assert A.is_dir() and (home["root"] / "gnss").is_dir()


# ---------------------------------------------------------------- the command
def test_cli_overview_and_typed_confirmation(home, monkeypatch, capsys):
    from insar_pipeline import cli
    A, _ = _finished(home, "alpha", release=True)
    assert cli.storage_main([]) == 0
    out = capsys.readouterr().out
    assert "alpha" in out and "Keep results only frees" in out and "real budget" in out
    assert cli.storage_main(["delete", "alpha"]) == 0 and A.is_dir()          # dry run
    answers = iter(["alpha2", "yes"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    assert cli.storage_main(["delete", "alpha", "--apply"]) == 1 and A.is_dir()
    assert "does not match" in capsys.readouterr().out
    answers = iter(["alpha", "no"])
    assert cli.storage_main(["delete", "alpha", "--apply"]) == 1 and A.is_dir()
    answers = iter(["alpha", "yes"])
    assert cli.storage_main(["delete", "alpha", "--apply"]) == 0 and not A.exists()
    assert cli.storage_main(["delete", "nosuch"]) == 1


def test_cli_keep_results(home, monkeypatch, capsys):
    from insar_pipeline import cli
    A, _ = _finished(home, "alpha")
    assert cli.storage_main(["keep-results", "alpha"]) == 0 and (A / "slc").exists()
    monkeypatch.setattr("builtins.input", lambda _p="": "yes")
    assert cli.storage_main(["keep-results", "alpha", "--apply"]) == 0
    assert not (A / "slc").exists() and (A / "deliverables" / "client_report.json").exists()


def test_dispatcher_and_help_know_storage():
    from insar_pipeline import dashboard
    assert "insar storage" in dashboard.USAGE and "keep-results" in dashboard.USAGE
    assert ("g", "storage", "Storage") in dashboard.ProjectsScreen.BINDINGS


# ---------------------------------------------------------------- the dashboard
def _pilot(app, body, size=(180, 55)):
    async def go():
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await body(app, pilot)
    asyncio.run(go())


async def _settle(app, pilot, n=4):
    for _ in range(n):
        await app.workers.wait_for_complete()
        await pilot.pause()


def _select(app, name):
    from textual.widgets import DataTable
    t = app.screen.query_one("#storage", DataTable)
    keys = [r.value for r in t.rows]
    t.move_cursor(row=keys.index(name))


def test_dashboard_storage_screen_deletes_in_the_background(home):
    from insar_pipeline import dashboard as dash
    A, _ = _finished(home, "alpha")
    B, _ = _finished(home, "beta", release=True)

    async def body(app, pilot):
        await _settle(app, pilot)
        await pilot.press("g")
        await _settle(app, pilot)
        assert isinstance(app.screen, dash.StorageScreen)
        assert set(app.screen.info) == {"alpha", "beta"}
        assert "free disk" in str(app.screen.query_one("#stitle").render())
        _select(app, "alpha")
        await pilot.press("d")
        await _settle(app, pilot)
        assert isinstance(app.screen, dash.StorageConfirm)
        assert "Delete project: alpha" in str(app.screen.query_one("#storage-plan").render())
        assert not app.screen.query("#storage-name")                            # not delivered: no name
        await pilot.click("#yes")
        await _settle(app, pilot, 8)
        assert isinstance(app.screen, dash.StorageScreen)
        assert not A.exists() and set(app.screen.info) == {"beta"}
        # delivered: the name must be typed
        _select(app, "beta")
        await pilot.press("d")
        await _settle(app, pilot)
        assert app.screen.query("#storage-name")
        await pilot.click("#yes")
        await pilot.pause()
        assert isinstance(app.screen, dash.StorageConfirm) and B.is_dir()          # refused in the modal
        await pilot.click("#storage-name")
        await pilot.press("b", "e", "t", "a", "enter")
        await _settle(app, pilot, 8)
        assert not B.exists(), (type(app.screen), [str(n.message) for n in app._notifications])
    _pilot(dash.InsarDashboard(), body)


def test_dashboard_keep_results_shows_why_not_on_an_unfinished_project(home):
    from insar_pipeline import dashboard as dash
    A, _ = _finished(home, "alpha")
    (A / "quality_assessment.json").unlink()
    before = _tree(A)

    async def body(app, pilot):
        await _settle(app, pilot)
        await pilot.press("g")
        await _settle(app, pilot)
        _select(app, "alpha")
        await pilot.press("k")
        await _settle(app, pilot)
        assert isinstance(app.screen, dash.StorageConfirm)
        assert not app.screen.query("#yes")                                        # only Close
        assert "NOT AVAILABLE" in str(app.screen.query_one("#storage-plan").render())
        await pilot.click("#no")
        await pilot.pause()
    _pilot(dash.InsarDashboard(), body)
    assert _tree(A) == before
