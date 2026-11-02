"""A menu-driven front end for the whole process (the old numbered menu; the
dashboard - `insar` - is the main one).

    python -m insar_pipeline.cli [config.yaml]        (or insar --menu)
    python -m insar_pipeline.cli --version
    python -m insar_pipeline.cli injection-test <config.yaml> [--out-dir D] [--sites N] [--sizes 0.3,0.6] ...
    python -m insar_pipeline.cli cross-check <config.yaml> [--out-dir D] [--force]
    python -m insar_pipeline.cli validate <config.yaml> --protocol <protocol.yaml> [--out DIR] ...
    python -m insar_pipeline.cli storage [keep-results|delete <name> [--apply]]

Numbers pick actions, letters pick utilities. Runs are started, followed and
stopped exactly as the dashboard does it (runs.start / runs.state /
runs.stop): one run per project under the project lock, the run recorded in
<project>/run.json, its log under <project>/logs/, its end in
<project>/run_result.json. So a run started here shows in the dashboard and
the other way round, and a second start on the same project is refused.
This menu used to keep its own job list of bare PIDs in
~/.insar_pipeline_cli.json, launch the scripts/ runners (which copied the
stage chain) and stop by PID (review 5 Oct 2026). The last config used is
still remembered in that file.
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PKG = Path(__file__).resolve().parent            # /home/kylew/insar_pipeline
PARENT = PKG.parent                              # the package is imported from here
STATE = Path.home() / ".insar_pipeline_cli.json"
PY = sys.executable
# One-stage jobs this menu can run under the project lock (`--stage NAME`),
# each through the pipeline's own stage function so the order and rules are
# the pipeline's: figures used to run `figures && panel_history`, the reverse
# of pipeline.run_figures, so the summary table was drawn from the previous
# run's histories.
STAGES = {"quality_assessment": "run_quality_assessment", "figures": "run_figures"}

BOLD, DIM, CYAN, ORANGE, GREEN, RED, END = "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[32m", "\033[31m", "\033[0m"


def say(msg="", c=""):
    print(f"{c}{msg}{END}" if c else msg)


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:                                    # noqa: BLE001
        return {}


def save_state(st):
    st = {k: v for k, v in st.items() if k != "jobs"}   # the old PID job list is gone
    STATE.write_text(json.dumps(st, indent=1))


def load_cfg(path):
    sys.path.insert(0, str(PARENT))
    from insar_pipeline.config import load_config
    return load_config(path)


def _runs():
    sys.path.insert(0, str(PARENT))
    from insar_pipeline import runs
    return runs


def choose_config(st):
    default = st.get("config", "")
    say("\nConfig file (a .yaml written by the project builder, or an existing one).")
    if default:
        say(f"  Enter for the last one: {default}", DIM)
    v = input("  path> ").strip() or default
    if not v:
        return None
    p = Path(v).expanduser()
    if not p.exists():
        say(f"  {p} does not exist.", RED); return st.get("config")
    try:
        cfg = load_cfg(str(p))
        say(f"  OK: {cfg.backend} route, {cfg.start_date} → {cfg.end_date}, project {cfg.project_dir}", GREEN)
    except Exception as exc:                             # noqa: BLE001
        say(f"  The config does not load: {exc}", RED); return st.get("config")
    st["config"] = str(p); save_state(st)
    return str(p)


def logs_dir(cfg):
    d = Path(cfg.project_dir) / "logs"; d.mkdir(parents=True, exist_ok=True); return d


def _what(rec, article="a"):
    return f"{article} {rec['kind']} run" if rec.get("kind") else f"{article} run"


def start_run(cfg, path, kind):
    """Start a run kind (runs.KINDS) detached, the way the dashboard does."""
    runs = _runs()
    try:
        rec = runs.start(str(Path(path).resolve()), cfg.project_dir, kind)
    except RuntimeError as exc:
        say(f"  Not started: {exc}", RED)
        return None
    say(f"  started pid {rec['pid']}; log {rec['log']}", GREEN)
    say("  It keeps running if you quit this menu, and shows in the dashboard. "
        "'l' follows the log, '9' shows status, 's' stops it.", DIM)
    return rec


def start_stage(cfg, path, stage):
    """One pipeline stage in the background, under the project lock."""
    runs = _runs()
    if runs.state(cfg.project_dir) == "running":
        held = runs.lock_holder(cfg.project_dir) or runs.record(cfg.project_dir) or {}
        say(f"  Not started: {_what(held)} is going in {cfg.project_dir} (pid {held.get('pid', '?')}).", RED)
        return None
    log = logs_dir(cfg) / f"{stage}_{datetime.now():%Y%m%d_%H%M%S}.log"
    with open(log, "w") as fh:
        runs.write_log_label(fh)         # the log's first line names the code (ERR-04, 16 Oct 2026)
        p = subprocess.Popen([PY, "-m", "insar_pipeline.cli", "--stage", stage, str(Path(path).resolve())], stdin=subprocess.DEVNULL,
                             stdout=fh, stderr=subprocess.STDOUT, cwd=str(PARENT), start_new_session=True,
                             env={**os.environ, "PYTHONPATH": str(PARENT), "PYTHONUNBUFFERED": "1"})
    say(f"  started pid {p.pid}; log {log}", GREEN)
    return p


def stage_main(stage, path) -> int:
    """`python -m insar_pipeline.cli --stage <stage> <config>`: the pipeline's
    own stage function, with the teardown and the project lock. 0 ran,
    1 failed (the log says why), 3 another run holds the project."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    sys.path.insert(0, str(PARENT))
    from insar_pipeline import pipeline, procs, runs
    if stage not in STAGES:
        say(f"unknown stage {stage!r}; one of {list(STAGES)}", RED)
        return 2
    cfg = load_cfg(path)
    procs.install_teardown()
    try:
        with runs.project_lock(cfg.project_dir, kind=stage, config=str(path)):
            return 0 if getattr(pipeline, STAGES[stage])(cfg) else 1
    except RuntimeError as exc:
        if "already running" not in str(exc):
            raise
        say(f"Not started: {exc}", RED)
        return 3


# The optional checks a finished project can be given by hand, each the
# module's own CLI (injection_test.main / crosscheck.main) under the project
# lock, so it cannot overlap a run that is rewriting the products it reads.
TOOLS = {"injection-test": ("injection_test", "injection_test"), "cross-check": ("crosscheck", "crosscheck")}


def tool_main(name, argv) -> int:
    """`python -m insar_pipeline.cli injection-test|cross-check <config> ...`.
    0 ran, 1 failed, 2 usage, 3 another run holds the project."""
    import importlib
    sys.path.insert(0, str(PARENT))
    from insar_pipeline import procs, runs
    module, kind = TOOLS[name]
    if not argv or argv[0].startswith("-"):
        say(f"usage: python -m insar_pipeline.cli {name} <config.yaml> [options]  (--help for the options)", RED)
        return 2
    cfg = load_cfg(argv[0])
    procs.install_teardown()
    mod = importlib.import_module(f"insar_pipeline.{module}")
    try:
        with runs.project_lock(cfg.project_dir, kind=kind, config=str(argv[0])):
            try:
                return int(mod.main(argv) or 0)
            except Exception as exc:                     # noqa: BLE001 - the module wrote its 'failed' record
                say(f"{name} failed: {exc}", RED)
                return 1
    except RuntimeError as exc:
        if "already running" not in str(exc):
            raise
        say(f"Not started: {exc}", RED)
        return 3


def validate_main(argv) -> int:
    """`python -m insar_pipeline.cli validate <config> --protocol <yaml> [--out DIR] ...`:
    the validation harness (validation.py, PLAN phase 1 V1) on a finished
    run. It only reads the project, so it takes no project lock (that would
    write the lock file into the project); it refuses while a run holds the
    project instead, because the products it reads may be half rewritten.
    0 written, 1 not made (the message says why), 2 usage, 3 a run is going."""
    sys.path.insert(0, str(PARENT))
    from insar_pipeline import runs, validation
    if argv[:1] in (["-h"], ["--help"]):
        return int(validation.main(["--help"]) or 0)        # argparse prints the options and exits 0
    if not argv:
        say("usage: python -m insar_pipeline.cli validate <config.yaml> --protocol <protocol.yaml> [--out DIR] "
            "[--run ID] [--cache DIR] [--ga-xlsx FILE] [--points FILE] [--up FILE] [--outlines FILE] "
            "[--lines FILE] [--no-sensitivity] [--no-epoch-coverage]   (--help for what each does)", RED)
        return 2
    if argv[0].startswith("-"):
        say("usage: the config comes first: validate <config.yaml> --protocol <protocol.yaml> ...", RED)
        return 2
    try:
        cfg = load_cfg(argv[0])
    except Exception as exc:                                  # noqa: BLE001 - said in words, not a traceback
        say(f"Validation not made: the config {argv[0]} does not load ({exc}).", RED)
        return 1
    if runs.state(cfg.project_dir) == "running":
        say(f"Not started: a run is going in {cfg.project_dir}; validate its products after it ends.", RED)
        return 3
    return int(validation.main(argv, cfg=cfg) or 0)


def storage_main(argv) -> int:
    """`insar storage [keep-results|delete] ...` (storage.py): the overview of
    every project, and keep-results-only / delete, each a dry run unless
    --apply. 0 done or shown, 1 not done, 2 usage."""
    sys.path.insert(0, str(PARENT))
    from insar_pipeline import storage
    try:
        return int(storage.main(argv) or 0)
    except SystemExit as exc:                                 # argparse: usage or --help
        return int(exc.code or 0)


def run_fg(cmd):
    say(f"  $ {' '.join(str(c) for c in cmd)}", DIM)
    return subprocess.call(cmd, cwd=str(PARENT), env={**os.environ, "PYTHONPATH": str(PARENT)})


def follow(log, lines=30):
    say(f"  {log}  (Ctrl-C returns to the menu)", DIM)
    try:
        subprocess.call(["tail", "-n", str(lines), "-f", str(log)])
    except KeyboardInterrupt:
        print()


# ---------------------------------------------------------------- actions
def act_builder(_cfg, _path):
    say("\n  Starting the project builder at http://localhost:8765 (a local server: OpenStreetMap maps and")
    say("  place search refuse pages opened as plain files). Ctrl-C returns to the menu and stops it.", DIM)
    say("  Draw or upload the project area, the mine outlines and any report areas, check the archive,", GREEN)
    say("  then create the project: the builder writes its config itself. Choose '2' here to load it.", GREEN)
    try:
        subprocess.call([PY, "-m", "insar_pipeline.map_builder"], cwd=str(PARENT),
                        env={**os.environ, "PYTHONPATH": str(PARENT)})
    except KeyboardInterrupt:
        print()


def act_edit(cfg, path):
    editor = os.environ.get("EDITOR") or ("nano" if shutil.which("nano") else "vi")
    run_fg([editor, path])


def act_check(cfg, path):
    say("\n  Loading the config, checking credentials, sizing the run (downloads nothing)...")
    sys.path.insert(0, str(PARENT))
    try:
        from insar_pipeline import stack_backend as sb
        problems = sb.check_weather_credentials(cfg) if cfg.mintpy_tropo_method == "pyaps" else []
        for p in problems:
            say(f"  ! {p.get('text', p) if isinstance(p, dict) else p}", ORANGE)
    except Exception as exc:                             # noqa: BLE001
        say(f"  credential check skipped: {exc}", DIM)
    run_fg([PY, "-m", "insar_pipeline.pipeline", path, "--plan-only"])


def act_resume(cfg, path):
    runs = _runs()
    try:
        kind, why = runs.suggest_kind(cfg.project_dir, cfg)
        say(f"\n  From the files on disk: {why} -> {runs.KINDS.get(kind, (kind,))[0]}")
    except Exception as exc:                             # noqa: BLE001
        say(f"  could not read the project's state: {exc}", ORANGE)
    if input("  start now? [y/N] ").strip().lower() == "y":
        start_run(cfg, path, "resume")


def act_full(cfg, path):
    say("\n  Full pipeline: download → coregister → [phase linking] → unwrap → MintPy → deliverables → assessment → figures → report.")
    say("  This takes hours to days depending on the record. Read the plan ('3') first; after a crash use 'r' (resume).", DIM)
    if input("  start now? [y/N] ").strip().lower() == "y":
        start_run(cfg, path, "full")


def act_mintpy(cfg, path):
    say("\n  MintPy onward on a finished stack (new reference pixel, thresholds, exclusions), then deliverables, assessment, figures, report.")
    if input("  start now? [y/N] ").strip().lower() == "y":
        start_run(cfg, path, "mintpy")


def act_miaplpy(cfg, path):
    say("\n  MiaplPy phase linking on the coregistered stack (run_01..run_12 done, real merged SLCs), then deliverables onward.")
    say("  Refused if the coregistration is not finished; 'r' (resume) picks the right start.", DIM)
    if input("  start now? [y/N] ").strip().lower() == "y":
        start_run(cfg, path, "miaplpy")


def act_qa(cfg, path):
    start_stage(cfg, path, "quality_assessment")


def act_figures(cfg, path):
    start_stage(cfg, path, "figures")


def act_page(cfg, path):
    out = Path(cfg.project_dir) / "deliverables" / "summary_page.html"
    rc = run_fg([PY, str(PKG / "scripts" / "collect_results.py"), path])
    if rc == 0:
        figs = Path(cfg.project_dir) / "deliverables" / "figures"
        cmd = [PY, str(PKG / "scripts" / "run_summary_page.py"), str(Path(cfg.project_dir) / "run_results.json"), str(out)]
        if figs.exists():
            cmd += ["--figures", str(figs)]
        if run_fg(cmd) == 0:
            say(f"  page: {out}", GREEN)


def act_status(cfg, path):
    P = Path(cfg.project_dir)
    say(f"\n  Project {P}", BOLD)
    items = [("downloads (SAFEs)", cfg.slc_dir, "*.SAFE"), ("stack run files", cfg.stack_dir / "run_files", "run_*"),
             ("unwrapped pairs", cfg.stack_dir / "merged" / "interferograms", "*/filt_fine.unw"),
             ("MintPy stack", cfg.mintpy_dir / "inputs", "ifgramStack.h5"), ("velocity (geocoded)", cfg.mintpy_dir / "geo", "geo_velocity.h5"),
             ("deliverables", P / "deliverables", "*.tif"), ("quality assessment", P, "quality_assessment.json"),
             ("figures", P / "deliverables" / "figures", "*.png"), ("panel history", P / "deliverables", "panel_history.json"),
             ("site history", P / "deliverables", "site_history.json"), ("summary page", P / "deliverables", "summary_page.html")]
    for label, d, pat in items:
        hits = sorted(Path(d).glob(pat)) if Path(d).exists() else []
        if hits:
            newest = max(hits, key=lambda x: x.stat().st_mtime)
            say(f"  {GREEN}✓{END} {label:22s} {len(hits):5d}   newest {datetime.fromtimestamp(newest.stat().st_mtime):%Y-%m-%d %H:%M}")
        else:
            say(f"  {DIM}·{END} {label:22s}     -")
    runs = _runs()
    st = runs.state(P)
    rec = runs.record(P) or {}
    if st == "running":
        held = runs.lock_holder(P) or {}
        who = rec if runs.alive(rec) else held
        since = who.get("started")
        say(f"\n  running: {who.get('kind') or 'run'}  pid {who.get('pid', '?')}"
            + (f"  since {datetime.fromtimestamp(since):%d %b %H:%M}" if since else "")
            + (f"  log {who['log']}" if who.get("log") else ""), BOLD)
    else:
        res = runs.result(P) or {}
        say(f"\n  last run: {st}" + (f" ({res.get('kind')})" if res.get("kind") else ""), BOLD)
        if st == "interrupted":
            info = runs.interruption(P, cfg) or {}
            if info.get("text"):
                say(f"  {info['text']}", ORANGE)
        elif st in ("failed", "stopped") and res.get("plain_cause"):
            say(f"  {res['plain_cause']}", ORANGE)
            if res.get("next_action"):
                say(f"  {res['next_action']}", DIM)
    qa = P / "quality_assessment.json"
    if qa.exists():
        q = json.loads(qa.read_text())
        # counted as the verdict counts them (D2, 7 Oct 2026): out of the
        # applicable checks, naming any that cannot apply
        from insar_pipeline.quality_assessment import check_counts
        say(f"\n  verdict: {q.get('verdict')}  ({check_counts(q)['text']})", BOLD)


def act_log(cfg, path):
    runs = _runs()
    rec = runs.record(cfg.project_dir) or {}
    logs = sorted(logs_dir(cfg).glob("*.log"), key=lambda x: x.stat().st_mtime)
    if rec.get("log") and runs.alive(rec) and Path(rec["log"]).exists():
        follow(rec["log"])
    elif logs:
        follow(logs[-1])
    else:
        say("  no logs yet.", DIM)


def act_stop(cfg, path):
    runs = _runs()
    P = cfg.project_dir
    if runs.state(P) != "running":
        out = runs.stop(P)                   # also stops commands a dead earlier run left behind
        say(f"  {out}.", DIM); return
    rec = runs.record(P) or {}
    who = rec if runs.alive(rec) else (runs.lock_holder(P) or {})
    v = input(f"  stop {_what(who, 'the')} (pid {who.get('pid', '?')})? [y/N] ").strip().lower()
    if v == "y":
        say(f"  {runs.stop(P)}. The run stops its commands and leaves resumable state ('r' resumes).", ORANGE)


MENU = [
    ("1", "New project: open the project builder (map: area, mines, report areas)", act_builder, False),
    ("2", "Choose or edit the config file", None, False),
    ("3", "Check the config and size the run (downloads nothing)", act_check, True),
    ("r", "Resume: pick up where the last run stopped (worked out from the files on disk)", act_resume, True),
    ("4", "Run the full pipeline", act_full, True),
    ("5", "Run MintPy onward on a finished stack", act_mintpy, True),
    ("p", "Phase linking (MiaplPy) on a finished coregistration, then deliverables onward", act_miaplpy, True),
    ("6", "Re-run the quality assessment", act_qa, True),
    ("7", "Redraw figures, panel and site histories", act_figures, True),
    ("8", "Build the summary page", act_page, True),
    ("9", "Project status", act_status, True),
    ("l", "Follow the latest log", act_log, True),
    ("s", "Stop the running job", act_stop, True),
    ("q", "Quit", None, False),
]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--stage"]:
        if len(argv) != 3:
            say("usage: python -m insar_pipeline.cli --stage quality_assessment|figures <config.yaml>", RED)
            sys.exit(2)
        sys.exit(stage_main(argv[1], argv[2]))
    if argv[:1] == ["--version"]:
        sys.path.insert(0, str(PARENT))
        from insar_pipeline import version
        print(version.label())
        return
    if argv[:1] and argv[0] in TOOLS:
        sys.exit(tool_main(argv[0], argv[1:]))
    if argv[:1] == ["validate"]:
        sys.exit(validate_main(argv[1:]))
    if argv[:1] == ["storage"]:
        sys.exit(storage_main(argv[1:]))
    st = load_state()
    path = argv[0] if argv else st.get("config", "")
    cfg = None
    if path and Path(path).exists():
        try:
            cfg = load_cfg(path); st["config"] = path; save_state(st)
        except Exception as exc:                         # noqa: BLE001
            say(f"config {path} does not load: {exc}", RED); path = ""
    while True:
        say(f"\n{BOLD}InSAR pipeline{END}  {DIM}config:{END} {path or '(none - choose 1 or 2)'}")
        for key, label, _, _ in MENU:
            say(f"  {CYAN}{key}{END}  {label}")
        choice = input("> ").strip().lower()
        if choice == "q":
            return
        if choice == "2":
            path = choose_config(st) or path
            if path:
                cfg = load_cfg(path)
                if input("  edit it now? [y/N] ").strip().lower() == "y":
                    act_edit(cfg, path); cfg = load_cfg(path)
            continue
        entry = next((m for m in MENU if m[0] == choice), None)
        if not entry or entry[2] is None:
            continue
        _, _, fn, needs_cfg = entry
        if needs_cfg and cfg is None:
            say("  choose a config first (2).", ORANGE); continue
        try:
            fn(cfg, path)
        except KeyboardInterrupt:
            print()
        except Exception as exc:                         # noqa: BLE001
            say(f"  failed: {exc}", RED)


if __name__ == "__main__":
    main()
