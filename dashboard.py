"""Full-screen dashboard: projects, a new area in three ways, preflight, and
a live view of a running pipeline.

    ./scripts/insar                 (or: python -m insar_pipeline.dashboard [config.yaml])
    ./scripts/insar --menu          the old numbered menu (cli.py)

Everything shown about a run comes from what is on disk: the run record
(runs.py), the progress feed each stage appends to (progress.py), the run's
own end record (run_result.json), the log, and the products. Runs are
detached, so quitting the dashboard never stops one; opening the project
again reattaches.

Every start of a run that processes (Whole pipeline, Resume, Phase linking
onward, Re-unwrap, MintPy onward) goes through the preflight screen with that
run kind: the disk budget against the planner's peak, memory, Earthdata and
ERA5 access, the patched MiaplPy. Until 5 Oct 2026 only the `n` flow reached
it; s -> Whole pipeline on a builder-made project started a 1-2 day run with
none of those checks, while the builder's own text said "the preflight
checks run first".

The area figures are the client report's own (client_report.json): the rate
with its 95% range and the report's reading ("within normal ground
variation" / "movement detected"). The projects table used to show the rate
alone under "relative to GNSS" and the run view an unlabelled 1 sigma, so a
coworker reading the dashboard could quote "-0.22 in/yr" for an area the
report calls indistinguishable from farmland. The run view's totals carry
their 95% range too (total_text, F8), and the only ground it calls a place
to check is a feature the report lists (feature_lines, F2: a stand-out the
quality assessment found alone stays in its technical record).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (Button, Checkbox, DataTable, Footer, Header, Input, Label, Markdown, OptionList,
                             RichLog, Select, Sparkline, Static, TabbedContent, TabPane)
from textual.widgets.option_list import Option

from . import areas, progress, runs

MM_IN = 25.4
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
STATE_STYLE = {"running": "bold cyan", "done": "green", "failed": "bold red", "stopped": "yellow",
               "interrupted": "bold yellow", "idle": "dim", "invalid": "red"}
STATUS_ICON = {"start": "▶", "running": "▶", "done": "✓", "failed": "✗", "skipped": "–"}
STAGE_LABELS = {
    "pipeline": "Pipeline", "run": "Run", "checks": "Checks before the run", "disk": "Disk watch",
    "download": "Download Sentinel-1 data", "orbits": "Orbit files", "stage_inputs": "Stage the inputs",
    "era5_prefetch": "ERA5 weather download (in the background)",
    "disp_search": "Find DISP-S1 granules",
    "disp_prefetch": "Download DISP-S1 windows", "disp_stitch": "Stitch reference epochs + unwrap checks",
    "disp_resample": "Resample onto the record grid", "mintpy_troposphere": "Troposphere (weather delays)",
    "mintpy_deramp": "Deramp (deforming area excluded)", "mintpy_dem_error": "DEM-error correction",
    "mintpy_reference_date": "Reference date", "mintpy_velocity": "Rate fit", "disp_products": "Products into place",
    "deliverables": "Deliverables (GeoTIFF, KMZ, points)", "quality_assessment": "Quality assessment",
    "figures": "Figures, panel + site histories", "client_report": "GNSS tie + client report",
    "provenance": "Provenance + release status", "crosscheck": "DISP-S1 cross-check (optional)",
    "injection_test": "Injection-recovery test (optional)",
    "validation": "Validation against ground truth (experiment runs)",
}
# The kinds whose start goes through the preflight screen: everything that
# processes. "products" and "plan" start directly (nothing to download, no
# multi-hour compute ahead of a check that could have failed first).
PREFLIGHT_KINDS = ("resume", "full", "miaplpy", "reunwrap", "mintpy")
# What a lock holder's kind means when it is not a pipeline run.
LOCK_KIND_WORDS = {"cleanup": "cleaning up", "delete": "deleting the project", "report": "rebuilding the report", "injection_test":
                   "injection-recovery test", "crosscheck": "DISP-S1 cross-check"}
# The user guide the docs lane writes (`?` opens it).
GUIDE_NAMES = ("USER_GUIDE.md", "docs/USER_GUIDE.md")
Z95_FALLBACK = 1.96          # client_report.Z95, when that module cannot be imported


def stage_label(key: str) -> str:
    if key in STAGE_LABELS:
        return STAGE_LABELS[key]
    if key.startswith("run_"):
        parts = key.split("_")
        return f"Stack {parts[1]}: " + " ".join(parts[2:])
    if key.startswith("miaplpy_"):
        return "Phase linking: " + key[8:].replace("_", " ")
    if key.startswith("mintpy_"):
        return "MintPy: " + key[7:].replace("_to_", " → ").replace("_", " ")
    return key.replace("_", " ")


def fmt_dur(s: float | None) -> str:
    if s is None:
        return ""
    s = int(max(0, s))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def bar(done: int | None, total: int | None, width: int = 22) -> Text:
    if not total:
        return Text("")
    f = max(0.0, min(1.0, (done or 0) / total))
    n = int(round(f * width))
    t = Text("█" * n, style="cyan")
    t.append("░" * (width - n), style="grey35")
    t.append(f" {100 * f:3.0f}%", style="bold")
    return t


def rate_text(mm: float | None) -> str:
    """A rate in the client report's headline precision (client_report
    rate_cell: 3 decimals of an inch). No bare ± here: the dashboard used to
    print the 1 sigma unlabelled next to the report's 95% range."""
    if mm is None:
        return "—"
    return f"{mm / MM_IN:+.3f} in/yr"


def range_text(rng) -> str:
    """'(-0.80 to +0.36 in/yr)': the report's 95% range at its precision
    (client_report range_cell: 2 decimals)."""
    if not rng or rng[0] is None or rng[1] is None:
        return ""
    return f"({rng[0] / MM_IN:+.2f} to {rng[1] / MM_IN:+.2f} in/yr)"


def anchored_of(rep: dict) -> bool | None:
    """Whether the report's figures carry the GNSS background: client_report
    'anchored' (every build since Sep 2026), else the gnss block's (the 5 Oct
    contract), else any area's anchored_to_gnss. None when the file does not say."""
    if "anchored" in rep:
        return bool(rep["anchored"])
    g = rep.get("gnss")
    if isinstance(g, dict) and "anchored" in g:
        return bool(g["anchored"])
    A = [a for a in rep.get("areas") or [] if isinstance(a, dict) and "anchored_to_gnss" in a]
    return any(a["anchored_to_gnss"] for a in A) if A else None


def _z95() -> float:
    try:
        from .client_report import Z95
        return float(Z95)
    except Exception:                                         # noqa: BLE001
        return Z95_FALLBACK


def area_view(a: dict, anchored: bool | None = True) -> dict:
    """One report area as the client report states it: {rate, range, range_mm,
    reading, style}. The range and the reading come from client_report.json
    (range95_mm_yr, reading, testable: the fields the report renders from);
    a report written before those fields existed gets exactly what its own
    HTML printed - vertical ± client_report.Z95 x the 1 sigma (range_cell) and
    the table's tag (detected -> 'Movement detected', else 'Within normal
    ground variation') - so the dashboard never derives a figure of its own."""
    mm = a.get("vertical_mm_yr")
    full = a.get("kind") == "full"
    if "range95_mm_yr" in a:
        rng = a.get("range95_mm_yr")
        rng = tuple(rng) if rng else None
    else:
        u = a.get("vertical_uncertainty_1sigma_mm_yr")
        rng = (mm - _z95() * u, mm + _z95() * u) if mm is not None and u is not None else None
    reading = a.get("reading")
    if a.get("testable") is False:
        why = a.get("not_testable_reason")
        reading = reading or ("Not testable against ordinary ground" + (f": {why}" if why else ""))
        rng = None if "range95_mm_yr" not in a else rng
    if not reading:
        if full:
            reading = ("Region moves with the GNSS network" if anchored
                       else "Whole area, relative to its own farmland")
        elif a.get("detected"):
            reading = "Movement detected"
        else:
            reading = "Within normal ground variation"
    low = reading.lower()
    style = ("dim" if a.get("testable") is False or "not testable" in low else
             "bold yellow" if ("detected" in low and "not" not in low) or "stands out" in low else "green")
    return {"rate": rate_text(mm), "range": range_text(rng), "range_mm": rng, "reading": reading, "style": style}


def area_line(a: dict, anchored: bool | None = True) -> Text:
    v = area_view(a, anchored)
    t = Text(f"{a.get('name', '?')}: ", style="bold")
    t.append(v["rate"])
    if v["range"]:
        t.append(" " + v["range"], style="dim")
    t.append(" · ")
    t.append(v["reading"][:1].lower() + v["reading"][1:] if v["reading"] else "", style=v["style"])
    return t


def _ri():
    """report_insights, or None on a partial install (the dashboard then
    prints the plain words)."""
    try:
        from . import report_insights
        return report_insights
    except Exception:                                         # noqa: BLE001
        return None


def _signed(x: float, nd: int = 2) -> str:
    """'+0.38' / '-1.47', never '-0.00' (report_insights.signed)."""
    ri = _ri()
    return ri.signed(x, nd) if ri else f"{float(x):+.{nd}f}"


def ground_of(rep: dict) -> str:
    """The comparison ground's name in the report's words ('farmland')."""
    ri = _ri()
    try:
        return ri.ground_word(rep) if ri else "farmland"
    except Exception:                                         # noqa: BLE001
        return "farmland"


def total_text(a: dict) -> str:
    """An area's measured total as the client report gives it, with its 95%
    range (F8, 7 Oct 2026: the run view printed '+0.53 in total' bare while
    the page, PDF and KMZ never print a total without its range - E9). The
    range is the report's own, as its summary table reads it:
    total_range95_mm, else (an area that could not be compared with
    farmland) untested_total_range95_mm. A report built before those fields
    existed says to rebuild it; no range is derived here. Not "built before
    7 Oct 2026": both trust-check copies' reports were generated on 7 Oct
    (09:00) without the fields, and the run view printed that beside
    "Generated 2026-10-07T09:00". '' when the area has no total."""
    tot = a.get("total_mm")
    if tot is None:
        return ""
    s = f"{_signed(tot / MM_IN)} in total"
    trng = a.get("total_range95_mm") or a.get("untested_total_range95_mm")
    if trng and len(trng) >= 2 and None not in trng[:2]:
        return s + f" (95% range {_signed(trng[0] / MM_IN)} to {_signed(trng[1] / MM_IN)} in)"
    if "total_range95_mm" not in a and "untested_total_range95_mm" not in a:
        return s + " (no 95% range: this report was built before totals carried one; rebuild it)"
    return s + " (no 95% range)"


def feature_verdict(f: dict) -> tuple[str, str]:
    """(words, style) for one listed feature, by the page's own rule
    (client_report.page's summary): not tested against chance -> a
    specialist's look; stands_out_clearly True -> 'stands out clearly' (E7);
    any other listed feature -> 'worth a look'."""
    if f.get("stands_out", True) is None:
        return "not tested against chance: a specialist's look first", "dim"
    if f.get("stands_out_clearly") is True:
        return "stands out clearly", "bold yellow"
    return "worth a look", "yellow"


def feature_lines(rep: dict) -> list[tuple[str, str]]:
    """[(line, style)] for the run view: the features the client report
    lists (client_report.json 'features', in its order and numbering), the
    only ground the dashboard calls an area to check on the ground.

    F2 (7 Oct 2026): the run view printed "One area to check on the ground"
    whenever the whole area held a quality-assessment stand-out, also when
    the report lists no feature (Site A: the QA's 4 km stand-out beside
    the page's "No local feature stands out"). A stand-out found only by the
    QA stays in the technical record; one the report lists is shown as that
    feature. [] for a report built before features were recorded."""
    if "features" not in rep and "features_assessed" not in rep:
        return []
    ground = ground_of(rep)
    if rep.get("features_assessed") is False:
        return [(f"Local features could not be assessed (too little ordinary {ground})", "dim")]
    feats = [f for f in rep.get("features") or [] if isinstance(f, dict)]
    if not feats:
        return [(f"No local feature stands out from ordinary {ground}", "dim")]
    n = len(feats)
    out = [("One feature to check on the ground" if n == 1 else f"{n} features to check on the ground",
            "bold yellow")]
    ri = _ri()
    for i, f in enumerate(feats, 1):
        try:
            kind, where = ri.feature_kind(f), ri.lalo_txt(f["lat"], f["lon"])
        except Exception:                                     # noqa: BLE001 - the plain words
            kind = f"{'sinking' if (f.get('sign') or 0) < 0 else 'rising'} ground"
            where = f"{f.get('lat', 0):.3f}, {f.get('lon', 0):.3f}"
        v = f.get("vertical_relative_mm_yr", f.get("mean_rel_vertical_mm_yr"))
        rate = ("no rate (ground reshaped)" if f.get("ground_reshaped") else
                f"{_signed(v / MM_IN)} in/yr relative to the surrounding {ground}" if v is not None else "")
        words, style = feature_verdict(f)
        out.append((f"  Feature {i}: {kind}, about {f.get('area_km2') or 0:.1f} km² near {where}"
                    + (f"; {rate}" if rate else "") + f" · {words}", style))
    return out


def _load_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


HOLD = "hold for specialist"
REPORT_OUT_OF_DATE = "(report out of date)"


def _mtime_or_none(f: Path) -> float | None:
    try:
        return f.stat().st_mtime
    except OSError:
        return None


def send_rule_shown(project_dir) -> tuple[str | None, bool]:
    """(the send rule's value - 'send' / 'send with notes' / 'hold for
    specialist' - or None, whether the client report is out of date).

    The report's rule (the QA's re-evaluated with the GNSS tie, which runs
    after the QA, as the release gate reads it) counts only when
    client_report.json is at least as new as quality_assessment.json
    (decision D7, 7 Oct 2026; provenance._send_rule reads them in the same
    order). Otherwise, or when the report has no rule of its own:
    summary.json's when written after this QA and re-evaluated with the tie
    (send_rule_gnss_considered; otherwise it is a copy of the QA's, possibly
    an older one), then quality_assessment.json's own, which never saw the
    tie. A report older than the QA is out of date: the rule shown is then
    the newer QA's, and the report must be rebuilt before it can go out.
    "Older" is the release gate's own test (release.report_predates_qa,
    which takes files written within QA_AFTER_REPORT_GRACE_S of each other
    as written together), so the mark and the gate's hard blocker - FINAL /
    DRAFT beside it - never disagree.
    The value is None when none of them has one (a QA written before the
    rule existed)."""
    P = Path(project_dir)
    D = P / "deliverables"
    rep_f, sum_f, qa_f = D / "client_report.json", D / "summary.json", P / "quality_assessment.json"
    qa_t = _mtime_or_none(qa_f)

    def value(d):
        sr = d.get("send_rule") if isinstance(d, dict) else None
        v = sr.get("value") if isinstance(sr, dict) else sr      # summary.json stores the value alone
        return v if isinstance(v, str) and v else None

    rep_t = _mtime_or_none(rep_f)
    try:
        from .release import QA_AFTER_REPORT_GRACE_S as grace
    except Exception:                                         # noqa: BLE001 - no gate: the plain comparison
        grace = 0.0
    stale = rep_t is not None and qa_t is not None and qa_t > rep_t + grace
    if rep_t is not None and not stale:
        v = value(_load_json(rep_f))
        if v:
            return v, False
    S = _load_json(sum_f)
    sum_t = _mtime_or_none(sum_f)
    if isinstance(S, dict) and S.get("send_rule_gnss_considered") is True and sum_t is not None \
            and (qa_t is None or sum_t >= qa_t):
        v = value(S)
        if v:
            return v, stale
    return value(_load_json(qa_f)), stale


def qa_send_rule(project_dir) -> str | None:
    """The send rule as the dashboard shows it (send_rule_shown): the
    report's when the report is at least as new as the quality assessment,
    else the newer QA's marked '(report out of date)'. None when there is
    none."""
    v, stale = send_rule_shown(project_dir)
    return f"{v} {REPORT_OUT_OF_DATE}" if v and stale else v


_RELEASE_CACHE: dict = {}


def release_status(cfg_or_dir) -> dict | None:
    """release.status for the project, cached on the mtimes of the files it
    reads (the projects table refreshes every 5 s). None when release.py is
    not importable or the project has no client report."""
    P = Path(getattr(cfg_or_dir, "project_dir", cfg_or_dir))
    # everything release.blockers reads, the velocity products for its staleness test included,
    # the report's own signal-to-noise raster, the time-series run's unfinished marker (in the
    # config's MintPy folder and the project's) and the settings file naming who prepared it
    M = Path(getattr(cfg_or_dir, "mintpy_dir", P / "mintpy"))
    D = P / "deliverables"
    files = (D / "client_report.json", D / "review.json",
             P / "quality_assessment.json", D / "gnss_tie.json",
             M / "geo" / "geo_velocity.h5", M / "velocity.h5",
             D / "technical" / "cross_check.json", D / "crosscheck.json",
             D / "rate_signal_to_noise.tif", D / "config_snapshot.yaml",
             *dict.fromkeys((M / "run_incomplete.json", P / "mintpy" / "run_incomplete.json")))
    if not files[0].exists():
        return None
    try:
        from .settings import settings_file
        files += (settings_file(),)
    except Exception:                                         # noqa: BLE001 - no settings file to watch
        pass
    # a loaded config's own file: its route and cross_check_disp change the
    # answer, and the run screen reloads it after an edit
    src = getattr(cfg_or_dir, "_source_path", None)
    if src:
        files += (Path(src),)
    key = tuple(f.stat().st_mtime if f.exists() else 0 for f in files)
    # the issues already made (reissue_without_note, M3): a final_* folder or
    # a package added or removed changes the answer with no watched file
    # touched (final check of 7 Oct 2026)
    key += tuple(sorted(x.name for x in D.glob("final_*"))) + ("|",) + tuple(
        sorted(x.name for x in (D / "releases").glob("*")))
    # a folder alone and a loaded config are judged differently (the cross-check
    # counts only when the config turns it on), so they are cached apart
    slot = (str(P), hasattr(cfg_or_dir, "project_dir"))
    hit = _RELEASE_CACHE.get(slot)
    if hit and hit[0] == key:
        return hit[1]
    try:
        from . import release
        st = release.status(cfg_or_dir)
    except Exception:                                         # noqa: BLE001 - shown as unknown
        return None
    _RELEASE_CACHE[slot] = (key, st)
    return st


def project_row(p) -> dict:
    """The projects-table cells for one runs.Project: {verdict, results}.
    Verdict: the send rule when there is one (send_rule_shown; else the
    report's verdict), marked '(report out of date)' when the report is older
    than the quality assessment, with DRAFT/FINAL from the release gate. Results: each area as the client
    report reads it, '(not GNSS-anchored)' when the figures carry no GNSS
    background (the column header said 'relative to GNSS' for every project)."""
    out = {"verdict": Text(p.verdict or ""), "results": Text(p.error or "—", style="red" if p.error else "dim")}
    if not p.project_dir:
        return out
    P = Path(p.project_dir)
    rep = _load_json(P / "deliverables" / "client_report.json")
    sr, stale = send_rule_shown(P)
    v = Text(sr or p.verdict or "", style="bold yellow" if sr == HOLD else "")
    if stale:
        v.append((" " if v.plain else "") + REPORT_OUT_OF_DATE, style="bold yellow")
    st = release_status(P) if rep else None
    if st is not None:
        v.append("  FINAL" if st.get("final") else "  DRAFT", style="green" if st.get("final") else "yellow")
    out["verdict"] = v
    if rep:
        anch = anchored_of(rep)
        t = Text()
        for i, a in enumerate(x for x in rep.get("areas") or [] if isinstance(x, dict)):
            if i:
                t.append("   ")
            t.append_text(area_line(a, anch))
        if anch is False:
            t.append("   (not GNSS-anchored)", style="yellow")
        out["results"] = t if t.plain else Text("—", style="dim")
    return out


def open_path(path: Path) -> str:
    """Open a file in the Windows browser from WSL, else xdg-open."""
    if shutil.which("wslpath") and shutil.which("explorer.exe"):
        win = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True).stdout.strip()
        subprocess.Popen(["explorer.exe", win], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return win
    if shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(path)


def edit_config(app, path: str) -> None:
    """The project's YAML in $EDITOR (nano, else vi), the dashboard suspended."""
    editor = os.environ.get("EDITOR") or ("nano" if shutil.which("nano") else "vi")
    with app.suspend():
        subprocess.call([editor, path])


def guide_path() -> Path | None:
    from .settings import PKG
    return next((PKG / n for n in GUIDE_NAMES if (PKG / n).exists()), None)


def cleaned_dates(cfg) -> int:
    """Dates whose coregistered SLCs the phase-linked merge replaced with a
    <date>.cleaned marker. A Whole pipeline run on such a project reuses the
    plan and skips those dates (stack_backend.coregistered_dates: their
    archives are not fetched again); before 5 Oct one re-assembled every
    cleaned date (3 h, ~111 GB on Site B). It still goes through every
    stage and checks the archive first, so the choice warns."""
    d = Path(getattr(cfg, "stack_dir", Path(cfg.project_dir) / "stack")) / "coreg_secondarys"
    try:
        return sum(1 for f in d.iterdir() if f.suffix == ".cleaned" and f.stem.isdigit() and len(f.stem) == 8)
    except OSError:
        return 0


# ---------------------------------------------------------------- run timing
def run_end(rec: dict | None, st: str, res: dict | None, L: dict, now: float | None = None) -> float | None:
    """When the run on record ended: now while it runs; else the user's stop,
    the run's own run_result.json, or its last progress event. The header used
    to count (stopped or now) - started, and run.json has 'stopped' only after
    a user stop, so Site B's 3.5 h run read 146 h a week later."""
    if not rec or not rec.get("started"):
        return None
    now = time.time() if now is None else now
    if st == "running":
        return now
    since = rec["started"] - 1
    if rec.get("stopped"):
        return float(rec["stopped"])
    if res and res.get("finished", 0) >= since and res.get("pid") in (None, rec.get("pid")):
        return float(res["finished"])
    mine = [e["t"] for e in L.values() if e.get("t", 0) >= since]
    return max(mine) if mine else float(rec["started"])


def warnings_line(res: dict | None) -> str | None:
    """'N warnings, M distinct - see the log' from run_result.json's
    'warnings' (pipeline.WarningCounter; 16 Oct 2026, readiness review
    ERR-02: a run that logged 860 WARNINGs read 'done' with nothing said),
    or None when the run logged none or predates the count."""
    w = (res or {}).get("warnings")
    if not isinstance(w, dict) or not w.get("count"):
        return None
    n, m, err = int(w["count"]), int(w.get("distinct") or 0), int(w.get("errors") or 0)
    return (f"{n} warning{'' if n == 1 else 's'}" + (f" ({err} error{'' if err == 1 else 's'})" if err else "")
            + f", {m}{'+' if w.get('distinct_capped') else ''} distinct - see the log")


def result_for_run(rec: dict | None, res: dict | None) -> dict | None:
    """run_result.json when it belongs to the run on record (not an older one)."""
    if not res:
        return None
    if not rec:
        return res
    if res.get("finished", 0) >= rec.get("started", 0) - 1 and res.get("pid") in (None, rec.get("pid")):
        return res
    return None


def stage_rows(L: dict, rec: dict | None, st: str, now: float | None = None) -> list[dict]:
    """The stage table, one dict per stage: {key, icon, style, label, done,
    total, elapsed, remaining, note}. A row from before this run is said to
    be one ('earlier run'), not just dimmed beside the same ✗; a failure the
    run carried on past (2 of 804 misreg pairs) is marked tolerated; a stage
    resumed over verified work says how many items were verified from the
    last run; elapsed stops at the stage's last event once the run is not
    running, and Remaining shows only while it runs."""
    now = time.time() if now is None else now
    since = (rec or {}).get("started", 0) - 1
    running = st == "running"
    mine = sorted((e for k, e in L.items() if k != "run" and e.get("t", 0) >= since), key=lambda e: e["t"])
    out = []
    for key, ev in L.items():
        if key == "run":
            continue
        stale = ev["t"] < since
        status = ev.get("status", "")
        style = "dim" if stale else {"done": "green", "failed": "bold red", "skipped": "dim",
                                     "running": "cyan", "start": "cyan"}.get(status, "")
        icon = STATUS_ICON.get(status, "·")
        msg = (ev.get("msg") or "").strip()
        notes = []
        if stale:
            notes.append("earlier run")
        if status == "failed" and not stale and key != "pipeline" \
                and any(e["t"] > ev["t"] and e.get("status") == "done" and e.get("stage") != key for e in mine):
            icon, style = "!", "yellow"
            notes.append("the run carried on past this (tolerated)")
        if ev.get("skipped") and not stale:
            notes.append(f"{int(ev['skipped'])} verified from the last run")
        if not stale and status in ("running", "start") and st in ("interrupted", "failed", "stopped"):
            # the run is gone but this stage never finished: not a live ▶ in
            # running cyan (ops-9: a dead run's unwrap at 400/804 kept both)
            icon, style = "■", "yellow"
            notes.insert(0, "interrupted" if st == "interrupted" else f"interrupted (run {st})")
        if msg:
            notes.append(msg)
        live = running and not stale and status in ("running", "start")
        end = now if live else ev["t"]
        done, total = ev.get("done"), ev.get("total")
        out.append({"key": key, "icon": icon, "style": style, "label": stage_label(key), "done": done,
                    "total": total, "elapsed": fmt_dur(end - ev.get("started", ev["t"])),
                    "remaining": fmt_dur(progress.eta_seconds(ev, now=now)) if live else "",
                    "note": " · ".join(notes)[:160], "live": live})
    return out


def sublog_of(L: dict, rec: dict | None, st: str) -> Path | None:
    """The per-step log a running stage names in its progress message
    ('log: <path>', MiaplPy's steps): during phase linking the run's own log
    is silent for hours while that file moves."""
    if st != "running":
        return None
    since = (rec or {}).get("started", 0) - 1
    live = [e for k, e in L.items() if e.get("t", 0) >= since and e.get("status") in ("start", "running")
            and str(e.get("msg") or "").startswith("log: ")]
    if not live:
        return None
    p = Path(str(max(live, key=lambda e: e["t"])["msg"])[5:].strip())
    return p if p.exists() else None


# ---------------------------------------------------------------- resources
class Resources:
    """Disk (the real budget: settings.disk_budget_gb - on WSL the Windows
    drive's free space + what the vhdx already holds there that ext4 does not
    use, the figure the planner and preflight use), memory, load and network,
    sampled on each refresh."""

    def __init__(self):
        self._net = None
        self.net_hist: list[float] = [0.0] * 40

    def sample(self, project_dir: str | None = None) -> dict:
        out = {}
        where = project_dir if project_dir and Path(project_dir).exists() else "/"
        here = shutil.disk_usage(where)
        out["used_gb"], out["free_gb"] = here.used / 1e9, here.free / 1e9
        try:
            from . import settings
            b = settings.disk_budget_gb(where)
            out["budget_gb"], out["budget_source"] = float(b["budget"]), str(b.get("source") or "")
            if b.get("c_free") is not None:
                out["c_free_gb"], out["vhdx_gb"] = b["c_free"], b.get("vhdx_gb")
                out["vhdx_headroom_gb"] = b.get("vhdx_headroom_gb")
        except Exception:                                     # noqa: BLE001 - shown without the budget
            pass
        try:
            mi = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()}
            out["mem_used_gb"] = (mi["MemTotal"] - mi["MemAvailable"]) / 1e6
            out["mem_total_gb"] = mi["MemTotal"] / 1e6
        except (OSError, KeyError, ValueError):
            pass
        try:
            out["load"] = float(Path("/proc/loadavg").read_text().split()[0])
            out["cpus"] = os.cpu_count()
        except OSError:
            pass
        try:
            rx = sum(int(l.split()[1]) for l in Path("/proc/net/dev").read_text().splitlines()[2:]
                     if not l.strip().startswith("lo:"))
            now = time.time()
            if self._net:
                rate = (rx - self._net[0]) / max(1e-3, now - self._net[1]) / 1e6
                self.net_hist = self.net_hist[1:] + [max(0.0, rate)]
                out["net_mb_s"] = rate
            self._net = (rx, now)
        except (OSError, ValueError, IndexError):
            pass
        return out

    @staticmethod
    def render(r: dict) -> Text:
        t = Text()
        t.append("Disk  ", style="bold")
        t.append(f"{r.get('free_gb', 0):.0f} GB free on this filesystem\n")
        if "budget_gb" in r:
            # the pipeline's own watchdog warns below 60 GB and stops below 20
            style = "green" if r["budget_gb"] > 150 else "yellow" if r["budget_gb"] > 60 else "bold red"
            t.append("      real budget ", style="dim")
            t.append(f"{r['budget_gb']:.0f} GB", style=style)
            if r.get("c_free_gb") is not None and r.get("vhdx_headroom_gb") is not None:
                # P0.2: the vhdx is sparse, so its length is not space on C:
                t.append(f"  (C: {r['c_free_gb']:.0f} GB free + {r['vhdx_headroom_gb']:.0f} GB unused "
                         "in the WSL disk)", style="dim")
            elif r.get("c_free_gb") is not None:
                t.append(f"  (C: {r['c_free_gb']:.0f} GB free, vhdx {r.get('vhdx_gb') or 0:.0f} GB)", style="dim")
            t.append("\n")
        if "mem_used_gb" in r:
            t.append("RAM   ", style="bold")
            t.append(f"{r['mem_used_gb']:.1f} / {r['mem_total_gb']:.0f} GB\n")
        if "load" in r:
            t.append("CPU   ", style="bold")
            t.append(f"load {r['load']:.1f} on {r['cpus']} cores\n")
        if "net_mb_s" in r:
            t.append("Net   ", style="bold")
            t.append(f"{r['net_mb_s']:.1f} MB/s in\n")
        return t


# ---------------------------------------------------------------- modals
class Confirm(ModalScreen[bool]):
    BINDINGS = [("escape", "no", "Cancel")]

    def __init__(self, question: str, yes: str = "Yes"):
        super().__init__()
        self.question, self.yes = question, yes

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.question, id="question", markup=False)
            with Horizontal(id="buttons"):
                yield Button(self.yes, variant="error", id="yes")
                yield Button("Cancel", id="no")

    @on(Button.Pressed, "#yes")
    def action_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def action_no(self) -> None:
        self.dismiss(False)


class StorageConfirm(ModalScreen[dict | None]):
    """The plan of a storage action (storage.keep_results_only /
    delete_project) and its confirmation. need_name: the project's name must
    be typed (it went to a client, or its report is FINAL). archive_choice:
    show the 'keep a small archive' tick (on by default). Returns
    {archive, name} or None; with refused reasons it shows them and only
    closes."""
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, text: str, yes: str, need_name: str | None = None,
                 archive_choice: bool = False, refused: bool = False):
        super().__init__()
        self.title_text, self.text, self.yes = title, text, yes
        self.need_name, self.archive_choice, self.refused = need_name, archive_choice, refused

    def compose(self) -> ComposeResult:
        with Vertical(id="signdialog"):
            with VerticalScroll(id="signbody"):
                yield Static(f"[b]{escape(self.title_text)}[/b]")
                yield Static(self.text, markup=False, id="storage-plan")
                if not self.refused:
                    if self.archive_choice:
                        yield Checkbox("Keep a small archive first (results, report, logs, config) in _archive/",
                                       value=True, id="storage-archive")
                    if self.need_name:
                        yield Static(f"[yellow]This project went to a client or its report is FINAL.[/yellow] "
                                     f"Type its name ([b]{escape(self.need_name)}[/b]) to delete it:")
                        yield Input(placeholder=self.need_name, id="storage-name")
                yield Static(id="storage-error")
            with Horizontal(id="buttons"):
                if not self.refused:
                    yield Button(self.yes, variant="error", id="yes")
                yield Button("Close" if self.refused else "Cancel", id="no")

    @on(Input.Submitted, "#storage-name")
    @on(Button.Pressed, "#yes")
    def confirm(self) -> None:
        name = None
        if self.need_name:
            name = self.query_one("#storage-name", Input).value.strip()
            if name != self.need_name:
                self.query_one("#storage-error", Static).update(
                    f"[red]The name does not match: type {escape(self.need_name)} exactly.[/red]")
                return
        archive = self.query_one("#storage-archive", Checkbox).value if self.archive_choice else True
        self.dismiss({"archive": archive, "name": name})

    @on(Button.Pressed, "#no")
    def action_cancel(self) -> None:
        self.dismiss(None)


class ChooseRun(ModalScreen[str | None]):
    """What to run. The restart that fits the files on disk
    (runs.suggest_kind) comes first as 'Recommended', with its reason; the
    rest sit under 'Advanced'. A 'Whole pipeline' pick after a phase-linked
    crash re-assembled every cleaned date (3 h, ~111 GB on Site B), and
    'Phase linking onward' after a half-finished merge linked a shortened
    stack for good: the list no longer offers them as equals."""
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, backend: str, recommended: tuple | None = None, cleaned: int = 0):
        super().__init__()
        self.backend = backend
        self.recommended = recommended
        self.cleaned = cleaned

    def kinds(self) -> list[str]:
        out = []
        for k in runs.KINDS:
            if k in ("miaplpy", "reunwrap") and self.backend != "miaplpy":
                continue
            if k in ("mintpy", "plan") and self.backend == "disp":
                continue
            if k in ("crosscheck", "cross_check", "injection_test") and self.backend == "disp":
                continue
            out.append(k)
        return out

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("[b]What to run[/b]  (it runs detached: closing the dashboard never stops it)")
            ol = OptionList(id="kinds")
            rec = self.recommended[0] if self.recommended and self.recommended[0] in runs.KINDS else None
            if rec:
                t = Text("Recommended: ", style="bold green")
                t.append(runs.KINDS[rec][0])
                t.append(f"\n  {self.recommended[1]}", style="dim")
                ol.add_option(Option(t, id=rec))
                ol.add_option(Option(Text("Advanced: only when you know why the recommendation does not fit",
                                          style="dim italic"), id="_advanced", disabled=True))
            for k in self.kinds():
                # Resume works out the same restart as the recommendation, at start time
                if k == rec or (rec and k == "resume"):
                    continue
                label = runs.KINDS[k][0]
                if k == "full" and self.cleaned:
                    label += f"  (! {self.cleaned} dates cleaned after coregistration)"
                ol.add_option(Option(("    " if rec else "") + label, id=k))
            yield ol
            if self.cleaned:
                yield Static(f"[yellow]![/yellow] Whole pipeline goes through every stage again. It reuses this "
                             f"project's plan and skips the {self.cleaned} dates already coregistered and cleaned "
                             "after the merge, and commands already finished; but it checks the archive first and "
                             "refuses to start if acquisitions are missing there, unless allow_missing_dates lists "
                             "them. Use the recommendation unless you mean that.", id="cleaned-note")
            yield Button("Cancel", id="cancel")

    @on(OptionList.OptionSelected)
    def chosen(self, ev: OptionList.OptionSelected) -> None:
        if ev.option.id and not ev.option.id.startswith("_"):
            self.dismiss(ev.option.id)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class HelpScreen(ModalScreen[None]):
    """`?`: the user guide (USER_GUIDE.md beside the package), or the keys
    when the guide is not there."""
    BINDINGS = [("escape", "close", "Close"), ("q", "close", "Close"), ("question_mark", "close", "Close")]

    KEYS = """**Keys**

- Projects: Enter opens the live view, b the map builder, m the results map, n a new area, v results, f files,
  g storage (every project's size; k keep results only, d delete a project)
- Live view: s start (the recommended restart first; processing runs go through the preflight), x stop,
  p preflight, e edit the config, v results, o open the report, r sign off, k build the client package, f files
- Esc goes back, q quits; runs keep going when the dashboard closes
"""

    def compose(self) -> ComposeResult:
        g = guide_path()
        with Vertical(id="helpdialog"):
            with VerticalScroll(id="helpbody"):
                if g is None:
                    yield Static("[yellow]User guide not found[/yellow] (USER_GUIDE.md next to the software). "
                                 "The keys:", id="guide-missing")
                    yield Markdown(self.KEYS)
                else:
                    try:
                        text = g.read_text(errors="replace")
                    except OSError as exc:
                        text = f"User guide not readable ({exc}).\n\n" + self.KEYS
                    yield Static(f"[dim]{g}[/dim]", id="guide-path")
                    yield Markdown(text, id="guide")
            yield Button("Close", id="close")

    @on(Button.Pressed, "#close")
    def action_close(self) -> None:
        self.dismiss(None)


# The release gate's hard blockers a sign-off itself settles (it ticks the
# features, records the reviewer, binds to these numbers and names the
# specialist of a held report). Every other hard blocker is still there after
# signing, and the sign-off dialogs (here and app_server's) say so first.
SIGNOFF_CLEARS = ("feature_unchecked", "not_signed_off", "signoff_stale", "qa_send_rule")


def signoff_hold(blockers: list) -> str:
    """The gate's hard 'hold for specialist' line, or '' (16 Oct 2026: the
    hold is hard until a named specialist releases it at sign-off)."""
    return next((b["text"] for b in blockers if b.get("code") == "qa_send_rule" and b.get("hard")), "")


def signoff_leaves(blockers: list) -> list[str]:
    """The hard blockers that still stop a final release whatever is signed."""
    return [b["text"] for b in blockers if b.get("hard") and b.get("code") not in SIGNOFF_CLEARS]


class SignOffScreen(ModalScreen[dict | None]):
    """Record a reviewer's sign-off of the current client report
    (release.sign_off): the reviewer's name and a tick for every feature the
    report shows (checked against imagery, dates, the ground), and, for a
    report held for a specialist, the specialist's name. Returns
    {reviewer, notes, features, specialist} or None."""
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, project_dir, rep: dict, blockers: list | None = None):
        super().__init__()
        self.project_dir = Path(project_dir)
        self.rep = rep
        self.blockers = blockers or []

    def _features(self) -> list[dict]:
        try:
            from . import release
            return release.features(self.rep)
        except Exception:                                     # noqa: BLE001
            return []

    def _held(self) -> str:
        return signoff_hold(self.blockers)

    def compose(self) -> ComposeResult:
        try:
            from .settings import identity
            who = identity().get("preparer", "")
        except Exception:                                     # noqa: BLE001
            who = ""
        feats = self._features()
        with Vertical(id="signdialog"):
            with VerticalScroll(id="signbody"):
                yield Static("[b]Sign off this client report[/b]\nYour name goes on the report; the sign-off "
                             "binds to these numbers, so any rebuild that changes one voids it.")
                other = signoff_leaves(self.blockers)
                if other:
                    yield Static("[yellow]Still blocking a final release whatever you sign:[/yellow]\n"
                                 + "\n".join(f"  • {x}" for x in other[:8]), markup=True, id="sign-blockers")
                yield Input(value=who, placeholder="reviewer's name", id="reviewer")
                held = self._held()
                if held:
                    yield Static("[yellow]Held for a specialist.[/yellow] " + escape(held)
                                 + "\nThe specialist who read these reasons types their name here (it may be you, "
                                   "if you are that specialist).", markup=True, id="sign-hold")
                    yield Input(placeholder="specialist's name", id="specialist")
                if feats:
                    yield Static("Tick each feature once you have checked it (imagery, dates, the ground):")
                    for f in feats:
                        yield Checkbox(f["label"], value=False, id=f"feat{f['n']}")
                else:
                    yield Static("[dim]The report shows no features to check.[/dim]")
                yield Input(placeholder="notes (optional)", id="notes")
                yield Static(id="sign-error")
            with Horizontal(id="buttons"):
                yield Button("Sign off, rebuild the report and package", variant="success", id="sign")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#sign")
    def sign(self) -> None:
        who = self.query_one("#reviewer", Input).value.strip()
        feats = self._features()
        ticked = [f["n"] for f in feats if self.query_one(f"#feat{f['n']}", Checkbox).value]
        err = self.query_one("#sign-error", Static)
        if not who:
            err.update("[red]Type the reviewer's name.[/red]")
            return
        if len(ticked) != len(feats):
            err.update(f"[red]Every feature must be checked before signing off ({len(ticked)} of "
                       f"{len(feats)} ticked).[/red]")
            return
        specialist = self.query_one("#specialist", Input).value.strip() if self._held() else ""
        if self._held() and not specialist:
            err.update("[red]This report is held for a specialist: type the specialist's name.[/red]")
            return
        self.dismiss({"reviewer": who, "notes": self.query_one("#notes", Input).value.strip(), "features": ticked,
                      **({"specialist": specialist} if specialist else {})})

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------- release actions
REBUILD_CODE = ("import logging, sys\n"
                "logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')\n"
                "from insar_pipeline.config import load_config\n"
                "from insar_pipeline import pipeline\n"
                "sys.exit(0 if pipeline.run_client_report(load_config(sys.argv[1])) else 1)\n")


def rebuild_report(config: str, project_dir) -> tuple[bool, str]:
    """Rebuild the GNSS tie, the client report and KMZ and the provenance
    stages (pipeline.run_client_report) in a child process under the project
    lock, so no run starts on the project meanwhile and the dashboard's
    memory stays its own. (ok, log path). Raises RuntimeError when a run
    holds the project."""
    P = Path(project_dir)
    logs = P / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"report_{time.strftime('%Y%m%d_%H%M%S')}.log"
    env = {**os.environ, "PYTHONPATH": str(runs.PARENT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "PYTHONUNBUFFERED": "1"}
    with runs.project_lock(P, kind="report", config=str(config)):
        with open(log, "a") as fh:
            runs.write_log_label(fh)     # the log's first line names the code (ERR-04, 16 Oct 2026)
            p = subprocess.run([runs.PY, "-c", REBUILD_CODE, str(config)], stdin=subprocess.DEVNULL, stdout=fh,
                               stderr=subprocess.STDOUT, cwd=str(runs.PARENT), env=env)
    return p.returncode == 0, str(log)


def build_package(cfg) -> tuple[Path | None, str]:
    """package.build once the release gate says final, then the outcome read
    back: the release folder with its MANIFEST.json. (path or None, message)."""
    from . import package, release
    if runs.state(cfg.project_dir) == "running":
        # a run rewrites the GeoTIFFs and points files while the signed report
        # still reads final: the package would mix the two
        return None, "A run is going on this project; build the client package when it ends."
    st = release.status(cfg)
    if not st.get("final"):
        hard = [b["text"] for b in st.get("blockers", []) if b.get("hard")]
        return None, "Not final, so no client package: " + ("; ".join(hard[:4]) or "see release.status")
    out = Path(package.build(cfg))
    if not (out / "MANIFEST.json").exists():
        return None, f"package.build returned {out} but it holds no MANIFEST.json"
    return out, f"Client package: {out}"


def sign_off_and_release(cfg, config: str, data: dict, say=print, rebuild=rebuild_report) -> dict:
    """The Sign off action, each step checked by its outcome:
    release.sign_off (refuses a partial acknowledgement, reads the record back
    bound to the numbers) -> rebuild the report (it then renders without the
    DRAFT mark) -> release.status -> package.build only when final.
    {signed, rebuilt, final, package, message}."""
    from . import release
    out = {"signed": None, "rebuilt": None, "final": False, "package": None, "message": ""}
    P = Path(cfg.project_dir)
    out["signed"] = str(release.sign_off(P, data["reviewer"], data.get("notes", ""), data.get("features") or (),
                                         specialist=data.get("specialist", "")))
    say(f"Signed off by {data['reviewer']}"
        + (f" (released for the specialist hold by {data['specialist']})" if data.get("specialist") else "")
        + "; rebuilding the client report…")
    ok, log = rebuild(config, P)
    out["rebuilt"] = ok
    if not ok:
        out["message"] = f"The sign-off is recorded, but the report rebuild failed: see {log}"
        return out
    st = release.status(cfg)
    out["final"] = bool(st.get("final"))
    if not out["final"]:
        hard = [b["text"] for b in st.get("blockers", []) if b.get("hard")]
        out["message"] = ("Signed off, but the report is still a DRAFT: " + "; ".join(hard[:4])
                          + (" (the rebuild changed the numbers: review and sign again)"
                             if any(b.get("code") == "signoff_stale" for b in st.get("blockers", [])) else ""))
        return out
    pkg, msg = build_package(cfg)
    out["package"] = str(pkg) if pkg else None
    out["message"] = msg
    return out


# ---------------------------------------------------------------- projects
class ProjectsScreen(Screen):
    BINDINGS = [Binding("enter", "open", "Open", show=True), ("b", "builder", "Map builder"), ("n", "new", "New area"),
                ("v", "results", "Results"), ("m", "viewer", "Results map"), ("f", "files", "Files"), ("g", "storage", "Storage"), ("r", "refresh", "Refresh"),
                ("q", "app.quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        from . import settings
        yield Static(f"[b]Projects[/b]  [dim]every config the pipeline knows; Enter opens the live view, "
                     f"b opens the map builder, m the results map, n a new area here, f browses files, g storage (free disk), "
                     f"? the guide  ·  "
                     f"projects folder {settings.projects_root()}[/dim]", id="title")
        yield DataTable(id="projects", cursor_type="row", zebra_stripes=True)
        yield Static(id="detail")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one(DataTable)
        t.add_columns("Project", "Route", "State", "Verdict", "Area results (vertical, per client report)", "Updated")
        self.action_refresh()
        self.set_interval(5, self.action_refresh)

    def action_refresh(self) -> None:
        self._load()

    @work(thread=True, exclusive=True, group="projects")
    def _load(self) -> None:
        rows = []
        for c in runs.known_configs():
            p = runs.project_summary(c)
            rows.append((p, project_row(p)))
        self.app.call_from_thread(self._fill, rows)

    def _fill(self, rows) -> None:
        t = self.query_one(DataTable)
        keep = t.cursor_row
        t.clear()
        for p, cells in rows:
            upd = datetime.fromtimestamp(p.updated).strftime("%d %b %H:%M") if p.updated else ""
            t.add_row(Text(p.name, style="bold"), areas.ROUTE_NAMES.get(p.backend, p.backend),
                      Text(p.state, style=STATE_STYLE.get(p.state, "")),
                      cells["verdict"], cells["results"], upd, key=p.config)
        if rows:
            t.move_cursor(row=min(keep, len(rows) - 1))

    def action_open(self) -> None:
        t = self.query_one(DataTable)
        if t.row_count:
            key = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
            self.app.push_screen(project_screen(key))

    @on(DataTable.RowSelected)
    def _selected(self, ev: DataTable.RowSelected) -> None:
        self.app.push_screen(project_screen(ev.row_key.value))

    def action_new(self) -> None:
        self.app.push_screen(NewAreaScreen())

    def action_builder(self) -> None:
        # draw/upload the area and mines on a map; the project it creates shows up here
        from . import map_builder
        url = map_builder.serve()
        map_builder.open_in_browser(url)
        self.notify(f"Map builder at {url}. Projects it creates appear here within seconds.")

    def action_viewer(self) -> None:
        # every project's measurement points on a map: click a point, draw a polygon, export it
        from . import map_builder, results_viewer
        url = results_viewer.serve()
        map_builder.open_in_browser(url)
        self.notify(f"Results map at {url}. A project appears there once its client KMZ has been built.")

    def action_results(self) -> None:
        t = self.query_one(DataTable)
        if not t.row_count:
            return
        cfgp = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
        from .config import load_config
        try:
            ok = (Path(load_config(cfgp).project_dir) / "deliverables" / "client_report.json").exists()
        except Exception:                                    # noqa: BLE001
            ok = False
        if not ok:
            self.notify("That project has no finished results yet.", severity="warning")
            return
        from .results_screen import ResultsScreen
        self.app.push_screen(ResultsScreen(cfgp))

    def action_files(self) -> None:
        from .files_screen import FilesScreen
        self.app.push_screen(FilesScreen())

    def action_storage(self) -> None:
        self.app.push_screen(StorageScreen())


# ---------------------------------------------------------------- storage
class StorageScreen(Screen):
    """`g` on the projects screen: every project on this machine with its
    size, state, report and whether it went to a client, what 'keep results
    only' would free, and the real disk budget (storage.inventory). k keeps
    the results only (cleanup levels 1-3 on a finished project), d deletes the
    whole project after a small archive. Each shows its plan first; the work
    runs in a background thread, so the dashboard never freezes on a long
    delete."""
    BINDINGS = [("k", "keep", "Keep results only"), ("d", "delete", "Delete project"),
                ("r", "refresh", "Refresh"), ("escape", "back", "Back")]

    def __init__(self):
        super().__init__()
        self.info: dict = {}
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("[b]Storage[/b]  [dim]measuring every project...[/dim]", id="stitle")
        yield DataTable(id="storage", cursor_type="row", zebra_stripes=True)
        yield Static(id="sdetail")
        yield Static(id="sprog")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one(DataTable)
        t.add_columns("Project", "Size", "Status", "Last run", "Report", "Delivered?", "Keep results only frees")
        self.action_refresh()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        self._load()

    @work(thread=True, exclusive=True, group="storage")
    def _load(self) -> None:
        from . import storage
        try:
            inv = storage.inventory()
        except Exception as exc:                              # noqa: BLE001
            self.app.call_from_thread(self.notify, f"Could not measure the projects: {exc}", severity="error")
            return
        self.app.call_from_thread(self._fill, inv)

    def _fill(self, inv: dict) -> None:
        from . import storage
        b = inv.get("budget") or {}
        free = f"{b['budget']:.1f} GB" if b.get("budget") is not None else "not measured"
        total = sum(p.size or 0 for p in inv["projects"])
        self.query_one("#stitle", Static).update(
            f"[b]Storage[/b]  free disk (the real budget): [b]{free}[/b]  ·  {len(inv['projects'])} projects, "
            f"{storage.gb(total)}  ·  [dim]k keep results only, d delete project, r refresh, Esc back  ·  "
            f"{escape(inv['root'])}[/dim]")
        t = self.query_one(DataTable)
        keep = t.cursor_row
        t.clear()
        self.info = {}
        for p in inv["projects"]:
            self.info[p.name] = p
            last = " ".join(x for x in (p.last_kind, p.last_status,
                                        datetime.fromtimestamp(p.last_finished).strftime("%d %b %H:%M")
                                        if p.last_finished else "") if x) or "-"
            t.add_row(Text(p.name, style="bold"), storage.gb(p.size),
                      Text(storage.status_text(p), style=STATE_STYLE.get(p.state, "")), last,
                      Text(p.release or "-", style="green" if p.release == "FINAL" else ""),
                      Text(storage.delivered_text(p), style="bold yellow" if p.delivered else ""),
                      storage.reclaim_text(p), key=p.name)
        if inv["projects"]:
            t.move_cursor(row=min(max(keep, 0), len(inv["projects"]) - 1))
        self._detail()

    def _selected(self):
        t = self.query_one(DataTable)
        if not t.row_count:
            return None
        return self.info.get(t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value)

    @on(DataTable.RowHighlighted)
    def _highlighted(self, _ev) -> None:
        self._detail()

    def _detail(self) -> None:
        from . import storage
        p = self._selected()
        d = self.query_one("#sdetail", Static)
        if p is None:
            d.update("")
            return
        t = Text()
        t.append(f"{p.path}\n", style="bold")
        t.append(f"config: {p.config or 'none known'}\n", style="dim")
        if p.folders:
            t.append("  ".join(f"{k} {storage.gb(v)}" for k, v in list(p.folders.items())[:8]) + "\n")
        if p.reclaimable is None and p.reclaimable_why:
            t.append(f"Keep results only is not available: {p.reclaimable_why}.\n", style="yellow")
        if p.error:
            t.append(f"Could not read everything: {p.error}\n", style="red")
        d.update(t)

    def _start(self) -> str | None:
        if self.busy:
            self.notify("A storage action is still going; wait for it to finish.", severity="warning")
            return None
        p = self._selected()
        return p.name if p else None

    # ---- keep results only
    def action_keep(self) -> None:
        name = self._start()
        if name:
            self._plan_keep(name)

    @work(thread=True, exclusive=True, group="storage-plan")
    def _plan_keep(self, name: str) -> None:
        from . import storage
        try:
            plan = storage.keep_results_only(name)
        except Exception as exc:                              # noqa: BLE001
            self.app.call_from_thread(self.notify, f"No plan: {exc}", severity="error")
            return
        self.app.call_from_thread(self._ask_keep, plan)

    def _ask_keep(self, plan: dict) -> None:
        from . import storage
        refused = bool(plan["refused"]) or not plan["items"]
        text = storage.keep_plan_text(plan)

        def done(ans) -> None:
            if ans:
                self._run(f"Keeping the results of {plan['project']}",
                          lambda cb: storage.keep_results_only(plan["project"], apply=True))
        self.app.push_screen(StorageConfirm(f"Keep results only: {plan['project']}", text,
                                            f"Delete {storage.gb(plan['bytes'])} of processing files",
                                            refused=refused), done)

    # ---- delete
    def action_delete(self) -> None:
        name = self._start()
        if name:
            self._plan_delete(name)

    @work(thread=True, exclusive=True, group="storage-plan")
    def _plan_delete(self, name: str) -> None:
        from . import storage
        try:
            plan = storage.delete_project(name, archive=True)
        except Exception as exc:                              # noqa: BLE001
            self.app.call_from_thread(self.notify, f"No plan: {exc}", severity="error")
            return
        self.app.call_from_thread(self._ask_delete, plan)

    def _ask_delete(self, plan: dict) -> None:
        from . import storage
        text = storage.delete_plan_text(plan)

        def done(ans) -> None:
            if ans:
                self._run(f"Deleting {plan['project']}",
                          lambda cb: storage.delete_project(plan["project"], archive=ans["archive"], apply=True,
                                                            confirm_name=ans["name"], progress=cb))
        self.app.push_screen(StorageConfirm(f"Delete project: {plan['project']}", text,
                                            f"Delete {plan['project']} ({storage.gb(plan['bytes'])})",
                                            need_name=plan["project"] if plan["needs_name"] else None,
                                            archive_choice=True, refused=bool(plan["refused"])), done)

    # ---- the work, off the UI thread
    def _progress(self, text: str) -> None:
        try:
            self.query_one("#sprog", Static).update(Text(text, style="bold cyan"))
        except Exception:                                     # noqa: BLE001 - the screen was closed
            pass

    def _run(self, what: str, fn) -> None:
        self.busy = True
        self._progress(f"{what}...")
        app = self.app

        def cb(done, total, text) -> None:
            app.call_from_thread(self._progress, f"{what}: {text} ({done}/{total})")

        def work_() -> None:
            try:
                plan = fn(cb)
            except Exception as exc:                          # noqa: BLE001
                app.call_from_thread(self._finished, what, None, str(exc))
                return
            app.call_from_thread(self._finished, what, plan, "")
        app.run_worker(work_, thread=True, group="storage-act", exclusive=False)

    def _finished(self, what: str, plan: dict | None, err: str) -> None:
        from . import storage
        self.busy = False
        if err or plan is None:
            msg, sev = f"{what} failed: {err}", "error"
        elif plan.get("refused"):
            msg, sev = f"{what}: not done. " + " ".join(plan["refused"]), "error"
        elif "freed" in plan:
            msg = f"{what}: freed {storage.gb(plan['freed'])}."
            if plan.get("failed"):
                msg += f" {len(plan['failed'])} items could not be deleted (see the cleanup docs)."
            sev = "warning" if plan.get("failed") else "information"
        else:
            msg = f"{what}: done, {storage.gb(plan['bytes'])} freed." + (
                f" Archive in {plan['archive_dir']}." if plan.get("archived") else "")
            sev = "information"
        self.app.notify(msg, severity=sev, timeout=12)
        self._progress(msg)
        try:
            self.action_refresh()
        except Exception:                                     # noqa: BLE001 - the screen was closed
            pass


# ---------------------------------------------------------------- run view
def project_screen(config: str) -> Screen:
    """The screen for a project in the list: a combined project (combine.py)
    has its own, every other project the run screen."""
    from . import combine
    return CombinedScreen(config) if combine.is_combined_config(config) else RunScreen(config)


def combined_text(config: str) -> Text:
    """A combined project's page in the terminal: its sources, whether it is
    current, each area's reading (combine.brief, as the app shows them)."""
    from . import combine
    cc = combine.load(config)
    rec = combine.read_report(config)
    why = combine.stale_reason(config)
    t = Text()
    t.append(f" {Path(config).stem} ", style="bold reverse")
    t.append("  combined ascending + descending\n\n", style="dim")
    t.append(f"Ascending:  {Path(cc.ascending).stem}\nDescending: {Path(cc.descending).stem}\n", style="bold")
    t.append("It reads both projects and never changes them.\n\n", style="dim")
    if rec is None:
        t.append("Not combined yet: c combines.\n", style="bold yellow")
        return t
    if why:
        t.append(f"Out of date: {why}. c combines again.\n\n", style="bold yellow")
    g = (rec.get("comparison_ground") or {}).get("words") or "the surrounding ground"
    t.append(f"Vertical from both passes together, relative to {g}"
             + (", GNSS-anchored" if rec.get("anchored_to_gnss") else "") + "; likely range is 95%.\n"
             "East-west is indicative (not a tested figure) until its re-test is signed off.\n\n", style="dim")
    for a, b in zip(rec.get("areas") or [], combine.brief(rec)):
        t.append(f"{b['name']}\n", style="bold")
        t.append(f"  {b['rate']}\n")
        if b["range"]:
            t.append(f"  likely range {b['range']}\n", style="dim")
        if b["total"]:
            t.append(f"  {b['total']}\n", style="dim")
        t.append(f"  {b['reading']}\n", style="yellow" if b["tone"] == "warn" else "green" if b["tone"] == "ok" else "dim")
        if a.get("note"):
            t.append(f"  {a['note']}\n", style="dim")
    t.append(f"\nCombined {rec.get('generated', '')}", style="dim")
    return t


class CombinedScreen(Screen):
    """A combined project: what it reads, its readings, and c to combine."""
    BINDINGS = [("c", "combine", "Combine"), ("escape", "back", "Back"), ("q", "app.quit", "Quit")]

    def __init__(self, config: str):
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll():
            yield Static(id="cbody")
        yield Footer()

    def on_mount(self) -> None:
        self.show()

    def show(self) -> None:
        try:
            self.query_one("#cbody", Static).update(combined_text(self.config))
        except Exception as exc:                             # noqa: BLE001
            self.query_one("#cbody", Static).update(Text(f"config does not load: {exc}", style="red"))

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_combine(self) -> None:
        self.app.push_screen(Confirm("Read both projects and build the combined view? Neither project is changed. "
                                     "It takes a minute or two.", "Combine"),
                             lambda yes: self._combine() if yes else None)

    @work(thread=True, exclusive=True, group="combine")
    def _combine(self) -> None:
        from . import combine
        self.app.call_from_thread(self.notify, "Combining…")
        try:
            combine.run(self.config)
            self.app.call_from_thread(self.notify, "Combined.")
        except Exception as exc:                             # noqa: BLE001
            self.app.call_from_thread(self.notify, str(exc), severity="error")
        self.app.call_from_thread(self.show)


class RunScreen(Screen):
    BINDINGS = [("s", "start", "Start…"), ("x", "stop", "Stop"), ("p", "preflight", "Preflight"),
                ("e", "edit", "Edit config"), ("v", "results", "Results"), ("o", "report", "Open report"),
                ("r", "signoff", "Sign off"), ("k", "package", "Package"), ("f", "folder", "Files"),
                ("c", "combine", "Combine"), ("escape", "back", "Back"), ("q", "app.quit", "Quit")]

    def __init__(self, config: str):
        super().__init__()
        self.config = config
        self.cfg = None
        self.res = Resources()
        self.log_path: Path | None = None
        self.log_pos = 0
        self.tick = 0
        self.others: list = []
        self.busy = ""                       # a sign-off / package in progress
        self.combine_status = None           # combine.status_for, looked up once (it reads every config)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="runtitle")
        with Horizontal(id="runbody"):
            with Vertical(id="left"):
                yield DataTable(id="stages", cursor_type="none", zebra_stripes=True)
                yield RichLog(id="log", max_lines=2000, wrap=False, markup=False, highlight=True)
            with Vertical(id="right"):
                yield Static(id="resources")
                yield Label("download MB/s", classes="dim")
                yield Sparkline([0.0], id="net")
                yield Static(id="results")
        yield Footer()

    def on_mount(self) -> None:
        from .config import load_config
        try:
            self.cfg = load_config(self.config)
        except Exception as exc:                             # noqa: BLE001
            self.query_one("#runtitle", Static).update(Text(f"config does not load: {exc}", style="red"))
            return
        self.query_one("#stages", DataTable).add_columns("", "Stage", "Progress", "Done", "Elapsed", "Remaining", "Note")
        self.refresh_view()
        self.set_interval(1.0, self.refresh_view)
        self._combine_lookup()

    @work(thread=True, exclusive=True, group="combine-status")
    def _combine_lookup(self) -> None:
        from . import combine
        try:
            self.combine_status = combine.status_for(self.cfg, fresh=True)
        except Exception:                                     # noqa: BLE001 - the screen works without it
            self.combine_status = None

    def action_combine(self) -> None:
        cs = self.combine_status
        if self.cfg is None or not cs:
            self.notify("No finished project from the other pass covers these report areas.", severity="warning")
            return
        if cs["kind"] == "combined":
            self.app.push_screen(CombinedScreen(cs["config"]))
            return
        self.app.push_screen(Confirm(f"Combine with {cs['name']} ({cs['direction']} pass)? It makes a new combined "
                                     "project; neither project is changed. It takes a minute or two.", "Combine"),
                             lambda yes: self._combine_with(cs["config"]) if yes else None)

    @work(thread=True, exclusive=True, group="combine")
    def _combine_with(self, partner: str) -> None:
        from . import combine
        self.app.call_from_thread(self.notify, "Combining…")
        try:
            path = combine.combine_with(self.config, partner)
        except Exception as exc:                             # noqa: BLE001
            self.app.call_from_thread(self.notify, str(exc), severity="error")
            return
        self._combine_lookup()
        self.app.call_from_thread(self.app.push_screen, CombinedScreen(str(path)))

    # ---- periodic refresh
    def state_words(self, P: Path, st: str) -> str:
        if st == "running":
            held = runs.lock_holder(P) or {}
            if held.get("kind") in LOCK_KIND_WORDS:
                return LOCK_KIND_WORDS[held["kind"]]
        return st

    def refresh_view(self) -> None:
        if self.cfg is None:
            return
        self.tick += 1
        now = time.time()
        P = Path(self.cfg.project_dir)
        rec = runs.record(P)
        st = runs.state(P)
        L = progress.latest(P)
        res = result_for_run(rec, runs.result(P))
        spin = SPIN[self.tick % len(SPIN)] if st == "running" else ""
        head = Text()
        head.append(f" {Path(self.config).stem.replace('config_', '')} ", style="bold reverse")
        head.append(f"  {areas.ROUTE_NAMES.get(self.cfg.backend, self.cfg.backend)}  ", style="dim")
        head.append(f"{spin} {self.state_words(P, st)}", style=STATE_STYLE.get(st, ""))
        if self.busy:
            head.append(f"   {self.busy}", style="bold cyan")
        if rec:
            end = run_end(rec, st, res, L, now)
            head.append(f"   {rec.get('kind')} run, pid {rec.get('pid')}, started "
                        f"{datetime.fromtimestamp(rec['started']):%d %b %H:%M}"
                        f" ({'running ' if st == 'running' else 'took '}{fmt_dur((end or now) - rec['started'])})",
                        style="dim")
        if st == "interrupted":
            try:
                info = runs.interruption(P, self.cfg) or {}
            except Exception:                                 # noqa: BLE001
                info = {}
            if info.get("text"):
                head.append(f"\n  {info['text']}", style="yellow")
        elif res and st in ("done", "failed", "stopped"):
            if res.get("plain_cause"):
                head.append(f"\n  {res['plain_cause']}", style="red" if st == "failed" else "")
            if res.get("next_action"):
                head.append("\n  What now: ", style="bold")
                head.append(str(res["next_action"]))
            wl = warnings_line(res)
            if wl:
                head.append(f"\n  {wl}", style="yellow")
            if st == "failed" and res.get("log"):
                head.append(f"\n  log: {res['log']}", style="dim")
        if self.tick % 5 == 1:
            try:
                self.others = runs.live_runs(exclude=P)
            except Exception:                                 # noqa: BLE001
                self.others = []
        if self.others:
            head.append("\n  ! Also running here: " + "; ".join(
                f"{Path(o['project']).name} ({o.get('kind') or 'run'}, pid {o.get('pid')})" for o in self.others)
                + ". Runs share the cores, memory and disk; each sizes its pools as if alone.", style="yellow")
        if self.cfg.backend == "disp":
            head.append("\n  DISP-S1 screening: quick, not official; it writes no client report unless "
                        "disp_client_report is set.", style="yellow")
        head.append(f"\n  {P}", style="dim")
        self.query_one("#runtitle", Static).update(head)
        self._stages(L, rec, st, now)
        if self.tick % 2 == 1:
            self._sample(str(P))
        if self.tick % 5 == 1:
            self._results(P)
        self._follow_log(rec, sublog_of(L, rec, st))

    @work(thread=True, exclusive=True, group="resources")
    def _sample(self, where: str) -> None:
        r = self.res.sample(where)
        self.app.call_from_thread(self._show_resources, r)

    def _show_resources(self, r: dict) -> None:
        self.query_one("#resources", Static).update(Resources.render(r))
        self.query_one("#net", Sparkline).data = list(self.res.net_hist)

    def _stages(self, L: dict, rec, st: str, now: float) -> None:
        t = self.query_one("#stages", DataTable)
        t.clear()
        if not L:
            t.add_row("", Text("No progress recorded yet: runs started from here (s) report each stage "
                               "as it finishes.", style="dim"), "", "", "", "", "")
        for r in stage_rows(L, rec, st, now):
            icon = SPIN[self.tick % len(SPIN)] if r["live"] and r["icon"] == "▶" else r["icon"]
            t.add_row(Text(icon, style=r["style"]), Text(r["label"], style=r["style"]), bar(r["done"], r["total"]),
                      f"{r['done']}/{r['total']}" if r["total"] else "", r["elapsed"], r["remaining"],
                      Text(r["note"], style="dim"))

    def _results(self, P: Path) -> None:
        w = self.query_one("#results", Static)
        d = _load_json(P / "deliverables" / "client_report.json")
        if not d:
            w.update(Text("\nResults appear here when the client report is built.", style="dim"))
            return
        anch = anchored_of(d)
        t = Text("\nResults ", style="bold")
        sr, stale = send_rule_shown(P)
        t.append(f"({sr or d.get('verdict', '')})", style="bold yellow" if sr == HOLD else "dim")
        if stale:
            t.append(f" {REPORT_OUT_OF_DATE}: rebuild the report", style="bold yellow")
        t.append("\n")
        rs = release_status(self.cfg)
        if rs is not None:
            hard = [b for b in rs.get("blockers", []) if b.get("hard")]
            if rs.get("final"):
                t.append(f"FINAL, reviewed by {rs.get('reviewed_by') or '?'}; k builds the client package\n",
                         style="green")
            else:
                t.append(f"DRAFT: {len(hard)} thing(s) to settle before it goes to a client\n", style="bold yellow")
                for b in hard[:3]:
                    t.append(f"  • {b['text'][:140]}\n", style="yellow")
                t.append("  r signs off once they are settled\n", style="dim")
        t.append("Vertical" + (", GNSS-anchored" if anch else ", relative to farmland (not GNSS-anchored)"
                               if anch is False else "") + "; likely range is 95%\n", style="dim")
        for a in d.get("areas", []):
            if not isinstance(a, dict):
                continue
            v = area_view(a, anch)
            t.append(f"{a.get('name', '?')}\n", style="bold")
            t.append(f"  {v['rate']}")
            tot = total_text(a)                    # with its 95% range (F8)
            if tot:
                t.append(f"   {tot}", style="dim")
            t.append("\n")
            if v["range"]:
                t.append(f"  likely range {v['range'][1:-1]}\n", style="dim")
            elif a.get("untested_range95_mm_yr") and range_text(a["untested_range95_mm_yr"]):
                # an area that could not be compared: the range that can be
                # stated, said so (the page's own wording, E9) - beside its
                # total's range (F8)
                t.append(f"  likely range {range_text(a['untested_range95_mm_yr'])[1:-1]}, not compared with "
                         f"{ground_of(d)}\n", style="dim")
            t.append(f"  {v['reading']}\n", style=v["style"])
        # the report's listed features only: a QA-only stand-out is not an
        # area to check (F2)
        for line, style in feature_lines(d):
            t.append(line + "\n", style=style)
        t.append(f"\nGenerated {d.get('generated', '')}; o opens the report", style="dim")
        cs = getattr(self, "combine_status", None)       # a stand-in screen (tests) has none
        if cs and cs["kind"] == "combined":
            t.append(f"\nAlso measured from the other pass: combined in {Path(cs['config']).stem} ({cs['status']}); "
                     "c opens it", style="bold")
        elif cs:
            t.append(f"\n{cs['name']} looks at the same ground from the {cs['direction']} pass; c combines the two "
                     "into vertical and east-west", style="bold")
        w.update(t)

    def _follow_log(self, rec, sub: Path | None = None) -> None:
        lg = sub or (Path(rec["log"]) if rec and rec.get("log") else None)
        if lg is None:
            # a run started outside the dashboard: its newest log, if any
            logs = sorted((Path(self.cfg.project_dir) / "logs").glob("*.log"), key=lambda f: f.stat().st_mtime)
            lg = logs[-1] if logs else None
        log = self.query_one("#log", RichLog)
        if lg != self.log_path:
            self.log_path, self.log_pos = lg, 0
            log.clear()
            if sub is not None:
                log.write(f"--- following the running step's own log: {sub}")
            if lg and lg.exists():
                for line in runs.tail(lg, 200):
                    log.write(line)
                self.log_pos = lg.stat().st_size
            return
        if lg and lg.exists():
            size = lg.stat().st_size
            if size < self.log_pos:
                self.log_pos = 0
            if size > self.log_pos:
                with open(lg, "rb") as fh:
                    fh.seek(self.log_pos)
                    chunk = fh.read(min(size - self.log_pos, 512_000)).decode(errors="replace")
                self.log_pos += len(chunk.encode(errors="replace"))
                for line in chunk.splitlines()[-300:]:
                    log.write(line)

    # ---- actions
    def action_back(self) -> None:
        self.app.pop_screen()

    def action_start(self) -> None:
        if self.cfg is None:
            return
        P = Path(self.cfg.project_dir)
        if runs.state(P) == "running":
            self.notify("A run is already going here.", severity="warning")
            return
        try:
            rec = runs.suggest_kind(P, self.cfg)
        except Exception:                                     # noqa: BLE001 - the list still works
            rec = None
        n_clean = cleaned_dates(self.cfg)

        def launch(kind):
            if kind in PREFLIGHT_KINDS:
                self.app.push_screen(PreflightScreen(self.config, kind, from_run=True))
                return
            try:
                r = runs.start(self.config, P, kind)
                self.notify(f"Started {kind} (pid {r['pid']}).")
            except Exception as exc:                         # noqa: BLE001
                self.notify(str(exc), severity="error")

        def go(kind):
            if not kind:
                return
            if kind == "full" and n_clean:
                self.app.push_screen(Confirm(
                    f"Whole pipeline goes through every stage again: it reuses this project's plan, skips the "
                    f"{n_clean} dates already coregistered and cleaned after the merge and the commands already "
                    "finished, and checks the archive first - it refuses to start if acquisitions are missing "
                    "there, unless allow_missing_dates in the config lists them. The recommended restart "
                    f"({(rec or ('resume',))[0]}) starts where the work stopped. Start the whole pipeline anyway?",
                    "Whole pipeline"), lambda yes: launch(kind) if yes else None)
                return
            launch(kind)
        self.app.push_screen(ChooseRun(self.cfg.backend, rec, n_clean), go)

    def action_preflight(self) -> None:
        if self.cfg is not None:
            self.app.push_screen(PreflightScreen(self.config, "resume", from_run=True))

    def action_edit(self) -> None:
        if self.cfg is None:
            return
        edit_config(self.app, self.config)
        from .config import load_config
        try:
            self.cfg = load_config(self.config)
            self.notify("Config reloaded.")
        except Exception as exc:                             # noqa: BLE001
            self.notify(f"The config no longer loads: {exc}", severity="error")

    def action_stop(self) -> None:
        if self.cfg is None or runs.state(self.cfg.project_dir) != "running":
            self.notify("Nothing running.")
            return
        held = runs.lock_holder(self.cfg.project_dir)
        if held and held.get("pid") == os.getpid():
            # the "run" is this dashboard's own report rebuild or package;
            # stopping it would signal the dashboard itself
            doing = (self.busy or "rebuilding the report").rstrip("…")
            self.notify(f"No run to stop: the dashboard is {doing}; wait for it to finish.", severity="warning")
            return

        def go(yes):
            if yes:
                self._stop()
        self.app.push_screen(Confirm("Stop the run? The runners tear down their children and leave "
                                     "resumable state; re-starting picks up where it stopped.", "Stop it"), go)

    @work(thread=True)
    def _stop(self) -> None:
        how = runs.stop(self.cfg.project_dir)
        self.app.call_from_thread(self.notify, f"Run {how}.")

    def action_report(self) -> None:
        """Open the client report; a DRAFT (or a QA 'hold for specialist', or
        a report older than the QA) asks first, listing what stands between
        it and a client."""
        if self.cfg is None:
            return
        P = Path(self.cfg.project_dir)
        f = P / "deliverables" / "client_report.html"
        if not f.exists():
            self.notify("No client report yet.", severity="warning")
            return
        st = release_status(self.cfg)
        sr, stale = send_rule_shown(P)
        hard_b = [b for b in (st or {}).get("blockers", []) if b.get("hard")]
        # the gate's own "the report predates the quality assessment" line,
        # when it has one, is the one said - first, so it is never cut off
        # below - and the dialog does not repeat it in its own words
        gate_stale = [b for b in hard_b if b.get("code") == "report_older_than_qa"]
        hard = [b["text"] for b in gate_stale + [b for b in hard_b if b not in gate_stale]]
        if (st is not None and not st.get("final")) or sr == HOLD or stale:
            q = ("This client report is a DRAFT: it is not ready to go to a client.\n\n"
                 + "\n".join(f"• {x}" for x in hard[:8])
                 + (f"\n…and {len(hard) - 8} more" if len(hard) > 8 else "")
                 + (f"\n\nThe quality assessment says '{sr}'." if sr and (sr != "send" or stale) else "")
                 + ("\n\nThe report is older than the quality assessment (report out of date): rebuild the "
                    "report." if stale and not gate_stale else "")
                 + "\n\nOpen it for internal review?")
            self.app.push_screen(Confirm(q, "Open the draft"),
                                 lambda yes: self.notify(f"Opening {open_path(f)}") if yes else None)
            return
        self.notify(f"Opening {open_path(f)}")

    def action_signoff(self) -> None:
        if self.cfg is None:
            return
        P = Path(self.cfg.project_dir)
        rep = _load_json(P / "deliverables" / "client_report.json")
        if not rep:
            self.notify("No client report to sign off yet.", severity="warning")
            return
        if runs.state(P) == "running":
            self.notify("Wait for the run to finish before signing off.", severity="warning")
            return
        st = release_status(self.cfg) or {}

        def go(data):
            if data:
                self._sign(data)
        self.app.push_screen(SignOffScreen(P, rep, st.get("blockers")), go)

    @work(thread=True, exclusive=True, group="release")
    def _sign(self, data: dict) -> None:
        self.busy = "signing off and rebuilding the report…"
        try:
            out = sign_off_and_release(self.cfg, self.config, data,
                                       say=lambda m: self.app.call_from_thread(self.notify, m))
            sev = "information" if out.get("package") else "warning"
            self.app.call_from_thread(self.notify, out["message"], severity=sev, timeout=20)
        except Exception as exc:                             # noqa: BLE001
            self.app.call_from_thread(self.notify, f"Not signed off: {exc}", severity="error", timeout=20)
        finally:
            self.busy = ""
            _RELEASE_CACHE.clear()

    def action_package(self) -> None:
        if self.cfg is None:
            return
        if runs.state(self.cfg.project_dir) == "running":
            self.notify("Wait for the run to finish before building the client package.", severity="warning")
            return
        self._package()

    @work(thread=True, exclusive=True, group="release")
    def _package(self) -> None:
        self.busy = "building the client package…"
        try:
            path, msg = build_package(self.cfg)
            self.app.call_from_thread(self.notify, msg, severity="information" if path else "warning", timeout=20)
        except Exception as exc:                             # noqa: BLE001
            self.app.call_from_thread(self.notify, f"Package not built: {exc}", severity="error", timeout=20)
        finally:
            self.busy = ""

    def action_results(self) -> None:
        if self.cfg is None:
            return
        if not (Path(self.cfg.project_dir) / "deliverables" / "client_report.json").exists():
            self.notify("Results need a finished run (no client report yet).", severity="warning")
            return
        from .results_screen import ResultsScreen
        self.app.push_screen(ResultsScreen(self.config))

    def action_folder(self) -> None:
        if self.cfg is not None:
            from .files_screen import FilesScreen
            self.app.push_screen(FilesScreen(Path(self.cfg.project_dir)))


# ---------------------------------------------------------------- new area
def gnss_text(g: dict) -> Text:
    """The archive check's GNSS lines from areas.gnss_preview: the stations the
    tie will USE (inside the tie's radius, enough years over the dates) apart
    from those nearby it will not. The old text listed every station within
    50 km as if the report were anchored to all of them."""
    t = Text()
    r = g.get("radius_km")
    rows = g.get("stations") or []
    used = [x for x in rows if x.get("will_use")]
    unused = [x for x in rows if x.get("will_use") is False]
    unknown = [x for x in rows if x.get("will_use") is None]
    fmt = lambda xs: ", ".join(f"{x.get('site') or x.get('id')} {float(x.get('distance_km') or 0):.0f} km" for x in xs[:8])  # noqa: E731
    if used:
        t.append(f"GNSS used for the anchor (within {r:.0f} km, enough years over these dates): {fmt(used)}\n")
    if unused:
        t.append(f"GNSS nearby, not used: {fmt(unused)}\n", style="dim")
    if unknown:
        t.append(f"GNSS within {r:.0f} km (spans not checked): {fmt(unknown)}\n")
    if not rows:
        t.append(f"No GNSS station within {r:.0f} km: results stay relative to the area's own farmland.\n",
                 style="yellow")
    if g.get("warning"):
        t.append(f"! {g['warning']}\n", style="yellow")
    return t


def frame_text(row: dict | None, track, direction) -> Text:
    """The DISP-S1 frame for the chosen track (areas.frame_for_track: largest
    share of the processing box on that track), warning below
    disp_backend.MIN_FRAME_OVERLAP. Chosen by granule count alone, the other
    track's frame could win."""
    try:
        from .disp_backend import MIN_FRAME_OVERLAP
    except Exception:                                         # noqa: BLE001
        MIN_FRAME_OVERLAP = 0.95
    if row is None:
        return Text(f"No OPERA DISP-S1 frame on {str(direction).lower()} track {track}: the screening route "
                    "cannot run on this track.\n", style="yellow")
    ov = row.get("overlap_fraction")
    t = Text(f"DISP-S1 frame for track {track}: F{row.get('frame')} ({row.get('granules')} granules"
             + (f", covers {100 * ov:.0f}% of the box" if ov is not None else "") + ")\n")
    if ov is not None and ov < MIN_FRAME_OVERLAP:
        t.append(f"! it covers only {100 * ov:.0f}% of the processing box (under {100 * MIN_FRAME_OVERLAP:.0f}%): "
                 "part of the area would have no data.\n", style="yellow")
    return t


class NewAreaScreen(Screen):
    BINDINGS = [("escape", "back", "Back"), ("ctrl+b", "builder", "Map builder")]
    PAD_KM = 5.0          # areas.write_project's processing margin: the box the frame must cover

    def __init__(self):
        super().__init__()
        self.area: areas.Area | None = None
        self.places: list = []
        self.archive: dict | None = None

    def compose(self) -> ComposeResult:
        # the record ends a week back by default: ERA5 is published ~5 days
        # late, and a record ending inside that lag stops at the troposphere step
        end = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        yield Header(show_clock=True)
        with VerticalScroll(id="newbody"):
            yield Static("[b]1  Where[/b]  [dim]three ways in; each ends in the same outline[/dim]")
            with TabbedContent(id="how"):
                with TabPane("Coordinates", id="t-coords"):
                    yield Static("[dim]'lat, lon' with a radius, two corners 'lat1, lon1, lat2, lon2' "
                                 "(the report-area order), a WKT POLYGON, or GeoJSON[/dim]")
                    with Horizontal(classes="row"):
                        yield Input(placeholder="38.6773, -87.5286", id="coords")
                        yield Input(value="3", placeholder="radius km", id="radius", classes="short")
                        yield Button("Use", id="use-coords", variant="primary")
                with TabPane("Place name", id="t-place"):
                    with Horizontal(classes="row"):
                        yield Input(placeholder="Vincennes, Indiana", id="place")
                        yield Input(value="5", placeholder="radius km", id="place-radius", classes="short")
                        yield Button("Search", id="search", variant="primary")
                    yield OptionList(id="places")
                with TabPane("Draw or import", id="t-poly"):
                    yield Static("[dim]Easiest: the map builder (ctrl+b) - draw or upload the area and the mine "
                                 "shapefiles and it creates the project itself. Or load any GeoJSON / KML / KMZ / "
                                 "shapefile / zipped shapefile / GeoPackage here. "
                                 "Windows paths are fine (paste from Explorer's 'Copy as path').[/dim]")
                    with Horizontal(classes="row"):
                        yield Input(placeholder="/path/to/project_zone.geojson", id="polyfile")
                        yield Button("Load", id="load-poly", variant="primary")
                        yield Button("Open map builder", id="builder")
            yield Static(id="areainfo")
            yield Static("[b]2  What the archive holds[/b]  [dim]the end date defaults to a week ago "
                         "(ERA5 weather data is published ~5 days late)[/dim]")
            with Horizontal(classes="row"):
                yield Input(value="2016-01-01", id="start", classes="short")
                yield Input(value=end, id="end", classes="short")
                yield Button("Check archive + GNSS", id="check", variant="primary")
            yield Static(id="archiveinfo")
            yield Static("[b]3  The project[/b]  [dim]the name heads the area's row in the client report[/dim]")
            with Horizontal(classes="row"):
                yield Input(placeholder="project name, e.g. Gibson SE check", id="name")
                yield Select([(v, k) for k, v in areas.ROUTES.items()], value="miaplpy", id="route", allow_blank=False)
            with Horizontal(classes="row"):
                yield Select([], prompt="track (check the archive first)", id="track")
                yield Input(placeholder="optional: deforming-area polygon file (kept out of the deramp and "
                                        "the farmland comparison)", id="deforming")
            yield Static(id="trackinfo")
            with Horizontal(classes="row"):
                yield Button("Write config and preflight", id="write", variant="success")
            yield Static(id="writeinfo")
        yield Footer()

    # ---- area input
    def _set_area(self, a: areas.Area) -> None:
        self.area = a
        w, s, e, n = a.bbox
        wk, hk = a.size_km()
        self.query_one("#areainfo", Static).update(
            f"[green]✓[/green] [b]{a.label}[/b] ({a.source}): {s:.5f}..{n:.5f} N, {w:.5f}..{e:.5f} E, "
            f"{wk:.1f} × {hk:.1f} km" + ("" if a.is_box else f", {len(a.rings)} polygon(s)"))
        if not self.query_one("#name", Input).value:
            self.query_one("#name", Input).value = a.label.split(",")[0][:40]

    def _err(self, where: str, exc: Exception) -> None:
        self.query_one(where, Static).update(Text(f"✗ {exc}", style="red"))

    @on(Button.Pressed, "#use-coords")
    @on(Input.Submitted, "#coords")
    def use_coords(self) -> None:
        try:
            r = float(self.query_one("#radius", Input).value or 2)
            self._set_area(areas.parse_coordinates(self.query_one("#coords", Input).value, radius_km=r))
        except Exception as exc:                             # noqa: BLE001
            self._err("#areainfo", exc)

    @on(Button.Pressed, "#search")
    @on(Input.Submitted, "#place")
    def search(self) -> None:
        q = self.query_one("#place", Input).value.strip()
        if q:
            self.query_one("#areainfo", Static).update(f"[cyan]searching OpenStreetMap for {q!r}…[/cyan]")
            self._geocode(q)

    @work(thread=True, exclusive=True, group="geo")
    def _geocode(self, q: str) -> None:
        try:
            hits = areas.geocode(q)
            self.app.call_from_thread(self._show_places, hits)
        except Exception as exc:                             # noqa: BLE001
            self.app.call_from_thread(self._err, "#areainfo", exc)

    def _show_places(self, hits) -> None:
        self.places = hits
        ol = self.query_one("#places", OptionList)
        ol.clear_options()
        for i, h in enumerate(hits):
            ol.add_option(Option(f"{h['name']}  [dim]({h['kind']}, {h['lat']:.4f}, {h['lon']:.4f})[/dim]", id=str(i)))
        self.query_one("#areainfo", Static).update(
            f"{len(hits)} place(s) found; pick one." if hits else "[yellow]nothing found[/yellow]")

    @on(OptionList.OptionSelected, "#places")
    def pick_place(self, ev: OptionList.OptionSelected) -> None:
        h = self.places[int(ev.option.id)]
        r = float(self.query_one("#place-radius", Input).value or 0)
        a = areas.parse_coordinates(f"{h['lat']},{h['lon']}", radius_km=r) if r > 0 else h["area"]
        a.label, a.source = h["name"].split(",")[0], "place name"
        self._set_area(a)

    @on(Button.Pressed, "#load-poly")
    @on(Input.Submitted, "#polyfile")
    def load_poly(self) -> None:
        try:
            self._set_area(areas.load_polygon(self.query_one("#polyfile", Input).value.strip()))
        except Exception as exc:                             # noqa: BLE001
            self._err("#areainfo", exc)

    @on(Button.Pressed, "#builder")
    def action_builder(self) -> None:
        # served from localhost: OpenStreetMap refuses pages opened as files
        from . import map_builder
        url = map_builder.serve()
        map_builder.open_in_browser(url)
        self.notify(f"Map builder at {url} (stays up while the dashboard is open)")

    # ---- archive
    @on(Button.Pressed, "#check")
    def check(self) -> None:
        if self.area is None:
            self.notify("Choose the area first.", severity="warning")
            return
        self.query_one("#archiveinfo", Static).update("[cyan]asking ASF, OPERA and the GNSS catalogue…[/cyan]")
        self._check(self.area, self.query_one("#start", Input).value, self.query_one("#end", Input).value)

    @work(thread=True, exclusive=True, group="archive")
    def _check(self, area, start, end) -> None:
        try:
            # the frames' cover is measured on the box the project will process
            r = areas.archive_check(area, start, end, box=area.padded(self.PAD_KM).bbox)
            try:
                r["gnss"] = areas.gnss_preview(area, start, end)
            except Exception as exc:                         # noqa: BLE001
                r["gnss_error"] = str(exc)
            self.app.call_from_thread(self._show_archive, r)
        except Exception as exc:                             # noqa: BLE001
            self.app.call_from_thread(self._err, "#archiveinfo", exc)

    def _show_archive(self, r) -> None:
        self.archive = r
        t = Text()
        if not r["tracks"]:
            t.append("No Sentinel-1 IW scenes over this area in those dates.\n", style="bold red")
        for tr in r["tracks"][:6]:
            yrs = " ".join(f"{y[2:]}:{n}" for y, n in sorted(tr["by_year"].items()))
            good = tr["dates"] >= 60
            t.append(f"{'●' if good else '○'} {tr['direction'].title():10s} track {tr['track']:3d}  "
                     f"{tr['dates']:4d} dates   ", style="green" if good else "dim")
            t.append(yrs + "\n", style="dim")
        detail = r.get("disp_frame_detail") or []
        fr = r.get("disp_frames") or {}
        if detail:
            t.append("OPERA DISP-S1 frames: " + ", ".join(
                f"F{d.get('frame')} ({str(d.get('direction') or '?').title()} {d.get('track')}, "
                f"{100 * (d.get('overlap_fraction') or 0):.0f}% of the box)" for d in detail[:6]) + "\n")
        elif fr and "error" not in fr:
            t.append("OPERA DISP-S1 frames: " + ", ".join(f"F{k} ({v} granules)" for k, v in fr.items()) + "\n")
        else:
            t.append("No OPERA DISP-S1 frame here (its published scope is North America from mid-2016, "
                     "with a few frames elsewhere).\n", style="yellow")
        t.append("Sub-swaths are looked up for the chosen track when the project is written, with a check that "
                 "its bursts cover the whole box and every report area.\n", style="dim")
        if r.get("gnss") is not None:
            t.append_text(gnss_text(r["gnss"]))
        elif r.get("gnss_error"):
            t.append(f"GNSS stations could not be checked: {r['gnss_error'][:120]}\n", style="yellow")
        desc = [x for x in r["tracks"] if x["direction"] == "DESCENDING" and x["dates"] >= 60]
        asc = [x for x in r["tracks"] if x["direction"] == "ASCENDING" and x["dates"] >= 60]
        if asc and desc:
            t.append("Both directions are well covered: vertical and east-west can be separated.\n", style="green")
        self.query_one("#archiveinfo", Static).update(t)
        self.query_one("#track", Select).set_options(
            [(f"{x['direction'].title()} {x['track']} ({x['dates']} dates)", f"{x['direction']}:{x['track']}")
             for x in r["tracks"]])
        if r["tracks"]:
            self.query_one("#track", Select).value = f"{r['tracks'][0]['direction']}:{r['tracks'][0]['track']}"
        self._track_info()

    def _chosen_frame(self, direction: str, track: int) -> dict | None:
        detail = (self.archive or {}).get("disp_frame_detail") or []
        if hasattr(areas, "frame_for_track"):
            return areas.frame_for_track(detail, int(track), direction)
        rows = [d for d in detail if d.get("track") == int(track)
                and str(d.get("direction") or "").lower() == direction.lower()]
        return max(rows, key=lambda d: (d.get("overlap_fraction") or 0.0, d.get("granules") or 0)) if rows else None

    @on(Select.Changed, "#track")
    @on(Select.Changed, "#route")
    @on(Input.Changed, "#name")
    def _track_info(self, *_) -> None:
        """What the chosen route, track and name will make: the DISP-S1 frame
        on that track, and the folder (a screening never shares the official
        project's folder: areas.project_key adds _disp_screening)."""
        try:
            info = self.query_one("#trackinfo", Static)
            route = self.query_one("#route", Select).value
            tv = self.query_one("#track", Select).value
        except Exception:                                     # noqa: BLE001 - not composed yet
            return
        t = Text()
        name = self.query_one("#name", Input).value.strip()
        if name:
            key = areas.project_key(name, route) if hasattr(areas, "project_key") else areas.slug(name)
            t.append(f"Project folder and config: {key}", style="dim")
            if route == "disp":
                t.append("  (DISP-S1 screenings get their own folder, never the official project's)", style="dim")
            t.append("\n")
        if route == "disp" and self.archive is not None and tv not in (None, Select.BLANK):
            direction, track = str(tv).split(":")
            t.append_text(frame_text(self._chosen_frame(direction, int(track)), track, direction))
        info.update(t)

    # ---- write
    @on(Button.Pressed, "#write")
    def write(self) -> None:
        if self.area is None or self.archive is None:
            self.notify("Choose the area and check the archive first.", severity="warning")
            return
        name = self.query_one("#name", Input).value.strip() or self.area.label
        route = self.query_one("#route", Select).value
        tv = self.query_one("#track", Select).value
        if tv in (None, Select.BLANK):
            self.notify("Pick a track.", severity="warning")
            return
        direction, track = str(tv).split(":")
        frame, asks = "", []
        if route == "disp":
            row = self._chosen_frame(direction, int(track))
            if row is None:
                self.notify(f"No DISP-S1 frame on {direction.lower()} track {track}; choose a track that has one, "
                            "or the phase-linked route.", severity="error")
                return
            frame = str(row.get("frame") or "")
            ov = row.get("overlap_fraction")
            try:
                from .disp_backend import MIN_FRAME_OVERLAP
            except Exception:                                 # noqa: BLE001
                MIN_FRAME_OVERLAP = 0.95
            if ov is not None and ov < MIN_FRAME_OVERLAP:
                asks.append((f"DISP-S1 frame F{frame} covers only {100 * ov:.0f}% of the processing box: part of "
                             "the area will have no data. Write the project anyway?", "Write it"))
        have = None
        try:
            have = areas.existing_project(name, route)
        except Exception:                                     # noqa: BLE001 - write_project refuses anyway
            pass
        if have:
            asks.append((f"A project called {name!r} already exists ({have.get('config') or have.get('project_dir')}; "
                         f"holds {', '.join(have.get('holds') or []) or 'no results yet'}).\n\nReplace its config? "
                         "The current one is kept as a .bak copy beside it; what the folder holds stays and the "
                         "next run reuses or overwrites it.", "Replace the config"))
        args = dict(name=name, route=route, direction=direction, track=int(track), frame=frame,
                    overwrite=bool(have), deforming=self.query_one("#deforming", Input).value.strip(),
                    start=self.query_one("#start", Input).value, end=self.query_one("#end", Input).value)

        def ask(i):
            if i == len(asks):
                self.query_one("#writeinfo", Static).update(Text("writing the project (looking up the bursts "
                                                                 "for this track)…", style="cyan"))
                self._write(args)
                return
            self.app.push_screen(Confirm(*asks[i]), lambda yes: ask(i + 1) if yes else None)
        ask(0)

    @work(thread=True, exclusive=True, group="write")
    def _write(self, a: dict) -> None:
        try:
            deforming = areas.load_polygon(a["deforming"], "deforming area") if a["deforming"] else None
            # swaths=None: write_project looks the sub-swaths up for THIS track
            # and refuses bursts that leave part of the box or an area uncovered
            # (the old call passed the first track's swaths for any track).
            # The area row carries the project's name, not the outline's label.
            out = areas.write_project(
                a["name"], self.area, a["route"], a["start"], a["end"], track=a["track"], direction=a["direction"],
                disp_frame=a["frame"], swaths=None,
                report_areas=[("Full Satellite Imaging", "full"), (a["name"], self.area)],
                deforming=deforming, overwrite=a["overwrite"])
        except Exception as exc:                             # noqa: BLE001
            self.app.call_from_thread(self._err, "#writeinfo", exc)
            return
        self.app.call_from_thread(self._written, out)

    def _written(self, out) -> None:
        runs._remember(out)
        self.query_one("#writeinfo", Static).update(Text(f"✓ wrote {out}", style="green"))
        self.app.push_screen(PreflightScreen(str(out), "full"))

    def action_back(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------- preflight
def _level_mark(level: str) -> str:
    return {"ok": "[green]✓[/green] ", "warn": "[yellow]![/yellow] ", "fail": "[red]✗[/red] ",
            "info": "[dim]·[/dim] "}.get(level, "")


def preflight_checks(cfg, config: str, kind: str = "full", say=None) -> list[tuple[str, str]]:
    """The checks before a run of `kind`, as [(level, text)] with level ok /
    warn / info / fail; only 'fail' blocks the start. `say` receives the
    planner's lines. Kinds: resume (checked as the restart it resolves to),
    full, miaplpy, reunwrap, mintpy, products.

    - Earthdata: doctor.earthdata_login_status (data_acquisition's coded
      check: the login accepted by Earthdata AND ASF), only when the run
      downloads (full); 'unreachable' (an outage) is a warning, not a fail.
    - ERA5: stack_backend.check_weather_credentials; only its 'fail' level
      blocks ('warn' = could not verify), the publication-lag note and the
      local ERA5 cache.
    - Disk: settings.disk_budget_gb (on WSL C: free + what the vhdx holds on
      C: that ext4 does not use) against
      the planner's peak minus what the project already holds (full), the
      phase-linking re-check (miaplpy), or the run's own watchdog limits.
    - Memory: the planner's products-stage peak against this machine's RAM.
    - The patched MiaplPy with the project's own mini-stack / time lag.
    - Report-area names the client should not see; other runs going."""
    say = say or (lambda line: None)
    out: list[tuple[str, str]] = []
    add = lambda level, text: out.append((level, text))      # noqa: E731
    P = Path(cfg.project_dir)
    resolved = kind
    if kind == "resume":
        try:
            resolved, why = runs.suggest_kind(P, cfg)
            add("info", f"Resume will run '{runs.KINDS.get(resolved, (resolved,))[0].split(':')[0]}': {why}")
        except Exception as exc:                              # noqa: BLE001
            resolved = "full"
            add("warn", f"could not work out the restart from the files ({exc}); checked as a whole run")
    add("ok", f"config loads: {cfg.backend} route, {cfg.start_date} → {cfg.end_date}, track {cfg.track}")
    if cfg.backend == "disp":
        add("warn", "DISP-S1 is fast screening; official numbers come from the phase-linked route")
    elif cfg.backend != "miaplpy":
        add("warn", f"the {cfg.backend} route is for comparison only; official numbers come from the "
                    "phase-linked route")
    # the study (domains.py): what it will and will not give; nothing for the default
    from . import domains
    st = domains.study_of(cfg)
    if st is not domains.DEFAULT:
        from . import evidence
        for line in evidence.card(st, route=cfg.backend):
            add("info", line)
        if st.wording.held_sentence:
            add("warn", f"the {st.key} study gives a screening report: it is sent only after your sign-off, and "
                        "its figures show where to look, not survey values")
        else:
            add("warn", f"the {st.key} study's client report is held from release until its wording is checked")
        # a field-wide vertical figure in a GNSS frame needs >= 3 datum stations (needs datum_stations>=3)
        if any("datum_stations>=3" in o.needs for o in st.outputs):
            n = len([x for x in (getattr(cfg, "reference_gnss_stations", None) or []) if str(x).strip()])
            if str(getattr(cfg, "reference_mode", "") or "").lower() != "gnss_absolute" or n < 3:
                add("info", "the field-wide vertical figure needs at least 3 GNSS stations (reference_mode: "
                            "gnss_absolute, reference_gnss_stations; set with e); without them every figure is "
                            "relative to the surrounding ground")
        if st.directions == "both_required":
            add("warn", "this study needs both viewing directions for client figures; this project has one")

    # downloads: the login Earthdata and ASF both accept, not a .netrc line;
    # servers that did not answer say nothing about the login (a warning)
    if resolved == "full":
        try:
            from .doctor import EARTHDATA_UNREACHABLE, earthdata_login_status
            code, msg = earthdata_login_status()
            if code == "unreachable":
                add("warn", f"Earthdata login (downloads) not checked: {EARTHDATA_UNREACHABLE} ({str(msg)[:160]})")
            else:
                add("ok" if code == "ok" else "fail", "Earthdata login (downloads): " + str(msg)[:200])
        except (ImportError, AttributeError):
            netrc = Path.home() / ".netrc"
            good = netrc.exists() and "urs.earthdata.nasa.gov" in netrc.read_text(errors="replace")
            add("ok" if good else "fail", "Earthdata login in ~/.netrc (downloads; not verified with ASF)")
        except Exception as exc:                              # noqa: BLE001
            add("warn", f"Earthdata login could not be checked ({str(exc)[:160]})")

    # ERA5: every processing kind reaches MintPy's troposphere step
    if str(cfg.mintpy_tropo_method).lower() == "pyaps" and resolved in ("full", "miaplpy", "reunwrap", "mintpy") \
            and cfg.backend != "disp":
        from . import stack_backend as sb
        try:
            probs = sb.check_weather_credentials(cfg)
        except Exception as exc:                              # noqa: BLE001
            probs = [{"level": "warn", "text": f"could not check ({exc})"}]
        if not probs:
            add("ok", "ERA5 access (token and licence)")
        for p in probs:
            level = p.get("level", "fail") if isinstance(p, dict) else "fail"
            text = p.get("text", str(p)) if isinstance(p, dict) else str(p)
            add("fail" if level == "fail" else "warn",
                ("ERA5: " if level == "fail" else "ERA5 (could not verify, not blocking): ") + str(text)[:300])
        try:
            lag = sb.era5_lag_note(cfg)
            if lag:
                add("warn", lag)
        except Exception:                                     # noqa: BLE001
            pass
        try:
            pre = sb.era5_prefetch_plan(cfg)
            if pre.get("ok"):
                left = int(pre["needed"]) - int(pre["present"])
                add("info", f"ERA5 cache: {pre['present']} of {pre['needed']} weather files already on disk "
                            f"({pre['dir']})" + (f"; {left} more download during the run" if left > 0 else ""))
            elif pre.get("reason"):
                add("info", f"ERA5 cache: not settled yet ({pre['reason']})")
        except Exception:                                     # noqa: BLE001
            pass

    # HRRR through RAiDER (mintpy_tropo_method raider): its environment, and that the
    # model serves every date the series keeps (HRRR: continental US, from 2016-07-15)
    if str(cfg.mintpy_tropo_method).lower() == "raider" and resolved in ("full", "miaplpy", "reunwrap", "mintpy") \
            and cfg.backend != "disp":
        from . import settings as st, tropo_disp
        model = str(getattr(cfg, "weather_model", "") or "").upper()
        env = str(getattr(cfg, "raider_env", "") or "")
        if env and not (st.env_prefix(env) / "bin" / "raider.py").exists():
            add("fail", f"{model} weather delays need the {env} environment with RAiDER in it; it is not installed "
                        "(run install/install.sh)")
        excluded = {str(d) for d in (getattr(cfg, "mintpy_exclude_dates", None) or [])}
        try:
            from . import stack_backend as sb
            dates = sorted(d for d in (sb.planned_dates(cfg) or set()) if d not in excluded)
        except Exception:                                     # noqa: BLE001
            dates = []
        planned = bool(dates)
        dates = dates or [str(cfg.start_date).replace("-", "")]
        try:
            ok, msg = tropo_disp.check_model_range(cfg, dates)
        except Exception as exc:                              # noqa: BLE001
            ok, msg = True, f"{model}: could not check the dates ({exc})"
        if ok and not planned:
            # the start date stood in for the dates, which are not planned yet
            msg = (f"{model} covers this area and the record from its start ({cfg.start_date}); a date it "
                   "cannot serve would stop the run at its weather step (no partial correction)")
        add("ok" if ok else "fail", f"weather delays: {msg}" + ("" if ok else
            " (mintpy_exclude_dates can leave the earlier dates out)"))

    # the patched MiaplPy, with this project's own overlap
    if cfg.backend == "miaplpy" and resolved in ("full", "miaplpy"):
        from .miaplpy_backend import verify_patched_miaplpy
        try:
            info = verify_patched_miaplpy(cfg.miaplpy_env, ministack=getattr(cfg, "miaplpy_ministack_size", None),
                                          time_lag=getattr(cfg, "miaplpy_time_lag", None))
            add("ok", f"patched MiaplPy {info.get('miaplpy_version')} verified "
                      f"(boundary error {info.get('ratio')}x inside; upstream reads ~3x)")
        except Exception as exc:                              # noqa: BLE001
            add("fail", f"MiaplPy is not the patched build: run patches/miaplpy/build_miaplpy.sh ({str(exc)[:120]})")

    # names the client report would show
    try:
        from .client_report import parse_areas
        for name, spec in parse_areas(cfg):
            if spec.lower() == "full":
                continue
            prob = areas.client_name_problem(name)
            if prob:
                add("warn", f"report area {prob} - press e to rename it before the report is built")
    except Exception:                                         # noqa: BLE001
        pass

    try:
        others = runs.live_runs(exclude=P)
    except Exception:                                         # noqa: BLE001
        others = []
    if others:
        add("warn", "also running on this machine: " + "; ".join(
            f"{Path(o['project']).name} ({o.get('kind') or 'run'})" for o in others)
            + ". They share the cores, memory and disk; two big runs can run out of memory.")

    # disk and memory
    from . import settings
    try:
        b = settings.disk_budget_gb(P)          # a folder not made yet: its parent's filesystem
        budget, src = float(b["budget"]), str(b.get("source") or "")
    except Exception as exc:                                  # noqa: BLE001
        add("fail", f"disk budget could not be measured ({exc})")
        return out
    try:
        from .pipeline import DISK_STOP_GB, DISK_WARN_GB
    except ImportError:
        DISK_WARN_GB, DISK_STOP_GB = 60.0, 20.0
    if cfg.backend == "disp":
        need = 40.0                   # ~7 MB per granule window + the time series files
        add("ok" if budget > need else "warn" if budget > 0.5 * need else "fail",
            f"disk budget {budget:.0f} GB against ~{need:.0f} GB for a DISP-S1 screening ({src})")
    elif resolved == "full":
        from . import pipeline, stack_backend as sb
        say("Running the planner (downloads nothing)…")
        try:
            plan = sb.plan_stack(cfg)
        except Exception as exc:                              # noqa: BLE001
            add("fail", f"the planner failed: {str(exc)[:300]}")
            return out
        for line in sb.format_plan(plan).splitlines():
            say(line)
        have = pipeline.project_gb(P) if P.exists() else 0.0
        need = max(0.0, float(plan.est_peak_gb) - have)
        text = (f"disk: projected peak {plan.est_peak_gb:.0f} GB"
                + (f", {have:.0f} GB already in the project, so {need:.0f} GB more" if have >= 1 else "")
                + f"; budget {budget:.0f} GB ({src})")
        add("fail" if need > budget else "warn" if budget - need < DISK_WARN_GB else "ok",
            text + (" - free space or shorten the dates/area" if need > budget else
                    f" - under {DISK_WARN_GB:.0f} GB to spare at the peak" if budget - need < DISK_WARN_GB else ""))
        if plan.est_ram_gb and plan.mem_gb:
            frac = plan.est_ram_gb / plan.mem_gb
            add("fail" if frac > 0.9 else "warn" if frac > 0.5 else "ok",
                f"memory: the products stage needs ~{plan.est_ram_gb:.0f} GB of this machine's "
                f"{plan.mem_gb:.0f} GB" + (" - it will not fit; shrink the area or coarsen the posting"
                                           if frac > 0.9 else " - run nothing else alongside it" if frac > 0.5 else ""))
    elif resolved == "miaplpy":
        from . import stack_backend as sb
        try:
            d = sb.check_phase_linking_disk(cfg, raise_on_short=False)
            add("ok" if d.get("ok", True) else "fail",
                f"disk for phase linking: ~{d.get('need_gb', 0):.0f} GB more against a budget of "
                f"{budget:.0f} GB ({src})")
        except Exception as exc:                              # noqa: BLE001
            add("warn" if budget > DISK_WARN_GB else "fail",
                f"disk budget {budget:.0f} GB ({src}); the phase-linking need could not be worked out ({exc})")
    else:
        # MintPy / re-unwrap / products write tens of GB at most; the run's
        # own watchdog stops it below DISK_STOP_GB
        add("fail" if budget < DISK_STOP_GB else "warn" if budget < DISK_WARN_GB else "ok",
            f"disk budget {budget:.0f} GB ({src}); the run stops itself below {DISK_STOP_GB:.0f} GB")
    if resolved != "full" and cfg.backend != "disp":
        try:
            from . import stack_backend as sb
            plan = sb.StackPlan()
            plan.n_dates = len(sb.planned_dates(cfg) or ())
            sb._plan_memory(cfg, plan)
            if plan.est_ram_gb and plan.mem_gb:
                frac = plan.est_ram_gb / plan.mem_gb
                add("fail" if frac > 0.9 else "warn" if frac > 0.5 else "ok",
                    f"memory: the products stage needs ~{plan.est_ram_gb:.0f} GB of this machine's "
                    f"{plan.mem_gb:.0f} GB")
        except Exception:                                     # noqa: BLE001 - an estimate, not a gate
            pass
    return out


class PreflightScreen(Screen):
    """The checks before a run of a given kind, and its Start button (disabled
    while any check fails). From the run view (s, p) Start returns to it;
    from the new-area flow it opens the run view."""
    BINDINGS = [("escape", "back", "Back"), ("e", "edit", "Edit config"), ("r", "recheck", "Check again")]

    def __init__(self, config: str, kind: str = "full", from_run: bool = False):
        super().__init__()
        self.config = config
        self.kind = kind
        self.from_run = from_run
        self.ok = False
        self.results: list = []

    def compose(self) -> ComposeResult:
        label = runs.KINDS.get(self.kind, (self.kind,))[0]
        yield Header(show_clock=True)
        with VerticalScroll():
            yield Static(f"[b]Preflight[/b]  {label}\n[dim]{self.config}[/dim]")
            yield Static("[cyan]checking…[/cyan]", id="checks")
            yield RichLog(id="plan", max_lines=400, wrap=True, markup=False)
            with Horizontal(classes="row"):
                yield Button("Start the run", id="go", variant="success", disabled=True)
                yield Button("Open the live view without starting", id="view")
        yield Footer()

    def on_mount(self) -> None:
        self._run_checks()

    @work(thread=True, exclusive=True, group="preflight")
    def _run_checks(self) -> None:
        from .config import load_config
        self.ok = False
        self.app.call_from_thread(self._lock_start)
        try:
            cfg = load_config(self.config)
        except Exception as exc:                             # noqa: BLE001
            self.app.call_from_thread(self.query_one("#checks", Static).update,
                                      Text(f"✗ config: {exc}", style="red"))
            return
        say = lambda line: self.app.call_from_thread(self.query_one("#plan", RichLog).write, line)  # noqa: E731
        try:
            res = preflight_checks(cfg, self.config, self.kind, say=say)
        except Exception as exc:                             # noqa: BLE001
            res = [("fail", f"the checks themselves failed: {exc}")]
        self.results = res
        self.ok = not any(level == "fail" for level, _ in res)
        text = "\n".join(_level_mark(level) + escape(msg) for level, msg in res)
        if not self.ok:
            text += "\n\n[red]Fix what is marked ✗ (e edits the config, r checks again).[/red]"
        self.app.call_from_thread(self._done, text)

    def _lock_start(self) -> None:
        self.query_one("#go", Button).disabled = True

    def _done(self, text: str) -> None:
        self.query_one("#checks", Static).update(text)
        self.query_one("#go", Button).disabled = not self.ok

    @on(Button.Pressed, "#go")
    def go(self) -> None:
        if not self.ok:
            return
        from .config import load_config
        cfg = load_config(self.config)
        try:
            runs.start(self.config, cfg.project_dir, self.kind)
        except Exception as exc:                             # noqa: BLE001
            self.notify(str(exc), severity="error")
            return
        if self.from_run:
            self.app.pop_screen()            # back to the run view it came from: no second RunScreen
        else:
            self.app.switch_screen(RunScreen(self.config))

    @on(Button.Pressed, "#view")
    def view(self) -> None:
        if self.from_run:
            self.app.pop_screen()
        else:
            self.app.switch_screen(RunScreen(self.config))

    def action_edit(self) -> None:
        edit_config(self.app, self.config)
        self._run_checks()

    def action_recheck(self) -> None:
        self.query_one("#checks", Static).update("[cyan]checking…[/cyan]")
        self.query_one("#plan", RichLog).clear()
        self._run_checks()

    def action_back(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------- app
class InsarDashboard(App):
    TITLE = "InSAR pipeline"
    SUB_TITLE = "ground motion from Sentinel-1"
    BINDINGS = [Binding("question_mark", "help", "Guide", show=True)]
    CSS = """
    Screen { background: $surface; }
    #title { padding: 1 2 0 2; }
    #projects, #storage { height: 1fr; margin: 1 2; }
    #stitle { padding: 1 2 0 2; height: auto; }
    #sdetail, #sprog { height: auto; padding: 0 2; }
    #detail { height: auto; padding: 0 2; }
    #runtitle { padding: 1 2 0 2; height: auto; }
    #runbody { height: 1fr; }
    #left { width: 3fr; padding: 0 1; }
    #right { width: 1fr; min-width: 34; padding: 1 1; border-left: tall $primary 30%; }
    #stages { height: auto; max-height: 55%; }
    #log { height: 1fr; border: round $primary 40%; margin-top: 1; }
    #net { height: 3; margin-bottom: 1; }
    #results { height: auto; }
    .dim { color: $text-muted; }
    #newbody { padding: 1 2; }
    #newbody > Static { margin-top: 1; }
    .row { height: auto; margin: 0 0 1 0; }
    .row Input { width: 1fr; }
    .row Select { width: 1fr; }
    .short { width: 16; max-width: 16; }
    #places { height: auto; max-height: 8; }
    Confirm, ChooseRun, HelpScreen, SignOffScreen, StorageConfirm { align: center middle; }
    #dialog { width: 80; height: auto; max-height: 90%; padding: 1 2; border: thick $primary; background: $panel; }
    #helpdialog, #signdialog { width: 100; height: 90%; padding: 1 2; border: thick $primary; background: $panel; }
    #helpbody, #signbody { height: 1fr; }
    #buttons { height: auto; margin-top: 1; }
    #buttons Button { margin-right: 2; }
    #kinds { height: auto; max-height: 16; margin: 1 0; }
    #fpath { padding: 1 2 0 2; height: auto; }
    #rbody { padding: 1 2; }
    #rstats, #rannual { height: auto; margin-top: 1; }
    #rprog { margin: 1 0; }
    .lbl { width: auto; padding: 1 1 0 1; color: $text-muted; }
    #fbody { height: 1fr; }
    #ftree { width: 2fr; margin: 1 0 0 1; }
    #fprev { width: 3fr; padding: 1 2; border-left: tall $primary 30%; }
    """

    def __init__(self, config: str | None = None):
        super().__init__()
        self.start_config = config

    def on_mount(self) -> None:
        self.push_screen(ProjectsScreen())
        if self.start_config:
            self.push_screen(project_screen(str(Path(self.start_config).resolve())))
        self._version()

    @work(thread=True, exclusive=True, group="version")
    def _version(self) -> None:
        # which code is running, in the header (version.label hashes the package's .py files)
        try:
            from . import version
            lab = version.label()
        except Exception:                                     # noqa: BLE001
            return
        self.call_from_thread(setattr, self, "sub_title", lab)

    def action_help(self) -> None:
        if not isinstance(self.screen, HelpScreen):
            self.push_screen(HelpScreen())


USAGE = """insar                      the dashboard
insar <config.yaml>        the dashboard, opened on that project
insar app [--no-window]    the same in a browser window: projects, live runs on a map, results
insar builder              the map builder: draw/upload the area and mines, create the project
insar viewer               the results map: every measurement point, click for its movement, draw polygons, export
insar doctor [--projects DIR]   check the installation (and set the projects folder)
insar cleanup <config> [--level 1|2|3] [--delete [--force] [--yes]]   free disk space on a finished project
insar studies [<study>]    the studies a project can answer: what each will and will not give
insar events <config> [--between FROM TO] [--record] [--polygons F] [--lines F] [--points F] [--out DIR]
                           change maps, dated-panel peaks, line profiles and values at points (a study's
                           wells and faults included) from a finished run
insar storage              every project on this machine: size, state, delivered?, what can be freed; the free disk
insar storage keep-results <name> [--apply]   keep a finished project's results, delete its processing files
insar storage delete <name> [--no-archive] [--apply]   delete a whole project (a small archive is kept first)
insar results <config> --area NAME|COORDS|FILE [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--export]
insar injection-test <config> [options]   plant known movements and measure how well they come back
insar cross-check <config> [options]      compare the record with OPERA DISP-S1 (an independent product)
insar validate <config> --protocol <yaml> [options]   compare a finished run with ground truth under a fixed protocol
insar diagnose <project> [--out DIR]   zip the run records, logs and doctor output for the custodian
insar --version            which version and code this is
insar --menu               the old numbered menu"""


def main() -> None:
    sub = sys.argv[1] if len(sys.argv) > 1 else ""
    if sub in ("-h", "--help", "help"):
        print(USAGE)
        return
    if sub == "--version":
        from . import version
        print(version.label())
        return
    if sub == "results":
        from . import results
        sys.exit(results.main(sys.argv[2:]))
    if sub == "diagnose":
        from . import diagnose
        sys.exit(diagnose.main(sys.argv[2:]))
    if sub in ("injection-test", "cross-check"):
        # under the project lock and the child teardown, like every run (cli.tool_main)
        from . import cli
        sys.exit(cli.tool_main(sub, sys.argv[2:]))
    if sub == "validate":
        # read-only on the project, so no lock; refuses while a run holds it (cli.validate_main)
        from . import cli
        sys.exit(cli.validate_main(sys.argv[2:]))
    if sub == "events":
        # read-only on the products; refuses while a run holds the project (events.main)
        from . import events
        sys.exit(events.main(sys.argv[2:]))
    if sub == "studies":
        # read-only: every study, what it will and will not give (domains.py, evidence.yaml)
        from . import cli
        sys.exit(cli.studies_main(sys.argv[2:]))
    if sub == "storage":
        # overview of every project and the two ways to free disk (storage.py); dry run unless --apply
        from . import cli
        sys.exit(cli.storage_main(sys.argv[2:]))
    if sub == "app":
        from . import app_server
        sys.exit(app_server.main(sys.argv[2:]))
    if sub == "viewer":
        from . import results_viewer
        results_viewer.main()
        return
    if sub in ("doctor", "builder", "cleanup"):
        sys.argv = [f"insar {sub}"] + sys.argv[2:]
        if sub == "doctor":
            from . import doctor
            sys.exit(doctor.main())
        if sub == "builder":
            from . import map_builder
            map_builder.main()
            return
        from .scripts import cleanup
        sys.exit(cleanup.main())
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--menu" in sys.argv:
        from . import cli
        sys.argv = [sys.argv[0]] + args
        cli.main()
        return
    InsarDashboard(args[0] if args else None).run()


if __name__ == "__main__":
    main()
