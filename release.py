"""The send gate: is this client report ready to leave the building?

Until 5 Oct 2026 a dashboard run built the client report and KMZ
unconditionally once QA had run. The Site B report went out of the
pipeline with a raw file-stem area name ('project_zone'), an unverified
0.5 m feature HANDOFF said needed an imagery check "before anything goes to
a client", and the bare token "overall: REVIEW" - and nothing recorded who,
if anyone, had looked at it.

Kyle's decision (5 Oct 2026): the client report is a DRAFT unless
release.is_final(...) is true. Final = no hard blockers AND a reviewer
sign-off (deliverables/review.json) bound to the current numbers. The
binding is a hash of client_report.json without timestamps and paths, so any
rebuild that changes a number voids the sign-off automatically - the
sign-off is tied to the outcome, not to the act of signing. The record lives
in deliverables/, not in the YAML, so editing the config never looks like a
review.

Hard blockers stop a release; soft ones go on the reviewer's list. Hard is
kept to integrity and naming (a gate that blocks every farmland site teaches
people to ignore it): network not connected; a report area under half
measured (or, in a report from before 5 Oct 2026, with no comparison with
ordinary ground); an area name that looks like a file stem; a feature not
checked in the review; a screening-route report; a report of a study other
than mining subsidence, whose wording is not checked yet
('study_not_released', 8 Nov 2026); stale inputs (including a
signal-to-noise raster that is not the report's); a report older than the
quality assessment; a time-series run that did not finish; a reissue that
does not say so (decision M3, 7 Oct 2026: the project already issued a
report - deliverables/final_* or a final package in deliverables/releases/ -
with other numbers, and this one carries no report_revision note, or one
older than the latest issue: 'reissue_without_note'); no sign-off for
these numbers. Each flagged QA check is soft, with its note, and so is a
report area that cannot be tested (decision D15, 7 Oct 2026: the report says
so plainly and gives it no verdict, which is an honest answer the reviewer
must read and accept; a hard block made such a report impossible to issue).
Two more soft items come from the trust review (7 Oct 2026): an area (or the
ground on a mapped mine layer) whose local or coherence-matched farmland
disagrees with the scene's ('null_local_differs', E6), and real farmland that makes a stronger patch
than every simulated copy ('feature_null_misspecified', E8). And one from its
follow-up (F7(b)): GNSS stations the report rests on that were not screened
for equipment changes and earthquakes, because NGL's steps file could not be
read or the tie predates the screening ('gnss_steps_not_screened',
gnss_tie.unscreened_reason). And from the readiness review (16 Oct 2026,
ERR-01): the parts the report build could not work out (client_report.json
'omitted') - one soft reading item ('report_parts_omitted'), hard
('report_part_failed') for the two whose absence makes a printed figure or
the gate's own input wrong (OMITTED_HARD).

    blockers(cfg, rep=None, qa=None) -> [{code, text, hard}]
    null_checks_differ(rep) -> [(area name, [why, ...])]
    feature_null_check(rep) -> (check | None, source | None)
    omitted_items(rep) -> [{code, text, hard, page_text?}]
    numbers_hash(rep) -> str
    earlier_issues(project_dir, rep=None) -> [{label, date, path}]
    review_record(project_dir) -> dict | None
    sign_off(project_dir, reviewer, notes="", acknowledged_features=(), rep=None, specialist="") -> Path
    is_final(cfg, rep=None, qa=None) -> bool
    report_predates_qa(project_dir, rep=None) -> str | None

    python -m insar_pipeline.release <config.yaml>                    # list blockers
    python -m insar_pipeline.release <config.yaml> --sign "Name" --features 1,2,3 [--notes "..."]
                                     [--specialist "Name"]    # a report held for a specialist
"""
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from . import domains

REVIEW_FILE = "review.json"
MIN_MEASURED_PERCENT = 50.0

# Keys whose values change on every rebuild without any number changing, or
# that only say where or by what the report was built. Dropped (at any depth)
# before hashing; everything else in client_report.json is "the numbers".
# 'written' is the GNSS tie's own timestamp (gnss_tie._write, copied into
# rep['gnss']) and 'data_file_modified' its stations' download times: the
# sign-off rebuild re-runs the tie, so with them in the hash no sign-off ever
# survived its own rebuild and no report could become FINAL (7 Oct 2026).
# 'software' is the build label the records carry since 16 Oct 2026
# (readiness review OUT-01): the GNSS tie's reaches rep['gnss'], and a code
# change alone must not void a sign-off any more than 'version' does.
_VOLATILE_KEYS = {"generated", "generated_utc", "project", "release", "version", "finished", "started",
                  "credits", "log", "path", "file", "written", "data_file_modified", "software"}
_ABS_PATH = re.compile(r"^(/|[A-Za-z]:[\\/]|\\\\)")

log = logging.getLogger(__name__)


class _Failed(str):
    """A _stale line that is a check failure (its report-page wording is PAGE_CHECK_FAILED)."""


# The DRAFT report page lists the hard lines; where a line's own text is for
# the reviewer (exception text, the QA's technical hold reasons), its
# 'page_text' is what the page shows (16 Oct 2026).
PAGE_CHECK_FAILED = ("One of the release checks could not run; the release gate (the dashboard, or python -m "
                     "insar_pipeline.release) says which and why.")
PAGE_HOLD = ("The quality assessment holds this report for a specialist, who must read its reasons and be named "
             "at sign-off.")
PAGE_PART_MISSING = ("Part of this report could not be worked out on this build; the release gate (the dashboard, "
                     "or python -m insar_pipeline.release) says which and why.")

# The report blocks (client_report.OPTIONAL_BLOCKS) whose absence makes a
# printed figure, or the gate's own input, wrong - their failure is a hard
# blocker (16 Oct 2026, readiness review ERR-01). Every other block the
# build left out is one soft reading item: the page says it is missing and
# what is printed stays right. Only blocks made in client_report.compute
# belong here (client_report._refresh_omitted_items).
OMITTED_HARD = {
    # _with_point_uncertainty then keeps the QA's split-half repeatability
    # as the 'Reading the numbers' single-point figure
    "point_figures": "without it the page quotes the quality assessment's split-half repeatability as the "
                     "single-point uncertainty, 20-40% smaller than the rasters' and the KMZ's (E18)",
    # _send_rule then falls back to the QA's own rule, written before the tie
    "send_rule": "without it the report keeps the quality assessment's own send rule, which never saw the GNSS "
                 "tie, so a failed tie would not hold the report for a specialist",
}
OMITTED_CODES = ("report_part_failed", "report_parts_omitted", "omitted_not_read")
# Left out of numbers_hash: what they make (credits, version, who prepared
# it) is not in the hash either (_VOLATILE_KEYS), and the ones the page
# makes are recorded after compute's release block has hashed the record -
# hashed, they would make that block disagree with the record on disk.
_OMITTED_NOT_HASHED = {"credits", "version", "figure_credits", "footer_version", "footer_identity",
                       "credits_section"}


def _check_failed(what: str, exc: BaseException) -> str:
    """The line for a release check that raised instead of answering. Until
    16 Oct 2026 the gate read such a check as 'nothing to report', so a hard
    blocker (an unannounced reissue, a stale GNSS tie) vanished on an error
    and the report could go FINAL (readiness review ERR-01). Now the gate
    fails closed: the check's line says it could not run, as hard as the
    check it stands for. Lower-case, no full stop (callers add them)."""
    log.warning("release check '%s' could not run: %s: %s", what, type(exc).__name__, exc)
    return _Failed(f"the release check for {what} could not run ({type(exc).__name__}: {str(exc)[:160]}); "
                   "fix what that says, then check the release again")


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _load(p: Path):
    try:
        return json.loads(Path(p).read_text()) if Path(p).exists() else None
    except (OSError, ValueError):
        return None


def _project_dir(cfg_or_dir) -> Path:
    return Path(getattr(cfg_or_dir, "project_dir", cfg_or_dir))


def _report(cfg, rep):
    return rep if rep is not None else _load(_project_dir(cfg) / "deliverables" / "client_report.json")


def _qa(cfg, qa):
    return qa if qa is not None else _load(_project_dir(cfg) / "quality_assessment.json")


# ------------------------------------------------------------------ the binding
def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items()
                if k not in _VOLATILE_KEYS and not (isinstance(v, str) and _ABS_PATH.match(v))}
    if isinstance(obj, list):
        return [_strip(v) for v in obj if not (isinstance(v, str) and _ABS_PATH.match(v))]
    return obj


def numbers_hash(rep: dict) -> str:
    """sha256 of client_report.json's content without timestamps, paths, the
    release block, the credits or the code version: identical numbers hash
    identically whenever and wherever they were rebuilt, and any changed
    number, area, feature or wording of a figure changes the hash.

    The 'omitted' list (16 Oct 2026, ERR-01) counts by which blocks were
    left out, not by their error text (which may carry an address or a
    temporary path), and an empty one not at all: a report that left
    nothing out hashes as it did before the list existed. The credits,
    version and identity blocks do not count (_OMITTED_NOT_HASHED)."""
    rep = dict(rep or {})
    if "omitted" in rep:
        om = rep.pop("omitted")
        blocks = sorted(str(o.get("block")) for o in om or [] if isinstance(o, dict)
                        and o.get("block") not in _OMITTED_NOT_HASHED) if isinstance(om, list) else om
        if blocks:
            rep["omitted"] = blocks
    canon = json.dumps(_strip(rep), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode()).hexdigest()


# ------------------------------------------------------------------ features
def features(rep: dict) -> list[dict]:
    """[{n, label}] for every feature the report shows, numbered as the
    report numbers them (Feature 1 = the first entry of rep['features'])."""
    obs = [o.get("title", "") for o in (rep or {}).get("observations", []) or []
           if isinstance(o, dict) and str(o.get("title", "")).startswith("Feature ")]
    out = []
    for i, f in enumerate((rep or {}).get("features", []) or [], start=1):
        label = next((t for t in obs if t.startswith(f"Feature {i}:")), None)
        if not label and isinstance(f, dict):
            where = (f"near {abs(f['lat']):.3f} {'N' if f['lat'] >= 0 else 'S'}, "
                     f"{abs(f['lon']):.3f} {'E' if f['lon'] >= 0 else 'W'}") if "lat" in f and "lon" in f else ""
            kind = "sinking" if (f.get("sign") or 0) < 0 else "rising" if (f.get("sign") or 0) > 0 else "moving"
            label = f"Feature {i}: {kind} ground" + (f", about {f['area_km2']} km²" if f.get("area_km2") else "") \
                + (f" {where}" if where else "")
        out.append({"n": i, "label": label or f"Feature {i}"})
    return out


def _feature_ids(ack) -> set[int]:
    """Accepts 1, '1', 'Feature 1', 'feature-1'."""
    out = set()
    for a in ack or ():
        m = re.search(r"(\d+)", str(a))
        if m:
            out.add(int(m.group(1)))
    return out


# ------------------------------------------------------------------ review record
def review_record(project_dir) -> dict | None:
    """deliverables/review.json, or None when there is none (or it does not parse)."""
    rec = _load(_project_dir(project_dir) / "deliverables" / REVIEW_FILE)
    return rec if isinstance(rec, dict) else None


def sign_off(project_dir, reviewer: str, notes: str = "", acknowledged_features=(), rep: dict | None = None,
             specialist: str = "") -> Path:
    """Record a reviewer's sign-off of the current client_report.json.

    The record holds the numbers hash, so it is void the moment a rebuild
    changes a number. Every feature the report shows must be acknowledged
    (checked against imagery / dates / the ground); a partial
    acknowledgement is refused rather than recorded as a sign-off. A report
    the quality assessment holds for a specialist needs the specialist's
    name (who read the hold reasons; may be the reviewer); it is recorded
    with those reasons, and the release gate accepts it only for them.
    Returns the path after reading the record back and confirming it binds."""
    P = _project_dir(project_dir)
    reviewer = (reviewer or "").strip()
    specialist = (specialist or "").strip()
    if not reviewer:
        raise ValueError("a sign-off needs the reviewer's name")
    rep = _report(P, rep)
    if not rep:
        raise FileNotFoundError(f"no client report to sign off: {P / 'deliverables' / 'client_report.json'}")
    sr = _send_rule(P, rep, _qa(P, None))
    if sr.get("value") == HOLD and not specialist:
        why = "; ".join(str(x).rstrip(".") for x in sr.get("reasons") or [])
        raise ValueError(f"the quality assessment holds this report for a specialist{f' ({why})' if why else ''}: "
                         "a specialist must read those reasons and be named at sign-off "
                         "(dashboard: the Specialist field; terminal: --specialist \"Name\")")
    feats = features(rep)
    ack = _feature_ids(acknowledged_features)
    missing = [f["label"] for f in feats if f["n"] not in ack]
    if missing:
        raise ValueError("every feature must be checked before signing off; not acknowledged: " + "; ".join(missing))
    unknown = sorted(ack - {f["n"] for f in feats})
    if unknown:
        raise ValueError(f"acknowledged feature(s) {unknown} are not in this report (it has {len(feats)})")
    try:
        from . import version
        code = {"version": version.__version__, "code_hash": version.code_hash()}
    except Exception:                                         # noqa: BLE001
        code = None
    rec = {"reviewer": reviewer, "date": datetime.now().isoformat(timespec="seconds"), "notes": notes or "",
           "acknowledged_features": sorted(ack), "feature_labels": [f["label"] for f in feats],
           "numbers_hash": numbers_hash(rep), "report_generated": rep.get("generated"), "code": code}
    if specialist:
        rec["specialist"] = {"name": specialist, "send_rule": sr.get("value"), "reasons": _hold_reasons(sr)}
    path = P / "deliverables" / REVIEW_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Keep the trail: an earlier sign-off is moved aside, never overwritten.
        old = _load(path) or {}
        stamp = re.sub(r"[^0-9]", "", str(old.get("date", "")))[:12] or datetime.now().strftime("%Y%m%d%H%M")
        hist = path.parent / "reviews"
        hist.mkdir(exist_ok=True)
        path.replace(hist / f"review_{stamp}.json")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    tmp.replace(path)
    back = review_record(P)
    if not back or back.get("numbers_hash") != numbers_hash(rep):
        raise RuntimeError(f"{path} does not read back bound to the current report")
    return path


# ------------------------------------------------------------------ reissues (M3, 7 Oct 2026)
_ISSUE_DATE = re.compile(r"(20\d\d)(\d\d)(\d\d)")


def _issue_date(d: Path, manifest: dict | None) -> str:
    """The date a copy was issued (YYYY-MM-DD): from its folder name
    (final_20260928, releases/20261008_1200), else its manifest, else the
    folder's own time."""
    m = _ISSUE_DATE.search(d.name)
    if m:
        try:
            return datetime(int(m[1]), int(m[2]), int(m[3])).date().isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(str((manifest or {}).get("created"))).date().isoformat()
    except ValueError:
        return datetime.fromtimestamp(d.stat().st_mtime).date().isoformat()


_ISSUED_HASHES: dict = {}


def _issued_hash(f: Path) -> str | None:
    """numbers_hash of an issued copy's client_report.json, cached on its
    size and time (issued copies do not change, and the dashboard asks every
    few seconds); None when there is none or it does not parse."""
    try:
        st = f.stat()
    except OSError:
        return None
    key = (str(f), st.st_mtime_ns, st.st_size)
    if key not in _ISSUED_HASHES:
        old = _load(f)
        _ISSUED_HASHES[key] = numbers_hash(old) if isinstance(old, dict) else None
    return _ISSUED_HASHES[key]


def earlier_issues(project_dir, rep: dict | None = None) -> list[dict]:
    """The reports this project has already issued with numbers other than
    rep's: every deliverables/final_* folder (assembled by hand before
    packages existed) and every final package in deliverables/releases/ (not
    a <stamp>_DRAFT, not one still being written). A copy whose report hashes
    as rep's is this very report already packaged, not an earlier issue; a
    copy whose report cannot be read counts as an issue.
    [{label, date (YYYY-MM-DD), path}], oldest first."""
    D = _project_dir(project_dir) / "deliverables"
    here = numbers_hash(rep) if rep else None
    out = []
    for d in sorted(D.glob("final_*")) + sorted((D / "releases").glob("*")):
        if not d.is_dir() or d.name.startswith(".") or d.name.upper().endswith("_DRAFT"):
            continue
        man = _load(d / "MANIFEST.json")
        man = man if isinstance(man, dict) else None
        if man is not None and man.get("final") is False:
            continue
        # the manifest's hash, and the copied report's hashed by this code
        # (a later change to what the hash leaves out must not turn the
        # package of these very numbers into an "earlier" issue)
        hashes = {man.get("numbers_hash")} if man else set()
        for f in (d / "technical" / "client_report.json", d / "client_report.json"):
            h = _issued_hash(f)
            if h:
                hashes.add(h)
        if here and here in hashes:
            continue
        out.append({"label": d.relative_to(D).as_posix(), "date": _issue_date(d, man), "path": str(d)})
    return sorted(out, key=lambda x: x["date"])


def _cfg_note(cfg) -> dict | None:
    """config.report_revision of a loaded config; None for a project folder."""
    from .config import report_revision
    return report_revision(cfg) if hasattr(cfg, "project_dir") else None


def _revision_note(cfg, rep: dict | None):
    """(the report_revision note, normalised by config.report_revision, or
    None; True when it is the report's own copy). The report's copy when it
    carries one - what the page, PDF and KMZ printed; else the loaded
    config's; else, for a caller with only the project folder (the
    dashboard's projects table), the config snapshot the products run wrote."""
    from .config import report_revision
    if isinstance(rep, dict) and "report_revision" in rep:
        return report_revision(rep.get("report_revision") or {}), True
    if hasattr(cfg, "project_dir"):
        return _cfg_note(cfg), False
    import yaml
    try:
        snap = yaml.safe_load((_project_dir(cfg) / "deliverables" / "config_snapshot.yaml").read_text()) or {}
        return report_revision(snap.get("report_revision") or {}), False
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        return None, False


def _day(iso: str) -> str:
    try:
        d = datetime.fromisoformat(iso)
        return f"{d.day} {d:%b %Y}"
    except ValueError:
        return iso


def _reissue_problem(cfg, rep: dict | None, issued: list[dict]) -> str | None:
    """Why this report cannot go out as a reissue, or None. Only a project
    that has already issued a report with other numbers is a reissue."""
    if not issued:
        return None
    last = issued[-1]
    where = f"{_day(last['date'])} (deliverables/{last['label']})"
    note, own = _revision_note(cfg, rep)
    if note is None:
        cfg_note = _cfg_note(cfg) if own else None
        if cfg_note:
            return (f"This report replaces the one issued {where}, and the project's settings now carry its "
                    f"revision note ({cfg_note['heading']}), but the report was built before the note was set; "
                    "rebuild the report so the page, the PDF and the KMZ print it.")
        return (f"This report replaces the one issued {where}, with other figures, but says nothing about it: set "
                "report_revision in the project's settings (its number, the date of the issue it replaces and "
                "what changed, in plain sentences), then rebuild the report so the page, the PDF and the KMZ "
                "print it.")
    if note["supersedes"] < last["date"]:
        return (f"The revision note says this report replaces the one issued {note['supersedes_text']}, but a later "
                f"report was issued {where}; update report_revision (number, date and changes against that "
                "issue), then rebuild the report.")
    if note["supersedes"] != last["date"]:
        # a date after the latest issue names no report the client holds
        # (a typo such as 2026-09-29 for 2026-09-28 would be printed as it is)
        return (f"The revision note says this report replaces the one issued {note['supersedes_text']}, but no "
                f"report was issued that day; the latest issue is {where}. Correct the date in report_revision, "
                "then rebuild the report.")
    cfg_note = _cfg_note(cfg) if own else None
    if cfg_note and cfg_note != note:
        # the note was edited after the build: the page still prints the old one
        return ("The revision note in the project's settings is not the one this report prints (it was changed "
                "after the report was built); rebuild the report so the page, the PDF and the KMZ print it.")
    return None


# ------------------------------------------------------------------ the gate
def _looks_like_file_stem(name: str, spec: str = "") -> bool:
    """'project_zone', 'area_a1.geojson', or the polygon file's own stem.
    A client-facing name has capitals or spaces ('Area A1')."""
    n = (name or "").strip()
    if not n:
        return True
    if re.search(r"\.(geojson|json|shp|kml|kmz|gpkg)$", n, re.I):
        return True
    if spec and not str(spec).startswith(("full", "bbox")) and Path(str(spec)).stem == n and n == n.lower():
        return True
    return bool(re.fullmatch(r"[a-z0-9]+([_\-][a-z0-9]+)+", n))


def _area_measured_percent(a: dict):
    for k in ("measured_percent", "coverage_percent"):
        if a.get(k) is not None:
            return float(a[k])
    return None


def _area_testable(a: dict):
    if "testable" in a:
        return bool(a["testable"]), a.get("not_testable_reason")
    # Reports built before 5 Oct 2026 carry no 'testable': an area was tested
    # when the farmland comparison produced a verdict.
    if a.get("detected") is not None or a.get("z") is not None:
        return True, None
    return False, "the report carries no comparison with ordinary ground for this area"


def _is_screening(cfg, rep) -> bool:
    if (rep or {}).get("route") == "screening":
        return True
    try:
        from . import config as _c
        return bool(_c.is_screening(cfg))
    except Exception:                                         # noqa: BLE001
        return str(getattr(cfg, "backend", "")) == "disp"


def _record_is_current(cfg, rec, option: str):
    """crosscheck.record_is_current, never raising: True / False, or the
    exception when the check itself failed (the gate then says so rather
    than reading it as 'not this run's', ERR-01, 16 Oct 2026). A record the
    check cannot date (not a dict, no readable 'generated') is not this run's."""
    try:
        from . import crosscheck
        return bool(crosscheck.record_is_current(cfg, rec, option))
    except Exception as exc:                                  # noqa: BLE001 - the gate never crashes
        return exc


MINTPY_INCOMPLETE = "run_incomplete.json"     # deliverables.RUN_INCOMPLETE, written by time_series


def _mintpy_incomplete(cfg) -> dict | None:
    """The marker a time-series run leaves while it is part-way ({stamp,
    started}; {} when it does not parse), or None when there is none."""
    P = _project_dir(cfg)
    for d in dict.fromkeys((Path(getattr(cfg, "mintpy_dir", P / "mintpy")), P / "mintpy")):
        f = d / MINTPY_INCOMPLETE
        if f.exists():
            rec = _load(f)
            return rec if isinstance(rec, dict) else {}
    return None


def _stale(cfg, rep) -> list[str]:
    """Inputs newer than what was built from them."""
    P = _project_dir(cfg)
    out = []
    mintpy = Path(getattr(cfg, "mintpy_dir", P / "mintpy"))
    vel = mintpy / "geo" / "geo_velocity.h5"
    if not vel.exists():
        vel = mintpy / "velocity.h5"
    vt = vel.stat().st_mtime if vel.exists() else None
    gen = (rep or {}).get("generated")
    if vt is not None and gen:
        try:
            g = datetime.fromisoformat(str(gen)).timestamp()
            # 'generated' has minute resolution
            if vt > g + 60:
                out.append(f"the velocity product ({vel.name}, {datetime.fromtimestamp(vt):%d %b %H:%M}) is newer "
                           f"than the report ({gen}); rebuild the report")
        except ValueError:
            pass
    qa_f = P / "quality_assessment.json"
    if vt is not None and qa_f.exists() and qa_f.stat().st_mtime < vt:
        out.append("the quality assessment is older than the velocity product; re-run it")
    tie = _load(P / "deliverables" / "gnss_tie.json")
    dates = (rep or {}).get("dates") or []
    anchored = bool(((rep or {}).get("gnss") or {}).get("anchored", (rep or {}).get("anchored")))
    window_off = bool(anchored and tie and tie.get("record_window") and dates
                      and list(tie["record_window"]) != list(dates[:2]))
    if window_off:
        out.append(f"the GNSS tie covers {tie['record_window'][0]}-{tie['record_window'][1]} but the report "
                   f"{dates[0]}-{dates[1]}; re-run the tie")
    # A tie from before the regional uncertainty floor (5 Oct 2026) is stale
    # whether or not the report used it: a report built on it prints too
    # narrow a range (Site A +/- 0.73 vs 1.01 mm/yr), and one built since
    # fell back to relative-only figures until the tie is re-run.
    try:
        from . import gnss_tie
        old_tie = gnss_tie.stale_reason(tie)
    except Exception as exc:                                  # noqa: BLE001 - the gate never crashes; fails closed
        old_tie = None
        out.append(_check_failed("whether the GNSS tie is out of date", exc))
        window_off = True                                     # one line for it: the next check reads the same tie
    if old_tie:
        out.append(old_tie + ", then rebuild the report")
    # A tie the report could not use because it is older than the velocity
    # product or covers another window (gnss_tie.check) turned the whole
    # report relative-only with nothing else to say so (7 Oct 2026: the
    # velocity product re-run, the report rebuilt without the tie). Only a
    # tie that had a regional motion to lose matters.
    elif not window_off and isinstance(tie, dict) and isinstance(tie.get("background"), dict) and dates:
        try:
            from . import gnss_tie
            # the dashboard asks with the project folder alone
            c = cfg if hasattr(cfg, "project_dir") else SimpleNamespace(project_dir=P, mintpy_dir=mintpy)
            if gnss_tie.radius_km(c) > 0 and gnss_tie.check(c, dates)[0] is None:
                out.append("re-run the GNSS tie (it does not match the current velocity product), then rebuild "
                           "the report")
        except Exception as exc:                              # noqa: BLE001 - the gate never crashes; fails closed
            out.append(_check_failed("whether the GNSS tie matches the current velocity product", exc))
    snr = _snr_not_the_reports(P, rep)
    if snr:
        out.append(snr)
    return out


def _snr_not_the_reports(P: Path, rep: dict | None) -> str | None:
    """Why deliverables/rate_signal_to_noise.tif is not the report's, or None.

    The export writes it first from the first farmland pass; the client
    report writes it again from its own farmland (WRITTEN_BY=client_report)
    just before client_report.json. When the report stage does not get that
    far the first-pass file stays beside an unchanged, signed report, and
    the class centres can differ by about 1 mm/yr between the passes
    (Site B), so a cell stands out in data/ and not in the report. A
    raster with no product tags at all predates them (5 Oct 2026) and says
    nothing about which pass wrote it, so only its age is judged.

    The files on disk are compared, so only for the report on disk: a
    report still being built (client_report.compute's release block) writes
    the raster itself straight after, and the gate read from disk after the
    build (the KMZ, the package, the dashboard) catches a rewrite that failed."""
    D = P / "deliverables"
    f, rj = D / "rate_signal_to_noise.tif", D / "client_report.json"
    if not f.exists() or not rj.exists():
        return None
    if rep is not None and _load(rj) != rep:
        return None
    why = ("the signal-to-noise raster rate_signal_to_noise.tif is not the report's own (the export rewrote it "
           "after the report, or the report's rewrite failed); rebuild the report")
    try:
        import rasterio
        with rasterio.open(f) as ds:
            tags = ds.tags()
        # a minute's grace, as for the velocity product above
        newer = f.stat().st_mtime > rj.stat().st_mtime + 60
    except Exception:                                         # noqa: BLE001 - unreadable is not the report's
        return why
    first_pass = str(tags.get("FARMLAND", "")).startswith("first pass") or (
        "QUANTITY" in tags and tags.get("WRITTEN_BY") != "client_report")
    return why if newer or first_pass else None


# A QA re-run cannot land within this many seconds of the report it follows
# (the QA stage alone took 20-30 s on Site B, 6 Oct 2026, and a run holds
# the project lock), so files written within it - by one script, one copy or
# one test fixture - are taken as written together, not as a re-run.
QA_AFTER_REPORT_GRACE_S = 2.0


def report_predates_qa(project_dir, rep: dict | None = None) -> str | None:
    """Why client_report.json is older than quality_assessment.json, or None.

    The report's send rule is the QA's re-evaluated with the GNSS tie
    (client_report._send_rule), and its hold reasons and quality notes are
    the QA's, so the report speaks for the QA only while client_report.json
    is at least as new as quality_assessment.json (decision D7, 7 Oct 2026;
    provenance._send_rule takes the report's rule in the same order). A QA
    re-run after the report left the report, and this gate's send-rule
    line, giving the previous assessment's verdict with nothing to say so.

    The files' modification times are compared, as provenance compares
    them (here allowing QA_AFTER_REPORT_GRACE_S), and only for the report
    on disk: a report still being built (client_report.compute's release
    block) has just read the QA as it is."""
    P = _project_dir(project_dir)
    rj, qa_f = P / "deliverables" / "client_report.json", P / "quality_assessment.json"
    try:
        rt, qt = rj.stat().st_mtime, qa_f.stat().st_mtime
    except OSError:
        return None
    if qt <= rt + QA_AFTER_REPORT_GRACE_S or (rep is not None and _load(rj) != rep):
        return None
    return (f"The report predates the quality assessment (re-run {datetime.fromtimestamp(qt):%d %b %H:%M}, after "
            f"the report was built {datetime.fromtimestamp(rt):%d %b %H:%M}), so its send rule and quality notes "
            "are the previous assessment's; rebuild the report.")


def _rule_after_qa(P: Path, qa: dict | None) -> dict:
    """The send rule that speaks for a QA re-run after the report (D7), read
    in provenance._send_rule's order, which the dashboard's send_rule_shown
    follows: summary.json's - the QA's rule re-evaluated with the GNSS tie,
    which quality_assessment.write refreshes straight after the QA - when it
    was written after this QA; else the QA's own, which never saw the tie.
    Until this, the gate's line read the QA's own rule, so after a re-run
    with an unusable tie it said 'send with notes' beside the dashboard's
    and provenance's 'hold for specialist'. summary.json keeps the value
    alone: the QA's reasons go with it only when the tie left the value as
    the QA's."""
    own = (qa or {}).get("send_rule")
    own = own if isinstance(own, dict) else {}
    try:
        from . import provenance
        sr, src = provenance._send_rule(P, P / "quality_assessment.json", qa or {})
    except Exception:                                         # noqa: BLE001 - the gate never crashes
        return own
    if src != "summary.json" or not isinstance(sr, dict) or not sr.get("value") or sr["value"] == own.get("value"):
        return own
    return {"value": sr["value"], "reasons": ["its own rule re-evaluated with the GNSS tie, as "
                                              "deliverables/summary.json records it"]}


HOLD = "hold for specialist"


def _send_rule(P: Path, rep: dict | None, qa: dict | None, old_report: str | None = None) -> dict:
    """The send rule that speaks for this report, {} when there is none: the
    report's own (the QA's re-evaluated with the GNSS tie), or the QA's own
    for a report from before the rule; when the report predates the QA
    (old_report, report_predates_qa), the rule that speaks for the re-run QA."""
    if old_report:
        sr = _rule_after_qa(P, qa)
    else:
        sr = (rep or {}).get("send_rule")
        sr = sr if isinstance(sr, dict) else ((qa or {}).get("send_rule") or {})
    return sr if isinstance(sr, dict) else {}


def _hold_reasons(sr: dict) -> list[str]:
    return sorted(str(x).strip() for x in (sr or {}).get("reasons") or [])


def specialist_release(rev: dict | None, sr: dict) -> dict | None:
    """The specialist's release of a held report recorded in the review
    (sign_off(specialist=...)), when it was given for this hold: the same
    rule and the same reasons. None otherwise."""
    sp = (rev or {}).get("specialist")
    if not (isinstance(sp, dict) and str(sp.get("name") or "").strip()):
        return None
    if sp.get("send_rule") != HOLD or sorted(sp.get("reasons") or []) != _hold_reasons(sr):
        return None
    return sp


_STOPPED_RE = re.compile(r"\b(stopped moving|has stopped|no longer moving|no longer moves)\b", re.I)


def study_report_problems(rep: dict) -> list[tuple[str, str]]:
    """Hard problems in a study's screening report (A4, PLAN_oil_gas_slopes.md):
    a number printed for a point inside a fast zone; a rising-ground rate
    printed as a number (the uplift check, Gate 2, has not run); "stopped
    moving" anywhere in the record's words (the detection-power check, Gate
    2b, has not run)."""
    out = []
    pts = rep.get("study_points") or []
    if any(r.get("refused") and (r.get("shown_vertical_mm_yr") is not None
                                 or r.get("relative_vertical_mm_yr") is not None) for r in pts):
        out.append(("study_fast_zone_number", "A point inside a fast zone carries a number; the satellite cannot "
                                              "follow movement there, so no number may be given."))
    if any(r.get("detected") and (r.get("shown_vertical_mm_yr") or 0) > 0 for r in pts):
        out.append(("study_rising_number", "A rising-ground rate is printed as a number; rising ground is given no "
                                           "rate until the uplift check is done."))
    words = json.dumps({k: rep.get(k) for k in ("study_points", "study_events", "features", "observations",
                                                 "answer", "areas")}, default=str)
    if _STOPPED_RE.search(words):
        out.append(("study_stopped_moving", "The report says ground stopped moving; that needs the detection-power "
                                            "check, which has not been done."))
    return out


def blockers(cfg, rep: dict | None = None, qa: dict | None = None) -> list[dict]:
    """Plain-English reasons the client report is not ready to send.

    [{code, text, hard}]: hard ones stop the release, soft ones are the
    reviewer's reading list. rep / qa are read from the project when not given."""
    P = _project_dir(cfg)
    rep, qa = _report(cfg, rep), _qa(cfg, qa)
    out: list[dict] = []

    def add(code, text, hard, page=None):
        out.append({"code": code, "text": text, "hard": bool(hard), **({"page_text": page} if page else {})})

    from .config import is_experiment
    if is_experiment(cfg):
        add("experiment_run", "This project is a validation experiment (purpose: experiment): its result is the "
                              "validation against ground truth in validation/, and nothing from it goes to a client.",
            True)
    if not rep:
        add("no_report", "There is no client report yet (deliverables/client_report.json).", True)
        return out

    # The network: one connected piece, or the record is several separately
    # referenced records.
    net = ((qa or {}).get("checks") or {}).get("network") or {}
    if net.get("connected") is False:
        add("network_not_connected",
            "The interferogram network is not one connected piece, so parts of the record are referenced "
            "separately and their movements cannot be compared.", True)

    if _is_screening(cfg, rep):
        add("screening_route",
            "This report comes from the screening route (OPERA DISP-S1), not the official phase-linked "
            "processing; screening results are not sent to clients.", True)

    # a study other than the default (domains.py) still carries the mining study's
    # wording until its own passes Gate 1b (review_2026-11-07/PLAN.md), so its
    # report is never final; the default study and configs without `study` add nothing
    # either side saying another study holds it: a config edited back to the default
    # must not release a report built for another study
    named = [str(v).strip().lower() for v in (getattr(cfg, "study", ""), rep.get("study")) if v]
    other = [s for s in named if s != domains.DEFAULT_STUDY]
    if other:
        try:
            held = domains.study(other[0]).wording.held_sentence
        except KeyError:
            held = ""
        if held and len(named) == 2 and named[0] == named[1]:
            # a study whose own wording is checked (A3/A4 of PLAN_oil_gas_slopes.md,
            # decision G1, 9 Nov 2026): a screening report, sent only with the
            # reviewer's sign-off (the sign-off rule below), so a soft reading item
            add("study_screening", held, False)
            for code, text in study_report_problems(rep):
                add(code, text, True)
        else:
            add("study_not_released",
                f"This report answers the '{other[0]}' study, whose report wording has not been checked yet; it "
                "is not sent to clients until it is.", True)

    for a in rep.get("areas", []) or []:
        if not isinstance(a, dict):
            continue
        name = str(a.get("name", ""))
        if a.get("kind") != "full" and _looks_like_file_stem(name, str(a.get("spec", ""))):
            add("area_name", f"Report area '{name}' is named like a file, not a place; give it the name the "
                             "client uses (report_areas in the project's settings).", True)
        if a.get("kind") != "full":
            ok, why = _area_testable(a)
            if not ok and a.get("testable") is False:
                # soft (D15, 7 Oct 2026): the report says "Not testable" and
                # gives no verdict, an honest answer the reviewer reads and accepts
                why = str(why or "").strip().rstrip(".")
                add("area_not_testable", f"'{name}' is reported as not testable, with no verdict: it could not "
                                         "be compared with ordinary ground"
                    + (f". Reason given: {why[:1].upper() + why[1:]}" if why else "")
                    + ". Read that in the report and accept it before signing off.", False)
            elif not ok:
                # a report from before 'testable' (5 Oct 2026) with no comparison
                # does not say it is not testable: it prints a figure with nothing to read it by
                add("area_not_compared", f"'{name}' carries no comparison with ordinary ground, so its figure "
                                         "cannot be read; rebuild the report with the current code.", True)
        mp = _area_measured_percent(a)
        if mp is None and a.get("kind") != "full":
            add("area_measured_unknown", f"The report does not say how much of '{name}' was measured; rebuild "
                                         "the report with the current code.", True)
        elif mp is not None and mp < MIN_MEASURED_PERCENT:
            add("area_under_measured", f"Only {mp:.0f}% of '{name}' gave usable measurements (at least "
                                       f"{MIN_MEASURED_PERCENT:.0f}% is needed for an area figure).", True)

    # GNSS stations inside the processed area that disagree with the record.
    for chk in ((rep.get("gnss") or {}).get("inside_checks") or []):
        if isinstance(chk, dict) and chk.get("consistent") is False:
            add("gnss_inside_inconsistent", f"The GNSS station {chk.get('site', '?')} inside the area does not "
                                            "agree with the radar record there.", True)

    # Who prepared it (clientout-9): settings.identity() was empty on the
    # working machine, so a FINAL report could go out naming no firm and no
    # preparer. Contact stays optional.
    try:
        from . import settings
        ident = settings.identity()
    except Exception:                                         # noqa: BLE001 - unknown = not set
        ident = {}
    missing = [k for k in ("firm", "preparer") if not str(ident.get(k) or "").strip()]
    if missing:
        add("identity_missing", "The report does not say who prepared it (" + " and ".join(missing) + " not set): "
                                "set them with insar doctor --firm \"...\" --preparer \"...\".", True)

    # A time-series run that retired the previous products and has not yet
    # written its own (time_series: mintpy/run_incomplete.json): what is on
    # disk is not one run's products, whatever the report says.
    inc = _mintpy_incomplete(cfg)
    if inc is not None:
        when = inc.get("started") or inc.get("stamp")
        if isinstance(when, (int, float)):
            when = datetime.fromtimestamp(when).strftime("%d %b %H:%M")
        add("mintpy_incomplete", "The time-series run" + (f" started {str(when)[:16].replace('T', ' ')}" if when else "")
            + " did not finish, so the products on disk are not one run's; finish or re-run it, then rebuild "
              "the report.", True)

    stale = _stale(cfg, rep)
    # The QA re-run after the report (D7, 7 Oct 2026). _stale judges the QA
    # only against the velocity product, so this is not one of its lines.
    old_report = report_predates_qa(P, rep)
    if old_report:
        add("report_older_than_qa", old_report, True)
    # what the report itself found stale when it was built (client_report
    # stale_inputs: a QA, summary or uncertainty model older than the
    # velocity product) - its printed numbers came from them, so it stays
    # stale until rebuilt after they are re-run (tests-4). Its QA line goes
    # when the QA has been re-run since the report: what is left to do is
    # the rebuild report_older_than_qa asks for (and _stale's own QA line
    # says so if the re-run QA is still older than the velocity product).
    for s in (rep.get("stale_inputs") or []):
        s = str(s).strip().rstrip(".")
        if s and not (s.startswith("the quality assessment") and (
                old_report or any(x.startswith("the quality assessment") for x in stale))):
            stale.append(s + "; then rebuild the report")
    for s in stale:
        add("stale_inputs", s[0].upper() + s[1:] + ".", True, PAGE_CHECK_FAILED if isinstance(s, _Failed) else None)

    # A reissue that does not say so (M3, 7 Oct 2026): Site A's rebuilt
    # report changed its headline and every figure with nothing on the page,
    # the PDF or the KMZ to tell a client holding the 28 Sep issue why.
    page = None
    try:
        why = _reissue_problem(cfg, rep, earlier_issues(P, rep))
    except Exception as exc:                                  # noqa: BLE001 - the gate never crashes; fails closed
        why, page = _cap(_check_failed("an earlier issue of this report", exc)) + ".", PAGE_CHECK_FAILED
    if why:
        add("reissue_without_note", why, True, page)

    # The review: every feature checked, bound to these numbers.
    rev = review_record(P)
    h = numbers_hash(rep)
    bound = bool(rev) and rev.get("numbers_hash") == h
    ack = _feature_ids(rev.get("acknowledged_features")) if bound else set()
    for f in features(rep):
        if f["n"] not in ack:
            add("feature_unchecked", f"{f['label']} has not been checked (imagery, dates, ground) and recorded "
                                     "in the review.", True)
    if not rev:
        add("not_signed_off", "Not yet reviewed and signed off.", True)
    elif not bound:
        add("signoff_stale", f"The sign-off by {rev.get('reviewer', '?')} on {str(rev.get('date', '?'))[:10]} was "
                             "for an earlier build of these numbers; review and sign again.", True)

    # Soft: each flagged quality check, with its plain note.
    for k, c in ((qa or {}).get("checks") or {}).items():
        if isinstance(c, dict) and c.get("pass") is False:
            note = str(c.get("plain") or c.get("note") or "").strip().rstrip(".")
            add(f"qa_{k}", f"Quality check flagged: {k.replace('_', ' ')}" + (f" - {note}" if note else "") + ".",
                False)
    # Soft: the tests' own checks - an area's local or coherence-matched
    # farmland disagrees with the scene's (E6), the feature copies are
    # lighter than the real farmland (E8). The gate never crashes on them.
    try:
        _diagnostic_items(rep, add)
    except Exception as exc:                                  # noqa: BLE001 - the gate never crashes; says so
        add("diagnostics_not_read", _cap(_check_failed("the report's own comparison checks", exc)) + ".", False)
    # What the report build left out (ERR-01, 16 Oct 2026): hard for the
    # blocks in OMITTED_HARD, one soft reading item for the rest. Fails
    # closed: a list that cannot be read may hide a hard one.
    try:
        for it in omitted_items(rep):
            add(it["code"], it["text"], it["hard"], it.get("page_text"))
    except Exception as exc:                                  # noqa: BLE001 - the gate never crashes; fails closed
        add("omitted_not_read", _cap(_check_failed("the parts the report left out", exc)) + ".", True,
            PAGE_CHECK_FAILED)
    # Soft (F7(b), 7 Oct 2026): the GNSS stations the report rests on were
    # not screened for equipment changes and earthquakes - NGL's steps file
    # could not be read, or the tie predates the screening. Read from the
    # tie the report used (rep['gnss']); the anchored figures stand, with
    # that caveat for the reviewer.
    try:
        from . import gnss_tie
        unscreened = gnss_tie.unscreened_reason(rep.get("gnss"))
    except Exception as exc:                                  # noqa: BLE001 - the gate never crashes; says so
        unscreened = _cap(_check_failed("GNSS equipment-change and earthquake screening", exc)) + "."
    if unscreened:
        add("gnss_steps_not_screened", unscreened, False)
    # the report's send rule first: the QA's re-evaluated with the GNSS tie,
    # which runs after the QA (client_report._send_rule); the QA's own for a
    # report built before 5 Oct 2026. A rule with no value (a QA from before
    # the rule existed) gives no send verdict at all: re-evaluated on
    # Site B's 29 Sep QA it said 'send with notes' where the current QA
    # holds the report for a specialist, so it is a hard blocker.
    # A report older than the QA does not speak for it (D7, 7 Oct 2026): the
    # rule that speaks for the re-run QA then, marked, in the order the
    # dashboard and provenance read it (_rule_after_qa).
    # 'hold for specialist' is hard (16 Oct 2026, readiness review QUA-03):
    # the user guide says the reviewer does not sign a held report off
    # alone, but the line was soft and any name made it FINAL. It clears only
    # when the sign-off bound to these numbers records the specialist who
    # read these hold reasons (sign_off(specialist=...)).
    sr = _send_rule(P, rep, qa, old_report)
    why = "; ".join(str(x).rstrip(".") for x in sr.get("reasons") or [])
    if "value" in sr and sr["value"] is None:
        add("send_rule_missing", "The quality assessment gives no send verdict"
            + (f" ({why})" if why else "") + ": re-run it, then rebuild the report.", True)
    elif sr.get("value") == HOLD:
        sp = specialist_release(rev, sr) if bound and not old_report else None
        if sp:
            add("qa_send_rule", f"The quality assessment says '{HOLD}'" + (f": {why}" if why else "")
                + f"; released by the specialist {sp['name']} at sign-off.", False)
        else:
            add("qa_send_rule", f"The quality assessment says '{HOLD}'" + (" (report out of date)" if old_report else "")
                + (f": {why}" if why else "") + ". A specialist must read these reasons and record the release at "
                "sign-off (dashboard r: the Specialist field; terminal: --specialist \"Name\").", True, PAGE_HOLD)
    elif sr.get("value") and sr["value"] != "send":
        add("qa_send_rule", f"The quality assessment says '{sr['value']}'"
            + (" (report out of date)" if old_report else "") + (f": {why}" if why else "") + ".", False)
    # crosscheck.py writes deliverables/technical/cross_check.json; only a
    # completed comparison can disagree ('no independent product' is not one),
    # and only this run's (crosscheck.record_is_current): one older than the
    # velocity product checked a previous MintPy run, and one left behind
    # after cross_check_disp was turned off is not this run's (7 Oct 2026).
    # A caller that passes only the project folder (the dashboard's projects
    # table) carries no option, so the line is left out there; it is soft, so
    # FINAL / DRAFT does not change.
    ccf = next((f for f in (P / "deliverables" / "technical" / "cross_check.json",
                            P / "deliverables" / "crosscheck.json") if f.exists()), None)
    cc = (_load(ccf) or {}) if ccf else {}
    if cc:
        cur = _record_is_current(cfg, cc, "cross_check_disp")
        if isinstance(cur, Exception):
            add("cross_check_unread", _cap(_check_failed("whether the satellite cross-check is this run's", cur))
                + ".", False)
        if cur is not True:
            cc = {}
    if cc and cc.get("status", "complete") == "complete" and (
            cc.get("indicative_pass") is False or cc.get("agree") is False
            or cc.get("within_tolerance") is False
            or any(isinstance(a, dict) and a.get("agree") is False for a in cc.get("areas") or [])):
        add("cross_check_disagrees", "The independent satellite cross-check does not agree with this record "
                                     f"everywhere; read {ccf.relative_to(P)}.", False)
    return out


# ------------------------------------------------------------------ the tests' own checks (7 Oct 2026)
# The keys of a report area that hold its local and coherence-matched nulls
# (significance.null_checks, copied by client_report; contract C1).
NULL_CHECK_KEYS = ("null_local", "null_coherence_matched")


def _tested_entries(rep: dict | None) -> list[tuple[str, dict]]:
    """(name, entry) for every tested client verdict that carries the nulls:
    each report area, and the ground on each mapped mine layer
    (client_report's mine_ground, tested by the same significance.assess
    through tested_fields, its verdict printed under Observations)."""
    rep = rep or {}
    out = [(str(a.get("name", "")), a) for a in rep.get("areas", []) or [] if isinstance(a, dict)]
    out += [(f"Ground on the mapped {m.get('label') or m.get('key') or 'mine layer'}", m)
            for m in rep.get("mine_ground", []) or [] if isinstance(m, dict)]
    return out


def _null_units(rep: dict | None) -> str:
    """The units client_report gives the copied nulls ('units', 7 Oct 2026):
    significance's figures, line of sight, beside a page that is vertical."""
    for _, a in _tested_entries(rep):
        for k in NULL_CHECK_KEYS:
            n = a.get(k)
            if isinstance(n, dict) and n.get("units"):
                return str(n["units"]).strip()
    return "mm/yr along the satellite's line of sight"


def null_checks_differ(rep: dict | None) -> list[tuple[str, list[str]]]:
    """[(area name, [why, ...])] for each report area - or the ground on a
    mapped mine layer, whose verdict comes from the same test - whose local
    null (farmland copies within 10 km) or coherence-matched null (copies of
    its mean coherence +/-0.025) disagrees with the pooled null it was
    tested against (E6, trust review 7 Oct 2026): a centre more than a
    quarter of the 95% threshold away, or a spread outside 0.8-1.25 times.
    The 5% false-alarm rate and the 4-in-5 detection are scene averages
    (finding #11: 0% to over 20% across 5 km blocks), so this is where they
    may not hold. Both nulls' reasons are given, the one that agrees too, so
    the reviewer can tell a regional shift from ground of another quality.
    Nothing switches: the report keeps the pooled test."""
    out = []
    for name, a in _tested_entries(rep):
        nulls = [a.get(k) for k in NULL_CHECK_KEYS if isinstance(a.get(k), dict)]
        if any(n.get("disagrees") is True for n in nulls):
            out.append((name, [str(n["why"]).strip() for n in nulls if n.get("why")]))
    return out


def feature_null_check(rep: dict | None) -> tuple[dict | None, str | None]:
    """(check, source): whether the real comparison ground's strongest patch
    beats every simulated copy of the feature search (E8, contract C4;
    report_insights.feature_null_check).

    Read from the report's 'feature_null_check', else from the copy inside
    'feature_search' (the null keeps one). A report built before 7 Oct 2026
    has neither, but its feature_search kept each side's copy maxima and the
    real farmland's strongest patch, so the same comparison is made from
    them (report_insights.feature_null_check's rule: tested side, real > 0,
    real > the strongest copy) - the delivered Site A report reads 87.1
    against a copy maximum of 61.2. (None, None) when the report made no
    feature search."""
    rep = rep or {}
    fs = rep.get("feature_search") if isinstance(rep.get("feature_search"), dict) else {}
    for src, chk in (("client_report.json feature_null_check", rep.get("feature_null_check")),
                     ("client_report.json feature_search.feature_null_check", fs.get("feature_null_check"))):
        if isinstance(chk, dict) and "real_beats_all_copies" in chk:
            return chk, src
    if not fs:
        return None, None
    if not fs.get("tested"):
        return {"real_beats_all_copies": False, "tested": False, "sides": [],
                "why": "the chance test could not be made"}, "recomputed from feature_search"
    beats, bits = [], []
    for name, s in (fs.get("sides") or {}).items():
        try:
            if not isinstance(s, dict) or not s.get("tested") or not s.get("max_mass"):
                continue
            real, top = float(s.get("farmland_strongest_mass") or 0.0), float(max(s["max_mass"]))
        except (TypeError, ValueError):
            continue
        if real > 0 and real > top:
            beats.append(name)
            bits.append(f"{name}: the real comparison ground's strongest patch (strength {real:.1f}) is stronger "
                        f"than the strongest of all {len(s['max_mass'])} simulated copies ({top:.1f})")
    why = ("; ".join(bits) + ". Ordinary ground here makes patches the copies never do - movement no layer maps, "
           "or copies too light - so the chance figures may be too low" if beats else
           "the real comparison ground's strongest patch is within what the simulated copies produce")
    return ({"real_beats_all_copies": bool(beats), "tested": True, "sides": beats, "why": why,
             "report_predates_check": True}, "recomputed from feature_search (report built before the check)")


def _diagnostic_items(rep: dict, add) -> None:
    """The soft items from the tests' own checks (trust review 7 Oct 2026)."""
    for name, whys in null_checks_differ(rep):
        add("null_local_differs",
            f"'{name}': ordinary farmland near this area, or of its coherence, does not behave like the farmland of "
            "the whole scene it was tested against. " + " ".join(w if w.endswith(".") else w + "." for w in whys)
            + f" (Figures in {_null_units(rep)}, as the test computes them; the page's are vertical.) Look at the "
              "farmland round the area on the map: a shifted centre is a regional offset that can push a reading "
              "near the threshold either way; a different spread means the 95% threshold is too tight or too "
              "loose there. Read the area's result, above all one close to the threshold, with that in mind.",
            False)
    chk, src = feature_null_check(rep)
    if isinstance(chk, dict) and chk.get("real_beats_all_copies") is True:
        why = str(chk.get("why") or "").strip().rstrip(".")
        old = bool(chk.get("report_predates_check")) or str(src or "").startswith("recomputed")
        what = ("real movement (an unmapped mine, spoil or a pit) belongs in the mine layers, which takes it out of "
                "the comparison ground on the next build; if it is not real, the copies are too light and every "
                "feature's chance figure is optimistic.")
        add("feature_null_misspecified",
            "The chance test for features may be too lenient for this ground - "
            + (why + ". " if why else "the real comparison ground makes a stronger patch than every simulated copy. ")
            + ("This report was built before that check (7 Oct 2026), so it does not say where the patch is and may "
               "still call a feature 'stands out clearly': rebuild it with the current code, then look at the patch "
               "it names on the map and imagery - " if old else
               "Look at that patch on the map and imagery: ") + what,
            False)


def omitted_items(rep: dict | None) -> list[dict]:
    """[{code, text, hard, page_text?}] for client_report.json's 'omitted'
    list (client_report._optional, 16 Oct 2026, readiness review ERR-01):
    'report_part_failed' (hard) for each block in OMITTED_HARD, and one
    'report_parts_omitted' (soft) naming every other block the build left
    out with its error. [] for a report that left nothing out, or one built
    before the list existed."""
    om = (rep or {}).get("omitted") or []
    if not isinstance(om, list):
        raise TypeError(f"client_report.json 'omitted' is a {type(om).__name__}, not a list")
    om = [o for o in om if isinstance(o, dict)]
    out, soft = [], []
    for o in om:
        what = str(o.get("what") or o.get("block") or "a part of the report")
        err = str(o.get("error") or "no error recorded").strip()
        if o.get("block") in OMITTED_HARD:
            out.append({"code": "report_part_failed", "hard": True, "page_text": PAGE_PART_MISSING,
                        "text": f"The report could not work out {what} ({err}): {OMITTED_HARD[o['block']]}. "
                                "Fix that (the report stage's log has the error), then rebuild the report."})
        else:
            soft.append(f"{what} ({err})")
    if soft:
        out.append({"code": "report_parts_omitted", "hard": False,
                    "text": f"This build of the report left out {len(soft)} part{'' if len(soft) == 1 else 's'} it "
                            f"could not work out: {'; '.join(soft)}. The page lists "
                            f"{'it' if len(soft) == 1 else 'them'} under 'Left out of this build'; fix the error "
                            "and rebuild, or accept the report without "
                            f"{'it' if len(soft) == 1 else 'them'} before signing off."})
    return out


def hard_blockers(cfg, rep=None, qa=None) -> list[dict]:
    return [b for b in blockers(cfg, rep, qa) if b["hard"]]


def is_final(cfg, rep: dict | None = None, qa: dict | None = None) -> bool:
    """No hard blockers (which include 'no sign-off bound to these numbers')."""
    rep = _report(cfg, rep)
    if not rep:
        return False
    rev = review_record(_project_dir(cfg))
    if not rev or rev.get("numbers_hash") != numbers_hash(rep):
        return False
    return not hard_blockers(cfg, rep, qa)


def status(cfg, rep: dict | None = None, qa: dict | None = None) -> dict:
    """{final, blockers, review} for client_report.json's 'release' block."""
    rep = _report(cfg, rep)
    b = blockers(cfg, rep, qa)
    rev = review_record(_project_dir(cfg))
    final = bool(rep) and not any(x["hard"] for x in b)
    return {"final": final, "blockers": b,
            "reviewed_by": rev.get("reviewer") if final and rev else None,
            "review_date": rev.get("date") if final and rev else None}


def main(argv=None):
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description="Release gate for a project's client report.")
    ap.add_argument("config")
    ap.add_argument("--sign", metavar="REVIEWER", help="record your sign-off of the current report")
    ap.add_argument("--features", default="", help="comma-separated feature numbers you checked, e.g. 1,2,3")
    ap.add_argument("--notes", default="")
    ap.add_argument("--specialist", default="", metavar="NAME",
                    help="for a report held for a specialist: who read the hold reasons and releases it")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.sign:
        p = sign_off(cfg.project_dir, a.sign, a.notes, [x for x in a.features.split(",") if x.strip()],
                     specialist=a.specialist)
        print(f"Signed off: {p}")
        print("Rebuild the client report (dashboard: products) so it renders without the DRAFT mark.")
    b = blockers(cfg)
    hard = [x for x in b if x["hard"]]
    print("FINAL - ready to package" if not hard else f"DRAFT - {len(hard)} thing(s) to settle before sending:")
    for x in b:
        print(("  [must] " if x["hard"] else "  [read] ") + x["text"])
    return 0 if not hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
