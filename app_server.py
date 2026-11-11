"""The pipeline app: the dashboard's work in a browser window, for people who
would rather not live in a terminal.

    insar app                 (on Windows: the "InSAR Pipeline" desktop icon,
                               install/windows/setup.ps1)

Pages: Projects (every project, its state and its results in the report's
words), a project's live view (the run's phases and stages with progress
and time left, a map that fills in as the run produces data, the log, the
results), and This computer (disk budget, memory, other runs, storage).
New projects are drawn in the map builder (map_builder.py) and results are
explored in the results viewer (results_viewer.py); this server starts both
and links to them rather than copying them.

Nothing here is a second implementation of the pipeline's rules. Run state,
stages, restarts and the preflight checks are the dashboard's and runs.py's
own functions (stage_rows, suggest_kind, preflight_checks, runs.start /
runs.stop); the area figures are client_report.json's, read through
dashboard.area_view / total_text / feature_lines, so the app and the
terminal dashboard can never disagree. A run started here is the same
detached run the dashboard starts, so either can watch it and closing the
window never stops it.

The live map draws what is on disk when it is there:
  study box, area of interest, mines and report areas  - from the config
  the satellite frame                                  - once the data search ran (slc/frame_footprint.json)
  measurement quality, line-of-sight velocity          - once MintPy geocodes (mintpy/geo/), then the products
  measurement points, GNSS stations                    - once the products and the GNSS tie are built
The rasters are the products drawn as pictures, labelled a look: the
figures quoted to a client are the report's.

Security is the results viewer's (16 Oct 2026, readiness review NET-06): the
server binds to 127.0.0.1 only, refuses other web pages (Host/Origin), takes
an API call only with the secret written into the page it served this
session, serves an exact list of files under a Content-Security-Policy, and
nothing it runs takes a shell string. Starting a run that processes needs a
passing preflight for that run kind on this server within PREFLIGHT_TTL_S,
the same gate as the dashboard's Start button.
"""
from __future__ import annotations

import base64
import http.server
import io
import json
import math
import os
import re
import secrets
import shutil
import socket
import sys
import threading
import time
from datetime import datetime
from functools import partial
from pathlib import Path

PKG = Path(__file__).resolve().parent
PAGE = "app.html"
PORT = 8767
SERVE_DIR = Path.home() / ".cache" / "insar_app"
ASSET_DIR = PKG / "app_assets"
VENDOR = PKG / "viewer_assets" / "vendor"
MM_IN = 25.4
M_IN = 1000 / MM_IN
PREFLIGHT_TTL_S = 15 * 60
MAX_RASTER_PX = 1100          # the longer side of a map picture
MAX_POINTS = 25000            # measurement points drawn on the live map (the viewer draws them all)
MAX_VERTICES = 6000           # per outline layer on the map
_SERVER: dict = {}
_PREFLIGHT: dict = {}         # (config, kind) -> (time, ok)
_RASTERS: dict = {}           # (path, layer) -> (mtime, payload)
_PREFLIGHT_LOCK = threading.Lock()
_STOP = threading.Event()     # set by Quit in the app: main() returns

ASSETS = {
    "assets/app.css": ASSET_DIR / "app.css", "assets/app.js": ASSET_DIR / "app.js",
    "assets/icon.png": ASSET_DIR / "icon.png",
    # the pixel fonts (SIL OFL 1.1, app_assets/fonts/LICENSES.txt): served from here, never a font CDN
    **{f"assets/fonts/{n}": ASSET_DIR / "fonts" / n for n in ("PressStart2P.woff2", "VT323.woff2", "LICENSES.txt")},
    **{f"assets/vendor/{n}": VENDOR / n for n in (
        "leaflet.min.js", "leaflet.min.css", "LICENSES.txt", "images/layers.png", "images/layers-2x.png",
        "images/marker-icon.png", "images/marker-icon-2x.png", "images/marker-shadow.png")},
}
CONTENT_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".png": "image/png",
                 ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8", ".woff2": "font/woff2"}
TILE_HOSTS = ("https://basemap.nationalmap.gov", "https://server.arcgisonline.com", "https://tile.openstreetmap.org",
              "https://*.tile.opentopomap.org")
# the results section shows the results viewer (results_viewer.PORT) in a frame: that one server only
VIEWER_ORIGINS = ("http://localhost:8766", "http://127.0.0.1:8766")
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
       f"img-src 'self' data: blob: {' '.join(TILE_HOSTS)}; connect-src 'self'; font-src 'self'; "
       f"frame-src {' '.join(VIEWER_ORIGINS)}; "
       "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

# The run's stages grouped into the phases the page shows as steps. A stage
# key not listed falls into "products" (everything after the time series).
PHASES = {
    "data": "Get the data", "stack": "Line up the images", "link": "Phase linking",
    "unwrap": "Unwrap", "series": "Time series", "products": "Products & report",
}
ROUTE_PHASES = {"miaplpy": ("data", "stack", "link", "unwrap", "series", "products"),
                "disp": ("data", "series", "products")}
DEFAULT_PHASES = ("data", "stack", "unwrap", "series", "products")
DATA_STAGES = {"checks", "download", "orbits", "stage_inputs", "era5_prefetch", "disp_search", "disp_prefetch"}
SERIES_STAGES = {"weather_raider", "disp_stitch", "disp_resample", "disp_products"}
NOT_A_STAGE = {"run", "pipeline", "disk"}
# Stages that run beside the others (the ERA5 prefetch): one still going does
# not hold its phase "active" once the run has moved past it.
BACKGROUND = {"era5_prefetch"}


def phase_of(key: str) -> str | None:
    if key in NOT_A_STAGE:
        return None
    if key in DATA_STAGES:
        return "data"
    if key.startswith("miaplpy_"):
        return "link"
    if key.startswith("mintpy_") or key in SERIES_STAGES or key.startswith("tropo"):
        return "series"
    if key.startswith("run_"):
        return "unwrap" if "unwrap" in key else "stack"
    return "products"


def _load_json(p) -> dict | list | None:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _mtime(p) -> float | None:
    try:
        return Path(p).stat().st_mtime
    except OSError:
        return None


def _when(t) -> str:
    return datetime.fromtimestamp(t).strftime("%d %b %H:%M") if t else ""


class _CombinedCfg:
    """A combined project (combine.py) seen through the pipeline config the
    pages expect: its own folder, backend "combined", and everything else -
    the study box, the report areas, their outlines - from its ascending
    project, which it never changes."""

    def __init__(self, path: str):
        from . import combine
        from .config import load_config
        self.cc = combine.load(path)
        self._asc = load_config(str(self.cc.ascending))
        self.project_dir = str(self.cc.project_dir)
        self._outline_dir = str(self._asc.project_dir)
        self._source_path = str(path)
        self._combined = True
        self.backend = "combined"

    def __getattr__(self, k):
        return getattr(self._asc, k)


def _is_combined(config: str) -> bool:
    from . import combine
    return combine.is_combined_config(config)


def _load_cfg(config: str):
    from .config import load_config
    from . import runs
    known = {str(c) for c in runs.known_configs()}
    if str(Path(config).resolve()) not in known:
        # only a project the pipeline already lists: the page never names a file of its own
        raise ValueError("not one of this computer's projects")
    if _is_combined(str(Path(config).resolve())):
        return _CombinedCfg(str(Path(config).resolve()))
    return load_config(str(Path(config).resolve()))


def _name(config: str) -> str:
    return Path(config).stem.replace("config_", "")


# ---------------------------------------------------------------- projects
def _areas_brief(rep: dict | None) -> list[dict]:
    from . import dashboard as db
    if not rep:
        return []
    anch = db.anchored_of(rep)
    out = []
    for a in rep.get("areas") or []:
        if not isinstance(a, dict):
            continue
        v = db.area_view(a, anch)
        out.append({"name": a.get("name", "?"), "rate": v["rate"], "range": v["range"], "reading": v["reading"],
                    "tone": _tone(v["style"]), "total": db.total_text(a)})
    return out


def _tone(style: str) -> str:
    return "warn" if "yellow" in style else "muted" if "dim" in style else "ok"


def _live_stage(P: Path, rec: dict | None, st: str) -> dict | None:
    """The stage a running run is on, for the projects list."""
    from . import dashboard as db, progress
    if st != "running":
        return None
    rows = [r for r in db.stage_rows(progress.latest(P), rec, st) if r["live"]]
    if not rows:
        return None
    r = rows[-1]
    return {"label": r["label"], "done": r["done"], "total": r["total"], "remaining": r["remaining"]}


def kept_records(names) -> list[dict]:
    """What survives of a project whose folder was deleted: the validation
    reports kept in the pipeline's review_*/results/<name>/ (the experiments,
    VALIDATION.md), the slim archive storage.delete_project leaves in
    <projects>/_archive/<name>_<date>/, and its lines in the run history.
    [{kind, path, what, finished, status, purpose}], newest first."""
    from . import runs
    from .settings import projects_root
    out, seen = [], set()
    for name in dict.fromkeys(n for n in names if n):
        for d in sorted(PKG.glob(f"review_*/results/{name}")):
            if d.is_dir() and d not in seen:
                seen.add(d)
                rr = _load_json(d / "run_result.json") or {}
                reports = sorted(x.name for x in d.iterdir() if x.is_dir())
                out.append({"kind": "results", "path": str(d), "what": "validation report" + ("s" if len(reports) > 1 else "")
                            + (f" ({', '.join(reports[:4])}{'…' if len(reports) > 4 else ''})" if reports else ""),
                            "finished": _when(rr.get("finished")), "t": rr.get("finished") or 0,
                            "status": rr.get("status"), "purpose": rr.get("purpose")})
        try:
            arch = sorted((projects_root() / "_archive").glob(f"{name}_*"))
        except Exception:                                     # noqa: BLE001 - no projects folder
            arch = []
        # storage._archive_dest names them <name>_<YYYYmmdd>[_n]: site_* alone would also
        # catch site_lc_full_20261023 (another project's archive)
        stamp = re.compile(re.escape(name) + r"_\d{8}(_\d+)?")
        for d in arch:
            if d.is_dir() and d not in seen and stamp.fullmatch(d.name):
                seen.add(d)
                out.append({"kind": "archive", "path": str(d), "what": "slim archive (deliverables, logs, config)",
                            "finished": _when(d.stat().st_mtime), "t": d.stat().st_mtime, "status": None, "purpose": None})
        try:
            hist = [r for r in runs.run_registry(project=name)]
        except Exception:                                     # noqa: BLE001
            hist = []
        if hist:
            last = hist[-1]
            out.append({"kind": "history", "path": "", "what": f"{len(hist)} run(s) in the run history, last a {last.get('kind')} run "
                        f"{last.get('status')} {str(last.get('finished') or '')[:10]}", "finished": str(last.get("finished") or "")[:16],
                        "t": 0, "status": last.get("status"), "purpose": None})
    return sorted(out, key=lambda r: -r["t"])


def api_home(_: dict) -> dict:
    from . import areas, dashboard as db, runs
    from .config import load_config
    out = []
    for c in runs.known_configs():
        p = runs.project_summary(c)
        row = {"config": str(c), "name": p.name, "route": areas.ROUTE_NAMES.get(p.backend, p.backend),
               "backend": p.backend, "state": p.state, "error": p.error, "updated": _when(p.updated),
               "updated_t": p.updated or 0}
        folder = Path(p.project_dir) if p.project_dir else None
        if folder is None or not folder.exists():
            # a folder deleted after its run (the experiments), or a config that no longer loads:
            # said so, with what survives, instead of a blank "not started"
            kept = kept_records([p.name, folder.name if folder else ""])
            if p.state == "invalid":
                # project_summary keeps the first line (the file's path); the reason is below it
                try:
                    load_config(str(c))
                except Exception as exc:                      # noqa: BLE001 - the reason, as the loader words it
                    lines = [x.strip() for x in str(exc).splitlines()[1:] if x.strip()]
                    row["error"] = "; ".join(lines[:3])[:300] or row["error"]
                row.update(gone="config", kept=kept)
            elif kept:
                row.update(state="removed", gone="folder", kept=kept)
        if p.backend == "combined":
            from . import combine
            row.update(gone=None, kept=None, combined=True, areas=combine.brief(combine.read_report(c)),
                       out_of_date="OUT OF DATE" in (p.verdict or ""))
            try:
                b = _CombinedCfg(str(c)).bbox
                row["center"], row["bbox"] = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2], list(b)
            except Exception:                                 # noqa: BLE001 - listed without a place
                pass
            out.append(row)
            continue
        if p.project_dir and row.get("gone") is None:
            P = Path(p.project_dir)
            rep = _load_json(P / "deliverables" / "client_report.json")
            sr, stale = db.send_rule_shown(P)
            rs = db.release_status(P) if rep else None
            row.update({"send_rule": sr, "report_out_of_date": stale, "areas": _areas_brief(rep),
                        "final": None if rs is None else bool(rs.get("final")),
                        "live": _live_stage(P, runs.record(P), p.state)})
            try:
                b = load_config(str(c)).bbox
                row["center"] = [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]
                row["bbox"] = list(b)
            except Exception:                                 # noqa: BLE001 - listed without a place
                pass
        out.append(row)
    return {"projects": out, "builder": _builder_url(), "now": time.time()}


# ---------------------------------------------------------------- one project
def _phases(rows: list[dict], backend: str, st: str) -> list[dict]:
    order = ROUTE_PHASES.get(backend, DEFAULT_PHASES)
    by = {k: [] for k in order}
    for r in rows:
        ph = phase_of(r["key"])
        if ph in by:
            by[ph].append(r)
        elif ph is not None and order:
            by[order[-1]].append(r)
    out = []
    for k in order:
        rs = by[k]
        cur = [r for r in rs if "earlier run" not in r["note"]]
        fore = [r for r in cur if r["key"] not in BACKGROUND]
        background = any(r["live"] for r in cur if r["key"] in BACKGROUND)
        if any(r["live"] for r in fore) or (background and not fore):
            status = "active"
        elif any(r["icon"] == "✗" for r in cur):
            status = "failed"
        elif any(r["icon"] == "■" for r in cur):
            status = "stopped"
        elif cur:
            status = "done"
        elif rs:
            status = "earlier"
        else:
            status = "waiting" if st == "running" else "not run"
        counted = [r for r in (fore or cur) if r["total"]]
        frac = None
        if counted:
            frac = sum(min(1.0, (r["done"] or 0) / r["total"]) for r in counted) / len(counted)
        out.append({"key": k, "label": PHASES[k], "status": status, "stages": len(rs), "fraction": frac,
                    "background": background and status != "active"})
    # a phase the run went past without a stage of its own (a restart from later on) is not "waiting"
    last = max((i for i, p in enumerate(out) if p["status"] not in ("waiting", "not run")), default=-1)
    for p in out[:max(last, 0)]:
        if p["status"] in ("waiting", "not run"):
            p["status"] = "skipped"
    return out


def _stage_list(rows: list[dict]) -> list[dict]:
    tone = {"✓": "ok", "✗": "bad", "!": "warn", "■": "warn", "–": "muted"}
    out = []
    for r in rows:
        stale = "earlier run" in r["note"]
        out.append({"key": r["key"], "label": r["label"], "phase": phase_of(r["key"]), "done": r["done"],
                    "total": r["total"], "elapsed": r["elapsed"], "remaining": r["remaining"], "note": r["note"],
                    "live": r["live"], "earlier": stale,
                    "tone": "live" if r["live"] else "muted" if stale else tone.get(r["icon"], "muted"),
                    "icon": r["icon"]})
    return out


def _results(cfg, P: Path) -> dict | None:
    from . import dashboard as db
    rep = _load_json(P / "deliverables" / "client_report.json")
    if not rep:
        return None
    sr, stale = db.send_rule_shown(P)
    rs = db.release_status(cfg) or {}
    hard = [b.get("text", "") for b in rs.get("blockers", []) if b.get("hard")]
    anch = db.anchored_of(rep)
    return {
        "send_rule": sr, "hold": sr == db.HOLD, "out_of_date": stale, "verdict": rep.get("verdict"),
        "final": bool(rs.get("final")), "reviewed_by": rs.get("reviewed_by"), "blockers": hard,
        "frame": ("Vertical, GNSS-anchored" if anch else "Vertical, relative to farmland (not GNSS-anchored)"
                  if anch is False else "Vertical") + "; likely range is 95%",
        "areas": _areas_brief(rep),
        "features": [{"text": t.strip(), "tone": _tone(s)} for t, s in db.feature_lines(rep)],
        "generated": rep.get("generated", ""),
        "has_report": (P / "deliverables" / "client_report.html").exists(),
        "has_points": _points_file(cfg).exists(),
    }


def _combined_viewer_ready(cc) -> bool:
    from . import combine_viewer as cv
    return cv.viewer_path(cc).exists()


def _combined_project(config: str, cfg) -> dict:
    """A combined project's page: its two sources, whether it is current,
    the readings, its layers. No stages: it is one step (Combine)."""
    from . import combine
    cc = cfg.cc
    rec = combine.read_report(config)
    why = combine.stale_reason(config)
    notes = [{"tone": "info", "text": f"Combines {_name(str(cc.ascending))} (ascending pass) and "
                                      f"{_name(str(cc.descending))} (descending pass). It reads both and never "
                                      "changes them."}]
    if rec is None:
        notes.append({"tone": "warn", "text": "Not combined yet: press Start, then Combine."})
    elif why:
        notes.append({"tone": "warn", "text": f"Out of date: {why}. Press Start, then Combine."})
    notes.append({"tone": "info", "text": "The vertical decides. East-west is indicative (not a tested figure) "
                                          "until its re-test is signed off."})
    head = {"name": _name(config), "config": str(config), "dir": str(cc.project_dir), "backend": "combined",
            "route": "combined ascending + descending", "state": "done" if rec else "idle",
            "state_words": "done" if rec else "idle", "notes": notes, "bbox": list(cfg.bbox),
            "sources": [{"name": _name(str(cc.ascending)), "direction": "ascending"},
                        {"name": _name(str(cc.descending)), "direction": "descending"}]}
    results = None
    if rec:
        g = rec.get("comparison_ground") or {}
        results = {"verdict": None, "final": None, "send_rule": None, "out_of_date": bool(why), "hold": False,
                   "blockers": [], "combined": True,
                   "frame": ("Vertical from both passes together" + (", GNSS-anchored" if rec.get("anchored_to_gnss")
                                                                      else ", relative")
                             + f"; relative to {g.get('words') or 'the surrounding ground'}; likely range is 95%; "
                               "east-west indicative"),
                   "areas": combine.brief(rec),
                   "features": [{"text": f"{a['name']}: {a.get('sentence')}"
                                 + (f" {a['note']}" if a.get("note") else ""), "tone": "muted"}
                                for a in rec.get("areas") or []],
                   "generated": rec.get("generated", ""), "has_report": False,
                   "has_points": _combined_viewer_ready(cc)}
    return {"head": head, "phases": [], "stages": [], "results": results, "layers": layer_list(cfg),
            "now": time.time()}


def api_project(body: dict) -> dict:
    from . import areas, dashboard as db, progress, runs
    config = body["config"]
    cfg = _load_cfg(config)
    if getattr(cfg, "_combined", False):
        return _combined_project(str(Path(config).resolve()), cfg)
    P = Path(cfg.project_dir)
    now = time.time()
    rec = runs.record(P)
    st = runs.state(P)
    L = progress.latest(P)
    res = db.result_for_run(rec, runs.result(P))
    rows = db.stage_rows(L, rec, st, now)
    head = {"name": _name(config), "config": str(config), "dir": str(P), "backend": cfg.backend,
            "route": areas.ROUTE_NAMES.get(cfg.backend, cfg.backend), "state": st, "state_words": st,
            "notes": [], "bbox": list(cfg.bbox)}
    if st == "running":
        held = runs.lock_holder(P) or {}
        head["state_words"] = db.LOCK_KIND_WORDS.get(held.get("kind"), st)
    if rec:
        end = db.run_end(rec, st, res, L, now)
        head.update({"kind": rec.get("kind"), "pid": rec.get("pid"), "started": _when(rec.get("started")),
                     "took": db.fmt_dur((end or now) - rec["started"]) if rec.get("started") else ""})
    if st == "interrupted":
        try:
            info = runs.interruption(P, cfg) or {}
        except Exception:                                     # noqa: BLE001
            info = {}
        if info.get("text"):
            head["notes"].append({"tone": "warn", "text": info["text"]})
    elif res and st in ("done", "failed", "stopped"):
        if res.get("plain_cause"):
            head["notes"].append({"tone": "bad" if st == "failed" else "info", "text": res["plain_cause"]})
        if res.get("next_action"):
            head["notes"].append({"tone": "info", "text": "What now: " + str(res["next_action"])})
        wl = db.warnings_line(res)
        if wl:
            head["notes"].append({"tone": "warn", "text": wl})
    try:
        others = runs.live_runs(exclude=P)
    except Exception:                                         # noqa: BLE001
        others = []
    if others:
        head["notes"].append({"tone": "warn", "text": "Also running on this computer: " + "; ".join(
            f"{Path(o['project']).name} ({o.get('kind') or 'run'})" for o in others)
            + ". Runs share the cores, memory and disk; each sizes itself as if alone."})
    if not P.exists():
        kept = kept_records([_name(config), P.name])
        head["notes"].append({"tone": "warn", "text": "This project's folder is not on disk any more"
                              + (": it was deleted after its run. Kept: " + "; ".join(k["what"] for k in kept) + "." if kept else ".")
                              + " Start runs it again from the beginning."})
    if cfg.backend == "disp":
        head["notes"].append({"tone": "info", "text": "DISP-S1 screening: quick, not official; it writes no "
                                                      "client report unless disp_client_report is set."})
    try:
        from . import combine
        cs = combine.status_for(cfg)
    except Exception:                                         # noqa: BLE001 - the page works without it
        cs = None
    if cs:
        head["combine"] = {k: cs.get(k) for k in ("kind", "name", "config", "direction", "status")}
        if cs["kind"] == "combined":
            head["combine"]["name"] = _name(cs["config"])       # the page links it: "Also measured from …"
        else:
            head["notes"].append({"tone": "info", "text": f"{cs['name']} looks at the same ground from the "
                                                          f"{cs['direction']} pass: the two can be combined into "
                                                          "vertical and east-west movement."})
    results = _results(cfg, P)
    if results is not None:
        results["signoff"] = _signoff_job(config)
    return {"head": head, "phases": _phases(rows, cfg.backend, st), "stages": _stage_list(rows),
            "results": results, "layers": layer_list(cfg), "now": now}


def api_resources(body: dict) -> dict:
    where = None
    if body.get("config"):
        where = str(_load_cfg(body["config"]).project_dir)
    r = _RESOURCES.sample(where)
    r["net_hist"] = list(_RESOURCES.net_hist)
    return r


class _Res:
    """dashboard.Resources, built on first use (it samples the network between calls)."""
    _r = None

    def __getattr__(self, k):
        if _Res._r is None:
            from .dashboard import Resources
            _Res._r = Resources()
        return getattr(_Res._r, k)


_RESOURCES = _Res()


def api_log(body: dict) -> dict:
    """The run's log from byte `pos` (the last 200 lines when pos is absent or
    the log changed), following the running step's own log as the dashboard does."""
    from . import dashboard as db, progress, runs
    cfg = _load_cfg(body["config"])
    P = Path(cfg.project_dir)
    rec = runs.record(P)
    st = runs.state(P)
    sub = db.sublog_of(progress.latest(P), rec, st)
    lg = sub or (Path(rec["log"]) if rec and rec.get("log") else None)
    if lg is None:
        logs = sorted((P / "logs").glob("*.log"), key=lambda f: f.stat().st_mtime)
        lg = logs[-1] if logs else None
    if lg is None or not lg.exists():
        return {"path": None, "lines": [], "pos": 0, "reset": True}
    size = lg.stat().st_size
    pos = body.get("pos")
    if body.get("path") != str(lg) or not isinstance(pos, int) or pos > size:
        return {"path": str(lg), "lines": runs.tail(lg, 200), "pos": size, "reset": True, "sub": sub is not None}
    if size == pos:
        return {"path": str(lg), "lines": [], "pos": pos, "reset": False, "sub": sub is not None}
    with open(lg, "rb") as fh:
        fh.seek(pos)
        chunk = fh.read(min(size - pos, 512_000))
    return {"path": str(lg), "lines": chunk.decode(errors="replace").splitlines()[-400:], "pos": pos + len(chunk),
            "reset": False, "sub": sub is not None}


# ---------------------------------------------------------------- starting and stopping
def _kinds(backend: str) -> list[str]:
    from . import runs
    out = []
    for k in runs.KINDS:
        if k in ("miaplpy", "reunwrap") and backend != "miaplpy":
            continue
        if k in ("mintpy", "plan") and backend == "disp":
            continue
        out.append(k)
    return out


def api_run_options(body: dict) -> dict:
    """What can be started here: the recommended restart first (runs.suggest_kind),
    the rest as advanced, which of them go through the preflight checks."""
    from . import dashboard as db, runs
    cfg = _load_cfg(body["config"])
    if getattr(cfg, "_combined", False):
        return {"options": [{"kind": "combine", "title": "Combine",
                             "detail": "Read both projects and build the vertical and east-west view (a minute "
                                       "or two). Neither project is changed.",
                             "recommended": True, "preflight": False, "warning": ""}],
                "reason": "", "running": False}
    P = Path(cfg.project_dir)
    try:
        rec = runs.suggest_kind(P, cfg)
    except Exception:                                         # noqa: BLE001 - the list still works
        rec = None
    n_clean = db.cleaned_dates(cfg)
    rk = rec[0] if rec and rec[0] in runs.KINDS else None
    opts = []
    for k in _kinds(cfg.backend):
        if rk and k == "resume":
            continue                       # Resume works out the same restart as the recommendation
        label, _ = runs.KINDS[k]
        title, _, detail = label.partition(": ")
        opts.append({"kind": k, "title": title, "detail": detail, "recommended": k == rk,
                     "preflight": k in db.PREFLIGHT_KINDS,
                     "warning": (f"Whole pipeline goes through every stage again. It reuses this project's plan and "
                                 f"skips the {n_clean} dates already coregistered and cleaned after the merge, but "
                                 "checks the archive first and refuses to start if acquisitions are missing there. "
                                 "Use the recommendation unless you mean that.") if k == "full" and n_clean else ""})
    opts.sort(key=lambda o: not o["recommended"])
    return {"options": opts, "reason": rec[1] if rec else "", "running": runs.state(P) == "running"}


def api_preflight(body: dict) -> dict:
    from . import dashboard as db
    config, kind = body["config"], body.get("kind", "full")
    cfg = _load_cfg(config)
    plan: list[str] = []
    try:
        res = db.preflight_checks(cfg, config, kind, say=plan.append)
    except Exception as exc:                                  # noqa: BLE001
        res = [("fail", f"the checks themselves failed: {exc}")]
    ok = not any(level == "fail" for level, _ in res)
    with _PREFLIGHT_LOCK:
        _PREFLIGHT[(str(Path(config).resolve()), kind)] = (time.time(), ok)
    return {"ok": ok, "checks": [{"level": lv, "text": t} for lv, t in res], "plan": plan[-200:]}


def api_start(body: dict) -> dict:
    from . import dashboard as db, runs
    config, kind = body["config"], body.get("kind")
    if kind == "combine":
        cfg = _load_cfg(config)
        if not getattr(cfg, "_combined", False):
            raise ValueError("Combine applies to a combined project only.")
        from . import combine
        with _COMBINE_LOCK:
            combine.run(str(Path(config).resolve()))
        return {"started": "combine", "pid": None, "finished": True}
    if kind not in runs.KINDS:
        raise ValueError(f"unknown run kind {kind!r}")
    cfg = _load_cfg(config)
    if kind not in _kinds(cfg.backend):
        raise ValueError(f"{kind!r} does not apply to the {cfg.backend} route")
    P = Path(cfg.project_dir)
    if runs.state(P) == "running":
        raise ValueError("A run is already going here.")
    if kind in db.PREFLIGHT_KINDS:
        with _PREFLIGHT_LOCK:
            t, ok = _PREFLIGHT.get((str(Path(config).resolve()), kind), (0, False))
        if not ok or time.time() - t > PREFLIGHT_TTL_S:
            raise ValueError("Run the preflight checks for this run first (they pass, then Start).")
    if kind == "full" and db.cleaned_dates(cfg) and not body.get("confirm_cleaned"):
        raise ValueError("Whole pipeline on a project with cleaned dates needs the confirmation box ticked.")
    r = runs.start(str(Path(config).resolve()), P, kind)
    with _PREFLIGHT_LOCK:
        _PREFLIGHT.pop((str(Path(config).resolve()), kind), None)    # one passing check, one start
    return {"started": kind, "pid": r.get("pid")}


_COMBINE_LOCK = threading.Lock()


def api_combine_with(body: dict) -> dict:
    """Combine a project with the partner the page offered (combine.status_for):
    writes configs/<name>_combined.yaml and builds it. Only the offered pair."""
    from . import combine
    cfg = _load_cfg(body["config"])
    offer = combine.status_for(cfg) or {}
    if offer.get("kind") != "offer" or str(Path(offer["config"]).resolve()) != str(Path(body.get("partner", "")).resolve()):
        raise ValueError("That pair is not on offer here any more: reload the page.")
    with _COMBINE_LOCK:
        path = combine.combine_with(str(Path(body["config"]).resolve()), offer["config"])
    return {"config": str(path), "name": _name(str(path))}


def api_stop(body: dict) -> dict:
    from . import runs
    cfg = _load_cfg(body["config"])
    P = Path(cfg.project_dir)
    if runs.state(P) != "running":
        return {"result": "nothing running"}
    held = runs.lock_holder(P)
    if held and held.get("pid") == os.getpid():
        raise ValueError("No run to stop: this app itself holds the project (a report rebuild); wait for it.")
    return {"result": runs.stop(P)}


# ---------------------------------------------------------------- sign-off (30 Oct 2026)
# The terminal dashboard's Sign off in the app, through the same functions:
# api_signoff_form says what the reviewer must settle (dashboard.SIGNOFF_CLEARS
# and its helpers), api_signoff runs dashboard.sign_off_and_release. The
# sign-off itself (release.sign_off: the reviewer's name, every feature ticked,
# a named specialist for a held report, bound to the numbers) is answered at
# once; the report rebuild and the client package after it take minutes, so
# they go on in a thread the project page follows (api_project's
# results.signoff).
_SIGNOFF: dict = {}           # resolved config -> {state, steps, message, final, package, started}
_SIGNOFF_LOCK = threading.Lock()
SIGNOFF_WAIT_S = 60           # for release.sign_off's answer; the rebuild is not waited for


def _signoff_job(config: str) -> dict | None:
    with _SIGNOFF_LOCK:
        j = _SIGNOFF.get(str(Path(config).resolve()))
        return {**j, "steps": list(j["steps"])} if j else None


def _signoff_busy() -> bool:
    with _SIGNOFF_LOCK:
        return any(j["state"] == "working" for j in _SIGNOFF.values())


def api_signoff_form(body: dict) -> dict:
    """What signing off this project's current report asks for: the features
    to tick, the hold for a specialist, the hard blockers a sign-off does not
    settle, and the preparer's name as the reviewer's default."""
    from . import dashboard as db, release, runs, settings
    cfg = _load_cfg(body["config"])
    if getattr(cfg, "_combined", False):
        raise ValueError("A combined project has no client report to sign off.")
    P = Path(cfg.project_dir)
    rep = _load_json(P / "deliverables" / "client_report.json")
    if not rep:
        raise ValueError("No client report to sign off yet.")
    blockers = (db.release_status(cfg) or {}).get("blockers", [])
    return {"reviewer": settings.identity().get("preparer", ""),
            "features": [{"n": f["n"], "label": f["label"]} for f in release.features(rep)],
            "held": db.signoff_hold(blockers), "other_blockers": db.signoff_leaves(blockers),
            "running": runs.state(P) == "running", "job": _signoff_job(body["config"])}


def api_signoff(body: dict) -> dict:
    """Sign off the current report, then rebuild it and build the client
    package when final (dashboard.sign_off_and_release). A sign-off that
    release.sign_off refuses raises here, with nothing recorded; once it is
    recorded the answer is the job, and the page follows the rest."""
    from . import dashboard as db, runs
    config = str(Path(body["config"]).resolve())
    cfg = _load_cfg(config)
    if getattr(cfg, "_combined", False):
        raise ValueError("A combined project has no client report to sign off.")
    if runs.state(cfg.project_dir) == "running":
        raise ValueError("Wait for the run to finish before signing off.")
    feats = body.get("features") or []
    if not isinstance(feats, list) or not all(isinstance(n, int) and not isinstance(n, bool) for n in feats):
        raise ValueError("features must be a list of feature numbers")
    data = {"reviewer": str(body.get("reviewer") or "").strip()[:120],
            "notes": str(body.get("notes") or "").strip()[:2000], "features": feats}
    specialist = str(body.get("specialist") or "").strip()[:120]
    if specialist:
        data["specialist"] = specialist
    with _SIGNOFF_LOCK:
        if (_SIGNOFF.get(config) or {}).get("state") == "working":
            raise ValueError("A sign-off is already going on this project.")
        job = {"state": "working", "steps": [], "message": "", "final": False, "package": None,
               "started": time.time()}
        _SIGNOFF[config] = job
    answered = threading.Event()

    def say(msg):
        with _SIGNOFF_LOCK:
            job["steps"].append(str(msg))
        answered.set()                 # its first line comes once the sign-off is recorded

    def work():
        try:
            out = db.sign_off_and_release(cfg, config, data, say=say, rebuild=db.rebuild_report)
            upd = {"state": "done", "message": out["message"], "final": out["final"], "package": out["package"]}
        except Exception as exc:                              # noqa: BLE001 - said on the page
            upd = {"state": "failed", "message": (f"The sign-off is recorded, but: {exc}" if job["steps"]
                                                  else f"Not signed off: {exc}")}
        finally:
            db._RELEASE_CACHE.clear()
        with _SIGNOFF_LOCK:
            job.update(upd)
        answered.set()

    threading.Thread(target=work, name="signoff", daemon=True).start()
    answered.wait(SIGNOFF_WAIT_S)
    with _SIGNOFF_LOCK:
        if job["state"] == "failed" and not job["steps"]:
            _SIGNOFF.pop(config, None)                        # nothing recorded: nothing to follow
            raise ValueError(job["message"].removeprefix("Not signed off: "))
    return {"job": _signoff_job(config)}


def api_open(body: dict) -> dict:
    """Open the report, the project folder or the log on this computer (Windows
    Explorer from WSL). Only these three paths, never one the page names."""
    from . import dashboard as db, runs
    cfg = _load_cfg(body["config"])
    P = Path(cfg.project_dir)
    what = body.get("what")
    if what == "report":
        f = P / "deliverables" / "client_report.html"
    elif what == "folder":
        f = P
    elif what == "deliverables":
        f = P / "deliverables"
    elif what == "log":
        rec = runs.record(P) or {}
        f = Path(rec["log"]) if rec.get("log") else None
    else:
        raise ValueError(f"cannot open {what!r}")
    if f is None or not f.exists():
        raise ValueError("That is not there yet.")
    return {"opened": db.open_path(f)}


def api_open_kept(body: dict) -> dict:
    """Open one of a removed project's kept records (kept_records), by its index."""
    from . import dashboard as db
    c = str(Path(body["config"]).resolve())
    from . import runs
    if c not in {str(x) for x in runs.known_configs()}:
        raise ValueError("not one of this computer's projects")
    p = runs.project_summary(c)
    folder = Path(p.project_dir) if p.project_dir else None
    kept = [k for k in kept_records([p.name, folder.name if folder else ""]) if k["path"]]
    i = int(body.get("index", 0))
    if not 0 <= i < len(kept):
        raise ValueError("no such record")
    f = Path(kept[i]["path"])
    report = next(iter(sorted(f.glob("*/validation.md"))), None)      # the report itself, when there is one
    return {"opened": db.open_path(report or f)}


def api_forget(body: dict) -> dict:
    """Take a project off the remembered list (runs.forget). The yaml is not
    touched; one that lives in a configs folder the pipeline reads stays listed."""
    from . import runs
    c = str(Path(body["config"]).resolve())
    hit = runs.forget(c)
    still = c in {str(x) for x in runs.known_configs()}
    return {"forgot": hit, "still_listed": still}


def api_viewer(body: dict) -> dict:
    from . import results_viewer
    url = results_viewer.serve()
    if body.get("config"):
        url += "#project=" + _name(body["config"])
    return {"url": url}


def _builder_url() -> str:
    from . import map_builder
    return map_builder.serve()


def api_builder(_: dict) -> dict:
    return {"url": _builder_url()}


def api_storage(_: dict) -> dict:
    """storage.inventory as plain rows (read-only here: freeing disk is done
    with `insar storage` or the dashboard's g, which show what goes first)."""
    from . import storage
    inv = storage.inventory(sizes=True)
    rows = []
    for p in inv.get("projects", []):
        d = p.__dict__ if hasattr(p, "__dict__") else dict(p)
        rows.append({k: v for k, v in d.items() if isinstance(v, (str, int, float, bool, type(None)))})
    return {"projects": rows, "overview": storage.overview_text(inv)}


def api_quit(_: dict) -> dict:
    """Stop this app's server (runs carry on: they are detached; a sign-off's
    report rebuild runs in this app, so it is waited for)."""
    if _signoff_busy():
        raise ValueError("A sign-off is rebuilding a report in this app: quit when it has finished.")
    threading.Timer(0.3, _STOP.set).start()     # after this answer has gone out
    return {"quitting": True}


# ---------------------------------------------------------------- the live map
def _points_file(cfg) -> Path:
    from . import field_points as fp
    return fp.viewer_path(cfg)


# The rasters the map can show: the client package's own files (package.DATA_FILES
# says what each means; the line-of-sight ones are in the report's frame, relative
# to ordinary ground), then the technical ones. While a run is still in its time
# series, before the products exist, MintPy's geocoded files stand in for two of
# them, labelled raw. The raw los_velocity_in_per_year.tif is never shown: ordinary
# farmland reads about -0.3 in/yr in it (package.py), against ~0 in the client raster.
#   style: div (two-sided, red = down / away), seq (one-sided), mask (1 = flagged), cat (classes)
RASTERS = (
    {"id": "vertical", "label": "Vertical movement", "group": "Movement", "units": "in/yr", "style": "div",
     "files": ("deliverables/vertical_movement_in_per_yr.tif",), "after": "Products & report"},
    {"id": "vertical_change", "label": "Vertical change at the average rate", "group": "Movement", "units": "in",
     "style": "div", "files": ("deliverables/vertical_change_at_average_rate_in_*.tif",
                               "deliverables/vertical_movement_total_in_*.tif"), "after": "Products & report"},
    {"id": "los_velocity", "label": "Line-of-sight velocity", "group": "Movement", "units": "in/yr", "style": "div",
     "files": ("deliverables/line_of_sight_velocity_in_per_yr.tif",), "mintpy": ("geo/geo_velocity.h5", "velocity", M_IN),
     "after": "Time series"},
    {"id": "los_change", "label": "Line-of-sight change, first to last image", "group": "Movement", "units": "in",
     "style": "div", "files": ("deliverables/line_of_sight_cumulative_displacement_in_*.tif",), "after": "Products & report"},
    {"id": "snr", "label": "Signal-to-noise", "group": "Quality", "units": "movement / noise", "style": "seq",
     "range": (0, 3), "files": ("deliverables/rate_signal_to_noise.tif",), "after": "Products & report"},
    {"id": "vertical_unc", "label": "Vertical uncertainty (1 sigma)", "group": "Quality", "units": "in/yr", "style": "seq",
     "files": ("deliverables/vertical_movement_uncertainty_1sigma_in_per_yr.tif",), "after": "Products & report"},
    {"id": "coherence", "label": "Measurement quality (temporal coherence)", "group": "Quality", "units": "0 to 1",
     "style": "seq", "range": (0, 1), "files": ("deliverables/quality_temporal_coherence.tif",),
     "mintpy": ("geo/geo_temporalCoherence.h5", "temporalCoherence", 1.0), "after": "Time series"},
    {"id": "unwrap", "label": "Unwrapping consistency", "group": "Quality", "units": "0 to 1", "style": "seq",
     "range": (0, 1), "files": ("deliverables/quality_unwrapping_consistency.tif",), "after": "Products & report"},
    {"id": "too_fast", "label": "Too fast to measure", "group": "Quality", "units": "", "style": "mask",
     "files": ("deliverables/quality_too_fast_to_measure.tif",), "after": "Products & report"},
    {"id": "position_unc", "label": "Position uncertainty", "group": "Quality", "units": "m", "style": "seq",
     "files": ("deliverables/quality_position_uncertainty_m.tif",), "after": "Products & report"},
    {"id": "dem_error", "label": "Elevation-model error estimate", "group": "Technical", "units": "m", "style": "div",
     "files": ("deliverables/dem_difference_m.tif",), "after": "Products & report"},
    {"id": "landcover", "label": "Land cover (ESA WorldCover)", "group": "Context", "units": "", "style": "cat",
     "files": ("context/landcover_worldcover.tif",), "after": "Products & report"},
    # a combined project's own (combine.py: ascending + descending); only it lists them
    {"id": "combined_vertical", "label": "Vertical movement, both directions", "group": "Combined", "units": "in/yr",
     "style": "div", "files": ("deliverables/vertical_movement_combined_in_per_yr.tif",), "after": "Combine",
     "combined": True},
    {"id": "combined_east", "label": "East-west movement (indicative)", "group": "Combined", "units": "in/yr",
     "style": "div_ew", "files": ("deliverables/east_west_movement_INDICATIVE_in_per_yr.tif",), "after": "Combine",
     "combined": True},
    {"id": "combined_vertical_unc", "label": "Vertical uncertainty, both directions (1 sigma)", "group": "Combined",
     "units": "in/yr", "style": "seq", "files": ("deliverables/vertical_movement_combined_uncertainty_1sigma_in_per_yr.tif",),
     "after": "Combine", "combined": True},
)
RASTER = {r["id"]: r for r in RASTERS}
# A study's before/after change map (events.write_event_products, A3 of
# review_2026-11-07/PLAN_oil_gas_slopes.md): line-of-sight change in mm between
# the window's two dates, relative to ordinary ground. Listed only when the
# project has made one (never pending), so a project without event products
# lists exactly what it did before.
EVENT_RASTER = {"id": "los_event_change", "label": "Change between two dates (screening)", "group": "Study results",
                "units": "mm", "style": "div", "files": ("deliverables/events/los_change_mm_*.tif",),
                "after": "Products & report"}
RASTER[EVENT_RASTER["id"]] = EVENT_RASTER
# a study's single-file layer fields (areas._write_study_layers) and their geometry
STUDY_FIELDS = (("event_polygons", "polygon"), ("event_lines", "line"), ("fast_zone_geojson", "polygon"),
                ("reference_stable_geojson", "polygon"))
# a well's reading in the report (client_report.study_point_entries): key, words, colour
WELL_STATUS = (("moving", "On moving ground (detected)", "#e03131"),
               ("not_distinguishable", "Not distinguishable from ordinary ground", "#f2f2f2"),
               ("not_measurable", "Not measurable here", "#8c8c8c"),
               ("no_number", "No number (too fast for this satellite)", "#000000"),
               ("no_report", "No number yet (the products run makes it)", "#bdbdbd"))
# ESA WorldCover 2021 classes and their published colours
WORLDCOVER = {10: ("Tree cover", "#006400"), 20: ("Shrubland", "#ffbb22"), 30: ("Grassland", "#ffff4c"),
              40: ("Cropland", "#f096ff"), 50: ("Built-up", "#fa0000"), 60: ("Bare / sparse", "#b4b4b4"),
              70: ("Snow and ice", "#f0f0f0"), 80: ("Water", "#0064c8"), 90: ("Herbaceous wetland", "#0096a0"),
              95: ("Mangroves", "#00cf75"), 100: ("Moss and lichen", "#fae6a0")}


def raster_file(cfg, lid: str) -> tuple[Path | None, bool]:
    """(the file for raster layer lid, whether it is MintPy's raw stand-in)."""
    spec = RASTER[lid]
    P = Path(cfg.project_dir)
    for pat in spec["files"]:
        hits = sorted(P.glob(pat), key=lambda f: f.stat().st_mtime)
        if hits:
            return hits[-1], False
    if spec.get("mintpy"):
        f = Path(cfg.mintpy_dir) / spec["mintpy"][0]
        if f.exists():
            return f, True
    return None, False


def _about(name: str) -> str:
    """What the client package says a product file is (package.DATA_FILES / TECH_FILES)."""
    try:
        from . import package
        for rx, words in tuple(package.DATA_FILES) + tuple(package.TECH_FILES):
            if rx.match(name):
                return words
    except Exception:                                         # noqa: BLE001 - no description
        pass
    return ""


def _outline_files(cfg) -> list[tuple[str, str, str, Path]]:
    """(id, label, style, path) of the config's own polygon files."""
    out = []
    P = Path(getattr(cfg, "_outline_dir", None) or cfg.project_dir)
    aoi = P / "context" / "area_of_interest.geojson"
    if aoi.exists():
        out.append(("aoi", "Area of interest", "aoi", aoi))
    for key, label in (("underground_mines_geojson", "Underground mines"), ("surface_mines_geojson", "Surface mines"),
                       ("other_movement_geojson", "Other known movement")):
        v = getattr(cfg, key, "") or ""
        if v and Path(v).exists():
            out.append((key.replace("_geojson", ""), label, "mine", Path(v)))
    from .config import split_area_spec
    for i, spec in enumerate(getattr(cfg, "report_areas", ()) or ()):
        try:
            name, kind, val = split_area_spec(spec)
        except ValueError:
            continue
        if kind == "geojson" and Path(val).exists():
            out.append((f"report_{i}", f"Report area: {name}", "report", Path(val)))
    return out


def _short(label: str) -> str:
    """A layer kind's label for the map: 'Wells (status, plug date, type by column)' -> 'Wells'."""
    return label.split(" (", 1)[0]


def study_layers(cfg) -> list[dict]:
    """The project's study layers from its config (feature_layers, event_polygons,
    event_lines, fast_zone_geojson, reference_stable_geojson): [{id, label,
    geometry, style, kind, files}], labelled from the study's layer kinds
    (domains.py). [] when the config names none (a default-study project)."""
    from . import domains
    try:
        st = domains.study_of(cfg)
    except KeyError:
        st = domains.study(domains.DEFAULT_STUDY)
    by_field = {L.config_field: L for L in st.layer_kinds if L.config_field != "feature_layers"}
    by_kind = {L.key: L for L in st.layer_kinds}
    out: dict = {}

    def add(key, label, geometry, field, f):
        if not f or not Path(f).exists():
            return
        style = ("study_point" if geometry == "point" else "study_line" if geometry == "line"
                 else "fast_zone" if field == "fast_zone_geojson" else "stable" if field == "reference_stable_geojson"
                 else "study_poly")
        row = out.setdefault(f"study_{key}", {"id": f"study_{key}", "label": label, "geometry": geometry,
                                              "style": style, "kind": key, "files": []})
        row["files"].append(Path(f))
    for spec in getattr(cfg, "feature_layers", ()) or ():
        try:
            k, f = domains.split_feature_layer(spec)
        except ValueError:
            continue
        L = by_kind.get(k)
        add(k, _short(L.label) if L else k.replace("_", " ").capitalize(), L.geometry if L else "point",
            "feature_layers", f)
    for field, geometry in STUDY_FIELDS:
        L = by_field.get(field)
        add(L.key if L else field, _short(L.label) if L else field.replace("_geojson", "").replace("_", " ").capitalize(),
            L.geometry if L else geometry, field, getattr(cfg, field, "") or "")
    return list(out.values())


def well_status(row: dict | None) -> str:
    """A study point's reading (client_report.study_point_entries): detected
    moving, not distinguishable, not measurable, or no number (refused)."""
    if row is None:
        return "no_report"
    if row.get("refused"):
        return "no_number"
    if row.get("relative_vertical_mm_yr") is None:
        return "not_measurable"
    return "moving" if row.get("detected") is True else "not_distinguishable"


def _study_points(cfg, L: dict) -> dict:
    """A study's point layer (wells) for the map: each point with its reading
    from client_report.json's study_points when the products have run, else
    as listed in its file (no number yet)."""
    from . import events
    P = Path(cfg.project_dir)
    rep = _load_json(P / "deliverables" / "client_report.json") or {}
    rows = [r for r in rep.get("study_points") or [] if isinstance(r, dict) and r.get("layer") == L["kind"]]
    words = {k: w for k, w, _ in WELL_STATUS}
    pts = []
    if rows:
        for r in rows:
            if r.get("lat") is None:
                continue
            st = well_status(r)
            v, rng = r.get("relative_vertical_mm_yr"), r.get("range95_mm_yr")
            pts.append({"id": r.get("id"), "lat": r["lat"], "lon": r["lon"], "status": st, "status_words": words[st],
                        "attributes": r.get("attributes") or {}, "value_mm_yr": v,
                        "value_in_yr": round(v / MM_IN, 3) if v is not None else None,
                        "range_mm_yr": rng, "range_in_yr": [round(x / MM_IN, 3) for x in rng] if rng else None,
                        "shared_with": r.get("shared_with") or [], "tier": r.get("tier"),
                        "reason": r.get("refused_reason") or r.get("not_testable_reason") or ""})
    else:
        attrs = ()
        try:
            from . import domains
            attrs = next((k.attributes for k in domains.study_of(cfg).layer_kinds if k.key == L["kind"]), ())
        except KeyError:
            pass
        for p in events.study_points([(L["kind"], str(f)) for f in L["files"]]):
            a = p.get("attributes") or {}
            pts.append({"id": p["id"], "lat": p["lat"], "lon": p["lon"], "status": "no_report",
                        "status_words": words["no_report"],
                        "attributes": {k: (None if a.get(k) in (None, "") else str(a[k])) for k in attrs},
                        "value_mm_yr": None, "value_in_yr": None, "range_mm_yr": None, "range_in_yr": None,
                        "shared_with": [], "tier": None, "reason": ""})
    seen = {p["status"] for p in pts}
    return {"kind": "study_points", "points": pts, "from_report": bool(rows),
            "legend": {"classes": [{"name": w, "color": c} for k, w, c in WELL_STATUS if k in seen], "units": ""}}


def _study_geojson(files: list) -> dict:
    feats = []
    for f in files:
        feats += _geojson(Path(f))["features"]
    return {"type": "FeatureCollection", "features": feats}


def layer_list(cfg) -> list[dict]:
    """Every layer the live map can show, in drawing order, with whether it
    is there yet and, if not, which part of the run makes it."""
    from .config import split_area_spec
    P = Path(cfg.project_dir)
    comb = bool(getattr(cfg, "_combined", False))
    L = [{"id": "bbox", "label": "Study box (what gets processed)", "group": "Project", "ready": True,
          "style": "bbox", "since": _when(_mtime(getattr(cfg, "_source_path", "") or P))}]
    for lid, label, style, f in _outline_files(cfg):
        L.append({"id": lid, "label": label, "group": "Project", "ready": True, "style": style,
                  "since": _when(_mtime(f))})
    for i, spec in enumerate(getattr(cfg, "report_areas", ()) or ()):
        try:
            name, kind, _ = split_area_spec(spec)
        except ValueError:
            continue
        if kind == "box":
            L.append({"id": f"report_{i}", "label": f"Report area: {name}", "group": "Project", "ready": True,
                      "style": "report", "since": ""})
    for sl in study_layers(cfg):                             # a study's own layers (wells, pads, faults, ...)
        L.append({"id": sl["id"], "label": sl["label"], "group": "Study layers", "ready": True, "style": sl["style"],
                  "since": _when(max(_mtime(f) or 0 for f in sl["files"]))})
    if comb:
        for spec in RASTERS:
            if spec.get("combined"):
                f, _ = raster_file(cfg, spec["id"])
                L.append({"id": spec["id"], "label": spec["label"], "group": spec["group"], "ready": f is not None,
                          "style": "raster", "since": _when(_mtime(f)) if f else "", "after": spec["after"],
                          "source": "combined products" if f else "", "about": ""})
        return L
    ff = P / "slc" / "frame_footprint.json"
    L.append({"id": "frame", "label": "Satellite frame (the bursts found)", "group": "Data", "ready": ff.exists(),
              "style": "frame", "since": _when(_mtime(ff)), "after": "Get the data",
              # the data search writes it; a project downloaded before it did never will
              "note": "" if ff.exists() or not any((P / "slc").glob("*.SAFE")) else
              "not recorded by this project's download (an older run)"})
    for spec in RASTERS:
        if spec.get("combined"):
            continue
        f, raw = raster_file(cfg, spec["id"])
        L.append({"id": spec["id"], "label": spec["label"] + (" (raw, not yet referenced)" if raw else ""),
                  "group": spec["group"], "ready": f is not None, "style": "raster", "since": _when(_mtime(f)) if f else "",
                  "after": spec["after"], "source": ("MintPy, before the products" if raw else "products") if f else "",
                  "about": _about(f.name) if f is not None and not raw else ""})
    water = P / "context" / "osm_water.geojson"
    if water.exists():
        L.append({"id": "water", "label": "Water (OpenStreetMap)", "group": "Context", "ready": True, "style": "water",
                  "since": _when(_mtime(water))})
    f, _ = raster_file(cfg, EVENT_RASTER["id"])
    if f is not None:                                        # a study's before/after map, once made
        L.append({"id": EVENT_RASTER["id"], "label": EVENT_RASTER["label"], "group": EVENT_RASTER["group"], "ready": True,
                  "style": "raster", "since": _when(_mtime(f)), "after": EVENT_RASTER["after"], "source": "event products",
                  "about": f"line-of-sight change in mm between the two dates in its name ({f.name}), relative to "
                           "ordinary ground; a screening product"})
    rj = P / "deliverables" / "client_report.json"
    L.append({"id": "features", "label": "Features the report lists", "group": "Results so far",
              "ready": rj.exists(), "style": "feature", "since": _when(_mtime(rj)), "after": "Products & report"})
    pf = _points_file(cfg)
    L.append({"id": "points", "label": "Measurement points (vertical)", "group": "Results so far",
              "ready": pf.exists(), "style": "points", "since": _when(_mtime(pf)), "after": "Products & report"})
    gf = P / "deliverables" / "gnss_tie.json"
    L.append({"id": "gnss", "label": "GNSS stations", "group": "Results so far", "ready": gf.exists(),
              "style": "gnss", "since": _when(_mtime(gf)), "after": "Products & report"})
    return L


def _thin_coords(c, k: int):
    """Every k-th vertex of every ring (rings stay closed), at any nesting depth."""
    if c and isinstance(c[0], (int, float)):
        return c
    if c and isinstance(c[0], list) and c[0] and isinstance(c[0][0], (int, float)):
        if len(c) > 4 * k:
            return c[::k] + [c[0]]
        return c
    return [_thin_coords(x, k) for x in c]


def _count(c) -> int:
    if c and isinstance(c[0], (int, float)):
        return 1
    return sum(_count(x) for x in c) if c else 0


def _geojson(f: Path) -> dict:
    gj = json.loads(f.read_text())
    if gj.get("type") == "Feature":
        gj = {"type": "FeatureCollection", "features": [gj]}
    elif gj.get("type") != "FeatureCollection":
        gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": gj}]}
    feats = [ft for ft in gj.get("features") or [] if isinstance(ft, dict) and ft.get("geometry")]
    n = sum(_count(ft["geometry"].get("coordinates") or []) for ft in feats)
    k = -(-n // MAX_VERTICES) if n > MAX_VERTICES else 1
    out = []
    for ft in feats:
        g = ft["geometry"]
        names = {kk: vv for kk, vv in (ft.get("properties") or {}).items() if kk.lower() in ("name", "label")}
        out.append({"type": "Feature", "properties": names,
                    "geometry": {"type": g.get("type"), "coordinates": _thin_coords(g.get("coordinates") or [], k)}})
    return {"type": "FeatureCollection", "features": out}


def _box(w, s, e, n, name="") -> dict:
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": name},
            "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}]}


def read_raster(f: Path, lid: str, max_px: int = MAX_RASTER_PX, bounds=None):
    """(array, (west, south, east, north)) of a layer's file at most max_px on a
    side; with bounds (W, S, E, N) only that window (the export's map frame)."""
    import numpy as np
    spec = RASTER[lid]
    if f.suffix == ".tif":
        import rasterio
        from rasterio.windows import from_bounds
        with rasterio.open(f) as d:
            if str(d.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
                raise ValueError(f"{f.name} is not in latitude/longitude ({d.crs})")
            win = None
            if bounds is not None:
                win = from_bounds(*bounds, transform=d.transform).round_offsets().round_lengths()
                win = win.intersection(rasterio.windows.Window(0, 0, d.width, d.height))
            h_, w_ = (int(win.height), int(win.width)) if win is not None else (d.height, d.width)
            step = max(1, math.ceil(max(w_, h_) / max_px))
            a = d.read(1, window=win, out_shape=(max(1, math.ceil(h_ / step)), max(1, math.ceil(w_ / step))),
                       resampling=rasterio.enums.Resampling.nearest).astype("float32")
            if d.nodata is not None and not np.isnan(d.nodata):
                a[a == d.nodata] = np.nan
            b = rasterio.windows.bounds(win, d.transform) if win is not None else tuple(d.bounds)
            if spec["style"] == "cat":
                a[a == 0] = np.nan
            return a, (b[0], b[1], b[2], b[3])
    import h5py
    path, ds, scale = spec["mintpy"]
    with h5py.File(f, "r") as h:
        at = h.attrs
        if "Y_FIRST" not in at:
            raise ValueError(f"{f.name} is not geocoded yet")
        W, Lh = int(at["WIDTH"]), int(at["LENGTH"])
        step = max(1, math.ceil(max(W, Lh) / max_px))
        a = h[ds][::step, ::step].astype("float32") * scale
        x0, y0, dx, dy = (float(at[k]) for k in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP"))
    a[a == 0] = np.nan                                         # MintPy's outside-the-data fill
    return a, (x0, y0 + dy * Lh, x0 + dx * W, y0)


def colour(a, lid: str, bw: bool = False, alpha: float = 0.85):
    """(rgba array 0-1, legend) for a layer: div symmetric about 0 at its 98th
    percentile (red = down / away from the satellite), seq over its range or
    2nd-98th percentile, mask in one colour, cat in WorldCover's colours. bw:
    the same in greys (for black-and-white prints)."""
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.colors import to_rgba
    spec = RASTER[lid]
    ok = np.isfinite(a)
    st = spec["style"]
    if st == "cat":
        rgba = np.zeros(a.shape + (4,))
        classes = []
        for v, (name, col) in WORLDCOVER.items():
            m = ok & (a == v)
            if m.any():
                c = to_rgba(col)
                if bw:
                    g = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                    c = (g, g, g, 1)
                rgba[m] = (*c[:3], alpha)
                classes.append({"name": name, "color": "#%02x%02x%02x" % tuple(int(255 * x) for x in c[:3])})
        return rgba, {"classes": classes, "units": ""}
    if st == "mask":
        rgba = np.zeros(a.shape + (4,))
        m = ok & (a > 0)
        rgba[m] = (0.0, 0.0, 0.0, 0.9) if bw else (0.93, 0.2, 0.6, 0.9)
        return rgba, {"classes": [{"name": spec["label"], "color": "#000000" if bw else "#ed3399"}], "units": ""}
    if st in ("div", "div_ew"):
        lim = float(np.nanpercentile(np.abs(a[ok]), 98)) if ok.any() else 1.0
        # black and white: one straight ramp, black = down / away, white = up / towards, mid-grey ~ 0.
        # east-west (div_ew): red = east, blue = west - neither direction is "bad"
        vmin, vmax = -(lim or 1.0), (lim or 1.0)
        cmap = "gray" if bw else ("RdBu_r" if st == "div_ew" else "RdBu")
    else:
        if spec.get("range"):
            vmin, vmax = spec["range"]
        else:
            vmin, vmax = (np.nanpercentile(a[ok], [2, 98]).tolist() if ok.any() else (0.0, 1.0))
            if vmax <= vmin:
                vmax = vmin + 1
        cmap = "gray_r" if bw else "viridis"
    cm = colormaps[cmap]
    t = np.clip((np.where(ok, a, vmin) - vmin) / (vmax - vmin), 0, 1)
    rgba = cm(t)
    rgba[..., 3] = np.where(ok, alpha, 0.0)
    stops = [cm(i / 6) for i in range(7)]
    return rgba, {"min": vmin, "max": vmax, "units": spec["units"], "style": st,
                  "colors": ["#%02x%02x%02x" % tuple(int(255 * c) for c in x[:3]) for x in stops]}


def api_layer(body: dict) -> dict:
    cfg = _load_cfg(body["config"])
    P = Path(cfg.project_dir)
    lid = str(body.get("id"))
    if lid == "bbox":
        w, s, e, n = cfg.bbox
        return {"kind": "geojson", "data": _box(w, s, e, n, "Study box")}
    if lid == "frame":
        d = _load_json(P / "slc" / "frame_footprint.json") or {}
        b = d.get("bbox_wsen")
        if not b:
            raise ValueError("no satellite frame yet")
        return {"kind": "geojson", "data": _box(*b, name=f"{d.get('distinct_bursts', '?')} distinct bursts"),
                "note": f"{d.get('n_bursts', '?')} bursts over {d.get('distinct_bursts', '?')} burst positions"}
    for oid, label, _, f in _outline_files(cfg):
        if oid == lid:
            return {"kind": "geojson", "data": _geojson(f)}
    if lid.startswith("study_"):
        sl = next((x for x in study_layers(cfg) if x["id"] == lid), None)
        if sl is None:
            raise ValueError(f"no layer {lid!r}")
        if sl["geometry"] == "point":
            return _study_points(cfg, sl)
        return {"kind": "geojson", "data": _study_geojson(sl["files"])}
    if lid.startswith("report_"):
        from .config import split_area_spec
        i = int(lid.split("_", 1)[1])
        name, kind, val = split_area_spec(cfg.report_areas[i])
        if kind == "box":
            s, w, n, e = val
            return {"kind": "geojson", "data": _box(w, s, e, n, name)}
    if lid in RASTER:
        f, raw = raster_file(cfg, lid)
        if f is None:
            raise ValueError("not made yet")
        key = (str(f), lid)
        m = _mtime(f)
        hit = _RASTERS.get(key)
        if hit is None or hit[0] != m:
            from matplotlib.image import imsave
            a, (w, s, e, n) = read_raster(f, lid)
            rgba, legend = colour(a, lid)
            buf = io.BytesIO()
            imsave(buf, (rgba * 255).astype("uint8"), format="png")
            hit = _RASTERS[key] = (m, {"kind": "image", "png": base64.b64encode(buf.getvalue()).decode(),
                                       "bounds": [[s, w], [n, e]], "legend": legend, "raw": raw,
                                       "source": f.name, "made": _when(m)})
        return hit[1]
    if lid == "water":
        return {"kind": "geojson", "data": _geojson(P / "context" / "osm_water.geojson")}
    if lid == "features":
        from . import dashboard as db
        rep = _load_json(P / "deliverables" / "client_report.json") or {}
        lines = [t.strip() for t, _ in db.feature_lines(rep)][1:]
        feats = [f for f in rep.get("features") or [] if isinstance(f, dict) and f.get("lat") is not None]
        return {"kind": "features", "features": [{"lat": f["lat"], "lon": f["lon"], "n": i + 1,
                                                    "text": lines[i] if i < len(lines) else f"Feature {i + 1}"}
                                                   for i, f in enumerate(feats)]}
    if lid == "points":
        return _points(body["config"])
    if lid == "gnss":
        g = _load_json(P / "deliverables" / "gnss_tie.json") or {}
        used = set(g.get("stations_used") or [])
        st = [{"site": s.get("site"), "lat": s.get("lat"), "lon": s.get("lon"), "used": s.get("site") in used,
               "km": s.get("distance_km")} for s in g.get("stations") or [] if s.get("lat") is not None]
        return {"kind": "gnss", "stations": st, "anchored": g.get("anchored")}
    raise ValueError(f"no layer {lid!r}")


def _points(config: str) -> dict:
    """The viewer's own table (results_viewer._table): the same colour value as
    its movement layer, every n-th point past MAX_POINTS."""
    import numpy as np
    from . import results_viewer as rv
    _, fd, stale = rv._table(str(Path(config).resolve()))
    m = fd["meta"]
    val = (fd["rel_v"] if m.get("relative_colours") else fd["rate"]) / MM_IN
    n = int(fd["lat"].size)
    k = max(1, math.ceil(n / MAX_POINTS))
    lat, lon, v = fd["lat"][::k], fd["lon"][::k], val[::k]
    ok = np.isfinite(v) & np.isfinite(lat) & np.isfinite(lon)
    lim = float(np.nanpercentile(np.abs(v[ok]), 98)) if ok.any() else 1.0
    return {"kind": "points", "lat": np.round(lat[ok], 6).tolist(), "lon": np.round(lon[ok], 6).tolist(),
            "v": np.round(v[ok], 4).tolist(), "limit": lim or 1.0, "n": n, "shown": int(ok.sum()),
            "relative": bool(m.get("relative_colours")), "stale": stale}


# ---------------------------------------------------------------- files and projects
def api_files(body: dict) -> dict:
    from . import file_manager as fm
    return fm.list_dir(_load_cfg(body["config"]), body.get("rel", ""))


def api_plan_delete(body: dict) -> dict:
    from . import file_manager as fm
    return fm.plan_delete(_load_cfg(body["config"]), list(body.get("rels") or []))


def api_delete_files(body: dict) -> dict:
    from . import file_manager as fm
    return fm.delete(_load_cfg(body["config"]), list(body.get("rels") or []), body.get("confirm_name", ""))


def api_project_space(body: dict) -> dict:
    """Keep results only / delete the project: storage.py's own plans (a dry run
    unless apply). The app asks for the project's name for every delete, not
    only for one that went to a client."""
    from . import storage
    cfg = _load_cfg(body["config"])
    P = str(Path(cfg.project_dir))
    action, apply = body.get("action"), bool(body.get("apply"))
    if action == "keep":
        plan = storage.keep_results_only(P, apply=apply)
    elif action == "delete":
        name = storage.find(P)[0]
        if apply and (body.get("confirm_name") or "").strip() != name:
            return {"refused": [f"Type the project's name ({name}) to delete it."], "applied": False, "project": name}
        plan = storage.delete_project(P, archive=bool(body.get("archive", True)), apply=apply,
                                      confirm_name=body.get("confirm_name"))
    else:
        raise ValueError(f"unknown action {action!r}")
    return json.loads(json.dumps(plan, default=str))


# ---------------------------------------------------------------- map export and the company
def api_export(body: dict) -> dict:
    from . import map_export
    cfg = _load_cfg(body["config"])
    opts = {k: body.get(k) for k in ("page", "orientation", "bw", "format", "bounds", "layers", "basemap", "title",
                                     "subtitle", "notes", "logo")}
    b = opts.get("bounds")
    if b is not None and (not isinstance(b, list) or len(b) != 4 or not all(isinstance(v, (int, float)) for v in b)):
        raise ValueError("bounds must be [west, south, east, north]")
    f = map_export.export(cfg, str(Path(body["config"]).resolve()), opts)
    opened = ""
    if body.get("open", True):
        from . import dashboard as db
        opened = db.open_path(f)
    return {"file": str(f), "opened": opened, "name": f.name}


LOGO_MAX_BYTES = 5_000_000


def api_company(body: dict) -> dict:
    """Who the reports and the map exports say prepared them (settings: firm,
    preparer, contact) and the company logo the exports carry. Read, or set
    the fields given; a logo arrives as a data: URL and is kept, re-encoded as
    PNG, beside the settings file."""
    from . import settings
    upd = {k: str(body[k]).strip()[:120] for k in ("firm", "preparer", "contact") if k in body}
    if body.get("logo_data"):
        from PIL import Image
        head, _, data = str(body["logo_data"]).partition(",")
        if not head.startswith("data:image/") or ";base64" not in head:
            raise ValueError("the logo must be a PNG or JPEG image")
        raw = base64.b64decode(data)
        if len(raw) > LOGO_MAX_BYTES:
            raise ValueError("the logo is over 5 MB")
        im = Image.open(io.BytesIO(raw))
        im.load()
        if im.width > 4000 or im.height > 4000:
            im.thumbnail((4000, 4000))
        dest = settings.settings_file().parent / "company_logo.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.convert("RGBA").save(dest)
        upd["company_logo"] = str(dest)
    if body.get("remove_logo"):
        upd["company_logo"] = ""
    if upd:
        settings.save(**upd)
    cur = settings.load()
    logo = cur.get("company_logo") or ""
    out = {**settings.identity(), "logo": None}
    if logo and Path(logo).is_file():
        out["logo"] = "data:image/png;base64," + base64.b64encode(Path(logo).read_bytes()).decode()
    return out


API = {"/api/open_kept": api_open_kept, "/api/forget": api_forget, "/api/export": api_export, "/api/company": api_company,
       "/api/files": api_files, "/api/plan_delete": api_plan_delete, "/api/delete_files": api_delete_files,
       "/api/project_space": api_project_space,
       "/api/home": api_home, "/api/project": api_project, "/api/resources": api_resources, "/api/log": api_log,
       "/api/run_options": api_run_options, "/api/preflight": api_preflight, "/api/start": api_start,
       "/api/stop": api_stop, "/api/open": api_open, "/api/viewer": api_viewer, "/api/builder": api_builder,
       "/api/storage": api_storage, "/api/layer": api_layer, "/api/quit": api_quit,
       "/api/combine_with": api_combine_with, "/api/signoff_form": api_signoff_form, "/api/signoff": api_signoff}


# ---------------------------------------------------------------- server
class _Handler(http.server.SimpleHTTPRequestHandler):
    port = PORT
    token = ""
    assets: dict | None = None

    def log_message(self, *a):
        pass

    def _local(self) -> bool:
        ok = {f"localhost:{self.port}", f"127.0.0.1:{self.port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in ok and (origin is None or origin.split("//", 1)[-1] in ok)

    def _send(self, code: int, obj) -> None:
        data = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0].split("#")[0]
        if path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/" + PAGE)
            self.end_headers()
            return
        if not self._local():
            return self._send(403, {"error": "only this computer's browser may open the app"})
        if path == "/" + PAGE:
            from .map_builder import send_page
            return send_page(self, PAGE, {"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff"})
        rel = path.lstrip("/")
        f = ASSETS.get(rel)                          # an exact name, never a path joined from the request
        data = (self.assets or {}).get(rel) if self.assets is not None else (
            f.read_bytes() if f is not None and f.is_file() else None)
        if f is None or data is None:
            return self._send(404, {"error": "not served"})
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(f.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        from .map_builder import STALE_PAGE, token_ok
        path = self.path.split("?")[0]
        fn = API.get(path)
        if fn is None:
            return self._send(404, {"error": "no such endpoint"})
        if not self._local():
            return self._send(403, {"error": "only the app page on this computer may call this"})
        if not token_ok(self.headers, self.token):
            return self._send(403, {"error": STALE_PAGE})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 8 * 1024 * 1024:                         # a company logo, as a data: URL
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, fn(body))
        except Exception as exc:                              # noqa: BLE001
            self._send(400, {"error": str(exc) or type(exc).__name__})


def _port_busy(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def serve(port: int = PORT) -> str:
    """Start the server in a daemon thread (once) and return the page URL. The
    page and its files are read now and served as they were (results_viewer.serve)."""
    url = f"http://localhost:{port}/{PAGE}"
    if "httpd" in _SERVER or _port_busy(port):
        return url
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ASSET_DIR / PAGE, SERVE_DIR / PAGE)
    assets = {rel: f.read_bytes() for rel, f in ASSETS.items() if f.is_file()}
    handler = type("Handler", (_Handler,), {"port": port, "token": secrets.token_urlsafe(32), "assets": assets})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(handler, directory=str(SERVE_DIR)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _SERVER["httpd"] = httpd
    return url


def open_window(url: str) -> None:
    """The page in its own app window (Edge --app on Windows: no tabs or address
    bar), else the default browser."""
    edge = None
    for d in ("/mnt/c/Program Files (x86)/Microsoft/Edge/Application", "/mnt/c/Program Files/Microsoft/Edge/Application"):
        if Path(d, "msedge.exe").exists():
            edge = str(Path(d, "msedge.exe"))
            break
    if edge:
        import subprocess
        subprocess.Popen([edge, f"--app={url}", "--window-size=1440,920"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        return
    from .map_builder import open_in_browser
    open_in_browser(url)


USAGE = """insar app [--no-window] [--port N]

The pipeline in a browser window: projects, live runs with a map that fills
in as data arrives, results. Runs are detached: closing the window never
stops one. --no-window serves without opening a window."""


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in argv):
        print(USAGE)
        return 0
    port = PORT
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    already = _port_busy(port)
    url = serve(port)
    if "--no-window" not in argv:
        open_window(url)
    if already:
        print(f"The app is already running at {url}; opened a window on it.")
        return 0
    print(f"InSAR pipeline app at {url}  (closing the window leaves it running; Ctrl-C or Quit in the app stops it)")
    try:
        _STOP.wait()
    except KeyboardInterrupt:
        print()
    _SERVER["httpd"].shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
