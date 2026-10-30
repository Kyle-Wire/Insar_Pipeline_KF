"""Storage: what every project on this machine takes, and two ways to free it.

    insar storage                                   the overview (read-only)
    insar storage keep-results <name> [--apply]     keep the results, delete the processing files
    insar storage delete <name> [--no-archive] [--apply]

Built on what exists, with no deletion rules of its own:
  - "Keep results only" is `insar cleanup --level 3` (scripts/cleanup.py):
    its targets(), its guards (never while a run holds the project, only on a
    FINISHED project) and its delete(), under the project lock. Logs,
    releases, the sign-off, the delivered products and the point products are
    never touched (cleanup.PROTECTED).
  - "Delete project" first copies a slim archive to
    <projects folder>/_archive/<name>_<YYYYmmdd>/ (the deliverables without
    files over 50 MB, the client report, the validation, the logs, the run
    records and the config), checks every copy, then deletes the project
    folder under the project lock and takes its config off the remembered
    list. The config yaml itself is kept unless it lives inside the folder.
    A project that went to a client (deliverables/releases/ or final_*) or
    whose report is FINAL needs its name typed as well.

Every action is a dry run unless apply=True (--apply): the plan says what
would go and how many GB. Sizes are apparent bytes (like `du -sb`), symlinks
not followed, hard links counted once; GB = 1e9 as everywhere else.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import runs, settings

log = logging.getLogger(__name__)

GB = settings.GB
ARCHIVE_DIR = "_archive"
SKIP_NAMES = {"configs", "gnss"}
MARKERS = ("run.json", "mintpy", "deliverables", "slc")
BIG_FILE_BYTES = 50_000_000          # files over this are left out of the archive (and listed)
# run_result_history.jsonl: every earlier run's end record (16 Oct 2026, readiness review ERR-04)
ARCHIVE_TOP_FILES = ("run.json", "run_result.json", "run_result_history.jsonl", "quality_assessment.json",
                     "config_snapshot.yaml")
ARCHIVE_MARGIN_GB = 2.0              # the archive must fit the real budget with this to spare


# ---------------------------------------------------------------- sizes
def du(path) -> int:
    """Apparent bytes under path, like `du -sb`: symlinks not followed (and
    not counted), each hard-linked file once."""
    p = Path(path)
    try:
        st = p.lstat()
    except OSError:
        return 0
    if p.is_symlink():
        return 0
    if not p.is_dir():
        return st.st_size
    total, seen, stack = 0, set(), [str(p)]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_symlink():
                        continue
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                        continue
                    s = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if s.st_nlink > 1:
                    k = (s.st_dev, s.st_ino)
                    if k in seen:
                        continue
                    seen.add(k)
                total += s.st_size
    return total


def by_folder(P: Path) -> dict:
    """{top-level folder: bytes}, loose files at the top summed as '(files)'; largest first."""
    out: dict = {}
    try:
        entries = list(P.iterdir())
    except OSError:
        return out
    for e in entries:
        if e.is_symlink():
            continue
        key = e.name if e.is_dir() else "(files)"
        out[key] = out.get(key, 0) + du(e)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def gb(n) -> str:
    """Bytes as GB for people: '181.3 GB', '0.04 GB', '-'."""
    if n is None:
        return "-"
    g = n / GB
    return f"{g:.1f} GB" if g >= 0.1 or g == 0 else f"{g:.2f} GB"


# ---------------------------------------------------------------- finding projects
@dataclass
class ProjectInfo:
    name: str
    path: str
    config: str = ""
    size: int | None = None
    folders: dict = field(default_factory=dict)
    backend: str = ""
    route: str = ""
    state: str = "idle"
    running: bool = False
    last_kind: str = ""
    last_status: str = ""
    last_finished: float | None = None
    purpose: str = ""
    release: str = ""                 # FINAL / DRAFT (client_report.json) or ""
    delivered: list = field(default_factory=list)   # releases/<x>, final_<x> folders
    reclaimable: int | None = None
    reclaimable_why: str = ""         # why "keep results only" is not available
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _load_cfg(path):
    from .config import load_config
    return load_config(str(path))


def _protected_dirs() -> list[Path]:
    """Folders storage never deletes or lists as a project."""
    root = settings.projects_root()
    out = [root, root / "configs", root / "gnss", root / ARCHIVE_DIR, settings.configs_dir(),
           Path.home(), Path("/"), runs.PKG, runs.PARENT]
    return [p.absolute() for p in out]


def _looks_like_project(d: Path) -> bool:
    return any((d / m).exists() for m in MARKERS)


def _config_map() -> dict:
    """{resolved project folder: (config path, cfg)} for every config the
    pipeline knows (the most recently used first wins)."""
    out: dict = {}
    for c in runs.known_configs():
        try:
            cfg = _load_cfg(c)
        except Exception:                                    # noqa: BLE001 - a broken config is not a project here
            continue
        try:
            P = Path(cfg.project_dir).resolve()
        except OSError:
            continue
        out.setdefault(P, (str(c), cfg))
    return out


def _config_from_records(P: Path):
    """The config a project's own run records name, when it loads and points back here."""
    for f in (runs.RESULTFILE, runs.RUNFILE):
        try:
            c = json.loads((P / f).read_text()).get("config")
        except (OSError, ValueError, AttributeError):
            continue
        if c and Path(c).is_file():
            try:
                cfg = _load_cfg(c)
                if Path(cfg.project_dir).resolve() == P.resolve():
                    return str(c), cfg
            except Exception:                                # noqa: BLE001
                continue
    return None, None


def project_dirs() -> list[tuple[str, Path, str, object]]:
    """[(name, folder, config path or '', cfg or None)]: every folder under the
    projects folder that looks like a project (run.json, mintpy/,
    deliverables/ or slc/), plus the existing project folders of the configs
    the pipeline knows. Skips configs/, gnss/, _archive and '_' / '.' folders."""
    root = settings.projects_root()
    cmap = _config_map()
    blocked = set(_protected_dirs())
    found: dict = {}
    try:
        kids = sorted(root.iterdir()) if root.is_dir() else []
    except OSError:
        kids = []
    for d in kids:
        if d.name in SKIP_NAMES or d.name.startswith(("_", ".")) or d.is_symlink() or not d.is_dir():
            continue
        if _looks_like_project(d) or d.resolve() in cmap:
            found[d.resolve()] = d
    for P in cmap:
        if P.is_dir() and P not in found and P.absolute() not in blocked \
                and not P.name.startswith(("_", ".")) and P.name not in SKIP_NAMES:
            found[P] = P
    out, names = [], set()
    for P, d in found.items():
        if P.absolute() in blocked:
            continue
        c, cfg = cmap.get(P, (None, None))
        if c is None:
            c, cfg = _config_from_records(P)
        name = d.name
        if name in names:                                     # same folder name elsewhere
            name = f"{d.name} ({d.parent.name})"
        names.add(name)
        out.append((name, d, c or "", cfg))
    return sorted(out, key=lambda t: t[0].lower())


def _read_json(f: Path):
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def delivered(P: Path) -> list[str]:
    """What went to a client: deliverables/releases/<x> and deliverables/final_<x>."""
    D = P / "deliverables"
    out = []
    rel = D / "releases"
    if rel.is_dir():
        out += [f"releases/{x.name}" for x in sorted(rel.iterdir())]
    out += [x.name for x in sorted(D.glob("final_*")) if x.is_dir()] if D.is_dir() else []
    return out


def release_state(P: Path) -> str:
    """FINAL / DRAFT from client_report.json's release block, '' without a report."""
    rep = _read_json(P / "deliverables" / "client_report.json")
    if not isinstance(rep, dict):
        return ""
    return "FINAL" if (rep.get("release") or {}).get("final") else "DRAFT"


def _is_final(P: Path, cfg) -> bool:
    """FINAL by the stored report, or by the release gate itself when a sign-off exists."""
    if release_state(P) == "FINAL":
        return True
    if cfg is not None and (P / "deliverables" / "review.json").exists():
        try:
            from . import release
            return bool(release.is_final(cfg))
        except Exception:                                    # noqa: BLE001 - unknown counts as final below
            return True
    return False


def _purpose(P: Path, cfg) -> str:
    if cfg is not None:
        try:
            from .config import is_experiment
            return "experiment" if is_experiment(cfg) else "client"
        except Exception:                                    # noqa: BLE001
            pass
    if (P / "deliverables" / "experiment_not_for_clients").exists():
        return "experiment"
    return "client" if cfg is not None else "?"


def _keep_results_targets(cfg):
    from .scripts import cleanup
    return cleanup.targets(cfg, 3)


def _not_available(P: Path, cfg) -> str:
    """Why 'keep results only' cannot run now ('' when it can)."""
    if cfg is None:
        return "no config file is known for this project"
    if runs.state(P) == "running":
        return "a run is going"
    from .scripts import cleanup
    why = cleanup.unfinished(cfg)
    if why:
        return "not finished: " + "; ".join(why)
    return ""


def describe(name: str, P: Path, cfg_path: str, cfg, sizes: bool = True) -> ProjectInfo:
    info = ProjectInfo(name=name, path=str(P), config=cfg_path)
    try:
        if cfg is not None:
            info.backend = str(getattr(cfg, "backend", "") or "")
        res = runs.result(P) or {}
        info.route = str(res.get("route") or "")
        info.backend = info.backend or str(res.get("backend") or "")
        info.last_kind = str(res.get("kind") or "")
        info.last_status = str(res.get("status") or "")
        info.last_finished = res.get("finished")
        info.state = runs.state(P)
        info.running = info.state == "running"
        info.purpose = _purpose(P, cfg)
        info.release = release_state(P)
        info.delivered = delivered(P)
        if sizes:
            info.folders = by_folder(P)
            info.size = sum(info.folders.values())
        info.reclaimable_why = _not_available(P, cfg)
        if not info.reclaimable_why:
            items = _keep_results_targets(cfg)
            info.reclaimable = sum(du(p) for _, p, _ in items)
    except Exception as exc:                                 # noqa: BLE001 - one bad project must not hide the rest
        info.error = str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__
    return info


def inventory(sizes: bool = True) -> dict:
    """{root, budget (settings.disk_budget_gb), projects: [ProjectInfo]}."""
    root = settings.projects_root()
    projects = [describe(n, P, c, cfg, sizes=sizes) for n, P, c, cfg in project_dirs()]
    try:
        budget = settings.disk_budget_gb(root)
    except Exception as exc:                                 # noqa: BLE001
        budget = {"budget": None, "source": f"not measured ({exc})"}
    return {"root": str(root), "budget": budget, "projects": projects}


def find(name: str):
    """(name, folder, config path, cfg) of one project, by its name in the
    overview or its folder path. KeyError with the names known otherwise."""
    allp = project_dirs()
    for t in allp:
        if t[0] == name:
            return t
    try:
        want = Path(name).expanduser().resolve()
    except OSError:
        want = None
    for t in allp:
        if want is not None and t[1].resolve() == want:
            return t
    known = ", ".join(t[0] for t in allp) or "none"
    raise KeyError(f"No project called '{name}'. Projects: {known}.")


# ---------------------------------------------------------------- keep results only
def keep_results_only(name: str, apply: bool = False) -> dict:
    """Delete the processing files of a finished project and keep its results:
    cleanup levels 1-3 through scripts/cleanup.py (targets, unfinished,
    delete) under the project lock. Dry run unless apply=True.

    {project, path, config, items: [{level, path, why, bytes}], bytes,
     refused: [reasons], notes: [what is lost], applied, freed, failed}"""
    from .scripts import cleanup
    pname, P, cfg_path, cfg = find(name)
    plan = {"project": pname, "path": str(P), "config": cfg_path, "items": [], "bytes": 0,
            "refused": [], "notes": [], "applied": False, "freed": 0, "failed": []}
    if cfg is None:
        plan["refused"].append("No config file is known for this project, so it cannot tell its results "
                               "from its processing files. Use 'insar cleanup <config>' or delete the project.")
        return plan
    if runs.state(P) == "running":
        h = runs.lock_holder(P) or runs.record(P) or {}
        what = f"A {h['kind']} run" if h.get("kind") else "A run"
        plan["refused"].append(f"{what} is going in this project (pid {h.get('pid', '?')}). "
                               "Wait until it finishes, or stop it first.")
    why = cleanup.unfinished(cfg)
    if why:
        plan["refused"].append("The project is not finished: " + "; ".join(why)
                               + ". Finish the run or rebuild the products ('Products only') first.")
    items = cleanup.targets(cfg, 3)
    for lv, p, w in items:
        n = du(p)
        plan["items"].append({"level": lv, "path": str(p), "why": w, "bytes": n})
        plan["bytes"] += n
    if any(lv >= 2 for lv, _, _ in items):
        plan["notes"].append(cleanup.LOST_AT_LEVEL2)
    if any(lv >= 3 for lv, _, _ in items):
        plan["notes"].append(cleanup.LOST_AT_LEVEL3)
    if not apply or plan["refused"] or not items:
        return plan
    try:
        with runs.project_lock(P, kind="cleanup", config=cfg_path):
            again = cleanup.unfinished(cfg)               # a run may have ended badly since the plan
            if again:
                plan["refused"].append("The project is not finished: " + "; ".join(again) + ".")
                return plan
            freed, failed = cleanup.delete(cfg, items)
    except RuntimeError as exc:
        if "already running" not in str(exc):
            raise
        plan["refused"].append(f"Not done: {exc}.")
        return plan
    plan.update(applied=True, freed=freed, failed=[(str(p), r) for p, r in failed])
    return plan


# ---------------------------------------------------------------- delete project
def _always_keep(rel: Path) -> bool:
    """Deliverables archived whatever their size: the client report and its
    KMZ/PDF, and what went to the client (releases/, final_*)."""
    parts = rel.parts
    if parts[0] == "releases" or parts[0].startswith("final_"):
        return True
    if len(parts) == 1:
        n = rel.name
        return (n.startswith("client_report.") or n.startswith("client_area_summaries")
                or (n.endswith(".kmz") and "ground_movement" in n))
    return False


def _walk_files(d: Path):
    for root, dirs, files in os.walk(d, followlinks=False):
        r = Path(root)
        dirs[:] = [x for x in dirs if not (r / x).is_symlink()]
        for f in files:
            p = r / f
            if not p.is_symlink() and p.is_file():
                yield p


def archive_plan(P: Path, cfg_path: str = "") -> tuple[list, list]:
    """([(source, path in the archive, bytes)], [(path in the project, bytes) left out])."""
    keep, skipped = [], []
    D = P / "deliverables"
    if D.is_dir():
        for f in _walk_files(D):
            rel = f.relative_to(D)
            n = f.stat().st_size
            if n > BIG_FILE_BYTES and not _always_keep(rel):
                skipped.append((f"deliverables/{rel}", n))
            else:
                keep.append((f, Path("deliverables") / rel, n))
    V = P / "validation"
    if V.is_dir():
        for f in _walk_files(V):
            rel = f.relative_to(V)
            if "gnss_cache" in rel.parts:
                continue
            n = f.stat().st_size
            if n > BIG_FILE_BYTES:
                skipped.append((f"validation/{rel}", n))
            else:
                keep.append((f, Path("validation") / rel, n))
    L = P / "logs"
    if L.is_dir():
        for f in _walk_files(L):
            keep.append((f, Path("logs") / f.relative_to(L), f.stat().st_size))
    for f in [P / n for n in ARCHIVE_TOP_FILES] + sorted(P.glob("provenance*")):
        if f.is_file() and not f.is_symlink():
            keep.append((f, Path(f.name), f.stat().st_size))
    if cfg_path and Path(cfg_path).is_file():
        c = Path(cfg_path)
        keep.append((c, Path("config") / c.name, c.stat().st_size))
    seen, out = set(), []
    for src, dst, n in keep:
        if dst not in seen:
            seen.add(dst)
            out.append((src, dst, n))
    return out, sorted(skipped, key=lambda t: -t[1])


def _archive_dest(name: str) -> Path:
    base = settings.projects_root() / ARCHIVE_DIR
    stem = f"{name.replace(' ', '_').replace('/', '_')}_{time.strftime('%Y%m%d')}"
    d, i = base / stem, 2
    while d.exists():
        d, i = base / f"{stem}_{i}", i + 1
    return d


def _unsafe(P: Path, others: list[Path]) -> str:
    """Why folder P must not be deleted ('' when it may)."""
    A = P.absolute()
    if P.is_symlink():
        return f"{P} is a link, not a folder"
    if not P.is_dir():
        return f"{P} is not a folder"
    for q in _protected_dirs():
        if A == q or A in q.parents:
            return f"{P} holds {q}, which is never deleted"
    for o in others:
        o = o.absolute()
        if o != A and A in o.parents:
            return f"{P} holds another project ({o})"
    return ""


def delete_project(name: str, archive: bool = True, apply: bool = False, confirm_name: str | None = None,
                   progress=None) -> dict:
    """Delete a whole project folder, after a slim archive (archive=True).
    Dry run unless apply=True. A project that went to a client (releases/,
    final_*) or whose report is FINAL needs confirm_name == its name.
    progress(done, total, text) is called while it works.

    {project, path, config, bytes, archive, archive_dir, archive_bytes,
     archive_files, skipped: [(path, bytes)], needs_name, why_name, refused,
     notes, applied, archived, deleted, forgot}"""
    pname, P, cfg_path, cfg = find(name)
    others = [t[1] for t in project_dirs() if t[1].resolve() != P.resolve()]
    plan = {"project": pname, "path": str(P), "config": cfg_path, "bytes": du(P), "archive": archive,
            "archive_dir": "", "archive_bytes": 0, "archive_files": 0, "skipped": [], "needs_name": False,
            "why_name": "", "refused": [], "notes": [], "applied": False, "archived": False,
            "deleted": False, "forgot": False}
    bad = _unsafe(P, others)
    if bad:
        plan["refused"].append(f"Refused: {bad}.")
    if runs.state(P) == "running":
        h = runs.lock_holder(P) or runs.record(P) or {}
        plan["refused"].append(f"A run is going in this project (pid {h.get('pid', '?')}). "
                               "Stop it first (live view, x), then delete.")
    went = delivered(P)
    final = _is_final(P, cfg)
    if went or final:
        plan["needs_name"] = True
        plan["why_name"] = ("it went to a client (" + ", ".join(went) + ")" if went else "its client report is FINAL")
    if cfg_path:
        inside = P.resolve() in Path(cfg_path).resolve().parents
        plan["notes"].append(f"The config {cfg_path} is " + ("inside the folder and goes with it"
                             + (" (a copy is kept in the archive)" if archive else "") if inside else
                             "kept, so the project can be run again from it (it stays on the projects "
                             "screen with nothing on disk; delete the yaml by hand to remove it)"))
    if cfg is not None:
        M = Path(getattr(cfg, "mintpy_dir", P / "mintpy")).resolve()
        if M.exists() and P.resolve() not in M.parents and M != P.resolve():
            plan["notes"].append(f"The MintPy folder {M} is outside the project folder and is NOT deleted.")
    if archive:
        files, skipped = archive_plan(P, cfg_path)
        plan.update(archive_dir=str(_archive_dest(pname)), archive_bytes=sum(n for _, _, n in files),
                    archive_files=len(files), skipped=[(s, n) for s, n in skipped])
    if not apply:
        return plan
    if plan["needs_name"] and (confirm_name or "").strip() != pname:
        plan["refused"].append(f"Type the project's name ({pname}) to delete it: {plan['why_name']}.")
    if archive:
        try:
            free = settings.disk_budget_gb(settings.projects_root()).get("budget")
        except Exception:                                    # noqa: BLE001
            free = None
        if free is not None and plan["archive_bytes"] / GB > free - ARCHIVE_MARGIN_GB:
            plan["refused"].append(f"The archive ({gb(plan['archive_bytes'])}) does not fit the free space "
                                   f"({free:.1f} GB). Delete without the archive, or free space first.")
    if plan["refused"]:
        return plan
    say = progress or (lambda *_: None)
    try:
        with runs.project_lock(P, kind="delete", config=cfg_path or None):
            if archive:
                dest = Path(plan["archive_dir"])
                ok, problems = _write_archive(P, dest, files, skipped, cfg_path, pname, say)
                if not ok:
                    plan["refused"].append("The archive could not be written, so nothing was deleted: "
                                           + "; ".join(problems[:5]))
                    return plan
                plan["archived"] = True
            left = _remove_folder(P, say)
            if left:
                plan["refused"].append("Some files could not be deleted: " + "; ".join(left[:5]))
                return plan
            plan["deleted"] = not P.exists()
    except RuntimeError as exc:
        if "already running" not in str(exc):
            raise
        plan["refused"].append(f"Not deleted: {exc}.")
        return plan
    if cfg_path:
        plan["forgot"] = runs.forget(cfg_path)
    plan["applied"] = True
    say(1, 1, "done")
    return plan


def _write_archive(P: Path, dest: Path, files, skipped, cfg_path, name, say) -> tuple[bool, list]:
    problems = []
    tmp = dest.with_name(dest.name + ".partial")
    try:
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        total = len(files)
        for i, (src, rel, n) in enumerate(files):
            if i % 25 == 0:
                say(i, total, f"archiving {rel}")
            out = tmp / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            if out.stat().st_size != src.stat().st_size:
                problems.append(f"{rel}: copy is a different size")
        manifest = {"project": name, "source": str(P), "config": cfg_path,
                    "archived": time.strftime("%Y-%m-%d %H:%M"), "files": len(files),
                    "bytes": sum(n for _, _, n in files),
                    "left_out_over_50MB": [{"path": s, "bytes": n} for s, n in skipped]}
        (tmp / "ARCHIVE.json").write_text(json.dumps(manifest, indent=1))
        (tmp / "README.txt").write_text(
            f"Archive of the project '{name}' ({P}), made {manifest['archived']} before the project was deleted.\n"
            "It holds the deliverables (files over 50 MB left out, except the client report, its KMZ/PDF and "
            "what went to the client), the validation, the logs, the run records and the config.\n"
            "ARCHIVE.json lists what was left out. The processing files are gone; a new run would start "
            "from the download.\n")
    except OSError as exc:
        problems.append(str(exc))
    if problems:
        shutil.rmtree(tmp, ignore_errors=True)
        return False, problems
    os.replace(tmp, dest)
    return True, []


def _remove_folder(P: Path, say) -> list[str]:
    """Delete everything in P, then P. The lock file goes last (it is held)."""
    left = []
    kids = [k for k in P.iterdir() if k.name != runs.LOCKFILE]
    for i, k in enumerate(kids):
        say(i, len(kids), f"deleting {k.name}")
        try:
            if k.is_dir() and not k.is_symlink():
                shutil.rmtree(k)
            else:
                k.unlink()
        except OSError as exc:
            left.append(f"{k}: {exc}")
    if left:
        return left
    try:
        lock = P / runs.LOCKFILE
        if lock.exists():
            lock.unlink()
        P.rmdir()
    except OSError as exc:
        left.append(f"{P}: {exc}")
    return left


# ---------------------------------------------------------------- the command
def _when(t) -> str:
    return time.strftime("%d %b %Y %H:%M", time.localtime(t)) if t else ""


def status_text(p: ProjectInfo) -> str:
    if p.error:
        return "unreadable"
    s = "RUNNING" if p.running else p.state
    if p.purpose == "experiment":
        s += ", experiment"
    return s


def delivered_text(p: ProjectInfo) -> str:
    return ("yes: " + ", ".join(p.delivered)) if p.delivered else "no"


def reclaim_text(p: ProjectInfo) -> str:
    """The table's 'keep results only frees' cell: GB, or a short reason."""
    if p.reclaimable is not None:
        return gb(p.reclaimable)
    w = p.reclaimable_why
    short = ("run going" if w.startswith("a run") else "not finished" if w.startswith("not finished")
             else "no config" if w.startswith("no config") else w or p.error or "?")
    return f"n/a ({short})"


def overview_text(inv: dict) -> str:
    b = inv["budget"]
    rows = [("Project", "Size", "Status", "Last run", "Report", "Delivered?", "Keep results only frees")]
    for p in inv["projects"]:
        last = " ".join(x for x in (p.last_kind, p.last_status, _when(p.last_finished)) if x) or "-"
        rows.append((p.name, gb(p.size), status_text(p), last, p.release or "-", delivered_text(p), reclaim_text(p)))
    n = len(rows[0]) - 1
    w = [max(len(r[i]) for r in rows) for i in range(n)]
    lines = [f"Projects folder: {inv['root']}"]
    if b.get("budget") is not None:
        lines.append(f"Free disk (the real budget): {b['budget']:.1f} GB  [{b.get('source', '')}]")
    else:
        lines.append(f"Free disk: {b.get('source', 'not measured')}")
    lines.append("")
    for i, r in enumerate(rows):
        lines.append("  ".join(r[j].ljust(w[j]) for j in range(n)) + "  " + r[n])
        if i == 0:
            lines.append("  ".join("-" * w[j] for j in range(n)) + "  " + "-" * len(r[n]))
    tot = sum(p.size or 0 for p in inv["projects"])
    lines.append("")
    for p in inv["projects"]:
        if p.reclaimable is None and (p.reclaimable_why or p.error):
            lines.append(f"{p.name}: keep results only is not available - {p.reclaimable_why or p.error}.")
    lines.append(f"{len(inv['projects'])} projects, {gb(tot)} in all.")
    lines.append("Free space:  insar storage keep-results <name>   (keep the results, delete the processing files)")
    lines.append("             insar storage delete <name>         (delete the whole project, a small archive kept)")
    lines.append("Both show the plan first; add --apply to do it.")
    return "\n".join(lines)


def keep_plan_text(plan: dict) -> str:
    out = [f"Keep results only: {plan['project']} ({plan['path']})"]
    if plan["items"]:
        out.append(f"Deletes {len(plan['items'])} items, {gb(plan['bytes'])}:")
        for it in sorted(plan["items"], key=lambda x: -x["bytes"]):
            out.append(f"  {gb(it['bytes']):>9}  {it['path']}  ({it['why']})")
    else:
        out.append("Nothing to delete: only the results are left.")
    out.append("Kept: the deliverables, the client report, the point products, the logs, the releases, "
               "the sign-off, the current MintPy products and the config.")
    out += plan["notes"]
    for r in plan["refused"]:
        out.append(f"NOT AVAILABLE: {r}")
    return "\n".join(out)


def delete_plan_text(plan: dict) -> str:
    out = [f"Delete project: {plan['project']} ({plan['path']}), {gb(plan['bytes'])}"]
    if plan["archive"]:
        out.append(f"First a small archive ({plan['archive_files']} files, {gb(plan['archive_bytes'])}) goes to "
                   f"{plan['archive_dir']}")
        if plan["skipped"]:
            out.append(f"Left out of the archive (over 50 MB each, {len(plan['skipped'])} files, "
                       f"{gb(sum(n for _, n in plan['skipped']))}):")
            for s, n in plan["skipped"][:12]:
                out.append(f"  {gb(n):>9}  {s}")
            if len(plan["skipped"]) > 12:
                out.append(f"  ... and {len(plan['skipped']) - 12} more (listed in the archive's ARCHIVE.json)")
    else:
        out.append("NO archive: nothing of the project is kept.")
    out += plan["notes"]
    if plan["needs_name"]:
        out.append(f"This project needs its name typed to delete it: {plan['why_name']}.")
    for r in plan["refused"]:
        out.append(f"REFUSED: {r}")
    return "\n".join(out)


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def main(argv=None) -> int:
    """`insar storage [keep-results|delete] ...`. 0 done or shown, 1 not done
    (refused, not confirmed, or something failed), 2 usage."""
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="insar storage", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    k = sub.add_parser("keep-results", help="keep the results, delete the processing files (a finished project)")
    k.add_argument("name")
    k.add_argument("--apply", action="store_true", help="do it (default: show the plan only)")
    d = sub.add_parser("delete", help="delete a whole project (a small archive is kept first)")
    d.add_argument("name")
    d.add_argument("--no-archive", action="store_true", help="keep nothing of it")
    d.add_argument("--apply", action="store_true", help="do it (default: show the plan only)")
    a = ap.parse_args(argv)
    if a.cmd is None:
        print("Measuring every project (a few seconds)...", flush=True)
        print(overview_text(inventory()))
        return 0
    try:
        find(a.name)
    except KeyError as exc:
        print(exc.args[0])
        return 1
    if a.cmd == "keep-results":
        plan = keep_results_only(a.name)
        print(keep_plan_text(plan))
        if plan["refused"] or not plan["items"]:
            return 1 if plan["refused"] else 0
        if not a.apply:
            print("\nDry run: nothing deleted. Add --apply to do it.")
            return 0
        if _ask(f"\nType yes to delete {gb(plan['bytes'])} of processing files: ").lower() != "yes":
            print("Nothing deleted.")
            return 1
        plan = keep_results_only(a.name, apply=True)
        if plan["refused"]:
            print("\n".join(f"REFUSED: {r}" for r in plan["refused"]))
            return 1
        print(f"Freed {gb(plan['freed'])}.")
        for p, r in plan["failed"]:
            print(f"  NOT deleted: {p}: {r}")
        return 1 if plan["failed"] else 0
    archive = not a.no_archive
    plan = delete_project(a.name, archive=archive)
    print(delete_plan_text(plan))
    if plan["refused"]:
        return 1
    if not a.apply:
        print("\nDry run: nothing deleted. Add --apply to do it.")
        return 0
    name = None
    if plan["needs_name"]:
        name = _ask(f"\nType the project's name ({plan['project']}) to confirm: ")
        if name != plan["project"]:
            print("The name does not match. Nothing deleted.")
            return 1
    if _ask(f"Type yes to delete {plan['path']} ({gb(plan['bytes'])}): ").lower() != "yes":
        print("Nothing deleted.")
        return 1

    def show(done, total, text):
        print(f"  [{done}/{total}] {text}", flush=True)
    plan = delete_project(a.name, archive=archive, apply=True, confirm_name=name, progress=show)
    if plan["refused"]:
        print("\n".join(f"REFUSED: {r}" for r in plan["refused"]))
        return 1
    print(f"Deleted {plan['path']} ({gb(plan['bytes'])})."
          + (f" Archive: {plan['archive_dir']}" if plan["archived"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())


def auto_keep_results(cfg, why: str) -> dict | None:
    """'Keep results only' on its own once a project is truly finished (13
    Oct 2026): called by package.build after a FINAL client package and by
    pipeline.run_from after a finished experiment run, both outside the
    project lock. Off with config auto_keep_results: false. Uses
    keep_results_only and so every cleanup guard (never while a run holds the
    project, only on a finished project, never logs/, releases/, results).
    Never raises: a refusal or an error is logged and the project left as it
    is. Returns the plan (or None when off or not attempted)."""
    if not getattr(cfg, "auto_keep_results", True):
        log.info("auto_keep_results is off: processing files kept (free them with 'insar storage keep-results')")
        return None
    P = Path(cfg.project_dir)
    try:
        plan = keep_results_only(str(P), apply=True)
    except Exception as exc:                                  # noqa: BLE001 - never fail the caller
        log.warning("Automatic 'keep results only' not done for %s: %s", P.name, exc)
        return None
    if plan.get("refused"):
        log.info("Automatic 'keep results only' not done for %s: %s", P.name, " ".join(plan["refused"]))
    elif plan.get("applied"):
        log.info("Kept the results of %s and freed %.1f GB of processing files, because %s. Only 'Products "
                 "only' is sure to work on it now; set auto_keep_results: false in its config to keep them.",
                 P.name, plan.get("freed", 0) / 1e9, why)
        if plan.get("failed"):
            log.warning("Some processing files of %s were not deleted: %s", P.name,
                        "; ".join(f"{p}: {w}" for p, w in plan["failed"][:5]))
    return plan
