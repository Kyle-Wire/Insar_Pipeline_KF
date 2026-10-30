"""Files inside one project, for the app's Files page: list a folder with its
sizes, and delete chosen files or folders from a button.

No deletion rule of its own beyond these:
  - nothing scripts/cleanup.py protects (cleanup.protected: the logs, the
    releases and sign-off, the delivered point products, mintpy/geo, the run
    records, context/, dem/, anything outside the project) and nothing in
    deliverables/final_*, nor the project's config file, nor the project
    folder itself;
  - never while a run holds the project, and the delete itself holds the
    project lock (kind "cleanup"), so a run cannot start part-way through;
  - a plan first (plan_delete: what goes, how many bytes, what is refused),
    then delete() re-checks everything against the disk;
  - the project's name typed when anything in deliverables/ is chosen or the
    total passes NAME_ABOVE_BYTES;
  - every deletion is appended to <project>/logs/deletions.jsonl (what, how
    big, when, by which program), which itself is never deleted (logs/).

Deleting here is permanent: WSL has no recycle bin. Whole-project actions
(keep results only, delete with an archive) are storage.py's.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from . import runs

NAME_ABOVE_BYTES = 50_000_000_000
# kept here besides cleanup.PROTECTED: what the release gate and the run history read
KEEP = ("quality_assessment.json", "run_result_history.jsonl", "deliverables/client_report.json",
        "deliverables/gnss_tie.json", "deliverables/provenance.json")
LOGFILE = "deletions.jsonl"
# what the top-level folders hold, in plain words (the page says it beside the name)
ABOUT = {
    "slc": "downloaded satellite images (SAFE)", "orbits": "satellite orbit files", "dem": "the elevation model",
    "stack": "the coregistered image stack", "interferograms": "interferograms",
    "miaplpy": "phase linking (its inputs and outputs)", "mintpy": "the time series (MintPy)",
    "deliverables": "what goes to the client: report, KMZ, rasters, points", "logs": "run logs (kept)",
    "context": "the project's map layers (kept)", "tropo_corrections": "weather delay files",
    "validation": "validation against ground truth", "superseded": "products a later run replaced",
}


def _protect_reason(cfg, p: Path) -> str:
    """Why p must not be deleted from the Files page ('' when it may)."""
    from .scripts import cleanup
    P = Path(cfg.project_dir).absolute()
    a = Path(os.path.abspath(p))
    if a == P:
        return "the project folder itself (use Delete project)"
    if P not in a.parents and not any(r == a or r in a.parents for r in cleanup._roots(cfg)):
        return "outside the project"
    src = getattr(cfg, "_source_path", "")
    if src and Path(os.path.abspath(src)) == a:
        return "the project's config"
    try:
        rel = a.relative_to(P)
    except ValueError:
        rel = None
    if rel is not None and len(rel.parts) >= 2 and rel.parts[0] == "deliverables" and rel.parts[1].startswith("final_"):
        return "an issued (final) copy of the deliverables"
    if rel is not None and str(rel) in KEEP:
        return "kept: the release status reads it"
    if cleanup.protected(cfg, a):
        for r in cleanup.PROTECTED:
            q = P / r
            if a == q or q in a.parents:
                return f"kept by the pipeline ({r})"
        return "kept by the pipeline"
    return ""


def _resolve(cfg, rel: str) -> Path:
    """A path the page names, relative to the project; refused if it climbs out."""
    P = Path(cfg.project_dir).absolute()
    rel = (rel or "").strip().lstrip("/")
    p = Path(os.path.abspath(P / rel))
    if p != P and P not in p.parents:
        raise ValueError("that is outside the project")
    return p


def _sizes(d: Path, timeout: float = 25.0) -> dict:
    """{child name: apparent bytes} for d's children in one `du` pass (None on timeout)."""
    try:
        out = subprocess.run(["du", "-sb", "--", *[str(c) for c in d.iterdir()]], capture_output=True, text=True,
                             timeout=timeout).stdout
    except (subprocess.TimeoutExpired, OSError):
        return {}
    res = {}
    for line in out.splitlines():
        n, _, path = line.partition("\t")
        try:
            res[Path(path).name] = int(n)
        except ValueError:
            pass
    return res


def list_dir(cfg, rel: str = "") -> dict:
    d = _resolve(cfg, rel)
    if not d.is_dir():
        raise ValueError("not a folder")
    P = Path(cfg.project_dir).absolute()
    sizes = _sizes(d) if any(True for _ in d.iterdir()) else {}
    items = []
    for c in sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        try:
            st = c.lstat()
        except OSError:
            continue
        kind = "link" if c.is_symlink() else "dir" if c.is_dir() else "file"
        why = _protect_reason(cfg, c)
        items.append({"name": c.name, "rel": str(c.relative_to(P)), "kind": kind,
                      "bytes": sizes.get(c.name, st.st_size if kind == "file" else None), "mtime": st.st_mtime,
                      "protected": why, "about": ABOUT.get(c.name, "") if d == P else ""})
    return {"rel": str(d.relative_to(P)) if d != P else "", "items": items,
            "total": sum(i["bytes"] or 0 for i in items), "running": runs.state(P) == "running"}


def plan_delete(cfg, rels: list, locked: bool = False) -> dict:
    """locked: this process holds the project lock (delete's re-check), which is
    itself what runs.state would read as a run going."""
    P = Path(cfg.project_dir).absolute()
    plan = {"items": [], "bytes": 0, "refused": [], "needs_name": False, "name": P.name}
    if not locked and runs.state(P) == "running":
        plan["refused"].append("A run is going in this project: stop it first, or wait for it to finish.")
    for rel in dict.fromkeys(rels or []):
        try:
            p = _resolve(cfg, rel)
        except ValueError as exc:
            plan["refused"].append(f"{rel}: {exc}")
            continue
        if not (p.exists() or p.is_symlink()):
            plan["refused"].append(f"{rel}: not there any more")
            continue
        why = _protect_reason(cfg, p)
        if why:
            plan["refused"].append(f"{rel}: {why}")
            continue
        from .storage import du
        n = du(p)
        plan["items"].append({"rel": str(p.relative_to(P)), "bytes": n, "kind": "dir" if p.is_dir() and not p.is_symlink() else "file"})
        plan["bytes"] += n
        if p.relative_to(P).parts[0] == "deliverables":
            plan["needs_name"] = True
    if plan["bytes"] > NAME_ABOVE_BYTES:
        plan["needs_name"] = True
    if not plan["items"] and not plan["refused"]:
        plan["refused"].append("Nothing chosen.")
    return plan


def delete(cfg, rels: list, confirm_name: str = "", program: str = "app") -> dict:
    """Delete what plan_delete allows, re-checked under the project lock.
    Refuses the whole request if any chosen item is refused."""
    P = Path(cfg.project_dir).absolute()
    plan = plan_delete(cfg, rels)
    plan.update(applied=False, freed=0, failed=[])
    if plan["refused"]:
        return plan
    if plan["needs_name"] and (confirm_name or "").strip() != plan["name"]:
        plan["refused"].append(f"Type the project's name ({plan['name']}) to delete these.")
        return plan
    src = getattr(cfg, "_source_path", "") or None
    try:
        with runs.project_lock(P, kind="cleanup", config=src):
            again = plan_delete(cfg, rels, locked=True)      # the disk again, now that nothing can start
            if again["refused"]:
                plan["refused"] = again["refused"]
                return plan
            logf = P / "logs" / LOGFILE
            logf.parent.mkdir(parents=True, exist_ok=True)
            for it in again["items"]:
                p = P / it["rel"]
                try:
                    if p.is_symlink() or not p.is_dir():
                        p.unlink()
                    else:
                        shutil.rmtree(p)
                    plan["freed"] += it["bytes"]
                    ok = True
                except OSError as exc:
                    plan["failed"].append([it["rel"], str(exc)])
                    ok = False
                with open(logf, "a") as fh:
                    fh.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "path": it["rel"], "bytes": it["bytes"],
                                         "deleted": ok, "by": program}) + "\n")
    except RuntimeError as exc:
        if "already running" not in str(exc):
            raise
        plan["refused"].append(f"Not deleted: {exc}.")
        return plan
    plan["applied"] = True
    return plan
