"""Which code made this result.

There is no git on this machine, so the version string alone cannot tell two
builds apart: client_report.py and report_insights.py changed on 29 Sep 2026
after pyproject's version was last set (2026.9.28), and the Site A (28 Sep)
and Site B (29 Sep) reports already differ in wording with nothing in
either saying which code produced it. The code hash is the identity that
does: a sha256 over every .py file of the package, so any edit - comments
included - changes it. It identifies the code exactly, not its scientific
behaviour; two hashes that differ may still give identical numbers.

    __version__      the release number (pyproject.toml beside this file, else
                     the installed metadata)
    code_hash()      12 hex characters over the package's *.py and the pages
                     and viewer scripts it serves
    describe(...)    both, plus the MiaplPy build and patch check and the
                     snaphu version when they are cheap to find
"""
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

# Folders inside the package directory that are not the code that runs:
# test files, the untouched pre-fix backup, review records, project context
# and the install bundle. A hash over them would change when a review note
# is added and stay the same when it should not.
_SKIP_DIRS = ("tests", "__pycache__", "backups", "context", "install", "report_assets")
_SKIP_PREFIX = ("review_", ".")


def _read_version() -> str:
    # pyproject.toml beside the code first: an editable install keeps the
    # metadata of the day it was installed, which is how a stale number
    # would end up in a record. The metadata is the fallback for a wheel.
    pp = PACKAGE_DIR / "pyproject.toml"
    if pp.exists():
        try:
            import tomllib
            v = tomllib.loads(pp.read_text()).get("project", {}).get("version")
            if v:
                return str(v)
        except Exception:                                    # noqa: BLE001
            pass
    try:
        import importlib.metadata as md
        return md.version("insar-pipeline")
    except Exception:                                        # noqa: BLE001
        return "unknown"


__version__ = _read_version()


# what the hash covers besides the .py files: the pages the code serves and
# the results viewer's scripts and styles (19 Oct 2026: its charts are code
# that draws client figures, so a change to them is a new build too)
_PAGE_GLOBS = ("*.html", "viewer_assets/**/*.js", "viewer_assets/**/*.css")


def _code_files(root: Path) -> list[Path]:
    out = []
    files = list(root.rglob("*.py")) + [p for g in _PAGE_GLOBS for p in root.glob(g)]
    for p in files:
        rel = p.relative_to(root).parts
        if any(part in _SKIP_DIRS or part.startswith(_SKIP_PREFIX) for part in rel[:-1]):
            continue
        out.append(p)
    return sorted(set(out), key=lambda p: p.relative_to(root).as_posix())


def code_hash(root: Path | str | None = None) -> str:
    """12 hex characters of a sha256 over the package's .py files and the
    pages and viewer scripts it serves (_PAGE_GLOBS; relative path + bytes,
    in path order). Computed once per process for the package
    itself; pass `root` to hash another tree."""
    if root is None:
        return _package_hash()
    return _hash_tree(Path(root))


def _hash_tree(root: Path) -> str:
    h = hashlib.sha256()
    for p in _code_files(root):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


@lru_cache(maxsize=1)
def _package_hash() -> str:
    return _hash_tree(PACKAGE_DIR)


# ------------------------------------------------------------------ MiaplPy and snaphu
def _env_prefix(env: str) -> Path | None:
    """The conda env's folder, found from this interpreter's own location
    (<base>/envs/<name>/bin/python) - no subprocess."""
    if not env:
        return None
    if Path(env).is_dir():
        return Path(env)
    exe = Path(sys.executable).resolve()
    for base in (exe.parents[2], exe.parents[1]):
        cand = base / env if base.name == "envs" else base / "envs" / env
        if cand.is_dir():
            return cand
    return None


def miaplpy_version(env: str = "miaplpy") -> str | None:
    """The MiaplPy installed in the conda env, read from its dist-info folder
    (a file read; `conda run` would cost seconds). '+overlapdatum' marks the
    patched build (patches/miaplpy)."""
    prefix = _env_prefix(env)
    if prefix is None:
        return None
    for d in sorted(prefix.glob("lib/python3*/site-packages/miaplpy-*.dist-info")):
        m = re.match(r"miaplpy-(.+)\.dist-info$", d.name)
        if m:
            return m.group(1)
    return None


_PATCH_LINE = re.compile(r"MiaplPy patch verified \(([^)]*)\): step error across mini-stack boundaries ([0-9.]+)x")


def miaplpy_patch_check(project_dir: Path | str | None) -> dict | None:
    """The functional patch check's result for a run (check_miaplpy_patch.py,
    run by miaplpy_backend.verify_patched_miaplpy before phase linking).

    A record file is preferred (miaplpy/patch_check.json, when the backend
    writes one); otherwise the newest 'MiaplPy patch verified' line in the
    project's logs, which is where the result has gone so far. The ratio is
    the step error across mini-stack boundaries over the error inside them:
    upstream 3.1-3.5x, patched 1.2-1.3x, pass line 2.0."""
    if not project_dir:
        return None
    P = Path(project_dir)
    for rec in (P / "miaplpy" / "patch_check.json", P / "miaplpy" / ".status" / "patch_check.json"):
        if rec.exists():
            try:
                d = json.loads(rec.read_text())
                d.setdefault("source", str(rec))
                return d
            except (OSError, ValueError):
                pass
    best = None
    logs = sorted(P.glob("logs/*.log"), key=lambda p: p.stat().st_mtime)
    for lf in logs:
        if lf.stat().st_size > 200e6:          # a run log is kB-MB; never stream a runaway one
            continue
        try:
            with open(lf, errors="replace") as f:
                for line in f:
                    m = _PATCH_LINE.search(line)
                    if m:
                        best = {"patched": True, "miaplpy_version": m.group(1), "ratio": float(m.group(2)),
                                "logged": line[:23].strip(), "source": str(lf)}
        except OSError:
            continue
    return best


@lru_cache(maxsize=4)
def snaphu_version(explicit: str = "") -> str | None:
    """'snaphu v2.0.7' from the binary the unwrapper uses (its usage text;
    milliseconds), or None when there is none."""
    import subprocess
    try:
        from .unwrap_crop import find_snaphu
        exe = find_snaphu(explicit)
        r = subprocess.run([exe], capture_output=True, text=True, timeout=10)
        return next((l.strip() for l in (r.stdout + r.stderr).splitlines() if "snaphu v" in l), None)
    except Exception:                                        # noqa: BLE001
        return None


def describe(cfg=None, project_dir: Path | str | None = None) -> dict:
    """The identity of the code, for provenance.json, client_report.json,
    MANIFEST.json and the dashboard. Everything here is cheap (file reads and
    one snaphu usage call); nothing runs in the MiaplPy env."""
    env = getattr(cfg, "miaplpy_env", "miaplpy") if cfg is not None else "miaplpy"
    pdir = project_dir or (getattr(cfg, "project_dir", None) if cfg is not None else None)
    mv = miaplpy_version(env)
    out = {"version": __version__, "code_hash": code_hash(),
           "python": ".".join(map(str, sys.version_info[:3])),
           "miaplpy": {"version": mv, "patched_build": bool(mv and "overlapdatum" in mv), "env": env}
           if mv else None,
           "snaphu": snaphu_version(getattr(cfg, "stack_unw_snaphu_bin", "") or "")}
    chk = miaplpy_patch_check(pdir) if pdir else None
    if chk and out["miaplpy"] is not None:
        out["miaplpy"]["patch_check"] = chk
    elif chk:
        out["miaplpy"] = {"version": chk.get("miaplpy_version"), "patch_check": chk}
    return out


def label() -> str:
    """One line for a footer or a header: 'insar_pipeline 2026.9.28 (code 1a2b3c4d5e6f)'."""
    return f"insar_pipeline {__version__} (code {code_hash()})"


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2))
