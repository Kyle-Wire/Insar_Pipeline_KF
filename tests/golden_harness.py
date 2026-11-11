"""Golden builds for the domain-studies plan (Phase 0 S0.2 and Gate 0, 7 Nov
2026): the client report and KMZ built so that the same inputs give the same
files on any day, after any code change that should change nothing.

    python tests/golden_harness.py make  --root R [NAME ...]           # write fixture projects
    python tests/golden_harness.py repin --root R [NAME ...]           # restore their pinned file times
    python tests/golden_harness.py build --fixture NAME --root R --out-dir D
    python tests/golden_harness.py build --config CONFIG.yaml --out-dir D

`build` runs client_report and then client_kmz with --out-dir D, each in a
fresh process (as the two CLIs run; the report path keeps in-process caches),
with these pinned:

  package     always imported as `insar_pipeline` from THIS checkout (a
              checkout under another name, e.g. a worktree, is reached through
              a link); a build that loaded any package module from elsewhere
              fails (exit 4): the fixtures and the builders must run one code
  clock       datetime.now/today/utcnow in every insar_pipeline module = FROZEN
              (patched in the package only: numpy, h5py and matplotlib keep
              the real class); TZ=UTC; SOURCE_DATE_EPOCH = FROZEN for the PDF
  code        version.__version__, code_hash, the MiaplPy and snaphu
              versions and the python version: constants. A refactor
              changes the code hash, which every output prints (page footer,
              PNG text, figure and PDF stamp, doc.kml); pinned, only what the
              code DOES shows up in a diff
  who         settings.identity() = a placeholder firm and preparer (the
              release blocker 'identity_missing' and the footer read it)
  registry    fixtures only: combine.pointer_for = None (it reads every
              config on the machine); a real config keeps the real lookup
  network     any socket connect or name lookup is refused and recorded;
              the build then fails (exit 3) rather than quietly differing
  numerics    OMP/OPENBLAS/MKL threads = 1, PYTHONHASHSEED = 0, MPLBACKEND=Agg

Each build also writes golden_inputs.json: what the build read besides the
code (a fixture's files with their hashes and times; a real project's file
sizes and times, and its combined-project pointer). compare_goldens.py
reports a change there as INPUTS CHANGED, apart from output differences, so
a re-run project or a copied fixture is not mistaken for a code change. A
fixture whose pinned file times were lost (a copy without times) is refused
before building: `repin` restores them.

Fixtures: tests/golden_fixtures.py. Comparison:
review_2026-11-07/phase0/compare_goldens.py (outside the repo record).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
import tempfile
from datetime import date as _real_date, datetime as _real_datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PKG = "insar_pipeline"
FROZEN = _real_datetime(2026, 11, 7, 12, 0, 0)    # naive, UTC (the builds run with TZ=UTC)
PINNED = {"version": "golden", "code_hash": "golden000000", "miaplpy": "golden+overlapdatum",
          "snaphu": "snaphu golden", "python": "golden"}
IDENTITY = {"firm": "Golden Test Firm", "preparer": "Golden Preparer", "contact": ""}

ENV = {"TZ": "UTC", "SOURCE_DATE_EPOCH": str(int(FROZEN.replace(tzinfo=timezone.utc).timestamp())),
       "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "PYTHONHASHSEED": "0", "INSAR_NOTIFY": "off"}
INPUTS = "golden_inputs.json"
_REAL_SKIP = ("logs", "__pycache__")                       # a real project's dirs that are not report inputs
_REAL_SKIP_NAMES = ("progress.jsonl", ".run.lock", "run.json")


# ------------------------------------------------------------------ the package (both processes)
def _package_root() -> Path:
    """The folder to put first on sys.path so that `insar_pipeline` is THIS
    checkout: its parent when it is named insar_pipeline, else a fixed link
    folder (a worktree is named after its branch)."""
    if REPO.name == PKG:
        return REPO.parent
    d = Path(tempfile.gettempdir()) / f"golden_pkg_{hashlib.sha1(str(REPO).encode()).hexdigest()[:10]}"
    d.mkdir(exist_ok=True)
    link = d / PKG
    if link.is_symlink() and link.resolve() != REPO:
        link.unlink()
    if not link.exists():
        link.symlink_to(REPO, target_is_directory=True)
    return d


def _use_this_checkout() -> None:
    sys.path.insert(0, str(_package_root()))
    sys.path.insert(1, str(HERE))


def _foreign_modules() -> list[str]:
    """Package modules (and the fixture modules) loaded from anywhere but
    this checkout: an editable install of another checkout, for one."""
    root = REPO.resolve()
    bad = []
    for name, mod in list(sys.modules.items()):
        if not (name == PKG or name.startswith(PKG + ".") or name in ("golden_fixtures", "test_fix_L5")):
            continue
        f = getattr(mod, "__file__", None)
        if f and not Path(f).resolve().is_relative_to(root):
            bad.append(f"{name}: {f}")
    return bad


# ------------------------------------------------------------------ pins (child process)
class FrozenDateTime(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN if tz is None else FROZEN.replace(tzinfo=timezone.utc).astimezone(tz)

    @classmethod
    def today(cls):
        return FROZEN

    @classmethod
    def utcnow(cls):
        return FROZEN


class FrozenDate(_real_date):
    @classmethod
    def today(cls):
        return FROZEN.date()


def _freeze(module) -> None:
    for attr, real, fake in (("datetime", _real_datetime, FrozenDateTime), ("date", _real_date, FrozenDate)):
        if getattr(module, attr, None) is real:
            setattr(module, attr, fake)


class _FreezeOnImport(importlib.abc.MetaPathFinder):
    """Freezes the clock in each package module as it is imported (lazy
    imports included: client_report loads several modules on first use)."""
    def find_spec(self, name, path, target=None):
        if not name.startswith(PKG + "."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            _freeze(module)
        spec.loader.exec_module = exec_module
        return spec


NET_ATTEMPTS: list[str] = []


def _tripwire() -> None:
    import socket

    def refuse(what):
        def f(*a, **k):
            NET_ATTEMPTS.append(f"{what} {a[1:2] if what == 'connect' else a[:1]} {sorted(k)}")
            raise OSError(f"golden build: network refused ({what})")
        return f
    socket.socket.connect = refuse("connect")
    socket.socket.connect_ex = refuse("connect")
    socket.getaddrinfo = refuse("getaddrinfo")
    socket.create_connection = refuse("create_connection")


def _pin(fixture: bool) -> None:
    _use_this_checkout()
    sys.meta_path.insert(0, _FreezeOnImport())
    _tripwire()
    version = __import__(f"{PKG}.version", fromlist=["x"])
    version.__version__ = PINNED["version"]
    version._package_hash = lambda: PINNED["code_hash"]
    version.miaplpy_version = lambda env="miaplpy": PINNED["miaplpy"]
    version.snaphu_version = lambda explicit="": PINNED["snaphu"]
    _describe = version.describe
    version.describe = lambda *a, **k: {**_describe(*a, **k), "python": PINNED["python"]}
    settings = __import__(f"{PKG}.settings", fromlist=["x"])
    settings.identity = lambda: dict(IDENTITY)
    if fixture:
        combine = __import__(f"{PKG}.combine", fromlist=["x"])
        combine.pointer_for = lambda cfg: None
    for name, mod in list(sys.modules.items()):              # anything imported before the hook
        if name.startswith(PKG + ".") and mod is not None:
            _freeze(mod)


def _cfg(a):
    if a.fixture:
        import golden_fixtures as gf
        return gf.config(Path(a.root).resolve(), a.fixture)
    load_config = __import__(f"{PKG}.config", fromlist=["x"]).load_config
    return load_config(a.config)


# ------------------------------------------------------------------ inputs record
def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree(d: Path, hashes: bool) -> dict:
    out = {}
    if not d.exists():
        return out
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d)
        if not p.is_file() or (not hashes and (rel.parts[0] in _REAL_SKIP or p.name in _REAL_SKIP_NAMES)):
            continue
        st = p.stat()
        out[rel.as_posix()] = [st.st_size, st.st_mtime_ns] + ([_sha(p)] if hashes else [])
    return out


def _inputs(cfg, fixture: bool) -> dict:
    """What this build reads besides the code. A fixture: every project file
    (size, time, sha256). A real project: its files' sizes and times (the
    products are GB; a re-run changes the times), the combined project's
    pointer and the files of every project that pointer reads."""
    P = Path(cfg.project_dir)
    rec = {"project": str(P), "files": _tree(P, hashes=fixture)}
    if not fixture:
        combine = __import__(f"{PKG}.combine", fromlist=["x"])
        ptr = combine.pointer_for(cfg)
        rec["combined_pointer"] = ptr
        if ptr:
            from importlib import import_module
            load_config = import_module(f"{PKG}.config").load_config
            cc = combine.load(ptr["config"])
            rec["combined_config_sha256"] = _sha(Path(ptr["config"]))
            for direction in ("ascending", "descending"):
                src = Path(load_config(str(getattr(cc, direction))).project_dir)
                if src.resolve() != P.resolve():
                    rec[f"source_{direction}"] = {"project": str(src), "files": _tree(src, hashes=False)}
            cdir = Path(getattr(cc, "project_dir", "") or "")
            if str(cdir):
                rec["combined_project"] = {"project": str(cdir), "files": _tree(cdir, hashes=False)}
    return rec


def _one(a) -> int:
    import logging
    # what the CLIs log (their main() sets INFO), without the logging clock:
    # compare_goldens compares the messages, so a new warning or fallback shows
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    _pin(bool(a.fixture))
    cfg = _cfg(a)
    out = Path(a.out_dir)
    if a.which == "report":
        (out / INPUTS).write_text(json.dumps(_inputs(cfg, bool(a.fixture)), indent=1, sort_keys=True, default=str))
        cr = __import__(f"{PKG}.client_report", fromlist=["x"])
        # a preview build leaves out the report-consistent rasters (the GeoTIFF
        # deliverables, in display units); keep compute's result and write
        # them into the build folder too (write_rasters' out_dir: never the
        # project's deliverables)
        kept = {}
        _compute = cr.compute

        def compute(c):
            kept["r"] = _compute(c)
            return kept["r"]
        cr.compute = compute
        print(cr.build(cfg, None, out))
        rep, ctx, _masks, _outlines = kept["r"]
        for p in cr.write_rasters(cfg, rep, ctx, out_dir=out / "rasters"):
            print(p)
    else:
        kmz = __import__(f"{PKG}.client_kmz", fromlist=["x"])
        print(kmz.build(cfg, out_dir=out))
    foreign = _foreign_modules()
    if foreign:
        print("golden build: package modules loaded from outside this checkout "
              f"({REPO}):\n  " + "\n  ".join(foreign[:20]), file=sys.stderr)
        return 4
    if NET_ATTEMPTS:
        (out / f"net_attempts_{a.which}.json").write_text(json.dumps(NET_ATTEMPTS, indent=1))
        print(f"golden build: {len(NET_ATTEMPTS)} network attempts refused: {NET_ATTEMPTS[:5]}", file=sys.stderr)
        return 3
    return 0


# ------------------------------------------------------------------ parent
def _env() -> dict:
    env = dict(os.environ, **ENV)
    env.setdefault("PROJ_DATA", str(Path(sys.executable).resolve().parents[1] / "share" / "proj"))
    return env


def _fixtures():
    _use_this_checkout()
    import golden_fixtures as gf
    return gf


def _build(a) -> int:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        print(f"{out} is not empty: a golden build starts from an empty folder", file=sys.stderr)
        return 2
    if a.fixture:
        gf = _fixtures()
        wrong = gf.check_times(gf.project_dir(Path(a.root).resolve(), a.fixture), FROZEN)
        if wrong:
            print(f"{a.fixture}: {len(wrong)} project files lost their pinned times (e.g. {wrong[:3]}); "
                  f"restore them with: python {Path(__file__).name} repin --root {a.root} {a.fixture}",
                  file=sys.stderr)
            return 2
    src = ["--fixture", a.fixture, "--root", str(Path(a.root).resolve())] if a.fixture else ["--config", str(a.config)]
    for which in ("report", "kmz"):
        with open(out / f"{which}.log", "w") as log:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_one", which, *src,
                                "--out-dir", str(out)], env=_env(), stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            print(f"{which} failed ({r.returncode}): see {out / (which + '.log')}", file=sys.stderr)
            return r.returncode
    return 0


def _make(a) -> int:
    gf = _fixtures()
    for n in a.names or list(gf.FIXTURES):
        print(gf.write(Path(a.root).resolve(), n, FROZEN))
    foreign = _foreign_modules()
    if foreign:
        print("fixtures written with package modules from outside this checkout:\n  " + "\n  ".join(foreign[:20]),
              file=sys.stderr)
        return 4
    return 0


def _repin(a) -> int:
    gf = _fixtures()
    for n in a.names or list(gf.FIXTURES):
        P = gf.project_dir(Path(a.root).resolve(), n)
        if P.exists():
            gf.pin_times(P, FROZEN)
            print(P)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="golden_harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make", "repin"):
        m = sub.add_parser(name)
        m.add_argument("--root", required=True, type=Path)
        m.add_argument("names", nargs="*")
    for name in ("build", "_one"):
        b = sub.add_parser(name)
        if name == "_one":
            b.add_argument("which", choices=("report", "kmz"))
        g = b.add_mutually_exclusive_group(required=True)
        g.add_argument("--fixture")
        g.add_argument("--config", type=Path)
        b.add_argument("--root", type=Path)
        b.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd in ("build", "_one") and a.fixture and not a.root:
        ap.error("--fixture needs --root")
    return {"make": _make, "repin": _repin, "build": _build, "_one": _one}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
