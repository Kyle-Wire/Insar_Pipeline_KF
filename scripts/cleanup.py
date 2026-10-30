"""Free disk space on a FINISHED project, in levels. Shows what it would
delete (and how much) unless --delete is given.

    insar cleanup <config.yaml>                        # level 1, dry run
    insar cleanup <config.yaml> --level 2              # levels 1+2, dry run
    insar cleanup <config.yaml> --level 2 --delete     # asks for a typed 'yes' at a terminal
    (or python -m insar_pipeline.scripts.cleanup ... with the same arguments)

Level 1 - history and scratch:
  each earlier MintPy run that a re-run moved into mintpy/superseded/
  run_<stamp>/ (listed with what its RETIRED.json says it holds; deleted only
  on a finished project, like levels 2-3, since it is the only way back from
  a re-run), the pre-5-Oct-2026 measurement_points_egms.gpkg once
  points_attributes.gpkg replaces it (its geometry is off the globe),
  superseded_* / snapshot folders under deliverables/ and mintpy/, MintPy's
  diagnostic plots (mintpy/pic), MintPy's intermediate radar-grid time series
  (every timeseries*.h5 except the final one of the chain this project
  actually ran - read from mintpy/run_products.json, the geocoded final
  series and the correction tokens, never a fixed name - and except the
  series before and after the deramp until a quality assessment newer than
  both has measured the removed tilt from them), the DISP-S1
  download caches, per-pair ISCE2 scratch of the old topsApp ('local') route.
Level 2 - processing inputs. Frees the most. Afterwards only 'Products only'
  is sure to work: 'Phase linking onward', 'Re-unwrap' and 'MintPy onward'
  need the inputs deleted here, and 'Whole pipeline' downloads and
  coregisters everything again (~2 days for a 10-year stack):
  miaplpy/inputs + miaplpy/inverted (the full-resolution SLC stack and the
  linked phases), stack/merged (coregistered images and interferograms),
  downloaded SAFEs, orbits, cached ERA5 weather files. The small records at
  the top of each of these folders (slc/dropped_short_dates.json,
  frame_footprint.json, assembly status, phase-linking markers) and
  stack/SAFE_files.txt are copied to logs/records/ first; a folder whose
  records cannot be copied is not deleted.
Level 3 - large point deliverables of the old layout (the per-point CSV,
  its GeoPackage, the time-series KMZ). 'Products only' rebuilds them only
  when the config asks for them (points_full_csv / full_timeseries_kmz).
NISAR projects (backend nisar, nisar_backend.cleanup_targets): level 1 also
  lists whole GUNW granules and crops no longer used in nisar_gunw/; level 2
  the cropped windows MintPy loaded, offered only once
  mintpy/inputs/ifgramStack.h5 exists (a re-run reads them again).

Guards (review 5 Oct 2026 - this used to rmtree whatever it listed):
  - nothing is deleted while a run holds the project (dashboard, terminal or
    script run: runs.state 'running' or the project lock), not even with
    --force; --delete takes the project lock itself, so no run can start
    while it deletes.
  - levels 2 and 3 only on a FINISHED project: the last run ended 'done' and
    quality_assessment.json and (on routes that have one) client_report.json
    are newer than the velocity product. --force overrides this check only.

Never touched at any level: logs/ (every run's log is the only record of
the downloads, the coregistration, dropped dates and the MiaplPy patch
check - cleanup and archive_project used to disagree about this),
deliverables/releases/ (packaged sends), deliverables/screening_not_for_clients/,
deliverables/review.json and reviews/, config_snapshot.yaml, the point products
(points_attributes.gpkg, points_timeseries.parquet), mintpy/geo/ (the current
products), the deliverables' rasters, figures, reports and
JSON, the final geocoded MintPy products, mintpy/inputs/ifgramStack.h5 and
the geometry, the run records and the config.

Exit codes: 0 done (or dry run), 1 something was not deleted or the
question was not answered 'yes', 2 refused (a run is going, or the project
is not finished).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline.config import load_config            # noqa: E402

# MintPy appends one token per correction step to the series' name; the same
# tokens deliverables._timeseries_by_tokens ranks. A fixed final name
# ("timeseries_ERA5_ramp_demErr.h5") deleted the real final series of every
# project run without ERA5 or without deramping as "intermediate".
TS_TOKENS = ("SET", "ERA5T", "ERA5", "ERAI", "MERRA2", "MERRA", "NARR", "HRRRAK", "HRRR", "GMAO", "ramp", "demErr")
# Small records kept from a level-2 folder before it goes (top level only).
RECORD_SUFFIXES = (".json", ".txt", ".csv", ".log", ".complete")
RECORD_MAX_BYTES = 2_000_000
EXTRA_RECORDS = ("stack/SAFE_files.txt",)
RECORDS_DIR = Path("logs") / "records"
# Relative to the project: never a target, whatever a glob matches.
PROTECTED = ("logs", "deliverables/releases", "deliverables/screening_not_for_clients",
             "deliverables/review.json", "deliverables/reviews", "deliverables/config_snapshot.yaml",
             "deliverables/points_attributes.gpkg", "deliverables/points_timeseries.parquet",
             "deliverables/points_README.txt", "mintpy/geo",
             "run.json", "run_result.json", ".run.lock",
             ".run_children.json", "progress.jsonl", "context", "dem")

LOST_AT_LEVEL2 = ("After level 2 only 'Products only' is sure to work. 'Phase linking onward', 'Re-unwrap' "
                  "and 'MintPy onward' need the inputs deleted here, and 'Whole pipeline' downloads and "
                  "coregisters everything again (~2 days for a 10-year stack).")
LOST_AT_LEVEL3 = ("Level 3 files come back with 'Products only' only when the config asks for them "
                  "(points_full_csv / full_timeseries_kmz).")


def size(p: Path) -> int:
    if p.is_symlink():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _series(d: Path, prefix: str) -> list[Path]:
    """Displacement series in d (residual files are fit diagnostics, not series)."""
    if not d.is_dir():
        return []
    return [f for f in d.glob(f"{prefix}timeseries*.h5") if "Residual" not in f.name]


def _by_tokens(files: list[Path]) -> Path | None:
    def score(f: Path) -> tuple:
        return (sum(1 for t in TS_TOKENS if f"_{t}" in f.stem), len(f.stem))
    return max(files, key=score) if files else None


def final_series(cfg) -> set[str]:
    """Names (in mintpy/) of the radar-grid time series to keep: the final
    series of the chain this project ran. Every source that can name it is
    used and the union kept - keeping one file too many costs a few GB,
    deleting the final series costs a MintPy re-run:
      - mintpy/run_products.json (time_series writes it after a successful
        run): timeseries_radar, and the radar twin of timeseries_geo;
      - the radar twin of the most corrected geocoded series (geo/geo_X.h5 ->
        X.h5), which is what the deliverables read;
      - the most corrected radar series by its correction tokens;
      - the newest radar series: within one MintPy run each correction writes
        a new file after the last, so the newest is that run's final one
        (this is what catches a stale, more-tokened file from an older chain)."""
    M = Path(cfg.mintpy_dir)
    keep: set[str] = set()
    try:
        rp = json.loads((M / "run_products.json").read_text())
    except (OSError, ValueError):
        rp = {}
    if rp.get("timeseries_radar"):
        keep.add(Path(rp["timeseries_radar"]).name)
    if rp.get("timeseries_geo"):
        keep.add(Path(rp["timeseries_geo"]).name.removeprefix("geo_"))
    geo_best = _by_tokens(_series(M / "geo", "geo_"))
    if geo_best is not None:
        keep.add(geo_best.name.removeprefix("geo_"))
    radar = [f for f in _series(M, "") if f.parent == M]
    best = _by_tokens(radar)
    if best is not None:
        keep.add(best.name)
    if radar:
        keep.add(max(radar, key=_mtime).name)
    return keep


def ramp_pair_needed(cfg) -> set[str]:
    """Names in mintpy/ of a series and its deramped copy (timeseries_ERA5.h5 +
    timeseries_ERA5_ramp.h5) kept until the quality assessment has measured
    the removed ramp from them: its 'before deramp' sensitivity needs both,
    and a QA run after cleanup can only say 'not on disk' (Site A)."""
    M = Path(cfg.mintpy_dir)
    qa_f = Path(cfg.project_dir) / "quality_assessment.json"
    qa_t = _mtime(qa_f)
    try:
        qa = json.loads(qa_f.read_text())
    except (OSError, ValueError):
        qa = {}
    # measured, not merely run: a QA from before 5 Oct 2026 has no ramp block
    measured = str(((qa.get("ramp_sensitivity") or {}) if isinstance(qa, dict) else {})
                   .get("before_deramp") or "").startswith("measured")
    keep: set[str] = set()
    for rp in M.glob("timeseries*_ramp.h5"):
        if "Residual" in rp.name:
            continue
        base = rp.with_name(rp.name[:-len("_ramp.h5")] + ".h5")
        if base.exists() and (not measured or qa_t < max(_mtime(base), _mtime(rp))):
            keep |= {base.name, rp.name}
    return keep


def _retired_note(d: Path) -> str:
    try:
        r = json.loads((d / "RETIRED.json").read_text())
    except (OSError, ValueError):
        r = {}
    n = len(r.get("files") or [])
    return ("products of an earlier MintPy run" + (f" ({n} files, retired {r.get('stamp', d.name[4:])}"
                                                    + (f": {r['why']}" if r.get("why") else "") + ")" if n else "")
            + "; delete once the newer run is accepted")


def needs_finished(cfg, lv: int, p: Path) -> bool:
    """Items deleted only on a finished project: levels 2-3, and the
    previous MintPy runs kept in mintpy/superseded/run_<stamp>/."""
    if lv >= 2:
        return True
    sup = Path(cfg.mintpy_dir).absolute() / "superseded"
    p = Path(p).absolute()
    return p == sup or sup in p.parents


def _roots(cfg) -> list[Path]:
    """Where targets may live: the project, and mintpy_dir when the config
    puts it elsewhere (time_series_dir)."""
    return [Path(cfg.project_dir).absolute(), Path(cfg.mintpy_dir).absolute()]


def protected(cfg, p: Path) -> bool:
    """True for anything cleanup must never delete (PROTECTED, the config
    file's folder entries) or that lies outside the project."""
    p = Path(p).absolute()
    P = Path(cfg.project_dir).absolute()
    for rel in PROTECTED:
        q = P / rel
        if p == q or q in p.parents:
            return True
    return not any(p == r or r in p.parents for r in _roots(cfg)) or p in _roots(cfg)


def targets(cfg, level: int) -> list[tuple[int, Path, str]]:
    P, M, D = Path(cfg.project_dir), Path(cfg.mintpy_dir), Path(cfg.project_dir) / "deliverables"
    out: list[tuple[int, Path, str]] = []

    def add(lv, p, why):
        if (p.exists() or p.is_symlink()) and not protected(cfg, p) and all(p != q for _, q, _ in out):
            out.append((lv, p, why))
    # ---- level 1
    # MintPy re-runs move the previous run's products into
    # mintpy/superseded/run_<stamp>/ (time_series.retire_downstream_products).
    # Each is listed on its own with what RETIRED.json says it holds; they are
    # the only way back from a re-run, so main() deletes them only on a
    # finished project (the newer run accepted), like levels 2-3.
    for d in sorted((M / "superseded").glob("run_*")) if (M / "superseded").is_dir() else []:
        add(1, d, _retired_note(d))
    for d in list(D.glob("superseded*")) + [x for x in M.glob("superseded*") if x != M / "superseded"]:
        add(1, d, "superseded products")
    for d in D.glob("*_20[0-9][0-9][0-9][0-9][0-9][0-9]"):
        if d.is_dir() and not d.name.startswith(("final_", "superseded")):
            add(1, d, "snapshot of an earlier version")
    add(1, M / "pic", "MintPy diagnostic plots")
    keep = final_series(cfg) | ramp_pair_needed(cfg)
    for f in M.glob("timeseries*.h5"):
        if f.name not in keep:
            add(1, f, "intermediate time series")
    add(1, M / "velocityERA5.h5", "diagnostic (troposphere-only rate)")
    add(1, P / "granule_cache", "DISP-S1 download cache")
    add(1, P / "corrections_cache", "DISP-S1 download cache")
    pairs = Path(getattr(cfg, "pairs_dir", P / "interferograms"))
    if pairs.is_dir():
        try:
            from insar_pipeline.isce_processing import _SCRATCH_DIRS
        except Exception:                                     # noqa: BLE001 - the retired route's list
            _SCRATCH_DIRS = []
        for pair in sorted(pairs.iterdir()):
            for name in _SCRATCH_DIRS:
                add(1, pair / name, "per-pair ISCE2 scratch (old topsApp route)")
    # logs/ are never targets (PROTECTED): the coregistration log is the only
    # record of the download, the dropped dates and the orbit staging.
    # ---- level 2
    add(2, P / "miaplpy" / "inputs", "full-resolution SLC stack (phase-linking input)")
    add(2, P / "miaplpy" / "inverted", "linked phases (phase-linking output, already exported)")
    add(2, P / "stack" / "merged", "coregistered images and interferograms")
    add(2, P / "stack" / "coreg_secondarys", "coregistered bursts")
    add(2, P / "slc", "downloaded images")
    add(2, P / "orbits", "orbit files")
    tropo = Path(getattr(cfg, "tropo_dir", P / "tropo_corrections"))
    if tropo.is_dir():
        for d in sorted(tropo.iterdir()):
            add(2, d / "weather_files", "cached weather model files (re-downloadable from CDS)")
    # The NISAR GUNW route (8 Oct 2026): its own list (nisar_backend.cleanup_targets).
    # Level 1: whole granules and unused crops; level 2: the crops MintPy loaded,
    # offered only once mintpy/inputs/ifgramStack.h5 holds what they gave.
    if str(getattr(cfg, "backend", "")).lower() == "nisar":
        from insar_pipeline import nisar_backend
        loaded = (Path(cfg.mintpy_dir) / "inputs" / "ifgramStack.h5").exists()
        for lv, p, why in nisar_backend.cleanup_targets(cfg):
            if lv >= 2 and not loaded:
                continue
            add(lv, p, why)
    # The pre-5-Oct-2026 point GeoPackage put UTM metres in EPSG:4326 (off the
    # globe); once points_attributes.gpkg replaces it, it is stale.
    if (D / "points_attributes.gpkg").exists():
        add(1, D / "measurement_points_egms.gpkg",
            "superseded point GeoPackage (broken geometry; replaced by points_attributes.gpkg)")
    # ---- level 3
    for pat in ("measurement_points_egms.csv", "measurement_points_egms.gpkg", "geo_timeseries_*.kmz"):
        for f in D.glob(pat):
            add(3, f, "large point deliverable (rebuilt by 'Products only' only if the config asks)")
    return [t for t in out if t[0] <= level]


def _velocity_time(cfg) -> float | None:
    M = Path(cfg.mintpy_dir)
    vel = [f for f in (M / "velocity.h5", M / "geo" / "geo_velocity.h5") if f.exists()]
    return max(_mtime(f) for f in vel) if vel else None


def unfinished(cfg) -> list[str]:
    """Plain reasons the project is not finished; [] when it is. Finished =
    the last run ended 'done' (or no run record says otherwise) and the
    products were built from the current MintPy result: the quality
    assessment and, on routes that have one, the client report are newer
    than the velocity product."""
    from insar_pipeline import runs
    P = Path(cfg.project_dir)
    why = []
    st = runs.state(P)
    if st in ("failed", "stopped", "interrupted"):
        res = runs.result(P) or {}
        why.append(f"the last run ended '{st}'" + (f" during {res['stage']}" if res.get("stage") else ""))
    backend = getattr(cfg, "backend", "")
    products_route = backend in ("stack", "miaplpy") or (backend == "disp" and getattr(cfg, "disp_record_chain", False))
    vel_t = _velocity_time(cfg)
    if products_route and vel_t is None:
        why.append("MintPy has produced no velocity yet (mintpy/velocity.h5)")
    need = []
    if products_route:
        need.append(P / "quality_assessment.json")
        try:
            from insar_pipeline.pipeline import client_report_allowed
            allowed = client_report_allowed(cfg)[0]
        except Exception:                                     # noqa: BLE001 - same rule, by hand
            allowed = backend != "disp" or bool(getattr(cfg, "disp_client_report", False))
        if allowed:
            need.append(P / "deliverables" / "client_report.json")
    for f in need:
        rel = f.relative_to(P)
        if not f.exists():
            why.append(f"{rel} is missing")
        elif vel_t is not None and _mtime(f) < vel_t:
            why.append(f"{rel} is older than the velocity product (the products were not rebuilt "
                       "after the last MintPy run)")
    return why


def save_records(cfg, d: Path) -> tuple[bool, list[Path]]:
    """Copy the small records at the top of folder d into logs/records/<d>/
    and check each copy. (ok, copies); not ok means d must not be deleted."""
    P = Path(cfg.project_dir)
    try:
        rel = d.absolute().relative_to(P.absolute())
    except ValueError:
        rel = Path(d.name)
    dest = P / RECORDS_DIR / str(rel).replace("/", "__")
    copies = []
    try:
        files = [f for f in d.iterdir() if f.is_file() and not f.is_symlink()
                 and f.suffix in RECORD_SUFFIXES and f.stat().st_size <= RECORD_MAX_BYTES]
        for f in files:
            dest.mkdir(parents=True, exist_ok=True)
            out = dest / f.name
            shutil.copy2(f, out)
            if not out.is_file() or out.stat().st_size != f.stat().st_size:
                return False, copies
            copies.append(out)
    except OSError as exc:
        print(f"  could not copy the records of {d}: {exc}")
        return False, copies
    return True, copies


def _save_extra_records(cfg) -> None:
    P = Path(cfg.project_dir)
    for rel in EXTRA_RECORDS:
        f = P / rel
        if f.is_file():
            dest = P / RECORDS_DIR / rel.replace("/", "__")
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
            except OSError as exc:
                print(f"  could not copy {f}: {exc}")


def delete(cfg, items) -> tuple[int, list[tuple[Path, str]]]:
    """Delete the items and check each is gone. (bytes freed, [(path, why not)])."""
    failed: list[tuple[Path, str]] = []
    freed = 0
    if any(lv >= 2 for lv, _, _ in items):
        _save_extra_records(cfg)
    for lv, p, _ in items:
        if protected(cfg, p):                    # belt and braces: targets() filtered already
            failed.append((p, "protected"))
            continue
        n = size(p)
        if lv >= 2 and p.is_dir() and not p.is_symlink():
            ok, copies = save_records(cfg, p)
            if not ok:
                failed.append((p, "its records could not be copied to logs/records, so it was kept"))
                continue
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
        except OSError as exc:
            failed.append((p, str(exc)))
            continue
        if p.exists() or p.is_symlink():
            failed.append((p, "still there after deleting"))
        else:
            freed += n
    return freed, failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--level", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--delete", action="store_true", help="actually delete (default: show only)")
    ap.add_argument("--force", action="store_true",
                    help="levels 2-3 on a project whose last run did not finish (never while a run is going)")
    ap.add_argument("--yes", action="store_true", help="do not ask for a typed 'yes' before level 2-3")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    P = Path(cfg.project_dir)
    if not P.is_dir():
        print(f"No project folder at {P}.")
        return 1
    from insar_pipeline import runs
    busy = runs.state(P) == "running"
    if busy:
        h = runs.lock_holder(P) or runs.record(P) or {}
        what = f"A {h['kind']} run" if h.get("kind") else "A run"
        busy_text = (f"{what} is going in {P} (pid {h.get('pid', '?')}): cleanup would delete what it is "
                     "reading. Wait until it finishes, or stop it first.")
        if a.delete:
            print(f"REFUSED: {busy_text}")
            return 2
        print(f"NOTE: {busy_text} --delete is refused until then.\n")
    items = targets(cfg, a.level)
    if not items:
        print("Nothing to clean at this level.")
        return 0
    total = 0
    for lv, p, why in items:
        n = size(p)
        total += n
        print(f"  L{lv}  {human(n):>9}  {p}   ({why})")
    print(f"\n  {len(items)} items, {human(total)} {'to delete' if a.delete else 'would be freed (dry run: add --delete)'}")
    deep = sorted({lv for lv, _, _ in items if lv >= 2})
    gated = [p for lv, p, _ in items if needs_finished(cfg, lv, p)]
    if 2 in deep:
        print(f"\n  {LOST_AT_LEVEL2}")
    if 3 in deep:
        print(f"  {LOST_AT_LEVEL3}")
    why = unfinished(cfg) if gated else []
    if why:
        print("\n  This project is not finished: " + "; ".join(why) + ".")
        if not a.force:
            print("  Levels 2-3 and earlier MintPy runs (mintpy/superseded/) are "
                  + ("REFUSED" if a.delete else "refused with --delete")
                  + " until it is (finish or rebuild the products first, or add --force).")
            if a.delete:
                return 2
    if not a.delete:
        return 0
    if gated and not a.yes and sys.stdin is not None and sys.stdin.isatty():
        if input("\n  Type yes to delete these files: ").strip().lower() != "yes":
            print("  Nothing deleted.")
            return 1
    try:
        # Held while deleting: every launcher takes this lock, so no run can
        # start on the project halfway through (and a run that started after
        # the check above is caught here).
        with runs.project_lock(P, kind="cleanup", config=str(a.config)):
            freed, failed = delete(cfg, items)
    except RuntimeError as exc:
        if "already running" not in str(exc):
            raise
        print(f"REFUSED: {exc}")
        return 2
    print(f"\n  freed {human(freed)}; {len(items) - len(failed)} of {len(items)} items deleted.")
    if failed:
        print("  NOT deleted:")
        for p, reason in failed:
            print(f"    {p}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
