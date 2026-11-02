"""Local change layer (Route B) for a stack-route project processed at fine looks.

    python scripts/run_local_change.py <config.yaml> [--steps partners,filter,steps,summary]
                                       [--compare [name=]export.csv ...] [--workers N]
                                       [--box name=lat,lon,radius_m ...] [--period YYYYMMDD-YYYYMMDD ...]
                                       [--tag NAME --sigma-km KM] [--out DIR]

Writes <project>/local_change/: partner flags, filtered consecutive-date pairs,
steps.h5 (one wrapped step per pair, mm LOS) and GeoTIFFs (rate while observed,
accepted share, counted displacement and steps, noise, t statistic, and the
significance-masked rate); local_change.json records the settings and outcome.
Screening only: the layer failed its pre-registered validation at Site D on
22 Oct 2026 (USER_GUIDE 6.10); use it to point to places to check by other means.
--compare sets vendor point exports beside ours over the dates BOTH cover:
per group the share of steps kept and the median rate while observed, then the
group-to-group differences. Those are NOT datum-free: each group keeps its own
steps, so what they are read against moves them. At Site D (22 Oct 2026) a 4 km
against an 8 km reference moved this layer's differences by up to 0.56 in/yr
(configuration 2); moving the reference point moved the official series'
differences by up to 0.27 in/yr over the vendor's 20 months
(compare_vendor_points.py --reference-box).
--box samples every cell within radius_m of a point (e.g. a town or a spoil heap)
over each --period (default: the whole record), for checks such as "stable
ground stays still". --tag/--sigma-km re-run only the steps and summary into
local_change/variants/<tag>/ (e.g. an 8 km reference to test scale stability),
reusing the partners and filtered pairs. Every comparison is also written to
<out>/compare.json (default: the run's folder).

The project needs real merged SLCs (stack_virtual_merge: "False") at fine
looks. The 12-day pairs come from the stack route's fine.int or, when there
are none, are formed from the merged SLCs, so the project can merge the
phase-linked way and stop:
    python scripts/run_pipeline.py <miaplpy-backend config> --to-step merge_reference_secondary_slc
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

from insar_pipeline import local_change as lc, vendor_points as vp      # noqa: E402
from insar_pipeline.config import load_config                       # noqa: E402

IN = 25.4


def theirs_over(ex, t0: str, t1: str) -> tuple[float, float]:
    """the vendor's median kept share and rate while observed (in/yr) over columns in (t0, t1]."""
    cols = [j for j, d in enumerate(ex.dates) if t0 < d <= t1]
    if not cols:
        return np.nan, np.nan
    j0 = max([j for j, d in enumerate(ex.dates) if d <= t0] or [0])
    C = vp.carried(ex)
    steps = vp.column_steps_days(ex.dates)
    disp = C[:, cols[-1]] - C[:, j0]
    yrs = (ex.valid[:, cols] * steps[cols]).sum(1) / vp.YEAR_DAYS
    has = ex.valid[:, cols].any(1)
    if not has.any():
        return np.nan, np.nan
    rate = np.where(yrs > 0.05, disp / np.maximum(yrs, 1e-9), np.nan)
    return float(ex.valid[:, cols][has].mean()), float(np.nanmedian(rate[has]))


def _finite(v):
    """JSON-safe copy: NaN and inf become null."""
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def box_points(lat: float, lon: float, radius_m: float, step_m: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Points on a step_m grid within radius_m of (lat, lon)."""
    n = int(radius_m // step_m)
    dy, dx = np.mgrid[-n:n + 1, -n:n + 1] * step_m
    keep = dy ** 2 + dx ** 2 <= radius_m ** 2
    return (lat + dy[keep] / 111320.0,
            lon + dx[keep] / (111320.0 * np.cos(np.radians(lat))))


def parse_box(arg: str) -> tuple[str, float, float, float]:
    name, _, rest = arg.partition("=")
    try:
        la, lo, r = (float(x) for x in rest.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--box {arg!r}: want name=lat,lon,radius_m") from None
    return name, la, lo, r


def parse_period(arg: str) -> tuple[str, str]:
    a, _, b = arg.partition("-")
    if not (len(a) == 8 and len(b) == 8 and a.isdigit() and b.isdigit() and a < b):
        raise argparse.ArgumentTypeError(f"--period {arg!r}: want YYYYMMDD-YYYYMMDD")
    return a, b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--steps", default=",".join(lc.STEPS))
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--box", action="append", type=parse_box, default=[])
    ap.add_argument("--period", action="append", type=parse_period, default=[])
    ap.add_argument("--tag", default=None, help="variant name: re-run steps,summary into local_change/variants/<tag>")
    ap.add_argument("--sigma-km", type=float, default=None, help="reference width for this run (with --tag)")
    ap.add_argument("--out", default=None, help="folder for compare.json (default: the run's folder)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    if a.sigma_km is not None and not a.tag:
        ap.error("--sigma-km changes the steps: give a --tag so the base run is not overwritten")
    if a.tag and a.steps == ",".join(lc.STEPS):
        a.steps = "steps,summary"
    steps = tuple(x.strip() for x in a.steps.split(",") if x.strip())
    bad = [s for s in steps if s not in lc.STEPS]
    if bad:
        ap.error(f"unknown --steps {bad}; choose from {lc.STEPS}")
    exports = []
    for arg in a.compare:
        label, _, path = arg.rpartition("=") if "=" in arg and not Path(arg).exists() else ("", "", arg)
        if not Path(path).exists():
            ap.error(f"--compare file not found: {path}")
        exports.append((label or Path(path).stem, path))
    cfg = load_config(a.config)
    if a.sigma_km is not None:
        cfg.local_change_sigma_km = a.sigma_km
    run_dir = lc.variant_dir(cfg, a.tag) if a.tag else lc.out_dir(cfg)
    if steps:
        rec = lc.run(cfg, steps=steps, workers=a.workers, variant=a.tag)
        print(f"local change written to {run_dir} ({rec.get('seconds')} s)")
    sf = run_dir / "steps.h5"
    out = {"steps_file": str(sf), "sigma_km": float(cfg.local_change_sigma_km), "exports": {}, "differences": {},
           "boxes": {}}
    if exports or a.box:
        import h5py
        with h5py.File(sf, "r") as h:
            pairs = [x.decode() for x in h["pair"][:]]
    if exports:
        res = {}
        print(f"\n{'group':12s} {'window':17s} {'kept%':>6s} {'(vendor)':>6s} {'rate/obs in/yr':>15s} {'(vendor)':>7s}  near")
        for label, path in exports:
            ex = vp.read_export(path)
            t0 = max(ex.dates[0], pairs[0][:8]); t1 = min(ex.dates[-1], pairs[-1][9:])
            g = lc.sample(cfg, sf, ex.lat, ex.lon, start=t0, end=t1)
            share_t, rate_t = theirs_over(ex, t0, t1)
            ours_share = float(np.nanmean(g["accepted_share"])) if np.isfinite(g["accepted_share"]).any() else np.nan
            res[label] = (float(np.nanmedian(g["rate_mm_yr"])) / IN if np.isfinite(g["rate_mm_yr"]).any() else np.nan, rate_t)
            out["exports"][label] = {"file": path, "window": [t0, t1], "ours_kept_share_of_pairs": ours_share,
                                     "theirs_kept_share_of_dates": share_t, "ours_rate_in_yr": res[label][0],
                                     "theirs_rate_in_yr": rate_t, "near": int(g["near"].sum()), "points": len(g["near"])}
            fmt = lambda v, w, p=2, pct=False: (f"{100 * v:{w}.0f}" if pct else f"{v:+{w}.{p}f}") if np.isfinite(v) else f"{'n/a':>{w}s}"  # noqa: E731
            print(f"{label[:12]:12s} {t0}-{t1} {fmt(ours_share, 6, pct=True)} {fmt(share_t, 6, pct=True)} "
                  f"{fmt(res[label][0], 15)} {fmt(rate_t, 7)}  {int(g['near'].sum())}/{len(g['near'])}")
        names = list(res)
        print("\ndifferences (in/yr while observed): ours | vendor")
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                o = res[x][0] - res[y][0]; t = res[x][1] - res[y][1]
                print(f"  {x} - {y}: {o:+.2f} | " + (f"{t:+.2f}" if np.isfinite(t) else "n/a"))
                out["differences"][f"{x} - {y}"] = {"ours_in_yr": o, "theirs_in_yr": t}
    if a.box:
        periods = a.period or [(pairs[0][:8], pairs[-1][9:])]
        print(f"\n{'box':14s} {'period':17s} {'cells':>6s} {'kept%':>6s} {'rate/obs mm/yr':>15s} {'p10':>7s} {'p90':>7s}")
        for name, la, lo, r in a.box:
            plat, plon = box_points(la, lo, r)
            out["boxes"][name] = {"lat": la, "lon": lo, "radius_m": r, "periods": {}}
            for t0, t1 in periods:
                g = lc.sample(cfg, sf, plat, plon, start=t0, end=t1)
                rate = g["rate_mm_yr"][g["near"]]
                fin = rate[np.isfinite(rate)]
                kept = float(np.nanmean(g["accepted_share"][g["near"]])) if g["near"].any() else np.nan
                row = {"cells": int(g["near"].sum()), "kept_share": kept,
                       "rate_mm_yr_median": float(np.median(fin)) if fin.size else None,
                       "rate_mm_yr_p10": float(np.percentile(fin, 10)) if fin.size else None,
                       "rate_mm_yr_p90": float(np.percentile(fin, 90)) if fin.size else None}
                out["boxes"][name]["periods"][f"{t0}-{t1}"] = row
                f = lambda v: f"{v:+7.1f}" if v is not None else f"{'n/a':>7s}"  # noqa: E731
                print(f"{name[:14]:14s} {t0}-{t1} {row['cells']:6d} {100 * kept:6.0f} {f(row['rate_mm_yr_median']):>15s} "
                      f"{f(row['rate_mm_yr_p10'])} {f(row['rate_mm_yr_p90'])}")
    if exports or a.box:
        od = Path(a.out) if a.out else run_dir
        od.mkdir(parents=True, exist_ok=True)
        tmp = od / "compare.json.tmp"
        tmp.write_text(json.dumps(_finite(out), indent=1, allow_nan=False))
        tmp.replace(od / "compare.json")
        print(f"\ncomparisons written to {od / 'compare.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
