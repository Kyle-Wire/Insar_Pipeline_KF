"""Long-pair feature check: how a feature moves against the ground around it, year by year.

    python scripts/run_feature_check.py <config.yaml> [--area name=lat,lon,radius_m,ring_in_m,ring_out_m ...]
           [--vendor name=export.csv[,buffer_m,ring_in_m,ring_out_m] ...] [--sep-days 182,365]
           [--dem-error geo_demErr.h5] [--landcover worldcover.tif] [--start YYYYMMDD] [--end YYYYMMDD] [--out FILE]
           [--groups 40]

Reads the project's merged full-resolution SLCs (any route that keeps them, e.g. the phase-linked-style merge)
and writes one JSON (default <project>/feature_check/feature_check.json): per area and separation, the pairs
used, each pair's change, and a velocity per calendar year (mm/yr LOS, + towards the satellite) against the
ring, with a jackknife standard error. See insar_pipeline/feature_check.py for the method and its limits.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline import feature_check as fc            # noqa: E402
from insar_pipeline.config import load_config             # noqa: E402


def parse_area(arg: str) -> dict:
    name, _, rest = arg.partition("=")
    try:
        la, lo, r, ri, ro = (float(x) for x in rest.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--area {arg!r}: want name=lat,lon,radius_m,ring_in_m,ring_out_m") from None
    return {"name": name, "lat": la, "lon": lo, "radius_m": r, "ring_m": (ri, ro)}


def parse_vendor(arg: str) -> dict:
    from insar_pipeline import vendor_points as vp
    name, _, rest = arg.partition("=")
    parts = rest.split(",")
    path = parts[0]
    if not Path(path).exists():
        raise argparse.ArgumentTypeError(f"--vendor {arg!r}: {path} not found")
    buf, ri, ro = (float(x) for x in parts[1:4]) if len(parts) >= 4 else (20.0, 400.0, 1200.0)
    ex = vp.read_export(path)
    ok = np.isfinite(ex.lat) & np.isfinite(ex.lon)
    return {"name": name, "lat": float(np.median(ex.lat[ok])), "lon": float(np.median(ex.lon[ok])),
            "points": np.c_[ex.lat[ok], ex.lon[ok]].tolist(), "buffer_m": buf, "ring_m": (ri, ro), "source": path}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--area", action="append", type=parse_area, default=[])
    ap.add_argument("--vendor", action="append", type=parse_vendor, default=[])
    ap.add_argument("--sep-days", default="182,365")
    ap.add_argument("--tol-days", type=float, default=18.0)
    ap.add_argument("--dem-error", default=None)
    ap.add_argument("--baselines", default=None)
    ap.add_argument("--landcover", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--groups", type=int, default=40,
                    help="jackknife blocks of start dates (default 40, ~3 months over 10 years; "
                         "the 22 Oct 2026 check used 10 with earlier code)")
    a = ap.parse_args(argv)
    areas = a.area + a.vendor
    if not areas:
        ap.error("give at least one --area or --vendor")
    if a.groups < 3:
        ap.error("--groups must be at least 3 (a jackknife needs three blocks)")
    cfg = load_config(a.config)
    stack = fc.Stack(Path(cfg.stack_dir), int(cfg.stack_range_looks))
    seps = tuple(float(x) for x in a.sep_days.split(","))
    dem = Path(a.dem_error) if a.dem_error else None
    bl = Path(a.baselines) if a.baselines else Path(cfg.stack_dir) / "baselines"
    rec = {"config": a.config, "separations_days": seps, "tolerance_days": a.tol_days,
           "dem_error": str(dem) if dem else None, "landcover": a.landcover, "jackknife_groups": a.groups,
           "areas": {}}
    coh = lambda x: f"{x:.2f}" if x is not None else "n/a"      # noqa: E731 - None: no pairs at this separation
    for area in areas:
        r = fc.check_area(stack, area, seps, a.tol_days, dem, bl if dem else None,
                          Path(a.landcover) if a.landcover else None, a.start, a.end, n_groups=a.groups)
        rec["areas"][area["name"]] = r
        print(f"\n{area['name']}: {r['feature_pixels']} feature px, {r['ring_pixels']} ring px "
              f"(ring {r['ring_m'][0]:.0f}-{r['ring_m'][1]:.0f} m), DEM-error term {'on' if r['dem_error'] else 'off'}")
        for sep, s in r["separations"].items():
            print(f"  ~{sep}-day pairs: {s['pairs_kept']}/{s['pairs']} kept; coherence feature {coh(s['coherence_feature_median'])}, "
                  f"ring {coh(s['coherence_ring_median'])}; whole-record rate "
                  + (f"{s['whole_record_rate_mm_yr']:+.1f} mm/yr" if s['whole_record_rate_mm_yr'] is not None else "n/a"))
            if s["velocity_mm_yr"]:
                cells = [f"{y}: " + (f"{v:+5.1f}" if v is not None else "  n/a")
                         + (f"±{e:.1f}" if e is not None else ("±n/a" if v is not None else ""))
                         for y, v, e in zip(s["years"], s["velocity_mm_yr"], s["se_mm_yr"])]
                print("    " + " | ".join(cells))
                if any(v is not None and e is None for v, e in zip(s["velocity_mm_yr"], s["se_mm_yr"])):
                    print("    ±n/a: no error, weakly determined (a refit without one block can't solve the year); "
                          "don't judge it")
    out = Path(a.out) if a.out else Path(cfg.project_dir) / "feature_check" / "feature_check.json"
    print(f"\nwritten {fc.write(out, rec)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
