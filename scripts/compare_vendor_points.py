"""Compare a vendor's point exports with our product, like for like.

    python scripts/compare_vendor_points.py <config.yaml> [name=]<export.csv> [...] [--out DIR]

Each export (a vendor "cumulative_displacement_Line_of_Sight" CSV) is read,
its rate identity checked (rate = displacement / accepted time), and our
series at the same points put through the same bookkeeping (insar_pipeline.
vendor_points): only intervals the vendor accepted count, over the dates both products
have. Group-to-group differences (for example pit edge minus farm) are the
fair comparison between products, but they are NOT datum-free: the groups
accept different intervals, so moving the reference (--reference-box) moved
them by up to 0.27 in/yr at Site D (22 Oct 2026 review). Quote that sensitivity.

Writes <out>/vendor_comparison.json and vendor_points_<name>.csv
(default out: <project>/analysis/vendor_compare - outside deliverables/, which
the client packager copies). Read-only on the project.
"""
import argparse
import csv
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

from insar_pipeline import vendor_points as vp                           # noqa: E402
from insar_pipeline.config import load_config                        # noqa: E402
from insar_pipeline.crosscheck import product, timeseries_file       # noqa: E402

log = logging.getLogger("compare_vendor_points")
PAIR_KEYS = ("their_rate_observed_in_yr", "our_rate_observed_in_yr", "their_gated_in", "our_gated_in",
             "our_straight_line_in_yr")


def name_of(path: Path) -> str:
    s = path.stem.replace("cumulative_displacement_Line_of_Sight", "").strip(" -_()")
    return s or path.stem


def cell_context(cfg, rows, cols, shape, attrs) -> dict:
    """Our coherence, acceptance and land cover at the export's cells."""
    from insar_pipeline import deliverables as dl, landcover
    import h5py
    out = {}
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    r, c = rows[ok], cols[ok]
    with h5py.File(product(cfg, "temporalCoherence.h5"), "r") as f:
        tc = f["temporalCoherence"][:]
    out["our_coherence_median"] = round(float(np.nanmedian(tc[r, c])), 2) if ok.any() else None
    try:
        out["our_accepted_share"] = round(float(dl.load_mask(cfg)[r, c].mean()), 3)
    except Exception as exc:                                             # noqa: BLE001
        out["our_accepted_share"] = None
        log.warning("product mask unavailable (%s)", exc)
    got = landcover.classes_on_grid(cfg, attrs, shape)
    if got is not None:
        cls, scheme = got
        labels = landcover.LABELS[scheme]
        u, n = np.unique(cls[r, c], return_counts=True)
        out["land_cover_percent"] = {labels.get(int(k), str(int(k))): round(100.0 * v / n.sum(), 1)
                                     for k, v in zip(u, n) if v}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference-box", default=None, metavar="S,N,W,E",
                    help="re-measure our series against the median of the built-up, coherent cells in this "
                         "box (e.g. a town) instead of the product's reference pixel")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    ts = timeseries_file(cfg)
    out_dir = Path(a.out) if a.out else Path(cfg.project_dir) / "analysis" / "vendor_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    import h5py
    with h5py.File(ts, "r") as f:
        attrs = dict(f.attrs)
        shape = f["timeseries"].shape[1:]
    ref_series, ref_note = None, None
    if a.reference_box:
        from insar_pipeline import landcover
        box = tuple(float(x) for x in a.reference_box.split(","))
        with h5py.File(product(cfg, "temporalCoherence.h5"), "r") as f:
            tc = f["temporalCoherence"][:]
        keep = tc >= cfg.min_temporal_coherence
        got = landcover.classes_on_grid(cfg, attrs, shape)
        if got is not None:
            cls, scheme = got
            keep &= np.isin(cls, (50,) if scheme == "worldcover" else (21, 22, 23, 24))
        ref_series, n_ref = vp.area_series(ts, box, keep)
        ref_note = {"box_south_north_west_east": list(box), "cells": n_ref,
                    "rule": "median of built-up cells with temporal coherence >= the product threshold"}
        log.info("Our series re-measured against %d built-up cells in %s", n_ref, box)
    groups = {}
    for arg in a.exports:
        label, _, path = arg.rpartition("=") if "=" in arg and not Path(arg).exists() else ("", "", arg)
        p = Path(path)
        label = label or name_of(p)
        ex = vp.read_export(p)
        ours, dates, rows, cols = vp.ours_at_points(ts, ex.lat, ex.lon)
        if ref_series is not None:
            ours = ours - ref_series[None, :]
        g = vp.gated(ex, ours, dates)
        s = vp.summarize(ex, g)
        s.update(cell_context(cfg, rows, cols, shape, attrs))
        s["centre"] = [round(float(np.median(ex.lat)), 5), round(float(np.median(ex.lon)), 5)]
        s["file"] = p.name
        groups[label] = s
        with open(out_dir / f"vendor_points_{label.replace(' ', '_')}{'_refbox' if ref_note else ''}.csv", "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["point_id", "latitude", "longitude", "their_listed_rate_in_yr", "accepted_years_common",
                        "their_gated_in", "our_gated_in", "our_straight_line_in_yr"])
            for i, pid in enumerate(ex.ids):
                w.writerow([pid, ex.lat[i], ex.lon[i], ex.rate[i], round(float(g["years"][i]), 4),
                            round(float(g["their_d"][i]), 4), round(float(g["our_d"][i]), 4),
                            None if not np.isfinite(g["our_fit"][i]) else round(float(g["our_fit"][i]), 4)])
    pairs = {}
    for x, y in combinations(groups, 2):
        cx, cy = groups[x]["common"], groups[y]["common"]
        pairs[f"{x} minus {y}"] = {k: (round(cx[k] - cy[k], 3) if cx.get(k) is not None and cy.get(k) is not None
                                       else None) for k in PAIR_KEYS}
    rec = {"product": str(ts), "reference": ref_note or "the product's reference pixel", "units": "inches LOS, negative = away from the satellite",
           "method": vp.__doc__.strip().split("\n\n")[0], "groups": groups, "differences": pairs}
    tag = "_refbox" if ref_note else ""
    (out_dir / f"vendor_comparison{tag}.json").write_text(json.dumps(rec, indent=1))
    hdr = f"{'group':12s} {'pts':>5s} {'acc%':>5s} {'listed':>7s} {'theirs/obs':>10s} {'ours/obs':>9s} " \
          f"{'their_gated':>11s} {'our_gated':>9s} {'our_line':>8s} {'corr':>5s} {'sd o/t':>6s}"
    print(hdr)
    for k, s in groups.items():
        c = s["common"]
        print(f"{k[:12]:12s} {s['points']:5d} {100 * c['accepted_share']:5.0f} {s['their_listed_rate_in_yr']:+7.2f} "
              f"{c['their_rate_observed_in_yr']:+10.2f} {c['our_rate_observed_in_yr']:+9.2f} "
              f"{c['their_gated_in']:+11.2f} {c['our_gated_in']:+9.2f} {c['our_straight_line_in_yr']:+8.2f} "
              f"{c['interval_median_corr'] if c['interval_median_corr'] is not None else float('nan'):5.2f} "
              f"{c['interval_median_sd_ratio_ours_over_theirs'] or float('nan'):6.2f}")
    print("\ndifferences (in/yr or in):  theirs/obs  ours/obs  their_gated  our_gated  our_line")
    for k, d in pairs.items():
        print(f"{k:28s} " + "  ".join(f"{d[x]:+.2f}" if d[x] is not None else "  n/a" for x in PAIR_KEYS))
    print(f"\nwrote {out_dir / f'vendor_comparison{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
