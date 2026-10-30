"""Compare independent products of the same site cell for cell, each read the
way the pipeline reads it: class-matched rate relative to its own far-field
farmland (significance.class_matched_field), on the coarsest product's grid.

    python scripts/compare_products.py <cfg_first> <cfg_second> [<cfg_third> ...]

The first config's grid is the comparison grid (put the coarsest product
first: DISP-S1 at 30 m); finer products are block-averaged onto it when their
grid nests (same origin, integer factor), else sampled by nearest cell.
Writes <first project>/deliverables/product_comparison.json and
figures/product_comparison.png. Reports, for every pair of products:
  * robust scatter and correlation of the relative rates, over all shared
    cells, by the first product's coherence class, and after 1 km averaging
    (field noise averages away; ground motion does not);
  * each product's three report areas (client_report.json);
  * the fixed candidate window from the delivered record's multi-scale search,
    measured in every product against that product's own 4 km farmland windows.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

from insar_pipeline.config import load_config                      # noqa: E402
from insar_pipeline.crosscheck import (                            # noqa: E402  (the shared comparison core)
    agree, block, grid, los_up_median, onto, rate_over_window, robust_sd, timeseries_file, window,
    WINDOW_TOLERANCE_DAYS)
from insar_pipeline.panel_history import _Context                  # noqa: E402

log = logging.getLogger("compare")


def label(cfg) -> str:
    return {"disp": "OPERA DISP-S1", "miaplpy": "Phase-linked record", "stack": "Classic record"}.get(cfg.backend, cfg.backend)


def _days(a: str, b: str) -> int:
    from datetime import datetime
    return abs((datetime.strptime(a, "%Y%m%d") - datetime.strptime(b, "%Y%m%d")).days)


def main(paths):
    cfgs = [load_config(p) for p in paths]
    ctxs = [_Context(c) for c in cfgs]
    names = [label(c) for c in cfgs]
    # one date window for every product
    d0 = max(ctx.dates[0] for ctx in ctxs)
    d1 = min(ctx.dates[-1] for ctx in ctxs)
    refit = []
    for n, ctx in zip(names, ctxs):
        if _days(ctx.dates[0], d0) > WINDOW_TOLERANCE_DAYS or _days(ctx.dates[-1], d1) > WINDOW_TOLERANCE_DAYS:
            log.info("%s: refitting the rate over the common window %s-%s (record %s-%s)", n, d0, d1,
                     ctx.dates[0], ctx.dates[-1])
            v = rate_over_window(timeseries_file(ctx.cfg), d0, d1) * 1000.0
            ctx.vel = v
            ctx.good = ctx.good & np.isfinite(v)
            ctx.r_cm, ctx.class_means = ctx.sg.class_matched_field(v, ctx.tc, ctx.far)
            refit.append(n)
    # each product's own LOS -> vertical factor, from its own geometry
    ups = [los_up_median(c, ctx.good) for c, ctx in zip(cfgs, ctxs)]
    G = grid(ctxs[0].attrs)
    rel = []
    for ctx in ctxs:
        f = np.where(ctx.good, ctx.r_cm, np.nan)
        rel.append(f if grid(ctx.attrs) == G else onto(f, grid(ctx.attrs), G))
    # compare inside the AOI only: the geocoded stack grids extend past it
    from insar_pipeline import significance as sg
    aoi = sg.aoi_mask(cfgs[0].bbox, ctxs[0].attrs, rel[0].shape)
    rel = [np.where(aoi, f, np.nan) for f in rel]
    kcls = ctxs[0].kcls
    labels = ["0.4-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    px_km = abs(G["Y_STEP"]) * 110.95
    k1 = max(1, int(round(1.0 / px_km)))
    out = {"grid": G, "common_window": [d0, d1], "refitted_to_common_window": refit,
           "products": [{"name": n, "config": str(p), "project": str(c.project_dir), "backend": c.backend,
                         "los_up": round(u, 4) if u else None}
                        for n, p, c, u in zip(names, paths, cfgs, ups)], "pairs": []}
    for i in range(len(ctxs)):
        for j in range(i + 1, len(ctxs)):
            a, b = rel[i], rel[j]
            rec = {"a": names[i], "b": names[j], "all_cells": agree(a, b), "by_coherence_class_of_" + names[0]: []}
            for k, lab in enumerate(labels):
                sel = kcls == k
                r = agree(np.where(sel, a, np.nan), np.where(sel, b, np.nan))
                if r.get("n", 0) >= 200:
                    rec["by_coherence_class_of_" + names[0]].append({"tc": lab, **r})
            rec["one_km_means"] = agree(block(a, k1), block(b, k1))
            out["pairs"].append(rec)
            log.info("%s vs %s: all cells %s; 1 km %s", names[i], names[j], rec["all_cells"], rec["one_km_means"])

    # the three report areas, as each product's client report gives them
    out["report_areas"] = []
    for n, c in zip(names, cfgs):
        f = Path(c.project_dir) / "deliverables" / "client_report.json"
        if f.exists():
            d = json.loads(f.read_text())
            out["report_areas"].append({"product": n, "verdict": d.get("verdict"), "areas": [
                {k: a.get(k) for k in ("name", "cells", "vertical_mm_yr", "vertical_uncertainty_1sigma_mm_yr",
                                       "relative_vertical_mm_yr", "detection_limit_mm_yr", "total_mm", "z")}
                for a in d.get("areas", [])]})

    # the candidate window, fixed where the delivered record found it
    cand = None
    for c in cfgs:
        if c.backend == "miaplpy":
            q = json.loads((Path(c.project_dir) / "quality_assessment.json").read_text())
            cand = (q["findings"].get("monitoring_point") or {}).get("standout")
    if cand:
        lat, lon, km = cand["lat"], cand["lon"], cand["scale_km"]
        k4 = max(1, int(round(km / px_km)))
        rows = []
        for n, ctx, f, up in zip(names, ctxs, rel, ups):
            v, ncell = window(f, G, lat, lon, km)
            far = np.where(onto(ctx.far.astype(float), grid(ctx.attrs), G) >= 0.5, f, np.nan) \
                if grid(ctx.attrs) != G else np.where(ctx.far, f, np.nan)
            null = block(far, k4)
            null = null[np.isfinite(null)]
            sd = robust_sd(null)
            q = json.loads((Path(ctx.cfg.project_dir) / "quality_assessment.json").read_text())
            own = (q["findings"].get("monitoring_point") or {}).get("multi_scale") or []
            own4 = next((m for m in own if abs(m["scale_km"] - km) < 1e-6), None)
            rows.append({"product": n, "window_relative_los_mm_yr": round(v, 2),
                         "window_relative_vertical_mm_yr": round(v / up, 2) if up else None, "cells": ncell,
                         "farmland_4km_windows": int(null.size), "farmland_spread_1sigma_mm_yr": round(sd, 2),
                         "z": round(v / sd, 2) if sd else None,
                         "rank_among_farmland_windows": int((null < v).sum()) + 1,
                         "own_deepest_4km": ({k: own4[k] for k in ("lat", "lon", "rate_relative_mm_yr", "gap_in_spreads", "stands_out")}
                                             if own4 else None)})
            log.info("candidate window in %s: %.2f mm/yr LOS (z %.2f, %d farmland windows)", n, v, v / sd if sd else np.nan, null.size)
        out["candidate_window"] = {"lat": lat, "lon": lon, "scale_km": km, "products": rows}

    P = Path(cfgs[0].project_dir) / "deliverables"
    (P / "product_comparison.json").write_text(json.dumps(out, indent=1))
    figure(out, rel, names, G, cand, P / "figures" / "product_comparison.png", k1, cfgs[0].bbox)
    log.info("wrote %s", P / "product_comparison.json")


def figure(out, rel, names, G, cand, path, k1, bbox):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.ndimage import uniform_filter
    n = len(rel)
    ext = (G["X_FIRST"], G["X_FIRST"] + G["WIDTH"] * G["X_STEP"], G["Y_FIRST"] + G["LENGTH"] * G["Y_STEP"], G["Y_FIRST"])

    def smooth(f, k):
        v = np.where(np.isfinite(f), f, 0.0); w = np.isfinite(f).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = uniform_filter(v, k) / uniform_filter(w, k)
        s[uniform_filter(w, k) < 0.3] = np.nan
        return s
    k5 = max(3, k1 // 2)
    sm = [smooth(f, k5) for f in rel]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cols = max(n, len(pairs))
    fig, axs = plt.subplots(2, cols, figsize=(5.2 * cols, 10), squeeze=False)
    for i in range(cols):
        ax = axs[0, i]
        if i < n:
            im = ax.imshow(sm[i], extent=ext, cmap="RdBu", norm=TwoSlopeNorm(0, -10, 10), interpolation="nearest")
            ax.set_title(f"{names[i]}\nrelative rate, ~{k5 * abs(G['Y_STEP']) * 111:.1f} km smoothing", fontsize=10)
        else:
            ax.axis("off")
    for i, (a, b) in enumerate(pairs):
        ax = axs[1, i]
        im2 = ax.imshow(sm[a] - sm[b], extent=ext, cmap="PuOr", norm=TwoSlopeNorm(0, -6, 6), interpolation="nearest")
        pr = next(p for p in out["pairs"] if p["a"] == names[a] and p["b"] == names[b])
        ax.set_title(f"{names[a]} minus {names[b]}\ncells: r {pr['all_cells'].get('correlation')}, "
                     f"1 km: r {pr['one_km_means'].get('correlation')}", fontsize=10)
    for i in range(len(pairs), cols):
        axs[1, i].axis("off")
    for ax in axs.ravel():
        if ax.axison:
            ax.tick_params(labelsize=7)
            ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
            if cand:
                dlat = cand["scale_km"] / 110.95 / 2; dlon = cand["scale_km"] / (111.32 * np.cos(np.radians(cand["lat"]))) / 2
                ax.add_patch(plt.Rectangle((cand["lon"] - dlon, cand["lat"] - dlat), 2 * dlon, 2 * dlat,
                                           fill=False, ec="k", lw=1.2, ls="--"))
    fig.colorbar(im, ax=axs[0, :n].tolist(), fraction=0.02, pad=0.01).set_label("mm/yr LOS, relative to farmland of the same coherence", fontsize=8)
    if pairs:
        fig.colorbar(im2, ax=axs[1, :len(pairs)].tolist(), fraction=0.02, pad=0.01).set_label("difference, mm/yr", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
