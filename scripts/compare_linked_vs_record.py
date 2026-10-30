"""
Phase-linked run vs the classic record over the SAME dates.

Fits a rate to the record's corrected geocoded series restricted to the dates
the phase-linked run contains, samples both on the record's grid (nearest
cell), and reports: agreement on high-coherence ground, the rate-by-record-
coherence-class table for BOTH (the distributed-scatterer bias test, using
the record's 10-year temporal coherence as the class label for both), and
the same figures over the focus sites. Writes JSON + a PNG beside the
phase-linked run's deliverables.

usage: python compare_linked_vs_record.py <linked config> <record config>
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from insar_pipeline.config import load_config          # noqa: E402
from insar_pipeline.panel_history import parse_focus_sites  # noqa: E402


def geo_grid(a):
    x0, y0 = float(a["X_FIRST"]), float(a["Y_FIRST"]); dx, dy = float(a["X_STEP"]), float(a["Y_STEP"])
    L, W = int(a["LENGTH"]), int(a["WIDTH"])
    return x0, y0, dx, dy, L, W


def series_file(mintpy_dir):
    c = sorted((mintpy_dir / "geo").glob("geo_timeseries*.h5"), key=lambda p: -len(p.name))
    return c[0]


def fit_rate(ts, dates_yyyymmdd):
    from datetime import datetime as dt
    t = np.array([(dt.strptime(d, "%Y%m%d") - dt.strptime(dates_yyyymmdd[0], "%Y%m%d")).days / 365.25 for d in dates_yyyymmdd])
    A = np.vstack([t, np.ones_like(t)]).T
    y = ts.reshape(len(t), -1)
    ok = np.isfinite(y).all(axis=0)
    out = np.full(y.shape[1], np.nan, np.float32)
    coef, *_ = np.linalg.lstsq(A, y[:, ok], rcond=None)
    out[ok] = coef[0]
    return out.reshape(ts.shape[1:]) * 1000.0     # m/yr -> mm/yr


def main(linked_cfg, record_cfg):
    L = load_config(linked_cfg); R = load_config(record_cfg)
    lf = series_file(L.mintpy_dir); rf = series_file(R.mintpy_dir)
    with h5py.File(lf) as f:
        ld = [d.decode() for d in f["date"][:]]; lts = f["timeseries"][:]; la = dict(f.attrs)
    with h5py.File(rf) as f:
        rd = [d.decode() for d in f["date"][:]]
        idx = [rd.index(d) for d in ld if d in rd]
        common = [rd[i] for i in idx]
        rts = f["timeseries"][idx]; ra = dict(f.attrs)
    if len(common) < 4:
        raise SystemExit(f"only {len(common)} common dates")
    lts = lts[[ld.index(d) for d in common]]
    # both relative to the first common date, then referenced to the record's reference pixel
    lts = lts - lts[0]; rts = rts - rts[0]
    l_rate = fit_rate(lts, common); r_rate = fit_rate(rts, common)
    # sample the linked rate on the record grid (nearest)
    lx0, ly0, ldx, ldy, lL, lW = geo_grid(la); rx0, ry0, rdx, rdy, rL, rW = geo_grid(ra)
    yy, xx = np.mgrid[0:rL, 0:rW]
    lat = ry0 + yy * rdy; lon = rx0 + xx * rdx
    li = np.rint((lat - ly0) / ldy).astype(int); lj = np.rint((lon - lx0) / ldx).astype(int)
    inside = (li >= 0) & (li < lL) & (lj >= 0) & (lj < lW)
    l_on_r = np.full((rL, rW), np.nan, np.float32); l_on_r[inside] = l_rate[li[inside], lj[inside]]
    # reference both to the record's reference pixel
    ri, rj = int(round((float(ra["REF_LAT"]) - ry0) / rdy)), int(round((float(ra["REF_LON"]) - rx0) / rdx))
    l_on_r -= np.nanmedian(l_on_r[ri - 2:ri + 3, rj - 2:rj + 3]); r_rate -= np.nanmedian(r_rate[ri - 2:ri + 3, rj - 2:rj + 3])
    with h5py.File(R.mintpy_dir / "geo" / "geo_temporalCoherence.h5") as f:
        rtc = f["temporalCoherence"][:]
    ltc_file = L.mintpy_dir / "geo" / "geo_temporalCoherence.h5"
    with h5py.File(ltc_file) as f:
        ltc_own = f["temporalCoherence"][:]
    ltc = np.full((rL, rW), np.nan, np.float32); ltc[inside] = ltc_own[li[inside], lj[inside]]
    good = np.isfinite(l_on_r) & np.isfinite(r_rate)
    out = {"common_dates": common, "n_common": len(common), "linked_series": lf.name, "record_series": rf.name,
           "note": "rates fitted over the common dates only; both re-referenced to the record's reference pixel; "
                   "coherence classes use the RECORD's 10-year temporal coherence so the same ground is compared"}
    def stats(m, name):
        d = (l_on_r - r_rate)[m]
        return {"cells": int(m.sum()), "linked_median_mm_yr": round(float(np.nanmedian(l_on_r[m])), 2),
                "record_median_mm_yr": round(float(np.nanmedian(r_rate[m])), 2),
                "difference_median_mm_yr": round(float(np.nanmedian(d)), 2),
                "difference_robust_scatter_mm_yr": round(float(1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d)))), 2),
                "correlation": round(float(np.corrcoef(l_on_r[m], r_rate[m])[0, 1]), 3) if m.sum() > 10 else None}
    out["all_cells"] = stats(good, "all")
    out["record_high_coherence_ge_0_7"] = stats(good & (rtc >= 0.7), "hi")
    out["by_record_coherence_class"] = []
    for lo, hi in ((0.0, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.01)):
        m = good & (rtc >= lo) & (rtc < hi)
        if m.sum() > 50:
            out["by_record_coherence_class"].append({"record_tc": f"{lo:.2f}-{min(hi, 1):.2f}", **stats(m, "cls")})
    cls = out["by_record_coherence_class"]
    if len(cls) >= 2:
        out["low_minus_high_record_mm_yr"] = round(cls[0]["record_median_mm_yr"] - cls[-1]["record_median_mm_yr"], 2)
        out["low_minus_high_linked_mm_yr"] = round(cls[0]["linked_median_mm_yr"] - cls[-1]["linked_median_mm_yr"], 2)
    out["linked_coverage_percent_ge_0_4"] = round(100 * float(np.nanmean(ltc[np.isfinite(ltc)] >= 0.4)), 1)
    out["record_coverage_percent_ge_0_4"] = round(100 * float(np.nanmean(rtc >= 0.4)), 1)
    sites = {}
    for name, S, Wl, N, E, _gj in parse_focus_sites(L):
        m = good & (lat >= S) & (lat <= N) & (lon >= Wl) & (lon <= E)
        sites[name] = stats(m, name)
    out["focus_sites"] = sites
    dest = L.project_dir / "deliverables" / "experiments"; dest.mkdir(parents=True, exist_ok=True)
    (dest / "linked_vs_record.json").write_text(json.dumps(out, indent=2))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    ext = (rx0, rx0 + rW * rdx, ry0 + rL * rdy, ry0)
    for ax, arr, t in ((axs[0], r_rate, f"classic record, {len(common)} dates"), (axs[1], l_on_r, "phase-linked, same dates")):
        im = ax.imshow(arr, extent=ext, cmap="RdBu_r", vmin=-40, vmax=40); ax.set_title(t + " (mm/yr vs record reference)")
        plt.colorbar(im, ax=ax, fraction=0.04)
    im = axs[2].imshow(l_on_r - r_rate, extent=ext, cmap="PuOr", vmin=-30, vmax=30); axs[2].set_title("linked minus classic")
    plt.colorbar(im, ax=axs[2], fraction=0.04)
    fig.tight_layout(); fig.savefig(dest / "linked_vs_record.png", dpi=110); plt.close(fig)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
