"""
T1 (8 Oct 2026): the 17 Tahmoor West GNSS units, digitised from MSEC's
figures in Tahmoor West Report 8 (MSEC1263 Rev 21, Aug 2023; Fig. G07-R21 to
Fig. GSR17S-R21, pages 89-105). Each figure plots the change in easting,
northing and height (+ve = subsidence) since the start of LW W1 (or the
unit's installation): every daily value and the 7-day running average.

Two independent passes (protocol A, truth.tahmoor_west_units.digitising):
  A  the PDF's own vector drawing: every plotted vertex, axes calibrated from
     the vector tick marks and their numeric labels (gnss_plot_vector.py);
  B  a 600 dpi render: axes calibrated from the gridlines found in the image,
     the 7-day curve read column by column from its colour
     (gnss_plot_raster.py).
Pass A is the value written; A - B is the digitising repeatability.

Digitising sigma per point (protocol A amendment 2026-10-08/p23-1, 8 Oct
2026, before any result): hypot(two-pass RMS, half the plotted line width in
data units, axis-calibration residual). The line width is the stroke width
of the 7-day-average paths read (pt) times the value axis's mm per pt; the
calibration residual is the value axis's RMS at its labelled ticks combined
with the date axis's RMS (days) times the curve's slope at the epoch (mm per
day). The rule as first fixed (two-pass RMS, at least half the gridline
spacing) is kept beside it in the *_as_first_fixed columns.

Checks written to checks/gnss_checks.json:
  - MSEC's 7-day running average equals the centred 7-day mean of the
    plotted daily values (so the daily vertices and the date calibration are
    right, and the average is centred, not trailing);
  - Report 7 (MSEC1263 Rev 20, Nov 2022) plots the same units with other
    axis scales: pass A on Report 7 against Report 8 at the shared epochs;
  - numbers printed in the reports' text (Report 8: Site 24 about 660 mm,
    +16 mm in Jan-Jun 2023; Sites 7, 8, 9 +5, +2, +4 mm in Jan-Jun 2023);
  - unit positions on the Picton-Mittagong Loop Line against the chainages
    in the figure titles.

Positions: the units are drawn as circles on the 1:10000 monitoring layout
MSEC1263-01 (Report 8 page 88), which is tied by an affine fit to the LW
W1-W3 corners read on the MGA-gridded MSEC drawings (panels.geojson). Each
figure also has an inset map with LW W1-W4 and the unit as a red diamond;
the inset, tied the same way to the W1-W4 corners, picks the unit's circle
(the next circle is always more than 50 m away) and checks it (they agree
within 2.4 m).

Outputs (review_2026-10-08/truth/):
  tahmoor_west_gnss_units.csv     ground_truth.load_points_csv format (wide):
                                  one row per unit and epoch (each unit's
                                  first 7-day value, then the 1st and 15th of
                                  every month while the unit has data)
  tahmoor_west_gnss_daily.csv     everything plotted: every day's daily value
                                  and 7-day average, per component (pass A)
  tahmoor_west_gnss_calibration.json   per figure: axis calibrations of both
                                  passes, styles, position fit, sigmas
  checks/                         overlays of the digitised epochs on the
                                  rendered figures, gnss_checks.json
"""
from __future__ import annotations

import json
import os
import tempfile
import math
from datetime import date
from pathlib import Path

import numpy as np

import drawgeo as dg
import gnss_plot_raster as gr
import gnss_plot_vector as gv
import pdfvec
from sources import SOURCES

HERE = Path(__file__).resolve().parent
TRUTH = HERE.parent
SRC = TRUTH / "sources"
CHECKS = TRUTH / "checks"
WORK = Path(os.environ.get("T1_WORKDIR", Path(tempfile.gettempdir()) / "t1_digitise"))   # SVG/PNG renders
WORK.mkdir(parents=True, exist_ok=True)
R8 = SRC / "tahmoor_west_report_8.pdf"
R7 = SRC / "tahmoor_west_report_7.pdf"
R8_NAME = "Tahmoor West Report 8 (1 Jan - 30 Jun 2023), Appendix: MSEC1263 Rev 21 (Aug 2023)"
R7_NAME = "Tahmoor West Report 7 (1 Jan - 31 Dec 2022), Appendix: MSEC1263 Rev 20 (Nov 2022)"
COMP = {"height": "u", "easting": "e", "northing": "n"}
SIGN = {"height": -1.0, "easting": 1.0, "northing": 1.0}      # height is plotted +ve = subsidence
MIN_DAYS_IN_WEEK = 4                                          # an epoch needs data on >= 4 of the 7 days

# Report 8 page -> figure; Report 7 page of the same figure
UNITS = [
    ("G07", 89, 179), ("G08", 90, 180), ("G09", 91, 181), ("G10", 92, 182), ("G11", 93, 183),
    ("G12A", 94, 184), ("G13", 95, 185), ("G15", 96, 186), ("G16", 97, 187), ("G17", 98, 188),
    ("G18", 99, 189), ("G19", 100, 190), ("G20", 101, 191), ("G21", 102, 192), ("G24", 103, 193),
    ("GSR17N", 104, 194), ("GSR17S", 105, 195),
]
# notes printed on the figures that change how a stretch of a series can be used
FIGURE_NOTES = {
    "G13": ("2022-03-29", "after the gap of 10 Jan - 29 Mar 2022: Fig. G13 says the unit was disturbed in early "
                          "January 2022 (printed '2002'), relocated on 28 March 2022 and 'results normalised', so values "
                          "after the gap are joined to those before it and any movement during the gap is not measured"),
}
PMLL_CHAINAGE_KM = {"G07": 87.850, "G08": 88.110, "G09": 88.550, "G10": 89.000, "G19": 89.440, "G11": 89.629}


# ---------------------------------------------------------------- helpers
def daily_reconstructed(fig, panel):
    """{day: value} for every integer day inside a drawn daily subpath. The
    plotting program drops a vertex that lies on the straight line between
    its neighbours (equal or evenly changing mm values), so days inside a
    subpath are interpolated along it (exact for dropped collinear points);
    a break between subpaths is missing data and is not filled."""
    out = {}
    for days, vals in fig.curves.get((panel, "daily"), []):
        d = np.round(days).astype(int)
        if d.size == 1:
            out[int(d[0])] = float(vals[0])
            continue
        for k in range(d[0], d[-1] + 1):
            out[k] = float(np.interp(k, days, vals))
    return out


def daily_observed(fig, panel):
    """{day: value} at the plotted daily vertices only (no interpolation)."""
    out = {}
    for days, vals in fig.curves.get((panel, "daily"), []):
        for d, v in zip(np.round(days).astype(int), vals):
            out[int(d)] = float(v)
    return out


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x))) if x.size else float("nan")


def epochs_for(first_day: int, last_day: int):
    """The unit's first 7-day value, then the 1st and 15th of each month."""
    d0 = gv.from_day(first_day)
    out = [first_day]
    y, m = d0.year, d0.month
    while True:
        for dd in (1, 15):
            t = gv.day_number(date(y, m, dd))
            if t > last_day:
                return out
            if t > first_day:
                out.append(t)
        m += 1
        if m == 13:
            y, m = y + 1, 1


def inset_position(paths, panel_corners_mga):
    """Affine fit from the inset's four panel outlines (W1, W2, W3: #b3b3b3; W4
    as mined: #e6e6e6) to the MGA corners; returns (E, N), the fit and the
    diamond's page centre."""
    box = (102.0, 76.0, 217.0, 213.5)
    quads = []
    for p in paths:
        col = pdfvec.colour_name(p.stroke)
        if col not in ("#b3b3b3", "#e6e6e6"):
            continue
        for s in p.subpaths:
            s = np.asarray(s, float)
            if len(s) == 5 and np.hypot(*(s[0] - s[-1])) < 0.05:
                s = s[:-1]
            if len(s) != 4 or not (box[0] <= s[:, 0].min() and s[:, 0].max() <= box[2]
                                   and box[1] <= s[:, 1].min() and s[:, 1].max() <= box[3]):
                continue
            quads.append(s)
    if len(quads) != 4:
        raise ValueError(f"inset: {len(quads)} panel outlines found, 4 expected")
    quads.sort(key=lambda q: q[:, 0].mean())                         # west to east: W1, W2, W3, W4
    page, mga = [], []
    for q, ref in zip(quads, panel_corners_mga):
        c = q.mean(axis=0)
        page.append(q[np.argsort(np.arctan2(q[:, 0] - c[0], -(q[:, 1] - c[1])))])
        cm = ref.mean(axis=0)
        mga.append(ref[np.argsort(np.arctan2(ref[:, 0] - cm[0], ref[:, 1] - cm[1]))])
    labels = [f"{w} corner {k}" for w in ("W1", "W2", "W3", "W4") for k in range(4)]
    A = dg.fit_affine(np.vstack(page), np.vstack(mga), labels)
    dia = None
    for p in paths:
        if pdfvec.colour_name(p.fill) == "#ff0000":
            for s in p.subpaths:
                s = np.asarray(s, float)
                # the diamond (4.3 pt across) may straddle the inset's frame: test its centre only
                cx, cy = s[:, 0].mean(), s[:, 1].mean()
                if len(s) >= 4 and float(np.ptp(s, axis=0).max()) < 8 and box[0] - 3 <= cx <= box[2] + 3 \
                        and box[1] - 3 <= cy <= box[3] + 3:
                    u = np.unique(np.round(s, 3), axis=0)
                    dia = u.mean(axis=0)
    if dia is None:
        raise ValueError("inset: no red diamond")
    en = A(dia)[0]
    return en, A, dia


def layout_circles():
    """The GNSS units drawn on the monitoring layout MSEC1263-01 (Report 8
    page 88, 1:10000): dark red circles (#b80000 outline of ~70 vertices, about
    3 pt across). Their page centres go through the drawing's affine fit
    (georef_drawings.json, tied to LW W1-W3) to MGA."""
    paths = pdfvec.read_paths(pdfvec.page_svg(R8, 88, WORK))
    rec = json.loads((TRUTH / "georef_drawings.json").read_text())["MSEC1263-01"]["georef"]
    M = np.array(rec["matrix"])
    cen = []
    for p in paths:
        if pdfvec.colour_name(p.stroke) != "#b80000":
            continue
        for s in p.subpaths:
            s = np.asarray(s, float)
            if len(s) >= 30:
                e = np.ptp(s, axis=0)
                if 2 < e.max() < 6 and abs(e[0] - e[1]) < 0.4:
                    cen.append(s.mean(axis=0))
    cen = np.array(cen)
    return np.c_[cen, np.ones(len(cen))] @ M.T, cen, rec["rms_m"]


def panel_corners():
    fc = json.loads((TRUTH / "panels.geojson").read_text())
    by = {f["properties"]["id"]: np.array(f["properties"]["mga94_corners"]) for f in fc["features"]}
    return [by["TW_W1"], by["TW_W2"], by["TW_W3"], by["TW_W4"]]


# ---------------------------------------------------------------- one figure
def digitise(code, page, page7, corners):
    fig = gv.extract(R8, page, WORK)
    gr.WORKDIR = WORK
    rb = gr.extract(fig, WORK)
    paths = pdfvec.read_paths(pdfvec.page_svg(R8, page, WORK))
    en, A, dia = inset_position(paths, corners)
    title = fig.title.split("Monitoring", 1)[-1].strip()
    cal = {"figure": fig.figure_id, "page": page, "title": title, "styles": {k: list(v[:2]) for k, v in fig.styles.items()},
           "x_axis": {"day_number_epoch": "2000-01-01", "a": fig.x_axis[0], "b_days_per_pt": fig.x_axis[1],
                      "rms_days": round(fig.x_rms_days, 3), "ticks": len(fig.x_ticks),
                      "first_last_tick": [fig.x_ticks[0][0], fig.x_ticks[-1][0]]},
           "panels": {}, "position": {"inset_affine_rms_m": round(A.rms_m, 2), "diamond_page_pt": [round(float(x), 2) for x in dia],
                                       "mga94": [round(float(x), 1) for x in en]},
           "warnings": list(fig.warnings)}
    comps = {}
    for name, pn in fig.panels.items():
        ad, av = gv.merged(fig, name, "avg7")
        dobs = daily_observed(fig, name)
        drec = daily_reconstructed(fig, name)
        ad_i = np.round(ad).astype(int)
        # check 1: MSEC's 7-day average is the centred mean of the daily values
        res = []
        for d, v in zip(ad_i, av):
            w = [drec[k] for k in range(d - 3, d + 4) if k in drec]
            if len(w) == 7:
                res.append(v - np.mean(w))
        # the unit's own daily scatter about its 7-day average
        scat = [dobs[d] - v for d, v in zip(ad_i, av) if d in dobs]
        r = rb[name]
        cal["panels"][name] = {
            "pass_A": {"value_mm = a + b * y_pt": [pn.a, pn.b], "labelled_ticks": len(pn.ticks),
                       "rms_at_ticks_mm": round(pn.rms_mm, 3), "gridline_mm": pn.gridline_mm,
                       "minor_tick_mm": round(pn.minor_mm, 2) if pn.minor_mm else None,
                       "y_range_pt": [pn.y_top, pn.y_bottom]},
            "pass_B": {"gridlines_used": r["y_n"], "rms_at_gridlines_mm": round(r["y_rms_mm"], 3),
                       "date_gridlines_used": r["x_n"], "rms_days": round(r["x_rms_days"], 3),
                       "columns_read": r["n_columns"], "dpi": gr.DPI},
            "avg7_vs_centred_daily_mean": {"n": len(res), "rms_mm": round(float(np.sqrt(np.mean(np.square(res)))), 3)
                                           if res else None},
            "daily_scatter_about_avg7_mm": round(robust_sigma(scat), 2),
        }
        widths = fig.curve_widths.get((name, "avg7"), [])
        comps[name] = dict(ad=ad, av=av, drec=drec, dobs=dobs, rb=r, panel=pn,
                           gnss_sigma=robust_sigma(scat),
                           line_width_pt=float(np.median(widths)) if widths else float(fig.styles["avg7"][1]),
                           line_widths_read=len(widths))
    # Report 7 (independent publication, other axis scales)
    fig7 = gv.extract(R7, page7, WORK)
    cal["report7"] = {"figure": fig7.figure_id, "page": page7, "x_rms_days": round(fig7.x_rms_days, 3)}
    comps7 = {name: gv.merged(fig7, name, "avg7") for name in fig7.panels}
    return fig, cal, comps, comps7, en


def series_at(ad, av, day):
    """MSEC's 7-day average at a day: linear between the plotted vertices
    (the plotting program dropped collinear vertices)."""
    if day < ad[0] - 0.5 or day > ad[-1] + 0.5:
        return float("nan")
    return float(np.interp(day, ad, av))


def raster_at(r, day):
    m = np.abs(r["days"] - day) <= 0.5
    return float(np.median(r["values"][m])) if m.any() else float("nan")


def main():
    corners = panel_corners()
    CHECKS.mkdir(exist_ok=True)
    rows, daily_rows, calib, checks = [], [], {}, {"units": {}}
    positions = {}
    circles, circles_pt, layout_rms = layout_circles()
    for code, page, page7 in UNITS:
        fig, cal, comps, comps7, en_inset = digitise(code, page, page7, corners)
        uid = f"TW_{code}"
        # position: the unit's circle on the 1:10000 monitoring layout (nearest to the inset's diamond;
        # the next circle is always more than 50 m away, so the match is unambiguous)
        d = np.hypot(*(circles - en_inset).T)
        j = int(np.argmin(d))
        en = circles[j]
        lon, lat = dg.mga_to_lonlat(en, 28356)[0]
        positions[uid] = en
        pos_sigma = 5.0
        cal["position"].update({
            "source": "circle on drawing MSEC1263-01 (Report 8 page 88), affine to LW W1-W3 corners "
                      f"(rms {layout_rms:.2f} m); checked against the figure's own inset map",
            "layout_circle_page_pt": [round(float(x), 2) for x in circles_pt[j]],
            "mga94": [round(float(x), 1) for x in en],
            "inset_mga94": [round(float(x), 1) for x in en_inset],
            "inset_minus_layout_m": round(float(d[j]), 1),
            "next_circle_m": round(float(np.sort(d)[1]), 1),
            "position_sigma_note": "5 m (1 sigma): the two drawings agree within 2.4 m; panel georeference 1-2 m"})
        # epochs: the first 7-day value, then the 1st and 15th, while there are data
        hd = comps["height"]
        first, last = int(round(hd["ad"][0])), int(round(hd["ad"][-1]))
        ep = epochs_for(first, last)
        per = {}
        for name, c in comps.items():
            vals, why = {}, {}
            for d in ep:
                n_days = sum(1 for k in range(d - 3, d + 4) if k in c["drec"])
                if n_days < MIN_DAYS_IN_WEEK:
                    why[d] = f"only {n_days} of the 7 days around it have data"
                    continue
                a = series_at(c["ad"], c["av"], d)
                b = raster_at(c["rb"], d)
                own = [c["drec"][k] for k in range(d - 3, d + 4) if k in c["drec"]]
                d7 = series_at(*comps7[name], d) if name in comps7 else float("nan")
                vals[d] = (a, b, float(np.mean(own)), d7)
            ab = np.array([v[0] - v[1] for v in vals.values() if math.isfinite(v[0]) and math.isfinite(v[1])])
            rms = float(np.sqrt(np.mean(ab ** 2))) if ab.size else float("nan")
            half = c["panel"].gridline_mm / 2.0
            dig_v1 = max(rms, half)                       # protocol A as first fixed: at least the half-gridline
            # amendment 2026-10-08/p23-1: per point, hypot(repeatability, half line width, calibration residual)
            half_line = 0.5 * c["line_width_pt"] * abs(c["panel"].b)
            cal_pt, dig = {}, {}
            for d in vals:
                # the curve's slope at the epoch (mm/day): central over +-1 day, one-sided at an end of the curve
                lo, mid, hi = (series_at(c["ad"], c["av"], d + k) for k in (-1, 0, 1))
                if math.isfinite(lo) and math.isfinite(hi):
                    slope = abs(hi - lo) / 2.0
                elif math.isfinite(mid) and (math.isfinite(lo) or math.isfinite(hi)):
                    slope = abs((hi if math.isfinite(hi) else lo) - mid)
                else:
                    slope = 0.0
                cal_pt[d] = float(np.hypot(c["panel"].rms_mm, fig.x_rms_days * slope))
                dig[d] = float(np.sqrt((rms if math.isfinite(rms) else 0.0) ** 2 + half_line ** 2 + cal_pt[d] ** 2))
            gn = c["gnss_sigma"]
            sig = {d: float(np.hypot(dig[d], gn)) for d in vals}
            per[name] = dict(vals=vals, why=why, rms=rms, n_ab=int(ab.size), max_ab=float(np.max(np.abs(ab))) if ab.size else None,
                             half=half, dig_v1=dig_v1, sigma_v1=float(np.hypot(dig_v1, gn)), gnss=gn,
                             half_line=half_line, calib=cal_pt, dig=dig, sigma=sig)
            cal["panels"][name]["two_pass"] = {"epochs": int(ab.size), "rms_A_minus_B_mm": round(rms, 2),
                                               "max_abs_A_minus_B_mm": round(per[name]["max_ab"], 2) if ab.size else None}

            def span(x):
                v = list(x.values())
                return {"median": round(float(np.median(v)), 2), "max": round(float(np.max(v)), 2)} if v else None
            cal["panels"][name]["sigma"] = {
                "rule": "amendment 2026-10-08/p23-1: digitising sigma = hypot(two-pass RMS, half the plotted line "
                        "width in data units, axis-calibration residual) per point; sigma = hypot(digitising sigma, "
                        "the unit's daily scatter)",
                "digitising_rms_mm": round(rms, 2),
                "avg7_line_width_pt": round(c["line_width_pt"], 3), "avg7_paths_read": c["line_widths_read"],
                "half_line_width_mm": round(half_line, 2),
                "value_axis_rms_mm": round(c["panel"].rms_mm, 3), "date_axis_rms_days": round(fig.x_rms_days, 3),
                "calibration_mm": span(cal_pt), "digitising_sigma_mm": span(dig),
                "gnss_sigma_mm": round(gn, 2), "sigma_mm": span(sig),
                "as_first_fixed": {"rule": "two-pass RMS, at least half the plot's gridline spacing (protocol A as "
                                           "first fixed)",
                                   "half_gridline_mm": half, "digitising_sigma_mm": round(dig_v1, 2),
                                   "sigma_mm": round(per[name]["sigma_v1"], 2)}}
            # daily record (pass A)
            for k in sorted(c["dobs"]):
                daily_rows.append((uid, gv.from_day(k).isoformat(), COMP[name], SIGN[name] * c["dobs"][k], None))
            for d, v in zip(np.round(c["ad"]).astype(int), c["av"]):
                daily_rows.append((uid, gv.from_day(int(d)).isoformat(), COMP[name], None, SIGN[name] * v))
        # rows
        prov = (f"{SOURCES[R8.name]}; page {page}; {cal['figure']}; pass A vector: y = a + b*y_pt fitted to the labelled "
                f"ticks (rms <= {max(cal['panels'][n]['pass_A']['rms_at_ticks_mm'] for n in comps):.2f} mm), date = a + b*x_pt "
                f"fitted to {cal['x_axis']['ticks']} month ticks (rms {cal['x_axis']['rms_days']} d); pass B 600 dpi raster "
                f"gridlines; value = MSEC 7-day running average; position = the unit's circle on monitoring layout MSEC1263-01 "
                f"(Report 8 page 88), affine to LW W1-W3 corners, {cal['position']['inset_minus_layout_m']} m from the "
                f"figure's inset-map diamond; details in tahmoor_west_gnss_calibration.json")
        for d in ep:
            u = per["height"]["vals"].get(d)
            e = per["easting"]["vals"].get(d)
            n = per["northing"]["vals"].get(d)
            if u is None and e is None and n is None:
                continue
            f = lambda x, s: (round(s * x[0], 1) if x is not None and math.isfinite(x[0]) else None)

            def pt(name, key):
                v = per[name][key].get(d)
                return round(v, 2) if v is not None else None
            rows.append({
                "id": uid, "lat": round(float(lat), 6), "lon": round(float(lon), 6), "date": gv.from_day(d).isoformat(),
                "up_mm": f(u, -1.0), "east_mm": f(e, 1.0), "north_mm": f(n, 1.0),
                "sigma_mm": pt("height", "sigma"),          # protocol A's column; = sigma_up_mm
                "sigma_up_mm": pt("height", "sigma"), "sigma_east_mm": pt("easting", "sigma"),
                "sigma_north_mm": pt("northing", "sigma"),
                "digitising_sigma_mm": pt("height", "dig"),
                "digitising_sigma_east_mm": pt("easting", "dig"),
                "digitising_sigma_north_mm": pt("northing", "dig"),
                "digitising_rms_mm": round(per["height"]["rms"], 2),
                "digitising_rms_east_mm": round(per["easting"]["rms"], 2),
                "digitising_rms_north_mm": round(per["northing"]["rms"], 2),
                "half_line_width_mm": round(per["height"]["half_line"], 2),
                "half_line_width_east_mm": round(per["easting"]["half_line"], 2),
                "half_line_width_north_mm": round(per["northing"]["half_line"], 2),
                "calibration_mm": pt("height", "calib"),
                "calibration_east_mm": pt("easting", "calib"),
                "calibration_north_mm": pt("northing", "calib"),
                "gnss_sigma_mm": round(per["height"]["gnss"], 2),
                "gnss_sigma_east_mm": round(per["easting"]["gnss"], 2),
                "gnss_sigma_north_mm": round(per["northing"]["gnss"], 2),
                "half_gridline_mm": per["height"]["half"],
                "half_gridline_east_mm": per["easting"]["half"],
                "half_gridline_north_mm": per["northing"]["half"],
                "sigma_up_as_first_fixed_mm": round(per["height"]["sigma_v1"], 1),
                "sigma_east_as_first_fixed_mm": round(per["easting"]["sigma_v1"], 1),
                "sigma_north_as_first_fixed_mm": round(per["northing"]["sigma_v1"], 1),
                "pass_b_up_mm": round(-u[1], 1) if u is not None and math.isfinite(u[1]) else None,
                "kind": "gnss", "role": "validation",   # ground_truth.KINDS; digitised, see provenance
                "position_sigma_m": round(pos_sigma, 1),
                "site": cal["title"],
                "note": (FIGURE_NOTES[code][1] if code in FIGURE_NOTES
                         and gv.from_day(d).isoformat() >= FIGURE_NOTES[code][0] else None),
                "source_report": R8_NAME, "page": page, "figure": cal["figure"], "provenance": prov,
            })
        if code in FIGURE_NOTES:
            cal["figure_note"] = {"from": FIGURE_NOTES[code][0], "note": FIGURE_NOTES[code][1]}
        cal["epochs"] = {"first": gv.from_day(first).isoformat(), "last": gv.from_day(last).isoformat(),
                         "listed": len(ep),
                         "dropped": {COMP[nm]: {gv.from_day(k).isoformat(): w for k, w in per[nm]["why"].items()}
                                     for nm in per if per[nm]["why"]}}
        cal["position"].update({"lat": round(float(lat), 6), "lon": round(float(lon), 6), "position_sigma_m": round(pos_sigma, 1)})
        # checks: Report 7 and own centred mean, per component
        ck = {}
        for name in per:
            v = per[name]["vals"]
            d78 = [x[0] - x[3] for x in v.values() if math.isfinite(x[0]) and math.isfinite(x[3])]
            dmean = [x[0] - x[2] for x in v.values() if math.isfinite(x[0])]
            ck[COMP[name]] = {"report8_minus_report7_mm": {"n": len(d78), "median": round(float(np.median(d78)), 2) if d78 else None,
                                                           "rms": round(float(np.sqrt(np.mean(np.square(d78)))), 2) if d78 else None,
                                                           "max_abs": round(float(np.max(np.abs(d78))), 2) if d78 else None},
                              "avg7_minus_own_centred_mean_mm": {"rms": round(float(np.sqrt(np.mean(np.square(dmean)))), 2),
                                                                 "max_abs": round(float(np.max(np.abs(dmean))), 2)}}
        checks["units"][uid] = ck
        calib[uid] = cal
        overlay(fig, per, page, uid)
        print(f"{uid:10s} {cal['title'][:55]:55s} lat {lat:.5f} lon {lon:.5f} epochs {len(ep):3d} "
              f"U rms(A-B) {per['height']['rms']:.2f} sigU {np.median(list(per['height']['sigma'].values())):.1f} "
              f"(as first fixed {per['height']['sigma_v1']:.1f}) gnssU {per['height']['gnss']:.1f} "
              f"R8-R7 U {ck['u']['report8_minus_report7_mm']}")
    write_outputs(rows, daily_rows, calib, checks, positions)


def write_outputs(rows, daily_rows, calib, checks, positions):
    import csv
    cols = list(rows[0].keys())
    hdr = ("# Tahmoor West GNSS units, digitised 8 Oct 2026 (PLAN T1) from Tahmoor West Report 8 (MSEC1263 Rev 21), "
           "Fig. G07-R21 to GSR17S-R21. Values: MSEC's 7-day running average (pass A, vector), mm since the start of "
           "LW W1 or the unit's installation; up = -(plotted height change); east/north in the mine's survey frame. "
           "sigma_*_mm = sqrt(digitising_sigma^2 + gnss_sigma^2) per point, digitising_sigma = hypot(digitising_rms "
           "(the measured two-pass RMS), half_line_width (half the plotted line width in mm), calibration (the "
           "axis-calibration residual at the point)) as protocol A amendment 2026-10-08/p23-1 states; "
           "sigma_*_as_first_fixed_mm use max(two-pass RMS, half_gridline) as protocol A was first fixed. "
           "See README.md.\n")
    with open(TRUTH / "tahmoor_west_gnss_units.csv", "w", newline="") as f:
        f.write(hdr)
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    # daily: merge daily and avg7 per (id, date, component)
    merged = {}
    for uid, d, comp, dv, av in daily_rows:
        k = (uid, d, comp)
        m = merged.setdefault(k, [None, None])
        if dv is not None:
            m[0] = round(dv, 2)
        if av is not None:
            m[1] = round(av, 2)
    with open(TRUTH / "tahmoor_west_gnss_daily.csv", "w", newline="") as f:
        f.write("# Every value plotted in Tahmoor West Report 8 Fig. G07-R21 .. GSR17S-R21, read from the PDF's vector "
                "paths (pass A). daily_mm: the plotted daily value (only days with a plotted vertex); avg7_mm: MSEC's "
                "7-day running average where it has a vertex. Component u = up (= -plotted height change), e, n. "
                "Dates rounded to the day (calibration rms < 0.1 d).\n")
        w = csv.writer(f)
        w.writerow(["id", "date", "component", "daily_mm", "avg7_mm"])
        for (uid, d, comp), (dv, av) in sorted(merged.items()):
            w.writerow([uid, d, comp, "" if dv is None else dv, "" if av is None else av])
    (TRUTH / "tahmoor_west_gnss_calibration.json").write_text(json.dumps(calib, indent=1) + "\n")
    checks["positions"] = pmll_check(positions)
    checks["printed_numbers"] = printed_checks()
    (CHECKS / "gnss_checks.json").write_text(json.dumps(checks, indent=1) + "\n")


def pmll_check(positions):
    """Units on the Picton-Mittagong Loop Line: their spacing along the line
    against the differences of the chainages in the figure titles."""
    ids = [f"TW_{c}" for c in PMLL_CHAINAGE_KM]
    out = []
    ordered_ids = sorted(ids, key=lambda i: PMLL_CHAINAGE_KM[i[3:]])
    for a, b in zip(ordered_ids, ordered_ids[1:]):
        ch = (PMLL_CHAINAGE_KM[b[3:]] - PMLL_CHAINAGE_KM[a[3:]]) * 1000
        st = float(np.hypot(*(positions[b] - positions[a])))
        out.append({"from": a, "to": b, "chainage_diff_m": round(ch, 0), "straight_line_m": round(st, 1),
                    "note": "the units stand beside the track, not on it, so the straight line between two can "
                            "differ from the chainage difference by tens of metres; on MSEC1263-01 Sites 7, 8, 9, 10 "
                            "and 19 sit beside the pegs of their title chainage, Site 11 beside peg ~89.69 km "
                            "(title 89.629 km): checked by eye on a 500 dpi render"})
    return out


def printed_checks():
    """Report 8 text against the digitised series."""
    import csv
    rows = list(csv.DictReader((l for l in open(TRUTH / "tahmoor_west_gnss_daily.csv") if not l.startswith("#"))))

    def avg7(uid, d, comp="u"):
        """MSEC's 7-day average at day d: linear between its plotted vertices
        (collinear vertices were dropped by the plotting program), only when a
        vertex lies within 7 days; (days to the nearest vertex, value, date)."""
        pts = sorted((date.fromisoformat(r["date"]), float(r["avg7_mm"])) for r in rows
                     if r["id"] == uid and r["component"] == comp and r["avg7_mm"])
        if not pts:
            return None
        t = np.array([(x[0] - d).days for x in pts], float)
        k = int(np.argmin(np.abs(t)))
        if abs(t[k]) > 7:
            return None
        v = float(np.interp(0.0, t, [x[1] for x in pts]))
        return (int(abs(t[k])), v, d.isoformat())

    out = []
    d1, d2 = date(2023, 1, 1), date(2023, 6, 30)
    for uid, printed in (("TW_G24", 16), ("TW_G07", 5), ("TW_G08", 2), ("TW_G09", 4)):
        a, b = avg7(uid, d1), avg7(uid, d2)
        if a and b:
            sub = -(b[1] - a[1])
            out.append({"unit": uid, "report8_text": f"additional {printed} mm of vertical subsidence between 1 January "
                                                     f"and 30 June 2023",
                        "digitised_mm": round(sub, 1), "dates_used": [a[2], b[2]], "difference_mm": round(sub - printed, 1)})
    a = avg7("TW_G24", date(2023, 6, 30))
    out.append({"unit": "TW_G24", "report8_text": "Site 24 'has recorded approximately 660 mm subsidence'",
                "digitised_mm": round(-a[1], 1) if a else None, "date_used": a[2] if a else None,
                "difference_mm": round(-a[1] - 660, 1) if a else None})
    a = avg7("TW_G24", date(2022, 11, 13))
    out.append({"unit": "TW_G24", "report8_text": "Report 7 (MSEC1263 Rev 20, plots to 15 Nov 2022): Site 24 'has "
                                                  "recorded approximately 640 mm subsidence'",
                "digitised_mm": round(-a[1], 1) if a else None, "date_used": a[2] if a else None,
                "difference_mm": round(-a[1] - 640, 1) if a else None})
    return out


def overlay(fig, per, page, uid):
    """Draw the listed epochs (pass A values) back onto the rendered figure."""
    from PIL import Image, ImageDraw
    png = WORK / f"{R8.stem}_p{page}_{gr.DPI}.png"
    im = Image.open(png).convert("RGB")
    s = gr.DPI / 72.0
    im = im.resize((im.size[0] // 3, im.size[1] // 3))
    s /= 3
    dr = ImageDraw.Draw(im)
    xa, xb = fig.x_axis
    for name, pr in per.items():
        pn = fig.panels[name]
        for d, v in pr["vals"].items():
            if not math.isfinite(v[0]):
                continue
            x = (d - xa) / xb
            y = (v[0] - pn.a) / pn.b
            dr.ellipse([x * s - 3, y * s - 3, x * s + 3, y * s + 3], outline=(255, 0, 0), width=1)
    im.save(CHECKS / f"overlay_{uid}.png")


if __name__ == "__main__":
    main()
