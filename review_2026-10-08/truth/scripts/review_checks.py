"""
Independent review of T1 (8 Oct 2026): two checks that use nothing the
digitising itself used. Writes checks/review_checks.json.

1. Appin outlines against a drawing by another hand and in a stated datum.
   South32's APNSVY00237 (LW709-711 and LW905 Extraction Plan variation,
   "Approved Plan", 8 Feb 2023, 1:6000, "MGA2020 COORDINATES") is a surveyor's
   CAD drawing, not an MSEC one. Its MGA2020 grid is drawn at an angle to the
   page. The grid is fitted to its 12 long grid lines (#989898). The 1000 m
   spacing is the drawing's own 1:6000 scale (472.45 pt). The line values come
   from the labels beside them ("E 28 600" = 286 000 E, "N62 180 00" =
   6 218 000 N). The fit is checked against the drawing's "APPLICATION AREA
   CORNERS (MGA2020)" table: every corner (A-X) must lie on the red
   application-area outline. The table values were read from a 110 dpi render;
   they are glyphs, not text. The LW709 and LW905 voids (light-grey fills,
   "EXTRACTION APPROVED AND CURRENTLY MINING") then go through the fit. Their
   vertices are compared with the corners in panels.geojson, which came from
   MSEC1394-01 and MSEC1334-01. Two readings of the MSEC datum are tried:
   MGA94 (as T1 assumed) and MGA2020.

2. The date axis of the 17 GNSS figures against the mining dates. T1
   calibrated each date axis from its month ticks. Every figure also draws
   the mining period as a pink bar (#ffcccc) under the plots. The bar's ends,
   put through that calibration, must fall on the published dates in
   panels.geojson: W1 start 15 Nov 2019, W3 start 13 Sep 2021, W4 start
   16 May 2022, W4 end 13 Sep 2022. G12A's bar starts where its axis starts
   (1 Jun 2021), so only its end is checked. This tests the absolute date
   calibration. The centred 7-day mean test in gnss_build.py cannot: the
   daily values and the 7-day average share one axis.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
from pyproj import Transformer

import gnss_plot_vector as gv
import pdfvec

HERE = Path(__file__).resolve().parent
TRUTH = HERE.parent
SRC = TRUTH / "sources"
WORK = Path(os.environ.get("T1_WORKDIR", Path(tempfile.gettempdir()) / "t1_digitise"))
WORK.mkdir(parents=True, exist_ok=True)
S32 = SRC / "appin_APNSVY00237_LW709-711_905_EP_variation_approved_plan_MGA2020.pdf"
R8 = SRC / "tahmoor_west_report_8.pdf"

# APPLICATION AREA CORNERS (MGA2020), read from a 110 dpi render of the table on APNSVY00237
CORNERS_MGA2020 = {
    "A": (286097.3, 6216121.4), "B": (285555.8, 6217242.9), "C": (285713.4, 6217320.2), "D": (285805.9, 6217128.7),
    "E": (290019.9, 6219194.7), "F": (289994.8, 6219246.0), "G": (290083.5, 6219287.5), "H": (290165.4, 6219392.0),
    "I": (290465.3, 6219345.2), "J": (291111.9, 6218838.3), "K": (290988.0, 6218680.3), "L": (290873.0, 6218624.0),
    "M": (290991.4, 6218382.4), "N": (288624.8, 6217222.9), "O": (288464.7, 6217549.7), "P": (288336.1, 6217486.7),
    "Q": (288174.0, 6217817.5), "R": (287827.4, 6217647.6), "S": (288047.3, 6217199.1), "T": (288270.3, 6216927.8),
    "U": (288286.7, 6216508.9), "V": (286527.4, 6216439.8), "W": (286384.8, 6216370.9), "X": (286428.1, 6216281.1)}
# grid lines named by the page point of one end, and the value of the label beside it
E_ANCHORS = {(750.7, 57.1): 286000.0, (2855.5, 57.1): 290000.0}
N_ANCHORS = {(57.0, 63.7): 6218000.0, (57.0, 589.9): 6217000.0, (1116.7, 57.1): 6219000.0}
SCALE_M_PER_PT = 6000 * 0.0254 / 72          # 1:6000


def _r(x, n=2):
    return round(float(x), n)


def south32_check() -> dict:
    paths = pdfvec.read_paths(pdfvec.page_svg(S32, 1, WORK))
    e_lines, n_lines = [], []
    for p in paths:
        if pdfvec.colour_name(p.stroke) != "#989898":
            continue
        for s in p.subpaths:
            s = np.asarray(s, float)
            if len(s) == 2 and np.hypot(*(s[1] - s[0])) > 800:
                d = np.abs(s[1] - s[0])
                (e_lines if d[1] > d[0] else n_lines).append(s)

    def order(lines):
        d = lines[0][1] - lines[0][0]
        nrm = np.array([d[1], -d[0]]) / np.hypot(*d)
        lines.sort(key=lambda s: float(np.mean(s @ nrm)))
        return np.array([float(np.mean(s @ nrm)) for s in lines])

    pe, pn = order(e_lines), order(n_lines)
    spacing = float(np.median(np.abs(np.r_[np.diff(pe), np.diff(pn)])))

    def nearest(lines, pt):
        return int(np.argmin([min(np.hypot(*(s[0] - pt)), np.hypot(*(s[1] - pt))) for s in lines]))

    def values(lines, anchors):
        (pa, va), (pb, vb) = list(anchors.items())[:2]
        ka, kb = nearest(lines, np.array(pa)), nearest(lines, np.array(pb))
        step = (vb - va) / (kb - ka)
        vals = {k: va + (k - ka) * step for k in range(len(lines))}
        for pt, v in anchors.items():                        # every labelled line must agree
            assert vals[nearest(lines, np.array(pt))] == v, (pt, v)
        return vals, step

    ev, estep = values(e_lines, E_ANCHORS)
    nv, nstep = values(n_lines, N_ANCHORS)

    def fit(lines, vals):
        A, b = [], []
        for k, s in enumerate(lines):
            for x, y in s:
                A.append([x, y, 1.0])
                b.append(vals[k])
        c, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
        return c, float(np.sqrt(np.mean((np.array(A) @ c - b) ** 2)))

    ce, rms_e = fit(e_lines, ev)
    cn, rms_n = fit(n_lines, nv)

    def to_mga2020(xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        return np.c_[xy @ ce[:2] + ce[2], xy @ cn[:2] + cn[2]]

    # the corner table against the red application-area outline (dashed, so its vertices sit near, not on, the corners)
    red = np.unique(np.round(np.array([q for p in paths if pdfvec.colour_name(p.stroke) == "#ff0000"
                                       for s in p.subpaths for q in np.asarray(s).tolist()]), 2), axis=0)
    redm = to_mga2020(red)
    corner_d = {k: _r(np.min(np.hypot(redm[:, 0] - e, redm[:, 1] - n)))
                for k, (e, n) in CORNERS_MGA2020.items()}
    off = np.array([redm[np.argmin(np.hypot(redm[:, 0] - e, redm[:, 1] - n))] - (e, n)
                    for e, n in CORNERS_MGA2020.values()])

    def fill_vertices(box):
        x0, x1, y0, y1 = box
        pts = [q for p in paths if pdfvec.colour_name(p.fill) == "#dbdbdb" for s in p.subpaths
               if x0 <= np.min(np.asarray(s)[:, 0]) and np.max(np.asarray(s)[:, 0]) <= x1
               and y0 <= np.min(np.asarray(s)[:, 1]) and np.max(np.asarray(s)[:, 1]) <= y1
               for q in np.asarray(s).tolist()]
        return np.unique(np.round(np.array(pts), 2), axis=0)

    grp = {f["properties"]["id"]: np.array(f["properties"]["mga94_corners"])
           for f in json.loads((TRUTH / "panels.geojson").read_text())["features"]}
    to94 = Transformer.from_crs(7856, 28356, always_xy=True)          # GDA2020 / MGA 56 -> GDA94 / MGA 56
    panels = {}
    for pid, box, label in (("AP_LW709", (1500, 2900, 1000, 1250), "LONGWALL 709 VOID 2615m x 324m"),
                            ("AP_LW905", (850, 1400, 1100, 1500), "LONGWALL 905 VOID 858m x 324m")):
        v20 = to_mga2020(fill_vertices(box))
        v94 = np.array(to94.transform(v20[:, 0], v20[:, 1])).T
        rec = {"drawing_label": label, "south32_vertices": int(len(v20))}
        for name, v in (("msec_as_mga94", v94), ("msec_as_mga2020", v20)):
            d = [float(np.min(np.hypot(*(v - c).T))) for c in grp[pid]]
            vec = np.mean([v[np.argmin(np.hypot(*(v - c).T))] - c for c in grp[pid]], axis=0)
            rec[name] = {"corner_to_nearest_south32_vertex_m": [_r(x, 1) for x in d],
                         "max_m": _r(max(d), 1), "mean_offset_south32_minus_t1_m": [_r(x) for x in vec]}
        panels[pid] = rec
    shift = np.array(to94.transform(288000.0, 6217000.0)) - (288000.0, 6217000.0)
    return {
        "drawing": "APNSVY00237 (South32, Appin LW709-711 and LW905 Extraction Plan VARIATION, Approved Plan, "
                   "8 Feb 2023, 1:6000, MGA2020)",
        "source_pdf": S32.name,
        "grid": {"e_lines": len(e_lines), "n_lines": len(n_lines), "spacing_pt": _r(spacing, 2),
                 "spacing_m_at_1_6000": _r(spacing * SCALE_M_PER_PT, 1), "step_e_m": estep, "step_n_m": nstep,
                 "fit_rms_m": [_r(rms_e, 3), _r(rms_n, 3)]},
        "corner_table_check": {"corners": len(corner_d), "max_m": max(corner_d.values()),
                               "mean_offset_outline_minus_table_m": [_r(x) for x in off.mean(axis=0)],
                               "per_corner_m": corner_d,
                               "note": "the outline is dashed: a dash end lies within a few metres of each corner"},
        "gda2020_to_gda94_shift_here_m": [_r(x) for x in shift],
        "panels": panels,
    }


def date_bar_check() -> dict:
    fc = {f["properties"]["id"]: f["properties"] for f in json.loads((TRUTH / "panels.geojson").read_text())["features"]}
    published = {"W1 start": fc["TW_W1"]["mined_start"], "W3 start": fc["TW_W3"]["mined_start"],
                 "W4 start": fc["TW_W4"]["mined_start"], "W4 end": fc["TW_W4"]["mined_end"]}
    out = {}
    for page in range(89, 106):
        fig = gv.extract(R8, page, WORK)
        a, b = fig.x_axis
        paths = pdfvec.read_paths(pdfvec.page_svg(R8, page, WORK))
        words = pdfvec.read_words(R8, page)
        bars = []
        for p in paths:
            if pdfvec.colour_name(p.fill) != "#ffcccc":
                continue
            for s in p.subpaths:
                s = np.asarray(s, float)
                if np.ptp(s[:, 0]) > 5 and np.ptp(s[:, 1]) < 15:
                    x0, x1, yc = s[:, 0].min(), s[:, 0].max(), s[:, 1].mean()
                    labels = sorted({w["text"] + " " + n["text"] for w in words for n in words
                                     if w["text"] == "LW" and n["text"].startswith("W") and abs(n["yc"] - w["yc"]) < 1
                                     and 0 < n["x0"] - w["x1"] < 6 and x0 <= w["xc"] <= x1 and abs(w["yc"] - yc) < 8})
                    bars.append((a + b * x0, a + b * x1, labels))
        if not bars:
            out[fig.figure_id] = {"bar": None}
            continue
        d0, d1, labels = min(bars, key=lambda t: t[0])[0], max(bars, key=lambda t: t[1])[1], bars[0][2]
        first = "W1 start" if "LW W1" in labels else "W3 start" if "LW W3" in labels else "W4 start"
        axis_start = a + b * min(x for _, x, _ in fig.x_ticks)
        rec = {"labels": labels, "bar_start": gv.from_day(d0).isoformat(), "bar_end": gv.from_day(d1).isoformat(),
               "end_vs_W4_end_days": _r(d1 - gv.day_number(date.fromisoformat(published["W4 end"])))}
        if abs(d0 - axis_start) > 1.0:            # the bar starts inside the axis: its start is a mining date
            key = "start_vs_" + first.replace(" ", "_") + "_days"
            rec[key] = _r(d0 - gv.day_number(date.fromisoformat(published[first])))
        else:
            rec["start"] = "at the axis start (the unit's record starts after the mining began): not checked"
        out[fig.figure_id] = rec
    diffs = [abs(v) for r in out.values() for k, v in r.items() if k.endswith("_days")]
    return {"published": published, "figures": out, "max_abs_days": max(diffs), "checked_ends": len(diffs)}


def main():
    rec = {"what": "independent review checks of T1 (8 Oct 2026); see scripts/review_checks.py",
           "south32_appin_outlines": south32_check(), "gnss_date_axis_vs_mining_bars": date_bar_check()}
    (TRUTH / "checks" / "review_checks.json").write_text(json.dumps(rec, indent=1) + "\n")
    s = rec["south32_appin_outlines"]
    print("grid", s["grid"], "corner table max", s["corner_table_check"]["max_m"])
    for pid, p in s["panels"].items():
        print(pid, "as MGA94", p["msec_as_mga94"]["max_m"], "as MGA2020", p["msec_as_mga2020"]["max_m"])
    g = rec["gnss_date_axis_vs_mining_bars"]
    print("date bars: ends checked", g["checked_ends"], "max |days|", g["max_abs_days"])


if __name__ == "__main__":
    main()
