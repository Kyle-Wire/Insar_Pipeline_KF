"""
Pass A of the Tahmoor West GNSS digitising (T1, 8 Oct 2026): read the MSEC
GNSS figures from the PDF's own vector drawing.

Each figure (e.g. Report 8 page 89, "Fig. G07-R21") has three stacked panels
(change in easting, northing and height, in mm) against a date axis. The
plotting program wrote every daily value as a vertex of a stroked path, and
the 7-day running average as another path. This module:

  1. finds the left axis, its major tick marks and the numeric tick labels,
     splits them into panels (a panel is a run of evenly spaced labels) and
     fits value = a + b * y per panel (least squares; the fit's residual at
     the ticks is the y calibration error);
  2. finds the date tick marks on the bottom axis and their labels
     ("Nov-19" = 1 Nov 2019, "01-Aug-21" = 1 Aug 2021) and fits
     day = a + b * x (residual = x calibration error, in days);
  3. reads the curve styles from the figure's legend ("Daily change ..." and
     "7 day running average"), takes every path in those styles whose
     vertices run left to right, and assigns it to the panel it lies in;
  4. converts the vertices to (date, value).

Nothing is resampled or smoothed here. Units: page points (pt) in, mm and
days out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

import pdfvec

DATE_RE = re.compile(r"^(?:(\d\d)-)?([A-Z][a-z]{2})-(\d\d)$")
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
EPOCH = date(2000, 1, 1)


def parse_axis_date(text: str) -> date | None:
    m = DATE_RE.match(text)
    if not m:
        return None
    day = int(m.group(1)) if m.group(1) else 1
    return datetime.strptime(f"{day:02d}-{m.group(2)}-{m.group(3)}", "%d-%b-%y").date()


def day_number(d: date) -> int:
    return (d - EPOCH).days


def from_day(x: float) -> date:
    return EPOCH + timedelta(days=int(round(x)))


@dataclass
class Panel:
    name: str                      # easting | northing | height
    a: float                       # value = a + b * y   (mm, y in pt)
    b: float
    y_top: float
    y_bottom: float
    ticks: list                    # [(value, y_tick, y_label)]
    rms_mm: float                  # calibration residual at the ticks
    gridline_mm: float             # spacing of the labelled ticks (gridlines)
    minor_mm: float | None = None  # spacing of the minor ticks, when found


@dataclass
class Figure:
    pdf: str
    page: int
    title: str
    figure_id: str
    x_axis: tuple                  # (a, b): day = a + b * x
    x_rms_days: float
    x_ticks: list                  # [(date iso, x_tick, x_label)]
    panels: dict = field(default_factory=dict)
    styles: dict = field(default_factory=dict)       # daily / avg7 -> (colour, width)
    curves: dict = field(default_factory=dict)       # (panel, kind) -> list of (days[], values[])
    curve_widths: dict = field(default_factory=dict) # (panel, kind) -> stroke widths (pt) of the paths read
    warnings: list = field(default_factory=list)


def _segments(paths, colour=None, width=None, tol=0.02):
    out = []
    for p in paths:
        if p.stroke is None:
            continue
        if colour is not None and pdfvec.colour_name(p.stroke) != colour:
            continue
        if width is not None and abs(p.width - width) > tol:
            continue
        for s in p.subpaths:
            for i in range(len(s) - 1):
                out.append((s[i][0], s[i][1], s[i + 1][0], s[i + 1][1]))
    return np.array(out) if out else np.zeros((0, 4))


def _dark(c) -> bool:
    return c is not None and max(c) < 0.15


def find_left_axis(paths):
    """x of the left axis = the common right end of the short horizontal dark tick marks."""
    segs = np.array([(x0, y0, x1, y1) for p in paths if _dark(p.stroke)
                     for s in p.subpaths for (x0, y0), (x1, y1) in zip(s[:-1], s[1:])])
    hor = segs[(np.abs(segs[:, 1] - segs[:, 3]) < 1e-3) & (np.abs(segs[:, 2] - segs[:, 0]) < 9)
               & (np.abs(segs[:, 2] - segs[:, 0]) > 1)]
    right = np.round(np.maximum(hor[:, 0], hor[:, 2]), 1)
    vals, cnt = np.unique(right, return_counts=True)
    axis_x = float(vals[np.argmax(cnt)])
    ticks = hor[np.abs(np.maximum(hor[:, 0], hor[:, 2]) - axis_x) < 0.15]
    return axis_x, ticks


def calibrate_panels(paths, words, warnings):
    axis_x, ticks = find_left_axis(paths)
    lengths = np.round(np.abs(ticks[:, 2] - ticks[:, 0]), 1)
    major_len = lengths.max()
    major = ticks[lengths >= major_len - 0.3]
    minor = ticks[lengths < major_len - 0.3]
    my = np.sort(major[:, 1])
    nums = [w for w in words if NUM_RE.match(w["text"]) and w["x1"] < axis_x - 1 and w["x0"] > axis_x - 45]
    pairs = []
    for w in nums:
        j = int(np.argmin(np.abs(my - w["yc"])))
        if abs(my[j] - w["yc"]) < 6:
            pairs.append((float(w["text"]), float(my[j]), w["yc"]))
    pairs.sort(key=lambda t: t[1])
    # split into panels: constant value step per pt
    runs, cur = [], [pairs[0]]
    for p in pairs[1:]:
        q = cur[-1]
        if p[1] - q[1] > 1e-6:
            s = (p[0] - q[0]) / (p[1] - q[1])
            if len(cur) >= 2:
                s0 = (cur[-1][0] - cur[-2][0]) / (cur[-1][1] - cur[-2][1])
                if np.sign(s) != np.sign(s0) or abs(s - s0) > 0.02 * abs(s0):
                    runs.append(cur)
                    cur = [p]
                    continue
            elif abs(p[0] - q[0]) == 0:
                runs.append(cur)
                cur = [p]
                continue
        cur.append(p)
    runs.append(cur)
    runs = [r for r in runs if len(r) >= 3]
    panels = {}
    titles = {"easting": None, "northing": None, "height": None}
    for k in titles:
        hits = [w for w in words if w["text"].lower() == k]
        titles[k] = hits[0]["yc"] if hits else None
    for r in runs:
        v = np.array([t[0] for t in r]); y = np.array([t[1] for t in r])
        b, a = np.polyfit(y, v, 1)
        res = v - (a + b * y)
        name = min((k for k in titles if titles[k] is not None),
                   key=lambda k: abs(titles[k] - y.mean()) if titles[k] is not None else 1e9)
        if name in panels:
            warnings.append(f"two label runs matched panel '{name}'")
        steps = np.diff(np.sort(v))
        mn = minor[(minor[:, 1] >= y.min() - 0.5) & (minor[:, 1] <= y.max() + 0.5)]
        minor_mm = None
        if len(mn) >= 2:
            ally = np.sort(np.r_[mn[:, 1], y])
            d = np.diff(ally)
            minor_mm = float(abs(b) * np.median(d[d > 0.3]))
        panels[name] = Panel(name, float(a), float(b), float(y.min()), float(y.max()),
                             [tuple(map(float, t)) for t in r], float(np.sqrt(np.mean(res ** 2))),
                             float(np.median(steps)), minor_mm)
    return axis_x, panels


def calibrate_dates(paths, words, panels, warnings):
    ybot = max(p.y_bottom for p in panels.values())
    segs = np.array([(x0, y0, x1, y1) for p in paths if _dark(p.stroke)
                     for s in p.subpaths for (x0, y0), (x1, y1) in zip(s[:-1], s[1:])])
    ver = segs[(np.abs(segs[:, 0] - segs[:, 2]) < 1e-3) & (np.abs(segs[:, 3] - segs[:, 1]) < 9)
               & (np.minimum(segs[:, 1], segs[:, 3]) > ybot + 2)]
    labels = [(parse_axis_date(w["text"]), w) for w in words]
    labels = [(d, w) for d, w in labels if d is not None and w["y0"] > ybot]
    if not labels:
        raise ValueError("no date labels below the panels")
    ylab = min(w["y0"] for _, w in labels)
    ver = ver[np.maximum(ver[:, 1], ver[:, 3]) <= ylab + 1]
    tx = np.unique(np.round(ver[:, 0], 2))
    pts = []
    for d, w in labels:
        if len(tx):
            j = int(np.argmin(np.abs(tx - w["xc"])))
            if abs(tx[j] - w["xc"]) < 3:
                pts.append((day_number(d), float(tx[j]), w["xc"], d.isoformat()))
                continue
        warnings.append(f"date label {w['text']} has no tick mark; its text centre is used")
        pts.append((day_number(d), w["xc"], w["xc"], d.isoformat()))
    pts.sort(key=lambda t: t[1])
    dn = np.array([t[0] for t in pts], float); x = np.array([t[1] for t in pts])
    b, a = np.polyfit(x, dn, 1)
    res = dn - (a + b * x)
    return (float(a), float(b)), float(np.sqrt(np.mean(res ** 2))), [(t[3], t[1], t[2]) for t in pts]


def legend_style(paths, words, first_word: str, phrase_word: str | None = None):
    """(colour, width) of the 2-point legend sample left of the legend text."""
    cands = [w for w in words if w["text"] == first_word]
    if phrase_word:
        cands = [w for w in cands if any(v["text"] == phrase_word and abs(v["yc"] - w["yc"]) < 2
                                         and 0 < v["x0"] - w["x1"] < 60 for v in words)]
    for w in cands:
        best = None
        for p in paths:
            if p.stroke is None or len(p.points) != 2:
                continue
            (x0, y0), (x1, y1) = p.points
            if abs(y0 - y1) > 1e-3 or abs(y0 - w["yc"]) > 4:
                continue
            gap = w["x0"] - max(x0, x1)
            if 0 < gap < 20 and (best is None or gap < best[0]):
                best = (gap, pdfvec.colour_name(p.stroke), round(p.width, 3), (min(x0, x1), y0, max(x0, x1)))
        if best:
            return best[1], best[2], best[3]
    return None


def inset_boxes(paths, panels):
    """Boxes of the insets drawn over the top panel (the polar movement plot and the
    location map): closed 4-corner paths, filled or stroked, larger than 60 x 60 pt,
    reaching above the top panel."""
    top = min(p.y_top for p in panels.values())
    boxes = []
    for p in paths:
        for s, closed in zip(p.subpaths, p.closed):
            s = np.asarray(s)
            if len(s) not in (4, 5):
                continue
            x0, y0, x1, y1 = s[:, 0].min(), s[:, 1].min(), s[:, 0].max(), s[:, 1].max()
            if x1 - x0 > 60 and y1 - y0 > 60 and y0 < top - 5 and y1 > top:
                boxes.append((float(x0), float(y0), float(x1), float(y1)))
    return boxes


def extract(pdf: Path, page: int, workdir: Path) -> Figure:
    svg = pdfvec.page_svg(pdf, page, workdir)
    paths = pdfvec.read_paths(svg)
    words = pdfvec.read_words(pdf, page)
    warnings: list = []
    fig_id = next((f"{a['text']} {b['text']}" for a, b in zip(words, words[1:])
                   if a["text"] == "Fig." and b["text"].startswith("G")), "")
    title = " ".join(w["text"] for w in words if w["y1"] < 58)
    axis_x, panels = calibrate_panels(paths, words, warnings)
    xa, xrms, xticks = calibrate_dates(paths, words, panels, warnings)
    fig = Figure(str(pdf), page, title, fig_id, xa, xrms, xticks, panels, warnings=warnings)
    daily = legend_style(paths, words, "Daily")
    avg7 = legend_style(paths, words, "running") or legend_style(paths, words, "7", "day")
    if daily is None or avg7 is None:
        raise ValueError(f"page {page}: legend styles not found (daily={daily}, avg7={avg7})")
    fig.styles = {"daily": daily, "avg7": avg7}
    right = max(t[1] for t in xticks)
    insets = inset_boxes(paths, panels)
    for kind, (col, wid, sample) in fig.styles.items():
        for p in paths:
            if p.stroke is None or pdfvec.colour_name(p.stroke) != col or abs(p.width - wid) > 0.01:
                continue
            for s in p.subpaths:
                s = np.asarray(s)
                if len(s) == 2 and abs(s[0, 1] - sample[1]) < 0.01 and abs(s[0, 0] - sample[0]) < 0.01:
                    continue                                   # the legend sample itself
                dx = np.diff(s[:, 0])
                if len(s) >= 2 and (dx < -0.05).any():
                    continue                                   # not a time series (polar track)
                if insets and any(r[0] - .5 <= s[:, 0].min() and s[:, 0].max() <= r[2] + .5 and
                                  r[1] - .5 <= s[:, 1].min() and s[:, 1].max() <= r[3] + .5 for r in insets):
                    continue                                   # inside an inset (polar plot, map): hidden on the page
                if s[:, 0].min() < axis_x - 0.5 or s[:, 0].max() > right + 0.5:
                    continue
                ym = float(np.median(s[:, 1]))
                pan = [k for k, pn in panels.items() if pn.y_top - 9 <= ym <= pn.y_bottom + 9]
                if len(pan) != 1:
                    warnings.append(f"{kind} path at y={ym:.1f} fits {len(pan)} panels; skipped")
                    continue
                pn = panels[pan[0]]
                days = xa[0] + xa[1] * s[:, 0]
                vals = pn.a + pn.b * s[:, 1]
                if (s[:, 1] < pn.y_top - 0.5).any() or (s[:, 1] > pn.y_bottom + 0.5).any():
                    warnings.append(f"{kind} {pan[0]}: {int(((s[:, 1] < pn.y_top - .5) | (s[:, 1] > pn.y_bottom + .5)).sum())}"
                                    " vertices lie outside the labelled axis range (kept: the vector holds them)")
                fig.curves.setdefault((pan[0], kind), []).append((days, vals))
                fig.curve_widths.setdefault((pan[0], kind), []).append(float(p.width))
    for k in list(fig.curves):
        fig.curves[k].sort(key=lambda dv: dv[0][0])
    return fig


def merged(fig: Figure, panel: str, kind: str):
    parts = fig.curves.get((panel, kind), [])
    if not parts:
        return np.zeros(0), np.zeros(0)
    d = np.concatenate([p[0] for p in parts]); v = np.concatenate([p[1] for p in parts])
    o = np.argsort(d, kind="stable")
    return d[o], v[o]
