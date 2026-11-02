"""
Pass B of the Tahmoor West GNSS digitising (T1, 8 Oct 2026): read the same
figures again from a rendered image, with an independent calibration.

The page is rendered at 600 dpi (pdftoppm). The axes are calibrated from the
image itself: gridlines are found as pixel rows (or columns) that are dark or
grey across most of a panel, and each is given the value of the nearest tick
label (label text from pdftotext). The 7-day running average curve is then
read column by column as the median row of the pixels in its legend colour.

Shared with pass A: only the label text (which value a gridline has), the
legend colour, and roughly where each panel is (to look in the right place).
Not shared: every coordinate used for calibration or for the curve.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import gnss_plot_vector as gv
import pdfvec

DPI = 600
WORKDIR = None                 # set by the caller (scratch folder for SVG and PNG renders)
SCALE = DPI / 72.0             # px per pt


def render(pdf: Path, page: int, workdir: Path) -> np.ndarray:
    out = workdir / f"{pdf.stem}_p{page}_{DPI}"
    png = Path(str(out) + ".png")
    if not png.exists():
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(DPI), "-png", "-singlefile",
                        str(pdf), str(out)], check=True, capture_output=True)
    return np.asarray(Image.open(png).convert("RGB")).astype(np.int16)


def _line_mask(img):
    """Pixels that belong to a dark or grey line (not white, little colour)."""
    mx, mn = img.max(axis=2), img.min(axis=2)
    return (mx < 215) & (mx - mn < 30)


def _clusters(idx, weights):
    """Group consecutive indices; return weighted centres."""
    out, cur = [], []
    for i in idx:
        if cur and i != cur[-1] + 1:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return [float(np.average(c, weights=weights[c])) for c in out]


def calibrate(img, panel, x0_pt, x1_pt, words, axis_x):
    r0, r1 = int((panel.y_top - 3) * SCALE), int((panel.y_bottom + 3) * SCALE)
    c0, c1 = int((x0_pt + 2) * SCALE), int((x1_pt - 2) * SCALE)
    lm = _line_mask(img[r0:r1, c0:c1])
    frac_r = lm.mean(axis=1)
    rows = _clusters(list(np.where(frac_r > 0.6)[0]), frac_r)
    rows = [r + r0 for r in rows]
    labels = [w for w in words if gv.NUM_RE.match(w["text"]) and w["x1"] < axis_x - 1 and w["x0"] > axis_x - 45
              and panel.y_top - 6 <= w["yc"] <= panel.y_bottom + 6]
    pairs = []
    for w in labels:
        if not rows:
            break
        j = int(np.argmin([abs(r - w["yc"] * SCALE) for r in rows]))
        if abs(rows[j] - w["yc"] * SCALE) < 6 * SCALE:
            pairs.append((float(w["text"]), rows[j]))
    v = np.array([p[0] for p in pairs]); r = np.array([p[1] for p in pairs])
    b, a = np.polyfit(r, v, 1)
    rms = float(np.sqrt(np.mean((v - (a + b * r)) ** 2)))
    return (float(a), float(b)), rms, len(pairs)


def calibrate_x(img, panel, x0_pt, x1_pt, words, ybot_pt):
    r0, r1 = int((panel.y_top + 3) * SCALE), int((panel.y_bottom - 3) * SCALE)
    c0, c1 = int((x0_pt - 3) * SCALE), int((x1_pt + 3) * SCALE)
    lm = _line_mask(img[r0:r1, c0:c1])
    frac_c = lm.mean(axis=0)
    cols = [c + c0 for c in _clusters(list(np.where(frac_c > 0.6)[0]), frac_c)]
    labels = [(gv.parse_axis_date(w["text"]), w) for w in words]
    labels = [(d, w) for d, w in labels if d is not None and w["y0"] > ybot_pt]
    pairs = []
    for d, w in labels:
        j = int(np.argmin([abs(c - w["xc"] * SCALE) for c in cols]))
        if abs(cols[j] - w["xc"] * SCALE) < 3 * SCALE:
            pairs.append((gv.day_number(d), cols[j]))
    dn = np.array([p[0] for p in pairs], float); c = np.array([p[1] for p in pairs])
    b, a = np.polyfit(c, dn, 1)
    rms = float(np.sqrt(np.mean((dn - (a + b * c)) ** 2)))
    return (float(a), float(b)), rms, len(pairs)


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def read_curve(img, panel, x0_pt, x1_pt, colour, other_colours, tol=40.0, exclude=()):
    r0, r1 = int((panel.y_top - 8) * SCALE), int((panel.y_bottom + 8) * SCALE)
    c0, c1 = int((x0_pt + 1) * SCALE), int((x1_pt - 1) * SCALE)
    sub = img[r0:r1, c0:c1].astype(float)
    target = hex_rgb(colour)
    d = np.sqrt(((sub - target) ** 2).sum(axis=2))
    ok = d < tol
    for oc in other_colours:
        ok &= d < np.sqrt(((sub - hex_rgb(oc)) ** 2).sum(axis=2))
    for (bx0, by0, bx1, by1) in exclude:            # legend boxes: their sample lines are not data
        ok[max(0, int(by0 * SCALE) - r0 - 2):max(0, int(by1 * SCALE) - r0 + 3),
           max(0, int(bx0 * SCALE) - c0 - 2):max(0, int(bx1 * SCALE) - c0 + 3)] = False
    cols, vals = [], []
    for j in range(ok.shape[1]):
        rr = np.where(ok[:, j])[0]
        if rr.size:
            cols.append(j + c0)
            vals.append(float(np.median(rr)) + r0)
    return np.array(cols), np.array(vals)


def legend_boxes(fig: gv.Figure, words):
    """Page boxes (pt) of the legends: the smallest closed rectangle around each
    legend's sample lines (read from the vector page; used only as a mask)."""
    paths = pdfvec.read_paths(pdfvec.page_svg(Path(fig.pdf), fig.page, WORKDIR))
    boxes = []
    for kind in ("daily", "avg7"):
        sx0, sy, sx1 = fig.styles[kind][2]
        best = None
        for p in paths:
            for s in p.subpaths:
                s = np.asarray(s)
                if len(s) not in (4, 5):
                    continue
                bx0, by0, bx1, by1 = s[:, 0].min(), s[:, 1].min(), s[:, 0].max(), s[:, 1].max()
                if bx0 <= sx0 and sx1 <= bx1 and by0 <= sy <= by1 and (bx1 - bx0) < 400:
                    area = (bx1 - bx0) * (by1 - by0)
                    if best is None or area < best[0]:
                        best = (area, (bx0, by0, bx1, by1))
        if best:
            boxes.append(best[1])
        else:
            boxes.append((sx0 - 2, sy - 3, sx1 + 2, sy + 3))
    return boxes


def extract(fig: gv.Figure, workdir: Path, axis_x: float | None = None):
    """{panel: dict(days, values, y_cal, y_rms, x_cal, x_rms)} for the 7-day curve."""
    pdf = Path(fig.pdf)
    img = render(pdf, fig.page, workdir)
    words = pdfvec.read_words(pdf, fig.page)
    xs = [t[1] for t in fig.x_ticks]
    x0_pt, x1_pt = min(xs), max(xs)
    axis_x = axis_x if axis_x is not None else x0_pt
    ybot = max(p.y_bottom for p in fig.panels.values())
    col7 = fig.styles["avg7"][0]
    others = [fig.styles["daily"][0], "#808080", "#000000", "#ffffff"]
    paths = pdfvec.read_paths(pdfvec.page_svg(pdf, fig.page, WORKDIR))
    masks = legend_boxes(fig, words) + gv.inset_boxes(paths, fig.panels)   # not data: legends, polar plot, map
    out = {}
    for name, pn in fig.panels.items():
        (ya, yb), yrms, ny = calibrate(img, pn, x0_pt, x1_pt, words, axis_x)
        (xa, xb), xrms, nx = calibrate_x(img, pn, x0_pt, x1_pt, words, ybot)
        cols, rows = read_curve(img, pn, x0_pt, x1_pt, col7, others, exclude=masks)
        out[name] = dict(days=xa + xb * cols, values=ya + yb * rows, y_rms_mm=yrms, y_n=ny,
                         x_rms_days=xrms, x_n=nx, n_columns=int(cols.size))
    return out
