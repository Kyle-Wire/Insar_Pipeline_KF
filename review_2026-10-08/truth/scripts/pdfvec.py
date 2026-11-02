"""
Read vector drawings and words from one PDF page (T1 digitising, 8 Oct 2026).

Poppler's pdftocairo writes the page as SVG; every stroked or filled path is
returned with its colour, width and points in PDF page units (pt, origin top
left, y down), after its transform. Curves keep their end points only (the
plots and drawings use straight segments; a Bezier's control points are not
on the line). Words and their boxes come from pdftotext -bbox in the same
units. Nothing is rasterised here: these are the drawing's own coordinates.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SVG = "{http://www.w3.org/2000/svg}"
_NUM = r"-?\d*\.?\d+(?:[eE][-+]?\d+)?"


@dataclass
class VPath:
    stroke: tuple | None          # (r, g, b) in 0..1, None = not stroked
    fill: tuple | None
    width: float
    subpaths: list = field(default_factory=list)   # [np.ndarray (n, 2)] page pt
    closed: list = field(default_factory=list)     # bool per subpath

    @property
    def points(self) -> np.ndarray:
        return np.vstack(self.subpaths) if self.subpaths else np.zeros((0, 2))

    def bbox(self):
        p = self.points
        return (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) if len(p) else None


def _rgb(s: str | None):
    if not s or s == "none":
        return None
    m = re.match(r"rgb\(([^)]*)\)", s)
    if not m:
        return None
    vals = [v.strip() for v in m.group(1).split(",")]
    return tuple(round(float(v.rstrip("%")) / 100.0, 4) if v.endswith("%") else round(float(v) / 255.0, 4)
                 for v in vals)


def _matrix(s: str | None) -> np.ndarray:
    M = np.eye(3)
    if not s:
        return M
    m = re.search(r"matrix\(([^)]*)\)", s)
    if m:
        a, b, c, d, e, f = (float(v) for v in re.split(r"[ ,]+", m.group(1).strip()))
        M = np.array([[a, c, e], [b, d, f], [0, 0, 1]])
    return M


def _parse_d(d: str):
    toks = re.findall(r"[MLCZmlcz]|" + _NUM, d)
    subs, closed, cur = [], [], []
    i, cmd = 0, None
    while i < len(toks):
        t = toks[i]
        if t in "MLCZmlcz":
            cmd = t
            i += 1
            if cmd in "Zz":
                if cur:
                    subs.append(cur)
                    closed.append(True)
                    cur = []
            continue
        if cmd in ("M", "L"):
            x, y = float(toks[i]), float(toks[i + 1])
            i += 2
            if cmd == "M" and cur:
                subs.append(cur)
                closed.append(False)
                cur = []
            cur.append((x, y))
            if cmd == "M":
                cmd = "L"
        elif cmd == "C":
            x, y = float(toks[i + 4]), float(toks[i + 5])
            i += 6
            cur.append((x, y))
        else:
            raise ValueError(f"path command {cmd!r} not handled")
    if cur:
        subs.append(cur)
        closed.append(False)
    return subs, closed


def page_svg(pdf: Path, page: int, workdir: Path | None = None) -> Path:
    workdir = Path(workdir or tempfile.mkdtemp())
    out = workdir / f"{Path(pdf).stem}_p{page}.svg"
    if not out.exists():
        subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), str(pdf), str(out)],
                       check=True, capture_output=True)
    return out


def read_paths(svg: Path) -> list[VPath]:
    root = ET.parse(svg).getroot()
    out: list[VPath] = []

    def walk(el, M, inherited_fill, in_defs):
        tag = el.tag.replace(SVG, "")
        if tag in ("defs", "clipPath", "symbol", "mask"):
            return
        M = M @ _matrix(el.get("transform"))
        fill = el.get("fill", inherited_fill)
        if tag == "path" and el.get("d"):
            subs, closed = _parse_d(el.get("d"))
            pts = []
            for s in subs:
                a = np.c_[np.asarray(s, float), np.ones(len(s))] @ M.T
                pts.append(a[:, :2])
            out.append(VPath(stroke=_rgb(el.get("stroke")), fill=_rgb(fill) if el.get("stroke") is None else
                             _rgb(el.get("fill")), width=float(el.get("stroke-width", "0") or 0) *
                             float(np.sqrt(abs(np.linalg.det(M[:2, :2])))), subpaths=pts, closed=closed))
        for ch in el:
            walk(ch, M, fill, in_defs)

    walk(root, np.eye(3), None, False)
    return out


def read_words(pdf: Path, page: int) -> list[dict]:
    """[{text, x0, y0, x1, y1, xc, yc}] in page pt (origin top left)."""
    r = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-bbox", str(pdf), "-"],
                       check=True, capture_output=True, text=True)
    words = []
    for m in re.finditer(r'<word xMin="([^"]+)" yMin="([^"]+)" xMax="([^"]+)" yMax="([^"]+)">([^<]*)</word>',
                         r.stdout):
        x0, y0, x1, y1 = (float(m.group(k)) for k in range(1, 5))
        t = (m.group(5).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))
        words.append(dict(text=t, x0=x0, y0=y0, x1=x1, y1=y1, xc=(x0 + x1) / 2, yc=(y0 + y1) / 2))
    return words


def colour_name(c) -> str:
    return "none" if c is None else "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)
