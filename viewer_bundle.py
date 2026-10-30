"""What the results viewer reads, and the date-window arithmetic it does -
written once here, in Python, as the reference the browser's
viewer_assets/viewer_core.js is tested against.

    bundle(cfg)                     the header (meta, the report's figures,
                                    the stamp) and the arrays the page needs,
                                    from the viewer table (field_points) and
                                    client_report.json
    encode(header, arrays)          the binary the live viewer sends
                                    (/api/bundle); decode() reads it back
    window_index(dates, a, b)       the images of a date window and whether
                                    it can be measured (results.MIN_DATES
                                    images over results.MIN_YEARS)
    window_figures(fd, i0, i1)      every point's figures over the window

Three kinds of figure, and the page labels each (19 Oct 2026):

  report    the whole record: the stored figures - the very numbers of the
            client report, the KMZ and the rasters - never recomputed
  tested    a window results.analyse measured: its null table is passed in
            (`table`), so every figure here is analyse's
  look      any other window: the noise of a single point is measured on a
            sample of the report's farmland cells (field_points._far_sample),
            not on all of it, so the signal-to-noise is a close estimate, not
            the tested figure

Why each point's OWN cell (cell_los) and not the 3 x 3 cells around it
(series_los): the stored rate is the cell's own; the 3 x 3 average differs by
up to 12.5 mm/yr on two projects. The own-cell series is the cell
minus farmland of its class at each image (field_points._frame), and every
good cell is finite on every image, so its slope over any window IS
results.analyse's relative rate for that window (the farmland class mean of
the cells' slopes is the slope of the farmland class mean), and its last
minus first image is the delivered line-of-sight change.

Units: LOS mm and mm/yr relative to farmland of the same coherence class
(no GNSS); vertical = LOS / the up component (+ the record's regional GNSS
rate when the report is anchored - the same for every window).
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

MIN_DATES = 6          # results.MIN_DATES
MIN_YEARS = 0.25       # results.MIN_YEARS
END_DAYS = 30          # results.analyse ends(): images within 30 days of each end
Z95 = 1.96             # the per-point threshold of the report, the rasters and the KMZ
MAD_SD = 1.4826        # significance.farmland_cell_spread
FORMAT = 1


# ------------------------------------------------------------------ windows
def window_index(dates: list[str], start: str | None = None, end: str | None = None) -> tuple[int, int]:
    """(first, last) image index inside [start, end], both YYYYMMDD and
    inclusive, as results.analyse selects them (the dates are in order)."""
    a = (start or dates[0]).replace("-", "")[:8]
    b = (end or dates[-1]).replace("-", "")[:8]
    sel = [i for i, d in enumerate(dates) if a <= d <= b]
    if not sel:
        raise ValueError(f"no images between {a} and {b}")
    return sel[0], sel[-1]


def window_problem(t, i0: int, i1: int) -> str | None:
    """Why a window cannot be measured (results.analyse's rule), or None."""
    n = i1 - i0 + 1
    if n < MIN_DATES:
        return f"only {n} images in the window; at least {MIN_DATES} are needed"
    span = float(t[i1] - t[i0])
    if span < MIN_YEARS:
        return f"the window spans {span * 12:.1f} months of images; at least 3 months are needed"
    return None


def _days(dates: list[str]) -> np.ndarray:
    d0 = datetime.strptime(dates[0], "%Y%m%d")
    return np.array([(datetime.strptime(d, "%Y%m%d") - d0).days for d in dates])


# ------------------------------------------------------------------ the noise
def nearest(table: dict, i: int):
    """table[i], or the nearest listed class's (ties to the lower), as
    significance._nearest_class and client_report.cell_sigma_map."""
    if i in table:
        return table[i]
    return table[min(table, key=lambda j: (abs(j - i), j))] if table else None


def null_table(emp: dict, spread: dict) -> dict:
    """significance.cell_null_by_class: per class, the larger of its
    repeatability (the nearest measured class's) and its farmland cell
    spread (its own, when measured), to 3 decimals."""
    emp = {int(k): float(v) for k, v in (emp or {}).items() if v is not None}
    spread = {int(k): float(v) for k, v in (spread or {}).items() if v is not None}
    out = {}
    for i in range(5):
        vals = [v for v in (nearest(emp, i), spread.get(i)) if v is not None and np.isfinite(v)]
        if vals:
            out[i] = round(max(vals), 3)
    return out


def per_point_null(table: dict, kcls: np.ndarray) -> np.ndarray:
    """client_report.cell_sigma_map on the points: the class's null, or the
    nearest listed class's; NaN below the first class."""
    table = {int(k): float(v) for k, v in (table or {}).items() if v is not None}
    out = np.full(kcls.shape, np.nan)
    if not table:
        return out
    for i in range(5):
        out[kcls == i] = nearest(table, i)
    return out


def mad_sd(x: np.ndarray) -> float:
    return float(MAD_SD * np.median(np.abs(x - np.median(x))))


def far_sample(fd: dict) -> tuple[np.ndarray, np.ndarray]:
    """(series, class) of the farmland cells an untested window's noise is
    measured on: the field points that are the report's farmland, then the
    stored top-up (field_points._far_sample)."""
    sel = (fd["group"] == 0) & fd["far"].astype(bool)
    S = np.vstack([fd["cell_los"][sel], fd["far_los"]]) if fd["far_los"].size else fd["cell_los"][sel]
    K = np.concatenate([fd["kcls"][sel], fd["far_cls"]]).astype(int)
    return S, K


# ------------------------------------------------------------------ the figures
def window_figures(fd: dict, i0: int, i1: int, table: dict | None = None) -> dict:
    """Every point's figures over images i0..i1 (inclusive), as numpy arrays:

      kind              'report' (the whole record, stored figures), 'tested'
                        (table given: results.analyse's null) or 'look'
      los               LOS mm/yr relative to farmland of the point's class
      vertical_rel      los / up
      vertical          + the regional GNSS rate when anchored
      null              the single-point farmland null, LOS mm/yr
      sd_vertical       null / up, then with the GNSS uncertainty
      snr               |los| / null (the raster's scale: 1.96 stands out)
      signal_to_threshold  |los| / (1.96 null): 1 or more stands out
      change_los        last image minus first, LOS mm (the raster's)
      change_los_ends   images within 30 days of each end, mean minus mean
                        (results.analyse's 'between window ends')
      table             the null per class used; spread / n_sample per class
    """
    m = fd["meta"]
    dates = list(m["dates"])
    T = len(dates)
    whole = i0 == 0 and i1 == T - 1
    # the whole record is the stored figures, whatever its length (a window
    # rule for a record would refuse what the report itself states)
    t = np.asarray(m.get("t") if m.get("t") is not None else [], float)
    why = None if whole else window_problem(t, i0, i1)
    if why:
        raise ValueError(why)
    up = np.asarray(fd["up"], float)
    anchored = bool(m.get("anchored"))
    bg_v = float(m.get("bg_v") or 0.0) if anchored else 0.0
    bg_u = m.get("bg_u") if anchored else None
    C = fd.get("cell_los")
    years = float(t[i1] - t[i0]) if t.size == T else float(m.get("span") or 0.0)
    out: dict = {"i0": i0, "i1": i1, "first": dates[i0], "last": dates[i1], "n_images": i1 - i0 + 1, "years": years}
    if C is not None and C.size and t.size == T:
        Cw = C[:, i0:i1 + 1].astype(float)
        out["change_los"] = Cw[:, -1] - Cw[:, 0]
        d = _days(dates)[i0:i1 + 1]
        first, last = d - d[0] <= END_DAYS, d[-1] - d <= END_DAYS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # an all-NaN point: NaN, said so
            out["change_los_ends"] = np.nanmean(Cw[:, last], 1) - np.nanmean(Cw[:, first], 1)
    if whole:
        out["kind"] = "report"
        los = np.asarray(fd["los_rel"], float)
        null = np.where(np.isfinite(fd["sd_v"]), np.asarray(fd["sd_v"], float) * up, np.nan)
        out["table"] = {int(k): float(v) for k, v in (m.get("rep_cell_null_by_class") or {}).items()
                        if v is not None}
        out["change_los"] = np.asarray(fd["change_los"], float)
    else:
        if C is None or not C.size:
            raise ValueError("this viewer table has no own-cell series (built before 19 Oct 2026): rebuild the "
                             "client KMZ for date windows")
        tw = t[i0:i1 + 1]
        los = slopes(tw, C[:, i0:i1 + 1])
        if table is not None:
            out["kind"] = "tested"
            tab = {int(k): float(v) for k, v in table.items() if v is not None}
        else:
            out["kind"] = "look"
            S, K = far_sample(fd)
            rf = slopes(tw, S[:, i0:i1 + 1])
            true_n = {int(k): int(v) for k, v in (m.get("far_cells_by_class") or {}).items()}
            spread, n_s = {}, {}
            for j in range(5):
                x = rf[(K == j) & np.isfinite(rf)]
                n_s[j] = int(x.size)
                if true_n.get(j, 0) >= int(m.get("min_spread_cells") or 200) and x.size:
                    spread[j] = round(mad_sd(x), 3)
            tab = null_table(m.get("emp_by_class") or {}, spread)
            out["spread"], out["n_sample"] = spread, n_s
        out["table"] = tab
        null = per_point_null(tab, np.asarray(fd["kcls"], int))
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(null) & (null > 0)
        out["los"] = los
        out["vertical_rel"] = los / up
        out["vertical"] = los / up + bg_v
        out["null"] = null
        out["sd_vertical"] = np.where(ok, np.hypot(null / up, bg_u or 0.0), np.nan)
        out["snr"] = np.where(ok & np.isfinite(los), np.abs(los) / null, np.nan)
        out["signal_to_threshold"] = np.where(ok & np.isfinite(los), np.abs(los) / (Z95 * null), np.nan)
    return out


def point_series(fd: dict, j: int, i0: int, i1: int, vertical: bool = True) -> np.ndarray:
    """Point j's own-cell series over images i0..i1 (mm), zeroed on the
    window's own straight-line fit at the window's first image, so the
    fitted line starts at 0 and ends at rate x years: vertical = LOS / up +
    the regional GNSS rate x time when anchored (field_points.
    series_vertical_in's frame, on the window and the own cell)."""
    m = fd["meta"]
    t = np.asarray(m["t"], float)[i0:i1 + 1]
    y = np.asarray(fd["cell_los"][j, i0:i1 + 1], float)
    ok = np.isfinite(y)
    zero = 0.0
    if ok.sum() >= 3:
        b, a = np.polyfit(t[ok], y[ok], 1)
        zero = a + b * t[0]
    up = float(fd["up"][j]) if vertical else 1.0
    bg = float(m.get("bg_v") or 0.0) if vertical and m.get("anchored") else 0.0
    return (y - zero) / up + bg * (t - t[0])


def mean_series(fd: dict, rows, i0: int, i1: int, vertical: bool = True) -> np.ndarray:
    """The mean of point_series over many points (NaN-aware), mm."""
    rows = list(rows)
    if not rows:
        return np.full(i1 - i0 + 1, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.vstack([point_series(fd, j, i0, i1, vertical) for j in rows]), axis=0)


def window_table(fd: dict, W: dict) -> dict:
    """A field_points table holding window W's figures (window_figures) in
    place of the record's, for the exports that read a table
    (client_kmz.point_parts, the viewer's CSV): the rates, the null and the
    change are the window's, and the series - the chart - is each point's
    own cell over the window, so a balloon's chart is the series its numbers
    come from. The whole record keeps the table as it is."""
    if W["kind"] == "report":
        return fd
    i0, i1 = W["i0"], W["i1"]
    m = dict(fd["meta"])
    t = np.asarray(m["t"], float)[i0:i1 + 1]
    if m.get("t_fit"):
        tf = np.asarray(m["t_fit"], float)[i0:i1 + 1]
        m["t_fit"] = [float(x) for x in tf - tf[0]]
    m.update({"t": [float(x) for x in t - t[0]], "dates": list(m["dates"][i0:i1 + 1]), "span": float(t[-1] - t[0]),
              "window": {"first": W["first"], "last": W["last"], "kind": W["kind"], "table": W["table"]}})
    up = np.asarray(fd["up"], float)
    out = dict(fd, meta=m, rate=W["vertical"], rel_v=W["vertical_rel"], los_rel=W["los"], snr=W["snr"],
               sd_v=np.where(np.isfinite(W["null"]), W["null"] / up, np.nan), change_los=W["change_los"],
               series_los=np.asarray(fd["cell_los"][:, i0:i1 + 1]),
               cell_los=np.asarray(fd["cell_los"][:, i0:i1 + 1]))
    return out


def slopes(t, Y) -> np.ndarray:
    from .field_points import slopes as _s
    return _s(np.asarray(t, float), np.asarray(Y, float), MIN_DATES)


# ------------------------------------------------------------------ the bundle
# the stored figures go as float64, so the whole record is the table's to
# the bit; the series are float32 in the table already
LIVE_ARRAYS = (("lat", "<f8"), ("lon", "<f8"), ("group", "u1"), ("up", "<f8"), ("los_rel", "<f8"),
               ("rel_v", "<f8"), ("rate", "<f8"), ("sd_v", "<f8"), ("snr", "<f8"), ("tc", "<f4"),
               ("unwrap", "<f4"), ("fast", "u1"), ("change_los", "<f8"), ("kcls", "i1"), ("far", "u1"),
               ("cell_los", "<f4"), ("far_cls", "i1"), ("far_los", "<f4"), ("frame_los", "<f4"),
               # a combined project's east-west (combine_viewer)
               ("east", "<f8"), ("east_change", "<f8"))


def report_state(cfg, rep: dict) -> dict:
    """What the page shows about the report: DRAFT or FINAL, the route, the
    lines every download carries (client_report.stamp_lines)."""
    from . import client_kmz as ck
    from . import client_report as cr
    from .config import route_tag
    try:
        final, blockers = ck.release_state(cfg, rep)
    except Exception:                                    # noqa: BLE001 - a draft, never a pass
        final, blockers = False, ["the release state could not be read"]
    try:
        stamp = cr.stamp_lines(cfg, rep)
    except Exception:                                    # noqa: BLE001
        stamp = [f"{rep.get('title') or ''} ground movement report"]
    try:                                                 # the code drawing the page's charts (OUT-01)
        from .deliverables import _software
        software = _software()
    except Exception:                                    # noqa: BLE001
        software = None
    return {"final": bool(final), "blockers": blockers, "route": route_tag(cfg), "stamp": stamp,
            "screening": bool(ck._screening(cfg, rep)), "software": software}


def bundle(cfg, fd: dict, rep: dict, stale: str | None = None) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    """(header, [(name, array)]) for the page. The header carries the
    table's meta, the report's state and its recorded dates; the arrays are
    the table's, typed for the browser."""
    m = fd["meta"]
    v2 = int(m.get("table_version") or 1) >= 2 and "cell_los" in fd
    arrays = []
    for name, dt in LIVE_ARRAYS:
        if name in fd:
            arrays.append((name, np.ascontiguousarray(np.asarray(fd[name]).astype(dt))))
    rd = rep.get("record_dates") or {}
    window_note = None
    if not v2:
        window_note = ("These points were built before date windows (19 Oct 2026): rebuild the client KMZ "
                       "(Products in the dashboard) to measure any window.")
    else:
        rc, cc = m.get("whole_record_check_mm_yr"), m.get("whole_record_change_check_mm")
        if rc is None or rc > 0.01 or cc is None or cc > 0.01:
            window_note = (f"The points' own series do not reproduce the delivered figures over the whole record "
                           f"(rate {rc} mm/yr, change {cc} mm): date windows are off. Rebuild the client KMZ.")
        elif m.get("far_partial_nan_cells"):
            window_note = None                         # analyse's identity holds for finite cells; said in meta
    meta = dict(m)
    meta.update({"stale": stale, "window_note": window_note, "windows": bool(v2 and window_note is None),
                 "bbox": list(cfg.bbox), "n": int(np.asarray(fd["lat"]).size),
                 "dropped": _dropped(rd), "available_images": rd.get("available"),
                 "recent_window": rep.get("recent_window"), **report_state(cfg, rep)})
    meta.pop("project_dir", None)
    header = {"format": FORMAT, "meta": meta, "arrays": []}
    return header, arrays


def _dropped(rd: dict) -> dict:
    """{YYYYMMDD: why} of the images the record left out
    (client_report.json record_dates.dropped: a dict, or a plain list)."""
    d = rd.get("dropped") or rd.get("dropped_dates") or {}
    if isinstance(d, dict):
        return {str(k): str(v) for k, v in d.items()}
    return {str(k): "" for k in d}


def encode(header: dict, arrays: list[tuple[str, np.ndarray]]) -> bytes:
    """4-byte little-endian header length, the header (JSON, its `arrays`
    listing name/dtype/shape/offset/nbytes), padding to 8, then each array's
    bytes at an 8-aligned offset from the start of the data."""
    specs, blobs, off = [], [], 0
    for name, a in arrays:
        b = a.tobytes()
        specs.append({"name": name, "dtype": a.dtype.str, "shape": list(a.shape), "offset": off, "nbytes": len(b)})
        pad = (-len(b)) % 8
        blobs.append(b + b"\0" * pad)
        off += len(b) + pad
    h = dict(header, arrays=specs)
    hb = json.dumps(clean(h), default=_json_default, allow_nan=False).encode()
    hb += b" " * ((-(4 + len(hb))) % 8)
    return len(hb).to_bytes(4, "little") + hb + b"".join(blobs)


def decode(data: bytes) -> tuple[dict, dict]:
    n = int.from_bytes(data[:4], "little")
    h = json.loads(data[4:4 + n])
    base = 4 + n
    arrs = {s["name"]: np.frombuffer(data, dtype=s["dtype"], count=int(np.prod(s["shape"])) if s["shape"] else 1,
                                     offset=base + s["offset"]).reshape(s["shape"]) for s in h["arrays"]}
    return h, arrs


def clean(o):
    """o with every non-finite float None, so the JSON is strict (a NaN in
    a page's JSON is not JSON)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
