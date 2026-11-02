"""Event products (F5, PLAN phase 2, 8 Oct 2026): what the ground did between
two dates, over each mapped panel, along a line and at survey epochs.

Why. The rate products (velocity, the client report's rates) fit one
straight line per cell over the whole record. A longwall panel moves the
ground by tens of centimetres over a few months and then stops; a straight
line through a five-year record spreads that over the record and reads a
few cm/yr. Nothing here fits a rate: every figure is a difference of the
delivered time series (deliverables.timeseries_file) between two of its
dates.

    cumulative_between(cfg, d0, d1) -> (map, attrs)
        the change map between two dates
    panel_max(cfg, polygons_geojson) -> [{panel, max_mm, date_of_max, lat, lon, ...}]
        per polygon, the largest robust subsidence and the date it is reached
    profile(cfg, line_geojson, dates) -> table
        changes along a polyline, in bins
    values_at(cfg, points, epochs, radius_m) -> table
        the value at any epoch at points (survey marks), interpolated

and write_event_products(cfg) writes all of it to deliverables/events/ when
the config sets event_products (deliverables.export_all calls it).

Config keys (all off by default; nothing else in the products moves):

    event_products   [] (off) | true (one window: the whole record) | a list
                     of windows, each "YYYY-MM-DD:YYYY-MM-DD", [start, end]
                     or {name: ..., start: ..., end: ...}. One change map
                     per window.
    event_polygons   GeoJSON (lon/lat) of panels or other areas. A polygon
                     whose properties carry its own dates (start/end, or
                     mined_start/mined_end, event_start/event_end) is
                     measured over them; any other polygon over each window
                     of event_products.
    event_lines      GeoJSON (lon/lat) of lines (survey lines, cross-lines):
                     a profile of every window of event_products along each.

Conventions, in every output:

  Units and sign. Millimetres along the line of sight (LOS), positive
  towards the satellite (MintPy's convention): subsidence reads negative.
  The "vertical_assumed" columns divide by the up component of the line of
  sight; they assume the ground moved vertically, which it does not on a
  longwall's flanks.

  Datum. The product's own: relative to its reference pixel. The reference
  date cancels in a change. In reference_mode gnss_absolute the time series
  stays on the reference pixel too (deliverables.gnss_absolute_reference);
  the validation harness double-differences against a reference station.
  A change on this datum carries whatever the datum and the wider ground
  did: on the Site A record the whole field reads about -140 mm over
  ten years against the reference pixel (the coherence-class drift the
  client figures take off by comparing with farmland). So each panel also
  gets the same statistics of the ground 1-2 km around it over the same
  dates, and the panel's figures less them, like for like:
  max_mm_relative_to_ring (the panel's 5th percentile less the ring's) and
  median_mm_relative_to_ring (median less median). Never a percentile less
  a median: on ground that moves like its surroundings that reads the
  spread of the cells (about -1.6 sigma), not subsidence.

  Dates. Every date in the outputs is YYYY-MM-DD (file names use the run's
  YYYYMMDD). A window snaps OUTWARD to the run's dates by default: its start to
  the last SAR date on or before it, its end to the first on or after it,
  so the change covers at least the window asked for, and the dates used
  are written beside every figure. snap="nearest" takes the nearest dates;
  snap="interpolate" interpolates linearly between SAR dates at most
  max_gap_days (24, protocol A's survey-epoch rule) apart, and gives no
  value across a longer gap. A window reaching beyond the record is
  measured from the record's first or to its last date and says so
  ("clamped").

  Reliable cell. Temporal coherence at or above the config's
  min_temporal_coherence (or `floor`) and a finite value at both dates;
  nothing is filled. A point or a line bin is the median of its reliable
  cells (radius 100 m for points; 100 m either side of a line in 50 m bins
  along it, protocol A) and needs at least 5 cells and 2 independent radar
  pixels (deliverables.radar_pixel_ids), the protocols' minimum.

  Aliasing. Where C-band cannot follow the motion (a steep, fast bowl) the
  time series is wrong by whole fringes and these figures with it,
  usually too small. Each panel and point carries the share of its cells
  that quality_aliasing_risk.tif (F2: 0 no risk, 1 risk, 2 likely aliased;
  a lower bound) and the older quality_too_fast_to_measure.tif flag, when
  those layers exist and are not older than the time series. Panels and
  maps also carry the F2 flag of their own interval (aliasing.risk_between,
  when the diagnostic is on for the config). A large flagged share means
  the change is a lower bound, not a measurement.

    python -m insar_pipeline.events <config.yaml> [--between D0 D1]...
        [--polygons F] [--lines F] [--points CSV [--radius M]] [--out DIR]

reads a finished project (it refuses while a run holds it) and writes the
same outputs, by default into <project>/deliverables/events/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

EVENTS_DIR = "events"
MAX_GAP_DAYS = 24.0           # protocol A epoch_matching.survey_epochs
POINT_RADIUS_M = 100.0        # protocols' sampling.radius_m (Sentinel-1)
LINE_BUFFER_M = 100.0         # protocol A sampling.lines
LINE_BIN_M = 50.0
MIN_CELLS = 5                 # protocols' sampling minimum (Sentinel-1)
MIN_INDEPENDENT = 2
PEAK_PERCENTILE = 5.0         # a robust minimum (protocol A appin_peak_ratio)
LOCATION_RADIUS_M = 100.0     # the deepest place: the mean change of reliable cells within this radius
RING_M = (1000.0, 2000.0)     # the ground around a panel: 1-2 km outside it, other polygons (+1 km) left out
SNAPS = ("outer", "nearest", "interpolate")
MAX_BLOCK_BYTES = 256e6       # read the cube in epoch blocks no larger than this
SIGN = ("mm along the line of sight, positive towards the satellite (MintPy): negative = away from the "
        "satellite, subsidence on a near-vertical signal")
ALIAS_LAYER = "quality_aliasing_risk.tif"
TOO_FAST_LAYER = "quality_too_fast_to_measure.tif"

_START_KEYS = ("start", "event_start", "mined_start", "date_start", "start_date", "from")
_END_KEYS = ("end", "event_end", "mined_end", "date_end", "end_date", "to")
_NAME_KEYS = ("name", "panel", "Name", "NAME", "id", "ID", "label", "line", "title", "objectid", "OBJECTID",
              "fid")
_EPOCH0 = date(1970, 1, 1)


# ------------------------------------------------------------------ dates
def _s(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _day(x) -> int:
    """Days since 1970-01-01 of a date given as YYYY-MM-DD or YYYYMMDD (text
    or a number), a date or datetime, a numpy datetime64 or a pandas
    Timestamp. Anything else (a day-first 05/02/2022 among them) is refused,
    never guessed."""
    if isinstance(x, np.datetime64):
        return int(x.astype("datetime64[D]").astype(np.int64))
    if hasattr(x, "to_datetime64"):                                  # pandas Timestamp
        return int(np.datetime64(x.to_datetime64(), "D").astype(np.int64))
    if isinstance(x, datetime):
        x = x.date()
    if isinstance(x, date):
        return (x - _EPOCH0).days
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        x = str(int(x))
    s = str(x).strip()
    if re.fullmatch(r"\d{8}", s):
        s = f"{s[:4]}-{s[4:6]}-{s[6:]}"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}([T ].*)?", s):
        raise ValueError(f"{x!r} is not a date: write YYYY-MM-DD (or YYYYMMDD)")
    try:
        return (date.fromisoformat(s[:10]) - _EPOCH0).days
    except ValueError:
        raise ValueError(f"{x!r} is not a date: write YYYY-MM-DD (or YYYYMMDD)") from None


def _iso(day) -> str:
    return (_EPOCH0 + timedelta(days=int(day))).isoformat()


def _ymd(day) -> str:
    return _iso(day).replace("-", "")


def parse_windows(spec) -> list[dict]:
    """event_products as [{name, start, end}] (ISO dates; None = the record's
    own first or last date). Empty for "off". Raises ValueError in words."""
    if spec is None or spec is False or (isinstance(spec, (str, list, tuple, dict)) and not spec):
        return []
    if spec is True:
        return [{"name": "record", "start": None, "end": None}]
    items = list(spec) if isinstance(spec, (list, tuple)) else [spec]
    out = []
    for it in items:
        name = None
        if isinstance(it, dict):
            name = it.get("name")
            a, b = it.get("start"), it.get("end")
            if a is None and b is None and "start" in it and "end" in it:
                # an already-read whole-record window ({name, start: None, end: None})
                out.append({"name": str(name or "record"), "start": None, "end": None})
                continue
            if a is None or b is None:
                raise ValueError(f"event window {it!r} needs both a start and an end")
        elif isinstance(it, (list, tuple)):
            if len(it) != 2:
                raise ValueError(f"event window {list(it)!r} needs exactly two dates [start, end]")
            a, b = it
        elif isinstance(it, str):
            s = it.strip()
            if s.lower() in ("record", "all", "full", "whole record"):
                out.append({"name": "record", "start": None, "end": None})
                continue
            parts = [p for p in re.split(r"\s*(?:\.\.|/|:|\s+to\s+)\s*", s) if p]
            if len(parts) != 2:
                raise ValueError(f"event window {it!r}: write it as YYYY-MM-DD:YYYY-MM-DD")
            a, b = parts
        elif it is True:
            out.append({"name": "record", "start": None, "end": None})
            continue
        else:
            raise ValueError(f"event window {it!r}: write it as YYYY-MM-DD:YYYY-MM-DD, [start, end] or "
                             "{name, start, end}")
        da, db = _day(a), _day(b)
        if db <= da:
            raise ValueError(f"event window {it!r} ends on or before it starts")
        out.append({"name": str(name) if name else f"{_iso(da)}_{_iso(db)}", "start": _iso(da), "end": _iso(db)})
    seen: dict = {}
    for w in out:
        n = w["name"]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            w["name"] = f"{n}#{seen[n]}"
    return out


def _snap(days: np.ndarray, d: int, side: str, snap: str, max_gap: float) -> dict:
    """Which SAR date(s) stand for date `d`: {i0, i1, w (the weight of i1),
    used (text), clamped, why (None, or why there is no value)}. `side` is
    'start' or 'end', for snap 'outer'."""
    n = days.size
    rec = {"i0": None, "i1": None, "w": 0.0, "clamped": False, "why": None}
    if snap == "outer":
        if side == "start":
            j = int(np.searchsorted(days, d, side="right")) - 1
            if j < 0:
                j, rec["clamped"] = 0, True
        else:
            j = int(np.searchsorted(days, d, side="left"))
            if j >= n:
                j, rec["clamped"] = n - 1, True
        rec.update(i0=j, i1=j)
    elif snap == "nearest":
        j = int(np.argmin(np.abs(days - d)))
        rec.update(i0=j, i1=j, clamped=bool(d < days[0] or d > days[-1]))
    elif snap == "interpolate":
        j = int(np.searchsorted(days, d, side="left"))
        if j < n and days[j] == d:
            rec.update(i0=j, i1=j)
        elif j == 0 or j >= n:
            rec["why"] = "outside the run's dates"
        elif days[j] - days[j - 1] > max_gap:
            rec["why"] = f"the SAR dates around it are more than {max_gap:g} days apart"
        else:
            rec.update(i0=j - 1, i1=j, w=float((d - days[j - 1]) / (days[j] - days[j - 1])))
    else:
        raise ValueError(f"snap = {snap!r}; expected one of {list(SNAPS)}")
    if rec["i0"] is not None:
        a, b = _iso(days[rec["i0"]]), _iso(days[rec["i1"]])
        rec["used"] = a if a == b else f"{a} .. {b} (interpolated)"
    else:
        rec["used"] = None
    return rec


# ------------------------------------------------------------------ the product
class _Product:
    """The delivered time series of one run, with its temporal coherence,
    read as it is used (the cube is never held whole)."""

    def __init__(self, cfg, floor=None):
        import h5py
        from . import deliverables as dlv
        self.cfg, self.dlv = cfg, dlv
        self.ts_path = Path(dlv.timeseries_file(cfg))
        with h5py.File(self.ts_path, "r") as f:
            self.attrs = {k: _s(v) for k, v in f.attrs.items()}
            self.dates = [_s(d) for d in f["date"][:]]
            ds = f["timeseries"]
            self.shape = tuple(ds.shape[1:])
            self.depth = int(ds.chunks[0]) if ds.chunks else 1
        if not self.dates:
            raise ValueError(f"{self.ts_path} holds no dates")
        self.days = np.array([_day(d) for d in self.dates], dtype=np.int64)
        if np.any(np.diff(self.days) <= 0):
            raise ValueError(f"the dates of {self.ts_path.name} are not in increasing order")
        unit = str(self.attrs.get("UNIT", "m")).strip().lower()
        self.scale = {"m": 1000.0, "mm": 1.0, "cm": 10.0}.get(unit, 1000.0)
        tc_path = dlv._product(cfg, "temporalCoherence.h5")
        if not Path(tc_path).exists():
            raise FileNotFoundError(f"No temporal coherence ({tc_path}): the reliability rule needs it")
        tc, _ = dlv._read(tc_path, "temporalCoherence")
        if tuple(tc.shape) != self.shape:
            raise ValueError(f"the temporal coherence is {tuple(tc.shape)}, the time series {self.shape}")
        self.tcoh = np.asarray(tc, dtype=float)
        self.floor = float(getattr(cfg, "min_temporal_coherence", 0.4) if floor is None else floor)
        self.reliable = np.isfinite(self.tcoh) & (self.tcoh >= self.floor)
        self._geom: dict = {}
        self._flags = None
        self.flags_note: list = []
        self._interval_risk: dict = {}

    # ---- layers
    def geometry(self, name):
        if name not in self._geom:
            arr = self.dlv.geometry_field(self.cfg, name)
            self._geom[name] = (None if arr is None or tuple(arr.shape) != self.shape
                                else np.asarray(arr, dtype=float))
        return self._geom[name]

    def flags(self) -> dict:
        """{layer name: array} of the flag layers on this grid (missing ones
        left out): the aliasing risk (F2) and the old too-fast flag. A layer
        older than the time series is from an earlier inversion and is not
        used (flags_note says so)."""
        if self._flags is None:
            self._flags = {}
            D = Path(self.cfg.project_dir) / "deliverables"
            ts_mtime = self.ts_path.stat().st_mtime
            for name in (ALIAS_LAYER, TOO_FAST_LAYER):
                p = D / name
                if not p.exists():
                    continue
                if p.stat().st_mtime < ts_mtime:
                    msg = f"{name} is older than {self.ts_path.name} (an earlier inversion's): not used"
                    log.warning("Event products: %s", msg)
                    self.flags_note.append(msg)
                    continue
                try:
                    import rasterio
                    with rasterio.open(p) as src:
                        a = src.read(1).astype(float)
                        nd = src.nodata
                    if tuple(a.shape) != self.shape:
                        log.warning("%s is %s, the time series %s: not used", name, a.shape, self.shape)
                        self.flags_note.append(f"{name} is {a.shape}, the time series {self.shape}: not used")
                        continue
                    if nd is not None:
                        a[a == nd] = np.nan
                    a[(a < 0) | (a > 2)] = np.nan
                    self._flags[name] = a
                except Exception as exc:                          # noqa: BLE001
                    log.warning("%s not read (%s)", name, exc)
                    self.flags_note.append(f"{name} not read ({type(exc).__name__}: {exc})")
        return self._flags

    def interval_risk(self, i0: int, i1: int):
        """The F2 aliasing flag for the one interval between epochs i0 and
        i1 (aliasing.risk_between: the highest level over the exported pairs
        spanning part of it, protocol A's flag definition), as floats with
        NaN outside the radar crop; None when the diagnostic is off for this
        config or there is no pre-unwrap record. The whole-record layer
        flags more than any one interval does."""
        key = (int(i0), int(i1))
        if key not in self._interval_risk:
            a = None
            try:
                from . import aliasing
                fn = getattr(aliasing, "risk_between", None)
                on = getattr(aliasing, "enabled", None)
                if fn is not None and (on is None or on(self.cfg)):
                    r = fn(self.cfg, self.dates[key[0]], self.dates[key[1]])
                    if r is not None and tuple(np.shape(r)) == self.shape:
                        a = np.asarray(r, dtype=float)
                        a[(a < 0) | (a > 2)] = np.nan
                    elif r is not None:
                        self.flags_note.append(f"the interval aliasing flag is {np.shape(r)}, the time series "
                                               f"{self.shape}: not used")
            except Exception as exc:                              # noqa: BLE001 - said in the record
                msg = f"the interval aliasing flag was not read ({type(exc).__name__}: {exc})"
                log.warning("Event products: %s", msg)
                self.flags_note.append(msg)
            self._interval_risk[key] = a
        return self._interval_risk[key]

    def datum(self) -> dict:
        a = self.attrs
        try:
            text = self.dlv.reference_label(self.cfg)
        except Exception:                                         # noqa: BLE001
            text = ""
        return {"relative_to": "the product's reference pixel" + (f": {text}" if text else ""),
                "reference": {k: a.get(k) for k in ("REF_LAT", "REF_LON", "REF_Y", "REF_X", "REF_DATE")},
                "note": "the reference date cancels in a change; the reference pixel does not"}

    # ---- reading
    def read(self, idx, window=None) -> np.ndarray:
        """(len(idx), rows, cols) in mm for the epochs `idx`, in the order
        given, over window = (r0, r1, c0, c1) or the whole grid."""
        import h5py
        idx = [int(i) for i in idx]
        r0, r1, c0, c1 = window or (0, self.shape[0], 0, self.shape[1])
        uniq = sorted(set(idx))
        with h5py.File(self.ts_path, "r") as f:
            blk = np.asarray(f["timeseries"][uniq, r0:r1, c0:c1], dtype=float) * self.scale
        pos = {v: k for k, v in enumerate(uniq)}
        return blk[[pos[i] for i in idx]]

    def iter_range(self, i0: int, i1: int, window):
        """(i, 2-D mm) for epochs i0..i1 over `window`, read in chunk-aligned
        blocks no larger than MAX_BLOCK_BYTES."""
        import h5py
        r0, r1, c0, c1 = window
        per = max(1, (r1 - r0) * (c1 - c0) * 8)              # held as float64
        step = max(1, int(MAX_BLOCK_BYTES // per))
        if step >= self.depth:
            step = step // self.depth * self.depth
        with h5py.File(self.ts_path, "r") as f:
            d = f["timeseries"]
            a = i0
            while a <= i1:
                b = min(i1 + 1, a + step)
                blk = np.asarray(d[a:b, r0:r1, c0:c1], dtype=float) * self.scale
                for k in range(blk.shape[0]):
                    yield a + k, blk[k]
                a = b

    def cells(self, rows, cols) -> np.ndarray:
        """(n_dates, k) in mm for k cells."""
        if not np.size(rows):
            return np.empty((len(self.dates), 0))
        cube = self.dlv._Cube(self.ts_path, "timeseries")
        return cube[:, np.asarray(rows, np.intp), np.asarray(cols, np.intp)].astype(float) * self.scale

    def iso(self, i: int) -> str:
        """The run's date i as YYYY-MM-DD (every date in the outputs is ISO)."""
        return _iso(self.days[int(i)])

    def value_at(self, s: dict, window=None) -> tuple[np.ndarray | None, list]:
        """The 2-D value (mm) at a snapped date over `window`, and the epoch
        indices read."""
        if s["i0"] is None:
            return None, []
        if s["i0"] == s["i1"] or s["w"] == 0.0:
            return self.read([s["i0"]], window)[0], [s["i0"]]
        a, b = self.read([s["i0"], s["i1"]], window)
        return (1.0 - s["w"]) * a + s["w"] * b, [s["i0"], s["i1"]]

    def window_epochs(self, start, end, snap="outer", max_gap=MAX_GAP_DAYS) -> tuple[dict, dict]:
        """The snapped start and end of a window (None = the record's first
        or last date)."""
        d0 = int(self.days[0]) if start is None else _day(start)
        d1 = int(self.days[-1]) if end is None else _day(end)
        if d1 <= d0:
            raise ValueError(f"the window {_iso(d0)} .. {_iso(d1)} ends on or before it starts")
        s0 = _snap(self.days, d0, "start", snap, max_gap)
        s1 = _snap(self.days, d1, "end", snap, max_gap)
        for s, d, what in ((s0, d0, "start"), (s1, d1, "end")):
            if s["why"]:
                raise ValueError(f"the window's {what} {_iso(d)} has no value: {s['why']}")
        # where each end sits on the run's dates (an index, fractional when interpolated)
        p0 = s0["i0"] + s0["w"] * (s0["i1"] - s0["i0"])
        p1 = s1["i0"] + s1["w"] * (s1["i1"] - s1["i0"])
        if p1 <= p0:
            raise ValueError(f"the window {_iso(d0)} .. {_iso(d1)} holds no interval between the run's dates "
                             f"({self.dates[0]} .. {self.dates[-1]})")
        s0["requested"], s1["requested"] = _iso(d0), _iso(d1)
        return s0, s1


def _product(cfg, floor=None, prod=None) -> _Product:
    if prod is not None and (floor is None or float(floor) == prod.floor):
        return prod
    return _Product(cfg, floor=floor)


# ------------------------------------------------------------------ geometry
def _geojson_epsg(data: dict) -> int | None:
    """The legacy GeoJSON 'crs' member's EPSG (None = lon/lat, RFC 7946)."""
    crs = data.get("crs") if isinstance(data, dict) else None
    if not isinstance(crs, dict):
        return None
    name = str((crs.get("properties") or {}).get("name", ""))
    if "CRS84" in name.upper():
        return None
    m = re.search(r"EPSG:+(\d+)", name, re.I)
    if not m:
        return None
    code = int(m.group(1))
    return None if code == 4326 else code


def _read_geojson(src) -> dict:
    if isinstance(src, dict):
        return src
    p = Path(str(src)).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    try:
        return json.loads(p.read_text())
    except ValueError as exc:
        raise ValueError(f"{p.name} is not valid GeoJSON ({exc})") from None


def load_features(src, kind: str) -> list[dict]:
    """The polygon or line features of a GeoJSON (a path or a dict), in
    lon/lat: [{name, geom (shapely), props, index}]. kind 'polygon' takes
    Polygon/MultiPolygon, 'line' LineString/MultiLineString. Names come from
    the properties (name, panel, id, ...), else '<kind>_<n>', made unique."""
    from shapely.geometry import shape
    from shapely.ops import transform
    data = _read_geojson(src)
    want = {"polygon": ("Polygon", "MultiPolygon"), "line": ("LineString", "MultiLineString")}[kind]
    if data.get("type") == "FeatureCollection":
        feats = data.get("features") or []
    elif data.get("type") == "Feature":
        feats = [data]
    elif data.get("type") in want + ("Point", "MultiPoint", "GeometryCollection"):
        feats = [{"type": "Feature", "geometry": data, "properties": {}}]
    else:
        raise ValueError(f"not a GeoJSON FeatureCollection, Feature or geometry (type {data.get('type')!r})")
    epsg = _geojson_epsg(data)
    to_ll = None
    if epsg:
        from pyproj import Transformer
        to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform
    out, names = [], {}
    for k, ft in enumerate(feats):
        g = (ft or {}).get("geometry")
        if not g or g.get("type") not in want:
            continue
        geom = shape(g)
        if to_ll is not None:
            geom = transform(to_ll, geom)
        if geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        if max(abs(minx), abs(maxx)) > 180 or max(abs(miny), abs(maxy)) > 90:
            raise ValueError("the coordinates are not longitude/latitude: GeoJSON is lon/lat (RFC 7946); a "
                             "projected file must name its CRS")
        if kind == "polygon" and not geom.is_valid:
            from shapely.validation import make_valid
            geom = make_valid(geom)
        props = dict(ft.get("properties") or {})
        name = next((str(props[c]).strip() for c in _NAME_KEYS if props.get(c) not in (None, "")), "")
        name = name or f"{kind}_{k + 1}"
        names[name] = names.get(name, 0) + 1
        if names[name] > 1:
            name = f"{name}#{names[name]}"
        out.append({"name": name, "geom": geom, "props": props, "index": k})
    return out


def _own_window(props: dict, name: str) -> dict | None:
    """A feature's own dates (start/end, mined_start/mined_end, ...), or None."""
    a = next((props[k] for k in _START_KEYS if props.get(k) not in (None, "")), None)
    b = next((props[k] for k in _END_KEYS if props.get(k) not in (None, "")), None)
    if a is None and b is None:
        return None
    if a is None or b is None:
        raise ValueError(f"{name}: its properties give only one of its dates (start and end are both needed)")
    try:
        da, db = _day(a), _day(b)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from None
    if db <= da:
        raise ValueError(f"{name}: its window {_iso(da)} .. {_iso(db)} ends on or before it starts")
    return {"name": f"{_iso(da)}_{_iso(db)}", "start": _iso(da), "end": _iso(db)}


def _grid_epsg(attrs: dict, shape, hint_lonlat=None) -> int | None:
    """None for a lat/lon grid, else the EPSG of the projected grid."""
    from . import deliverables as dlv
    if dlv._is_geographic(attrs):
        return None
    from .stack_qc import product_epsg
    return int(product_epsg(attrs, lonlat=hint_lonlat, shape=shape))


def _to_grid(geom_ll, epsg):
    if epsg is None:
        return geom_ll
    from pyproj import Transformer
    from shapely.ops import transform
    return transform(Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform, geom_ll)


def _window(attrs: dict, shape, geom_grid) -> tuple | None:
    """(r0, r1, c0, c1) of the grid cells that can lie inside geom_grid
    (grid coordinates), or None off the grid."""
    x0, dx, y0, dy = (float(attrs[k]) for k in ("X_FIRST", "X_STEP", "Y_FIRST", "Y_STEP"))
    minx, miny, maxx, maxy = geom_grid.bounds
    ca, cb = sorted(((minx - x0) / dx, (maxx - x0) / dx))
    ra, rb = sorted(((miny - y0) / dy, (maxy - y0) / dy))
    r0, r1 = max(0, int(math.floor(ra)) - 1), min(shape[0], int(math.ceil(rb)) + 1)
    c0, c1 = max(0, int(math.floor(ca)) - 1), min(shape[1], int(math.ceil(cb)) + 1)
    if r0 >= r1 or c0 >= c1:
        return None
    return r0, r1, c0, c1


def _centres(attrs: dict, window) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre grid coordinates (X, Y) over the window (2-D)."""
    r0, r1, c0, c1 = window
    rr, cc = np.mgrid[r0:r1, c0:c1]
    return (float(attrs["X_FIRST"]) + (cc + 0.5) * float(attrs["X_STEP"]),
            float(attrs["Y_FIRST"]) + (rr + 0.5) * float(attrs["Y_STEP"]))


def _grid_to_lonlat(X, Y, epsg):
    if epsg is None:
        return np.asarray(X, float), np.asarray(Y, float)
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(X, Y)
    return np.asarray(lon, float), np.asarray(lat, float)


def _polygon_cells(prod: _Product, geom_ll) -> tuple | None:
    """(window, inside mask over the window) of the cells whose centres lie
    in the polygon, or None when none do."""
    import shapely
    c = geom_ll.representative_point()
    epsg = _grid_epsg(prod.attrs, prod.shape, (c.x, c.y))
    g = _to_grid(geom_ll, epsg)
    win = _window(prod.attrs, prod.shape, g)
    if win is None:
        return None
    X, Y = _centres(prod.attrs, win)
    inside = shapely.contains_xy(g, X, Y)
    if not inside.any():
        return None
    return win, inside


def _local_metric(lon0: float, lat0: float):
    """(forward, inverse) lon/lat <-> metres on an azimuthal equidistant
    plane centred on (lon0, lat0): distances within a few km to <0.1%."""
    from pyproj import Transformer
    aeqd = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    return (Transformer.from_crs("EPSG:4326", aeqd, always_xy=True).transform,
            Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True).transform)


def _pixel_metres(attrs: dict, shape) -> tuple[float, float]:
    from .significance import pixel_metres
    return pixel_metres(dict(attrs, LENGTH=shape[0]))


def _disk(radius_m: float, py: float, px: float) -> np.ndarray:
    hy, hx = max(1, int(radius_m // py)), max(1, int(radius_m // px))
    yy, xx = np.mgrid[-hy:hy + 1, -hx:hx + 1]
    return (np.hypot(yy * py, xx * px) <= radius_m).astype(float)


def _flag_shares(prod: _Product, rows, cols) -> dict:
    """Share of the cells flagged by each flag layer present."""
    out = {}
    for name, a in prod.flags().items():
        v = a[rows, cols] if np.size(rows) else np.array([])
        v = v[np.isfinite(v)]
        if not v.size:
            out[name] = {"cells": 0}
            continue
        out[name] = {"cells": int(v.size), "share_flagged": round(float((v >= 1).mean()), 4)}
        if name == ALIAS_LAYER:
            out[name]["share_likely_aliased"] = round(float((v >= 2).mean()), 4)
    return out


def _interval_share(prod: _Product, i0: int, i1: int, rows, cols) -> dict | None:
    """Share of the cells flagged by the aliasing flag of the interval i0..i1
    (_Product.interval_risk), or None when there is none."""
    a = prod.interval_risk(i0, i1)
    if a is None:
        return None
    v = a[rows, cols] if np.size(rows) else np.array([])
    v = v[np.isfinite(v)]
    rec = {"interval": f"{prod.iso(i0)} .. {prod.iso(i1)}", "cells": int(v.size),
           "what": "aliasing.risk_between: the highest F2 level over the exported pairs spanning part of the interval"}
    if v.size:
        rec.update(share_flagged=round(float((v >= 1).mean()), 4), share_likely_aliased=round(float((v >= 2).mean()), 4))
    return rec


# ------------------------------------------------------------------ cumulative maps
def cumulative_between(cfg, d0, d1, *, snap: str = "outer", floor=None, max_gap_days: float = MAX_GAP_DAYS,
                       _prod=None) -> tuple[np.ndarray, dict]:
    """
    The LOS change (mm, float32) of every cell between two dates, NaN where
    the cell is not reliable (temporal coherence below the floor, or no
    value at either date). The dates snap as the module docstring says
    (default: outward to the run's dates). d0 or d1 None = the record's
    first or last date.

    Returns (map, attrs): the product's attributes with START_DATE/END_DATE
    set to the SAR dates used, plus EVENT_* keys (the dates asked for and
    used, the snap rule, whether the window was clamped to the record, the
    floor, the reliable-cell count, the datum and the sign).
    """
    prod = _product(cfg, floor, _prod)
    s0, s1 = prod.window_epochs(d0, d1, snap=snap, max_gap=max_gap_days)
    v0, _ = prod.value_at(s0)
    v1, _ = prod.value_at(s1)
    ch = v1 - v0
    ok = prod.reliable & np.isfinite(ch)
    out = np.where(ok, ch, np.nan).astype(np.float32)
    e0 = prod.iso(s0["i0"]) if not s0["w"] else s0["used"]
    e1 = prod.iso(s1["i1"]) if not s1["w"] else s1["used"]
    attrs = dict(prod.attrs)
    attrs.update({
        "START_DATE": prod.dates[s0["i0"]], "END_DATE": prod.dates[s1["i1"]],
        "EVENT_REQUESTED_START": s0["requested"], "EVENT_REQUESTED_END": s1["requested"],
        "EVENT_EPOCH_START": e0, "EVENT_EPOCH_END": e1, "EVENT_SNAP": snap,
        "EVENT_CLAMPED": "yes" if (s0["clamped"] or s1["clamped"]) else "no",
        "EVENT_COHERENCE_FLOOR": f"{prod.floor:g}", "EVENT_RELIABLE_CELLS": str(int(ok.sum())),
        "EVENT_UNITS": "mm", "EVENT_SIGN": SIGN,
        "EVENT_DATUM": prod.datum()["relative_to"],
        "EVENT_MEASURED": "the difference of the delivered time series between the two dates; no rate is assumed",
        "UNIT": "mm",
    })
    if s0["clamped"] or s1["clamped"]:
        log.warning("Event window %s .. %s reaches beyond the record (%s .. %s): measured over %s .. %s",
                    s0["requested"], s1["requested"], prod.iso(0), prod.iso(-1), e0, e1)
    return out, attrs


# ------------------------------------------------------------------ panels
def _ring_area(geom_ll, union_ll, ring_m):
    """The ground ring_m = (inner, outer) metres outside a polygon, less
    every polygon of its file grown by `inner` (other panels are not
    ground at rest), in lon/lat; None when nothing is left."""
    from shapely.ops import transform
    inner, outer = (float(v) for v in ring_m)
    c = geom_ll.centroid
    fwd, inv = _local_metric(c.x, c.y)
    g = transform(fwd, geom_ll)
    ring = g.buffer(outer).difference(transform(fwd, union_ll).buffer(inner))
    return None if ring.is_empty else transform(inv, ring)


def _ring_change(prod: _Product, ring_cells, i0: int, i1: int, percentile: float = PEAK_PERCENTILE) -> dict:
    """The change (mm) of the reliable ring cells between epochs i0 and i1:
    their median and their `percentile`th percentile (robust_mm, the
    statistic of the panel's max_mm, so the two compare like for like)."""
    if ring_cells is None:
        return {"cells": 0}
    win, inside = ring_cells
    a, b = prod.read([i0, i1], win)
    ch = b - a
    ok = inside & prod.reliable[win[0]:win[1], win[2]:win[3]] & np.isfinite(ch)
    n = int(ok.sum())
    if n < MIN_CELLS:
        return {"cells": n, "median_mm": None, "robust_mm": None}
    return {"cells": n, "median_mm": round(float(np.median(ch[ok])), 2),
            "robust_mm": round(float(np.percentile(ch[ok], percentile)), 2)}


def _panel_one(prod: _Product, feat: dict, cells, win_spec: dict, source: str, percentile: float,
               min_cells: int, location_radius_m: float, ring_cells=None, ring_m=None) -> dict:
    name = feat["name"]
    row = {"panel": name, "max_mm": None, "date_of_max": None, "lat": None, "lon": None,
           "window": {"name": win_spec.get("name"), "source": source,
                      "requested_start": win_spec.get("start"), "requested_end": win_spec.get("end")},
           "percentile": percentile, "units": "mm LOS", "sign": SIGN}
    if cells is None:
        row["reason"] = "no cell of the product's grid has its centre inside this polygon"
        return row
    win, inside = cells
    try:
        s0, s1 = prod.window_epochs(win_spec.get("start"), win_spec.get("end"), snap="outer")
    except ValueError as exc:
        row["reason"] = str(exc)
        return row
    i0, i1 = s0["i0"], s1["i1"]
    row["window"].update(epoch_start=prod.iso(i0), epoch_end=prod.iso(i1),
                         clamped=bool(s0["clamped"] or s1["clamped"]))
    r0, r1, c0, c1 = win
    rel = inside & prod.reliable[r0:r1, c0:c1]
    row["n_cells_in_polygon"] = int(inside.sum())
    base = None
    series, at = [], []
    deepest = (np.inf, None, None)
    for i, ep in prod.iter_range(i0, i1, win):
        if base is None:
            base = ep
            ok_base = rel & np.isfinite(base)
            row["n_reliable_cells"] = int(ok_base.sum())
            row["coverage_share"] = round(float(ok_base.sum() / max(inside.sum(), 1)), 4)
            continue
        ch = ep - base
        ok = ok_base & np.isfinite(ch)
        n = int(ok.sum())
        rec = {"date": prod.iso(i), "n_cells": n, "robust_mm": None, "median_mm": None}
        if n >= min_cells:
            v = ch[ok]
            rec["robust_mm"] = round(float(np.percentile(v, percentile)), 2)
            rec["median_mm"] = round(float(np.median(v)), 2)
            j = int(np.argmin(np.where(ok, ch, np.inf)))
            if ch.flat[j] < deepest[0]:
                deepest = (float(ch.flat[j]), prod.iso(i), np.unravel_index(j, ch.shape))
        series.append(rec)
        at.append(i)
    row["series"] = series
    good = [k for k, r in enumerate(series) if r["robust_mm"] is not None]
    if not good:
        row["reason"] = (f"no date in {prod.iso(i0)} .. {prod.iso(i1)} with {min_cells} or more reliable cells "
                         f"inside the polygon (temporal coherence >= {prod.floor:g})")
        return row
    kmax = min(good, key=lambda k: series[k]["robust_mm"])
    mx = series[kmax]["robust_mm"]
    imax = at[kmax]
    row.update(max_mm=mx, date_of_max=prod.iso(imax), median_mm_at_max=series[kmax]["median_mm"],
               n_cells_at_max=series[kmax]["n_cells"])
    # the deepest value on the window's last date: the ground may have gone on sinking after it
    row["max_at_window_end"] = bool(kmax == len(series) - 1 and mx < 0)
    if mx < 0:
        half = next((series[k]["date"] for k in good if series[k]["robust_mm"] <= mx / 2.0), None)
        row["date_half"] = half
    if deepest[2] is not None:
        rr, cc = deepest[2]
        lon, lat = prod.dlv._rc_to_lonlat(prod.attrs, r0 + rr, c0 + cc)
        row["deepest_cell"] = {"mm": round(deepest[0], 2), "date": deepest[1],
                               "lat": round(float(lat), 6), "lon": round(float(lon), 6),
                               "note": "one cell: noise-dominated, the robust figure is max_mm"}
    # where: the reliable cell whose neighbourhood (location_radius_m) sank most at the date of the maximum
    ep = prod.read([i0, imax], win)
    ch = ep[1] - ep[0]
    ok = ok_base & np.isfinite(ch)
    from scipy.ndimage import correlate
    py, px = _pixel_metres(prod.attrs, prod.shape)
    k = _disk(location_radius_m, py, px)
    s = correlate(np.where(ok, ch, 0.0), k, mode="constant")
    n = correlate(ok.astype(float), k, mode="constant")
    need = min(min_cells, int(k.sum()))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok & (n >= need), s / np.maximum(n, 1), np.nan)
    if np.isfinite(mean).any():
        j = int(np.nanargmin(mean))
        how = f"the reliable cell whose reliable neighbours within {location_radius_m:g} m sank most (their mean)"
    else:
        j = int(np.nanargmin(np.where(ok, ch, np.nan)))
        mean = np.where(ok, ch, np.nan)
        how = "the deepest reliable cell (too few neighbours for a neighbourhood mean)"
    rr, cc = np.unravel_index(j, ch.shape)
    lon, lat = prod.dlv._rc_to_lonlat(prod.attrs, r0 + rr, c0 + cc)
    row.update(lat=round(float(lat), 6), lon=round(float(lon), 6), location_mean_mm=round(float(mean.flat[j]), 2),
               location_how=how)
    lu = prod.geometry("los_up")
    R, C = np.nonzero(ok)
    R, C = R + r0, C + c0
    if lu is not None and R.size:
        up = float(np.nanmedian(lu[R, C]))
        if np.isfinite(up) and up > 0.1:
            row["los_up"] = round(up, 4)
            row["vertical_assumed_mm"] = round(mx / up, 2)
    if R.size:
        ids, _ = prod.dlv.radar_pixel_ids(prod.cfg, R, C, prod.shape)
        row["n_independent_at_max"] = int(np.unique(ids[ids >= 0]).size)
    fl = _flag_shares(prod, R, C)
    iv = _interval_share(prod, i0, imax, R, C)
    if iv is not None:
        fl["aliasing_risk_interval"] = iv
    if fl:
        row["flags"] = fl
    if ring_m:
        rg = _ring_change(prod, ring_cells, i0, imax, percentile)
        row["ring"] = {"from_m": ring_m[0], "to_m": ring_m[1], "cells": rg["cells"], "median_mm": rg.get("median_mm"),
                       "robust_mm": rg.get("robust_mm"), "percentile": percentile,
                       "what": ("the change of the reliable ground this far outside the polygon (other polygons of "
                                "the file, grown by the inner distance, left out) over the same dates, as the median "
                                f"and the {percentile:g}th percentile (robust_mm, max_mm's statistic): what the "
                                "product's datum and the ground around the panel did")}
        if rg.get("robust_mm") is not None:
            # like for like: a percentile less the same percentile (a percentile less a median would read the
            # cells' spread, about -1.6 sigma, as subsidence on ground that moves like its surroundings)
            row["max_mm_relative_to_ring"] = round(mx - rg["robust_mm"], 2)
        if rg.get("median_mm") is not None and series[kmax]["median_mm"] is not None:
            row["median_mm_relative_to_ring"] = round(series[kmax]["median_mm"] - rg["median_mm"], 2)
    row["note"] = (f"max_mm = the {percentile:g}th percentile of the reliable cells' change since "
                   f"{prod.iso(i0)} (a robust minimum), at its most negative date, on the product's datum; "
                   "max_mm_relative_to_ring = max_mm less the same percentile of the ring around it over the same "
                   "dates, median_mm_relative_to_ring the medians' difference (like for like: about 0 on ground "
                   "that moves like its surroundings); a lower bound where the motion aliases (flags)")
    return row


def panel_max(cfg, polygons_geojson, *, windows=None, floor=None, percentile: float = PEAK_PERCENTILE,
              min_cells: int = MIN_CELLS, location_radius_m: float = LOCATION_RADIUS_M, ring_m=RING_M,
              _prod=None) -> list[dict]:
    """
    For each polygon: the largest robust subsidence over its window and the
    date it is reached.

      window      the polygon's own dates (properties start/end, or
                  mined_start/mined_end, event_start/event_end) when it has
                  them; otherwise each window in `windows` (parse_windows
                  forms; None = the whole record). Snapped outward: from
                  the last SAR date on or before the start to the first on
                  or after the end.
      max_mm      at each date after the window's first, the `percentile`th
                  (5th) percentile of the change since that first date over
                  the polygon's reliable cells (a robust minimum; 5 or more
                  cells); max_mm is the most negative of them (LOS mm,
                  negative = away from the satellite)
      date_of_max that date (ISO); date_half, the first date at which half
                  of it was reached; max_at_window_end when it is the
                  window's last date (the ground may have gone on sinking
                  after it: a longer window shows whether it did)
      lat, lon    where: the reliable cell whose reliable neighbours within
                  location_radius_m (100 m) sank most at date_of_max
      ring        the change, over the same dates, of the reliable ground
                  ring_m (1-2 km) outside the polygon, every polygon of the
                  file grown by 1 km left out: its median and its
                  `percentile`th percentile (robust_mm). Like for like:
                  max_mm_relative_to_ring = max_mm minus the ring's
                  robust_mm, median_mm_relative_to_ring = median_mm_at_max
                  minus the ring's median; both about 0 on ground that moves
                  like its surroundings (a percentile less a median would
                  read the cells' spread as subsidence). max_mm is on the
                  product's datum (the reference pixel), which carries
                  whatever the datum and the wider ground did (on farmland,
                  the record's coherence-class drift); the ring figures take
                  that off. ring_m None: no ring.

    Each row also carries the window used, the cell counts and coverage,
    the change over time ('series'), the deepest single cell, the vertical
    figure under the vertical-motion assumption, and the share of the
    panel's cells under each flag layer (aliasing: a lower bound there).
    Returns [{panel, max_mm, date_of_max, lat, lon, ...}], one per polygon
    and window, in file order; max_mm None with a 'reason' when there is no
    value.
    """
    prod = _product(cfg, floor, _prod)
    feats = load_features(polygons_geojson, "polygon")
    if not feats:
        raise ValueError(f"{polygons_geojson}: no Polygon or MultiPolygon feature")
    wins = parse_windows(windows) if windows is not None else [{"name": "record", "start": None, "end": None}]
    union = None
    if ring_m:
        from shapely.ops import unary_union
        union = unary_union([f["geom"] for f in feats])
    rows = []
    for ft in feats:
        own = _own_window(ft["props"], ft["name"])
        cells = _polygon_cells(prod, ft["geom"])
        ring_cells = None
        if ring_m:
            area = _ring_area(ft["geom"], union, ring_m)
            ring_cells = _polygon_cells(prod, area) if area is not None else None
        for w, src in ([(own, "the polygon's own dates")] if own else
                       [(w, "the whole record" if w["start"] is None else "event window") for w in wins]):
            rows.append(_panel_one(prod, ft, cells, w, src, percentile, min_cells, location_radius_m,
                                   ring_cells=ring_cells, ring_m=ring_m))
    return rows


# ------------------------------------------------------------------ profiles
def profile(cfg, line_geojson, dates, *, buffer_m: float = LINE_BUFFER_M, bin_m: float = LINE_BIN_M, floor=None,
            snap: str = "outer", min_cells: int = MIN_CELLS, min_independent: int = MIN_INDEPENDENT,
            max_gap_days: float = MAX_GAP_DAYS, _prod=None):
    """
    The LOS change along each line of a GeoJSON, in bins.

      dates   the first is the baseline, every later one a date to measure
              the change to; a single date is measured from the record's
              first date. Snapped as the module docstring says (outer: the
              baseline back, the others forward).
      cells   reliable cells whose centres lie within buffer_m (100 m) of
              the line and alongside it (not beyond its ends), binned by
              their distance along it (bin_m, 50 m):
              the median of a bin with min_cells (5) cells and
              min_independent (2) radar pixels, else no value.

    Returns a pandas DataFrame, one row per line, bin and date: line, bin,
    distance_m (the bin's middle along the line; the last bin can be
    shorter than bin_m and its middle is that of its own length), lat, lon (that point on
    the line), baseline (asked), baseline_epoch (used), date (asked), epoch
    (used), change_mm (LOS, median), n_cells, n_independent, spread_mm
    (1.4826 x MAD), p05_mm, vertical_assumed_mm, status ('ok' or why not).
    df.attrs holds the datum, sign and settings.
    """
    import pandas as pd
    import shapely
    from shapely.ops import linemerge, transform
    prod = _product(cfg, floor, _prod)
    feats = load_features(line_geojson, "line")
    if not feats:
        raise ValueError(f"{line_geojson}: no LineString or MultiLineString feature")
    dates = [dates] if isinstance(dates, (str, date, int, np.datetime64)) else list(dates or [])
    if not dates:
        raise ValueError("profile needs at least one date")
    base_req = None if len(dates) == 1 else dates[0]
    targets = dates if len(dates) == 1 else dates[1:]
    d_base = int(prod.days[0]) if base_req is None else _day(base_req)
    sb = _snap(prod.days, d_base, "start", snap, max_gap_days)
    if sb["why"]:
        raise ValueError(f"the baseline {_iso(d_base)} has no value: {sb['why']}")
    st = []
    for d in targets:
        dd = _day(d)
        if dd <= d_base:
            raise ValueError(f"profile date {_iso(dd)} is not after the baseline {_iso(d_base)}")
        st.append((dd, _snap(prod.days, dd, "end", snap, max_gap_days)))
    lu = prod.geometry("los_up")
    rows = []
    for ft in feats:
        geom = ft["geom"]
        if geom.geom_type == "MultiLineString":
            geom = linemerge(geom)
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for pk, part in enumerate(parts):
            lname = ft["name"] if len(parts) == 1 else f"{ft['name']}#{pk + 1}"
            c = part.centroid
            fwd, inv = _local_metric(c.x, c.y)
            line_m = transform(fwd, part)
            if line_m.length <= 0:
                continue
            area_ll = transform(inv, line_m.buffer(buffer_m + 2 * max(_pixel_metres(prod.attrs, prod.shape))))
            epsg = _grid_epsg(prod.attrs, prod.shape, (c.x, c.y))
            win = _window(prod.attrs, prod.shape, _to_grid(area_ll, epsg))
            nb = max(1, int(math.ceil(line_m.length / bin_m)))
            # each bin's middle along the line (the last bin may be shorter than bin_m)
            mids = [(b * bin_m + min((b + 1) * bin_m, line_m.length)) / 2.0 for b in range(nb)]
            centres = [line_m.interpolate(m) for m in mids]
            clon, clat = inv(np.array([p.x for p in centres]), np.array([p.y for p in centres]))
            if win is None:
                sel = np.zeros((0,), bool)
                X = Y = np.zeros((0,))
            else:
                X, Y = _centres(prod.attrs, win)
                lon, lat = _grid_to_lonlat(X, Y, epsg)
                mx, my = fwd(lon, lat)
                mx, my = np.asarray(mx), np.asarray(my)
                pts = shapely.points(mx, my)
                sel = ((shapely.distance(pts, line_m) <= buffer_m) & _alongside(line_m, mx, my)
                       & prod.reliable[win[0]:win[1], win[2]:win[3]])
            if sel.any():
                along = shapely.line_locate_point(line_m, pts[sel])
                b_of = np.minimum((along // bin_m).astype(int), nb - 1)
                R, C = np.nonzero(sel)
                R, C = R + win[0], C + win[2]
                ids, _ = prod.dlv.radar_pixel_ids(prod.cfg, R, C, prod.shape)
                vb, _ = prod.value_at(sb, win)
                vb = vb[sel]
                up = lu[R, C] if lu is not None else np.full(R.size, np.nan)
            for dd, s in st:
                if s["why"]:
                    for b in range(nb):
                        rows.append(_profile_row(lname, b, mids[b], clat[b], clon[b], d_base, sb, dd, s, prod,
                                                 status=f"no value: {s['why']}"))
                    continue
                if sel.any():
                    v1, _ = prod.value_at(s, win)
                    ch = v1[sel] - vb
                for b in range(nb):
                    if not sel.any():
                        rows.append(_profile_row(lname, b, mids[b], clat[b], clon[b], d_base, sb, dd, s, prod,
                                                 status="no reliable cell within the buffer"))
                        continue
                    m = (b_of == b) & np.isfinite(ch)
                    v = ch[m]
                    n = int(v.size)
                    nind = int(np.unique(ids[m][ids[m] >= 0]).size)
                    extra = {"n_cells": n, "n_independent": nind}
                    if n:
                        med = float(np.median(v))
                        extra.update(spread_mm=round(1.4826 * float(np.median(np.abs(v - med))), 2),
                                     p05_mm=round(float(np.percentile(v, 5)), 2))
                    if n >= min_cells and nind >= min_independent:
                        u = float(np.nanmedian(up[m])) if np.isfinite(up[m]).any() else float("nan")
                        extra.update(change_mm=round(med, 2),
                                     vertical_assumed_mm=round(med / u, 2) if math.isfinite(u) and u > 0.1 else None)
                        rows.append(_profile_row(lname, b, mids[b], clat[b], clon[b], d_base, sb, dd, s, prod,
                                                 status="ok", **extra))
                    else:
                        rows.append(_profile_row(lname, b, mids[b], clat[b], clon[b], d_base, sb, dd, s, prod,
                                                 status=(f"too few reliable cells ({n} cells, {nind} radar pixels; "
                                                         f"{min_cells} and {min_independent} needed)"), **extra))
    cols = ["line", "bin", "distance_m", "lat", "lon", "baseline", "baseline_epoch", "date", "epoch", "change_mm",
            "n_cells", "n_independent", "spread_mm", "p05_mm", "vertical_assumed_mm", "status"]
    df = pd.DataFrame(rows, columns=cols)
    df.attrs.update({"units": "mm LOS", "sign": SIGN, "datum": prod.datum()["relative_to"],
                     "buffer_m": buffer_m, "bin_m": bin_m, "coherence_floor": prod.floor, "snap": snap,
                     "min_cells": min_cells, "min_independent": min_independent,
                     "vertical_assumed": "LOS / up component of the line of sight: assumes vertical motion"})
    return df


def _alongside(line_m, x, y, tol_m: float = 0.01) -> np.ndarray:
    """True for points beside the line, not beyond its ends: the projection
    onto the first segment is not before the start, onto the last not after
    the end (1 cm tolerance, so a point on an end is in). Without this the
    round ends of a buffer pile cells from beyond the line into the first
    and last bins (half a disc of 100 m: about 57 cells at 15 m posting,
    against ~45 in a 50 m bin)."""
    c = np.asarray(line_m.coords, dtype=float)
    (ax, ay), (bx, by) = c[0], c[1]
    (px, py), (qx, qy) = c[-2], c[-1]
    da = np.hypot(bx - ax, by - ay) or 1.0
    db = np.hypot(qx - px, qy - py) or 1.0
    before = ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / da < -tol_m
    after = ((x - qx) * (qx - px) + (y - qy) * (qy - py)) / db > tol_m
    return ~before & ~after


def _profile_row(line, b, dist_m, lat, lon, d_base, sb, dd, s, prod, status, **extra) -> dict:
    return {"line": line, "bin": b, "distance_m": round(float(dist_m), 1), "lat": round(float(lat), 6),
            "lon": round(float(lon), 6), "baseline": _iso(d_base), "baseline_epoch": sb["used"], "date": _iso(dd),
            "epoch": s["used"], "status": status, **extra}


# ------------------------------------------------------------------ points
def _points_list(points) -> list[dict]:
    """[{id, lat, lon, epochs (or None)}] from: a CSV path
    (ground_truth.load_points_csv: each point's own dates), a GeoJSON path
    or dict of Point features, {id: {lat, lon}} or {id: (lat, lon)}, a list
    of {id, lat, lon} dicts, of (Point, DataFrame) pairs, or of (lat, lon)
    pairs, or a pandas DataFrame with id, lat, lon[, date]."""
    import pandas as pd
    out = []
    if isinstance(points, (str, Path)):
        p = Path(str(points)).expanduser()
        if p.suffix.lower() == ".csv":
            from . import ground_truth
            points = ground_truth.load_points_csv(p)
        else:
            points = _read_geojson(p)
    if isinstance(points, dict) and points.get("type") in ("FeatureCollection", "Feature", "Point"):
        data = points
        feats = (data.get("features") or [] if data.get("type") == "FeatureCollection"
                 else [data] if data.get("type") == "Feature" else [{"geometry": data, "properties": {}}])
        epsg = _geojson_epsg(data)
        to_ll = None
        if epsg:
            from pyproj import Transformer
            to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform
        for k, ft in enumerate(feats):
            g = (ft or {}).get("geometry") or {}
            if g.get("type") != "Point":
                continue
            props = ft.get("properties") or {}
            pid = next((str(props[c]) for c in _NAME_KEYS if props.get(c) not in (None, "")), f"point_{k + 1}")
            x, y = (float(v) for v in g["coordinates"][:2])
            if to_ll is not None:
                x, y = to_ll(x, y)
            if abs(x) > 180 or abs(y) > 90:
                raise ValueError(f"{pid}: the coordinates are not longitude/latitude (GeoJSON is lon/lat; a "
                                 "projected file must name its CRS)")
            out.append({"id": pid, "lat": float(y), "lon": float(x), "epochs": None})
        return out
    if isinstance(points, pd.DataFrame):
        df = points.rename(columns={c: str(c).lower() for c in points.columns})
        tcol = "date" if "date" in df.columns else "time" if "time" in df.columns else None
        for pid, g in df.groupby("id", sort=False):
            out.append({"id": str(pid), "lat": float(g["lat"].iloc[0]), "lon": float(g["lon"].iloc[0]),
                        "epochs": list(g[tcol]) if tcol else None})
        return out
    if isinstance(points, dict):
        for pid, v in points.items():
            if isinstance(v, dict):
                out.append({"id": str(pid), "lat": float(v["lat"]), "lon": float(v["lon"]), "epochs": v.get("epochs")})
            else:
                out.append({"id": str(pid), "lat": float(v[0]), "lon": float(v[1]), "epochs": None})
        return out
    for k, it in enumerate(points or []):
        if isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], dict) and hasattr(it[1], "columns"):
            pt, df = it
            ep = sorted(set(df["time"])) if "time" in df.columns else None
            out.append({"id": str(pt["id"]), "lat": float(pt["lat"]), "lon": float(pt["lon"]), "epochs": ep})
        elif isinstance(it, dict):
            out.append({"id": str(it.get("id", f"point_{k + 1}")), "lat": float(it["lat"]), "lon": float(it["lon"]),
                        "epochs": it.get("epochs")})
        else:
            out.append({"id": f"point_{k + 1}", "lat": float(it[0]), "lon": float(it[1]), "epochs": None})
    return out


def values_at(cfg, points, epochs=None, radius_m: float = POINT_RADIUS_M, *, floor=None,
              min_cells: int = MIN_CELLS, min_independent: int = MIN_INDEPENDENT,
              max_gap_days: float = MAX_GAP_DAYS, _prod=None):
    """
    The time series at points, at any epochs (survey or digitised dates).

      at a SAR date  the median of the reliable cells whose centres lie
                     within radius_m of the point (validation.cells_near,
                     the harness's own selection), with min_cells cells and
                     min_independent radar pixels at that date, else no
                     value ("masked")
      at an epoch    that series interpolated linearly between the SAR dates
                     around it when they are at most max_gap_days (24) apart;
                     a SAR date itself is taken as it is. Outside the run's
                     dates or across a longer gap there is no value, and
                     the row says why.

    epochs: None = each point's own dates (a CSV of truth points) or else
    every SAR date; a list = the same epochs for every point; {id: [...]}
    per point.

    Returns a pandas DataFrame, one row per point and epoch: id, lat, lon,
    epoch (ISO), value_mm (LOS on the product's datum), change_mm (since
    the point's first epoch in this table with a value), status ('ok',
    'masked' = no reliable value, 'no value'), why, sar_before, sar_after,
    weight_after, n_cells, n_independent, spread_mm (1.4826 x MAD), los_east,
    los_north, los_up, incidence_deg, vertical_assumed_mm,
    cells_within_radius, reliable_cells, and the flag layers' share of the
    point's reliable cells (aliasing_risk_share, too_fast_share; None when
    the layer is absent).
    """
    import pandas as pd
    from .validation import cells_near
    prod = _product(cfg, floor, _prod)
    pts = _points_list(points)
    if not pts:
        raise ValueError("no points")
    le, ln, lu = (prod.geometry(k) for k in ("los_east", "los_north", "los_up"))
    inc = prod.geometry("incidence")
    flags = prod.flags()
    found = [cells_near(prod.attrs, prod.shape, p["lat"], p["lon"], radius_m) for p in pts]
    R = np.concatenate([f[0] for f in found]) if found else np.array([], int)
    C = np.concatenate([f[1] for f in found]) if found else np.array([], int)
    all_ids, _ = prod.dlv.radar_pixel_ids(prod.cfg, R, C, prod.shape) if R.size else (np.array([], np.int64), "")
    rows, k0 = [], 0
    for p, (r, c, _) in zip(pts, found):
        ids = all_ids[k0:k0 + r.size]
        k0 += r.size
        rel = prod.reliable[r, c] if r.size else np.array([], bool)
        rr, cc, ids = r[rel], c[rel], ids[rel]
        ts = prod.cells(rr, cc)
        fin = np.isfinite(ts)
        n_cells = fin.sum(1)
        n_ind = (np.array([np.unique(ids[fin[i] & (ids >= 0)]).size for i in range(ts.shape[0])])
                 if ts.shape[1] else np.zeros(len(prod.dates), int))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(ts, axis=1) if ts.shape[1] else np.full(len(prod.dates), np.nan)
            spread = (1.4826 * np.nanmedian(np.abs(ts - med[:, None]), axis=1) if ts.shape[1]
                      else np.full(len(prod.dates), np.nan))
        ok = (n_cells >= min_cells) & (n_ind >= min_independent)
        series = np.where(ok, med, np.nan)
        gsel = (rr, cc) if rr.size else (r, c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vec = [float(np.nanmedian(g[gsel])) if g is not None and np.size(gsel[0]) else None for g in (le, ln, lu)]
            incd = float(np.nanmedian(inc[gsel])) if inc is not None and np.size(gsel[0]) else None
        if all(v is not None and math.isfinite(v) for v in vec):
            norm = math.sqrt(sum(v * v for v in vec))          # a unit vector, as the harness's
            vec = [round(v / norm, 5) for v in vec] if norm > 0 else vec
        fsh = {}
        for name, key in ((ALIAS_LAYER, "aliasing_risk_share"), (TOO_FAST_LAYER, "too_fast_share")):
            a = flags.get(name)
            if a is None or not rr.size:
                fsh[key] = None
                continue
            v = a[rr, cc]
            v = v[np.isfinite(v)]
            fsh[key] = round(float((v >= 1).mean()), 4) if v.size else None
        ep = epochs.get(p["id"]) if isinstance(epochs, dict) else epochs
        if ep is None:
            ep = p["epochs"] if p["epochs"] is not None else list(prod.dates)
        first = None
        for e in sorted({_day(x) for x in ep}):
            s = _snap(prod.days, e, "end", "interpolate", max_gap_days)
            rec = {"id": p["id"], "lat": p["lat"], "lon": p["lon"], "epoch": _iso(e), "value_mm": None,
                   "change_mm": None, "status": "no value", "why": s["why"], "sar_before": None, "sar_after": None,
                   "weight_after": None, "n_cells": None, "n_independent": None, "spread_mm": None,
                   "los_east": vec[0], "los_north": vec[1], "los_up": vec[2], "incidence_deg": incd,
                   "vertical_assumed_mm": None, "cells_within_radius": int(r.size), "reliable_cells": int(rr.size),
                   **fsh}
            if s["why"] is None:
                a, b, w = s["i0"], s["i1"], s["w"]
                rec.update(sar_before=prod.iso(a), sar_after=prod.iso(b), weight_after=round(w, 4),
                           n_cells=int(min(n_cells[a], n_cells[b])), n_independent=int(min(n_ind[a], n_ind[b])))
                va, vb = series[a], series[b]
                if np.isfinite(va) and np.isfinite(vb):
                    v = float((1.0 - w) * va + w * vb)
                    rec.update(value_mm=round(v, 2), status="ok", why=None,
                               spread_mm=round(float(max(spread[a], spread[b])), 2))
                    if first is None:
                        first = v
                    rec["change_mm"] = round(v - first, 2)
                    if vec[2] is not None and math.isfinite(vec[2]) and vec[2] > 0.1:
                        rec["vertical_assumed_mm"] = round(v / vec[2], 2)
                else:
                    rec.update(status="masked", why=(f"fewer than {min_cells} reliable cells or {min_independent} "
                                                     "radar pixels at the SAR date(s)"))
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.attrs.update({"units": "mm LOS", "sign": SIGN, "datum": prod.datum()["relative_to"], "radius_m": radius_m,
                     "coherence_floor": prod.floor, "min_cells": min_cells, "min_independent": min_independent,
                     "max_gap_days": max_gap_days,
                     "vertical_assumed": "LOS / up component of the line of sight: assumes vertical motion"})
    return df


# ------------------------------------------------------------------ settings
def settings(cfg) -> dict:
    """The config's event keys, read (event_products, event_polygons, event_lines)."""
    return {"windows": parse_windows(getattr(cfg, "event_products", None)),
            "polygons": str(getattr(cfg, "event_polygons", "") or "").strip(),
            "lines": str(getattr(cfg, "event_lines", "") or "").strip()}


def check_settings(cfg) -> list[str]:
    """Problems with the event keys, in words (config.validate_config). A
    missing file is validate_inputs' message, not repeated here."""
    out = []
    try:
        wins = parse_windows(getattr(cfg, "event_products", None))
    except ValueError as exc:
        return [f"event_products: {exc}"]
    for key, kind in (("event_polygons", "polygon"), ("event_lines", "line")):
        v = str(getattr(cfg, key, "") or "").strip()
        if not v:
            continue
        if not wins:
            out.append(f"{key} is set but event_products is empty, so nothing would be written: set event_products "
                       "to true (the whole record) or to a list of windows (YYYY-MM-DD:YYYY-MM-DD)")
        if not Path(v).expanduser().exists():
            continue
        try:
            feats = load_features(v, kind)
            if not feats:
                out.append(f"{key} {v} holds no {kind} feature")
            if kind == "polygon":
                for ft in feats:
                    _own_window(ft["props"], ft["name"])
        except (ValueError, KeyError, TypeError) as exc:
            out.append(f"{key} {v}: {exc}")
    return out


# ------------------------------------------------------------------ writing
def _sha256(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# the files write_event_products makes: only these are ever moved aside
_OUR_FILES = re.compile(r"^(los_change_mm_.+\.(tif|png|tif\.aux\.xml)|panel_max\.(csv|json)|panels\.png|"
                        r"profiles\.csv|profile_.+\.png|values_at\.csv|README\.txt|events\.json)$")


def _set_aside(out_dir: Path) -> Path | None:
    """Move an earlier write's files to <parent>/superseded/<name>_<stamp>/
    (moved, never deleted), so nothing in out_dir is from another run. Only
    files this module makes are moved: a folder given on the command line
    (--out) may hold other things, and they stay where they are."""
    out_dir = Path(out_dir)
    ours = [p for p in out_dir.iterdir() if p.is_file() and _OUR_FILES.match(p.name)] if out_dir.is_dir() else []
    if not ours:
        return None
    dest = out_dir.parent / "superseded" / f"{out_dir.name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in ours:
        shutil.move(str(p), str(dest / p.name))
    log.info("Earlier event products moved to %s", dest)
    return dest


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "window"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, date, datetime)):
        return str(o)
    return str(o)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_map(path: Path, arr, attrs, title: str, overlays=()) -> None:
    plt = _plt()
    x0, dx, y0, dy = (float(attrs[k]) for k in ("X_FIRST", "X_STEP", "Y_FIRST", "Y_STEP"))
    ny, nx = arr.shape
    extent = (x0, x0 + nx * dx, y0 + ny * dy, y0)
    v = arr[np.isfinite(arr)]
    lim = float(np.percentile(np.abs(v), 98)) if v.size else 1.0
    lim = lim if lim > 0 else 1.0
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(arr, extent=extent, cmap="RdBu", vmin=-lim, vmax=lim, interpolation="nearest")
    for g, label in overlays:
        geoms = list(g.geoms) if hasattr(g, "geoms") else [g]
        for gg in geoms:
            if gg.geom_type == "Polygon":
                xs, ys = gg.exterior.xy
            else:
                xs, ys = gg.xy
            ax.plot(xs, ys, color="k", lw=0.8)
        if label:
            c = g.representative_point() if g.geom_type.endswith("Polygon") else g.interpolate(0.5, normalized=True)
            ax.text(c.x, c.y, label, fontsize=6, ha="center")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("LOS change (mm), negative = away from the satellite")
    ax.set_title(title, fontsize=9)
    from . import deliverables as dlv
    geographic = dlv._is_geographic(attrs)
    ax.set_aspect(1.0 / max(math.cos(math.radians(y0 + ny * dy / 2.0)), 0.1) if geographic else "equal")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_panels(path: Path, rows: list[dict]) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in rows:
        s = [x for x in r.get("series") or [] if x["robust_mm"] is not None]
        if not s:
            continue
        t = [datetime.strptime(x["date"], "%Y-%m-%d") for x in s]
        ax.plot(t, [x["robust_mm"] for x in s], marker="o", ms=2, lw=1,
                label=f"{r['panel']} ({r['window'].get('epoch_start')} ..)")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_ylabel(f"LOS change since the window's start (mm, {PEAK_PERCENTILE:g}th percentile of the panel)")
    ax.set_title("Panels: robust deepest change over each window", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_profiles(path: Path, df, line: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sub = df[df["line"] == line]
    for (wname, d), g in sub.groupby(["window", "date"], sort=False):
        ax.plot(g["distance_m"], g["change_mm"], marker="o", ms=2, lw=1,
                label=f"{g['baseline_epoch'].iloc[0]} -> {g['epoch'].iloc[0]}")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xlabel("distance along the line (m)")
    ax.set_ylabel("LOS change (mm, median per bin)")
    ax.set_title(f"Profile along {line}", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


README = """Event products (insar_pipeline.events, F5)

What the ground did between two dates, not a rate. Every figure is the
difference of the delivered time series between two of its dates.

  los_change_mm_<window>_<from>_<to>.tif  change map of one window (GeoTIFF;
                                    its tags give the dates asked for and used)
  los_change_mm_<window>_<from>_<to>.png  a quick look at it
  panel_max.csv / panel_max.json    per polygon and window: the largest robust
                                    subsidence (max_mm), its date, where, the
                                    flags; the json adds each panel's series
  panels.png                        the panels' robust change over time
  profiles.csv, profile_<line>.png  binned change along each line
  values_at.csv                     points at their epochs (command line --points)
  events.json                       what was asked, what was used, any error

Units and sign: {sign}.
Datum: {datum}. A change on it carries whatever the datum and the wider
ground did (on farmland, the record's coherence-class drift: the whole field
can read tens of mm over a few years). panel_max gives the change of the
ground 1-2 km around each panel over the same dates (ring_median_mm, and
ring_robust_mm, the same percentile as max_mm) and the panel's figures less
them, like for like: max_mm_relative_to_ring (percentile less percentile) and
median_mm_relative_to_ring (median less median). Both read about 0 where the
panel moves like the ground around it; read them when the ring has cells.
Dates: a window snaps outward to the run's dates (start: the last SAR date on
or before it; end: the first on or after it); the dates used are in every
file. "clamped" = the window reaches beyond the record.
Reliable cell: temporal coherence >= {floor:g} and a value at both dates.
A panel's max_mm is the {pct:g}th percentile of its reliable cells' change (a
robust minimum); a line bin is the median of its cells within {buf:g} m of the
line, {bin:g} m bins along it, with {mc} cells and {mi} radar pixels at least.
The vertical_assumed columns divide by the up component of the line of sight:
they assume vertical motion, which longwall flanks do not have.
Where the motion aliases (a steep, fast bowl in C-band) the change is wrong
by whole fringes, usually too small: see the flag shares beside each panel,
map and point (quality_aliasing_risk.tif, F2, over the whole record; the
_interval columns are the F2 flag of the figure's own dates; the older
too-fast flag). A large flagged share = a lower bound, not a measurement.
A flag layer older than the time series is not used (events.json).
"""


def write_event_products(cfg, out_dir=None, *, windows=None, polygons=None, lines=None, points=None,
                         radius_m: float = POINT_RADIUS_M) -> dict | None:
    """
    deliverables/events/ (or out_dir): the change map of each window of
    event_products (GeoTIFF + PNG), panel_max over event_polygons, profiles
    of each window along event_lines, values_at for `points` (command line
    only), README.txt and events.json. The keyword arguments replace the
    config's keys. An earlier write in the folder is moved to superseded/.
    Each part is made on its own: a failure is logged, written to
    events.json and the rest still made. Returns the events.json record, or
    None when there is nothing to make (no window).
    """
    st = settings(cfg)
    wins = parse_windows(windows) if windows is not None else st["windows"]
    polys = st["polygons"] if polygons is None else str(polygons or "")
    lns = st["lines"] if lines is None else str(lines or "")
    if not wins and not points:
        return None
    from . import deliverables as dlv
    from .areas import formula_safe_frame
    out = Path(out_dir) if out_dir else Path(cfg.project_dir) / "deliverables" / EVENTS_DIR
    prod = _Product(cfg)
    moved = _set_aside(out)
    out.mkdir(parents=True, exist_ok=True)
    # 'software': which code wrote the folder - its CSVs carry no '#' lines to
    # say it (16 Oct 2026, readiness review OUT-01)
    man = {"written": datetime.now().isoformat(timespec="seconds"), "status": "written", "software": dlv._software(),
           "product": {"timeseries": str(prod.ts_path), "dates": len(prod.dates), "first": prod.iso(0),
                       "last": prod.iso(-1), **prod.datum()},
           "coherence_floor": prod.floor, "units": "mm LOS", "sign": SIGN,
           "settings": {"windows": wins, "polygons": polys or None, "polygons_sha256": _sha256(polys) if polys else None,
                        "lines": lns or None, "lines_sha256": _sha256(lns) if lns else None},
           "earlier_write_moved_to": str(moved) if moved else None,
           "maps": [], "panels": [], "profiles": None, "values_at": None, "files": [], "errors": []}
    overlays = []
    for src, kind in ((polys, "polygon"), (lns, "line")):
        if src:
            try:
                overlays += [(f["geom"], f["name"]) for f in load_features(src, kind)]
            except Exception as exc:                              # noqa: BLE001 - reported with its part below
                log.warning("%s not read for the map overlay (%s)", src, exc)

    def fail(part, exc):
        msg = f"{part}: {type(exc).__name__}: {exc}"
        log.error("Event products: %s", msg)
        man["errors"].append(msg)

    for w in wins:
        try:
            arr, attrs = cumulative_between(cfg, w["start"], w["end"], _prod=prod)
            e0, e1 = attrs["START_DATE"], attrs["END_DATE"]
            stem = f"los_change_mm_{_safe(w['name'])}_{e0}_{e1}" if w["name"] not in (
                "record", f"{w['start']}_{w['end']}") else f"los_change_mm_{e0}_{e1}"
            tif = out / f"{stem}.tif"
            tags = dlv.product_tags(
                cfg, attrs, "line-of-sight change between two dates", "mm",
                "the reference pixel (the product's datum; the reference date cancels in a change)",
                sign="Negative = motion away from the satellite along the line of sight (subsidence on a "
                     "near-vertical signal). Positive = toward the satellite.",
                **{k.lower(): v for k, v in attrs.items() if k.startswith("EVENT_")})
            dlv._write_geotiff(tif, arr, attrs, tags=tags, units="mm",
                               description=(f"LOS change (mm) from {attrs['EVENT_EPOCH_START']} to "
                                            f"{attrs['EVENT_EPOCH_END']} (window asked: "
                                            f"{attrs['EVENT_REQUESTED_START']} .. {attrs['EVENT_REQUESTED_END']}), "
                                            "the difference of the delivered time series; no rate assumed. "
                                            f"Cells with temporal coherence below {prod.floor:g} are empty."))
            man["files"].append(tif.name)
            epsg = None
            try:
                ov = []
                if overlays:
                    c = overlays[0][0].representative_point()
                    epsg = _grid_epsg(prod.attrs, prod.shape, (c.x, c.y))
                    ov = [(_to_grid(g, epsg), name) for g, name in overlays]
                png = out / f"{stem}.png"
                _plot_map(png, arr, attrs, f"LOS change {attrs['EVENT_EPOCH_START']} -> {attrs['EVENT_EPOCH_END']} "
                                           f"(window {w['name']})", ov)
                man["files"].append(png.name)
            except Exception as exc:                              # noqa: BLE001
                fail(f"quick-look of {w['name']}", exc)
            fin = np.isfinite(arr)
            v = arr[fin]
            R, C = np.nonzero(fin)
            entry = {"window": w, "file": tif.name, "epoch_start": attrs["EVENT_EPOCH_START"],
                     "epoch_end": attrs["EVENT_EPOCH_END"], "clamped": attrs["EVENT_CLAMPED"] == "yes",
                     "reliable_cells": int(v.size),
                     "p01_mm": round(float(np.percentile(v, 1)), 2) if v.size else None,
                     "median_mm": round(float(np.median(v)), 2) if v.size else None,
                     "flags": _flag_shares(prod, R, C)}
            iv = _interval_share(prod, prod.dates.index(e0), prod.dates.index(e1), R, C)
            if iv is not None:
                entry["flags"]["aliasing_risk_interval"] = iv
            man["maps"].append(entry)
        except Exception as exc:                                  # noqa: BLE001
            fail(f"window {w['name']}", exc)

    if polys:
        try:
            rows = panel_max(cfg, polys, windows=wins or None, _prod=prod)
            man["panels"] = [{k: v for k, v in r.items() if k != "series"} for r in rows]
            import pandas as pd
            flat = []
            for r in rows:
                f = r.get("flags") or {}
                flat.append({"panel": r["panel"], "window": r["window"].get("name"),
                             "window_source": r["window"].get("source"),
                             "requested_start": r["window"].get("requested_start"),
                             "requested_end": r["window"].get("requested_end"),
                             "epoch_start": r["window"].get("epoch_start"), "epoch_end": r["window"].get("epoch_end"),
                             "clamped": r["window"].get("clamped"), "max_mm": r["max_mm"],
                             "date_of_max": r["date_of_max"], "date_half": r.get("date_half"),
                             "max_at_window_end": r.get("max_at_window_end"), "lat": r["lat"], "lon": r["lon"],
                             "location_mean_mm": r.get("location_mean_mm"),
                             "vertical_assumed_mm": r.get("vertical_assumed_mm"),
                             "ring_median_mm": (r.get("ring") or {}).get("median_mm"),
                             "ring_robust_mm": (r.get("ring") or {}).get("robust_mm"),
                             "ring_cells": (r.get("ring") or {}).get("cells"),
                             "max_mm_relative_to_ring": r.get("max_mm_relative_to_ring"),
                             "median_mm_relative_to_ring": r.get("median_mm_relative_to_ring"),
                             "median_mm_at_max": r.get("median_mm_at_max"),
                             "n_cells_in_polygon": r.get("n_cells_in_polygon"),
                             "n_reliable_cells": r.get("n_reliable_cells"), "coverage_share": r.get("coverage_share"),
                             "n_independent_at_max": r.get("n_independent_at_max"),
                             "aliasing_risk_share": (f.get(ALIAS_LAYER) or {}).get("share_flagged"),
                             "likely_aliased_share": (f.get(ALIAS_LAYER) or {}).get("share_likely_aliased"),
                             "aliasing_risk_share_interval": (f.get("aliasing_risk_interval") or {}).get(
                                 "share_flagged"),
                             "likely_aliased_share_interval": (f.get("aliasing_risk_interval") or {}).get(
                                 "share_likely_aliased"),
                             "too_fast_share": (f.get(TOO_FAST_LAYER) or {}).get("share_flagged"),
                             "reason": r.get("reason")})
            # polygon, window and line names come from the client's files: never a
            # formula in a spreadsheet (16 Oct 2026, readiness review SEC-05)
            formula_safe_frame(pd.DataFrame(flat)).to_csv(out / "panel_max.csv", index=False)
            (out / "panel_max.json").write_text(json.dumps({"units": "mm LOS", "sign": SIGN, "panels": rows},
                                                           indent=1, default=_json_default))
            man["files"] += ["panel_max.csv", "panel_max.json"]
            try:
                _plot_panels(out / "panels.png", rows)
                man["files"].append("panels.png")
            except Exception as exc:                              # noqa: BLE001
                fail("panels.png", exc)
        except Exception as exc:                                  # noqa: BLE001
            fail(f"panels ({polys})", exc)

    if lns:
        try:
            import pandas as pd
            frames = []
            for w in wins or [{"name": "record", "start": None, "end": None}]:
                s0, s1 = prod.window_epochs(w["start"], w["end"])
                df = profile(cfg, lns, [prod.dates[s0["i0"]], prod.dates[s1["i1"]]], _prod=prod)
                df.insert(0, "window", w["name"])
                frames.append(df)
            prof = pd.concat(frames, ignore_index=True)
            formula_safe_frame(prof).to_csv(out / "profiles.csv", index=False)
            man["files"].append("profiles.csv")
            man["profiles"] = {"lines": sorted(set(prof["line"])), "rows": int(len(prof)),
                               "bins_with_value": int((prof["status"] == "ok").sum()), **frames[0].attrs}
            for ln in sorted(set(prof["line"])):
                try:
                    png = out / f"profile_{_safe(ln)}.png"
                    _plot_profiles(png, prof, ln)
                    man["files"].append(png.name)
                except Exception as exc:                          # noqa: BLE001
                    fail(f"profile_{ln}.png", exc)
        except Exception as exc:                                  # noqa: BLE001
            fail(f"profiles ({lns})", exc)

    if points:
        try:
            va = values_at(cfg, points, None, radius_m, _prod=prod)
            formula_safe_frame(va).to_csv(out / "values_at.csv", index=False)
            man["files"].append("values_at.csv")
            man["values_at"] = {"points": int(va["id"].nunique()), "rows": int(len(va)),
                                "with_value": int((va["status"] == "ok").sum()), **va.attrs}
        except Exception as exc:                                  # noqa: BLE001
            fail("values_at", exc)

    (out / "README.txt").write_text(README.format(
        sign=SIGN, datum=prod.datum()["relative_to"], floor=prod.floor, pct=PEAK_PERCENTILE, buf=LINE_BUFFER_M,
        bin=LINE_BIN_M, mc=MIN_CELLS, mi=MIN_INDEPENDENT))
    man["files"].append("README.txt")
    man["flag_layers"] = {"used": sorted(prod.flags()), "notes": sorted(set(prod.flags_note)),
                          "what": ("share of each panel's / map's reliable cells under the flag layers: "
                                   f"{ALIAS_LAYER} (F2, the whole record; 0 none found, 1 risk, 2 likely aliased; a "
                                   f"lower bound), {TOO_FAST_LAYER} (from the unwrapped velocity; a lower bound) and "
                                   "aliasing_risk_interval (the F2 flag of the figure's own interval, "
                                   "aliasing.risk_between)")}
    if man["errors"]:
        man["status"] = "written with errors" if len(man["files"]) > 1 else "failed"
    (out / "events.json").write_text(json.dumps(man, indent=1, default=_json_default))
    log.info("Event products: %s in %s (%d file(s)%s)", man["status"], out, len(man["files"]),
             f"; {len(man['errors'])} error(s), see events.json" if man["errors"] else "")
    return man


def record_failure(out_dir, exc) -> None:
    """events.json saying the event products could not be made at all."""
    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / "events.json").write_text(json.dumps(
            {"written": datetime.now().isoformat(timespec="seconds"), "status": "failed",
             "errors": [f"{type(exc).__name__}: {exc}"]}, indent=1))
    except OSError as e:
        log.warning("Could not record the event-product failure in %s (%s)", out, e)


# ------------------------------------------------------------------ command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m insar_pipeline.events",
        description="Event products of a finished run: change maps between dates, the largest subsidence and its "
                    "date per panel, profiles along lines, values at survey epochs. Reads the run's products; "
                    "writes into --out (default <project>/deliverables/events). The options replace the "
                    "config's event_products / event_polygons / event_lines.")
    ap.add_argument("config")
    ap.add_argument("--between", nargs=2, action="append", metavar=("FROM", "TO"),
                    help="a window (YYYY-MM-DD YYYY-MM-DD); repeat for more")
    ap.add_argument("--record", action="store_true", help="add the whole record as a window")
    ap.add_argument("--polygons", help="GeoJSON of panels (lon/lat)")
    ap.add_argument("--lines", help="GeoJSON of lines (lon/lat)")
    ap.add_argument("--points", help="truth points (CSV as ground_truth.load_points_csv, or GeoJSON points)")
    ap.add_argument("--radius", type=float, default=POINT_RADIUS_M, help="point sampling radius (m), default 100")
    ap.add_argument("--out", help="output folder")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .config import ConfigError, load_config
    try:
        cfg = load_config(a.config)
    except (ConfigError, OSError) as exc:
        print(f"Not made: the config does not load ({exc})")
        return 1
    try:
        from . import runs
        if runs.state(cfg.project_dir) == "running":
            print(f"Not made: a run is going in {cfg.project_dir}; make event products after it ends.")
            return 3
    except Exception:                                             # noqa: BLE001 - no run registry is no run
        pass
    wins = None
    if a.between or a.record:
        wins = ([{"name": "record", "start": None, "end": None}] if a.record else []) + \
               [{"name": f"{x}_{y}", "start": x, "end": y} for x, y in (a.between or [])]
    try:
        man = write_event_products(cfg, a.out, windows=wins, polygons=a.polygons, lines=a.lines, points=a.points,
                                   radius_m=a.radius)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Not made: {exc}")
        return 1
    if man is None:
        print("Nothing to make: no window (event_products, --between or --record) and no --points.")
        return 2
    print(f"Event products {man['status']}: {', '.join(man['files'])}")
    for e in man["errors"]:
        print(f"  error: {e}")
    return 0 if man["status"] == "written" else 1


if __name__ == "__main__":
    raise SystemExit(main())
