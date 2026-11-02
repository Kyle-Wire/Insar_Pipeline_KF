"""
Combined view: an ascending and a descending project of the same ground,
cross-referenced into vertical and east-west movement (28 Oct 2026).

WHY. One look direction turns every movement into "vertical": east-west
movement reads as sinking from one side and as rising from the other. On
the first site combined the ascending report read -1.75 and the descending +1.68 mm/yr
"vertical" (relative): the same ground, opposite answers. Two directions
together separate vertical from east-west (decompose.py's solve).

WHAT A COMBINED PROJECT IS. A project of its own, so it is plain that two
projects are cross-referenced: configs/<name>_combined.yaml

    kind: combined
    name: Site combined
    project_dir: <projects root>/site_combined
    ascending: <configs>/site.yaml
    descending: <configs>/site_descending.yaml

It reads the two finished projects and writes only into its own folder: the
sources are never changed (their reports carry one pointer line, written by
their own report build: pointer_for).

THE METHOD (review_2026-10-28/PLAN.md, tested in its Phase 0):
  1. each run's report field: the class-matched relative LOS rate on the
     report's own farmland (panel_history._Context + results.report_farmland,
     the recipe the viewer and the points table use), reliable cells only;
  2. both on decompose.py's common grid (15 m, lat/lon);
  3. Up and East solved per cell with each run's own LOS vector; north is
     not estimated (its leak into Up is about -0.14 mm per mm);
  4. Up anchored with the regional GNSS vertical as the single-run report is;
     East stays relative to the surrounding ground;
  5. Up and East each tested against the comparison ground
     (significance.assess): ground in both runs away from the mines and the
     report areas, which the code calls "farmland" (far_wide) whatever it
     is. The records name it as the single-project report does
     (ground_label, landcover.comparison_ground): the first combined site's is built-up
     and grassland, with no cropland; "farmland" only where cropland
     dominates;
  6. the reading of each report area (decision()).

EAST-WEST IS INDICATIVE (Kyle, 28 Oct 2026). Phase 0's injection test caught
planted east-west movement at its stated 80% limit only 63% of the time
(70% required), so until the pre-registered re-test (PLAN.md P0.3b) passes:
the decision rests on the vertical; east-west is shown as "possible east-west
movement (indicative, not a tested figure)"; no east-west threshold,
detection limit or uncertainty goes into the client record. The full East
test is kept in deliverables/technical/combined_east_test.json.

Runs whose records start or end more than WINDOW_TOLERANCE_DAYS apart are
refused (each run's rate is over its own record; a refit over the common
window is on the plan's later list).

    python -m insar_pipeline.combine configs/<name>_combined.yaml
    python -m insar_pipeline.combine --new <asc.yaml> <desc.yaml>   # writes the config
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

KIND = "combined"
REPORT = "combined_report.json"
EAST_TEST = "combined_east_test.json"
WINDOW_TOLERANCE_DAYS = 45                 # crosscheck.WINDOW_TOLERANCE_DAYS: one revisit either side is not a difference
MIN_CELLS = 5                              # an area needs this many measured cells from EACH direction
MM_IN = 25.4
EAST_STATUS = ("indicative: the east-west test has not yet passed its injection check (review_2026-10-28 "
               "P0.3), so no east-west threshold, detection limit or uncertainty is given")
EAST_PHRASE = "possible east-west movement (indicative, not a tested figure)"
# files whose change makes a combined result out of date
SOURCE_FILES = ("velocity.h5", "temporalCoherence.h5", "maskTempCoh.h5")
_KEYS = {"kind", "name", "project_dir", "ascending", "descending"}


class CombinedConfigError(ValueError):
    pass


# ------------------------------------------------------------------ config

@dataclass
class CombinedConfig:
    path: Path
    name: str
    project_dir: Path
    ascending: Path
    descending: Path

    @property
    def deliverables(self) -> Path:
        return self.project_dir / "deliverables"


def _raw(path) -> dict | None:
    import yaml
    try:
        d = yaml.safe_load(Path(path).read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    return d if isinstance(d, dict) else None


def is_combined_config(path) -> bool:
    d = _raw(path)
    return bool(d) and str(d.get("kind", "")).lower() == KIND


def _expand(v) -> str:
    from . import settings
    settings.export_env()
    return os.path.expanduser(os.path.expandvars(str(v)))


def load(path) -> CombinedConfig:
    path = Path(path)
    d = _raw(path)
    if d is None:
        raise CombinedConfigError(f"{path}: not a readable YAML file of settings")
    if str(d.get("kind", "")).lower() != KIND:
        raise CombinedConfigError(f"{path}: not a combined config (kind: {KIND})")
    unknown = sorted(set(d) - _KEYS)
    if unknown:
        raise CombinedConfigError(f"{path}: unknown setting(s) {unknown}; a combined config holds only "
                                  f"{sorted(_KEYS)}")
    missing = [k for k in ("project_dir", "ascending", "descending") if not d.get(k)]
    if missing:
        raise CombinedConfigError(f"{path}: missing {missing}")
    cc = CombinedConfig(path=path, name=str(d.get("name") or path.stem),
                        project_dir=Path(_expand(d["project_dir"])),
                        ascending=Path(_expand(d["ascending"])), descending=Path(_expand(d["descending"])))
    for k in ("ascending", "descending"):
        if not getattr(cc, k).exists():
            raise CombinedConfigError(f"{path}: the {k} project's config {getattr(cc, k)} is not on disk")
    return cc


def write_config(asc_path, desc_path, out: Path | None = None, name: str | None = None) -> Path:
    """configs/<asc project>_combined.yaml for two existing project configs."""
    from . import settings
    from .config import load_config
    a, d = load_config(str(asc_path)), load_config(str(desc_path))
    _check_directions(a, d)
    stem = Path(a.project_dir).name
    out = Path(out) if out else settings.configs_dir() / f"{stem}_combined.yaml"
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    title = name or stem.replace("_", " ").title() + " combined"
    text = ("# Combined view: ascending + descending of the same ground (combine.py). A project of its\n"
            "# own: it reads the two projects below and never changes them.\n"
            f"kind: {KIND}\nname: {title}\n"
            f"project_dir: {Path(settings.projects_root()) / (stem + '_combined')}\n"
            f"ascending: {Path(asc_path).resolve()}\ndescending: {Path(desc_path).resolve()}\n")
    out.write_text(text)
    _STATUS.clear()
    return out


def _check_directions(a, d) -> None:
    da, dd = str(getattr(a, "orbit_direction", "")).lower(), str(getattr(d, "orbit_direction", "")).lower()
    if (da, dd) != ("ascending", "descending"):
        raise CombinedConfigError(f"the first project must look from an ascending pass and the second from a "
                                  f"descending one (they are {da or '?'} and {dd or '?'})")


# ------------------------------------------------------------------ sources

@dataclass
class Source:
    """One look direction as the combination reads it."""
    label: str                      # "ascending" / "descending"
    cfg: object
    rep: dict
    attrs: dict
    rel: np.ndarray                 # class-matched relative LOS rate, mm/yr, reliable cells only (NaN elsewhere)
    tc: np.ndarray
    good: np.ndarray
    far_wide: np.ndarray
    los_e: np.ndarray
    los_n: np.ndarray
    los_u: np.ndarray
    areas: list = field(default_factory=list)       # [(name, spec)]
    ctx: object = None                               # the report's _Context (the viewer reads the cube through it)


def load_source(cfg, label: str) -> Source:
    """A finished project's report field, on the report's own farmland."""
    from . import client_report as cr
    from . import deliverables as dl
    from .panel_history import _Context
    from .results import report_farmland
    rp = Path(cfg.project_dir) / "deliverables" / "client_report.json"
    if not rp.exists():
        raise FileNotFoundError(f"the {label} project has no client report yet ({rp}): run its Products first")
    rep = json.loads(rp.read_text())
    ctx = _Context(cfg)
    if list(rep.get("dates") or [])[:1] != ctx.dates[:1] or list(rep.get("dates") or [])[-1:] != ctx.dates[-1:]:
        raise RuntimeError(f"the {label} project's client report is from another record than its products: "
                           "rebuild its Products first")
    report_farmland(ctx, rep)
    vecs = [dl.geometry_field(cfg, k) for k in ("los_east", "los_north", "los_up")]
    if any(v is None for v in vecs):
        raise FileNotFoundError(f"the {label} project has no per-cell look geometry")
    e, n, u = (np.asarray(v, np.float32) for v in vecs)
    rel = np.where(ctx.good, ctx.r_null, np.nan).astype(np.float32)
    return Source(label=label, cfg=cfg, rep=rep, attrs=dict(ctx.attrs), rel=rel, tc=np.asarray(ctx.tc, np.float32),
                  good=np.asarray(ctx.good, bool), far_wide=np.asarray(ctx.far_wide, bool),
                  los_e=e, los_n=n, los_u=u, areas=cr.parse_areas(cfg), ctx=ctx)


def _area_geometry(spec: str):
    """Comparable form of an area spec: 'full', a box, or a GeoJSON's coordinates."""
    s = spec.strip()
    if s.lower() == "full":
        return "full"
    if s.lower().endswith(".geojson"):
        try:
            gj = json.loads(Path(s).read_text())
        except (OSError, ValueError):
            return ("missing", s)
        return [json.dumps(f.get("geometry"), sort_keys=True) for f in gj.get("features", [])]
    return tuple(round(float(v), 7) for v in s.split(","))


def check_areas(a: Source, d: Source) -> list[tuple[str, str]]:
    """The report areas, which both projects must share (same names, same shapes)."""
    na, nd = [x for x, _ in a.areas], [x for x, _ in d.areas]
    if na != nd:
        raise CombinedConfigError(f"the two projects have different report areas ({na} against {nd}); a combined "
                                  "view needs the same areas in both")
    for (name, sa), (_, sd) in zip(a.areas, d.areas):
        if _area_geometry(sa) != _area_geometry(sd):
            raise CombinedConfigError(f"report area '{name}' has a different outline in the two projects")
    return a.areas


def _days(x: str, y: str) -> int:
    f = lambda s: date(int(s[:4]), int(s[4:6]), int(s[6:8]))  # noqa: E731
    return abs((f(x) - f(y)).days)


def check_window(a: Source, d: Source) -> dict:
    ra, rd = a.rep["dates"], d.rep["dates"]
    ds, de = _days(ra[0], rd[0]), _days(ra[-1], rd[-1])
    if ds > WINDOW_TOLERANCE_DAYS or de > WINDOW_TOLERANCE_DAYS:
        raise CombinedConfigError(
            f"the records cover different years (ascending {ra[0]}-{ra[-1]}, descending {rd[0]}-{rd[-1]}): "
            f"they must start and end within {WINDOW_TOLERANCE_DAYS} days of each other")
    return {"ascending": [ra[0], ra[-1]], "descending": [rd[0], rd[-1]],
            "common": [max(ra[0], rd[0]), min(ra[-1], rd[-1])], "start_days_apart": ds, "end_days_apart": de}


# ------------------------------------------------------------------ the combination

class Grids:
    """Both sources on decompose.py's common grid."""

    def __init__(self, sources):
        from . import decompose as dc
        self.dc = dc
        self.src = [dc.Grid.from_attrs(s.attrs, s.rel.shape) for s in sources]
        self.grid = dc.common_grid(self.src)
        self.attrs = self.grid.attrs()
        self.rs = [dc._Resampler(g, self.grid) for g in self.src]

    def on(self, i: int, arr) -> np.ndarray:
        R = self.rs[i]
        r0, r1, c0, c1 = R.window
        return R(np.asarray(arr)[r0:r1, c0:c1])

    def vectors(self, i: int, s: Source):
        R = self.rs[i]
        r0, r1, c0, c1 = R.window
        return R.vectors(s.los_e[r0:r1, c0:c1], s.los_n[r0:r1, c0:c1], s.los_u[r0:r1, c0:c1])


def combine_fields(sources, G: Grids | None = None) -> dict:
    """Up / East (mm/yr, relative to the comparison ground) and what the tests need, on the
    common grid."""
    from . import decompose as dc
    G = G or Grids(sources)
    vec = [G.vectors(i, s) for i, s in enumerate(sources)]
    L = np.stack([G.on(i, s.rel) for i, s in enumerate(sources)])
    sol = dc.solve_up_east(L, np.stack([v[0] for v in vec]), np.stack([v[2] for v in vec]),
                           north_c=np.stack([v[1] for v in vec]))
    good = np.logical_and.reduce([G.on(i, s.good.astype(np.float32)) >= 0.5 for i, s in enumerate(sources)])
    far = np.logical_and.reduce([G.on(i, s.far_wide.astype(np.float32)) >= 0.5 for i, s in enumerate(sources)])
    with np.errstate(invalid="ignore"):
        tcs = np.stack([G.on(i, s.tc) for i, s in enumerate(sources)])
        tc = np.where(np.isfinite(tcs).all(0), tcs.mean(0), np.nan).astype(np.float32)
    return {"grid": G, "up": sol["up"], "east": sol["east"], "condition": sol["condition"],
            "leak_up": sol["leak_up"], "los": L, "vectors": vec, "good": good & np.isfinite(sol["up"]),
            "far_wide": far, "tc": tc}


def footprint_on(attrs: dict, shape, spec: str, bbox) -> np.ndarray:
    """An area's whole outline on a grid (client_report._mask's rule)."""
    from . import significance as sg
    from .stack_backend import _polygon_mask
    s = spec.strip()
    if s.lower() == "full":
        return sg.aoi_mask(bbox, attrs, shape)
    if s.lower().endswith(".geojson"):
        return _polygon_mask(Path(s), shape, attrs).astype(bool)
    s_, w, n, e = (float(v) for v in s.split(","))
    return sg.aoi_mask((min(w, e), min(s_, n), max(w, e), max(s_, n)), attrs, shape)


def area_footprint(G: Grids, spec: str, bbox) -> np.ndarray:
    return footprint_on(G.attrs, G.grid.shape, spec, bbox)


def assess(F: dict, field_: np.ndarray, footprint: np.ndarray) -> dict:
    from . import significance as sg
    reg = footprint & F["good"] & np.isfinite(field_)
    return sg.assess(field_, F["far_wide"] & np.isfinite(field_), reg, tc=F["tc"], attrs=F["grid"].attrs)


def _single(src: Source, name: str) -> dict:
    a = next((x for x in src.rep.get("areas") or [] if x.get("name") == name), {}) or {}
    return {"project": Path(src.cfg.project_dir).name, "direction": src.label,
            "cells": a.get("cells"), "reading": a.get("reading"), "detected": a.get("detected"),
            "excess_vertical_mm_yr": a.get("excess_vertical_mm_yr"), "vertical_mm_yr": a.get("vertical_mm_yr"),
            "testable": a.get("testable")}


def _words(text):
    """significance's sentences said "farmland" for the comparison ground,
    whatever it was, until 28 Oct 2026 (they say "ordinary ground" now); a
    source record built before then still does, and the combined record says
    ground (Kyle, 28 Oct 2026)."""
    if not isinstance(text, str):
        return text
    return (text.replace("ordinary farmland", "ordinary ground").replace("the farmland measured",
                                                                          "the comparison ground measured")
            .replace("farmland", "comparison ground"))


def _fmt(mm: float) -> str:
    return f"{mm / MM_IN:+.2f} in/yr ({mm:+.1f} mm/yr)"


def decision(up: dict, east: dict, singles: list, east_los: list, bg_v: float | None) -> dict:
    """The reading of one report area (PLAN.md Method 6, with Kyle's 28 Oct
    decision: the vertical decides; east-west is indicative).

    up / east: significance.assess of the combined fields over the area.
    singles: each direction's own report row (_single). east_los: each
    direction's median east component of its look vector over the area, to
    tell whether east-west movement explains a one-direction reading."""
    out = {"note": None}
    if not up.get("testable"):
        out.update(reading="Not testable", sentence=_words(up.get("not_testable_reason")) or "This area could not be tested.")
        return out
    x = float(up["excess_mm_yr"])
    vert = x + (bg_v or 0.0)
    e_flag = bool(east.get("testable") and east.get("distinguishable_from_farmland"))
    if up.get("distinguishable_from_farmland"):
        word = "sinking" if x < 0 else "rising"
        out.update(reading=f"Vertical movement ({word})",
                   sentence=f"Vertical movement: {word}, about {_fmt(vert)}, more than ordinary ground of its size.")
        if e_flag:
            out["sentence"] += f" Also {EAST_PHRASE}."
    elif e_flag:
        out.update(reading="No vertical movement detected; " + EAST_PHRASE,
                   sentence=("No vertical movement detected (the two directions together). There is "
                             + EAST_PHRASE + "."))
    else:
        out.update(reading="No movement detected",
                   sentence="No movement detected: both directions together read like ordinary ground of its size.")
    seen = [s for s in singles if s.get("detected")]
    if seen and not up.get("distinguishable_from_farmland"):
        ex_east = east.get("excess_mm_yr")
        explained = []
        for s, e_c in zip(singles, east_los):
            if not s.get("detected") or ex_east is None or e_c is None or s.get("excess_vertical_mm_yr") is None:
                continue
            explained.append(np.sign(e_c * ex_east) == np.sign(s["excess_vertical_mm_yr"]))
        which = " and ".join(s["direction"] for s in seen)
        out["note"] = (f"Seen from the {which} direction alone; "
                       + ("explained by east-west movement, which one direction reads as vertical."
                          if explained and all(explained) else "not confirmed by the two directions together."))
    return out


def ground_label(cfg, attrs: dict, far: np.ndarray) -> tuple[str, dict]:
    """The comparison ground in words, by the single-project report's rule
    (landcover.comparison_ground): the source config's control_ground_label
    when someone set it (not the default "farmland"); else "farmland" where
    cropland dominates the measured ground, "ground (mostly built-up and
    grassland)" elsewhere, "ground" without a land-cover map - after "the
    surrounding". Kyle, 28 Oct 2026: the first combined site has no farmland, so the
    default word was wrong there. Returns (words, comparison_ground's record:
    label, source, cover_percent, scheme)."""
    from . import landcover
    g = landcover.comparison_ground(cfg, attrs, far)
    return f"the surrounding {g['label']}", g


def _east_client(east: dict) -> dict:
    """What the client record carries about east-west: the movement and the
    indicative flag only - no threshold, limit or uncertainty."""
    return {"movement_mm_yr": east.get("excess_mm_yr"),
            "movement_in_yr": (round(east["excess_mm_yr"] / MM_IN, 3) if east.get("excess_mm_yr") is not None
                               else None),
            "stands_out_indicative": (bool(east.get("distinguishable_from_farmland")) if east.get("testable")
                                      else None),
            "status": "indicative", "sign": "positive = moving east, negative = moving west"}


def _up_client(up: dict, bg_v, bg_u) -> dict:
    if not up.get("testable"):
        return {"testable": False, "not_testable_reason": _words(up.get("not_testable_reason")),
                "measured_mean_mm_yr": up.get("value_mm_yr")}
    x = float(up["excess_mm_yr"])
    vert = x + (bg_v or 0.0)
    half = up["threshold_95_mm_yr"] if bg_v is None else math.hypot(up["threshold_95_mm_yr"], 1.96 * (bg_u or 0.0))
    return {"testable": True, "relative_vertical_mm_yr": round(x, 2),
            "vertical_mm_yr": round(vert, 2), "vertical_in_yr": round(vert / MM_IN, 3),
            "range95_mm_yr": [round(vert - half, 2), round(vert + half, 2)],
            "decision_threshold_mm_yr": up.get("threshold_95_mm_yr"),
            "detection_limit_sinking_mm_yr": up.get("detection_limit_sink_mm_yr"),
            "detection_limit_rising_mm_yr": up.get("detection_limit_rise_mm_yr"),
            "detected": bool(up.get("distinguishable_from_farmland")),
            "p_two_sided": up.get("p_two_sided"),
            "comparison_ground_spread_1sigma_mm_yr": up.get("farmland_spread_1sigma_mm_yr"),
            "anchored_to_gnss": bg_v is not None}


def _gnss(src: Source) -> tuple:
    rep = src.rep
    if not rep.get("anchored"):
        return None, None
    bg = (rep.get("gnss") or {}).get("background") or {}
    return bg.get("vertical_mm_yr"), bg.get("vertical_uncertainty_mm_yr")


def analyse(sources, areas, bbox) -> dict:
    """Per-area combined readings; nothing written."""
    F = combine_fields(sources)
    G = F["grid"]
    gv = [_gnss(s) for s in sources]
    anchored = all(v[0] is not None for v in gv)
    bg_v = gv[0][0] if anchored else None
    bg_u = max(v[1] or 0.0 for v in gv) if anchored else None
    rows, east_rows = [], []
    for name, spec in areas:
        fp = area_footprint(G, spec, bbox)
        reg = fp & F["good"]
        singles = [_single(s, name) for s in sources]
        n_cells = int(reg.sum())
        # each direction's measured cells on its OWN grid: the common grid
        # holds only the ground both directions cover
        own = [int((footprint_on(s.attrs, s.good.shape, spec, bbox) & s.good).sum()) for s in sources]
        if n_cells < MIN_CELLS:
            one = [s for s, n in zip(singles, own) if n >= MIN_CELLS]
            rows.append({"name": name, "cells_both": n_cells, "cells_each": own, "combined": False,
                         "reading": ("Measured from one direction only" if one else "Not measurable"),
                         "sentence": ((f"Only the {one[0]['direction']} pass measured this area: its own reading "
                                       f"stands ({one[0]['reading']}).") if one else
                                      "Too few points measured from both directions."),
                         "single_runs": singles})
            continue
        up, east = assess(F, F["up"], fp), assess(F, F["east"], fp)
        e_los = [float(np.nanmedian(v[0][reg])) if reg.any() else None for v in F["vectors"]]
        dec = decision(up, east, singles, e_los, bg_v)
        rows.append({"name": name, "cells_both": n_cells, "cells_each": own, "combined": True,
                     "reading": dec["reading"], "sentence": dec["sentence"], "note": dec["note"],
                     "vertical": _up_client(up, bg_v, bg_u), "east_west": _east_client(east),
                     "single_runs": singles})
        east_rows.append({"name": name, "east": {k: v for k, v in east.items() if k not in ("local_null", "coherence_null")},
                          "local_null": east.get("local_null"), "coherence_null": east.get("coherence_null")})
    ground, cover = ground_label(sources[0].cfg, G.attrs, F["far_wide"])
    return {"F": F, "areas": rows, "east_rows": east_rows, "anchored": anchored, "bg_v": bg_v, "bg_u": bg_u,
            "ground": ground, "ground_cover": cover}


# ------------------------------------------------------------------ records and staleness

_HASHES: dict = {}           # (path, mtime_ns, size) -> sha256: the pages ask every few seconds


def _sha256(p: Path) -> str | None:
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key not in _HASHES:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _HASHES[key] = h.hexdigest()
    return _HASHES[key]


def source_files(cfg) -> dict:
    """{name: sha256} of the files a combined result is built from."""
    from . import deliverables as dl
    out = {n: _sha256(dl._product(cfg, n)) for n in SOURCE_FILES}
    out["client_report.json"] = _sha256(Path(cfg.project_dir) / "deliverables" / "client_report.json")
    return out


def stale_reason(path) -> str | None:
    """None when the combined result is current; else why not, in words."""
    from .config import load_config
    cc = load(path)
    try:
        rec = json.loads((cc.deliverables / REPORT).read_text())
    except (OSError, ValueError):
        return "not run yet"
    for s in rec.get("sources") or []:
        try:
            now = source_files(load_config(s["config"]))
        except Exception as exc:                              # noqa: BLE001 - a missing source is a reason
            return f"the {s.get('direction')} project cannot be read ({str(exc).splitlines()[0]})"
        if now != s.get("files"):
            changed = sorted(k for k in now if now.get(k) != (s.get("files") or {}).get(k))
            return f"the {s.get('direction')} project has changed since ({', '.join(changed)}): run it again"
    return None


# ------------------------------------------------------------------ rasters

def _write_rasters(cc: CombinedConfig, sources, res: dict, window: dict) -> list:
    from . import deliverables as dl
    from . import significance as sg
    F, G = res["F"], res["F"]["grid"]
    D = cc.deliverables
    T = D / "technical"
    T.mkdir(parents=True, exist_ok=True)
    good = F["good"]
    bg_v, bg_u = res["bg_v"], res["bg_u"]
    anchored = bg_v is not None
    d0, d1 = window["common"]
    span = _days(d0, d1) / 365.25
    sd_up, _ = sg.farmland_cell_spread(F["up"], F["tc"], F["far_wide"])
    sd_e, _ = sg.farmland_cell_spread(F["east"], F["tc"], F["far_wide"])
    vert = F["up"] + (bg_v or 0.0)
    ground = res["ground"]
    rel_to = (f"{ground}, of the same coherence class, in both directions"
              + (", + the regional GNSS motion" if anchored else ""))
    common = {"COMBINED_FROM": f"{sources[0].label}: {Path(sources[0].cfg.project_dir).name}; "
                               f"{sources[1].label}: {Path(sources[1].cfg.project_dir).name}",
              "NORTH": "not estimated (both passes fly close to north-south)", "PROJECT": cc.project_dir.name,
              "ROUTE": "combined ascending + descending (combine.py)"}
    specs = (
        (D, "vertical_movement_combined_in_per_yr.tif", vert / MM_IN, "vertical velocity (combined)", "in/yr",
         "Vertical movement per year from the ascending and descending passes together, relative to " + rel_to
         + ". East-west movement is separated out; north is not estimated.", True, None),
        (D, "vertical_movement_combined_uncertainty_1sigma_in_per_yr.tif",
         np.hypot(sd_up, bg_u or 0.0) / MM_IN, "vertical velocity uncertainty (1 sigma, combined)", "in/yr",
         "Per-cell 1-sigma: the cell spread of the combined vertical over the comparison ground, by coherence class"
         + (", with the GNSS uncertainty" if anchored else ""), True, "unsigned (a 1-sigma uncertainty)"),
        (D, f"vertical_change_combined_at_average_rate_in_{d0}_{d1}.tif", vert * span / MM_IN,
         "vertical change at the average rate (combined)", "in",
         f"Average rate x {span:.2f} years per cell - not a measured change.", True, None),
        (D, "east_west_movement_INDICATIVE_in_per_yr.tif", F["east"] / MM_IN, "east-west velocity (indicative)",
         "in/yr", f"East-west movement per year relative to {ground}, positive = east. INDICATIVE: "
         "not a tested figure.", False, "positive = moving east, negative = moving west"),
        (D, f"east_west_change_INDICATIVE_at_average_rate_in_{d0}_{d1}.tif", F["east"] * span / MM_IN,
         "east-west change at the average rate (indicative)", "in",
         f"Average east-west rate x {span:.2f} years per cell, positive = east. INDICATIVE.", False,
         "positive = moving east, negative = moving west"),
        (T, "east_west_movement_uncertainty_1sigma_mm_per_yr.tif", sd_e, "east-west velocity uncertainty (1 sigma)",
         "mm/yr", "Technical only: the cell spread of the combined east-west over the comparison ground, by coherence class.", False,
         "unsigned (a 1-sigma uncertainty)"),
    )
    out = []
    for folder, name, arr, qty, units, desc, vertical, sign in specs:
        a = np.where(good, arr, np.nan).astype(np.float32)
        tags = dl.product_tags(sources[0].cfg, {**G.attrs, "START_DATE": d0, "END_DATE": d1}, qty, units,
                               relative_to=rel_to if vertical else f"{ground}, in both directions",
                               vertical=vertical, sign=sign)
        tags.update(common)
        tags.pop("REFERENCE_PIXEL", None)
        dl._write_geotiff(folder / name, a, G.attrs, tags=tags, description=desc, units=units)
        out.append(folder / name)
    return out


def run(path) -> Path:
    """Build the combined project; returns its combined_report.json."""
    from . import version
    from .config import load_config
    cc = load(path)
    a_cfg, d_cfg = load_config(str(cc.ascending)), load_config(str(cc.descending))
    _check_directions(a_cfg, d_cfg)
    before = [source_files(a_cfg), source_files(d_cfg)]
    sources = [load_source(a_cfg, "ascending"), load_source(d_cfg, "descending")]
    areas = check_areas(*sources)
    window = check_window(*sources)
    res = analyse(sources, areas, list(a_cfg.bbox))
    F = res["F"]
    # opposite look directions in fact, not only in the configs' words
    e_med = [float(np.nanmedian(v[0])) for v in F["vectors"]]
    if not (e_med[0] < 0 < e_med[1]):
        raise CombinedConfigError(f"the look directions are not ascending (east component < 0) and descending "
                                  f"(> 0): {e_med[0]:.2f} and {e_med[1]:.2f}")
    after = [source_files(a_cfg), source_files(d_cfg)]
    if after != before:
        raise RuntimeError("a source project changed while the combination was read: run it again when it is done")
    cc.deliverables.mkdir(parents=True, exist_ok=True)
    rasters = _write_rasters(cc, sources, res, window)
    G = F["grid"]
    rec = {
        "kind": KIND, "name": cc.name, "config": str(cc.path), "generated": datetime.now().isoformat(timespec="seconds"),
        "what": ("vertical and east-west movement from an ascending and a descending project of the same ground; "
                 "the vertical decides, east-west is indicative"),
        "sources": [{"direction": s.label, "config": str(p), "project_dir": str(s.cfg.project_dir),
                     "track": getattr(s.cfg, "track", None), "dates": s.rep.get("dates"),
                     "verdict": s.rep.get("verdict"), "files": f}
                    for s, p, f in zip(sources, (cc.ascending, cc.descending), before)],
        "window": window, "anchored_to_gnss": res["anchored"],
        "gnss_background_vertical_mm_yr": res["bg_v"], "gnss_background_uncertainty_mm_yr": res["bg_u"],
        "grid": {**G.attrs, "posting_m_approx": [round(x, 2) for x in G.grid.posting_m()]},
        "cells_solved": int(F["good"].sum()), "median_condition_number": round(float(np.nanmedian(F["condition"])), 3),
        "north_leak_into_vertical_median": round(float(np.nanmedian(F["leak_up"])), 3),
        "comparison_ground": {"words": res["ground"], "cells_in_both_directions": int(F["far_wide"].sum()),
                              "land_cover": res["ground_cover"],
                              "note": ("ordinary ground away from the mines and the report areas, the same in both "
                                       "directions; the code calls it 'farmland' whatever it is")},
        "east_west_status": EAST_STATUS, "areas": res["areas"],
        "rasters": [str(p.relative_to(cc.project_dir)) for p in rasters],
        "north": ("North-south movement is not estimated: both passes fly close to north-south. Its leak into the "
                  "vertical is about {:.2f} mm per mm of northward movement.".format(
                      float(np.nanmedian(F["leak_up"])))),
        "software": {"version": version.__version__, "code": version.code_hash()},
    }
    (cc.deliverables / "technical" / EAST_TEST).write_text(json.dumps(
        {"what": "the full comparison-ground test of the combined east-west field, per area - technical only",
         "status": EAST_STATUS, "areas": res["east_rows"]}, indent=1, default=str))
    out = cc.deliverables / REPORT
    out.write_text(json.dumps(rec, indent=1, default=str))
    _STATUS.clear()
    # the results viewer's table of the combined view (combine_viewer); a
    # failure there leaves the combined result standing and says so
    try:
        from . import combine_viewer
        combine_viewer.build(path)
    except Exception as exc:                                  # noqa: BLE001 - the map is extra, the result is not
        log.warning("The combined view's results map was not built: %s", exc)
    log.info("Combined view written to %s", out)
    return out


# ------------------------------------------------------------------ pointer for the source reports

def combined_configs() -> list[Path]:
    from . import runs
    return [c for c in runs.known_configs() if is_combined_config(c)]


def pointer_for(cfg) -> dict | None:
    """The combined project a source project belongs to, for its report's one
    line: {name, config, direction, status ('current' / 'out of date' / 'not
    run'), reason, sentence}. None when it belongs to none."""
    from .config import load_config
    me = Path(cfg.project_dir).resolve()
    for c in combined_configs():
        try:
            cc = load(c)
        except CombinedConfigError:
            continue
        for direction in ("ascending", "descending"):
            try:
                src = Path(load_config(str(getattr(cc, direction))).project_dir).resolve()
            except Exception:                                 # noqa: BLE001 - not this one
                continue
            if src != me:
                continue
            other = "descending" if direction == "ascending" else "ascending"
            why = stale_reason(c)
            status = "current" if why is None else ("not run" if why == "not run yet" else "out of date")
            base = (f"This ground is also measured from the {other} pass; the two are combined into vertical and "
                    f"east-west movement in '{cc.name}'")
            sentence = base + {"current": ".", "not run": " (not run yet).",
                               "out of date": f" (out of date: {why})."}[status]
            return {"name": cc.name, "config": str(c), "direction": direction, "status": status, "reason": why,
                    "sentence": sentence}
    return None


# ------------------------------------------------------------------ the program's offer (app and dashboard)

def _finished_window(cfg) -> list | None:
    try:
        rep = json.loads((Path(cfg.project_dir) / "deliverables" / "client_report.json").read_text())
    except (OSError, ValueError):
        return None
    d = rep.get("dates") or []
    return [d[0], d[-1]] if len(d) >= 2 else None


def partner_for(cfg) -> dict | None:
    """A finished project of the opposite look direction that can be combined
    with this one (PLAN D3): same report areas (names and outlines), records
    starting and ending within WINDOW_TOLERANCE_DAYS, both with a client
    report, and not already in a combined project. {config, name, direction}
    or None. The app and the dashboard both offer this, so they agree."""
    from . import client_report as cr
    from . import runs
    from .config import load_config
    mine = str(getattr(cfg, "orbit_direction", "")).lower()
    want = {"ascending": "descending", "descending": "ascending"}.get(mine)
    w0 = _finished_window(cfg)
    if want is None or w0 is None or pointer_for(cfg) is not None:
        return None
    me = Path(cfg.project_dir).resolve()
    areas = [(n, _area_geometry(s)) for n, s in cr.parse_areas(cfg)]
    for c in runs.known_configs():
        if is_combined_config(c):
            continue
        try:
            o = load_config(str(c))
        except Exception:                                     # noqa: BLE001 - not a candidate
            continue
        if Path(o.project_dir).resolve() == me or str(getattr(o, "orbit_direction", "")).lower() != want:
            continue
        w1 = _finished_window(o)
        if w1 is None or _days(w0[0], w1[0]) > WINDOW_TOLERANCE_DAYS or _days(w0[1], w1[1]) > WINDOW_TOLERANCE_DAYS:
            continue
        if [(n, _area_geometry(s)) for n, s in cr.parse_areas(o)] != areas:
            continue
        return {"config": str(c), "name": Path(c).stem.replace("config_", ""), "direction": want}
    return None


_STATUS: dict = {}           # project folder -> (time, status): the app asks every 2 s
STATUS_TTL_S = 30.0


def status_for(cfg, fresh: bool = False) -> dict | None:
    """What a project page says about combining: {"kind": "combined", ...
    pointer_for} when it is in a combined project, {"kind": "offer", ...
    partner_for} when it could be, else None. Kept STATUS_TTL_S (it reads
    every project's config); combining clears it."""
    key = str(Path(cfg.project_dir).resolve())
    hit = _STATUS.get(key)
    if hit and not fresh and time.time() - hit[0] < STATUS_TTL_S:
        return hit[1]
    ptr = pointer_for(cfg)
    if ptr:
        out = {"kind": "combined", **ptr}
    else:
        p = partner_for(cfg)
        out = {"kind": "offer", **p} if p else None
    _STATUS[key] = (time.time(), out)
    return out


def combine_with(config, partner) -> Path:
    """Write the combined config for a project and its partner (in ascending,
    descending order) and build it; returns the combined config's path."""
    from .config import load_config
    a, b = load_config(str(config)), load_config(str(partner))
    asc, desc = (config, partner) if str(a.orbit_direction).lower() == "ascending" else (partner, config)
    if str(b.orbit_direction).lower() == str(a.orbit_direction).lower():
        raise CombinedConfigError("both projects look from the same direction")
    path = write_config(asc, desc)
    run(path)
    return path


def brief(rec: dict | None) -> list[dict]:
    """One line per area of a combined report, for the project lists:
    {name, rate, range, total, reading, tone} (the app's area row)."""
    out = []
    for a in (rec or {}).get("areas") or []:
        v, ew = a.get("vertical") or {}, a.get("east_west") or {}
        if v.get("testable"):
            rate = f"{v['vertical_mm_yr'] / MM_IN:+.2f} in/yr vertical ({v['vertical_mm_yr']:+.1f} mm/yr)"
            lo, hi = v.get("range95_mm_yr") or (None, None)
            rng = f"{lo / MM_IN:+.2f} to {hi / MM_IN:+.2f} in/yr" if lo is not None else None
        else:
            rate, rng = "not testable", None
        e = ew.get("movement_mm_yr")
        total = (f"east-west {e / MM_IN:+.2f} in/yr ({e:+.1f} mm/yr), indicative" if e is not None else None)
        tone = ("warn" if v.get("detected") or ew.get("stands_out_indicative") else
                "muted" if not a.get("combined") or not v.get("testable") else "ok")
        out.append({"name": a.get("name", "?"), "rate": rate, "range": rng, "total": total,
                    "reading": a.get("reading", ""), "tone": tone})
    return out


def read_report(path) -> dict | None:
    try:
        return json.loads((load(path).deliverables / REPORT).read_text())
    except (OSError, ValueError, CombinedConfigError):
        return None


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m insar_pipeline.combine",
                                 description="Combine an ascending and a descending project into vertical and "
                                             "east-west movement.")
    ap.add_argument("config", nargs="?", help="a combined config (kind: combined)")
    ap.add_argument("--new", nargs=2, metavar=("ASC_CONFIG", "DESC_CONFIG"),
                    help="write configs/<name>_combined.yaml for two project configs (and stop)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.new:
        print(write_config(*a.new))
        return 0
    if not a.config:
        ap.error("give a combined config, or --new ASC DESC")
    out = run(a.config)
    rec = json.loads(Path(out).read_text())
    print(f"Combined view written to {out}")
    for r in rec["areas"]:
        print(f"  {r['name']}: {r['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
