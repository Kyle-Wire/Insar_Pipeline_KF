"""
A combined project (combine.py) in the results viewer (28 Oct 2026, Kyle:
"I'd also like the map results to be available for the combined results so
I can review and draw specific polygons").

The viewer reads one table per project (field_points: a point every ~100 m,
each point's figures and series). A combined project gets a table of the
same form, so the page, its points, its 15 m cells and its polygons work
unchanged:

  * the vertical slot holds the COMBINED vertical: `los_rel` = Up relative
    to the surrounding ground and `up` = 1, so every "LOS / up" the page
    computes is the combined vertical itself; `rate` adds the GNSS rate as
    the single-project table does;
  * east-west travels beside it (`east`, `series_east`); the page shows it
    where a single project shows line of sight, labelled east-west
    (indicative) - meta `combined` switches the words;
  * whole-record figures are the combined fields' (combine.combine_fields:
    the same numbers as the combined rasters and report);
  * each point's series: both projects' own report series for that ground
    (field_points.gather: the 3 x 3 cells against their farmland, and the
    point's own cell), the descending interpolated onto the ascending dates
    (decompose.interpolation_plan, both neighbours within one 24-day revisit; dates
    the descending pass cannot reach are left out), then Up and East solved
    date by date with each run's own look vector at that ground.

DATE WINDOWS ARE OFF for the combined view: the series are interpolated
between two passes, so their slopes do not reproduce the stored rates
exactly, and the viewer only measures windows when they do. The table has
no `cell_los` and the page says why (results_viewer).

Nothing here changes either source project.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SPACING_M = 100.0
# the descending neighbours a date is interpolated from: within one revisit.
# Sentinel-1 revisited every 24 days until 1B joined (late 2016): with 12
# days the series lost its first six months (the first site: 8 of 25 dates left out)
MAX_GAP_DAYS = 24
WINDOW_NOTE = ("Date windows are not available for the combined view: its series are interpolated between the "
               "two passes, so the figures are the whole record.")


def viewer_path(cc) -> Path:
    return Path(cc.project_dir) / "deliverables" / "viewer" / "field_points.npz"


def solve_series(asc, desc, epochs, asc_eu, desc_eu) -> tuple[np.ndarray, np.ndarray]:
    """(Up, East) series, (N, len(epochs)), zero at the first epoch.

    asc / desc: (N, T_asc) / (N, T_desc) relative LOS series (mm); epochs:
    [(ascending date index, descending interpolation plan)], the plan as
    decompose.interpolation_plan gives it; asc_eu / desc_eu: each run's
    (east, up) LOS components per row. LOS = east_c x East + up_c x Up, the
    two runs solved exactly (two equations, two unknowns)."""
    ae, au = (np.asarray(x, float) for x in asc_eu)
    de, du = (np.asarray(x, float) for x in desc_eu)
    La = np.stack([np.asarray(asc)[:, i] for i, _ in epochs], axis=1).astype(float)
    Ld = np.stack([sum(w * np.asarray(desc)[:, j] for j, w in plan) for _, plan in epochs], axis=1).astype(float)
    det = ae * du - de * au
    with np.errstate(invalid="ignore", divide="ignore"):
        E = (La * du[:, None] - Ld * au[:, None]) / det[:, None]
        U = (ae[:, None] * Ld - de[:, None] * La) / det[:, None]
    return (U - U[:, :1]).astype(np.float32), (E - E[:, :1]).astype(np.float32)


class Sources:
    """Both projects of a combined config, loaded once: the combined fields,
    each source's report context and cell maps, and the dates both reach."""

    def __init__(self, config):
        from . import combine
        from . import field_points as fp
        from . import significance as sg
        from .config import load_config
        self.config = str(config)
        self.cc = combine.load(config)
        a_cfg, d_cfg = load_config(str(self.cc.ascending)), load_config(str(self.cc.descending))
        self.src = [combine.load_source(a_cfg, "ascending"), combine.load_source(d_cfg, "descending")]
        self.F = combine.combine_fields(self.src)
        self.G = self.F["grid"]
        gv = [combine._gnss(s) for s in self.src]
        self.anchored = all(v[0] is not None for v in gv)
        self.bg_v = float(gv[0][0]) if self.anchored else 0.0
        self.bg_u = max(v[1] or 0.0 for v in gv) if self.anchored else None
        self.sd_up, self.sd_rows = sg.farmland_cell_spread(self.F["up"], self.F["tc"], self.F["far_wide"])
        self.kcls = sg.coherence_class(np.nan_to_num(self.F["tc"], nan=0.0))
        self.maps = [fp.cell_maps(s.cfg, s.rep, s.ctx) for s in self.src]
        self.ground, _ = combine.ground_label(a_cfg, self.G.attrs, self.F["far_wide"])
        # the dates: the ascending pass's, where the descending can be interpolated
        from . import decompose as dc
        da, dd = list(self.src[0].ctx.dates), list(self.src[1].ctx.dates)
        self.epochs = []
        for i, d in enumerate(da):
            plan = dc.interpolation_plan(dd, dc._d(d), MAX_GAP_DAYS)
            if plan is not None:
                self.epochs.append((i, plan))
        if len(self.epochs) < 3:
            raise RuntimeError("the two passes share too few dates for a series")
        self.dates = [da[i] for i, _ in self.epochs]
        self.t = sg.years(self.dates)
        self.t = self.t - self.t[0]
        self.span = float(self.t[-1])
        self.dropped_dates = [d for i, d in enumerate(da) if i not in {k for k, _ in self.epochs}]
        xs, ys = self.G.grid.centres()
        self.lon_c, self.lat_c = xs, ys

    # ---------------------------------------------------------------- cells
    def _on_source(self, k: int, rows, cols):
        """(rows, cols, ok) of the source cells under the common cells' centres."""
        s = self.src[k]
        a = s.attrs
        x0, y0, dx, dy = (float(a[n]) for n in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP"))
        lat, lon = self.lat_c[rows], self.lon_c[cols]
        r = np.floor((lat - y0) / dy).astype(int)
        c = np.floor((lon - x0) / dx).astype(int)
        ny, nx = s.good.shape
        ok = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
        rr, cc = np.clip(r, 0, ny - 1), np.clip(c, 0, nx - 1)
        ok &= s.ctx.good[rr, cc]
        return rr, cc, ok

    def usable(self, rows, cols) -> np.ndarray:
        rows, cols = np.asarray(rows, int), np.asarray(cols, int)
        ok = self.F["good"][rows, cols] & np.isfinite(self.F["up"][rows, cols])
        for k in range(2):
            ok &= self._on_source(k, rows, cols)[2]
        return ok

    def figures(self, rows, cols) -> dict:
        """Whole-record figures of common cells (the combined fields)."""
        rows, cols = np.asarray(rows, int), np.asarray(cols, int)
        up = self.F["up"][rows, cols].astype(float)
        east = self.F["east"][rows, cols].astype(float)
        sd = self.sd_up[rows, cols].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.where(sd > 0, np.abs(up) / sd, np.nan)
        return {"los_rel": up, "rel_v": up, "rate": up + self.bg_v, "east": east, "sd_v": sd, "snr": snr,
                "tc": self.F["tc"][rows, cols].astype(np.float32), "up": np.ones(rows.size),
                "change_los": up * self.span, "east_change": east * self.span,
                "unwrap": np.full(rows.size, np.nan, np.float32), "fast": np.zeros(rows.size, bool),
                "kcls": self.kcls[rows, cols].astype(np.int8),
                "lat": self.lat_c[rows], "lon": self.lon_c[cols]}

    def series(self, rows, cols) -> dict:
        """Up and East series (mm, zero at the first date) of usable common
        cells: from each source's 3 x 3 report series (`up3`, `east3`) and
        from its own cell (`up1`, `east1`)."""
        from . import field_points as fp
        rows, cols = np.asarray(rows, int), np.asarray(cols, int)
        per = []
        for k, s in enumerate(self.src):
            r, c, _ = self._on_source(k, rows, cols)
            m = self.maps[k]
            g = fp.gather(s.cfg, s.rep, s.ctx, [(r, c)], up_map=m["up"], null_map=m["null"], change=m["change"])
            per.append({"P": g["series_los"], "C": g["cell_los"],
                        "e": s.los_e[r, c].astype(float), "u": s.los_u[r, c].astype(float)})
        a, d = per
        out = {}
        for key, tag in (("P", "3"), ("C", "1")):
            U, E = solve_series(a[key], d[key], self.epochs, (a["e"], a["u"]), (d["e"], d["u"]))
            out["up" + tag], out["east" + tag] = U, E
        return out

    def table(self, rows, cols, with_series: bool = True) -> dict:
        """A viewer table (field_points' form) of the usable common cells."""
        rows, cols = np.asarray(rows, int), np.asarray(cols, int)
        keep = self.usable(rows, cols)
        rows, cols = rows[keep], cols[keep]
        fd = self.figures(rows, cols)
        fd.update({"row": rows.astype(np.int32), "col": cols.astype(np.int32), "group": np.zeros(rows.size, np.int16)})
        if with_series and rows.size:
            S = self.series(rows, cols)
            fd.update({"series_los": S["up3"], "series_east": S["east3"], "own_up": S["up1"], "own_east": S["east1"]})
        fd["meta"] = self.meta()
        return fd

    # ---------------------------------------------------------------- the table
    def meta(self) -> dict:
        from . import field_points as fp
        a0 = {}
        try:
            a0 = fp.load(fp.viewer_path(self.src[0].cfg))["meta"]
        except (OSError, ValueError, KeyError):
            pass
        far = self.F["far_wide"] & np.isfinite(self.F["up"]) & (self.sd_up > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            chance = float((np.abs(self.F["up"][far]) / self.sd_up[far] >= 1.96).mean() * 100) if far.any() else None
        return {"combined": True, "title": self.cc.name, "names": ["Study area"],
                "dates": list(self.dates), "t": [float(x) for x in self.t], "t_fit": [float(x) for x in self.t],
                "fit_exclude": [], "span": self.span, "anchored": self.anchored, "bg_v": self.bg_v,
                # the page says "ordinary <ground>": "ground (mostly built-up and grassland)"
                "bg_u": self.bg_u, "ground": self.ground.replace("the surrounding ", "", 1), "relative_colours": a0.get("relative_colours", True),
                "bins": a0.get("bins"), "lim_in": a0.get("lim_in"), "areas": a0.get("areas") or [], "features": [],
                "resolution": a0.get("resolution"), "spacing_m": SPACING_M,
                "snr_chance_pct": None if chance is None else round(chance, 1),
                "rep_cell_null_by_class": {str(r["class_index"]): r["farmland_cell_spread_mm_yr"] for r in self.sd_rows},
                "numbers_hash": "", "window_note": WINDOW_NOTE,
                "dates_left_out": self.dropped_dates,
                "sources": [{"direction": s.label, "project": Path(s.cfg.project_dir).name} for s in self.src],
                "east_west": "indicative: not a tested figure"}

    def points_table(self) -> dict:
        from . import field_points as fp
        px = math.sqrt(np.prod(self.G.grid.posting_m()))
        mask = self.F["good"].copy()
        rows, cols, spacing = fp.grid_cells(mask, px, SPACING_M)
        fd = self.table(rows, cols)
        fd["meta"]["spacing_m"] = spacing
        return fd


ARRAYS = ("row", "col", "group", "lat", "lon", "up", "los_rel", "rel_v", "rate", "sd_v", "snr", "tc", "unwrap",
          "fast", "change_los", "series_los", "kcls", "east", "east_change", "series_east")


def save(fd: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(part, meta=np.array(json.dumps(fd["meta"], default=str)), **{k: fd[k] for k in ARRAYS})
    part.replace(path)
    return path


def build(config) -> Path:
    """The combined project's viewer table (deliverables/viewer/field_points.npz)."""
    from . import combine
    S = Sources(config)
    fd = S.points_table()
    p = save(fd, viewer_path(S.cc))
    kmz = write_kmz(S.cc, fd)                     # the whole field as a KMZ (28 Oct 2026, Kyle)
    log.info("Combined KMZ: %s (%.1f MB)", kmz, kmz.stat().st_size / 1e6)
    log.info("Combined viewer table: %d points, %d dates (%d ascending dates left out: the descending pass is not "
             "within %d days either side)", fd["lat"].size, len(S.dates), len(S.dropped_dates), MAX_GAP_DAYS)
    combine._STATUS.clear()
    return p


# ---------------------------------------------------------------- exports (KMZ balloons)

def kmz_point_parts(fd: dict, j: int, name: str, bins, relative_colours: bool, start_year: float) -> dict:
    """Point j of a combined table as client_kmz.points_kml takes it: its
    style (the vertical's colour bin, faded within normal variation) and its
    balloon in point_description's layout (heading, rows, chart, footer), so
    the KMZ shares one balloon per style as every route's does."""
    import html
    from . import client_kmz as ck
    from . import field_points as fp
    m = fd["meta"]
    t = np.asarray(m["t"])
    anchored, bg_v, bg_u = bool(m.get("anchored")), float(m.get("bg_v") or 0.0), m.get("bg_u")
    lat, lon = float(fd["lat"][j]), float(fd["lon"][j])
    rate, rel, east = float(fd["rate"][j]), float(fd["rel_v"][j]), float(fd["east"][j])
    sd, snr, tc = float(fd["sd_v"][j]), float(fd["snr"][j]), float(fd["tc"][j])
    ground = m.get("ground") or "ground"
    u_ = math.hypot(sd, bg_u or 0.0) if np.isfinite(sd) else None
    chart = ck.svg_chart(t, fp.series_vertical_in(fd, j), f"{ck.latlon_text(lat, lon)}: vertical movement",
                         centre_in=(bg_v * t / ck.MM_IN) if anchored else None,
                         half_in=(ck.Z95 * u_ * t / ck.MM_IN) if u_ else None, start_year=start_year)
    i = lambda v: "—" if not np.isfinite(v) else f"{v / ck.MM_IN:+.2f}"                        # noqa: E731
    mm = lambda v: "" if not np.isfinite(v) else f" ({v:+.1f} mm)"                             # noqa: E731
    row = lambda a, b: f"<tr><td>{a}</td><td>{b}</td></tr>"                                    # noqa: E731
    rows = [row("Vertical, per year", f"<b>{i(rate)} in</b>{mm(rate)}"
                + (" with the regional GNSS motion" if anchored else "")),
            row(f"Compared with ordinary {html.escape(ground)}", f"{i(rel)} in/yr{mm(rel)}"),
            row("East-west, per year (indicative)", f"{i(east)} in/yr{mm(east)}; + = east"),
            row("Signal-to-noise ratio (vertical)",
                "—" if not np.isfinite(snr) else f"{snr:.1f} (1.96 or more stands out)"),
            row("Measurement quality", f"{tc:.2f} temporal coherence")]
    src = ", ".join(f"{s['project']} ({s['direction']})" for s in m.get("sources") or [])
    desc = (f"<b>{ck.POINT_TITLE}{ck.latlon_text(lat, lon)}</b><br>in {html.escape(name)}<br>"
            "<table style='font-size:12px'>" + "".join(rows) + "</table>" + chart
            + "<br><small>" + html.escape(
                f"Both passes together ({src}): east-west movement is separated out of the vertical; north-south "
                "is not measured. East-west is indicative, not a tested figure. Negative vertical = sinking.")
            + "</small>")
    col = rel if relative_colours else rate
    faded = not (np.isfinite(snr) and snr >= 1.96)
    return {"style": f"b{ck._bin(col / ck.MM_IN, bins)}{'f' if faded else ''}", "lat": lat, "lon": lon,
            "desc": desc, "fast": False}


def write_kmz(cc, fd: dict) -> Path:
    """The combined project's whole-field KMZ (deliverables/<name>_combined_DRAFT.kmz):
    every point of the viewer table, each with combine_viewer's balloon, through
    the client KMZ's shared-balloon writer; DRAFT because the combined vertical
    is not yet signed off as a client figure and east-west is indicative."""
    import html
    import zipfile
    from . import client_kmz as ck
    m = fd["meta"]
    bins = [(lo, hi, tuple(rgb), lab) for lo, hi, rgb, lab in (m.get("bins") or [])] or ck.point_bins(m.get("lim_in") or 0.6)
    y0 = ck.decimal_year(m["dates"][0])
    name = str(m.get("title") or "Combined view")
    styles, pms = ck.points_kml([kmz_point_parts(fd, j, name, bins, bool(m.get("relative_colours", True)), y0)
                                 for j in range(fd["lat"].size)], bins)
    src = ", ".join(f"{s['project']} ({s['direction']})" for s in m.get("sources") or [])
    kml = (f"<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
           f"<name>{html.escape(name)} - DRAFT</name><open>1</open>"
           f"<description><![CDATA[Vertical and east-west movement from both passes together ({html.escape(src)}), "
           f"{m['dates'][0]} to {m['dates'][-1]}. Colours: vertical per year. East-west is indicative, not a tested "
           f"figure; north-south is not measured. DRAFT: not yet signed off as client figures.]]></description>"
           f"{styles}<Folder><name>Measurement points ({fd['lat'].size}, about {m.get('spacing_m') or 100:.0f} m "
           f"apart)</name>{''.join(pms)}</Folder></Document></kml>")
    out = Path(cc.deliverables) / f"{Path(cc.project_dir).name}_DRAFT.kmz"
    part = out.with_name(out.name + ".partial")
    with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", ck._no_neg_zero(kml))
        z.writestr(ck.DOT, ck.dot_png())
    part.replace(out)
    return out
