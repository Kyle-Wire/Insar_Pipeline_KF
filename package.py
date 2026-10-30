"""The client package: what leaves the building, as a dated, checksummed copy.

Until 5 Oct 2026 nothing produced one. Site A's final_20260928/
(client/, data/, technical/, README.txt) was assembled by hand; Site B's
deliverables/ was one flat 22 GB folder in which the raw line-of-sight
raster (los_velocity_in_per_year.tif: ordinary farmland reads about -0.3
in/yr there, against +0.01 in the client raster) and the technical
summary_table.png sat beside the client report with nothing to tell them
apart. Rebuilds then overwrote everything in place, so the state a client
was sent could not be shown afterwards, and the hand-made package carried no
provenance at all.

    package.build(cfg, force_draft=False) -> Path

writes deliverables/releases/<YYYYmmdd_HHMM>/ with
    client/     numbered copies of the report, its figures, the KMZ and the
                per-area workbooks and charts
    data/       the GeoTIFFs in the report's own terms (vertical, relative to
                ordinary ground, GNSS-anchored when the report is)
    technical/  everything else: the JSON records, a fresh provenance.json
                (with the later stages), the config snapshot, the review,
                technical figures, and rasters/ for the raw line-of-sight
                and all-vertical-assumption products
    README.txt  generated from client_report.json
    MANIFEST.json  sha256 of every file + version.describe() + provenance
It refuses unless release.is_final(...) (force_draft writes <stamp>_DRAFT).

Files are COPIED, never hard-linked: html/json are rewritten in place by the
next rebuild, so a link would change the "delivered" copy (the Site A
data/ links survived only because the GeoTIFF writer recreates its files).
Unknown files go to technical/, never to client/. The package is built in a
hidden folder, checked (no raw product in client/ or data/, every manifest
hash re-read, no file sharing an inode with deliverables/), then renamed into
place and made read-only. scripts/cleanup.py must never touch releases/.
"""
import contextlib
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


def _sensor(backend) -> str:
    """'NISAR' or 'Sentinel-1' (config.sensor_name)."""
    from .config import sensor_name
    return sensor_name(str(backend or ""))


# Files that never go into a release: regenerable and huge (the EGMS
# CSV/GPKG are 8-18 GB, the MintPy time-series KMZ 3.8 GB on Site B), or
# working state. They are listed in the README by path and size instead.
NEVER_COPY = (re.compile(r"(^|/)measurement_points_egms\.(csv|gpkg)$"),
              re.compile(r"(^|/)points_full.*\.csv$"),
              re.compile(r"(^|/)geo_timeseries.*\.kmz$"),
              re.compile(r"\.(h5|tmp|partial|lock)$"),
              # a write in progress (x.partial.gpkg) or one that failed its own
              # read-back check (x.gpkg.INVALID, deliverables._write_points_gpkg)
              re.compile(r"\.partial\.[^/]+$|\.INVALID$"))
# Folders under deliverables/ that are other copies, not this build.
# viewer/: the results viewer's working copy of the points (field_points.npz), rebuilt with the KMZ
SKIP_DIRS = re.compile(r"^(releases|final_.*|superseded.*|package_.*|client_package.*|viewer|\..*)$")
TECH_COPY_CAP_BYTES = 200e6        # a technical file larger than this is listed, not copied

# What may never sit in client/ or data/: line-of-sight or all-vertical
# products, the technical table and figures, the point tables.
FORBIDDEN_CLIENT = re.compile(r"(^|/)(los_|.*ASSUMES_VERTICAL|summary_table|velocity_relative|cumulative_relative|"
                              r"measurement_points|geo_timeseries|points_attributes|points_timeseries|points\.csv)")

# data/: the rasters in the report's own terms, each with a one-line meaning.
DATA_FILES = (
    (re.compile(r"^vertical_movement_in_per_yr\.tif$"),
     "each cell's vertical movement per year (in/yr), relative to ordinary ground of the same kind"),
    (re.compile(r"^vertical_movement_uncertainty_1sigma_in_per_yr\.tif$"),
     "that cell's uncertainty (1 sigma, in/yr; x2 for a 95% range)"),
    (re.compile(r"^vertical_change_at_average_rate_in_\d{8}_\d{8}\.tif$"),
     "each cell's vertical change at its average rate (average rate × years, in) - not the measured change; "
     "the report's area totals are the measured change"),
    # the same raster under its name before 6 Oct 2026 (a report not rebuilt since)
    (re.compile(r"^vertical_movement_total_in_\d{8}_\d{8}\.tif$"),
     "each cell's vertical change at its average rate (average rate × years, in; older file name) - not the "
     "measured change; the report's area totals are the measured change"),
    (re.compile(r"^rate_signal_to_noise\.tif$"), "movement / noise for each cell"),
    # line of sight in the report's frame (8 Oct 2026); the raw los_* stay technical
    (re.compile(r"^line_of_sight_velocity_in_per_yr\.tif$"),
     "each cell's movement per year along the satellite's line of sight (in/yr), relative to ordinary ground of "
     "the same kind; + = towards the satellite; not converted to vertical"),
    (re.compile(r"^line_of_sight_cumulative_displacement_in_\d{8}_\d{8}\.tif$"),
     "each cell's line-of-sight change from the first image to the last (in), relative to ordinary ground of the "
     "same kind; + = towards the satellite; a two-date difference, so each date's noise is in it"),
    (re.compile(r"^quality_temporal_coherence\.tif$"),
     "how consistently each cell was measured (temporal coherence, 0-1; every imaged cell, including those "
     "below the threshold the figures use)"),
    (re.compile(r"^quality_unwrapping_consistency\.tif$"),
     "how well each cell's interferograms agree with its time series (0-1; the second quality check)"),
    (re.compile(r"^quality_too_fast_to_measure\.tif$"),
     "1 where the ground changes too fast between neighbouring cells for this satellite to measure"),
    (re.compile(r"^quality_position_uncertainty_m\.tif$"),
     "how far an elevation-model error can move each cell on the map (m)"),
    (re.compile(r"^quality_.*\.tif$"), "measurement quality layer (see the report)"),
    (re.compile(r"^disturbed_.*\.tif$"),
     "per cell, the longest stretch of consecutive images that look like noise (-1 = no data): a diagnostic, "
     "not a map of disturbed ground on its own"),
)
# technical/: named so the README says what each is (anything else is 'unclassified')
TECH_FILES = (
    (re.compile(r"^points_attributes\.gpkg$"),
     "every measured point (EPSG:4326), EGMS-style fields in mm and mm/yr, line of sight; join on pid"),
    (re.compile(r"^points_timeseries\.parquet$"),
     "each point's displacement series (GeoParquet, mm from its first valid date, line of sight)"),
    (re.compile(r"^points_README\.txt$"), "fields, units, sign, CRS and credits of the point products"),
    # E1 (ledger #1, 7 Oct 2026): it said 'ground during the record minus the
    # elevation model', but MintPy's estimate follows the movement rate
    (re.compile(r"^dem_difference_m\.tif$"),
     "MintPy's elevation-model error estimate: the height the inversion gives each cell minus the elevation "
     "model (m); it also takes up part of the ground movement, so it is not a map of reshaped ground on its own"),
)
REQUIRED_DATA = "vertical_movement_in_per_yr.tif"


class PackageError(RuntimeError):
    pass


def _load(p: Path):
    try:
        return json.loads(Path(p).read_text()) if Path(p).exists() else None
    except (OSError, ValueError):
        return None


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_") or "area"


def _title(cfg, rep) -> str:
    t = (rep or {}).get("title")
    if not t:
        try:
            from .report_insights import identity
            t = identity(cfg)["title"]
        except Exception:                                     # noqa: BLE001
            t = Path(cfg.project_dir).name
    return _safe(t)


def _kmz_numbers_hash(path: Path) -> str | None:
    """The numbers_hash a client KMZ embeds in doc.kml (client_kmz.build), or None."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            m = re.search(r"<Data name='numbers_hash'><value>([0-9a-f]+)</value>",
                          z.read("doc.kml").decode("utf-8", "replace"))
        return m.group(1) if m else None
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


def _pdf_problem(pdf: Path, rep_t: float | None, final: bool) -> str | None:
    """Why the area-summaries PDF is not this report's, or None: older than
    client_report.json (the PDF failed on the last build and the one before
    stayed), or, in a final package, still marked DRAFT in its metadata."""
    try:
        if rep_t is not None and pdf.stat().st_mtime < rep_t:
            return "older than the report (it was not rewritten on the last build); rebuild the report"
        if final and b"/Keywords (DRAFT)" in pdf.read_bytes():
            return "rendered as a DRAFT; rebuild the report now that it is signed off"
    except OSError:
        return "cannot be read; rebuild the report"
    return None


def _sheets(path: Path) -> list[str] | None:
    """The sheet names of a workbook (read-only open: the Points sheet is
    not read), or None when it cannot be opened."""
    try:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True)
        try:
            return list(wb.sheetnames)
        finally:
            wb.close()
    except Exception:                                         # noqa: BLE001 - unreadable = not sent
        return None


def _results_problem(books: list[Path], rep_t: float | None, summ_hash: str | None = None,
                     nh: str | None = None) -> str | None:
    """Why a Results export may not go to the client, or None. A workbook
    older than client_report.json carries figures from an earlier report
    (Site B's 29 Sep workbook beside the 6 Oct report), and one without
    the Notes sheet is from before the caveats were always written - the
    truncated 3-sheet file an interrupted export left (techout-2). An
    export whose summary.json records the report's numbers_hash is judged
    by that instead of its age: the sign-off rebuild rewrites
    client_report.json with the same numbers, which left every workbook
    exported before signing out of the final package (7 Oct 2026)."""
    for b in books:
        if summ_hash:
            if summ_hash != nh:
                return "exported from an earlier version of the report's numbers; export the area again (Results, v)"
        elif rep_t is not None and b.stat().st_mtime < rep_t:
            return "exported before the report was last rebuilt; export the area again (Results, v)"
        sh = _sheets(b)
        if sh is None:
            return "the workbook cannot be opened; export the area again"
        if "Notes" not in sh:
            return ("the workbook has no Notes sheet (an export from before 5 Oct 2026 or one that did not "
                    "finish); export the area again")
    return None


def _workbook_meaning(summ: dict, books: list[Path]) -> str:
    """The workbook's meaning from what it holds: a large area's export
    leaves the per-point table out (results.XLSX_MAX_POINTS; points.csv
    beside it), so it is not 'every measured point'."""
    pts = summ.get("points_sheet")
    if pts is None and books:
        sh = _sheets(books[0]) or []
        pts = "Points" in sh
    if pts:
        return "every measured point in the area, with statistics and yearly rates"
    return ("the area's statistics and yearly rates (too many points for a workbook: the per-point table is "
            "not included)")


def _stale_pair_figures(cfg, D: Path) -> dict:
    """{name: why} for the interferogram figures (figures/winter_pair.png,
    summer_pair.png) that show another kind of pair than this record's
    route builds on - figures._carry_pair_figures moves them to
    deliverables/superseded/ the next time the figures are drawn, and a
    package made before then left them in technical/figures/ under captions
    saying the record is built from them (Site A's raw ISCE pairs of 28
    Sep on its phase-linked record; final check of 7 Oct 2026)."""
    from . import figures as fg
    names = [n for n in fg.PAIR_FIGURES if (D / "figures" / n).exists()]
    if not names:
        return {}
    want = fg.PAIR_KIND_LINKED if fg._route(cfg) == "miaplpy" else fg.PAIR_KIND_RAW
    try:
        man = json.loads((D / "figures" / "figures.json").read_text())
    except (OSError, ValueError):
        man = []
    entries = {m.get("file"): m for m in man if isinstance(m, dict)} if isinstance(man, list) else {}
    out = {}
    for n in names:
        kind = fg._pair_kind(entries[n]) if n in entries else None
        if kind != want:
            out[n] = (f"shows {kind or 'an unrecorded kind of'} interferograms, not this record's {want} pairs; it "
                      "moves to deliverables/superseded/ when the figures are next drawn")
    return out


def _plan(cfg, rep: dict, draft: bool) -> tuple[list, list, list]:
    """(copies [(src, dest_rel, role, meaning)], not_copied [(src, bytes, why)],
    unclassified [dest_rel]) for everything under deliverables/."""
    D = Path(cfg.project_dir) / "deliverables"
    title = _title(cfg, rep)
    copies, not_copied, unclassified, claimed = [], [], [], set()

    def take(src: Path, dest: str, role: str, meaning: str = ""):
        copies.append((src, dest, role, meaning))
        claimed.add(src.resolve())

    # client/ - numbered in reading order
    n = 1
    report = D / "client_report.html"
    if draft and not report.exists() and (D / "client_report_DRAFT.html").exists():
        report = D / "client_report_DRAFT.html"
    if report.exists():
        take(report, f"client/{n}_{title}_ground_movement_report{'_DRAFT' if draft else ''}.html", "client",
             "the plain-language report (open in any browser; self-contained)")
        n += 1
    for fig, name, meaning in (("client_summary.png", "areas_summary.png", "the report areas in one table"),
                               ("client_map.png", "map_vertical_movement.png", "map of vertical movement"),
                               ("client_series.png", "movement_over_time.png", "the areas' movement over time")):
        p = D / "figures" / fig
        if p.exists():
            take(p, f"client/{n}_{name}", "client", meaning)
            n += 1
    rep_f = D / "client_report.json"
    rep_t = rep_f.stat().st_mtime if rep_f.exists() else None
    pdf = D / "client_area_summaries.pdf"            # client_report.PDF_NAME, one page per area
    if pdf.exists():
        # client_report.build writes the PDF just after client_report.json and
        # only logs a failure, so an older PDF (or one still marked DRAFT) is
        # the previous build's: renamed without DRAFT it went into a FINAL
        # package's client/ (7 Oct 2026)
        why = _pdf_problem(pdf, rep_t, final=not draft)
        if why:
            not_copied.append((pdf, pdf.stat().st_size, why))
            claimed.add(pdf.resolve())
        else:
            take(pdf, f"client/{n}_{title}_area_summaries{'_DRAFT' if draft else ''}.pdf", "client",
                 "one page per report area, for presenting")
            n += 1
    from . import release
    nh = release.numbers_hash(rep)
    for k in sorted(D.glob("*_ground_movement*.kmz")):
        if "DRAFT" in k.name and not draft:
            # a draft-marked KMZ never goes into a final package (nor into technical/ as 'unclassified')
            not_copied.append((k, k.stat().st_size, "a draft KMZ (a final package takes the one built after sign-off)"))
            claimed.add(k.resolve())
            continue
        # the KMZ carries the numbers it was built from (doc.kml numbers_hash);
        # one built before the report was last rebuilt states other figures -
        # in a draft package too (Site B's hash-less 29 Sep KMZ went into a
        # draft's client/ under a name without DRAFT)
        if _kmz_numbers_hash(k) != nh:
            not_copied.append((k, k.stat().st_size, "built from an earlier version of the report's numbers; rebuild the KMZ"))
            claimed.add(k.resolve())
            continue
        take(k, f"client/{n}_{title}_ground_movement_Google_Earth{'_DRAFT' if 'DRAFT' in k.name else ''}.kmz",
             "client", "open in Google Earth: the areas, measurement points and the map overlay")
        n += 1
    for rd in sorted(p for p in (D / "results").glob("*") if p.is_dir()) if (D / "results").exists() else []:
        summ = _load(rd / "summary.json") or {}
        s = summ.get("area_summary") or {}
        area = _safe(s.get("area") or rd.name)
        span = f"{s.get('first_image', '')} to {s.get('last_image', '')}".strip(" to")
        books = sorted(rd.glob("*.xlsx"))
        why = _results_problem(books, rep_t, summ.get("numbers_hash"), nh)
        if why:
            # the whole export stays out of client/: its charts are as old as its workbook
            for p in books + sorted(rd.glob("trend.png")) + sorted(rd.glob("chart.png")):
                not_copied.append((p, p.stat().st_size, why))
                claimed.add(p.resolve())
            continue
        for pat, name, meaning in (("*.xlsx", "points_yearly_rates_and_statistics.xlsx", _workbook_meaning(summ, books)),
                                   ("trend.png", "trend_graph.png", "the area's measured movement over the window"),
                                   ("chart.png", "yearly_rates_chart.png", "the points' spread and the yearly means")):
            for p in sorted(rd.glob(pat)):
                take(p, f"client/{n}_{area}_{name}", "client", meaning + (f" ({span})" if span else ""))
                n += 1

    # data/
    for p in sorted(D.glob("*.tif")):
        for pat, meaning in DATA_FILES:
            if pat.match(p.name):
                take(p, f"data/{p.name}", "data", meaning)
                aux = p.with_name(p.name + ".aux.xml")
                if aux.exists():
                    take(aux, f"data/{aux.name}", "data", "GIS display settings for the raster above")
                break

    # technical/ - the records first (some live outside deliverables/)
    P = Path(cfg.project_dir)
    if (P / "quality_assessment.json").exists():
        take(P / "quality_assessment.json", "technical/quality_assessment.json", "technical", "the quality assessment")
    stale_pairs = _stale_pair_figures(cfg, D)
    for p in sorted(D.rglob("*")):
        if not p.is_file() or p.resolve() in claimed:
            continue
        rel = p.relative_to(D)
        if (SKIP_DIRS.match(rel.parts[0]) and len(rel.parts) > 1) or rel.parts[0].startswith("."):
            continue
        if rel.parts[0] == "figures" and len(rel.parts) == 2 and rel.name in stale_pairs:
            not_copied.append((p, p.stat().st_size, stale_pairs[rel.name]))
            continue
        if re.match(r"client_report.*\.html$", rel.name):
            not_copied.append((p, p.stat().st_size, "another render of the report (the packaged one is in client/)"))
            continue
        if rel.name in ("provenance.json", "config_snapshot.yaml"):
            continue                                  # a fresh copy is written below
        rel_s = rel.as_posix()
        size = p.stat().st_size
        if any(r.search(rel_s) for r in NEVER_COPY):
            not_copied.append((p, size, "regenerable and large; stays in deliverables/"))
            continue
        if size > TECH_COPY_CAP_BYTES:
            not_copied.append((p, size, f"over {TECH_COPY_CAP_BYTES / 1e6:.0f} MB; stays in deliverables/"))
            continue
        tech = next((m for pat, m in TECH_FILES if pat.match(rel.name)), None) if len(rel.parts) == 1 else None
        if re.match(r"^(los_|vertical_velocity_.*ASSUMES_VERTICAL)", rel.name) and p.suffix in (".tif", ".xml"):
            dest, meaning = f"technical/rasters/{rel.name}", "RAW: line of sight against one reference pixel"
        elif tech is not None:
            dest, meaning = (f"technical/{'rasters' if rel.suffix == '.tif' else 'points'}/{rel.name}", tech)
        elif rel.suffix == ".json" or rel.parts[0] in ("figures", "results", "reviews", "experiments") \
                or rel.name.startswith("los_") or rel.suffix in (".csv", ".png", ".xml"):
            dest, meaning = f"technical/{rel_s}", ""
        else:
            dest, meaning = f"technical/{rel_s}", "unclassified"
            unclassified.append(dest)
        take(p, dest, "technical", meaning)
    return copies, not_copied, unclassified


def _raw_median_in_yr(path: Path):
    """The median cell of a raw LOS rate raster (in/yr), read decimated: the
    number the README quotes so 'do not send' carries the size of the error."""
    try:
        import numpy as np
        import rasterio
        with rasterio.open(path) as ds:
            a = ds.read(1, out_shape=(max(1, ds.height // 8), max(1, ds.width // 8)), masked=True).astype(float)
        v = a.compressed() if hasattr(a, "compressed") else a.ravel()
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else None
    except Exception:                                         # noqa: BLE001
        return None


# The technical record the report's footer names (client_report.
# TECHNICAL_RECORD_NOTE: "A technical record (quality assessment, GNSS tie,
# provenance) is kept with the project ..."), by their packaged names
TECHNICAL_RECORD = (("quality assessment", "technical/quality_assessment.json"),
                    ("GNSS tie", "technical/gnss_tie.json"),
                    ("provenance", "technical/provenance.json"))


def _page_record_note() -> str:
    """The page's own sentence about its technical record, so the README
    follows the page if its wording changes."""
    try:
        from .client_report import TECHNICAL_RECORD_NOTE
        return str(TECHNICAL_RECORD_NOTE)
    except (ImportError, AttributeError):
        return ("A technical record (quality assessment, GNSS tie, provenance) is kept with the project; the "
                "preparer can answer questions about any figure from it.")


def _record_note_lines(copies) -> list[str]:
    """What the page says about its technical record, and what the preparer
    does with it. The README said technical/ was 'do NOT send to the client'
    while the page offered that record 'on request' (final verification,
    7 Oct 2026); the page now says the record is kept with the project and
    the preparer answers questions from it, and the README quotes that
    sentence and names the files. Should the page offer the record on
    request again, the README says what to send - those records only."""
    import textwrap
    have = {d for _s, d, _r, _m in copies} | {"technical/provenance.json"}     # provenance is written fresh
    named = [f"{p.split('/', 1)[1]} ({what})" for what, p in TECHNICAL_RECORD if p in have]
    if not named:
        return []
    note = _page_record_note()
    files = ", ".join(named[:-1]) + (" and " if len(named) > 1 else "") + named[-1]
    if "on request" in note.lower():
        todo = (f"When the client asks, send {files} from technical/ with a covering note; never the raw rasters "
                "or the technical figures, which are not in the report's terms.")
    else:
        todo = (f"Answer the client's questions from {files} in technical/; they stay with the project and are "
                "not sent.")
    return textwrap.wrap(f'The report says: "{note}" {todo}', 98, initial_indent="  ", subsequent_indent="  ")


def _revision_lines(cfg, rep) -> list[str]:
    """The reissue note (decision M3, 7 Oct 2026: report_revision = {number,
    supersedes, changes}) as the page prints it, so the package README says
    which issue this replaces and what changed. [] for a first issue.
    Read with config.report_revision, the one reader the page, the PDF and
    the KMZ use, so the heading is theirs word for word ('replacing the
    report issued 28 Sep 2026', not the config's '2026-09-28'). The report's
    record comes first: it is what the packaged page printed."""
    from .config import report_revision as note
    rv = None
    for src in ((rep or {}).get("report_revision"), getattr(cfg, "report_revision", None)):
        if isinstance(src, dict) and src:
            rv = note(src)
            if rv:
                break
    if not rv or not rv.get("heading"):
        return []
    changes = [str(c).strip() for c in (rv.get("changes") or []) if str(c).strip()]
    return [rv["heading"] + (". What changed:" if changes else ".")] + [f"  - {c}" for c in changes] + [""]


def _readme(cfg, rep, qa, copies, not_copied, unclassified, rel_status, final, prov) -> str:
    from . import credits as cr
    from . import version
    P = Path(cfg.project_dir)
    d0, d1 = (list(rep.get("dates") or []) + ["", ""])[:2]
    fmt = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(str(d)) == 8 else str(d)    # noqa: E731
    title = rep.get("title") or P.name
    lines = [f"{title} ground movement - {'FINAL' if final else 'DRAFT, NOT FOR THE CLIENT'} package, "
             f"{datetime.now():%d %B %Y %H:%M}",
             f"Source: {rep.get('location', '')} - {_sensor(rep.get('backend', ''))} track {rep.get('track', '?')} "
             f"{str(rep.get('orbit', '')).lower()}, {rep.get('n_dates', '?')} images, {fmt(d0)} to {fmt(d1)}; "
             f"route: {rep.get('route') or (prov or {}).get('route') or rep.get('backend', '?')}.",
             f"Built by {version.label()}.", ""]
    lines += _revision_lines(cfg, rep)
    rev = rel_status.get("review")
    if final and rev:
        lines += [f"Reviewed and signed off by {rev.get('reviewer')} on {str(rev.get('date', ''))[:10]}"
                  + (f": {rev['notes']}" if rev.get("notes") else "."), ""]
    if not final:
        lines += ["THIS IS A DRAFT. It has not passed the release check. Before anything goes to the client:"]
        lines += [f"  - {b['text']}" for b in rel_status.get("blockers", []) if b["hard"]]
        lines += [""]
    soft = [b for b in rel_status.get("blockers", []) if not b["hard"]]
    if soft:
        lines += ["For the reviewer (does not stop the release):"] + [f"  - {b['text']}" for b in soft] + [""]
    areas = [a for a in rep.get("areas", []) or [] if isinstance(a, dict)]
    if areas:
        lines += ["Report areas (the figures are in the report; these lines are for finding them):"]
        for a in areas:
            v = a.get("vertical_mm_yr")
            rng = a.get("range95_mm_yr")
            txt = f"  {a.get('name')}: " + (f"{v / 25.4:+.2f} in/yr ({v:+.1f} mm/yr) vertical" if v is not None else "no figure")
            if rng and len(rng) == 2 and None not in rng:
                txt += f", 95% range {rng[0] / 25.4:+.2f} to {rng[1] / 25.4:+.2f} in/yr"
            if a.get("reading"):
                txt += f" - {a['reading']}"
            lines.append(txt)
        lines.append("")

    def block(role, head):
        rows = [(d, m) for _s, d, r, m in copies if r == role and not d.endswith(".aux.xml")]
        if not rows:
            return []
        w = max(len(Path(d).name) for d, _ in rows)
        return [head] + [f"  {Path(d).name.ljust(w)}  {m}".rstrip() for d, m in rows] + [""]
    lines += block("client", "client/      what goes to the client")
    lines += block("data", "data/        GeoTIFFs for GIS (QGIS/ArcGIS), in the report's own terms - "
                           "the numbers match the report and the KMZ")
    if any(r == "data" for *_x, r, _m in copies):
        lines += ["  Single cells are noisy; areas average that down, as the report's ranges show.", ""]
    lines += ["technical/   for specialists - not part of what is sent to the client",
              "  The JSON records (quality assessment, GNSS tie, client_report.json, provenance.json with the",
              "  software, settings and later stages, config_snapshot.yaml = the resolved settings), the review",
              "  record, technical figures (line of sight, relative to farmland, every mine: summary_table.png",
              "  is NOT the client's figures), and the per-area tables."]
    lines += _record_note_lines(copies)
    raw = [s for s, d, r, m in copies if d.startswith("technical/rasters/") and s.suffix == ".tif"]
    if raw:
        med = next((_raw_median_in_yr(s) for s in raw if s.name == "los_velocity_in_per_year.tif"), None)
        lines += ["", "technical/rasters/   RAW products: line of sight, against the one reference pixel, with the",
                  "  coherence-dependent farmland drift included"
                  + (f" - the median cell reads {med:+.2f} in/yr in los_velocity_in_per_year.tif" if med is not None else "")
                  + ".", "  Do NOT send these to the client or read report numbers off them:"]
        lines += [f"    {s.name}" for s in raw]
    if unclassified:
        lines += ["", "Unclassified (new files the packager does not know; put in technical/ - check before use):"]
        lines += [f"  {u}" for u in unclassified]
    if not_copied:
        lines += ["", "Left in deliverables/ (too large or regenerable to copy into every release):"]
        lines += [f"  {s.relative_to(P / 'deliverables')}  ({b / 1e6:,.0f} MB) - {why}" for s, b, why in not_copied]
    lines += ["", "MANIFEST.json lists every file with its sha256; a changed hash means a changed file.", ""]
    try:
        lines += [cr.text_block(cfg, rep, prov)]
    except Exception as exc:                                  # noqa: BLE001
        lines += [f"Data sources and credits: not generated ({exc})"]
    return "\n".join(lines) + "\n"


def _verify(root: Path, manifest: dict, sources: dict, final: bool) -> list[str]:
    """Re-read what was written and check it, rather than trusting the copy loop."""
    problems = []
    for sub in ("client", "data"):
        for p in (root / sub).rglob("*") if (root / sub).exists() else []:
            rel = p.relative_to(root).as_posix()
            if FORBIDDEN_CLIENT.search(rel.split("/", 1)[1]):
                problems.append(f"{rel} is a technical/raw product and must not be in {sub}/")
            if final and "DRAFT" in p.name:
                problems.append(f"{rel} is draft-marked in a final package")
    if not list((root / "client").glob("*_ground_movement_report*.html")):
        problems.append("client/ has no report")
    if not (root / "data" / REQUIRED_DATA).exists():
        problems.append(f"data/{REQUIRED_DATA} is missing")
    for need in ("technical/client_report.json", "technical/provenance.json", "README.txt"):
        if not (root / need).exists():
            problems.append(f"{need} is missing")
    for e in manifest["files"]:
        p = root / e["path"]
        if not p.exists():
            problems.append(f"{e['path']} listed but absent")
            continue
        if _sha256(p) != e["sha256"]:
            problems.append(f"{e['path']} does not match its manifest hash")
        st = p.stat()
        src = sources.get(e["path"])
        if st.st_nlink != 1 or (src is not None and src.exists() and os.stat(src).st_ino == st.st_ino):
            problems.append(f"{e['path']} shares its data with deliverables/ (a link, not a copy)")
    return problems


def build(cfg, force_draft: bool = False) -> Path:
    """Write deliverables/releases/<YYYYmmdd_HHMM>/ (or <stamp>_DRAFT with
    force_draft). Raises PackageError when a run holds the project, when the
    report is not final (and force_draft is False), when a required file is
    missing, or when the written package fails its own check; returns the
    folder otherwise.

    The whole build holds the project lock (runs.project_lock, kind
    'package'): a 'Products only' run rewrites the rasters and figures in
    place while the signed client_report.json stays final, so a package
    built meanwhile could copy half-written files and hash them into a
    FINAL manifest (7 Oct 2026)."""
    from . import runs
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(runs.project_lock(Path(cfg.project_dir), kind="package"))
        except RuntimeError as exc:
            raise PackageError(f"No package was built: a run is using this project ({exc}). Wait for it to "
                               "finish, then package.") from None
        out = _build(cfg, force_draft)
    # A FINAL package means the project is delivered: keep its results and
    # free the processing files (config auto_keep_results; 13 Oct 2026).
    # Outside the lock, which keep_results_only takes itself; never raises.
    if not out.name.endswith("_DRAFT"):
        from .storage import auto_keep_results
        auto_keep_results(cfg, "its FINAL client package was built")
    return out


def _build(cfg, force_draft: bool) -> Path:
    from . import provenance, release, version
    from .config import is_experiment
    P = Path(cfg.project_dir)
    D = P / "deliverables"
    if is_experiment(cfg):
        raise PackageError("No package was built: this project is a validation experiment (purpose: experiment). "
                           f"Its result is the validation against ground truth in {P / 'validation'}, not a client "
                           "package.")
    rep = _load(D / "client_report.json")
    if not rep:
        raise PackageError(f"no client report to package: {D / 'client_report.json'}")
    qa = _load(P / "quality_assessment.json")
    st = release.status(cfg, rep, qa)
    final = release.is_final(cfg, rep, qa)
    if not final and not force_draft:
        hard = [b["text"] for b in st["blockers"] if b["hard"]]
        raise PackageError("The report is not final, so no client package was built:\n  - " + "\n  - ".join(hard)
                           + "\n(build a draft package with force_draft=True)")
    if final:
        # The HTML must be the build that was signed: rendered after sign-off,
        # without the DRAFT mark. A report signed after it was rendered still
        # carries the watermark and must be rebuilt first.
        rel = rep.get("release") or {}
        html = D / "client_report.html"
        if not html.exists():
            raise PackageError(f"{html} is missing (rebuild the client report)")
        if (rel and rel.get("final") is False) or "DRAFT" in html.read_text(errors="replace"):
            raise PackageError("client_report.html was rendered as a DRAFT; rebuild the client report now that "
                               "it is signed off, then package")
    if not (D / REQUIRED_DATA).exists():
        raise PackageError(f"{D / REQUIRED_DATA} is missing; the data/ folder cannot be built")

    stamp = datetime.now().strftime("%Y%m%d_%H%M") + ("" if final else "_DRAFT")
    rels = D / "releases"
    rels.mkdir(parents=True, exist_ok=True)
    name, k = stamp, 2
    while (rels / name).exists():
        name, k = f"{stamp}_{k}", k + 1
    tmp = rels / f".{name}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        copies, not_copied, unclassified = _plan(cfg, rep, draft=not final)
        if final and not any(d.startswith("client/") and d.endswith(".kmz") for _s, d, _r, _m in copies):
            # every client report has its Google Earth file (pipeline.run_client_report)
            raise PackageError("There is no client KMZ for these numbers (it is missing, or built from earlier "
                               "numbers); rebuild the client report and KMZ, then package")
        sources = {}
        for src, dest, _role, _m in copies:
            out = tmp / dest
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)              # a copy: new inode, never a hard link
            sources[dest] = src
        # A fresh provenance record with the later stages, and the resolved config.
        prov = provenance.collect(cfg)
        prov["stages"] = provenance.stages(cfg)
        for key in ("disp_settings", "stack_settings", "miaplpy_settings", "miaplpy_outcome",
                    "stack_outcome", "unwrap", "records"):
            if prov.get(key) is None:
                prov.pop(key, None)
        (tmp / "technical").mkdir(exist_ok=True)
        (tmp / "technical" / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
        snap, snap_sha = provenance.config_snapshot(cfg)
        (tmp / "technical" / "config_snapshot.yaml").write_text(snap)
        st_out = {**st, "review": release.review_record(P)}
        (tmp / "README.txt").write_text(_readme(cfg, rep, qa, copies, not_copied, unclassified, st_out, final, prov))

        files = []
        for p in sorted(x for x in tmp.rglob("*") if x.is_file()):
            rel_p = p.relative_to(tmp).as_posix()
            role = rel_p.split("/", 1)[0] if "/" in rel_p else "readme"
            files.append({"path": rel_p, "bytes": p.stat().st_size, "sha256": _sha256(p), "role": role,
                          "source": str(sources[rel_p]) if rel_p in sources else "generated"})
        manifest = {"created": datetime.now().isoformat(timespec="seconds"), "project": str(P),
                    "release": name, "final": final, "numbers_hash": release.numbers_hash(rep),
                    "report_generated": rep.get("generated"), "review": release.review_record(P),
                    "blockers": st["blockers"], "version": version.describe(cfg),
                    "config_snapshot_sha256": snap_sha,
                    "provenance": {"file": "technical/provenance.json",
                                   "sha256": _sha256(tmp / "technical" / "provenance.json"),
                                   "route": prov.get("route"), "inputs": {k: v for k, v in prov.get("inputs", {}).items()
                                                                          if not isinstance(v, list)},
                                   "software": prov.get("software")},
                    "not_copied": [{"path": str(s), "bytes": b, "why": w} for s, b, w in not_copied],
                    "unclassified": unclassified, "files": files}
        (tmp / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str))
        problems = _verify(tmp, manifest, sources, final)
        if problems:
            raise PackageError("The package failed its own check:\n  - " + "\n  - ".join(problems))
        final_dir = rels / name
        os.replace(tmp, final_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    # Read-only files: a release is a record. The folders stay writable so a
    # draft can still be deleted by hand.
    for p in final_dir.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    return final_dir


def main(argv=None):
    import argparse
    from .config import load_config
    ap = argparse.ArgumentParser(description="Build the client package for a project.")
    ap.add_argument("config")
    ap.add_argument("--draft", action="store_true", help="build a DRAFT package even though the report is not final")
    a = ap.parse_args(argv)
    try:
        out = build(load_config(a.config), force_draft=a.draft)
    except PackageError as exc:
        print(exc)
        return 1
    print(f"Package: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
