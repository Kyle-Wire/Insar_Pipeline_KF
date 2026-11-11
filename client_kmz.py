"""A Google Earth file for the client: <project>/deliverables/<Title>_ground_movement[_SCREENING][_DRAFT].kmz

Every number in it is READ from client_report.json - the report's own fields,
never recomputed here - so the KMZ, the report, the workbook and the rasters
cannot disagree (clientout-2, clientout-3: until 5 Oct 2026 the KMZ worked out
its own rates, said "anchored to GNSS" whatever the tie said, and described
any stand-out with a fixed sentence written for Site A - the only patch
standing clear, outside every mapped panel, cause unknown - on Site B,
next to mapped mining, beside 3 reported features):

  Report areas        each area's outline; click for its rate, measured total,
                      95% range, 80%-power detection limit ("reliably detects,
                      4 times in 5": the sinking limit, and the rising one
                      when it differs by more than 10%), signal / threshold
                      and reading, exactly as the report gives them, and its
                      movement chart (the report's series; the chart's data
                      is in charts/*.csv). The average-rate total is printed
                      beside the measured one by the page's rule
                      (client_report.totals_differ)
  Features            the report's numbered features (same numbers, outline,
                      words - report_insights.feature_text - and chart).
                      The quality assessment's stand-out is drawn only as
                      the feature the report lists it under; one the QA
                      found alone is not a client placemark (F2, 7 Oct 2026)
  Measurement points  OFF when the file opens (26 Oct 2026: they made Google
                      Earth slow to load; the 15 m cell layer shows instead),
                      each balloon headed with its own point ("Point 40.12345
                      N, ..."), its area beneath. For every 15 m cell of a
                      chosen site and dates: the results viewer's "KMZ, every
                      15 m cell". The WHOLE study area, one point every ~100 m (about one
                      per independent radar measurement: each value combines
                      ground within ~90 m x 240 m), in ~3 km tiles that Google
                      Earth loads as they come into view (field_points; until
                      8 Oct 2026 ~400 points per area, so the field between
                      features looked empty), plus ~150 per feature. Coloured
                      on the client map's scale; faded where the point is
                      within what ordinary farmland does (the per-cell null of
                      its coherence class, client_report.json
                      cell_null_by_class: signal / threshold < 1). A point's
                      figures and verdict are its own cell's, by the rule of
                      rate_signal_to_noise.tif (|relative rate| / that cell's
                      null, 1.96 or more stands out), so a balloon reads what
                      the raster and the point table read at that cell (until
                      7 Oct 2026 the balloon tested the 3 x 3 mean against the
                      centre cell's null, and about 3% of points disagreed
                      with the raster). The balloon also gives the signal-to-
                      noise ratio and the line-of-sight rate and first-to-last
                      change (the client rasters line_of_sight_*), and an
                      inline chart of its movement over the record (the 3 x 3
                      cells around it, ~45 m; svg_chart - a PNG per point was
                      ~25 kB). A cell flagged in quality_too_fast_to_measure.tif
                      says so instead.
  Signal-to-noise     the same points coloured by signal-to-noise (off until
  points              ticked).
  Map layers          off until ticked, each with its legend: the report's map
                      (the same field, band and colours), line-of-sight
                      velocity and line-of-sight cumulative displacement
                      relative to ordinary ground (per 15 m cell, the client
                      rasters), and signal-to-noise per cell.
  Limitations         the page's own list (client_report.limitations) in the
                      Document balloon; every area, feature and point
                      balloon says that vertical is assumed from one track
                      and how much ground a value covers (E26, 7 Oct 2026:
                      the KMZ said neither, and its feature text pointed to
                      a Limitations section it did not have).
  GNSS stations, legend, data credits.

Figures are GNSS-anchored only when the report's are (rep["anchored"]);
otherwise every one says it is relative to the surrounding farmland. A
report that is not signed off (release.is_final) gives a _DRAFT file, and the
DISP-S1 route a _SCREENING one. A reissued report (config report_revision:
number, supersedes, changes) says "Revision N, replacing the report issued
<date>" and lists the changes in the Document balloon and its name.

Precision follows the page (E9): rates to 0.01 in/yr, area totals to 0.01
in and feature and point totals to 0.1 in, and every total with its 95%
range - a feature's, a point's or a chart title's total without one is not
printed (the title gives the rate alone); an area's, from a report made
before 7 Oct 2026, is printed as the page prints it. A feature's rate
carries the report's own 95% range (rate_range95_mm_yr, the page's; else
1.96 x the spread of ordinary farmland of its size), and its balloon says,
in the report's words (report_insights.OUTLINE_CAVEAT), that its outline was
drawn from the same data, so its figures read somewhat stronger than the
ground moved (M4). A value printed beside a limit (signal / threshold
against 1, unwrapping consistency against its gate) is rounded down only
when ordinary rounding would reach the limit.

    python -m insar_pipeline.client_kmz <config.yaml> [--out-dir preview/]

The full-resolution time-series KMZ MintPy writes (~15 GB here) is for
specialists; this one is a few tens of MB (Site B 31 MB, 37,740 field
points; Site A 50 MB, 66,189). Every build also writes the results
viewer's copy of the points (deliverables/viewer/field_points.npz,
results_viewer).
"""
from __future__ import annotations

import html
import io
import json
import logging
import math
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from . import domains

log = logging.getLogger(__name__)

MM_IN = 25.4
Z95 = 1.96
PER_AREA = 400            # _sample's default; report areas now take the field grid (FIELD_SPACING_M)
PER_FEATURE = 150
# The colour limit of a report written before 5 Oct 2026 (client_report.figures
# then: RdBu, +/- 0.6 in/yr, absolute). A newer report carries its map's own
# bins (rep["map_scale"]: edges from the data, relative to the comparison
# ground), and the points, the overlay and the legend use exactly those.
MAP_LIM_IN = 0.6
GREY = (0.86, 0.86, 0.86)                 # the map's "within normal ground variation"


# ------------------------------------------------------------------ the shared wording
def latlon_text(lat: float, lon: float, nd: int = 5) -> str:
    """'38.30440 N, 87.38430 W' - hemisphere-aware (abs(lon) + 'W' was right
    only west of Greenwich)."""
    return f"{abs(lat):.{nd}f} {'N' if lat >= 0 else 'S'}, {abs(lon):.{nd}f} {'W' if lon < 0 else 'E'}"


def _in(mm, nd=2, sign=True):
    from . import report_insights as ri
    return "—" if mm is None else (ri.signed(mm / MM_IN, nd) if sign else f"{abs(mm) / MM_IN:.{nd}f}")


def _no_neg_zero(text: str) -> str:
    from . import report_insights as ri
    return ri.no_negative_zero(text)


def _tidy(fig) -> None:
    """No '-0.00' on a chart (report_insights.tidy_figure)."""
    from . import report_insights as ri
    ri.tidy_figure(fig)


def _gnss_bg(rep: dict) -> dict:
    return ((rep.get("gnss") or {}).get("background") or {}) if rep.get("anchored") else {}


def basis_sentence(rep: dict) -> str:
    """What every figure is relative to, from the report's own anchoring
    (gap-licensing-3: 'anchored to the regional GNSS network' was printed
    whatever the tie said, '+0.00 mm/yr' when there was none). The regional
    rate is the report's own sentence (client_report.regional_gnss): with its
    uncertainty, and a direction word only when its 95% range excludes zero -
    'the region itself moves -1.9 mm/yr' stood beside a report that read the
    same rate as 'cannot be told apart from no movement' (outcome check,
    6 Oct 2026)."""
    from . import report_insights as ri
    ground = ri.ground_word(rep)
    if rep.get("anchored"):
        reg = ""
        try:
            from . import client_report as cr
            reg = cr.regional_gnss(rep).get("sentence") or ""
        except Exception as exc:                              # noqa: BLE001 - the sentence without the rate
            log.info("Regional GNSS sentence not built (%s)", exc)
        return ("Vertical movement, negative = sinking, anchored to the regional GNSS network: absolute movement."
                + (f" Regional movement: {reg[0].lower() + reg[1:]}" if reg else ""))
    why = rep.get("gnss_reason") or (rep.get("gnss") or {}).get("reason")
    return (f"Vertical movement, negative = sinking, relative to the area's ordinary {ground} - no GNSS anchor"
            + (f" ({str(why).rstrip('.')})" if why else "") + ", so the figures are not absolute and the movement "
            "of the study area as a whole is not measured.")


def totals_differ(tot, avg) -> bool:
    """Whether the average-rate total is printed beside the measured one:
    the page's and the PDF's rule (client_report.totals_differ, >= max(5 mm,
    5%)). Until 7 Oct 2026 the KMZ had its own (> max(2 mm, 5%), or the bend
    tested), and Site B's whole area showed two totals in Google Earth
    and one in the report (finding #37)."""
    if tot is None or avg is None:
        return False
    try:
        from . import client_report as cr
        return bool(cr.totals_differ(tot, avg))
    except (ImportError, AttributeError):                     # a partial install: the same numbers
        return abs(float(avg) - float(tot)) >= max(5.0, 0.05 * abs(float(tot)))


# M2 (7 Oct 2026): "next to mapped mine areas" only this close to a mapped edge
NEXT_TO_MINE_M = 500.0

# E5 (7 Oct 2026): the rising limit is given beside the sinking one only
# when it differs from it by more than this share
RISE_LIMIT_SHOWN = 0.10


def _rising_differs(rec: dict) -> bool:
    """The page's rule (client_report.rising_limit_differs): the rising
    limit is printed only when it differs from the sinking one by more than
    a tenth of it."""
    try:
        from . import client_report as cr
        return bool(cr.rising_limit_differs(rec))
    except (ImportError, AttributeError):                     # a partial install: the same rule
        sink, rise = rec.get("detection_limit_mm_yr"), rec.get("detection_limit_rise_mm_yr")
        return bool(sink and rise and rec.get("detection_limit_basis") == "sinking"
                    and abs(float(rise) - float(sink)) > RISE_LIMIT_SHOWN * float(sink))


def detection_text(rec: dict) -> str:
    """The 'Reliably detects (4 times in 5)' cell of a tested area (E5,
    contract C1): the SINKING limit (detection_limit_mm_yr, vertical, when
    detection_limit_basis says 'sinking'), and the rising one
    (detection_limit_rise_mm_yr) only when it differs by more than a tenth
    (_rising_differs, the page's rule). A report from before 7 Oct 2026 has
    no per-sign limit: its single limit was the worse sign's, printed as it
    was."""
    det, rise = rec.get("detection_limit_mm_yr"), rec.get("detection_limit_rise_mm_yr")
    if not det:
        return "—"
    if rec.get("detection_limit_basis") != "sinking":
        return f"sustained movement of {_in(det, 2, False)} in/yr ({det:.0f} mm/yr) or more"
    txt = f"sinking of {_in(det, 2, False)} in/yr ({det:.0f} mm/yr) or more"
    if _rising_differs(rec):
        txt += f"; rising of {_in(rise, 2, False)} in/yr ({rise:.0f} mm/yr) or more"
    return txt


def _total_range(rec: dict) -> list | None:
    """A total's 95% range (E9: total_range95_mm, or
    untested_total_range95_mm for an area that could not be compared with
    farmland); None when the record has none."""
    trng = rec.get("total_range95_mm") or rec.get("untested_total_range95_mm")
    return list(trng[:2]) if trng and len(trng) >= 2 and None not in trng[:2] else None


def _total_range_text(rec: dict, nd: int = 2) -> str:
    """', 95% a to b in' for a total that carries its range, as the page
    prints it: to 0.01 in for an area (client_report's table), to 0.1 in for
    a feature (report_insights.feature_text)."""
    trng = _total_range(rec)
    return f", 95% {_in(trng[0], nd)} to {_in(trng[1], nd)} in" if trng else ""


def avg_total_text(avg: float, rate_range: list | None, span: float | None, nd: int = 2) -> str:
    """'; a in (b mm) at the average rate (95% c to d in)': the average-rate
    total printed beside the measured one, with ITS range - the rate's 95%
    range times the years (E9: every total with its range; until 7 Oct 2026
    it stood bare). No range without the rate's range and the span."""
    txt = f"; {_in(avg, nd)} in ({avg:+.0f} mm) at the average rate"
    if rate_range and len(rate_range) >= 2 and None not in rate_range[:2] and span:
        lo, hi = (float(x) * float(span) for x in rate_range[:2])
        txt += f" (95% {_in(lo, nd)} to {_in(hi, nd)} in)"
    return txt


# M4 (7 Oct 2026): a feature's outline is drawn from the same data its
# figures are measured on
OUTLINE_NOTE = ("The outline was drawn from these same measurements, so its rate, total and range are for that "
                "outline and read somewhat stronger than the ground truly moved.")


def feature_rate_range(f: dict) -> list | None:
    """A feature's 95% rate range (M4): the report's own
    (rate_range95_mm_yr, on its footprint test's frame - the range the page
    and the PDF print in the feature's paragraph) when it carries one, else
    its rate +/- 1.96 x the spread of ordinary farmland of its size
    (farmland_spread_1sigma_mm_yr, vertical - the lighter band of its
    chart), as client_report falls back to. None without a spread. (Until
    the review of 7 Oct 2026 the report's own key was not read, and the
    balloon's table and its paragraph would have given two ranges.)"""
    for key in ("rate_range95_mm_yr", "range95_mm_yr"):
        rng = f.get(key)
        if rng and len(rng) >= 2 and None not in rng[:2]:
            return sorted([float(rng[0]), float(rng[1])])
    v = f.get("vertical_relative_mm_yr", f.get("mean_rel_vertical_mm_yr"))
    sd = f.get("farmland_spread_1sigma_mm_yr")
    if v is None or not sd or not np.isfinite(float(sd)) or float(sd) <= 0:
        return None
    return [float(v) - Z95 * float(sd), float(v) + Z95 * float(sd)]


def outline_note() -> str:
    """The outline caveat in the report's own words
    (report_insights.OUTLINE_CAVEAT, which the page and the PDF print), so
    the three cannot word it differently; OUTLINE_NOTE on a partial install."""
    try:
        from . import report_insights as ri
        return str(getattr(ri, "OUTLINE_CAVEAT", "") or OUTLINE_NOTE)
    except ImportError:
        return OUTLINE_NOTE


def _says_outline(text: str) -> bool:
    """Whether the report's own feature paragraph already gives the outline
    caveat (so the balloon does not say it twice)."""
    t = (text or "").lower()
    return "outline" in t and "stronger" in t


# ------------------------------------------------------------------ the reissue note (M3)
def _date_text(d) -> str:
    """'28 Sep 2026' from 2026-09-28, 20260928 or a date; any other text as given."""
    if d is None:
        return ""
    if hasattr(d, "strftime"):
        return f"{d.day} {d.strftime('%b %Y')}"
    s = str(d).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            x = datetime.strptime(s, fmt)
            return f"{x.day} {x.strftime('%b %Y')}"
        except ValueError:
            pass
    return s


def revision_record(cfg, rep: dict) -> dict | None:
    """The report's revision (M3, 7 Oct 2026): config.report_revision on the
    config's report_revision block {number, supersedes (the earlier issue's
    date), changes (plain sentences)}, else on the same note carried in
    client_report.json - the one reader the page, the PDF and the release
    gate use, so the KMZ cannot word it differently. None for a first issue
    (and for an incomplete note, which load_config refuses). Until then a
    reissued report (Site A's Revision 1 of the 28 Sep report) gave a
    different headline and figures with no word of what had changed."""
    try:
        from .config import report_revision as reader
    except ImportError:                                       # a partial install: read the block here
        reader = None
    rv = getattr(cfg, "report_revision", None) if cfg is not None else None
    if not rv:
        rv = (rep or {}).get("report_revision")
    if not rv:
        return None
    if reader is not None:
        return reader(rv if isinstance(rv, dict) else dict(getattr(rv, "__dict__", {}) or {}))
    if not isinstance(rv, dict):
        rv = dict(getattr(rv, "__dict__", {}) or {})
    n = rv.get("number")
    changes = [" ".join(str(x).split()) for x in (rv.get("changes") or []) if str(x or "").strip()]
    if n in (None, "") and not rv.get("supersedes") and not changes:
        return None
    return {"number": n, "supersedes": rv.get("supersedes"), "changes": changes}


def revision_heading(rv: dict | None) -> str:
    """'Revision 1, replacing the report issued 28 Sep 2026' (the note's own
    heading when it has one; '' for a first issue)."""
    if not rv:
        return ""
    if rv.get("heading"):
        return str(rv["heading"])
    n = rv.get("number")
    head = f"Revision {n}" if n not in (None, "") else "Revised report"
    when = _date_text(rv.get("supersedes_text") or rv.get("supersedes"))
    return head + (f", replacing the report issued {when}" if when else "")


def chance_text(f: dict) -> str:
    """A feature's selection-corrected chance in words (E8, contract C4):
    the report's chance['p_text'] ('less than 1 in 200' when it beat every
    one of the 199 copies, F3 - a bound, never 'about 0.005'; 'less than 1
    in 60' in a record made with 59), rebuilt from the copies for a record
    made before 7 Oct 2026. '' when the chance was not tested."""
    ch = f.get("chance") or {}
    if ch.get("p_text"):
        return str(ch["p_text"])
    p = ch.get("p", f.get("selection_p"))
    if p is None and ch.get("copies_as_strong") is None:
        return ""
    from . import report_insights as ri
    return ri.p_text(ch.get("copies_as_strong"), ch.get("copies"), p)


def _up_factor_text(rep: dict) -> str:
    k = rep.get("los_east_over_up")
    return f"east-west ground movement of 1 in/yr reads as about {float(k):.1f} in/yr" if k else ""


def vertical_caveat(rep: dict) -> str:
    """One sentence for every balloon that gives a vertical figure (E26):
    vertical is assumed from one satellite track (look angle, orbit and the
    east-west factor from the report: rep look_angle_deg, orbit,
    los_east_over_up), and the ground each value covers
    (rep['resolution']['text'], the page's own words)."""
    ang = rep.get("look_angle_deg")
    orbit = str(rep.get("orbit") or "").strip().lower()
    track = f"one {orbit} satellite track" if orbit in ("ascending", "descending") else "one satellite track"
    txt = ("Vertical assumes all the movement is up or down: the radar looks "
           + (f"about {float(ang):.0f}° from vertical " if ang else "at a slant ") + f"from {track}")
    k = _up_factor_text(rep)
    txt += (f", so {k} of vertical movement." if k else ".")
    res = (rep.get("resolution") or {}).get("text")
    return txt + (f" {res}" if res else "")


def see_limitations(text: str, rep: dict) -> str:
    """The report's feature paragraph for a balloon: its '(see
    Limitations)' points to a page section, so the caveat itself goes in its
    place (E26)."""
    if not text or "(see Limitations)" not in text:
        return text or ""
    k = _up_factor_text(rep)
    return text.replace("(see Limitations)", f"(from this track, {k} of vertical movement; see Limitations in this "
                                             "file's description)" if k else "(see Limitations in this file's "
                                                                            "description)")


def point_footer(resolution: dict | None = None) -> str:
    """The small print under a point (E26): the 3 x 3 cells its chart
    averages, the ground each value already combines (the look cell's
    support, rounded as the page rounds it) and the vertical assumption."""
    res = resolution or {}
    post = res.get("posting_m")
    cells = f"~{3 * float(post):.0f} m" if post else "~45 m"
    sup = res.get("support_m") or []
    # the figures are the point's own cell, as the rasters and the point
    # table give it (rate_signal_to_noise.tif's rule); the chart averages
    # the cells around it
    txt = (f"Negative = sinking. The figures above are this point's own {float(post):.0f} m cell"
           if post else "Negative = sinking. The figures above are this point's own cell")
    txt += f", as in the delivered rasters; the chart averages the 3 x 3 cells around it ({cells})"
    if len(sup) == 2 and all(isinstance(x, (int, float)) for x in sup):
        txt += (f"; each value already combines ground within about {round(float(sup[0]), -1):.0f} m east-west and "
                f"{round(float(sup[1]), -1):.0f} m north-south")
    return txt + ". Vertical assumes all the movement is up or down. Single points are noisy - see the area figures."


def limitations_items(cfg, rep: dict) -> list[str]:
    """The page's Limitations items (client_report.limitations, already
    HTML-escaped) for the Document balloon; failing that, the vertical
    caveat alone - never nothing (E26)."""
    try:
        from . import client_report as cr
        items = [str(x) for x in cr.limitations(cfg, rep) if x]
        if items:
            return items
    except Exception as exc:                                  # noqa: BLE001 - the caveat without the page's list
        log.info("Page limitations not read for the KMZ (%s)", exc)
    cav = vertical_caveat(rep)
    return [html.escape(cav)] if cav else []


def _screening(cfg, rep: dict) -> bool:
    try:
        from .config import is_screening
        if is_screening(cfg):
            return True
    except (ImportError, AttributeError):
        if str(getattr(cfg, "backend", "")).lower() == "disp":
            return True
    return str(rep.get("route") or "").lower() == "screening"


def release_state(cfg, rep: dict) -> tuple[bool, list[str]]:
    """(final, the hard blockers' plain text) - release.is_final on THIS
    report (a reviewer's sign-off bound to its numbers). A missing gate is a
    draft, never a pass."""
    try:
        from . import release
        final = bool(release.is_final(cfg, rep))
        st = release.status(cfg, rep) if hasattr(release, "status") else {"blockers": release.blockers(cfg, rep)}
        return final, [b["text"] for b in st.get("blockers") or [] if b.get("hard")]
    except (ImportError, AttributeError) as exc:
        log.info("Release gate not available (%s): the KMZ is a draft", exc)
        blk = rep.get("release") or {}
        return False, [b.get("text", "") for b in blk.get("blockers") or [] if b.get("hard")] or ["not signed off"]


def _title(cfg, rep: dict) -> str:
    t = (rep.get("title") or "").strip()
    if t:
        return t
    try:
        from . import report_insights as ri
        return ri._pretty(Path(cfg.project_dir).name)
    except Exception:                                         # noqa: BLE001
        return Path(cfg.project_dir).name


def kmz_name(cfg, rep: dict, final: bool, screening: bool) -> str:
    """<Title>_[SCREENING_]ground_movement[_DRAFT].kmz, the title from the
    report (report_insights.identity): 'site_d' gave 'Bear_ground_movement'
    from project_dir.split('_')[0]."""
    t = re.sub(r"\(screening\)", "", _title(cfg, rep), flags=re.I)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_") or "Project"
    return f"{safe}_{'SCREENING_' if screening else ''}ground_movement{'' if final else '_DRAFT'}.kmz"


def doc_name(cfg, rep: dict, final: bool, screening: bool) -> str:
    d0, d1 = rep["dates"]
    rv = revision_record(cfg, rep)
    rev = f", Revision {rv['number']}" if rv and rv.get("number") not in (None, "") else ""
    return (f"{'SCREENING – ' if screening else ''}{_title(cfg, rep)}: ground movement {d0[:4]}–{d1[:4]}{rev}"
            + ("" if final else " (DRAFT)"))


def revision_block(cfg, rep: dict) -> str:
    """The Document balloon's reissue note (M3): 'Revision N, replacing the
    report issued <date>' and the config's plain sentences of what changed;
    '' for a first issue."""
    rv = revision_record(cfg, rep)
    if not rv:
        return ""
    e = html.escape
    out = f"<b>{e(revision_heading(rv))}.</b>"
    if rv["changes"]:
        out += (" What changed:<ul style='font-size:12px'>"
                + "".join(f"<li>{e(c)}</li>" for c in rv["changes"]) + "</ul>")
    else:
        out += "<br>"
    return out


def doc_description(cfg, rep: dict, final: bool, blockers: list[str], screening: bool) -> str:
    """The Document balloon: what this is, how it is measured, what the
    figures are relative to, its release state and the data credits."""
    e = html.escape
    d0, d1 = rep["dates"]
    parts = []
    if screening:
        parts.append("<b>SCREENING ONLY</b>: made from OPERA DISP-S1 products to look for movement worth a full "
                     "processing. These figures are not the official measurement.<br>")
    if str(getattr(cfg, "backend", "")) == "nisar":
        parts.append("<b>COMPARISON - NISAR</b>: made from NASA-ISRO's PROVISIONAL NISAR interferograms, not yet "
                     "validated; for comparison with the Sentinel-1 results, not the official measurement.<br>")
    if not final:
        parts.append("<b>DRAFT - not for release.</b> "
                     + ("Still open: " + "; ".join(e(b.rstrip(".")) for b in blockers[:6]) + "." if blockers else
                        "Not yet signed off by a reviewer.") + "<br>")
    # a reissue says so, and what changed (M3, 7 Oct 2026)
    rev = revision_block(cfg, rep)
    if rev:
        parts.append(rev)
    loc = (rep.get("location") or "").strip()
    from .config import sensor_name
    src = ("OPERA DISP-S1 (from Sentinel-1 radar)" if screening else
           "NISAR radar (comparison: PROVISIONAL products, not validated)" if sensor_name(cfg) == "NISAR"
           else "Sentinel-1 radar")
    parts.append(f"{e(_title(cfg, rep))}{', ' + e(loc) if loc else ''}: ground movement from {src}, "
                 f"{rep.get('n_dates', '')} images {d0[:4]}-{d0[4:6]}-{d0[6:]} to {d1[:4]}-{d1[4:6]}-{d1[6:]}. "
                 f"{e(basis_sentence(rep))} Click an area, a feature or a point for its numbers and chart; the "
                 "numbers are the client report's.")
    # the page's Limitations, word for word (E26: the KMZ said nothing of the
    # vertical assumption or of the ground each value covers, and its feature
    # text pointed to a Limitations section it did not have)
    lim = limitations_items(cfg, rep)
    if lim:
        parts.append("<br><br><b>Limitations</b><ul style='font-size:12px'>"
                     + "".join(f"<li>{x}</li>" for x in lim) + "</ul>")
    try:
        from . import credits as cd
        block = cd.text_block(cfg, rep)
        parts.append("<br><br><small>" + "<br>".join(e(x) for x in block.splitlines()) + "</small>")
    except Exception as exc:                                  # noqa: BLE001 - never stop the KMZ for the credits
        log.warning("Credits not added to the KMZ (%s)", exc)
        parts.append("<br><br><small>" + ("NISAR data: NASA-ISRO, via ASF DAAC." if str(getattr(cfg, "backend", "")) == "nisar"
                                         else "Contains modified Copernicus Sentinel data.") + "</small>")
    return "".join(parts)


def _row(k: str, v: str) -> str:
    return f"<tr><td>{k}</td><td>{v}</td></tr>"


def area_description(rec: dict, rep: dict, png: str | None = None) -> str:
    """One report area's balloon, every figure the report's own field
    (vertical_mm_yr, total_mm, range95_mm_yr, detection_limit_mm_yr,
    signal_to_noise, reading); a not-testable area has no range and no
    verdict, and says why."""
    e = html.escape
    from . import report_insights as ri
    ground = e(ri.ground_word(rep))
    v, tot, avg = rec.get("vertical_mm_yr"), rec.get("total_mm"), rec.get("total_at_average_rate_mm")
    full = rec.get("kind") == "full"
    rows = []
    size = (f"{rec['footprint_km2']:,.1f} km&sup2; area, {rec.get('measured_percent', 0):.0f}% measured"
            if rec.get("footprint_km2") else f"{rec.get('area_km2') or 0:,.1f} km&sup2;")
    head = f"<b>{e(rec.get('display_name') or rec['name'])}</b> ({size}, {rec.get('cells') or 0:,} points)<br>"
    if v is not None:
        # to 0.01 in/yr, as the page (E9: 0.001 in claimed far more than the data have)
        rows.append(_row("Vertical, per year", f"<b>{_in(v)} in</b> ({v:+.1f} mm)"
                         + ("" if rep.get("anchored") else f" relative to the surrounding {ground}")))
    elif rec.get("relative_only"):
        rv = rec.get("relative_vertical_mm_yr")
        rows.append(_row("Vertical, per year", "relative only: no GNSS anchor, so the area as a whole has no figure"
                         + (f" (relative to its own {ground} {rv:+.1f} mm, ~0 by construction)" if rv is not None else "")))
    else:
        rows.append(_row("Vertical, per year", "—"))
    if tot is not None:
        extra = ""
        if totals_differ(tot, avg):
            rr_ = rec.get("range95_mm_yr") or rec.get("untested_range95_mm_yr")
            extra = avg_total_text(avg, rr_, rep.get("span_years"))
        rows.append(_row(f"Measured over the record ({rep.get('span_years', 0):.1f} yr)",
                         f"{_in(tot)} in ({tot:+.0f} mm){_total_range_text(rec)}{extra}"))
    rr, rw = rec.get("recent_rate_mm_yr"), rec.get("recent_window") or []
    if rr is not None and len(rw) == 2:
        rng_r = rec.get("recent_range95_mm_yr")
        rows.append(_row(f"Last two years ({rw[0][:4]}–{rw[1][:4]})", f"{_in(rr)} in/yr ({rr:+.1f} mm/yr)"
                         + (f", 95% {_in(rng_r[0])} to {_in(rng_r[1])} in/yr" if rng_r else "")))
    rng = rec.get("range95_mm_yr")
    if "range95_mm_yr" not in rec and v is not None and rec.get("vertical_uncertainty_1sigma_mm_yr"):
        sd = rec["vertical_uncertainty_1sigma_mm_yr"]           # a report built before 5 Oct 2026
        rng = [v - Z95 * sd, v + Z95 * sd]
    urng = None if rng else rec.get("untested_range95_mm_yr")
    rows.append(_row("95% range", f"{_in(rng[0])} to {_in(rng[1])} in/yr ({rng[0]:+.1f} to {rng[1]:+.1f} mm/yr)"
                     if rng else
                     # E9: an area that could not be compared with farmland
                     # still carries the uncertainty that can be stated
                     f"{_in(urng[0])} to {_in(urng[1])} in/yr ({urng[0]:+.1f} to {urng[1]:+.1f} mm/yr), not "
                     f"compared with ordinary {ground}" if urng else "—"))
    if full:
        g = rec.get("regional_gnss_vertical_mm_yr")
        if g is not None:
            gu = rec.get("regional_gnss_uncertainty_mm_yr")
            rows.append(_row("Regional ground motion (GNSS)", f"{_in(g)} in/yr ({g:+.1f}"
                             + (f" &plusmn; {gu:.1f} mm/yr, 1 sigma)" if gu is not None else " mm/yr)")))
    # a report written before 5 Oct 2026 has no `testable`: its limit was the
    # 2-sigma threshold and its S/N |z|, neither of which may be printed as
    # today's "reliably detects" / "signal / threshold"
    legacy = not full and "testable" not in rec
    if not full and not legacy:
        rows.append(_row("Reliably detects (4 times in 5)", detection_text(rec)))
    s = rec.get("signal_to_noise")
    if s is not None and not legacy:
        # the whole area is not tested against farmland: its number is its
        # own rate (satellite plus GNSS) over its 95% uncertainty (final
        # verification: '(1 or more stands out)' gave it the farmland test's words).
        # A value just under 1 is never shown as '1.0' beside a reading
        # that says it did not stand out (0.96 read '1.0 (1 or more stands out)')
        st = _beside_limit(float(s), 1.0, 1)
        rows.append(_row("Signal &divide; threshold", f"{st} (its rate, satellite plus GNSS, &divide; its 95% "
                         "uncertainty; 1 or more means its range excludes zero)" if full else
                         f"{st} (1 or more stands out)"))
    reading = rec.get("reading") or (("Movement detected" if rec.get("detected") else "Within normal ground variation")
                                     if legacy and rec.get("detected") is not None else "—")
    rows.append(_row("Reading", f"<b>{e(reading)}</b>"))
    notes = []
    if legacy:
        notes.append("This client report was made before 5 Oct 2026: rebuild it for the detection limit and the "
                     "signal / threshold.")
    if rec.get("testable") is False and rec.get("not_testable_reason") and not str(rec.get("reading", "")).startswith(
            ("Not testable", "Not measurable")):
        notes.append(rec["not_testable_reason"])
    if rec.get("testable") is False and not full:
        notes.append("The figure is this area's movement only; it was not checked against ordinary ground.")
    try:
        from . import client_report as cr
        tn = cr.thin_null_sentence(rec, ri.ground_word(rep)) if rec.get("testable") else ""
    except (ImportError, AttributeError):
        tn = ""
    if tn:
        notes.append(tn.strip())
    if rec.get("resolution_caveat"):
        notes.append(str(rec["resolution_caveat"]))
    desc = (head + "<table style='font-size:12px'>" + "".join(rows) + "</table>"
            + "".join(f"<small>{e(n)}</small><br>" for n in notes))
    if png:
        desc += f"<img src='{png}' width='486'/>"
    # the page's caveats travel with the figure (E26): vertical is assumed
    # from one track, and each value covers more ground than its point
    cav = vertical_caveat(rep)
    return desc + f"<br><small>{e(basis_sentence(rep))}</small>" + (f"<br><small>{e(cav)}</small>" if cav else "")


def feature_heading(i: int, f: dict) -> str:
    """The report's Observations heading for feature i, built by the same
    words (report_insights.feature_kind: 'change on surface-mine ground'
    where radar cannot follow the earthmoving, else sinking/rising ground)."""
    try:
        from . import report_insights as ri
        kind, where = ri.feature_kind(f), ri.lalo_txt(f["lat"], f["lon"])
    except (ImportError, AttributeError):                     # a partial install: the plain words
        kind, where = f"{'sinking' if f['sign'] < 0 else 'rising'} ground", latlon_text(f["lat"], f["lon"], 3)
    return f"Feature {i}: {kind}, about {f['area_km2']:.1f} km² near {where}"


# a study point's colour by its reading (KML aabbggrr): sinking red, within
# variation grey-green, not measurable grey, no number (fast zone) black
_STUDY_POINT_STYLES = (("sp_moving", "ff1c1cd6"), ("sp_quiet", "ff6a9a4a"), ("sp_untested", "ff9a9a9a"),
                       ("sp_refused", "ff202020"))


def _study_point_style(r: dict) -> str:
    if r.get("refused"):
        return "sp_refused"
    if not r.get("testable"):
        return "sp_untested"
    return "sp_moving" if r.get("detected") else "sp_quiet"


def study_folders(cfg, rep: dict) -> str:
    """The study's own folders (A3, PLAN_oil_gas_slopes.md): one placemark per
    listed point (wells) with the report's reading, and the dated pads'
    outlines with their largest change - the record's numbers
    (client_report.study_point_entries / study_event_entries), word for word.
    "" for the default study or a record without them, so its KMZ is as
    before."""
    from . import client_report as cr
    e = html.escape
    pts, evs = rep.get("study_points"), rep.get("study_events")
    if not pts and not (evs or {}).get("pads"):
        return ""
    out = []
    if pts:
        styles = "".join(f"<Style id='{k}'><IconStyle><color>{c}</color><scale>0.9</scale><Icon><href>"
                         "https://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle>"
                         "</Style>" for k, c in _STUDY_POINT_STYLES)
        pms = []
        for r in pts:
            attrs = "".join(f"<br>{e(k.replace('_', ' ').capitalize())}: {e(str(v))}"
                            for k, v in (r.get("attributes") or {}).items() if v)
            shared = (f"<br>Shares one value with {e(', '.join(map(str, r['shared_with'])))} (too close to tell apart)"
                      if r.get("shared_with") else "")
            pms.append(f"<Placemark><name>{e(str(r['id']))}</name><styleUrl>#{_study_point_style(r)}</styleUrl>"
                       f"<description><![CDATA[{e(cr._well_reading(r))}{attrs}{shared}<br>Ground within "
                       f"{int(r.get('radius_m') or 100)} m of the point; not a rate for the point itself. "
                       f"Tier: {e(str(r.get('tier') or '').lower() or 'not stated')}.]]></description>"
                       f"<Point><coordinates>{r['lon']:.6f},{r['lat']:.6f},0</coordinates></Point></Placemark>")
        label = next((r.get("layer_label") for r in pts if r.get("layer_label")), "Listed points").split(" (")[0]
        out.append(f"<Folder><name>{e(label)} (screening)</name><open>0</open>{styles}{''.join(pms)}</Folder>")
    pads = (evs or {}).get("pads") or []
    src = str(getattr(cfg, "event_polygons", "") or "").strip()
    if pads and src:
        from . import events as ev_mod
        geoms = {}
        try:
            for ft in ev_mod.load_features(src, "polygon"):
                geoms[ft["name"]] = ft["geom"]
        except (OSError, ValueError, KeyError, TypeError):
            geoms = {}
        pms = []
        for p_ in pads:
            g = geoms.get(p_.get("name"))
            if g is None:
                continue
            rings = []
            for poly in getattr(g, "geoms", [g]):
                coords = " ".join(f"{x:.6f},{y:.6f},0" for x, y in poly.exterior.coords)
                rings.append(f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates>"
                             "</LinearRing></outerBoundaryIs></Polygon>")
            v = p_.get("max_relative_vertical_mm")
            val = ("no value" if v is None else
                   f"{abs(v) / MM_IN:.2f} in {'down' if v < 0 else 'up'} ({v:+.1f} mm) at its deepest, relative to the "
                   "ground around it")
            pms.append(f"<Placemark><name>{e(str(p_.get('name')))}</name><styleUrl>#sp_pad</styleUrl>"
                       f"<description><![CDATA[{e(str(p_.get('window_start')))} to {e(str(p_.get('window_end')))}: "
                       f"{e(val)}; reached by {e(str(p_.get('date_of_max') or '-'))}. Tier: "
                       f"{e(str(p_.get('tier') or '').lower() or 'not stated')}.]]></description>"
                       + (rings[0] if len(rings) == 1 else f"<MultiGeometry>{''.join(rings)}</MultiGeometry>")
                       + "</Placemark>")
        if pms:
            out.append("<Folder><name>Dated pads (screening)</name><open>0</open><Style id='sp_pad'><LineStyle>"
                       "<color>ff0080ff</color><width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle>"
                       f"</Style>{''.join(pms)}</Folder>")
    return "".join(out)


def feature_description(i: int, f: dict, rep: dict, png: str | None = None) -> str:
    """Feature i's balloon: the report's own paragraph (f['text'] =
    report_insights.feature_text(i, f)) word for word, its figures and chart.
    Feature figures are relative to the surrounding comparison ground, always."""
    from . import report_insights as ri
    e = html.escape
    v = f.get("vertical_relative_mm_yr", f.get("mean_rel_vertical_mm_yr"))
    tot, avg = f.get("total_relative_mm"), f.get("total_at_average_rate_mm")
    text = f.get("text")
    if not text:
        try:
            text = ri.feature_text(i, f, domains.wording_of(rep))
        except (KeyError, TypeError, AttributeError) as exc:  # a report built before 5 Oct 2026
            log.warning("Feature %d has no report text (%s)", i, exc)
            text = ""
    if f.get("ground_reshaped"):
        # dug or built up by metres: the report gives no rate (gap-surface_mine_site-7)
        desc = (f"<b>{e(feature_heading(i, f))}</b><br>{e(see_limitations(text, rep))}<br>"
                "<table style='font-size:12px'>" + _row("Vertical, per year", "no rate: the ground was reshaped "
                                                       "during the record") + "</table>")
        return desc + ("<br><small>Outline: the feature's measured cells, simplified.</small>" if f.get("ring") else
                       "<br><small>Outline: the feature's bounding box (approximate).</small>")
    ground = e(ri.ground_word(f) if f.get('control_ground_label') else ri.ground_word(rep))
    rows = [_row("Vertical, per year", f"<b>{_in(v)} in</b> ({v:+.1f} mm) relative to the surrounding {ground}")]
    # the rate's 95% range from the spread of ordinary farmland of its size
    # (M4: the feature's rate was the one figure printed without one)
    frng = feature_rate_range(f)
    if frng:
        rows.append(_row("95% range", f"{_in(frng[0])} to {_in(frng[1])} in/yr ({frng[0]:+.1f} to {frng[1]:+.1f} "
                         f"mm/yr), from the spread of ordinary {ground} of its size"))
    if tot is not None:
        # one rule with the page and the PDF (client_report.totals_differ);
        # to 0.1 in, as the feature's own paragraph (E9: '-10.55 in' sat
        # beside 'moved 10.6 in' in the same balloon)
        # E9: a total is never printed without its range (a record from
        # before 7 Oct 2026 has none: the row is left out)
        if _total_range(f):
            rows.append(_row("Measured over the record", f"{_in(tot, 1)} in ({tot:+.0f} mm)" + _total_range_text(f, 1)
                             + (avg_total_text(avg, frng, rep.get("span_years"), 1) if totals_differ(tot, avg)
                                else "")))
    rr, rw = f.get("recent_rate_mm_yr"), f.get("recent_window") or []
    if rr is not None and len(rw) == 2:
        rng_r = f.get("recent_range95_mm_yr")
        rows.append(_row(f"Last two years ({rw[0][:4]}–{rw[1][:4]})", f"{_in(rr)} in/yr ({rr:+.1f} mm/yr)"
                         + (f", 95% {_in(rng_r[0])} to {_in(rng_r[1])} in/yr" if rng_r else "")))
    chance = chance_text(f)
    if chance:
        rows.append(_row(f"Chance of a patch this strong in ordinary {ground}", e(chance)))
    desc = (f"<b>{e(feature_heading(i, f))}</b><br>{e(see_limitations(text, rep))}<br>"
            "<table style='font-size:12px'>" + "".join(rows) + "</table>")
    if not _says_outline(text):
        desc += f"<small>{e(outline_note())}</small><br>"
    if png:
        desc += f"<img src='{png}' width='486'/>"
    cav = vertical_caveat(rep)
    return desc + ("<br><small>Outline: the feature's measured cells, simplified.</small>" if f.get("ring") else
                   "<br><small>Outline: the feature's bounding box (approximate).</small>") + (
        f"<br><small>{e(cav)}</small>" if cav else "")


def standout_text(so: dict, n_features: int, mine_layers: list, ground: str = "farmland") -> str:
    """Words for the quality assessment's deepest-window stand-out. Not
    drawn in the client KMZ since F2 (7 Oct 2026): a stand-out the report
    lists is shown as its feature, one the QA found alone stays in the
    technical record. Built from its data only (gap-geolocation-1,
    methodology-12): sinking or rising by its sign; its basis (the deepest
    1 km square, relative to farmland, with the regional motion when
    anchored); its relation to the mapped mines from the flags; its strength
    as the QA's selection-aware gap, not a bare signal / noise. It never
    claims the patch is the only one or that its cause is unknown."""
    v = so.get("vertical_mm_yr")
    vr = so.get("vertical_relative_mm_yr")
    ref = vr if vr is not None else v
    verb = "Sinking" if (ref or 0) < 0 else "Rising"
    km = so.get("scale_km") or 1
    txt = (f"The deepest {km:.0f} km square the quality assessment found, centred {latlon_text(so['lat'], so['lon'], 4)}. "
           f"{verb} about {_in(ref, 2, False)} in/yr ({abs(ref or 0):.0f} mm/yr) relative to the surrounding {ground}"
           + (f" ({_in(v, 2)} in/yr including the regional motion)" if v is not None and vr is not None and v != vr else "")
           + ".")
    gap = so.get("gap_in_spreads")
    if gap is not None:
        txt += (f" It lies {gap:.1f} {ground} spreads beyond the deepest ordinary {ground} window of the same size.")
    if "inside_mapped_panels" in so or "within_panel_buffer" in so:
        buf = so.get("buffer_m")
        if so.get("inside_mapped_panels"):
            mine = "It lies inside a mapped mine area."
        elif so.get("within_panel_buffer") and buf is not None and float(buf) > NEXT_TO_MINE_M:
            # M2 (7 Oct 2026): 'next to' only within 500 m of a mapped mine
            # edge; the QA's buffer is 2 km on Site A
            mine = f"It lies within {float(buf) / 1000:.1f} km of a mapped mine area."
        elif so.get("within_panel_buffer"):
            mine = "It lies next to mapped mine areas."
        elif mine_layers:
            mine = "It lies outside every mapped mine area and its buffer."
        else:
            mine = "No mine map was supplied for this project."
    elif so.get("in_mapped_mine"):
        mine = "It lies in or next to mapped mine ground."
    elif mine_layers:
        mine = "It lies away from the mapped mine areas."
    else:
        mine = "No mine map was supplied for this project."
    txt += " " + mine
    if n_features:
        txt += f" The report lists {n_features} feature{'s' if n_features > 1 else ''}: see the Features folder."
    return txt + " Worth checking on the ground."


def point_values(rate: float, rel_v: float, sd_v: float, bg_u: float | None, anchored: bool) -> dict:
    """A point's figures, one rule with the report's per-cell test: the 95%
    range of its movement beyond ordinary farmland (rel_v +/- 1.96 x the
    cell null, vertical) and signal / threshold = |rel_v| / (1.96 x null), so
    the range excludes zero exactly when signal / threshold >= 1 (faded
    below; empirical-3: a confident range sat beside 'within normal
    variation'). The absolute range adds the GNSS uncertainty. A cell with
    no finite rate has no test (faded), as rate_signal_to_noise.tif leaves
    it empty."""
    ok = rel_v is not None and np.isfinite(rel_v) and rate is not None and np.isfinite(rate)
    thr = Z95 * sd_v if ok and sd_v and np.isfinite(sd_v) and sd_v > 0 else None
    s2t = abs(rel_v) / thr if thr else None
    u = math.hypot(sd_v, bg_u or 0.0) if thr else None
    return {"rate": rate, "rel": rel_v, "threshold": thr, "signal_to_threshold": s2t,
            "faded": s2t is None or s2t < 1.0,
            "rel_range": [rel_v - thr, rel_v + thr] if thr else None,
            "range": [rate - Z95 * u, rate + Z95 * u] if u else None, "anchored": anchored}


def _beside_limit(x: float, limit: float, nd: int = 2) -> str:
    """x to nd decimals, never shown at or above `limit` when it is below
    it (0.996 against 1 is '0.99', not '1.00 (within ...)'). Only a value
    that would round up to the limit is rounded down: 0.68 is '0.68' (to 1
    decimal '0.7', as the page prints it), not '0.6'."""
    txt = f"{x:.{nd}f}"
    if x < limit and float(txt) >= limit:
        # only a limit on the printed grid (1, 0.7) can be reached this way:
        # the value just below it on that grid
        txt = f"{limit - 10 ** -nd:.{nd}f}"
    return txt


def point_description(name: str, lat: float, lon: float, pv: dict, span: float, tc: float, png: str | None,
                      too_fast: bool = False, ground: str = "farmland", unwrap: float | None = None,
                      gate: float = 0.7, resolution: dict | None = None, los: dict | None = None,
                      chart: str | None = None, span_label: str = "over the record") -> str:
    """A point's balloon. `resolution` (client_report.json resolution:
    posting_m, support_m) sizes the footer: the 3 x 3 cells the chart
    averages, and the ground each value already combines (about 90 m x
    240 m on the Sentinel-1 routes) - until 7 Oct 2026 the footer said only
    '~45 m', and a point read as a 45 m measurement (E26).

    los: {"rate": relative LOS mm/yr, "change": relative LOS mm first to
    last image, "d0", "d1"} - the line-of-sight rows (the client rasters
    line_of_sight_*); chart: inline HTML (svg_chart) in place of a PNG."""
    e = html.escape
    ground = e(ground or "farmland")
    # the point itself first (26 Oct 2026, a coworker's review of the Site E
    # KMZ: every balloon was headed with its report area, "Full Satellite
    # Imaging", and read as the area's overview rather than the clicked spot)
    head = f"<b>{POINT_TITLE}{latlon_text(lat, lon)}</b><br>in {e(name)}<br>"
    if too_fast:
        return (head + "<b>Not measurable - too fast for this satellite.</b> The ground here changes faster across a "
                "few tens of metres than the radar can follow between passes, so the value at this point is not a "
                f"measurement (quality_too_fast_to_measure.tif).<br><small>Measurement quality: {tc:.2f} temporal "
                "coherence.</small>")
    rate, rel = pv["rate"], pv["rel"]
    s2t = pv["signal_to_threshold"]
    if rate is None or not np.isfinite(rate):
        rows = [_row("Vertical, per year", "— (no rate at this cell)")]
    else:
        # rates to 0.01 in/yr as the page; the total at the average rate
        # with its 95% range (the rate's range x the years), to 0.1 in
        # (E9: '+2.26 in' stood bare beside a rate range of -0.46 to +0.89 in/yr)
        rows = [_row("Vertical, per year", f"<b>{_in(rate)} in</b> ({rate:+.1f} mm)"
                     + ("" if pv["anchored"] else f" relative to the surrounding {ground}"))]
        rng = pv["range"]
        if rng:
            rows.append(_row(f"At the average rate {span_label} ({span:.1f} yr)",
                             f"{_in(rate * span, 1)} in ({rate * span:+.0f} mm), 95% {_in(rng[0] * span, 1)} to "
                             f"{_in(rng[1] * span, 1)} in"))
    if pv["range"] and pv["anchored"]:
        rows.append(_row("95% range, this point (with the regional motion)",
                         f"{_in(pv['range'][0])} to {_in(pv['range'][1])} in/yr"))
    if pv["rel_range"]:
        # the line the reading below belongs to: its range excludes zero
        # exactly when signal / threshold >= 1
        rows.append(_row(f"Compared with ordinary {ground}", f"{_in(rel)} in/yr, 95% {_in(pv['rel_range'][0])} to "
                         f"{_in(pv['rel_range'][1])} in/yr"))
    if s2t is not None:
        # to 0.01, so a value just under 1 never reads '1.0 (within ...)'
        rows.append(_row("Signal &divide; threshold", f"{_beside_limit(s2t, 1.0)} "
                         + ("(stands out)" if s2t >= 1 else "(within normal variation for a single point)")))
        # the same test as a signal-to-noise ratio (rate_signal_to_noise.tif,
        # the SNR layer): signal / threshold x 1.96
        rows.append(_row("Signal-to-noise ratio", f"{_beside_limit(s2t * Z95, Z95)} (1.96 or more stands out)"))
    if los and los.get("rate") is not None and np.isfinite(los["rate"]):
        rows.append(_row(f"Line of sight, per year (vs ordinary {ground})",
                         f"{_in(los['rate'])} in ({los['rate']:+.1f} mm); + = towards the satellite"))
        if los.get("change") is not None and np.isfinite(los["change"]):
            rows.append(_row(f"Line of sight, {_date_text(los.get('d0'))} to {_date_text(los.get('d1'))}",
                             f"{_in(los['change'], 1)} in ({los['change']:+.0f} mm), first image to last "
                             "(each image carries its own noise: see the chart)"))
    rows.append(_row("Measurement quality", f"{tc:.2f} temporal coherence"))
    if unwrap is not None and np.isfinite(unwrap):
        # the second quality check, per point (phaselinked-6); 0.699 read
        # '0.70 (below 0.7 ...)' (final verification preview, 7 Oct 2026)
        rows.append(_row("Unwrapping consistency", f"{_beside_limit(unwrap, gate)} " + (
            f"(below {gate:.1f}: fails the second quality check - treat this point with extra care)" if unwrap < gate
            else "(passes the second quality check)")))
    desc = head + "<table style='font-size:12px'>" + "".join(rows) + "</table>"
    if png:
        desc += f"<img src='{png}' width='486'/>"
    elif chart:
        desc += chart
    return desc + f"<br><small>{e(point_footer(resolution))}</small>"


# ------------------------------------------------------------------ colours (the client map's)
def _cmap_rgb(v_in: float, lim: float) -> tuple:
    from matplotlib import colormaps
    from matplotlib.colors import TwoSlopeNorm
    return tuple(colormaps["RdBu"](TwoSlopeNorm(0, -lim, lim)(np.clip(v_in, -lim, lim)))[:3])


def _kml(rgb, alpha: float = 1.0) -> str:
    r, g, b = (int(round(255 * x)) for x in rgb)
    return f"{int(round(255 * alpha)):02x}{b:02x}{g:02x}{r:02x}"


def point_bins(lim: float = MAP_LIM_IN) -> list[tuple[float, float, tuple, str]]:
    """Seven bins on the client map's own scale (RdBu over +/- lim in/yr,
    red = sinking): edges at lim/6, lim/3 and 2 lim/3 each side (0.1, 0.2,
    0.4 at the map's 0.6), each coloured as the map colours its middle
    value, the outer ones as the map's saturated ends; the middle bin is the
    map's grey. Until 5 Oct 2026 the points had their own palette."""
    ed = [lim / 6, lim / 3, 2 * lim / 3]
    edges = [-1e9, -ed[2], -ed[1], -ed[0], ed[0], ed[1], ed[2], 1e9]
    mids = [-lim, -(ed[1] + ed[2]) / 2, -(ed[0] + ed[1]) / 2, 0.0, (ed[0] + ed[1]) / 2, (ed[1] + ed[2]) / 2, lim]
    out = []
    for i in range(7):
        lo, hi = edges[i], edges[i + 1]
        rgb = GREY if i == 3 else _cmap_rgb(mids[i], lim)
        lab = (f"< {-ed[2]:+.2f}" if i == 0 else f"> {ed[2]:+.2f}" if i == 6 else f"{lo:+.2f} to {hi:+.2f}")
        out.append((lo, hi, rgb, lab))
    return out


def scale_bins(ms: dict | None) -> list[tuple[float, float, tuple, str]] | None:
    """The client map's own bins (rep["map_scale"], report_insights.map_scale,
    coloured by client_report._map_cmap): in in/yr, RELATIVE to the
    comparison ground like the map (the GNSS regional rate is in the
    balloons, not the colours); the band is the grey middle bin and the open
    ends take the map's extend-arrow colours. None for a report without one
    (before 5 Oct 2026: point_bins)."""
    if not ms or not ms.get("edges_mm_yr"):
        return None
    from . import client_report as cr
    edges = [float(x) for x in ms["edges_mm_yr"]]
    cmap, _ = cr._map_cmap(edges, int(ms.get("n_below") or 0))
    ed_in = [x / MM_IN for x in edges]
    out = [(-1e9, ed_in[0], tuple(cmap.get_under()[:3]), f"< {ed_in[0]:+.2f}")]
    for i in range(len(ed_in) - 1):
        out.append((ed_in[i], ed_in[i + 1], tuple(cmap(i)[:3]), f"{ed_in[i]:+.2f} to {ed_in[i + 1]:+.2f}"))
    out.append((ed_in[-1], 1e9, tuple(cmap.get_over()[:3]), f"> {ed_in[-1]:+.2f}"))
    return out


def _bin(v_in: float, bins) -> int:
    for i, (lo, hi, _, _) in enumerate(bins):
        if lo <= v_in < hi:
            return i
    return len(bins) - 1


# ------------------------------------------------------------------ geometry
def _sample(mask: np.ndarray, px_m: float, per: int = PER_AREA) -> list[tuple[int, int]]:
    """Cells nearest the centre of each block of a regular grid, inside mask;
    block size chosen for at most ~per points, never finer than 2 cells."""
    n = int(mask.sum())
    if not n:
        return []
    area = n * px_m * px_m
    k = max(2, int(round(math.sqrt(area / per) / px_m)))
    rows, cols = np.where(mask)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    out = []
    for r in range(r0, r1, k):
        for c in range(c0, c1, k):
            blk = mask[r:r + k, c:c + k]
            if not blk.any():
                continue
            yy, xx = np.where(blk)
            j = int(np.argmin((yy - k / 2) ** 2 + (xx - k / 2) ** 2))
            out.append((r + int(yy[j]), c + int(xx[j])))
    return out


def _polygon(rings) -> str:
    return "".join("<Polygon><outerBoundaryIs><LinearRing><coordinates>"
                   + " ".join(f"{x:.6f},{y:.6f},0" for x, y in np.asarray(ring)[:, :2])
                   + "</coordinates></LinearRing></outerBoundaryIs></Polygon>" for ring in rings)


# ------------------------------------------------------------------ charts
def _chart(dates, y_in, title, centre_in=None, half_in=None, band_label="95% range of the rate",
           centre_label="regional ground motion (GNSS)", line_label="this location") -> bytes:
    """A movement chart: the series, a dashed regional line when anchored,
    and a band only when there is one (never a zero-width band)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=90)
    c = np.zeros(len(dates)) if centre_in is None else np.asarray(centre_in)
    if half_in is not None and np.nanmax(np.abs(half_in)) > 0:
        ax.fill_between(dates, c - half_in, c + half_in, color="#c9d3db", alpha=0.7, lw=0, label=band_label)
    if centre_in is not None:
        ax.plot(dates, c, color="#5b6672", lw=1, ls="--", label=centre_label)
    y = np.asarray(y_in, dtype=float)
    fin = y[np.isfinite(y)]
    ax.plot(dates, y, color="#a8432f" if fin.size and fin[-1] < 0 else "#2d5f7c", lw=1.3, label=line_label)
    ax.axhline(0, color="#8a95a1", lw=0.6)
    ax.set_ylabel("inches"); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.legend(fontsize=7, loc="lower left", framealpha=0.85)
    # fixed margins, not tight_layout: a third of each chart's time, and the
    # KMZ draws ~400 per area
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.11)
    buf = io.BytesIO(); _tidy(fig); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()


def _report_chart(dates, t, it: dict, bg_line: float, title: str) -> bytes:
    """An area's or feature's chart drawn by the report's own code
    (client_report._series_axes on a client_report.series_items entry): the
    same line, straight average-rate line and the two bands of series_band
    (clientout-6, stats-8; until 5 Oct 2026 the KMZ drew a zero-width rate
    cone of its own). The reading goes on the right as in the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import client_report as cr
    fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=90)
    cr._series_axes(ax, dates, t, it, bg_line, compact=True)
    ax.set_title(title, fontsize=8.5, loc="left")
    tag = ("Movement detected" if it.get("detected") else "Within normal ground variation" if it.get("decision")
           else "Not tested against ordinary ground" if not it.get("feature") else "")
    if tag:
        ax.set_title(tag, fontsize=8, loc="right",
                     color="#a8432f" if it.get("detected") else "#3d7a4c" if it.get("decision") else "#6b7580")
    ax.tick_params(labelsize=8)
    lg = ax.get_legend()
    if lg is not None:
        for tx in lg.get_texts():
            tx.set_fontsize(6.5)
    fig.subplots_adjust(left=0.11, right=0.89, top=0.84, bottom=0.1)
    buf = io.BytesIO(); _tidy(fig); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()


def _full_chart(dates, t, rec: dict, name: str, bg_line: float, anchored: bool, ground: str = "farmland") -> bytes:
    """The whole study area's chart: its series (satellite + GNSS when
    anchored), its straight average-rate line, the 95% range of that rate
    (range95_mm_yr x t, from the report) and the GNSS regional line. The
    whole area is not tested against its own farmland, so it has no
    'ordinary ground' band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = np.array([np.nan if q is None else q for q in rec["series_vertical_mm"]], dtype=float) / MM_IN
    v, rng = rec.get("vertical_mm_yr"), rec.get("range95_mm_yr")
    fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=90)
    t = np.asarray(t, dtype=float)
    if v is not None and rng and rng[0] is not None and rng[1] is not None:
        ax.fill_between(dates, rng[0] * t / MM_IN, rng[1] * t / MM_IN, color="#c9d3db", lw=0,
                        label="95% range of the area's rate")
    if anchored and bg_line:
        ax.plot(dates, bg_line * t / MM_IN, color="#5b6672", lw=1, ls="--", label="regional ground motion (GNSS)")
    col = "#a8432f" if (v if v is not None else np.nanmean(y[-5:]) if np.isfinite(y).any() else 0) < 0 else "#2d5f7c"
    ax.plot(dates, y, color=col, lw=0.9, alpha=0.75, label=f"{name}, every image")
    if v is not None:
        ax.plot(dates, v * t / MM_IN, color=col, lw=2.2, label="its average rate")
    ax.axhline(0, color="#8a95a1", lw=0.6)
    ax.set_ylabel("inches"); ax.grid(alpha=0.25); ax.tick_params(labelsize=8)
    ax.secondary_yaxis("right", functions=(lambda q: q * MM_IN, lambda q: q / MM_IN)).set_ylabel("mm")
    ax.set_title(series_title(name, v, rec.get("total_mm"), not anchored, ground, _total_range(rec)), fontsize=8.5,
                 loc="left")
    ax.legend(fontsize=6.5, loc="best", ncol=2, framealpha=0.9)
    fig.subplots_adjust(left=0.11, right=0.89, top=0.84, bottom=0.1)
    buf = io.BytesIO(); _tidy(fig); fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()


def _csv(dates_str, y_mm, label: str) -> bytes:
    """The chart's data (the report's series, vertical mm), so the chart can
    be checked and reused."""
    lines = [f"date,{label}_vertical_mm"]
    for d, v in zip(dates_str, y_mm):
        lines.append(f"{d},{'' if v is None or not np.isfinite(v) else f'{v:.2f}'}")
    return ("\n".join(lines) + "\n").encode()


def series_title(name: str, v: float | None, tot: float | None, relative: bool, ground: str = "farmland",
                 tot_range: list | None = None, nd_total: int = 2) -> str:
    """A chart's title: the rate to 0.01 in/yr as the page's chart
    (client_report._series_axes), and on a second line the total measured
    over the record only with its 95% range (E9: until 7 Oct 2026 the KMZ
    printed '-1.012 in/yr' and '-10.55 in measured over the record' with no
    range). nd_total: 2 for an area, 1 for a feature, as the page."""
    from . import report_insights as ri
    if v is None:
        return f"{name}: relative only (no GNSS anchor)"
    ok = tot is not None and tot_range and len(tot_range) >= 2 and None not in tot_range[:2]
    return (f"{name}: {ri.signed(v / MM_IN)} in/yr ({v:+.1f} mm/yr)"
            + (f",\n{ri.signed(tot / MM_IN, nd_total)} in measured over the record (95% "
               f"{ri.signed(tot_range[0] / MM_IN, nd_total)} to {ri.signed(tot_range[1] / MM_IN, nd_total)} in)"
               if ok else "")
            + (f"  (relative to the surrounding {ri.ground_noun(ground)})" if relative else ""))


# ------------------------------------------------------------------ inline charts (one per point)
SVG_W, SVG_H, SVG_MAX_VERTICES = 460, 170, 60
SVG_FRAME_MARK = "<!--frame-->"


def svg_chart(t, y_in, title: str, centre_in=None, half_in=None,
              band_label: str = "ordinary ground at this point's noise (95%)", start_year: float | None = None) -> str:
    """A point's movement chart as inline SVG (~2 kB): a PNG per point was
    ~25 kB, and the whole-field KMZ has tens of thousands of points. Same
    content as _chart: the series (in), its 95% band of ordinary ground
    widening with time, zero, the years along the bottom. Decimated to at
    most SVG_MAX_VERTICES by averaging consecutive dates (the line's shape,
    not one noisy image). t: years from the first date (sg.years);
    start_year: that date as a decimal year, for the year ticks."""
    t = np.asarray(t, float)
    y = np.asarray(y_in, float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return ""
    tt, yy = t[ok], y[ok]
    if tt.size > SVG_MAX_VERTICES:
        edges = np.linspace(0, tt.size, SVG_MAX_VERTICES + 1).astype(int)
        tt = np.array([tt[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
        yy = np.array([yy[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    lo_b = hi_b = None
    if half_in is not None:
        c = np.zeros_like(t) if centre_in is None else np.asarray(centre_in, float)
        h = np.asarray(half_in, float)
        if np.isfinite(h).all():
            lo_b, hi_b = c - h, c + h
    # the axis follows the series (and zero), not the band: the band of a
    # noisy cell reaches several inches by the end of a ten-year record and
    # flattened the line (Site B preview, 8 Oct 2026); it is clipped to
    # the plot, so a chart filled grey reads 'within ordinary ground'
    vals = [float(yy.min()), float(yy.max()), 0.0]
    ymin, ymax = min(vals), max(vals)
    pad = max(0.05, 0.08 * (ymax - ymin))
    ymin, ymax = ymin - pad, ymax + pad
    L, R, T, B = 46, 8, 20, 22
    W, H = SVG_W - L - R, SVG_H - T - B
    t0 = float(t[0])
    t1 = float(t[-1]) if t[-1] > t[0] else t0 + 1.0
    X = lambda v: L + (v - t0) / (t1 - t0) * W                         # noqa: E731
    Y = lambda v: T + (ymax - v) / (ymax - ymin) * H                   # noqa: E731
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{SVG_W}' height='{SVG_H}' "
           f"style='font:10px sans-serif;background:#fff'>"
           f"<clipPath id='pc'><rect x='{L}' y='{T}' width='{W}' height='{H}'/></clipPath>",
           f"<text x='{L}' y='13' font-size='11' font-weight='bold'>{html.escape(title)}</text>"]
    if lo_b is not None:
        # the band's edges are straight lines from the first date: four
        # corners, clipped to the plot
        pts = f"{X(t[0]):.0f},{Y(hi_b[0]):.0f} {X(t[-1]):.0f},{Y(hi_b[-1]):.0f} " \
              f"{X(t[-1]):.0f},{Y(lo_b[-1]):.0f} {X(t[0]):.0f},{Y(lo_b[0]):.0f}"
        out.append(f"<polygon points='{pts}' fill='#d8dee6' clip-path='url(#pc)'/>")
    out.append(f"<line x1='{L}' x2='{L + W}' y1='{Y(0):.1f}' y2='{Y(0):.1f}' stroke='#888' stroke-width='0.7'/>")
    span = ymax - ymin
    step = next((s for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if span / s <= 5), 100)
    nd = 2 if step < 0.1 else 1
    v = math.ceil(ymin / step) * step
    while v <= ymax + 1e-9:
        lab = f"{v:+.{nd}f}" if abs(v) > step / 2 else f"{0:.{nd}f}"
        out.append(f"<text x='{L - 4}' y='{Y(v) + 3:.1f}' text-anchor='end' fill='#555'>{lab}</text>")
        v += step
    # the line as a path of relative whole-pixel steps (half the bytes of absolute points)
    px_ = [(int(round(X(a))), int(round(Y(b)))) for a, b in zip(tt, yy)]
    out.append(f"<path fill='none' stroke='#b2182b' stroke-width='1.4' d='M{px_[0][0]},{px_[0][1]}l"
               + " ".join(f"{x - px_[k][0]},{y - px_[k][1]}" for k, (x, y) in enumerate(px_[1:])) + "'/>")
    out.append(f"<text x='{L + 4}' y='{T + H - 4}' fill='#555'>inches"
               + (f"; grey = {html.escape(band_label)}" if lo_b is not None else "") + "</text>")
    # what follows is the same for every point of a record (the years and
    # the frame): points_kml writes it once, in the shared balloon
    out.append(SVG_FRAME_MARK)
    if start_year is not None:
        ystep = 1 if (t1 - t0) <= 6 else 2
        for yr in range(int(math.floor(start_year)) + 1, int(start_year + t1) + 1, ystep):
            x = X(yr - start_year)
            out.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{T + H}' y2='{T + H + 3}' stroke='#555'/>"
                       f"<text x='{x:.1f}' y='{T + H + 14}' text-anchor='middle' fill='#555'>{yr}</text>")
    out.append(f"<rect x='{L}' y='{T}' width='{W}' height='{H}' fill='none' stroke='#999' stroke-width='0.6'/>")
    out.append("</svg>")
    return "".join(out)


def decimal_year(d: str) -> float:
    x = datetime.strptime(str(d), "%Y%m%d")
    return x.year + (x.timetuple().tm_yday - 1) / 365.25



# ------------------------------------------------------------------ the field's points and layers
FIELD_SPACING_M = 100.0     # field_points.SPACING_M: ~ one point per independent radar measurement
# 13 Oct 2026: the points are written straight into doc.kml, always loaded.
# The 8 Oct file put them in ~3 km tiles behind Regions and drew 94% of
# them as a 40%-opaque light-grey dot (the faded 'within normal variation'
# bin): on imagery the KMZ looked blank (Kyle, Site B rebuild).
POINT_SCALE, POINT_FADED_ALPHA = 0.5, 0.6
CELL_ALPHA, CELL_FADED_ALPHA = 0.95, 0.4
OVERLAY_TILE = 2048         # pixels per GroundOverlay side (Google Earth's safe texture size)
# signal-to-noise classes (unsigned): below 1, 1 to 1.96, 1.96 (stands out) to 3, 3 or more
SNR_BINS = ((0.0, 1.0, (0.80, 0.80, 0.80), "below 1"), (1.0, Z95, (0.996, 0.878, 0.545), "1 to 1.96"),
            (Z95, 3.0, (0.957, 0.427, 0.263), "1.96 to 3: stands out"),
            (3.0, 1e9, (0.647, 0.0, 0.149), "3 or more: stands out clearly"))
DOT = "dot.png"
POINT_TITLE = "Point "      # a point balloon's heading: "Point 40.12345 N, 100.12345 W"


def dot_png() -> bytes:
    """The points' icon, inside the KMZ (no fetch from Google's servers): a
    white disc with a thin dark rim, tinted by each style's colour."""
    from PIL import Image, ImageDraw
    n = 64
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((2, 2, n - 3, n - 3), fill=(255, 255, 255, 255), outline=(90, 90, 90, 255), width=3)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return buf.getvalue()


def point_parts(fd: dict, j: int, name: str, bins, relative_colours: bool, d0: str, d1: str,
                start_year: float, span_label: str = "over the record") -> dict:
    """Point j of a field_points table: its style and its balloon - the
    point's own cell's figures (the rule of rate_signal_to_noise.tif and the
    point table), the line-of-sight rows of the client rasters, and its chart
    (the 3 x 3 cells around it) inline - as point_description writes it."""
    from . import field_points as fp
    m = fd["meta"]
    t = np.asarray(m["t"])
    anchored, bg_v, bg_u = bool(m["anchored"]), float(m["bg_v"]), m.get("bg_u")
    lat, lon = float(fd["lat"][j]), float(fd["lon"][j])
    rel_v, rate, sd_v = float(fd["rel_v"][j]), float(fd["rate"][j]), float(fd["sd_v"][j])
    pv = point_values(rate, rel_v, sd_v, bg_u, anchored)
    fast = bool(fd["fast"][j])
    chart = None
    if not fast:
        u_ = math.hypot(sd_v, bg_u or 0.0) if np.isfinite(sd_v) else None
        chart = svg_chart(t, fp.series_vertical_in(fd, j), f"{latlon_text(lat, lon)}: movement {span_label}",
                          centre_in=(bg_v * t / MM_IN) if anchored else None,
                          half_in=(Z95 * u_ * t / MM_IN) if u_ else None, start_year=start_year)
    unwrap = float(fd["unwrap"][j])
    desc = point_description(name, lat, lon, pv, float(m["span"]), float(fd["tc"][j]), None, fast,
                             ground=m.get("ground") or "farmland", unwrap=unwrap if np.isfinite(unwrap) else None,
                             gate=float(m.get("gate") or 0.7), resolution=m.get("resolution"),
                             los={"rate": float(fd["los_rel"][j]), "change": float(fd["change_los"][j]), "d0": d0,
                                  "d1": d1}, chart=chart, span_label=span_label)
    col_v = rel_v if relative_colours else rate               # the map's frame
    style = "nm" if fast else f"b{_bin(col_v / MM_IN, bins)}{'f' if pv['faded'] else ''}"
    return {"style": style, "lat": lat, "lon": lon, "desc": desc, "fast": fast}


_ROW = re.compile(r"<tr><td>(.*?)</td><td>(.*?)</td></tr>", re.S)


def _split_balloon(desc: str):
    """point_description's text -> ('in <area>' line, lat/lon, [(label, value)], chart, footer)."""
    a = desc.index("<table")
    b = desc.index("</table>") + len("</table>")
    hm = re.match(r"<b>" + re.escape(POINT_TITLE) + r"(.*?)</b><br>(.*?)<br>$", desc[:a], re.S)
    rest = desc[b:]
    k = rest.rindex("<br><small>")
    return hm.group(2), hm.group(1), _ROW.findall(desc[a:b]), rest[:k], rest[k:]


def _data(k: str, v: str) -> str:
    v = _no_neg_zero(v)
    body = f"<![CDATA[{v}]]>" if ("<" in v or "&" in v) else v
    return f"<Data name='{k}'><value>{body}</value></Data>"


def points_kml(parts: list[dict], bins) -> tuple[str, list[str]]:
    """(styles, placemarks) for many points with ONE shared balloon: the row
    labels and the footer are written once, in each style's BalloonStyle,
    and every point carries only its values (ExtendedData, which Google
    Earth puts into the $[...] fields). The text Google Earth shows is
    point_description's, word for word (expand_placemarks, tested); a point
    whose balloon lacks a row the others have shows '—' there. About 1.5 kB a
    point instead of ~4.9 kB, so the whole field fits in doc.kml."""
    labels, footer = [], ""
    split = []
    for p in parts:
        if p["fast"]:
            split.append(None)
            continue
        s = _split_balloon(p["desc"])
        split.append(s)
        footer = footer or s[4]
        for lab, _ in s[2]:
            if lab not in labels:
                labels.append(lab)
    key = {lab: f"r{i}" for i, lab in enumerate(labels)}
    # the chart's opening tag and its frame (years, outline) are the same
    # for every point of a record: written once, around $[c]
    chart_head = chart_frame = ""
    for sp in split:
        c = sp[3] if sp else ""
        if c.startswith("<svg") and SVG_FRAME_MARK in c and sp[1] in c:
            # up to and including the title, whose only per-point part is the lat/lon ($[ll])
            end_title = c.index("</text>") + len("</text>")
            chart_head = c[:end_title].replace(sp[1], "$[ll]")
            chart_frame = c[c.index(SVG_FRAME_MARK):]
            break
    template = (f"<b>{POINT_TITLE}$[ll]</b><br>$[h]<br><table style='font-size:12px'>"
                + "".join(f"<tr><td>{lab}</td><td>$[{key[lab]}]</td></tr>" for lab in labels)
                + f"</table>{chart_head}$[c]{chart_frame}" + footer)

    def inner(c: str, ll: str) -> str:
        head = chart_head.replace("$[ll]", ll)
        if chart_head and c.startswith(head) and c.endswith(chart_frame):
            return c[len(head):len(c) - len(chart_frame)]
        return ""                                             # no chart (fewer than 3 dates): the frame alone
    bal = f"<BalloonStyle><text><![CDATA[{_no_neg_zero(template)}]]></text></BalloonStyle>"
    styles = "".join(
        f"<Style id='b{i}{suf}'><IconStyle><color>{_kml(rgb, POINT_FADED_ALPHA if suf else 1.0)}</color>"
        f"<scale>{POINT_SCALE}</scale><Icon><href>{DOT}</href></Icon></IconStyle>"
        f"<LabelStyle><scale>0</scale></LabelStyle>{bal}</Style>"
        for i, (_, _, rgb, _) in enumerate(bins) for suf in ("", "f"))
    # the icons over https (16 Oct 2026, readiness review DAT-05): Google
    # Earth fetched them over plain http
    styles += (f"<Style id='nm'><IconStyle><color>ff808080</color><scale>{POINT_SCALE}</scale><Icon><href>"
               "https://maps.google.com/mapfiles/kml/shapes/open-diamond.png</href></Icon></IconStyle>"
               "<LabelStyle><scale>0</scale></LabelStyle><BalloonStyle><text><![CDATA[$[x]]]></text></BalloonStyle></Style>")
    pms = []
    for p, s in zip(parts, split):
        if s is None:
            data = _data("x", p["desc"])
        else:
            vals = dict(s[2])
            data = (_data("h", s[0]) + _data("ll", s[1])
                    + "".join(_data(key[lab], vals.get(lab, "—")) for lab in labels) + _data("c", inner(s[3], s[1])))
        pms.append(f"<Placemark><styleUrl>#{p['style']}</styleUrl><ExtendedData>{data}</ExtendedData>"
                   f"<Point><coordinates>{p['lon']:.6f},{p['lat']:.6f},0</coordinates></Point></Placemark>")
    return styles, pms


def expand_placemarks(kml: str) -> list[tuple[str, str, str, str]]:
    """What Google Earth shows: (style, balloon html, lon, lat) for every
    point placemark, its style's BalloonStyle text with the $[...] fields
    filled from its ExtendedData (a placemark with its own description is
    taken as it is)."""
    tmpl = {sid: t for sid, t in re.findall(
        r"<Style id='(\w+)'>(?:(?!</Style>).)*?<BalloonStyle><text><!\[CDATA\[(.*?)\]\]></text>", kml, re.S)}
    out = []
    for m in re.finditer(r"<Placemark><styleUrl>#(\w+)</styleUrl>(?:<ExtendedData>(.*?)</ExtendedData>|"
                         r"<description><!\[CDATA\[(.*?)\]\]></description>)<Point><coordinates>([-\d.]+),([-\d.]+),0",
                         kml, re.S):
        sid, ext, desc, lon, lat = m.groups()
        if ext is not None:
            vals = {k: (a if a else html.unescape(b)) for k, a, b in re.findall(
                r"<Data name='(\w+)'><value>(?:<!\[CDATA\[(.*?)\]\]>|([^<]*))</value></Data>", ext, re.S)}
            desc = re.sub(r"\$\[(\w+)\]", lambda q: vals.get(q.group(1), q.group(0)), tmpl.get(sid, ""))
        out.append((sid, desc, lon, lat))
    return out


def point_placemark(fd: dict, j: int, name: str, bins, relative_colours: bool, d0: str, d1: str,
                    start_year: float) -> str:
    """One point with its own full description (no shared style needed)."""
    p = point_parts(fd, j, name, bins, relative_colours, d0, d1, start_year)
    return (f"<Placemark><styleUrl>#{p['style']}</styleUrl><description><![CDATA[{p['desc']}]]></description>"
            f"<Point><coordinates>{p['lon']:.6f},{p['lat']:.6f},0</coordinates></Point></Placemark>")


def cell_rgba_vertical(field_in: np.ndarray, snr: np.ndarray, fast, good: np.ndarray, bins) -> np.ndarray:
    """Every measured cell in the points' colours: the map bin of its
    vertical rate (in/yr, the points' frame), fainter where it is within
    normal variation (signal-to-noise below 1.96), grey where too fast."""
    rgba = np.zeros(field_in.shape + (4,))
    ok = good & np.isfinite(field_in)
    stand = ok & np.isfinite(snr) & (snr >= Z95)
    for i, (lo, hi, rgb, _) in enumerate(bins):
        lo_ = -np.inf if i == 0 else lo
        hi_ = np.inf if i == len(bins) - 1 else hi
        sel = ok & (field_in >= lo_) & (field_in < hi_)
        rgba[sel, :3] = rgb
        rgba[sel, 3] = np.where(stand[sel], CELL_ALPHA, CELL_FADED_ALPHA)
    if fast is not None:
        rgba[fast & good] = (0.5, 0.5, 0.5, 0.9)
    return rgba


def nice_limit(v: np.ndarray, pct: float = 98.0) -> float:
    """A round colour limit: the pct-th percentile of |v|, up to 1, 2 or 5 x 10^k."""
    v = np.abs(np.asarray(v, float))
    v = v[np.isfinite(v)]
    x = float(np.percentile(v, pct)) if v.size else 1.0
    if not x > 0:
        return 1.0
    p = 10 ** math.floor(math.log10(x))
    return next(m * p for m in (1, 2, 2.5, 5, 10) if m * p >= x)


def diverging_rgba(field: np.ndarray, lim: float) -> np.ndarray:
    """RdBu (red = negative = away from the satellite), clipped at +/- lim."""
    from matplotlib import colormaps
    from matplotlib.colors import TwoSlopeNorm
    rgba = colormaps["RdBu"](TwoSlopeNorm(0, -lim, lim)(np.clip(np.nan_to_num(field), -lim, lim)))
    rgba[~np.isfinite(field)] = (0, 0, 0, 0)
    return rgba


def snr_rgba(snr: np.ndarray) -> np.ndarray:
    rgba = np.zeros(snr.shape + (4,))
    for lo, hi, rgb, _ in SNR_BINS:
        sel = np.isfinite(snr) & (snr >= lo) & (snr < hi)
        rgba[sel] = (*rgb, 0.45 if lo == 0 else 1.0)
    return rgba


def _field_overlay(cfg, ctx, rgba: np.ndarray, tile: int = OVERLAY_TILE) -> list[tuple[bytes, tuple]]:
    """A per-cell RGBA field cropped to the study box (as _overlay crops the
    report's map), at full resolution - one 15 m cell per pixel - as PNG
    pieces of at most tile x tile pixels: [(png, (west, south, east, north))]."""
    from PIL import Image
    a = ctx.attrs
    y0, x0, dy, dx = (float(a[q]) for q in ("Y_FIRST", "X_FIRST", "Y_STEP", "X_STEP"))
    w, s_, e, n = cfg.bbox
    ny, nx = rgba.shape[:2]
    r0, r1 = max(0, int((n - y0) / dy)), min(ny, int(math.ceil((s_ - y0) / dy)))
    c0, c1 = max(0, int((w - x0) / dx)), min(nx, int(math.ceil((e - x0) / dx)))
    out = []
    for ra in range(r0, r1, tile):
        for ca in range(c0, c1, tile):
            rb, cb = min(r1, ra + tile), min(c1, ca + tile)
            sub = rgba[ra:rb, ca:cb]
            if not (sub[..., 3] > 0).any():
                continue
            im = Image.fromarray((np.clip(sub, 0, 1) * 255).astype(np.uint8), "RGBA")
            buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
            out.append((buf.getvalue(), (x0 + ca * dx, y0 + rb * dy, x0 + cb * dx, y0 + ra * dy)))
    return out


TREE_RGB, TREE_ALPHA = (0.10, 0.30, 0.12), 0.75


def tree_rgba(wood: np.ndarray) -> np.ndarray:
    """Wooded cells as dark-green diagonal stripes (every other 2-cell band),
    so the layer under them still shows through."""
    ny, nx = wood.shape
    rr, cc = np.ogrid[:ny, :nx]
    stripe = ((rr + cc) % 4) < 2
    out = np.zeros((ny, nx, 4), np.float32)
    on = wood & stripe
    out[on, 0], out[on, 1], out[on, 2] = TREE_RGB
    out[on, 3] = TREE_ALPHA
    return out


def tree_folder(pieces: list, ground: str = "farmland") -> str:
    """The Tree cover overlay: off at first, outside the one-at-a-time
    layers so it can lie over any of them."""
    ov = "".join(
        f"<GroundOverlay><name>Tree cover ({i + 1})</name><visibility>0</visibility><drawOrder>3</drawOrder>"
        f"<Icon><href>layers/trees_{i}.png</href></Icon><LatLonBox><north>{bn}</north><south>{bs}</south>"
        f"<east>{be}</east><west>{bw}</west></LatLonBox></GroundOverlay>"
        for i, (_, (bw, bs, be, bn)) in enumerate(pieces))
    note = ("Green stripes: tree cover (ESA WorldCover 2021). This satellite (C-band) sees the canopy, not the "
            f"ground beneath it, so movement read there is less certain. Trees are left out of the ordinary {ground} "
            "every figure is compared with.")
    return (f"<Folder><name>Tree cover (lay over any map layer)</name><visibility>0</visibility>"
            f"<description><![CDATA[{html.escape(note)}]]></description><styleUrl>#hide</styleUrl>{ov}</Folder>")


def _scale_legend(title: str, lim: float, note: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import TwoSlopeNorm
    fig, ax = plt.subplots(figsize=(2.6, 0.9), dpi=100)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.55, bottom=0.42)
    grad = np.linspace(-lim, lim, 256)[None, :]
    ax.imshow(grad, aspect="auto", cmap=colormaps["RdBu"], norm=TwoSlopeNorm(0, -lim, lim), extent=(-lim, lim, 0, 1))
    ax.set_yticks([]); ax.set_xticks([-lim, 0, lim])
    ax.set_xticklabels([f"{-lim:+g}", "0", f"{lim:+g}"], fontsize=8)
    ax.set_title(title, fontsize=8, loc="left")
    fig.text(0.08, 0.02, note, fontsize=6.5, color="#55616e", va="bottom")
    fig.patch.set_facecolor("white"); fig.patch.set_alpha(0.9)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


def _snr_legend(chance_pct) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.6, 1.6), dpi=100)
    ax.axis("off")
    ax.set_title("Signal-to-noise ratio\n|movement| / noise of ordinary ground", fontsize=8, loc="left")
    for i, (_, _, rgb, lab) in enumerate(SNR_BINS):
        ax.scatter([0.06], [0.85 - i * 0.17], s=60, color=rgb, edgecolor="k", lw=0.4)
        ax.text(0.16, 0.85 - i * 0.17, lab, va="center", fontsize=8)
    if isinstance(chance_pct, (int, float)):
        ax.text(0.0, 0.0, f"{chance_pct:.1f}% of ordinary ground stands out\nby chance", fontsize=6.5,
                color="#55616e", va="bottom")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.patch.set_facecolor("white"); fig.patch.set_alpha(0.9)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


def layers_guide(ground: str = "farmland", cell_m: float = 15.0) -> str:
    """What each folder holds, at the top of the file. cell_m: the product
    grid's posting (15 m on the Sentinel-1 routes and, since 26 Oct 2026,
    NISAR; 30 m on DISP-S1)."""
    g = html.escape(ground)
    items = [("Report areas", "each area's outline; click it for the report's figures and chart"),
             ("Features", "the report's numbered features"),
             ("Measurement points: vertical movement", "off when the file opens - tick it to show them. The whole "
              "study area, one point every ~100 m; click any point for its own figures, its signal-to-noise and "
              "its movement over the record"),
             ("Map layers (one at a time)", f"every measured {cell_m:.0f} m map cell, shown when the file opens; switch to "
              f"line-of-sight velocity, line-of-sight cumulative displacement, signal-to-noise compared with ordinary "
              f"{g}, or the report's smoothed map (each carries its own legend)")]
    return ("<p><b>In this file</b></p><ul>"
            + "".join(f"<li><b>{html.escape(a)}</b>: {b}</li>" for a, b in items) + "</ul>")


# ------------------------------------------------------------------ build
def _read_flag(paths, shape) -> np.ndarray | None:
    for p in paths:
        if p.exists():
            try:
                import rasterio
                with rasterio.open(p) as r:
                    a = r.read(1)
                if a.shape == tuple(shape):
                    return a == 1
                log.warning("%s is %s, the product grid %s: not used", p.name, a.shape, tuple(shape))
            except Exception as exc:                          # noqa: BLE001
                log.warning("%s not read (%s)", p.name, exc)
    return None


def build(cfg, out: Path | None = None, out_dir: Path | None = None, rep: dict | None = None, ctx=None,
          viewer_only: bool = False) -> Path:
    """The client KMZ. out_dir: a preview folder - the KMZ is written there
    (and a client_report.json there, a preview's, is read first); the
    deliverables are not touched. ctx: client_report.compute's context
    (already on the report's farmland); a fresh one is put there
    (results.report_farmland). The results viewer's copy of the points
    (<deliverables or out_dir>/viewer/field_points.npz) is written on the
    way; viewer_only stops there and returns its path."""
    from . import client_report as cr
    from . import report_insights as ri
    from . import results as rs
    D = Path(cfg.project_dir) / "deliverables"
    rep = rep if rep is not None else rs.report_record(cfg, out_dir)
    if not rep:
        raise FileNotFoundError(f"No client_report.json in {out_dir or D}: build the client report first")
    final, blockers = release_state(cfg, rep)
    screening = _screening(cfg, rep)
    if ctx is None:
        from .panel_history import _Context
        ctx = _Context(cfg)
        rs.report_farmland(ctx, rep)
    sg, attrs = ctx.sg, ctx.attrs
    anchored = bool(rep.get("anchored"))
    ground = ri.ground_word(rep)          # the comparison ground's name in every client string (generality-16)
    bg = _gnss_bg(rep)
    bg_v = float(bg.get("vertical_mm_yr") or 0.0) if anchored else 0.0
    up_map = cr._los_up(cfg, ctx.vel.shape)
    try:
        model = json.loads((D / "uncertainty_model.json").read_text())
    except (OSError, ValueError):
        model = {}
    # the per-cell farmland null by class INDEX (empirical-11: by list
    # position a missing class shifted every sigma; a made-up 5 mm/yr filled
    # the gaps)
    null_map = cr.cell_sigma_map(rep.get("cell_null_by_class"), ctx.kcls, model)
    # the second quality check and the too-fast flag per point are read by
    # field_points.gather (phaselinked-6, gap-surface_mine_site-8)
    from . import deliverables as dl
    py, px = sg.pixel_metres(attrs)
    px_m = math.sqrt(py * px)
    lim = float(rep.get("map_colour_limit_in_per_yr") or MAP_LIM_IN)
    # the client map's own bins (relative to the comparison ground); a
    # report from before 5 Oct 2026 has none: its fixed absolute scale
    bins = scale_bins(rep.get("map_scale"))
    relative_colours = bins is not None
    bins = bins or point_bins(lim)

    # ---- report areas and the report's listed features (no QA-only stand-out, F2)
    groups = []
    for name, spec in cr.parse_areas(cfg):
        got = cr._mask(ctx, spec)
        rec = next((a for a in rep["areas"] if a["name"] == name), None)
        # the name the client reads ('project_zone' -> 'Project Zone'; the
        # release gate still holds the report until it is renamed)
        shown = (rec or {}).get("display_name") or name
        groups.append({"kind": "area", "name": shown, "key": name, "mask": got[0] & ctx.good, "rings": got[1],
                       "rec": rec})
    feats = rep.get("features") or []
    for i, f in enumerate(feats, 1):
        ring = f.get("ring") or (rs._bbox_ring(f["bbox"]) if f.get("bbox") else None)
        if not ring:
            continue
        groups.append({"kind": "feature", "name": f"Feature {i}", "i": i, "f": f, "rings": [np.asarray(ring)],
                       "mask": rs.ring_mask(ring, attrs, ctx.good.shape) & ctx.good})
    # the quality assessment's stand-out is drawn only as the feature the
    # report lists it under (F2, 7 Oct 2026): one the QA found alone stays in
    # its technical record. Until then a stand-out no feature held was drawn
    # as "Deepest patch (quality assessment) ... Worth checking on the
    # ground" beside a page saying no feature stands out (Site A's 4 km
    # stand-out once Feature 1 was no longer listed)
    # ---- the points: a ~100 m grid over the ground of every report area
    # (field_points; until 8 Oct 2026 ~400 per area, so the field between
    # features looked empty) and each listed feature's own denser sample
    from . import field_points as fp
    field_mask = np.zeros(ctx.good.shape, bool)
    for g in groups:
        if g["kind"] == "area":
            field_mask |= g["mask"]
    fr, fc, spacing = fp.grid_cells(field_mask, px_m, FIELD_SPACING_M)
    cells = [(fr, fc)]
    feat_gi = [gi for gi, g in enumerate(groups) if g["kind"] == "feature"]
    for gi in feat_gi:
        s = _sample(groups[gi]["mask"], px_m, PER_FEATURE)
        cells.append((np.array([r for r, _ in s], int), np.array([c for _, c in s], int)))
    change = fp.relative_change(ctx)
    fd = fp.gather(cfg, rep, ctx, cells, up_map=up_map, null_map=null_map, change=change)
    full_rec = next((a for a in rep.get("areas", []) if a.get("kind") == "full"), None)
    field_name = (full_rec or {}).get("display_name") or (full_rec or {}).get("name") or "Study area"
    names = [field_name] + [groups[gi]["name"] for gi in feat_gi]
    log.info("Client KMZ: %d points (%d over the field at ~%.0f m, %d in %d features)", fd["row"].size, fr.size,
             spacing, fd["row"].size - fr.size, len(feat_gi))
    # the viewer's copy (results_viewer): the same table, never recomputed there
    vpath = fp.save(fd, fp.viewer_path(cfg, out_dir), extra={
        "names": names, "spacing_m": spacing, "bins": [[lo, hi, list(rgb), lab] for lo, hi, rgb, lab in bins],
        "relative_colours": relative_colours, "lim_in": lim,
        "areas": [{"name": g["name"], "rings": [np.asarray(r)[:, :2].round(6).tolist() for r in g["rings"]]}
                  for g in groups if g["kind"] == "area"],
        "features": [{"name": g["name"], "rings": [np.asarray(r)[:, :2].round(6).tolist() for r in g["rings"]]}
                     for g in groups if g["kind"] == "feature"]})
    log.info("Viewer points: %s", vpath)
    if viewer_only:
        return vpath

    files: dict[str, bytes] = {}
    t = ctx.t
    dates = [datetime.strptime(d, "%Y%m%d") for d in ctx.dates]
    d0, d1 = ctx.dates[0], ctx.dates[-1]
    start_year = decimal_year(d0)
    parts = [point_parts(fd, j, names[int(fd["group"][j])], bins, relative_colours, d0, d1, start_year)
             for j in range(fd["row"].size)]
    pstyles, pms = points_kml(parts, bins)
    field_pms = [pm for j, pm in enumerate(pms) if fd["group"][j] == 0]
    feat_pm_pts = {gi: [pm for j, pm in enumerate(pms) if fd["group"][j] == g + 1] for g, gi in enumerate(feat_gi)}
    files[DOT] = dot_png()
    pts_n = int(fd["row"].size)

    # ---- area and feature placemarks with their charts (the
    # report's own chart items: client_report.series_items)
    dstr = list(ctx.dates)
    area_pm, feat_pm = [], []
    try:
        items = {it["key"]: it for it in cr.series_items(rep, max_features=None)}
    except (KeyError, TypeError, ValueError) as exc:          # a report from before 5 Oct 2026
        log.info("Charts without the report's bands (%s)", exc)
        items = {}
    bg_line = bg_v if anchored else 0.0
    for gi, g in enumerate(groups):
        coords = _polygon(g["rings"])
        if g["kind"] == "area":
            rec = g["rec"]
            if rec is None:
                desc = f"<b>{html.escape(g['name'])}</b><br>Not in the client report (it was built before this area was added)."
            else:
                png = None
                ser = rec.get("series_vertical_mm")
                if ser:
                    y = np.array([np.nan if q is None else q for q in ser], dtype=float)
                    v = rec.get("vertical_mm_yr")
                    png = f"charts/area{gi}.png"
                    title = series_title(g["name"], v, rec.get("total_mm"), not anchored, ground, _total_range(rec))
                    if g["key"] in items:
                        files[png] = _report_chart(dates, t, items[g["key"]], bg_line, title)
                    elif rec.get("kind") == "full" or "testable" in rec:
                        files[png] = _full_chart(dates, t, rec, g["name"], bg_line, anchored, ground)
                    else:                                     # a report from before 5 Oct 2026
                        unc = rec.get("vertical_uncertainty_1sigma_mm_yr")
                        files[png] = _chart(dates, y / MM_IN, title, centre_in=(bg_v * t / MM_IN) if anchored else None,
                                            half_in=(Z95 * unc * t / MM_IN) if unc else None, line_label=g["name"])
                    files[f"charts/area{gi}.csv"] = _csv(dstr, y, "area")
                desc = area_description(rec, rep, png)
            area_pm.append(f"<Placemark><name>{html.escape(g['name'])}</name><styleUrl>#area{min(gi, 3)}</styleUrl>"
                           f"<description><![CDATA[{desc}]]></description><MultiGeometry>{coords}</MultiGeometry></Placemark>")
        elif g["kind"] == "feature":
            f, i = g["f"], g["i"]
            png = None
            if f.get("series_vertical_mm"):
                y = np.array([np.nan if q is None else q for q in f["series_vertical_mm"]], dtype=float)
                sd = f.get("farmland_spread_1sigma_mm_yr")
                png = f"charts/feature{i}.png"
                title = series_title(f"Feature {i}", f.get("vertical_relative_mm_yr"), f.get("total_relative_mm"), True,
                                     ground, _total_range(f), nd_total=1)
                if f"Feature {i}" in items:
                    files[png] = _report_chart(dates, t, items[f"Feature {i}"], 0.0, title)
                elif not f.get("ground_reshaped"):
                    files[png] = _chart(dates, y / MM_IN, title, half_in=(Z95 * sd * t / MM_IN) if sd else None,
                                        line_label=f"Feature {i}", band_label=f"range of ordinary {ground} (95%)")
                else:                                         # reshaped ground: the report gives no rate or chart
                    png = None
                files[f"charts/feature{i}.csv"] = _csv(dstr, y, "relative")
            desc = feature_description(i, f, rep, png)
            feat_pm.append(f"<Placemark><name>Feature {i}</name><styleUrl>#feature</styleUrl>"
                           f"<description><![CDATA[{desc}]]></description><MultiGeometry>{coords}</MultiGeometry></Placemark>")

    # ---- map layers, per 15 m cell at full resolution: every measured cell
    # in the points' colours (ON, so the whole field shows the moment the
    # file opens), line of sight, signal-to-noise and the report's map
    files["overlay.png"], box = _overlay(cfg, ctx, up_map, bg_v, rep, lim, px_m)
    files["legend.png"] = _legend(rep, bins, lim, relative_colours)
    in_box = ctx.good & (ctx.aoi if getattr(ctx, "aoi", None) is not None else True)
    los_v_in = ctx.r_cm / MM_IN
    chg_in = change / MM_IN
    with np.errstate(invalid="ignore", divide="ignore"):
        snr_map = np.where(np.isfinite(null_map) & (null_map > 0), np.abs(ctx.r_cm) / null_map, np.nan)
        vert_in = (ctx.r_cm / up_map + (0.0 if relative_colours else bg_v)) / MM_IN
    fast_map = fp._read_flag([D / "quality_too_fast_to_measure.tif", D / "technical" / "quality_too_fast_to_measure.tif"],
                             ctx.good.shape)
    lim_v, lim_c = nice_limit(los_v_in[in_box]), nice_limit(chg_in[in_box])
    layers = {
        "cells": _field_overlay(cfg, ctx, cell_rgba_vertical(vert_in, snr_map, fast_map, ctx.good, bins)),
        "los_velocity": _field_overlay(cfg, ctx, diverging_rgba(np.where(ctx.good, los_v_in, np.nan), lim_v)),
        "los_change": _field_overlay(cfg, ctx, diverging_rgba(np.where(ctx.good, chg_in, np.nan), lim_c)),
        "snr": _field_overlay(cfg, ctx, snr_rgba(np.where(ctx.good, snr_map, np.nan)))}
    # tree cover, laid over any layer (landcover, 15 Oct 2026): C-band sees the canopy there
    from . import landcover
    wood = landcover.wooded_on_grid(cfg, ctx.attrs, ctx.good.shape)
    if wood is not None and wood.any():
        layers["trees"] = _field_overlay(cfg, ctx, tree_rgba(wood))
    for k, pieces in layers.items():
        for i, (png, _) in enumerate(pieces):
            files[f"layers/{k}_{i}.png"] = png
    files["legend_los_velocity.png"] = _scale_legend(f"Line of sight, in/yr\n(relative to ordinary {ground})",
                                                     lim_v, "+ = towards the satellite\nred = away (sinking)")
    files["legend_los_change.png"] = _scale_legend(
        f"Line of sight, in, {_date_text(d0)}\nto {_date_text(d1)} (relative to ordinary {ground})", lim_c,
        "first image to last; single images\ncarry their own noise")
    files["legend_snr.png"] = _snr_legend(fd["meta"].get("snr_chance_pct"))

    gnss = "".join(
        f"<Placemark><name>GNSS {html.escape(st['site'])}</name><styleUrl>#gnss</styleUrl><description><![CDATA["
        f"GNSS station {html.escape(st['site'])}, {st['distance_km']:.0f} km from the area centre; vertical "
        f"{st['vertical_over_record_mm_yr']['rate_mm_yr']:+.1f}"
        # with its uncertainty (every figure carries one, E9)
        + (f" &plusmn; {float(st['vertical_over_record_mm_yr']['sigma_mm_yr']):.1f} mm/yr (1 sigma)"
           if st["vertical_over_record_mm_yr"].get("sigma_mm_yr") is not None else " mm/yr")
        + " over the record's dates"
        + (", used for the regional motion" if st.get("used_for_background") else "") + "]]></description>"
        f"<Point><coordinates>{st['lon']:.5f},{st['lat']:.5f},0</coordinates></Point></Placemark>"
        for st in (rep.get("gnss") or {}).get("stations", []) if st.get("vertical_over_record_mm_yr"))
    styles = pstyles
    area_cols = ["ff30251c", "ff4a7a0f", "ffa03f6b", "ff00a5ff"]
    styles += "".join(f"<Style id='area{i}'><LineStyle><color>{c}</color><width>2.5</width></LineStyle>"
                      f"<PolyStyle><fill>0</fill></PolyStyle></Style>" for i, c in enumerate(area_cols))
    styles += ("<Style id='feature'><LineStyle><color>ff1c2530</color><width>2.5</width></LineStyle>"
               "<PolyStyle><color>401a1a8b</color></PolyStyle></Style>")
    styles += ("<Style id='gnss'><IconStyle><scale>1.1</scale><Icon><href>https://maps.google.com/mapfiles/kml/shapes/"
               "triangle.png</href></Icon></IconStyle></Style>")
    styles += "<Style id='hide'><ListStyle><listItemType>checkHideChildren</listItemType></ListStyle></Style>"
    styles += "<Style id='radio'><ListStyle><listItemType>radioFolder</listItemType></ListStyle></Style>"
    w, s_, e, n = box
    feat_folders = "".join(
        f"<Folder><name>{html.escape(groups[gi]['name'])} - points</name><open>0</open>{''.join(feat_pm_pts[gi])}</Folder>"
        for gi in feat_gi)
    chance = fd["meta"].get("snr_chance_pct")
    field_note = (f"{len(field_pms):,} points, one every ~{spacing:.0f} m over the whole study area - about one per "
                  f"independent radar measurement (each value combines ground within about 90 m x 240 m). Colour = "
                  f"vertical movement per year on the report map's scale; lighter = within normal variation for a "
                  f"single point. Click a point for its figures and its movement over the record.")
    snr_note = ("Signal-to-noise ratio: |movement relative to ordinary " + ground + "| / the noise of ground of the "
                "same kind. 1.96 or more stands out"
                + (f"; {chance:.1f}% of ordinary {ground} does so by chance, so only a larger share means anything"
                   if isinstance(chance, (int, float)) else "") + ". It says how clearly the ground moved, not which way.")
    obs = ri.observations(rep)
    feat_note = "" if feats else next((o["text"] for o in obs if o.get("kind") == "feature"), "")
    try:
        from . import release
        nh = release.numbers_hash(rep)
    except (ImportError, AttributeError):
        nh = ""

    def layer(name, key, legend, note, visible=False):
        ov = "".join(
            f"<GroundOverlay><name>{html.escape(name)} ({i + 1})</name><visibility>{int(visible)}</visibility>"
            f"<drawOrder>1</drawOrder><Icon><href>layers/{key}_{i}.png</href></Icon><LatLonBox><north>{bn}</north>"
            f"<south>{bs}</south><east>{be}</east><west>{bw}</west></LatLonBox></GroundOverlay>"
            for i, (_, (bw, bs, be, bn)) in enumerate(layers[key]))
        leg = ("" if not legend else
               f"<ScreenOverlay><name>Legend</name><visibility>{int(visible)}</visibility><Icon><href>{legend}</href></Icon>"
               f"<overlayXY x='1' y='0' xunits='fraction' yunits='fraction'/><screenXY x='0.99' y='0.06' "
               f"xunits='fraction' yunits='fraction'/><size x='0' y='0' xunits='pixels' yunits='pixels'/></ScreenOverlay>")
        return (f"<Folder><name>{html.escape(name)}</name><visibility>{int(visible)}</visibility>"
                f"<description><![CDATA[{html.escape(note)}]]></description><styleUrl>#hide</styleUrl>{ov}{leg}</Folder>")

    kml = (f"<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
           f"<name>{html.escape(doc_name(cfg, rep, final, screening or rep.get('study') is not None))}</name><open>1</open>"
           f"<LookAt><longitude>{(w + e) / 2:.6f}</longitude><latitude>{(s_ + n) / 2:.6f}</latitude><altitude>0</altitude>"
           f"<range>{1.6 * max((n - s_) * 111e3, (e - w) * 111e3 * math.cos(math.radians((s_ + n) / 2))):.0f}</range>"
           f"<tilt>0</tilt><heading>0</heading></LookAt>"
           f"<description><![CDATA[{doc_description(cfg, rep, final, blockers, screening)}{layers_guide(ground, px_m)}]]></description>"
           f"<ExtendedData><Data name='numbers_hash'><value>{nh}</value></Data>"
           # the code that wrote the file (16 Oct 2026, readiness review OUT-01)
           f"<Data name='software'><value>{html.escape(dl._software())}</value></Data>"
           f"<Data name='final'><value>{str(final).lower()}</value></Data>"
           f"<Data name='route'><value>{html.escape(str(rep.get('route') or ''))}</value></Data></ExtendedData>{styles}"
           f"<ScreenOverlay><name>Legend</name><Icon><href>legend.png</href></Icon>"
           f"<overlayXY x='0' y='0' xunits='fraction' yunits='fraction'/><screenXY x='0.01' y='0.06' xunits='fraction' yunits='fraction'/>"
           f"<size x='0' y='0' xunits='pixels' yunits='pixels'/></ScreenOverlay>"
           f"<Folder><name>Report areas</name><open>1</open>{''.join(area_pm)}</Folder>"
           f"<Folder><name>Features</name><open>1</open>"
           + (f"<description><![CDATA[{html.escape(feat_note)}]]></description>" if feat_note else "")
           + f"{''.join(feat_pm)}</Folder>"
           + study_folders(cfg, rep)
           # the points are off when the file opens (26 Oct 2026, Kyle: they made
           # Google Earth slow to load); the 15 m cell layer below is what shows
           + f"<Folder><name>Measurement points: vertical movement (every ~{spacing:.0f} m)</name>"
           f"<visibility>0</visibility><open>0</open>"
           f"<styleUrl>#hide</styleUrl><description><![CDATA[{html.escape(field_note)}]]></description>"
           f"{''.join(field_pms)}</Folder>"
           + (f"<Folder><name>Feature points</name><visibility>0</visibility><open>0</open>{feat_folders}</Folder>"
              if feat_folders else "")
           + "<Folder><name>Map layers (one at a time)</name><open>1</open><styleUrl>#radio</styleUrl>"
           + layer(f"Vertical movement, every measured {px_m:.0f} m cell", "cells", None,
                   "Every measured cell in the points' colours (the legend at the bottom left); lighter = within "
                   "normal variation for that cell.", visible=True)
           + layer("Line-of-sight velocity (relative)", "los_velocity", "legend_los_velocity.png",
                   f"Each cell's line-of-sight rate relative to ordinary {ground} of the same kind (the client raster "
                   f"line_of_sight_velocity_in_per_yr.tif). + = towards the satellite.")
           + layer("Line-of-sight cumulative displacement (relative)", "los_change", "legend_los_change.png",
                   f"Each cell's line-of-sight change from the first image ({_date_text(d0)}) to the last "
                   f"({_date_text(d1)}), relative to ordinary {ground} of the same kind (the client raster "
                   f"line_of_sight_cumulative_displacement_in_{d0}_{d1}.tif).")
           + layer("Signal-to-noise ratio", "snr", "legend_snr.png", snr_note)
           + f"<GroundOverlay><name>The report's map (smoothed; grey = normal variation)</name><visibility>0</visibility>"
           f"<color>d0ffffff</color><drawOrder>1</drawOrder><Icon><href>overlay.png</href></Icon><LatLonBox><north>{n}</north>"
           f"<south>{s_}</south><east>{e}</east><west>{w}</west></LatLonBox></GroundOverlay>"
           + "</Folder>"
           + (tree_folder(layers["trees"], ground) if layers.get("trees") else "")
           + f"<Folder><name>GNSS stations</name>{gnss}</Folder></Document></kml>")
    if out is None:
        folder = Path(out_dir) if out_dir else D
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / kmz_name(cfg, rep, final, screening)
    out = Path(out)
    part = out.with_name(out.name + ".partial")
    try:
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as z:
            # no '-0.00' anywhere in the client text (report_insights.no_negative_zero),
            # and "farmland" only where the comparison ground is farmland (ri.say_ground)
            z.writestr("doc.kml", _no_neg_zero(ri.say_ground(kml, ground)))
            for name, data in files.items():
                z.writestr(name, _no_neg_zero(ri.say_ground(data, ground)) if isinstance(data, str) else data)
        with zipfile.ZipFile(part) as z:                      # the file holds what was written
            if z.testzip() is not None or "doc.kml" not in z.namelist():
                raise RuntimeError(f"{part.name} did not write completely")
        os.replace(part, out)
    finally:
        if part.exists():
            part.unlink()
    if out_dir is None and out.parent.resolve() == D.resolve():
        # One client KMZ in deliverables/: the other release state's file (or
        # one under an earlier title) states other figures, under a name that
        # may not say DRAFT - Site B's 29 Sep Site B_ground_movement.kmz
        # sat beside the 6 Oct _DRAFT one (7 Oct 2026). Moved aside, never
        # deleted; the package skips superseded*.
        others = [k.name for k in D.glob("*ground_movement*.kmz") if k.name != out.name]
        if others:
            dl._set_aside(D, others, "kmz", "another release state or an earlier build of the client KMZ")
    log.info("Client KMZ: %s (%.1f MB, %d points, %d features%s%s)", out, out.stat().st_size / 1e6, pts_n,
             len(feat_pm), "" if final else ", DRAFT", ", SCREENING" if screening else "")
    return out


def _overlay(cfg, ctx, up_map, bg_v, rep, lim, px_m) -> tuple[bytes, tuple]:
    """The report's map as a transparent PNG over the study box: the same
    smoothed field (report_insights' smoothing at rep['map_smoothing_m']),
    the same grey band (rep['map_band'], centred on farmland - stats-7) and
    the same bins and colours as client_report.figures (rep['map_scale'],
    relative to the comparison ground)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    from matplotlib.colors import TwoSlopeNorm
    from PIL import Image
    from . import report_insights as ri
    rel_v = ctx.r_cm / up_map
    ok = np.isfinite(rel_v) & ctx.good
    k = max(3, int(round(float(rep.get("map_smoothing_m") or 500.0) / px_m)))
    if hasattr(ri, "_smooth"):
        sm = ri._smooth(rel_v, ok, k)
    else:
        sm = ri.smoothed_field(ctx, up_map)["field"]
    a = ctx.attrs
    y0, x0, dy, dx = (float(a[q]) for q in ("Y_FIRST", "X_FIRST", "Y_STEP", "X_STEP"))
    w, s_, e, n = cfg.bbox
    r0, r1 = max(0, int((n - y0) / dy)), int((s_ - y0) / dy)
    c0, c1 = max(0, int((w - x0) / dx)), int((e - x0) / dx)
    sub = sm[r0:r1, c0:c1]
    ms = rep.get("map_scale") or {}
    if ms.get("edges_mm_yr"):
        # the client map's own bins and colours, on the field relative to the
        # comparison ground (clientout-5: the report's map frame)
        from . import client_report as cr
        cmap, norm = cr._map_cmap([float(x) for x in ms["edges_mm_yr"]], int(ms.get("n_below") or 0))
        rgba = cmap(norm(np.nan_to_num(sub)))
    else:                                                     # a report from before 5 Oct 2026: absolute, fixed
        vin = (sub + bg_v) / MM_IN
        rgba = colormaps["RdBu"](TwoSlopeNorm(0, -lim, lim)(np.clip(np.nan_to_num(vin), -lim, lim)))
    band = rep.get("map_band") or {}
    if band.get("low_mm_yr") is not None:
        quiet = (sub > band["low_mm_yr"]) & (sub < band["high_mm_yr"])
    elif rep.get("map_band_mm_yr"):
        quiet = np.abs(sub) < rep["map_band_mm_yr"]
    else:
        quiet = np.zeros(sub.shape, bool)                     # no band measured: nothing is called normal
    rgba[quiet] = (*GREY, 0.55)
    rgba[~np.isfinite(sub)] = (0, 0, 0, 0)
    im = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    box = (x0 + c0 * dx, y0 + r1 * dy, x0 + c1 * dx, y0 + r0 * dy)
    return buf.getvalue(), box


def _legend(rep, bins, lim, relative: bool = False) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import report_insights as ri
    n = len(bins)
    step = min(0.075, 0.52 / max(n, 1))
    fig, ax = plt.subplots(figsize=(2.9, 3.9 + max(0, n - 7) * 0.25), dpi=100)
    ax.axis("off")
    ground = ri.ground_noun(ri.ground_word(rep))                  # already in brackets: no land-cover note
    ax.set_title(f"Vertical movement, in/yr\n(relative to the surrounding {ground})" if relative
                 else "Vertical movement, in/yr", fontsize=9, loc="left")
    for i, (_, _, rgb, lab) in enumerate(bins):
        ax.scatter([0.08], [0.95 - i * step], s=70, color=rgb, edgecolor="k", lw=0.4)
        ax.text(0.2, 0.95 - i * step, f"{lab}  (red = sinking)" if i == 0 else lab, va="center", fontsize=8)
    ax.text(0.02, 0.95 - n * step - 0.02, "faded = within normal variation for a single\npoint (signal ÷ threshold < 1)\n"
            "◇ = not measurable: too fast for this satellite", fontsize=7, color="#55616e", va="top")
    band = rep.get("map_band") or {}
    if band.get("low_mm_yr") is not None:
        txt = (f"Map overlay: the same colours; grey = within\nnormal ground variation "
               f"({band['low_mm_yr'] / MM_IN:+.2f} to {band['high_mm_yr'] / MM_IN:+.2f} in/yr)")
    else:
        txt = f"Map overlay: the same colours (no grey band:\ntoo little {ground} to measure its range)"
    ax.text(0.02, 0.02, txt, fontsize=7, color="#55616e", va="bottom")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.patch.set_facecolor("white"); fig.patch.set_alpha(0.9)
    buf = io.BytesIO(); _tidy(fig); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


def main(argv=None) -> int:
    import argparse
    import sys
    from .config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m insar_pipeline.client_kmz", description="The client Google Earth file")
    ap.add_argument("config")
    ap.add_argument("--out-dir", help="preview folder; deliverables untouched (a client_report.json there is used)")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    print(build(load_config(a.config), out_dir=Path(a.out_dir) if a.out_dir else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
