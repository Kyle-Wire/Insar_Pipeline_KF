"""
What the client report says about a project, worked out from the project's
own products, so the report reads right for any area - not just the one it
was first written for.

  identity(cfg)            title and place line (config, else the folder name
                           and a cached reverse geocode of the area's centre)
  mine_layers(cfg)         mapped mines by type: underground workings, surface
                           mines, other known movement (or one untyped layer)
  smoothed_field(ctx)      the report map's field: class-matched vertical rate
                           relative to farmland, averaged over ~500 m, and the
                           band ordinary farmland stays within at that scale
                           (farmland's own median and 2.5/97.5% quantiles,
                           once real movement is taken out of the farmland)
  find_features(...)       patches that go well beyond that band, largest first,
                           each with where it sits relative to the mapped mines
  feature_null(...)        the same search run on ordinary farmland only: how
                           far chance alone reaches in an area this size
  selection_p(f, null)     the chance that farmland alone produces a patch at
                           least this strong somewhere in the searched area;
                           sets f["stands_out_clearly"] (both tests, E7; the
                           QA's own calibration sound, H1: qa_standout_doubt)
  relation_to_mines(...)   inside / next to (within NEAR_MINE_KM) / the distance
                           to the nearest mapped mine (M2)
  passes_selection(k, n)   the exact Monte Carlo rule that decides listing:
                           (1 + k) / (1 + n), times the patches re-tested,
                           at most SELECTION_ALPHA / 2, in integers (G1, G2)
  feature_null_check(null) whether the simulated copies are as strong as the
                           real farmland (E8; a reviewer item when not)
  leading_candidates(...)  the first pass's strongest candidates, each re-tested
                           with itself left out of the farmland (F1;
                           match_candidate, leave_out_result; the re-tests
                           corrected for their number, G2: leave_out_chance)
  p_text(k, n)             the chance in words: "less than 1 in 200" when a
                           feature beats every copy (a bound), never "0.005"
  dem_change(...)          "ground reshaped since the elevation model", tested
                           after removing what the movement itself puts into
                           MintPy's DEM error (E1); reshaped_since_dem(f)
  cells_standing_out(...)  the share of single cells beyond the per-cell rule,
                           on the report's own (second-pass) farmland (E14);
                           cells_sentence() words it ("the same as chance"
                           only inside chance's 95% binomial range, F7a)
  trend_shape(t, series)   steady / began around / already sinking at the start /
                           fastest in / slowing - each word only when it beats
                           farmland's year-to-year spread ("fastest in" against
                           the same choice of years made on farmland's own
                           noise, E2; "began" only after ordinary years, M1)
  corrections_applied(...) what was removed from the signal, in plain words,
                           from what the run RECORDS it applied
  feature_text(i, f)       the Observations paragraph for one feature
  observations(rep)        the report's feedback: features, mine ground, and
                           the data-quality notes that apply to this record
  map_scale(...)           the client map's colour bins, from the data (5 Oct
                           2026: a fixed +/-0.6 in/yr scale saturated 30% of
                           Site B's coloured cells, the deepest at -2.9)
  map_patches(...)         every coloured patch inside a report area, named as
                           a feature or as within chance
  basemap(...)             a cached public-domain map (USGS The National Map)
                           under the client map; None offline, never an error

Nothing here changes a number in the technical products; it reads them and
chooses words. Wording is deliberately cautious: a feature is "worth a look",
and the cause is suggested from the mapped context, never asserted. A
direction word ("sinking", "rising") is used for an area only when its 95%
range excludes zero (direction()).
"""
from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from . import domains

log = logging.getLogger(__name__)
MM_IN = 25.4
Z95 = 1.959964
MIN_FEATURE_KM2 = 0.25      # smaller than this is a few fields; chance makes plenty
STRONG = 2.0                # a feature's peak must pass this many farmland bands
MAX_FEATURES = 5
# the default study's wording and limits live in domains.py (S0.4, 8 Nov 2026)
KIND_LABEL = dict(domains.DEFAULT.wording.feature_noun)
# the nearest one of each, for "1.8 km from the nearest mapped mine area"
NEAREST_LABEL = dict(domains.DEFAULT.wording.nearest_noun)
# "Next to" mapped mining - and the mining causes and sideways-movement
# caveat that go with it - only this close to a mapped outline (M2, final
# check of 7 Oct 2026; relation_to_mines). 2 km until then.
NEAR_MINE_KM = domains.DEFAULT.limits.near_km
# The band needs enough farmland to be a measurement: ~40 independent 500 m
# patches. Below it the map has no grey band and no feature can be assessed.
MIN_BAND_KM2 = 10.0
# The calibrated feature search (review of 5 Oct 2026, stats-6). Until then a
# searched-for feature had to beat 1.25 x a GAUSSIAN expectation of the
# deepest 1 km patch, a margin set by eye; Site B Feature 3 cleared it by
# 0.5 mm/yr while ordinary farmland of an equal area produced a patch of the
# same strength (-19.4 against -20.4 mm/yr), and with no QA chance level the
# gate fell open (chance 0, every patch "clears what chance alone produces").
SELECTION_ALPHA = 0.05      # for the whole search, both signs: 0.025 a side
# The Monte Carlo rule (G1, calibration of 7 Oct 2026): a patch beats chance
# when p = (1 + k) / (1 + n) <= SELECTION_ALPHA / 2, compared in integers
# (passes_selection) - with 199 copies k <= 4, an exact 2.5% a side because
# 200 x 0.025 is a whole number. Until then the rule was p < 0.025 (k <= 3,
# 2.0% a side): Site B's 2.2 km² patch, 4 of 199 copies as strong
# (p 5/200), sat on the line and was not listed.
# Simulated farmland copies; beating all of them gives p = 1/200 (F3,
# follow-up of 7 Oct 2026: 59 until then, when a listing could hang on one
# copy - Site B's 2.2 km² patch beat all 59 by 57.0 to 54.8, and 4 of 199
# are as strong as it). The rule was 199 if the products run grows by under
# 10 minutes on Site B, else 99. Measured on the Site B copy
# (work_fu_features/time_null.py, fu_dump.py): a null takes 32 s at 59, 47 s
# at 99, 89 s at 199 copies; with F1's leave-out re-tests the report makes 4
# nulls, and its compute goes from 2.2 to 6.7 minutes (+4.5; 1.2 minutes
# before F1, so +5.5 for the round).
N_SURROGATES = 199
# The copies' spatial correlation (E8, trust review 7 Oct 2026): the farmland's
# measured autocorrelation is kept at full strength out to the largest window
# any search here uses (the quality assessment's 4 km stand-out window) and
# tapered to zero by SURROGATE_MAX_LAG_M. Until 7 Oct a cosine taper over
# 0-3 km cut it at every lag: Site B farmland 0.30 at 1.5 km and 0.23 at
# 2 km, the copies 0.15 and 0.06 (audit_features/covar.py).
SURROGATE_MATCH_LAG_M = 4000.0
SURROGATE_MAX_LAG_M = 6000.0
# measured against generated at these lags; feature_null records both
SURROGATE_CHECK_LAGS_M = (500.0, 1000.0, 2000.0, 4000.0)
SURROGATE_MATCH_TOL = 0.05  # largest |farmland - copies| correlation still called a match
MIN_NULL_KM2 = 50.0         # the farmland the null is measured on: at least this, and 1/4 of the searched area
NULL_EXCLUDE_M = 1000.0     # every candidate is kept this far out of the null (2 x the smoothing)
TAIL_PROB = 0.05            # the null's outer 5% a side is modelled from farmland's bulk, not read off
# Leave-candidate-out (F1, follow-up of 7 Oct 2026; the principle of E28: an
# area must not be compared with itself). On the first pass a large real
# patch inside the farmland widens the band, shifts the class centres and
# feeds the copies, and so tests itself against itself: Site A Feature 1
# (23.7 km², 38.241 N 87.647 W) scored 88.6 with 13 of 59 copies as strong,
# was never listed, so never cut, and the second pass - where it scores
# 115.4 against copies up to 88.8 - never ran. The LEAVE_OUT_LEADING
# strongest candidates (cluster mass, at the map's one search scale) that do
# not already beat chance are each re-tested on farmland with that candidate
# (+ client_report.FARMLAND_SECOND_PASS_M) taken out, the copies built from
# that farmland, under the same selection-corrected rule; one that passes is
# cut and the second pass runs as before.
# G2 (calibration of 7 Oct 2026): those re-tests are L more tries at the
# same area, so each is corrected for their number (Bonferroni): with L
# candidates re-tested, one passes when min(1, L x p_left_out) <= SELECTION_ALPHA
# / 2 (passes_selection(k, n, tests=L)), and the chance printed for a feature
# listed this way is that adjusted p (leave_out_chance). Uncorrected, pure
# noise listed a feature in 5.6% of fields at 540 m correlation (1.9% from
# the first pass alone; work_fu_features_rev/sim_null.py).
LEAVE_OUT_LEADING = 3
# Year-by-year pace (methodology-6 / stats-12). A calendar year is used only
# when its dates span 0.9 yr: Site B's Jan-Jun 2026 (0.45 yr, half a
# seasonal cycle) set the "recent" rate of every feature. A shape word needs
# its contrast of yearly rates to pass 2 sigma of the same contrast on
# farmland patches of the same footprint (client_report passes them in).
MIN_YEAR_SPAN = 0.9
CONTRAST_Z = 2.0
MIN_YEARLY_NULL = 10        # farmland placements needed for the per-year null
# "Fastest in <years>" (E2, trust review 7 Oct 2026): the fast years are
# picked from the data, so the contrast is tested against the same pick made
# on FASTEST_DRAWS simulated steady movements (the record's mean pace plus
# farmland's year-to-year noise, its covariance measured on the placements):
# said only above that null's FASTEST_Q quantile. Tested as if chosen in
# advance it was printed for 43-66% of steady movements (nominal 2.3%;
# audit_features_verify/trend_v.py) - Site A Feature 1's "fastest in
# 2019-2022" among them. A covariance from under MIN_FASTEST_NULL
# placements is too noisy to set the point at all: the word is withheld.
# The covariance is itself estimated from the placements, so the plug-in
# null is a little light: at its 97.5% point steady movements were named
# for 2.8-5.2% with 20-40 placements (400 each, work_ri/fastest_fa.py); at
# its 99% point 1.2-2.2%, under the 2.3% the other shape words allow (the
# old test: 45-75%).
FASTEST_Q = 0.99
FASTEST_DRAWS = 2000
MIN_FASTEST_NULL = 20
FASTEST_SEED = 0
# Without placements a single year's rate is taken as this many times the
# record's farmland spread: per-year sd 5.0-10.9 mm/yr LOS against 4.2 for the
# whole record on 116 farmland patches of 2 km2 (Site B, 5 Oct 2026); the
# upper end, so the fallback can only withhold a word, never add one.
YEARLY_NOISE_FACTOR = 2.6


# A feature at least this share on a surface-mine layer is "change on
# surface-mine ground" (relation_to_mines' own "on" rule).
ON_LAYER_SHARE = domains.DEFAULT.limits.on_layer_share
# Mining-induced horizontal movement at a subsidence trough's flanks, as a
# share of the vertical (the textbook 0.3-0.4 of NCB-style profiles; the
# review's methodology-15 figure). It sets the "+/- X%" of the LOS limitation.
MINING_HORIZONTAL_SHARE = domains.DEFAULT.limits.horizontal_share
# "Ground reshaped since the elevation model" (E1, trust review 7 Oct 2026):
# MintPy's DEM error takes up the part of the movement its quadratic model
# does not fit (bperp runs -242 to +232 m and correlates with time), so over
# farmland the 500 m DEM error follows the 500 m rate - Theil-Sen 0.23 m per
# mm/yr on Site B, r 0.33. The flag tests what is left after that: the
# feature's DEM error minus the farmland fit at its own rate, against
# DEM_RATE_SIGMAS robust sd of the farmland's residuals. Site B Feature 1
# read -14.9 m (3.6 sd raw) but -7.2 m after its -33 mm/yr (1.8 sd): not
# flagged; a refit of its own series with a model that follows the movement
# gave -1.8 to -2.8 m (audit_recompute_verify/f0_tsrefit.py).
DEM_RATE_SIGMAS = 3.0
DEM_FIT_SAMPLE = 3000       # farmland cells in the robust fit (fixed seed)
DEM_FIT_MIN = 1000          # fewer farmland cells: no fit, no flag
# Single cells standing out (E14): the per-cell rule of the S/N raster
# (|class-matched relative rate| / that cell's farmland null >= 1.96).
CELL_Z = 1.96


# ------------------------------------------------------------------ numbers in client text
# "-0.00" is never printed (review outcome check, 6 Oct 2026: Site A's
# whole-area range read "-0.15 to -0.00 in/yr" - a value that rounds to zero
# carries no direction). signed() is the formatter; no_negative_zero() is the
# net applied to every finished client text (report HTML, PNG labels, KMZ,
# results notes), so an f-string written later cannot bring it back.
_SIGNED_ZERO = re.compile(r"(?<![\w.])(?:[-\u2212](?=0)|\+(?=0\.0))(0(?:\.0+)?)(?!\.?\d)")


def signed(x, nd: int = 2) -> str:
    """'+0.38' / '-1.47'; a value that rounds to zero is '0.00' (no sign).
    Formatted first, then tested: 0.5 at nd=0 rounds half to even, to 0, and
    printed '+0' when the test came before the rounding (final check of 7
    Oct 2026: Site A's years-since-mining table read '(+0 mm)' beside
    '(0 mm)')."""
    s = f"{float(x):+.{nd}f}"
    return s[1:] if float(s) == 0 else s


def no_negative_zero(text):
    """'-0.00' / '-0' / '+0.00' -> '0.00' / '0' / '0.00' in finished text.
    Only a zero standing alone is touched: '-0.05', '2016-01-05', '-87.38'
    and base64 ('+0' with no decimal point) are left as they are."""
    if not isinstance(text, str) or "0" not in text:
        return text
    return _SIGNED_ZERO.sub(r"\1", text)


def tidy_text(obj):
    """no_negative_zero over every string in a JSON-like structure (numbers
    are left alone): client_report.json's sentences reach the page, the KMZ
    and the results workbook verbatim."""
    if isinstance(obj, str):
        return no_negative_zero(obj)
    if isinstance(obj, dict):
        return {k: tidy_text(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tidy_text(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(tidy_text(v) for v in obj)
    return obj


def tidy_figure(fig, ground: str | None = None) -> None:
    """no_negative_zero over a matplotlib figure's titles, labels, legends and
    annotations before it is saved (tick labels are the formatter's and never
    show a signed zero); with `ground` (the comparison ground's label), its
    "farmland" said as that ground too (say_ground)."""
    try:
        from matplotlib.text import Text
        for t in fig.findobj(Text):
            s = t.get_text()
            if s:
                u = no_negative_zero(say_ground(s, ground, single_words=True) if ground else s)
                if u != s:
                    t.set_text(u)
    except Exception as exc:                                  # noqa: BLE001 - a label is never worth a crash
        log.debug("figure text not tidied (%s)", exc)


# ------------------------------------------------------------------ wording helpers
def ground_word(d) -> str:
    """What the report calls the comparison ground (cfg or rep:
    control_ground_label, default "farmland" - the Indiana coalfield's)."""
    v = d.get("control_ground_label") if isinstance(d, dict) else getattr(d, "control_ground_label", "")
    return str(v or "").strip() or "farmland"


def ground_noun(label) -> str:
    """The comparison ground's words without their land-cover note, for a
    sentence that already has one in brackets or that names the ground
    again: 'ground (mostly built-up and grassland)' -> 'ground'."""
    return re.sub(r"\s*\(.*\)\s*$", "", str(label or "")).strip() or "farmland"


# names someone gave (areas, mines, the project) are never reworded
_NAMED_KEYS = frozenset({"name", "title", "location", "project", "mine_layers", "label", "path", "file"})


def say_ground(obj, label, single_words: bool = False):
    """The sentences in `obj` (a string, or a JSON-like record) with the
    word "farmland" said as the comparison ground's noun (ground_noun of
    `label`, the report's control_ground_label; 28 Oct 2026). Much of the
    record's prose and every page literal was written for the Indiana
    coalfield and says "farmland" for the comparison ground. Unchanged when
    the label is itself farmland; strings with no space (identifiers, and
    values such as reference_mode 'farmland') and the names under
    _NAMED_KEYS are left alone (quality_assessment.reword_ground's word
    rule: farmland_spread_1sigma_mm_yr is not touched). single_words: a
    one-word string is text too (a figure's legend entry)."""
    noun = ground_noun(label)
    if "farmland" in noun.lower():
        return obj
    from .quality_assessment import reword_ground

    def walk(o):
        if isinstance(o, str):
            return reword_ground(o, noun) if single_words or " " in o else o
        if isinstance(o, dict):
            return {k: (v if k in _NAMED_KEYS else walk(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, tuple):
            return tuple(walk(v) for v in o)
        return o
    return walk(obj)


def lat_txt(lat: float, nd: int = 3) -> str:
    """'38.304 N' / '12.100 S' - the hemisphere from the sign, not assumed."""
    return f"{abs(float(lat)):.{nd}f} {'N' if float(lat) >= 0 else 'S'}"


def lon_txt(lon: float, nd: int = 3) -> str:
    return f"{abs(float(lon)):.{nd}f} {'W' if float(lon) < 0 else 'E'}"


def lalo_txt(lat: float, lon: float, nd: int = 3) -> str:
    return f"{lat_txt(lat, nd)}, {lon_txt(lon, nd)}"


def direction(v, rng, words=("sinking", "rising")) -> str:
    """A direction word only when the 95% range excludes zero (stats-9,
    clientout-10: 'the region is sinking' was printed for -1.9 mm/yr with a
    range of -4.4 to +0.7); '' otherwise, including when there is no range -
    a figure that was never compared with anything is not a direction."""
    if v is None or not rng or rng[0] is None or rng[1] is None:
        return ""
    lo, hi = float(rng[0]), float(rng[1])
    if lo > 0 or hi < 0:
        return words[0] if float(v) < 0 else words[1]
    return ""


# ------------------------------------------------------------------ identity
def _pretty(name: str) -> str:
    """'site_a_pl' -> 'Site A'; a screening project keeps saying so
    ('site_d_disp_screening' -> 'Site D (screening)')."""
    name = re.sub(r"^config_", "", name)
    screening = bool(re.search(r"_disp_screening$", name))
    name = re.sub(r"_disp_screening$", "", name)
    name = re.sub(r"_(pl|p\d|disp|stack|miaplpy|test)$", "", name)
    return name.replace("_", " ").strip().title() + (" (screening)" if screening else "")


def display_name(name: str, spec: str = "") -> str:
    """The name a client reads for a report area. One that looks like a file
    stem ('project_zone', 'area_a1.geojson') is shown tidied ('Project
    Zone') - release.blockers still holds the report until it is renamed in
    the project settings (clientout-13, empirical-13); a real name is kept as
    typed."""
    n = (name or "").strip()
    try:
        from .release import _looks_like_file_stem
        stem = _looks_like_file_stem(n, spec)
    except (ImportError, AttributeError):
        stem = bool(re.fullmatch(r"[a-z0-9]+([_\-][a-z0-9]+)+", n))
    if not stem:
        return n
    n = re.sub(r"\.(geojson|json|shp|kml|kmz|gpkg)$", "", n, flags=re.I)
    return re.sub(r"[_\-]+", " ", n).strip().title() or name


def reverse_geocode(lat: float, lon: float, cache: Path | None = None, timeout: int = 15) -> str:
    """'Gibson County, Indiana' for a point (OpenStreetMap Nominatim, one
    request, cached beside the project). '' when offline."""
    if cache and cache.exists():
        try:
            return json.loads(cache.read_text()).get("place", "")
        except ValueError:
            pass
    # the firm/contact as set NOW (insar-doctor --firm/--contact), not as
    # they were when areas was first imported
    from . import areas
    ua = areas.user_agent() if hasattr(areas, "user_agent") else areas.USER_AGENT
    url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(
        {"lat": f"{lat:.5f}", "lon": f"{lon:.5f}", "format": "jsonv2", "zoom": 8, "addressdetails": 1})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Language": "en"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            a = (json.loads(r.read().decode()) or {}).get("address") or {}
    except Exception as exc:                                  # noqa: BLE001
        log.info("Place lookup skipped (%s)", exc)
        return ""
    parts = [a.get("county") or a.get("state_district") or a.get("city") or a.get("town"),
             a.get("state") or a.get("region"), None if a.get("country_code") == "us" else a.get("country")]
    place = ", ".join(p for p in parts if p)
    if cache and place:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"lat": lat, "lon": lon, "place": place}))
    return place


def identity(cfg) -> dict:
    P = Path(cfg.project_dir)
    title = (getattr(cfg, "project_title", "") or "").strip() or _pretty(P.name)
    place = (getattr(cfg, "project_location", "") or "").strip()
    if not place:
        w, s, e, n = cfg.bbox
        place = reverse_geocode(0.5 * (s + n), 0.5 * (w + e), P / "context" / "location.json")
    return {"title": title, "location": place}


# ------------------------------------------------------------------ mines
# areas.MINE_KIND_LABEL (what the builder writes into mine_layer_sources) -> layer key
_SOURCE_KEY = {"underground mine workings": "underground", "surface mines": "surface",
               "other known ground movement": "other", "other known movement": "other",
               "dated mined areas": "dated"}


# A date that says when a FILE was obtained, not what it maps: 'file dated
# 2023-03-28' (what the builder records when nobody typed a source),
# 'retrieved 21 Sep 2026', 'downloaded 2026-09-21', 'accessed on Sep 21, 2026',
# 'retrieved in 2026'.
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DAYS = r"\d{1,2}(?:\s*[-\u2013]\s*\d{1,2})?"
_DATE = (rf"(?:\d{{4}}-\d{{1,2}}(?:-\d{{1,2}})?|\d{{8}}|\d{{1,2}}[/.]\d{{1,2}}[/.]\d{{2,4}}"
         rf"|{_DAYS}[\s-]+{_MONTH}[\s,-]+\d{{4}}|{_MONTH}\s+{_DAYS},?\s+\d{{4}}|{_MONTH}\s+\d{{4}}|\d{{4}})")
_OBTAINED = re.compile(rf"\b(?:file\s+dated|retrieved|downloaded|accessed|obtained|fetched)\b[\s:]*(?:(?:on|in)\s+)?{_DATE}",
                       re.I)


def layer_vintage(text: str) -> int | None:
    """The year a layer is mapped to, from its "source, vintage" text
    ('Indiana Geological & Water Survey, Coal_Mines_Underground_Dec2016' ->
    2016; '..., annual mined-area polygons 2000-2025, file dated 2026-09-21'
    -> 2025). None when the text gives none - or gives only the date the
    FILE was saved or obtained ('file dated 2023-03-28', 'retrieved 21 Sep
    2026', 'downloaded 2026-09-21'): that says when it was saved or fetched,
    not what it maps, and is never read as the vintage."""
    t = str(text or "")
    if not t or "not recorded" in t.lower():
        return None
    t = _OBTAINED.sub("", t)
    ys = [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", t)]
    return max(ys) if ys else None


def layer_sources(cfg) -> dict:
    """{layer key: "source, vintage"} from cfg.mine_layer_sources (L4b's builder
    writes one "Label: source, vintage" per kept layer); the untyped 'mapped'
    layer carries all of them, each after its plain label: 'dated mined areas
    (Indiana Department of Natural Resources, ...)' (gap-licensing-5). Until
    7 Oct 2026 it joined the internal keys ('dated: Indiana ...'), which the
    report printed as "mine areas - dated: Indiana ..."."""
    out, plain = {}, {}
    for s in getattr(cfg, "mine_layer_sources", ()) or ():
        label, _, rest = str(s).partition(":")
        key = _SOURCE_KEY.get(label.strip().lower())
        if key and rest.strip():
            out[key] = rest.strip()
            plain[key] = label.strip().lower()
    if out:
        out["mapped"] = "; ".join(f"{plain[k]} ({v})" for k, v in out.items() if k != "mapped")
    return out


def mine_layers(cfg) -> list[dict]:
    """[{key, label, path, source, vintage}] - typed layers when the project has
    them, else the one untyped deramp-exclusion layer (which is what older
    projects carry). 'other' (ground already known to move: a haul road, a
    slip, an active cut) is its own layer when the project names one
    (other_movement_geojson); until 5 Oct 2026 it was only merged into the
    deramp exclusion, so relation_to_mines could never say "on ground already
    known to move" once a typed layer existed (gap-surface_mine_site-7).
    source/vintage: what the builder recorded for the layer (layer_sources),
    None when nothing was."""
    src = layer_sources(cfg)
    out = []
    for key, attr in (("underground", "underground_mines_geojson"), ("surface", "surface_mines_geojson"),
                      ("other", "other_movement_geojson")):
        p = (getattr(cfg, attr, "") or "").strip()
        if p and Path(p).exists():
            out.append({"key": key, "label": KIND_LABEL[key], "path": Path(p)})
    d = (getattr(cfg, "deramp_exclude_geojson", "") or "").strip()
    if not out and d and Path(d).exists():
        out.append({"key": "mapped", "label": KIND_LABEL["mapped"], "path": Path(d)})
    for L in out:
        L["source"] = src.get(L["key"])
        L["vintage"] = layer_vintage(L["source"]) if L["source"] else None
    return out


# the study-neutral name (S0.4); callers move to it in Phase 1a
study_layers = mine_layers


def polygon_masks(ctx, layers: list[dict]) -> dict:
    """key -> boolean grid of the layer's polygons (not limited to coherent cells)."""
    from .stack_backend import _polygon_mask
    return {L["key"]: _polygon_mask(L["path"], ctx.good.shape, ctx.attrs).astype(bool) for L in layers}


# ------------------------------------------------------------------ field and features
def _smooth(v: np.ndarray, ok: np.ndarray, k: int) -> np.ndarray:
    """Mean of v over the ok cells of a k x k window; NaN where under 30% of it
    is ok. float32 (mm/yr to 1e-6): two of these on a 6 M-cell grid were
    a third of the report's memory in float64."""
    from scipy.ndimage import uniform_filter
    num = uniform_filter(np.where(ok, v, 0.0).astype(np.float32), k)
    den = uniform_filter(ok.astype(np.float32), k)
    num /= np.maximum(den, np.float32(1e-9))
    num[~(ok & (den > 0.3))] = np.nan
    return num


def farmland_band(farv: np.ndarray, method: str = "quantile") -> tuple[float, float, float]:
    """
    (centre, low, high) of ordinary farmland at the map's scale: its median
    and its 2.5% and 97.5% quantiles (method "quantile"), or 1.96 x each
    side's robust spread (method "robust": 1.4826 x the median one-sided
    deviation - a half-normal's median is 0.6745 sigma).

    Why not zero +/- 2 x MAD (until 5 Oct 2026, stats-7): farmland's median
    is not zero (+1.17 mm/yr on Site B with mean class centres, +1.05 on
    the cleaned farmland with median ones) and its sinking side is wider, so
    ordinary farmland showed 7.0% red against 0.6% blue under a caption
    promising "one in twenty". The quantiles must be taken on farmland
    CLEANED of real movement (smoothed_field does): the in-box farmland
    holds unmapped subsidence (Site B Features 2 and 3, 2.6% of it) that
    IS the 2.5% tail - uncleaned, the sinking edge sat at -17.2 mm/yr and hid
    Feature 3. The robust form is the fallback when too little clean
    farmland is left for quantiles to mean anything.
    """
    farv = np.asarray(farv, dtype=float)
    farv = farv[np.isfinite(farv)]
    if not farv.size:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(farv))
    if method == "quantile":
        lo, hi = np.quantile(farv, [0.025, 0.975])
        return med, float(lo), float(hi)
    below, above = med - farv[farv < med], farv[farv > med] - med
    s_lo = 1.4826 * float(np.median(below)) if below.size else 0.0
    s_hi = 1.4826 * float(np.median(above)) if above.size else 0.0
    return med, med - Z95 * s_lo, med + Z95 * s_hi


def smoothed_field(ctx, los_up, smooth_m: float = 500.0) -> dict:
    """The report map's field (vertical mm/yr relative to farmland of the same
    coherence class, averaged over ~smooth_m), the band ordinary farmland stays
    inside at that scale (farmland_band: the median and the 2.5/97.5%
    quantiles of the farmland cleaned of strong patches - two passes, below),
    and `null_field`: the same smoothing of the
    placement field (significance.placement_field - farmland outside the box
    re-centred on its own class means), which the feature null is measured on.

    los_up is the LOS -> vertical factor (a map or a number); there is no
    default - a vertical figure converted with another site's incidence is
    5-15% off (deliverables.los_up_or_raise).

    `band_ok` is False (and the band NaN) when the in-box farmland covers
    less than MIN_BAND_KM2: then nothing can be called "within normal
    variation" and no feature can be assessed."""
    if los_up is None:
        raise ValueError("smoothed_field needs the LOS -> vertical factor (deliverables.los_up_or_raise)")
    up = los_up
    py, px = ctx.sg.pixel_metres(ctx.attrs)
    k = max(3, int(round(smooth_m / np.sqrt(py * px))))
    rel_v = ctx.r_cm / up
    sm = _smooth(rel_v, np.isfinite(rel_v) & ctx.good, k)
    r_null = getattr(ctx, "r_null", None)
    if r_null is not None and r_null is not ctx.r_cm:
        nv = r_null / up
        null_sm = _smooth(nv, np.isfinite(nv) & ctx.good, k)
    else:
        null_sm = sm
    farm = ctx.far & np.isfinite(sm)
    farv = sm[farm]
    km2 = float(farm.sum()) * py * px / 1e6
    out = {"field": sm, "null_field": null_sm, "smoothing_m": int(round(k * np.sqrt(py * px))),
           "pixel_m": (float(py), float(px)), "farmland_km2": round(km2, 1)}
    if km2 < MIN_BAND_KM2:
        out.update({"band_ok": False, "centre_mm_yr": float("nan"), "band_lo_mm_yr": float("nan"),
                    "band_hi_mm_yr": float("nan"), "band_mm_yr": float("nan"),
                    "reason": (f"only {km2:.1f} km² of ordinary {ground_word(getattr(ctx, 'cfg', None) or {})} "
                               "in the study area at the map's scale "
                               f"({MIN_BAND_KM2:g} km² needed to measure its normal range)")})
        return out
    # Two passes. Real movement inside the farmland (Site B Features 2 and
    # 3, unmapped) is exactly the tail a band must not be set by, so: a
    # first band (median +/- 1.96 robust sd), the patches the search would
    # report on the farmland itself, those patches + NULL_EXCLUDE_M out,
    # and the final band on what is left (farmland_band).
    med = float(np.median(farv))
    s0 = 1.4826 * float(np.median(np.abs(farv - med)))
    first = dict(out, band_ok=True, centre_mm_yr=med, band_lo_mm_yr=med - Z95 * s0, band_hi_mm_yr=med + Z95 * s0)
    strong = np.zeros(sm.shape, bool)
    for sign in (-1, 1):
        _, width = _side(first, sign)
        lab, idx, size, _, peak = _components(sm, farm, first, sign, py * px / 1e6)
        keep = idx[(size >= MIN_FEATURE_KM2) & (sign * (peak - med) >= STRONG * width)]
        if keep.size:
            strong |= np.isin(lab, keep)
    n_strong = _count_patches(strong)
    clean = farm & ~ctx.sg.euclidean_buffer(strong, ctx.attrs, NULL_EXCLUDE_M) if n_strong else farm
    cv = sm[clean]
    method = "quantile"
    if float(clean.sum()) * py * px / 1e6 < MIN_BAND_KM2:
        cv, method = farv, "robust"
    c, lo, hi = farmland_band(cv, method)
    out.update({"band_ok": True, "centre_mm_yr": c, "band_lo_mm_yr": lo, "band_hi_mm_yr": hi,
                # the old single half-width, kept for readers of client_report.json
                "band_mm_yr": 0.5 * (hi - lo),
                "band_basis": (("farmland median and 2.5/97.5% quantiles" if method == "quantile" else
                                "farmland median and 1.96 x each side's robust spread")
                               + f", measured on the in-box farmland less {n_strong} strong patch(es) + {NULL_EXCLUDE_M:g} m"),
                # the share of ordinary farmland the map colours by chance, each side
                "farmland_beyond_low_percent": round(100 * float(np.mean(cv <= lo)), 1),
                "farmland_beyond_high_percent": round(100 * float(np.mean(cv >= hi)), 1),
                "farmland_km2_for_band": round(float(cv.size) * py * px / 1e6, 1)})
    return out


def _count_patches(mask: np.ndarray) -> int:
    """Number of connected patches in a boolean grid."""
    from scipy import ndimage
    return int(ndimage.label(mask)[1]) if mask.any() else 0


def _side(sm: dict, sign: int) -> tuple[float, float]:
    """(edge of the band on this side, its distance from the centre)."""
    c = sm["centre_mm_yr"]
    edge = sm["band_lo_mm_yr"] if sign < 0 else sm["band_hi_mm_yr"]
    return edge, abs(edge - c)


def relation_to_mines(comp: np.ndarray, poly: dict, pixel_m: tuple, near_km: float | None = None) -> dict:
    """Where a feature sits: share inside each mapped layer, distance to the
    nearest one, and a phrase for the report.

    "Next to" (key near_<layer>, which brings the mining causes and the
    sideways-movement caveat) only within NEAR_MINE_KM of a mapped outline
    (M2, final check of 7 Oct 2026); farther out the distance is given
    plainly with no mining attribution (key "none"). Until then it reached
    2 km: Site A Feature 1, 1.84 km from mapped workings, read "next to
    mapped mine areas ... It is beside mapped mining", beyond typical
    angle-of-draw reach and outside the report's own 2 km mine buffer."""
    near_km = NEAR_MINE_KM if near_km is None else float(near_km)
    from scipy.ndimage import distance_transform_edt
    if not poly:
        return {"phrase": domains.DEFAULT.wording.no_layer_phrase, "inside": {}, "nearest_km": None, "key": None}
    n = comp.sum()
    inside = {k: round(float((comp & m).sum()) / max(1, n), 2) for k, m in poly.items()}
    best = max(inside, key=inside.get)
    ys, xs = np.nonzero(comp)
    sl = (slice(max(0, ys.min() - 400), ys.max() + 400), slice(max(0, xs.min() - 400), xs.max() + 400))
    dist = {}
    for k, m in poly.items():
        sub = m[sl]
        if sub.any():
            d = distance_transform_edt(~sub, sampling=pixel_m)
            dist[k] = float(d[comp[sl]].min()) / 1000.0
    nearest = min(dist, key=dist.get) if dist else None
    if inside[best] >= 0.3:
        phrase = {"surface": "on mapped surface-mine ground", "underground": "over mapped underground workings",
                  "other": "on ground already known to move", "mapped": "on mapped mine ground"}[best]
        key = best
    elif nearest is not None and dist[nearest] <= near_km:
        phrase = f"next to mapped {KIND_LABEL[nearest]}, {dist[nearest]:.1f} km at the closest" \
            if dist[nearest] > 0.05 else f"at the edge of mapped {KIND_LABEL[nearest]}"
        key = "near_" + nearest
    elif nearest is not None:
        phrase = f"{dist[nearest]:.1f} km from the nearest mapped {NEAREST_LABEL[nearest]}"
        key = "none"
    else:
        phrase = "away from every mapped mine"
        key = "none"
    return {"phrase": phrase, "inside": inside, "nearest_km": round(dist[nearest], 2) if nearest else None,
            "nearest": nearest, "key": key}


# the study-neutral name (S0.4); callers move to it in Phase 1a
relation_to_layers = relation_to_mines


def _components(field: np.ndarray, where: np.ndarray, sm: dict, sign: int, px_km2: float):
    """(labels, ids, size km2, mean, peak) of the connected patches of `field`
    within `where` that lie beyond the band on side `sign`."""
    from scipy import ndimage
    edge, _ = _side(sm, sign)
    hot = where & (sign * (field - edge) >= 0)
    lab, n = ndimage.label(hot)
    if not n:
        return lab, np.array([], int), np.array([]), np.array([]), np.array([])
    idx = np.arange(1, n + 1)
    size = ndimage.sum(hot, lab, idx) * px_km2
    mean = ndimage.mean(np.where(hot, field, 0.0), lab, idx)
    peak = (ndimage.minimum if sign < 0 else ndimage.maximum)(np.where(hot, field, 0.0), lab, idx)
    return lab, idx, np.asarray(size), np.asarray(mean), np.asarray(peak)


def qa_standout_doubt(cal: dict | None) -> str | None:
    """
    Why the quality assessment's own stand-out calibration cannot confirm a
    feature (H1, final check of 7 Oct 2026), or None when it can. The QA
    flags it itself:
      copies_match_within_band False  its simulated farmland does not vary
                                      like the real farmland at every scale
                                      it reads (window-spread ratio outside
                                      its band, 0.8-1.25)
      candidate_in_farmland True      its copies were built from farmland
                                      that still holds the candidate
    Either way its threshold is set on the wrong yardstick, so a stand-out
    against it supports no "stands out clearly" (selection_p): Site A
    Feature 1's 4 km gap 2.20 cleared 2.02 on copies whose 4 km spread
    was 2.60 against the farmland's 1.88 (ratio 0.72); against 200 rolled
    real fields 15% reach 2.20. A calibration that did not run is no doubt
    here: calibrated_standouts then gives no stand-out at all.
    """
    cal = cal or {}
    if not cal.get("tested"):
        return None
    if cal.get("candidate_in_farmland") is True:
        return ("the quality assessment's simulated farmland was built from ground that still holds this patch, "
                "so its check is not independent of it")
    if cal.get("copies_match_within_band") is False:
        try:
            from .quality_assessment import STANDOUT_MATCH as band
        except Exception:                                     # noqa: BLE001 - the words only
            band = (0.8, 1.25)
        off = []
        for k, v in (cal.get("copies_match") or {}).items():
            r = v.get("ratio") if isinstance(v, dict) else None
            if r is None or not band[0] <= float(r) <= band[1]:
                off.append(f"{k} km: {float(r):.2f}" if r is not None else f"{k} km: not measured")
        return ("the quality assessment's simulated farmland does not vary like the real farmland at the scale of "
                "its check" + (f" (spread ratio {'; '.join(off)}; {band[0]:g}-{band[1]:g} expected)" if off else ""))
    return None


def _standout_list(standout) -> list[dict]:
    """The quality assessment's stand-outs as a list: one dict (its
    `standout`, the largest gap) or a list of them (every scale that passed,
    quality_assessment.calibrated_standouts), each with lat/lon. An entry
    that says it does NOT stand out (`stands_out` False or None - the
    calibration could not run) is dropped."""
    if not standout:
        return []
    items = [standout] if isinstance(standout, dict) else list(standout)
    return [s for s in items if isinstance(s, dict) and s.get("lat") is not None and s.get("lon") is not None
            and s.get("stands_out", True) is True]


def find_features(ctx, sm: dict, poly: dict, aoi: np.ndarray, standout: dict | list | None = None,
                  limit: int | None = MAX_FEATURES) -> list[dict] | None:
    """Connected patches of the smoothed field beyond the farmland band, at
    least MIN_FEATURE_KM2, whose strongest value lies STRONG times the band's
    width (that side's) beyond farmland's centre; `quality_standout` marks
    a patch holding one of the quality assessment's tested stand-outs
    (`standout`: one dict or the list of every scale that passed its
    calibrated rule - quality_assessment.calibrated_standouts; Site B's
    2 km stand-out lies in Feature 2, its 1 km one in Feature 1). Largest
    effect first; `limit` None keeps every candidate (the calibrated search
    tests them all).

    The stand-out no longer excuses a patch from STRONG (E7, trust review 7
    Oct 2026): the simulated copies count only patches that pass it, so a
    patch kept without it cannot be compared with them (Gaussian null field
    1001 on Site B: a 1.6 km² QA patch below STRONG with p = 1.0 was
    listed "stands out clearly").

    None - not [] - when the band could not be measured: "nothing found" and
    "could not look" must never read the same (stats-2)."""
    if not sm.get("band_ok", True) or not np.isfinite(sm.get("band_mm_yr", np.nan)):
        return None
    field = sm["field"]
    c = sm["centre_mm_yr"]
    py, px = sm["pixel_m"]
    px_km2 = py * px / 1e6
    lat_of = lambda r: float(ctx.attrs["Y_FIRST"]) + (r + 0.5) * float(ctx.attrs["Y_STEP"])   # noqa: E731
    lon_of = lambda c_: float(ctx.attrs["X_FIRST"]) + (c_ + 0.5) * float(ctx.attrs["X_STEP"])  # noqa: E731
    # MintPy's own lat/lon -> row/col (floor, +0.01 cell): int() truncated
    # towards zero and could put a centre coordinate on the neighbouring
    # cell (gap-geolocation-9)
    so_items = _standout_list(standout)
    so_rcs = [ctx.sg.lalo_to_rc(ctx.attrs, float(s["lat"]), float(s["lon"])) for s in so_items]
    found = []
    fin = np.isfinite(field) & aoi
    for sign in (-1, 1):
        _, width = _side(sm, sign)
        lab, idx, size, mean_, peak = _components(field, fin, sm, sign, px_km2)
        for i, a_km2, mn, pk in zip(idx, size, mean_, peak):
            held = [s for s, rc in zip(so_items, so_rcs)
                    if 0 <= rc[0] < lab.shape[0] and 0 <= rc[1] < lab.shape[1] and lab[rc] == i]
            has_so = bool(held)
            if a_km2 < MIN_FEATURE_KM2 or sign * (pk - c) < STRONG * width:
                continue
            # H1: the QA's own calibration in doubt for every stand-out it
            # holds (client_report._calibrated_standouts sets it from
            # qa_standout_doubt) - one without a doubt confirms it
            doubts = [s.get("calibration_doubt") for s in held]
            doubt = doubts[0] if held and all(doubts) else None
            comp = lab == i
            ys, xs = np.nonzero(comp)
            found.append({"sign": sign, "area_km2": round(float(a_km2), 2), "mean_rel_vertical_mm_yr": round(float(mn), 1),
                          "peak_rel_vertical_mm_yr": round(float(pk), 1),
                          # the search statistic: how far the patch's mean lies beyond farmland's centre
                          "beyond_farmland_mm_yr": round(float(sign * (mn - c)), 2),
                          "lat": round(lat_of(ys.mean()), 4), "lon": round(lon_of(xs.mean()), 4),
                          "bbox": [round(lon_of(xs.min()), 4), round(lat_of(ys.max()), 4),
                                   round(lon_of(xs.max()), 4), round(lat_of(ys.min()), 4)],
                          "quality_standout": bool(has_so), "quality_standout_doubt": doubt,
                          "band_mm_yr": round(float(width), 1),
                          "mines": relation_to_mines(comp, poly, sm["pixel_m"]),
                          # its land cover: trees are canopy to C-band (landcover, 15 Oct 2026)
                          "land_cover": _land_cover(ctx, comp, field), "_mask": comp})
    found.sort(key=lambda f: -(abs(f["mean_rel_vertical_mm_yr"]) * f["area_km2"] + 1e6 * f["quality_standout"]))
    return found if limit is None else found[:limit]


def _land_cover(ctx, mask: np.ndarray, field: np.ndarray | None = None) -> dict | None:
    """landcover.share_of for a feature or area; None without a map."""
    cfg = getattr(ctx, "cfg", None)
    if cfg is None:
        return None
    from . import landcover
    try:
        return landcover.share_of(cfg, ctx.attrs, mask, None if field is None else np.isfinite(field))
    except Exception as exc:                                     # noqa: BLE001 - a label, never a failure
        log.warning("Land cover of an outline: %s", exc)
        return None


def _lag_taper(shape, max_lag: int, flat_lag: int = 0) -> np.ndarray:
    """Taper over circular lags of an FFT grid: 1 out to flat_lag cells, a
    cosine from there to max_lag, zero beyond (flat_lag 0: a cosine over
    the whole range, which until 7 Oct 2026 halved the copies' correlation
    at half the range)."""
    ny, nx = shape
    dy = np.minimum(np.arange(ny), ny - np.arange(ny))
    dx = np.minimum(np.arange(nx), nx - np.arange(nx))
    r = np.hypot(dy[:, None], dx[None, :])
    flat = min(max(int(flat_lag), 0), max_lag - 1)
    x = np.clip((r - flat) / max(max_lag - flat, 1), 0.0, 1.0)
    return np.where(r <= max_lag, 0.5 * (1 + np.cos(np.pi * x)), 0.0)


def _axis_correlation(C: np.ndarray, lags_m, py: float, px: float) -> list:
    """C (circular autocovariance, lag 0 at [0, 0]) as a correlation at each
    lag in metres, the mean of the row and column directions; None beyond
    the grid."""
    out = []
    c0 = float(C[0, 0])
    for lag in lags_m:
        ly, lx = int(round(lag / py)), int(round(lag / px))
        if c0 <= 0 or ly >= C.shape[0] // 2 or lx >= C.shape[1] // 2:
            out.append(None)
            continue
        out.append(round(0.5 * (float(C[ly, 0]) + float(C[0, lx])) / c0, 3))
    return out


def _search_stats(fld: np.ndarray, where: np.ndarray, sm: dict, px_km2: float) -> dict:
    """{sign: (largest cluster mass, largest mean beyond the centre)} of the
    patches the feature search would report in `fld` (size >= MIN_FEATURE_KM2,
    peak STRONG band widths beyond the centre); 0 when there are none."""
    c = sm["centre_mm_yr"]
    out = {}
    for sign in (-1, 1):
        _, width = _side(sm, sign)
        _, _, size, mean_, peak = _components(fld, where, sm, sign, px_km2)
        ok = (size >= MIN_FEATURE_KM2) & (sign * (peak - c) >= STRONG * width)
        beyond = sign * (mean_[ok] - c)
        mass = (beyond - width) * size[ok]
        out[sign] = (float(mass.max()) if mass.size else 0.0, float(beyond.max()) if beyond.size else 0.0)
    return out


def _gpd_tail(x_u: float, x_10: float, x_05: float) -> tuple[float, float]:
    """(shape xi >= 0, scale) of a generalised Pareto tail through the 75%,
    90% and 95% points of one side (quantile spacings, no fit to the
    extremes): r = (x05 - xu) / (x10 - xu) = (5^xi - 1) / (2.5^xi - 1). A
    Gaussian gives r = 1.60 < ln 5 / ln 2.5 = 1.76, so xi = 0 - an
    exponential tail of 0.60 sd against the Gaussian's true 0.42: heavier,
    the safe error. Never lighter than exponential."""
    d1, d2 = max(x_10 - x_u, 1e-9), max(x_05 - x_u, 1e-9)
    r = d2 / d1
    if r <= np.log(5) / np.log(2.5):
        return 0.0, d2 / np.log(5)
    lo, hi = 1e-6, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if (5 ** mid - 1) / (2.5 ** mid - 1) < r else (lo, mid)
    xi = min(0.5 * (lo + hi), 1.0)
    return xi, xi * d2 / (5 ** xi - 1)


def _tailed_quantiles(vals: np.ndarray, u: np.ndarray, q: np.ndarray, tail: float = TAIL_PROB) -> np.ndarray:
    """q (the quantiles of the sorted farmland values `vals` at probabilities
    u) with the outer `tail` on each side MODELLED (_gpd_tail through that
    side's 75/90/95% points) instead of read off. Read off, the tails are
    wrong either way: with the candidates left in, a real feature's own
    values give the simulated copies features as strong as it (no power);
    with them cut out, pure noise loses exactly the extremes it is tested
    on (the candidates ARE its extremes) and the null is too weak. The
    75/90/95% points barely move when a feature of a few per cent of the
    farmland is in or out."""
    q05, q10, q25, q75, q90, q95 = np.quantile(vals, [tail, 2 * tail, 0.25, 0.75, 1 - 2 * tail, 1 - tail])
    out = q.copy()
    for side, (xu, x10, x05) in (("hi", (q75, q90, q95)), ("lo", (-q25, -q10, -q05))):
        xi, beta = _gpd_tail(xu, x10, x05)
        p = (1 - u) if side == "hi" else u
        sel = p < tail
        pp = p[sel] / 0.25
        x = xu + (beta / xi * (pp ** -xi - 1) if xi > 0 else beta * np.log(1 / pp))
        out[sel] = x if side == "hi" else -x
    return out


def cluster_mass(f: dict, sm: dict) -> float:
    """A feature's cluster mass, mm/yr x km2: how far its mean lies beyond the
    band edge times its area. The search keeps patches of any size from 0.25
    km2, and a small patch reaches a given strength by chance far more often
    than a large one (its mean averages fewer independent fields), so the
    test statistic must carry size as well as strength (cluster-mass
    inference: Bullmore et al. 1999; Hayasaka & Nichols 2004)."""
    _, width = _side(sm, f["sign"])
    return float((f["beyond_farmland_mm_yr"] - width) * f["area_km2"])


def feature_null(ctx, sm: dict, candidates: list[dict] | None, aoi: np.ndarray,
                 n_surrogates: int | None = None, seed: int = 0, exclude_m: float = NULL_EXCLUDE_M,
                 min_null_km2: float = MIN_NULL_KM2) -> dict:
    """
    What ordinary farmland alone produces under the feature search's own
    rules, in an area the size of the one searched: the calibrated chance
    level (stats-6, review of 5 Oct 2026).

    One realisation of farmland is not enough: the searched area and the
    farmland around it are about the same size, so beating the strongest
    farmland patch means p ~ 0.5, and a fitted tail on the 9-23 patches it
    holds swung from bounded to heavy between the two projects (GPD shape
    -1.5 to +1.2). So the null is SIMULATED from the farmland itself
    (amplitude-adjusted Fourier surrogates, Theiler et al. 1992):

      1. farmland: ctx.far_wide (coherent cells outside every mapped mine +
         buffer, inside and outside the box, on the placement field so the
         outside ring sits on its own class centres) minus every candidate
         the real search found, grown by exclude_m (2 x the smoothing);
      2. its spatial autocorrelation, measured on the gappy field (masked
         FFT, on normal scores), kept as measured out to
         SURROGATE_MATCH_LAG_M and tapered to zero by SURROGATE_MAX_LAG_M
         (E8); `covariance_check` records the farmland's correlation and
         the copies' at SURROGATE_CHECK_LAGS_M;
      3. n_surrogates Gaussian fields with that autocorrelation over the
         searched area, each rank-mapped onto the values of ALL the
         farmland (candidates included) with the outer TAIL_PROB a side
         modelled from its bulk (_tailed_quantiles), so the copies carry
         farmland's skew but neither lose pure noise's own extremes nor
         borrow a real feature's;
      4. the search's rules on each copy (the band, MIN_FEATURE_KM2, STRONG)
         and the largest cluster mass per side (cluster_mass).

    A feature's selection-corrected p = (1 + copies whose largest patch is at
    least as strong) / (1 + n_surrogates), per side; it stands out at or
    below SELECTION_ALPHA / 2 (both signs are searched; passes_selection,
    G1). Deterministic (fixed seed), so a rebuild gives the same numbers and
    keeps a sign-off valid.
    Real movement left in the farmland (an unmapped mine outside the box)
    widens the null: the safe error.

    Calibration (5 Oct 2026, 92 km² synthetic farmland, 60 m cells, 20
    fields each): ANY patch standing out in 0/20 Gaussian fields (300 m
    correlation), 1/20 with 540 m correlation, 1/20 skewed (sinking side
    1.6 x wider), 1/20 t5-tailed, 1/20 two-scale - 4/100 against the 3.3%
    the level allows; a 1 km bowl four band widths deep stood out in 94/100.
    The candidates' values read off the cleaned farmland gave 9/24
    "features" on pure noise; read off all of it, 1/20 bowls at 540 m.

    Fails closed (`tested` False with a reason) when the band could not be
    measured or the farmland is under max(min_null_km2, a quarter of the
    searched area): a feature then is never called "beyond chance".

    `feature_null_check` (feature_null_check(); contract C4): whether the
    real farmland's own strongest patch beats every copy - then the copies
    are lighter than this ground and nothing is "stands out clearly".

    n_surrogates None: N_SURROGATES, read when called (F3).
    """
    from scipy import fft as sfft
    n_surrogates = N_SURROGATES if n_surrogates is None else int(n_surrogates)
    py, px = sm["pixel_m"]
    px_km2 = py * px / 1e6
    field = sm.get("null_field")
    field = sm["field"] if field is None else field
    searched = np.isfinite(sm["field"]) & aoi
    A = float(searched.sum()) * px_km2
    out = {"tested": False, "method": "amplitude-adjusted Fourier surrogates of the farmland field",
           "statistic": "cluster mass (mean beyond the band edge x area), mm/yr x km²",
           "searched_area_km2": round(A, 1), "null_area_km2": 0.0, "exclude_m": exclude_m,
           "n_surrogates": int(n_surrogates), "seed": int(seed), "sides": {}}
    if not sm.get("band_ok", True) or not np.isfinite(sm.get("band_mm_yr", np.nan)):
        out["reason"] = sm.get("reason") or "the farmland band could not be measured"
        return out
    domain = getattr(ctx, "far_wide", None)
    domain = (ctx.far if domain is None else domain) & np.isfinite(field)
    domain_all = domain
    if candidates:
        cand = np.zeros(domain.shape, bool)
        for f in candidates:
            cand |= f["_mask"]
        domain = domain & ~ctx.sg.euclidean_buffer(cand, ctx.attrs, exclude_m)
    D = float(domain.sum()) * px_km2
    out["null_area_km2"] = round(D, 1)
    need = max(min_null_km2, 0.25 * A)
    if D < need or not searched.any():
        out["reason"] = (f"only {D:.0f} km² of ordinary {ground_word(getattr(ctx, 'cfg', None) or {})} is left "
                         "to measure chance on "
                         f"({need:.0f} km² needed)")
        return out
    c = sm["centre_mm_yr"]
    L = max(4, int(round(SURROGATE_MAX_LAG_M / np.sqrt(py * px))))
    # 2. masked autocorrelation: correlate the zero-filled field and the mask,
    # divide lag by lag by the number of cell pairs behind it
    sh = (sfft.next_fast_len(domain.shape[0] + L), sfft.next_fast_len(domain.shape[1] + L))
    # on NORMAL SCORES: a skewed or heavy-tailed field correlates less than the
    # Gaussian field beneath it, and surrogates built on the raw correlation
    # cluster their extremes too little (t3 farmland: 97% false alarms)
    from scipy.stats import norm
    zs = np.zeros(domain.shape)
    zv = field[domain]
    order = np.argsort(zv, kind="stable")
    rank = np.empty(zv.size)
    rank[order] = np.arange(zv.size)
    zs[domain] = norm.ppf((rank + 0.5) / zv.size)
    del zv, order, rank
    Fx = sfft.rfft2(zs, sh)
    del zs
    Cx = sfft.irfft2(Fx * np.conj(Fx), sh)
    del Fx
    Fm = sfft.rfft2(domain.astype(np.float64), sh)
    Cm = sfft.irfft2(Fm * np.conj(Fm), sh)
    del Fm
    C = np.where(Cm > 1000.0, Cx / np.maximum(Cm, 1.0), 0.0)
    del Cx, Cm
    # E8: the farmland's own correlation, as measured, at the check lags
    measured = _axis_correlation(C, SURROGATE_CHECK_LAGS_M, py, px)
    Lf = int(round(SURROGATE_MATCH_LAG_M / np.sqrt(py * px)))
    C *= _lag_taper(sh, L, Lf)
    out["correlation_at_smoothing"] = round(float(C[0, min(sm["smoothing_m"] // max(int(px), 1), C.shape[1] - 1)]
                                                  / C[0, 0]), 3) if C[0, 0] > 0 else None
    # 3. the generation grid: the searched area's bounding box, padded by the lag
    rows, cols = np.flatnonzero(searched.any(1)), np.flatnonzero(searched.any(0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    gs = (sfft.next_fast_len(r1 - r0 + L), sfft.next_fast_len(c1 - c0 + L))
    hy, hx = min(L, gs[0] // 2 - 1), min(L, gs[1] // 2 - 1)
    Cg = np.zeros(gs)
    for ys in (slice(0, hy + 1), slice(-hy, None)):
        for xs in (slice(0, hx + 1), slice(-hx, None)):
            Cg[ys, xs] = C[ys, xs]
    del C
    amp = np.sqrt(np.maximum(sfft.rfft2(Cg).real, 0.0))
    del Cg
    # E8 "measure it; match it": the copies' own correlation (exactly what
    # the generator gives: the inverse transform of its power) against the
    # farmland's, at the check lags
    copies = _axis_correlation(sfft.irfft2(amp * amp, gs), SURROGATE_CHECK_LAGS_M, py, px)
    diffs = [abs(a - b) for a, b in zip(measured, copies) if a is not None and b is not None]
    out["covariance_check"] = {
        "lags_m": [int(x) for x in SURROGATE_CHECK_LAGS_M], "farmland": measured, "copies": copies,
        "max_difference": round(max(diffs), 3) if diffs else None,
        "matched": bool(diffs) and max(diffs) <= SURROGATE_MATCH_TOL,
        "matched_to_m": SURROGATE_MATCH_LAG_M, "tapered_to_zero_by_m": SURROGATE_MAX_LAG_M,
        "basis": ("correlation of normal scores, mean of the row and column directions; farmland as measured on "
                  "the comparison ground, copies as generated")}
    # the values the copies take: ALL the farmland's (candidates included,
    # so pure noise keeps its own extremes), the outer TAIL_PROB a side
    # modelled from the bulk (so a real feature's own values do not end up
    # in the copies) - _tailed_quantiles
    vals = np.sort((field - c)[domain_all])
    sub = searched[r0:r1, c0:c1]
    nsub = int(sub.sum())
    u = (np.arange(nsub) + 0.5) / nsub
    q = _tailed_quantiles(vals, u, vals[np.minimum((u * vals.size).astype(int), vals.size - 1)])
    del vals
    smz = dict(sm, centre_mm_yr=0.0, band_lo_mm_yr=sm["band_lo_mm_yr"] - c, band_hi_mm_yr=sm["band_hi_mm_yr"] - c)
    rng = np.random.default_rng(seed)
    maxima = {-1: [], 1: []}
    fld = np.full(sub.shape, np.nan)
    for _ in range(int(n_surrogates)):
        g = sfft.irfft2(amp * sfft.rfft2(rng.standard_normal(gs)), gs)[:r1 - r0, :c1 - c0]
        o = np.argsort(g[sub], kind="stable")
        v = np.empty(nsub)
        v[o] = q
        fld[sub] = v
        for sign, mm in _search_stats(fld, sub, smz, px_km2).items():
            maxima[sign].append(mm)
    # the real farmland's own strongest patches under the same rules (one
    # realisation; E8: stronger than every copy means the copies are lighter
    # than this ground - feature_null_check)
    real = _search_stats(field, domain, sm, px_km2)
    for sign in (-1, 1):
        mass = np.array([m for m, _ in maxima[sign]])
        beyond = np.array([b for _, b in maxima[sign]])
        s_ = {"tested": True, "edge_mm_yr": round(_side(sm, sign)[1], 2),
              "max_mass": [round(float(x), 3) for x in np.sort(mass)],
              "median_max_mass": round(float(np.median(mass)), 2),
              "median_max_beyond_mm_yr": round(float(np.median(beyond)), 1),
              "farmland_strongest_mass": round(real[sign][0], 2),
              "farmland_strongest_beyond_mm_yr": round(real[sign][1], 1)}
        if real[sign][0] > 0 and mass.size and real[sign][0] > float(mass.max()):
            s_["farmland_strongest_patch"] = _strongest_patch(field, domain, sm, sign, px_km2, ctx.attrs)
        out["sides"]["sinking" if sign < 0 else "rising"] = s_
    out["tested"] = True
    out["feature_null_check"] = feature_null_check(out)
    return out


def _strongest_patch(fld: np.ndarray, where: np.ndarray, sm: dict, sign: int, px_km2: float, attrs) -> dict | None:
    """Where the strongest patch of `fld` on side `sign` lies (the search's
    rules, as _search_stats): {mass, area_km2, beyond_mm_yr, lat, lon}."""
    c = sm["centre_mm_yr"]
    _, width = _side(sm, sign)
    lab, idx, size, mean_, peak = _components(fld, where, sm, sign, px_km2)
    ok = (size >= MIN_FEATURE_KM2) & (sign * (peak - c) >= STRONG * width)
    if not ok.any():
        return None
    beyond = sign * (mean_ - c)
    mass = np.where(ok, (beyond - width) * size, -np.inf)
    j = int(np.argmax(mass))
    ys, xs = np.nonzero(lab == idx[j])
    try:
        lat = float(attrs["Y_FIRST"]) + (ys.mean() + 0.5) * float(attrs["Y_STEP"])
        lon = float(attrs["X_FIRST"]) + (xs.mean() + 0.5) * float(attrs["X_STEP"])
    except (KeyError, TypeError, ValueError):
        lat = lon = None
    return {"mass": round(float(mass[j]), 2), "area_km2": round(float(size[j]), 2),
            "beyond_mm_yr": round(float(beyond[j]), 1),
            "lat": round(lat, 4) if lat is not None else None, "lon": round(lon, 4) if lon is not None else None}


def feature_null_check(null: dict | None) -> dict:
    """
    Whether the simulated copies are as strong as the real comparison
    ground (E8, trust review 7 Oct 2026; contract C4): {real_beats_all_copies,
    why, sides, covariance_matched}. When the real farmland's own strongest
    patch (feature_null's `farmland_strongest_mass`, the search's rules on
    the comparison ground less every candidate + exclude_m) is stronger than
    the strongest patch in every copy, ordinary ground of this area makes
    patches the copies never do: the null is mis-specified for it - unmapped
    movement in the comparison ground, or copies too light. Then no feature
    may be called "stands out clearly" (selection_p) and release lists it
    for the reviewer (soft). Site A as delivered (copies cut at 3 km):
    87.1 against a copy maximum of 61.2 (a 5.7 km² patch at 38.442 N,
    87.843 W, in Illinois outside the box, near a surface mine no layer
    maps); with the copies matched to 4 km (7 Oct 2026) they reach 88.8 and
    122 (second and first pass), and it no longer fires.

    `covariance_matched` (feature_null's covariance_check) is recorded beside
    it; it does not gate anything by itself.
    """
    null = null or {}
    if not null.get("tested"):
        return {"real_beats_all_copies": False, "tested": False, "sides": [], "covariance_matched": None,
                "why": "the chance test could not be made" + (f" ({null['reason']})" if null.get("reason") else "")}
    beats, bits = [], []
    for name, s in (null.get("sides") or {}).items():
        if not s.get("tested") or not s.get("max_mass"):
            continue
        real, top = float(s.get("farmland_strongest_mass") or 0.0), float(max(s["max_mass"]))
        if real > 0 and real > top:
            beats.append(name)
            p = s.get("farmland_strongest_patch") or {}
            where = (f", {p['area_km2']:.1f} km² near {lalo_txt(p['lat'], p['lon'])}"
                     if p.get("lat") is not None and p.get("area_km2") is not None else "")
            bits.append(f"{name}: the real comparison ground's strongest patch (strength {real:.1f}{where}) is "
                        f"stronger than the strongest of all {len(s['max_mass'])} simulated copies ({top:.1f})")
    cc = null.get("covariance_check") or {}
    out = {"real_beats_all_copies": bool(beats), "tested": True, "sides": beats,
           "covariance_matched": cc.get("matched")}
    out["why"] = ("; ".join(bits) + ". Ordinary ground here makes patches the copies never do - movement no layer "
                  "maps, or copies too light - so the chance figures may be too low and no feature is called "
                  "'stands out clearly'." if beats else
                  "the real comparison ground's strongest patch is within what the simulated copies produce")
    return out


def passes_selection(k: int, n: int, alpha_side: float | None = None, tests: int = 1) -> bool:
    """The exact Monte Carlo rule (G1, calibration of 7 Oct 2026): a patch
    with k of n copies as strong beats chance when min(1, tests x (1 + k)
    / (1 + n)) <= alpha_side (SELECTION_ALPHA / 2 by default), compared in
    whole numbers - tests x (1 + k) <= floor(alpha_side x (1 + n)) - so no
    rounding of p or of the level can move a patch across the line.
    Its size is exactly alpha_side when (1 + n) x alpha_side is a whole
    number (199 copies: k <= 4 passes). tests: the number of tries the
    p is corrected for (G2: the leave-out re-tests made, Bonferroni)."""
    a = SELECTION_ALPHA / 2 if alpha_side is None else float(alpha_side)
    return max(int(tests), 1) * (1 + int(k)) <= int(np.floor(a * (1 + int(n)) + 1e-9))


def adjusted_p(k: int, n: int, tests: int = 1) -> float:
    """min(1, tests x (1 + k) / (1 + n)): the Bonferroni-adjusted Monte
    Carlo p (G2)."""
    return min(1.0, max(int(tests), 1) * (1 + int(k)) / (1 + int(n)))


def adjusted_p_text(k: int | None, n: int | None, tests: int = 1, p: float | None = None) -> str:
    """The adjusted chance in words (G2): p_text with the adjusted count,
    "about tests x (1 + k) in n + 1" ("about 3 in 200" for a patch that
    beat all 199 copies when 3 patches were re-tested); with one test it
    is p_text itself ("less than 1 in 200" when it beat every copy)."""
    if k is None or not n:
        return p_text(k, n, p)
    t = max(int(tests), 1)
    if t == 1:
        return p_text(k, n, p)
    return p_text(min(int(n), t * (1 + int(k)) - 1), n)


def p_text(k: int | None, n: int | None, p: float | None = None) -> str:
    """The chance in words (E8, trust review 7 Oct 2026). A feature that
    beats every copy has a chance below 1 in n + 1 - a bound, not an
    estimate, so it is never printed as "about 0.02" (Site A Feature 1
    beat 59 copies by 1.9 times; 0 of 413 seed-pooled copies came near it).
    Otherwise "about k + 1 in n + 1"."""
    if k is None or not n:
        return f"about {p:.2f}" if p is not None else "not known"
    k, n = int(k), int(n)
    return f"less than 1 in {n + 1}" if k == 0 else f"about {k + 1} in {n + 1}"


def selection_p(f: dict, sm: dict, null: dict) -> dict:
    """
    The selection-corrected chance for feature f (find_features): the
    share of simulated farmland copies of the searched area whose strongest
    patch on f's side has at least f's cluster mass, (1 + k) / (1 + n). It
    stands out when p <= SELECTION_ALPHA / 2, compared in whole numbers
    (passes_selection, G1: k <= 4 of 199; until 7 Oct p < 0.025, k <= 3) -
    the only test that decides whether a feature is listed (E7). Untested
    (fail closed) when the null is. `chance_level_mm_yr`: how far beyond
    farmland's centre the strongest chance patch reaches in half the copies
    (for the wording); `p_text` the chance in words (p_text()).

    Sets f["stands_out_clearly"] (E7, trust review 7 Oct 2026; contract C4):
    True only when f beats chance here AND holds the quality assessment's
    stand-out (`quality_standout`, the QA's multi-scale rule calibrated for
    trying 3 scales) AND the copies are not lighter than the real farmland
    (feature_null_check). Until 7 Oct the QA stand-out alone made a feature
    "stands out clearly", overriding this test: on Gaussian farmland copies
    that label appeared in 4.5% of reports, 10-15% on real-like noise, on
    patches this test scored p = 1.0.

    H1 (final check of 7 Oct 2026): the QA stand-out counts only when its
    own calibration is sound - f["quality_standout_doubt"] (find_features,
    from qa_standout_doubt: the QA's copies do not vary like the real
    farmland, or were built from farmland still holding the candidate)
    withholds "clearly", and `clearly_withheld` / `clearly_withheld_by`
    ("qa_calibration" or "null_check") say why. Site A Feature 1 then
    reads "worth a look".
    """
    side = (null.get("sides") or {}).get("sinking" if f["sign"] < 0 else "rising") or {}
    if not null.get("tested") or not side.get("tested"):
        f["stands_out_clearly"] = False
        return {"tested": False, "p": None, "stands_out": None, "p_text": None, "stands_out_clearly": False,
                "reason": side.get("reason") or null.get("reason") or "no farmland null"}
    m = cluster_mass(f, sm)
    M = np.asarray(side["max_mass"], dtype=float)
    k = int(np.sum(M >= m))
    p = (1 + k) / (1 + M.size)
    stands = passes_selection(k, int(M.size))
    check = null.get("feature_null_check") or feature_null_check(null)
    doubt = f.get("quality_standout_doubt")
    clearly = bool(stands and f.get("quality_standout") and not doubt and not check.get("real_beats_all_copies"))
    f["stands_out_clearly"] = clearly
    out = {"tested": True, "p": round(float(p), 4), "stands_out": stands,
           "p_text": p_text(k, int(M.size), p), "stands_out_clearly": clearly,
           "cluster_mass": round(m, 2), "copies_as_strong": k, "copies": int(M.size),
           "chance_level_mm_yr": side.get("median_max_beyond_mm_yr"),
           "null_area_km2": null.get("null_area_km2"), "searched_area_km2": null.get("searched_area_km2")}
    if stands and f.get("quality_standout") and not clearly:
        out["clearly_withheld"] = doubt or check.get("why")
        out["clearly_withheld_by"] = "qa_calibration" if doubt else "null_check"
    return out


def copies_percentile(m: float | None, max_mass) -> float | None:
    """The share of the copies' strongest patches (one side's max_mass) that
    are weaker than strength m, in per cent: 100 when m beats every copy."""
    M = np.asarray(max_mass if max_mass is not None else [], dtype=float)
    if m is None or not M.size:
        return None
    return round(100.0 * float(np.mean(M < float(m))), 1)


def leading_candidates(cands: list[dict] | None, sm: dict, n: int = LEAVE_OUT_LEADING) -> list[dict]:
    """The first pass's n strongest candidates by the chance test's own
    statistic (cluster_mass), strongest first: the ones re-tested with
    themselves left out of the farmland (F1). The report searches at one
    scale (the map's smoothing), so these are its leading candidates at
    every scale it uses."""
    return sorted(cands or [], key=lambda f: -cluster_mass(f, sm))[:max(int(n), 0)]


def match_candidate(f: dict, cands: list[dict] | None) -> dict | None:
    """The candidate of a re-run search that is f's patch: the one on f's
    side sharing the most cells with f's `_mask`; None when no candidate
    overlaps it (the patch no longer passes the search's rules)."""
    best, n_best = None, 0
    for g in cands or []:
        if g.get("sign") != f.get("sign") or g.get("_mask") is None or f.get("_mask") is None:
            continue
        n = int(np.count_nonzero(g["_mask"] & f["_mask"]))
        if n > n_best:
            best, n_best = g, n
    return best


def leave_out_result(g: dict | None, sm: dict | None, null: dict | None, retested: int = 1) -> dict:
    """One leave-candidate-out test's record (F1): the candidate as the
    search re-run on farmland without it found it (g, its `chance` from
    selection_p against that farmland's copies): {strength (cluster mass),
    copies, copies_as_strong, copies_percentile, p, retested, p_adjusted,
    p_text, p_adjusted_text, passed}.

    retested (G2, calibration of 7 Oct 2026): L, the number of leading
    candidates re-tested this way (client_report._leave_candidate_out);
    passed when min(1, L x p) <= SELECTION_ALPHA / 2 (passes_selection
    with tests=L): with 199 copies and L = 3 a patch must beat every copy,
    L = 2 allows 1 copy as strong, L = 1 the first pass's 4. p is the
    left-out test's own; p_adjusted (and p_adjusted_text, "about 3 in 200")
    the corrected one, which is printed for a feature listed this way
    (leave_out_chance). False, with a reason, when the patch is no longer a
    candidate or the test could not be made."""
    null = null or {}
    L = max(int(retested or 1), 1)
    if not null.get("tested"):
        return {"tested": False, "passed": False, "retested": L,
                "reason": null.get("reason") or "the chance test could not be made on that farmland"}
    if g is None or sm is None:
        return {"tested": True, "passed": False, "strength": None, "retested": L,
                "reason": "with itself left out the patch no longer passes the search's rules"}
    ch = g.get("chance") or selection_p(g, sm, null)
    side = (null.get("sides") or {}).get("sinking" if g["sign"] < 0 else "rising") or {}
    m = ch.get("cluster_mass")
    k, n = ch.get("copies_as_strong"), ch.get("copies")
    tested = bool(ch.get("tested")) and k is not None and bool(n)
    return {"tested": bool(ch.get("tested")),
            "passed": bool(tested and passes_selection(k, n, tests=L)),
            "strength": m, "copies": n, "copies_as_strong": k,
            "copies_percentile": copies_percentile(m, side.get("max_mass")),
            "copies_max": round(float(max(side["max_mass"])), 2) if side.get("max_mass") else None,
            "p": ch.get("p"), "retested": L,
            "p_adjusted": round(adjusted_p(k, n, L), 4) if tested else None,
            "p_text": ch.get("p_text"),
            "p_adjusted_text": adjusted_p_text(k, n, L, ch.get("p")) if tested else None,
            "area_km2": g.get("area_km2"),
            "lat": g.get("lat"), "lon": g.get("lon"), "null_area_km2": null.get("null_area_km2")}


def leave_out_chance(ch: dict | None, left_out: dict | None) -> dict:
    """The chance a feature listed BECAUSE of its leave-out re-test carries
    (G2, calibration of 7 Oct 2026): the second pass lists it by its own
    test, but that test, like the re-test, measures it against farmland
    without it; so its printed chance is the re-test's corrected for the
    number re-tested - p = p_adjusted, copies_as_strong and copies the
    re-test's, p_text its p_adjusted_text ("about 3 in 200") - with the
    second pass's own figures kept as `second_pass` and the re-test's as
    `leave_out` {p, copies_as_strong, copies, retested, p_adjusted}.
    stands_out and stands_out_clearly stay the second pass's. ch unchanged
    when the re-test carries no adjusted p."""
    ch = dict(ch or {})
    lo = left_out or {}
    if lo.get("p_adjusted") is None or not ch.get("tested"):
        return ch
    second = {k: ch.get(k) for k in ("p", "copies_as_strong", "copies", "p_text", "cluster_mass")}
    ch.update({"p": lo["p_adjusted"], "copies_as_strong": lo.get("copies_as_strong"), "copies": lo.get("copies"),
               "p_text": lo.get("p_adjusted_text") or ch.get("p_text"),
               "listed_on": "leave-out re-test", "retested": lo.get("retested"),
               "leave_out": {k: lo.get(k) for k in ("p", "copies_as_strong", "copies", "retested", "p_adjusted",
                                                     "strength")},
               "second_pass": second})
    return ch


def first_pass_chance(ch: dict | None, first: dict | None) -> dict:
    """The chance a feature listed on the FIRST pass carries (final check of
    7 Oct 2026: the chance printed is the test that listed it). The first
    pass's test lists and cuts it; the second pass, made on farmland
    without it, keeps it listed but scores it against copies that never
    held it - Site B Feature 1 read "less than 1 in 200" there against
    "about 2 in 200" (1 of 199 copies) on the test that listed it, while
    Site A printed its listing test. p, copies_as_strong, copies,
    p_text and cluster_mass become the first pass's (`listed_on` "first
    pass"), the second pass's are kept as `second_pass`; stands_out and
    stands_out_clearly stay the second pass's. ch unchanged when the first
    pass carries no tested chance."""
    ch = dict(ch or {})
    fp = first or {}
    if not fp.get("tested") or fp.get("p") is None or not ch.get("tested"):
        return ch
    second = {k: ch.get(k) for k in ("p", "copies_as_strong", "copies", "p_text", "cluster_mass")}
    ch.update({k: fp.get(k) for k in ("p", "copies_as_strong", "copies", "p_text", "cluster_mass")})
    ch.update({"listed_on": "first pass", "second_pass": second})
    return ch


# ------------------------------------------------------------------ trend
def yearly_rates(t: np.ndarray, series_mm: np.ndarray, years, min_span: float = MIN_YEAR_SPAN) -> dict:
    """{year: rate} for every calendar year whose dates span at least min_span
    years (6 dates at least). Accepts one series or a (n_series, n_dates)
    stack (then each value is an array over the series)."""
    y = np.atleast_2d(np.asarray(series_mm, dtype=float))
    yrs = np.asarray(years)
    out = {}
    for yr in sorted(set(int(v) for v in yrs)):
        m = yrs == yr
        tt = t[m]
        if m.sum() < 6 or np.ptp(tt) < min_span:
            continue
        sub = y[:, m]
        tc = tt - tt.mean()
        ok = np.isfinite(sub)
        n = ok.sum(1)
        w = np.where(ok, tc, 0.0)
        ybar = np.where(n > 0, np.nansum(sub, 1) / np.maximum(n, 1), np.nan)
        num = np.nansum(w * (sub - ybar[:, None]), 1)
        den = (w * w).sum(1) - (w.sum(1) ** 2) / np.maximum(n, 1)
        r = np.where((n >= 6) & (den > 0), num / np.where(den > 0, den, 1.0), np.nan)
        out[yr] = r if np.ndim(series_mm) > 1 else float(r[0])
    return out


def _fastest_null_q(v: np.ndarray, cov: np.ndarray | None, sigmas,
                    n_rows: int | None = None) -> tuple[float | None, str | None]:
    """(FASTEST_Q quantile of the 'fastest' z under steady movement, why it
    could not be had). v: the yearly rates in the direction of the movement;
    cov: farmland's year-to-year covariance (None: too few placements);
    sigmas(W): the contrast sigma for each row of weights W - the same
    yardstick the observed contrast is divided by. Each draw is the record's
    mean pace plus a draw of farmland's noise, its fast years picked as the
    data's are (>= 0.7 x its fastest year); a draw that names no years (all
    of them fast) counts as no claim. n_rows: the sigma was MEASURED on that
    many placements, so each draw measures its own on as many simulated
    placements (with the sigma taken as known, 35-40 placements' t-like
    tails named "fastest" for 6-9% of steady movements). Fixed seed: a
    rebuild gives the same words."""
    if cov is None:
        return None, (f"fewer than {MIN_FASTEST_NULL} farmland patches of the same footprint: too few to test "
                      "years picked from the data")
    n = v.size
    lam, U = np.linalg.eigh(np.asarray(cov, dtype=float))
    Lf = U * np.sqrt(np.clip(lam, 0.0, None))
    rng = np.random.default_rng(FASTEST_SEED)
    V0 = float(np.mean(v)) + rng.standard_normal((FASTEST_DRAWS, n)) @ Lf.T
    fast = V0 >= 0.7 * V0.max(axis=1, keepdims=True)
    nf = fast.sum(axis=1)
    no = n - nf
    W = np.where(fast, 1.0 / np.maximum(nf, 1)[:, None], -1.0 / np.maximum(no, 1)[:, None])
    c = (W * V0).sum(axis=1)
    if n_rows:
        R0 = rng.standard_normal((FASTEST_DRAWS, int(n_rows), n)) @ Lf.T
        s = np.std(np.einsum("kij,kj->ki", R0, W), axis=1, ddof=1)
        del R0
    else:
        s = np.asarray(sigmas(W), dtype=float)
    z = np.full(FASTEST_DRAWS, -1e9)
    ok = (no > 0) & (s > 0)
    z[ok] = c[ok] / s[ok]
    return float(np.quantile(z, FASTEST_Q)), None


def trend_shape(t: np.ndarray, series_mm: np.ndarray, years: list[int], noise_mm_yr: float = 0.0,
                null_yearly: dict | None = None) -> dict:
    """Describe how a movement unfolded from its per-year rates: when it began,
    when it was fastest, and whether it is slowing or speeding up now - each
    only when it beats the same contrast on ordinary farmland.

    Years: calendar years whose dates span MIN_YEAR_SPAN (a half year carries
    half a seasonal cycle). The whole-record gate stays: no words unless the
    record's rate is 3 x `noise_mm_yr` (its farmland spread).

    Contrasts of yearly rates, in the direction of the movement:
      began   mean from the onset year on - mean before it
      fastest mean of the fastest years - mean of the others
      pace    mean of the last two full years - mean of the earlier ones
    Each is stated only when it exceeds CONTRAST_Z sigma of the same contrast
    over `null_yearly` = {"years": [...], "rates": (n_placements, n_years)},
    the yearly rates of farmland patches of the same footprint and dates (a
    constant offset between patches cancels in every contrast). Without at
    least MIN_YEARLY_NULL placements a year's rate is taken as
    YEARLY_NOISE_FACTOR x noise_mm_yr, independent between years; with
    neither, no shape word is given (`shape` "untested").

    "Began around <year>" also needs the years before it to be ordinary (M1,
    final check of 7 Oct 2026): their mean against the same placements'
    (median and sd of the placements' mean over those years; the
    approximate yardstick: 0 and its sd) within CONTRAST_Z sigma. Beyond it
    on the movement's side, the feature "was already sinking when the record
    starts (<first year>)", and the tested onset contrast becomes "sped up
    around <year>" - "between <last year used> and <year>" when years
    between them were not used. Site B Feature 2 read "began around
    2019" while 2016-2017 averaged -20 mm/yr, z -3.75 against 40 farmland
    placements (2018, 0.85 yr of dates, not used). Beyond it on the other
    side (moving the other way before the onset) no onset words at all:
    neither "began" nor "runs through the whole record" is true of it.
    contrasts["before_onset"] records the test. The pace part names no years ("its recent pace"):
    the page gives the last two years' rate with its own window beside it
    (feature_text), and two windows, 2024-2025 and 2024-2026, read as one.

    "fastest" also has to beat its selection (E2, trust review 7 Oct 2026):
    the fast years are those at 0.7 x the record's fastest year or more,
    picked from the data, so its z (contrast / sigma) must exceed the
    FASTEST_Q quantile of the same pick and z on FASTEST_DRAWS steady
    movements at the record's mean pace plus farmland's year-to-year noise
    (covariance of the placements' yearly rates). Under MIN_FASTEST_NULL
    placements no years are named - and never from the approximate
    yardstick used under MIN_YEARLY_NULL placements (F7c, follow-up of 7
    Oct 2026: its independent years of YEARLY_NOISE_FACTOR x noise_mm_yr
    named "fastest in" on no farmland evidence at all). contrasts["fastest"]
    records z, the null's quantile, and why the words were withheld."""
    ok = np.isfinite(series_mm)
    if ok.sum() < 12:
        return {"text": "", "yearly_mm_yr": {}}
    rate = float(np.polyfit(t[ok], series_mm[ok], 1)[0])
    per = yearly_rates(t, np.where(ok, series_mm, np.nan), years)
    per = {y: v for y, v in per.items() if np.isfinite(v)}
    dropped = sorted(set(int(y) for y in years) - set(per))
    out = {"rate_mm_yr": round(rate, 2), "yearly_mm_yr": {int(k): round(v, 1) for k, v in per.items()},
           "years_not_used": dropped}
    if len(per) < 3 or abs(rate) < max(3 * noise_mm_yr, 1e-9):
        out["text"] = ""
        return out
    ys_ = sorted(per)
    sgn = np.sign(rate)
    v = np.array([per[y] * sgn for y in ys_])                 # positive = in the direction of the movement
    # the null: each contrast's sigma over farmland patches of the same footprint
    R = None
    if null_yearly and null_yearly.get("rates") is not None:
        cols = [list(null_yearly["years"]).index(y) if y in list(null_yearly["years"]) else None for y in ys_]
        if None not in cols:
            Rm = np.asarray(null_yearly["rates"], dtype=float)[:, cols]
            Rm = Rm[np.all(np.isfinite(Rm), axis=1)]
            if Rm.shape[0] >= MIN_YEARLY_NULL:
                R = Rm
    if R is not None:
        sigmas = lambda W: np.std(R @ W.T, axis=0, ddof=1)    # noqa: E731 - one sigma per row of W
        cov = np.atleast_2d(np.cov(R, rowvar=False)) if R.shape[0] >= MIN_FASTEST_NULL else None
        out["yearly_null"] = f"{R.shape[0]} farmland patches of the same footprint"
    elif noise_mm_yr > 0:
        s_y = YEARLY_NOISE_FACTOR * noise_mm_yr
        sigmas = lambda W: s_y * np.linalg.norm(W, axis=1)     # noqa: E731
        # F7(c): the approximate yardstick (fewer than MIN_YEARLY_NULL
        # placements, or none) never names fast years - its plug-in
        # independent-year covariance is not farmland's
        cov = None
        out["yearly_null"] = f"approximate: a year's rate taken as {YEARLY_NOISE_FACTOR:g} x the record's farmland spread"
    else:
        out.update({"text": "", "shape": "untested", "yearly_null": None})
        return out
    sigma = lambda w: float(sigmas(np.atleast_2d(w))[0])        # noqa: E731

    def contrast(a_idx, b_idx):
        """(value, sigma) of mean(v[a]) - mean(v[b])."""
        w = np.zeros(len(ys_))
        w[a_idx] = 1.0 / len(a_idx)
        w[b_idx] = -1.0 / len(b_idx)
        return float(w @ v), sigma(w)

    tested = {}
    vmax = v.max()
    if vmax <= 0:
        # No year moved the record's way: its whole-record rate comes from
        # steps between the years, not a movement within them (the first
        # city site, 30 Oct 2026: "fastest" had no years and divided by zero).
        out.update({"text": "", "shape": "untested",
                    "untested_because": "no single year moved in the direction of the whole-record rate"})
        return out
    fast = [i for i, x in enumerate(v) if x >= 0.7 * vmax]
    start = next((i for i, x in enumerate(v) if x >= 0.3 * vmax), 0)
    began = sped_up = None
    already = opposite = False
    if start > 0:
        pre = list(range(start))
        c_, s_ = contrast(list(range(start, len(ys_))), pre)
        tested["began"] = {"contrast_mm_yr": round(c_, 1), "sigma_mm_yr": round(s_, 1)}
        # M1: the years before the onset against farmland of the same footprint
        w = np.zeros(len(ys_))
        w[pre] = 1.0 / len(pre)
        level = float(w @ v)
        centre = float(np.median(R @ w)) * sgn if R is not None else 0.0
        s_pre = sigma(w)
        z_pre = (level - centre) / s_pre if s_pre > 0 else 0.0
        tested["before_onset"] = {"years": [ys_[i] for i in pre], "mean_mm_yr": round(level * sgn, 1),
                                  "farmland_mm_yr": round(centre * sgn, 1), "sigma_mm_yr": round(s_pre, 1),
                                  "z": round(z_pre, 2)}
        when = (f"around {ys_[start]}" if ys_[start] - ys_[start - 1] == 1 else
                f"between {ys_[start - 1]} and {ys_[start]}")
        if z_pre > CONTRAST_Z:
            already = True
            if c_ > CONTRAST_Z * s_:
                sped_up = when
        elif abs(z_pre) <= CONTRAST_Z and c_ > CONTRAST_Z * s_:
            began = ys_[start]
            began_when = when
        elif z_pre < -CONTRAST_Z:
            # the years before moved the OTHER way beyond farmland: neither
            # "began" (M1: not ordinary years) nor "runs through the whole
            # record" (review of the final check, 7 Oct 2026) - no onset words
            opposite = True
    fastest = None
    others = [i for i in range(len(ys_)) if i not in fast]
    if others:
        c_, s_ = contrast(fast, others)
        z = c_ / s_ if s_ > 0 else 0.0
        q, why = _fastest_null_q(v, cov, sigmas, n_rows=R.shape[0] if R is not None else None)
        if R is None:
            why = (f"fewer than {MIN_YEARLY_NULL} farmland patches of the same footprint back the test (an "
                   "approximate yardstick): years picked from the data are not named")
        tested["fastest"] = {"contrast_mm_yr": round(c_, 1), "sigma_mm_yr": round(s_, 1), "z": round(z, 2),
                             "null_z_q": round(q, 2) if q is not None else None, "null_quantile": FASTEST_Q}
        if why:
            tested["fastest"]["withheld"] = why
        elif c_ > CONTRAST_Z * s_ and z > q:
            fastest = [ys_[fast[0]], ys_[fast[-1]]]
            out["fastest_years"] = [ys_[i] for i in fast]
    recent_idx = list(range(len(ys_) - 2, len(ys_)))
    recent = float(np.mean(v[recent_idx]))
    c_, s_ = contrast(recent_idx, list(range(len(ys_) - 2)))
    tested["pace"] = {"contrast_mm_yr": round(c_, 1), "sigma_mm_yr": round(s_, 1)}
    whole = abs(rate)
    if c_ < -CONTRAST_Z * s_ and recent < 0.6 * whole:
        shape = "slowing"
    elif c_ > CONTRAST_Z * s_ and recent > 1.4 * whole:
        shape = "accelerating"
    else:
        shape = "steady"
    if began:
        parts = [f"it began {began_when}"]
    elif already:
        # M1: moving beyond farmland from the record's first year on
        parts = [f"it was already {'sinking' if sgn < 0 else 'rising'} when the record starts ({int(min(years))})"]
        if sped_up:
            parts.append(f"it sped up {sped_up}")
    elif opposite:
        parts = []
    else:
        parts = ["it runs through the whole record"]
    if fastest:
        fy = out["fastest_years"]
        if len(fy) == 1:
            when = str(fy[0])
        elif fast == list(range(fast[0], fast[-1] + 1)):          # consecutive years used
            when = f"{fy[0]}–{fy[-1]}"
        else:                                                     # not a run: name them (2021, 2022 and 2025)
            when = ", ".join(str(y) for y in fy[:-1]) + f" and {fy[-1]}"
        parts.append(f"it was fastest in {when}")
    y1, y2 = ys_[recent_idx[0]], ys_[recent_idx[-1]]
    # no years named here (one recent window on the page: feature_text
    # gives the last two years' rate with its own dates); recent_years
    # keeps the calendar years this contrast used
    if shape == "slowing":
        parts.append("has slowed since" if fastest else "it has slowed in recent years")
    elif shape == "accelerating":
        parts.append("it has sped up in recent years")
    else:
        parts.append("its recent pace is not measurably different from the earlier years")
    out.update({"shape": shape, "recent_mm_yr": round(recent * sgn, 1), "recent_years": [y1, y2],
                "contrasts": tested})
    if began:
        out["began"] = began
    if already:
        out["already_moving_at_start"] = True
        if sped_up:
            out["sped_up"] = sped_up
    if opposite:
        out["moved_the_other_way_before"] = ys_[start]
    if fastest:
        out["fastest"] = fastest
    out["text"] = (", ".join(parts[:-1]) + " and " + parts[-1]) if len(parts) > 1 else parts[0]
    return out


def cause_hint(relation_key: str, sign: int, shape: str | None, reshaped_since_dem: bool = False) -> str:
    """A cautious suggestion of what usually looks like this, from the mapped
    context. On surface-mine ground radar cannot follow ground dug out or
    built up by metres (a quarter wavelength, 14 mm, between neighbouring
    looks per revisit is its limit), so there the figure is a sign of
    change, not a settlement rate - said, not left to the reader
    (gap-surface_mine_site-7). Ground reshaped since the elevation model
    (gap-surface_mine_site-6) adds that earthmoving is the likely setting;
    feature_text keeps that clause only when dem_change's test with the
    movement taken out flagged the feature (E1, reshaped_since_dem)."""
    base = _cause_base(relation_key, sign, shape)
    if reshaped_since_dem:
        base += RESHAPED_HINT
    return base


# cause_hint's earthmoving clause; feature_text drops it unless dem_change's
# rate-removed test flagged the feature (E1)
RESHAPED_HINT = (" Ground reshaped since the elevation model was made usually means earthmoving: spoil or fill "
                 "settling as it compacts, or excavation.")


def dem_change(dem_sm: np.ndarray | None, rate_sm: np.ndarray | None, m: np.ndarray, far: np.ndarray,
               smoothing_m: float = 500.0, seed: int = 0) -> dict | None:
    """
    Was a feature's ground reshaped since the elevation model was made (E1,
    trust review 7 Oct 2026)? dem_sm: MintPy's DEM height error averaged over
    smoothing_m (client_report._dem_smoothed); rate_sm: the report's
    vertical rate relative to farmland at the same scale (smoothed_field's
    `field`, mm/yr); m: the feature's measured cells; far: the comparison
    ground.

    MintPy's DEM error takes up the movement its deformation model does not
    fit, so it follows the rate. The farmland's DEM error is fitted against
    its rate (Theil-Sen on DEM_FIT_SAMPLE cells, fixed seed; intercept the
    median), and the feature's residual - the median over its cells of DEM
    error minus the fit at the cell's rate, less the farmland residuals'
    median - is flagged beyond DEM_RATE_SIGMAS robust sd of the farmland
    residuals. `flag` is that test (what client_report reads); the raw gate
    (the feature's median against DEM_RATE_SIGMAS robust sd of farmland's,
    the rule until 7 Oct) is kept as `raw_flag` with its numbers. With no
    rate field or too little farmland: `rate_removed` False and `flag`
    False - no claim. None without the layer or with under 20 cells.
    """
    if dem_sm is None or m.sum() < 20:
        return None
    okd = np.isfinite(dem_sm)
    ref = dem_sm[far & okd]
    v = dem_sm[m & okd]
    if ref.size < DEM_FIT_MIN or v.size < 20:
        return None
    c = float(np.median(ref))
    rsd = 1.4826 * float(np.median(np.abs(ref - c)))
    med = float(np.median(v))
    thr = DEM_RATE_SIGMAS * rsd
    out = {"flag": False, "rate_removed": False, "raw_flag": bool(rsd > 0 and abs(med - c) > thr),
           "feature_median_m": round(med, 1), "farmland_median_m": round(c, 1),
           "farmland_robust_sd_m": round(rsd, 2), "threshold_m": round(thr, 1), "smoothing_m": float(smoothing_m)}
    del ref, v
    if rate_sm is None:
        out["reason"] = "no rate field to take the movement out of the DEM error"
        return out
    ok = okd & np.isfinite(rate_sm)
    fx, fy = rate_sm[far & ok].astype(np.float64), dem_sm[far & ok].astype(np.float64)
    mx, my = rate_sm[m & ok].astype(np.float64), dem_sm[m & ok].astype(np.float64)
    if fx.size < DEM_FIT_MIN or mx.size < 20:
        out["reason"] = "too few cells with both a rate and a DEM error"
        return out
    from scipy.stats import theilslopes
    rng = np.random.default_rng(seed)
    pick = rng.choice(fx.size, min(DEM_FIT_SAMPLE, fx.size), replace=False)
    b = float(theilslopes(fy[pick], fx[pick])[0])
    a = float(np.median(fy - b * fx))
    res = fy - (a + b * fx)
    rc = float(np.median(res))
    rr = 1.4826 * float(np.median(np.abs(res - rc)))
    resid = float(np.median(my - (a + b * mx))) - rc
    rthr = DEM_RATE_SIGMAS * rr
    rate_f = float(np.median(mx))
    out.update({"flag": bool(rr > 0 and abs(resid) > rthr), "rate_removed": True,
                "slope_m_per_mm_yr": round(b, 3), "intercept_m": round(a, 2),
                "feature_rate_mm_yr": round(rate_f, 1), "expected_from_rate_m": round(a + b * rate_f, 1),
                "residual_m": round(resid, 1), "residual_robust_sd_m": round(rr, 2),
                "residual_threshold_m": round(rthr, 1), "residual_sigmas": round(resid / rr, 2) if rr > 0 else None,
                "basis": (f"MintPy DEM height error averaged over {smoothing_m:.0f} m, less what the comparison "
                          f"ground's fit against its rate ({b:.2f} m per mm/yr, Theil-Sen) gives at the feature's "
                          f"rate; flagged when that residual is beyond {DEM_RATE_SIGMAS:g} robust sd of the "
                          "comparison ground's residuals")})
    return out


def cells_standing_out(r_rel: np.ndarray, cell_null: np.ndarray, reported: np.ndarray, far: np.ndarray,
                       z: float = CELL_Z) -> dict:
    """
    The share of single cells that stand out, on the report's own frame
    (E14, trust review 7 Oct 2026): r_rel the class-matched relative LOS rate
    after the second farmland pass (ctx.r_cm once features that beat chance
    are cut from the farmland), cell_null the per-cell farmland null in the
    same units (client_report.cell_sigma_map of rep['cell_null_by_class']),
    reported the study area's measured cells, far the comparison ground of
    that pass (the chance rate). A cell stands out when |r_rel| / cell_null
    >= z - rate_signal_to_noise.tif's rule. Until 7 Oct the sentence quoted
    summary.json's FIRST-pass shares (Site B 6.8% against 7.1% by chance;
    the delivered second-pass raster gives 6.7% against 5.6%).
    """
    ok = np.isfinite(r_rel) & np.isfinite(cell_null) & (cell_null > 0)
    flag = np.zeros(ok.shape, bool)
    flag[ok] = np.abs(r_rel[ok]) >= z * cell_null[ok]
    a, b = reported & ok, far & ok
    pct = lambda w: round(100 * float(flag[w].mean()), 1) if w.any() else None   # noqa: E731
    return {"cells_standing_out_percent": pct(a), "cells_standing_out_by_chance_percent": pct(b),
            "cells": int(a.sum()), "comparison_cells": int(b.sum()),
            "rule": (f"|rate relative to the comparison ground of the same coherence class| / that cell's farmland "
                     f"null >= {z:g}, on the report's own comparison ground (second pass)")}


def chance_share_range(chance_pct: float | None, n_cells: int | None) -> list | None:
    """The 95% binomial range, in per cent, of the share of n_cells
    independent radar cells that stand out when each does so with chance's
    probability chance_pct (F7a, follow-up of 7 Oct 2026): [2.5%, 97.5%]
    points of Binomial(n_cells, chance_pct / 100) over n_cells. None without
    a cell count or a chance share."""
    if chance_pct is None or not n_cells or int(n_cells) <= 0 or not np.isfinite(float(chance_pct)):
        return None
    from scipy.stats import binom
    n = int(n_cells)
    p = min(max(float(chance_pct) / 100.0, 0.0), 1.0)
    lo, hi = binom.ppf([0.025, 0.975], n, p)
    return [round(100.0 * float(lo) / n, 2), round(100.0 * float(hi) / n, 2)]


def cells_sentence(stand: float | None, chance: float | None, n_cells: int | None = None) -> str:
    """The whole-area row's single-cell sentence from cells_standing_out's
    shares ('' when either is missing). 'The same as chance' only when the
    share lies inside chance_share_range for the study area's n_cells radar
    resolution cells (F7a, follow-up of 7 Oct 2026; a fixed 1.5 percentage
    points until then, 16 times the binomial half-range for 5.6% by chance
    on Site B's 242,891 cells: 5.51-5.69%). Without a cell count the
    range cannot be had and the two shares are given side by side, never
    called the same."""
    if stand is None or chance is None:
        return ""
    rng = chance_share_range(chance, n_cells)
    if rng is not None and rng[0] <= float(stand) <= rng[1]:
        return (f"At the scale of single points, {stand:.1f}% of the measured ground stands out, the same as chance "
                f"alone gives ({chance:.1f}%).")
    return (f"At the scale of single points, {stand:.1f}% of the measured ground stands out, against {chance:.1f}% "
            "by chance.")


def _cause_base(relation_key: str, sign: int, shape: str | None) -> str:
    """The default study's suggested cause (domains.GROUND_SUBSIDENCE causes)."""
    return domains.cause_text(domains.DEFAULT, relation_key, sign, shape)


# ------------------------------------------------------------------ corrections
# run_products.json tokens (time_series.run_products_record) -> plain words
_CORRECTION_WORDS = (
    ("ERA5", "weather delays (ERA5 atmospheric model)"),
    ("HRRR", "weather delays (NOAA HRRR weather model)"),
    ("GACOS", "weather delays (GACOS)"),
    ("tropHgt", "weather delays (height-correlated, estimated from the data)"),
    ("ramp", domains.DEFAULT.wording.correction_ramp),
    ("demErr", "differences between the elevation model and today's ground"),
    ("SET", "solid-earth tides"),
    ("GIM", "ionosphere (global ionosphere maps)"),
    ("ion", "ionosphere (split-spectrum estimate)"),
)


def _era5_words(rec: dict | None, dates: list[str] | None = None,
                label: str = "weather delays (ERA5 atmospheric model)") -> str | None:
    """The ERA5 line when the weather model covered only some dates
    (corrections.verify_era5_coverage with era5_allow_partial): 'for N of M
    dates' and which were left - the most recent ones when ERA5 had not been
    published for them yet. None when every date was corrected or nothing
    was recorded."""
    rec = rec or {}
    tot, done = rec.get("dates_total"), rec.get("dates_corrected")
    if not tot or done is None or int(done) >= int(tot):
        return None
    miss = list(rec.get("missing_dates") or [])
    k = int(tot) - int(done)
    base = f"{label} for {done} of {tot} dates"
    dates = sorted(str(d) for d in (dates or []))
    if dates and miss and sorted(miss) == dates[-len(miss):]:
        return base + (f"; the most recent {k} {'were' if k > 1 else 'was'} not corrected because the weather model "
                       "was not yet published for them")
    return base + (f"; {k} date{'s were' if k > 1 else ' was'} not corrected"
                   + (f" ({', '.join(miss[:6])}{' and more' if len(miss) > 6 else ''})" if miss else ""))


def _series_dates(ts_file: Path | None) -> list[str]:
    """The dates of the time series in use ([] when it cannot be read)."""
    if not ts_file or not Path(ts_file).exists():
        return []
    try:
        import h5py
        with h5py.File(ts_file, "r") as f:
            return [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]] if "date" in f else []
    except (OSError, ImportError):
        return []


def corrections_applied(cfg, ts_file: Path | None) -> list[str]:
    """What was removed from the signal - from what the run RECORDS it applied,
    never from what the config asked for (timeseries-9: the tide and
    ionosphere lines used to come from corrections.json, the decision written
    before MintPy's steps ran).

    1. mintpy/run_products.json, when its time series is the file in use: its
       `corrections_applied` tokens (tide = SET, GIM ionosphere read back from
       the final file's IONO_GIM_APPLIED attribute) and its ERA5 date coverage
       ("for N of M dates" when the weather model missed some).
    2. Otherwise the file in use: the tokens in its name (MintPy names each
       step's output: _ERA5, _ramp, _demErr, _SET) and its IONO_GIM_APPLIED
       attribute.
    3. The screening route (DISP-S1): tides and ionosphere only when the
       stitch MEASURED them subtracted from every date
       (corrections.json disp_corrections.status "all"); an older record
       without that block says nothing either way."""
    name = Path(ts_file).name if ts_file else ""
    tokens, era5 = None, None
    try:
        rp = json.loads((Path(cfg.mintpy_dir) / "run_products.json").read_text())
        used = {Path(p).name for p in (rp.get("timeseries_geo"), rp.get("timeseries_radar")) if p}
        toks = rp.get("corrections_applied")
        if isinstance(toks, list) and (not name or name in used):
            tokens = [str(x) for x in toks]
            era5 = rp.get("era5")
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    if tokens is None:
        tokens = [w for w, _ in _CORRECTION_WORDS if w not in ("GIM", "ion") and w in name]
        if ts_file and Path(ts_file).exists():
            try:
                import h5py
                with h5py.File(ts_file, "r") as f:
                    if f.attrs.get("IONO_GIM_APPLIED"):
                        tokens.append("GIM")
            except (OSError, ImportError):
                pass
    corr = {}
    try:
        corr = json.loads((Path(cfg.mintpy_dir) / "corrections.json").read_text())
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    if era5 is None and ("ERA5" in tokens or "HRRR" in tokens):
        era5 = corr.get("troposphere")                       # written by verify_era5_coverage for this run
    out = []
    ramp = domains.study_of(cfg).wording.correction_ramp
    for tok, words in _CORRECTION_WORDS:
        if tok == "ion" and "GIM" in tokens:
            continue
        if tok == "ramp":
            words = ramp
        if tok in tokens:
            if tok in ("ERA5", "HRRR") and str((era5 or {}).get("model") or "ERA5").upper() == tok:
                words = _era5_words(era5, _series_dates(ts_file), words) or words
            if words not in out:
                out.append(words)
    if getattr(cfg, "backend", "") == "disp":
        st = (corr.get("disp_corrections") or {}).get("status") if isinstance(corr.get("disp_corrections"), dict) else None
        if st == "all":
            out.append("tides and ionosphere (OPERA product layers)")
    return out


# ------------------------------------------------------------------ feedback
def quality_notes(Q: dict, rep: dict, up: float) -> list[str]:
    """Plain-language notes for the checks that bear on how to read the numbers."""
    C = Q.get("checks") or {}
    notes = []
    # Two numbers, not one (methodology-3): the coherence gate alone says a
    # cell RETURNED a measurement, not that it is reliable; reliable_percent
    # (QA since 5 Oct 2026) is the share that also passes MintPy's unwrapping
    # consistency - 43% on site_a_pl against 97% at the gate. An older QA
    # file has only the first, and is worded as such.
    cv = C.get("coverage") or {}
    cov, rel = cv.get("aoi_percent_at_threshold"), cv.get("reliable_percent")
    if cov is not None:
        # whether the second check gates anything, said (final check of 7 Oct
        # 2026: "52% passed both" beside "ground that changes too much gives
        # no measurement" read as if the rest were not measured; with
        # min_unwrap_consistency 0 every measured point is in the figures)
        sg2 = cv.get("second_gate") or {}
        gate = sg2.get("applied")
        # "pass both" counts the second check at the QA's reliable level; a
        # config gate set looser than that still lets some failing points in
        # (review of the final check, 7 Oct 2026)
        try:
            from .quality_assessment import RELIABLE_UNWRAP_CONSISTENCY as level
        except Exception:                                     # noqa: BLE001 - the words only
            level = 0.7
        strict = sg2.get("applied_threshold") is None or float(sg2["applied_threshold"]) >= level - 1e-9
        use = ("" if rel is None or gate is None else
               " Only points that pass both are used in the figures." if gate and strict else
               " The figures leave out only the points that fail a looser form of the second check, so some that "
               "fail it are still used." if gate else
               f" All measured points are used in the area figures; those that fail the second check are noisier, "
               f"and that noise is part of the {ground_word(rep)} comparison the ranges come from.")
        notes.append(f"{cov:.0f}% of the study area returned a measurement"
                     + (f"; {rel:.0f}% passed both quality checks (the second is how consistently the radar phase "
                        "could be followed through time)" if rel is not None else "")
                     + ". Ground that changes too much between images gives no measurement." + use)
    ground = ground_word(rep)
    limits = [float(x) for x in rep.get("detection_limits_mm_yr") or [] if x]
    rp = C.get("rate_precision") or {}
    if rp and not rp.get("pass", True) and rp.get("empirical_median_mm_yr"):
        notes.append(f"A single measurement point over {ground} is uncertain by about "
                     f"{rp['empirical_median_mm_yr'] / up / MM_IN:.2f} in/yr (1 sigma), so the report works with areas, "
                     "where that noise averages out, rather than with single points.")
    ds = C.get("distributed_scatterer_bias") or {}
    if ds and not ds.get("pass", True):
        notes.append(ds_bias_note(ds, up, ground, limits))
    rs = C.get("reference_stability") or {}
    if rs and not rs.get("pass", True) and rs.get("value_mm_yr") is not None:
        # the measured drift, never "a little": Site B's -3.1 mm/yr LOS
        # (-0.16 in/yr) was three times the yardstick and above the project
        # zone's detection limit (empirical-12, clientout-18)
        d = float(rs["value_mm_yr"]) / up
        yd = rs.get("yardstick_mm_yr")
        big = bool(limits) and abs(d) >= min(limits)
        notes.append(f"The reference point drifts against the wider area by about {signed(d / MM_IN)} in/yr "
                     f"({signed(d, 1)} mm/yr"
                     + (f"; good practice is under {float(yd) / up / MM_IN:.2f} in/yr, {float(yd) / up:.1f} mm/yr"
                        if yd else "") + ")"
                     + (", more than the smallest movement the report areas can detect" if big else "")
                     + f". Every figure is measured against the surrounding {ground}"
                     + (" and tied to GNSS" if rep.get("anchored") else "")
                     + ", not against the reference point, so the drift does not enter them.")
    nw = C.get("network") or {}
    if nw and not nw.get("connected", True):
        notes.append("The image network is not fully connected; some dates could not be tied to the rest.")
    se = C.get("single_epoch_noise") or {}
    if se and not se.get("pass", True) and se.get("robust_mm"):
        notes.append(f"Single images are noisier than usual (about {se['robust_mm'] / MM_IN:.2f} in); rates over "
                     "the record are reliable, individual dates less so.")
    if not rep.get("anchored"):
        why = (rep.get("gnss_reason") or "").strip().rstrip(".")
        notes.append("No GNSS station could anchor the figures" + (f" ({why[0].lower() + why[1:]})" if why else "")
                     + f", so they are relative to the area's own {ground} rather than absolute, and the movement of "
                       "the study area as a whole is not measured.")
    return notes


def _tc_low(label) -> float | None:
    """0.7 from a coherence-class label '0.7-0.8'."""
    m = re.match(r"\s*([0-9.]+)", str(label or ""))
    return float(m.group(1)) if m else None


def ds_bias_note(ds: dict, up: float, ground: str = "farmland", limits: list | None = None) -> str:
    """
    The distributed-scatterer bias note WITH its size (methodology-9: "read
    slightly more negative ... which removes it" was printed for Site B,
    whose lower-quality farmland read 14.3 mm/yr LOS below the best points).

    Size = median rate of the dominant coherence class (most points) minus the
    point-weighted median class rate of the classes at 0.7 and above - not
    low_minus_high, whose top class (967 points on Site B) is too small to
    stand for good ground. Vertical, in/yr. "slightly" only when that is under
    half the smallest report-area detection limit. The cancellation is stated
    as what it is: an assumption that the effect is the same on the site as on
    the comparison ground.
    """
    rows = [r for r in ds.get("velocity_by_coherence_class") or []
            if r.get("median_mm_yr") is not None and r.get("n")]
    generic = (f"Lower-quality {ground} points read more negative than high-quality ones (a known effect of seasonal "
               f"soil and crop changes and of errors in following the radar phase through time). Every figure "
               f"compares ground with {ground} of the same quality, which cancels it as long as the effect is the "
               f"same at the site as on the comparison {ground}.")
    if not rows:
        return generic
    dom = max(rows, key=lambda r: r["n"])
    high = [r for r in rows if (_tc_low(r.get("tc")) or 0.0) >= 0.7]
    if not high or dom in high:
        return generic
    order = sorted(high, key=lambda r: r["median_mm_yr"])
    w = np.cumsum([r["n"] for r in order], dtype=float)
    ref = order[int(np.searchsorted(w, 0.5 * w[-1]))]["median_mm_yr"]
    diff = (float(dom["median_mm_yr"]) - float(ref)) / up
    share = 100.0 * dom["n"] / sum(r["n"] for r in rows)
    slight = bool(limits) and abs(diff) < 0.5 * min(limits)
    return (f"Most {ground} points ({share:.0f}% of them, coherence {dom.get('tc')}) read about "
            f"{abs(diff) / MM_IN:.2f} in/yr ({abs(diff):.0f} mm/yr) {'slightly ' if slight else ''}"
            f"{'more negative' if diff < 0 else 'more positive'} than the best points. This comes partly from "
            f"seasonal soil and crop effects and partly from errors in following the radar phase through time. "
            f"Every figure compares ground with {ground} of the same quality, which cancels it as long as the effect "
            f"is the same at the site as on the comparison {ground}.")


def _past_or_present(f: dict) -> str:
    """'moves' for a steady pace; 'averaged' when the pace changed."""
    tr = f.get("trend") or {}
    return "averaged" if (f.get("nonlinear") or tr.get("shape") in ("slowing", "accelerating")) else "moves"


def reshaped_since_dem(f: dict) -> bool:
    """The feature's ground was reshaped since the elevation model - by the
    test with the movement taken out (dem_change: `rate_removed` and
    `flag`), never by the raw DEM error alone (E1), and not when the ground
    was reshaped during the record (that says more)."""
    dc = f.get("dem_change") or {}
    return bool(dc.get("rate_removed") and dc.get("flag") and not f.get("ground_reshaped")
                and dc.get("feature_median_m") is not None)


def _recent_clause(f: dict) -> str:
    """'; about 0.41 in/yr (10 mm/yr) down over 2024–2026 (95% range ...)'
    when the last two years' 95% range excludes zero; otherwise that its
    rate then cannot be told apart from no movement (E11, trust review 7 Oct
    2026: Site B Features 1 and 2 read "about 0.38 in/yr (10 mm/yr) over
    2024–2026" after "down", with ranges of -1.01 to +0.25 and -0.93 to
    +0.12 in/yr). No range: no rate."""
    rec = f.get("recent_rate_mm_yr")
    rw = f.get("recent_window") or []
    if rec is None or len(rw) != 2:
        return ""
    yrs = f"{str(rw[0])[:4]}–{str(rw[1])[:4]}"
    rng = f.get("recent_range95_mm_yr")
    if not rng or rng[0] is None or rng[1] is None:
        return f"; its rate over {yrs} could not be tested"
    lo, hi = float(rng[0]), float(rng[1])
    r_in = f"95% range {signed(lo / MM_IN)} to {signed(hi / MM_IN)} in/yr"
    if not (lo > 0 or hi < 0):
        return f"; over {yrs} its rate cannot be told apart from no movement ({r_in})"
    # the word follows the range (the test's frame); a measured value on the
    # other side of zero (a range centred on the excess over farmland's
    # median, not on the value) is not printed beside it
    word = "down" if hi < 0 else "up"
    if (rec < 0) != (hi < 0):
        return f"; over {yrs} it moved {word} ({r_in})"
    return f"; about {abs(rec) / MM_IN:.2f} in/yr ({abs(rec):.0f} mm/yr) {word} over {yrs} ({r_in})"


def rate_range_words(v, rng) -> str:
    """'95% range 0.39 to 0.91 in/yr' for a feature's rate (M4, final check
    of 7 Oct 2026: the footer says every figure carries its uncertainty,
    and feature rates carried none) - sizes when the whole range lies on
    the rate's side of zero, signed otherwise; '' for a record without it."""
    try:
        lo, hi = sorted(float(x) for x in rng)
    except (TypeError, ValueError):
        return ""
    if v < 0 and hi < 0 or v > 0 and lo > 0:
        a, b = sorted((abs(lo), abs(hi)))
        return f"95% range {a / MM_IN:.2f} to {b / MM_IN:.2f} in/yr"
    return f"95% range {signed(lo / MM_IN)} to {signed(hi / MM_IN)} in/yr, negative = down"


# M4 (final check of 7 Oct 2026): a feature's outline is drawn where the
# measurements go furthest beyond farmland, so the same noise that drew it
# is inside its average: on farmland copies with each feature planted and
# its outline redrawn, the outline's average read about 3 mm/yr stronger
# than the ground inside it moved, and the 95% range held it 77-91% of the
# time, every miss on the stronger side (verify_stats_adv/v2_feat.py).
OUTLINE_CAVEAT = ("Its outline was drawn from these same measurements, so its rate, total and ranges are for that "
                  "outline and read somewhat stronger than the ground there truly moved.")


def _total_range_clause(tot, rng) -> str:
    """' (95% range a to b in)' for a feature's measured total (E9: every
    total a client reads carries its range; client_report.total_range on
    the feature's footprint test). Said as sizes of the movement when the
    whole range lies on the total's side of zero, signed otherwise; '' for
    a record without the range (built before 7 Oct 2026)."""
    try:
        lo, hi = sorted(float(x) for x in rng)
    except (TypeError, ValueError):
        return ""
    if tot < 0 and hi < 0 or tot > 0 and lo > 0:
        a, b = sorted((abs(lo), abs(hi)))
        return f" (95% range {a / MM_IN:.1f} to {b / MM_IN:.1f} in)"
    return f" (95% range {lo / MM_IN:+.1f} to {hi / MM_IN:+.1f} in, negative = down)"


def _average_total_range(f: dict, v, avg):
    """The 95% range of a feature's total at its average rate: the record's
    own (total_at_average_rate_range95_mm = the rate's range x the years),
    else, for a record built before it was kept, the rate's 95% range times
    the years that total was worked over (avg / rate). None without the
    rate's range (final check of 7 Oct 2026: the page printed this total
    bare while the KMZ gave its range)."""
    if f.get("total_at_average_rate_range95_mm"):
        return f["total_at_average_rate_range95_mm"]
    rng = f.get("rate_range95_mm_yr")
    try:
        span = float(avg) / float(v)
        lo, hi = sorted(float(x) for x in rng)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not (span > 0 and np.isfinite(span)):
        return None
    return [lo * span, hi * span]


def feature_text(i: int, f: dict, wording=None) -> str:
    """The Observations paragraph for feature i (1-based). Totals are the
    MEASURED change over the record (significance.record_change); when the
    pace changed, the rate is said in the past tense as an average and the
    last two years' rate follows it (Site B Feature 1: "moves about 1.56
    in/yr ... 16.3 in over the record" read 2.4 x its current pace and 22%
    over its own series) - with its 95% range, and a direction only when
    that range excludes zero (_recent_clause, E11).

    The verdict (E7, E8; trust review 7 Oct 2026): "stands out clearly"
    only for f["stands_out_clearly"] (selection_p: the calibrated chance
    test AND the calibrated QA stand-out AND copies as strong as the real
    farmland); any other feature that beats chance is "worth a look rather
    than a confirmed finding". The chance is said as selection_p's p_text
    ("less than 1 in 200" when it beats every one of 199 copies - a bound,
    never "about 0.005"); for a feature listed on its leave-out re-test
    (chance["listed_on"], leave_out_chance) the re-test's chance corrected
    for the number of patches re-tested (G2: "about 3 in 200"). "Its
    ground was reshaped since the elevation model" and the
    earthmoving hint need dem_change's test with the movement taken out
    (reshaped_since_dem, E1).

    Final check of 7 Oct 2026: the rate carries its 95% range
    (rate_range95_mm_yr, M4) and the outline caveat follows the figures
    (OUTLINE_CAVEAT); the last two years' clause is given for every
    feature, steady or not; "stands out clearly" not when the QA's own
    calibration is flagged (H1: clearly_withheld_by "qa_calibration");
    the QA's test is "a second, separate check", not its name; the total
    at the average rate, when it is printed, carries its range as the KMZ
    gives it (the rate's range times the years, _average_total_range)."""
    sink = f["sign"] < 0
    v = f.get("vertical_relative_mm_yr", f["mean_rel_vertical_mm_yr"])
    tot = f.get("total_relative_mm")
    verb = _past_or_present(f)
    ground = ground_word(f)
    mines = f.get("mines") or {}
    if f.get("ground_reshaped"):
        # dug or built up by metres: the radar's number is not a measurement
        # of it (verify note: gated on the measured elevation change, not on
        # every rising feature - slow heave on coherent ground is real)
        txt = ("Radar cannot measure ground being dug out or built up by metres, and this ground's height differs "
               "from the elevation model by that much: it changed during the record, and no rate is given for it")
    else:
        rr = rate_range_words(v, f.get("rate_range95_mm_yr"))
        txt = (f"It {verb} about {abs(v) / MM_IN:.2f} in/yr ({abs(v):.0f} mm/yr{'; ' + rr if rr else ''}) "
               f"{'down' if sink else 'up'} relative to the surrounding {ground}"
               + (" over the whole record" if verb == "averaged" else ""))
        # the last two years for every feature, steady or not (final check
        # of 7 Oct 2026: the KMZ gave Site A Feature 1's 2024-2026 rate
        # and the page did not)
        txt += _recent_clause(f)
        if tot is not None:
            txt += (f". Measured over the record it moved {abs(tot) / MM_IN:.1f} in ({abs(tot):.0f} mm)"
                    + _total_range_clause(tot, f.get("total_range95_mm")))
            avg = f.get("total_at_average_rate_mm")
            # the page's and the KMZ's rule (client_report.totals_differ, E20 #37)
            from . import client_report as cr
            if cr.totals_differ(tot, avg):
                txt += (f"; at its average rate it would have moved {abs(avg) / MM_IN:.1f} in"
                        + _total_range_clause(avg, _average_total_range(f, v, avg)))
        txt += ". " + OUTLINE_CAVEAT.rstrip(".")
    txt += ". " + where_sentence(mines, wording)
    from . import landcover
    txt += landcover.wooded_sentence(f.get("land_cover"))
    dc = f.get("dem_change") or {}
    reshaped = reshaped_since_dem(f)
    if reshaped:
        # gap-surface_mine_site-6, E1: said in the feature's own words, after
        # the part of the DEM error its movement explains is taken out
        # both differences from the comparison ground's own median, so the
        # three figures add up; the movement's part can point the other way
        # (a residual of +14 m on ground whose rate explains -12 m read "the
        # remaining 14 m" of a 2 m difference until the review of 7 Oct)
        base = float(dc.get("farmland_median_m") or 0.0)
        raw = float(dc["feature_median_m"]) - base
        mv = float(dc["expected_from_rate_m"]) - base
        if abs(mv) < 0.5:
            mv_txt = f"ordinary {ground} moving as fast as this hardly differs from it"
        else:
            mv_txt = (f"ordinary {ground} moving as fast as this differs by about {abs(mv):.0f} m"
                      + (" the other way" if mv * raw < 0 and abs(raw) >= 0.5 else "") + " from the movement alone")
        own = (f"its height differs from the model by about {abs(raw):.0f} m" if abs(raw) >= 0.5 else
               "its height matches the model to within a metre")
        txt += (f" Its ground was reshaped since the elevation model was made: averaged over "
                f"{float(dc.get('smoothing_m') or 500):.0f} m {own}, where {mv_txt}. That leaves about "
                f"{abs(float(dc['residual_m'])):.0f} m the "
                f"movement does not explain, more than ordinary {ground} shows (under "
                f"{float(dc['residual_threshold_m']):.0f} m).")
    key = str(mines.get("key") or "")
    if key in ("surface", "underground", "mapped") or key.startswith("near_"):
        # one ascending view reads east-west movement as vertical (stats-16,
        # methodology-15); mining moves the ground sideways at a bowl's edges
        los = f.get("los_relative_mm_yr")
        txt += (" Near mining the ground also moves sideways, which this single satellite view partly reads as "
                "vertical" + (f"; the radar itself measured {abs(los):.0f} mm/yr along its line of sight" if los else "")
                + " (see Limitations).")
    tr = f.get("trend") or {}
    if tr.get("text"):
        txt += " " + tr["text"][0].upper() + tr["text"][1:] + "."
    ch = f.get("chance") or {}
    hint = f.get("cause_hint", "") or ""
    if not reshaped:
        hint = hint.replace(RESHAPED_HINT.strip(), "").strip()
    # "stands out clearly" only as selection_p decided it (E7, contract C4);
    # a record without that decision (built before 7 Oct 2026, when the
    # uncalibrated QA rule alone gave the label) gets the weaker wording
    clearly = f.get("stands_out_clearly")
    if clearly is None:
        clearly = ch.get("stands_out_clearly")
    clearly = bool(clearly) and ch.get("stands_out") is True
    if ch.get("tested") and ch.get("p") is not None:
        k, n = ch.get("copies_as_strong"), ch.get("copies")
        pt = ch.get("p_text") or p_text(k, n, ch["p"])
        L = int(ch.get("retested") or 1) if ch.get("listed_on") == "leave-out re-test" else 1
        if n and L > 1:
            # G2: listed on its leave-out re-test, the chance corrected for
            # the number of patches re-tested that way
            copies = (f"in {n} simulated copies of this area's own ordinary {ground}, made with this patch left out, "
                      f"{'none' if not k else k} produced a patch as large and as strong; as {L} of the strongest "
                      f"patches were each tested this way, the chance is {pt}")
        else:
            copies = (f"in {n} simulated copies of this area's own ordinary {ground}, {'none' if not k else k} "
                      f"produced a patch as large and as strong (a chance of {pt})" if n else
                      f"the chance of a patch this strong anywhere in the area is {pt}")
        second = f"a second, separate check against the deepest patch ordinary {ground} produces anywhere in an area this size"
        if clearly:
            txt += " " + hint
            txt += (f" It stands out clearly from what ordinary {ground} produces by chance: {copies}, and it also "
                    f"passes {second}.")
        elif ch.get("stands_out") is False:
            # within chance: no cause is suggested for what chance explains
            txt += (" It was found by searching the whole area for patches beyond the normal range; " + copies
                    + f": ordinary {ground} produces patches like this in an area this size, so it is within chance.")
        else:
            txt += " " + hint
            why = ""
            if f.get("quality_standout") and ch.get("clearly_withheld"):
                if ch.get("clearly_withheld_by") == "qa_calibration":
                    # H1: that check's own copies are not a close match to
                    # this farmland, so it cannot confirm the feature
                    why = (f". It also passes {second}, but that check's simulated {ground} is not a close enough "
                           f"match to the real {ground} here to confirm it")
                else:
                    why = (f". It also passes {second}, but ordinary {ground} elsewhere in this area holds a patch "
                           "stronger than any of the simulated copies, so the chance figure may be too low")
            txt += (" It was found by searching the whole area for patches beyond the normal range; " + copies
                    + why + ": worth a look rather than a confirmed finding.")
    else:
        txt += (" It could not be tested against what chance alone produces in an area this size"
                + (f" ({ch['reason']})" if ch.get("reason") else "") + ", so no cause is suggested: it needs a "
                "specialist's look before anything is read into it.")
    return " ".join(txt.split())


def observations(rep: dict) -> list[dict]:
    """[{title, text}] - the report's feedback, strongest first."""
    obs = []
    ground = ground_word(rep)
    for i, f in enumerate(rep.get("features") or [], 1):
        head = f"Feature {i}: {feature_kind(f)}, about {f['area_km2']:.1f} km² near {lalo_txt(f['lat'], f['lon'])}"
        obs.append({"title": head, "text": f.get("text") or feature_text(i, f, domains.wording_of(rep)),
                    "kind": "feature"})
    weak = rep.get("features_within_chance") or []
    if weak:
        # "weaker" only when every one is slower than every listed feature
        # (final check of 7 Oct 2026: Site A's 4 "weaker" patches
        # included two faster than Feature 1 - -18.8 and -17.8 against
        # -16.3 mm/yr - on 1.9 and 2.6 km² against 23.7; the search ranks
        # by rate x area, not by rate)
        fastest = max(abs(w.get("mean_rel_vertical_mm_yr") or 0) for w in weak)
        biggest = max(float(w.get("area_km2") or 0) for w in weak)
        feats = rep.get("features") or []
        rates = [abs(f.get("vertical_relative_mm_yr", f.get("mean_rel_vertical_mm_yr")) or 0)
                 for f in feats if not f.get("ground_reshaped")]
        sizes = [float(f.get("area_km2") or 0) for f in feats]
        # "smaller" likewise only when each is smaller than every listed
        # feature (Site B's 4.35 km² patch is larger than its Feature 2)
        word = ("weaker" if rates and fastest < min(rates) else "smaller" if sizes and biggest < min(sizes)
                else "other")
        many = len(weak) > 1
        obs.append({"title": f"{len(weak)} {word} patch{'es' if many else ''} within chance",
                    "text": (f"{len(weak)} further patch{'es' if many else ''} beyond the normal range "
                             f"{'were' if many else 'was'} found ("
                             + (f"up to {fastest / MM_IN:.2f} in/yr, over areas up to {biggest:.1f} km²" if many else
                                f"about {fastest / MM_IN:.2f} in/yr over {biggest:.1f} km²")
                             + f"); ordinary {ground} produces patches like "
                             + ("these" if many else "this") + " in an area this size, so "
                             + ("they are" if many else "it is") + " not listed as features."),
                    "kind": "feature"})
    from . import landcover
    for m in rep.get("mine_ground") or []:
        if m.get("note") and m.get("relative_vertical_mm_yr") is None:
            continue
        v = m["relative_vertical_mm_yr"]
        meas = (f"{m['area_km2']:.1f} of {m['footprint_km2']:.1f} km² measured" if m.get("footprint_km2")
                else f"{m['area_km2']:.1f} km² measured")
        if m.get("testable") is False:
            verdict = ("it could not be compared with ordinary ground" + (f" ({m['not_testable_reason'].rstrip('.')})"
                                                                          if m.get("not_testable_reason") else "") + ".")
        else:
            verdict = ("that stands out from ordinary ground." if m.get("detected") else
                       "within the range ordinary ground of the same size varies.")
            # a thin or square-patch comparison says so, as the report areas'
            # do (methodology-2: Site A's mine ground rests on ~7
            # independent placements and printed no caveat)
            if m.get("null_thin") or m.get("null_shape") == "compact_square":
                from . import client_report as cr
                verdict += cr.thin_null_sentence(m, ground, what="this comparison")
        # its 95% range on the test's frame, as a report area's (M4, final
        # check of 7 Oct 2026: the footer promises every figure its range)
        mr = m.get("range95_mm_yr")
        mr_txt = (f"; 95% range {signed(mr[0] / MM_IN)} to {signed(mr[1] / MM_IN)} in/yr"
                  if mr and None not in mr else "")
        if not mr_txt and m.get("testable") is False:
            # nothing to state a range from: said, not left to the footer
            verdict = verdict.rstrip(".") + ", so no 95% range can be given for it."
        obs.append({"title": f"Ground on the mapped {m['label']}",
                    "text": (f"{meas}. On average it moves {signed(v / MM_IN)} in/yr ({signed(v, 1)} mm/yr{mr_txt}) "
                             f"relative to the surrounding {ground}: " + verdict
                             + landcover.wooded_sentence(m.get("land_cover"), "This ground")
                             + (" Individual features on it are listed above." if any(
                                 (f["mines"].get("key") or "") == m["key"] for f in rep.get("features") or []) else "")),
                    "kind": "mines"})
    if rep.get("features_assessed") is False:
        obs.insert(0, {"title": "Features could not be assessed",
                       "text": (f"The study area holds too little ordinary {ground} to measure its normal range"
                                + (f" ({rep['features_reason']})" if rep.get("features_reason") else "")
                                + ", so no search for local features could be made. This is not a finding that "
                                  "nothing moves."), "kind": "feature"})
    elif not rep.get("features"):
        obs.insert(0, {"title": "No feature stands out",
                       "text": f"No patch of ground larger than a few fields moves beyond what ordinary {ground} "
                               "produces by chance anywhere in the study area.", "kind": "feature"})
    return obs


def where_sentence(mines: dict | None, wording=None) -> str:
    """'It lies over mapped underground workings.' - or, with no mine
    outlines, that it cannot be related to mining ('It lies no mine outlines
    were supplied' was printed until 5 Oct 2026)."""
    mines = mines or {}
    if not mines.get("key") and not mines.get("inside"):
        return (wording or domains.DEFAULT.wording).no_layer_feature
    return f"It lies {mines.get('phrase', '')}."


def feature_kind(f: dict) -> str:
    """The feature's headline. 'sinking/rising ground' everywhere but on a
    surface mine: there a patch at least ON_LAYER_SHARE on the layer is
    'change on surface-mine ground' - pits, haul roads and fill move by
    metres, which radar cannot follow, so 'rising ground, 0.8 in/yr' would
    present earthmoving as a measured rate (gap-surface_mine_site-7)."""
    inside = ((f.get("mines") or {}).get("inside") or {})
    if f.get("ground_reshaped") or float(inside.get("surface") or 0.0) >= ON_LAYER_SHARE:
        return "change on surface-mine ground" if float(inside.get("surface") or 0.0) >= ON_LAYER_SHARE \
            else "ground reshaped during the record"
    return f"{'sinking' if f['sign'] < 0 else 'rising'} ground"


# ------------------------------------------------------------------ the client map
# Readable bin edges in in/yr (the map is labelled in inches first).
NICE_IN = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0)
MAX_BINS_PER_SIDE = 4
# The darkest colour starts at the 99th percentile of the coloured cells, not
# their maximum: one runaway cell must not flatten the map.
SCALE_PERCENTILE = 99.0
MIN_PATCH_KM2 = 0.25        # MIN_FEATURE_KM2: smaller coloured specks are not listed


def map_scale(field_rel: np.ndarray, where: np.ndarray, lo_b: float, hi_b: float) -> dict:
    """
    The client map's colour bins (clientout-5, generality-4). Until 5 Oct
    2026 the scale was fixed at +/-0.6 in/yr behind a +/-0.40 in/yr grey
    band, so colour separated only 0.40-0.60 in/yr: on Site B 30% of the
    coloured cells (to -2.9 in/yr) were the same saturated red as a chance
    patch of -0.55.

    field_rel: the map's field, vertical mm/yr relative to the comparison
    ground (the frame of the grey band and of every feature figure). where:
    the cells drawn. The grey band [lo_b, hi_b] is the middle bin; beyond it
    readable steps (NICE_IN, the same on both sides so red and blue shades
    mean the same size) up to `top` = the smallest step at or above the
    coloured cells' 99th percentile and 1.5 x the band's wider edge, at most
    MAX_BINS_PER_SIDE bins a side, the last open-ended (extend arrows).
    Returns edges in mm/yr (ascending, lo_b and hi_b included), top, the
    peak cell and the share of coloured cells beyond top.
    """
    v = np.asarray(field_rel, dtype=float)
    ok = where & np.isfinite(v)
    col = ok & ((v <= lo_b) | (v >= hi_b))
    vin = np.abs(v[col]) / MM_IN
    edge_in = max(abs(lo_b), abs(hi_b)) / MM_IN
    p = float(np.percentile(vin, SCALE_PERCENTILE)) if vin.size else edge_in * 1.5
    need = max(p, 1.5 * edge_in)
    top = next((x for x in NICE_IN if x >= need - 1e-9), NICE_IN[-1])

    def side(e_in):
        inner = [x for x in NICE_IN if 1.2 * e_in < x < top - 1e-9]
        while len(inner) > MAX_BINS_PER_SIDE - 1:
            inner = inner[1::2] if len(inner) > 2 else inner[-1:]
        return inner + [top]

    neg = [-x * MM_IN for x in reversed(side(abs(lo_b) / MM_IN))]
    pos = [x * MM_IN for x in side(abs(hi_b) / MM_IN)]
    edges = neg + [float(lo_b), float(hi_b)] + pos
    out = {"edges_mm_yr": [round(x, 2) for x in edges], "top_in_per_yr": top, "n_below": len(neg),
           "band_mm_yr": [round(float(lo_b), 2), round(float(hi_b), 2)], "frame": "relative",
           "coloured_cells": int(col.sum()),
           "beyond_top_percent": round(100.0 * float(np.mean(vin > top)), 1) if vin.size else 0.0,
           "peak_mm_yr": None, "peak_rc": None}
    if col.any():
        vv = np.where(col, np.abs(v), -np.inf)
        r, c = np.unravel_index(int(np.argmax(vv)), v.shape)
        out.update({"peak_mm_yr": round(float(v[r, c]), 1), "peak_rc": [int(r), int(c)]})
    return out


def map_patches(field_rel: np.ndarray, lo_b: float, hi_b: float, areas: dict, features: list,
                px_km2: float, attrs: dict, min_km2: float = MIN_PATCH_KM2, limit: int = 12) -> list[dict]:
    """
    Every coloured patch of at least min_km2 that touches a report area
    (clientout-5: a 1.97 km2 patch at -13.8 mm/yr over mapped panels inside
    Site B's own project zone was drawn the same red as Feature 1 and never
    mentioned). areas: {name: footprint mask}; features: the listed features
    (with bbox) - a patch overlapping one is that feature, any other is
    within what ordinary ground produces by chance (the calibrated search
    found nothing in it). Strongest first, at most `limit`.
    """
    from scipy import ndimage
    v = np.asarray(field_rel, dtype=float)
    fin = np.isfinite(v)
    lat_of = lambda r: float(attrs["Y_FIRST"]) + (r + 0.5) * float(attrs["Y_STEP"])    # noqa: E731
    lon_of = lambda c: float(attrs["X_FIRST"]) + (c + 0.5) * float(attrs["X_STEP"])    # noqa: E731
    out = []
    for sign, hot in ((-1, fin & (v <= lo_b)), (1, fin & (v >= hi_b))):
        lab, n = ndimage.label(hot)
        if not n:
            continue
        idx = np.arange(1, n + 1)
        size = ndimage.sum(hot, lab, idx) * px_km2
        for name, m in areas.items():
            touch = np.unique(lab[m & hot])
            for i in touch[touch > 0]:
                if size[i - 1] < min_km2:
                    continue
                comp = lab == i
                ys, xs = np.nonzero(comp)
                lat, lon = lat_of(ys.mean()), lon_of(xs.mean())
                pw, pe = lon_of(xs.min()), lon_of(xs.max())
                ps, pn = lat_of(ys.max()), lat_of(ys.min())
                # the same patch as a listed feature: same sign, and either
                # centre inside the other's box
                feat = next((k for k, f in enumerate(features, 1) if f.get("sign") == sign and f.get("bbox") and (
                    (f["bbox"][0] <= lon <= f["bbox"][2] and f["bbox"][1] <= lat <= f["bbox"][3])
                    or (pw <= f.get("lon", 1e9) <= pe and ps <= f.get("lat", 1e9) <= pn))), None)
                out.append({"area": name, "sign": sign, "area_km2": round(float(size[i - 1]), 2),
                            "inside_area_km2": round(float((comp & m).sum()) * px_km2, 2),
                            "mean_rel_vertical_mm_yr": round(float(v[comp].mean()), 1),
                            "lat": round(lat, 4), "lon": round(lon, 4), "feature": feat})
    out.sort(key=lambda p: -abs(p["mean_rel_vertical_mm_yr"]) * p["area_km2"])
    return out[:limit]


# The basemap: USGS The National Map topographic tiles - a US Government work,
# public domain, no key (Kyle, 5 Oct 2026: USGS by default). Cached as
# fetched; a second build needs no network, and with none it is skipped.
BASEMAP_URL = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"
BASEMAP_CREDIT = "Basemap: USGS The National Map (public domain)"
BASEMAP_MAX_TILES = 64
BASEMAP_TIMEOUT_S = 10


def _tile_xy(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Web-Mercator tile coordinates (fractional) of a point at zoom z."""
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n
    return float(x), float(y)


def basemap(bbox, cache_dirs: list, write_dir: Path | None, fetch: bool = True,
            target_px: int = 1400) -> dict | None:
    """
    A light map of roads, towns and rivers under the client map
    (clientout-12), as {"image": (H, W, 3) float array on a regular lon/lat
    grid, "extent": (w, e, s, n), "credit", "zoom", "tiles", "fetched"}, or
    None when it cannot be had - offline, a tile missing, or disabled. Never
    raises: the map is drawn without it.

    Tiles are looked up in each of cache_dirs (read-only is fine: a preview
    reads the project's cache) and, when fetch is set, the missing ones are
    downloaded into write_dir. The Web-Mercator mosaic is resampled row by row
    onto the map's lon/lat grid, so towns sit where the radar cells are.
    """
    try:
        w, s, e, n = (float(v) for v in bbox)
        z = int(np.clip(round(np.log2(target_px / 256.0 * 360.0 / max(e - w, 1e-6))), 6, 15))
        x0, y0 = _tile_xy(n, w, z)
        x1, y1 = _tile_xy(s, e, z)
        tx = range(int(np.floor(x0)), int(np.floor(x1)) + 1)
        ty = range(int(np.floor(y0)), int(np.floor(y1)) + 1)
        if len(tx) * len(ty) > BASEMAP_MAX_TILES:
            log.info("Basemap skipped: %d tiles at zoom %d", len(tx) * len(ty), z)
            return None
        from matplotlib import image as mpimg
        rows, fetched = [], 0
        for ty_ in ty:
            row = []
            for tx_ in tx:
                rel = Path("usgs_topo") / str(z) / f"{tx_}_{ty_}.jpg"
                p = next((Path(d) / rel for d in cache_dirs if d and (Path(d) / rel).exists()), None)
                if p is None:
                    if not fetch or write_dir is None:
                        return None
                    p = Path(write_dir) / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    from . import areas
                    ua = areas.user_agent() if hasattr(areas, "user_agent") else areas.USER_AGENT
                    req = urllib.request.Request(BASEMAP_URL.format(z=z, x=tx_, y=ty_), headers={"User-Agent": ua})
                    with urllib.request.urlopen(req, timeout=BASEMAP_TIMEOUT_S) as r:
                        data = r.read()
                    tmp = p.with_suffix(".part")
                    tmp.write_bytes(data)
                    tmp.replace(p)
                    fetched += 1
                img = mpimg.imread(p, format="jpg")
                img = img[..., :3].astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)
                if img.shape[:2] != (256, 256):
                    return None
                row.append(img)
            rows.append(np.concatenate(row, axis=1))
        mosaic = np.concatenate(rows, axis=0)
        H = max(64, int(round((n - s) / (e - w) * mosaic.shape[1] / np.cos(np.radians(0.5 * (s + n))))))
        Wd = mosaic.shape[1]
        lats = np.linspace(n, s, H)
        lons = np.linspace(w, e, Wd)
        ys = np.array([_tile_xy(la, w, z)[1] for la in lats])
        xs = np.array([_tile_xy(s, lo, z)[0] for lo in lons])
        ri_ = np.clip(((ys - ty.start) * 256).astype(int), 0, mosaic.shape[0] - 1)
        ci_ = np.clip(((xs - tx.start) * 256).astype(int), 0, mosaic.shape[1] - 1)
        return {"image": mosaic[ri_][:, ci_], "extent": (w, e, s, n), "credit": BASEMAP_CREDIT,
                "zoom": z, "tiles": len(tx) * len(ty), "fetched": fetched}
    except Exception as exc:                                  # noqa: BLE001 - the map is drawn without it
        log.info("Basemap not drawn (%s)", exc)
        return None
