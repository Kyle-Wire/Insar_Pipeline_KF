"""The study registry: what a study offers and how its report is worded.

S0.4 of review_2026-11-07/PLAN.md (8 Nov 2026). One study for now, the default
`ground_subsidence` (mining subsidence on farmland), whose wording and limits
are today's: report_insights.KIND_LABEL, NEAREST_LABEL, NEAR_MINE_KM,
ON_LAYER_SHARE, MINING_HORIZONTAL_SHARE and _cause_base, the ramp line of
_CORRECTION_WORDS, the "Comparison ground" glossary line of client_report and
areas.MINE_KIND_LABEL are read from here, so the default study's reports are
byte-identical to those before the registry (Gate 0(b), the golden builds of
tests/golden_harness.py).

Rules of this module:
  - pure: frozen dataclasses and tuples, standard library only, no I/O, and
    nothing here imports the rest of the package (areas, config and
    report_insights import it);
  - no verdicts, gate results or ceilings: an output names the evidence ids it
    rests on, and evidence.yaml (read by evidence.py) holds what those ids
    showed. Tiers are computed from both by evidence.output_tier;
  - presets only for PRESET_KEYS, applied once when a project is created;
    reference_mode and control_ground_label are never presets.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# Stamped into every new project config by areas.write_project; a config without
# it (written before the registry) is version 0 and reads the default study.
REGISTRY_VERSION = 1
DEFAULT_STUDY = "ground_subsidence"

PRESET_KEYS = ("site_profile", "gnss_validation", "expected_max_rate_mm_yr", "aliasing_diagnostic")
NEVER_PRESET = ("reference_mode", "control_ground_label")

LAYER_ROLES = ("known_movement", "source", "asset", "reference", "event")
LAYER_PRODUCTS = ("relation", "profile", "polygon_peak", "values_at", "history")
OUTPUT_KINDS = ("map", "table", "series", "profile", "flag")
# the spatial scale of the motion an output reports or an evidence entry scored,
# smallest first: a smaller output than its evidence is outside the regime
FOOTPRINTS = ("local", "panel", "compact", "area")
SIGNS = ("down", "up", "east", "any")
COMPONENTS = ("los", "vertical", "east")
# fast regimes: C-band reads them low, so an output inside one is refused
FAST_REGIMES = ("active_longwall", "compact_dome", "cyclic_steam_pad", "diatomite", "sinkhole_edge", "fast_zone")
CANNOT_TAGS = ("physics", "sensor", "geometry", "failed")
NOT_YET_TAGS = ("validation", "producer")
DIRECTIONS = ("one", "both_recommended", "both_required")
BUILDER_STATUS = ("offered", "not_yet")


@dataclass(frozen=True)
class LayerKind:
    key: str
    label: str
    geometry: str                       # polygon | line | point
    config_field: str                   # an existing config field, else "feature_layers"
    role: str                           # LAYER_ROLES
    regime: str = ""                    # "" or one of FAST_REGIMES (refused inside)
    buffer_m: float = 0.0               # lines and points, before any deramp exclusion
    attributes: tuple = ()              # columns read when present, never a form field
    product: str = "relation"           # LAYER_PRODUCTS


@dataclass(frozen=True)
class Output:
    key: str
    label: str
    kind: str                           # OUTPUT_KINDS
    producer: str                       # the existing function that makes it
    quantity: str                       # the quantity evidence.yaml grades
    evidence: tuple = ()                # evidence.yaml ids; () means no evidence (RESEARCH)
    sign: str = "down"                  # SIGNS
    footprint: str = "area"             # FOOTPRINTS
    component: str = "los"              # COMPONENTS
    needs: tuple = ()                   # e.g. "both_directions", "datum_stations>=3"
    client_words: tuple = ()            # ((tier, words), ...); () keeps today's wording
    automatic: bool = True              # made by every products run; False = only when its settings are given
    how: str = ""                       # for automatic=False: what turns it on


@dataclass(frozen=True)
class Cause:
    """One suggested cause: the first rule matching (sign, relation, shape) wins.
    relation: a relation_to_mines key; "near" matches every near_<layer>."""
    sign: str                           # "up" | "down"
    relation: str
    text: str
    shape: str = "*"                    # a trend shape, or "*"


@dataclass(frozen=True)
class Wording:
    feature_noun: tuple                 # ((layer key, noun), ...): "over mapped underground workings"
    nearest_noun: tuple                 # ((layer key, noun), ...): "1.8 km from the nearest mapped mine area"
    source_label: tuple                 # ((layer key, label), ...): mine_layer_sources' credit line
    comparison_ground: str              # glossary text; {ground} = report_insights.ground_word
    correction_ramp: str                # the ramp line of the corrections list
    causes: tuple                       # Cause rules, in order
    fallback_cause: str                 # when no rule matches
    # the defaults below are today's (default study) words, printed unchanged;
    # a study with its own wording replaces them (A3 of PLAN_oil_gas_slopes.md)
    no_layer_sentence: str = "No mine outlines were supplied, so features cannot be related to mining."
    no_layer_feature: str = "No mine outlines were supplied, so it cannot be related to mining."
    no_layer_phrase: str = "no mine outlines were supplied for this project"       # a feature's "where" clause
    unmeasured_or: str = "water, woodland, active workings or fresh spoil"     # "the rest (...) changes too much"
    unmeasured_list: str = "water, woodland, active workings, fresh spoil"     # "Ground that could not be measured (...)"
    known_movement_noun: str = "mapped mine or known movement"               # "more than 0.5 km from any ..."
    figure_kinds: str = "area, feature and mine-ground figures"               # the disclaimer's last sentence
    held_sentence: str = ""                                                   # release.py's study hold (new studies)
    limits: tuple = ()                  # Phase 1a: limitation sentences
    no_prediction: str = ""             # Phase 1a


@dataclass(frozen=True)
class Limits:
    near_km: float                      # "next to" a layer only this close (relation_to_mines)
    on_layer_share: float               # a feature this share on a layer is "on" it (feature_kind)
    horizontal_share: float             # sideways movement as a share of the vertical (the LOS caveat)


@dataclass(frozen=True)
class Study:
    key: str
    label: str
    question: str
    builder_status: str                 # BUILDER_STATUS
    layer_kinds: tuple
    outputs: tuple
    wording: Wording
    limits: Limits
    builder_reason: str = ""            # why not_yet
    cannot: tuple = ()                  # ((text, CANNOT_TAGS), ...)
    not_yet: tuple = ()                 # ((text, NOT_YET_TAGS), ...)
    directions: str = "one"             # DIRECTIONS
    qa_profile: str = ""                # a quality_assessment site profile, "" = from the config
    report_sections: tuple = ()         # () = today's sections
    kmz_folders: tuple = ()             # () = today's folders
    presets: tuple = ()                 # ((PRESET_KEYS key, value), ...), applied once at creation
    grandfathered: bool = False         # D5: computed tiers recorded, not enforced (but the longwall refusal)
    fixed_sentences: tuple = ()         # printed as written on the card (e.g. the response-tier sentence)

    def output(self, key: str) -> Output:
        for o in self.outputs:
            if o.key == key:
                return o
        raise KeyError(f"study {self.key!r} has no output {key!r}")


_COMBINE = "a combined project (both directions), built with the app's combine action"

# ------------------------------------------------------------- the default study
_MINE_LAYERS = (
    LayerKind("underground", "Underground mine workings", "polygon", "underground_mines_geojson", "known_movement"),
    LayerKind("surface", "Surface mines", "polygon", "surface_mines_geojson", "known_movement"),
    LayerKind("other", "Other known ground movement", "polygon", "other_movement_geojson", "known_movement"),
    LayerKind("dated", "Dated mined areas", "polygon", "mined_year_geojson", "known_movement",
              attributes=("mined_year", "mine_name"), product="history"),
)

_SUBSIDENCE_CAUSES = (
    Cause("up", "surface",
          "Rising ground on a surface mine is usually spoil or stockpiles being built up. Radar cannot "
          "follow ground built up by metres, so treat the figure as a sign that this ground changed during "
          "the record, not as a measured rate; the mine's records would show what was placed there."),
    Cause("up", "*",
          "Rising ground is usually spoil or stockpiles being built up, swelling soils, or rebound after "
          "pumping stops."),
    Cause("down", "surface",
          "Backfilled spoil in reclaimed surface mines compacts under its own weight for years, fastest at "
          "first and slowing over time, which matches this record. If the ground was dug out or filled during "
          "the record, radar cannot follow it and the figure is not a settlement rate. The mine's mining and "
          "reclamation dates would settle which it is.", shape="slowing"),
    Cause("down", "surface",
          "Backfilled spoil in reclaimed surface mines compacts under its own weight for years, fastest at "
          "first and slowing over time. If the ground was dug out or filled during the record, radar cannot "
          "follow it and the figure is not a settlement rate. The mine's mining and reclamation dates would "
          "settle which it is."),
    Cause("down", "underground",
          "Subsidence over underground workings usually follows the mining of each panel; the panels' "
          "mining dates would show whether the timing matches."),
    Cause("down", "near",
          "It is beside mapped mining; mining just outside the mapped outlines, spoil, or water-level "
          "changes from nearby operations are possible causes."),
    Cause("down", "other", "It lies on ground already known to move."),
)

GROUND_SUBSIDENCE = Study(
    key="ground_subsidence",
    label="Mining subsidence",
    question="Is ground over the mines sinking; how fast; since when; is it slowing?",
    builder_status="offered",
    layer_kinds=_MINE_LAYERS,
    outputs=(
        Output("multi_year_rate", "Multi-year rate over report areas", "table", "client_report.compute",
               "multi_year_rate", evidence=("B2",)),
        Output("multi_year_rate_vertical", "Multi-year vertical rate maps", "map", "deliverables.export_all",
               "vertical_rate", evidence=("B2", "B1_B3"), component="vertical"),
        Output("multi_year_total", "Multi-year totals", "table", "client_report.compute",
               "multi_year_total", evidence=("B1_B3",)),
        Output("series", "Time series of report areas", "series", "client_report.compute",
               "series", evidence=("B2", "B1_B3")),
        Output("single_year_rate", "Single-year rates", "table", "client_report.compute",
               "single_year_rate", evidence=("B1_B3",)),
        Output("feature_search", "Unmapped features", "table", "report_insights.find_features",
               "multi_year_rate", evidence=("B2",), footprint="local"),
        Output("trend_shape", "Trend-shape words", "table", "report_insights.trend_shape",
               "trend_shape"),
        Output("rising_ground", "Rising-ground wording", "table", "report_insights.find_features",
               "multi_year_rate", evidence=("B2",), sign="up", footprint="local"),
        Output("east_west", "East-west movement (combined project)", "map", "combine.run",
               "east_detection", evidence=("P0.3b",), sign="east", footprint="compact", component="east",
               needs=("both_directions",), automatic=False, how=_COMBINE),
        Output("local_change", "Local change layer", "map", "local_change.run",
               "near_pair_agreement", evidence=("LC",), sign="any", footprint="local", automatic=False,
               how="scripts/run_local_change.py on a finished record"),
    ),
    wording=Wording(
        feature_noun=(("underground", "underground workings"), ("surface", "surface mines"),
                      ("other", "other known movement"), ("mapped", "mine areas")),
        nearest_noun=(("underground", "underground workings"), ("surface", "surface mine"),
                      ("other", "ground known to move"), ("mapped", "mine area")),
        source_label=tuple((L.key, L.label) for L in _MINE_LAYERS),
        comparison_ground=("Ordinary {ground} around the site, away from mapped mines, measured in the same record. "
                           "Every area is compared with it."),
        correction_ramp="long-range tilts from orbits and atmosphere (a planar ramp, fitted away from mapped mines)",
        causes=_SUBSIDENCE_CAUSES,
        fallback_cause=("The mapped mines do not explain it. Groundwater pumping, drainage, soft soils, construction or "
                        "unmapped mining are possible causes; a site visit would settle it."),
    ),
    limits=Limits(near_km=0.5, on_layer_share=0.3, horizontal_share=0.35),
    cannot=(("active longwall subsidence on C-band (evidence A)", "sensor"),
            ("single-year rates (evidence B1_B3)", "failed")),
    not_yet=(("rising ground as a measured rate", "validation"),),
    grandfathered=True,
)

# ------------------------------------------------------------------ new studies
# review_2026-11-07/PLAN.md, "The domains" (sections 2-15), added 8 Nov 2026.
# Their tiers come from evidence.output_tier: a new study is at most SCREENING
# until a transfer is signed (rule 6), and a study not offered is RESEARCH.
# Their reports still use the default study's wording until each study's
# wording passes Gate 1b, so release.py holds every client report of a study
# other than the default (study_not_released).

def _L(key, label, geometry, field, role, **kw):
    return LayerKind(key, label, geometry, field, role, **kw)


_STABLE = _L("stable_area", "Stable reference area (used when the reference is set to it)", "polygon",
             "reference_stable_geojson", "reference")
_FLANKS = ("single-track vertical on bowl and dome flanks: sideways motion projects with opposite signs on "
           "either side", "geometry")
_EVENTS = "set event_products (and the layer) in the config"
# A1 of PLAN_oil_gas_slopes.md: write_project turns the event products on when the layer is given
_WITH_PADS = "made when the dated pads layer is given"


def _rate(key="multi_year_rate", label="Multi-year rate over report areas", evidence=("B2",), **kw):
    return Output(key, label, "table", "client_report.compute", "multi_year_rate", evidence=evidence, **kw)


def _vertical(label="Multi-year vertical rate map", evidence=("B2", "B1_B3"), **kw):
    return Output("multi_year_rate_vertical", label, "map", "deliverables.export_all", "vertical_rate",
                  evidence=evidence, component="vertical", **kw)


def _series(label="Time series of report areas", **kw):
    return Output("series", label, "series", "client_report.compute", "series", evidence=("B2", "B1_B3"), **kw)


def _profile(label):
    return Output("profile", label, "profile", "events.profile", "multi_year_rate", evidence=("B2",),
                  footprint="local", automatic=False, how=_EVENTS)


def _values_at(label):
    return Output("values_near_points", label, "table", "events.values_at", "multi_year_rate", evidence=("B2",),
                  footprint="local", automatic=False, how="the events command with --points (not in a products run yet)")


def _before_after(label):
    return Output("before_after", label, "table", "events.cumulative_between", "series", evidence=("B2", "B1_B3"),
                  footprint="local", automatic=False, how=_EVENTS)


_FEATURES = Output("feature_search", "Unmapped features", "table", "report_insights.find_features",
                   "multi_year_rate", evidence=("B2",), footprint="local")
_TREND = Output("trend_shape", "Trend-shape words", "table", "report_insights.trend_shape", "trend_shape")
_RISING = Output("rising_ground", "Rising ground (not yet checked)", "table", "report_insights.find_features",
                 "multi_year_rate", evidence=("B2",), sign="up", footprint="local")
_SINGLE_YEAR = Output("single_year_rate", "Single-year rates", "table", "client_report.compute",
                      "single_year_rate", evidence=("B1_B3",))
_NO_ABSOLUTE = ("absolute rates without GNSS: InSAR is relative and the whole area may move", "physics")


def _new(key, label, question, layers, outputs, wording=None, **kw):
    return Study(key=key, label=label, question=question, layer_kinds=layers, outputs=outputs,
                 wording=wording or GROUND_SUBSIDENCE.wording, limits=GROUND_SUBSIDENCE.limits, **kw)


BASIN_SUBSIDENCE = _new(
    "basin_subsidence", "Groundwater and basin subsidence",
    "Is the town, suburb or farmland over a pumped aquifer sinking; how fast; since when; is it slowing?",
    (_L("pumping_area", "Known pumping or fill areas", "polygon", "other_movement_geojson", "known_movement"),
     _STABLE,
     _L("dated_change", "Dated changes (e.g. a pumping change)", "polygon", "event_polygons", "event",
        attributes=("start", "end"), product="polygon_peak")),
    (_rate(evidence=("B2", "B1_B3")), _vertical(), Output("multi_year_total", "Multi-year totals", "table",
     "client_report.compute", "multi_year_total", evidence=("B1_B3",)), _series(), _SINGLE_YEAR,
     _FEATURES, _TREND, _RISING),
    builder_status="offered",
    cannot=(_NO_ABSOLUTE, ("single-year rates (evidence B1_B3)", "failed")),
    not_yet=(("the seasonal aquifer swing: the official route has no seasonal term", "producer"),
             ("rain-fed cropland and dense urban ground: no evidence on that ground", "validation"),
             ("rates above the validated envelope (evidence B2)", "validation")),
    qa_profile="groundwater_basin",
    presets=(("site_profile", "groundwater_basin"),),
)

INFRASTRUCTURE = _new(
    "infrastructure", "Buildings and infrastructure",
    "Is my district, corridor, airport, dam or levee on ground moving faster than the region round it; "
    "did it start or stop?",
    (_L("asset_line", "Asset lines (roads, rail, pipelines, levees)", "line", "event_lines", "asset",
        buffer_m=100.0, product="profile"),
     _L("asset_footprint", "Asset footprints", "polygon", "event_polygons", "asset", attributes=("start", "end"),
        product="polygon_peak"),
     _L("fill_area", "Fill or construction areas", "polygon", "other_movement_geojson", "known_movement"),
     _STABLE),
    (_rate(), _vertical(), _series(), _profile("Profile along asset lines"),
     _values_at("Values near asset points"), _before_after("Before and after a construction date"), _FEATURES,
     _RISING, _TREND),
    builder_status="offered",
    cannot=(("integrity or safety: radar measures displacement, not condition", "physics"),
            ("one building, pier or slab: smaller than the radar's support", "sensor"),
            ("along-axis motion of north-south bridges", "geometry"),
            ("a line threshold finer than the validated rate error (evidence B2)", "sensor")),
    not_yet=(("dense urban ground: the urban validation site (V1) has not run", "validation"),
             ("thermal motion of spans: no thermal term", "producer")),
    directions="both_recommended",
)

OIL_GAS_WORDING = replace(
    GROUND_SUBSIDENCE.wording,
    comparison_ground=("Ordinary {ground} around the site, away from the mapped pads, fast zones and other known "
                       "movement, measured in the same record. Every area is compared with it."),
    correction_ramp=("long-range tilts from orbits and atmosphere (a planar ramp, fitted away from the mapped pads "
                     "and fast zones)"),
    causes=(Cause("up", "*", "Rising ground has not yet been checked as a measured rate on this route; injection, "
                             "swelling soils or rebound after pumping stops are possible causes."),),
    fallback_cause=("Its cause is not known from these data. Fluid withdrawal, groundwater pumping, soft soils or "
                    "construction are possible causes; field records or a site visit would settle it."),
    no_layer_sentence=("Features are not related to field operations here; values near the listed wells and pads "
                       "are given in their own section."),
    no_layer_feature="It is not related to field operations here; see the values near the listed wells and pads.",
    no_layer_phrase="not related to field operations here",
    unmeasured_or="water, woodland, fresh earthworks or active pads",
    unmeasured_list="water, woodland, fresh earthworks, active pads",
    known_movement_noun="mapped pad, fast zone or known movement",
    figure_kinds="area, feature and well figures",
    held_sentence=("This report answers the oil and gas fields study. It is a screening report: it is sent only "
                   "after the reviewer signs it off, and its figures show where to look, not survey values."),
)

OIL_GAS = _new(
    "oil_gas", "Oil and gas fields",
    "Is the field sinking; where is the bowl; did the rate change after a shut-in or another operational date; "
    "which listed wells sit on moving ground?",
    (_L("wells", "Wells (status, plug date, type by column)", "point", "feature_layers", "source",
        buffer_m=100.0, attributes=("status", "plug_date", "type"), product="values_at"),
     _L("pads", "Dated pads or patterns", "polygon", "event_polygons", "source", attributes=("start", "end"),
        product="polygon_peak"),
     _L("faults", "Faults", "line", "feature_layers", "asset", product="profile"),
     _L("fast_zone", "Fast zones (diatomite, cyclic steam)", "polygon", "fast_zone_geojson", "known_movement",
        regime="fast_zone"),
     _STABLE),
    (_vertical(label="Vertical map in a GNSS frame", needs=("datum_stations>=3",)), _rate(label="Per-lease rates"),
     _series(), Output("pad_peak", "Series and peak change at dated pads", "table", "events.panel_max", "series",
                       evidence=("B2", "B1_B3"), footprint="local", automatic=False, how=_WITH_PADS),
     replace(_before_after("Before and after an operational date"), how=_WITH_PADS),
     replace(_values_at("Values near listed wells (not a per-well rate)"), how="made when the wells layer is given"),
     _FEATURES, _RISING, _TREND),
    builder_status="offered",
    cannot=(("which well, where wells are closer than the radar's support", "sensor"),
            ("diatomite or cyclic-steam motion: faster than C-band can follow", "sensor"),
            ("casing-shear prediction", "physics"),
            ("that a well is sound: deep leaks may show nothing at the surface, and no uplift proves nothing",
             "physics"),
            _FLANKS),
    not_yet=(("EOR heave (steam, CO2, waterflood): rising ground waits for the uplift check (Gate 2)", "validation"),
             ("uplift bullseyes over shallow leaks or plugged wells (Gate 2)", "validation"),
             ("'stopped moving' after a shut-in: needs the detection-power check (Gate 2b)", "validation")),
    directions="both_recommended",
    wording=OIL_GAS_WORDING,
)

CCUS_BASELINE = _new(
    "ccus_baseline", "CCUS baseline (CO2 storage, CO2-EOR)",
    "Was the site moving before injection; what is the smallest change you could see?",
    (_L("area_of_review", "Area of review", "polygon", "event_polygons", "source"),
     _L("injectors", "Injectors (start date by column)", "point", "feature_layers", "source",
        attributes=("start_date",), product="values_at"),
     _L("legacy_wells", "Legacy wells", "point", "feature_layers", "asset", product="values_at"),
     _L("predicted_footprint", "Predicted footprint", "polygon", "feature_layers", "reference"),
     _STABLE),
    (_rate(label="Pre-injection baseline rate"), _vertical(label="Pre-injection baseline vertical map"),
     _series(label="Pre-injection series")),
    builder_status="offered",
    cannot=(("containment proof: deep storage predicts surface change below the noise floor", "physics"),
            ("leakage at depth", "physics"),
            ("an annual figure (evidence B1_B3)", "failed")),
    not_yet=(("change since injection began (Gate 2)", "validation"),
             ("'no change beyond X' (Gate 2b)", "validation"),
             ("values in fixed buffers round the injectors: no producer yet", "producer"),
             ("turning off the unmapped-feature table, which the report still shows", "producer"),
             ("farmland coherence on C-band; L-band would help once NISAR is re-tested", "validation")),
    directions="both_recommended",
)

SALT_CAVERNS = _new(
    "salt_caverns", "Salt cavern storage",
    "How fast is the bowl growing; near which caverns; do the figures match our benchmarks?",
    (_L("caverns", "Cavern outlines", "polygon", "event_polygons", "source", product="polygon_peak"),
     _L("dome", "Dome outline", "polygon", "other_movement_geojson", "known_movement"),
     _L("benchmarks", "Benchmarks", "point", "feature_layers", "reference", product="values_at"),
     _L("facility_lines", "Facility lines", "line", "event_lines", "asset", product="profile"),
     _L("sinkhole_zone", "Sinkhole-prone ground", "polygon", "fast_zone_geojson", "known_movement",
        regime="sinkhole_edge"),
     _L("off_dome_reference", "Off-dome reference", "polygon", "reference_stable_geojson", "reference")),
    (_vertical(label="Vertical map against the off-dome reference", needs=("reference_mode: stable_polygon",)),
     _rate(label="Bowl rate"),
     Output("multi_year_total", "Bowl totals", "table", "client_report.compute", "multi_year_total",
            evidence=("B1_B3",)),
     Output("cavern_peak", "Values near each cavern", "table", "events.panel_max", "series",
            evidence=("B2", "B1_B3"), footprint="local", automatic=False, how=_EVENTS),
     _series(), _FEATURES, _RISING),
    builder_status="offered",
    cannot=(("which cavern, where caverns are closer than about twice the radar's support", "sensor"),
            ("a sinkhole or collapse warning: fast, compact and with little precursor", "sensor"),
            ("convergence volume from one direction", "geometry"),
            _FLANKS),
    not_yet=(("storage-cycle motion and its lag: no seasonal term", "producer"),
             ("a benchmark comparison table: no producer yet", "producer"),
             ("coherence on Gulf wetland", "validation"),
             ("convergence from both directions with a model", "producer")),
    directions="both_recommended",
)

COASTAL = _new(
    "coastal", "Coastal defence and sea-level rise",
    "How fast is the land under my port, levee or shoreline sinking in a GNSS frame?",
    (_L("defence_lines", "Defence lines (levees, sea walls)", "line", "event_lines", "asset", buffer_m=100.0,
        product="profile"),
     _L("tide_gauges", "Tide gauges", "point", "feature_layers", "reference", product="values_at"),
     _L("fill_area", "Fill areas", "polygon", "other_movement_geojson", "known_movement"),
     _STABLE),
    (_vertical(label="Vertical land motion in a GNSS frame", needs=("datum_stations>=3",)),
     _rate(), _profile("Profile along defence lines"),
     Output("up_combined", "Up map from both directions", "map", "combine.run", "vertical_rate",
            evidence=("B2", "B1_B3"), component="vertical", needs=("both_directions",), automatic=False,
            how=_COMBINE),
     _FEATURES, _RISING),
    builder_status="offered",
    cannot=(("sea-level projections or freeboard: ocean and climate models, outside InSAR", "physics"),
            ("tide-gauge representativeness: no gauge tie", "physics"),
            _NO_ABSOLUTE,
            ("narrow crests, which sit inside one radar value", "sensor"),
            ("water and marsh, which lose coherence", "sensor")),
    not_yet=(("client-grade rates (about a millimetre a year): needs the second-direction validation (V2)",
              "validation"),
             ("the seasonal aquifer swing: no seasonal term", "producer"),
             ("levee and embankment slope creep: needs slope products", "producer"),
             ("faces under trees while the wooded mask is on", "producer")),
    directions="both_required",
)


def _research(key, label, question, why, layers=(), outputs=(), **kw):
    return _new(key, label, question, layers, outputs, builder_status="not_yet", builder_reason=why, **kw)


SLOPES = _research(
    "slopes", "Slopes and landslides", "Which slopes are moving; how fast downslope; which can the satellite not see?",
    "missing producers: downslope projection, visibility layer, summer stacks",
    cannot=(("north-south motion", "geometry"), ("motion faster than C-band can follow", "sensor"),
            ("a sudden-failure warning", "physics"), ("shadow and layover", "geometry")),
    not_yet=(("forested slopes", "validation"), ("downslope velocity", "producer")))
PERMAFROST = _research(
    "permafrost", "Permafrost", "How much does ground under my road or pad settle each thaw season?",
    "missing producers: summer-only stacks, per-season reference, snow mask",
    cannot=(("the net trend across winters on C-band: snow", "sensor"), ("boreal forest", "sensor"),
            ("thaw depth or ice content: needs a model", "physics")),
    not_yet=(("the seasonal cycle, read as noise today", "producer"),))
DISPOSAL = _research(
    "saltwater_disposal", "Saltwater disposal injection",
    "Is ground round my disposal wells rising; since when; did it slow after a curtailment?",
    "its headline is rising ground, which waits for the uplift check (Gate 2)",
    layers=(_L("wells", "Disposal wells", "point", "feature_layers", "source", product="values_at"),
            _L("compact_dome", "Compact domes", "polygon", "fast_zone_geojson", "known_movement",
               regime="compact_dome")),
    outputs=(_vertical(label="Basin vertical map in a GNSS frame", needs=("datum_stations>=3",), sign="up"),),
    cannot=(("pressure at depth", "physics"), ("which operator caused a dome", "physics"),
            ("compact steep-gradient domes", "sensor"), ("'quiet means safe'", "physics")),
    not_yet=(("any rising number before Gate 2", "validation"),),
    directions="both_required")
INDUCED_SEISMICITY = _research(
    "induced_seismicity", "Induced-seismicity risk (deformation side)",
    "Where is the deformation relative to wells, faults and response radii?",
    "no fault-slip check exists; deformation is context only",
    cannot=(("seismic hazard", "physics"), ("any forecast", "physics"),
            ("small earthquakes, below about magnitude five", "sensor"),
            ("placing a well in a response tier: levels are set by earthquake magnitude", "physics")),
    fixed_sentences=("Regulatory response levels are set by earthquake magnitude; this map is context and does "
                     "not place a well in any response tier.",),
    directions="both_required")
EARTHQUAKES = _research(
    "earthquakes", "Earthquakes", "Did the ground at our site move after the quake; how much; which way?",
    "missing producers: co-event pairs, step term, offset tracking",
    cannot=(("timing or forecast", "physics"), ("small earthquakes, below about magnitude five", "sensor"),
            ("near-field metres", "sensor"),
            ("north-south strike-slip", "geometry"), ("rapid response: a full record takes days", "sensor")))
VOLCANOES = _research(
    "volcanoes", "Volcanoes", "Is it inflating; how fast; speeding up; magmatic or atmosphere?",
    "missing producers: topography-correlation check, changepoint, source model; observatory levels take precedence",
    cannot=(("an eruption forecast", "physics"), ("fast eruptive motion", "sensor"),
            ("a source from one direction", "geometry")),
    not_yet=(("forested volcanoes", "validation"),))

STUDIES = (GROUND_SUBSIDENCE, BASIN_SUBSIDENCE, INFRASTRUCTURE, OIL_GAS, CCUS_BASELINE, SALT_CAVERNS, COASTAL,
           SLOPES, PERMAFROST, DISPOSAL, INDUCED_SEISMICITY, EARTHQUAKES, VOLCANOES)
STUDY_KEYS = tuple(s.key for s in STUDIES)
DEFAULT = GROUND_SUBSIDENCE


def study(key: str | None) -> Study:
    """The registered study for a config's `study` key ("" or None = the default)."""
    key = (key or DEFAULT_STUDY).strip().lower()
    for s in STUDIES:
        if s.key == key:
            return s
    raise KeyError(f"no study {key!r}; known: {list(STUDY_KEYS)}")


def study_of(cfg) -> Study:
    """The study a project config asks for; a config written before the
    registry (no `study`) gets the default."""
    return study(getattr(cfg, "study", "") or DEFAULT_STUDY)


def cause_text(s: Study, relation_key: str, sign: int, shape: str | None) -> str:
    """The suggested cause for a feature: report_insights._cause_base."""
    want = "up" if sign > 0 else "down"
    for c in s.wording.causes:
        if c.sign != want:
            continue
        if c.relation == "near":
            hit = relation_key.startswith("near_")
        else:
            hit = c.relation == "*" or c.relation == relation_key
        if hit and (c.shape == "*" or c.shape == shape):
            return c.text
    return s.wording.fallback_cause


def wording_of(cfg_or_record) -> Wording:
    """The wording a report uses: the study of a config, or of a report record
    (its "study" key; a record without one is the default study's)."""
    if isinstance(cfg_or_record, dict):
        key = str(cfg_or_record.get("study") or DEFAULT_STUDY).strip().lower()
        try:
            return study(key).wording
        except KeyError:
            return DEFAULT.wording
    return study_of(cfg_or_record).wording


def feature_layer_kinds(s: Study) -> tuple:
    """The layer kinds a project gives through `feature_layers` ("kind:path")."""
    return tuple(L.key for L in s.layer_kinds if L.config_field == "feature_layers")


def split_feature_layer(spec: str) -> tuple:
    """'wells:/path/wells.geojson' -> ('wells', '/path/wells.geojson')."""
    kind, sep, path = str(spec).partition(":")
    if not sep or not kind.strip() or not path.strip():
        raise ValueError(f"{spec!r}: expected 'kind:path'")
    return kind.strip().lower(), path.strip()


def layer_files(s: Study, cfg) -> list:
    """[(layer kind, path)] the project gives for each of the study's layer kinds."""
    out = []
    given = {}
    for spec in getattr(cfg, "feature_layers", ()) or ():
        k, p = split_feature_layer(spec)
        given.setdefault(k, []).append(p)
    for L in s.layer_kinds:
        if L.config_field == "feature_layers":
            out += [(L.key, p) for p in given.get(L.key, [])]
        else:
            p = (getattr(cfg, L.config_field, "") or "").strip()
            if p:
                out.append((L.key, p))
    return out

