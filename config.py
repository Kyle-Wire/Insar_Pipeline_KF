"""Configuration handling for the InSAR pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
import collections.abc
import datetime as _dt
import json
import logging
import os
import re
import typing

import yaml

from . import domains

log = logging.getLogger(__name__)

# Default radius (km) for anchoring rates to continuous GNSS (gnss_tie.py).
# One number for the config default, the builder (areas.write_project), its
# "GNSS within N km" text and the dashboard, which disagreed before (50 vs 40).
GNSS_TIE_RADIUS_KM = 40.0

# What each processing route's numbers are (Kyle, 28 Sep 2026): the phase-linked
# record is the official result; DISP-S1 is screening only; the classic stack
# and the older routes are for comparison. Reports, KMZs and the release gate
# read this instead of testing backend names themselves.
ROUTE_TAG = {"miaplpy": "official", "disp": "screening", "stack": "comparison",
             "local": "comparison", "hyp3": "comparison", "nisar": "comparison"}


def route_tag(cfg) -> str:
    """'official' | 'screening' | 'comparison' for this config's route."""
    return ROUTE_TAG.get(str(getattr(cfg, "backend", "")).lower(), "comparison")


# The satellite each route's numbers come from, for client wording and the
# data credits; every route but nisar reads Sentinel-1 (26 Oct 2026: the
# NISAR route's products said "Sentinel-1" and carried the Copernicus notice).
SENSOR = {"nisar": "NISAR"}


def sensor_name(cfg_or_backend) -> str:
    """'NISAR' or 'Sentinel-1' for a config, or a backend name."""
    b = cfg_or_backend if isinstance(cfg_or_backend, str) else getattr(cfg_or_backend, "backend", "")
    return SENSOR.get(str(b or "").lower(), "Sentinel-1")


def is_screening(cfg) -> bool:
    """True on the DISP-S1 route, whose numbers are screening only."""
    return str(getattr(cfg, "backend", "")).lower() == "disp"


# What a run is for (config `purpose`, V3 8 Oct 2026). "client" is every run
# before that date: the products stage ends with the client report and the
# release status. "experiment" is a validation run: deliverables, quality
# assessment, figures and the validation harness, never a client report.
PURPOSES = ("client", "experiment")
# The kinds of ground the quality assessment knows (config `site_profile`;
# the table of what each one explains is quality_assessment.SITE_PROFILES).
# "farmland" is the default and today's behaviour.
SITE_PROFILES = ("farmland", "urban", "mixed", "mine_longwall", "groundwater_basin")


def is_experiment(cfg) -> bool:
    """True for a validation run (purpose: experiment): no client report,
    release status or package is ever built for it."""
    return str(getattr(cfg, "purpose", "client") or "client").strip().lower() == "experiment"


def site_profile(cfg) -> str:
    """The config's site_profile, lower case; "farmland" when unset."""
    return str(getattr(cfg, "site_profile", "farmland") or "farmland").strip().lower()


def _sized(name: str):
    """Default pool size for this machine (settings.sized_parallelism): pool
    sizes change speed, never numbers, so a config that does not set one gets
    what this machine can hold (12/3/24 workers, 16 GB MintPy on the 23 GB /
    24-core machine they were measured on) instead of OOM on a 16 GB laptop."""
    def make():
        try:
            from . import settings
            r = settings.machine_resources()
            return settings.sized_parallelism(r["mem_gb"], r["cores"])[name]
        except Exception:                                    # noqa: BLE001 - never block a config load
            return {"stack_num_proc": 12, "stack_num_proc_topo": 3, "stack_num_proc_unwrap": 24,
                    "mintpy_max_memory_gb": 16.0}[name]
    return field(default_factory=make)


@dataclass
class PipelineConfig:
    project_dir: Path
    bbox: Sequence[float]              # (min_lon, min_lat, max_lon, max_lat)
    start_date: str                    # "YYYY-MM-DD"
    end_date: str                      # "YYYY-MM-DD"
    track: Optional[int] = None        # relative orbit number; None = search all
    orbit_direction: str = "descending"  # "ascending" or "descending"
    dem_path: Optional[Path] = None    # if None, auto-download a Copernicus DEM
    max_perp_baseline: float = 200.0   # meters — see note in pair_selection.py
    max_temp_baseline: int = 48        # days
    weather_model: str = "ERA5"        # for RAiDER
    raider_env: str = ""               # conda env to run raider.py in; empty = current
    skip_existing: bool = True         # resume-friendly: skip steps already done
    do_esd: bool = True                # Enhanced Spectral Diversity azimuth refinement
    esd_coherence_threshold: float = 0.7   # ISCE2 default is 0.85. MEASURED: at
                                            # 0.85 five of 87 pairs (all 36-48
                                            # day) failed "no points left for
                                            # reliable ESD" over farmland; 0.7
                                            # gave 87/87.
    cleanup_intermediates: bool = True  # delete per-pair scratch once merged/ is written
    exclude_pairs: tuple = ()          # pair names to drop from the time series

    # --- Backend ---
    # "local"  topsApp per pair. Full parameter control; slow; use for
    #          testing and anything needing non-default processing.
    # "hyp3"   Sentinel-1 interferograms ordered from ASF HyP3. Use for
    #          client work - faster and far less to maintain.
    # "disp"   OPERA DISP-S1: JPL's ready-made displacement time series,
    #          30 m, North America (a few frames elsewhere, e.g. Sydney),
    #          mid-2016 on. For client work consider
    #          OPERA's own opera-utils (rebase_reference + MintPy export)
    #          instead of this backend's reader - same data, maintained by
    #          the product authors.
    #          No ordering, no credits, no inversion.
    # "nisar"  Pre-made NISAR L2 GUNW products from ASF. No ordering step.
    #          L-band, left-looking: process separately from Sentinel-1,
    #          never in the same inversion.
    backend: str = "local"
    hyp3_looks: str = "20x4"           # HyP3 range x azimuth looks

    # --- NISAR GUNW route (backend nisar; M2, 8 Oct 2026; nisar_backend.py) ---
    # Defaults are nisar_backend.SETTINGS (a test keeps the two equal).
    # nisar_allow_beta: also search the BETA collection (never mixed with
    # PROVISIONAL in one stack). nisar_polarization: "" = the product's own
    # co-pol (HH, else VV). nisar_ionosphere: "split_spectrum" (the product's
    # screen, subtracted by MintPy) or "no". nisar_chain_gap: a broken chain
    # keeps its "longest_segment", or "fail" stops. nisar_fetch: "window"
    # (read the box over HTTPS) or "download" (whole granule, cropped on
    # arrival, then deleted). nisar_posting_m: the common lattice (m).
    # nisar_crop_margin_deg: around the AOI. nisar_min_valid_ratio: a pair
    # below this share of the group's median valid area is dropped.
    nisar_allow_beta: bool = False
    nisar_polarization: str = ""
    nisar_ionosphere: str = "split_spectrum"
    nisar_chain_gap: str = "longest_segment"
    nisar_fetch: str = "window"
    nisar_posting_m: float = 15.0
    nisar_crop_margin_deg: float = 0.02
    nisar_min_valid_ratio: float = 0.5
    # The phase-linked NISAR route (nisar_pl.py, 28 Oct 2026): the GUNW route
    # with each crop's unwrapped phase replaced by dolphin's phase linking of
    # the 5 m L2 GSLC stack, output every nisar_posting_m / 5 pixels.
    # nisar_pl_window: the phase-linking window (pixels, odd); 11 chosen on
    # the first test site by the plan's pre-registered rule (Gate 2).
    nisar_phase_linked: bool = False
    nisar_pl_window: int = 11

    # --- Network ---
    # Each acquisition is paired with this many later ones. 1 gives a
    # sequential chain, which has no closed loops - temporal coherence then
    # reads 1.000 everywhere by construction and tells you nothing. A
    # minimum redundancy of 3 is a common published choice and is what
    # makes the quality product meaningful.
    n_neighbors: int = 3

    # --- Extent ---
    # Crop every layer to the AOI before stacking. HyP3 multi-burst
    # products do not all come out the same size - a single run can span
    # several row/column counts depending on where each date's bursts
    # fall - and MintPy fails outright when the geometry file and the
    # interferogram stack differ by even one row. Subsetting forces a
    # common grid, and trims the products to the area of interest, which
    # also speeds up the inversion.
    #
    # Leave True unless you deliberately want the full product extent.
    subset_to_aoi: bool = True

    # --- Reference point ---
    # Every displacement is relative to this pixel. MintPy picks one
    # automatically when this is unset, which is fine for testing and not
    # acceptable for a deliverable: change the reference and every number
    # changes. Set it somewhere argued to be stable, as "lat,lon".
    reference_lalo: str = ""
    # Other stable-ground candidates (from stack_qc.select_reference_candidates);
    # the quality assessment reports each headline figure re-referenced to
    # every one of them, so the reader sees how much the choice matters.
    reference_candidates_lalo: tuple = ()
    # --- Phase linking (backend "miaplpy") ---
    # Distributed-scatterer processing (MiaplPy): one consistent phase history
    # per cell estimated from the whole coherence matrix instead of from a
    # network of pairs, which is what removes the short-baseline bias at its
    # source. Runs after topsStack coregistration; needs the coregistered
    # merged SLCs on disk (stack_virtual_merge False, coreg not cleaned).
    miaplpy_env: str = "miaplpy"                     # conda env with MiaplPy + MintPy + ISCE2
    miaplpy_ministack_size: int = 10                 # images per mini-stack in sequential phase linking
    miaplpy_range_window: int = 15                   # SHP search window (full-res pixels)
    miaplpy_azimuth_window: int = 15
    # Pixels not phase-linked (15 Oct 2026, landcover.skip_on_points): "water"
    # (default; permanent water ~25 m in from the shore), "water_and_dense_woods"
    # (also woods ~50 m in from their edge: opt-in, for heavily forested areas),
    # or "none". Needs the land-cover map; applied only when linking starts.
    miaplpy_skip: str = "water"
    miaplpy_phase_linking_method: str = "sequential_EMI"   # EVD | EMI | PTA | sequential_* | SBW
    miaplpy_num_processors: int = 8                  # parallel patches in phase linking
    # Phase linking on the GPU (11 Nov 2026; pl_gpu.py, review_2026-11-10/PLAN.md
    # Phase 2): the same per-pixel algorithm as our patched MiaplPy build,
    # batched with JAX in miaplpy_gpu_env; it writes MiaplPy's own patch files,
    # so concatenation and everything after are unchanged, and MiaplPy on the
    # CPU then links any patch the GPU did not. "auto" (default since 11 Nov 2026,
    # Kyle: the GPU when one is usable and its self-check against MiaplPy on the
    # CPU passes, else the CPU; a GPU failure leaves the patches it did not
    # finish to the CPU), "on" (a GPU is required), "off". Only sequential_EMI/EMI.
    # The first real run with it is Gate 2(b) of the plan (MintPy velocity and
    # GNSS tie against a CPU-linked record).
    miaplpy_gpu: str = "auto"
    miaplpy_gpu_env: str = "dolphin"                 # conda env with CUDA JAX (the NISAR route's)
    miaplpy_gpu_memory_gb: float = 4.0               # device memory one batch may use (Windows shares the GPU)
    # MiaplPy's --time_lag. In the PATCHED library (patches/miaplpy, built by build_miaplpy.sh)
    # (25 Sep 2026) it is the window OVERLAP of the sequential estimator: each
    # mini-stack of miaplpy_ministack_size images is solved together with the
    # previous miaplpy_time_lag images and chained to the previous window through
    # them, which is what keeps one datum across seasons on ground whose
    # compressed images do not stay coherent (measured 0.13-0.35 here). Upstream
    # MiaplPy uses the same flag only as the coherence band of its SBW method.
    miaplpy_time_lag: int = 3
    # Interferogram network exported from the linked phase series: N nearest
    # connections per date (12/24/36 days at N = 3), which unwrap well and give
    # MintPy closed loops for the phase-closure correction. 0 exports a
    # single-reference star (one pair per date, no loops) - poor over farmland,
    # where a five-year linked pair is as decorrelated as a classic one.
    miaplpy_network_connections: int = 3
    # Disk on the phase-linked route (all measured on the 2023 test, 18 dates):
    # a real merged SLC is 0.84 GB per date and MiaplPy's own stack, patches
    # and phase series add ~0.8 GB per date on top. Each product is deleted
    # as soon as the next one has been verified, so a decade-long record
    # peaks near the coregistration peak instead of twice it.
    miaplpy_delete_merged_slc_after_load: bool = True     # secondary <date>.slc.full once slcStack.h5 holds finite data for every date
    miaplpy_delete_patches_after_concatenate: bool = True  # inverted/PATCHES once phase_series.h5 exists
    # 11 Nov 2026: each merged <date>.slc.full is rewritten as a sparse file
    # holding only the phase-linking window (miaplpy_backend.sparsify_merged_slc):
    # same name, size and header, zeros outside the window, ~crop_fraction of the
    # disk. A later change of area that reaches outside the window is refused
    # before loading (re-merge those dates). Off = whole-frame merged SLCs.
    miaplpy_sparse_merged_slc: bool = True
    # MEASURED 24 Sep 2026 on the 273-date Site A stack: MiaplPy's
    # sequential_EMI anchors every mini-stack to the compressed image of the
    # FIRST mini-stack, so over decorrelating farmland each mini-stack carries
    # a random per-pixel datum offset (block coherence 0.9 inside a mini-stack,
    # 0.4 across any boundary). Re-running phase linking with another method
    # needs slcStack.h5; keeping it (0.28 GB per date) is far cheaper than
    # re-coregistering. Delete it only once the export's datum check passes.
    miaplpy_delete_slc_stack_after_concatenate: bool = False
    # --- Fast motion: aliasing (F2, F1, F3; 8 Oct 2026; aliasing.py has the details) ---
    # F2: before unwrapping, each exported pair is checked for aliasing (wrapped
    # phase gradient above half a cycle per look cell, coherence holes);
    # written to <project>/aliasing/ and, by aliasing.export_products, to
    # deliverables/quality_aliasing_risk.tif + aliasing_risk.json. Changes no
    # number. "auto" (default) = on for purpose experiment, site_profile
    # mine_longwall or a config with fast zones; "on"; "off".
    aliasing_diagnostic: object = "auto"
    # The deforming polygons its statistics and the F1 preflight use; empty =
    # deramp_exclude_geojson. Properties max_rate_mm_yr / halfwidth_m per
    # polygon override the two settings below.
    aliasing_zone_geojson: str = ""
    # F1 preflight in the run plan: the peak subsidence rate expected (mm/yr,
    # vertical; for a longwall the rate while the face passes, not the yearly
    # average; 0 = not set) and the trough half-width it builds over (m;
    # 0 = half the narrow side of each zone polygon).
    expected_max_rate_mm_yr: float = 0.0
    expected_trough_halfwidth_m: float = 0.0
    # F3: inside these polygons only each date's fast_zone_connections nearest
    # pairs reach MintPy (the longer ones alias first); originals kept beside
    # each pair and put back when this changes. Empty = off.
    fast_zone_geojson: str = ""
    fast_zone_connections: int = 1
    # F3 / A1b: re-link ANOTHER project's coregistered stack and SLC stack
    # (that project's folder) with this config's phase-linking windows, into
    # this project's own folder. Start with 'Phase linking onward'. Empty = off.
    miaplpy_relink_from: str = ""
    # Where the delivered time series lives when it is not <project>/mintpy.
    # Empty = default (every route now writes there).
    time_series_dir: str = ""
    # Named sites to report in full (series, rate, cumulative, a vendor-style
    # 126-point cluster) as "name:S,W,N,E" in degrees - e.g. the parcel a
    # monitoring contractor reports on, so the comparison is on the same ground.
    focus_sites_lalo: tuple = ()
    # The areas the CLIENT REPORT gives subsidence figures for, in order:
    # "Name:full" (the whole study area), "Name:S,W,N,E" (a box, degrees) or
    # "Name:/path/to/polygon.geojson". Empty = the whole study area plus
    # every focus site.
    report_areas: tuple = ()
    # Anchor absolute rates to the continuous GNSS stations within this
    # radius (km) of the AOI centre that overlap the record by >= 3 years
    # (gnss_tie.py). 0 disables the tie.
    gnss_tie_radius_km: float = GNSS_TIE_RADIUS_KM
    # What the report calls the comparison ground every area is measured
    # against (the class-matched null): "farmland" over the Indiana coalfield;
    # "pasture", "forest", "open ground" elsewhere.
    control_ground_label: str = "farmland"
    # Basemap under the client map: "usgs" = USGS The National Map topo tiles
    # (public domain, cached in <project>/context/basemap); "" or "none" =
    # no basemap (outside USGS coverage, or an operator that does not want one).
    report_basemap: str = "usgs"
    # Where each mine layer came from, one "Label: source, vintage" string per
    # layer (the builder writes them; the report prints them under the map).
    mine_layer_sources: tuple = ()
    # A REISSUE of a report the client already has (decision M3, 7 Oct 2026).
    # Leave empty for a first issue. For a reissue give
    #   report_revision:
    #     number: 1
    #     supersedes: "2026-09-28"      # the date of the issue it replaces
    #     changes:                      # what moved and why, in plain sentences
    #       - "..."
    # The page, the PDF and the KMZ print "Revision 1, replacing the report
    # issued 28 Sep 2026" and the changes (report_revision() below). The
    # release gate holds a report with no note once the project has an earlier
    # issue (deliverables/final_*, or a final package in deliverables/releases/:
    # release 'reissue_without_note'), so a client holding the first report is
    # never handed different figures with nothing to explain them.
    report_revision: dict = field(default_factory=dict)
    # OPTIONAL dated mine polygons for any site (panel_history): a GeoJSON whose
    # features carry the year each panel/pit was mined or reclaimed in
    # mined_year_field and its name in mined_name_field. Empty = no panel
    # timeline (nothing site-specific is hard-coded any more).
    mined_year_geojson: str = ""
    mined_year_field: str = ""
    mined_name_field: str = ""
    # OPTIONAL land-cover raster (NLCD-like GeoTIFF of class codes) used as a
    # second matching axis for the comparison ground, so cells are compared
    # with ground of the same coherence class AND cover. Empty = coherence only.
    landcover_raster: str = ""
    # Label wooded cells (15 Oct 2026): C-band sees canopy, not the ground. With a
    # land-cover map (landcover_raster, or ESA WorldCover fetched once into
    # context/) trees are left out of the comparison ground, each feature and
    # area states its wooded share, and the KMZ has a Tree cover layer.
    landcover_label_wooded: bool = True
    # Leave wooded cells out of the delivered products (21 Oct 2026): maps,
    # rasters, points, KMZ, viewer and area figures. Their relative values
    # read as rising (one project: +2.5 in median over 10 years, 85% of wooded
    # cells positive), because woods carry less of the farmland bias that the
    # comparison removes. The quality assessment still uses every cell. Needs
    # landcover_label_wooded and a land-cover map; without one nothing changes.
    mask_wooded: bool = True

    # --- What the run is for, and the kind of ground (V3, 8 Oct 2026) ---
    # "client" (default): the products stage ends with the GNSS tie, the client
    # report and KMZ and the release status, as it always has.
    # "experiment": a validation run, not a client project. The products stage
    # runs the deliverables, the quality assessment and the figures, then the
    # validation harness when validation_protocol is set, and builds no client
    # report, no client KMZ and no release status; client files an earlier run
    # left in deliverables/ are moved to deliverables/experiment_not_for_clients/.
    # The dashboard's report rebuild refuses too (pipeline.run_client_report),
    # so after a products run there is no client report to sign off or package.
    purpose: str = "client"
    # The kind of ground, which says which quality flags the ground itself
    # explains (quality_assessment.SITE_PROFILES). "farmland" (default) is the
    # Indiana behaviour, the only one whose flags have been checked against
    # real records. "urban", "mixed", "mine_longwall" and "groundwater_basin"
    # name the flags the ground explains there but still hold the send rule
    # for them, because the figures' way of coping has only been checked on
    # farmland; "groundwater_basin" also says that figures relative to the far
    # field cancel motion shared by the whole area.
    site_profile: str = "farmland"
    # The study the project answers (domains.py; S0.4, 8 Nov 2026): which
    # outputs and wording its reports use. "ground_subsidence" (default) is the
    # mining study and today's behaviour; a config without it behaves exactly so.
    study: str = domains.DEFAULT_STUDY
    # The study registry version write_project stamped (domains.REGISTRY_VERSION);
    # 0 = written before the registry existed.
    study_registry_version: int = 0
    # The study's layers that have no field of their own, as "kind:path" (a
    # GeoJSON per entry), e.g. "wells:/data/wells.geojson"; the kinds a study
    # takes are domains.feature_layer_kinds(study).
    feature_layers: tuple = ()
    # The fixed validation protocol (a YAML under review_2026-10-08/) that the
    # validation harness reads after the quality assessment. Experiment runs
    # only (purpose: experiment); empty = no validation stage.
    validation_protocol: str = ""
    # Digitised or survey truth points (CSV, ground_truth.load_points_csv)
    # the validation reads beside the GNSS, as `insar validate --points`
    # (e.g. the Tahmoor West units of experiment A). --points wins. Empty =
    # none.
    validation_points: str = ""

    # --- What relative figures are relative to (B1, 8 Oct 2026) ---
    # significance.REFERENCE_SETTINGS holds each key's meaning; the trade-offs
    # of the three modes are in significance.reference_mode's docstring.
    # "farmland" (default; "farmland_relative" is the same): each cell minus
    # far-field ground of the same coherence class, as every Indiana record.
    # It cannot see motion that covers the whole area. "stable_polygon": the
    # comparison ground is the polygon in reference_stable_geojson (e.g. a
    # ring 1-3 km outside the angle of draw). "gnss_absolute": rates put on a
    # datum fitted to the GNSS stations in reference_gnss_stations (one offset
    # per date plus one constant per station, optionally a plane; each station
    # scored leave-one-out), written to deliverables/reference_datum.json.
    # Validation stations of an experiment must never be named here: the
    # harness warns when a scored station set the product's datum.
    reference_mode: str = "farmland"
    reference_stable_geojson: str = ""
    reference_gnss_stations: list = field(default_factory=list)
    reference_gnss_plane: bool = False
    reference_gnss_radius_m: float = 100.0
    reference_gnss_match_days: float = 3
    reference_gnss_cache_dir: str = ""
    # gnss_absolute only (8 Oct 2026, plate_motion.py): the frame of the GNSS
    # datum. "plate" (default): each station's plate motion (ITRF2020-PMM,
    # Altamimi et al. 2023) is removed, so plate-stable ground reads still;
    # "itrf" keeps NGL's IGS20 series as they are. reference_gnss_plate:
    # "auto" (from where the stations are) or a plate code such as NOAM,
    # AUST, PCFC (needed outside the auto regions). The validation harness
    # always compares IGS20 with IGS20 and does not read these.
    reference_gnss_frame: str = "plate"
    reference_gnss_plate: str = "auto"

    # --- Event products (F5, 8 Oct 2026; events.py has the details) ---
    # What the ground did between two dates, per panel and along lines,
    # written to deliverables/events/; nothing assumes a constant rate. Off by
    # default (nothing else in the products changes). event_products: true
    # (one window, the whole record) or a list of windows, each
    # "YYYY-MM-DD:YYYY-MM-DD", [start, end] or {name, start, end}: one change
    # map each. event_polygons: GeoJSON (lon/lat) of panels: the largest
    # robust subsidence and its date per polygon, over the polygon's own dates
    # (properties start/end or mined_start/mined_end) or else each window.
    # event_lines: GeoJSON (lon/lat) of lines: a binned profile of each window.
    event_products: object = field(default_factory=list)
    event_polygons: str = ""
    event_lines: str = ""

    # --- Quality thresholds ---
    # MintPy's default is 0.7, the "reliable pixel" threshold in Yunjun et
    # al. (2019). Literature drops this to 0.2-0.5 over vegetated or coastal
    # ground where 0.7 masks almost everything. 0.4 is what this farmland
    # site supports at 12-day C-band (Site A: 84.5% of the AOI at 0.4,
    # 6.6% at 0.7); say so in the report rather than making a map look fuller.
    min_temporal_coherence: float = 0.4
    # Second quality gate (Kyle, 5 Oct 2026: measure, then decide). > 0 also
    # drops cells whose MintPy temporal coherence of the unwrapped network
    # (geo/geo_temporalCoherence_mintpy.h5, i.e. loop-closure consistency) is
    # below this. 0.0 = off: the layer is delivered and its effect on coverage
    # recorded in the QA (coverage.second_gate at 0.4/0.5/0.7), but report
    # figures are not gated by it.
    min_unwrap_consistency: float = 0.0
    # Spatial coherence floor for including a pair in the network. Written
    # to mintpy.network.minCoherence, which MintPy only READS when
    # mintpy.network.coherenceBased = yes - and this package never sets
    # that, so today this field is inert. Turning coherence-based dropping
    # on at summer coherence ~0.3 would strip the summer network down to the
    # minimum spanning tree with no triplets; leave it off.
    min_spatial_coherence: float = 0.4
    # Coherence a pixel needs to qualify as the automatic reference point.
    # MintPy's default is 0.85, which finer multilooking (10x2, 5x1) often
    # cannot reach - fewer looks means less averaging and noisier pixels -
    # and over farmland it may find nothing at all. Ignored when
    # reference_lalo is set.
    reference_min_coherence: float = 0.85

    # --- Deliverables ---
    # Vendor reports Kleinfelder receives use inches; the InSAR literature
    # uses millimetres. "in" or "mm".
    display_units: str = "in"

    # --- Report identity: the client report's title and place line ---
    project_title: str = ""            # "" = the project folder's name, tidied
    project_location: str = ""         # "" = looked up from the map (county, state), cached in context/location.json

    # A COUNT only, kept for comparison with older QA records: how many cells
    # read faster than this (mm/yr, whatever display_units says). It is not the
    # physical limit. What unwrapping limits is the DIFFERENCE between
    # neighbouring cells about one look (~41 m) apart: a quarter wavelength per
    # revisit, ~422 mm/yr at a 12-day revisit (C-band, 5.55 cm). A broad,
    # smooth bowl can sink faster than that and still be measured; a steep
    # longwall edge moving metres per year cannot be measured at C-band at
    # all, however this is set. The QA measures the real limit and flags the
    # cells beyond it (quality_assessment.json 'plausibility',
    # stack_qc.rate_gradient_flags).
    max_plausible_velocity_mm_yr: float = 500.0
    # Grid spacing (pixels) for the KMZ displacement points. Every pixel
    # would make an unusable file; this samples it down.
    kmz_point_step: int = 20

    # Number of points to average in the cumulative displacement graph.
    # Vendor reports plot a selected cluster inside the deforming area
    # rather than the whole AOI - an AOI-wide mean is dominated by stable
    # ground and understates the movement people care about.
    graph_n_points: int = 126

    # Where the selected-point cluster sits, as "lat,lon". Empty centres it
    # on the deepest point of the smoothed rate field anywhere in the scene.
    #
    # PIN THIS for any vendor comparison. Over a 50 km footprint the scene
    # minimum is usually NOT the feature the client is paying attention to -
    # on the Site A run it landed 9 km outside the monitoring polygon,
    # so the graph described a different piece of ground than the vendor's.
    graph_center_lalo: str = ""

    # A pixel must be valid in at least this fraction of epochs to join the
    # cluster. A tight cluster decorrelates together: without this, the
    # Site A graph ran to nan in its last two epochs because all 126
    # points went blank on the same dates.
    #
    # THIS FILTER BIASES THE GRAPH SHALLOW, unavoidably. Fast-subsiding
    # ground is exactly the ground that decorrelates, so any completeness
    # requirement preferentially keeps slow pixels. Measured at the
    # Site A bowl:
    #
    #     no filter   228 m radius   -25.4 mm/yr   but the trace ends nan
    #     0.6         716 m radius    -9.8 mm/yr   complete trace
    #     0.7        1052 m radius    -5.1 mm/yr   complete trace
    #
    # So treat the plotted trace as a CONSERVATIVE floor, and make any
    # vendor comparison on the velocity field at a stated averaging scale
    # instead - velocity is fitted per pixel over whatever epochs exist and
    # does not suffer this dropout bias.
    graph_min_valid_fraction: float = 0.6

    # Rate at which a pixel counts as "deforming" in the summary. Site
    # specific: 1 in/yr is reasonable for slow settlement, far too low for
    # active longwall mining, and far too high for a site where a quarter
    # inch a year matters. Expressed in the display units.
    deforming_threshold: float = 1.0

    # DISP-S1 only: apply OPERA's recommended_mask on top of the phase
    # quality threshold. It is deliberately conservative and discards
    # roughly four fifths of the grid over farmland. Over a site whose
    # deformation is already known, phase quality alone may be the better
    # filter - but turning this off means keeping pixels OPERA itself
    # flagged, so say so in any report.
    disp_use_recommended_mask: bool = True

    # DISP-S1 only: abandon a single granule read that has made no progress
    # for this many seconds, then retry it on a fresh connection. 0 disables
    # the watchdog.
    #
    # This is a backstop, not a tuning knob. A normal 50 km window read takes
    # ~2 minutes; the failure it exists for is unbounded, not slow. fsspec
    # waits on its loop thread with no timeout, so a half-closed keepalive
    # connection can park the run indefinitely - one stalled 6 hours at
    # granule 192 of 254 without logging anything. Keep it well above the
    # slowest legitimate read so it never fires on a merely slow granule.
    disp_read_timeout_s: int = 900

    # DISP-S1 only: process this frame and discard every other one. "" keeps
    # whatever the AOI intersects.
    #
    # SET THIS whenever the AOI touches more than one frame. Each DISP-S1
    # frame is an independent time series on its own grid with its own
    # reference epochs, and build_stack() locks onto the grid of the first
    # granule it reads, then skips the rest as shape mismatches. Which frame
    # wins is decided by a date sort, not by coverage - so an AOI that is
    # 100% inside frame A can silently produce a time series of frame B,
    # complete with a velocity map and coverage stats, if B happens to sort
    # first. The warning in search() tells you the AOI spans frames; it
    # cannot tell you which frame you actually want. That is this setting.
    #
    # Give the digits only, as they appear in the granule name: "04861" from
    # OPERA_L3_DISP-S1_IW_F04861_VV_...
    disp_frame_id: str = ""

    # DISP-S1 only: keep every Nth pixel on read. 1 reads native 30 m.
    #
    # The point of this is a WIDE, COARSE run, which is the only way to
    # estimate a long-wavelength ramp against GNSS. Over the 50 km
    # Site A box only 3 GNSS stations fall inside, two of them the
    # same site 80 m apart and the third a known unwrapping artefact - not
    # enough to fit a 3-parameter plane. Over the full DISP-S1 frame
    # (275 x 223 km) there are 30 stations, 25 of them usable, which gives
    # 22 degrees of freedom.
    #
    # Decimation is what makes that affordable: the full frame at 30 m is
    # ~76 GB of stack, at 300 m it is 0.6 GB - cheaper than the fine run.
    # A ramp is by definition long-wavelength, so throwing away resolution
    # costs nothing for this purpose.
    #
    # Do NOT use a decimated run for deformation: subsampling does not
    # average, it just skips pixels, so single-pixel noise survives intact
    # while the signal is undersampled.
    disp_decimate: int = 1

    # DISP-S1 only: keep at most this many epochs per year. 0 keeps every
    # granule. Subsampling cuts read time roughly proportionally and is
    # fine for a long slow signal, but it coarsens the time series - do
    # not subsample when you need to resolve a short event.
    #
    # Granules at reference-epoch boundaries are always kept regardless:
    # they anchor the stitching, and dropping one puts an offset error
    # into every epoch after it.
    disp_max_per_year: int = 0

    # DISP-S1 only: drop pixels flagged connected_component_labels == 0
    # (SNAPHU could not unwrap them reliably) in the epoch where they are
    # flagged. Safe to leave on: these drops no longer feed the
    # reference-epoch carry, which is what used to turn a 15% per-granule
    # mask into 99% loss over a decade. Turn it off only to reproduce the
    # unfiltered product.
    disp_mask_connected_components: bool = True

    # DISP-S1 only: fraction of epochs a pixel must be valid in to get a
    # velocity and a quality value. Applied to BOTH, so the raster and the
    # summary statistics describe the same pixels.
    #
    # OPERA's validation work uses 0.9. That is right for validating the
    # product against GNSS and wrong for monitoring farmland, where it
    # removes most of the site. 0.5 keeps a pixel that has half a decade
    # of record. Raise it towards 0.9 for a deliverable and report what
    # the coverage cost was.
    disp_min_valid_fraction: float = 0.5

    # DISP-S1 only: subtract the correction layers the product supplies but
    # does NOT itself apply - solid earth tide and ionospheric delay. Both
    # layers state "Subtract from /displacement to apply as a correction",
    # and the granule metadata records both corrections as None. Leaving
    # this false means they stay in the result.
    #
    # There is no tropospheric layer to apply. DISP-S1 is uncorrected for
    # troposphere, which is the dominant error source at these scales.
    disp_apply_corrections: bool = True

    # DISP-S1 only: fit an annual sine and cosine alongside the linear rate.
    # Over farmland, soil moisture and crop cycles put a real annual
    # oscillation into the phase; fitting a straight line through it leaves
    # that oscillation in the residual and can bias the rate where the
    # record does not cover whole years evenly. Needs a stack of at least
    # two years to be identifiable.
    disp_seasonal_model: bool = True

    # DISP-S1 only: fit a quadratic term and report acceleration. EGMS
    # publishes one per measurement point, so the field exists in the
    # schema either way - this controls whether it is estimated.
    #
    # OFF by default because it TRADES AGAINST THE RATE: over a short
    # record a steady trend and a gentle curve are nearly
    # indistinguishable, and fitting both inflates the error bar on each.
    # Worth turning on only for a decade-length stack, and only when
    # someone actually wants to know whether motion is speeding up.
    disp_acceleration_model: bool = False

    # DISP-S1 only: load the per-frame DISP-S1-STATIC product for per-pixel
    # line-of-sight vectors and the layover/shadow mask. Without it, the
    # incidence angle is a single frame-average number, which is wrong by
    # up to ~7 degrees at the swath edges - about 10% on any conversion
    # from line-of-sight to vertical.
    disp_use_static_layers: bool = True

    # DISP-S1 is a screening route (ROUTE_TAG). It writes no client report or
    # client KMZ unless this is set, so a screening number cannot reach a
    # client by default.
    disp_client_report: bool = False

    # --- Validation ---
    # Compare against GNSS at export time. Where a station falls inside the
    # AOI this calibrates the reference point onto an absolute frame; where
    # none does, it bounds how fast the reference pixel could plausibly be
    # moving, using regional vertical rates. Either is worth having, and a
    # product with neither should say so.
    gnss_validation: bool = True
    gnss_search_radius_km: float = 250.0

    # Ground argued to be stable, as (min_lon, min_lat, max_lon, max_lat).
    # The scatter of velocity across it is an EMPIRICAL precision figure
    # that absorbs the reference point, the uncorrected troposphere and
    # any orbital ramp - all of which the formal velocityStd excludes and
    # all of which are larger. Quote the larger of the two as the minimum
    # detectable velocity.
    #
    # Unset means the empirical precision is NOT measured and summary.json
    # says so. There is deliberately no fallback: selecting the "least
    # deforming" area from the result itself is selection on the outcome and
    # understates (tests/test_rate_and_geometry.py asserts it stays removed).
    stable_area_bbox: tuple = ()

    # The vendor figures for this site, drawn on the coverage-vs-threshold
    # plot so it is visible where their coverage falls on our curve. An
    # AUDIT LINE, not a target: matching someone's coverage by tuning a
    # threshold imports an error budget nobody can see, and their
    # acceptance criteria are not published. Leave unset if there is no
    # vendor product to compare against.
    vendor_coverage_percent: float = 0.0
    vendor_points: int = 0

    # DISP-S1 only: reject residual unwrapping errors that the per-granule
    # connected-component mask missed, by looking for whole-cycle jumps
    # (2.77 cm in C-band line of sight) that revert at the next epoch.
    # Nothing physical moves 2.8 cm in twelve days and exactly back again.
    disp_flag_unwrap_jumps: bool = True

    # DISP-S1 only: estimate and remove the tropospheric delay with RAiDER.
    # DISP-S1 applies no tropospheric correction and ships no layer for
    # one, and troposphere is the dominant error source at these scales.
    #
    # OFF BY DEFAULT because of what it costs, not because it is optional:
    # every epoch is a separate ERA5 request to the Copernicus CDS, they
    # queue, and 254 of them can take longer than streaming the InSAR.
    # Delays are cached per date, so an interrupted run resumes.
    #
    # Expect little benefit over a small, flat AOI: ERA5's ~30 km grid
    # resolves almost no structure across 7 km, and a spatially uniform
    # delay cancels against the reference point anyway. The run reports
    # the spatial spread of what it removed so this can be judged. Needs
    # CDS credentials in ~/.cdsapirc and the ERA5 licence accepted.
    disp_tropo_correction: bool = False

    # DISP-S1 only: subtract whole-cycle steps that the reference-epoch
    # carry baked in, where the run shows a clear excess of them at
    # boundaries over ordinary epochs. On the Site A stack that excess
    # is 8.4% against a 1.1% control - an eightfold enrichment, so the
    # steps are the stitching rather than noise.
    #
    # OFF BY DEFAULT because it erases real displacement wherever the
    # ground genuinely moved a whole cycle (27.7 mm in C-band line of
    # sight) between consecutive epochs. On a 12-day stack that is a rate
    # of 840 mm/yr; Site A runs ~25 mm/yr, so the two are nowhere near
    # each other. Over active longwall mining they would be.
    disp_repair_boundary_steps: bool = False

    # Remove a scene-wide DATUM SHIFT at a reference-epoch boundary.
    #
    # Distinct from disp_repair_boundary_steps, which hunts whole-cycle
    # unwrapping errors. This one is SMALLER than a cycle and so invisible
    # to that test, but it moves every epoch after it and biases the rate
    # across the whole record.
    #
    # Site A 2025-01-17: DISP-S1 rebased during a winter coverage
    # collapse and the carry was estimated from epochs with 6-16% of
    # pixels valid. It came out +39.0 mm, the largest of sixteen
    # boundaries. Everything after sat ~25 mm low, which made stable
    # farmland 15 km from any deformation read -32.7 mm/yr and inflated
    # the real subsidence bowl from -15.2 to -18.4 mm/yr.
    #
    # Safe to leave on. A spatially uniform offset is exactly what InSAR
    # cannot measure - referencing to a pixel already removes any uniform
    # motion - so subtracting the scene median at a boundary can only
    # remove artefact, never signal. It does NOT fix a spatially varying
    # carry error; that needs the boundary treated with suspicion.
    # OFF by default. The failure it repairs is real, but the repair has
    # its own failure mode: an earlier version subtracted the whole
    # scene-median step and accumulated a spurious +19 mm/yr uplift across
    # the entire map. It now removes only the anomaly relative to the
    # local trend and triggers only on a large step AND poor coverage -
    # but a boundary a human has looked at still beats an automatic
    # correction that quietly rewrites a decade. Turn on deliberately,
    # and check the control ground afterwards.
    disp_repair_boundary_datum: bool = False

    # DISP-S1 only: instead of fitting a rate to the stitched product
    # directly, resample it onto a lat/lon grid and run the SAME MintPy
    # correction chain the phase-linked record uses - ERA5 troposphere
    # (pyaps), linear deramp with the deforming area excluded, DEM-error
    # correction, minimum-residual reference date, linear rate - then the
    # same deliverables, QA, figures, GNSS tie and client report. This is
    # what makes a DISP-S1 run comparable with a stack run cell for cell
    # (disp_chain.py). Off keeps the direct fit (build_stack).
    disp_record_chain: bool = False

    # Solid-earth tide and ionosphere (corrections.py). Both vary over
    # hundreds of km; across a small area they are a uniform shift plus a
    # tilt that the reference point and deramp remove (measured on
    # Site A, 30 x 24 km: < 0.003 mm/yr). "auto" applies them when the
    # processed box's longer side reaches large_area_km. "yes"/"gim" forces,
    # "no" skips. The ionosphere uses IGS global ionosphere maps; the DISP-S1
    # route always subtracts the product's own layers.
    solid_earth_tides: str = "auto"
    ionosphere_correction: str = "auto"
    large_area_km: float = 100.0

    # disp_record_chain: an existing MintPy geocoded file whose lat/lon grid
    # the DISP-S1 series is resampled onto, coarsened by
    # disp_target_grid_factor (2 turns the record's 15 m posting into 30 m,
    # DISP-S1's own). "" builds a ~30 m grid from the bbox.
    disp_target_grid_file: str = ""
    disp_target_grid_factor: int = 2

    # disp_record_chain: a MintPy geo geometry file (height, incidence and
    # azimuth angle, slant range) to block-average onto the target grid.
    # "" uses the DISP-S1-STATIC layers instead.
    disp_geometry_file: str = ""

    # disp_record_chain: topsStack baselines/ directory for the DEM-error
    # step. DISP-S1 v1.0 ships a perpendicular_baseline layer that is all
    # zeros, so the baselines have to come from an orbit stack of the same
    # track. Epochs without a baseline are dropped (logged). "" skips the
    # DEM-error correction and says so in the product.
    disp_baseline_dir: str = ""

    # disp_record_chain: directory holding (or to receive) the ERA5/ folder
    # of pyaps3 grib files. Point it at an existing project's mintpy/ to
    # reuse its downloads; "" uses this project's mintpy/.
    disp_weather_dir: str = ""

    # DISP-S1 only: granules read concurrently (separate processes) before
    # the stack is assembled from the on-disk cache. Each read is a remote
    # window out of a ~440 MB file, so the run is network-bound; 1 keeps
    # the old sequential behaviour.
    disp_prefetch_workers: int = 6
    # After QA, cross-check the official record against OPERA DISP-S1 over the
    # same ground (crosscheck.py). Off by default: it downloads the DISP-S1
    # stack. The anchor-epoch check is applied first.
    cross_check_disp: bool = False
    # After QA, plant synthetic sustained movement of known size in the real
    # field and measure how often the report's rule finds it
    # (injection_test.py). No positive-control site exists, so this is the
    # recovery test. Off by default (CPU time); also `insar injection-test`.
    injection_test: bool = False
    max_tropo_std_cm: float = 0.0     # exclude pairs whose differential tropospheric
                                        # delay varies more than this (cm). 0 = no gating

    # ------------------------------------------------------------------
    # topsStack backend (backend: "stack")
    # ------------------------------------------------------------------
    # WHY THIS EXISTS. The "local" backend runs one topsApp per pair, which
    # re-coregisters from scratch every time and exposes no multilook, no
    # filter strength and no unwrapper cost mode. For a 250-epoch stack that
    # is both slow and uncontrollable. topsStack coregisters once against a
    # single reference, then forms every interferogram off that stack.
    #
    # stackSentinel.py generates run_files/ + configs/; we execute them.
    # Flag names in the comments below are stackSentinel's own.

    stack_swaths: str = "2"            # -n  sub-swaths, e.g. "2" or "1 2 3".
                                        # Each IW sub-swath is ~80 km wide. Check the
                                        # AOI does not straddle a boundary: if it does,
                                        # a single swath silently drops half the area.
    stack_polarization: str = "vv"     # -p
    stack_reference_date: str = ""     # -m YYYYMMDD; "" = first date.
                                        # MATTERS MORE THAN IT LOOKS: every SLC is
                                        # coregistered to this date and geom_reference
                                        # is derived from it. Prefer a mid-stack, low
                                        # perpendicular-baseline, winter (bare-field,
                                        # high-coherence) date over the default first.
    stack_num_connections: int = 3     # -c  temporal neighbours per date.
                                        # stackSentinel pairs by TIME ONLY - it has no
                                        # baseline criterion at all. Generate slightly
                                        # over-connected here and prune on perpendicular
                                        # baseline in MintPy's modify_network, which is
                                        # where max_perp_baseline finally gets used.
    stack_num_overlap_connections: int = 3   # -O  burst overlaps for NESD

    # Multilooking. Sentinel-1 IW SLC is 2.330 m slant range x ~13.9 m azimuth;
    # ground range is 2.330/sin(theta) ~ 3.71 m at 38.9 deg, so square pixels
    # need range ~ 3.75 x azimuth. Effective looks ENL ~ 0.53 * r * a once
    # oversampling is accounted for (range ~1.4x, azimuth ~1.35x).
    #
    #   r x a   posting           ENL    phase sigma @ coh 0.4   LOS sigma
    #   4 x 1   14.8 x 13.9 m     ~2.1   1.12 rad                4.9 mm
    #   8 x 2   29.7 x 27.8 m     ~8.5   0.56 rad                2.5 mm
    #  11 x 3   40.8 x 41.7 m     ~17    0.39 rad                1.7 mm
    #  22 x 6   81.6 x 83.4 m     ~70    0.19 rad                0.9 mm
    #
    # 11x3 IS THE DEFAULT ON MEASUREMENT. The table above assumes coherence
    # 0.4; Site A summer pairs measured 0.30, where 8x2 gives phase sigma
    # ~0.77 rad - close to the ~1 rad point where snaphu's cost surface stops
    # having a clear minimum. 11x3 gives ~0.55 rad. Residue density scales as
    # phase-sigma squared, so finer posting makes an unwrapping problem WORSE.
    # Deliver 15 m by geocoding (mintpy_geocode_lalo_step), not by
    # multilooking, and say in the report that 15 m is the POSTING and ~41 m
    # is the RESOLUTION. The trade is the active panel edge: the unwrappable
    # gradient is pi per pixel (13.9 mm), which the active face exceeds at any
    # C-band posting; the settled bowl is what this product measures.
    stack_range_looks: int = 11        # -r
    stack_azimuth_looks: int = 3       # -z

    stack_filter_strength: float = 0.4  # -f  Goldstein-Werner alpha, via
                                        # topsStack/FilterAndCoherence.py runFilter.
                                        # NOTE the coherence estimator alongside it
                                        # (ICU phase sigma) has FIXED internal windows -
                                        # there is no coherence-window knob to expose.

    # Unwrapping. This is the whole reason for the backend.
    #
    # topsStack/Stack.py:11-12 hardcodes noMCF='False', defoMax='2', and
    # unwrap.py:236 runUnwrapMcf maps that to costMode='SMOOTH',
    # initMethod='MCF', initOnly=True - snaphu's statistical cost optimisation
    # NEVER RUNS. That is the same weak unwrapper the topsApp path uses.
    # There is no CLI flag for it, so we rewrite the generated configs.
    #
    # stack_unw_full_cost=True gives costMode='DEFO', initMethod='MST',
    # maxComponents=20 - the real thing. 5-10x slower and worth measuring.
    stack_unw_method: str = "snaphu"   # -u  snaphu | icu
    stack_unw_full_cost: bool = True   # -> configs nomcf : True
    stack_unw_defomax: float = 4.0     # -> configs defomax, in cycles.
    # Which snaphu runs the unwrap. "isce": the snaphu 1.4.2 embedded in ISCE,
    # driven by topsStack's unwrap.py - which hard-codes 20 connected
    # components and passes a minimum component fraction of int(0.01) = 0, so
    # every Site A pair hit the cap. "snaphu2": the standalone snaphu 2.x
    # binary with every parameter below set explicitly (review 27 Sep 2026).
    stack_unw_engine: str = "isce"
    stack_unw_cost_mode: str = "DEFO"            # snaphu2 only: DEFO or SMOOTH
    stack_unw_max_components: int = 64           # snaphu2 only
    stack_unw_min_component_fraction: float = 0.001   # snaphu2 only
    # snaphu2 only: looks behind each coherence value (NCORRLOOKS). 0 = range x
    # azimuth looks / 0.64, right for a boxcar estimate. An adaptively filtered
    # stack (partner-pixel averaging, 22 Oct 2026) estimates coherence over many
    # more samples; left at the boxcar value, snaphu reads its honest low
    # coherence as noise.
    stack_unw_ncorrlooks: float = 0.0
    # Local change layer (Route B, 22 Oct 2026; local_change.py, scripts/
    # run_local_change.py): vendor-style gated 12-day steps at full resolution,
    # read without unwrapping, on a project with real merged SLCs at fine looks
    # (4 x 1): the stack route to the filter step, or the phase-linked-style
    # merge (--to-step merge_reference_secondary_slc), whose pairs the layer
    # forms from the merged SLCs. Says how a spot moves against its surroundings
    # (within about local_change_sigma_km); regional figures stay with the
    # GNSS-anchored record. SCREENING ONLY: on 22 Oct 2026 it failed its
    # pre-registered validation against the vendor at Site D in both
    # configurations (configuration 1: kept share, group differences;
    # configuration 2: kept share, town control, 4 vs 8 km reference
    # stability); not to be quoted as a finding. Its 12-day steps keep the
    # farmland drift.
    local_change_window: int = 5               # partner search radius in pixels (11 x 11 window)
    local_change_partner_corr: float = 0.35    # log-amplitude time-series correlation for a partner
    local_change_partner_dmu: float = 0.15     # mean amplitude within +-15% for a partner
    local_change_sigma_km: float = 4.0         # width of the smooth reference each step is read against
    local_change_threshold: float = 0.3        # step coherence needed for a step to count
    local_change_era5: str = ""                # ERA5.h5 (geometryRadar.h5 beside it); empty = <mintpy>/inputs/ERA5.h5
    local_change_min_snr: float = 3.0          # significance as a Gaussian sigma (3 -> p 0.0027, two-sided t-test)
    local_change_min_steps: int = 10           # counted steps a pixel needs before it can be significant
    local_change_min_partners: int = 4         # partners a pixel needs before any of its steps count
    local_change_noise_alpha: float = 0.05     # a step counts only above the coherence noise passes with this probability
    local_change_eff_sample_ratio: float = 0.65    # share of the rlks x (partners+1) samples that are independent
    local_change_dem_error: str = ""           # MintPy geo_demErr.h5 (geo_geometryRadar.h5 beside it): remove each step's DEM-error residual; empty = none
    local_change_baselines: str = ""           # topsStack baselines/ folder for that term; empty = <stack>/baselines
    # partners chosen per period of this many days over the whole record (22 Oct 2026: chosen once over 10
    # years, real neighbours on rotating cropland fall below the correlation test - median 3 partners of 120
    # at Site D; ~2-year periods put the test back at the ~50 dates it was set on); 0 = one period
    local_change_partner_window_days: float = 730.0
    local_change_ps_dispersion: float = 0.15   # amplitude sd/mean below this = persistent scatterer: counts without partners (speckle alone ~0.3 at 4 x 1 looks, so at 0.15 the path is effectively off: 0.04% of pixels at Site D); 0 = off
    local_change_out: str = ""                 # output folder name under the project; empty = local_change
    stack_unw_snaphu_bin: str = ""               # snaphu2 only; empty = PATH, then any conda env's bin/snaphu
    # Phase-linked route: the coherence written as filt_fine.cor, which
    # snaphu's cost model and MintPy's inversion weights read. "phasor": how
    # consistent the linked phase is across each look window (not a sample
    # coherence). "sample": the standard sample coherence of the two dates'
    # SLCs over each look window, from the kept SLC stack.
    miaplpy_pair_coherence: str = "phasor"
    # Per-pixel whole-cycle correction of unwrapping errors from loop closure
    # after bridging (time_series.closure_integer_correction). MintPy's own
    # phase_closure only corrects regions unwrapped in every pair, which
    # farmland never has.
    mintpy_closure_correction: bool = False
                                        # The hardcoded 2 cycles is 55 mm, far too
                                        # small for a longwall face. Sweep this on
                                        # panel-crossing pairs before production.

    stack_virtual_merge: str = "True"  # -V  virtual rasters for merged bursts

    # Ionosphere: OFF, deliberately. Delay scales as 1/f^2, so C-band is ~18x
    # smaller than L-band. At 38 N, away from the equatorial anomaly and the
    # auroral oval, differential ionospheric phase is typically under 1 cm and
    # very long-wavelength - which the linear deramp removes for free. The
    # split-spectrum estimate at C-band has noise comparable to the signal it
    # is estimating, and the sub-band interferograms have sqrt(2) worse SNR, so
    # a bad sub-band unwrap INJECTS error. Cost would be +80-150% wall clock.
    # Revisit only if per-date deramp coefficients show a strong across-track
    # ramp correlating with independent TEC.
    stack_ionosphere: bool = False     # --param_ion
    stack_ion_num_connections: int = 3  # --num_connections_ion

    # Parallelism. THESE ARE OUR RUNNER'S POOL SIZES, NOT --num_proc.
    # stackSentinel is always invoked with --num_proc 1: at higher values
    # Stack.write_wrapper_config2run_file emits " &" suffixes and bare "wait"
    # lines, a shell batch structure that cannot be resumed into or have
    # failures isolated within. We parallelise per command instead.
    #
    # MEASURED at Site A (24 cores, /usr/bin/time -v per command):
    #   topo         one process uses ~24 threads (2368% CPU)
    #   geo2rdr      ~8 threads each; resample ~7
    #   igram/merge/filter   I/O-bound at 21-43% CPU, ~500 MB/s on one disk
    #   snaphu       single-threaded, 98% CPU, peak RSS 0.53 GB (0.93 GB at 8x2)
    # The runner sets OMP_NUM_THREADS = cores // workers for each pool, so the
    # geometry stages at 3 workers x 8 threads fill the machine without
    # oversubscribing it, and unwrap can run one snaphu per core.
    #
    # Unset, they are SIZED FOR THIS MACHINE (settings.sized_parallelism) from
    # its RAM and cores: 12 / 3 / 24 on the 23 GB / 24-core machine above, fewer
    # on a laptop where 12 merges at 1.27 GB peak RSS each (measured,
    # site_b run_12) or 24 snaphu would be OOM-killed. A value in the YAML
    # always wins. Pool sizes change speed only, never the numbers.
    stack_num_proc: int = _sized("stack_num_proc")             # igram/merge/filter:
                                        # disk-bound, more workers past ~12 queue on the disk
    stack_num_proc_topo: int = _sized("stack_num_proc_topo")   # topo/geo2rdr/resample: multithreaded
    stack_num_proc_unwrap: int = _sized("stack_num_proc_unwrap")   # snaphu: one thread, ~0.5 GB each
    stack_cores: int = 0               # physical cores to divide between a
                                        # pool's workers; 0 = os.cpu_count()

    # Kill any single unwrap that runs longer than this, in minutes, quarantine
    # its partial output and treat the pair as failed. 0 = no limit.
    #
    # MEASURED at Site A: unwrap time is set by coherence. Pairs at median
    # coherence 0.447 took a median 1.3 min; pairs at 0.305 ran 40 min to
    # 3h51 each, and full-DEFO snaphu has no upper bound on low-coherence
    # input. 16 of 84 pairs held 85% of the unwrap CPU and the single longest
    # pair WAS the wall clock. MintPy down-weights low coherence anyway, so the
    # slowest pairs are also the ones contributing least. The cost of a limit
    # is connectivity: a date whose every pair times out drops out of the time
    # series, so the runner checks the network stays connected after unwrap
    # and names any orphaned date. Off by default; production uses 60.
    stack_unwrap_timeout_min: float = 0.0

    # Unwrap only the AOI (+margin) instead of the whole burst frame. A burst
    # is the full sub-swath in range, so the merged frame is ~90 km wide
    # around a 24 km AOI; at Site A 86% of every interferogram was outside
    # the usable area and snaphu solved all of it every time. The wrapper in
    # unwrap_crop.py crops, runs the same topsStack unwrap.py, and pads the
    # result back to full-frame files so MintPy's metadata stays correct.
    stack_crop_unwrap: bool = True
    stack_bbox_margin_deg: float = 0.05   # -b margin for burst selection AND
                                           # the unwrap crop margin around bbox
    stack_cleanup_burst_igrams: bool = True   # delete per-burst IW*/ after merge

    # Disk cleanups for long stacks. MEASURED at Site A per date: 0.86 GB of
    # coregistered SLCs, 0.86 GB of geo2rdr offsets (dead once that date is
    # resampled), 0.18 GB of overlap SLCs (dead once run_08 has inverted the
    # misregistration); per pair: 0.158 GB of ESD + coarse interferogram
    # (dead after run_08), 0.050 GB of unfiltered fine.int/fine.cor (dead
    # once filtered). At 274 dates / 820 pairs that is ~510 GB of dead
    # weight held for the whole run without these. Each cleanup checks its
    # consumer's OUTPUT exists first and leaves a .cleaned marker the
    # runner accepts on resume. geo2rdr and resample run as one per-date
    # chain so the offsets never pile up across the stack.
    # THE TRADE for coreg-after-filter: adding dates or changing looks later
    # means a fresh coregistration (and re-download). Fine for production;
    # leave off while a stack is still being iterated on.
    stack_cleanup_offsets: bool = False
    stack_cleanup_misreg_inputs: bool = False
    stack_cleanup_coreg_after_filter: bool = False
    stack_cleanup_unfiltered_igrams: bool = False

    # Concurrent burst2safe / orbit downloads. Measured serially: ~93 s per
    # date, almost all of it ASF latency; the disk is idle.
    download_workers: int = 4

    # Assemble each acquisition from its individual BURSTS instead of
    # downloading whole SAFE archives.
    #
    # MEASURED over Site A: a full SAFE is ~5.3 GB and carries three
    # sub-swaths and both polarisations. This AOI needs IW2/VV only - 3 bursts,
    # ~450 MB - so over 90% of every download was discarded. 31 dates go from
    # 274 GB to 13.9 GB; 275 dates from 1,240 GB to ~124 GB, which is what
    # turns the production run from "needs batched download-and-delete" into
    # something that simply fits. Ran 31 dates with zero failures; the output
    # is deterministic (same bursts -> same SAFE name, hash included).
    #
    # THE TRADE: the assembled SAFE contains only the swaths and polarisation
    # requested, so widening either later means re-downloading. Verified that
    # ISCE's Sentinel1 reader parses the result and that stackSentinel accepts
    # it - note it globs *.zip FIRST and only falls back to *.SAFE, so the two
    # must never be mixed in one directory.
    stack_use_burst2safe: bool = True
    burst2safe_env: str = "burst2safe"   # conda env holding burst2safe; it
                                          # pins dependencies that conflict
                                          # with ISCE, hence the separate env

    # Delete each SAFE once its date is coregistered. MEASURED on P0: the
    # SAFEs were 32 GB of a 45 GB project - 71% - and nothing downstream of
    # coregistration reads them (except the REFERENCE date's, whose burst
    # SLCs are VRTs into its SAFE; cleanup_safes protects it).
    #
    # THE TRADE: redoing coregistration then means re-downloading. A date is
    # only deleted once its coreg_secondarys output exists, so the risk is
    # confined to deliberately reprocessing from scratch, not to a resumed run.
    stack_delete_safes_after_coreg: bool = True
    # SAFEs kept regardless, most recent first. stackSentinel's incremental
    # path re-reads recent acquisitions to extend a stack, so a rolling window
    # keeps "add more dates later" cheap. 0 keeps none.
    stack_keep_recent_safes: int = 6
    # Delete the per-burst geometry (stack/geom_reference, ~5.8 GB and fixed
    # regardless of stack length) once every date is coregistered. Only needed
    # to coregister NEW dates; merged/geom_reference, which MintPy reads, is
    # a separate and much smaller product.
    stack_cleanup_burst_geometry: bool = False
    stack_use_gpu: bool = False        # -useGPU. Needs a CUDA build of ISCE2 in
                                        # stack_conda_env; the conda-forge build is
                                        # CPU-only. Accelerates topo/geo2rdr/resample
                                        # ONLY - snaphu has no GPU implementation, so
                                        # it does not touch the unwrapping bottleneck.
    stack_conda_env: str = ""          # run stack commands in this conda env;
                                        # "" = current. Used to reach a CUDA build
                                        # without disturbing the working env.

    # ------------------------------------------------------------------
    # MintPy options used by the stack path
    # ------------------------------------------------------------------
    # No mintpy.deramp line is emitted anywhere else in this package, so "auto"
    # means no ramp removal at all. Against a defect that grows with distance
    # from the reference point that is a conspicuous omission.
    #
    # linear, not quadratic: over an 80 km frame a quadratic surface has enough
    # freedom to absorb real regional subsidence, and the mine sits inside the
    # frame. Run quadratic once as a sensitivity test, not as the default.
    mintpy_deramp: str = "linear"      # no | linear | quadratic
    deramp_exclude_geojson: str = ""   # polygon(s) to EXCLUDE from the ramp fit.
                                        # Empty searches <project>/deliverables/
                                        # context/ then the package's context/.
                                        # Getting this wrong is silent: the plane
                                        # is fitted through the deformation and
                                        # then subtracted from it, shrinking the
                                        # signal with nothing in the output to
                                        # show it happened.
    mintpy_deramp_mask: str = ""       # path to a mask with the mine ZEROED.
                                        # MintPy's default maskFile is maskTempCoh.h5,
                                        # which includes the subsidence bowl. Fitting a
                                        # plane through real deformation biases the
                                        # plane and then subtracts that bias from the
                                        # bowl. Empty = build one from the mine
                                        # footprint plus a buffer.
    mintpy_unwrap_error_method: str = "bridging+phase_closure"
                                        # topsStack writes filt_fine.unw.conncomp, so
                                        # bridging is genuinely available here - unlike
                                        # the topsApp path. This is the direct attack
                                        # on the diagnosed defect.
    mintpy_geocode_lalo_step: str = "-0.000135,0.000172"
                                        # 15 m x 15 m at 38.27 N:
                                        # lat 15/110900, lon 15/(111320*cos 38.27).
                                        # POSTING, not resolution. See the looks table.
    # Dates MintPy leaves out of the series (mintpy.network.excludeDate: every
    # pair touching one is dropped before the inversion), as YYYYMMDD. For
    # dates a weather model cannot serve - HRRR starts 2016-07-15 (24 Oct 2026).
    mintpy_exclude_dates: list = field(default_factory=list)
    mintpy_tropo_method: str = "pyaps"  # pyaps | raider | height_correlation | no.
                                        # raider: weather_model (HRRR: NOAA, AWS, no
                                        # queue, CONUS from 2016-07-15) through RAiDER
                                        # in raider_env, written as inputs/<MODEL>.h5
                                        # for MintPy (weather_raider.py; 24 Oct 2026).
                                        # pyaps works natively in radar coordinates via
                                        # geometryRadar.h5, which is what this path
                                        # produces. RAiDER (tropo_correction.py) needs
                                        # GEOCODED per-pair files and so stays on the
                                        # DISP/HyP3 paths only. Needs ~/.cdsapirc.
    # ERA5 is published about 5 days after real time, so a record ending today
    # has its last dates without a weather-model delay. By default the run STOPS
    # after the troposphere step and names those dates (corrections.
    # verify_era5_coverage), so a partly corrected series never reaches geo/
    # under the ERA5 name. True delivers anyway: the dates are left uncorrected
    # and recorded in corrections.json and run_products.json (era5_missing_dates).
    era5_allow_partial: bool = False
    # MintPy's own parallelism (dask local cluster in invert_network). A
    # 31-date stack inverted 595k px x 82 pairs single-process in 57 s, so this is
    # irrelevant below a few hundred pairs; production (~820 pairs, 275
    # dates) is where it starts to matter. 0 workers = leave MintPy at its
    # defaults (no cluster).
    mintpy_num_workers: int = 0
    # MintPy's maxMemory (a chunking ceiling, not an allocation). Unset = sized
    # for this machine: 70% of its RAM, at most 16 GB (16 on the 23 GB machine).
    mintpy_max_memory_gb: float = _sized("mintpy_max_memory_gb")

    # ------------------------------------------------------------------
    # Site measurements in the quality assessment (quality_assessment.py)
    # ------------------------------------------------------------------
    # OPTIONAL named event, as YYYYMMDD. Empty (the default) means no event:
    # every year is treated alike and panel_history measures each panel
    # across its own CMIS mining year instead. Set both dates only to test a
    # specific reported event; the pre window is acceptance_window_days
    # before event_start, the post window the same span after event_end.
    acceptance_event_start: str = ""
    acceptance_event_end: str = ""
    acceptance_window_days: int = 30
    # OPTIONAL polygon of one named panel to report alongside the workings.
    # Empty (the default) reports no single panel; panel_history covers all.
    acceptance_panel_geojson: str = ""
    # Mapped mines by type (the map builder writes them). Both are also inside
    # deramp_exclude_geojson; kept apart so the report can say whether a
    # feature sits on surface-mine ground, over underground workings, or away
    # from both. Either may be empty.
    underground_mines_geojson: str = ""
    surface_mines_geojson: str = ""
    # OPTIONAL ground already known to move that is not a mine (landslide,
    # karst, a settling fill; the builder's "Other known ground movement"
    # layer). Also inside deramp_exclude_geojson; kept apart so the report can
    # say a feature lies on ground already known to move. Empty = none.
    other_movement_geojson: str = ""
    # Far field = coherent pixels outside the mine polygon plus this buffer.
    acceptance_far_field_buffer_m: float = 2000.0
    # Buffer (m) around the mapped mines and known movement that is kept out
    # of the deramp plane fit (build_deramp_mask, a Euclidean buffer). The QA's
    # ramp sensitivity reruns the fit at other buffers to show what it costs.
    deramp_buffer_m: float = 2000.0
    # Acquisition dates (YYYYMMDD) that may be missing from the processed
    # stack without stopping the run: the stage that finds a date dropped
    # (download, coregistration, merge) refuses unless the date is listed
    # here. ("all",) accepts any missing date (and the QA lists them).
    allow_missing_dates: tuple = ()
    # Acquisition dates (YYYYMMDD) left out of the Sentinel-1 stack on purpose:
    # a date that downloads and coregisters but cannot be aligned (29 Oct 2026,
    # the first city site: one date's burst-overlap image came out empty, so
    # every ESD pair with it failed). A new plan gets them as stackSentinel -x;
    # a plan already under way keeps its other dates' work - the excluded
    # dates' commands are skipped and their part-made coregistered images
    # moved to stack/excluded_dates/. They count as accepted missing dates and
    # are not downloaded. The stack's reference date cannot be excluded.
    stack_exclude_dates: tuple = ()
    # Dates ASF serves in a form burst2safe can never assemble (a known
    # catalogue defect on every attempt: "max() arg is an empty sequence",
    # duplicate burst versions) are added to allow_missing_dates in the
    # config file automatically and the run carries on - up to 5% of the
    # dates (at least 3). Network, login, disk or other failures still stop
    # the run. false = stop on these too and list them by hand.
    auto_accept_catalogue_defects: bool = True
    # Once a project is finished - its FINAL client package built, or an
    # experiment run ended with its validation - delete its processing files
    # and keep the results ('keep results only', insar storage). Afterwards
    # only 'Products only' is sure to work. false = keep everything.
    auto_keep_results: bool = True

    # --- Point products (deliverables.py) ---
    # Compact by default: points_attributes.gpkg (~20 columns, EPSG:4326
    # points) + points_timeseries.parquet. The per-point CSV of every epoch
    # (8-14 GB on a decade record) and the MintPy time-series KMZ only on request.
    points_full_csv: bool = False
    full_timeseries_kmz: bool = False

    @property
    def stack_dir(self) -> Path:
        """topsStack working directory: run_files/, configs/, coreg_secondarys/ ..."""
        return self.project_dir / "stack"

    @property
    def aoi_wkt(self) -> str:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return (
            f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
            f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
        )

    @property
    def slc_dir(self) -> Path:
        return self.project_dir / "slc"

    @property
    def orbit_dir(self) -> Path:
        return self.project_dir / "orbits"

    @property
    def dem_dir(self) -> Path:
        return self.project_dir / "dem"

    @property
    def pairs_dir(self) -> Path:
        return self.project_dir / "interferograms"

    @property
    def tropo_dir(self) -> Path:
        return self.project_dir / "tropo_corrections"

    @property
    def mintpy_dir(self) -> Path:
        if getattr(self, "time_series_dir", ""):
            return Path(self.time_series_dir)
        return self.project_dir / "mintpy"

    def ensure_dirs(self) -> None:
        for d in (
            self.slc_dir, self.orbit_dir, self.dem_dir,
            self.pairs_dir, self.tropo_dir, self.mintpy_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


BACKENDS = ("local", "hyp3", "disp", "nisar", "stack", "miaplpy")
_ENUMS = {
    "backend": BACKENDS,
    "orbit_direction": ("ascending", "descending"),
    "display_units": ("in", "mm"),
    "mintpy_deramp": ("no", "linear", "quadratic"),
    "stack_unw_method": ("snaphu", "icu"),
    "mintpy_tropo_method": ("pyaps", "raider", "height_correlation", "no"),
    "solid_earth_tides": ("auto", "yes", "no"),
    "ionosphere_correction": ("auto", "gim", "no"),
    # anything but "snaphu2" used to fall silently to ISCE's snaphu 1.4.2 and
    # its 20-component cap (unwrap_crop.engine_lines), so a typo is an error
    "stack_unw_engine": ("isce", "snaphu2"),
    "stack_unw_cost_mode": ("defo", "smooth"),
    "stack_polarization": ("vv", "vh", "hh", "hv"),
    "stack_virtual_merge": ("true", "false"),
    "mintpy_unwrap_error_method": ("auto", "no", "bridging", "phase_closure", "bridging+phase_closure"),
    "miaplpy_pair_coherence": ("phasor", "sample"),
    "miaplpy_gpu": ("auto", "on", "off"),
    "purpose": PURPOSES,
    "site_profile": SITE_PROFILES,
    "study": domains.STUDY_KEYS,
}
# MiaplPy matches these names exactly (case included). sequential_* is the
# patched overlapping-window estimator the official route uses.
MIAPLPY_METHODS = ("EMI", "EVD", "PTA", "sequential_EMI", "sequential_EVD", "sequential_PTA", "SBW")
_UNIT_INTERVAL = ("min_temporal_coherence", "min_spatial_coherence",
                  "reference_min_coherence", "esd_coherence_threshold", "min_unwrap_consistency")
_NON_NEGATIVE = ("gnss_tie_radius_km", "gnss_search_radius_km", "acceptance_far_field_buffer_m",
                 "deramp_buffer_m", "stack_unwrap_timeout_min", "stack_bbox_margin_deg",
                 "mintpy_num_workers", "miaplpy_network_connections", "max_tropo_std_cm",
                 "disp_max_per_year", "disp_read_timeout_s", "stack_keep_recent_safes",
                 "vendor_coverage_percent", "vendor_points")
_POSITIVE = ("max_perp_baseline", "max_temp_baseline", "max_plausible_velocity_mm_yr",
             "mintpy_max_memory_gb", "large_area_km", "acceptance_window_days", "kmz_point_step",
             "graph_n_points", "deforming_threshold", "miaplpy_gpu_memory_gb")
# One Sentinel-1 IW swath is ~80 km, a whole track ~250 km (~2.5 deg at these
# latitudes); a coregistered stack over more than that is a typo, not a site.
MAX_STACK_BBOX_DEG = 2.5
# Every route reads Sentinel-1 (IW TOPS) or NISAR; a stack track number is 1-175.
_S1_TRACKS = range(1, 176)


class ConfigError(ValueError):
    pass


def _as_date(v, fmt: str):
    try:
        return _dt.datetime.strptime(str(v), fmt).date()
    except ValueError:
        return None


def split_area_spec(spec: str, allow_full: bool = True) -> tuple:
    """
    One report_areas / focus_sites_lalo entry -> (name, kind, value) with kind
    "full", "box" (value = (S, W, N, E)) or "geojson" (value = the path as
    written). Raises ValueError in plain words. The report splits each entry at
    the FIRST ':', so a name may not contain one ("Pit 3: north:..." is read
    as the area "Pit 3" with the spec "north:...").
    """
    spec = str(spec)
    if ":" not in spec:
        raise ValueError(f"{spec!r} has no ':' - write it as 'Name:full', 'Name:S,W,N,E' or 'Name:/path/area.geojson'")
    name, val = (x.strip() for x in spec.split(":", 1))
    if not name:
        raise ValueError(f"{spec!r} has no name before the ':'")
    if val.lower() == "full":
        if not allow_full:
            raise ValueError(f"{spec!r}: 'full' is not a site; give S,W,N,E or a .geojson")
        return name, "full", None
    if val.lower().endswith(".geojson"):
        return name, "geojson", val
    try:
        s, w, n, e = (float(v) for v in val.split(","))
    except ValueError:
        hint = " (is there a ':' in the name?)" if ":" in val else ""
        raise ValueError(f"{spec!r}: {val!r} is not 'full', 'S,W,N,E' or a .geojson path{hint}") from None
    if not (-90 <= s <= 90 and -90 <= n <= 90 and -180 <= w <= 180 and -180 <= e <= 180) or s == n or w == e:
        raise ValueError(f"{spec!r}: {val!r} is not a box of S,W,N,E in degrees")
    return name, "box", (min(s, n), min(w, e), max(s, n), max(w, e))


def _overlaps(box_swne, bbox) -> bool:
    s, w, n, e = box_swne
    W, S, E, N = bbox
    return w < E and e > W and s < N and n > S


def _geojson_bounds(path: Path):
    """(S, W, N, E) over every coordinate in the file, or None if it has none."""
    xs, ys = [], []

    def walk(c):
        if c and isinstance(c[0], (int, float)):
            xs.append(float(c[0])); ys.append(float(c[1]))
        else:
            for x in c or ():
                walk(x)
    doc = json.loads(Path(path).read_text())
    feats = doc.get("features") or ([doc] if doc.get("type") == "Feature" else [{"geometry": doc}])
    for f in feats:
        walk((f.get("geometry") or {}).get("coordinates") or [])
    return (min(ys), min(xs), max(ys), max(xs)) if xs else None


# ------------------------------------------------------------------ reissues (M3, 7 Oct 2026)
REVISION_KEYS = ("number", "supersedes", "changes")


def _revision_date(v) -> Optional[_dt.date]:
    """The issue date from 'YYYY-MM-DD' or a YAML date; None otherwise."""
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, _dt.date):
        return v
    return _as_date(str(v).strip(), "%Y-%m-%d") if v is not None else None


def revision_problems(rev) -> list:
    """What is wrong with a report_revision value, in plain words. Empty for
    a complete note, and for none at all (a first issue)."""
    if rev is None or rev == {}:
        return []
    if not isinstance(rev, dict):
        return [f"report_revision must be a block with {', '.join(REVISION_KEYS)}; got {rev!r}"]
    out = []
    extra = sorted(set(map(str, rev)) - set(REVISION_KEYS))
    if extra:
        out.append(f"report_revision: unknown key(s) {extra}; it takes {list(REVISION_KEYS)}")
    n = rev.get("number")
    whole = isinstance(n, int) and not isinstance(n, bool) or isinstance(n, str) and n.strip().isdigit()
    if not whole or int(n) < 1:
        out.append(f"report_revision number = {n!r} must be a whole number: 1 for the first reissue, 2 for the next")
    d = _revision_date(rev.get("supersedes"))
    if d is None:
        out.append(f"report_revision supersedes = {rev.get('supersedes')!r} must be the date the report it "
                   "replaces was issued, as YYYY-MM-DD")
    elif d > _dt.date.today() + _dt.timedelta(days=1):
        out.append(f"report_revision supersedes = {d} is in the future; give the date the earlier report was issued")
    ch = rev.get("changes")
    if not isinstance(ch, (list, tuple)) or not ch:
        out.append("report_revision changes must be a list of plain sentences: what changed since that issue, and why")
    elif any(not isinstance(c, str) or not c.strip() for c in ch):
        out.append("report_revision changes: every entry must be a sentence of text; none may be empty")
    return out


def report_revision(cfg_or_note) -> Optional[dict]:
    """The revision note to print, or None for a first issue (and for a note
    that is incomplete - load_config refuses those). Takes a config, or the
    note itself (the YAML block, or this function's own output, which a
    report may carry):
        {number, supersedes: 'YYYY-MM-DD', supersedes_text: '28 Sep 2026',
         heading: 'Revision 1, replacing the report issued 28 Sep 2026',
         changes: [plain sentences]}
    The page, the PDF and the KMZ print heading and changes as they are, and
    the release gate reads the same note, so they cannot word it differently."""
    rev = cfg_or_note if isinstance(cfg_or_note, dict) else getattr(cfg_or_note, "report_revision", None)
    if not isinstance(rev, dict) or not rev:
        return None
    rev = {k: rev.get(k) for k in REVISION_KEYS}
    if revision_problems(rev):
        return None
    n, d = int(rev["number"]), _revision_date(rev["supersedes"])
    when = f"{d.day} {d:%b %Y}"
    return {"number": n, "supersedes": d.isoformat(), "supersedes_text": when,
            "heading": f"Revision {n}, replacing the report issued {when}",
            "changes": [" ".join(c.split()) for c in rev["changes"]]}


def validate_config(cfg: "PipelineConfig") -> list:
    """Return every problem found in the values themselves; empty means the
    config is consistent. What a run needs from outside the config - files that
    exist, a track - is checked by validate_inputs (load_config runs both)."""
    problems = []
    backend = str(getattr(cfg, "backend", "")).lower()
    for name, allowed in _ENUMS.items():
        v = getattr(cfg, name, None)
        if v is not None and str(v).lower() not in allowed:
            problems.append(f"{name} = {v!r}; expected one of {list(allowed)}")
    v = getattr(cfg, "study_registry_version", 0)
    if isinstance(v, int) and not isinstance(v, bool) and not 0 <= v <= domains.REGISTRY_VERSION:
        problems.append(f"study_registry_version = {v}; this program knows versions 0-{domains.REGISTRY_VERSION} "
                        "(a newer config needs a newer program)")
    for name in _UNIT_INTERVAL:
        v = getattr(cfg, name, None)
        if v is not None and not (0.0 <= float(v) <= 1.0):
            problems.append(f"{name} = {v!r} is outside [0, 1]")
    for name in _NON_NEGATIVE:
        v = getattr(cfg, name, None)
        if v is not None and float(v) < 0:
            problems.append(f"{name} = {v!r} must be >= 0")
    for name in _POSITIVE:
        v = getattr(cfg, name, None)
        if v is not None and not float(v) > 0:
            problems.append(f"{name} = {v!r} must be > 0")
    # Ranges with a stated basis (16 Oct 2026, readiness review HND-03: 7 and
    # 40000 loaded). Only where the basis is clear; other settings keep the
    # sign checks above.
    # stack_filter_strength is the Goldstein-Werner alpha (topsStack
    # FilterAndCoherence.runFilter -> goldsteinWerner(alpha)): each patch's
    # spectrum is weighted by its smoothed magnitude to the power alpha, which
    # the method defines on [0, 1] - 0 no filtering, 1 the strongest
    # (Goldstein & Werner 1998, GRL 25, 4035). 7 is a typo for 0.7.
    fs = getattr(cfg, "stack_filter_strength", None)
    if fs is not None and not 0.0 <= float(fs) <= 1.0:
        problems.append(f"stack_filter_strength = {fs!r} is outside [0, 1]: it is the Goldstein-Werner filter "
                        "alpha (0 = no filtering, 1 = the strongest)")
    bad_dates = [d for d in (getattr(cfg, "mintpy_exclude_dates", None) or [])
                 if not re.fullmatch(r"(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", str(d).strip())]
    if bad_dates:
        problems.append(f"mintpy_exclude_dates must be dates written YYYYMMDD (e.g. '20160322'); not {bad_dates}")
    # raider: weather_raider.py runs RAiDER with weather_model; one RAiDER
    # serves (tropo_disp.MODEL_START), or it fails date by date hours in
    if str(getattr(cfg, "mintpy_tropo_method", "")) == "raider":
        wm = str(getattr(cfg, "weather_model", "") or "").upper()
        if wm not in ("HRRR", "HRRRAK", "GMAO", "MERRA2", "ERA5", "ERA5T"):
            problems.append(f"mintpy_tropo_method = 'raider' needs weather_model to be one RAiDER serves "
                            f"(HRRR, HRRRAK, GMAO, MERRA2, ERA5, ERA5T), not {wm!r}")
    # gnss_tie_radius_km: at most the regional-floor radius (gnss_tie.
    # REGIONAL_FLOOR_RADIUS_KM, 100 km). The tie's uncertainty is floored by the
    # scatter of the stations within that radius and the tie does not anchor
    # without it (gnss_tie.regional_scatter, tie), so stations farther out
    # would move the background without being in the region its quoted
    # uncertainty describes. 40000 km takes in every station on Earth.
    from .gnss_tie import REGIONAL_FLOOR_RADIUS_KM
    tie_r = getattr(cfg, "gnss_tie_radius_km", None)
    if tie_r is not None and float(tie_r) > REGIONAL_FLOOR_RADIUS_KM:
        problems.append(f"gnss_tie_radius_km = {tie_r!r} is more than {REGIONAL_FLOOR_RADIUS_KM:g} km, the region "
                        "whose stations set the tie's uncertainty; use 0-"
                        f"{REGIONAL_FLOOR_RADIUS_KM:g} (0 switches the tie off)")
    bbox = list(cfg.bbox or [])
    bbox_ok = False
    if len(bbox) != 4:
        problems.append(f"bbox must be [min_lon, min_lat, max_lon, max_lat]; got {bbox}")
    else:
        w, s, e, n = (float(v) for v in bbox)
        if not (w < e and s < n):
            problems.append(f"bbox {bbox} is not (W < E, S < N)")
        elif not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90):
            problems.append(f"bbox {bbox} is outside lon/lat range")
        else:
            bbox_ok = True
            if backend in ("stack", "miaplpy") and max(e - w, n - s) > MAX_STACK_BBOX_DEG:
                problems.append(f"bbox {bbox} is {e - w:.2f} x {n - s:.2f} degrees; a coregistered stack covers at "
                                f"most one track (~{MAX_STACK_BBOX_DEG} degrees) - check the corners")
    for name in ("stack_range_looks", "stack_azimuth_looks", "stack_num_proc",
                 "stack_num_proc_topo", "stack_num_proc_unwrap", "download_workers",
                 "miaplpy_num_processors", "disp_prefetch_workers", "disp_decimate",
                 "disp_target_grid_factor", "stack_num_connections"):
        v = getattr(cfg, name, 1)
        if int(v) < 1:
            problems.append(f"{name} = {v!r} must be >= 1")

    # dates: the format every route parses, in order, and not in the future
    # (a future end date makes the record change every time the run is repeated)
    d0, d1 = _as_date(cfg.start_date, "%Y-%m-%d"), _as_date(cfg.end_date, "%Y-%m-%d")
    for name, d in (("start_date", d0), ("end_date", d1)):
        if d is None:
            problems.append(f"{name} = {getattr(cfg, name)!r} must be YYYY-MM-DD")
    if d0 and d1:
        if d0 >= d1:
            problems.append(f"start_date {cfg.start_date} is not before end_date {cfg.end_date}")
        if d1 > _dt.date.today() + _dt.timedelta(days=1):
            problems.append(f"end_date {cfg.end_date} is in the future; use the last date wanted "
                            "(at most today) so a rerun covers the same record")
    for name in ("acceptance_event_start", "acceptance_event_end", "stack_reference_date"):
        v = getattr(cfg, name, "")
        if v and _as_date(v, "%Y%m%d") is None:
            problems.append(f"{name} = {v!r} must be YYYYMMDD")
    ev0, ev1 = (_as_date(getattr(cfg, k, ""), "%Y%m%d") for k in ("acceptance_event_start", "acceptance_event_end"))
    if ev0 and ev1 and ev0 > ev1:
        problems.append(f"acceptance_event_start {cfg.acceptance_event_start} is after acceptance_event_end")
    ref_d = _as_date(getattr(cfg, "stack_reference_date", ""), "%Y%m%d")
    if ref_d and d0 and d1 and not (d0 <= ref_d <= d1):
        problems.append(f"stack_reference_date {cfg.stack_reference_date} is outside {cfg.start_date} .. {cfg.end_date}")
    for v in cfg.allow_missing_dates or ():
        if str(v).lower() != "all" and _as_date(v, "%Y%m%d") is None:
            problems.append(f"allow_missing_dates entry {v!r} must be YYYYMMDD (or 'all')")
    for v in cfg.stack_exclude_dates or ():
        if _as_date(v, "%Y%m%d") is None:
            problems.append(f"stack_exclude_dates entry {v!r} must be YYYYMMDD")
    if ref_d and ref_d.strftime("%Y%m%d") in {str(v).strip() for v in cfg.stack_exclude_dates or ()}:
        problems.append(f"stack_exclude_dates lists {cfg.stack_reference_date}, the stack's reference date "
                        "(stack_reference_date); every other date is aligned to it")

    swaths = str(getattr(cfg, "stack_swaths", "2")).split()
    if not swaths or any(t not in ("1", "2", "3") for t in swaths):
        problems.append(f"stack_swaths = {cfg.stack_swaths!r}; give sub-swath numbers 1-3 separated by "
                        "spaces, e.g. '2' or '1 2'")

    def _latlon(name, v):
        try:
            lat, lon = (float(x) for x in str(v).split(","))
        except ValueError:
            problems.append(f"{name} = {v!r} is not 'lat,lon'")
            return
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            problems.append(f"{name} = {v!r} is not 'lat,lon' in degrees")
        elif bbox_ok and not (bbox[1] <= lat <= bbox[3] and bbox[0] <= lon <= bbox[2]):
            problems.append(f"{name} = {v!r} is outside the bbox {bbox} (is it 'lat,lon'?)")
    if str(cfg.reference_lalo or "").strip():
        _latlon("reference_lalo", cfg.reference_lalo)
    for v in getattr(cfg, "reference_candidates_lalo", ()) or ():
        _latlon("reference_candidates_lalo entry", v)
    if str(getattr(cfg, "graph_center_lalo", "") or "").strip():
        _latlon("graph_center_lalo", cfg.graph_center_lalo)

    sab = list(getattr(cfg, "stable_area_bbox", ()) or ())
    if sab:
        if len(sab) != 4:
            problems.append(f"stable_area_bbox must be [min_lon, min_lat, max_lon, max_lat]; got {sab}")
        else:
            sw, ss, se, sn = (float(v) for v in sab)
            if not (sw < se and ss < sn and -180 <= sw <= 180 and -90 <= ss <= 90 and -90 <= sn <= 90):
                problems.append(f"stable_area_bbox {sab} is not [min_lon, min_lat, max_lon, max_lat] (W < E, S < N)")
            elif bbox_ok and not _overlaps((ss, sw, sn, se), bbox):
                problems.append(f"stable_area_bbox {sab} does not overlap the bbox {bbox}")

    # report areas and focus sites: parsed exactly as the report parses them
    for field_name, allow_full in (("report_areas", True), ("focus_sites_lalo", False)):
        seen = {}
        for spec in getattr(cfg, field_name, ()) or ():
            try:
                name, kind, val = split_area_spec(spec, allow_full)
            except ValueError as exc:
                problems.append(f"{field_name}: {exc}")
                continue
            if name.lower() in seen:
                problems.append(f"{field_name}: the name {name!r} is used twice")
            seen[name.lower()] = val
            if kind == "box" and bbox_ok and not _overlaps(val, bbox):
                problems.append(f"{field_name}: {name!r} ({val[0]},{val[1]},{val[2]},{val[3]}) is outside the bbox "
                                f"{bbox} - the box is S,W,N,E (latitude first)")
        paths = [v for v in seen.values() if isinstance(v, str)]
        for p in sorted({p for p in paths if paths.count(p) > 1}):
            problems.append(f"{field_name}: several areas share the polygon file {p}")

    # a reissue's note: a half-written one would print a heading with no
    # changes, or the gate would read it as no note at all
    problems += revision_problems(getattr(cfg, "report_revision", None))

    # the validation harness runs only in experiment mode: a protocol on a
    # client project would be read by nothing, which looks like a validation
    if str(getattr(cfg, "validation_protocol", "") or "").strip() and not is_experiment(cfg):
        problems.append("validation_protocol is set but purpose is not 'experiment': the validation harness runs "
                        "only on experiment runs. Set purpose: experiment, or remove validation_protocol.")

    # the study (domains.py): a study not offered yet runs only as an experiment,
    # and feature_layers names only that study's kinds
    sk = str(getattr(cfg, "study", "") or domains.DEFAULT_STUDY).strip().lower()
    if sk in domains.STUDY_KEYS:
        st = domains.study(sk)
        if st.builder_status != "offered" and not is_experiment(cfg):
            problems.append(f"study {sk!r} is not offered for client projects yet ({st.builder_reason}); "
                            "run it with purpose: experiment, or choose another study")
        kinds = domains.feature_layer_kinds(st)
        for spec in getattr(cfg, "feature_layers", ()) or ():
            try:
                k, _p = domains.split_feature_layer(spec)
            except ValueError as exc:
                problems.append(f"feature_layers: {exc}")
                continue
            if k not in kinds:
                problems.append(f"feature_layers: {k!r} is not a layer of the {sk!r} study"
                                + (f" (it takes {list(kinds)})" if kinds else " (it takes none)"))

    # what relative figures are relative to (B1): the mode, and what each mode
    # needs (significance.check_reference_settings). A missing polygon file is
    # validate_inputs' message (_INPUT_FILES), not repeated here.
    from .significance import check_reference_settings
    problems += [p for p in check_reference_settings(cfg)
                 if not (p.startswith("reference_stable_geojson ") and p.endswith(" does not exist"))]

    # event products (F5): the windows read as dates, the layers hold polygons
    # or lines with readable dates (a missing file is validate_inputs')
    from .events import check_settings as check_event_settings
    problems += check_event_settings(cfg)

    # fast motion (F2, F1, F3): values only; no file is required here
    from .aliasing import check_settings as check_aliasing_settings
    problems += check_aliasing_settings(cfg)

    # the NISAR route's settings (M2): values only
    problems += _nisar_problems(cfg)

    if backend == "miaplpy":
        # MiaplPy reads real merged SLC files (moved here from run_miaplpy, so
        # the builder and preflight see them before a run starts)
        if str(cfg.stack_virtual_merge).lower() != "false":
            problems.append("backend miaplpy needs stack_virtual_merge: \"False\" (MiaplPy reads real merged SLC files)")
        if cfg.stack_cleanup_coreg_after_filter:
            problems.append("backend miaplpy needs stack_cleanup_coreg_after_filter: false "
                            "(phase linking reads the coregistered SLCs)")
        problems += _miaplpy_problems(cfg)
    return problems


def _miaplpy_problems(cfg) -> list:
    """MiaplPy settings. MEASURED 5 Oct 2026 with the patched library's own
    sequential estimator (check_miaplpy_patch.py's synthetic farmland, two
    seeds, mini-stacks of 10): boundary/inside step-error ratio 20-22 at
    miaplpy_time_lag 0 (the datum defect back, caught only by the datum gate
    after ~12 h of linking), 1.06-1.08 at 1, 1.1-1.2 at 2, 1.2-1.3 at 3, 1.6
    at 5. The overlap must also leave each window a new image, so it is below
    the mini-stack size; and MiaplPy's mini-stack count is n_images // size
    with no floor (invert.pyx), so a mini-stack of < 2 images links nothing."""
    out = []
    m = str(cfg.miaplpy_phase_linking_method)
    if m not in MIAPLPY_METHODS:
        out.append(f"miaplpy_phase_linking_method = {m!r}; expected one of {list(MIAPLPY_METHODS)} (case matters)")
    size, lag = int(cfg.miaplpy_ministack_size), int(cfg.miaplpy_time_lag)
    if size < 2:
        out.append(f"miaplpy_ministack_size = {size} must be >= 2")
    if not (1 <= lag < max(size, 2)):
        out.append(f"miaplpy_time_lag = {lag} must be at least 1 and less than miaplpy_ministack_size ({size}); "
                   "in the patched MiaplPy it is the window overlap, and 0 brings back the datum defect")
    for name in ("miaplpy_range_window", "miaplpy_azimuth_window"):
        if int(getattr(cfg, name)) < 1:
            out.append(f"{name} = {getattr(cfg, name)!r} must be >= 1")
    return out


NISAR_CHOICES = {"nisar_ionosphere": ("split_spectrum", "no"), "nisar_chain_gap": ("longest_segment", "fail"),
                 "nisar_fetch": ("window", "download"), "nisar_polarization": ("", "HH", "VV")}


def _nisar_problems(cfg) -> list:
    """The nisar_* settings (nisar_backend.SETTINGS), in words. Checked on
    every config (they are inert unless backend is nisar)."""
    out = []
    for key, allowed in NISAR_CHOICES.items():
        v = str(getattr(cfg, key, allowed[0]) or "").strip()
        if (v.upper() if key == "nisar_polarization" else v) not in allowed:
            shown = ", ".join(repr(a) for a in allowed)
            out.append(f"{key} {v!r} is not one of {shown}")
    try:
        if bool(getattr(cfg, "nisar_phase_linked", False)):
            w = int(getattr(cfg, "nisar_pl_window", 11))
            if w < 3 or w % 2 == 0:
                out.append(f"nisar_pl_window = {w}: an odd number of pixels, at least 3")
            k = float(getattr(cfg, "nisar_posting_m", 15.0)) / 5.0
            if k < 1 or abs(k - round(k)) > 1e-6:
                out.append("nisar_phase_linked needs nisar_posting_m to be a multiple of 5 (the GSLC's 5 m pixels)")
            if not str(getattr(cfg, "reference_lalo", "") or "").strip():
                out.append("nisar_phase_linked needs reference_lalo (the phase-linked time series' reference point)")
        if not float(getattr(cfg, "nisar_posting_m", 80.0)) > 0:
            out.append("nisar_posting_m must be above 0 (metres; the grid NISAR is put on - 15 m like the Sentinel-1 "
                       "products; the GUNW itself is posted at 80 m)")
        if not float(getattr(cfg, "nisar_crop_margin_deg", 0.02)) >= 0:
            out.append("nisar_crop_margin_deg must be 0 or more (degrees around the AOI)")
        if not 0.0 <= float(getattr(cfg, "nisar_min_valid_ratio", 0.5)) <= 1.0:
            out.append("nisar_min_valid_ratio must be between 0 and 1 (a share of the median valid area)")
    except (TypeError, ValueError) as exc:
        out.append(f"a nisar_* setting is not a number ({exc})")
    return out


def miaplpy_stack_problem(cfg, n_dates: int) -> Optional[str]:
    """For the phase-linked runner once the scene search has counted the
    dates: MiaplPy splits n_dates // miaplpy_ministack_size mini-stacks with no
    floor of one, so fewer dates than one mini-stack gives every linked phase
    0 and a late, misleading failure. None when the stack is long enough."""
    size = int(getattr(cfg, "miaplpy_ministack_size", 10))
    if int(n_dates) < size:
        return (f"{n_dates} dates is fewer than one mini-stack (miaplpy_ministack_size = {size}): MiaplPy would link "
                f"nothing. Lengthen the date range or set miaplpy_ministack_size <= {n_dates}.")
    return None


# config fields that name a file or folder that must exist when set
_INPUT_FILES = ("dem_path", "deramp_exclude_geojson", "mintpy_deramp_mask", "underground_mines_geojson",
                "surface_mines_geojson", "other_movement_geojson", "acceptance_panel_geojson", "mined_year_geojson", "landcover_raster",
                "disp_target_grid_file", "disp_geometry_file", "disp_baseline_dir", "stack_unw_snaphu_bin",
                "validation_protocol", "validation_points", "reference_stable_geojson", "event_polygons", "event_lines",
                "aliasing_zone_geojson", "fast_zone_geojson")


def validate_inputs(cfg: "PipelineConfig") -> list:
    """
    What a run needs from outside the config itself; empty means it is all
    there. A missing polygon used to be silent: the mine mask came out empty
    (significance.panel_mask) and the mines joined the farmland null, or the
    run stopped mid-way at the deramp or at the very end at the report.
    """
    problems = []
    backend = str(getattr(cfg, "backend", "")).lower()
    if backend in ("stack", "miaplpy"):
        t = getattr(cfg, "track", None)
        if t is None or str(t).strip() == "":
            problems.append(f"track is required on the {backend} route (a stack is coregistered on ONE relative orbit; "
                            "the builder's archive check lists them)")
        elif not (str(t).isdigit() and int(t) in _S1_TRACKS):
            problems.append(f"track = {t!r} is not a Sentinel-1 relative orbit (1-175)")
    for name in _INPUT_FILES:
        v = getattr(cfg, name, None)
        if v and str(v).strip() and not Path(str(v)).exists():
            problems.append(f"{name}: {v} does not exist")
    if cfg.mined_year_geojson and not cfg.mined_year_field:
        problems.append("mined_year_geojson is set but mined_year_field (the property holding the year) is not")
    vp = str(getattr(cfg, "validation_protocol", "") or "").strip()
    if vp and not Path(vp).is_absolute():
        # the harness opens it from wherever the run starts (the dashboard
        # starts runs in the package's parent folder), as with report polygons
        problems.append(f"validation_protocol {vp} is a relative path; give it in full "
                        "(or as $INSAR_PIPELINE/review_2026-10-08/...)")

    deramps = backend in ("stack", "miaplpy") or (backend == "disp" and getattr(cfg, "disp_record_chain", False))
    if deramps and str(cfg.mintpy_deramp).lower() != "no" and not (cfg.deramp_exclude_geojson or cfg.mintpy_deramp_mask):
        proj_ctx = Path(cfg.project_dir) / "deliverables" / "context"
        if not (proj_ctx.exists() and any(proj_ctx.glob("*active*.geojson"))):
            problems.append(f"mintpy_deramp = {cfg.mintpy_deramp!r} needs deramp_exclude_geojson (the mines and known "
                            "movement kept out of the ramp fit) - or set mintpy_deramp: no. Without it the plane is "
                            "fitted through the deformation and subtracted from it.")

    bbox = list(cfg.bbox or [])
    for spec in getattr(cfg, "feature_layers", ()) or ():
        try:
            kind, val = domains.split_feature_layer(spec)
        except ValueError:
            continue                                       # validate_config reports it
        p = Path(val)
        if re.match(r"^[A-Za-z]:[\\/]|^\\\\", val):
            problems.append(f"feature_layers: {kind!r} layer {val} is a Windows path; give it as this machine "
                            "sees it (e.g. /mnt/c/...)")
        elif not p.is_absolute():
            problems.append(f"feature_layers: {kind!r} layer {val} is a relative path; give it in full")
        elif not p.exists():
            problems.append(f"feature_layers: {kind!r} layer {val} does not exist")
        else:
            try:
                b = _geojson_bounds(p)
            except (OSError, ValueError, AttributeError, TypeError, IndexError) as exc:
                problems.append(f"feature_layers: {kind!r} layer {p} is not readable GeoJSON ({exc})")
                continue
            if b is None:
                problems.append(f"feature_layers: {kind!r} layer {p} has no coordinates")
            elif len(bbox) == 4 and not _overlaps(b, bbox):
                problems.append(f"feature_layers: {kind!r} layer {p} lies outside the bbox {bbox}")
    for field_name, allow_full in (("report_areas", True), ("focus_sites_lalo", False)):
        for spec in getattr(cfg, field_name, ()) or ():
            try:
                name, kind, val = split_area_spec(spec, allow_full)
            except ValueError:
                continue                                   # validate_config reports it
            if kind != "geojson":
                continue
            p = Path(val)
            if not p.is_absolute():
                # the report opens it from wherever the run starts, not from
                # the project folder: a relative path passed here and failed there
                problems.append(f"{field_name}: {name!r} polygon {val} is a relative path; give it in full "
                                "(or as $INSAR_PROJECTS/...)")
                continue
            if not p.exists():
                problems.append(f"{field_name}: {name!r} polygon {val} does not exist")
                continue
            try:
                b = _geojson_bounds(p)
            except (OSError, ValueError, AttributeError, TypeError, IndexError) as exc:
                problems.append(f"{field_name}: {name!r} polygon {p} is not readable GeoJSON ({exc})")
                continue
            if b is None:
                problems.append(f"{field_name}: {name!r} polygon {p} has no coordinates")
            elif len(bbox) == 4 and not _overlaps(b, bbox):
                problems.append(f"{field_name}: {name!r} polygon {p} lies outside the bbox {bbox}")
    return problems


def _yaml_scalars(raw: dict) -> dict:
    """YAML reads an unquoted 2016-01-01 as a date and 20210219 as an int;
    the pipeline compares and formats these fields as strings."""
    out = dict(raw)
    for k in ("start_date", "end_date"):
        if isinstance(out.get(k), (_dt.date, _dt.datetime)):
            out[k] = out[k].strftime("%Y-%m-%d")
    for k in ("stack_reference_date", "acceptance_event_start", "acceptance_event_end"):
        if isinstance(out.get(k), (_dt.date, _dt.datetime)):
            out[k] = out[k].strftime("%Y%m%d")
        elif isinstance(out.get(k), int) and not isinstance(out.get(k), bool):
            out[k] = str(out[k])
    for k in ("allow_missing_dates", "stack_exclude_dates"):
        if isinstance(out.get(k), list):
            out[k] = [v.strftime("%Y%m%d") if isinstance(v, _dt.date) else str(v) for v in out[k]]
    rev = out.get("report_revision")
    if isinstance(rev, dict) and isinstance(rev.get("supersedes"), (_dt.date, _dt.datetime)):
        out["report_revision"] = {**rev, "supersedes": rev["supersedes"].strftime("%Y-%m-%d")}
    elif rev is None and "report_revision" in out:
        out["report_revision"] = {}                 # "report_revision:" with nothing under it = a first issue
    return out


# ------------------------------------------------------------------ value types (16 Oct 2026, readiness review HND-03)
# Values used to reach PipelineConfig as YAML typed them, unchecked: the text
# 'no' or 'false' in a true/false setting read as True (any non-empty text is
# true), a quoted '0.6' stayed text, 2.7 stayed 2.7 in a count, 'high' for a
# number failed later as a bare ValueError that callers catching ConfigError
# never saw. _check_types holds each value to its field's declared type. str,
# dict and object fields are left exactly as YAML read them: several pass text
# through to ISCE/MintPy (stack_virtual_merge 'False', mintpy_deramp 'no'),
# and validate_config checks their values.
_BOOL_WORDS = {"true": True, "yes": True, "on": True, "1": True,
               "false": False, "no": False, "off": False, "0": False}
# PyYAML follows YAML 1.1, which reads a number with an exponent and no dot
# (1e-3, 2E5) as TEXT. Written without quotes it is plainly a number, so a
# number field takes it as one; quoted, it is text like any other.
_EXP_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+")


def _yaml_key_lines(text: str) -> dict:
    """{top-level key: (line, plain)}: the line each setting is written on (for
    the messages) and whether its value was written without quotes."""
    try:
        node = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return {}
    out = {}
    if isinstance(node, yaml.MappingNode):
        for k, v in node.value:
            if isinstance(k, yaml.ScalarNode):
                out.setdefault(k.value, (k.start_mark.line + 1, isinstance(v, yaml.ScalarNode) and v.style is None))
    return out


def _field_kind(tp) -> tuple:
    """(kind, optional) of a PipelineConfig annotation. kind: bool, int, float,
    "path", "numbers" (Sequence[float]: bbox), "list" (tuple / list fields), or
    None for the types left as YAML read them (str, dict, object)."""
    optional = False
    if typing.get_origin(tp) is typing.Union:
        args = typing.get_args(tp)
        rest = [a for a in args if a is not type(None)]
        optional, tp = len(rest) < len(args), rest[0]
    if tp in (bool, int, float):
        return tp, optional
    if tp is Path:
        return "path", optional
    if typing.get_origin(tp) is collections.abc.Sequence:
        return "numbers", optional
    if tp in (tuple, list):
        return "list", optional
    return None, optional


def _number(v, plain: bool = False):
    """v when YAML gave a number (never a bool); an unquoted 1e-3 as its float;
    None for anything else."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if plain and isinstance(v, str) and _EXP_NUMBER.fullmatch(v.strip()):
        return float(v)
    return None


def _quotes_hint(v, plain: bool) -> str:
    """' - remove the quotes' for a number written in quotes ('0.6')."""
    num = r"\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*"
    return " - remove the quotes" if isinstance(v, str) and not plain and re.fullmatch(num, v) else ""


def _check_types(raw: dict, lines: dict) -> tuple:
    """(values, problems): every value held to its field's declared type.
    true/false settings take true/false/yes/no/on/off/1/0 in any case, quoted
    or not, as the setting they name; counts take whole numbers (2.0 -> 2, never
    2.7 or a bool); numbers take numbers, not text; None only where the field
    is Optional, except that an empty list setting is an empty list (as an empty
    report_revision is no revision). Values already of the right type are
    returned unchanged (an int in a float field stays that int)."""
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(PipelineConfig)}
    out, problems = dict(raw), []
    for name, v in raw.items():
        f = fields.get(name)
        if f is None:
            continue
        kind, optional = _field_kind(f.type)
        if kind is None:
            continue
        line, plain = lines.get(name, (None, False))
        at = f" (line {line})" if line else ""
        if v is None or (optional and kind in (int, float) and isinstance(v, str) and not v.strip()):
            if optional:
                out[name] = None                      # e.g. track: "" - validate_inputs reads it as not set
            elif kind == "list":
                out[name] = f.type()                  # "exclude_pairs:" with nothing under it = none
            else:
                problems.append(f"{name}{at} is empty; give a value" +
                                ("" if name in ("project_dir", "bbox") else ", or delete the line for the default"))
        elif kind is bool:
            if isinstance(v, bool):
                continue
            word = str(v).strip().lower() if isinstance(v, (str, int)) else None
            if word in _BOOL_WORDS:
                out[name] = _BOOL_WORDS[word]
            else:
                problems.append(f"{name}{at} = {v!r} is not true or false (write true or false; yes/no, on/off "
                                "and 1/0 are read the same)")
        elif kind is int:
            n = _number(v, plain)
            if isinstance(n, int) or isinstance(n, float) and n.is_integer():
                out[name] = n if isinstance(n, int) else int(n)
            else:
                problems.append(f"{name}{at} = {v!r} must be a whole number{_quotes_hint(v, plain)}")
        elif kind is float:
            n = _number(v, plain)
            if n is None:
                problems.append(f"{name}{at} = {v!r} must be a number{_quotes_hint(v, plain)}")
            else:
                out[name] = n
        elif kind == "path":
            if not isinstance(v, str) or not (optional or v.strip()):
                problems.append(f"{name}{at} = {v!r} must be a path to a folder or file")
        elif kind == "numbers" or (kind == "list" and name == "stable_area_bbox"):
            if not isinstance(v, (list, tuple)) or any(_number(x) is None for x in v):
                problems.append(f"{name}{at} = {v!r} must be numbers, [min_lon, min_lat, max_lon, max_lat]")
        elif kind == "list" and not isinstance(v, (list, tuple)):
            problems.append(f"{name}{at} = {v!r} must be a list: one '- item' per line, or [item, item]")
    return out, problems


def _recorded_reference(cfg) -> None:
    """
    Carry the reference choice a run recorded (miaplpy_backend.choose_reference
    -> stack/reference_choice.json) into every later process - MintPy onward,
    products only, results - so they all reference the same pixel:
      - no reference in the YAML: use the run's choice;
      - the YAML reference is the one the run REJECTED (unwrapped in < 80% of
        pairs) and the YAML still says it: use the run's replacement. Before 5
        Oct 2026 only the first case was handled, so 'MintPy onward' went back
        to the rejected pixel and then failed telling the user to move it.
    A YAML reference that differs from the recorded one was set after the run
    and wins.
    """
    if cfg.backend not in ("stack", "miaplpy"):
        return
    try:
        rc = json.loads((cfg.stack_dir / "reference_choice.json").read_text())
    except (OSError, ValueError):
        return
    try:
        yaml_ref = str(cfg.reference_lalo or "").strip()
        conf = rc.get("configured") or None
        chosen = rc.get("chosen") or None
        if not yaml_ref:
            ch = chosen or (conf if rc.get("kept_configured") else None)
            if ch:
                cfg.reference_lalo = f"{ch['lat']:.6f},{ch['lon']:.6f}"
                cfg.reference_auto = True            # not in the YAML: chosen by the run's rule
            return
        if rc.get("kept_configured") or not chosen or not conf:
            return
        lat, lon = (float(v) for v in yaml_ref.split(","))
        # the run parsed the same YAML string; 1e-5 deg (~1 m) absorbs printing
        if abs(lat - float(conf["lat"])) < 1e-5 and abs(lon - float(conf["lon"])) < 1e-5:
            cfg.reference_lalo = f"{chosen['lat']:.6f},{chosen['lon']:.6f}"
            cfg.reference_auto = True
            cfg.reference_yaml_rejected = yaml_ref   # for provenance: what the YAML asked for
            log.warning("reference_lalo %s in the YAML was rejected by the run (unwrapped in %.0f%% of pairs); "
                        "using the run's choice %s recorded in %s", yaml_ref,
                        100 * float(conf.get("fraction_of_pairs_unwrapped", 0)), cfg.reference_lalo,
                        cfg.stack_dir / "reference_choice.json")
    except (KeyError, TypeError, ValueError):
        pass


class _DuplicateKey(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    """yaml.safe_load, except that a key given twice in one block is an error.
    PyYAML keeps the last one silently: on 7 Oct 2026 a second report_revision
    block written under the first in config_site_a_pl.yaml replaced it
    with nothing to say so, and a setting edited in one place and repeated
    lower down would be ignored the same way. Only keys written out count: a
    YAML merge ('<<: *defaults') is left to PyYAML, as yaml.safe_load reads it,
    and a written key may override a merged one."""

    def construct_mapping(self, node, deep=False):
        seen = {}
        if isinstance(node, yaml.MappingNode):
            for key_node, _v in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    continue
                key = self.construct_object(key_node, deep=deep)
                line = key_node.start_mark.line + 1
                try:
                    if key in seen:
                        where = f"line {line}" if seen[key] == line else f"lines {seen[key]} and {line}"
                        raise _DuplicateKey(f"the setting {key!r} is given twice ({where}); YAML would keep only "
                                            "the second without a word - remove one")
                    seen[key] = line
                except TypeError:                     # an unhashable key: PyYAML reports it itself
                    continue
        return super().construct_mapping(node, deep)


def _yaml_problem(exc) -> str:
    """A YAML syntax error in words, with the line and column it was found at."""
    mark = getattr(exc, "problem_mark", None)
    where = f"line {mark.line + 1}, column {mark.column + 1}: " if mark is not None else ""
    what = "; ".join(str(x) for x in (getattr(exc, "context", None), getattr(exc, "problem", None)) if x)
    return (f"{where}not valid YAML ({what or exc}). Check that line and the one above it: the indentation, "
            "a missing ':' after a setting's name, an unclosed quote or bracket.")


def load_config(path: str) -> PipelineConfig:
    import dataclasses
    import difflib

    # Every problem with the file is a ConfigError naming the file and, where
    # there is one, the line (16 Oct 2026, readiness review HND-03): a YAML
    # syntax error used to escape as a raw yaml ParserError, a wrong type as a
    # bare ValueError/TypeError, and callers catching ConfigError missed both.
    with open(path) as f:
        try:
            text = f.read()
        except UnicodeDecodeError as exc:
            raise ConfigError(f"{path}: not a text file ({exc})") from None
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader) or {}
    except _DuplicateKey as exc:
        raise ConfigError(f"{path}: {exc}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: {_yaml_problem(exc)}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: a config is settings written one 'name: value' per line; this file holds a "
                          f"YAML {type(raw).__name__}")
    known = {f.name for f in dataclasses.fields(PipelineConfig)}
    unknown = sorted(set(raw) - known, key=str)
    if unknown:
        hints = []
        for k in unknown:
            close = difflib.get_close_matches(str(k), known, n=1)
            hints.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ConfigError(f"{path}: unknown setting(s): " + ", ".join(hints))
    missing = [k for k in ("project_dir", "bbox", "start_date", "end_date") if k not in raw]
    if missing:
        raise ConfigError(f"{path}: missing required setting(s): {missing}")
    from . import settings                       # sets $INSAR_PIPELINE / $INSAR_PROJECTS
    settings.export_env()
    raw, problems = _check_types(_yaml_scalars(raw), _yaml_key_lines(text))
    if problems:
        raise ConfigError(f"{path}:\n  " + "\n  ".join(problems))

    def _expand(v):
        if isinstance(v, str) and ("$" in v or v.startswith("~")):
            return os.path.expanduser(os.path.expandvars(v))
        if isinstance(v, str) and (v.startswith(("\\\\", "//wsl")) or (len(v) > 2 and v[1] == ":" and v[2] in "\\/")):
            from .areas import normalize_path          # a Windows path pasted from Explorer
            return str(normalize_path(v))
        if isinstance(v, str) and ":~" in v:
            # "Name:~/areas/x.geojson" (report_areas, focus sites; "$VAR" is expanded above)
            name, rest = v.split(":", 1)
            return f"{name}:{os.path.expanduser(rest)}"
        if isinstance(v, list):
            return [_expand(x) for x in v]
        return v
    raw = {k: _expand(v) for k, v in raw.items()}
    raw["project_dir"] = Path(raw["project_dir"])
    if raw.get("dem_path"):
        raw["dem_path"] = Path(raw["dem_path"])
    cfg = PipelineConfig(**raw)
    try:
        problems = validate_config(cfg) + validate_inputs(cfg)
    except (TypeError, ValueError, AttributeError) as exc:
        # the types are checked above, so this is a value no check foresaw:
        # still the config's problem, and still a ConfigError for the caller
        raise ConfigError(f"{path}: a setting could not be checked ({type(exc).__name__}: {exc})") from exc
    if problems:
        raise ConfigError(f"{path}:\n  " + "\n  ".join(problems))
    _recorded_reference(cfg)
    # The exact YAML this config came from, for provenance (its sha256 and
    # whether it was edited after the product). A plain attribute, not a field:
    # it is not a setting and must never be written back into a config.
    cfg._source_path = Path(path).resolve()
    return cfg
