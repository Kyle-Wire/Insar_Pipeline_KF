"""Synthetic golden fixture projects (domain-studies plan, Phase 0 S0.2, 7 Nov
2026): one small geocoded project per report path that the real projects on
disk cannot reach (no mine layers there).

Each fixture is tests/test_fix_L5.make_project's farmland (120 x 120 cells
of ~100 m, 280 dates over 9.2 years, seed 7, the unmapped sinking FEATURE,
the Test Field and the 70%-unmeasured Wet Pit), plus one change:

  base               nothing added: no mine layers, the official route
  sendrule_farmland  QA record with a send rule and two flagged checks
  longwall           the same QA record with the mine_longwall site profile
  ug_over            underground workings over the feature
  ug_near            underground workings ~0.3 km beside it
  ug_far             underground workings ~5 km away
  surface_on         surface mines over the feature, other movement elsewhere
  other_on           other known movement over the feature, surface elsewhere
  surface_reshaped   surface_on + a DEM-error layer: ground reshaped on a mine
  reshaped_off_mine  base + the DEM-error layer: reshaped with no mine layer
  rising             a steady +20 mm/yr LOS rising patch beside the feature
  surface_rising     rising + surface mines over the rising patch
  dated              four dated mined areas (years-since-mining table)
  disp               the DISP-S1 screening route
  disp_qa            disp + a DISP-shaped QA record (stitch, network, dates)
  ug_over_lc         ug_over + a land-cover map, trees on ~40% of the workings
  surface_on_lc      surface_on + a land-cover map, trees on ~40% of the surface mine

Nothing here downloads anything. Every random draw is seeded. File times
are pinned (golden_harness.FROZEN), so the staleness checks see inputs
written before the report, in a fixed order. Names in the data are
synthetic.

write(root, name) writes the project once; config(root, name) is the
PipelineConfig the harness builds with (it writes nothing).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_fix_L5 as L5                                    # noqa: E402  (the fixture this builds on)

from insar_pipeline import significance as sg               # noqa: E402
from insar_pipeline.config import PipelineConfig            # noqa: E402

SEED = 7
RISE = (slice(60, 75), slice(95, 110))           # 15 x 15 cells, > 1 km from FEATURE
RISE_LOS_MM_YR = 20.0                            # listed as Feature 1 on seed 7 (k = 0 of 199)
OTHER_BOX = (slice(55, 75), slice(50, 70))       # farmland clear of FEATURE, FIELD_BOX, PIT_BOX
UG_OVER = (slice(28, 47), slice(68, 87))         # the detected patch is ~3.2 km2, larger than FEATURE
UG_NEAR = (slice(28, 47), slice(88, 95))         # ~0.3 km east of the detected patch (< 0.5 km)
UG_FAR = (slice(100, 115), slice(95, 115))       # ~5 km away
DATED = (("North pit", 2018, (slice(55, 65), slice(5, 15))),
         ("North pit", 2019, (slice(55, 65), slice(20, 30))),
         ("South pit", 2020, (slice(105, 115), slice(60, 70))),
         ("South pit", 2021, (slice(105, 115), slice(85, 95))))
DATED_LOS_MM_YR = -6.0                            # settling from each area's mining year on
LC_WOODS = (slice(100, 120), slice(0, 15))        # a wood away from every report area and layer
LC_GRASS = (slice(0, 4), slice(30, 120))          # a strip of grassland along the north edge

SOURCES = {"underground": "Underground mine workings: Synthetic survey, workings mapped to 2016",
           "surface": "Surface mines: Synthetic survey surface pits, 2019",
           "other": "Other known ground movement: Synthetic notes, 2020",
           "dated": "Dated mined areas: Synthetic state agency, annual mined-area polygons 2018-2021"}
LAYER_FIELD = {"underground": "underground_mines_geojson", "surface": "surface_mines_geojson",
               "other": "other_movement_geojson"}

_QA_SENDRULE = {"send_rule": {},
                "checks": {"distributed_scatterer_bias": {"pass": False, "low_minus_high_coherence_mm_yr": -4.4,
                                                          "velocity_by_coherence_class": []},
                           "reference_stability": {"pass": False}}}
_LONGWALL = {"name": "mine_longwall", "where": "at a longwall mining site", "ground": "pasture",
             "expected": ["rate_precision", "distributed_scatterer_bias"], "handled": False,
             "whole_area_moves": False}

# name -> what it adds to the base project
FIXTURES: dict[str, dict] = {
    "base": {},
    "sendrule_farmland": {"qa": _QA_SENDRULE},
    "longwall": {"qa": {**_QA_SENDRULE, "site_profile": _LONGWALL}},
    "ug_over": {"layers": {"underground": [UG_OVER]}},
    "ug_near": {"layers": {"underground": [UG_NEAR]}},
    "ug_far": {"layers": {"underground": [UG_FAR]}},
    "surface_on": {"layers": {"surface": [L5.FEATURE], "other": [OTHER_BOX]}},
    "other_on": {"layers": {"other": [L5.FEATURE], "surface": [OTHER_BOX]}},
    "surface_reshaped": {"layers": {"surface": [L5.FEATURE], "other": [OTHER_BOX]}, "dem_error": True},
    "reshaped_off_mine": {"dem_error": True},
    "rising": {"rise": True},
    "surface_rising": {"rise": True, "layers": {"surface": [RISE]}},
    "dated": {"dated": True},
    "disp": {"cfg": {"backend": "disp"}, "corrections": True},
    "disp_qa": {"cfg": {"backend": "disp"}, "corrections": True, "disp_qa": True},
    # the land-cover paths (wooded share, trees left out of the products and
    # the comparison ground, the Tree cover KMZ layer, the WorldCover credit)
    "ug_over_lc": {"layers": {"underground": [UG_OVER]},
                   "landcover": [(UG_OVER[0], slice(68, 76)), LC_WOODS]},
    "surface_on_lc": {"layers": {"surface": [L5.FEATURE], "other": [OTHER_BOX]},
                      "landcover": [(L5.FEATURE[0], slice(70, 76)), LC_WOODS]},
}


def project_dir(root, name) -> Path:
    return Path(root) / name / "proj"


def _ring(sl):
    s, w, n, e = (float(v) for v in L5._box_spec(sl).split(","))
    return [[w, s], [e, s], [e, n], [w, n], [w, s]]


def _fc(features):
    return {"type": "FeatureCollection", "features": features}


def _poly(sl, props=None):
    return {"type": "Feature", "properties": props or {}, "geometry": {"type": "Polygon", "coordinates": [_ring(sl)]}}


def config(root, name) -> PipelineConfig:
    """The fixture's config (what make_project gives, the basemap off, plus the
    fixture's layers). Paths are absolute: report_insights.mine_layers and
    significance.panel_mask check deramp_exclude_geojson as given."""
    spec = FIXTURES[name]
    P = project_dir(root, name)
    ctx = P / "context"
    kw = dict(project_dir=P, bbox=L5.BBOX, start_date="2016-01-01", end_date="2025-03-31",
              backend="miaplpy", acceptance_far_field_buffer_m=500.0,
              project_title="Testsite", project_location="Test County, Indiana",
              report_areas=("Full Satellite Imaging:full", f"Test Field:{L5._box_spec(L5.FIELD_BOX)}",
                            f"Wet Pit:{L5._box_spec(L5.PIT_BOX)}"),
              report_basemap="none")
    sources = []
    layers = spec.get("layers") or {}
    for key in ("underground", "surface", "other"):          # areas.write_project's order
        if key in layers:
            kw[LAYER_FIELD[key]] = str(ctx / f"{key}.geojson")
            sources.append(SOURCES[key])
    if layers:
        kw["deramp_exclude_geojson"] = str(ctx / "deforming_area.geojson")   # the builder's union
    if spec.get("dated"):
        kw.update(mined_year_geojson="context/mine_panels.geojson", mined_year_field="mined_year",
                  mined_name_field="mine_name", deramp_exclude_geojson=str(ctx / "mine_panels.geojson"))
        sources.append(SOURCES["dated"])
    if sources:
        kw["mine_layer_sources"] = tuple(sources)
    kw.update(spec.get("cfg") or {})
    return PipelineConfig(**kw)


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) and v else v
    return out


def write(root, name, frozen: datetime) -> Path:
    """Write the fixture project (refuses to overwrite one that exists)."""
    spec = FIXTURES[name]
    cfg = config(root, name)
    P = cfg.project_dir
    if P.exists():
        raise FileExistsError(f"{P} exists: a golden's project is written once")
    geo = cfg.mintpy_dir / "geo"
    dates = L5._dates()
    t = sg.years(dates)
    rng = np.random.default_rng(SEED)
    NY, NX = L5.NY, L5.NX
    # --- as L5.make_project (same draws, same order) ---
    rate = gaussian_filter(rng.normal(0, 1, (NY, NX)), 3)
    rate = rate / rate.std() * 2.0
    ts = np.empty((len(dates), NY, NX), np.float32)
    for i in range(len(dates)):
        atm = gaussian_filter(rng.normal(0, 1, (NY, NX)), 6)
        ts[i] = (rate * t[i] + atm / atm.std() * 3.0 + rng.normal(0, 2.0, (NY, NX))
                 + 6.0 * np.sin(2 * np.pi * t[i])) / 1000.0
    ts[:, L5.FEATURE[0], L5.FEATURE[1]] += (L5.feature_truth(t) / 1000.0)[:, None, None]
    # --- the fixture's plants, before the velocity fit ---
    if spec.get("rise"):
        ts[:, RISE[0], RISE[1]] += (RISE_LOS_MM_YR * (t - t[0]) / 1000.0)[:, None, None]
    if spec.get("dated"):
        d0 = datetime.strptime(dates[0], "%Y%m%d")
        t_abs = d0.year + (d0.timetuple().tm_yday - 1) / 365.25 + t
        for _mine, year, sl in DATED:
            ts[:, sl[0], sl[1]] += (DATED_LOS_MM_YR * np.clip(t_abs - year, 0, None) / 1000.0)[:, None, None]
    vel = np.polyfit(t, ts.reshape(len(t), -1), 1)[0].reshape(NY, NX).astype(np.float32)
    A = dict(L5._attrs(), START_DATE=dates[0], END_DATE=dates[-1], REF_DATE=dates[0])
    mask = np.ones((NY, NX), bool)
    pit = np.zeros((NY, NX), bool)
    pit[L5.PIT_BOX] = True
    rr, cc = np.nonzero(pit)
    mask[rr[: int(0.7 * rr.size)], cc[: int(0.7 * rr.size)]] = False
    L5._h5(geo / "geo_velocity.h5", {"velocity": vel}, A)
    L5._h5(geo / "geo_timeseries_ERA5_ramp_demErr.h5", {"timeseries": ts}, A, dates)
    L5._h5(geo / "geo_maskTempCoh.h5", {"mask": mask}, A)
    L5._h5(geo / "geo_temporalCoherence.h5", {"temporalCoherence": np.full((NY, NX), 0.95, np.float32)}, A)
    L5._h5(geo / "geo_geometryRadar.h5", {"incidenceAngle": np.full((NY, NX), 39.5, np.float32),
                                          "azimuthAngle": np.full((NY, NX), -100.0, np.float32),
                                          "height": np.zeros((NY, NX), np.float32)}, A)
    if spec.get("dem_error"):
        dh = np.random.default_rng(SEED + 1000).normal(0, 1.0, (NY, NX)).astype(np.float32)
        dh[L5.FEATURE] += 30.0
        L5._h5(geo / "geo_demErr.h5", {"dem": dh}, A)
    # --- layers ---
    ctx = P / "context"
    layers = spec.get("layers") or {}
    if layers:
        ctx.mkdir(parents=True, exist_ok=True)
        union = []
        for key in ("underground", "surface", "other"):
            if key in layers:
                feats = [_poly(sl) for sl in layers[key]]
                (ctx / f"{key}.geojson").write_text(json.dumps(_fc(feats)))
                union += feats
        (ctx / "deforming_area.geojson").write_text(json.dumps(_fc(union)))
    if spec.get("dated"):
        ctx.mkdir(parents=True, exist_ok=True)
        (ctx / "mine_panels.geojson").write_text(json.dumps(_fc(
            [_poly(sl, {"mined_year": year, "mine_name": mine}) for mine, year, sl in DATED])))
    if spec.get("landcover"):
        _write_landcover(ctx / "landcover_worldcover.tif", spec["landcover"])
    if spec.get("corrections"):
        (cfg.mintpy_dir / "corrections.json").write_text(json.dumps(
            {"solid_earth_tides": True, "ionosphere": "product layer",
             "disp_corrections": {"status": "all", "epochs": len(dates), "epochs_corrected": len(dates)}}))
    # --- the JSON records, as make_project writes them ---
    D = P / "deliverables"
    (D / "figures").mkdir(parents=True, exist_ok=True)
    (D / "summary.json").write_text(json.dumps({"total_measured_points": NY * NX, "track": 92,
                                                "orbit_direction": "ASCENDING", "look_angle_deg": 39.5}))
    (D / "uncertainty_model.json").write_text(json.dumps({
        "by_class": [{"tc": "0.9-1.0", "class_index": 4, "empirical_mm_yr": 1.0, "n": 14000}],
        "cell_null_by_class": {"4": 2.5}}))
    qa = {"checks": {"coverage": {"aoi_percent_at_threshold": 97.0, "pass": True},
                     "network": {"connected": True, "pass": True}},
          "findings": {}, "passed": 2, "flagged": 0, "verdict": "MEETS PRACTICE"}
    if spec.get("qa"):
        qa = _merge(qa, spec["qa"])
    if spec.get("disp_qa"):
        from insar_pipeline import quality_assessment as qam
        dc = {"stitch": {"epochs": len(dates), "skipped_granules": [dates[40]], "lost_anchors": [],
                         "boundary_datum": {}}}
        qa = _merge(qa, {"send_rule": {},
                         "checks": {"network": {"network_type": "OPERA DISP-S1: per-frame displacement, "
                                                                "stitched in time", "pass": None},
                                    "stitch": qam.stitch_check(dc, cfg)},
                         # two skipped (the plural wording), one without a baseline (the singular)
                         "dates": {"available": len(dates) + 3, "processed": len(dates),
                                   "dropped": {"20180110": qam._DISP_SKIPPED, "20180122": qam._DISP_SKIPPED,
                                               "20190105": qam._DISP_NO_BASELINE}}})
    (P / "quality_assessment.json").write_text(json.dumps(qa))
    (D / "gnss_tie.json").write_text(json.dumps({
        "record_window": [dates[0], dates[-1]], "stations": [], "inside_checks": [], "los_up_median": L5.UP,
        "uncertainty_floor_mm_yr": L5.BG_U,
        "background": {"vertical_mm_yr": L5.BG_V, "vertical_uncertainty_mm_yr": L5.BG_U,
                       "los_mm_yr": L5.BG_V * L5.UP, "uncertainty_basis": "regional floor"}}))
    pin_times(P, frozen)
    return P


def _write_landcover(path: Path, trees) -> None:
    """A WorldCover-scheme class map on the product grid, one class cell per
    product cell, where production downloads it (context/, landcover.ensure):
    cropland (40) with tree cover (10) on the given boxes and grassland (30)
    along the north edge. No compression options that carry a time."""
    import rasterio
    from rasterio.transform import from_origin
    from insar_pipeline import landcover
    cls = np.full((L5.NY, L5.NX), 40, np.uint8)
    cls[LC_GRASS] = 30
    for sl in trees:
        cls[sl] = 10
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=L5.NX, height=L5.NY, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(L5.X0, L5.Y0, L5.DX, abs(L5.DY)), nodata=0) as dst:
        dst.write(cls, 1)
        dst.update_tags(source=landcover.WORLDCOVER_CREDIT, scheme="worldcover")


def _expected_time(p: Path, frozen: datetime) -> float:
    t0 = (frozen - timedelta(days=2)).replace(tzinfo=timezone.utc).timestamp()   # FROZEN is UTC
    return t0 + (10 if p.suffix == ".json" else 0)


def check_times(P: Path, frozen: datetime) -> list[str]:
    """Project files whose time is not the pinned one (a copy that dropped
    the times): the staleness checks read them, so the build is refused."""
    return [str(p.relative_to(P)) for p in sorted(P.rglob("*"))
            if p.is_file() and abs(p.stat().st_mtime - _expected_time(p, frozen)) > 1e-3]


def pin_times(P: Path, frozen: datetime) -> None:
    """Inputs two days before the frozen report time; the JSON records ten
    seconds after the rasters (so none reads as older than the velocity)."""
    for p in sorted(P.rglob("*")):
        if p.is_file():
            t = _expected_time(p, frozen)
            os.utime(p, (t, t))
