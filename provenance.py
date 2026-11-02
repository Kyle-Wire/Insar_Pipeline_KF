"""Record what produced a result, so the result can be defended later.

Written next to the deliverables at export time. Software versions,
granules, thresholds and the reference point are cheap to capture while
the run is happening and effectively impossible to reconstruct months
later when someone asks why a number is what it is.

Until 5 Oct 2026 every route-specific block was gated on backend == "stack",
so the official (phase-linked, miaplpy) route recorded n_pairs 0, no dates,
no MiaplPy build or settings and "not recorded by the processor" for the
unwrap-error and tropospheric corrections - while stack/SAFE_files.txt (270
granules), mintpy/inputs/ifgramStack.h5 (804 pairs), unwrap_error_applied.json
and the patch check in the run log all said otherwise. The record now reads
those records: the network from ifgramStack.h5 (which survives cleanup), the
MiaplPy settings as requested AND as run (miaplpyApp.cfg), the patch check,
the unwrap engine from the per-pair unwrap records, the DEM source, a
snapshot of the resolved config with its sha256, the code's version and hash
(version.describe()), who ran it where, and - via record_stages() after the
client report - the QA verdict, the GNSS tie and the report that followed.
From 7 Oct 2026 (trust review) the stages also carry what the tests found
about themselves: each area's local and coherence-matched nulls and
pace-change test (the mapped mine ground's nulls too), each feature's
chance and "stands out clearly", the
feature copies against the real farmland, the GNSS series' screening, and
the injection test's decision rule. From 16 Oct 2026 (readiness review
OUT-02) 'builds' lists every build that wrote the record, appended across
runs: a products-only rerun no longer erases the build of the processing
run (build_history, add_build).
"""
import hashlib
import json
import logging
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import PipelineConfig

log = logging.getLogger(__name__)

# The routes that coregister with ISCE2 topsStack and unwrap pairs under
# stack/merged/interferograms. Every block below that describes coregistration,
# pairs or unwrapping applies to both; miaplpy adds phase linking on top.
TOPSSTACK_ROUTES = ("stack", "miaplpy")


def _version(module: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(module)
    except Exception:  # noqa: BLE001
        return "unknown"


def _isce_version() -> str:
    # Import, don't shell out: `topsApp.py --help` makes ISCE write isce.log
    # into the current directory on every export.
    try:
        import isce
        v = getattr(isce, "__version__", None) or getattr(isce, "release_version", None)
        if v:
            return str(v)
    except Exception:                                    # noqa: BLE001
        pass
    return _version("isce2")


def _load(p: Path):
    """A JSON record, or None when absent or unreadable."""
    try:
        return json.loads(Path(p).read_text()) if Path(p).exists() else None
    except (OSError, ValueError):
        return None


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def _effective_unwrap_error_method(cfg: PipelineConfig) -> str:
    """What the unwrap-error correction actually did, not what was asked for.

    mintpy/unwrap_error_applied.json is the run's own record (written when
    phase_closure is declined, as it was on Site B and Site A); it
    wins over re-deriving the answer from stack-route logic, which on the
    phase-linked route described a network that is not the one inverted."""
    rec = _load(cfg.mintpy_dir / "unwrap_error_applied.json")
    if rec and rec.get("applied"):
        return str(rec["applied"])
    try:
        from .stack_backend import _unwrap_error_method
        return _unwrap_error_method(cfg)
    except Exception:                                    # noqa: BLE001
        return cfg.mintpy_unwrap_error_method


def _ifgram_network(cfg: PipelineConfig) -> tuple[list, list, str | None]:
    """(dates, pairs, source) of the network MintPy loaded, from the 'date'
    dataset of mintpy/inputs/ifgramStack.h5 - a (n, 2) slice of a few kB,
    never the phase. It survives every cleanup level, unlike
    stack/merged/interferograms and coreg_secondarys (Site A no longer
    has either), so it is the primary record on every route that has it."""
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if not stack.exists():
        return [], [], None
    try:
        import h5py
        with h5py.File(stack, "r") as f:
            d12 = [(_s(a), _s(b)) for a, b in f["date"][:]]
    except Exception as exc:                                  # noqa: BLE001
        log.debug("ifgramStack.h5 date not read: %s", exc)
        return [], [], None
    return sorted({d for p in d12 for d in p}), [f"{a}_{b}" for a, b in d12], "mintpy/inputs/ifgramStack.h5"


def _h5_dates(path: Path) -> list | None:
    if not path.exists():
        return None
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return [_s(x) for x in f["date"][:]] if "date" in f else None
    except Exception:                                         # noqa: BLE001
        return None


def _safe_list(cfg: PipelineConfig) -> list:
    """Granules the stack was generated from (stack/SAFE_files.txt). The
    SAFEs themselves are deleted after coregistration, so the surviving
    archives (7 on Site B) are not the record; this list (270) is."""
    f = cfg.stack_dir / "SAFE_files.txt"
    if not f.exists():
        return []
    return [Path(l.strip()).name for l in f.read_text().splitlines() if l.strip()]


def _outcome(cfg: PipelineConfig) -> dict:
    """
    What the run actually did, read back out of the product it wrote.

    Distinct from the config, which records what was asked for. A
    tropospheric correction can be requested and then declined because it
    made the series worse; a reference point can be requested and then
    rejected as too gappy. A provenance record that only repeats the
    config would assert both happened.
    """
    out: dict = {}
    try:
        import h5py
        # topsStack does not geocode; MintPy's geocode step writes
        # geo/geo_velocity.h5 instead. Without this the record silently
        # degrades to "no velocity.h5 to read" on the one backend whose
        # settings most need recording.
        vel = cfg.mintpy_dir / "geo" / "geo_velocity.h5"
        if not vel.exists():
            vel = cfg.mintpy_dir / "velocity.h5"
        if not vel.exists():
            return {"note": "no velocity.h5 to read"}
        out["product_read"] = str(vel)
        with h5py.File(vel, "r") as f:
            attrs = dict(f.attrs)
            out["datasets"] = sorted(f.keys())
    except Exception as exc:  # noqa: BLE001
        return {"note": f"could not read the product: {exc}"}

    # Two families of attribute. The upper-case ones are written by the
    # processor or by this package's DISP path; MintPy copies its RESOLVED
    # template into the product under lower-case mintpy.* keys, and those
    # are the proof of what the inversion and corrections did on the stack
    # path. An absent key is reported as absent, not dropped: a record that
    # silently omits "deramp" reads as "no deramp".
    NOT_RECORDED = "not recorded by the processor"
    for key, label in (
        ("RLOOKS", "range_looks"),
        ("ALOOKS", "azimuth_looks"),
        ("FILTER_STRENGTH", "goldstein_filter_strength"),
        ("UNW_COST_MODE", "unwrap_cost_mode"),
        ("DEFOMAX", "unwrap_defomax_cycles"),
        ("DERAMP", "deramp"),
        ("TROPO_CORRECTION", "tropospheric_correction"),
        ("RATE_MODEL", "rate_model"),
        ("RATE_SIGMA_INFLATION", "rate_sigma_inflation"),
        ("INCIDENCE_ANGLE", "incidence_angle_deg"),
        ("INCIDENCE_ANGLE_SOURCE", "incidence_angle_source"),
        ("REF_LALO", "reference_point"),
        ("REF_Y", "reference_row"),
        ("REF_X", "reference_col"),
        ("REF_DATE", "reference_date"),
        ("FRAME_ID", "frame_id"),
        ("PROCESSOR", "processor"),
        ("mintpy.networkInversion.obsDatasetName", "inversion_observation_dataset"),
        ("mintpy.networkInversion.numIfgram", "inversion_n_interferograms"),
        ("mintpy.networkInversion.weightFunc", "inversion_weight"),
        ("mintpy.networkInversion.maskDataset", "inversion_mask_dataset"),
        ("mintpy.unwrapError.method", "unwrap_error_method_applied"),
        ("mintpy.troposphericDelay.method", "tropospheric_method_applied"),
        ("mintpy.deramp", "deramp_applied"),
        ("mintpy.deramp.maskFile", "deramp_mask_applied"),
        ("mintpy.topographicResidual", "topographic_residual_applied"),
        ("mintpy.timeFunc.polynomial", "rate_polynomial_order"),
    ):
        v = attrs.get(key)
        out[label] = _s(v) if v not in (None, "", "None") else NOT_RECORDED

    # The correction chain is spelled out in the file name MintPy chose.
    try:
        from .deliverables import timeseries_file
        ts = timeseries_file(cfg)
        tokens = [t for t in ts.stem.replace("geo_", "").split("_")[1:] if t]
        out["timeseries_product"] = ts.name
        out["corrections_in_product_name"] = tokens      # e.g. ERA5, ramp, demErr
        out["geocoded"] = ts.parent.name == "geo"
    except Exception as exc:                                  # noqa: BLE001
        out["timeseries_product"] = f"unresolved ({exc})"

    # The run's own records fill what the product's attributes leave out.
    # MintPy does not copy the unwrap-error or troposphere method into
    # geo_velocity.h5 on these routes, so the record said "not recorded"
    # although unwrap_error_applied.json and the product name said which.
    # Each fallback names its source.
    ua = _load(cfg.mintpy_dir / "unwrap_error_applied.json")
    if out.get("unwrap_error_method_applied") == NOT_RECORDED and ua and ua.get("applied"):
        out["unwrap_error_method_applied"] = (f"{ua['applied']} (from mintpy/unwrap_error_applied.json; "
                                              f"requested {ua.get('requested', '?')})")
    if out.get("tropospheric_method_applied") == NOT_RECORDED:
        rp = _load(cfg.mintpy_dir / "run_products.json") or {}
        applied = rp.get("corrections_applied")
        toks = out.get("corrections_in_product_name")
        if isinstance(applied, list):
            out["tropospheric_method_applied"] = (
                _weather_label(applied) + " (from mintpy/run_products.json corrections_applied)")
            if rp.get("era5_missing_dates"):
                out["era5_missing_dates"] = rp["era5_missing_dates"]
        elif isinstance(toks, list):
            out["tropospheric_method_applied"] = _weather_label(toks) + " (from the time-series product name)"

    # The network as loaded and as kept.
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if stack.exists():
        try:
            with h5py.File(stack, "r") as f:
                n = int(f["unwrapPhase"].shape[0]) if "unwrapPhase" in f else int(f["date"].shape[0])
                keep = f["dropIfgram"][:].astype(bool) if "dropIfgram" in f else None
                kept = int(keep.sum()) if keep is not None else n
                dropped = []
                if keep is not None and "date" in f:
                    # MintPy stores pairs as an (n, 2) `date` dataset, not `date12`
                    d12 = [f"{_s(a)}_{_s(b)}" for a, b in f["date"][:]]
                    dropped = [d for d, k in zip(d12, keep) if not k]
                out["network"] = {
                    "interferograms_loaded": n, "interferograms_inverted": kept,
                    "dropped": dropped,
                    "unwrap_phase_datasets": sorted(k for k in f.keys()
                                                    if k.startswith("unwrapPhase")),
                }
        except Exception as exc:                              # noqa: BLE001
            out["network"] = f"could not read ifgramStack.h5 ({exc})"

    # Deramp mask: the pixel count, not the intention.
    dm = cfg.mintpy_dir / "maskDeramp.h5"
    if dm.exists():
        try:
            with h5py.File(dm, "r") as f:
                m = f["mask"][:].astype(bool)
                da = dict(f.attrs)
            out["deramp_mask_pixels_kept"] = int(m.sum())
            out["deramp_mask_pixels_total"] = int(m.size)
            # How the fit area was built and how evenly it surrounds the area
            # (stack_backend.build_deramp_mask, from 5 Oct 2026; older masks
            # carry none of these and the keys are simply absent).
            for k in ("DERAMP_BUFFER_M", "DERAMP_BUFFER_KIND", "DERAMP_QUADRANTS", "DERAMP_QUADRANT_SHARE"):
                if k in da:
                    v = da[k]
                    v = v.decode() if isinstance(v, bytes) else v
                    if k == "DERAMP_QUADRANT_SHARE":
                        try:
                            v = json.loads(v)
                        except (TypeError, ValueError):
                            pass
                    out[f"deramp_mask_{k[7:].lower()}"] = v.item() if hasattr(v, "item") else v
        except Exception as exc:                              # noqa: BLE001
            out["deramp_mask_pixels_kept"] = f"unreadable ({exc})"
    return out


def _weather_label(tokens) -> str:
    """'pyaps (ERA5)', 'RAiDER (HRRR)' or 'none' from a correction chain's tokens."""
    from .corrections import WEATHER_TOKENS
    toks = [str(t) for t in tokens]
    w = next((t for t in WEATHER_TOKENS if t in toks), None)
    if w is None:
        w = next((t for t in WEATHER_TOKENS if any(t.lower() in x.lower() for x in toks)), None)
    return "none" if w is None else (f"pyaps ({w})" if w.startswith(("ERA", "MERRA", "NARR")) else f"RAiDER ({w})")


def _stack_outcome(cfg: PipelineConfig) -> dict:
    """What the topsStack stage actually produced, read from the stack dir."""
    out: dict = {}
    if cfg.backend not in TOPSSTACK_ROUTES:
        return out
    try:
        from . import stack_backend as sb
        from .unwrap_crop import WINDOW_FILE
    except Exception as exc:                                  # noqa: BLE001
        return {"note": f"stack_backend unavailable ({exc})"}
    s = cfg.stack_dir
    merged = s / "merged" / "interferograms"
    if not merged.exists():
        # After cleanup the pairs are gone; an empty scan would report an
        # unconnected zero-pair network for a run that inverted hundreds.
        return {"note": "stack/merged/interferograms is no longer on disk (cleanup); the network as "
                        "inverted is recorded under outcome.network and inputs (from ifgramStack.h5)"}
    pairs = sorted(d.name for d in merged.iterdir() if d.is_dir())
    unwrapped = [p for p in pairs if (merged / p / "filt_fine.unw").exists()
                 and (merged / p / "filt_fine.unw").stat().st_size > 0]
    out["pairs_merged"] = len(pairs)
    out["pairs_unwrapped"] = len(unwrapped)
    out["pairs_not_unwrapped"] = sorted(set(pairs) - set(unwrapped))
    try:
        out["pairs_quarantined"] = sb.quarantined_pairs(cfg)
    except Exception as exc:                                  # noqa: BLE001
        out["pairs_quarantined"] = f"not read ({exc})"
    dates = sorted({d for p in unwrapped for d in p.split("_")})
    try:
        connected, orphaned, comps = sb.check_network_connectivity(dates, unwrapped)
        out["dates_with_an_unwrapped_pair"] = len(dates)
        out["network_connected"] = connected
        out["orphaned_dates"] = orphaned
    except Exception as exc:                                  # noqa: BLE001
        out["network_connected"] = f"not checked ({exc})"
    # Re-verify the generated configs NOW rather than trusting a log line
    # from hours ago.
    try:
        problems = sb.verify_configs(cfg)
        out["configs_verified"] = not problems
        out["config_problems"] = problems[:10]
    except Exception as exc:                                  # noqa: BLE001
        out["configs_verified"] = f"not checked ({exc})"
    unw = sorted((s / "configs").glob("config_igram_unw*")) if (s / "configs").exists() else []
    if unw:
        try:
            kv = sb._read_config_kv(unw[0])
            out["unwrap_config_example"] = {k: kv.get(k) for k in
                                            ("nomcf", "defomax", "rlks", "alks", "method")}
        except Exception:                                     # noqa: BLE001
            pass
    win = s / WINDOW_FILE
    if win.exists():
        try:
            w = json.loads(win.read_text())
            out["unwrap_crop"] = {k: w.get(k) for k in
                                  ("r0", "r1", "c0", "c1", "crop_px", "frame_px",
                                   "crop_fraction", "frame_dem_valid_fraction")}
        except json.JSONDecodeError:
            out["unwrap_crop"] = "crop_window.json unreadable"
    else:
        out["unwrap_crop"] = "none (full-frame unwrap)"
    dem = cfg.dem_dir / "dem_extent.json"
    if dem.exists():
        try:
            out["dem_extent"] = json.loads(dem.read_text())
        except json.JSONDecodeError:
            pass
    hist = s / "unwrap_history.json"
    if hist.exists():
        try:
            h = json.loads(hist.read_text())
            vals = sorted(h.values())
            if vals:
                out["unwrap_seconds"] = {
                    "n": len(vals), "median": round(vals[len(vals) // 2], 1),
                    "max": round(vals[-1], 1), "sum_hours": round(sum(vals) / 3600, 2)}
        except json.JSONDecodeError:
            pass
    return out


def _unwrap_record(cfg: PipelineConfig) -> dict:
    """The unwrapper as it ran, from the per-pair unwrap_crop.json records
    (engine, snaphu version, cost mode). These are intermediates cleanup
    removes, so the summary is copied here while they exist."""
    merged = cfg.stack_dir / "merged" / "interferograms"
    if not merged.exists():
        return {"note": "per-pair unwrap records removed by cleanup"}
    seen: dict = {}
    n = 0
    for f in sorted(merged.glob("*/unwrap_crop.json")):
        r = _load(f) or {}
        n += 1
        key = (str(r.get("engine")), str(r.get("unwrapper")), str(r.get("cost")),
               str(r.get("defomax")), str(r.get("max_components")))
        seen[key] = seen.get(key, 0) + 1
    if not n:
        return {"note": "no unwrap_crop.json records (ISCE's own unwrap step)"}
    combos = [{"engine": k[0], "unwrapper": k[1], "cost": k[2], "defomax_cycles": k[3],
               "max_components": k[4], "pairs": c} for k, c in sorted(seen.items(), key=lambda kv: -kv[1])]
    return {"pairs_with_record": n, "settings": combos,
            "uniform": len(combos) == 1}


def _parse_miaplpy_cfg(path: Path) -> dict:
    """miaplpy.inversion.* / interferograms.* / multiprocessing lines of the
    template MiaplPy actually ran with - what was requested is the config,
    this is what the tool was given (and 'auto' is recorded as 'auto')."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        k, v = (x.strip() for x in line.split("=", 1))
        if k.startswith(("miaplpy.inversion.", "miaplpy.interferograms.networkType",
                         "miaplpy.interferograms.referenceDate", "miaplpy.multiprocessing.",
                         "miaplpy.subset.", "miaplpy.timeseries.tempCohType")):
            out[k] = v
    return out


def _miaplpy_settings(cfg: PipelineConfig) -> dict:
    work = cfg.project_dir / "miaplpy"
    tpl = next((p for p in (work / "miaplpyApp.cfg", work / "inputs" / "miaplpyApp.cfg") if p.exists()), None)
    as_run = _parse_miaplpy_cfg(tpl) if tpl else {}
    try:
        from .miaplpy_backend import stack_reference, LINKED_COHERENCE_LAYER
        ref, layer = stack_reference(cfg), LINKED_COHERENCE_LAYER
    except Exception:                                         # noqa: BLE001
        ref, layer = cfg.stack_reference_date or "", "unknown"
    return {
        "phase_linking_method": cfg.miaplpy_phase_linking_method,
        "ministack_size": cfg.miaplpy_ministack_size,
        "range_window": cfg.miaplpy_range_window,
        "azimuth_window": cfg.miaplpy_azimuth_window,
        "time_lag": getattr(cfg, "miaplpy_time_lag", None),
        "network_connections": getattr(cfg, "miaplpy_network_connections", None),
        "num_processors": cfg.miaplpy_num_processors,
        "pair_coherence": getattr(cfg, "miaplpy_pair_coherence", "phasor"),
        "env": cfg.miaplpy_env,
        "stack_reference_date": ref or "(not resolved)",
        "linked_coherence_layer": layer,
        "as_run_template": str(tpl) if tpl else "absent",
        "as_run": as_run,
    }


def _miaplpy_outcome(cfg: PipelineConfig) -> dict:
    """What phase linking did: the patch check, the datum check, the
    coherence the pairs carry, the reference and the linked stack's dates."""
    from . import version
    work = cfg.project_dir / "miaplpy"
    merged = cfg.stack_dir / "merged"
    slc_dates = _h5_dates(work / "inputs" / "slcStack.h5")
    ph_dates = _h5_dates(work / "inverted" / "phase_series.h5")
    status = sorted(p.name for p in (work / ".status").glob("*.done.json")) if (work / ".status").exists() else []
    pc = _load(merged / "pair_coherence.json")
    return {
        "patch_check": version.miaplpy_patch_check(cfg.project_dir)
        or "absent (no patch_check.json and no 'MiaplPy patch verified' line in logs/)",
        "linked_datum_check": _load(merged / "linked_datum_check.json") or "absent",
        "pair_coherence": ({"kind": pc.get("kind"), "pairs": len(pc.get("pairs", []))} if pc
                           else "no pair_coherence.json: phasor coherence (the default) on every pair"),
        "reference_choice": _load(cfg.stack_dir / "reference_choice.json") or "absent",
        "linking_fingerprint": _load(work / "inverted" / "linking_fingerprint.json") or "absent",
        "export_manifest": _export_manifest(merged / "interferograms" / "export_manifest.json"),
        "phase_series_complete": (work / "inverted" / "phase_series.complete").exists(),
        "completed_steps": status,
        "n_dates_slc_stack": len(slc_dates) if slc_dates is not None else "slcStack.h5 absent",
        "n_dates_phase_series": len(ph_dates) if ph_dates is not None else "phase_series.h5 absent",
        "unwrap": _unwrap_record(cfg),
    }


# The epoch of the ground the elevation model describes. geo_demErr.h5 (the
# DEM-error layer) is today's ground minus THIS surface, so reshaped ground
# (spoil, pits, reclamation since then) shows there. Copernicus DEM GLO-30/90
# is the edited TanDEM-X DEM, from acquisitions of Dec 2010 - Jan 2015.
_DEM_EPOCH = {"glo_30": "TanDEM-X acquisitions 2011-2015 (Copernicus DEM GLO-30)",
              "glo_90": "TanDEM-X acquisitions 2011-2015 (Copernicus DEM GLO-90)"}


def _export_manifest(p: Path):
    """The export key the unwrapped pairs were made under (miaplpy_backend):
    which linked series, so a re-link that changed it is visible. The pair
    list itself is in inputs.pairs."""
    m = _load(p)
    if not isinstance(m, dict):
        return "absent"
    return {k: (len(v) if k == "pairs" and isinstance(v, (list, dict)) else v)
            for k, v in m.items() if k != "series_probe"}


def _acquisitions(cfg: PipelineConfig) -> dict | None:
    """N available, M processed, K left out and why, from the download's own
    record (slc/assembly_status.json, data_acquisition). None when there is
    no record (DISP and older projects)."""
    a = _load(cfg.slc_dir / "assembly_status.json")
    if not isinstance(a, dict):
        return None
    dropped = dict(a.get("dropped") or {})
    for d in a.get("accepted_failures") or []:
        dropped.setdefault(str(d), "could not be downloaded (accepted in allow_missing_dates)")
    failed = a.get("failed") or {}
    return {"catalogue_dates": len(a.get("catalogue_dates") or []),
            "processed": len(a.get("processed") or []),
            "left_out": dropped,
            "failed": failed,
            "recovered_from_burst_extractor": a.get("recovered") or [],
            "accepted_missing": a.get("accepted_missing") or [],
            "source": str(cfg.slc_dir / "assembly_status.json")}


def _mintpy_run(cfg: PipelineConfig) -> dict | None:
    """Which MintPy run the numbers came from (mintpy/run_products.json,
    written by time_series only after a verified run), in a few lines; the
    whole record is under records.run_products."""
    rp = _load(cfg.mintpy_dir / "run_products.json")
    if not isinstance(rp, dict):
        return None
    era5 = rp.get("era5") if isinstance(rp.get("era5"), dict) else {}
    return {"run_stamp": rp.get("run_stamp"), "started": rp.get("started"), "finished": rp.get("finished"),
            "superseded_dir": rp.get("superseded_dir"),
            "corrections_applied": rp.get("corrections_applied"),
            "era5_dates_corrected": era5.get("dates_corrected"), "era5_dates_total": era5.get("dates_total"),
            "era5_missing_dates": rp.get("era5_missing_dates"),
            "removed_ramp": rp.get("removed_ramp"),
            "second_gate": rp.get("second_gate"),
            "linked_noise_level": rp.get("linked_noise_level"),
            "unwrap_error": rp.get("unwrap_error")}


def _resolution(cfg: PipelineConfig) -> dict | None:
    """Posting vs effective resolution of THIS product (resolution.py), with
    the measured similar-pixel counts on the phase-linked route, so the real
    support of each value can be recovered from the deliverables."""
    try:
        from . import deliverables as dl
        from .resolution import effective_resolution
        try:
            _, attrs = dl._read(dl._product(cfg, "velocity.h5"), "velocity")
        except Exception:                                     # noqa: BLE001
            attrs = None
        return effective_resolution(cfg, attrs, measure_shp=cfg.backend == "miaplpy")
    except Exception as exc:                                  # noqa: BLE001
        log.debug("resolution not recorded: %s", exc)
        return None


def _dem_record(cfg: PipelineConfig) -> dict:
    ext = _load(cfg.dem_dir / "dem_extent.json") or {}
    err = cfg.mintpy_dir / "geo" / "geo_demErr.h5"
    dem_error = {"dem_error_layer": str(err) if err.exists() else "absent",
                 "dem_error_meaning": "ground surface during the record minus the elevation model, m"}
    if cfg.dem_path:
        return {"source": "user-supplied", "file": str(cfg.dem_path),
                "epoch": "not recorded (user-supplied DEM)", **dem_error}
    src = ext.get("source")
    if cfg.backend == "disp":
        epoch = "OPERA DISP-S1's own (not recorded here)"
    else:
        epoch = _DEM_EPOCH.get(src or "glo_30", "not recorded")
    return {"source": {"glo_30": "Copernicus DEM GLO-30", "glo_90": "Copernicus DEM GLO-90"}.get(src, src)
            or ("Copernicus DEM GLO-30 (default; dem_extent.json absent)" if cfg.backend != "disp"
                else "OPERA DISP-S1's own"),
            "epoch": epoch,
            "tool": f"dem_stitcher {_version('dem_stitcher')}",
            "bbox_wsen": ext.get("bbox_wsen"), **dem_error}


def _records(cfg: PipelineConfig) -> dict:
    """The run's own JSON records, copied verbatim (each 'absent' when
    missing) so the provenance file carries them after cleanup."""
    P, S, M = cfg.project_dir, cfg.stack_dir, cfg.mintpy_dir
    out = {}
    for label, p in (("stack_generation", S / "generation.json"),
                     ("reference_choice", S / "reference_choice.json"),
                     ("dem_extent", cfg.dem_dir / "dem_extent.json"),
                     ("unwrap_error_applied", M / "unwrap_error_applied.json"),
                     ("reference_excluded_pairs", M / "reference_excluded_pairs.json"),
                     ("network_connectivity", M / "network_connectivity.json"),
                     ("corrections", M / "corrections.json"),
                     ("run_products", M / "run_products.json"),
                     ("deramp_mask", M / "deramp_mask.json"),
                     ("deramp_effect", M / "deramp_effect.json"),
                     ("era5_prefetch", M / "era5_prefetch.json"),
                     ("misreg_network", S / "misreg_network.json"),
                     ("assembly_status", cfg.slc_dir / "assembly_status.json")):
        rec = _load(p)
        out[label] = rec if rec is not None else "absent"
    era5 = sorted((P / "tropo_corrections").rglob("*.grb")) if (P / "tropo_corrections").exists() else []
    era5 += sorted((M / "ERA5").glob("*.grb")) if (M / "ERA5").exists() else []
    out["era5_files_on_disk"] = len({f.name for f in era5})
    return out


def config_snapshot(cfg) -> tuple[str, str]:
    """(yaml text, sha256) of the RESOLVED config - every field with the
    value this run used, defaults included. The YAML on disk can be edited
    after the run (Site A's was, the day after its record); this cannot."""
    import dataclasses
    import yaml

    def plain(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (tuple, list)):
            return [plain(x) for x in v]
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        return v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
    if dataclasses.is_dataclass(cfg):
        d = {f.name: plain(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
    else:
        d = {k: plain(v) for k, v in vars(cfg).items() if not k.startswith("_")}
    text = yaml.safe_dump(d, sort_keys=True, default_flow_style=False)
    return text, hashlib.sha256(text.encode()).hexdigest()


def _source_config(cfg) -> dict:
    """The YAML the run was launched with, its sha256 and whether it was
    edited after the product was made."""
    path = getattr(cfg, "_source_path", None) or getattr(cfg, "config_path", None)
    run = _load(cfg.project_dir / "run.json") or {}
    if not path and run.get("config"):
        path = run["config"]
    if not path or not Path(path).exists():
        return {"path": str(path) if path else "unknown", "sha256": None}
    p = Path(path)
    out = {"path": str(p), "sha256": _sha256_file(p),
           "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
    vel = cfg.mintpy_dir / "geo" / "geo_velocity.h5"
    if vel.exists() and p.stat().st_mtime > vel.stat().st_mtime:
        out["edited_after_product"] = True
        out["note"] = ("the YAML was modified after the velocity product was written; config_snapshot "
                       "below is the resolved config at the time this record was written")
    return out


def _env_fingerprint(prefix: Path | None) -> dict | None:
    """A lock fingerprint of a conda env: sha256 over its conda-meta package
    list (name-version-build of every installed package) - what
    `conda list --explicit` would pin, without seconds of subprocess."""
    if prefix is None or not (Path(prefix) / "conda-meta").is_dir():
        return None
    names = sorted(p.name for p in (Path(prefix) / "conda-meta").glob("*.json"))
    return {"prefix": str(prefix), "packages": len(names),
            "sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]}


def portable(obj):
    """obj with the user's home folder written '~' in every string, at any
    depth. provenance.json goes to the client (the package's technical/
    folder), and every path in it carried the operator's login name
    (16 Oct 2026, readiness review DAT-07). Nothing reads these paths back."""
    h = str(Path.home()).rstrip("/")
    if len(h) < 2:
        return obj
    # the home folder as a whole path: at the start, or after '=' / ':' / a
    # space (--config=/home/x/..., PATH-like lists), never inside another path
    home = re.compile(r"(?<![^\s=:])" + re.escape(h) + r"(?=/|$|[\s:])")

    def f(o):
        if isinstance(o, Path):
            o = str(o)
        if isinstance(o, str):
            return home.sub("~", o)
        if isinstance(o, dict):
            return {k: f(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [f(v) for v in o]
        return o
    return f(obj)


def _run_identity(cfg) -> dict:
    """Who ran it and how. The operator is the preparer the machine's
    settings name (insar doctor --preparer), the person the report names;
    until 16 Oct 2026 it was the OS login and the computer's name, which went
    to the client in the package (readiness review DAT-07)."""
    run = _load(cfg.project_dir / "run.json") or {}
    try:
        from . import settings
        who = str(settings.identity().get("preparer") or "").strip()
    except Exception:                                         # noqa: BLE001 - unknown = not set
        who = ""
    out = {"operator": who or "not set (insar doctor --preparer \"Name\")", "command": list(sys.argv)}
    if run.get("started"):
        try:
            out["run_started"] = datetime.fromtimestamp(float(run["started"])).isoformat(timespec="seconds")
        except (TypeError, ValueError):
            pass
    if run.get("kind"):
        out["run_kind"] = run["kind"]
    return out


# The builds that made this project's records, oldest first (16 Oct 2026,
# readiness review OUT-02). A products-only rerun rewrote provenance.json
# with its own code version and hash, so the build that did the processing
# run was recorded nowhere. The list is appended to, never replaced: write()
# adds the export's build and record_stages() the report stages' when they
# ran outside that run or with other code. {run_kind, started, step,
# version, code_hash, written_utc}; run_kind and started are run.json's when
# this process is that run, else None (a report rebuilt from the dashboard).
# Kept small: the first entry (usually the processing run's) and the last
# MAX_BUILDS - 1.
MAX_BUILDS = 50
_SAME_BUILD = ("run_kind", "started", "version", "code_hash")


def build_history(prev: dict | None) -> list[dict]:
    """The 'builds' list of an earlier provenance record. A record from
    before the list gives one entry, from its own software and run blocks -
    which is how a project processed before 16 Oct 2026 keeps the build of
    its processing run through its next products-only rerun."""
    prev = prev if isinstance(prev, dict) else {}
    got = prev.get("builds")
    if isinstance(got, list):
        return [b for b in got if isinstance(b, dict)]
    sw, run = prev.get("software") or {}, prev.get("run") or {}
    if not (sw.get("insar_pipeline") or sw.get("code_hash")):
        return []
    return [{"run_kind": run.get("run_kind"), "started": run.get("run_started"), "step": "export",
             "version": sw.get("insar_pipeline"), "code_hash": sw.get("code_hash"),
             "written_utc": prev.get("generated_utc"), "from": "the provenance record before builds were kept"}]


def this_build(cfg, step: str) -> dict:
    """The entry for the code running now, at `step` ('export', 'report')."""
    from . import version
    run = _load(Path(cfg.project_dir) / "run.json") or {}
    ours = run.get("pid") == os.getpid()
    started = None
    if ours and run.get("started"):
        try:
            started = datetime.fromtimestamp(float(run["started"])).isoformat(timespec="seconds")
        except (TypeError, ValueError, OverflowError, OSError):
            pass
    return {"run_kind": run.get("kind") if ours else None, "started": started, "step": step,
            "version": version.__version__, "code_hash": version.code_hash(),
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def add_build(builds: list, entry: dict) -> list:
    """builds + entry, unless the last entry is the same run and code."""
    builds = list(builds or [])
    if builds and all(builds[-1].get(k) == entry.get(k) for k in _SAME_BUILD):
        return builds
    builds.append(entry)
    if len(builds) > MAX_BUILDS:
        builds = builds[:1] + builds[-(MAX_BUILDS - 1):]
    return builds


def _software(cfg, unwrap: dict | None = None) -> dict:
    from . import version
    desc = version.describe(cfg)
    sw = {
        "insar_pipeline": desc["version"],
        "code_hash": desc["code_hash"],
        "isce2": _isce_version(),
        "mintpy": _version("mintpy"),
        "raider": _version("RAiDER"),
        "asf_search": _version("asf_search"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "env_insar": _env_fingerprint(Path(sys.prefix)),
    }
    if cfg.backend == "miaplpy":
        sw["miaplpy"] = desc.get("miaplpy") or {"version": None, "note": "MiaplPy env not found"}
        sw["env_miaplpy"] = _env_fingerprint(version._env_prefix(cfg.miaplpy_env))
    # The snaphu that ran, from the per-pair records when they exist; the
    # binary found now only when they do not (it may since have changed).
    ran = [s.get("unwrapper") for s in (unwrap or {}).get("settings", []) if s.get("unwrapper") not in (None, "None")]
    if ran:
        sw["snaphu"] = ran[0]
        sw["snaphu_source"] = "stack/merged/interferograms/*/unwrap_crop.json"
    elif cfg.backend in TOPSSTACK_ROUTES and desc.get("snaphu"):
        sw["snaphu"] = desc["snaphu"]
        sw["snaphu_source"] = "the binary found at export (no per-pair record left)"
    return sw


def collect(cfg: PipelineConfig, extra: dict | None = None) -> dict:
    """Assemble the provenance record for this run."""
    topsstack = cfg.backend in TOPSSTACK_ROUTES
    # Inputs. The network MintPy loaded (ifgramStack.h5) is the primary
    # record on every route that has one. The fallbacks are the directory
    # globs: the stack path assembles .SAFE DIRECTORIES from bursts and
    # deletes most of them after coregistration, and its pairs live under
    # stack/merged/interferograms - the old *.zip / pairs_dir globs returned
    # 0 and 0 for a 31-scene, 84-pair run, and the miaplpy route fell to
    # them until 5 Oct 2026 (n_pairs 0 for an 804-pair run).
    slcs = []
    if cfg.slc_dir.is_dir():
        slcs = sorted(p.name for p in list(cfg.slc_dir.glob("S1*.zip"))
                      + list(cfg.slc_dir.glob("S1*.SAFE")))
    net_dates, net_pairs, net_src = _ifgram_network(cfg)
    dates_processed, pairs, src = [], [], None
    if net_pairs:
        dates_processed, pairs, src = net_dates, net_pairs, net_src
    elif topsstack:
        coreg = cfg.stack_dir / "coreg_secondarys"
        # A coregistered date is a directory, or a <date>.cleaned marker once
        # its burst SLCs were released after the merged SLC was verified.
        dates_processed = sorted({d.name.split(".")[0] for d in coreg.iterdir()
                                  if d.is_dir() or d.name.endswith(".cleaned")}) if coreg.exists() else []
        if cfg.stack_reference_date:
            dates_processed = sorted(set(dates_processed) | {cfg.stack_reference_date})
        merged = cfg.stack_dir / "merged" / "interferograms"
        pairs = sorted(d.name for d in merged.iterdir() if d.is_dir()) if merged.exists() else []
        src = "stack/coreg_secondarys + stack/merged/interferograms"
    else:
        pairs = sorted(d.name for d in cfg.pairs_dir.iterdir()
                       if d.is_dir()) if cfg.pairs_dir.is_dir() else []
        src = "pairs directory" if pairs else None
    granules = _safe_list(cfg) if topsstack else []
    nc = _load(cfg.mintpy_dir / "network_connectivity.json") or {}

    stack_settings = {
        "swaths": cfg.stack_swaths,
        "reference_date": cfg.stack_reference_date or "(first date)",
        "range_looks": cfg.stack_range_looks,
        "azimuth_looks": cfg.stack_azimuth_looks,
        "filter_strength_goldstein": cfg.stack_filter_strength,
        "unwrap_method": cfg.stack_unw_method,
        "unwrap_engine": getattr(cfg, "stack_unw_engine", "isce"),
        "unwrap_cost_mode": getattr(cfg, "stack_unw_cost_mode", None),
        "unwrap_max_components": getattr(cfg, "stack_unw_max_components", None),
        "unwrap_full_cost_snaphu": cfg.stack_unw_full_cost,
        "unwrap_defomax_cycles": cfg.stack_unw_defomax,
        "num_connections": cfg.stack_num_connections,
        "coregistration": _coregistration(cfg),
        "ionosphere_split_spectrum": cfg.stack_ionosphere,
        "gpu": cfg.stack_use_gpu,
        "deramp": cfg.mintpy_deramp,
        "deramp_mask_requested": (cfg.mintpy_deramp_mask or
                                  f"built from deramp_exclude_geojson + {cfg.deramp_buffer_m:g} m buffer"
                                  if cfg.mintpy_deramp != "no" else "n/a"),
        "deramp_exclude_geojson": cfg.deramp_exclude_geojson or "(package/project context)",
        "unwrap_crop_requested": getattr(cfg, "stack_crop_unwrap", False),
        "unwrap_timeout_min": getattr(cfg, "stack_unwrap_timeout_min", 0.0),
        # The EFFECTIVE method, not the requested one. phase_closure is
        # dropped automatically when the network has no closed loops, and
        # a record that repeated the config here would assert a correction
        # ran that did not - which is the exact failure this whole module
        # exists to prevent.
        "unwrap_error_method": _effective_unwrap_error_method(cfg),
        "unwrap_error_method_requested": cfg.mintpy_unwrap_error_method,
        "geocode_lalo_step": cfg.mintpy_geocode_lalo_step,
        "posting_note": _posting_note(cfg),
        "tropo_method": cfg.mintpy_tropo_method,
    } if topsstack else None

    miaplpy_settings = _miaplpy_settings(cfg) if cfg.backend == "miaplpy" else None
    miaplpy_outcome = None
    if cfg.backend == "miaplpy":
        try:
            miaplpy_outcome = _miaplpy_outcome(cfg)
        except Exception as exc:                              # noqa: BLE001
            miaplpy_outcome = {"note": f"not read ({exc})"}
    unwrap = (miaplpy_outcome or {}).get("unwrap") if isinstance(miaplpy_outcome, dict) else None
    if unwrap is None and topsstack:
        unwrap = _unwrap_record(cfg)
    snap_text, snap_sha = config_snapshot(cfg)

    record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_dir": str(cfg.project_dir),
        "aoi_bbox_min_lon_min_lat_max_lon_max_lat": list(cfg.bbox),
        "date_range": [cfg.start_date, cfg.end_date],
        "track": cfg.track,
        "orbit_direction": cfg.orbit_direction,
        "route": _route_tag(cfg),
        "run": _run_identity(cfg),
        # every build that wrote this project's record so far (OUT-02); write()
        # adds this one
        "builds": build_history(_load(Path(cfg.project_dir) / "deliverables" / "provenance.json")),

        "processing": {
            "backend": cfg.backend,
            "n_neighbors": cfg.n_neighbors,
            "do_esd": cfg.do_esd,
            "esd_coherence_threshold": cfg.esd_coherence_threshold,
            "weather_model": cfg.weather_model,
            "excluded_pairs": list(cfg.exclude_pairs or []),
            "max_tropo_std_cm": cfg.max_tropo_std_cm,
        },

        # Everything on the DISP path that changes the numbers. Recorded
        # because "what settings produced this?" is the first question
        # asked of any result that is later doubted, and reconstructing
        # the answer months afterwards is effectively impossible.
        #
        # These are what was REQUESTED. What actually happened is under
        # "outcome" below - a run can decline to apply a correction, and
        # the difference between asking and getting is exactly the thing
        # worth recording.
        "disp_settings": {
            "mask_connected_components": cfg.disp_mask_connected_components,
            "min_valid_fraction": cfg.disp_min_valid_fraction,
            "use_recommended_mask": cfg.disp_use_recommended_mask,
            "apply_corrections": cfg.disp_apply_corrections,
            "seasonal_model": cfg.disp_seasonal_model,
            "use_static_layers": cfg.disp_use_static_layers,
            "flag_unwrap_jumps": cfg.disp_flag_unwrap_jumps,
            "repair_boundary_steps": cfg.disp_repair_boundary_steps,
            "tropo_correction": cfg.disp_tropo_correction,
            "max_per_year": cfg.disp_max_per_year,
        } if cfg.backend == "disp" else None,

        # The topsStack settings, which the phase-linked route shares for
        # coregistration, looks and unwrapping. unwrap_full_cost is the one
        # that matters on the isce engine: ISCE hardcodes MCF-initialisation-
        # only unwrapping with no CLI flag, so a record that does not state
        # whether it was overridden cannot distinguish this backend from the
        # one it replaced.
        "stack_settings": stack_settings,
        # Phase linking, as requested (the config) and as run (the template
        # MiaplPy was given), and what it did.
        "miaplpy_settings": miaplpy_settings,
        "miaplpy_outcome": miaplpy_outcome,

        # What was asked for is above. What HAPPENED is here, read from the
        # products and the stack directory at export time.
        "outcome": _outcome(cfg),
        "stack_outcome": _stack_outcome(cfg) if topsstack else None,
        "unwrap": unwrap,
        "dem": _dem_record(cfg),
        "records": _records(cfg) if topsstack or cfg.backend == "disp" else None,

        # The two numbers that change every displacement value in the
        # product. Recorded explicitly because a result is not
        # interpretable without them.
        "reference_point_lalo": (
            cfg.reference_lalo
            or ("auto (quietest well-observed pixel)" if cfg.backend == "disp"
                else "auto (MintPy selected)")),
        # The pixel the product is actually referenced to. MintPy resolves
        # the configured lat/lon in radar coordinates; after geocoding the
        # datum can sit a pixel or two away (two rows on the Site A
        # record), so the configured value alone does not locate it.
        "reference_pixel": _reference_pixel(cfg),
        # The YAML asked for a reference the run rejected (unwrapped in too
        # few pairs); the one in effect is reference_choice.json's (config.
        # _recorded_reference). Absent when the YAML reference was used.
        "reference_yaml_rejected": ({"yaml": cfg.reference_yaml_rejected,
                                     "note": "YAML reference rejected by the run, replaced by "
                                             "stack/reference_choice.json"}
                                    if getattr(cfg, "reference_yaml_rejected", None) else None),
        "min_temporal_coherence": cfg.min_temporal_coherence,
        "resolution": _resolution(cfg),
        "acquisitions": _acquisitions(cfg),
        "mintpy_run": _mintpy_run(cfg),

        # This string is read by whoever has to defend a number, so it
        # must describe what the code does rather than what it once did.
        # It previously stated that the DISP-S1 backend negates its input;
        # it does not, and has not since the sign was settled against the
        # `displacement` layer attribute in the granules themselves.
        "sign_convention": (
            "Negative line-of-sight displacement means motion AWAY from the "
            "satellite; positive means motion TOWARD it. Subsidence "
            "therefore appears negative. This is MintPy's convention. "
            "OPERA DISP-S1 states the same convention on its `displacement` "
            "layer ('Positive values indicate apparent motion towards the "
            "platform'), so that backend passes values through unchanged "
            "rather than negating them."
        ),

        "software": _software(cfg, unwrap),

        # The config this record describes: the resolved values (snapshot
        # written beside this file as config_snapshot.yaml) and the YAML the
        # run was launched with.
        "config": {"snapshot_sha256": snap_sha, "snapshot_file": "config_snapshot.yaml",
                   "source": _source_config(cfg)},

        "inputs": {
            "n_dates_processed": len(dates_processed) if (dates_processed or topsstack) else None,
            "dates_processed": dates_processed if (dates_processed or topsstack) else None,
            "dates_excluded_before_inversion": nc.get("excluded_dates", []),
            "network_source": src,
            "n_scenes_used": len(granules) if granules else None,
            "scenes_used": granules or None,
            "scenes_used_source": "stack/SAFE_files.txt" if granules else None,
            "n_scene_archives_on_disk": len(slcs),
            "scene_archives_on_disk": slcs,
            "scene_archives_note": ("stack_delete_safes_after_coreg removes archives "
                                    "once coregistered; the processed dates are the "
                                    "record, not the surviving archives")
            if topsstack else None,
            "n_pairs": len(pairs),
            "pairs": pairs,
        },
    }
    record["inputs"] = {k: v for k, v in record["inputs"].items() if v is not None}
    if extra:
        record.update(extra)
    return portable(record)


def _route_tag(cfg) -> str:
    try:
        from . import config as _c
        return _c.route_tag(cfg)
    except Exception:                                         # noqa: BLE001
        return {"miaplpy": "official", "disp": "screening"}.get(cfg.backend, "comparison")


def _posting_note(cfg) -> str:
    """Posting vs resolution in one sentence, from resolution.py when it is
    available; the generic statement otherwise (never a site's numbers).
    From the product's own attributes, as _resolution reads them: without
    them the Sentinel-1 defaults gave Site B 'about 95 m east-west' here
    and 'about 90 m' in the resolution record beside it (7 Oct 2026)."""
    try:
        from . import deliverables as dl
        from .resolution import effective_resolution
        try:
            _, attrs = dl._read(dl._product(cfg, "velocity.h5"), "velocity")
        except Exception:                                     # noqa: BLE001
            attrs = None
        t = effective_resolution(cfg, attrs).get("text")
        if t:
            return t
    except Exception:                                         # noqa: BLE001
        pass
    return ("geocode_lalo_step sets POSTING, not resolution. At "
            f"{cfg.stack_range_looks}x{cfg.stack_azimuth_looks} looks the "
            "independent resolution cell is coarser than the delivered grid"
            + (", and phase linking averages over its search window as well." if cfg.backend == "miaplpy" else "."))


def _reference_pixel(cfg) -> dict | None:
    """REF_Y/REF_X and REF_LAT/REF_LON of the delivered velocity product."""
    try:
        from . import deliverables as dl
        from .significance import reference_pixel
        _, attrs = dl._read(dl._product(cfg, "velocity.h5"), "velocity")
        return reference_pixel(attrs, cfg) or None
    except Exception as exc:                                  # noqa: BLE001
        log.debug("reference pixel not read: %s", exc)
        return None


# ------------------------------------------------------------------ later stages
def _send_rule(P: Path, qa_f: Path, qa: dict) -> tuple:
    """(send rule, the file it was read from) for the stages record.

    The QA runs before the GNSS tie, so the rule it stores never saw the tie
    (gnss_considered False), and provenance could say 'send with notes'
    where the report and the release list hold the report for want of a
    GNSS anchor (7 Oct 2026). The report's rule (client_report._send_rule:
    the QA's re-evaluated with the tie) comes first when the report was built
    after this QA; then summary.json's, re-evaluated the same way
    (deliverables._apply_qa_and_gnss), when written after this QA; the QA's
    own otherwise. "After" for the report is the release gate's own test
    (release.report_predates_qa: files written within
    QA_AFTER_REPORT_GRACE_S of each other count as written together), so
    this record, the gate and the dashboard never take different rules."""
    D = P / "deliverables"
    try:
        qa_t = qa_f.stat().st_mtime
    except OSError:
        return qa.get("send_rule"), "quality_assessment.json"
    try:
        from .release import QA_AFTER_REPORT_GRACE_S as grace
    except Exception:                                         # noqa: BLE001 - no gate: the plain comparison
        grace = 0.0

    def after_qa(f: Path, allow: float = 0.0):
        try:
            return _load(f) if f.stat().st_mtime + allow >= qa_t else None
        except OSError:
            return None

    rep = after_qa(D / "client_report.json", grace)
    if isinstance(rep, dict) and isinstance(rep.get("send_rule"), dict):
        return rep["send_rule"], "client_report.json"
    S = after_qa(D / "summary.json")
    if isinstance(S, dict) and S.get("send_rule") and S.get("send_rule_gnss_considered") is True:
        return {"value": S["send_rule"], "gnss_considered": True}, "summary.json"
    return qa.get("send_rule"), "quality_assessment.json"


# What the report's own tests found about themselves (trust review, 7 Oct
# 2026), recorded so a figure can be defended later: per report area the
# per-sign detection limits (E5), the local and coherence-matched nulls (E6)
# and the calibrated pace-change test behind its total (E10: client_report's
# 'pace_change' block, {tested, pace_change_mm, farmland_p95_mm,
# farmland_copies, ...}, vertical); per feature the chance in words and
# whether it may be called "stands out clearly" (E7, E8); the same nulls for
# the ground on each mapped mine layer; and whether the feature copies are
# as strong as the real farmland.
_AREA_DIAGNOSTICS = ("detection_limit_mm_yr", "detection_limit_rise_mm_yr", "detection_limit_either_direction_mm_yr",
                     "detection_limit_basis", "null_local", "null_coherence_matched", "null_checks_note",
                     "total_method", "nonlinear", "pace_change")
_FEATURE_DIAGNOSTICS = ("stands_out_clearly", "quality_standout", "total_method", "nonlinear", "pace_change")
_MINE_GROUND_DIAGNOSTICS = ("detected", "detection_limit_mm_yr", "detection_limit_rise_mm_yr", "null_local",
                            "null_coherence_matched", "null_checks_note")
# report_insights.selection_p's keys: k is 'copies_as_strong', n 'copies'
_CHANCE_KEYS = ("tested", "p", "copies_as_strong", "copies", "p_text", "stands_out", "stands_out_clearly",
                "clearly_withheld",
                # G2: a feature listed on its leave-out re-test - p is the
                # re-test's corrected for the number re-tested
                "listed_on", "retested", "leave_out", "second_pass")


def _pick(entry: dict, keys) -> dict:
    """`keys` of a report entry, at its top level only: a nested block (an
    area's deepest patch or stand-out) is another place's figure."""
    return {k: entry[k] for k in keys if k in entry}


def _report_diagnostics(rep: dict) -> dict:
    """The client report's self-checks (see _AREA_DIAGNOSTICS), each as the
    report recorded it; a key the report does not carry is left out, and a
    report built before 7 Oct 2026, which carries none of them, says so."""
    areas = [{"name": a.get("name"), "kind": a.get("kind"), **_pick(a, _AREA_DIAGNOSTICS)}
             for a in rep.get("areas", []) or [] if isinstance(a, dict)]
    mines = [{"label": m.get("label"), **_pick(m, _MINE_GROUND_DIAGNOSTICS)}
             for m in rep.get("mine_ground", []) or [] if isinstance(m, dict)]
    feats = []
    for i, f in enumerate(rep.get("features", []) or [], start=1):
        if not isinstance(f, dict):
            continue
        ch = f.get("chance") if isinstance(f.get("chance"), dict) else {}
        feats.append({"n": i, "lat": f.get("lat"), "lon": f.get("lon"), **_pick(f, _FEATURE_DIAGNOSTICS),
                      **({"chance": {k: ch[k] for k in _CHANCE_KEYS if k in ch}} if ch else {})})
    from . import release
    chk, src = release.feature_null_check(rep)
    fs = rep.get("feature_search") if isinstance(rep.get("feature_search"), dict) else {}
    out = {"areas": areas, "mine_ground": mines, "features": feats,
           "feature_null_check": chk if chk is not None else "no feature search in the report",
           "feature_null_check_source": src,
           "feature_copies_covariance": fs.get("covariance_check", "not recorded")}
    if not any(k in a for a in areas for k in ("null_local", "null_coherence_matched", "pace_change")):
        out["note"] = ("the report carries no local or coherence-matched null and no calibrated pace-change test "
                       "(built before 7 Oct 2026, or no area could be tested)")
    return out


def _gnss_screening(tie: dict) -> dict | str:
    """How the GNSS series were cleaned before the background was estimated
    (E15, 7 Oct 2026): the estimator, the NGL steps file (and whether it was
    read), each station's equipment steps and dropped antenna heights, the
    neighbour outlier test and what it left out, the radius sensitivity
    (only from stations downloaded: 'not evaluated' otherwise), and each
    inside-station check over the years it shares with the record (E16)."""
    if not any(k in tie for k in ("estimator", "steps", "outlier_test")):
        return ("not recorded: a tie written before the 7 Oct 2026 screening (NGL steps file, antenna heights, "
                "neighbour outlier test)")
    ot = tie.get("outlier_test") if isinstance(tie.get("outlier_test"), dict) else {}
    pair_keys = ("sites", "distance_km", "common_years", "tested", "difference_mm_yr", "limit_mm_yr", "disagree",
                 "left_out")
    return {"estimator": tie.get("estimator"),
            "steps_file": tie.get("steps"),
            "stations": [{"site": s.get("site"), "used_for_background": s.get("used_for_background"),
                          **(s.get("screening") if isinstance(s.get("screening"), dict) else {}),
                          **({"left_out": s["left_out"]} if s.get("left_out") else {})}
                         for s in tie.get("stations", []) or [] if isinstance(s, dict)],
            "outlier_test": {"rule": ot.get("rule"), "z": ot.get("z"), "left_out": ot.get("left_out"),
                             "pairs": [{k: p[k] for k in pair_keys if k in p}
                                       for p in ot.get("pairs", []) or [] if isinstance(p, dict)]},
            "background_left_out": (tie.get("background") or {}).get("left_out"),
            "radius_sensitivity": {r: {k: e[k] for k in ("vertical_mm_yr", "sites", "evaluated", "status",
                                                         "missing_sites", "left_out") if k in e}
                                   for r, e in (tie.get("radius_sensitivity") or {}).items() if isinstance(e, dict)},
            "inside_checks": [{k: c[k] for k in ("site", "verdict", "consistent", "verdict_reason", "overlap")
                               if k in c} for c in tie.get("inside_checks", []) or [] if isinstance(c, dict)]}


def stages(cfg: PipelineConfig) -> dict:
    """What came after the export: the QA verdict, the GNSS tie, the client
    report and its review. provenance.json is written at the export stage,
    before these exist, so until 5 Oct 2026 its embedded summary said the
    GNSS check was 'skipped' while the client report was GNSS-anchored.
    record_stages() adds this section after the report; package.build() puts
    a fresh copy in every release."""
    P = cfg.project_dir
    D = P / "deliverables"
    vel = cfg.mintpy_dir / "geo" / "geo_velocity.h5"
    vel_t = vel.stat().st_mtime if vel.exists() else None

    def stamp(p: Path) -> dict:
        t = p.stat().st_mtime
        out = {"file": str(p), "modified": datetime.fromtimestamp(t).isoformat(timespec="seconds")}
        if vel_t is not None and t < vel_t:
            out["older_than_velocity_product"] = True
        return out

    out: dict = {}
    qa_f = P / "quality_assessment.json"
    qa = _load(qa_f)
    if qa:
        flagged = [k for k, c in (qa.get("checks") or {}).items() if isinstance(c, dict) and c.get("pass") is False]
        sr, sr_from = _send_rule(P, qa_f, qa)
        # counted as the verdict counts them (D2, 7 Oct 2026): a check that
        # cannot apply is named, not counted (read from 'checks', so a record
        # from before D2 is counted the same way)
        from .quality_assessment import check_counts
        cnt = check_counts(qa)
        out["quality_assessment"] = {**stamp(qa_f), "verdict": qa.get("verdict"), "passed": qa.get("passed"),
                                     "flagged": flagged, "not_applicable": cnt["not_applicable"],
                                     "checks_text": cnt["text"], "send_rule": sr, "send_rule_source": sr_from,
                                     # every acquisition available, processed or left out, and why
                                     "dates": qa.get("dates")}
    else:
        out["quality_assessment"] = "absent"
    tie_f = D / "gnss_tie.json"
    tie = _load(tie_f)
    if tie is not None:
        used = tie.get("stations_used") or [s.get("site") for s in tie.get("stations", [])
                                            if s.get("used_for_background")]
        bg = tie.get("background") or {}
        out["gnss_tie"] = {**stamp(tie_f), "source": tie.get("source"), "record_window": tie.get("record_window"),
                           "stations_used": used,
                           "background_vertical_mm_yr": bg.get("vertical_mm_yr") if bg else None,
                           "background_uncertainty_mm_yr": bg.get("vertical_uncertainty_mm_yr") if bg else None,
                           "uncertainty_basis": bg.get("uncertainty_basis") if bg else None,
                           "radius_km": tie.get("radius_km"),
                           "station_data_files": [{k: s.get(k) for k in ("site", "data_file", "data_file_modified")}
                                                  for s in tie.get("stations", []) if isinstance(s, dict)
                                                  and s.get("data_file_modified")] or None,
                           "anchored": bool(tie.get("anchored", bool(bg))) and bool(bg),
                           "reason": tie.get("reason")}
        try:
            from . import gnss_tie as _gt
            old_tie = _gt.stale_reason(tie)
        except Exception:                                     # noqa: BLE001
            old_tie = None
        if old_tie:
            # recorded, not used: the report reads it as no tie (gnss_tie.check)
            out["gnss_tie"].update({"anchored": False, "stale": old_tie})
        try:
            out["gnss_tie"]["screening"] = _gnss_screening(tie)
        except Exception as exc:                              # noqa: BLE001
            out["gnss_tie"]["screening"] = f"not read ({exc})"
    else:
        out["gnss_tie"] = "absent"
    rep_f = D / "client_report.json"
    rep = _load(rep_f)
    if rep is not None:
        info = {**stamp(rep_f), "generated": rep.get("generated"), "route": rep.get("route"),
                "areas": [a.get("name") for a in rep.get("areas", []) if isinstance(a, dict)],
                "release": rep.get("release")}
        try:
            from . import release
            info["numbers_hash"] = release.numbers_hash(rep)
            info["review"] = release.review_record(P) or "not signed off"
        except Exception as exc:                              # noqa: BLE001
            info["numbers_hash"] = f"not computed ({exc})"
        try:
            info["diagnostics"] = _report_diagnostics(rep)
        except Exception as exc:                              # noqa: BLE001
            info["diagnostics"] = f"not read ({exc})"
        out["client_report"] = info
    else:
        out["client_report"] = "absent"
    # The optional stages (crosscheck.py, injection_test.py, both L10) write
    # their records under deliverables/technical/. The random site draw (seed,
    # sizes, rates) is recorded so an injection test can be repeated exactly.
    # 'current' says whether the report, the QA and the release gate use it
    # (crosscheck.record_is_current: the option on, written no earlier than
    # the velocity product); a record that is not is kept here, not used.
    # The injection test is technical only since 7 Oct 2026 (E4): its record
    # has no client sentence ('sentence' None, kept here only from an older
    # record that has one) and says which rule 'found' uses (decision_rule),
    # with the zero-rate control and the farmland it planted on.
    for label, name, option, keys in (
            ("cross_check", "cross_check.json", "cross_check_disp",
             ("status", "available", "reason", "sentence", "indicative_pass", "sibling_project", "disp",
              "common_window", "gates_verdict")),
            ("injection_test", "injection_recovery.json", "injection_test",
             ("status", "seed", "sizes_km", "rates_mm_yr", "types", "decision_rule", "technical_summary", "control",
              "farmland", "rates_dropped", "sizes_beyond_default_km", "sentence", "not_covered", "reason"))):
        f = next((p for p in (D / "technical" / name, D / name.replace("cross_check", "crosscheck"))
                  if p.exists()), None)
        if f is None:
            continue
        rec = _load(f)
        try:
            from . import crosscheck
            current = bool(crosscheck.record_is_current(cfg, rec, option))
        except Exception:                                     # noqa: BLE001 - recorded as not used
            current = False
        out[label] = {**stamp(f), "current": current,
                      **({k: rec.get(k) for k in keys if k in rec and not (k == "sentence" and label == "injection_test"
                                                                          and rec.get(k) is None)}
                         if isinstance(rec, dict) else {"note": "unreadable"})}
    return portable(out)


def record_stages(cfg: PipelineConfig) -> Path:
    """Add (or refresh) the 'stages' section of deliverables/provenance.json
    after the QA, GNSS tie and client report have run. Writes a full record
    first if none exists. Returns the path, after reading it back."""
    path = cfg.project_dir / "deliverables" / "provenance.json"
    rec = _load(path)
    if rec is None:
        path = write(cfg, path.parent)
        rec = _load(path) or {}
    rec["stages"] = stages(cfg)
    rec["stages_recorded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # the report stages' build, when they ran outside the export's run or
    # with other code (a sign-off rebuild from the dashboard; OUT-02)
    rec["builds"] = add_build(build_history(rec), this_build(cfg, "report"))
    # The summary embedded at export time predates the QA verdict and the
    # GNSS tie; replace it with summary.json as it stands now (deliverables.
    # refresh_summary attaches both), read through the key mapping for files
    # written before 5 Oct 2026.
    try:
        from .deliverables import read_summary
        S = read_summary(cfg.project_dir / "deliverables")
        if S:
            rec["summary"] = S
    except Exception as exc:                                  # noqa: BLE001
        log.warning("provenance: summary.json not re-read (%s)", exc)
    _write_json(path, rec)
    if "stages" not in (_load(path) or {}):
        raise RuntimeError(f"{path} was written but does not read back with its stages")
    return path


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def write(cfg: PipelineConfig, out_dir: Path, extra: dict | None = None) -> Path:
    """Write provenance.json and config_snapshot.yaml alongside the deliverables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    record = collect(cfg, extra)
    # appended to the record this one replaces, never reset (OUT-02)
    record["builds"] = add_build(build_history(_load(out_dir / "provenance.json")), this_build(cfg, "export"))
    # Settings blocks for other backends are not part of this record.
    for key in ("disp_settings", "stack_settings", "miaplpy_settings", "miaplpy_outcome",
                "stack_outcome", "unwrap", "records", "reference_yaml_rejected", "resolution",
                "acquisitions", "mintpy_run"):
        if record.get(key) is None:
            record.pop(key)
    snap_text, snap_sha = config_snapshot(cfg)
    (out_dir / "config_snapshot.yaml").write_text(
        f"# Resolved configuration of {cfg.project_dir} as recorded in provenance.json\n"
        f"# sha256 of the YAML below (without these two lines): {snap_sha}\n" + snap_text)
    path = out_dir / "provenance.json"
    _write_json(path, record)
    log.info("Wrote %s", path)

    if not cfg.reference_lalo:
        chooser = ("the backend" if cfg.backend == "disp" else "MintPy")
        log.warning("No reference_lalo set - %s chose the reference point "
                    "automatically. Every displacement in this product is "
                    "relative to a pixel nobody argued was stable. Set "
                    "reference_lalo before anything leaves the building.",
                    chooser)
    return path


def _coregistration(cfg) -> str:
    """What the stack was generated with (a one-burst crop falls back to
    geometry; stack_backend.coregistration_used), else what do_esd asks."""
    try:
        from .stack_backend import coregistration_used
        return coregistration_used(cfg)
    except Exception:                                          # noqa: BLE001
        return "NESD" if cfg.do_esd else "geometry"
