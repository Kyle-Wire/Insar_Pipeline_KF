"""Gather every number the run summary needs into one JSON, read from the
products and records on disk - never from the log.

    python scripts/collect_results.py <config.yaml> [out.json] [--compare <other config.yaml>]

--compare adds a slim copy of a second, independently processed run's
assessment, mine and site figures and its like-for-like comparison with this
one (deliverables/experiments/linked_vs_record.json), for the page's
independent-check section.

Reads: quality_assessment.json, reference_candidates.json, deliverables/summary.json, deliverables/provenance.json,
stack/timing.json, stack/unwrap_history.json, stack/crop_window.json,
slc/dropped_short_dates.json, mintpy/reference_excluded_pairs.json,
mintpy/geo/geo_velocity.h5 attributes, and the .cleaned markers.

Also records what the page must not hard-code (it was written for one site and
printed "Gibson County Mine - Site A, Indiana", "11 x 3 looks ... 41 m" and
Site A's acquisition gaps on every project): the project's identity
(report_insights.identity), the effective resolution (resolution.py), the gaps
in the acquisition record and the pairs that bridge them (ifgramStack.h5), the
code version (version.describe), the credits and the release status.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _summary(P: Path):
    """deliverables/summary.json through deliverables.read_summary, so files
    written before 5 Oct 2026 read with the renamed keys."""
    try:
        from insar_pipeline.deliverables import read_summary
        return read_summary(P / "deliverables") or None
    except Exception:                                         # noqa: BLE001 - the raw file is still a record
        return _load(P / "deliverables" / "summary.json")


def collect(cfg) -> dict:
    import glob
    import statistics as st

    P = cfg.project_dir
    S = cfg.stack_dir
    out = {"project": str(P), "config": {
        "dates": [cfg.start_date, cfg.end_date], "bbox": list(cfg.bbox),
        "reference_lalo": cfg.reference_lalo, "reference_date": _stack_reference(cfg),
        "looks": [cfg.stack_range_looks, cfg.stack_azimuth_looks],
        "connections": cfg.stack_num_connections, "min_temporal_coherence": cfg.min_temporal_coherence,
        "unwrap_timeout_min": cfg.stack_unwrap_timeout_min, "crop_unwrap": cfg.stack_crop_unwrap,
        "swaths": cfg.stack_swaths, "orbit_direction": cfg.orbit_direction, "track": cfg.track,
        "filter_strength": cfg.stack_filter_strength, "unwrap_engine": getattr(cfg, "stack_unw_engine", "isce"),
        "unwrap_defomax": cfg.stack_unw_defomax, "unwrap_full_cost": cfg.stack_unw_full_cost,
        "esd_coherence_threshold": cfg.esd_coherence_threshold, "do_esd": cfg.do_esd,
        "cleanups": {k: getattr(cfg, k) for k in (
            "stack_cleanup_offsets", "stack_cleanup_misreg_inputs",
            "stack_cleanup_coreg_after_filter", "stack_cleanup_unfiltered_igrams",
            "stack_delete_safes_after_coreg", "stack_cleanup_burst_geometry")},
    }}
    out["quality"] = _load(P / "quality_assessment.json")
    out["reference_candidates"] = _load(P / "reference_candidates.json")
    out["summary"] = _summary(P)
    out["provenance"] = _load(P / "deliverables" / "provenance.json")
    out["timing"] = _load(S / "timing.json")
    out["crop_window"] = _load(S / "crop_window.json")
    out["dropped_short_dates"] = _load(cfg.slc_dir / "dropped_short_dates.json")
    out["assembly_status"] = _load(cfg.slc_dir / "assembly_status.json")
    out["reference_excluded_pairs"] = _load(cfg.mintpy_dir / "reference_excluded_pairs.json")
    out["network_connectivity"] = _load(cfg.mintpy_dir / "network_connectivity.json")
    out["panel_history"] = _load(P / "deliverables" / "panel_history.json")
    out["site_history"] = _load(P / "deliverables" / "site_history.json")
    out["dem_extent"] = _load(cfg.dem_dir / "dem_extent.json")
    out["uncertainty_model"] = _load(P / "deliverables" / "uncertainty_model.json")
    out["acquisitions_not_in_stack"] = _not_in_stack(cfg)
    out["identity"] = _identity(cfg)
    out["resolution"] = _resolution(cfg)
    out["network_gaps"] = _network_gaps(cfg)
    out["mine_layers"] = _mine_layers(cfg)
    try:
        from insar_pipeline import version
        out["version"] = version.describe(cfg)
    except Exception as exc:                                  # noqa: BLE001
        out["version"] = {"error": str(exc)}
    try:
        from insar_pipeline import credits
        out["credits"] = credits.credits(cfg, prov=out["provenance"])
    except Exception as exc:                                  # noqa: BLE001
        out["credits"] = [{"source": "unavailable", "text": f"credits not generated ({exc})", "url": ""}]
    try:
        from insar_pipeline import release
        out["release"] = release.status(cfg)
    except Exception as exc:                                  # noqa: BLE001
        out["release"] = {"error": str(exc)}
    hist = _load(S / "unwrap_history.json") or {}
    v = sorted(hist.values())
    if v:
        out["unwrap"] = {"n": len(v), "median_s": round(st.median(v), 1),
                         "p90_s": round(v[int(0.9 * len(v))], 1), "max_s": round(v[-1], 1),
                         "sum_cpu_h": round(sum(v) / 3600, 2),
                         "slowest": sorted(hist.items(), key=lambda kv: -kv[1])[:10]}
    out["cleaned_markers"] = {
        "coreg_dates": len(glob.glob(str(S / "coreg_secondarys" / "*.cleaned"))),
        "offsets": len(glob.glob(str(S / "coreg_secondarys" / "*" / "offsets.cleaned"))),
        "unfiltered_pairs": len(glob.glob(str(S / "merged" / "interferograms" / "*" / "fine.cleaned"))),
        "misreg_inputs": (S / "ESD.cleaned").exists(),
    }
    out["pairs_on_disk"] = len(glob.glob(str(S / "merged" / "interferograms" / "*" / "filt_fine.unw")))
    out["quarantined"] = sorted(p.name for p in (S / "quarantine_unwrap_timeout").iterdir()) \
        if (S / "quarantine_unwrap_timeout").exists() else []
    vel = cfg.mintpy_dir / "geo" / "geo_velocity.h5"
    if vel.exists():
        import h5py
        with h5py.File(vel, "r") as f:
            keys = ("mintpy.networkInversion.obsDatasetName", "mintpy.networkInversion.numIfgram",
                    "mintpy.deramp", "mintpy.deramp.maskFile", "REF_DATE", "REF_LAT", "REF_LON",
                    "mintpy.troposphericDelay.method", "WAVELENGTH", "ORBIT_DIRECTION")
            out["velocity_attrs"] = {k: (f.attrs[k].decode() if isinstance(f.attrs[k], bytes)
                                          else str(f.attrs[k])) for k in keys if k in f.attrs}
            out["velocity_shape"] = list(f["velocity"].shape)
    return out


def _stack_reference(cfg) -> str:
    """The stack's reference date: the config's, else the one reference/ was
    built from (left blank, topsStack picks the first date)."""
    try:
        from insar_pipeline.miaplpy_backend import stack_reference
        return stack_reference(cfg)
    except Exception:                                         # noqa: BLE001
        return cfg.stack_reference_date


def _identity(cfg) -> dict:
    """Title and place, the way the client report names the project
    (report_insights.identity: project_title / project_location, else the
    folder name and a cached reverse geocode)."""
    try:
        from insar_pipeline.report_insights import identity
        return identity(cfg)
    except Exception:                                         # noqa: BLE001
        return {"title": Path(cfg.project_dir).name.replace("_", " ").title(), "location": ""}


def _resolution(cfg) -> dict | None:
    """resolution.effective_resolution(cfg) - posting and what each value
    combines, measured for this route; None when the module is absent."""
    try:
        from insar_pipeline.resolution import effective_resolution
        r = effective_resolution(cfg)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
    except Exception:                                         # noqa: BLE001
        return None


def _mine_layers(cfg) -> list:
    try:
        from insar_pipeline.report_insights import mine_layers
        return [{"key": m["key"], "label": m["label"], "path": str(m["path"])} for m in mine_layers(cfg)]
    except Exception:                                         # noqa: BLE001
        return []


def _network_gaps(cfg, min_gap_days: int = 36) -> dict:
    """Gaps in the acquisition record (consecutive dates more than three
    12-day revisits apart) and the longest kept pair bridging each, with its
    perpendicular baseline - from ifgramStack.h5's date/bperp/dropIfgram
    (small datasets; the phase is never read)."""
    from datetime import datetime as _dt
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if not stack.exists():
        return {}
    import h5py
    import numpy as np
    with h5py.File(stack, "r") as f:
        d12 = [(a.decode(), b.decode()) for a, b in f["date"][:]]
        keep = f["dropIfgram"][:].astype(bool) if "dropIfgram" in f else np.ones(len(d12), bool)
        bperp = f["bperp"][:] if "bperp" in f else None
    day = lambda s: _dt.strptime(s, "%Y%m%d")                  # noqa: E731
    dates = sorted({d for p in d12 for d in p})
    gaps = []
    for a, b in zip(dates[:-1], dates[1:]):
        n = (day(b) - day(a)).days
        if n > min_gap_days:
            bridge = [(i, (day(q) - day(p)).days) for i, (p, q) in enumerate(d12)
                      if keep[i] and p <= a and q >= b]
            i_max = max(bridge, key=lambda x: x[1])[0] if bridge else None
            gaps.append({"from": a, "to": b, "days": n, "bridging_pairs": len(bridge),
                         "longest_bridge_days": max(x[1] for x in bridge) if bridge else None,
                         "max_bridge_bperp_m": (round(float(max(abs(bperp[i]) for i, _ in bridge)), 0)
                                                if bridge and bperp is not None else None),
                         "longest_bridge": "_".join(d12[i_max]) if i_max is not None else None})
    return {"min_gap_days": min_gap_days, "gaps": gaps}


def _not_in_stack(cfg) -> list:
    """Acquisitions assembled in slc/ that never entered the inverted stack.
    20161206 was assembled after the run files were generated and silently
    left out of the record; a date like that must be reported, not found by
    a reviewer."""
    import re
    stack = cfg.mintpy_dir / "inputs" / "ifgramStack.h5"
    if not stack.exists() or not cfg.slc_dir.exists():
        return []
    import h5py
    with h5py.File(stack, "r") as f:
        in_stack = {d.decode() for pair in f["date"][:] for d in pair}
    on_disk = set()
    for p in cfg.slc_dir.iterdir():
        m = re.search(r"_(\d{8})T\d{6}_", p.name)
        if m and p.name.endswith((".SAFE", ".zip")):
            on_disk.add(m.group(1))
    excluded = set()
    nc = _load(cfg.mintpy_dir / "network_connectivity.json") or {}
    excluded |= set(nc.get("excluded_dates", []))
    return sorted(on_disk - in_stack - excluded)


def _slim(d, drop=("series_mm", "envelope_mm", "dates")):
    if isinstance(d, dict):
        return {k: _slim(v, drop) for k, v in d.items() if k not in drop}
    if isinstance(d, list) and d and isinstance(d[0], (dict, list)):
        return [_slim(v, drop) for v in d]
    return d


_main_project = [""]


def comparison(cfg) -> dict:
    P = cfg.project_dir
    lvr = _load(P / "deliverables" / "experiments" / "linked_vs_record.json") or \
        _load(Path(_main_project[0]) / "deliverables" / "experiments" / "linked_vs_record.json") or {}
    return {"project": str(P), "backend": cfg.backend,
            "quality": _slim(_load(P / "quality_assessment.json") or {}, drop=("series_mm", "step_rms", "dates")),
            "mines": _slim((_load(P / "deliverables" / "panel_history.json") or {}).get("mines", {})),
            "workings": _slim((_load(P / "deliverables" / "panel_history.json") or {}).get("workings", {})),
            "sites": _slim((_load(P / "deliverables" / "site_history.json") or {}).get("sites", {})),
            "panels": {n: {"rate_mm_yr": p.get("rate_mm_yr"), "n_pixels": p.get("n_pixels"), "significance": p.get("significance")}
                       for n, p in ((_load(P / "deliverables" / "panel_history.json") or {}).get("panels") or {}).items()},
            "summary": _summary(P),
            "linked_vs_record": {k: v for k, v in lvr.items() if k not in ("linked_series", "record_series", "common_dates")}}


def main():
    import argparse
    from insar_pipeline.config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("out", nargs="?")
    ap.add_argument("--compare", help="config of an independently processed run of the same site")
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = collect(cfg)
    res["backend"] = cfg.backend
    _main_project[0] = str(cfg.project_dir)
    if a.compare:
        res["comparison"] = comparison(load_config(a.compare))
    dst = Path(a.out) if a.out else cfg.project_dir / "run_results.json"
    dst.write_text(json.dumps(res, indent=2, default=str))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
