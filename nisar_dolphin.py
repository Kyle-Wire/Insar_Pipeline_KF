"""
Phase linking and unwrapping of a NISAR GSLC stack with dolphin
(review_2026-10-28/PLAN_nisar_phase_linked.md, Phase 2; 28 Oct 2026).

dolphin (isce-framework, OPERA's phase linker: DISP-S1 from CSLC-S1, the
basis of DISP-NI) runs in its own conda env (`dolphin`). This module writes its
inputs from nisar_gslc's crops, its config with the plan's settings, runs it,
and records the time and peak memory:

  inputs     one complex64 GeoTIFF per usable date (<work>/slcs/<date>.slc.tif),
             on the GSLC's own 5 m UTM grid; the dates gslc_stack.json names as
             empty are left out
  phase linking  sequential EMI over mini-stacks of 10 (the Sentinel-1 route's
             size), GLRT SHP test, a WINDOW x WINDOW window (half-window
             (WINDOW-1)/2) on the native 5 m pixels, output every STRIDE-th
             pixel: 3 = 15 m, the Sentinel-1 grid (Kyle 28 Oct, after a stride-1
             21 x 21 run was killed for memory at 23.8 GB)
  network    nearest-3 interferograms by date index
  unwrap     snaphu, smooth cost, tiled, MST initialisation (INIT_METHOD): dolphin's
             default, MCF, runs snaphu's CS2 module, whose licence is "strictly
             noncommercial"; MST does not touch it (2 Nov 2026, readiness review
             DEP-04). Unwrapped results an earlier MCF run left in the work folder
             are moved aside first, since dolphin reuses outputs it finds
  inversion  L1 (dolphin's default), referenced at the Sentinel-1 project's
             reference point
  units      metres (wavelength c / centre frequency of the GSLC)

    python -m insar_pipeline.nisar_dolphin run GSLC_DIR WORK --window 11 --stride 3 --ref LAT,LON
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ENV = "dolphin"
C_M_S = 299792458.0
# snaphu's initialisation: MST. dolphin's default, MCF, needs snaphu's CS2 module,
# licensed "strictly noncommercial" (readiness review DEP-04, 2 Nov 2026).
INIT_METHOD = "mst"
MINISTACK = 10
BANDWIDTH = 3
STRIDE = 3                 # 5 m pixels -> 15 m output
WORKERS = 2                # chunks phase-linked at once: 4 peaked at 23.5 GB (11 x 11, stride 1)
# dolphin's --worker-settings.gpu-enabled (speed plan Phase 3, 11 Nov 2026): dolphin
# defaults it to False and then hides the GPU from JAX, so the route ran on the CPU
# even on a machine with CUDA JAX. "auto" = on when pl_gpu.py's probe finds a usable
# CUDA device in the env; "on" / "off" force it.
GPU = "auto"


def conda_run() -> list[str]:
    from . import settings
    exe = settings.conda_base() / "bin" / "conda"
    return [str(exe), "run", "--no-capture-output", "-n", ENV]


def write_inputs(gslc_dir: Path, work: Path) -> dict:
    """The usable crops as dated complex64 GeoTIFFs; returns their record."""
    import h5py
    import rasterio
    from rasterio.transform import from_origin
    gslc_dir, work = Path(gslc_dir), Path(work)
    rec = json.loads((gslc_dir / "gslc_stack.json").read_text())
    skip = {e["date"] for e in rec.get("empty") or []}
    out = work / "slcs"
    out.mkdir(parents=True, exist_ok=True)
    files, freq = [], None
    for r in sorted(rec["reads"], key=lambda r: r["date"]):
        if r["date"] in skip:
            continue
        tif = out / f"{r['date']}.slc.tif"
        with h5py.File(r["file"], "r") as f:
            x, y, epsg = f["x"][()], f["y"][()], int(f.attrs["epsg"])
            freq = freq or float(f.attrs.get("centerFrequency") or 0) or None
            if not tif.exists():
                slc = f["slc"][()]
                dx, dy = float(x[1] - x[0]), float(y[0] - y[1])
                T = from_origin(float(x[0]) - dx / 2, float(y[0]) + dy / 2, dx, dy)
                with rasterio.open(tif, "w", driver="GTiff", height=slc.shape[0], width=slc.shape[1], count=1,
                                   dtype="complex64", crs=f"EPSG:{epsg}", transform=T, tiled=True,
                                   blockxsize=256, blockysize=256) as d:
                    d.write(slc, 1)
        files.append(str(tif))
    wavelength = C_M_S / freq if freq else None
    return {"files": files, "left_out": sorted(skip), "epsg": epsg, "x": [float(x[0]), float(x[-1])],
            "y": [float(y[0]), float(y[-1])], "shape": [int(y.size), int(x.size)], "wavelength_m": wavelength}


def ref_rowcol(gslc_dir: Path, lat: float, lon: float) -> tuple[int, int]:
    import h5py
    from pyproj import Transformer
    rec = json.loads((Path(gslc_dir) / "gslc_stack.json").read_text())
    with h5py.File(rec["reads"][0]["file"], "r") as f:
        x, y, epsg = f["x"][()], f["y"][()], int(f.attrs["epsg"])
    xs, ys = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    return int(np.argmin(np.abs(y - ys))), int(np.argmin(np.abs(x - xs)))


def gpu_probe() -> dict:
    """pl_gpu.py probe in the dolphin env: {"gpu": bool, "device", ...}. Never raises."""
    try:
        p = subprocess.run(conda_run() + ["python", str(Path(__file__).resolve().parent / "pl_gpu.py"), "probe"],
                           capture_output=True, text=True, timeout=180)
        line = [ln for ln in p.stdout.splitlines() if ln.strip().startswith("{")]
        return json.loads(line[-1]) if p.returncode == 0 and line else {"gpu": False, "reason": f"probe rc={p.returncode}"}
    except Exception as exc:                                    # noqa: BLE001
        return {"gpu": False, "reason": f"{type(exc).__name__}: {exc}"[:300]}


def resolve_gpu(mode: str = GPU) -> dict:
    """{"enabled": bool, "mode", "device"/"reason"} for dolphin's GPU switch."""
    mode = str(mode).strip().lower()
    if mode == "off":
        return {"enabled": False, "mode": mode, "reason": "off"}
    pr = gpu_probe()
    if not pr.get("gpu"):
        if mode == "on":
            raise RuntimeError(f"dolphin GPU 'on' but no usable GPU in env '{ENV}': {pr.get('reason')}")
        return {"enabled": False, "mode": mode, "reason": pr.get("reason")}
    return {"enabled": True, "mode": mode, "device": pr.get("device"), "kind": pr.get("kind")}


def write_config(work: Path, inputs: dict, window: int, ref_rc: tuple[int, int], workers: int = WORKERS,
                 stride: int = STRIDE, gpu: bool = False) -> Path:
    """dolphin's YAML with the plan's settings (dolphin config, then the
    reference point set in the YAML)."""
    import yaml
    work = Path(work)
    hw = (int(window) - 1) // 2
    cfg_path = work / "dolphin_config.yaml"
    cmd = conda_run() + [
        "dolphin", "config", "--outfile", str(cfg_path), "--work-directory", str(work),
        "--slc-files", *inputs["files"],
        "--phase-linking.ministack-size", str(MINISTACK),
        "--phase-linking.half-window.x", str(hw), "--phase-linking.half-window.y", str(hw),
        "--output-options.strides.x", str(int(stride)), "--output-options.strides.y", str(int(stride)),
        "--interferogram-network.max-bandwidth", str(BANDWIDTH),
        "--unwrap-options.unwrap-method", "SNAPHU",
        "--unwrap-options.snaphu-options.ntiles", "2", "2",
        "--unwrap-options.snaphu-options.tile-overlap", "300", "300",
        "--unwrap-options.snaphu-options.n-parallel-tiles", "4",
        "--unwrap-options.n-parallel-jobs", "2",
        "--worker-settings.threads-per-worker", "4",
        "--worker-settings.n-parallel-bursts", str(workers),
        "--worker-settings.gpu-enabled" if gpu else "--worker-settings.no-gpu-enabled"]
    if inputs.get("wavelength_m"):
        cmd += ["--input-options.wavelength", f"{inputs['wavelength_m']:.6f}"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    y = yaml.safe_load(cfg_path.read_text())
    # the reference point is a cell of the OUTPUT grid (every stride-th input pixel)
    y.setdefault("timeseries_options", {})["reference_point"] = [int(ref_rc[0]) // int(stride),
                                                                 int(ref_rc[1]) // int(stride)]
    # never snaphu's noncommercial CS2 code: MST, not dolphin's default MCF (DEP-04)
    y.setdefault("unwrap_options", {}).setdefault("snaphu_options", {})["init_method"] = INIT_METHOD
    cfg_path.write_text(yaml.safe_dump(y, sort_keys=False))
    return cfg_path


def retire_other_init_outputs(work: Path) -> list[str]:
    """Move unwrapped results not made with INIT_METHOD (a run before 2 Nov 2026
    used dolphin's default, MCF) to <work>/superseded_<method>_<stamp>/, so dolphin
    cannot reuse them. Returns the folders moved. A work folder whose run record
    says INIT_METHOD is left alone."""
    work = Path(work)
    rec_f = work / "nisar_pl_run.json"
    try:
        prev = json.loads(rec_f.read_text()) if rec_f.exists() else {}
    except (OSError, ValueError):
        prev = {}
    if prev.get("init_method") == INIT_METHOD:
        return []
    moved = []
    dest = work / f"superseded_{prev.get('init_method') or 'mcf'}_{time.strftime('%Y%m%d_%H%M%S')}"
    for name in ("unwrapped", "timeseries"):
        d = work / name
        if d.exists():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(dest / name))
            moved.append(name)
    if moved:
        log.info("dolphin: moved %s (unwrapped with %s) to %s; they are made again with %s",
                 ", ".join(moved), prev.get("init_method") or "MCF", dest, INIT_METHOD)
    return moved


def run(gslc_dir: Path, work: Path, window: int, ref_lalo: tuple[float, float], workers: int = WORKERS,
        stride: int = STRIDE, gpu: str = GPU) -> dict:
    """Inputs, config, `dolphin run`; returns (and writes) the run record:
    seconds, peak memory, the settings."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    retired = retire_other_init_outputs(work)
    inputs = write_inputs(gslc_dir, work)
    rc = ref_rowcol(gslc_dir, *ref_lalo)
    g = resolve_gpu(gpu)
    log.info("dolphin on the %s", f"GPU ({g.get('device')})" if g["enabled"] else f"CPU ({g.get('reason')})")
    cfg = write_config(work, inputs, window, rc, workers, stride, gpu=g["enabled"])
    timer = shutil.which("time") or "/usr/bin/time"
    cmd = conda_run() + ["dolphin", "run", str(cfg)]
    if Path(timer).exists():
        cmd = [timer, "-v"] + cmd
    t0 = time.time()
    log_path = work / "dolphin_run.log"
    with open(log_path, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           env={**os.environ, "OMP_NUM_THREADS": "4", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    secs = time.time() - t0
    text = log_path.read_text(errors="replace")
    peak_kb = None
    for line in text.splitlines()[-40:]:
        if "Maximum resident set size" in line:
            peak_kb = int(line.split(":")[-1].strip())
    rec = {"window": int(window), "half_window": (int(window) - 1) // 2, "strides": [int(stride), int(stride)],
           "workers": int(workers), "gpu_enabled": g["enabled"], "gpu": g,
           "ministack": MINISTACK, "network": f"nearest-{BANDWIDTH}", "unwrap": f"snaphu (smooth, 2 x 2 tiles, {INIT_METHOD.upper()})",
           "init_method": INIT_METHOD, "retired": retired,
           "inversion": "L1", "reference_lalo": list(ref_lalo), "reference_rowcol_input": list(rc),
           "reference_rowcol_output": [rc[0] // int(stride), rc[1] // int(stride)],
           "dates": len(inputs["files"]), "left_out": inputs["left_out"], "shape": inputs["shape"],
           "wavelength_m": inputs["wavelength_m"], "returncode": p.returncode, "seconds": round(secs, 1),
           "peak_rss_gb": None if peak_kb is None else round(peak_kb / 1e6, 2), "log": str(log_path)}
    (work / "nisar_pl_run.json").write_text(json.dumps(rec, indent=1))
    if p.returncode:
        raise RuntimeError(f"dolphin failed (exit {p.returncode}); see {log_path}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m insar_pipeline.nisar_dolphin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("gslc_dir")
    r.add_argument("work")
    r.add_argument("--window", type=int, required=True)
    r.add_argument("--ref", required=True, help="LAT,LON of the reference point")
    r.add_argument("--workers", type=int, default=WORKERS)
    r.add_argument("--stride", type=int, default=STRIDE)
    r.add_argument("--gpu", choices=("auto", "on", "off"), default=GPU)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lat, lon = (float(v) for v in a.ref.split(","))
    rec = run(Path(a.gslc_dir), Path(a.work), a.window, (lat, lon), a.workers, a.stride, a.gpu)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------- Gate 2 (PLAN, defined before any result)

GROUND_BUFFER_M = 500.0
FLOOR = 0.4
CLOSURE_RAD = 0.5


def _read(path):
    import rasterio
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float64)
        nod = s.nodata
        meta = {"transform": s.transform, "crs": s.crs, "shape": a.shape}
    if nod is not None and not np.isnan(nod):
        a[a == nod] = np.nan
    return a, meta


def _region(geojson: Path, meta: dict) -> np.ndarray:
    import json as _json
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.geometry import shape
    from shapely.ops import transform as stransform
    tr = Transformer.from_crs(4326, meta["crs"].to_epsg(), always_xy=True).transform
    geoms = [stransform(tr, shape(f["geometry"])) for f in _json.loads(Path(geojson).read_text())["features"]]
    return rasterize(geoms, out_shape=meta["shape"], transform=meta["transform"], fill=0, default_value=1).astype(bool)


def gate2(work: Path, region_geojson: Path) -> dict:
    """The Phase 2 measures (PLAN_nisar_phase_linked, written before any
    result): temporal coherence over the comparison ground; triplet closure
    and the report area's footprint spread over its reliable cells."""
    from scipy.ndimage import distance_transform_edt
    from . import significance as sg
    from .significance import years
    work = Path(work)
    tcf = sorted((work / "interferograms").glob("temporal_coherence_average_*.tif"))
    tc, meta = _read(tcf[-1])
    px = abs(meta["transform"].a)
    region = _region(region_geojson, meta)
    near = distance_transform_edt(~region) * px <= GROUND_BUFFER_M
    ts_files = sorted((work / "timeseries").glob("[0-9]*_[0-9]*.tif"))
    dates = [ts_files[0].stem.split("_")[0]] + [f.stem.split("_")[1] for f in ts_files]
    T = np.array(years(dates)) - years(dates)[0]
    D = np.stack([np.zeros(meta["shape"])] + [_read(f)[0] for f in ts_files]) * 1000.0     # mm LOS
    valid = np.isfinite(tc) & (tc > 0) & np.isfinite(D).all(axis=0)
    ground = valid & ~near
    reliable = ground & (tc >= FLOOR)
    # straight-line rate per cell over the usable dates
    tt = T - T.mean()
    rate = np.tensordot(tt, D - D.mean(axis=0), axes=(0, 0)) / float((tt ** 2).sum())
    rate_rel = rate - np.nanmedian(rate[reliable])
    s = sg.assess(np.where(valid & (tc >= FLOOR), rate_rel, np.nan), reliable, region & valid & (tc >= FLOOR))
    # triplet closure of the unwrapped nearest-3 network, each referenced to
    # the reference point (snaphu leaves each with its own whole-cycle offset)
    rec = json.loads((work / "nisar_pl_run.json").read_text())
    rr, rc = rec.get("reference_rowcol_output") or rec.get("reference_rowcol")
    unw = {f.name.split(".")[0]: f for f in (work / "unwrapped").glob("*.unw.tif")}
    ok_n = tot_n = 0
    n_trip = 0
    for i in range(len(dates) - 2):
        k12, k23, k13 = f"{dates[i]}_{dates[i + 1]}", f"{dates[i + 1]}_{dates[i + 2]}", f"{dates[i]}_{dates[i + 2]}"
        if not all(k in unw for k in (k12, k23, k13)):
            continue
        a, b, c = (_read(unw[k])[0] for k in (k12, k23, k13))
        a, b, c = a - a[rr, rc], b - b[rr, rc], c - c[rr, rc]
        cl = a + b - c
        m = reliable & np.isfinite(cl)
        ok_n += int((np.abs(cl[m]) < CLOSURE_RAD).sum())
        tot_n += int(m.sum())
        n_trip += 1
    out = {"window": rec.get("window"), "stride": (rec.get("strides") or [1])[0], "posting_m": px,
           "comparison_ground_cells": int(ground.sum()), "reliable_ground_cells": int(reliable.sum()),
           "temporal_coherence_median_ground": round(float(np.median(tc[ground])), 3),
           "pass_temporal_coherence": bool(np.median(tc[ground]) >= 0.5),
           "triplets": n_trip, "closure_share": round(ok_n / tot_n, 4) if tot_n else None,
           "pass_closure": bool(tot_n and ok_n / tot_n >= 0.8),
           "footprint_cells": int((region & valid & (tc >= FLOOR)).sum()),
           "spread_1sigma_mm_yr": s.get("farmland_spread_1sigma_mm_yr"),
           "footprint_rate_rel_mm_yr": s.get("value_mm_yr"), "testable": s.get("testable"),
           "rate_ground_robust_sd_mm_yr": round(float(1.4826 * np.median(np.abs(rate_rel[reliable]))), 3),
           "seconds": rec.get("seconds"), "peak_rss_gb": rec.get("peak_rss_gb"), "dates": len(dates)}
    (work / "gate2.json").write_text(json.dumps(out, indent=1))
    return out


def window_rule(g11: dict, g21: dict) -> dict:
    """The pre-registered choice: 11 x 11 if its spread <= 1.25 x the 21 x 21
    spread and its closure share >= 80%; otherwise 21 x 21."""
    s11, s21 = g11.get("spread_1sigma_mm_yr"), g21.get("spread_1sigma_mm_yr")
    ok = s11 is not None and s21 is not None and s11 <= 1.25 * s21 and (g11.get("closure_share") or 0) >= 0.8
    return {"chosen_window": 11 if ok else 21, "spread11": s11, "spread21": s21,
            "ratio": None if not (s11 and s21) else round(s11 / s21, 3), "closure11": g11.get("closure_share")}
