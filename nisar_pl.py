"""
The phase-linked NISAR route (review_2026-10-28/PLAN_nisar_phase_linked.md,
Phase 3; 28 Oct 2026, Kyle: "have this set up in the same way as the
Sentinel-1 route").

`nisar_phase_linked: true` keeps the NISAR route exactly as it is - JPL's
GUNW crops on the project's lattice, the chain, the product's split-spectrum
ionosphere screen (D3), HRRR troposphere, tides, DEM error, MintPy, the
products - and replaces ONE layer of every kept crop: the unwrapped phase,
with the phase-linked one.

  1. nisar_gslc.fetch: the study box of every L2 GSLC (5 m, frequency A),
     reused when already read;
  2. nisar_dolphin.run: dolphin phase linking on the 5 m pixels, output every
     STRIDE-th pixel (nisar_posting_m / 5: 15 m -> 3), the window
     nisar_pl_window (11, chosen on the first test site by the plan's pre-registered
     rule, Gate 2), referenced at the config's reference point; reused when
     a finished run has the same settings and dates;
  3. swap_into_crops: each kept crop's unwrappedPhase, coherenceMagnitude and
     connectedComponents become dolphin's for the same pair (unwrapped
     12-day interferogram, its correlation, its snaphu components), put on
     the crop's lattice by nearest neighbour. JPL's own layers are kept in
     the crop under the same names + "GUNW", and the crop says
     phase_source = "dolphin". The ionosphere screen, troposphere and tide
     cubes, geometry and baselines stay JPL's.
  4. dolphin's temporal coherence on the lattice
     (nisar_gunw/phase_linked_temporal_coherence.tif) becomes the products'
     temporal coherence (time_series.apply_nisar_pair_coherence) - on a GUNW
     chain MintPy's is 1.0 by construction.

Checked on the first test site before the swap was written (28 Oct 2026): dolphin's
12-day interferograms against JPL's for the same pairs, averaged to 80 m:
r 0.93-0.95, slope 0.89-1.02, the same sign, 0.09-0.13 rad RMS apart; one
wavelength (24.196 cm) in both.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

GSLC_POSTING_M = 5.0
TC_FILE = "phase_linked_temporal_coherence.tif"
SOURCE = "dolphin"
U = "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/"


def stride_for(cfg) -> int:
    from .nisar_backend import setting
    posting = float(setting(cfg, "nisar_posting_m"))
    k = posting / GSLC_POSTING_M
    if k < 1 or abs(k - round(k)) > 1e-6:
        raise ValueError(f"nisar_phase_linked needs nisar_posting_m to be a multiple of the GSLC's 5 m "
                         f"(it is {posting:g} m)")
    return int(round(k))


def _ref_lalo(cfg):
    s = str(getattr(cfg, "reference_lalo", "") or "").strip()
    if not s:
        return None
    lat, lon = (float(v) for v in s.split(","))
    return lat, lon


def current_run(work: Path, window: int, stride: int, dates: list[str]) -> bool:
    """A finished dolphin run in work with these settings and these dates."""
    try:
        rec = json.loads((Path(work) / "nisar_pl_run.json").read_text())
    except (OSError, ValueError):
        return False
    have = sorted(p.stem.split(".")[0] for p in (Path(work) / "slcs").glob("*.slc.tif"))
    return (rec.get("returncode") == 0 and int(rec.get("window", -1)) == int(window)
            and list(rec.get("strides") or []) == [int(stride), int(stride)] and have == sorted(dates)
            and (Path(work) / "unwrapped").is_dir())


def prepare(cfg, P: Path) -> dict:
    """GSLC crops and the dolphin run for this project (P: <project>/nisar_gunw);
    each reused when current. Returns {gslc_dir, work, run}."""
    from . import nisar_dolphin as nd
    from . import nisar_gslc as ng
    from .nisar_backend import setting
    project = Path(cfg.project_dir)
    gslc_dir, work = project / "nisar_gslc", project / "dolphin"
    rec = json.loads(ng.fetch(cfg, gslc_dir, pol=str(setting(cfg, "nisar_polarization") or "HH")).read_text())
    if rec.get("failed"):
        raise RuntimeError(f"NISAR GSLC dates could not be read: {rec['failed']}")
    skip = {e["date"] for e in rec.get("empty") or []}
    dates = [r["date"] for r in rec["reads"] if r["date"] not in skip]
    window, stride = int(setting(cfg, "nisar_pl_window")), stride_for(cfg)
    if current_run(work, window, stride, dates):
        run = json.loads((work / "nisar_pl_run.json").read_text())
        log.info("Phase-linked NISAR: the dolphin run in %s is current (window %d, stride %d, %d dates) - reused",
                 work, window, stride, len(dates))
    else:
        ref = _ref_lalo(cfg)
        if ref is None:
            raise ValueError("nisar_phase_linked needs reference_lalo (dolphin's time series reference point)")
        log.info("Phase-linked NISAR: dolphin, window %d x %d, stride %d (%g m), %d dates", window, window, stride,
                 stride * GSLC_POSTING_M, len(dates))
        run = nd.run(gslc_dir, work, window, ref, stride=stride)
    return {"gslc_dir": gslc_dir, "work": work, "run": run, "dates": dates, "empty": sorted(skip)}


def _lattice_transform(x, y):
    from rasterio.transform import from_origin
    dx, dy = float(x[1] - x[0]), float(y[0] - y[1])
    return from_origin(float(x[0]) - dx / 2, float(y[0]) + dy / 2, dx, dy)


def _onto(src: Path, shape, transform, dtype=np.float32, nodata=0) -> np.ndarray:
    import rasterio
    from rasterio.warp import Resampling, reproject
    out = np.zeros(shape, dtype)
    with rasterio.open(src) as s:
        reproject(rasterio.band(s, 1), out, dst_transform=transform, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest, src_nodata=s.nodata if s.nodata is not None else nodata,
                  dst_nodata=nodata)
    return out


def swap_into_crops(crops: list[Path], work: Path) -> list[dict]:
    """Each crop's unwrapped phase, coherence and components -> dolphin's for
    the same pair (JPL's kept as *GUNW). Idempotent: a crop already swapped
    from this dolphin run is left alone."""
    import h5py
    import hashlib
    work = Path(work)
    run_id = hashlib.sha256((work / "nisar_pl_run.json").read_bytes()).hexdigest()[:16]
    out = []
    for crop in crops:
        with h5py.File(crop, "r+") as h:
            ref, sec = str(h.attrs["reference"]), str(h.attrs["secondary"])
            pol = sorted(k for k in h[U] if k in ("HH", "VV", "HV", "VH"))[0]
            g = U + pol + "/"
            if h.attrs.get("phase_source") == SOURCE and h.attrs.get("phase_source_run") == run_id:
                out.append({"crop": crop.name, "pair": f"{ref}_{sec}", "swapped": "already"})
                continue
            unw, sign = work / "unwrapped" / f"{ref}_{sec}.unw.tif", 1.0
            if not unw.exists():
                unw, sign = work / "unwrapped" / f"{sec}_{ref}.unw.tif", -1.0
            if not unw.exists():
                raise RuntimeError(f"dolphin made no interferogram for {ref}_{sec} (crop {crop.name})")
            stem = unw.name.split(".")[0]
            x, y = h[g + "xCoordinates"][()], h[g + "yCoordinates"][()]
            shape, T = (y.size, x.size), _lattice_transform(x, y)
            ph = _onto(unw, shape, T).astype(np.float32) * sign
            cc = _onto(unw.with_name(stem + ".unw.conncomp.tif"), shape, T, np.uint16)
            cor = _onto(work / "interferograms" / f"{stem}.int.cor.tif", shape, T)
            ph[(ph == 0) | (cc == 0)] = np.nan
            for name, new in (("unwrappedPhase", ph), ("coherenceMagnitude", cor.astype(np.float32)),
                              ("connectedComponents", cc)):
                keep = g + name + "GUNW"
                if keep not in h:
                    h.copy(h[g + name], keep)
                h[g + name][...] = new
            h.attrs.update({"phase_source": SOURCE, "phase_source_run": run_id,
                            "phase_source_note": f"unwrappedPhase / coherenceMagnitude / connectedComponents from "
                                                 f"dolphin ({work}); JPL's kept as *GUNW"})
            out.append({"crop": crop.name, "pair": f"{ref}_{sec}", "swapped": True,
                        "valid_share": round(float(np.isfinite(ph).mean()), 4)})
    return out


def write_temporal_coherence(work: Path, lattice_crop: Path, out: Path) -> Path:
    """dolphin's temporal coherence on the crops' lattice (GeoTIFF, EPSG:4326)."""
    import h5py
    import rasterio
    with h5py.File(lattice_crop, "r") as h:
        pol = sorted(k for k in h[U] if k in ("HH", "VV", "HV", "VH"))[0]
        x, y = h[U + pol + "/xCoordinates"][()], h[U + pol + "/yCoordinates"][()]
    shape, T = (y.size, x.size), _lattice_transform(x, y)
    src = sorted((Path(work) / "interferograms").glob("temporal_coherence_average_*.tif"))[-1]
    tc = _onto(src, shape, T)
    with rasterio.open(out, "w", driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="float32",
                       crs="EPSG:4326", transform=T, nodata=0) as d:
        d.write(tc.astype(np.float32), 1)
        d.update_tags(SOURCE=f"dolphin temporal coherence ({src.name}), nearest onto the crops' lattice")
    return out
