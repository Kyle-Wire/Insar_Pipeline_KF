"""
NISAR L2 GSLC: the single-look images the phase-linked NISAR route starts from
(review_2026-10-28/PLAN_nisar_phase_linked.md, Phase 1; 28 Oct 2026, Kyle:
"set it up the same way as the Sentinel-1 route").

WHY. The NISAR route so far read JPL's L2 GUNW interferograms, which exist only
at 80 m. The L2 GSLC is the same acquisition before any interferogram is made:
frequency A is 40 MHz, posted 5 m x 5 m on a UTM grid that every GSLC of a
track and frame shares (geocoded and coregistered by JPL), so a stack is ready
for phase linking as read - no coregistration and no merge, unlike
Sentinel-1's topsStack.

WHAT IS READ. A GSLC file is about 23 GB (the whole 240 km frame, HH and HV,
frequencies A and B). Only the study box (plus MARGIN_M) of frequency A's
co-pol layer and its mask are read, over HTTPS through the NISAR route's
windowed reader (nisar_backend.open_gunw: only the 512 x 512 chunks the window
touches). Measured 28 Oct 2026 on a 10 x 10 km box: 2,036 x 2,038 cells, 33 s.

    python -m insar_pipeline.nisar_gslc fetch CFG [--out DIR] [--workers N]
    python -m insar_pipeline.nisar_gslc preview DIR [--pair D1 D2]

Each date becomes <DIR>/crops/NISAR_GSLC_CROP_<track><dir>_<frame>_<date>.h5
(slc complex64, mask uint8, x, y, EPSG, the granule and window in attributes),
reused only when its granule, window, polarisation and crop version match.
gslc_stack.json records every read (bytes, seconds) and checks that all crops
share one grid.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

COLLECTION = "NISAR_L2_GSLC_PROVISIONAL_V1"      # BETA is never mixed in (the GUNW route's rule)
GRID = "science/LSAR/GSLC/grids/frequencyA/"
IDENT = "science/LSAR/identification/"
CROP_VERSION = 1
MARGIN_M = 500.0
# a date with less than this share of the box valid is named and left out
# (the first test site, 20260907: the acquisition is empty over the box - mask 0, no
# data - as the GUNW route found for the same date on 26 Oct 2026)
MIN_VALID_SHARE = 0.5
NAME_RE = re.compile(r"NISAR_L2_(?P<level>\w\w)_GSLC_(?P<cycle>\d{3})_(?P<track>\d{3})_(?P<dir>[AD])_(?P<frame>\d{3})_"
                     r"(?P<mode>\w{4})_(?P<pols>\w{4})_\w_(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_"
                     r"(?P<crid>\w+?)_\w_\w_\w_(?P<counter>\d{3})\.h5$")


def parse_name(name: str) -> dict | None:
    m = NAME_RE.search(str(name))
    if not m:
        return None
    d = m.groupdict()
    return {"track": int(d["track"]), "direction": "ascending" if d["dir"] == "A" else "descending",
            "frame": int(d["frame"]), "cycle": int(d["cycle"]), "mode": d["mode"], "pols": d["pols"],
            "date": d["start"][:8], "start": d["start"], "crid": d["crid"], "counter": int(d["counter"])}


# ---------------------------------------------------------------- search

def search(cfg) -> list[dict]:
    """GSLC granules over the study box and the config's window, one record each."""
    from .data_acquisition import cmr_search
    res = cmr_search(shortName=COLLECTION, intersectsWith=cfg.aoi_wkt, start=cfg.start_date or None,
                     end=cfg.end_date or None, maxResults=2000)
    out = []
    for r in res:
        p = r.properties if hasattr(r, "properties") else (r.get("properties") or {})
        name = p.get("fileName") or p.get("sceneName") or ""
        f = parse_name(name)
        if f is None:
            log.warning("Not a GSLC name, skipped: %s", name)
            continue
        size = (p.get("bytes") or {}).get(name, {}).get("bytes") if isinstance(p.get("bytes"), dict) else p.get("bytes")
        out.append({**f, "name": name, "url": p.get("url"), "bytes": size})
    return out


def choose_stack(recs: list[dict], track=None, direction=None) -> list[dict]:
    """One track and frame (the config's track and direction when given; else
    the frame with the most dates), one granule per date (the highest
    processing counter, then the latest composite release id)."""
    if track not in (None, ""):
        recs = [r for r in recs if r["track"] == int(track)]
    if direction:
        recs = [r for r in recs if r["direction"] == str(direction).lower()]
    if not recs:
        return []
    by = {}
    for r in recs:
        by.setdefault((r["track"], r["direction"], r["frame"]), set()).add(r["date"])
    key = max(by, key=lambda k: (len(by[k]), -k[2]))
    best = {}
    for r in recs:
        if (r["track"], r["direction"], r["frame"]) != key:
            continue
        o = best.get(r["date"])
        if o is None or (r["counter"], r["crid"]) > (o["counter"], o["crid"]):
            best[r["date"]] = r
    return [best[d] for d in sorted(best)]


# ---------------------------------------------------------------- reading

def window_for(x: np.ndarray, y: np.ndarray, epsg: int, bbox, margin_m: float = MARGIN_M) -> tuple:
    """(r0, r1, c0, c1) of the grid cells covering bbox (W, S, E, N, degrees)
    plus margin_m, clipped to the grid."""
    from pyproj import Transformer
    w, s, e, n = (float(v) for v in bbox)
    tr = Transformer.from_crs(4326, int(epsg), always_xy=True)
    lon = np.array([w, e, e, w, (w + e) / 2, (w + e) / 2, w, e])
    lat = np.array([s, s, n, n, s, n, (s + n) / 2, (s + n) / 2])
    xs, ys = tr.transform(lon, lat)
    x0, x1 = float(np.min(xs)) - margin_m, float(np.max(xs)) + margin_m
    y0, y1 = float(np.min(ys)) - margin_m, float(np.max(ys)) + margin_m
    cols = np.flatnonzero((x >= x0) & (x <= x1))
    rows = np.flatnonzero((y >= y0) & (y <= y1))
    if not cols.size or not rows.size:
        raise ValueError("the study box is not on this GSLC's grid")
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def valid_cells(mask: np.ndarray, slc: np.ndarray | None = None) -> np.ndarray:
    """Cells holding data: the GSLC mask's subswath number 1-254 (0 = no data,
    255 = fill; NISAR GSLC spec) and, when given, a non-zero SLC value."""
    m = (mask > 0) & (mask < 255)
    if slc is not None:
        m &= np.abs(slc) > 0
    return m


def valid_share(path: Path) -> float:
    import h5py
    with h5py.File(path, "r") as f:
        return float(valid_cells(f["mask"][()], f["slc"][()]).mean())


def _attr_text(v) -> str:
    if isinstance(v, np.ndarray):
        v = v.ravel()[0] if v.size else ""
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def crop_path(out_dir: Path, rec: dict) -> Path:
    return Path(out_dir) / "crops" / (f"NISAR_GSLC_CROP_T{rec['track']:03d}{rec['direction'][0].upper()}_"
                                       f"F{rec['frame']:03d}_{rec['date']}.h5")


def reusable(path: Path, rec: dict, bbox, margin_m: float, pol: str) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            a = f.attrs
            return (a.get("granule") == rec["name"] and int(a.get("crop_version", 0)) == CROP_VERSION
                    and a.get("polarization") == pol and math.isclose(float(a.get("margin_m", -1)), margin_m)
                    and np.allclose(np.asarray(a.get("bbox")), np.asarray(bbox, float)))
    except (OSError, TypeError, ValueError):
        return False


def read_crop(rec: dict, bbox, out: Path, pol: str = "HH", margin_m: float = MARGIN_M, auth=None) -> dict:
    """Read one granule's window of frequency A (pol) and its mask; write the
    crop. Returns the read record."""
    import h5py
    from . import nisar_backend as nb
    t0 = time.time()
    with nb.open_gunw(rec["url"], auth) as (h, fh):
        have = [_attr_text(p) for p in np.atleast_1d(h[GRID + "listOfPolarizations"][()])]
        if pol not in have:
            raise ValueError(f"{rec['name']}: polarisation {pol} is not in frequency A ({have})")
        x, y = h[GRID + "xCoordinates"][()], h[GRID + "yCoordinates"][()]
        epsg = int(np.asarray(h[GRID + "projection"][()]).ravel()[0])
        r0, r1, c0, c1 = window_for(x, y, epsg, bbox, margin_m)
        t_meta = time.time() - t0
        slc = h[GRID + pol][r0:r1, c0:c1]
        mask = h[GRID + "mask"][r0:r1, c0:c1] if GRID + "mask" in h else np.zeros(slc.shape, np.uint8)
        meta = {k: _attr_text(h[IDENT + k][()]) for k in ("zeroDopplerStartTime", "orbitPassDirection",
                                                           "trackNumber", "frameNumber", "productVersion")
                if IDENT + k in h}
        for k in ("rangeBandwidth", "centerFrequency", "xCoordinateSpacing", "yCoordinateSpacing"):
            if GRID + k in h:
                meta[k] = float(np.asarray(h[GRID + k][()]).ravel()[0])
        n_bytes = fh.bytes_read
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".partial")
    with h5py.File(part, "w") as f:
        f.create_dataset("slc", data=slc.astype(np.complex64), chunks=True, compression="gzip", compression_opts=1)
        f.create_dataset("mask", data=mask.astype(np.uint8), chunks=True, compression="gzip")
        f.create_dataset("x", data=x[c0:c1])
        f.create_dataset("y", data=y[r0:r1])
        f.attrs.update({"granule": rec["name"], "date": rec["date"], "epsg": epsg, "polarization": pol,
                        "window": [r0, r1, c0, c1], "bbox": np.asarray(bbox, float), "margin_m": float(margin_m),
                        "crop_version": CROP_VERSION, "bytes_read": int(n_bytes),
                        **{k: v for k, v in meta.items()}})
    part.replace(out)
    return {"date": rec["date"], "granule": rec["name"], "file": str(out), "shape": list(slc.shape),
            "window": [r0, r1, c0, c1], "epsg": epsg, "bytes_read": int(n_bytes), "seconds": round(time.time() - t0, 1),
            "metadata_seconds": round(t_meta, 1), "finite_share": float(np.isfinite(slc).mean()),
            "nonzero_share": float((np.abs(slc) > 0).mean()), "reused": False, **meta}


def _read_one(args):
    rec, bbox, out, pol, margin_m = args
    return read_crop(rec, bbox, Path(out), pol, margin_m)


def fetch(cfg, out_dir: Path, workers: int = 4, pol: str = "HH", margin_m: float = MARGIN_M) -> Path:
    """Read every date's window (in parallel processes: h5py reads hold one
    lock per process); write gslc_stack.json. Returns its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stack = choose_stack(search(cfg), getattr(cfg, "track", None), getattr(cfg, "orbit_direction", None))
    if not stack:
        raise RuntimeError("no GSLC over the study box for this track and direction in the window")
    bbox = list(cfg.bbox)
    todo, reads = [], []
    for rec in stack:
        p = crop_path(out_dir, rec)
        if p.exists() and reusable(p, rec, bbox, margin_m, pol):
            reads.append({"date": rec["date"], "granule": rec["name"], "file": str(p), "reused": True})
        else:
            todo.append((rec, bbox, str(p), pol, margin_m))
    failed = []
    if todo:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = {ex.submit(_read_one, a): a[0] for a in todo}
            for fu in as_completed(futs):
                rec = futs[fu]
                try:
                    r = fu.result()
                    reads.append(r)
                    log.info("GSLC %s: %s cells, %.1f MB read, %.0f s", r["date"], r["shape"], r["bytes_read"] / 1e6,
                             r["seconds"])
                except Exception as exc:                      # noqa: BLE001 - named in the record, not swallowed
                    failed.append({"date": rec["date"], "granule": rec["name"], "error": str(exc)[:300]})
                    log.warning("GSLC %s failed: %s", rec["date"], exc)
    reads.sort(key=lambda r: r["date"])
    empty = []
    for r in reads:
        r["valid_share"] = round(valid_share(Path(r["file"])), 4)
        if r["valid_share"] < MIN_VALID_SHARE:
            empty.append({"date": r["date"], "valid_share": r["valid_share"],
                          "cause": "the acquisition holds no valid data over the study box (the GSLC mask is 0 "
                                   "there)" if r["valid_share"] == 0 else
                                   f"only {100 * r['valid_share']:.0f}% of the study box holds data"})
    grid = same_grid([Path(r["file"]) for r in reads])
    disk = sum(Path(r["file"]).stat().st_size for r in reads)
    record = {"collection": COLLECTION, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
              "track": stack[0]["track"], "direction": stack[0]["direction"], "frame": stack[0]["frame"],
              "polarization": pol, "bbox": bbox, "margin_m": margin_m, "dates_found": len(stack),
              "dates_read": len(reads), "failed": failed, "empty": empty,
              "dates_usable": len(reads) - len(empty), "grid": grid, "disk_bytes": disk,
              "bytes_read": sum(int(r.get("bytes_read") or 0) for r in reads),
              "seconds": round(time.time() - t0, 1), "reads": reads}
    path = out_dir / "gslc_stack.json"
    path.write_text(json.dumps(record, indent=1, default=str))
    return path


def same_grid(files: list[Path]) -> dict:
    """Whether every crop has the first one's x / y / EPSG."""
    import h5py
    if not files:
        return {"same": None}
    with h5py.File(files[0], "r") as f:
        x0, y0, e0 = f["x"][()], f["y"][()], int(f.attrs["epsg"])
    bad = []
    for p in files[1:]:
        with h5py.File(p, "r") as f:
            if int(f.attrs["epsg"]) != e0 or f["x"].shape != x0.shape or f["y"].shape != y0.shape \
                    or not (np.array_equal(f["x"][()], x0) and np.array_equal(f["y"][()], y0)):
                bad.append(p.name)
    return {"same": not bad, "different": bad, "epsg": e0, "shape": [int(y0.size), int(x0.size)],
            "spacing_m": [float(x0[1] - x0[0]), float(y0[1] - y0[0])]}


# ---------------------------------------------------------------- the first look (Gate 1)

def _box(a: np.ndarray, k: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    if np.iscomplexobj(a):
        return uniform_filter(a.real, k) + 1j * uniform_filter(a.imag, k)
    return uniform_filter(a, k)


def coherence(a: np.ndarray, b: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """(coherence, interferogram phase) of two co-registered SLCs, k x k boxcar."""
    ab = a * np.conj(b)
    num = _box(ab, k)
    den = np.sqrt(_box(np.abs(a) ** 2, k) * _box(np.abs(b) ** 2, k))
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.abs(num) / den
    return np.clip(np.nan_to_num(g), 0, 1).astype(np.float32), np.angle(num).astype(np.float32)


def preview(out_dir: Path, pair: tuple | None = None, k: int = 5, landcover: Path | None = None) -> dict:
    """Every consecutive pair's median coherence (k x k), and for one pair (the
    first 12-day one, or `pair`) a coherence map and its interferogram as
    PNGs; medians by land cover when a WorldCover raster is given."""
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    rec = json.loads((out_dir / "gslc_stack.json").read_text())
    skip = {e["date"] for e in rec.get("empty") or []}
    files = {r["date"]: Path(r["file"]) for r in rec["reads"] if r["date"] not in skip}
    dates = sorted(files)
    with h5py.File(files[dates[0]], "r") as f:
        x, y, epsg = f["x"][()], f["y"][()], int(f.attrs["epsg"])
    cls = _landcover_on(landcover, x, y, epsg) if landcover else None
    pairs = []
    for d1, d2 in zip(dates[:-1], dates[1:]):
        with h5py.File(files[d1], "r") as f1, h5py.File(files[d2], "r") as f2:
            a, b = f1["slc"][()], f2["slc"][()]
            valid = valid_cells(f1["mask"][()], a) & valid_cells(f2["mask"][()], b)
        g, ph = coherence(a, b, k)
        days = (np.datetime64(f"{d2[:4]}-{d2[4:6]}-{d2[6:]}") - np.datetime64(f"{d1[:4]}-{d1[4:6]}-{d1[6:]}")).astype(int)
        row = {"pair": f"{d1}_{d2}", "days": int(days), "median_coherence": round(float(np.median(g[valid])), 3)}
        if cls is not None:
            row["by_cover"] = {name: round(float(np.median(g[valid & (cls == v)])), 3)
                               for v, name in WORLDCOVER.items() if (valid & (cls == v)).sum() > 1000}
        pairs.append(row)
        if pair is None and days <= 12 and "map" not in rec:
            pair = (d1, d2)
            rec["map"] = True
        if pair is not None and (d1, d2) == tuple(pair):
            ext = [x[0], x[-1], y[-1], y[0]]
            fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
            im0 = ax[0].imshow(g, cmap="gray", vmin=0, vmax=1, extent=ext)
            ax[0].set_title(f"Coherence {d1}-{d2} ({days} d), {k}x{k} boxcar at 5 m")
            fig.colorbar(im0, ax=ax[0], fraction=0.046)
            im1 = ax[1].imshow(np.where(g > 0.2, ph, np.nan), cmap="hsv", vmin=-np.pi, vmax=np.pi, extent=ext)
            ax[1].set_title("Interferogram phase (coherence > 0.2), one colour cycle = 12 cm LOS")
            fig.colorbar(im1, ax=ax[1], fraction=0.046)
            for a_ in ax:
                a_.set_xlabel(f"x (m, EPSG:{epsg})")
            fig.tight_layout()
            fig.savefig(out_dir / f"preview_{d1}_{d2}.png", dpi=110)
            plt.close(fig)
    rec.pop("map", None)
    res = {"k": k, "pairs": pairs, "preview_pair": list(pair) if pair else None,
           "dates_left_out": sorted(skip)}
    (out_dir / "gslc_preview.json").write_text(json.dumps(res, indent=1))
    return res


WORLDCOVER = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
              60: "bare / sparse", 80: "water"}


def _landcover_on(path: Path, x, y, epsg: int) -> np.ndarray:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    out = np.zeros((y.size, x.size), np.uint8)
    dx, dy = float(x[1] - x[0]), float(y[0] - y[1])
    T = from_origin(float(x[0]) - dx / 2, float(y[0]) + dy / 2, dx, dy)
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), out, dst_transform=T, dst_crs=f"EPSG:{epsg}", resampling=Resampling.nearest)
    return out


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m insar_pipeline.nisar_gslc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="read the study box of every GSLC date")
    f.add_argument("config")
    f.add_argument("--out", help="folder (default: <project>/nisar_gslc)")
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--pol", default="HH")
    p = sub.add_parser("preview", help="pair coherence and one interferogram")
    p.add_argument("dir")
    p.add_argument("--pair", nargs=2)
    p.add_argument("--landcover")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "fetch":
        from .config import load_config
        cfg = load_config(a.config)
        out = Path(a.out) if a.out else Path(cfg.project_dir) / "nisar_gslc"
        path = fetch(cfg, out, a.workers, a.pol)
        r = json.loads(path.read_text())
        for e in r["empty"]:
            print(f"left out {e['date']}: {e['cause']}")
        print(f"{r['dates_read']} of {r['dates_found']} dates, {r['dates_usable']} usable, {r['bytes_read'] / 1e9:.2f} GB read, "
              f"{r['disk_bytes'] / 1e9:.2f} GB on disk, {r['seconds'] / 60:.1f} min; same grid: {r['grid']['same']}")
        return 0 if not r["failed"] else 1
    res = preview(Path(a.dir), tuple(a.pair) if a.pair else None, landcover=Path(a.landcover) if a.landcover else None)
    for row in res["pairs"]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
