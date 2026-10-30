"""
Unwrap only the part of the frame that can be used.

WHY
---
stackSentinel's -b selects whole BURSTS. A burst is the full sub-swath in
range, so the merged frame is ~90 km wide around a 24 km AOI, without
the crop 86% of every interferogram lies outside the usable area (measured:
3.17 Mpx per pair, 0.43 Mpx inside the AOI). snaphu solved the whole
frame every time: full-frame, unwrapping was 4.1 h of a 5 h stack, and the
wall clock was set by ONE summer pair that ran 3 h 51 min.

Nothing in topsStack crops. So the unwrap step is replaced, pair by pair,
with: crop the filtered interferogram and coherence to a window, run the
SAME topsStack unwrap.py on the crop (so nomcf/defomax/looks from the
patched config still govern), then pad the result back into a full-frame
filt_fine.unw / .unw.conncomp. MintPy then sees files of the size its
metadata describes - STARTING_RANGE, per-pixel perpendicular baseline and
the demErr correction all stay correct with no surgery - and its own
subset.lalo crops to the AOI as before. Outside the window the phase is 0
and the connected component is 0, which MintPy already treats as
"not reliably unwrapped".

The window is computed ONCE per stack from the merged geometry
(compute_crop_window) and recorded in crop_window.json. Every pair is
checked against it before anything is cropped.

The command line this module answers to is what generate_run_files writes
into run_16_unwrap in place of `SentinelWrapper.py -c <config>`:

    python -m insar_pipeline.unwrap_crop --config <config_igram_unw_PAIR> \
        --window <stack>/crop_window.json
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

WINDOW_FILE = "crop_window.json"
NODATA_HEIGHT = -499.0      # ISCE fills height with -500 where the DEM is absent

_DTYPES = {"CFLOAT": np.complex64, "FLOAT": np.float32, "DOUBLE": np.float64,
           "BYTE": np.uint8, "SHORT": np.int16, "INT": np.int32}


# ----------------------------------------------------------------------
# ISCE raster I/O (raw + xml sidecar), enough for the six files we touch
# ----------------------------------------------------------------------
def _load_image(path: Path):
    from .isceenv import import_isce
    _, isceobj = import_isce()
    img = isceobj.createImage()
    img.load(str(path) + ".xml")
    # ISCE records an absolute file_name inside the xml; after a stack is
    # moved it points at the old location. Remember where we really read it.
    img._src_path = str(path)
    return img


def _read_raw(path: Path, img) -> np.ndarray:
    """Array of shape (length, width) or (length, bands, width) for BIL."""
    dt = _DTYPES[str(img.dataType).upper()]
    L, W, b = int(img.length), int(img.width), int(img.bands)
    mm = np.memmap(str(path), dtype=dt, mode="r")
    if b == 1:
        return np.asarray(mm).reshape(L, W)
    scheme = str(img.scheme).upper()
    if scheme == "BIL":
        return np.asarray(mm).reshape(L, b, W)
    if scheme == "BIP":
        return np.asarray(mm).reshape(L, W, b).transpose(0, 2, 1)
    return np.asarray(mm).reshape(b, L, W).transpose(1, 0, 2)          # BSQ


def _write_like(template_img, out: Path, arr: np.ndarray) -> None:
    """Write arr with template_img's dtype/bands/scheme and render xml + vrt."""
    from .isceenv import import_isce
    _, isceobj = import_isce()
    b = int(template_img.bands)
    scheme = str(template_img.scheme).upper()
    dt = _DTYPES[str(template_img.dataType).upper()]
    if b == 1:
        L, W = arr.shape
        raw = np.ascontiguousarray(arr, dtype=dt)
    else:
        L, _, W = arr.shape
        if scheme == "BIL":
            raw = np.ascontiguousarray(arr, dtype=dt)
        elif scheme == "BIP":
            raw = np.ascontiguousarray(arr.transpose(0, 2, 1), dtype=dt)
        else:
            raw = np.ascontiguousarray(arr.transpose(1, 0, 2), dtype=dt)
    raw.tofile(str(out))
    img = isceobj.createImage()
    img.load(getattr(template_img, "_src_path", str(template_img.filename)) + ".xml")
    img.setFilename(str(out))
    img.setWidth(W)
    img.setLength(L)
    img.setAccessMode("read")
    img.renderHdr()
    img.renderVRT()


def _unw_templates(unw: Path, win: dict):
    """ISCE image templates for a full-frame unw (2-band BIL float) and conncomp (byte):
    the existing files' own xml when they are there, else new ones."""
    from .isceenv import import_isce
    _, isceobj = import_isce()
    # The templates live beside the outputs under their own name: the caller
    # removes the old outputs (xml included) before writing the new ones, and
    # _write_like re-reads the template's xml at write time.
    out = []
    for path, bands, dtype, kind in ((unw, 2, "FLOAT", "unw"), (Path(str(unw) + ".conncomp"), 1, "BYTE", "cc")):
        tpl = Path(str(path) + ".template")
        if Path(str(path) + ".xml").exists():
            shutil.copyfile(str(path) + ".xml", str(tpl) + ".xml")
            out.append(_load_image(tpl)); continue
        img = isceobj.createImage()
        if kind == "unw":
            img.imageType = "unw"
        img.setFilename(str(tpl)); img.setWidth(win["width"]); img.setLength(win["length"])
        img.bands = bands; img.scheme = "BIL"; img.dataType = dtype; img.setAccessMode("read")
        img.renderHdr()
        out.append(_load_image(tpl))
    return out


def _read_gdal(path: Path) -> np.ndarray:
    from osgeo import gdal
    gdal.UseExceptions()
    for cand in (str(path) + ".vrt", str(path)):
        if os.path.exists(cand):
            ds = gdal.Open(cand)
            if ds is not None:
                return ds.GetRasterBand(1).ReadAsArray()
    raise FileNotFoundError(path)


# ----------------------------------------------------------------------
# the window
# ----------------------------------------------------------------------
def compute_crop_window(cfg, margin_deg: float | None = None, write: bool = True) -> dict:
    """
    Radar-coordinate window covering cfg.bbox +/- margin, restricted to
    pixels that have a DEM height. Written to <stack>/crop_window.json.

    Restricting to valid height matters twice: a bbox-only DEM leaves the
    rest of the frame at -500 and unwrappable only as noise, and a window
    that hugs the AOI is what makes the crop cheap.
    """
    stack = cfg.stack_dir
    geom = stack / "merged" / "geom_reference"
    lat = _read_gdal(geom / "lat.rdr")
    lon = _read_gdal(geom / "lon.rdr")
    hgt = _read_gdal(geom / "hgt.rdr")
    L, W = lat.shape
    if lon.shape != (L, W) or hgt.shape != (L, W):
        raise RuntimeError(f"merged geometry shapes disagree: lat {lat.shape} "
                           f"lon {lon.shape} hgt {hgt.shape}")

    m = cfg.stack_bbox_margin_deg if margin_deg is None else margin_deg
    w, s, e, n = cfg.bbox
    inside = ((lat >= s - m) & (lat <= n + m) & (lon >= w - m) & (lon <= e + m))
    valid = hgt > NODATA_HEIGHT
    sel = inside & valid
    if not sel.any():
        raise RuntimeError(
            "No merged-geometry pixel falls inside the AOI (+margin) with a "
            "valid DEM height. The bbox or the DEM is wrong for this frame.")
    rows = np.where(sel.any(axis=1))[0]
    cols = np.where(sel.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1

    n_inside_no_dem = int((inside & ~valid).sum())
    rec = {
        "r0": r0, "r1": r1, "c0": c0, "c1": c1,
        "length": L, "width": W,
        "crop_px": (r1 - r0) * (c1 - c0), "frame_px": L * W,
        "crop_fraction": round((r1 - r0) * (c1 - c0) / (L * W), 4),
        "bbox_wsen": list(cfg.bbox), "margin_deg": m,
        "aoi_px_without_dem": n_inside_no_dem,
        "frame_dem_valid_fraction": round(float(valid.mean()), 4),
        "geometry_mtime": os.path.getmtime(geom / "lat.rdr"),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not write:                       # local_change reads the window without changing what MintPy sees
        return rec
    out = stack / WINDOW_FILE
    out.write_text(json.dumps(rec, indent=2))
    log.info("Unwrap crop window rows %d:%d cols %d:%d = %d px, %.1f%% of the "
             "%dx%d frame (DEM valid over %.0f%% of the frame)", r0, r1, c0, c1,
             rec["crop_px"], 100 * rec["crop_fraction"], W, L,
             100 * rec["frame_dem_valid_fraction"])
    if n_inside_no_dem:
        log.warning("%d pixels inside the AOI+margin have NO DEM height and are "
                    "excluded from the window. The DEM does not cover the AOI "
                    "margin; get_dem() should have stitched the frame footprint.",
                    n_inside_no_dem)
    return rec


def load_window(path: Path) -> dict:
    rec = json.loads(Path(path).read_text())
    for k in ("r0", "r1", "c0", "c1", "length", "width"):
        if k not in rec:
            raise KeyError(f"{path} lacks {k}")
    return rec


# ----------------------------------------------------------------------
# per-pair: crop -> unwrap.py -> pad
# ----------------------------------------------------------------------
def parse_unwrap_config(path: Path) -> dict:
    """The key : value lines of a topsStack config_igram_unw_* file."""
    d = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line and not line.startswith("#") and not line.startswith("["):
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k:
                d[k] = v
    for need in ("ifg", "unw", "coh"):
        if not d.get(need):
            raise ValueError(f"{path} has no '{need}' entry")
    return d


def _unwrap_argv(conf: dict, ifg: Path, unw: Path, coh: Path) -> list:
    """Reproduce SentinelWrapper's dict->argv rule for unwrap.py."""
    argv = ["--ifg", str(ifg), "--unw", str(unw), "--coh", str(coh)]
    for k in ("nomcf", "reference", "defomax", "rlks", "alks", "rmfilter", "method"):
        v = conf.get(k, "")
        if v == "True":
            argv.append(f"--{k}")
        elif v and v != "False":
            argv += [f"--{k}", v]
    return argv


def find_snaphu(explicit: str = "") -> str:
    """The standalone snaphu 2 binary: explicit path, then PATH, then any conda env's bin/."""
    import glob
    if explicit and Path(explicit).is_file():
        return explicit
    w = shutil.which("snaphu")
    if w:
        return w
    for cand in sorted(glob.glob(str(Path(sys.executable).parents[2] / "*" / "bin" / "snaphu"))) + \
            sorted(glob.glob(str(Path(sys.executable).parents[1] / "bin" / "snaphu"))):
        if Path(cand).is_file():
            return cand
    raise FileNotFoundError("no standalone snaphu found: set stack_unw_snaphu_bin")


def _snaphu_meta(reference: str) -> dict:
    """Wavelength, altitude, earth radius and correlation looks exactly as topsStack's unwrap.py derives them."""
    from types import SimpleNamespace
    sys.path.insert(0, _tops_stack_dir())
    import unwrap as tops_unwrap                      # noqa: E402
    swaths = tops_unwrap.ut.getSwathList(reference)
    return tops_unwrap.extractInfo(os.path.join(reference, f"IW{swaths[0]}.xml"), SimpleNamespace(rglooks=1, azlooks=1))


def _run_snaphu2(conf: dict, ifg: np.ndarray, coh: np.ndarray, work: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Standalone snaphu 2 on the cropped pair with every parameter explicit.
    Geometry exactly as topsStack's unwrap.py gives snaphu 1.4.2 (so an
    engine comparison changes only what it means to change); connected
    components with a real minimum size and a limit the pairs do not hit.
    """
    rl, al = int(conf.get("rlks", 1)), int(conf.get("alks", 1))
    md = _snaphu_meta(conf["reference"])
    h, w = ifg.shape
    fi, fc, fu, fcc, fconf = (work / n for n in ("snaphu_in.c8", "snaphu_cor.f4", "snaphu_unw.f4", "snaphu_cc.u4", "snaphu.conf"))
    np.ascontiguousarray(ifg, dtype=np.complex64).tofile(fi)
    np.ascontiguousarray(np.nan_to_num(coh), dtype=np.float32).tofile(fc)
    corrlooks = float(conf.get("ncorrlooks") or 0) or rl * al / (0.8 * 0.8)
    fconf.write_text("\n".join([
        f"OUTFILE {fu}", f"CORRFILE {fc}", f"CONNCOMPFILE {fcc}",
        "INFILEFORMAT COMPLEX_DATA", "CORRFILEFORMAT FLOAT_DATA", "OUTFILEFORMAT FLOAT_DATA",
        "CONNCOMPOUTTYPE UINT", f"STATCOSTMODE {conf.get('cost', 'DEFO').upper()}", "INITMETHOD MST",
        f"DEFOMAX_CYCLE {float(conf.get('defomax', 4.0))}", f"LAMBDA {md['wavelength']}",
        f"ALTITUDE {md['altitude']}", f"EARTHRADIUS {md['earthRadius']}",
        f"NLOOKSRANGE {rl}", f"NLOOKSAZ {al}", f"NCORRLOOKS {corrlooks}",
        f"MAXNCOMPS {int(conf.get('maxcomp', 64))}", f"MINCONNCOMPFRAC {float(conf.get('minfrac', 0.001))}",
        "CONNCOMPTHRESH 300"]) + "\n")
    exe = find_snaphu(conf.get("snaphu", ""))
    import subprocess
    r = subprocess.run([exe, "-f", str(fconf), str(fi), str(w)], capture_output=True, text=True)
    if r.returncode or not fu.exists() or fu.stat().st_size != h * w * 4:
        raise RuntimeError(f"snaphu 2 failed ({r.returncode}): {(r.stderr or r.stdout)[-600:]}")
    unw = np.fromfile(fu, np.float32).reshape(h, w)
    cc = np.fromfile(fcc, np.uint32).reshape(h, w)
    if cc.max() > 255:
        raise RuntimeError(f"{cc.max()} connected components do not fit the byte conncomp file")
    for f in (fi, fc, fu, fcc):
        f.unlink(missing_ok=True)
    version = next((l for l in (r.stdout + r.stderr).splitlines() if "snaphu v" in l), "snaphu 2")
    return unw, cc.astype(np.uint8), version.strip()


def run_pair(config_path: Path, window_path: Path, keep_crop: bool = False) -> dict:
    conf = parse_unwrap_config(config_path)
    win = load_window(window_path)
    ifg, unw, coh = Path(conf["ifg"]), Path(conf["unw"]), Path(conf["coh"])
    pair_dir = ifg.parent
    r0, r1, c0, c1 = win["r0"], win["r1"], win["c0"], win["c1"]
    t0 = time.time()

    ifg_img, coh_img = _load_image(ifg), _load_image(coh)
    for name, img in (("ifg", ifg_img), ("coh", coh_img)):
        if (int(img.length), int(img.width)) != (win["length"], win["width"]):
            raise RuntimeError(
                f"{name} {img.filename} is {img.length}x{img.width}; the crop "
                f"window was computed on a {win['length']}x{win['width']} frame. "
                "Recompute crop_window.json (looks changed?) before unwrapping.")

    crop_ifg = pair_dir / "filt_fine_crop.int"
    crop_coh = pair_dir / "filt_fine_crop.cor"
    crop_unw = pair_dir / "filt_fine_crop.unw"
    for f in (crop_ifg, crop_coh, crop_unw):
        _remove_isce(f)
    _remove_isce(Path(str(crop_unw) + ".conncomp"))
    # The previous unwrap goes BEFORE anything can fail, not after snaphu
    # succeeds: a re-unwrap with new settings that died in between (component
    # overflow, MemoryError, an OOM kill) used to leave the old settings'
    # filt_fine.unw/.conncomp and unwrap_crop.json in place, and MintPy loaded
    # them under the new settings' name (5 Oct 2026 review). A failed pair now
    # has no unwrap at all, which the runner and MintPy's loader both see.
    # The snaphu 2 templates are then built from the window, exactly as for a
    # pair unwrapped for the first time.
    _remove_isce(unw)
    _remove_isce(Path(str(unw) + ".conncomp"))
    (pair_dir / "unwrap_crop.json").unlink(missing_ok=True)

    a = _read_raw(ifg, ifg_img)
    _write_like(ifg_img, crop_ifg, a[r0:r1, ..., c0:c1] if a.ndim == 3 else a[r0:r1, c0:c1])
    b = _read_raw(coh, coh_img)
    _write_like(coh_img, crop_coh, b[r0:r1, ..., c0:c1] if b.ndim == 3 else b[r0:r1, c0:c1])
    del a, b

    engine = conf.get("engine", "isce").lower()
    t1 = time.time()
    version = "snaphu (ISCE, via topsStack unwrap.py)"
    if engine == "snaphu2":
        # Standalone snaphu 2. The full-frame outputs keep the format MintPy
        # already reads: a 2-band BIL float unw (amplitude, phase) and a byte
        # conncomp, written from templates of the same shape.
        ci = _read_raw(crop_ifg, _load_image(crop_ifg))
        cc_src = Path(conf["corfile"]) if conf.get("corfile") else crop_coh
        if conf.get("corfile"):
            full_cor = _read_raw(cc_src, _load_image(cc_src))
            cor = full_cor[r0:r1, c0:c1]
        else:
            cor = _read_raw(crop_coh, _load_image(crop_coh))
        log.info("%s: unwrapping %dx%d crop with snaphu 2 (%s cost, %s components >= %s of the crop, coherence %s)",
                 pair_dir.name, r1 - r0, c1 - c0, conf.get("cost", "DEFO"), conf.get("maxcomp", 64),
                 conf.get("minfrac", 0.001), cc_src.name)
        ph, c, version = _run_snaphu2(conf, ci, cor, pair_dir)
        u = np.stack([np.abs(ci).astype(np.float32), ph], axis=1)          # (l, 2, w) BIL
        unw_img, cc_img = _unw_templates(unw, win)
    else:
        # The real thing, unchanged: topsStack's unwrap.py on the crop, in-process
        # exactly as SentinelWrapper would call it.
        sys.path.insert(0, _tops_stack_dir())
        import unwrap as tops_unwrap                      # noqa: E402
        argv = _unwrap_argv(conf, crop_ifg, crop_unw, crop_coh)
        log.info("%s: unwrapping %dx%d crop (%.0f%% of frame): unwrap.py %s",
                 pair_dir.name, r1 - r0, c1 - c0, 100 * win.get("crop_fraction", 0),
                 " ".join(argv[6:]))
        tops_unwrap.main(argv)
        unw_img = _load_image(crop_unw)
        cc_img = _load_image(Path(str(crop_unw) + ".conncomp"))
        u = _read_raw(crop_unw, unw_img)                          # (l, 2, w) BIL
        c = _read_raw(Path(str(crop_unw) + ".conncomp"), cc_img)  # (l, w)
    t_unw = time.time() - t1

    # Pad back to the full frame.
    if u.shape[0] != r1 - r0 or u.shape[-1] != c1 - c0:
        raise RuntimeError(f"unwrap.py output is {u.shape}, expected "
                           f"({r1 - r0}, 2, {c1 - c0})")
    full_u = np.zeros((win["length"], u.shape[1], win["width"]), dtype=u.dtype)
    full_u[r0:r1, :, c0:c1] = u
    full_c = np.zeros((win["length"], win["width"]), dtype=c.dtype)
    full_c[r0:r1, c0:c1] = c
    _remove_isce(unw)
    _remove_isce(Path(str(unw) + ".conncomp"))
    _write_like(unw_img, unw, full_u)
    _write_like(cc_img, Path(str(unw) + ".conncomp"), full_c)
    for base in (str(unw) + ".template", str(unw) + ".conncomp.template"):
        for ext in (".xml", ".vrt", ".hdr", ""):
            Path(base + ext).unlink(missing_ok=True)

    # Verify the END STATE - the files MintPy will load.
    exp_u = win["length"] * win["width"] * u.shape[1] * np.dtype(u.dtype).itemsize
    exp_c = win["length"] * win["width"] * np.dtype(c.dtype).itemsize
    got_u, got_c = unw.stat().st_size, Path(str(unw) + ".conncomp").stat().st_size
    if got_u != exp_u or got_c != exp_c:
        raise RuntimeError(f"padded output sizes wrong: unw {got_u} (want {exp_u}), "
                           f"conncomp {got_c} (want {exp_c})")
    n_cc = int((c > 0).sum())
    rec = {"pair": pair_dir.name, "window": [r0, r1, c0, c1],
           "crop_px": int((r1 - r0) * (c1 - c0)),
           "reliable_px": n_cc,
           "reliable_fraction_of_crop": round(n_cc / max(1, (r1 - r0) * (c1 - c0)), 4),
           "n_components": int(len(np.unique(c[c > 0]))),
           "unwrap_seconds": round(t_unw, 1), "total_seconds": round(time.time() - t0, 1),
           "nomcf": conf.get("nomcf"), "defomax": conf.get("defomax"),
           "rlks": conf.get("rlks"), "alks": conf.get("alks"),
           "engine": engine, "unwrapper": version, "cost": conf.get("cost", "DEFO" if engine == "isce" else None),
           "max_components": int(conf.get("maxcomp", 20 if engine == "isce" else 64)),
           "coherence_file": conf.get("corfile") or str(coh)}
    (pair_dir / "unwrap_crop.json").write_text(json.dumps(rec, indent=2))
    if not keep_crop:
        for f in (crop_ifg, crop_coh, crop_unw, Path(str(crop_unw) + ".conncomp")):
            _remove_isce(f)
    log.info("%s: unwrapped in %.0f s; %.1f%% of the crop in a nonzero component "
             "(%d components)", pair_dir.name, t_unw,
             100 * rec["reliable_fraction_of_crop"], rec["n_components"])
    return rec


def _remove_isce(path: Path) -> None:
    for suffix in ("", ".xml", ".vrt", ".aux.xml", ".rsc", ".hdr"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()


def _tops_stack_dir() -> str:
    from .stack_backend import ISCE_STACK
    return str(ISCE_STACK)


# ----------------------------------------------------------------------
# run-file rewrite
# ----------------------------------------------------------------------
ENGINE_KEYS = ("engine", "cost", "maxcomp", "minfrac", "ncorrlooks", "snaphu")


def engine_lines(cfg) -> dict:
    """The per-pair config keys that select and parameterise the unwrapping engine."""
    engine = str(getattr(cfg, "stack_unw_engine", "isce")).lower()
    if engine not in ("isce", "snaphu2"):
        # config validation refuses this at load; a config object built in code
        # must not fall through to ISCE's snaphu 1.4.2 and its 20-component cap
        raise ValueError(f"stack_unw_engine {engine!r} is not 'isce' or 'snaphu2'")
    if engine != "snaphu2":
        return {}                  # no keys = ISCE, so existing configs (and their markers) stay untouched
    return {"engine": "snaphu2", "cost": str(cfg.stack_unw_cost_mode).upper(),
            "maxcomp": str(int(cfg.stack_unw_max_components)),
            "minfrac": str(float(cfg.stack_unw_min_component_fraction)),
            **({"ncorrlooks": str(float(cfg.stack_unw_ncorrlooks))}
               if float(getattr(cfg, "stack_unw_ncorrlooks", 0) or 0) > 0 else {}),
            "snaphu": find_snaphu(getattr(cfg, "stack_unw_snaphu_bin", "") or "")}


def apply_engine(text: str, cfg) -> str:
    """Set the engine keys in a config_igram_unw_* text (replacing any already there)."""
    import re
    want = engine_lines(cfg)
    kept = [l for l in text.split("\n") if not re.match(rf"^({'|'.join(ENGINE_KEYS)})\s*:", l)]
    if not want:
        return "\n".join(kept)       # byte-identical when there were no engine keys
    return "\n".join(kept).rstrip("\n") + "\n" + "".join(f"{k} : {v}\n" for k, v in want.items())


def crop_command(cfg, config_path: Path) -> str:
    return (f"{sys.executable} -m insar_pipeline.unwrap_crop --config {config_path} "
            f"--window {cfg.stack_dir / WINDOW_FILE}")


def rewrite_unwrap_run_file(cfg) -> int:
    """
    Replace `SentinelWrapper.py -c config_igram_unw_*` lines with the crop
    wrapper. Idempotent; returns the number of lines changed. The original
    is kept as sentinelwrapper_run_16_unwrap for the record.
    """
    import re
    rfs = [p for p in (cfg.stack_dir / "run_files").glob("run_*")
           if p.is_file() and p.name.endswith("_unwrap")]
    changed = 0
    for rf in rfs:
        lines = rf.read_text().splitlines()
        out = []
        n = 0
        for line in lines:
            m = re.match(r"^\s*SentinelWrapper\.py\s+-c\s+(\S*config_igram_unw\S*)\s*&?\s*$", line)
            if m:
                out.append(crop_command(cfg, Path(m.group(1))))
                n += 1
            else:
                out.append(line)
        if n:
            # The backup must NOT start with "run_": stack_backend.run_files()
            # would execute it as a 17th run file - a full-frame unwrap of
            # every pair, after the cropped one.
            backup = rf.with_name("sentinelwrapper_" + rf.name)
            if not backup.exists():
                backup.write_text("\n".join(lines) + "\n")
            rf.write_text("\n".join(out) + "\n")
            changed += n
            log.info("%s: %d unwrap command(s) now run through unwrap_crop", rf.name, n)
    return changed


def restore_unwrap_run_file(cfg) -> int:
    """Undo rewrite_unwrap_run_file (stack_crop_unwrap turned off)."""
    n = 0
    for b in (cfg.stack_dir / "run_files").glob("sentinelwrapper_run_*_unwrap"):
        rf = b.with_name(b.name[len("sentinelwrapper_"):])
        rf.write_text(b.read_text())
        b.unlink()
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--window", required=True, type=Path)
    ap.add_argument("--keep-crop", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    rec = run_pair(a.config, a.window, keep_crop=a.keep_crop)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
