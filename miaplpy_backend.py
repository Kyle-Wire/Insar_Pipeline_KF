"""Phase linking with MiaplPy after topsStack coregistration.

Distributed scatterers (farmland) carry a short-baseline phase bias that no
network of pairs can remove; phase linking estimates one consistent phase
history per cell from the whole coherence matrix instead. MiaplPy (Mirzaee,
Amelung & Fattahi 2023) does that on the coregistered SLC stack; the
interferograms are formed from the linked phases (each date with its
miaplpy_network_connections nearest neighbours, 3 in production; 0 = a
single-reference star), unwrapped and
inverted with MintPy. MintPy runs on the stack's own interferogram folder
(stack/merged/interferograms, written in topsStack's layout - step 3) and
writes its products to <project>/mintpy (cfg.mintpy_dir; a config's
cfg.time_series_dir overrides it, empty by default), where everything
downstream (deliverables, quality assessment, figures, panel and site
histories) reads them, exactly as on the classic route. MiaplPy's own
network directory (miaplpy/inverted/...) holds only its linked phase series.

Sequence:
  1. topsStack run_01 .. run_12 (coregistration, merged full-res SLCs with a
     REAL merge - MiaplPy globs merged/SLC/*/*.slc.full)
  2. MiaplPy load_data, phase_linking, concatenate_patches on a crop of the
     full-resolution stack ALIGNED to the record's multilook grid (rows a
     multiple of the azimuth looks, columns of the range looks)
  3. interferograms (nearest-neighbour network, miaplpy_network_connections)
     formed HERE from MiaplPy's linked phase
     series, multilooked to the stack's 11x3 grid, padded to the full frame
     and written into stack/merged/interferograms/<pair>/ exactly as
     topsStack would (filt_fine.int, filt_fine.cor)
  4. the stack route's own unwrapper (cropped full-cost snaphu, verified
     outputs) and MintPy chain (reference check, connectivity, bridging,
     ERA5, deramp mask, geocode) - so the phase-linked record lands on the
     same grid, with the same products and assessment, as the classic one.
MiaplPy runs in its own conda environment as subprocesses; every step is
verified on disk. MiaplPy's own unwrapper is not used: it unwraps at full
resolution with every tile in memory (11 GB per interferogram, eleven
minutes each), which is neither needed for a 15 m product nor survivable on
this machine.
"""
import json
import logging
import re as _re
import os
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import PipelineConfig

log = logging.getLogger(__name__)

from .settings import conda_sh as _conda_sh
CONDA_SH = str(_conda_sh())      # this machine's conda (settings.py)


def work_dir(cfg: PipelineConfig) -> Path:
    return cfg.project_dir / "miaplpy"


def network_dir(cfg: PipelineConfig) -> Path:
    return work_dir(cfg) / "network_single_reference"


SKIP_MASK = "skipMask.h5"          # inputs/: 1 = phase-link this pixel, 0 = skip (landcover.skip_on_points)


def skip_mask_path(cfg: PipelineConfig) -> Path:
    return work_dir(cfg) / "inputs" / SKIP_MASK


def write_skip_mask(cfg: PipelineConfig) -> dict:
    """
    The pixels MiaplPy should not phase-link (miaplpy_skip; 15 Oct 2026), on
    the loaded stack's radar grid (inputs/geometryRadar.h5's lat/lon). Only
    before any patch is linked: a series is linked with one mask or none.
    Never raises - without a land-cover map every pixel is linked. Returns
    the record kept in the linking fingerprint.
    """
    import h5py
    work = work_dir(cfg)
    out = skip_mask_path(cfg)
    if _patch_flags(work) or (work / "inverted" / "phase_series.h5").exists():
        rec = {"mode": "as linked", "note": "linking had already started; the mask (or none) it began with is kept"}
        if out.exists():
            rec["file"] = str(out)
        return rec
    out.unlink(missing_ok=True)
    from . import landcover
    mode = landcover.skip_mode(cfg)
    if mode == "none":
        return {"mode": "none"}
    try:
        landcover.ensure(cfg)
        geo = work / "inputs" / "geometryRadar.h5"
        with h5py.File(geo, "r") as g:
            lat, lon = g["latitude"][:], g["longitude"][:]
            attrs = dict(g.attrs)
        got = landcover.skip_on_points(cfg, lat, lon, mode)
        if got is None:
            return {"mode": "none", "note": f"{mode} asked for, but there is no land-cover map: every pixel linked"}
        skip, rec = got
        if not skip.any():
            log.info("Phase-linking skip (%s): nothing to skip in this frame", mode)
            return {**rec, "file": None}
        part = out.with_name(out.name + ".partial")
        with h5py.File(part, "w") as f:
            f.create_dataset("mask", data=(~skip).astype(np.int8), compression="lzf")
            for k in ("LENGTH", "WIDTH", "REF_DATE", "FILE_PATH"):
                attrs.pop(k, None)
            f.attrs.update({k: v for k, v in attrs.items() if isinstance(v, (str, bytes, int, float, np.number))})
            f.attrs.update(FILE_TYPE="mask", LENGTH=str(skip.shape[0]), WIDTH=str(skip.shape[1]), DATA_TYPE="int8",
                           description=f"insar_pipeline phase-linking skip ({mode}): 1 = linked, 0 = skipped")
        part.replace(out)
        log.info("Phase-linking skip (%s): %.1f%% of pixels not linked (water %.1f%%, dense woods %.1f%%) -> %s",
                 mode, rec["skipped_percent"], rec["water_percent"], rec["dense_woods_percent"], out)
        return {**rec, "file": str(out)}
    except Exception as exc:                                     # noqa: BLE001 - a speed-up, never a failure
        out.unlink(missing_ok=True)
        log.warning("Phase-linking skip not applied (%s: %s); every pixel is linked", type(exc).__name__, exc)
        return {"mode": "none", "note": f"{mode} asked for, not applied: {type(exc).__name__}: {exc}"}


def load_skip_keep(cfg: PipelineConfig, shape) -> np.ndarray | None:
    """The linked pixels (True) on the phase series' grid, or None when no
    pixel was skipped."""
    p = skip_mask_path(cfg)
    if not p.exists():
        return None
    import h5py
    with h5py.File(p, "r") as f:
        keep = f["mask"][:].astype(bool)
    if keep.shape != tuple(shape):
        log.warning("%s is %s, the phase series %s: not used", p, keep.shape, tuple(shape))
        return None
    return keep


def write_template(cfg: PipelineConfig) -> Path:
    """MiaplPy + MintPy options for this project, every value from the config."""
    from . import corrections
    stack = cfg.stack_dir
    work = work_dir(cfg); work.mkdir(parents=True, exist_ok=True)
    y0, y1, x0, x1 = aligned_subset(cfg)
    era5 = cfg.project_dir / "mintpy" / "ERA5"
    era5.mkdir(parents=True, exist_ok=True)
    lines = [
        "# written by insar_pipeline.miaplpy_backend - edit the YAML config, not this file",
        f"miaplpy.multiprocessing.numProcessor = {cfg.miaplpy_num_processors}",
        "miaplpy.load.processor      = isce",
        "miaplpy.load.autoPath       = no",
        # 'auto' and 'no' both end up as a filter name h5py rejects (MiaplPy
        # substitutes MintPy's value for 'no'), and the stack file is then
        # created empty. lzf is a real h5py filter: a little slower to write,
        # and complex SLC data hardly shrinks, but the load succeeds.
        "miaplpy.load.compression    = lzf",
        "mintpy.load.compression     = lzf",
        f"miaplpy.load.metaFile       = {stack}/reference/IW*.xml",
        f"miaplpy.load.baselineDir    = {stack}/baselines",
        f"miaplpy.load.slcFile        = {stack}/merged/SLC/*/*.slc.full",
        f"miaplpy.load.demFile        = {stack}/merged/geom_reference/hgt.rdr.full",
        f"miaplpy.load.lookupYFile    = {stack}/merged/geom_reference/lat.rdr.full",
        f"miaplpy.load.lookupXFile    = {stack}/merged/geom_reference/lon.rdr.full",
        f"miaplpy.load.incAngleFile   = {stack}/merged/geom_reference/los.rdr.full",
        f"miaplpy.load.azAngleFile    = {stack}/merged/geom_reference/los.rdr.full",
        f"miaplpy.load.shadowMaskFile = {stack}/merged/geom_reference/shadowMask.rdr.full",
        # full-resolution rows/cols, a multiple of the looks, so that the
        # multilooked interferograms coincide with the stack's merged grid
        f"miaplpy.subset.yx           = {y0}:{y1},{x0}:{x1}",
        f"miaplpy.inversion.ministackSize        = {cfg.miaplpy_ministack_size}",
        f"miaplpy.inversion.rangeWindow          = {cfg.miaplpy_range_window}",
        f"miaplpy.inversion.azimuthWindow        = {cfg.miaplpy_azimuth_window}",
        f"miaplpy.inversion.phaseLinkingMethod   = {cfg.miaplpy_phase_linking_method}",
        f"miaplpy.inversion.sbw_connNum          = {cfg.miaplpy_time_lag}",
        # pixels not linked (write_skip_mask); MiaplPy links every pixel while the file does not exist
        f"miaplpy.inversion.mask                 = {skip_mask_path(cfg)}",
        "miaplpy.interferograms.networkType     = single_reference",
        f"miaplpy.interferograms.referenceDate   = {cfg.stack_reference_date or 'auto'}",
        "",
        "# MintPy (MiaplPy copies these into custom_smallbaselineApp.cfg)",
        f"mintpy.reference.lalo                  = {cfg.reference_lalo.strip() or 'auto'}",
        f"mintpy.reference.minCoherence          = {cfg.reference_min_coherence}",
        f"mintpy.network.tempBaseMax             = {cfg.max_temp_baseline}",
        f"mintpy.network.perpBaseMax             = {cfg.max_perp_baseline}",
        "mintpy.network.keepMinSpanTree         = yes",
        "mintpy.unwrapError.method              = no",
        f"mintpy.networkInversion.minTempCoh     = {cfg.min_temporal_coherence}",
        "mintpy.networkInversion.weightFunc     = var",
    ] + corrections.mintpy_tropo_lines(cfg, 38) + [
        f"mintpy.troposphericDelay.weatherDir    = {era5.parent}",
        f"mintpy.deramp                          = {cfg.mintpy_deramp}",
        f"mintpy.deramp.maskFile                 = {network_dir(cfg) / 'maskDeramp.h5'}",
        "mintpy.topographicResidual             = yes",
        f"mintpy.geocode.laloStep                = {cfg.mintpy_geocode_lalo_step}",
        "mintpy.save.kmz                        = no",
        "",
    ]
    # NOT miaplpyApp.cfg: that name is MiaplPy's own default template and a
    # custom file called that is ignored ("ignored if the default
    # miaplpyApp.cfg is input"). The Sen/A|D/T<track> suffix is the naming
    # MiaplPy parses for the sensor.
    direction = "A" if str(getattr(cfg, "orbit_direction", "")).lower().startswith("a") else "D"
    path = work / f"{cfg.project_dir.name}Sen{direction}T{getattr(cfg, 'track', 0) or 0}.cfg"
    path.write_text("\n".join(lines))
    log.info("MiaplPy template: %s", path)
    return path


def sync_workdir_templates(work: Path, tpl: Path) -> int:
    """
    Push this template's values into the work-dir templates MiaplPy actually
    reads on a resume.

    On the first run MiaplPy copies our values into its own miaplpyApp.cfg and
    custom_smallbaselineApp.cfg in the work dir; on every later run it reads
    THOSE and ignores the file it is given. A setting changed between runs
    (the unwrap tile size that decides how many snaphu processes run at once,
    say) would otherwise never take effect. Returns the number of lines changed.
    """
    ours = {}
    for ln in tpl.read_text().splitlines():
        if "=" in ln and not ln.lstrip().startswith("#"):
            k, v = ln.split("=", 1)
            ours[k.strip()] = v.split("#", 1)[0].strip()
    changed = 0
    for name in ("miaplpyApp.cfg", "custom_smallbaselineApp.cfg", "smallbaselineApp.cfg"):
        f = work / name
        if not f.exists():
            continue
        out = []
        for ln in f.read_text().splitlines():
            key = ln.split("=", 1)[0].strip() if "=" in ln and not ln.lstrip().startswith("#") else None
            if key in ours:
                cur = ln.split("=", 1)[1].split("#", 1)[0].strip()
                if cur != ours[key]:
                    comment = ("  #" + ln.split("#", 1)[1]) if "#" in ln.split("=", 1)[1] else ""
                    ln = f"{key:<40s} = {ours[key]}{comment}"
                    changed += 1
            out.append(ln)
        f.write_text("\n".join(out) + "\n")
    if changed:
        log.info("Synced %d setting(s) into MiaplPy's work-dir templates", changed)
    return changed


def aligned_subset(cfg: PipelineConfig) -> tuple:
    """
    The phase-linking crop in full-resolution pixels: the stack's unwrap crop
    window (multilooked rows r0:r1, cols c0:c1, from crop_window.json)
    scaled by the looks, so multilooked pixel (r, c) of the linked
    interferograms is exactly merged-grid pixel (r, c).
    """
    import json
    import re
    cw = json.loads((cfg.stack_dir / "crop_window.json").read_text())
    al, rl = int(cfg.stack_azimuth_looks), int(cfg.stack_range_looks)
    # the .xml sidecar, not the raster: delete_merged_slcs_after_load removes
    # the rasters and keeps the sidecars, and this runs after it
    xmls = sorted((cfg.stack_dir / "merged" / "SLC").glob("*/*.slc.full.xml"))
    if not xmls:
        raise RuntimeError(f"no merged SLC metadata (*.slc.full.xml) under {cfg.stack_dir / 'merged' / 'SLC'}")
    t = xmls[0].read_text()
    W = int(re.search(r'<property name="width">\s*<value>(\d+)', t).group(1))
    L = int(re.search(r'<property name="length">\s*<value>(\d+)', t).group(1))
    y0, y1 = cw["r0"] * al, min(cw["r1"] * al, L)
    x0, x1 = cw["c0"] * rl, min(cw["c1"] * rl, W)
    return y0, y1, x0, x1


def repoint_full_vrts(stack: Path) -> int:
    """
    Make every merged *.full.vrt reference the real merged *.full file.

    topsStack's merge writes <name>.full as a real raster (with -V False) AND
    a <name>.full.vrt that still points at the per-burst inputs. MiaplPy opens
    the .vrt (read_image), so once the per-burst geometry or the coregistered
    bursts are cleaned up those VRTs read as empty and MiaplPy dies on
    'NoneType has no attribute shape'. Rewriting each VRT against the real
    file (GDAL's ISCE driver reads it) keeps the disk savings and the chain.
    Returns the number rewritten.
    """
    from osgeo import gdal
    import numpy as np
    gdal.UseExceptions()
    n = 0
    targets = sorted((stack / "merged" / "geom_reference").glob("*.rdr.full")) + \
              sorted((stack / "merged" / "SLC").glob("*/*.slc.full"))
    for full in targets:
        vrt = full.with_name(full.name + ".vrt")
        # Decide from the VRT's own sources, not from a probe read: GDAL fills
        # a 4x4 corner block that lies outside every SimpleSource's DstRect
        # with NoData without ever opening the (deleted) burst files, so a
        # probe read passes on a VRT that returns NaN for the whole image.
        if vrt.exists():
            srcs = _re.findall(r"<SourceFilename[^>]*>([^<]+)</SourceFilename>", vrt.read_text())
            if srcs and all((vrt.parent / Path(x)).resolve() == full.resolve() for x in srcs):
                continue
        try:
            ds = gdal.Open(str(full))
            ok = ds is not None and ds.GetRasterBand(1).ReadAsArray(0, 0, 4, 4) is not None
        except Exception:                                  # noqa: BLE001
            ok = False
        if not ok:
            log.warning("%s cannot be read by GDAL - VRT left as is", full)
            continue
        gdal.Translate(str(vrt), str(full), format="VRT")
        ds = gdal.Open(str(vrt))
        bx, by = min(64, ds.RasterXSize // 2), min(64, ds.RasterYSize // 2)
        blk = ds.GetRasterBand(1).ReadAsArray(ds.RasterXSize // 2, ds.RasterYSize // 2, bx, by)
        if blk is None or not np.isfinite(blk).all():
            raise RuntimeError(f"{vrt} still reads NaN after re-pointing at {full}")
        n += 1
    if n:
        log.info("Re-pointed %d merged .full.vrt files at their real rasters", n)
    return n


def stack_reference(cfg: PipelineConfig) -> str:
    """The stack's reference date: the config's, or - when it is left blank
    ("" = topsStack picks the first date) - the date reference/ was built
    from. Site B (29 Sep 2026) left it blank, so every '== reference'
    guard compared against "" and protected nothing."""
    if cfg.stack_reference_date:
        return cfg.stack_reference_date
    from .stack_backend import reference_dates
    ref = cfg.stack_dir / "reference"
    hits = sorted(reference_dates(ref)) if ref.exists() else []
    return hits[0] if len(hits) == 1 else ""


def _merged_slc_verified(cfg: PipelineConfig, date: str) -> bool:
    """The real merged SLC of a date: xml present, file size == width*length*8."""
    import re
    full = cfg.stack_dir / "merged" / "SLC" / date / f"{date}.slc.full"
    xml = full.with_name(full.name + ".xml")
    if not full.exists() or not xml.exists():
        return False
    t = xml.read_text()
    w = int(re.search(r'<property name="width">\s*<value>(\d+)', t).group(1))
    l = int(re.search(r'<property name="length">\s*<value>(\d+)', t).group(1))
    if full.stat().st_size != w * l * 8:
        log.warning("%s is %d bytes, expected %d", full, full.stat().st_size, w * l * 8)
        return False
    return True


def cleanup_coreg_date(cfg: PipelineConfig, date: str) -> float:
    """
    Delete one date's coregistered bursts once its REAL merged SLC exists.

    With stack_virtual_merge False the merged <date>.slc.full is a full copy
    (0.84 GB per date), so the bursts under coreg_secondarys/<date>/ are dead
    weight. Called per date from the merge stage (stack_backend.execute) so
    the bursts and the merged copies never coexist for the whole record:
    that is the difference between a decade fitting on the disk and not.
    The reference date's bursts stay (MiaplPy copies the reference directory
    and MintPy reads reference/IW*.xml). Returns GB reclaimed.
    """
    import shutil
    if date == stack_reference(cfg) or not _merged_slc_verified(cfg, date):
        return 0.0
    bursts = cfg.stack_dir / "coreg_secondarys" / date
    if not bursts.exists():
        return 0.0
    size = sum(f.stat().st_size for f in bursts.rglob("*") if f.is_file())
    shutil.rmtree(bursts)
    (cfg.stack_dir / "coreg_secondarys" / f"{date}.cleaned").write_text(
        "merged real SLC verified; bursts removed by miaplpy_backend\n")
    return size / 1e9


def cleanup_coreg_after_merge(cfg: PipelineConfig) -> float:
    """Every date the per-date hook did not reach (a resumed run)."""
    reclaimed = sum(cleanup_coreg_date(cfg, full.parent.name)
                    for full in sorted((cfg.stack_dir / "merged" / "SLC").glob("*/*.slc.full")))
    if reclaimed:
        log.info("Reclaimed %.1f GB of coregistered bursts (real merged SLCs verified)", reclaimed)
    return reclaimed


def merge_date_from_command(cmd: str) -> str | None:
    """'SentinelWrapper.py -c .../config_merge_20230104' -> '20230104'; None for the geometry merges."""
    m = _re.search(r"config_merge_(\d{8})(?:\s|$)", cmd)
    return m.group(1) if m else None


def delete_merged_slcs_after_load(cfg: PipelineConfig) -> float:
    """
    Delete the secondary dates' merged SLCs once MiaplPy's slcStack.h5 holds
    finite data for every date (run_step verifies that before this runs).
    Nothing downstream reads them: the interferograms come from the phase
    series, the unwrapper from the interferograms and the merged geometry,
    MintPy from reference/IW*.xml, baselines/ and geom_reference/. The
    reference date's files stay for metadata. Returns GB reclaimed.
    """
    import h5py
    work = work_dir(cfg)
    stack_f = work / "inputs" / "slcStack.h5"
    if not stack_f.exists():
        # deleted after a verified concatenation (miaplpy_delete_slc_stack_after_concatenate):
        # this ran on the load that made it, and a resume has nothing to check against
        log.info("%s is gone (deleted after concatenation) - merged SLCs left as they are", stack_f)
        return 0.0
    with h5py.File(stack_f, "r") as f:
        loaded = {d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]}
        bad = _empty_slc_dates(f)
        if bad:
            raise RuntimeError(
                f"{stack_f} holds NaN/zero data for {len(bad)} date(s) ({bad[:6]}) - an interrupted or broken "
                "load - so the merged SLCs are NOT deleted. Delete that file and restart 'Phase linking onward': "
                "load_data reloads it from the merged SLCs, which are still on disk.")
    ref = stack_reference(cfg)
    if not ref:
        log.warning("Cannot tell the stack's reference date - keeping the merged SLCs")
        return 0.0
    reclaimed = 0.0
    for d in sorted((cfg.stack_dir / "merged" / "SLC").glob("*/")):
        date = d.name
        if date == ref or date not in loaded:
            continue
        for f in d.glob(f"{date}.slc.full*"):
            if f.suffix in (".full",):
                reclaimed += f.stat().st_size / 1e9
                f.unlink()
        (d / "DELETED_AFTER_MIAPLPY_LOAD").write_text("merged SLC removed once slcStack.h5 was verified\n")
    if reclaimed:
        log.info("Reclaimed %.1f GB of merged SLCs (MiaplPy's SLC stack verified for every date)", reclaimed)
    return reclaimed


# ----------------------------------------------------------------------
# completion: what a resume may skip and what a cleanup may delete
# ----------------------------------------------------------------------
# A file's EXISTENCE says nothing here. MiaplPy's concatenate step deletes the
# old phase_series.h5 and creates the new one (invert.pyx unpatch ->
# initiate_output) BEFORE writing any block: `phase` is created with no fill
# value, so unwritten blocks read 0.0 - finite, and a probe for NaN passes
# them - and temporalCoherence is initialised to -1. Each block is written
# phase first, temporalCoherence LAST, clamped to >= 0. So "temporalCoherence
# >= 0 everywhere" is exactly "every block was written", and a stop inside the
# ~9-minute concatenation (05 Oct 2026 review: Site B 9.2 min after 704.6
# min of linking) used to be accepted on resume and the 53 GB of PATCHES
# deleted. The same holds for slcStack.h5: MiaplPy creates the dataset at full
# size and fills it date by date, and on a re-run it does NOT refill a dataset
# that already exists.
STATUS_DIR = ".status"                       # miaplpy/.status/<step>.done.json
CONCAT_MARKER = "phase_series.complete"      # miaplpy/inverted/, after a clean step AND a full scan
SCAN_ROWS = 512                              # rows of temporalCoherence[0] per band of the scan


def _status_file(work: Path, step: str) -> Path:
    return work / STATUS_DIR / f"{step}.done.json"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _file_id(path: Path) -> dict:
    st = Path(path).stat()
    return {"bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _write_step_marker(cfg: PipelineConfig, step: str, tpl: Path | None, seconds: float | None,
                       source: str, **extra) -> Path:
    """miaplpy/.status/<step>.done.json: what was verified, when, from which template."""
    import hashlib
    work = work_dir(cfg)
    rec = {"step": step, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source,
           "seconds": round(seconds, 1) if seconds is not None else None,
           "template_sha1": hashlib.sha1(Path(tpl).read_bytes()).hexdigest()[:16] if tpl and Path(tpl).exists() else None,
           **extra}
    out = _status_file(work, step)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    return out


def scan_phase_series(work: Path) -> dict:
    """
    Read temporalCoherence[0] of inverted/phase_series.h5 in bands and say
    whether every block was written (none still at MiaplPy's initial -1).
    Only the 2-D layer is read: Site B's 2604 x 9240 crop scans in seconds.
    Returns {"ok", "reason", "dates", "shape", "unwritten_cells", "unwritten_fraction"}.
    """
    import h5py
    ps = work / "inverted" / "phase_series.h5"
    rec = {"ok": False, "reason": "", "dates": None, "shape": None,
           "unwritten_cells": None, "unwritten_fraction": None}
    if not ps.exists():
        rec["reason"] = f"{ps} does not exist"
        return rec
    try:
        with h5py.File(ps, "r") as f:
            missing = [k for k in ("phase", "temporalCoherence", "date") if k not in f]
            if missing:
                rec["reason"] = f"{ps.name} has no {', '.join(missing)} dataset - MiaplPy's concatenation did not finish"
                return rec
            n, L, W = f["phase"].shape
            nd = len(f["date"])
            tc = f["temporalCoherence"]
            rec.update(dates=int(nd), shape=[int(n), int(L), int(W)])
            if n != nd or tuple(tc.shape[1:]) != (L, W):
                rec["reason"] = f"{ps.name} is inconsistent: phase {f['phase'].shape}, {nd} dates, temporalCoherence {tc.shape}"
                return rec
            bad = 0
            for r0 in range(0, L, SCAN_ROWS):
                bad += int((tc[0, r0:r0 + SCAN_ROWS, :] < 0).sum())
    except (OSError, KeyError, ValueError) as exc:
        rec["reason"] = f"{ps} cannot be read ({exc}) - most likely a concatenation killed mid-write"
        return rec
    rec["unwritten_cells"] = bad
    rec["unwritten_fraction"] = round(bad / max(1, L * W), 6)
    if bad:
        rec["reason"] = (f"{100 * bad / max(1, L * W):.1f}% of {ps.name} was never written (temporalCoherence still "
                         "at MiaplPy's initial -1): the concatenation was interrupted")
        return rec
    rec["ok"] = True
    return rec


def _write_concat_marker(work: Path, scan: dict, source: str) -> Path:
    ps = work / "inverted" / "phase_series.h5"
    m = work / "inverted" / CONCAT_MARKER
    m.write_text(json.dumps({"phase_series": str(ps), **_file_id(ps), "dates": scan["dates"],
                             "shape": scan["shape"], "unwritten_cells": scan["unwritten_cells"],
                             "verified": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": source}, indent=2))
    return m


def phase_series_complete(work: Path, write_marker: bool = True) -> bool:
    """
    True only when inverted/phase_series.h5 is a COMPLETE concatenation: its
    marker (written after a clean concatenate step and a full scan) describes
    the file as it is now, or - a run that predates the marker, or a file
    whose size/mtime moved - a fresh full scan passes, and then the marker is
    written so the scan runs once. Never existence alone.
    """
    ps = work / "inverted" / "phase_series.h5"
    if not ps.exists():
        return False
    rec = _read_json(work / "inverted" / CONCAT_MARKER)
    if rec and rec.get("bytes") == ps.stat().st_size and rec.get("mtime_ns") == ps.stat().st_mtime_ns:
        return True
    scan = scan_phase_series(work)
    if not scan["ok"]:
        log.warning("phase_series.h5 is not a complete concatenation: %s", scan["reason"])
        return False
    if write_marker:
        _write_concat_marker(work, scan, "full scan on resume (no current marker: the run predates it or the "
                                         "file changed since)")
        log.info("phase_series.h5 scanned complete (%d dates, %s) - marker written", scan["dates"], scan["shape"][1:])
    return True


def _slc_stack_verified(cfg: PipelineConfig, write_marker: bool = True) -> bool:
    """
    inputs/slcStack.h5 is a complete load: the load_data marker describes the
    file as it is (same size), or - a run that predates the marker - every
    date passes _empty_slc_dates, and then the marker is written.
    """
    import h5py
    work = work_dir(cfg)
    f = work / "inputs" / "slcStack.h5"
    if not f.exists():
        return False
    rec = _read_json(_status_file(work, "load_data"))
    if rec and rec.get("slc_stack_bytes") == f.stat().st_size:
        return True
    try:
        with h5py.File(f, "r") as h:
            if "slc" not in h or h["slc"].shape[0] < 2:
                return False
            bad = _empty_slc_dates(h)
            shape = list(h["slc"].shape)
    except (OSError, KeyError, ValueError) as exc:
        log.warning("%s cannot be read (%s) - not a complete load", f, exc)
        return False
    if bad:
        log.warning("%s has NaN/zero data for %d date(s) %s - not a complete load", f, len(bad), bad[:6])
        return False
    if write_marker:
        _write_step_marker(cfg, "load_data", None, None,
                           "checked on resume: every date holds data (the run predates the marker)",
                           slc_stack_bytes=f.stat().st_size, shape=shape)
        log.info("slcStack.h5 checked complete on resume (%s) - marker written", shape)
    return True


def delete_slc_stack_after_concatenate(cfg: PipelineConfig) -> float:
    """inputs/slcStack.h5 once phase_series.h5 is a verified, complete
    concatenation (its marker, see phase_series_complete); nothing later reads it."""
    work = work_dir(cfg); f = work / "inputs" / "slcStack.h5"
    if not f.exists():
        return 0.0
    if f.is_symlink():
        log.info("%s is a link to another project's SLC stack (miaplpy_relink_from): nothing of this project's "
                 "to delete", f)
        return 0.0
    if not phase_series_complete(work):
        raise RuntimeError("phase_series.h5 is missing or not a complete concatenation - slcStack.h5 kept")
    note = _read_json(work / "inverted" / CONCAT_MARKER) or {}
    size = f.stat().st_size / 1e9
    f.unlink()
    (work / "inputs" / "slcStack.h5.DELETED_AFTER_CONCATENATE").write_text(
        f"phase_series.h5 verified complete: {note.get('dates')} dates, shape {note.get('shape')}, "
        f"0 unwritten cells ({note.get('source', '')}, {note.get('verified', '')})\n")
    log.info("Reclaimed %.1f GB: MiaplPy's SLC stack (phase_series.h5 verified complete)", size)
    return size


def delete_patches_after_concatenate(cfg: PipelineConfig) -> float:
    """inverted/PATCHES once phase_series.h5 is a verified, complete
    concatenation (its marker after a full scan), never on a probe."""
    import shutil
    work = work_dir(cfg)
    ps = work / "inverted" / "phase_series.h5"; patches = work / "inverted" / "PATCHES"
    if not ps.exists() or not patches.exists():
        return 0.0
    if not phase_series_complete(work):
        raise RuntimeError(f"{ps} is not a complete concatenation ({scan_phase_series(work)['reason']}) - "
                           "PATCHES kept; re-run concatenation ('Phase linking onward')")
    note = _read_json(work / "inverted" / CONCAT_MARKER) or {}
    size = sum(f.stat().st_size for f in patches.rglob("*") if f.is_file()) / 1e9
    shutil.rmtree(patches)
    (work / "inverted" / "PATCHES.cleaned").write_text(
        f"phase_series.h5 verified complete: {note.get('dates')} dates, shape {note.get('shape')}, 0 unwritten "
        f"cells ({note.get('source', '')}, {note.get('verified', '')}); patches removed\n")
    log.info("Reclaimed %.1f GB of phase-linking patches (phase_series.h5 verified complete)", size)
    return size


def _command_argv(cfg: PipelineConfig, args: list) -> list:
    """The miaplpyApp command line, run in its own conda environment."""
    work = work_dir(cfg)
    # MiaplPy's step commands import isceobj (unwrap_ifgram.py, prep_slc_isce)
    # which lives under ISCE's components directory - the same PYTHONPATH the
    # stack route sets for itself (isceenv). Resolved inside the env at run time.
    isce_env = ('ISCE_HOME=$(python -c "import isce, os; print(os.path.dirname(isce.__file__))" 2>/dev/null | tail -n 1) && '
                'export ISCE_HOME && export PYTHONPATH="$ISCE_HOME/components:$ISCE_HOME/applications:$PYTHONPATH" && '
                'export PATH="$ISCE_HOME/applications:$PATH" && ')
    # every value is quoted and the env name checked (16 Oct 2026, readiness
    # review SEC-02 / RUN-04: a project folder with a space split the command)
    from .procs import conda_env_arg
    env = conda_env_arg(cfg.miaplpy_env, "miaplpy_env")
    cmd = (f"source {shlex.quote(str(CONDA_SH))} && conda activate {env} && {isce_env}cd {shlex.quote(str(work))} && "
           f"miaplpyApp {' '.join(shlex.quote(str(a)) for a in args)}")
    return ["bash", "-lc", cmd]


# Supervision of a running step. Phase linking runs ~12 h (Site B 704.6
# min) and used to report only start and done: its log goes to
# miaplpy/miaplpy_<step>.log, not logs/, and MiaplPy farms patches out with
# multiprocessing.Pool.map, which blocks FOREVER if a worker is killed (an
# OOM kill leaves no traceback either). Every POLL_S the supervisor counts
# finished patches (PATCH_*/flag.npy, MiaplPy's own completion sign) against
# the patch grid, or parses "Concatenate block i/N", and emits it as
# progress; anything moving - the step log growing (MiaplPy's per-worker
# progress bars write every few seconds while a patch runs), a new flag, the
# output file growing - counts as activity. Nothing moving for the stall
# limit stops the step and says why.
POLL_S = 60.0
STALL_FLOOR_S = 3600.0          # never less than an hour of silence
STALL_PATCH_FACTOR = 5.0        # ... nor less than 5x the measured time of one round of patches
_CONCAT_RE = _re.compile(rb"Concatenate block (\d+)/(\d+)")


def patch_count(length: int, width: int, range_window: int, azimuth_window: int, patch_size: int = 200) -> int:
    """MiaplPy's patch grid (invert.pyx patch_slice): rows start every patch_size
    over [0, length - azimuth_window), columns over [0, width - range_window).
    Site B 2604 x 9240, 15 x 15 windows, 200 px patches -> 13 x 47 = 611,
    the number its concatenation wrote (its run-file generator printed 658,
    from ceil(L/200) x ceil(W/200), which is not what runs)."""
    nr = len(range(0, max(0, int(length) - int(azimuth_window)), int(patch_size)))
    nc = len(range(0, max(0, int(width) - int(range_window)), int(patch_size)))
    return nr * nc


def _patch_size(work: Path) -> int:
    """miaplpy.inversion.patchSize from the work-dir template MiaplPy reads ('auto' = 200)."""
    for name in ("miaplpyApp.cfg",):
        f = work / name
        if f.exists():
            m = _re.search(r"^\s*miaplpy\.inversion\.patchSize\s*=\s*(\S+)", f.read_text(), _re.M)
            if m and m.group(1).isdigit():
                return int(m.group(1))
    return 200


def _patch_total(cfg: PipelineConfig) -> int | None:
    try:
        y0, y1, x0, x1 = aligned_subset(cfg)
        return patch_count(y1 - y0, x1 - x0, int(cfg.miaplpy_range_window), int(cfg.miaplpy_azimuth_window),
                           _patch_size(work_dir(cfg)))
    except Exception as exc:                                  # noqa: BLE001
        log.debug("patch total unknown: %s", exc)
        return None


def _patch_flags(work: Path) -> list:
    """mtimes of the finished patches (PATCH_*/flag.npy)."""
    out = []
    for f in (work / "inverted" / "PATCHES").glob("PATCH_*/flag.npy"):
        try:
            out.append(f.stat().st_mtime)
        except OSError:
            pass
    return out


def _log_tail(log_path: Path, offset: int = 0, n: int = 25, nbytes: int = 65536) -> str:
    """The last n lines of a step log after `offset`, progress-bar runs split
    and shortened, for the run log and error messages."""
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as fh:
            fh.seek(max(offset, size - nbytes))
            text = fh.read().decode(errors="replace")
    except OSError:
        return ""
    lines = [ln.strip()[:200] for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def _activity(work: Path, step: str, log_path: Path) -> tuple:
    """Everything whose change means the step is alive."""
    def sig(p):
        try:
            st = Path(p).stat()
            return (st.st_size, st.st_mtime_ns)
        except OSError:
            return None
    flags = len(_patch_flags(work)) if step == "phase_linking" else 0
    return (sig(log_path), flags, sig(work / "inverted" / "phase_series.h5"), sig(work / "inputs" / "slcStack.h5"))


def _supervise(cfg: PipelineConfig, proc, step: str, log_path: Path, offset: int, t0: float) -> int:
    """Wait for the step, emitting progress, and stop it if it stalls."""
    import math
    import signal
    from . import progress
    work = work_dir(cfg)
    stage = f"miaplpy_{step}"
    total = _patch_total(cfg) if step == "phase_linking" else None
    flags0 = len(_patch_flags(work)) if step == "phase_linking" else 0
    workers = max(1, int(getattr(cfg, "miaplpy_num_processors", 1) or 1))
    # the stall clock is monotonic and skips gaps between polls far longer than
    # the poll interval: a laptop asleep overnight (WSL paused with Windows)
    # must not read as an hour of silence when it wakes
    last_sig, last_change = _activity(work, step, log_path), time.monotonic()
    last_done, last_info, t_last_flag, last_poll = None, 0.0, None, time.monotonic()
    while True:
        try:
            return proc.wait(timeout=POLL_S)
        except subprocess.TimeoutExpired:
            pass
        now, mono = time.time(), time.monotonic()
        if mono - last_poll > 5 * POLL_S + 30:
            log.info("MiaplPy %s: %.0f min between checks (machine asleep?) - stall clock restarted",
                     step, (mono - last_poll) / 60)
            last_change = mono
        last_poll = mono
        done = tot = None
        if step == "phase_linking":
            flags = _patch_flags(work)
            done, tot = len(flags), total
            if len(flags) > flags0:
                t_last_flag = max(flags)
        elif step == "concatenate_patches":
            try:
                with open(log_path, "rb") as fh:
                    fh.seek(max(offset, log_path.stat().st_size - 65536))
                    hits = _CONCAT_RE.findall(fh.read())
                if hits:
                    done, tot = int(hits[-1][0]) + 1, int(hits[-1][1])
            except OSError:
                pass
        if done is not None and done != last_done:
            progress.emit(cfg.project_dir, stage, "running", done=done, total=tot, msg=f"log: {log_path}")
            last_done = done
            if now - last_info >= 600:          # one line in the run log at most every 10 min
                log.info("MiaplPy %s: %d%s %s done (%.0f min) - log %s", step, done, f"/{tot}" if tot else "",
                         "patches" if step == "phase_linking" else "blocks", (now - t0) / 60, log_path)
                last_info = now
        sig = _activity(work, step, log_path)
        if sig != last_sig:
            last_sig, last_change = sig, mono
        limit = STALL_FLOOR_S
        new = (done or 0) - flags0 if step == "phase_linking" else 0
        if new > 0 and t_last_flag:
            # one round = `workers` patches in flight; Site B: ~23 min a round
            per_round = max(0.0, t_last_flag - t0) / math.ceil(new / workers)
            limit = max(STALL_FLOOR_S, STALL_PATCH_FACTOR * per_round)
        if mono - last_change > limit:
            for sg in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sg)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    continue
            tail = _log_tail(log_path, offset)
            log.error("MiaplPy %s stalled - last lines of %s:\n%s", step, log_path, tail)
            what = (f"no patch finished ({done}{'/' + str(tot) if tot else ''} done) and its log stayed silent"
                    if step == "phase_linking" else "its log and outputs stayed unchanged")
            raise RuntimeError(
                f"MiaplPy {step} made no progress for {(mono - last_change) / 60:.0f} min - {what} - so it was "
                "stopped. The usual cause is a phase-linking worker killed for lack of memory, after which MiaplPy "
                "waits forever without an error. Check free memory (lower miaplpy_num_processors if it ran out), "
                "then restart 'Phase linking onward': finished patches are kept and only the rest re-run.\n"
                f"Last lines of {log_path}:\n{tail}")


def _run(cfg: PipelineConfig, args: list[str], log_name: str) -> int:
    """miaplpyApp in its own conda environment; stdout+stderr to the work dir.
    Runs in its own process group, registered for teardown (a Stop kills it,
    not just this process), under _supervise (progress and stall detection)."""
    from . import procs as _procs
    work = work_dir(cfg)
    argv = _command_argv(cfg, args)
    log_path = work / log_name
    step = str(args[args.index("--dostep") + 1]) if "--dostep" in args[:-1] else ""
    log.info("Running: %s  (log %s)", argv[-1], log_path)
    t0 = time.time()
    _procs.install_teardown()            # idempotent; the runner should already have done it
    with open(log_path, "a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {argv[-1]}\n")
        fh.flush()                      # the child writes to the fd next; keep the header first
        offset = fh.tell()
        proc = _procs.popen_group(argv, stdout=fh, stderr=subprocess.STDOUT,
                                  env={**os.environ, "OMP_NUM_THREADS": "1"})
        try:
            rc = _supervise(cfg, proc, step, log_path, offset, t0)
        finally:
            _procs.unregister(proc.pid)
    log.info("miaplpyApp %s finished rc=%s in %.1f min", args[-1] if args else "", rc, (time.time() - t0) / 60)
    return rc


def _expect(paths, what):
    missing = [str(p) for p in paths if not Path(p).exists() or Path(p).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"MiaplPy {what} did not produce: {missing}")


def _step_failed(log_path: Path, start_offset: int) -> str | None:
    """miaplpyApp returns 0 even when a step's command dies; the traceback is
    only in the log. Returns the tail of the log after the step started if a
    traceback appeared, else None."""
    # byte offset into bytes: the log holds non-ASCII progress bars, so a
    # character slice would start too early and pick up older tracebacks
    text = log_path.read_bytes()[start_offset:].decode(errors="replace")
    if ("Traceback (most recent call last)" in text or "MemoryError" in text or "Killed" in text
            or _re.search(r"^ERROR \d+: .*(No such file|not recognized as)", text, _re.M)):
        return "\n".join(text.strip().splitlines()[-25:])
    return None


# what each MiaplPy step must leave behind (relative to work / network dir)
STEP_OUTPUTS = {
    "load_data": lambda w, n: [w / "inputs" / "slcStack.h5", w / "inputs" / "geometryRadar.h5"],
    "phase_linking": lambda w, n: [w / "inverted"],
    "concatenate_patches": lambda w, n: [w / "inverted" / "phase_series.h5"],
    "generate_ifgram": lambda w, n: [w / "inverted"],
    "unwrap_ifgram": lambda w, n: [w / "inverted"],
    "load_ifgram": lambda w, n: [n / "inputs" / "ifgramStack.h5"],
    "ifgram_correction": lambda w, n: [n / "inputs" / "ifgramStack.h5"],
    "invert_network": lambda w, n: [n / "timeseries.h5", n / "temporalCoherence.h5", n / "maskTempCoh.h5"],
    "timeseries_correction": lambda w, n: [n / "velocity.h5"],
}


def _ifgram_dirs(work: Path) -> list:
    return sorted(d for d in (work / "inverted").glob("interferograms_*/*_*") if (d / "filt_fine.int").exists())


# ----------------------------------------------------------------------
# linking fingerprint: which settings made the phase series on disk
# ----------------------------------------------------------------------
# phase_series.h5 records none of them (Site B's 86 attrs: no method,
# mini-stack or lag), and sync_workdir_templates rewrites MiaplPy's own
# templates to the CURRENT config on every run. So a series linked with one
# setting was resumed, gated and exported as if it had been linked with
# another: the datum gate read the method from the config, and its own advice
# ("switch to EMI") skipped both the re-linking and the gate (review 5 Oct
# 2026). The fingerprint is written when phase linking starts and completed
# after a verified concatenation; a resume compares it with the config and
# stops on any difference, saying what to delete.
# Since 16 Oct 2026 (readiness review OUT-03) the comparison also covers
# linked dates the SLC stack no longer holds (removed_dates) and the MiaplPy
# build (its version label, recorded from the patch check as miaplpy_version
# since 5 Oct). Neither is in FINGERPRINT_KEYS or the sha, so an export made
# from a series is still recognised as its own; a fingerprint without them
# (none recorded) compares as before; and the pipeline's code hash is
# deliberately not compared - any edit would force 12 h of re-linking.
FINGERPRINT = "linking_fingerprint.json"          # miaplpy/inverted/
FINGERPRINT_KEYS = ("method", "ministack_size", "time_lag", "range_window", "azimuth_window", "subset_yx")
_PL_ARGS = {"method": rb"--method\s+(\S+)", "ministack_size": rb"--mini_stack_size\s+(\d+)",
            "time_lag": rb"--time_lag\s+(\d+)", "range_window": rb"--range_window\s+(\d+)",
            "azimuth_window": rb"--azimuth_window\s+(\d+)", "patch_size": rb"--patch_size\s+(\d+)"}


def fingerprint_path(cfg: PipelineConfig) -> Path:
    return work_dir(cfg) / "inverted" / FINGERPRINT


def linking_settings(cfg: PipelineConfig) -> dict:
    """The settings that change the linked phase, from the config. Not the
    ones that do not (processors, cleanup flags): those must stay editable."""
    try:
        subset = [int(v) for v in aligned_subset(cfg)]
    except Exception:                                          # noqa: BLE001
        subset = None
    return {"method": str(cfg.miaplpy_phase_linking_method).strip(),
            "ministack_size": int(cfg.miaplpy_ministack_size), "time_lag": int(cfg.miaplpy_time_lag),
            "range_window": int(cfg.miaplpy_range_window), "azimuth_window": int(cfg.miaplpy_azimuth_window),
            "subset_yx": subset}


def _settings_from_log(work: Path) -> dict | None:
    """What the LAST phase_linking.py command in MiaplPy's own log ran with:
    the measured record for a series linked before the fingerprint existed."""
    lp = work / "miaplpy_phase_linking.log"
    if not lp.exists():
        return None
    lines = _re.findall(rb"phase_linking\.py [^\n]*--method[^\n]*", lp.read_bytes())
    if not lines:
        return None
    out = {}
    for k, rx in _PL_ARGS.items():
        m = _re.search(rx, lines[-1])
        if m:
            v = m.group(1).decode(errors="replace")
            out[k] = v if k == "method" else int(v)
    return out or None


def _series_facts(work: Path) -> dict:
    """Dates and subset as phase_series.h5 itself records them (SUBSET_* attrs)."""
    import h5py
    try:
        with h5py.File(work / "inverted" / "phase_series.h5", "r") as f:
            dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
            a = f.attrs
            subset = ([int(a["SUBSET_YMIN"]), int(a["SUBSET_YMAX"]), int(a["SUBSET_XMIN"]), int(a["SUBSET_XMAX"])]
                      if all(k in a for k in ("SUBSET_YMIN", "SUBSET_YMAX", "SUBSET_XMIN", "SUBSET_XMAX")) else None)
        return {"dates": dates, "subset_yx": subset}
    except (OSError, KeyError, ValueError):
        return {}


def _slc_stack_dates(work: Path) -> list | None:
    import h5py
    f = work / "inputs" / "slcStack.h5"
    if not f.exists():
        return None
    try:
        with h5py.File(f, "r") as h:
            return sorted(d.decode() if isinstance(d, bytes) else str(d) for d in h["date"][:])
    except (OSError, KeyError, ValueError):
        return None


def available_dates(cfg: PipelineConfig) -> list | None:
    """Dates the record could be linked from now: the loaded SLC stack's, plus
    every merged date that can be loaded (.slc.full on disk) or was loaded and
    then removed (DELETED_AFTER_MIAPLPY_LOAD). None when neither is known."""
    have = set(_slc_stack_dates(work_dir(cfg)) or [])
    slc = cfg.stack_dir / "merged" / "SLC"
    if slc.exists():
        for d in slc.iterdir():
            if d.is_dir() and _re.fullmatch(r"\d{8}", d.name) and (
                    (d / f"{d.name}.slc.full").exists() or (d / "DELETED_AFTER_MIAPLPY_LOAD").exists()):
                have.add(d.name)
    return sorted(have) if have else None


def read_linking_fingerprint(cfg: PipelineConfig) -> dict | None:
    return _read_json(fingerprint_path(cfg))


def fingerprint_sha(fp: dict) -> str:
    """Hash of what makes the series what it is (settings + dates), not of when or how it was recorded."""
    import hashlib
    core = {k: fp.get(k) for k in FINGERPRINT_KEYS + ("dates",)}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


def write_linking_fingerprint(cfg: PipelineConfig, status: str, source: str, settings: dict | None = None,
                              dates: list | None = None, patch_check: dict | None = None) -> dict:
    """inverted/linking_fingerprint.json: the settings and dates the series is
    (being) linked with, the MiaplPy build and its patch check."""
    fp = dict(settings or linking_settings(cfg))
    old = read_linking_fingerprint(cfg) or {}
    pc = (patch_check if patch_check is not None else
          old.get("patch_check") or _read_json(work_dir(cfg) / "patch_check.json") or {})
    fp.update({"dates": dates, "n_dates": len(dates) if dates else None, "status": status, "source": source,
               "miaplpy_version": pc.get("miaplpy_version"), "patch_check_ratio": pc.get("ratio"),
               "patch_check": pc or None, "written": time.strftime("%Y-%m-%dT%H:%M:%S")})
    fp["sha"] = fingerprint_sha(fp)
    p = fingerprint_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fp, indent=2))
    return fp


def _legacy_fingerprint(cfg: PipelineConfig, status: str = "complete") -> dict:
    """A series (status "complete") or finished patches (status "linking")
    from before the fingerprint: record what made them from MiaplPy's own log
    and the series' attrs (measured), the config only for what neither holds
    (named as 'assumed')."""
    work = work_dir(cfg)
    cfg_now = linking_settings(cfg)
    from_log = _settings_from_log(work) or {}
    facts = _series_facts(work) if status == "complete" else {}
    settings = {k: from_log.get(k, cfg_now.get(k)) for k in FINGERPRINT_KEYS}
    if facts.get("subset_yx"):
        settings["subset_yx"] = facts["subset_yx"]
    assumed = [k for k in FINGERPRINT_KEYS if k not in from_log and not (k == "subset_yx" and facts.get("subset_yx"))]
    src = ("measured: miaplpy_phase_linking.log" + (" + phase_series.h5 attrs" if facts else "") if from_log else
           "settings ASSUMED from the config (no phase_linking.py line in MiaplPy's log)")
    fp = write_linking_fingerprint(cfg, status, src + (f"; assumed from the config: {assumed}" if assumed else ""),
                                   settings=settings,
                                   dates=facts.get("dates") if status == "complete" else _slc_stack_dates(work))
    log.warning("The %s predates the linking fingerprint - recorded it from %s: %s",
                "phase series" if status == "complete" else "linked patches",
                "MiaplPy's log" if from_log else "the CONFIG (nothing better on disk)",
                {k: settings[k] for k in FINGERPRINT_KEYS})
    return fp


def _ensure_fingerprint(cfg: PipelineConfig, complete: bool) -> None:
    """Something is linked but has no fingerprint (a run before 5 Oct 2026):
    record it from what is measurable, so the comparison has a basis."""
    if fingerprint_path(cfg).exists():
        return
    if complete:
        _legacy_fingerprint(cfg, "complete")
    elif _patch_flags(work_dir(cfg)):
        _legacy_fingerprint(cfg, "linking")


def removed_dates(cfg: PipelineConfig, linked: list | None) -> list:
    """Linked dates the SLC stack no longer holds - the stack a re-link (or
    the rest of an interrupted one) reads, since MiaplPy loads nothing while
    inputs/slcStack.h5 exists. Without a readable stack (deleted after a
    verified concatenation, or a project cleaned for the archive) nothing is
    claimed: merged folders cleaned by hand say nothing about the record."""
    stacked = _slc_stack_dates(work_dir(cfg))
    if not linked or not stacked:
        return []
    return sorted(set(linked) - set(stacked))


def installed_miaplpy_build(cfg: PipelineConfig) -> str | None:
    """The MiaplPy build this run links with: the version label from the
    patch check run() writes before any step (patch_check.json)."""
    return (_read_json(work_dir(cfg) / "patch_check.json") or {}).get("miaplpy_version") or None


def check_linking_fingerprint(cfg: PipelineConfig) -> dict | None:
    """
    Compare the fingerprint of the series (or of the patches being linked)
    with the current config. Raises with plain steps on any difference -
    settings, dates added since linking or linked dates since removed from
    the SLC stack, or another MiaplPy build; never deletes anything. A value
    the fingerprint does not record is not compared. Returns the
    fingerprint, or None when there is none.
    """
    fp = read_linking_fingerprint(cfg)
    if not fp:
        return None
    now = linking_settings(cfg)
    changed = []
    for k in FINGERPRINT_KEYS:
        a, b = fp.get(k), now.get(k)
        if a is None or b is None:
            continue
        if (str(a).lower() != str(b).lower()) if k == "method" else (a != b):
            changed.append(f"{k}: linked with {a}, config now {b}")
    avail = available_dates(cfg)
    added = sorted(set(avail) - set(fp["dates"])) if fp.get("dates") and avail else []
    if added:
        changed.append(f"dates: {len(added)} date(s) on disk were not in the linking ({added[:4]}"
                       f"{'...' if len(added) > 4 else ''})")
    removed = removed_dates(cfg, fp.get("dates"))
    if removed:
        changed.append(f"dates: {len(removed)} linked date(s) are no longer in inputs/slcStack.h5 ({removed[:4]}"
                       f"{'...' if len(removed) > 4 else ''})")
    built, build_now = fp.get("miaplpy_version"), installed_miaplpy_build(cfg)
    rebuilt = bool(built and build_now and str(built).strip() != str(build_now).strip())
    if rebuilt:
        changed.append(f"MiaplPy build: linked with {built}, installed now {build_now}")
    if not changed:
        return fp
    work = work_dir(cfg)
    inv = work / "inverted"
    stack_kept = (work / "inputs" / "slcStack.h5").exists()
    raise RuntimeError(
        f"The phase series in {inv} was linked with different settings than the config now asks for - "
        + "; ".join(changed) + ". Resuming would export (and gate) the old series as if it were the new one, so "
        "nothing was run. Either put the settings back in the YAML"
        + (f" (and the MiaplPy build {built} back in env '{cfg.miaplpy_env}')" if rebuilt else "")
        + f", or re-link: delete the folder {inv} and restart 'Phase linking onward' (phase linking re-runs, ~12 h)"
        + ("; inputs/slcStack.h5 is kept and reused" if stack_kept else
           "; inputs/slcStack.h5 was deleted, so load_data reloads it from the merged SLCs - if those were "
           "removed too, coregistration has to run again ('Whole pipeline')")
        + (". New dates also need the SLC stack reloaded: delete inputs/slcStack.h5 as well."
           if added else ".") )


def _patches_complete(cfg: PipelineConfig) -> bool:
    """Every patch of MiaplPy's grid carries flag.npy (its own completion sign,
    which its concatenation checks too)."""
    total = _patch_total(cfg)
    flags = len(_patch_flags(work_dir(cfg)))
    return bool(total) and flags >= total


def step_done(cfg: PipelineConfig, step: str) -> bool:
    """
    Verified outcome on disk that lets a step be skipped on resume - never a
    file's existence (see STATUS_DIR / CONCAT_MARKER). For phase linking and
    concatenation the linking fingerprint must also match the config; a
    mismatch raises (check_linking_fingerprint).
    """
    work = work_dir(cfg); net = network_dir(cfg)
    if step == "load_data":
        if phase_series_complete(work):
            return True           # a later stage is verified; the stack may have been deleted
        return _slc_stack_verified(cfg)
    if step in ("phase_linking", "concatenate_patches"):
        complete = phase_series_complete(work)
        if complete or (work / "inverted" / "phase_series.h5").exists() or _patch_flags(work):
            _ensure_fingerprint(cfg, complete)
            check_linking_fingerprint(cfg)        # something linked: it must be what the config asks for
        if step == "concatenate_patches" or complete:
            return complete
        return _patches_complete(cfg)
    if step == "generate_ifgram":
        return len(_ifgram_dirs(work)) > 0
    if step == "unwrap_ifgram":
        dirs = _ifgram_dirs(work)
        return bool(dirs) and all((d / "filt_fine.unw").exists() and (d / "filt_fine.unw.conncomp").exists() for d in dirs)
    if step == "load_ifgram":
        return (net / "inputs" / "ifgramStack.h5").exists()
    return False          # ifgram_correction, invert_network, timeseries_correction always run (MintPy's own update mode applies)


def _empty_slc_dates(f) -> list:
    """Dates whose SLC is NaN or all-zero on an interior 256x256 block, or
    all-zero on a block of the LAST rows where other dates hold data.

    MiaplPy writes each date as one full-image write, which fills chunks top
    to bottom, so a load stopped inside a date leaves that date's last rows at
    the fill value 0 while its centre may already be written. Judged against
    the other dates, so ground that is legitimately empty there (outside the
    bursts) is the same for every date and flags none."""
    import numpy as np
    ds = f["slc"]; n, L, W = ds.shape
    dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
    r0, c0 = max(0, L // 2 - 128), max(0, W // 2 - 128)
    rb = max(0, L - 16)           # inside the last chunk row, which is written last or not at all
    bad, bottom_empty, bottom_any = [], [], False
    for i in range(n):
        blk = ds[i, r0:r0 + 256, c0:c0 + 256]
        if not np.isfinite(blk).all() or not np.any(blk != 0):
            bad.append(dates[i])
            continue
        has = bool(np.any(ds[i, rb:L, c0:c0 + 256] != 0))
        bottom_any |= has
        if not has:
            bottom_empty.append(dates[i])
    if bottom_any:
        bad += bottom_empty
    return sorted(bad, key=dates.index)


def _before_step(cfg: PipelineConfig, step: str) -> None:
    """What must be true, or cleared, before a step runs at all."""
    work = work_dir(cfg)
    if step == "load_data":
        f = work / "inputs" / "slcStack.h5"
        _status_file(work, "load_data").unlink(missing_ok=True)
        if f.exists():
            # not verified (step_done said no) and MiaplPy will not refill a
            # dataset that already exists: re-running would keep the hole
            raise RuntimeError(
                f"{f} is an incomplete load (interrupted, or dates with no data - see the log above), and MiaplPy "
                "does not refill an existing stack. Nothing was deleted. Delete that file and restart 'Phase "
                "linking onward': load_data reloads it from the merged SLCs.")
    if step == "phase_linking":
        inv = work / "inverted"
        if not fingerprint_path(cfg).exists() or (not (inv / "phase_series.h5").exists() and not _patch_flags(work)):
            # nothing linked yet (or nothing the old fingerprint describes): record what will be,
            # with THIS run's patch check - not an abandoned linking's, which the
            # build comparison would then hold against it (16 Oct 2026, OUT-03)
            fp = write_linking_fingerprint(cfg, "linking", "config at the start of phase linking",
                                           dates=_slc_stack_dates(work),
                                           patch_check=_read_json(work / "patch_check.json") or {})
            fp["skip"] = write_skip_mask(cfg)
            fingerprint_path(cfg).write_text(json.dumps(fp, indent=2))
    if step == "concatenate_patches":
        # MiaplPy deletes phase_series.h5 the moment this step starts; a marker
        # must never outlive the file it vouches for
        (work / "inverted" / CONCAT_MARKER).unlink(missing_ok=True)


def run_step(cfg: PipelineConfig, tpl: Path, step: str) -> None:
    """One miaplpyApp step, verified: no traceback in its log, outputs present,
    and - for the steps a resume may skip - a completion marker written only
    after that verification (miaplpy/.status/<step>.done.json; for the
    concatenation also inverted/phase_series.complete after a full scan)."""
    work = work_dir(cfg); net = network_dir(cfg)
    if step_done(cfg, step):
        log.info("MiaplPy step %s already complete on disk - skipped", step)
        return
    _before_step(cfg, step)
    log_path = work / f"miaplpy_{step}.log"
    offset = log_path.stat().st_size if log_path.exists() else 0
    from . import progress
    progress.emit(cfg.project_dir, f"miaplpy_{step}", "start", msg=f"log: {log_path}")
    t0 = time.time()
    try:
        rc = _run(cfg, [tpl, "--dir", work, "--dostep", step], log_path.name)
    except Exception as exc:                                  # noqa: BLE001 - stalled or could not start
        progress.emit(cfg.project_dir, f"miaplpy_{step}", "failed", msg=str(exc).splitlines()[0][:200])
        raise
    tail = _step_failed(log_path, offset)
    if rc != 0 or tail:
        progress.emit(cfg.project_dir, f"miaplpy_{step}", "failed", msg=f"rc={rc}")
        raise RuntimeError(f"MiaplPy step {step} failed (rc={rc}) - see {log_path}\n"
                           f"{tail or _log_tail(log_path, offset)}")
    _expect(STEP_OUTPUTS[step](work, net), step)
    if step in ("generate_ifgram", "unwrap_ifgram") and not step_done(cfg, step):
        progress.emit(cfg.project_dir, f"miaplpy_{step}", "failed", msg="outputs missing")
        dirs = _ifgram_dirs(work)
        missing = [d.name for d in dirs if not (d / ("filt_fine.unw" if step == "unwrap_ifgram" else "filt_fine.int")).exists()]
        raise RuntimeError(f"MiaplPy {step}: {len(missing)} of {len(dirs)} interferograms without output: {missing[:6]} - see {log_path}")
    if step == "load_data":
        import h5py
        with h5py.File(work / "inputs" / "slcStack.h5", "r") as f:
            if "slc" not in f or f["slc"].shape[0] < 2:
                progress.emit(cfg.project_dir, f"miaplpy_{step}", "failed", msg="no SLC stack")
                raise RuntimeError("MiaplPy load_data wrote no SLC stack (see the compression setting)")
            bad = _empty_slc_dates(f)
            if bad:
                progress.emit(cfg.project_dir, f"miaplpy_{step}", "failed", msg=f"{len(bad)} empty dates")
                raise RuntimeError(
                    f"MiaplPy load_data read NaN/zero SLCs for {len(bad)} date(s): {bad[:6]} - the merged "
                    "*.slc.full.vrt files point at inputs that no longer exist (see repoint_full_vrts); "
                    f"delete {work / 'inputs' / 'slcStack.h5'} and restart 'Phase linking onward'")
            shape = list(f["slc"].shape)
        _write_step_marker(cfg, step, tpl, time.time() - t0, "verified after the step: rc 0, clean log, every date holds data",
                           slc_stack_bytes=(work / "inputs" / "slcStack.h5").stat().st_size, shape=shape)
        log.info("SLC stack loaded: %s, every date holds finite data", shape)
    if step == "phase_linking":
        total, flags = _patch_total(cfg), len(_patch_flags(work))
        open_ = sorted(d.name for d in (work / "inverted" / "PATCHES").glob("PATCH_*")
                       if d.is_dir() and not (d / "flag.npy").exists())
        if open_:
            # started and never finished: certain, whatever the grid arithmetic says
            progress.emit(cfg.project_dir, f"miaplpy_{step}", "failed", msg=f"{len(open_)} patches unfinished")
            raise RuntimeError(f"MiaplPy phase_linking returned rc 0 with {len(open_)} patch(es) unfinished "
                               f"({open_[:4]}) - see {log_path}; restart 'Phase linking onward' to link the rest "
                               "(finished patches are kept)")
        if total and flags < total:
            # MiaplPy's own concatenation refuses unflagged patches; say it here first
            log.warning("MiaplPy phase_linking: %d patches flagged, the %dx%d-window grid has %d - the concatenation "
                        "will check every patch", flags, cfg.miaplpy_range_window, cfg.miaplpy_azimuth_window, total)
        _write_step_marker(cfg, step, tpl, time.time() - t0, "verified after the step: rc 0, clean log, no patch unfinished",
                           patches=flags, patch_total=total)
    if step == "concatenate_patches":
        scan = scan_phase_series(work)
        if not scan["ok"]:
            progress.emit(cfg.project_dir, f"miaplpy_{step}", "failed", msg=scan["reason"][:200])
            raise RuntimeError(f"MiaplPy concatenation ended rc 0 but {scan['reason']}. PATCHES were kept; "
                               f"restart 'Phase linking onward' to concatenate again - see {log_path}")
        _write_concat_marker(work, scan, "verified after the step: rc 0, clean log, full scan of temporalCoherence")
        _write_step_marker(cfg, step, tpl, time.time() - t0, "verified after the step: rc 0, clean log, full scan",
                           dates=scan["dates"], shape=scan["shape"])
        # complete the fingerprint written when linking started, with what the
        # series itself records (its dates and SUBSET_* attrs)
        old = read_linking_fingerprint(cfg) or {}
        settings = {**linking_settings(cfg), **{k: old[k] for k in FINGERPRINT_KEYS if old.get(k) is not None}}
        facts = _series_facts(work)
        if facts.get("subset_yx"):
            settings["subset_yx"] = facts["subset_yx"]
        fp = write_linking_fingerprint(cfg, "complete", old.get("source") or "config at the start of phase linking",
                                       settings=settings, dates=facts.get("dates"))
        if old.get("skip") is not None:                       # informational: not in FINGERPRINT_KEYS or the sha
            fp["skip"] = old["skip"]
            fingerprint_path(cfg).write_text(json.dumps(fp, indent=2))
        log.info("phase_series.h5 complete: %d dates, %s, every block written", scan["dates"], scan["shape"][1:])
    progress.emit(cfg.project_dir, f"miaplpy_{step}", "done")
    log.info("MiaplPy step %s verified", step)


def network_pairs(dates: list, connections: int, ref: str) -> list:
    """(A, B, ia, ib) with A earlier: N nearest connections per date, or the
    single-reference star when connections == 0."""
    n = len(dates)
    if connections <= 0:
        ri = dates.index(ref)
        return [tuple(sorted([ref, d])) + ((ri, k) if ref < d else (k, ri)) for k, d in enumerate(dates) if d != ref]
    out = []
    for a in range(n):
        for b in range(a + 1, min(n, a + connections + 1)):
            out.append((dates[a], dates[b], a, b))
    return out


def mini_stack_index(k: int, n: int, size: int) -> int:
    """Which sequential mini-stack image k belongs to (MiaplPy's partition:
    n // size stacks, the last one absorbing the remainder)."""
    total = max(1, n // size)
    return min(k // size, total - 1)


# The datum gate (check_datum_consistency). Measured on cells that are
# coherent on BOTH sides of a boundary, sampled across the whole crop, and each
# boundary judged against its own neighbouring within-window pairs. The first
# version took the median over every cell of one fixed central block with an
# absolute 0.5 floor: on Site B 14.4% of cells read < 0.5 across boundaries
# with no datum error at all (5 Oct 2026 review), so a surface mine filling
# the AOI centre, or a short record with one weak boundary (Site B's
# weakest: 0.48), could fail a healthy series after 12 h. Decorrelated ground
# reads low inside and across alike and tells nothing either way; only
# coherent cells can show a datum step.
GATE_COHERENT = 0.7            # both neighbouring within-window pairs at least this, per cell
GATE_RATIO = 0.6               # across / neighbours, median over boundaries
GATE_FLOOR = 0.5               # absolute floor on the across median ...
GATE_FLOOR_MIN_BOUNDARIES = 3  # ... applied only with this many judged boundaries
GATE_TILES = 4                 # up to 4 x 4 tiles spread over the crop
GATE_TILE_CELLS = 50           # each 50 x 50 multilooked cells (150 x 550 px at 3 x 11 looks; 16 tiles ~ the old block)


def _gate_tiles(L: int, W: int, al: int, rl: int) -> list:
    """(r0, r1, c0, c1) full-resolution tiles, look-aligned, spread over the crop."""
    th = min(GATE_TILE_CELLS * al, (L // al) * al)
    tw = min(GATE_TILE_CELLS * rl, (W // rl) * rl)
    if th <= 0 or tw <= 0:
        return []
    ny, nx = max(1, min(GATE_TILES, L // th)), max(1, min(GATE_TILES, W // tw))
    rows = sorted({int(r) - int(r) % al for r in np.linspace(0, L - th, ny)})
    cols = sorted({int(c) - int(c) % rl for c in np.linspace(0, W - tw, nx)})
    return [(r, r + th, c, c + tw) for r in rows for c in cols]


def _mines_lonlat(gj):
    """Union of the mapped mine polygons (lon/lat), or None when there are none."""
    import shapely
    from shapely.geometry import shape
    geoms = []
    for feat in json.loads(Path(gj).read_text()).get("features", []):
        g = (feat or {}).get("geometry") or {}
        if g.get("type") in ("Polygon", "MultiPolygon"):
            s = shape(g)
            s = s if s.is_valid else s.buffer(0)
            if not s.is_empty:
                geoms.append(s)
    return shapely.union_all(geoms) if geoms else None


def _gate_mine_masks(cfg, tiles: list, al: int, rl: int, subset) -> list | None:
    """Per tile, which multilooked cells lie inside a mapped mine (excluded
    from the gate); None when the mines or the geometry are not available."""
    import shapely
    gj = getattr(cfg, "deramp_exclude_geojson", None)
    if not gj or not Path(gj).exists() or not subset:
        return None
    try:
        cw = json.loads((cfg.stack_dir / "crop_window.json").read_text())
        Lm, Wm = cw["length"], cw["width"]
        G = cfg.stack_dir / "merged" / "geom_reference"
        lat = np.memmap(G / "lat.rdr", dtype=np.float64, mode="r", shape=(Lm, Wm))
        lon = np.memmap(G / "lon.rdr", dtype=np.float64, mode="r", shape=(Lm, Wm))
        mines = _mines_lonlat(gj)
        if mines is None:
            return None
        shapely.prepare(mines)
        out = []
        for r0, r1, c0, c1 in tiles:
            rr, cc = (int(subset[0]) + r0) // al, (int(subset[2]) + c0) // rl
            la = np.asarray(lat[rr:rr + (r1 - r0) // al, cc:cc + (c1 - c0) // rl])
            lo = np.asarray(lon[rr:rr + (r1 - r0) // al, cc:cc + (c1 - c0) // rl])
            out.append(shapely.contains_xy(mines, lo.ravel(), la.ravel()))
        return out
    except Exception as exc:                                   # noqa: BLE001 - the gate runs without it
        log.warning("Datum gate: mapped mines not excluded (%s)", exc)
        return None


def check_datum_consistency(cfg: PipelineConfig, ph, dates: list, linked: dict | None = None) -> dict:
    """
    Guard against the defect measured on 24 Sep 2026: with the sequential
    estimators upstream MiaplPy anchors every mini-stack to the first
    mini-stack's compressed image, and over ground that decorrelates across
    years each mini-stack ends up with its own random per-pixel datum.
    Consecutive-date pairs inside a mini-stack then read ~0.9 block coherence
    while the same pairs across a boundary read ~0.4, and no pair spanning a
    boundary can be unwrapped.

    The method and mini-stack size are the ones the series was LINKED with
    (`linked`, the linking fingerprint), never the config's: a config edited
    after linking used to switch the gate off (5 Oct 2026). Judged on cells
    coherent in both neighbouring within-window pairs (GATE_COHERENT), sampled
    over the whole crop, outside the mapped mines; raises when the median
    boundary reads below GATE_RATIO of its neighbours, or below GATE_FLOOR
    once GATE_FLOOR_MIN_BOUNDARIES boundaries are judged.
    """
    n, L, W = ph.shape
    al, rl = int(cfg.stack_azimuth_looks), int(cfg.stack_range_looks)
    lk = linked or {}
    method = str(lk.get("method") or cfg.miaplpy_phase_linking_method)
    size = int(lk.get("ministack_size") or cfg.miaplpy_ministack_size)
    seq = method.lower().startswith("sequential")
    subset = lk.get("subset_yx")
    if not subset:
        try:
            subset = list(aligned_subset(cfg))
        except Exception:                                      # noqa: BLE001
            subset = None
    tiles = _gate_tiles(L, W, al, rl)
    mines = _gate_mine_masks(cfg, tiles, al, rl, subset)
    parts = []
    for t, (r0, r1, c0, c1) in enumerate(tiles):
        h, w = (r1 - r0) // al, (c1 - c0) // rl
        blk = np.asarray(ph[:, r0:r1, c0:c1], dtype=np.float32)
        d = blk[:-1] - blk[1:]
        del blk
        re_ = np.cos(d).reshape(n - 1, h, al, w, rl).mean(axis=(2, 4))
        im_ = np.sin(d).reshape(n - 1, h, al, w, rl).mean(axis=(2, 4))
        del d
        c = np.hypot(re_, im_).reshape(n - 1, h * w).astype(np.float32)
        if mines is not None:
            c[:, mines[t]] = np.nan
        parts.append(c)
    C = np.concatenate(parts, axis=1) if parts else np.zeros((max(0, n - 1), 0), np.float32)
    ms = [mini_stack_index(k, n, size) for k in range(n)]
    within = [k for k in range(n - 1) if ms[k] == ms[k + 1]]
    cross = [k for k in range(n - 1) if ms[k] != ms[k + 1]]
    with np.errstate(all="ignore"):
        pair_med = np.nanmedian(C, axis=1) if C.shape[1] else np.full(n - 1, np.nan)
    w_all = float(np.nanmedian(pair_med[within])) if within else float("nan")
    x_all = float(np.nanmedian(pair_med[cross])) if cross else float("nan")
    cells = int(np.isfinite(C[0]).sum()) if C.shape[0] else 0
    min_cells = max(10, min(100, int(0.05 * cells)))
    wset, per = set(within), []
    for k in cross:
        nb = [j for j in (k - 1, k + 1) if j in wset]
        rec = {"pair": f"{dates[k]}_{dates[k + 1]}" if len(dates) > k + 1 else str(k), "judged": False}
        if not nb:
            per.append({**rec, "reason": "no within-window neighbour pair"}); continue
        with np.errstate(invalid="ignore"):
            sel = np.all(C[nb] >= GATE_COHERENT, axis=0) & np.isfinite(C[k])
        ns = int(sel.sum())
        if ns < min_cells:
            per.append({**rec, "coherent_cells": ns, "reason": f"fewer than {min_cells} coherent cells"}); continue
        x = float(np.median(C[k, sel])); wn = float(np.median(C[nb][:, sel].mean(axis=0)))
        per.append({**rec, "judged": True, "coherent_cells": ns, "across": round(x, 3), "neighbours": round(wn, 3),
                    "ratio": round(x / wn, 3) if wn > 0 else None})
    judged = [p for p in per if p["judged"] and p["ratio"] is not None]
    x_med = float(np.median([p["across"] for p in judged])) if judged else float("nan")
    w_med = float(np.median([p["neighbours"] for p in judged])) if judged else float("nan")
    r_med = float(np.median([p["ratio"] for p in judged])) if judged else float("nan")
    fail = bool(seq and judged and (r_med < GATE_RATIO or (len(judged) >= GATE_FLOOR_MIN_BOUNDARIES and x_med < GATE_FLOOR)))
    info = {"method": method, "mini_stack_size": size,
            "method_source": "linking fingerprint" if lk.get("method") else "config (no fingerprint)",
            "config_method": str(getattr(cfg, "miaplpy_phase_linking_method", "")),
            "config_mini_stack_size": getattr(cfg, "miaplpy_ministack_size", None),
            "consecutive_pairs_within_mini_stack": len(within), "consecutive_pairs_across_boundary": len(cross),
            # every sampled cell, as the first version measured (for comparison with older records)
            "median_block_coherence_within": round(w_all, 3), "median_block_coherence_across": round(x_all, 3),
            "ratio_across_over_within": round(x_all / w_all, 3) if within and cross and w_all > 0 else None,
            "coherent_cells": {"threshold": GATE_COHERENT, "min_cells_per_boundary": min_cells,
                               "boundaries_judged": len(judged), "boundaries_not_judged": len(per) - len(judged),
                               "median_across": round(x_med, 3) if judged else None,
                               "median_neighbours": round(w_med, 3) if judged else None,
                               "median_ratio": round(r_med, 3) if judged else None, "per_boundary": per},
            "rule": (f"fails when the median boundary ratio < {GATE_RATIO}, or the median across < {GATE_FLOOR} with "
                     f">= {GATE_FLOOR_MIN_BOUNDARIES} judged boundaries; sequential methods only"),
            "testable": bool(judged) if seq else False, "passed": not fail,
            "cells_sampled": cells, "tiles_full_res_rows_cols": [list(map(int, t)) for t in tiles],
            "mines_excluded": mines is not None}
    out = cfg.stack_dir / "merged" / "linked_datum_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    log.info("Linked-phase datum check (%s): on coherent cells consecutive pairs read %.2f across mini-stack "
             "boundaries vs %.2f for their within-window neighbours (median ratio %.2f, %d of %d boundaries judged; "
             "all cells %.2f / %.2f)", method, x_med, w_med, r_med, len(judged), len(cross), x_all, w_all)
    if seq and cross and not judged:
        log.warning("Datum gate could not judge any of the %d mini-stack boundaries: no cells coherent on both sides "
                    "of them in the sample - recorded in %s, not a pass", len(cross), out)
    if fail:
        work = work_dir(cfg) if hasattr(cfg, "project_dir") else Path("miaplpy")
        lag = getattr(cfg, "miaplpy_time_lag", "?")
        keep = ("inputs/slcStack.h5 is kept" if not getattr(cfg, "miaplpy_delete_slc_stack_after_concatenate", False)
                else "load_data reloads the SLC stack from the merged SLCs if they are still on disk")
        raise RuntimeError(
            f"MiaplPy's {method} phase series has its own datum in each mini-stack: on cells coherent either side "
            f"of a boundary, consecutive pairs across it read {x_med:.2f} block coherence against {w_med:.2f} for "
            f"their within-window neighbours (median ratio {r_med:.2f} over {len(judged)} boundaries). No pair "
            "spanning a boundary can be unwrapped from this series. Two things cause this: the windows are not tied "
            "together - the MiaplPy build is not the patched one, or its window overlap (miaplpy_time_lag, now "
            f"{lag}) is too short for this ground - or the ground changed abruptly at those dates (snow, flooding, "
            "ploughing) on most coherent cells. To re-link: run patches/miaplpy/check_miaplpy_patch.py "
            f"--ministack {size} --time-lag {lag} in the MiaplPy env (it must say patched), raise miaplpy_time_lag "
            f"(e.g. 3 -> 5), delete {work / 'inverted'} ({keep}), then restart 'Phase linking onward'. Do not switch "
            "to non-sequential EMI: it solves the whole stack per pixel and did not finish (HANDOFF.md 25 Sep, "
            f"'Phase linking'). Per-boundary figures: {out}")
    return info


def export_phase_linked_ifgrams(cfg: PipelineConfig) -> list:
    """
    Interferograms from MiaplPy's linked phase series, on the stack's
    multilook grid, as topsStack files.

    Pairs are the N nearest connections per date (miaplpy_network_connections,
    0 = single-reference star). A pair is named by date order (A_B with A
    earlier) and its phase is phi(A) - phi(B), the ISCE convention MintPy's
    reader expects. Unit-amplitude phasors are averaged over azimuth_looks x
    range_looks blocks: the mean's phase is the multilooked interferogram and
    its magnitude the coherence of the linked phase over the block
    (filt_fine.cor). Both are padded with zeros into the full merged frame so
    the unwrapper, the crop window, the geometry and MintPy see exactly the
    files the classic route produces. The datum check runs first.
    """
    import h5py
    from .isceenv import import_isce
    _, isceobj = import_isce()
    stack = cfg.stack_dir; work = work_dir(cfg)
    cw = json.loads((stack / "crop_window.json").read_text())
    Lm, Wm = cw["length"], cw["width"]
    al, rl = int(cfg.stack_azimuth_looks), int(cfg.stack_range_looks)
    y0, y1, x0, x1 = aligned_subset(cfg)
    ref = stack_reference(cfg)
    out_root = stack / "merged" / "interferograms"; out_root.mkdir(parents=True, exist_ok=True)
    written = []
    linked = read_linking_fingerprint(cfg)
    with h5py.File(work / "inverted" / "phase_series.h5", "r") as f:
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
        ph = f["phase"]
        n, L, W = ph.shape
        assert (L, W) == (y1 - y0, x1 - x0), f"phase_series is {L}x{W}, aligned crop is {y1 - y0}x{x1 - x0}: reload with the aligned subset"
        probe = ph[min(1, n - 1), L // 2 - 128:L // 2 + 128, W // 2 - 128:W // 2 + 128]
        if not np.isfinite(probe).any():
            raise RuntimeError("MiaplPy's phase_series.h5 is NaN over the image centre - phase linking ran on an "
                               "empty SLC stack; delete miaplpy/inputs/slcStack.h5 and miaplpy/inverted and rerun")
        check_datum_consistency(cfg, ph, dates, linked=linked)
        Lc, Wc = L // al, W // rl
        r0, c0 = y0 // al, x0 // rl
        pairs = network_pairs(dates, int(cfg.miaplpy_network_connections), ref)
        probe = _series_probe(ph)
        stale = _export_stale(cfg, out_root, linked, dates, probe)
        todo = [(a, b, ia, ib) for a, b, ia, ib in pairs
                if stale or not ((out_root / f"{a}_{b}" / "filt_fine.int").exists() and (out_root / f"{a}_{b}" / "filt_fine.cor").exists())]
        written = [f"{a}_{b}" for a, b, _, _ in pairs if (a, b) not in {(t[0], t[1]) for t in todo}]
        cache = {}

        # skipped pixels (write_skip_mask) carry their raw single-pixel phase:
        # they add nothing to a block, and a block of them only is empty
        keep = load_skip_keep(cfg, (L, W))
        keep = None if keep is None else keep[:Lc * al, :Wc * rl].astype(np.float32)

        def date_phase(k):
            if k not in cache:
                cache[k] = ph[k, :Lc * al, :Wc * rl].astype(np.float32)
                for old in [q for q in cache if q < k - max(1, int(cfg.miaplpy_network_connections))]:
                    cache.pop(old)
            return cache[k]

        for a, b, ia, ib in sorted(todo, key=lambda t: (t[2], t[3])):
            pair = f"{a}_{b}"; pdir = out_root / pair; pdir.mkdir(exist_ok=True)
            z = np.exp(1j * (date_phase(ia) - date_phase(ib)))
            if keep is not None:
                z = z * keep
            z = z.reshape(Lc, al, Wc, rl).mean(axis=(1, 3))
            ifg = np.zeros((Lm, Wm), np.complex64); cor = np.zeros((Lm, Wm), np.float32)
            ifg[r0:r0 + Lc, c0:c0 + Wc] = z.astype(np.complex64)
            cor[r0:r0 + Lc, c0:c0 + Wc] = np.abs(z).astype(np.float32)
            _write_isce(isceobj, pdir / "filt_fine.int", ifg, "CFLOAT")
            _write_isce(isceobj, pdir / "filt_fine.cor", cor, "FLOAT")
            written.append(pair)
    written = sorted(set(written))
    _write_export_manifest(out_root, linked, dates, written, "written" if todo else "unchanged", probe)
    others = sorted(d.name for d in out_root.iterdir() if d.is_dir() and _re.fullmatch(r"\d{8}_\d{8}", d.name)
                    and d.name not in set(written) and (d / "filt_fine.unw").exists())
    if others:
        # MintPy's loader globs */filt_fine.unw: these would enter the inversion
        log.warning("%d unwrapped pair folder(s) in %s are not in this network (%s...) - MintPy loads every "
                    "*/filt_fine.unw there; move them out if the network was changed on purpose",
                    len(others), out_root, others[:4])
    log.info("Wrote %d phase-linked, %dx%d-looked interferograms (%s) into %s", len(written), rl, al,
             "single-reference star" if int(cfg.miaplpy_network_connections) <= 0
             else f"{int(cfg.miaplpy_network_connections)} nearest connections per date", out_root)
    export_linked_temporal_coherence(cfg)
    return written


EXPORT_MANIFEST = "export_manifest.json"        # stack/merged/interferograms/


def _series_probe(ph) -> str:
    """A content hash of a few rows of three dates of the phase series: the
    same series reads the same wherever it is copied, and a re-link - even
    with identical settings - reads differently."""
    import hashlib
    n, L, W = ph.shape
    h = hashlib.sha256()
    for k in sorted({0, n // 2, n - 1}):
        for r in (L // 4, L // 2, (3 * L) // 4):
            h.update(np.ascontiguousarray(ph[k, r, ::max(1, W // 256)], dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _series_key(linked: dict | None, dates: list, probe: str = "") -> str:
    """The export's tie to the series: the fingerprint's hash (or, without
    one, the dates) together with the series' content probe."""
    import hashlib
    base = (linked.get("sha") or fingerprint_sha(linked)) if linked else fingerprint_sha({"dates": list(dates)})
    return hashlib.sha256(f"{base}:{probe}".encode()).hexdigest()[:16] if probe else base


def _write_export_manifest(out_root: Path, linked: dict | None, dates: list, pairs: list, what: str,
                           probe: str = "") -> Path:
    m = out_root / EXPORT_MANIFEST
    m.write_text(json.dumps({"fingerprint_sha": _series_key(linked, dates, probe), "series_probe": probe,
                             "linking_fingerprint_sha": (linked or {}).get("sha"), "n_dates": len(dates),
                             "pairs": list(pairs), "status": what,
                             "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    return m


def _export_stale(cfg: PipelineConfig, out_root: Path, linked: dict | None, dates: list, probe: str = "") -> bool:
    """
    True when the pair files on disk were exported from a DIFFERENT phase
    series (the manifest's fingerprint hash differs): every pair must then be
    rewritten and re-unwrapped. Their unwrap products - derived from the old
    interferograms and otherwise reused by file existence - are removed here,
    with the sample-coherence record that would skip them, and the reason is
    logged. Pair folders without a manifest (exports before 5 Oct 2026) are
    adopted as the current series' export.
    """
    rec = _read_json(out_root / EXPORT_MANIFEST)
    key = _series_key(linked, dates, probe)
    if rec is None or rec.get("fingerprint_sha") == key:
        return False
    removed = 0
    for d in sorted(out_root.iterdir()):
        if not (d.is_dir() and _re.fullmatch(r"\d{8}_\d{8}", d.name)):
            continue
        for p in list(d.glob("filt_fine.unw*")) + list(d.glob("filt_fine_crop.*")) + \
                [d / "unwrap_crop.json", d / "filt_fine.phasor.cor", d / "filt_fine.phasor.cor.xml",
                 d / "filt_fine.phasor.cor.vrt", d / "fast_zone_original.npz"]:
            if p.exists():
                p.unlink(); removed += 1
    (cfg.stack_dir / "merged" / "pair_coherence.json").unlink(missing_ok=True)
    log.warning("The exported pairs in %s came from a different phase series (manifest %s, series now %s): every "
                "pair is rewritten and re-unwrapped; %d stale unwrap file(s) removed", out_root,
                rec.get("fingerprint_sha"), key, removed)
    return True


def linked_temporal_coherence_path(cfg: PipelineConfig) -> Path:
    return cfg.stack_dir / "merged" / "linked_temporal_coherence.npz"


# The layer the route grades pixels on. MiaplPy writes two temporal
# coherences (lib/invert.pyx: temporalCoherence[0] -> tempCoh_average,
# [1] -> tempCoh_full). 'full' is the fit of the linked phase to the WHOLE
# stack's covariance: on this farmland, which keeps no coherence across a
# season, it is ~0.03 everywhere and grades nothing. 'average' is the fit
# within each mini-stack window (<= 13 images, ~5 months): a measure of how
# well the short-span phase is estimated, blind to datum errors between
# windows and to unwrapping errors. It is NOT comparable with MintPy's
# network-inversion temporal coherence on the small-baseline route, so
# coverage figures of the two routes must not be set side by side as like.
LINKED_COHERENCE_LAYER = "average"
_LAYER_ORDER = "miaplpy temporalCoherence index 0 = average (mini-stack), 1 = full (whole stack)"


def load_linked_temporal_coherence(cfg: PipelineConfig, layer: str = LINKED_COHERENCE_LAYER) -> np.ndarray:
    """
    One layer of merged/linked_temporal_coherence.npz. Files written before
    27 Sep 2026 carry no `layer_order` record and have the two names SWAPPED
    (index 0 was stored as 'full'); they are read the right way round here.
    """
    with np.load(linked_temporal_coherence_path(cfg)) as z:
        if "layer_order" in z.files:
            return z[layer]
        log.warning("%s predates the layer-order fix: its 'full' is MiaplPy's mini-stack AVERAGE and its "
                    "'average' the whole-stack figure - reading %r from %r", linked_temporal_coherence_path(cfg),
                    layer, {"average": "full", "full": "average"}[layer])
        return z[{"average": "full", "full": "average"}[layer]]


def export_linked_temporal_coherence(cfg: PipelineConfig) -> Path:
    """
    MiaplPy's phase-linking temporal coherence on the stack's multilook grid.

    A single-reference star has one interferogram per date, so MintPy's
    inversion has no residual and its temporalCoherence.h5 is 1 everywhere:
    it cannot grade pixels on this route. MiaplPy's own measure, the
    goodness of fit of the linked phase to the SLC covariance ('full' over
    the whole stack, 'average' over the ministacks), is the quality figure
    MiaplPy itself masks on. Block-averaged over the looks and padded with
    NaN outside the crop, both are stored under their MiaplPy names (see
    LINKED_COHERENCE_LAYER for which one the route uses, and why).
    """
    import json
    import h5py
    stack = cfg.stack_dir; work = work_dir(cfg)
    cw = json.loads((stack / "crop_window.json").read_text())
    Lm, Wm = cw["length"], cw["width"]
    al, rl = int(cfg.stack_azimuth_looks), int(cfg.stack_range_looks)
    y0, y1, x0, x1 = aligned_subset(cfg)
    out = {}
    with h5py.File(work / "inverted" / "phase_series.h5", "r") as f:
        if "temporalCoherence" not in f:
            raise RuntimeError("phase_series.h5 has no temporalCoherence dataset - MiaplPy's concatenate step did not finish")
        tc = f["temporalCoherence"]
        k, L, W = tc.shape
        assert (L, W) == (y1 - y0, x1 - x0)
        Lc, Wc = L // al, W // rl
        r0, c0 = y0 // al, x0 // rl
        # MiaplPy's order (lib/invert.pyx): index 0 is tempCoh_average, 1 is
        # tempCoh_full. Until 27 Sep 2026 this read ("full", "average") and
        # swapped the names.
        for i, name in enumerate(("average", "full")[:k]):
            a = tc[i, :Lc * al, :Wc * rl].astype(np.float32).reshape(Lc, al, Wc, rl).mean(axis=(1, 3))
            full = np.full((Lm, Wm), np.nan, np.float32)
            full[r0:r0 + Lc, c0:c0 + Wc] = a
            out[name] = full
    path = linked_temporal_coherence_path(cfg)
    np.savez_compressed(path, layer_order=np.array(_LAYER_ORDER), **out)
    use = out[LINKED_COHERENCE_LAYER]
    ins = use[np.isfinite(use)]
    log.info("Linked temporal coherence (%s) on the multilook grid: median %.3f, %.1f%% >= %.2f -> %s", LINKED_COHERENCE_LAYER,
             float(np.median(ins)), 100 * float((ins >= cfg.min_temporal_coherence).mean()),
             cfg.min_temporal_coherence, path)
    return path


def _write_isce(isceobj, path: Path, arr: np.ndarray, dtype: str) -> None:
    arr.tofile(str(path))
    img = isceobj.createImage()
    img.setFilename(str(path)); img.setWidth(arr.shape[1]); img.setLength(arr.shape[0])
    img.setDataType(dtype); img.setBands(1); img.setInterleavedScheme("BIL"); img.setAccessMode("read")
    img.renderHdr()


def write_pair_coherence(cfg: PipelineConfig, pairs: list) -> dict:
    """
    Set the coherence each exported pair carries (filt_fine.cor), which
    snaphu's cost model and MintPy's inversion weights read.

    The export writes the magnitude of the mean unit phasor of the LINKED
    phase over each look window - a measure of how smooth the linked phase
    is, not of how much the two dates decorrelated, and it runs high (median
    0.77 against 0.36 for the classic record). With
    miaplpy_pair_coherence = "sample" this writes the standard sample
    coherence |sum s1 s2*| / sqrt(sum|s1|^2 sum|s2|^2) of the two dates'
    SLCs over the same look window, from the kept SLC stack, and keeps the
    phasor figure beside it as filt_fine.phasor.cor. Streams the stack in
    chunk-aligned date blocks, holding only the dates the remaining pairs
    need. Recorded in merged/pair_coherence.json; idempotent. A pair whose
    coherence is rewritten loses its unwrap marker, so the unwrap step that
    follows unwraps it again with the new coherence.
    """
    import h5py
    from . import unwrap_crop as uc
    kind = str(getattr(cfg, "miaplpy_pair_coherence", "phasor")).lower()
    stack = cfg.stack_dir
    rec_f = stack / "merged" / "pair_coherence.json"
    done = json.loads(rec_f.read_text()) if rec_f.exists() else {"kind": "phasor", "pairs": []}
    ifg_dir = stack / "merged" / "interferograms"
    if kind != "sample":
        if done.get("kind") == "sample":
            raise RuntimeError("pairs carry sample coherence but miaplpy_pair_coherence is 'phasor': restore "
                               "filt_fine.phasor.cor over filt_fine.cor first")
        return done
    todo = [p for p in pairs if p not in set(done.get("pairs", []))]
    if not todo:
        return done
    cw = json.loads((stack / "crop_window.json").read_text())
    r0, r1, c0, c1, L, W = cw["r0"], cw["r1"], cw["c0"], cw["c1"], cw["length"], cw["width"]
    al, rl = int(cfg.stack_azimuth_looks), int(cfg.stack_range_looks)
    h, w = r1 - r0, c1 - c0
    slc_f = work_dir(cfg) / "inputs" / "slcStack.h5"
    if not slc_f.exists():
        raise RuntimeError(f"{slc_f} is needed for sample coherence (miaplpy_delete_slc_stack_after_concatenate: false)")
    todo_by_last = sorted(todo, key=lambda p: (p[9:], p[:8]))
    need = sorted({d for p in todo for d in p.split("_")})
    # snaphu's cost model read the old coherence, and nothing in a pair's
    # unwrap config names the coherence kind, so its "done" marker would
    # survive this: a switch to sample coherence re-used every old unwrap and,
    # with filt_fine.unw unchanged, MintPy kept the old stack's coherence while
    # every record said 'sample' (7 Oct 2026). Each rewritten pair loses its
    # marker first and is unwrapped again.
    markers = _unwrap_markers_by_pair(cfg)
    t0 = time.time()
    with h5py.File(slc_f, "r") as f:
        sd = [x.decode() for x in f["date"][:]]
        idx = {d: sd.index(d) for d in need}
        B = int(f["slc"].chunks[0]) if f["slc"].chunks else 1
        cache, k = {}, 0
        for b0 in range(0, len(sd), B):
            blk_dates = [d for d in need if b0 <= idx[d] < b0 + B]
            if not blk_dates:
                continue
            blk = f["slc"][b0:b0 + B]
            for d in blk_dates:
                a = blk[idx[d] - b0][:h * al, :w * rl]
                if a.shape != (h * al, w * rl):
                    raise RuntimeError(f"SLC stack crop {a.shape} does not match the look window {(h * al, w * rl)}")
                cache[d] = a
            # every pair whose later date is now loaded
            while k < len(todo_by_last) and todo_by_last[k][9:] in cache:
                p = todo_by_last[k]; a, bb = (cache[x] for x in p.split("_"))
                num = (a * np.conj(bb)).reshape(h, al, w, rl).sum(axis=(1, 3))
                pa = (np.abs(a) ** 2).reshape(h, al, w, rl).sum(axis=(1, 3))
                pb = (np.abs(bb) ** 2).reshape(h, al, w, rl).sum(axis=(1, 3))
                with np.errstate(invalid="ignore", divide="ignore"):
                    g = np.nan_to_num(np.abs(num) / np.sqrt(pa * pb)).astype(np.float32)
                full = np.zeros((L, W), np.float32); full[r0:r1, c0:c1] = g
                pd_ = ifg_dir / p
                cor, keep = pd_ / "filt_fine.cor", pd_ / "filt_fine.phasor.cor"
                img = uc._load_image(cor)
                if not keep.exists():
                    uc._write_like(img, keep, uc._read_raw(cor, img))
                for m in markers.get(p, []):
                    m.unlink(missing_ok=True)
                uc._write_like(img, cor, full)
                done["pairs"].append(p); k += 1
            # drop dates no remaining pair needs
            later = {x for p in todo_by_last[k:] for x in p.split("_")}
            for d in [d for d in cache if d not in later]:
                del cache[d]
            done["kind"] = "sample"
            rec_f.write_text(json.dumps(done))
    log.info("Sample coherence written for %d pairs in %.0f s (phasor figure kept as filt_fine.phasor.cor)",
             len(todo), time.time() - t0)
    return done


def _unwrap_markers_by_pair(cfg: PipelineConfig) -> dict:
    """{pair: [its run_16_unwrap status markers]}, matched on the config name
    in each marker's command (the interpreter path in the command can differ
    between launchers, so the marker's hash cannot be recomputed reliably)."""
    out = {}
    status = cfg.stack_dir / ".status" / "run_16_unwrap"
    for m in sorted(status.glob("*.done")) if status.is_dir() else []:
        try:
            mp = _re.search(r"config_igram_unw_(\d{8}_\d{8})", json.loads(m.read_text()).get("cmd", ""))
        except Exception:                                      # noqa: BLE001
            mp = None
        if mp:
            out.setdefault(mp.group(1), []).append(m)
    return out


def _unwrap_marker_ok(cfg: PipelineConfig, cmd: str, pair: str) -> bool:
    """The pair's run_16_unwrap status marker says done for THIS command and
    config contents (stack_backend's own marker rules), and the outputs it
    vouches for are on disk and the right size."""
    from . import stack_backend as sb
    if not cmd:
        return False
    rec = _read_json(sb._status_path(cfg, "run_16_unwrap", cmd))
    if not rec or rec.get("returncode") != 0:
        return False
    now = sb._config_sha1(cmd)
    if rec.get("config_sha1") and now and rec["config_sha1"] != now:
        return False
    return not sb.verify_outputs(cfg, "run_16_unwrap", cmd) and \
        (cfg.stack_dir / "merged" / "interferograms" / pair / "filt_fine.unw.conncomp").exists()


def unwrap_exported(cfg: PipelineConfig, pairs: list) -> dict:
    """
    Unwrap the exported pairs with the stack route's own runner: a
    config_igram_unw_<pair> per pair (nomcf, defomax, looks - the settings
    patch_configs verifies), a run_16_unwrap of unwrap_crop commands, the
    bounded pool, longest-first ordering and the output predicates.
    """
    from . import stack_backend as sb
    from . import unwrap_crop as uc
    from . import aliasing
    stack = cfg.stack_dir
    # F2 (8 Oct 2026): the wrapped pairs, before snaphu sees them - off unless
    # the config asks (aliasing.enabled); never stops the run
    aliasing.pre_unwrap(cfg, pairs, route="miaplpy")
    cfgdir = stack / "configs"; cfgdir.mkdir(exist_ok=True)
    lines, cmd_of = [], {}
    for pair in pairs:
        pdir = stack / "merged" / "interferograms" / pair
        c = cfgdir / f"config_igram_unw_{pair}"
        c.write_text("[Common]\n##########################\n###################################\n[Function-1]\nunwrap : \n"
                     f"ifg : {pdir / 'filt_fine.int'}\nunw : {pdir / 'filt_fine.unw'}\ncoh : {pdir / 'filt_fine.cor'}\n"
                     f"nomcf : {'True' if cfg.stack_unw_full_cost else 'False'}\nreference : {stack / 'reference'}\n"
                     f"defomax : {cfg.stack_unw_defomax}\nrlks : {cfg.stack_range_looks}\nalks : {cfg.stack_azimuth_looks}\n"
                     f"rmfilter : False\nmethod : {cfg.stack_unw_method}\n")
        c.write_text(uc.apply_engine(c.read_text(), cfg))
        cmd_of[pair] = uc.crop_command(cfg, c)
        lines.append(cmd_of[pair])
    run_dir = stack / "run_files"; run_dir.mkdir(exist_ok=True)
    rf = run_dir / "run_16_unwrap"
    if rf.exists() and not (run_dir / "run_16_unwrap.stacksentinel").exists():
        rf.rename(run_dir / "run_16_unwrap.stacksentinel")      # topsStack's own network, kept aside
    rf.write_text("\n".join(lines) + "\n")
    status = stack / ".status" / "run_16_unwrap"
    if status.exists():
        # keep the markers of pairs whose outputs are on disk (a resume must
        # not redo five hours of unwrapping); drop the rest
        keep = {p for p in pairs if (stack / "merged" / "interferograms" / p / "filt_fine.unw.conncomp").exists()}
        for m in status.glob("*"):
            try:
                rec = json.loads(m.read_text()); mp = _re.search(r"config_igram_unw_(\d{8}_\d{8})", rec.get("cmd", ""))
            except Exception:                                  # noqa: BLE001
                mp = None
            if not (mp and mp.group(1) in keep and rec.get("returncode") == 0):
                m.unlink()
    result = sb.execute_run_file(cfg, rf, fail_fast=False)
    # "Done" is the status marker - this command, these config contents,
    # returncode 0, outputs verified - not a .conncomp on disk: a re-unwrap with
    # new settings that failed for a pair used to leave the PREVIOUS settings'
    # unwrap in place, and it went into MintPy under the new settings' name
    # (5 Oct 2026 review). Every failed pair's leftovers are quarantined.
    failed = [p for p in pairs if not _unwrap_marker_ok(cfg, cmd_of.get(p, ""), p)]
    if failed:
        moved = []
        for p in failed:
            pdir = stack / "merged" / "interferograms" / p
            if any(pdir.glob("filt_fine.unw*")) or any(pdir.glob("filt_fine_crop.*")):
                try:
                    q = sb.quarantine_unwrap(cfg, cmd_of.get(p, ""), "unwrap failed (non-zero exit or outputs not "
                                             "verified) - moved so MintPy cannot load a stale or partial unwrap")
                    if q:
                        moved.append(p)
                except OSError as exc:
                    log.error("Could not quarantine %s: %s", p, exc)
        # stack_backend._run_one already quarantines a failed unwrap as it
        # finishes; count those too
        moved = sorted(set(moved) | {p for p in failed if (stack / "quarantine_unwrap_timeout" / p).is_dir()})
        left = [p for p in failed if (stack / "merged" / "interferograms" / p / "filt_fine.unw").exists()]
        raise RuntimeError(
            f"{len(failed)} of {len(pairs)} phase-linked pairs did not unwrap: {failed[:6]}"
            + (f"; their outputs were moved to {stack / 'quarantine_unwrap_timeout'} ({len(moved)} pairs)" if moved else "")
            + (f"; STILL in place for {left[:6]}" if left else "")
            + f". Logs under {stack / 'logs' / 'run_16_unwrap'}. Fix the cause and restart 'Re-unwrap' (or "
              "'Phase linking onward'): unwrapped pairs are kept, only these re-run.")
    dates = sorted({d for p in pairs for d in p.split("_")})
    ok, orphans, comps = sb.check_network_connectivity(dates, pairs)
    if not ok:
        raise RuntimeError(f"phase-linked single-reference network is not connected: orphans {orphans}")
    log.info("Unwrapped %d phase-linked pairs; network connected over %d dates", len(pairs), len(dates))
    # F3 (8 Oct 2026): inside fast_zone_geojson only the nearest pairs reach
    # MintPy; with no zones and no earlier zones this does nothing
    aliasing.apply_fast_zones(cfg, pairs)
    return result


KM_PER_DEG_LAT = 110.57


def _mines_km(gj, lat0: float):
    """The mapped mines' union in a local plane in km (x = lon * kx,
    y = lat * 110.57, kx at lat0), and kx; (None, kx) when there are none."""
    import shapely
    kx = float(np.cos(np.radians(lat0)) * 111.32)
    u = _mines_lonlat(gj)
    if u is None:
        return None, kx
    return shapely.transform(u, lambda c: c * np.array([kx, KM_PER_DEG_LAT])), kx


def distance_to_mines_km(gj, lat, lon, lat0: float | None = None) -> np.ndarray | float | None:
    """Distance in km from points to the nearest mapped mine's EDGE (0
    inside one); None when no mines are mapped. The one definition the
    reference choice records and the report can quote."""
    import shapely
    if not gj or not Path(gj).exists():
        return None
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    mines, kx = _mines_km(gj, float(np.nanmean(lat)) if lat0 is None else lat0)
    if mines is None:
        return None
    d = shapely.distance(mines, shapely.points(lon * kx, lat * KM_PER_DEG_LAT))
    return float(d) if np.ndim(d) == 0 else np.asarray(d, float)


def choose_reference(cfg: PipelineConfig, pairs: list, min_fraction: float = 0.8) -> dict:
    """
    Keep the configured reference pixel if it is unwrapped (connected
    component > 0) in at least min_fraction of the pairs; otherwise pick the
    far-field pixel with the highest linked temporal coherence among those
    unwrapped in at least 95% of the pairs, inside the AOI and more than
    acceptance_far_field_buffer_m (2 km) from the EDGE of every mapped mine,
    and set cfg.reference_lalo to it. MEASURED 24 Sep 2026: the record's
    reference (chosen under short classic pairs) was in component 0 in all
    272 pairs of a five-year star. The choice, the reason and the chosen
    pixel's measured distance to the nearest mapped mine are written to
    stack/reference_choice.json; the YAML is not edited.

    The far field used to be measured to polygon VERTICES (a KD-tree of ring
    corners) with a hard-coded 2 km: on a 5-vertex rectangle drawn in the
    builder a pixel 0.1 km outside a long edge measured 5 km, and the auto
    reference landed 0.25 km from a mine while the record said "> 2 km" (5
    Oct 2026 review). It is now a metric buffer of the polygons themselves,
    which also counts mines lying outside the crop.
    """
    stack = cfg.stack_dir
    cw = json.loads((stack / "crop_window.json").read_text()); L, W = cw["length"], cw["width"]
    r0, r1, c0, c1 = cw["r0"], cw["r1"], cw["c0"], cw["c1"]
    G = stack / "merged" / "geom_reference"
    lat = np.asarray(np.memmap(G / "lat.rdr", dtype=np.float64, mode="r", shape=(L, W))[r0:r1, c0:c1])
    lon = np.asarray(np.memmap(G / "lon.rdr", dtype=np.float64, mode="r", shape=(L, W))[r0:r1, c0:c1])
    cnt = np.zeros(lat.shape, np.int32)
    for pair in pairs:
        cc = stack / "merged" / "interferograms" / pair / "filt_fine.unw.conncomp"
        if cc.exists():
            cnt += np.memmap(cc, dtype=np.uint8, mode="r", shape=(L, W))[r0:r1, c0:c1] > 0
    n = len(pairs)
    buf_m = float(getattr(cfg, "acceptance_far_field_buffer_m", 2000.0) or 2000.0)
    buf_km = buf_m / 1000.0
    gj = getattr(cfg, "deramp_exclude_geojson", None)
    have_mines = bool(gj) and Path(gj).exists()
    if gj and not have_mines:
        log.warning("Mapped mines file %s not found - the reference cannot be checked against the %.1f km "
                    "far-field rule, and none is claimed", gj, buf_km)
    lat0 = float(np.nanmean(lat))

    def dist(la, lo):
        d = distance_to_mines_km(gj, la, lo, lat0) if have_mines else None
        return None if d is None else round(float(d), 2)

    auto = bool(getattr(cfg, "reference_auto", False))
    if str(cfg.reference_lalo or "").strip():
        ref_lat, ref_lon = (float(v) for v in str(cfg.reference_lalo).split(","))
        i, j = np.unravel_index(np.argmin((lat - ref_lat) ** 2 + (lon - ref_lon) ** 2), lat.shape)
        frac_cfg = float(cnt[i, j]) / max(1, n)
        d_cfg = dist(ref_lat, ref_lon)
        info = {"configured": {"lat": ref_lat, "lon": ref_lon, "fraction_of_pairs_unwrapped": round(frac_cfg, 3),
                               "distance_to_nearest_mapped_mine_km": d_cfg,
                               "source": "earlier automatic choice (reference_choice.json)" if auto else "the YAML"},
                "pairs": n, "kept_configured": frac_cfg >= min_fraction}
        near = d_cfg is not None and d_cfg < buf_km
        if near and frac_cfg >= min_fraction:
            if auto:
                # an automatic pick (possibly by the vertex rule before 5 Oct 2026)
                # inside the buffer: choose again under the measured rule
                log.warning("The reference in effect (%s) was chosen automatically and lies %.2f km from a mapped "
                            "mine, inside the %.1f km far field - choosing again", cfg.reference_lalo, d_cfg, buf_km)
                info["kept_configured"] = False
                frac_cfg = -1.0
            else:
                log.warning("The configured reference %s lies %.2f km from a mapped mine, inside the %.1f km far "
                            "field - kept because the YAML sets it, but every rate is relative to it", cfg.reference_lalo,
                            d_cfg, buf_km)
    else:
        # none configured (a new area; the builder leaves it optional): the rule below picks it
        frac_cfg = 0.0
        info = {"configured": None, "pairs": n, "kept_configured": False}
    info.update({"far_field_buffer_m": buf_m, "mapped_mines": str(gj) if have_mines else None})
    if frac_cfg >= min_fraction:
        log.info("Reference %s is unwrapped in %.0f%% of %d pairs - kept", cfg.reference_lalo, 100 * frac_cfg, n)
        # "chosen" is always the reference in effect: load_config reads it when the
        # YAML has none, so a kept earlier auto-choice must not drop out of the file
        info["chosen"] = {"lat": ref_lat, "lon": ref_lon, "fraction_of_pairs_unwrapped": round(frac_cfg, 3),
                          "distance_to_nearest_mapped_mine_km": info["configured"]["distance_to_nearest_mapped_mine_km"],
                          "far_field_buffer_m": buf_m,
                          "rule": "kept: the reference in effect is unwrapped in enough pairs"}
        (stack / "reference_choice.json").write_text(json.dumps(info, indent=2))
        return info
    tc = load_linked_temporal_coherence(cfg)[r0:r1, c0:c1]
    W_, S_, E_, N_ = cfg.bbox
    aoi = (lat > S_ + 0.01) & (lat < N_ - 0.01) & (lon > W_ + 0.01) & (lon < E_ - 0.01)
    far = np.ones(lat.shape, bool)
    if have_mines:
        import shapely
        mines, kx = _mines_km(gj, lat0)
        if mines is not None:
            # Exact distance-within against the prepared (indexed) polygons, not
            # a buffered polygon: buffering Site B's 38k-vertex mine union by
            # 2 km took 830 MB in GEOS and its arcs are chords inside the true
            # circle; dwithin took 0.3 s and ~90 MB on its 729k-pixel crop.
            shapely.prepare(mines)
            x, y = lon.ravel() * kx, lat.ravel() * KM_PER_DEG_LAT
            xa, ya, xb, yb = mines.bounds
            cand = np.flatnonzero((x >= xa - buf_km) & (x <= xb + buf_km) & (y >= ya - buf_km) & (y <= yb + buf_km))
            near = np.zeros(x.shape, bool)
            for k in range(0, cand.size, 200_000):
                idx = cand[k:k + 200_000]
                near[idx] = shapely.dwithin(mines, shapely.points(x[idx], y[idx]), buf_km + 1e-6)   # +1 mm: 'more than'
            far = ~near.reshape(lat.shape)          # strictly more than the buffer from every mine edge
    ok = aoi & far & (cnt >= 0.95 * n) & np.isfinite(tc)
    if not ok.any():
        ok = aoi & far & (cnt >= 0.8 * n) & np.isfinite(tc)
    if not ok.any():
        raise RuntimeError(
            f"No pixel in the AOI is both unwrapped in 80% of the {n} pairs and more than {buf_km:.1f} km from the "
            "edge of every mapped mine (acceptance_far_field_buffer_m) - the network cannot be referenced by the "
            "rule. Set reference_lalo in the YAML to a stable point away from the mines (a building, road junction "
            "or bedrock) and restart 'Phase linking onward': linking, export and unwrapping are kept.")
    # prefer a pixel whose 5x5 neighbourhood is also unwrapped everywhere (a lone bright pixel is a poor datum)
    from scipy.ndimage import minimum_filter, uniform_filter
    nb_min = minimum_filter(cnt, size=5); nb_tc = uniform_filter(np.nan_to_num(tc, nan=0.0), size=5)
    score = np.where(ok, nb_tc + 0.5 * tc + 0.2 * (nb_min >= 0.95 * n), -1.0)
    i, j = np.unravel_index(int(np.argmax(score)), lat.shape)
    new = f"{lat[i, j]:.6f},{lon[i, j]:.6f}"
    raw = distance_to_mines_km(gj, float(lat[i, j]), float(lon[i, j]), lat0) if have_mines else None
    if raw is not None and raw <= buf_km:
        # verify the outcome, not the mask: the record will quote this distance
        raise RuntimeError(f"Reference choice {new} measures {raw:.3f} km from a mapped mine, inside the "
                           f"{buf_km:.1f} km far field it was selected for - not used")
    d_new = None if raw is None else round(float(raw), 2)
    rule = (f"far field (> {buf_km:g} km from the edge of every mapped mine), inside the AOI, unwrapped in >=95% of "
            "pairs, highest linked temporal coherence with its 5x5 neighbourhood") if have_mines else \
           ("no mapped mines available - inside the AOI, unwrapped in >=95% of pairs, highest linked temporal "
            "coherence with its 5x5 neighbourhood (no distance to mines can be claimed)")
    info["chosen"] = {"lat": float(lat[i, j]), "lon": float(lon[i, j]), "linked_temporal_coherence": float(tc[i, j]),
                      "fraction_of_pairs_unwrapped": float(cnt[i, j]) / max(1, n),
                      "neighbourhood_5x5_min_fraction": float(nb_min[i, j]) / max(1, n),
                      "distance_to_nearest_mapped_mine_km": d_new, "far_field_buffer_m": buf_m,
                      "rule": rule}
    (stack / "reference_choice.json").write_text(json.dumps(info, indent=2))
    where = f"{d_new:.2f} km from the nearest mapped mine" if d_new is not None else "no mapped mines to measure from"
    if info["configured"] is None:
        log.info("No reference point configured - chose %s (%s, linked temporal coherence %.2f, unwrapped in "
                 "%.0f%% of %d pairs); recorded in %s", new, where, tc[i, j], 100 * cnt[i, j] / max(1, n), n,
                 stack / "reference_choice.json")
    else:
        log.warning("Reference %s %s - using %s instead (%s, linked temporal coherence %.2f, unwrapped in %.0f%% of "
                    "pairs); recorded in %s", cfg.reference_lalo,
                    "is inside the far-field buffer" if frac_cfg < 0 else
                    f"is unwrapped in only {100 * frac_cfg:.0f}% of {n} pairs", new, where,
                    tc[i, j], 100 * cnt[i, j] / max(1, n), stack / "reference_choice.json")
    cfg.reference_lalo = new
    return info


PATCH_CHECK = Path(__file__).resolve().parent / "patches" / "miaplpy" / "check_miaplpy_patch.py"


def _patch_check_command(env: str, extra: str = "") -> str:
    """The shell line that runs the patch check in `env`: the env name checked,
    the paths quoted (16 Oct 2026, readiness review SEC-02 / RUN-04)."""
    from .procs import conda_env_arg
    return (f"source {shlex.quote(str(CONDA_SH))} && conda activate {conda_env_arg(env, 'miaplpy_env')} && cd / && "
            f"python {shlex.quote(str(PATCH_CHECK))}{extra}")


def verify_patched_miaplpy(env: str, timeout: int = 300, ministack: int | None = None,
                           time_lag: int | None = None) -> dict:
    """Run the functional patch check in the MiaplPy env (about a second).

    Upstream MiaplPy gives every mini-stack its own random datum on farmland
    (patches/miaplpy/README.md); a run on it fails only after ~18 h, at the
    export's datum gate. This catches it before anything starts. With
    `ministack`/`time_lag` (the project's miaplpy_ministack_size and
    miaplpy_time_lag) the check runs the estimator with THAT window size and
    overlap - the template passes the time lag to the patched library as the
    overlap, and an overlap of 0 leaves the windows untied (ratio 21.95,
    measured 5 Oct 2026) although the default check passes. Returns the
    check's JSON; raises if the library is not the patched build or fails
    with these settings."""
    extra = ""
    if ministack is not None:
        extra += f" --ministack {int(ministack)}"
    if time_lag is not None:
        extra += f" --time-lag {int(time_lag)}"
    r = subprocess.run(["bash", "-c", _patch_check_command(env, extra)], capture_output=True, text=True,
                       timeout=timeout)
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("{")), "")
    try:
        info = json.loads(line)
    except ValueError:
        info = {"patched": False, "reason": (r.stderr or r.stdout)[-400:]}
    if not info.get("patched"):
        if info.get("ratio") is not None and str(info.get("miaplpy_version", "")).endswith("overlapdatum"):
            # The boundary error GROWS with the overlap from 1 up (mini-stack
            # 10, measured 6 Oct 2026: 1.08x at 1, 1.29x at 3, 1.86x at 6,
            # 2.11x at 7, 2.41x at 9), and an overlap of 0 leaves the windows
            # untied (21.95x): only 0 is fixed by raising it.
            try:
                lag = int(info.get("time_lag"))
            except (TypeError, ValueError):
                lag = None
            advice = ("Raise miaplpy_time_lag to at least 1 before phase linking (3 is the tested default)."
                      if lag is not None and lag < 1 else
                      "Lower miaplpy_time_lag before phase linking: this check's boundary error grows with the "
                      "overlap (with mini-stack 10 it passes for 1 to 6 and fails from 7); it must be at least 1, "
                      "and 3 is the tested default.")
            raise RuntimeError(
                f"The patched MiaplPy in conda env '{env}' does NOT tie the windows together with these settings "
                f"(mini-stack {info.get('ministack')}, window overlap miaplpy_time_lag {info.get('time_lag')}): "
                f"boundary step error {info.get('ratio')}x the error inside windows, pass line 2.0 ({info}). "
                + advice)
        raise RuntimeError(
            f"The MiaplPy in conda env '{env}' is NOT the patched build ({info}). Phase linking with it would "
            "give every 10-image window its own datum. Rebuild it: "
            "insar_pipeline/patches/miaplpy/build_miaplpy.sh " + env)
    return info


def _resume_guard(cfg: PipelineConfig) -> None:
    """
    A phase_series.h5 that is not a complete concatenation and has nothing
    left to be rebuilt from (the patches were deleted - by the code before
    5 Oct 2026, which trusted the file's existence) cannot be resumed: say
    what to delete instead of re-linking for 12 h unannounced. Deletes nothing.
    """
    work = work_dir(cfg); inv = work / "inverted"
    if not (inv / "phase_series.h5").exists() or phase_series_complete(work) or _patch_flags(work):
        return
    scan = scan_phase_series(work)
    stack_gone = not (work / "inputs" / "slcStack.h5").exists()
    raise RuntimeError(
        f"{inv / 'phase_series.h5'} is not a complete concatenation ({scan['reason']}) and the phase-linking "
        "patches it is built from are gone, so it cannot be finished. Nothing was deleted. To recover: delete "
        f"the folder {inv} and restart 'Phase linking onward' (phase linking re-runs, ~12 h)"
        + ("; inputs/slcStack.h5 was deleted as well, so load_data reloads it from the merged SLCs - if those were "
           "removed too, coregistration has to run again ('Whole pipeline')." if stack_gone else
           "; inputs/slcStack.h5 is kept and reused."))


def _merged_dates(cfg: PipelineConfig) -> list:
    """Dates whose full-resolution merged SLC is on disk (loadable now)."""
    slc = cfg.stack_dir / "merged" / "SLC"
    if not slc.exists():
        return []
    return sorted(d.name for d in slc.iterdir()
                  if d.is_dir() and _re.fullmatch(r"\d{8}", d.name) and (d / f"{d.name}.slc.full").exists())


def check_stack_dates(cfg: PipelineConfig) -> dict:
    """
    Before phase linking: is the record long enough to link at all
    (config.miaplpy_stack_problem: fewer dates than one mini-stack links
    nothing), and does an existing inputs/slcStack.h5 hold every date that is
    merged on disk? MiaplPy's load_data is skipped whenever the stack exists,
    so a date merged after it was loaded would silently never reach the phase
    series (stackrunner-1). Raises with the remedy; deletes nothing.
    """
    from .config import miaplpy_stack_problem
    work = work_dir(cfg)
    if phase_series_complete(work, write_marker=False):
        # linked already: nothing is loaded or linked again (a finished
        # project keeps neither the SLC stack nor the merged SLCs); a changed
        # date set is check_linking_fingerprint's to refuse
        return {"phase_series_complete": True, "missing_from_stack": []}
    stacked = _slc_stack_dates(work)
    merged = _merged_dates(cfg)
    n = len(stacked or []) or len(available_dates(cfg) or merged)
    prob = miaplpy_stack_problem(cfg, n) if n else None      # none at all: run() names the missing merge
    if prob:
        raise RuntimeError(prob)
    missing = sorted(set(merged) - set(stacked)) if stacked is not None else []
    if missing:
        raise RuntimeError(
            f"{len(missing)} merged date(s) are not in {work / 'inputs' / 'slcStack.h5'} ({', '.join(missing[:6])}"
            f"{' ...' if len(missing) > 6 else ''}): MiaplPy loads the stack once and would link without them. "
            "Delete miaplpy/inputs/slcStack.h5 and restart 'Phase linking onward' (load_data rebuilds it from the "
            "merged SLCs). Nothing was changed.")
    return {"dates": n, "slc_stack_dates": len(stacked) if stacked is not None else None,
            "merged_dates": len(merged), "missing_from_stack": missing}


# ----------------------------------------------------------------------
# re-linking another project's stack (F3, PLAN A1b; 8 Oct 2026)
# ----------------------------------------------------------------------
# A1b asks whether smaller phase-linking windows recover the longwall bowl
# that A1's 15 x 15 window (~90 x 240 m support) may smear. Re-linking INSIDE
# A1's project would overwrite A1's products, and a new project would cost a
# second 190 GB coregistration. So a new project (its own project_dir, its own
# miaplpy_range_window / miaplpy_azimuth_window) names A1's folder in
# miaplpy_relink_from and reads A1's coregistered stack and SLC stack through
# symbolic links. Everything it writes (linked phases, interferograms,
# unwrapping, MintPy, deliverables) lands in its own folder. Start it with
# "Phase linking onward" (python -m insar_pipeline.pipeline CONFIG --start
# miaplpy); a full start refuses (stack_backend.stage_inputs).
RELINK_RECORD = "relink_source.json"          # <project>/miaplpy/


def relink_source(cfg: PipelineConfig) -> Path | None:
    """The project whose stack this one re-links (miaplpy_relink_from), or None."""
    v = str(getattr(cfg, "miaplpy_relink_from", "") or "").strip()
    return Path(os.path.expanduser(v)) if v else None


def _xml_dims(xml: Path) -> tuple[int, int]:
    t = Path(xml).read_text()
    w = int(_re.search(r'<property name="width">\s*<value>(\d+)', t).group(1))
    h = int(_re.search(r'<property name="length">\s*<value>(\d+)', t).group(1))
    return w, h


def source_looks(source: Path) -> tuple[int, int] | None:
    """(range, azimuth) looks of a project's merged geometry: the full-
    resolution width and length over the multilooked ones."""
    G = Path(source) / "stack" / "merged" / "geom_reference"
    try:
        wf, lf = _xml_dims(G / "lat.rdr.full.xml")
        wm, lm = _xml_dims(G / "lat.rdr.xml")
    except (OSError, AttributeError, ValueError):
        return None
    return int(round(wf / max(wm, 1))), int(round(lf / max(lm, 1)))


def relink_problems(cfg: PipelineConfig) -> list:
    """What stops this project re-linking its source's stack; [] when nothing."""
    S = relink_source(cfg)
    if S is None:
        return []
    P = Path(cfg.project_dir)
    if not S.is_dir():
        return [f"miaplpy_relink_from {S} is not a folder"]
    if S.resolve() == P.resolve():
        return ["miaplpy_relink_from names this project itself"]
    out = []
    need = {S / "miaplpy" / "inputs" / "slcStack.h5":
            "its SLC stack (kept only with miaplpy_delete_slc_stack_after_concatenate: false, and deleted by "
            "cleanup level 2)",
            S / "stack" / "crop_window.json": "its unwrap crop window",
            S / "stack" / "reference": "its reference geometry (reference/IW*.xml)",
            S / "stack" / "baselines": "its baselines",
            S / "stack" / "merged" / "geom_reference" / "lat.rdr": "its merged geometry",
            S / "stack" / "merged" / "SLC": "its merged SLC metadata"}
    for path, what in need.items():
        if not path.exists():
            out.append(f"the source has no {path.relative_to(S)} ({what})")
    slc = S / "miaplpy" / "inputs" / "slcStack.h5"
    if slc.is_symlink():
        out.append(f"{slc} is itself a link: re-link from the project that holds the stack")
    from . import runs
    busy = runs.lock_holder(S)
    if busy is not None:
        # MiaplPy's phase linking opens the SLC stack for writing
        # (initiate_stacks: h5py mode 'a'), its own and a re-linked one alike
        out.append(f"the source project is running now (pid {busy.get('pid', '?')}, {busy.get('kind') or 'a run'}): "
                   "re-link only when it is idle")
    lk = source_looks(S)
    want = (int(cfg.stack_range_looks), int(cfg.stack_azimuth_looks))
    if lk and lk != want:
        out.append(f"the source's merged grid has {lk[0]} x {lk[1]} looks (range x azimuth); this config asks "
                   f"{want[0]} x {want[1]}: set stack_range_looks / stack_azimuth_looks to the source's")
    own = P / "miaplpy" / "inputs" / "slcStack.h5"
    if own.exists() and not own.is_symlink():
        out.append(f"{own} is this project's own SLC stack: a project either loads its own stack or re-links one")
    if (P / "stack" / "coreg_secondarys").exists():
        out.append(f"{P / 'stack' / 'coreg_secondarys'} exists: this project has a coregistration of its own")
    return out


def _link_targets(cfg: PipelineConfig) -> list:
    S = relink_source(cfg).resolve()
    P = Path(cfg.project_dir)
    PS, SS = P / "stack", S / "stack"
    links = [(PS / "reference", SS / "reference"), (PS / "baselines", SS / "baselines"),
             (PS / "merged" / "geom_reference", SS / "merged" / "geom_reference"),
             (PS / "merged" / "SLC", SS / "merged" / "SLC")]
    links += [(PS / "configs" / f.name, f) for f in sorted((SS / "configs").glob("config_merge_*"))]
    links += [(PS / ".status" / d.name, d) for d in sorted((SS / ".status").glob("run_*_merge_reference_secondary_slc"))]
    links += [(P / "miaplpy" / "inputs" / f.name, f) for f in sorted((S / "miaplpy" / "inputs").glob("*.h5"))]
    return links


def prepare_relink(cfg: PipelineConfig) -> dict:
    """
    Make this project read its source's coregistered stack and SLC stack:
    symbolic links for the reference, baselines, merged geometry, merged SLC
    metadata, merge records and miaplpy/inputs/*.h5; copies of the small
    crop_window.json and generation.json; the load_data marker (the SLC stack
    is the source's, checked). This code never writes in the source and never
    replaces anything of this project's own: a real file where a link belongs
    stops it. Idempotent. Recorded in miaplpy/relink_source.json.

    MiaplPy itself opens the SLC stack in append mode while it links
    (objects/inversion_utils.initiate_stacks, h5py mode 'a'), as it does in
    its own project: HDF5 rewrites the file's superblock and its mtime, not
    its data. So the source must be idle (relink_problems), and the check
    after linking compares the size and a content probe, not the mtime
    (check_relink_source_unchanged).
    """
    import shutil
    probs = relink_problems(cfg)
    if probs:
        raise RuntimeError(f"Cannot re-link {relink_source(cfg)} into {cfg.project_dir}: " + "; ".join(probs)
                           + ". Nothing was changed.")
    S = relink_source(cfg).resolve()
    P = Path(cfg.project_dir)
    made, kept = [], 0
    for dst, src in _link_targets(cfg):
        if dst.is_symlink():
            if Path(os.path.realpath(dst)) == Path(os.path.realpath(src)):
                kept += 1
                continue
            raise RuntimeError(f"{dst} links to {os.readlink(dst)}, not {src}. Remove that link (only the link "
                               "goes) and start again.")
        if dst.exists():
            raise RuntimeError(f"{dst} is a real file or folder of this project; the re-link puts a link there "
                               "and never overwrites data. Move it away, or use a fresh project_dir.")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst)
        made.append(str(dst.relative_to(P)))
    for name in ("crop_window.json", "generation.json"):
        src, dst = S / "stack" / name, P / "stack" / name
        if not src.exists():
            continue
        if dst.exists() and not dst.is_symlink():
            if dst.read_bytes() != src.read_bytes():
                raise RuntimeError(f"{dst} differs from the source's {src}: this project's unwrap grid would not "
                                   "be the source's. Use a fresh project_dir.")
            continue
        shutil.copy2(src, dst)
        made.append(str(dst.relative_to(P)))
    slc = S / "miaplpy" / "inputs" / "slcStack.h5"
    work = work_dir(cfg)
    rec = _read_json(_status_file(work, "load_data"))
    if not (rec and rec.get("slc_stack_bytes") == slc.stat().st_size):
        import h5py
        with h5py.File(slc, "r") as h:
            if "slc" not in h or h["slc"].shape[0] < 2:
                raise RuntimeError(f"{slc} holds no SLC stack")
            bad = _empty_slc_dates(h)
            shape = list(h["slc"].shape)
        if bad:
            raise RuntimeError(f"{slc} has NaN/zero data for {len(bad)} date(s) {bad[:6]}: not a complete load")
        _write_step_marker(cfg, "load_data", None, None, f"re-linked from {slc} (checked: every date holds data)",
                           slc_stack_bytes=slc.stat().st_size, shape=shape, relinked_from=str(S))
    out = {"source": str(S), "linked_or_copied_now": made, "links_already_there": kept,
           "source_slc_stack": {"path": str(slc), **_file_id(slc), "probe": slc_probe(slc)},
           "source_looks": list(source_looks(S) or []),
           "windows": [int(cfg.miaplpy_range_window), int(cfg.miaplpy_azimuth_window)],
           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    prev = _read_json(work / RELINK_RECORD) or {}
    if prev.get("source_slc_stack") and prev["source_slc_stack"].get("bytes") != out["source_slc_stack"]["bytes"]:
        log.warning("The source's SLC stack changed size since this project first linked it (%s -> %s bytes)",
                    prev["source_slc_stack"].get("bytes"), out["source_slc_stack"]["bytes"])
    out["first_linked"] = prev.get("first_linked") or out["written"]
    out["source_slc_stack_first"] = prev.get("source_slc_stack_first") or out["source_slc_stack"]
    work.mkdir(parents=True, exist_ok=True)
    (work / RELINK_RECORD).write_text(json.dumps(out, indent=1))
    log.info("Re-linking %s's stack (%d links made or copied now, %d already there); windows %s x %s",
             S.name, len(made), kept, cfg.miaplpy_range_window, cfg.miaplpy_azimuth_window)
    return out


def slc_probe(path: Path) -> str:
    """A content fingerprint of an SLC stack, read only: its shape, its date
    list and a 64 x 64 block at the centre of the first, middle and last
    dates. Unlike the mtime, it does not move when MiaplPy opens the file in
    append mode without writing data."""
    import hashlib
    import h5py
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        ds = f["slc"]
        n, L, W = ds.shape
        h.update(repr(ds.shape).encode())
        h.update(np.asarray(f["date"][:]).tobytes())
        r0, c0 = max(0, L // 2 - 32), max(0, W // 2 - 32)
        for k in sorted({0, n // 2, n - 1}):
            h.update(np.ascontiguousarray(ds[k, r0:r0 + 64, c0:c0 + 64]).tobytes())
    return h.hexdigest()[:24]


def check_relink_source_unchanged(cfg: PipelineConfig) -> dict:
    """After linking: the source's SLC stack holds what it held when first
    linked - same size, same content probe (slc_probe). Its mtime moves
    whenever MiaplPy opens it in append mode (prepare_relink), so a moved
    mtime alone is logged, not alarmed. A changed size or probe is logged
    as an error: something wrote data into the other project."""
    work = work_dir(cfg)
    rec = _read_json(work / RELINK_RECORD) or {}
    first = rec.get("source_slc_stack_first") or {}
    p = Path(first.get("path", "")) if first.get("path") else None
    if not p or not p.exists():
        return {"checked": False}
    now = _file_id(p)
    probe = slc_probe(p) if first.get("probe") else None
    same = now["bytes"] == first.get("bytes") and (probe is None or probe == first.get("probe"))
    if not same:
        log.error("The source's SLC stack %s changed while this project re-linked it (size %s -> %s, content "
                  "probe %s -> %s): something wrote into the other project. Check that project before using "
                  "either.", p, first.get("bytes"), now["bytes"], first.get("probe"), probe)
    elif now["mtime_ns"] != first.get("mtime_ns"):
        log.info("The source's SLC stack %s has a new mtime but the same size and content probe (MiaplPy opens it "
                 "in append mode, which rewrites HDF5's superblock)", p)
    return {"checked": True, "unchanged": same, "probe_compared": probe is not None,
            "mtime_moved": now["mtime_ns"] != first.get("mtime_ns")}


def relink_estimate(cfg: PipelineConfig) -> dict:
    """GB this project writes: MiaplPy's patches and phase series (scaled
    from the source's SLC stack by the measured per-date ratios), the
    full-frame pair files (filt_fine.int/.cor/.unw/.conncomp) and MintPy.
    An ESTIMATE; PLAN's 25 GB for A1b was unconfirmed."""
    import h5py
    from . import stack_backend as sb
    S = relink_source(cfg)
    slc = S / "miaplpy" / "inputs" / "slcStack.h5"
    gb_slc = slc.stat().st_size / 1e9
    with h5py.File(slc, "r") as h:
        n = int(h["slc"].shape[0])
    cw = json.loads((S / "stack" / "crop_window.json").read_text())
    frame = int(cw["length"]) * int(cw["width"])
    crop = (int(cw["r1"]) - int(cw["r0"])) * (int(cw["c1"]) - int(cw["c0"]))
    k = max(1, int(cfg.miaplpy_network_connections))
    pairs = k * n - k * (k + 1) // 2 if k > 0 else n - 1
    linking = gb_slc * (sb.PL_PATCHES_GB + sb.PL_PHASE_SERIES_GB) / sb.PL_SLCSTACK_GB
    pair_files = pairs * frame * (8 + 4 + 8 + 1) / 1e9
    mintpy = (pairs * crop * 9 + n * crop * 4 * 8) / 1e9
    total = linking + pair_files + mintpy + 3.0
    return {"dates": n, "pairs": pairs, "linking_gb": round(linking, 1), "pair_files_gb": round(pair_files, 1),
            "mintpy_gb": round(mintpy, 1), "total_gb": round(total, 1),
            "text": (f"Re-linking {S.name}'s stack writes ~{total:.0f} GB here (estimate): phase linking "
                     f"~{linking:.0f} GB, {pairs} pair files ~{pair_files:.0f} GB, MintPy ~{mintpy:.0f} GB; the "
                     f"source's {gb_slc:.0f} GB SLC stack is read, not copied.")}


def check_relink_disk(cfg: PipelineConfig) -> dict:
    """The re-link's own disk check before phase linking (stack_backend's
    counts a new SLC stack, which a re-link never writes)."""
    from . import stack_backend as sb
    est = relink_estimate(cfg)
    have = sb._allocated_gb(cfg.project_dir) if Path(cfg.project_dir).exists() else 0.0
    need = max(0.0, est["total_gb"] - have)
    budget, src = sb._disk_budget(Path(cfg.project_dir).parent)
    log.info("%s %.0f GB already here; budget %.0f GB (%s).", est["text"], have, budget, src)
    if need > budget:
        raise RuntimeError(f"{est['text']} With {have:.0f} GB already here, {need:.0f} GB more is needed and the "
                           f"disk budget is {budget:.0f} GB ({src}). Free space before phase linking.")
    return {**est, "need_gb": round(need, 1), "budget_gb": budget}


def relink_plan(cfg: PipelineConfig):
    """The run plan of a re-linking project (stack_backend.plan_stack): no
    scene search, no download, no coregistration."""
    import h5py
    from . import stack_backend as sb
    plan = sb.StackPlan(swaths=cfg.stack_swaths, bbox_snwe=sb._bbox_snwe(cfg), range_looks=cfg.stack_range_looks,
                        azimuth_looks=cfg.stack_azimuth_looks)
    probs = relink_problems(cfg)
    if probs:
        plan.notes.append("Cannot re-link: " + "; ".join(probs))
        return plan
    S = relink_source(cfg)
    with h5py.File(S / "miaplpy" / "inputs" / "slcStack.h5", "r") as h:
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in h["date"][:]]
    est = relink_estimate(cfg)
    plan.dates, plan.n_dates = dates, len(dates)
    plan.n_pairs = est["pairs"]
    plan.est_peak_gb = est["total_gb"]
    plan.free_gb, plan.budget_source = sb._disk_budget(Path(cfg.project_dir).parent)
    plan.project_gb = sb._allocated_gb(cfg.project_dir) if Path(cfg.project_dir).exists() else 0.0
    plan.notes.append(est["text"] + f" Windows {cfg.miaplpy_range_window} x {cfg.miaplpy_azimuth_window} "
                      f"(range x azimuth). Start with 'Phase linking onward' (--start miaplpy).")
    from . import aliasing
    note = aliasing.plan_note(cfg, dates)         # F1, as stack_backend.plan_stack adds it
    if note:
        plan.notes.append(note)
    return plan


def relink_start_message(cfg: PipelineConfig) -> str:
    return (f"This project re-links the stack of {relink_source(cfg)} (miaplpy_relink_from): it downloads and "
            "coregisters nothing. Start it with 'Phase linking onward' (python -m insar_pipeline.pipeline "
            f"{getattr(cfg, '_source_path', '<config>')} --start miaplpy). Nothing was changed.")


def run(cfg: PipelineConfig) -> Path:
    """Phase linking end to end: MiaplPy's linked phase series, then the
    interferograms formed from it. Returns cfg.stack_dir / "merged" /
    "interferograms" - the folder the unwrapper and MintPy read (the
    products then land in <project>/mintpy), not MiaplPy's network
    directory."""
    info = verify_patched_miaplpy(cfg.miaplpy_env, ministack=int(cfg.miaplpy_ministack_size),
                                  time_lag=int(cfg.miaplpy_time_lag))
    log.info("MiaplPy patch verified (%s) with mini-stack %s and window overlap %s: step error across mini-stack "
             "boundaries %.2fx inside", info.get("miaplpy_version"), info.get("ministack"), info.get("time_lag"),
             info.get("ratio"))
    work = work_dir(cfg); work.mkdir(parents=True, exist_ok=True)
    (work / "patch_check.json").write_text(json.dumps({**info, "checked": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                       "env": cfg.miaplpy_env}, indent=2))
    stack = cfg.stack_dir
    relinked = relink_source(cfg) is not None
    if relinked:
        # F3 / A1b: another project's coregistered stack and SLC stack, read
        # through links; nothing in that project is written
        prepare_relink(cfg)
    slcs = sorted((stack / "merged" / "SLC").glob("*/*.slc.full"))
    loaded = (work / "inputs" / "slcStack.h5").exists() or (work / "inverted" / "phase_series.h5").exists()
    if not slcs and not loaded:
        raise RuntimeError(
            f"No merged full-resolution SLCs under {stack / 'merged' / 'SLC'} - the miaplpy "
            "backend needs stack_virtual_merge: \"False\" and the coregistration steps run first.")
    log.info("Phase linking on %d merged SLCs", len(slcs))
    if not relinked:                  # both write in the stack, which a re-link only reads
        repoint_full_vrts(stack)          # VRTs must not depend on the bursts...
        cleanup_coreg_after_merge(cfg)    # ...before the bursts go
    tpl = write_template(cfg)
    if not (work / "inputs" / "slcStack.h5").exists() and not (work / "inverted" / "phase_series.h5").exists():
        # A fresh start. MiaplPy prefers an existing custom_smallbaselineApp.cfg
        # in the work dir over the template it is given, and updates its own
        # miaplpyApp.cfg from the custom one only on the first run; anything a
        # failed attempt left behind would silently override this template.
        for stale in ("custom_smallbaselineApp.cfg", "smallbaselineApp.cfg", "miaplpyApp.cfg"):
            (work / stale).unlink(missing_ok=True)
        for sub in ("inputs", "pic"):
            for f in (work / sub).glob("*.cfg") if (work / sub).exists() else []:
                f.unlink()
    else:
        sync_workdir_templates(work, tpl)
    _resume_guard(cfg)
    check_stack_dates(cfg)
    if not phase_series_complete(work, write_marker=False):
        # the planner had only the bbox estimate of the crop; the crop is
        # measured now, and a disk that fills ~20 h in leaves a broken series
        if relinked:
            check_relink_disk(cfg)        # the SLC stack is the source's, not written here
        else:
            from . import stack_backend as sb
            sb.check_phase_linking_disk(cfg)

    for step in ("load_data", "phase_linking", "concatenate_patches"):
        run_step(cfg, tpl, step)
        if step == "load_data" and cfg.miaplpy_delete_merged_slc_after_load and not relinked:
            delete_merged_slcs_after_load(cfg)    # (a re-link's merged/SLC is the source's: never touched)
    check_linking_fingerprint(cfg)            # the series about to be exported is the one the config asks for
    # every cleanup below requires the concatenation marker (a full scan), never existence
    if cfg.miaplpy_delete_patches_after_concatenate:
        delete_patches_after_concatenate(cfg)
    if cfg.miaplpy_delete_slc_stack_after_concatenate:
        delete_slc_stack_after_concatenate(cfg)
    if relinked:
        check_relink_source_unchanged(cfg)
    pairs = export_phase_linked_ifgrams(cfg)
    write_pair_coherence(cfg, pairs)          # no-op unless miaplpy_pair_coherence = "sample"
    unwrap_exported(cfg, pairs)
    choose_reference(cfg, pairs)
    log.info("Phase-linked interferograms unwrapped on the stack grid; MintPy runs through the stack route's chain")
    return cfg.stack_dir / "merged" / "interferograms"


def main(argv=None) -> int:
    """python -m insar_pipeline.miaplpy_backend relink CONFIG [--check]:
    set up (or only check) a project that re-links another project's stack."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m insar_pipeline.miaplpy_backend")
    ap.add_argument("action", choices=["relink"])
    ap.add_argument("config")
    ap.add_argument("--check", action="store_true", help="say what stops it; change nothing")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .config import load_config
    cfg = load_config(a.config)
    if relink_source(cfg) is None:
        print("miaplpy_relink_from is not set in this config.")
        return 2
    probs = relink_problems(cfg)
    if a.check or probs:
        print("\n".join(probs) if probs else "Nothing stops the re-link. " + relink_estimate(cfg)["text"])
        return 1 if probs else 0
    rec = prepare_relink(cfg)
    print(json.dumps({k: rec[k] for k in ("source", "links_already_there", "windows")}, indent=1))
    print(relink_estimate(cfg)["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
