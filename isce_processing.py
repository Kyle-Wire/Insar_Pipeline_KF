"""Stage 3: run ISCE2's topsApp.py for each interferometric pair.

NOTE: the topsApp.xml schema below covers the common properties, but
ISCE2's exact property names/behavior have shifted slightly across
releases, and this doesn't set swath/burst subsetting or multilook
factors — those are AOI-specific. Run `topsApp.py --help` and check
your installed version's documentation before trusting this blind on
a new dataset.
"""
import logging
import shutil
import subprocess
from pathlib import Path
from string import Template

from .config import PipelineConfig
from .gis_products import make_gis_products

log = logging.getLogger(__name__)

_TOPSAPP_TEMPLATE = Template("""\
<?xml version="1.0" encoding="UTF-8"?>
<topsApp>
  <component name="topsinsar">
    <property name="Sensor name">SENTINEL1</property>
    <component name="reference">
      <property name="safe">['$ref_safe']</property>
      <property name="output directory">reference</property>
    </component>
    <component name="secondary">
      <property name="safe">['$sec_safe']</property>
      <property name="output directory">secondary</property>
    </component>
    <property name="demFilename">$dem_path</property>
    <property name="geocode bounding box">[$south, $north, $west, $east]</property>
    <property name="do ESD">$do_esd</property>
    <property name="esd coherence threshold">$esd_threshold</property>
    <property name="do unwrap">True</property>
    <property name="unwrapper name">snaphu_mcf</property>
    <property name="do denseoffsets">False</property>
    <property name="geocode list">['merged/filt_topophase.unw',
                                    'merged/filt_topophase.flat',
                                    'merged/phsig.cor',
                                    'merged/los.rdr']</property>
  </component>
</topsApp>
""")


def _pair_dir(cfg: PipelineConfig, ref, sec) -> Path:
    ref_date = ref.properties["startTime"][:10].replace("-", "")
    sec_date = sec.properties["startTime"][:10].replace("-", "")
    return cfg.pairs_dir / f"{ref_date}_{sec_date}"


def write_tops_config(cfg: PipelineConfig, ref, sec, dem_path: Path) -> Path:
    pair_dir = _pair_dir(cfg, ref, sec)
    pair_dir.mkdir(parents=True, exist_ok=True)
    min_lon, min_lat, max_lon, max_lat = cfg.bbox
    xml = _TOPSAPP_TEMPLATE.substitute(
        ref_safe=str(cfg.slc_dir / ref.properties["fileName"]),
        sec_safe=str(cfg.slc_dir / sec.properties["fileName"]),
        orbit_dir=str(cfg.orbit_dir),
        dem_path=str(dem_path),
        do_esd=str(cfg.do_esd),
        esd_threshold=str(cfg.esd_coherence_threshold),
        south=min_lat, north=max_lat, west=min_lon, east=max_lon,
    )
    config_path = pair_dir / "topsApp.xml"
    config_path.write_text(xml)
    return config_path


def run_pair(cfg: PipelineConfig, ref, sec, dem_path: Path) -> bool:
    """Returns True on success. Failures are logged, not raised, so one bad
    pair doesn't take down the whole stack."""
    pair_dir = _pair_dir(cfg, ref, sec)
    unw_output = pair_dir / "merged" / "filt_topophase.unw.geo"

    if unw_output.exists() and cfg.skip_existing:
        log.info("Pair %s already processed, skipping", pair_dir.name)
        return True

    config_path = write_tops_config(cfg, ref, sec, dem_path)
    log.info("Running topsApp.py for %s", pair_dir.name)
    result = subprocess.run(
        ["topsApp.py", config_path.name, "--start=startup", "--end=geocode"],
        cwd=pair_dir,
        capture_output=True,
        text=True,
    )
    # Always write the log, not just on a non-zero exit: topsApp can exit
    # cleanly while still failing to produce the geocoded output (e.g. the
    # geocode step silently doing nothing), and without the log there's
    # nothing to diagnose.
    (pair_dir / "topsApp_run.log").write_text(result.stdout + result.stderr)

    if result.returncode != 0:
        log.error("topsApp.py exited with an error for %s (see topsApp_run.log)", pair_dir.name)
        return False

    if not unw_output.exists():
        log.error("topsApp.py finished cleanly for %s but produced no geocoded "
                  "output at %s — check the geocode step in topsApp_run.log",
                  pair_dir.name, unw_output)
        return False

    make_gis_products(pair_dir)
    if cfg.cleanup_intermediates:
        cleanup_pair(pair_dir)
    return True


# Scratch directories topsApp creates per pair. Everything needed
# downstream (by RAiDER, MintPy, or GIS) lives in merged/, so these can go
# once the pair is done — they dominate per-pair disk use, and a 20+ pair
# stack is not feasible without removing them.
#
# Note this makes a pair non-resumable: rerunning it means starting from
# scratch rather than from a pickled step. That's the trade for being able
# to process a long stack at all.
_SCRATCH_DIRS = [
    "secondary", "fine_coreg", "fine_interferogram",
    "fine_offsets", "coarse_coreg", "coarse_interferogram", "coarse_offsets",
    "ESD", "geom_reference", "misreg", "PICKLE",
]

# reference/ is NOT in the list above: its IW*.xml files carry platform
# metadata (satellite altitude and similar) that MintPy needs at load time
# and cannot recover from merged/. The bulk of that directory is burst
# rasters, which are safe to remove — so it gets pruned rather than deleted.
_REFERENCE_KEEP_SUFFIXES = (".xml",)


def cleanup_pair(pair_dir: Path) -> None:
    """Remove per-pair scratch, keeping merged/, the logs, and reference metadata."""
    freed = 0
    for name in _SCRATCH_DIRS:
        target = pair_dir / name
        if target.is_dir():
            freed += sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            shutil.rmtree(target, ignore_errors=True)

    # Prune reference/ rather than removing it: drop the burst rasters,
    # keep the small XML metadata MintPy reads.
    reference = pair_dir / "reference"
    if reference.is_dir():
        for f in reference.rglob("*"):
            if f.is_file() and not f.name.endswith(_REFERENCE_KEEP_SUFFIXES):
                freed += f.stat().st_size
                f.unlink(missing_ok=True)
    if freed:
        log.info("Cleaned %.1f GB of intermediates from %s",
                 freed / 1e9, pair_dir.name)


def run_all_pairs(cfg: PipelineConfig, pairs, dem_path: Path):
    succeeded, failed = [], []
    for ref, sec in pairs:
        ok = run_pair(cfg, ref, sec, dem_path)
        (succeeded if ok else failed).append((ref, sec))
    log.info("Interferogram formation: %d succeeded, %d failed", len(succeeded), len(failed))
    return succeeded, failed
