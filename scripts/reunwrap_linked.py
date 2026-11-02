"""Re-unwrap an existing phase-linked export with the configured unwrapping
settings, then MintPy -> deliverables -> assessment -> figures -> GNSS tie +
client report. The phase linking and the export are not redone.

    python scripts/reunwrap_linked.py <config.yaml>

It is `pipeline.launch(<config>, "reunwrap")` - the one stage chain every
launcher shares (pipeline.run_from start 'unwrap': it checks that every
expected pair was exported, then pair coherence -> unwrap -> reference ->
MintPy -> products), with the child-process teardown installed FIRST and
the project lock held for the whole run: a dashboard Stop reaches the up to
24 snaphu/unwrap_crop processes it starts in their own sessions (before
5 Oct 2026 they kept running), and a second run on the same project is
refused (exit 3) instead of unwrapping into the same files.

Before anything is overwritten, keep_previous() copies what a before/after
comparison needs and the re-run would OVERWRITE in place - the inversion's
coherence and masks and the small records - to <mintpy>/superseded_<date>/.
The velocity, the corrected time series, the DEM error and the whole of
mintpy/geo/ are not copied: the MintPy step moves the previous run's set
aside itself, by rename, into mintpy/superseded/run_<stamp>/ - the radar
ones as the re-unwrapped stack is reloaded
(time_series.clear_stale_mintpy_products; until 7 Oct 2026 it deleted
them), the rest before correct_LOD (time_series.retire_downstream_products).
The uncorrected inversion (timeseries.h5) is not kept. Copying them here as
well cost several GB per re-unwrap, and the fixed series name it copied
(timeseries_ERA5_ramp_demErr.h5) was not the final series of any chain run
without ERA5 or deramping.
"""
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger("reunwrap")

# What the re-run overwrites in place (the inversion runs before MintPy's
# retirement step) and a before/after comparison needs; small apart from the
# coherence layers. Everything downstream of the inversion is retired by rename.
SUPERSEDED = ("temporalCoherence.h5", "temporalCoherence_mintpy.h5", "maskTempCoh.h5", "maskConnComp.h5",
              "numTriNonzeroIntAmbiguity4unwrapPhase_bridging.h5", "smallbaselineApp.cfg",
              "reference_excluded_pairs.json", "network_connectivity.json", "unwrap_error_applied.json",
              "corrections.json")


def keep_previous(cfg) -> Path:
    """Copy what the re-run overwrites in place to <mintpy>/superseded_<date>/
    (once a day), with a note pointing at the retired products."""
    keep = cfg.mintpy_dir / f"superseded_{time.strftime('%Y%m%d')}"
    if not keep.exists():
        keep.mkdir(parents=True)
        for rel in SUPERSEDED:
            src = cfg.mintpy_dir / rel
            if src.exists():
                dst = keep / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        if (cfg.stack_dir / "unwrap_history.json").exists():
            shutil.copy2(cfg.stack_dir / "unwrap_history.json", keep / "unwrap_history.json")
        (keep / "README.txt").write_text(
            "Copied before a re-unwrap: the inversion's coherence and masks and the run records, which the\n"
            "re-run overwrites in place. The previous velocity and corrected time series (radar and geocoded),\n"
            "DEM error and the rest of geo/ are moved aside by the MintPy step itself into\n"
            f"{cfg.mintpy_dir / 'superseded'}/run_<stamp>/ (see its RETIRED.json). The uncorrected inversion\n"
            "(timeseries.h5) is not kept.\n")
        log.info("Previous inversion layers and records copied to %s", keep)
    return keep


def main(argv=None) -> int:
    """Teardown first, then the shared launcher with the backup as its
    `before` step. Returns the exit code (3: another run holds the project)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", force=True)
    from insar_pipeline import procs
    procs.install_teardown()              # first: before anything can start a child
    if not argv:
        raise SystemExit("usage: reunwrap_linked.py <config.yaml>")
    from insar_pipeline.config import load_config
    cfg = load_config(argv[0])
    if cfg.backend != "miaplpy":
        raise SystemExit("reunwrap_linked.py is for the phase-linked (miaplpy) backend")
    from insar_pipeline.pipeline import launch
    return int(launch(argv[0], "reunwrap", before=keep_previous) or 0)


if __name__ == "__main__":
    sys.exit(main())
