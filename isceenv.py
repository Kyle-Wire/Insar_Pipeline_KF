"""
Import ISCE without letting it silence this package's logging.

`import isce` runs logging.config.fileConfig() on ISCE's own logging.conf,
which (a) replaces the root handlers and (b) DISABLES every logger that
already exists - i.e. every insar_pipeline module imported before it. On
the P1 v2 run the DEM was re-stitched in-process, isce was imported for
the first time inside get_dem(), and the runner's log went silent for the
rest of the night while the stack processed normally underneath.

Use `isce, isceobj = import_isce()` anywhere instead of importing directly.
"""
import logging


def import_isce():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    was_enabled = {n for n, lg in logging.root.manager.loggerDict.items()
                   if isinstance(lg, logging.Logger) and not lg.disabled}
    import isce      # noqa: F401  (reconfigures logging as a side effect)
    import isceobj
    # Put back what was there, re-enable our loggers, and drop the handlers
    # ISCE added (its FileHandler writes isce.log into the current directory).
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
    for h in handlers:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(level)
    # every logger that was enabled before (a driver script's own logger
    # included - on 27 Sep 2026 a "reunwrap" logger went silent mid-run),
    # and every insar_pipeline logger created since
    for name, lg in logging.root.manager.loggerDict.items():
        if isinstance(lg, logging.Logger) and (name in was_enabled or name.startswith("insar_pipeline")):
            lg.disabled = False
    return isce, isceobj


def guard_logging() -> None:
    """
    Trigger ISCE's one-time logging configuration now, through import_isce,
    so that a later first import somewhere this package does not control -
    MintPy's readfile imports isceobj to read ISCE .xml files - finds isce
    already imported and changes nothing. No-op where ISCE is not installed.
    """
    import sys
    if "isce" in sys.modules:
        return
    try:
        import_isce()
    except ImportError:
        pass
