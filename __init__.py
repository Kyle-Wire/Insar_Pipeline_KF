"""insar_pipeline: Sentinel-1 ground-motion records from download to client report."""
# ISCE reconfigures logging (and disables existing loggers) the first time it is
# imported; import it here, through the guard, before anything else can.
from .isceenv import guard_logging as _guard_logging

_guard_logging()
