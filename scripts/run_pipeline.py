"""Launcher for a run from a terminal - the same as
`python -m insar_pipeline.pipeline <config.yaml> [pipeline args...]`:

    python scripts/run_pipeline.py <config.yaml> [pipeline args...]

The config is required (this used to default to an Site A config,
which no longer exists). Further arguments go to insar_pipeline.pipeline
(--resume, --start mintpy, --from-step unwrap, --to-step filter, --force ...).
The run takes the project lock, stops its commands on Ctrl-C/Stop, shows in
the dashboard, writes <project>/run_result.json and exits non-zero when any
step fails (3: another run holds the project).

Earthdata credentials: the urs.earthdata.nasa.gov entry of ~/.netrc is
exported into the environment when it is there. data_acquisition reads
~/.netrc itself as well and says plainly when the login is missing, so a
start that downloads nothing (products, MintPy onward) no longer needs it.
"""
import netrc
import os
import sys
from pathlib import Path

PKG_PARENT = str(Path(__file__).resolve().parents[2])   # the folder the package is imported from
sys.path.insert(0, PKG_PARENT)


def earthdata_env() -> bool:
    """Export the ~/.netrc Earthdata login unless the environment has one."""
    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        return True
    try:
        a = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("urs.earthdata.nasa.gov")
    except (OSError, netrc.NetrcParseError):
        a = None
    if not a:
        print("No urs.earthdata.nasa.gov entry in ~/.netrc: a step that downloads will say so.", flush=True)
        return False
    os.environ["EARTHDATA_USERNAME"], _, os.environ["EARTHDATA_PASSWORD"] = a
    # the login is not printed: run output is kept in logs and shared (16 Oct 2026,
    # readiness review DAT-07)
    print("Earthdata login: from ~/.netrc", flush=True)
    return True


def main(argv=None) -> int:
    from insar_pipeline.pipeline import script_main
    argv = sys.argv[1:] if argv is None else argv
    if argv and not str(argv[0]).startswith("-"):
        earthdata_env()
    return script_main(argv, None, __doc__)


if __name__ == "__main__":
    sys.exit(main())
