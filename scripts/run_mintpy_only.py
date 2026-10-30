"""MintPy onward (MintPy -> deliverables -> quality assessment -> figures ->
GNSS tie + client report) on a stack that is already unwrapped - the same as
`python -m insar_pipeline.pipeline <config.yaml> --start mintpy` and the
dashboard's "MintPy onward":

    python scripts/run_mintpy_only.py <config.yaml> [--force] [--no-notify]

One stage chain (pipeline.run_from): the route's settings (a star network
-> 'bridging' unwrap-error correction) and the precondition check (on the
phase-linked route every expected pair must be unwrapped) apply here too.
The config is required (no Site A default any more).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None) -> int:
    from insar_pipeline.pipeline import script_main
    return script_main(sys.argv[1:] if argv is None else argv, "mintpy", __doc__)


if __name__ == "__main__":
    sys.exit(main())
