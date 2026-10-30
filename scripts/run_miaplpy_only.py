"""Phase linking onward (MiaplPy -> unwrapping -> MintPy -> deliverables ->
assessment -> figures -> GNSS tie + client report) on a finished
coregistration - the same as `python -m insar_pipeline.pipeline
<config.yaml> --start miaplpy` and the dashboard's "Phase linking onward":

    python scripts/run_miaplpy_only.py <config.yaml> [--from-products] [--force] [--no-notify]

--from-products skips MiaplPy, unwrapping and MintPy and only re-runs the
deliverables onward (= --start products). The run refuses a coregistration
that is not finished (a shorter stack would be linked for good); 'Resume'
picks the right start from the files on disk.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None) -> int:
    from insar_pipeline.pipeline import script_main
    argv = list(sys.argv[1:] if argv is None else argv)
    start = "products" if "--from-products" in argv else "miaplpy"
    return script_main([a for a in argv if a != "--from-products"], start, __doc__)


if __name__ == "__main__":
    sys.exit(main())
