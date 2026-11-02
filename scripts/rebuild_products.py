"""Deliverables -> quality assessment -> panel/site history + figures -> GNSS
tie + client report from a finished MintPy run (no MintPy re-run) - the same
as `python -m insar_pipeline.pipeline <config.yaml> --start products` and the
dashboard's "Products only":

    python scripts/rebuild_products.py <config.yaml> [--no-notify]

Exits non-zero when any step fails, and says which in <project>/run_result.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None) -> int:
    from insar_pipeline.pipeline import script_main
    return script_main(sys.argv[1:] if argv is None else argv, "products", __doc__)


if __name__ == "__main__":
    sys.exit(main())
