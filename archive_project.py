"""Reclaim disk space from a finished project - an alias of `insar cleanup`
(scripts/cleanup.py), kept so the old command still works.

Usage:
    python -m insar_pipeline.archive_project <config.yaml> [--level LEVEL] [--apply]

This used to be a second cleanup tool with its own rules: it collected the
per-pair scratch of the retired topsApp ('local') route and slc/*.zip, so on
the burst2safe and phase-linked routes (SAFE folders, stack/, miaplpy/) it
found nothing, while its docstring and cleanup.py disagreed about what is
kept (review 5 Oct 2026). Now there is one tool and one policy - see
scripts/cleanup.py: dry run unless asked, never while a run holds the
project, levels 2-3 only on a finished project, logs and
deliverables/releases/ never touched.

The old level names map onto cleanup's levels:
    scratch -> 1  (history and scratch, incl. the old per-pair ISCE2 scratch)
    weather -> 2  (processing inputs, incl. the cached ERA5 weather files)
    slc     -> 2  (processing inputs, incl. the downloaded images)
    all     -> 2
1, 2 and 3 are passed through. --apply means --delete.
"""
import sys

LEVEL_MAP = {"scratch": "1", "weather": "2", "slc": "2", "all": "2"}


def translate(argv: list) -> list:
    """Old archive_project arguments -> scripts/cleanup.py arguments."""
    out, it = [], iter(argv)
    for a in it:
        if a == "--apply":
            out.append("--delete")
        elif a == "--level":
            v = next(it, "")
            out += ["--level", LEVEL_MAP.get(v, v)]
        elif a.startswith("--level="):
            v = a.split("=", 1)[1]
            out += ["--level", LEVEL_MAP.get(v, v)]
        else:
            out.append(a)
    return out


def main(argv=None) -> int:
    from .scripts import cleanup
    args = translate(list(sys.argv[1:] if argv is None else argv))
    print("archive_project is now an alias of `insar cleanup` (scripts/cleanup.py): "
          + " ".join(args), flush=True)
    return cleanup.main(args)


if __name__ == "__main__":
    sys.exit(main())
