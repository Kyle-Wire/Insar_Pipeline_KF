"""Estimate the size and cost of a run before spending anything.

Usage:
    python -m insar_pipeline.plan_run <config.yaml>

stack and miaplpy (the official phase-linked route): the same sizing as
`python -m insar_pipeline.pipeline <config> --plan-only` and the dashboard's
"Size the run only" (pipeline.run_from with plan_only: dates, disk peak
against the real budget; exits non-zero when the peak does not fit).
hyp3: the tracks available over the AOI, the number of acquisitions on each,
and how many HyP3 jobs the configured network would need. Submits nothing.

Fast motion (F1, 8 Oct 2026): when the config sets expected_max_rate_mm_yr or
runs the aliasing diagnostic (aliasing.enabled), the sizing also says whether
the intervals can follow the expected tilt (a quarter wavelength between
neighbouring look cells per interval; aliasing.detectability). On the stack
and miaplpy routes it is a note of the plan itself, so --plan-only and the
dashboard's sizing show it too.
"""
import logging
import sys
from collections import Counter

from .config import load_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    cfg = load_config(argv[0])

    # Dispatch on backend. This used to run a Sentinel-1 BURST search no
    # matter what the config said, so pointing it at a disp, nisar or stack
    # config produced HyP3 job counts and a 22 MB/product storage figure that
    # meant nothing for the run actually being planned - confidently wrong,
    # which is worse than refusing. A miaplpy config was then told "stack ->
    # supported, re-run with backend: stack", i.e. to switch the official
    # route to the comparison-only one (review 5 Oct 2026, stackrunner-16).
    # Both topsStack routes are sized by the pipeline's own plan step, so
    # this answer, --plan-only and the dashboard's can never disagree.
    if cfg.backend in ("stack", "miaplpy"):
        from . import pipeline
        return pipeline.run_from(cfg, "full", kind="plan", plan_only=True, config_path=str(argv[0]))
    if cfg.backend != "hyp3":
        print(f"backend is '{cfg.backend}'. This planner sizes the stack and miaplpy "
              f"(phase-linked, official) routes and models HyP3 job counts; neither applies.\n"
              f"  disp  -> screening route: no ordering step and no credits; cost is read "
              f"time and the stack size set by disp_decimate\n"
              f"  nisar -> products are pre-made; cost is download only")
        _detectability(cfg)
        return 0

    from . import hyp3_backend
    by_burst = hyp3_backend.find_bursts(cfg)
    if not by_burst:
        print("No bursts found for this AOI, date range and track.")
        return 1

    # Which tracks are present, and how many dates on each. Pick one:
    # ascending and descending are different geometries and must not share
    # an inversion.
    tracks = Counter()
    for scenes in by_burst.values():
        for s in scenes:
            tracks[(s.properties.get("pathNumber"),
                    (s.properties.get("flightDirection") or "?")[:3])] += 1

    print("\nTracks over this AOI:")
    for (path, direction), n in sorted(tracks.items()):
        print(f"   track {path} ({direction.lower()}): {n} bursts")
    if cfg.track is None and len(tracks) > 1:
        print("\n   `track` is unset and more than one is present. Set it in "
              "the config before processing - mixing tracks mixes geometries.")

    pairs = hyp3_backend.build_multiburst_pairs(
        by_burst, cfg.max_temp_baseline, cfg.n_neighbors)

    already = hyp3_backend.existing_products(cfg)
    todo = [p for p in pairs
            if (p[0].replace("-", ""), p[1].replace("-", "")) not in already]

    print(f"\n   {len(pairs)} pairs at n_neighbors={cfg.n_neighbors}")
    print(f"   {len(already)} already downloaded")
    print(f"   {len(todo)} jobs to submit")
    print(f"\n   At roughly 22 MB per product, {len(todo)} jobs is about "
          f"{len(todo) * 22 / 1024:.1f} GB of download.")
    if len(todo) > 200:
        print("\n   That is a large batch. Consider a shorter date range to "
              "validate the signal first, or n_neighbors=2, before "
              "committing the full stack.")
    dates = sorted({str(s.properties.get("startTime", ""))[:10].replace("-", "")
                    for scenes in by_burst.values() for s in scenes if s.properties.get("startTime")})
    _detectability(cfg, dates or None)
    return 0


def _detectability(cfg, dates=None) -> None:
    """The F1 preflight (aliasing.plan_note) when the config asks for it."""
    from . import aliasing
    note = aliasing.plan_note(cfg, dates)
    if note:
        print("\n  " + note)


if __name__ == "__main__":
    sys.exit(main())
