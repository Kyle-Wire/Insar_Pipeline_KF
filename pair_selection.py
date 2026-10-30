"""Stage 2: build an interferometric pair network from a scene list."""
import logging
from datetime import datetime

log = logging.getLogger(__name__)


def _acquisition_date(scene) -> datetime:
    return datetime.strptime(scene.properties["startTime"][:10], "%Y-%m-%d")


def build_sequential_network(scenes, max_temporal_days: int, n_neighbors: int = 3):
    """
    Pair each acquisition with its next `n_neighbors` in time, capped by
    max_temporal_days.

    Why n_neighbors matters more than it looks: with n_neighbors=1 the
    network is a simple chain with no closed loops, so the SBAS inversion
    is exactly determined and its residual is zero by construction. MintPy's
    temporal coherence then reads 1.000 everywhere and is measuring
    nothing - which also makes the reliability mask useless. Redundant
    connections create the loops that let phase closure detect unwrapping
    errors. A minimum of 3 is a common published choice.

    The cost is processing time: n_neighbors=3 is roughly three times the
    pairs, so three times the interferogram formation.

    This is still a time-based network, not a true small-baseline one:
    pairing by perpendicular baseline needs orbit state vectors that ISCE2
    only produces during processing. The practical path is to run this,
    then use the reported baselines to prune or extend.
    """
    scenes_sorted = sorted(scenes, key=_acquisition_date)
    pairs = []
    for i, ref in enumerate(scenes_sorted):
        added = 0
        for sec in scenes_sorted[i + 1:]:
            dt = (_acquisition_date(sec) - _acquisition_date(ref)).days
            if dt > max_temporal_days:
                break
            pairs.append((ref, sec))
            added += 1
            if added >= n_neighbors:
                break

    # Report redundancy so a thin network is visible before hours of
    # processing rather than after.
    counts = {}
    for ref, sec in pairs:
        for s in (ref, sec):
            key = _acquisition_date(s).date()
            counts[key] = counts.get(key, 0) + 1
    thin = [str(d) for d, c in sorted(counts.items()) if c < 2]

    log.info("Built %d pairs from %d scenes (n_neighbors=%d)",
             len(pairs), len(scenes_sorted), n_neighbors)
    if counts:
        log.info("Connections per acquisition: min %d, median %d, max %d",
                 min(counts.values()),
                 sorted(counts.values())[len(counts) // 2],
                 max(counts.values()))
    if thin:
        log.warning("These dates have fewer than 2 connections and will be "
                    "weakly constrained: %s", ", ".join(thin))
    if n_neighbors < 2:
        log.warning("n_neighbors=%d gives a network with no closed loops. "
                    "Temporal coherence will read 1.000 everywhere and the "
                    "quality product will be meaningless.", n_neighbors)
    return pairs
