"""Phase 1 on products already on disk: export and measure each project.

    python -m insar_pipeline.research.atmospheric_temperature.run_phase1 \\
        <project_dir> [<project_dir> ...] --out <dir> [--model HRRR] [--label NAME ...]

Writes <out>/<label>/radar_obs.h5, radar_obs.json, observability.json and
observability.png per project, and <out>/phase1_summary.json. Reads the
projects only; downloads nothing.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def plot(obs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sf = obs["structure_function_mm"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].loglog(sf["lag_m"], sf["residual_pair_median"], "o-", label="residual (obs - model), median pair")
    ax[0].loglog(sf["lag_m"], sf["residual_pair_p90"], ":", label="residual, 90th percentile")
    ax[0].loglog(sf["lag_m"], sf["model_pair_median"], "s-", label="model (HRRR), median pair")
    ax[0].set_xlabel("separation (m)")
    ax[0].set_ylabel("RMS difference, LOS (mm)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Short-interval date pairs")
    s = [p["slope_mm_per_km"] for p in obs["pairs"]]
    m = [p["model_slope_mm_per_km"] for p in obs["pairs"]]
    ax[1].plot(range(len(s)), s, ".", label="residual")
    ax[1].plot(range(len(m)), m, ".", alpha=0.5, label="model")
    ax[1].set_xlabel("date pair")
    ax[1].set_ylabel("slope against height (mm/km)")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"Relief {obs['height_m']['relief']:.0f} m")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    from . import observability, radar_obs
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("projects", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="HRRR")
    ap.add_argument("--label", nargs="*", help="output folder names, one per project (default: the folder name)")
    ap.add_argument("--no-digest", action="store_true", help="skip the sha256 of the observation file")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    labels = a.label or [Path(p).name for p in a.projects]
    if len(labels) != len(a.projects):
        ap.error("--label needs one name per project")
    out = Path(a.out)
    summary = {}
    for proj, lab in zip(a.projects, labels):
        d = out / lab
        rec = radar_obs.export(Path(proj), d, a.model, digest=not a.no_digest)
        obs = observability.assess(d, Path(proj))
        plot(obs, d / "observability.png")
        summary[lab] = {"check": rec["check_against_applied_correction"],
                        "relief_m": obs["height_m"]["relief"], "pairs": obs["pairs_used"],
                        "detectability_per_date": obs["detectability_per_date"]}
    (out / "phase1_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return 0 if all(s["check"]["passed"] for s in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
