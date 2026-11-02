"""Does the MiaplPy in this Python environment carry the overlapping-window
datum fix? A functional test, not a version check: it runs the library's own
compiled sequential estimator on a synthetic stack with a KNOWN phase history
and measures the datum error at mini-stack boundaries.

    conda run -n miaplpy python check_miaplpy_patch.py        # exit 0 = patched
    conda run -n miaplpy python check_miaplpy_patch.py --ministack 10 --time-lag 3

The synthetic ground decorrelates the way the Site A farmland does
(consecutive-date correlation 0.80, so images 10 dates apart share ~0.1 and a
season apart nothing). Upstream MiaplPy anchors each mini-stack through
compressed images of earlier ones, which on such ground carry no phase
information, so every boundary gets a random datum step; the patched library
joins windows through the images they share. The test compares the phase
error of consecutive-date differences ACROSS boundaries with the error INSIDE
mini-stacks. Measured 28 Sep 2026 over 3 seeds: upstream 0d5f4b0 3.1-3.5x,
patched 1.2-1.3x; the pass line is 2.0. Runs in about a second.

--ministack / --time-lag are the project's miaplpy_ministack_size and
miaplpy_time_lag. The pipeline writes the time lag into the template as
sbw_connNum, and the patched library uses it as the window OVERLAP
(process_patch_c passes it as `lag`), so the check has to run with the
configured value, not the library default of 3: measured 5 Oct 2026 with the
overlap passed explicitly, 3 -> ratio 1.29, 1 -> 1.08, 0 -> 21.95 (no shared
images, nothing ties the windows). A build whose estimator takes no overlap
argument is upstream's and fails.
Prints one JSON line.
"""
import argparse
import json
import sys

import numpy as np

SIZE, N_WIN, SAMPLES, PIXELS, RHO, SEED = 10, 4, 60, 150, 0.80, 7
OVERLAP = 3                      # the library default and the pipeline's default miaplpy_time_lag


def simulate(rng, n):
    """n dates x SAMPLES looks of one pixel: a true phase history times an
    AR(1) decorrelating scatterer, plus thermal noise."""
    truth = np.cumsum(rng.normal(0, 0.6, n))
    c = np.empty((n, SAMPLES), complex)
    c[0] = (rng.normal(size=SAMPLES) + 1j * rng.normal(size=SAMPLES)) / np.sqrt(2)
    for t in range(1, n):
        w = (rng.normal(size=SAMPLES) + 1j * rng.normal(size=SAMPLES)) / np.sqrt(2)
        c[t] = RHO * c[t - 1] + np.sqrt(1 - RHO ** 2) * w
    noise = 0.3 * (rng.normal(size=(n, SAMPLES)) + 1j * rng.normal(size=(n, SAMPLES))) / np.sqrt(2)
    return truth, (np.exp(1j * truth)[:, None] * c + noise).astype(np.complex64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ministack", type=int, default=SIZE, help="miaplpy_ministack_size (default 10)")
    ap.add_argument("--time-lag", type=int, default=OVERLAP, help="miaplpy_time_lag = window overlap (default 3)")
    a = ap.parse_args(argv)
    size, overlap = int(a.ministack), int(a.time_lag)
    try:
        import miaplpy
        from miaplpy.lib import utils as iut
    except Exception as exc:                                  # noqa: BLE001
        print(json.dumps({"patched": False, "reason": f"miaplpy does not import: {exc}",
                          "ministack": size, "time_lag": overlap}))
        return 2
    if size < 2:
        print(json.dumps({"patched": False, "reason": f"mini-stack size {size} is not a sequential estimator",
                          "ministack": size, "time_lag": overlap}))
        return 1
    n = size * N_WIN
    rng = np.random.default_rng(SEED)
    inside, across = [], []
    for _ in range(PIXELS):
        truth, z = simulate(rng, n)
        try:
            vec, sq, _q = iut.sequential_phase_linking_py(np.ascontiguousarray(z), b"sequential_EMI", size, N_WIN,
                                                          overlap)
        except TypeError:
            # upstream's signature has no overlap: the build is not the patched one
            print(json.dumps({"patched": False, "reason": "sequential_phase_linking_py takes no window overlap "
                              "(upstream MiaplPy, not the overlapdatum build)", "ministack": size,
                              "time_lag": overlap, "miaplpy_version": getattr(miaplpy, "__version__", "?"),
                              "module": miaplpy.__file__}))
            return 1
        vec = iut.datum_connect_py(sq, vec, size)          # the call sequence MiaplPy itself uses
        est = np.angle(np.asarray(vec))
        err = np.angle(np.exp(1j * (np.diff(est) - np.diff(truth))))
        for t in range(n - 1):
            (across if (t + 1) % size == 0 else inside).append(abs(err[t]))
    e_in, e_x = float(np.median(inside)), float(np.median(across))
    ratio = e_x / max(e_in, 1e-6)
    ok = ratio < 2.0
    print(json.dumps({"patched": ok, "median_step_error_inside_rad": round(e_in, 3),
                      "median_step_error_across_boundaries_rad": round(e_x, 3), "ratio": round(ratio, 2),
                      "ministack": size, "time_lag": overlap,
                      "miaplpy_version": getattr(miaplpy, "__version__", "?"), "module": miaplpy.__file__}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
