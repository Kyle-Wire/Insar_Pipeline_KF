# The MiaplPy patch: what it is, why, and how it is protected

**In one line.** Upstream MiaplPy's sequential phase linking joins its 10-image
windows through "compressed images" of earlier windows. On farmland those carry
no phase information across a season, so every window came out with its own
random offset per pixel. The patch joins each window to the previous one
through three images the two windows share instead.

## The problem it fixes

Sequential EMI (Ansari et al. 2017) splits a long stack into mini-stacks of 10
images. It solves each one, compresses it into one "compressed image", and then
ties all the windows into a single datum using those compressed images. That
works where the ground stays coherent for years: cities and bare rock, the
settings the method was designed and published for.

At Site A the compressed images of adjacent windows share 0.13–0.35
coherence, and anything a season apart is at the noise floor (0.07). There is no
common phase left to tie the windows through. The result was a random step
at every window boundary: consecutive-date coherence 0.72 inside a window,
0.40 across boundaries. No pair spanning a boundary could be unwrapped, and
the first production run failed at the reference check. This is not a coding
bug upstream. It is a design assumption, persistent coherence across windows,
that this ground breaks.

## What the patch changes (`overlap_window_datum.patch`)

Against upstream commit `0d5f4b0` (MiaplPy 0.2.1), in `src/miaplpy/lib/utils.pyx`
and `utils.pxd`:

- `sequential_phase_linking_cy`: window k is solved from images
  `[10k − overlap, 10k + 10)`, on its own, without compressed images. The
  `overlap` images at its start were also estimated by the previous window
  from the same data, so the offset between the windows is the circular mean
  of their phase differences on those images. `overlap` = the pipeline's
  `miaplpy_time_lag` (3).
- `datum_connect_cy` becomes an identity, because the windows are already in
  one datum. It is kept so the callers are unchanged.
- `version.py`: the build calls itself `0.2.1+overlapdatum`, so
  `pip show miaplpy` says which build is installed.

The per-pixel cost is unchanged.

Verified 25 Sep 2026 in simulation (boundary step error 0.22 → 0.03 rad) and on
real pixels (worst boundaries 0.48–0.58 → 0.66–0.85 coherence). The production
series passed the export's datum gate: 0.85 inside, 0.71 across.

## The speed patch (`speed.patch`, 10 Nov 2026): faster, same estimate

Applied on top of the datum patch. The build calls itself
`0.2.1+overlapdatum.fast`. It changes speed only:

- **Covariance through BLAS** (`utils.pyx`, `est_corr_cy`/`est_cov_cy`). The
  sample covariance `CCG · CCGᴴ / n` was a scalar triple loop. For the full
  N×N matrix of every pixel that was ~65% of phase-linking time at 445 dates,
  and it grows with N². It is now one complex64 matrix product.
- **One worker pool over all patches** (`miaplpyApp.py`, `phase_linking.py`).
  Upstream splits the patches into `ceil(patches / workers)` jobs and, on one
  machine, runs them one after another, each waiting for its slowest patch.
  On study_areav2 (445 dates) workers sat idle 21% of 11.8 h. Now every patch
  goes to the next free worker (`imap_unordered`, chunksize 1, a fresh process
  per patch).

**Same estimate, measured.** The test ran 6 real 16×16 boxes (3 places, 279
and 445 dates) plus a 32-patch `phase_linking.py` run end to end (60 dates),
each with the old build, the new build, and the old build on input nudged by
float32 rounding noise (relative 6e-8):

| | SHP / PS masks | phase p99 | phase max | tempCoh max |
|---|---|---|---|---|
| old vs new (boxes) | identical | ≤ 0.0023 rad | 0.029 rad | 4e-4 |
| old vs old + rounding noise (boxes) | identical | ≤ 0.0021 rad | 0.034 rad | 4e-4 |
| old vs new (32-patch run) | identical | 1.2e-4 rad | 0.065 rad | 8e-4 |
| old vs old + rounding noise (32-patch run) | 2 patches differ | 1.1e-4 rad | 0.158 rad | 3e-3 |

So the new build sits inside the old build's own float32 sensitivity. The
largest difference is ~0.3 mm line of sight at C-band, on one pixel.
`check_miaplpy_patch.py` gives the identical result (ratio 1.29). The linking
fingerprint therefore treats `.fast` as the same algorithm
(`miaplpy_backend.SPEED_ONLY_BUILD_SUFFIXES`). A series linked with either
build resumes under the other, and any other build label is still refused.

**Speed.** Old and new builds were run alternately, 3 runs each, on a 12×12
real box. Per pixel: 18.9 → 10.0 ms at 279 dates and 38.8 → 20.2 ms at 445
dates, so **1.9×**. With the scheduling fix, a whole run should be about 2.3×
faster. That figure has not yet been measured on a full run.

## How it is protected

1. **Rebuild recipe.** `build_miaplpy.sh [env]` clones upstream at the pinned
   commit, applies the patch, installs it into the env without touching
   dependencies, and runs the check. It works from GitHub, or offline with
   `MIAPLPY_SRC=/home/kylew/src/MiaplPy`. Tested 28 Sep 2026: a fresh GitHub
   clone plus the patch gives the passing build now installed.
2. **Pip lock.** The env's own pip config (`pip config --site`, applied whether
   or not the env is activated) points at `constraints.txt`, which contains
   `miaplpy==0.2.1+overlapdatum`. `pip install -U miaplpy` leaves it in place,
   and `pip install miaplpy==0.2.1` (upstream) is refused. Conda does not
   manage this package (it came from pip), so `conda update` does not touch it.
3. **Functional check before every run.** `check_miaplpy_patch.py` runs the
   installed library's own compiled estimator on a synthetic farmland-like
   stack with a known phase history. It compares the phase error across window
   boundaries with the error inside windows. Upstream 0d5f4b0 reads 3.1–3.5×
   and the patched build 1.2–1.3× (3 seeds); the pass line is 2.0. It takes
   about a second. (It does not tell `.fast` from the plain datum build: both
   pass. The installer compares the version label as well.)
   - It takes the project's own settings: `--ministack` (`miaplpy_ministack_size`)
     and `--time-lag` (`miaplpy_time_lag`, the window overlap). Measured 5 Oct
     2026 with the overlap passed explicitly: overlap 3 → 1.29× (pass), 1 →
     1.08×, 0 → 21.95× (fail: with no shared images nothing ties the windows),
     ministack 15 with overlap 5 → 1.86×. The config therefore refuses
     `miaplpy_time_lag` below 1 or not below the mini-stack size.
   - `miaplpy_backend.run()` calls it with the project's values before any
     phase-linking step, records the result in `miaplpy/patch_check.json`, and
     stops with the rebuild command if it fails.
   - The dashboard's preflight and `insar doctor` run it too (the doctor with
     each project's own values when they differ from 10 and 3).
4. **Last line of defence.** The export's datum gate (`check_datum_consistency`)
   still measures the real series after phase linking. Since 5 Oct 2026 it
   judges each mini-stack boundary on cells coherent on both sides (Site B:
   ratio 0.96, weakest boundary 0.82, pass line 0.6; per-boundary figures in
   `stack/merged/linked_datum_check.json`). What to do when it trips is in
   `INSAR_RUNBOOK.md`, section 11.

Run the check by hand, in the miaplpy environment:

    conda run -n miaplpy python ~/insar_pipeline/patches/miaplpy/check_miaplpy_patch.py --ministack 10 --time-lag 3

Exit 0 and `"patched": true` in the JSON line mean the installed build carries
the fix.

## If you ever want a newer MiaplPy

Rebase the patch onto the new upstream commit and update `UPSTREAM_COMMIT`
and `VERSION_LABEL` in `build_miaplpy.sh`. Run the script and the check must
pass. Upstream has not addressed this as of 0.2.1.

Backup of the build installed before 28 Sep: `backups/miaplpy_installed_20260928/` in Kyle's
copy of the package (not shipped in releases).

## Licence

MiaplPy is GPL-3.0-or-later (Copyright 2021 Sara Mirzaee). The patch is a
modification of it: the build labels itself `0.2.1+overlapdatum`, and this page
records what changed and when (25-28 Sep 2026). Using it inside the firm needs
nothing more. Giving the pipeline or this build to anyone outside the firm
needs a licence review first (`THIRD_PARTY.md`).

## Moving the package folder

The pip lock names `constraints.txt` in the folder that built MiaplPy.
`install/install.sh` writes a new `constraints.txt` in its own folder and
points the lock at it, so moving or replacing the package folder (an update)
never leaves pip pointing at a missing file. `constraints.txt` is written per
machine and is not shipped in releases.
