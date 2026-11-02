# Experiment runs A and B: configs, protocols, how to launch

> **28 Oct 2026:** the run configs this page lists (`exp_*.yaml`) were removed from the repository
> (Kyle). This page stays as the record of the runs: their protocols, the sha256s fixed for them
> (which `validation.py` still checks a config against) and the digitised deforming area in
> `context/`. To run one again, write its config from the table below into the projects folder's
> `configs/`.

Written 8 Oct 2026 (P0.1 of `review_2026-10-08/PLAN.md`), brought up to date after phases 1, 2 and 3
the same day. These are **validation runs, not client projects**. Their numbers are made by the
validation harness (`validation.py`, `insar validate`) from each project's products, under protocols
fixed before any data existed. Every config sets `purpose: experiment`: the products stage ends with
the harness under the config's `validation_protocol` and builds no client report, client KMZ, release
status or package. `site_profile` (`mine_longwall` at A, `groundwater_basin` at B) keeps the QA from
calling its flags "expected on farmland".

**No run has started, no experiment data has been downloaded and no result exists** (8 Oct 2026).

## What is here

Sentinel-1, phase-linked (the official route):

| file | run | track | window | dates | site |
|---|---|---|---|---|---|
| `exp_a1_appin_des147.yaml` | A1 | DES-147, IW1+IW2 (4 bursts) | 2017-11-01 → 2022-12-31 | 143 | Appin LW902 + Tahmoor West, NSW |
| `exp_a3_appin_asc9.yaml` | A3 | ASC-9, IW1 (3 bursts) | 2019-11-01 → 2022-12-31 | 96 | same box |
| `exp_b1_corcoran_asc137.yaml` | B1 | ASC-137, IW2 (3 bursts) | 2021-10-01 → 2026-09-30 | 192 | Corcoran station box, CA |
| `exp_b3_corcoran_des144.yaml` | B3 | DES-144, IW2 (3 bursts) | 2021-10-01 → 2026-09-30 | 188 | same box |
| `exp_b2_houston_asc34.yaml` | B-2a | ASC-34, IW2 (3 bursts) | 2021-10-01 → 2026-09-30 | 145 | Houston SW / Katy, TX |

OPERA DISP-S1 through the record's chain, each on its Sentinel-1 run's grid:

| file | run | frame | window | granules | site |
|---|---|---|---|---|---|
| `exp_a2_appin_disp_f39360.yaml` | A2 | F39360 (DES-147), A1's grid | 2017-11-01 → 2022-12-31 | 127 | A box (F39360 covers 62% of it) |
| `exp_b2_corcoran_disp_f36542.yaml` | B2 | F36542 (ASC-137), B1's grid | 2021-10-01 → 2026-03-28 | 167 | Corcoran station box |
| `exp_b4_corcoran_disp_f38502.yaml` | B4 | F38502 (DES-144), B3's grid | 2021-10-01 → 2026-02-14 | 153 | Corcoran station box |
| `exp_b2b_houston_disp_f08882.yaml` | B-2b | F08882 (ASC-34), B-2a's grid | 2021-10-01 → 2026-02-19 | 128 | Houston SW / Katy |

NISAR L2 GUNW, PROVISIONAL only (the route of `nisar_backend.py`; one geometry and one box per project):

| file | run | track | window | pairs (ASF, 8 Oct) | site |
|---|---|---|---|---|---|
| `exp_a5a_appin_nisar_asc167.yaml` | A5 | ASC-167 | 2026-06-01 → 2026-10-08 | 7 | A box |
| `exp_a5b_appin_nisar_asc95.yaml` | A5 | ASC-95 | same | 7 | A box |
| `exp_a5c_appin_nisar_des16.yaml` | A5 | DES-16 | same | 7 | A box |
| `exp_b5a_corcoran_nisar_asc34.yaml` | B5 | ASC-34 | 2025-11-01 → 2026-09-30 | 24 (one chain) | Corcoran station box |
| `exp_b5b_corcoran_nisar_des114.yaml` | B5 | DES-114 | same | 20 (one chain) | Corcoran station box |
| `exp_b5c_elnido_nisar_asc34.yaml` | B5 | ASC-34 | same | as Corcoran | El Nido-Madera box |
| `exp_b5d_elnido_nisar_des114.yaml` | B5 | DES-114 | same | as Corcoran | El Nido-Madera box |
| `exp_b5e_elnido_nisar_des42.yaml` | B5 (optional) | DES-42 | same | chain broken: the 19-date piece is kept | El Nido-Madera box |
| `exp_b2c_houston_nisar_asc62.yaml` | B-2c | ASC-62 | 2025-11-01 → 2026-09-30 | 25 (26 dates) | Houston SW / Katy |

Context and truth (PLAN T1, digitised 8 Oct 2026; how they were made: `review_2026-10-08/truth/README.md`):

| file | used by |
|---|---|
| `context/appin_tahmoor_deforming.geojson` | A1, A2, A3: the deramp exclusion, far field and reference-pixel rule (the digitised panels + angle of draw + 50 m) |
| `context/appin_tahmoor_deforming_APPROX.geojson` | nothing now; the hand-drawn envelopes of P0.1, kept for the record |
| `review_2026-10-08/truth/panels.geojson` | A1, A2, A3 `event_polygons`: the harness's stable ring and Tahmoor West peak, and the event products |
| `review_2026-10-08/truth/survey_lines.geojson` | A1, A2, A3 `event_lines`: the Appin peak ratio, and the event products' profiles |
| `review_2026-10-08/truth/tahmoor_west_gnss_units.csv` | A1, A3 `validation_points`: the 17 Tahmoor West GNSS units (protocol A `tahmoor_w_units`) |

Common to every config:

- `project_dir` is `$INSAR_PROJECTS/exp_<name>`, so the projects land in `~/insar_projects/exp_*`.
  Nothing there exists yet.
- `gnss_tie_radius_km: 0` and `gnss_validation: false`, because the tie would anchor to the stations
  being validated. GNSS enters only through the harness.
- The Sentinel-1 configs use the phase-linked settings the Indiana record ran with
  (`configs/config_site_a_pl.yaml`). Their dates are the ASF **burst** catalogue's on 8 Oct 2026,
  which is what the download fetches (PLAN's 194, 193 and 147 for the B runs counted SLC scenes; the
  burst catalogue lacks 2-5 scenes from Apr-Jul 2026). Bursts, coverage, stations and reference dates
  were checked with the pipeline's own burst search (metadata only):
  `review_2026-10-08/p0/protocol/burst_check.{py,log,json}`.
- The NISAR pair counts come from ASF metadata of 8 Oct 2026
  (`review_2026-10-08/p23/nisar/meta_probe_20261008.*`).

## The protocols come first

The three protocols are fixed: `review_2026-10-08/protocol_A.yaml` (A1-A5), `protocol_B1.yaml` (B1-B5)
and `protocol_B2.yaml` (Houston, B-2a to B-2c). Each one sets the AOI and window, which stations
calibrate or reference and which are scored, the sampling radius (100 m; 160 m for NISAR), the 0.4
coherence floor, the reference (datum) method, the models and the PLAN 5.3 thresholds.

They were fixed **before** any run and the text above each file's `amendments:` line must never be
edited. A change goes into the `amendments` list, with its date and reason, and every statistic is then
reported under both versions. `tests/test_p0_protocol.py`, `test_p1_finish.py` and `test_p23_amend.py`
fail if a file no longer matches the hashes below.

| protocol | sha256 (with the amendments of 8 Oct 2026, appended before any run) |
|---|---|
| `protocol_A.yaml` | `905261986b6864c7d25744b0babaf2610e0720c04ba2fb52aa86560360bb45de` |
| `protocol_B1.yaml` | `aefd45e0b86ae76219f7c2c88275a80f153cd084b1c4080c9e27eae6d3cfcc7a` |
| `protocol_B2.yaml` | `6efd8497b81bb66113293857ff506b1c6a64480a1d28fd5e6c429a2421975da0` |

Check: `sha256sum ~/insar_pipeline/review_2026-10-08/protocol_{A,B1,B2}.yaml`

**As first fixed.** Everything above each file's `amendments:` line is the text fixed on 8 Oct 2026
(after the independent review that day), byte for byte. These were the files' hashes then, when the
last line read `amendments: []`. The harness rebuilds that text (the file cut at `amendments:` plus
`amendments: []`) and checks it against this table (`validation.json`, `protocol.fixed_text`).

| protocol, as first fixed | sha256 |
|---|---|
| protocol_A.yaml (first fixed) | `6f2f23accccdc5d074462fe7149b339abde9a6035e6976cc6b261d2185930a69` |
| protocol_B1.yaml (first fixed) | `c567492b1d47c0d4a6ba397b3bac050d8cd5817d29382105c963bae4c61d88e1` |
| protocol_B2.yaml (first fixed) | `89e248e17c5d6693597a319fa9345e902e9bba3f438a4d24739af8c5a5bfd25f` |

**The amendments of 8 Oct 2026** were all appended before any experiment run, download or comparison
output existed. The first five are Kyle's answers to the decisions phase 1 left open (HANDOFF.md, "8 Oct:
validation experiments - phase 1"). The `p23-*` ones answer the decisions phases 2 and 3 left open; Kyle
delegated them, and Claude made them for him (`decided_by: Claude for Kyle (delegated 8 Oct)`). The
`configs` ones and `p23-7` are records only. Each carries its date, its reason and the `harness:` keys the
harness applies:

| id | in | what changes |
|---|---|---|
| `2026-10-08/1` | A | The 95% interval of an event change is computed by the harness from stable ground: the robust spread, about zero, of the same double-differenced change over 100 m discs on the stable ring (1-3 km outside the T1 panels + draw distance; without the outlines, on the ground more than 2 km from the deramp polygon). The route rule's uncertainty-calibration part is then graded instead of reading "not stated". |
| `2026-10-08/2` | B1, B2 | `york_slope` is graded on per-station water-year changes (B2: annual changes), with each year's mean over the stations removed so the reference pixel cancels. The rate-based York slope and `rate_correlation` are reported, not graded. |
| `2026-10-08/3` | B1, B2 | The plane check uses the joint fit and is graded (`plane_check`). Offset-only stays primary and alone enters the route verdict. |
| `2026-10-08/4` | A, B1, B2 | The GNSS daily sigma is the larger of the robust and the formal sigma. Which one was used is recorded. |
| `2026-10-08/6` | A | Confirmed: in a run with no CA19 epochs (A3, D1), the reference station's coverage is judged on the run's own dates. |
| `2026-10-08/configs` | A, B1, B2 | Record only: the DISP-S1 configs of A2, B2, B4 and B-2b, so the harness finds those runs. |
| `2026-10-08/a-deramp-polygon` | A | Record only, made by the phase 2/3 finishing pass (not decided by Kyle): A1, A2 and A3 keep the **digitised** polygon `context/appin_tahmoor_deforming.geojson` out of the deramp fit, instead of the hand-drawn APPROX envelopes. No graded quantity changes (every panel statistic already used the T1 outlines). The harness warns when an A run's config names another polygon. To undo: delete the amendment and point the three configs back at the APPROX file. |
| `2026-10-08/nisar-configs` | A, B1, B2 | Record only, made by the finishing pass: the NISAR configs of A5 (3 projects), B5 (5 projects) and B-2c, so the harness finds those runs. |
| `2026-10-08/p23-1` | A | Decided by Claude for Kyle (delegated). The Tahmoor West digitising error is, per point, hypot(two-pass repeatability, half the plotted line width in mm, axis-calibration residual), not the half-gridline floor: Up sigma 2.2-4.1 mm instead of 10-50 mm (the CSV carries every term and the first-fixed sigma). |
| `2026-10-08/p23-2` | A | Decided by Claude for Kyle. The coherence hole is relative: the cell's coherence (3 x 3 look-cell mean) at most 2/pi of its 1-3 km ring's median (ring median >= 0.4). The honest-or-flagged row passes only if the flag marks at most 20% of the stable-ground placements over the row's intervals; else "flag not selective" (fail). |
| `2026-10-08/p23-4` | A | Decided by Claude for Kyle. Site 13's rows from 2022-03-28 (after its relocation) are a separate unit, TW_G13b, with its own datum; no change across the gap. |
| `2026-10-08/p23-5` | A, B1, B2 | Decided by Claude for Kyle. F4's cycle-consistency test stays in C-band LOS (NISAR projected under the vertical-motion assumption); its accuracy comparison is in vertical (each sensor's LOS / cos of its own incidence). Both reported. |
| `2026-10-08/p23-6` | A | Decided by Claude for Kyle. A2 (DISP-S1) gets an accuracy verdict only, labelled "honesty not assessable (OPERA unwraps its own pairs; no per-pair aliasing record)". |
| `2026-10-08/p23-7` | A | Record only: the readings the phase 2/3 groups chose where the protocol is silent, accepted and stated. |

The harness reports every row under the amended protocol and, in `validation.json` `version_1`, under
the protocol as first fixed.

## Order, disk and time

Run **one phase-linked project at a time**: MiaplPy uses 20 of 24 cores. "Own peak" is the run
alone. "Peak with kept" adds what earlier steps keep after their cleanup. Sentinel-1 and DISP-S1
figures are from `review_2026-10-08/disk/plan_model.py` and `plan/b_aoi_check*.log`; the NISAR figures
are the NISAR route's (windowed reads, 8 Oct 2026; PLAN's 148 and 194 GB were whole granules).

| step | run | config | needs first | own peak GB | peak with kept GB | kept after GB | hours |
|---|---|---|---|---|---|---|---|
| 0 | Products-only on Site B and Site A, sign-offs, then the PLAN 4.1 cleanup | – | – | | | | ~1 |
| A1 | S1 DES-147, Appin + Tahmoor | `exp_a1_appin_des147.yaml` | – | 190 | 190 | 10 (+28 slcStack, see A1b) | 13–15 |
| A1b | A1 re-linked with smaller MiaplPy windows (F3) | **not written** (below) | A1 "done" | ~58 for phase linking (2.1 × A1's ~28 GB SLC stack) + pair files + MintPy (UNCONFIRMED) | | | |
| A2 | DISP-S1 F39360 on A1's grid, then level 1 | `exp_a2_appin_disp_f39360.yaml` | A1 "done" | ~1.4 | ~11 | ~11 | 2–4 |
| A3 | S1 ASC-9, same box | `exp_a3_appin_asc9.yaml` | – | 101 | 112 | 18 | ~9 |
| B1 | S1 ASC-137, Corcoran | `exp_b1_corcoran_asc137.yaml` | – | 227 | 245 | 38 | 24–29 |
| B2 | DISP-S1 F36542 on B1's grid (RAM ~9–10 GB: nothing else heavy), then level 1 | `exp_b2_corcoran_disp_f36542.yaml` | B1 "done" | ~6 | 44 | 39 | 2–4 |
| B3 | S1 DES-144, Corcoran | `exp_b3_corcoran_des144.yaml` | – | 226 | **265 (plan maximum)** | 59 | 24–29 |
| B4 | DISP-S1 F38502 on B3's grid, then level 1 | `exp_b4_corcoran_disp_f38502.yaml` | B3 "done" | ~5 | 64 | 60 | 2–4 |
| B-2a | S1 ASC-34, Houston SW / Katy | `exp_b2_houston_asc34.yaml` | B-1 shows no pipeline defect | 163 | 223 | 73 | 15–18 |
| B-2b | DISP-S1 F08882 on B-2a's grid | `exp_b2b_houston_disp_f08882.yaml` | B-2a "done" | ~5 | ~78 | 74 | 2–4 |
| A4 | S1 DES-147, 2025-07 → 2026-10 (LW711), C-band partner of A5 | **not written** (below) | – | 49 | 123 | 79 | ~4 |
| A5 | NISAR ASC-167, ASC-95, DES-16 over the A box | `exp_a5a…`, `exp_a5b…`, `exp_a5c…` | – (F4 needs A4) | < 1 each | | < 1 each | < 1 each, plus ERA5 |
| B5 | NISAR ASC-34 + DES-114 at Corcoran and El Nido, DES-42 at El Nido (optional) | `exp_b5a…` to `exp_b5e…` | – (F4 needs B1/B3) | < 1 each | | < 1 each | < 1 each, plus ERA5 |
| B-2c | NISAR ASC-62, Houston SW / Katy | `exp_b2c_houston_nisar_asc62.yaml` | – (F4 needs B-2a) | < 1 | | < 1 | < 1, plus ERA5 |

- **Plannable after the PLAN 4.1 cleanup: ~420 GB.** The largest moment is B3, at 265 GB.
  - With Site B cleaned to level 1 only (~250 GB plannable), A fits but B1 and B3 do not.
- **The NISAR runs are small and can run at any time**, also beside a phase-linked run (they need no
  MiaplPy), but **not beside a DISP-S1 run** (both stream from Earthdata). Each reads ~0.3-1.5 GB over
  the network as windows of the granules and never stores a whole granule. Their C-band comparison
  (F4, `lband`) needs the matching Sentinel-1 run "done": B5 with B1/B3, B-2c with B-2a, A5 with A4.
- **A1 keeps its slcStack** (`miaplpy_delete_slc_stack_after_concatenate: false`, ~28 GB), the one
  deliberate exception to PLAN 4.2's "true everywhere", because A1b re-links A1's own stack.
  - Until A1b has run, or been dropped, do not clean A1 to level 2. Run
    `du -sh ~/insar_projects/exp_a1_appin_des147` after A1 ends.
  - Decide on A1b before B1 starts.
- B-2 (Houston) runs only if the B-1 analysis shows no pipeline defect (PLAN 1B).
- Machine time is about 100–125 h for the Sentinel-1 and DISP-S1 steps; the NISAR runs add a few hours.
  With Kyle's checks between steps, the wall time is longer.

## Before every run

1. **Disk:** `python3 ~/insar_pipeline/review_2026-10-08/disk/real_budget.py`. REAL must be at least the
   step's **own peak + 30 GB** (REAL already excludes what earlier steps keep). The dashboard, planner,
   preflight and watchdog read the same figure (P0.2, landed 8 Oct 2026).
2. **No other phase-linked run is going** (Sentinel-1 steps), and no DISP-S1 run beside a NISAR run.
   Check the dashboard's Projects list.
3. **The protocols are unchanged:** run the `sha256sum` line above and compare with the table.
4. **A DISP-S1 config loads only after its Sentinel-1 run has ended "done"** (it reads that run's
   `mintpy/geo/`; before then it refuses with "disp_target_grid_file ... does not exist", on purpose).

## How to launch (dashboard)

The configs in `configs/experiments/` are **not in the dashboard's Projects list** until they have
been started once (the list reads the run registry `~/.insar_pipeline_cli.json`, `configs/*.yaml` and
the projects folder's `configs/`, not sub-folders). `insar <config>` opens the dashboard on that
project's live view:

```bash
# Sentinel-1, phase-linked (one at a time)
insar ~/insar_pipeline/configs/experiments/exp_a1_appin_des147.yaml           # A1
insar ~/insar_pipeline/configs/experiments/exp_a3_appin_asc9.yaml             # A3
insar ~/insar_pipeline/configs/experiments/exp_b1_corcoran_asc137.yaml        # B1
insar ~/insar_pipeline/configs/experiments/exp_b3_corcoran_des144.yaml        # B3
insar ~/insar_pipeline/configs/experiments/exp_b2_houston_asc34.yaml          # B-2a (Houston)
# OPERA DISP-S1, each after its Sentinel-1 run is done
insar ~/insar_pipeline/configs/experiments/exp_a2_appin_disp_f39360.yaml      # A2, after A1
insar ~/insar_pipeline/configs/experiments/exp_b2_corcoran_disp_f36542.yaml   # B2, after B1
insar ~/insar_pipeline/configs/experiments/exp_b4_corcoran_disp_f38502.yaml   # B4, after B3
insar ~/insar_pipeline/configs/experiments/exp_b2b_houston_disp_f08882.yaml   # B-2b, after B-2a
# NISAR GUNW (small; any time, not beside a DISP-S1 run)
insar ~/insar_pipeline/configs/experiments/exp_b5a_corcoran_nisar_asc34.yaml  # B5, Corcoran ASC-34 (run this one first)
insar ~/insar_pipeline/configs/experiments/exp_b5b_corcoran_nisar_des114.yaml # B5, Corcoran DES-114
insar ~/insar_pipeline/configs/experiments/exp_b5c_elnido_nisar_asc34.yaml    # B5, El Nido ASC-34
insar ~/insar_pipeline/configs/experiments/exp_b5d_elnido_nisar_des114.yaml   # B5, El Nido DES-114
insar ~/insar_pipeline/configs/experiments/exp_b5e_elnido_nisar_des42.yaml    # B5, El Nido DES-42 (optional)
insar ~/insar_pipeline/configs/experiments/exp_b2c_houston_nisar_asc62.yaml   # B-2c
insar ~/insar_pipeline/configs/experiments/exp_a5a_appin_nisar_asc167.yaml    # A5, ASC-167
insar ~/insar_pipeline/configs/experiments/exp_a5b_appin_nisar_asc95.yaml     # A5, ASC-95
insar ~/insar_pipeline/configs/experiments/exp_a5c_appin_nisar_des16.yaml     # A5, DES-16
```

Then, in the live view:

1. **Size it first.** Press `s` and choose **"Size the run only (downloads nothing)"**.
   - It starts at once, with no preflight. It queries ASF and writes the planner's peak, dates and
     bursts (or granules) into the log shown on the left, and creates the project folder.
   - Sentinel-1: check the dates, the bursts per date (4 for A1, 3 for every other run) and the peak
     against the tables above. The plan also prints F1's fast-motion note (the quarter-wavelength
     limit per interval); no config sets `expected_max_rate_mm_yr`, so it gives no site verdict.
2. **Start it.** Press `s` and choose **"Whole pipeline: data, processing, products, report"** (on a
   fresh project this is also the recommended choice). It opens the **Preflight** screen (Earthdata, the
   ERA5 token and licence, disk, memory and, for phase-linked runs, the patched MiaplPy). Press **"Start
   the run"** when nothing is marked ✗.
3. The run is detached: closing the dashboard never stops it. After the first start the project
   appears in the Projects list. If a run is interrupted, open the project and press `s`; the
   **Recommended** restart is worked out from the files on disk.

What each route does on the way (none of it needs a setting):

- **Sentinel-1 at A:** F2 (`aliasing_diagnostic: auto` is on for experiment runs) measures each
  exported pair for aliasing before snaphu and writes `<project>/aliasing/`. The products stage
  writes `deliverables/quality_aliasing_risk.tif` and `aliasing_risk.json` (0 no risk found, 1 risk,
  2 likely aliased, 255 outside the radar crop; a lower bound) and the event products in
  `deliverables/events/` (A1, A2, A3: the change over the whole record, each digitised panel's
  deepest subsidence and its date over its own mining dates, a profile along each survey line).
- **Sentinel-1 at B:** F2 runs too (experiment runs); there are no event layers.
- **NISAR:** the route searches PROVISIONAL GUNW only, keeps the config's track and direction, one
  granule per pair, reads each as a window over the box, resamples to one 80 m lat/lon grid, stages
  a GLO-30 DEM and lets MintPy's `prep_nisar` load the crops (its line-of-sight azimuth is corrected
  on the way). Every choice is in `nisar_gunw/nisar_stack.json`; the reliability gate (mean pair
  coherence >= 0.4, protocol_B1's NISAR rule) is `nisar_gunw/coherence_gate.tif`, which the harness
  reads. The route stops at the deliverables (no QA, figures or client report), then the harness.
  A chain with a missing pair keeps its longest piece (B5e) and lists what it dropped.

When a run ends "done", **do not read the QA verdict as the experiment's result**. Check that the run
finished, that its "Validation against ground truth" stage shows ✓, the dates it kept (the QA, or
`nisar_stack.json`, lists dropped dates) and the coverage. The result is the validation, read next.

## After each run

### 1. Read the validation

The products stage ran the harness under the config's protocol and wrote
`~/insar_projects/exp_<name>/validation/protocol_<id>_<run>/` (for example `protocol_A_A1/`,
`protocol_B1_B3/`, `protocol_B1_B5/`, `protocol_B2_B-2c/`):

- `validation.md`: the verdict, one line per graded row with its interval, the reason for every row
  not graded, the coherence-floor sensitivity, the protocol's power statement and the warnings;
- `validation.json`: everything, with the GNSS files' sha256, fetch dates and last epochs, the truth
  files used (`truth.points_file`, `events.outlines_file`, `events.lines_file`) and the aliasing flag
  read (`aliasing`);
- `validation_epochs.csv` and the plots.

It first says whether the protocol is the fixed one (the sha256 above). A row reads pass, usable with
stated caveats, fail, not testable or not computed, never a silent pass.

- **A1, A3 (and A2):** the configs give the harness the T1 files (`validation_points`,
  `event_polygons`, `event_lines`), so the stage's own validation already has the Tahmoor West
  units, the stable ring for amendment `2026-10-08/1`, the Tahmoor West peak and the Appin peak ratio.
  The honest-or-flagged row reads F2's flag for each comparison's own dates from
  `<project>/aliasing/pairs/` (only the pairs the record's `summary.json` lists); the whole-record
  tif stands in only when that record is missing, and the output says so. It passes only if the flag
  is selective (amendment `2026-10-08/p23-2`): at most 20% of the stable-ring discs flagged over the
  row's own intervals (`flag_selectivity` beside the row), else it reads "flag not selective". The
  Tahmoor West units carry the measured digitising sigma (`p23-1`) and Site 13 after its relocation is
  unit `TW_G13b` (`p23-4`); `validation.json` `truth.points_rules` says what was applied. A2 (DISP-S1)
  has no F2 record (OPERA unwraps): its honest-or-flagged row and its verdict read "honesty not
  assessable (OPERA unwraps its own pairs; no per-pair aliasing record)", and its verdict is an
  accuracy verdict only (ca19_lw902, bias, uncertainty calibration; amendment `p23-6`).
- **NISAR (A5, B5, B-2c):** the harness reads the mean pair coherence as its reliability quantity
  (`validation.json` `product.reliability_quantity`) and the line of sight from
  `mintpy/inputs/geometryGeo.h5`. The NISAR rows themselves (`comparisons.nisar`) are still "not
  computed"; the other rows run over the run's window (A5 has no window of its own, so its rows find
  no truth: A5's test is F4). **On the first NISAR result (B5a), check the sign:** CAHA and QCAP sink
  fast and at different rates (MIDAS Up about -160 and -67 mm/yr), so the station series rows
  (`series_rmse`, `series_correlation`) must agree in sign; a clearly negative correlation means the
  phase sign is wrong (the route's sign rests on ISCE3's convention; `nisar_backend.physical_sign_check`
  is the same test on a map).

### 2. Make it again, or with other truth (`insar validate`)

```bash
insar validate ~/insar_pipeline/configs/experiments/<config>.yaml \
    --protocol ~/insar_pipeline/review_2026-10-08/<protocol>.yaml \
    --out ~/insar_projects/exp_<name>/validation/<a new folder name>
```

- It only reads the project; it refuses while a run holds the project. Without `--out` it overwrites
  the stage's own folder. GNSS is downloaded into `<out>/gnss_cache` (`--cache DIR` to share one).
- `--ga-xlsx FILE`: protocol A takes the Geoscience Australia CEMP workbook first and downloads it
  itself; give the file only if that download fails (the run then says it fell back to NGL).
- `--points FILE`: digitised or survey truth (CSV: id, lat, lon, date, up_mm, sigma_mm, ...). Default:
  the config's `validation_points` (A1/A3: `review_2026-10-08/truth/tahmoor_west_gnss_units.csv`).
- `--outlines FILE`, `--lines FILE` (protocol A): the digitised panels and end-of-panel survey lines.
  Default: the config's `event_polygons` / `event_lines`. A line is matched to a protocol maximum by
  its `end_of_panel_maxima` list (panel and mm), as the T1 file has it.
- `--up FILE`: the decomposed Up series, for the secondary vertical rows (next step).

### 3. Up and East from both geometries (D1), once both runs of a site are done

```bash
python -m insar_pipeline.decompose \
    ~/insar_pipeline/configs/experiments/exp_a3_appin_asc9.yaml \
    ~/insar_pipeline/configs/experiments/exp_a1_appin_des147.yaml \
    --out ~/insar_projects/exp_d1_appin_a1a3
insar validate ~/insar_pipeline/configs/experiments/exp_a1_appin_des147.yaml \
    --protocol ~/insar_pipeline/review_2026-10-08/protocol_A.yaml \
    --up ~/insar_projects/exp_d1_appin_a1a3/vertical_timeseries_decomposed.h5 \
    --out ~/insar_projects/exp_a1_appin_des147/validation/protocol_A_A1_with_d1
```

At B use `exp_b1_corcoran_asc137.yaml` and `exp_b3_corcoran_des144.yaml` with `protocol_B1.yaml`.
The decomposition works over the window both runs share (A: 2019-11 to 2022-12), on monthly epochs;
North is not estimated, and every output says so. It reads `mintpy/geo/`, which cleanup level 2 keeps.

### 4. C-band against L-band (F4), once a NISAR run and its Sentinel-1 partner are done

```bash
python -m insar_pipeline.lband \
    ~/insar_pipeline/configs/experiments/exp_b1_corcoran_asc137.yaml \
    ~/insar_pipeline/configs/experiments/exp_b5a_corcoran_nisar_asc34.yaml \
    --out ~/insar_projects/exp_b5a_corcoran_nisar_asc34/lband_vs_b1
```

Pairs: B5a with B1, B5b with B3, B-2c with B-2a, A5a/b/c with A4 (`--plane` at A, where the NISAR runs
have no deramp). It writes `lband_cycle_consistency.json` and `lband_whole_cycle_count.tif`: each
interval's differences classed as agreeing (< 15 mm), a whole C-band cycle (within 5 mm of a non-zero
multiple of 27.7 mm: C-band aliasing) or other, with the chance level. The cycle test is in C-band LOS (the
NISAR change projected under the vertical-motion assumption); beside it, `accuracy_vertical` compares the two
in vertical (each sensor's LOS / cos of its own incidence, "assumes vertical motion"): amendment
`2026-10-08/p23-5`. Protocol B1 reports both, not graded.

### 5. Clean up

```bash
insar cleanup ~/insar_pipeline/configs/experiments/<config>.yaml --level 2           # dry run: read the list
insar cleanup ~/insar_pipeline/configs/experiments/<config>.yaml --level 2 --delete  # typed "yes"
```

- Level 2 keeps `mintpy/geo/`, the inputs MintPy loaded, the deliverables and `aliasing/` (the per-pair
  F2 record the harness reads), which is what the harness, D1, F4 and "Products only" need. It deletes
  `miaplpy/inputs` and `miaplpy/inverted`, `stack/merged`, the SAFEs, the orbits and the cached ERA5
  files, and refuses until the project's last run ended "done".
- **A1:** not before A1b is decided.
- DISP-S1 projects: `--level 1` (granule caches).
- NISAR projects: `insar cleanup` now offers `nisar_gunw/` too: level 1 lists whole granules and unused
  crops, level 2 the cropped windows, only once `mintpy/inputs/ifgramStack.h5` exists.
- Keep the level-2 projects until the experiments are written up.

## Not configured yet

**A1b (F3, smaller MiaplPy windows on A1's stack).** The machinery is in (`miaplpy_relink_from`,
`python -m insar_pipeline.miaplpy_backend relink CFG [--check]`, F3's `fast_zone_geojson`), but the
window sizes are Kyle's choice. To write it: copy `exp_a1_appin_des147.yaml` to a new name (e.g.
`exp_a1b_appin_des147_w9x9.yaml`), set `project_dir: $INSAR_PROJECTS/<that name>`,
`miaplpy_relink_from: $INSAR_PROJECTS/exp_a1_appin_des147` and smaller `miaplpy_range_window` /
`miaplpy_azimuth_window` (A1 has 15 × 15), keep the looks; optionally `fast_zone_geojson` (the digitised
polygon) with `fast_zone_connections: 1`. Then `python -m insar_pipeline.miaplpy_backend relink <cfg>
--check` (it prints the disk estimate; PLAN's "≤ 25 GB" is far too low: phase linking writes about
2.1 × A1's SLC stack) and start it with **"Phase linking onward"** (`--start miaplpy`); a full start is
refused. It writes nothing in A1. The harness finds it once a record-only amendment names it in
`run_configs` (`runs.A1b`; it is a sensitivity run, not graded).

**A4 (Sentinel-1 DES-147, 2025-07-01 → 2026-10-08, LW711)**, the C-band partner A5's F4 needs. It needs
its own deramp polygon (the panels mined in 2025-26: LW710/711 at Appin, the Tahmoor South panels after
S1A; none is digitised), a burst check and a reference date, and a record-only amendment naming it.

## Decisions in these configs, and why

**Reference pixel: left empty in all of them.**
- The run's own rule picks the pixel (`miaplpy_backend.choose_reference`, recorded in
  `stack/reference_choice.json`): unwrapped in ≥ 95% of pairs, with the highest linked temporal
  coherence. At A it must also be more than 2 km from the deforming polygon.
- The experiments' datum is set in the harness instead, where the pixel cancels: A double-differences
  to the stable station CA07; B uses station-pair double differences plus GNSS-absolute leave-one-out.
- At A, the QA also re-reports its headline figures re-referenced to CA07 and PCTN
  (`reference_candidates_lalo`).

**Deramp at B (Corcoran, Houston) and on every NISAR run: off.**
- The whole B box subsides (Corcoran about -40 to -290 mm/yr; Houston tilts about 8–10 mm/yr per
  10 km, the precision under test). A per-epoch plane would absorb the real gradient.
- With the deramp off, the run still writes `mintpy/deramp_effect.json`: the plane a deramp would have
  removed, as a diagnostic. Long-wavelength error is measured in the harness instead.
- The A5 NISAR runs have no deramp either, while A1 has one: `lband --plane` removes a plane from each
  C-minus-L difference.

**Deramp at A (Appin + Tahmoor): linear, with the digitised panels excluded.**
- The bowls are local (1–2 km) in a 21×18 km box.
- The exclusion polygon is `context/appin_tahmoor_deforming.geojson`, built from the T1 outlines by the
  APPROX file's rule: every panel mined, or completed in the six months before the window and still
  settling, in 2017-11-01 → 2022-12-31 (16 panels), each buffered by its angle of draw (tan 26.5° ×
  depth of cover, 204-356 m) plus 50 m; 22.9 km² against 185 km² for the hand-drawn envelopes, which
  also missed part of LW S1A and LW708B themselves. The run adds `deramp_buffer_m` (2 km) on top.
- Checked with the pipeline's own mask builder on a uniform grid of the processed crop: the fit keeps
  85% of the crop (APPROX: 59%) with 21–29% in each quadrant (the run refuses below 3 quadrants with
  5%); about 249 of the box's 376 km² lie more than 2 km from the polygon, for the reference rule
  (APPROX: 74). CA19 lies inside; CA07 (2.8 km), PCTN (1.2 km), MENA (2.4 km) and CRDX lie outside.
- The switch was made before any run (amendment `2026-10-08/a-deramp-polygon`). Changing the polygon
  once a run has started makes a new run. A2 must use the polygon A1 ran with.
- **Decided now, before any result, in case a run stops on these rules** (note the change and the
  run's message in the config's comment and here; the protocols' statistics do not change):
  - The deramp refuses with "The deramp fit ground does not surround the area" (possible if the bushland
    and reservoir ground of the Upper Nepean catchment and Dharawal is incoherent): set
    `deramp_buffer_m: 1000` and restart with the Recommended choice; if it still refuses, set
    `mintpy_deramp: "no"`; the CA07 double difference then carries the datum.
  - The reference rule stops with "No pixel in the AOI is both unwrapped in 80% ...": set
    `reference_lalo: "-34.2216,150.6392"` (CA07, the protocol's datum, 2.8 km from the polygon) and
    restart as the message says.

**Sub-swaths.**

| track | `stack_swaths` | bursts |
|---|---|---|
| DES-147 | `'1 2'` | 2 per sub-swath, equal counts |
| ASC-9 | `'1'` | |
| ASC-137 | `'2'` | |
| DES-144 | `'2'` | |
| ASC-34 | `'2'` | |

No box touches a sub-swath its config leaves out (burst check).

**Reference dates.** Each is a mid-record acquisition that has every burst, in the winter of its
hemisphere:

| run | reference date |
|---|---|
| A1 | 20200727 (DES-147 has no acquisition 2020-05-04 → 2020-07-15) |
| A3 | 20210607 |
| B1, B3 | 20240108 |
| B-2a | 20240301 |

**Looks, unwrapping and phase linking: as Indiana** (11×3 looks, 15 m posting at each site's
latitude, snaphu 2, sequential EMI, mini-stack 10, time lag 3, 3 connections). The experiment tests
the official route as it is; smaller windows for the longwall flanks are A1b's question.

**The DISP-S1 runs (A2, B2, B4, B-2b).**
- Each runs OPERA DISP-S1 through the record's MintPy chain (`disp_record_chain`) on its Sentinel-1 run's
  grid, coarsened 2x to 30 m, with that run's geometry, orbit-stack baselines (DEM error) and ERA5 files.
  The settings that are not the S1 run's are the Indiana DISP-S1 run's (`configs/config_site_a_disp.yaml`).
- Windows are the protocols' (`runs.A2`, `runs.B2`, `runs.B4`, `runs.B-2b`), which end at each frame's last
  secondary date. The harness finds each run through the `2026-10-08/configs` amendment.
- Frames were checked against ASF metadata on 8 Oct 2026: F36542, F38502 and F08882 cover 100% of their
  boxes. **F39360 covers 62% of the A box.** CA19, CA07, PCTN and MENA lie inside it, CA07 only 1.35 km
  from its edge; CRDX lies outside (its null-test entry reads "no InSAR value").
- The chain fills a cell's missing epochs by linear interpolation in time when the cell has values at
  >= 50% of the epochs (`disp_min_valid_fraction`), as in the Indiana DISP-S1 run. The harness cannot
  tell a filled epoch from an observed one. Whole missing epochs, such as F39360's 372-day hole, are not
  filled.

**The NISAR runs (A5, B5, B-2c).**
- One project per geometry and box: windowed reads of two boxes cost less than one read over both,
  and MintPy needs one geometry per stack.
- `ionosphere_correction: "no"` (the route subtracts the product's own split-band screen instead;
  the pipeline's IGS-map correction would correct twice), `mintpy_unwrap_error_method: bridging`
  (a chain has no triangles: phase closure does not apply), ERA5 as every other run.
- The `nisar_*` settings keep their defaults (`config.py`; README "New settings").
- The El Nido box is the El Nido-Madera box of protocol_B1 `site.nisar_extension` (36.744-37.115 N),
  because PLAN's box left out LSJ1.

## Open decisions for Kyle (none blocks A1)

- **The two record-only amendments of the finishing pass** (`2026-10-08/a-deramp-polygon`,
  `2026-10-08/nisar-configs`): confirm or undo.
- **A1b's windows and A4's design** (above).
- The digitising sigma, the coherence-hole rule and its selectivity check, Site 13, F4's frames, A2's
  verdict and the groups' readings were decided on 8 Oct 2026 (amendments `2026-10-08/p23-1` to `p23-7`;
  HANDOFF.md, "Protocol decisions (8 Oct)"). One consequence to read before A1's result: on Site B
  farmland the amended hole still marks 67% of the crop over a year (23% per interval between dates), so
  if Appin's stable ring behaves like it, protocol A's honest-or-flagged row will read "flag not
  selective".
