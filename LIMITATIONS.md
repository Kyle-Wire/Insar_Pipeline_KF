# What the program cannot yet do

The one place to read what is not validated, what is screening only, which
rules the program leaves to the reviewer, and the settings behind its
decisions. Written 2 Nov 2026. The evidence is in [VALIDATION.md](VALIDATION.md);
dated changes are in [CHANGELOG.md](CHANGELOG.md); how to use each feature is
in [USER_GUIDE.md](USER_GUIDE.md).

## 1. Not validated

These have not been checked against independent ground truth, or failed
when they were.

- **Fast longwall subsidence from C-band (Sentinel-1).** It fails. At Appin
  the C-band record caught only about 40% of a 219 mm drop at the panel GNSS
  station, and NASA's OPERA product about 25%. Quiet periods matched within a
  few mm; the failure is during fast sinking (VALIDATION.md, experiment A).
- **NISAR L-band.** Unverified. Time series at Corcoran agree with GNSS
  within 3.7-6.0 mm after less than a year, but the slope against GNSS rests
  on 3 stations, too few to judge. At Appin no public truth exists for 2026.
  Every NISAR client page is labelled "Comparison - NISAR, not validated".
- **Indiana-type rain-fed cropland.** Not tested as such. Its coherence is
  lower than Houston's, so expect wider error bars. The program measures and
  states its own noise.
- **Single-year rates.** Not to better than about 12 mm/yr (12-14 mm/yr error
  at Corcoran). Multi-year rates and totals are what passed.
- **NOAA HRRR weather correction.** The default for new continental-US
  projects since 24 Oct 2026 (`areas.weather_default`). It has not yet been
  checked against GNSS here: the validation used ERA5.
- **The wooded-cell mask** (`mask_wooded`, on by default since 21 Oct 2026).
  The validation predates it, so its figures were made with woods included.
- The comparison with California's official TRE-Altamira InSAR, and the
  stable-ring row at Appin, were never run (VALIDATION.md).

A rerun of Houston with today's defaults (HRRR, woods masked) is the planned
check of the last two items.

## 2. Screening only, not to be quoted as findings

Use these to decide where to look, then confirm by other means.

- **Local change layer** (`local_change.py`, USER_GUIDE 6.10). On 22 Oct 2026
  it failed its validation against the vendor at Site D, against criteria
  fixed in advance, in both configurations tried. Its 12-day steps keep the
  farmland drift.
- **Long-pair feature check** (`feature_check.py`, same day). Both town
  controls passed, but farmland, the spoil patch and the pit edge lose
  coherence within about 6 months, so it could not judge them. It works on
  coherent ground; it cannot yet confirm or reject a farmland feature.
- **East-west movement in a combined ascending + descending project**
  (28 Oct 2026). Indicative: no client threshold, limit or uncertainty. Its
  first injection test caught movement at the stated limit 63% of the time
  against 70% required. A pre-registered re-test passed (78%, 4.4% false
  alarms) on 38 sites instead of 60 and awaits sign-off. The vertical part of
  a combined project is what the report areas are read on.
- **OPERA DISP-S1 route** (`backend: disp`). A quick look at where movement
  might be. The release gate never releases its report. At Appin it missed
  more of the longwall drop than the main route (VALIDATION.md, A2).
- **Phase-linked NISAR route** (`nisar_phase_linked: true`, 28 Oct 2026).
  Its validation gate (Gate 3) is not passed: better than the GUNW route on
  every measure, but short of two thresholds (section 5).

## 3. Rules the program does not enforce

- **"No client longwall numbers from C-band" is an operator rule.** The code
  does not block it. Setting `site_profile: mine_longwall` changes the
  quality assessment's wording and turns on the aliasing (too-fast) check.
  A flag the ground explains then holds the report for a specialist, but a
  sign-off that names a specialist clears that hold. Nothing in the release
  step looks at the site profile or the radar band (checked 2 Nov 2026).
  **The reviewer must apply this rule at sign-off.**
- **NISAR reports.** The label "Comparison - NISAR, not validated" is the
  only guard. The release step has no NISAR-specific refusal (checked 2 Nov
  2026). Do not release a NISAR report as a client finding.
- **East-west from a combined project.** Nothing stops it being quoted. Treat
  it as indicative (section 2).

## 4. Provisional or known gaps

- **DISP-S1 stitch limit** (`ANCHOR_SUSPECT_SHARE`, `quality_assessment.py`).
  The share of cells with a whole-cycle error above which the stitch check
  fails is marked provisional in the code, "as the review recommended".
- **The disturbed-ground layer over-flags** (`disturbed_ground_runs.tif`,
  `time_series.write_disturbed_layer`). On phase-linked records it flags most
  reliable ground, so it is marked unusable there and not delivered. Its
  effect is recorded in the QA (`coverage.disturbed`) and never applied.
- **Dates with missing or duplicated bursts** cannot be assembled by
  burst2safe (9, 6 and 3 dates at B1, B3 and Houston). Not fixed; such dates
  are left out (VALIDATION.md, "Not fixed").
- **`min_spatial_coherence` has no effect.** It is accepted and written to
  MintPy, but MintPy reads it only when coherence-based pair dropping is
  turned on, and the program never turns that on (`config.py`).
- **`nisar_mining.py`** (L-band processing aimed at fast mining subsidence)
  is not run by any command. Treat it as experimental and untested against
  truth.

## 5. Settings and thresholds

"Basis" is the reason given beside the value in the code. "No recorded
basis" means the code gives none.

| Setting | Value | What it does | Basis |
|---|---|---|---|
| `min_temporal_coherence` | 0.4 | Cells below it are not measured | Measured at Site A: 84.5% of the area kept at 0.4, 6.6% at MintPy's 0.7. Literature uses 0.2-0.5 over vegetation |
| `min_unwrap_consistency` | 0.0 (off) | Second quality gate on unwrapping consistency; off means the layer is delivered and its effect recorded, figures are not gated | Decision of 5 Oct 2026: measure the effect first, then decide |
| `mask_wooded` | on | Leaves wooded cells out of every delivered product; the QA still uses every cell | Woods read as rising against farmland (85% of wooded cells positive on one project) |
| Detection power | 0.8 | "Reliably detects sinking of X": X would be caught 4 times in 5 (`significance._power_limit`) | Recorded as decision E5 / contract C1; no calibration recorded |
| `PACE_ALPHA` | 0.05 | False-alarm rate of the change-of-pace test on ordinary farmland (`significance.py`) | No recorded basis |
| `GRADIENT_LIMIT_CYCLES` | 0.5 cycle | Too-fast flag: neighbouring cells differ by more than half a cycle (about 422 mm/yr at 12 days; `aliasing.py`) | Fixed 8 Oct 2026, before any experiment data, in protocol A |
| `MIN_MEASURED_PERCENT` | 50% | Under it an area gets no figure and the report cannot be released (`client_report.py`, `release.py`) | No recorded basis |
| `ANCHOR_SUSPECT_SHARE` | 0.01 (1%) | DISP-S1 stitch check fails above this share of cells with a whole-cycle error | Provisional, as the review recommended |
| `GEOLOCATION_FLAG_M` | 10 m | Geolocation against mapped water is flagged above this offset | The reviewer measured 7.5 m or less on two records (half a 15 m cell) |
| Too-fast cells (`quality_assessment.py`) | 0.1% of the area / 1% of a report area; 5% of a report area | Above the first, a flag; above 5% of a report area, the report is held for a specialist | No recorded basis |
| `PHASE_LINKED_PEAK_GB` (`insar doctor`) | 300 GB | Expected peak disk of a phase-linked decade record | Measured at Site A and Site B, about 270 dates |
| Doctor disk warning | 350 GB | `insar doctor` warns when less is free | 300 GB peak plus room for the products |
| Watchdog warning / stop (`pipeline.py`) | 60 GB / 20 GB | During a run: warn below 60 GB free; stop cleanly below 20 GB on two samples in a row | A full Windows drive can corrupt files of every project. 20 GB covers the free-space figure's possible overstatement (9.5 GB measured 7 Oct 2026); 60 GB: no recorded basis |
| Doctor RAM minimum | 12 GB fail, 20 GB warn | Memory the WSL machine needs | A 15.9 GB machine was killed for memory; 24 GB has run every stage. Below 12 GB a decade stack cannot finish in a working week |
| `local_change_ps_dispersion` | 0.15 | Bright steady points count without partners | Speckle alone gives about 0.3, so 0.04% of points pass at Site D: effectively off |
| `local_change_min_partners` | 4 | Partners a pixel needs before its steps count | No recorded basis for the value (with 0-3 partners noise passed the old gate in 50-75% of steps) |
| `local_change_noise_alpha` | 0.05 | A step counts only above the coherence pure noise reaches with this probability | No recorded basis |
| `local_change_min_steps` | 10 | Counted steps a pixel needs before it can be significant | No recorded basis |
| NISAR Gate 3: spread ratio | 0.7 or less | Phase-linked against GUNW route spread (got 0.78: failed) | Set in the route's plan before the test; revisited unchanged once NISAR has two years |
| NISAR Gate 3: agreement with Sentinel-1 | r 0.3 or more | Up/East agreement (got 0.11: failed) | As above |
| NISAR Gate 3: GNSS station | passed (1.7 against 11.2 mm/yr off) | Agreement with a GNSS station | Limit not recorded in this checkout |

## 6. Known false positives and false negatives

- **C-band under-measures fast longwall sinking** (false negative): about 40%
  of a 219 mm drop at Appin; 129 mm short at the panel station.
- **The too-fast warning cannot single out bad values.** It marked 45-49% of
  stable ground against a 20% limit, about half, so a flagged cell is not
  necessarily wrong and an unflagged one not necessarily right.
- **Stable hilly or forested ground drifts falsely.** At Appin, stable
  stations read 6-10 mm/yr of false drift in hilly, forested reservoir
  country. Not seen at Corcoran or Houston.
- **Woods read as rising** before the mask (21 Oct 2026). On one project 85%
  of wooded cells read positive; leaving them out moved its study-area median
  relative rate from +2.5 to +1.45 mm/yr. With the mask on, movement under
  trees can go unseen (false negative); set `mask_wooded: false` where it
  matters.
- **Cropland drift in the local change layer.** Near two Site D towns,
  cropland read about 8-10 mm/yr lower than the few built-up points kept, in
  its 12-day steps. A check made after the fact; the size is uncertain.
- **Water and crops** lose coherence and are often unmeasured (USER_GUIDE
  6.6).
- **Mine outlines older than the workings**: newer workings cannot be
  recognised as mining (USER_GUIDE 6.6).
