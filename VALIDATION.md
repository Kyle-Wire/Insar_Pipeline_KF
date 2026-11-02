# Validation against independent ground truth (October 2026)

What the pipeline has been shown to measure correctly, what it has not, and the
evidence. Every result below comes from rules (protocols) written and fixed
**before** any experiment ran; later changes were dated amendments made before
any result existed. The protocols, their hashes and amendments are in
`review_2026-10-08/protocol_{A,B1,B2}.yaml` and `configs/experiments/README.md`.
The plan, site research and evidence are in `review_2026-10-08/` (`PLAN.md`).
Each run's full validation report (validation.md / .json, per-date CSV, plots)
is saved in `review_2026-10-08/results/exp_<run>/protocol_<id>_<run>/`; the
experiment project folders themselves were deleted after the experiments.

## What you can rely on

| Kind of job | Evidence | Use it? |
|---|---|---|
| **Slow subsidence, a few mm/yr** (the Indiana regime) | Houston: rate RMSE **2.9 mm/yr** (95% CI 0.5-4.4) against 7 GNSS stations; single stations differ by up to about **5.8 mm/yr**. Bias **-0.5 mm/yr**, stated uncertainties held 7 of 7. **PASS** | **Yes** - multi-year rates and totals, with the stated uncertainty |
| **Long-term subsidence, tens of cm** (groundwater basin) | Corcoran: time series within **6.6-9.3 mm** of 7 GNSS stations over 5 years (r 0.99-0.998), unbiased, scale correct; stated uncertainties held 6 of 6 (both directions). Protocol verdict: **FAIL** for both runs (B1 and B3, below) | **Only what passed:** time series and multi-year totals, within the stated uncertainty. Not single-year rates (12-14 mm/yr error) |
| **Vertical vs east-west** (both viewing directions combined) | Corcoran: vertical within **7.1 mm**, bias +0.4 mm/yr, scale 0.98. Assuming all motion is vertical from one direction was worse (9-12.5 mm, bias up to -5 mm/yr) | **Yes** where both directions exist; say "assumes vertical motion" otherwise |
| **Fast longwall subsidence** (active panels) | Appin: C-band captured only **~40%** of a 219 mm drop at the panel GNSS station; NASA's OPERA product ~25%. The warning flag marks ~half of stable ground, so it cannot single out bad values. **FAIL** | **No** - not with Sentinel-1 C-band. Do not give clients longwall numbers |
| **NISAR L-band** (new satellite) | Corcoran: time series within **3.7-6.0 mm** of GNSS after <1 year; the slope against GNSS was -4.9 from 3 stations (B5a), too small a sample to judge. Appin: holds signal over every mapped panel; one panel (LW709) moving -18 to -47 mm in 3 months, **unverified** (no public truth for 2026) | Promising, **not yet validated** for longwall or rates (too little data) |

**In one line:** for slow to moderate subsidence the pipeline's numbers and
their stated uncertainties agree with independent GNSS at two sites; for fast
longwall subsidence C-band data cannot keep up, and the program's own warning
cannot yet say which numbers are wrong.

## The experiments

### A. Longwall - Appin LW902 + Tahmoor West, NSW, Australia
- **Truth:** Geoscience Australia GNSS CA19 directly over LW902 (-326 mm Up,
  public daily series); 17 Tahmoor West GNSS units digitised from the mine's
  reports (measured digitising error 2-4 mm); digitised panel outlines and
  end-of-panel survey lines; stable stations CA07, PCTN, MENA, CRDX.
- **A1** (Sentinel-1 descending 147, 2017-2022, 143 dates): **fail**. CA19
  over LW902: 129 mm short of a 219 mm LOS drop; series RMS 64 mm; Tahmoor W
  units 30 mm RMS (usable with caveats); bias +58 mm; peak ratio 15-55%.
  Quiet periods matched within a few mm - the failure is during fast sinking.
- **A2** (NASA OPERA DISP-S1, same track): **fail**, worse (165 mm short).
  Two independent processing chains miss the same sinking: a C-band data
  limit, not a fault of this pipeline.
- **A3** (Sentinel-1 ascending 9, 2019-2022, 96 dates): **fail**. Caught ~2x
  more of the Tahmoor West peak than A1 (460 of 897 mm); bias +21 mm (pass);
  units 45 mm RMS.
- **Honesty test:** the aliasing (too-fast) flag marked 45-49% of stable
  ground against a 20% limit: "not selective".
- **Stable ground:** stable stations read 6-10 mm/yr of false drift in this
  hilly, forested, reservoir country (not seen at Corcoran or Houston).
- **A5a** (NISAR ascending 167, Jun-Sep 2026): exploratory, see above.

### B1. Long-term (groundwater) - Corcoran, San Joaquin Valley, California
- **Truth:** 7 NGL continuous GNSS stations in the box (live IGS20 series).
- **B1** (Sentinel-1 ascending 137, 2021-2026, 183 dates): series 9.3 mm,
  r 0.99, uncertainty 6/6, bias -2.4 mm/yr; fails the year-by-year pair-rate
  row (13.7 mm/yr, driven by station KDL2; 7.6 without it).
- **B3** (descending 144, 182 dates): passes every row except the pair-rate
  row (12.4 mm/yr): series 6.6 mm, r 0.998, bias +0.5 mm/yr, scale 1.00.
- **Why the pair-rate row fails:** one-year rates from ~6-7 mm-noise series
  cannot reach the 5 mm/yr limit. Multi-year rates would. The limit was fixed
  in advance and was not changed.
- **Also failed at B1** (independent recompute, 16 Oct 2026): the York slope
  check and the plane check (a per-date plane fitted to the stations).
- **Protocol verdict:** **FAIL** for both runs, B1 and B3. What held is the
  time series and the multi-year totals, within the stated uncertainty; that
  is the only use the table above allows.
- **D1** (B1+B3 split into vertical and east-west): vertical 7.1 mm, as above.
- **NISAR B5a/B5b:** see table. GNSS shows 2-5 cm/yr of real sideways motion
  here; NISAR (looks left) and Sentinel-1 (looks right) see it with opposite
  sign, which explains their ~60 mm disagreement away from the stations.

### B2. Slow subsidence - Houston (Katy), Texas
- **Truth:** 7 independent NGL GNSS sites.
- **B-2a** (Sentinel-1 ascending 34, 2021-2026, 141 dates): **PASS** - the
  route rule's rows all pass (pair-rate RMSE 2.9 mm/yr, 95% CI 0.5-4.4;
  single stations differ by up to about 5.8 mm/yr; bias -0.5 mm/yr, series
  6.4 mm, uncertainty 7/7). Series correlation (one qualifying station) and
  the York slope are weak tests where all stations move at similar small rates.
- **Failed at Houston** (independent recompute, 16 Oct 2026): the plane check.

## What this does and does not validate

- **Validated:** the processing chain (coregistration, phase linking,
  unwrapping, MintPy, ERA5, DEM error), the GNSS-referenced datum, the stated
  uncertainty model, and two-direction decomposition - against independent
  GNSS, in a suburban/mixed site (Houston) and an irrigated-farmland basin
  (Corcoran).
- **Not validated:**
  - Indiana-like **rain-fed cropland** specifically: its coherence is lower
    than Houston's, so expect wider (but stated) error bars. The method is the
    same; the program measures and reports its own noise.
  - **Fast longwall** with any sensor (C-band fails; L-band untested against
    truth).
  - **Single-year rates** to better than ~12 mm/yr.
  - The comparison with California's official TRE-Altamira InSAR (reader not
    written) and the stable-ring row at Appin.
- **Small samples:** 6-7 stations per site, so every figure has a wide
  confidence interval (e.g. Houston's rate RMSE of 2.9 mm/yr has a 95% CI
  of 0.5-4.4).

## Defaults changed since this validation

Two defaults for new projects changed after these experiments ran:
- **Weather correction:** new projects in the continental US use NOAA's HRRR
  weather model (since 24 Oct 2026, `areas.weather_default`). Elsewhere they
  keep ERA5.
- **Woods:** wooded cells are left out of the products by default (since
  21 Oct 2026, `mask_wooded`).

Every figure above was obtained with ERA5 and without the wooded mask. HRRR
has not yet been checked against GNSS here. The planned check is a rerun of
Houston with the current defaults. Until then, see
[LIMITATIONS.md](LIMITATIONS.md).

## Defects found and fixed during the experiments
- Validation's vertical comparison picked a truth source that ended before
  the window (GA workbook) and reported "0 common epochs" (9 Oct).
- NISAR route: MintPy's temporal coherence is 1.0 by construction on a
  12-day chain, so products' reliability masks screened nothing; the mean
  pair coherence now replaces it (`time_series.apply_nisar_pair_coherence`,
  also applied by "Products only" on finished runs; 12 Oct).
- Disk budget overstated free space by ~100-230 GB on a sparse WSL disk (8 Oct).
- Not fixed: ASF catalogue dates with missing or duplicated bursts cannot be
  assembled by burst2safe (9, 6 and 3 dates at B1, B3, Houston; listed in each
  config's `allow_missing_dates`). Duplicate bursts (two processing versions)
  could be de-duplicated in code to recover such dates.

## Optional runs not done
A5b, A5c (NISAR at Appin, other geometries) and B-2c (NISAR at Houston). They
would add evidence on L-band, but cannot validate it until ground truth for
2026 exists (e.g. survey data for the panel being mined) and NISAR chains
reach a year or more.
