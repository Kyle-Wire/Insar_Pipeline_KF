# Changelog

Newest first. The version is `pyproject.toml`'s; each release bumps it and is
tagged `v<version>` in git. Every output also records the 12-character code
hash of the build that made it (`insar --version`), which tells two builds
apart even when their version is the same.

"Numbers that move" says whether a rebuilt project's client figures change.

## [Unreleased]

### Disk: merged images keep only the area (11 Nov 2026, Kyle)
- **Sparse merged SLCs** (`miaplpy_sparse_merged_slc`, on). On the
  phase-linked route each date's merged `<date>.slc.full` is rewritten, right
  after its merge, as a sparse file holding only the phase-linking window:
  same name, size and header, the window's bytes checked bit for bit before
  the original is replaced. The window is written as soon as the lat, lon and
  height merges finish; a window left by another area or an older geometry is
  never used; a load that would read outside a kept window is refused. The run
  plan credits it: 10 years, 3 bursts, 36% crop went from a 465 GB to a
  363 GB projected peak (merged images 347 to 134 GB).
- Numbers that move: none (the bytes MiaplPy reads are the same).

### Readiness review fixes (2 Nov 2026, Kyle)
- **Releases hold no client data** (BLD-06). `make_release.sh` leaves out
  `tests/test_local_*.py` and the golden client numbers, and ships only the
  public Indiana layers from `context/` (it used to take every
  `context/*.geojson`, client report areas included, on the developer's
  machine); the build refuses if anything else is staged. `RELEASE.txt`,
  `install/README.md` and the runbook now say "internal: do not distribute
  outside the operating organisation" (they said the release contained client
  configurations, untrue since 28 Oct).
- **SBOM and file manifest in every release** (DEP-01, SEC-06).
  `scripts/make_sbom.py` writes `SBOM.cdx.json` (CycloneDX 1.5: every pinned
  package of the five environments, with package URLs); `MANIFEST.sha256`
  lists every shipped file, and `insar doctor` checks the installed files
  against it (`integrity.py`). `requirements.txt` is now a pinned reference
  list of what the code imports, not an install source.
- **`insar diagnose <project>`** (ERR-04): one zip for the custodian with the
  run records, the last three logs, the JSON records, the config,
  `insar --version` and `insar doctor --quick`; never `context/`, the report,
  KMZ, rasters or point files (`diagnose.py`).
- **Run-log level and size** (ERR-03): `INSAR_LOG_LEVEL` (DEBUG, INFO,
  WARNING, ERROR; default INFO); a run log over 50 MB rolls over to `.1`-`.3`
  and the run carries on writing under the same name (`runs.LogRoller`).
- **The build in more outputs** (OUT-01): the Results workbook's Summary
  sheet and properties, and every PNG figure (a `Software` text chunk,
  `version.png_metadata`).
- **The map builder serves its own Leaflet** (NET-04) from
  `viewer_assets/vendor/`, and uses system fonts: no page loads anything from
  cdnjs or Google Fonts now.
- **The report says what the software worked out** (HIL-02): every footer
  carries "The readings, verdicts, feature descriptions and possible causes in
  this report are calculated by the software from the measurements...", and
  the Observations say "Calculated by the software". The numbers hash does not
  change.
- **Lint** (QUA-02): ruff runs on every push with the rules in
  `pyproject.toml`; the 56 deliberate exceptions it found are marked, and
  CONTRIBUTING.md, "Lint", says which are acceptable. `nisar_mining.py` is
  labelled an unwired experiment.
- **Vendor comparison CSV** (SEC-05): the vendor's point IDs go through
  `formula_safe` like every other text cell.
- **NISAR phase-linked credits** (DEP-03): the data line names the GSLCs and
  the GUNW layers kept, and dolphin is credited.
- **Docs for people new to the program** (2 Nov 2026): the README now says
  plainly what the program does, what it is and isn't for, what a PC needs and
  where to start; its technical sections moved to `TECHNICAL_REFERENCE.md`.
  `install/README.md` is a step-by-step install for someone who has never used
  Ubuntu, with a troubleshooting table; `USER_GUIDE.md` opens with "Start
  here": a first project in ten steps, how long things take, a glossary and
  where to get help.
- **Docs**: `LIMITATIONS.md` (what the program cannot yet do, its settings
  and their basis, known false positives and negatives); `PRIVACY.md` (the
  personal and client information it keeps); VALIDATION.md corrected (an RMSE
  is not a bound; Corcoran's protocol verdict; NISAR's sign; the defaults
  changed since the validation); EXTERNAL_SERVICES.md and THIRD_PARTY.md
  brought up to the code (HRRR, NISAR GSLC reads, map tiles, dolphin, Herbie,
  CUDA); USER_GUIDE.md gains where the program writes, removing a client's
  data, the export contents and tracing a figure back to its source;
  CONTRIBUTING.md's component map covers every module.
- **No noncommercial snaphu code on the NISAR phase-linked route** (DEP-04).
  dolphin's default snaphu initialisation (MCF) runs snaphu's CS2 module,
  whose licence is "strictly noncommercial"; `nisar_dolphin.py` now sets MST
  (`INIT_METHOD`) in every dolphin config, as the Sentinel-1 routes already
  do, records it in `nisar_pl_run.json`, and moves unwrapped results of an
  earlier MCF run to `superseded_mcf_<stamp>/` before dolphin can reuse them.
- Numbers that move: phase-linked NISAR projects (comparison only) on their
  next run, from the unwrapping on; its Gate 3 figures of 28 Oct were made
  with MCF. Nothing else.

### Added
- **Sign off in the app** (30 Oct 2026, Kyle: the app had no sign-off; the
  page said to use the terminal dashboard). A **Sign off** button beside
  Report opens a dialog with the dashboard's questions: the reviewer's name,
  a tick for every feature, the specialist for a held report, notes, and
  first the hard blockers a sign-off does not settle. The server runs the
  dashboard's own `sign_off_and_release` (`app_server.api_signoff`), so the
  two cannot disagree: a refused sign-off is answered at once with nothing
  recorded; a recorded one rebuilds the report and builds the package in the
  background, one per project, and the Results tab and a message say how it
  ended. Quit waits for that rebuild. `dashboard.SIGNOFF_CLEARS`,
  `signoff_hold` and `signoff_leaves` hold the rule both dialogs use.
  USER_GUIDE.md section 4.7. No figure moves.
- **Releases are built and checked on GitHub** (30 Oct 2026, Kyle). A pushed
  tag `v<version>` runs `.github/workflows/release.yml`: the tag must match
  `pyproject.toml`'s version, the checks and the full suite must pass, then
  `install/make_release.sh` builds the tarball and a GitHub Release gets it
  with its `.sha256` and the Windows setup files. The full suite
  (`tests.yml`, pinned `insar` environment) also runs on pull requests and on
  `main`; every push still runs the one-minute `checks.yml`, which now also
  parses the app's script (`app_assets/app.js`). The repository stays
  private: the workflows fit GitHub's free Actions minutes for private
  repositories. CONTRIBUTING.md, "Commits, branches, releases". No figure
  moves.
  - Three tests failed on a fresh clone, so the suite could not pass on
    GitHub: the worker-clamp test assumed 24 cores (`test_fix_INT1.py`), the
    config-loading test asserted configs that left the repository on 28 Oct
    (`test_fix_L4.py`, now skips without them), and the docs-path test
    wanted the git-ignored `tests/test_local_*.py` (`test_fix_docs.py`).

### Changed
- **"The firm" is now "the operating organisation" in the release notice and
  the prose** (29 Oct 2026, stage 1). The tarball's
  `RELEASE.txt` line, the installer's closing message, `install/README.md`
  and the runbook now read "do not distribute outside the operating
  organisation": a release goes to whoever operates it.
  `tests/test_fix_oct7_dec_D14D18_cfg.py` asserts the new string.
  - The screening report's "not the firm's full processing" is now "not the
    full processing". No figure moves.
  - Not changed, on purpose: `THIRD_PARTY.md` and `patches/miaplpy/README.md`
    (the GPL-3 distribution boundary, waiting on the licensing answer) and the
    `--firm` setting, which is saved in users' config.

### Removed
- **The example and experiment configs left the repository** (28 Oct 2026,
  Kyle): `configs/config_example_pl.yaml`, `config_hyp3_example.yaml`,
  `config_nisar_example.yaml` and the 18 `configs/experiments/exp_*.yaml`.
  - Projects keep their configs in the projects folder's `configs/`, and
    releases ship none.
  - The tests that checked those configs against their protocols are gone.
  - Tests that only borrowed one as a base now write a small config of their
    own: tides and ionosphere, the recorded reference point, and the
    decompose command line.
  - `configs/experiments/README.md` stays as the record of the runs and their
    sha256s.
  - `.gitignore` and `check_repo_clean.sh` no longer make an exception for
    example configs.

### Changed
- **NOAA HRRR is the default weather model for new projects** (24 Oct 2026,
  Kyle: the Copernicus ERA5 queue had been clogged since 22 Oct;
  `areas.weather_default`). New phase-linked and classic projects in the
  continental US get `mintpy_tropo_method: raider`, `weather_model: HRRR`,
  `raider_env: raider-env`, and a start before HRRR's first day (2016-07-15)
  moves to it, said in the config's header. Elsewhere, on the DISP-S1 route,
  or without `raider-env`, they keep ERA5 and the header says why. The
  preflight checks `raider-env` and that HRRR serves every date the series
  keeps (`tropo_disp.check_model_range`). The installer builds `raider-env`
  from `install/envs/raider-env.yml` (exact conda-forge pins, Herbie from pip
  at 2026.9.2). Existing projects are untouched. The validation in
  VALIDATION.md was done with ERA5; HRRR is not yet checked against GNSS here.

### Added
- **Phase-linked NISAR route: NISAR the Sentinel-1 way** (28 Oct 2026, Kyle).
  Comparison only, like every NISAR route; **its validation gate (Gate 3) is
  not passed** (below).
  - **Why:** the NISAR route read JPL's L2 GUNW interferograms, which exist
    only at 80 m, so on the 15 m grid each value was an 80 m block.
  - **How:** `nisar_phase_linked: true` keeps the route and replaces one layer
    of every kept crop, the unwrapped phase:
    - `nisar_gslc.py` reads the study box of each NISAR L2 GSLC (40 MHz,
      5 m, one grid per frame) over HTTPS; about 60 s and 25 MB per date.
    - `nisar_dolphin.py` runs dolphin (OPERA's phase linker) in a new pinned
      env, `dolphin` (install/envs/dolphin.yml): sequential EMI over
      mini-stacks of 10, an 11 x 11 window (chosen by a pre-registered rule
      against 21 x 21), output every third pixel (15 m), nearest-3 network,
      snaphu, L1 inversion.
    - `nisar_pl.py` swaps dolphin's phase, coherence and components into the
      GUNW crops (JPL's kept as *GUNW). JPL's split-spectrum ionosphere
      screen, troposphere and tide cubes and geometry stay.
    - dolphin's temporal coherence becomes the products' quality layer.
  - **Products:** points 15 m apart; each value combines similar ground
    within about 55 m, and a strong reflector keeps its 5 m pixel.
  - **Gate 3** (first test site, 25 dates over 10 months), against the GUNW
    route:
    - spread 8.74 vs 11.2 mm/yr (ratio 0.78; 0.7 required);
    - agreement with Sentinel-1 Up/East r 0.11 vs 0.04 (0.3 required);
    - GNSS station 1.7 vs 11.2 mm/yr off (passed).
    - Better on every measure, short of two thresholds. They are revisited
      unchanged once NISAR has two years or more.
  - A NISAR project whose crops are on another grid than its config now says
    so after Products only (`nisar_backend.posting_note`).
  - Numbers that move: none (a new option, off by default).
- **Combined view: an ascending and a descending project into vertical and
  east-west movement** (28 Oct 2026, Kyle). Background: one look direction
  reads east-west movement as vertical, so on the first site combined the
  ascending report read -1.75 and the descending +1.68 mm/yr "vertical" over
  the same ground.
  - `combine.py`: a combined project of its own (`configs/<name>_combined.yaml`,
    `kind: combined`). It reads both projects and never changes them.
    - Method: each run's class-matched relative field on its report's own
      comparison ground; Up and East solved per 15 m cell; Up anchored with
      the regional GNSS rate; both tested against the comparison ground
      (`significance.assess`).
    - The vertical decides each report area's reading. East-west is
      **indicative**: no client threshold, limit or uncertainty. Its first
      injection test caught movement at the stated limit 63% of the time
      against 70%; a pre-registered re-test passed (78%, 4.4% false alarms) on
      38 sites instead of 60 and awaits sign-off.
    - The comparison ground is named by its land cover ("ground (mostly
      built-up and grassland)"), not "farmland".
    - The result goes out of date when a source's products change.
  - **In the program:**
    - Project list: "combined ascending + descending".
    - Source reports: one line under Independent tests.
    - App: a combined project page with layers (vertical; east-west red =
      east, blue = west). Start → Combine. "Combine with …" is offered when a
      finished project of the opposite pass shares the report areas.
    - Dashboard: a combined screen, and `c` on a source project.
  - **Results viewer (`combine_viewer.py`):** a combined table in
    field_points' form, with the vertical in the vertical slot and east-west
    where line of sight is. Points, 15 m cells, cell clicks and polygons work.
    - Series come from both projects' own report series, the descending
      interpolated onto the ascending dates within one 24-day revisit.
    - Date windows are off; the series' slopes are within about 0.9 mm/yr of
      the stored rates.
    - **Exports** (28 Oct 2026, Kyle): a polygon's CSV and KMZ, of the ~100 m
      points or of every 15 m cell, carry the vertical, the east-west
      (indicative) and each point's series; KMZ balloons share the client KMZ's
      layout.
    - Combine also writes the whole field as
      `deliverables/<name>_combined_DRAFT.kmz`.
    - Save as report area is still refused (save it on one of the two
      projects).
  - Numbers that move: none (new outputs only).
- **Every 15 m cell on the results viewer's map, when zoomed in** (26 Oct
  2026, Kyle).
  - At street level on the whole record, every measured 15 m cell replaces the
    ~100 m points, on the same layer and colour scale (`api_cells`, capped at
    150,000 cells in view). Site E: 25,924 cells around a view, 0.75 s.
  - A click gives the cell's own balloon (the KMZ's wording) and its own
    chart (`api_cell`, 0.1 s). `field_points.cells_from_frames` reads only the
    cell's 3 x 3 block of the time series, less the farmland frames `gather`
    stored. On Site E's points it reproduces the table exactly (the series
    to 0.000008 mm). `gather`'s per-cell figures moved into `cell_maps` /
    `cell_figures`, which both paths use.
  - A link can name a cell (`#cell=lat,lon`).
  - A drawn polygon's panel uses every measured 15 m cell inside
    (`api_polygon_cells`): its figures, average series, histogram, calendar
    years and CSV, over the chosen dates. A date window's noise uses the point
    table's own farmland sample, so cells and points agree. Site E's
    outline holds 48 cells against 2 points; 1.7 s once, then cached per
    polygon. Above 40,000 cells, it falls back to the ~100 m points and says so.
  - The polygon chart draws every cell's own series in light grey behind the
    average, which is their mean. Above 300 cells it draws an even sample of
    300 and says "300 of N". The axis follows the grey lines' 1st-99th
    percentile; the rest are clipped at the plot edge.
  - Date windows stay on the ~100 m points (a window's figures for every cell
    need the whole cube), and the page says so; the 15 m KMZ export follows
    the dates.
  - **Numbers that move:** none.

- **Projects not on disk** (24 Oct 2026, the app's Projects page): a project whose
  folder was deleted after its run (the validation experiments, VALIDATION.md) or whose
  config no longer loads is listed apart under "Not on disk" with what survives of it
  (kept validation reports in `review_*/results/<name>/`, slim archives in
  `_archive/<name>_<date>/`, its run history), an Open button for each and Forget
  (off the remembered list; the yaml is kept), instead of a blank "Not started". A
  broken config shows the loader's reason. Out of the totals and the map.
- **The app window** (24 Oct 2026, `app_server.py`, `app_assets/`, `insar app`,
  USER_GUIDE 5.1). The dashboard's work in a browser window (Edge app mode):
  - Projects on a map and as cards;
  - a project's run as six steps with live progress and time left;
  - stages, results, log and activity tabs;
  - Start (recommended restart first, preflight gate), Stop, report, results
    map, folder;
  - This computer.

  The live map draws what the run has made so far: the study box, area of
  interest, mines and report areas, then the satellite frame, the coherence
  and line-of-sight velocity pictures (MintPy's geocoded files, then the
  products), then the measurement points and GNSS stations. Every figure,
  state and restart comes from the dashboard's own functions, so the window
  and the terminal never disagree. A processing run starts only after a
  passing preflight of that kind on that server within 15 minutes. Security is
  the results viewer's: 127.0.0.1 only, per-session token, exact asset list,
  CSP. The logo is `app_assets/icon.png` / `install/windows/insar.ico`. The look is an 80s arcade cabinet in black and white:
  two pixel fonts shipped in `app_assets/fonts/` (Press Start 2P, VT323; SIL
  OFL 1.1), stepped pixel borders, a scanline. The one colour is the "mist": a
  64 x 24 interferogram app.js draws from a two-source Mogi model, pale, in
  hard blocks, on buttons (on hover), the page hero and live progress. The
  map's measurement layers keep their colour scales over a grey base map. Sign-off, package, config editing and freeing disk stay in the
  terminal dashboard for now.
- **Files, more map layers, official map exports** (24 Oct 2026, `file_manager.py`,
  `map_export.py`, the app's Files page and Export map):
  - **Files:** every project's folders with sizes; delete chosen files or
    folders from a button. Nothing `scripts/cleanup.py` protects, nor
    `final_*`, the QA, the GNSS tie or provenance. Never while a run holds the
    project, and the delete holds the project lock. The project's name must be
    typed for anything in `deliverables/` or over 50 GB. Every deletion goes
    in `logs/deletions.jsonl`. Keep results only / Delete project are
    `storage.py`'s; the app wants the name for every project delete.
  - **Map layers:** the client rasters in the report's frame (vertical,
    vertical change, line of sight, line-of-sight change, signal-to-noise,
    uncertainty, the four quality layers), the elevation-model error, ESA
    WorldCover, OSM water, the report's features; topographic and light-grey
    base maps. **Fixed:** the live map's velocity was the raw
    `los_velocity_in_per_year.tif` (farmland about -0.3 in/yr there); it is
    now `line_of_sight_velocity_in_per_yr.tif`, with MintPy's raw files only
    as labelled stand-ins mid-run.
  - **Export map:** Letter to ANSI D, A4, A3, portrait or landscape; colour or
    black and white (a black-to-white ramp, black = down); PDF or 300-dpi PNG.
    The layout has lat/lon ticks, a scale bar (ft/mi and m, true distance at
    the centre), a north arrow, a legend of the layers drawn, notes, data
    credits (`credits.short_line` plus the base map's) and a title block (the
    company logo and name, preparer, date, print scale, status). DRAFT goes
    across the map unless the report is FINAL. It is written to
    `<project>/exports/maps/` with a JSON of its options.
  - **Company** (This computer): firm, preparer and contact (the settings the
    reports already use) and a logo upload (kept as PNG beside the settings).
  - **Fun:** a soft ring through the mist under the pointer, and a pixel
    satellite passing every few minutes ("beep beep beep").
- **Windows setup and desktop icon** (24 Oct 2026, `install/windows/`).
  `Setup InSAR Pipeline.cmd` / `setup.ps1`:
  - installs Ubuntu in WSL;
  - sizes `.wslconfig` (an existing memory line is kept; it asks before
    `wsl --shutdown`);
  - copies the release into Ubuntu, checks its `.sha256` and runs
    `install.sh`;
  - makes an **InSAR Pipeline** shortcut on the desktop and in the Start menu.

  The shortcut runs `insar app` hidden. `make_release.sh` ships these files and
  puts the setup beside the release file.
- **Local change layer, vendor-style** (22 Oct 2026, `local_change.py`,
  `scripts/run_local_change.py`, USER_GUIDE 6.10). It is built from a
  full-resolution processing (4 × 1 looks) and needs no unwrapping:
  - each 12-day pair is averaged over each pixel's "partners", the neighbours
    whose brightness behaves alike, chosen afresh for each ~2-year period
    (`local_change_partner_window_days`, 730);
  - bright, steady points (persistent scatterers, `local_change_ps_dispersion`
    0.15) may count without partners. At these looks speckle alone gives
    about 0.3, so the test passes 0.04% of points at Site D: the path is
    effectively off;
  - the ERA5 weather change is removed, and optionally a DEM-error map's
    residual (`local_change_dem_error`);
  - each step is read as the wrapped phase against the pair's smooth
    surroundings (4 km);
  - only steps whose coherence passes count, so a failed step adds nothing;
  - the rate is counted movement ÷ counted time.

  **Pairs:** the pairs come from the stack route's `fine.int`, or, when there
  are none, are formed from the merged SLCs. The result is identical (a test
  checks it). So a project can merge the phase-linked way: about 241 GB at
  Site D against about 475 GB for the stack route.

  **Options:** `--box` / `--period` sample stable ground or a feature over
  chosen years; `--tag` / `--sigma-km` re-run the steps with another reference
  width into `local_change/variants/`; every comparison is saved to
  `compare.json`.

  **Fixes from the same-day review:** the first agreement with the vendor was
  overstated.
  - The quality gate passed noise at pixels with few partners: a coherence
    over a handful of samples clears 0.3 in 50-75% of pure-noise steps. The
    gate now depends on how many samples the coherence rests on
    (`local_change_min_partners`, `local_change_noise_alpha`).
  - The signal-to-noise mask treated a t statistic as Gaussian. Significance
    is now a t-test with at least `local_change_min_steps` steps, and its
    share counts only pixels that can be significant.
  - The kept-step comparison used different date ranges. `--compare` now uses
    only the dates both products cover.

  **Screening only.** On 22 Oct 2026 it failed its validation against the
  vendor at Site D, against criteria fixed in advance, in both configurations:
  configuration 1 on the kept share and the group differences; configuration 2
  on the kept share, a town control, and stability between a 4 km and an 8 km
  reference. It is not to be quoted as a finding. Its 12-day steps keep the
  farmland drift, and the step gate doesn't remove it: near two towns,
  cropland read about 8-10 mm/yr lower than the few built-up points kept
  (after the fact; size uncertain). Use it to point to places to check by
  other means.

  **Long-pair feature check** (same day, `feature_check.py`,
  `scripts/run_feature_check.py`, USER_GUIDE 6.10): a feature against a ring of
  ground around it, from pairs about 6 months and 1 year apart, where the
  farmland drift has faded; one rate per calendar year with a jackknife error.
  At Site D both town controls passed, but farmland, the spoil patch and the
  pit edge lose coherence within about 6 months, so it couldn't judge them.

  **Fixes from the final review (22 Oct 2026):**
  - The significance map filled cells whose pixels were all non-significant
    from a significant neighbour, so one significant pixel showed as a 3 × 3
    patch. A map cell now shows only where most of its radar pixels are
    significant, and only cells no radar pixel fell in are filled. Site D:
    64,295 significant map cells before, 19,020 after (configuration 2), and
    55,464 before, 19,423 after (configuration 1). Over the map's own area the
    map and the pixels now agree: 1.34% of map cells against 1.44% of the
    pixels that can be significant (configuration 2), 1.92% against 2.04%
    (configuration 1).
    `share_significant` in `local_change.json` (2.04% and 2.76%) counts the
    whole processed window, about three times the map's area, and doesn't
    change.
  - The feature check printed 0.0 mm/yr, or split one pair over two years, for
    a year its usable pairs didn't determine. Such a year now reads unknown.
    Its jackknife uses 40 blocks of start dates (~3 months) instead of 10. On
    6-month pairs with every pair kept, a noise-only year now passes twice its
    error 4-5% of the time, against about 7% with the 22 Oct code. With many
    pairs dropped it still passes too often: about 7% with half kept, 8-10%
    with a quarter, 11-13% with 5-10% kept (the pit edge kept 5%), so there
    that error is a lower bound too. On 1-year pairs it is 10-11%, about a
    point more than before, so that error is a lower bound. A year with a value
    but no error prints "±n/a" and is weakly determined. The record gives
    `pairs_used`, `groups_per_year` and `cover_used_years` (cover by the pairs
    used). The block count is recorded (`jackknife_groups`) and set with
    `--groups`. The 22 Oct errors can be reproduced only with that day's code;
    its verdicts didn't rest on them.
  - The feature check's command line no longer crashes on a separation with no
    pairs.

  Files are written atomically and checked before reuse. **Numbers that
  move:** none in any delivered product (a separate, optional layer).
- **`mintpy_exclude_dates`: dates left out of the series** (24 Oct 2026,
  YYYYMMDD list). MintPy drops every pair touching them before the inversion
  (`mintpy.network.excludeDate`). This is for dates a weather model cannot
  serve: HRRR starts 15 Jul 2016, and MintPy cannot apply weather delays
  without the series' first date. On one project, leaving out its 5 earlier
  dates removed 15 of 831 pairs (checked with MintPy's own network step on
  a copy of the stack). The rest stayed one connected network of 273 dates
  starting 1 Aug 2016. The network check no longer takes these dates for
  orphans. A date it does exclude for having no pair is added to the
  config's list instead of replacing it. run_result.json notes the dates
  left out. **Numbers that move:** none unless set. A project that sets it
  loses those dates from its record.
- **Weather delays from NOAA HRRR on the stack and phase-linked routes**
  (24 Oct 2026, `mintpy_tropo_method: raider` with `weather_model: HRRR` and
  `raider_env`; `weather_raider.py`).
  - **Why:** from 22 Oct the ERA5 queue at Copernicus was full ("Running:
    100 — Queued: 2397"). It refused most requests, and the ones it took
    waited 1 to 4.6 hours. HRRR comes from NOAA's public archive on AWS
    through RAiDER: no account, no queue, about 2 minutes a date, and a 3 km
    grid against ERA5's ~31 km. It covers the continental US from 15 Jul
    2016.
  - **How it reaches MintPy:** before MintPy's troposphere step, the run
    writes `mintpy/inputs/HRRR.h5` in the layout and sign of MintPy's own
    `ERA5.h5`. MintPy is told method `pyaps`, weather model `HRRR`, and its
    own `diff.py` applies the file (checked against MintPy itself). Each
    date's delay is sampled at every pixel's height and turned into the
    line of sight with its incidence angle. The series is named
    `timeseries_HRRR*`, and the report says "weather delays (NOAA HRRR
    weather model)".
  - **Checks:** a date without a delay stops the run unless
    `era5_allow_partial` is set, as with ERA5. A reference date before
    HRRR's archive starts stops the run with that reason, before MintPy
    meets it.
  - **Fetching ahead:** `python scripts/prefetch_weather.py <config>
    [workers]` fills the per-date cache (`tropo_corrections/HRRR/`), which
    the run reuses.
  - **Needs:** Herbie 2026.9.2 or later in the RAiDER environment. Herbie
    2024.8.0 with pandas 3 cannot read HRRR's index files before mid-July
    2018.

  **Numbers that move:** none unless set. A project switched to HRRR gets
  a different weather correction.
- **`stack_unw_ncorrlooks`** (22 Oct 2026, snaphu2 engine only): how many looks
  lie behind each coherence value handed to snaphu (`NCORRLOOKS`). The
  default, 0, keeps today's range × azimuth looks ÷ 0.64, which suits a
  boxcar estimate. An adaptively filtered stack (partner-pixel averaging)
  estimates coherence over many more samples. Left at the boxcar value,
  snaphu reads its honest low coherence as noise. **Numbers that move:**
  none unless set.
- **Like-for-like comparison with a vendor's point exports** (21 Oct 2026,
  `scripts/compare_vendor_points.py`, `vendor_points.py`).
  - **What the export counts:** the vendor's point CSV counts only the 12-day
    steps the vendor accepted for each point. A blank step adds nothing, and
    the series stays flat across it. Its rate is displacement ÷ accepted time
    (checked on 11,024 of 11,177 points in five exports).
  - **What the script does:** it puts our series at the same points through
    the same bookkeeping, over the dates both products have. It reports each
    group and the group-to-group differences. Those are not datum-free: each
    group keeps its own steps, so moving the reference (`--reference-box`)
    moved them by up to 0.27 in/yr at Site D.
  - **Usage:** `python scripts/compare_vendor_points.py <config>
    [name=]<export.csv> ...`. It writes `analysis/vendor_compare/` in the
    project, outside `deliverables/`, which the client packager copies.

  **Numbers that move:** none (read-only).
- **The results map takes any dates** (19 Oct 2026, the first step of
  making the results map the home base for results).
  - **Dates:** a bar at the top picks the window: presets, two images, or a
    drag across the image strip. The map, every figure, chart and download
    follow it.
  - **Figure kinds:** the whole record shows the report's stored figures.
    Any other window is computed in the browser from each point's own series
    and labelled a look. It reproduces `insar results` point by point
    (Site D, 2021: rates within 0.004 mm/yr). Only the single-point noise
    is estimated, from a sample of the report's farmland (within 1.5% there).
  - **Charts and downloads:** every chart has PNG, SVG, CSV and a table view.
    Downloads carry the report's stamp lines, the numbers hash, the build and
    DRAFT until sign-off.
  - **New on the map:** points and polygons together.
    - Polygons stay on the map beside the points you click.
    - Several can be drawn and named (**Draw a polygon** / **Draw a
      rectangle**).
    - Any mix of up to four points and polygons can be pinned to one compare
      chart, each in its own colour.
    - A polygon panel adds a histogram against farmland and yearly rates.
    - The page address keeps the view: dates, layer, polygons, pins and the
      selection.
  - **Updates and a running map:** a results map already running keeps
    serving the files it started with. A dashboard left open across this
    update served the new page with its old code, and the page came up
    unstyled.
  - **Window exports:** the polygon CSV and KMZ export follow the window.
  - **Under the hood:** the browser arithmetic (`viewer_assets/viewer_core.js`)
    is tested under node against the Python reference (`viewer_bundle.py`),
    which is tested against `results.analyse`. The viewer table
    (`field_points.npz`, version 2) adds each point's own-cell series, its
    coherence class, the report's farmland flag and a farmland sample.
    **Numbers that move: none.** Projects need **Products only** (or
    `python -m insar_pipeline.field_points <config>`) for the date windows;
    an older table shows the whole record and says so.
- **The results map serves its own files:**
  - Leaflet and Leaflet.draw are vendored (checked against cdnjs's published
    hashes), with no CDN and no Google Fonts;
  - an exact list of files is served under a Content-Security-Policy that
    runs scripts only from the server;
  - `make_release.sh` ships the files, and `insar doctor` checks they are
    there.
  - The code hash now covers the pages and the viewer scripts, so every hash
    changes once.
- `install/envs/insar.yml` pins Node.js 22 (the viewer's tests). CI parses
  the viewer's scripts and runs their tests.
- **The build label is in more outputs** (OUT-01, BLD-01):
  - the client KMZ and the results-map exports;
  - the parquet point table;
  - the JSON records: summary, GNSS tie, quality assessment, events;
  - CSVs that already had `#` lines (a `# software:` line);
  - every run log (first line);
  - run_result.json.

  The release gate's numbers hash ignores it, so a code change alone never
  voids a sign-off.
- **Run records:**
  - `run_result.json` keeps its history in `run_result_history.jsonl` (ERR-04),
    and counts the run's log warnings (ERR-02); the dashboard shows the count.
  - provenance.json keeps every build that worked on the project (`builds`,
    OUT-02).
  - An append-only registry, `<projects root>/run_history.jsonl`, lists every
    run with its build, so the outputs of a given build can be found (OUT-05).
- **Release tarballs:** a `.sha256` beside each one (BLD-05).
- **Tests:** the first tests of `gis_products` (TST-03).
- `EXTERNAL_SERVICES.md`: every outside service the pipeline contacts, what
  it sends, how it authenticates, and its approval status (readiness review
  NET-02).
- `install/README.md`: installing from the GitHub repository, with the SSH
  key made inside Ubuntu.

### Fixed
- **Weather dates are handled by the run, never by a config edit** (27 Oct
  2026, Kyle). Site E descending stopped after its inversion: "HRRR weather
  delays are missing for 3 of 292 dates". None was really unavailable. Runs
  stopped part-way had left cut-off HRRR downloads in weather_files/, which
  Herbie reused on every attempt ("cannot rename 'gh'", "'str' object has no
  attribute '_coord_names'"); a fresh download gave a cube for each.
  - `tropo_disp.delay_cube_for_date` tries a failed date once more from a
    clean download cache. On Site E it recovered all three dates.
  - Before the inversion, `time_series.exclude_weather_unavailable` fetches
    every stack date's weather (`weather_raider.check_dates`). It records the
    dates still without any, with the reason, in
    tropo_corrections/<MODEL>/unavailable_dates.json, and leaves them out of
    the series (excludeDate, modify_network re-run). The record is rewritten
    by every run, so a date that comes back is taken back. A run note lists
    the dates left out.
  - The coverage error's advice no longer says to edit mintpy_exclude_dates.
  - **Numbers that move:** none where every date has weather; a date without
    any is left out instead of stopping the run.
- **NISAR on the Sentinel-1 products' 15 m grid** (27 Oct 2026, Kyle: the
  NISAR maps looked coarser). The GUNW crops were put on an 80 m grid (the
  product's posting); they now go on the 15 m grid every Sentinel-1 product
  has, nearest-neighbour as MintPy fills Sentinel-1's (`nisar_posting_m`
  default 15). Maps, KMZ points (~105 m), the cell layer, figures and the
  viewer's cells then match the Sentinel-1 run's cell for cell. Each value
  still averages ~80 m, and the products say so ("Points are 15 m apart; each
  value is an average over about 80 m (NISAR's product, ...)",
  `resolution.py`). The KMZ's and viewer's "every measured 15 m cell" labels
  now use the grid's real posting. **Numbers that move:** a NISAR project on
  its next full run (new grid; the crops are read again).
- **Results viewer: date windows were off on Site E and Site E
  NISAR** (26 Oct 2026). The load check fitted each point's own series on
  the pipeline's time axis over every date. MintPy fitted the delivered
  velocity on its own decimal years, without the dates it judged too noisy
  (mintpy/exclude_date.txt: 20250510 at Site E). Over 10 months (NISAR)
  the two axes differ by 0.1% of a rate. So the check missed by 0.054 and
  0.13 mm/yr against its 0.01 limit. It now fits the delivered way
  (`field_points.fit_axis`; every project now passes within 0.00006). A table
  built earlier gets the axis on load (`add_fit_axis`) and needs no rebuild.
  Window figures are unchanged; they stay results.analyse's.
  **Numbers that move:** none.
- **Client KMZ: points off when the file opens, each balloon names its own
  point; the results viewer exports every 15 m cell of a chosen site and
  dates** (26 Oct 2026, a coworker's review of the Site E KMZ).
  - The ~100 m measurement points (9,454 at Site E, against 317,414
    measured 15 m cells) loaded with the file and made Google Earth slow. They
    are now off until ticked, as are the feature points. The 15 m cell layer
    shows on open, as before.
  - Every point balloon was headed with its report area ("Full Satellite
    Imaging"), so a click read as the area's overview, though the chart was the
    point's own. It is now headed "Point 40.12345 N, 100.12345 W", with "in
    <area>" beneath (`client_kmz.POINT_TITLE`).
  - The whole field at 15 m would be ~66 MB at Site E and ~560 MB at
    Site B, more than Google Earth Pro opens in one file. So the results
    viewer's polygon panel gains "KMZ, every 15 m cell": every measured cell
    inside the polygon over the dates chosen in the viewer. It is computed from
    the products by the client KMZ's own `field_points.gather`, up to 40,000
    cells (~9 km2, ~8 MB) (`results_viewer._full_table`). At Site E a
    ~1 km2 site gave 3,090 cells (against 64 points) in 0.5 MB and 3 s. Its
    cells that the ~100 m table also holds read identically.
  - **Numbers that move:** none.
- **The NISAR route gets the full product set, named and credited as NISAR**
  (26 Oct 2026, Kyle: the Site E NISAR comparison needs "all of the files
  and maps that we would normally get").
  - The products stage now runs the QA, figures and client report on `nisar`
    too (pipeline.py), so the run builds the report, the KMZ and the viewer.
  - Every route said "Sentinel-1" and carried the Copernicus Sentinel notice,
    so the NISAR run's GeoTIFFs and points_README credited Sentinel data they
    do not contain (`deliverables.source_line` even added the notice when it
    was missing). The satellite now comes from `config.sensor_name`. On NISAR
    the credit is NASA-ISRO via ASF DAAC (`credits.py`), and every client page,
    the KMZ included, is flagged "Comparison - NISAR, not validated", with a
    NISAR methods paragraph and limitation.
  - On routes already on the map grid (HyP3, NISAR), `geocode_extra_layers`
    left the extra layers "where they are", but every reader opens
    geo/geo_<name>, so none reached the products. They are now copied to those
    names. On NISAR, MintPy's temporal coherence is not: it is 1.0 by
    construction on a chain (`time_series.extra_layers_for`).
  - A project switched from purpose: experiment loses its stale
    EXPERIMENT_NOT_FOR_CLIENTS.txt when it builds its report.
  - Checked with Products only on a copy of Site E NISAR: the same files
    as the Sentinel-1 project except quality_unwrapping_consistency.tif and the two
    example-interferogram figures, which need Sentinel-1 stack files.
  - **Numbers that move:** none; files are added. HyP3 projects gain
    dem_difference_m.tif, quality_position_uncertainty_m.tif and
    quality_unwrapping_consistency.tif on their next products run.
- **A NISAR pair the product's own mask rejects no longer empties the whole
  stack** (26 Oct 2026, `nisar_backend.valid_cells`, `product_mask_keep`,
  `crop_validity`). At Site E the 2026-09-07 acquisition held no valid
  sample over the box: subswath digit 0 in the GUNW mask, coherence NaN, flat
  phase. Its phase was still finite in one connected component, so both pairs
  touching it passed the screen at 100% valid. MintPy's prep_nisar keeps only
  cells valid in EVERY pair, so it masked every cell. MintPy then stopped at
  plot_network with NaN coherence in all 28 pairs. A pair's valid share now
  also requires the product mask, decoded as prep_nisar decodes it. Crops kept
  from an earlier run are re-screened from their stored layers, without being
  read again. Site E keeps 24 pairs, 25 dates, 2025-10-30 to 2026-08-26
  (the chain breaks at 7 Sep; the longest piece is kept). Checked on a copy:
  MintPy loads it and gets through modify_network. **Numbers that move:**
  none for a stack without such a pair.
- **HRRR weather delays work on the NISAR route** (26 Oct 2026,
  `nisar_backend.acquisition_times`, `weather_raider.acquisition_time`). The
  HRRR step took the time of the pass from Sentinel-1 SAFE names only, so a
  NISAR run would have stopped at its weather step ("no SAFE names"). It now
  reads the NISAR crops' identification (zero-Doppler start and end of each
  date; Site E T019A: 12:11:15-12:11:44 UTC). RAiDER's run-config says
  `look_dir: left` on this route; zenith delays, as used here, do not read it.
  **Numbers that move:** none (no NISAR run had HRRR before).
- **A date missing from NOAA's HRRR archive is named as such, with the
  remedy** (26 Oct 2026, `corrections.verify_era5_coverage`,
  `weather_raider.archive_gaps`, `advice`). Site B stopped at the
  troposphere step because the archive holds no HRRR analysis for its
  2016-08-08 pass (23Z-01Z missing on AWS and Google). The message gave the
  ERA5 advice ("published about 5 days after real time ... the Copernicus
  queue ... wait and re-run"), which cannot help with a gap in the archive,
  and the app cut it off at 300 characters ("... or."). The run now reads
  RAiDER's log for each missing date, says when the archive itself lacks
  the file, and names the dates to put in `mintpy_exclude_dates`. The app
  shows the cause and the remedy separately (category "weather"). HRRR
  forecasts from the cycle before were tried as a stand-in and rejected: that
  evening a 1 h forecast was off from the analysis by twice the field it
  would correct (4.1 vs 2.0 mm spatial std). **Numbers that move:** none.
- **An empty or partial weather file no longer breaks every MintPy re-run**
  (24 Oct 2026, `time_series.retire_partial_weather_file`). MintPy applies
  any `inputs/ERA5.h5` that fits the grid, whatever dates it holds. A run
  that reached MintPy with no ERA5 files on disk wrote one holding no
  dates. Every re-run after it failed in MintPy's `diff.py` with
  "IndexError: list index out of range". A partial file corrected only its
  own dates. Before the troposphere step, a weather file missing any of the
  series' dates is now moved to `mintpy/superseded/run_<stamp>/inputs/`
  and built again. **Numbers that move:** none.
- **A stopped run also stops RAiDER** (24 Oct 2026, `tropo_disp`). RAiDER
  runs in its own process group, which the run's teardown kills. It was a
  plain child process and kept running after a stop. **Numbers that
  move:** none.
- **A full ERA5 queue no longer fails every weather file** (23 Oct 2026,
  `stack_backend.start_era5_prefetch`). While its ERA5 queue is full, the
  CDS turns new requests away ("Maximum number queued requests for this
  dataset has been reached"), then accepts some again minutes later. The
  ERA5 download counted each refusal as that date's failure and moved on.
  On a new project it "failed" 12 dates in its first minute, then had its 4
  accepted requests waiting in the queue, and the refused dates were never
  asked for again. A refusal now means waiting and asking again for the same
  date (after 1 minute, doubling to 15). `era5_prefetch.json` counts the
  refusals apart from the failures (`queue_full`), and the dashboard says
  "CDS queue full". Any other failure still counts at once. **Numbers that
  move:** none.
- **A finished run no longer stays alive while ERA5 downloads wait** (22 Oct
  2026, `pipeline.stop_era5_prefetch`, `stack_backend.start_era5_prefetch`).
  A phase-linked run stopped with `--to-step merge_reference_secondary_slc`
  wrote "done" in run_result.json and then stayed alive for ten minutes,
  until it was stopped by hand. Three of its weather downloads were still
  waiting in the Copernicus (CDS) queue, and Python waits for download
  threads like these before a process exits. Anything waiting for the run to
  end waited too, and the manual stop logged "run failed" after a run that
  had succeeded. Now:
  - a waiting download no longer holds the process;
  - the end of every run, however it ends, cancels its project's ERA5
    download. `mintpy/era5_prefetch.json` then says "cancelled", with what
    was fetched, what failed and what was left; MintPy fetches what is
    missing, as before;
  - a `--to-step` run starts no ERA5 download. The run that carries on
    ('full' or 'Phase linking onward') starts its own;
  - `era5_prefetch.json` is rewritten as each file starts and ends. It
    gives each failure's reason (without the per-request web address), a
    count per reason, and which files are in progress and since when. It
    used to be written only at the end, so this run's 265 failures of 270
    left no reason.

  Those 265 were refused by the CDS: "Maximum number queued requests for
  this dataset has been reached. Please configure your scripts to request
  data sequentially". Its queue for ERA5 was full. The first request was
  refused too, while nothing else of ours was queued, and an earlier run
  that night met the same refusal for 46 of 48 files. The few requests it
  accepted waited an hour or more. The credentials were not the cause:
  refused credentials answer 401 or 403. **Numbers that move:** none.
- **A NASA search timeout no longer stops a download as a "refused login"**
  (22 Oct 2026, `data_acquisition.auth_refused`). The 401/403 check matched
  "line 403" in a timeout's traceback. It stopped a 274-date download at date
  44 with "the password in ~/.netrc is wrong", though the login worked.
  Traceback line numbers are now ignored, and timeouts are treated as
  temporary. **Numbers that move:** none.
- **A slow NASA catalogue no longer kills a run before it starts** (22 Oct
  2026, `data_acquisition.cmr_search`). asf_search gives the catalogue 30 s.
  A 10-year burst search over Site D timed out twice in two minutes, each time
  before anything was downloaded, while small queries answered in 6 s. The
  scene and burst searches now allow 120 s and retry a timeout or a 502-504
  three times (after 30, 60 and 120 s); any other error stops at once.
  **Numbers that move:** none.
- **The stack-route disk plan counts real merged SLCs** (22 Oct 2026,
  `stack_backend.plan_stack`). With `stack_virtual_merge: "False"`, the merge
  writes a full-resolution SLC per date beside its aligned bursts, about
  0.84 GB per date per 3 bursts. The plan counted them only on the
  phase-linked route, whose merge swaps the two. A Site D full-record
  stack-route run planned 303 GB and would have needed about 475 GB at the
  merge, against a 399 GB budget. The plan now shows "end of the merge" and
  includes the merged SLCs in the peak and the steady state, so such a run is
  refused before it starts. **Numbers that move:** none.
- **An area spanning one burst per sub-swath is refused before anything is
  downloaded** (21 Oct 2026, `pipeline.burst_gate`). topsStack cannot merge a
  single-burst sub-swath, and ESD has no burst overlaps to measure. A 10 × 8 km
  experiment crop failed twice: first at ESD (run_05, every command wrote
  nothing), then, coregistered by geometry, at the merge (run_07, "min() arg is
  an empty sequence"). The run plan now says so and the run stops, naming the
  fix: widen the box north-south to two bursts and limit the analysis
  afterwards. `--force` overrides. When the download's own record shows one
  burst per sub-swath, coregistration also falls back to geometry, and
  provenance records the coregistration actually used. **Numbers that move:**
  none.
- **Woods read as rising and are now left out of the products** (21 Oct
  2026, `mask_wooded`, on by default). C-band sees the canopy, and woods
  carry less of the farmland bias that the relative figures remove, so a
  wooded cell's relative value read upward: one project's woods showed +2.5 in
  median over ten years, with 85% of wooded cells positive. With a land-cover
  map, wooded cells now drop out of every delivered map, raster, point file,
  KMZ, viewer layer and area figure; `client_report.json` records how many
  (`wooded_masked`). The wording for wooded areas now says the woods are left
  out. The quality assessment still uses every cell. `mask_wooded: false`
  restores the old behaviour. **Numbers that move:** coverage and point
  counts fall by the wooded share (that project: 30% of measured cells). Area
  figures shift toward the open ground (its study-area median relative
  rate goes from +2.5 to +1.45 mm/yr). **Products only** brings a project up
  to date.
- **A coherence class too small to measure had no line-of-sight change.**
  Its rate borrows the nearest measured class's centre, but
  `field_points._frame` gave it no frame. One project's delivered
  `line_of_sight_cumulative_displacement_*.tif` missed 132 cells of coherence
  0.90-0.98 that its velocity raster had; their KMZ balloons and viewer
  points had no change and no chart. `results.analyse`'s "change between
  window ends" was blank for those cells too. A cell below the lowest class
  now has no change, as it has no rate. **Numbers that move:** that project
  gains those 132 cells (every other cell is unchanged, to 0.00001 mm); Site D
  has none. **Products only** brings a project up to date; one without a
  borrowing class is unchanged.
- `insar doctor` allowed the test suite 10 minutes, and it takes about 9 on a
  fast PC, so an install on a slower PC could end "not ready" with nothing
  wrong. The limit is now 30 minutes.
- **Configuration is checked by type** (HND-03). True/false settings take
  true/false, yes/no, on/off or 1/0; the text "no" used to read as true.
  Counts must be whole numbers, and numbers must not be quoted text. A YAML
  syntax error names the line. `stack_filter_strength` must be 0-1, and
  `gnss_tie_radius_km` 0-100. All existing configs load unchanged.
- **A settings file that does not parse** is reported by `insar doctor` and
  is never overwritten; it used to be replaced, losing every setting in it.
- **Spreadsheet formula injection everywhere** (SEC-05). Every CSV and
  workbook writer neutralises text that starts like a formula (a leading
  apostrophe, the OWASP convention): mine and panel names from shapefile
  attributes, area and polygon names, ground-truth point names. Numbers are
  untouched.
- **A skipped unwrapping-error correction is no longer silent.** When the
  config asks for `bridging+phase_closure` and MintPy's phase closure finds
  no region unwrapped in every kept pair, it corrects nothing, and the
  inversion uses the bridging dataset. The skip was in `provenance.json` and
  one log warning only. Now it is:
  - flagged in the QA's new `unwrap_correction` block, with MintPy's reason;
  - a send-rule note, so the release list shows it (a note, not a hold);
  - in the QA text report;
  - in the run's notes in `run_result.json`.

  The client page does not change. Numbers that move: none. On a rebuilt
  report the send rule's reasons gain the note, so the numbers hash changes
  for projects where closure was skipped.
- **Report parts that fail are no longer silent** (ERR-01). A part of the
  report that could not be worked out is listed under "Left out of this
  build" and on the release gate. Two of them are hard blockers: the
  single-point uncertainty and the send rule re-evaluated with the GNSS tie.
  A failed geocode of the extra quality layers appears in the run's notes.
- **Shell commands** quote every path and check conda environment names
  (SEC-02); a project folder with a space broke phase linking.
- **Outline files** (shapefile zip, KML/KMZ, GeoJSON) are checked before
  parsing (SEC-03/04). The content must match the extension; there are
  limits on size (300 MB), unzipped size (1 GB), zip members (1,000),
  features (200,000) and vertices (10 million); KML with a DOCTYPE or ENTITY
  is refused. A broken file gives a plain message instead of a Python error.
- **The local map builder and results map** require a per-session token on
  every change request (NET-06). A page left open across a restart of the
  server must be reloaded.
- **The Earthdata cookie jar** is private to the user
  (`~/.cache/insar_pipeline/earthdata/`); it was a fixed file in shared
  /tmp (DAT-03). `scripts/run_pipeline.py` no longer prints the Earthdata
  user name (DAT-07).
- **Phase-linking resume** also stops when the MiaplPy build has changed, or
  when linked dates have left the SLC stack (OUT-03). Existing projects
  match as before.
- **The two map rasters of a pair** are each tried on their own; one
  unreadable input used to cost the other (TST-03).
- **Two projects in one process** could share the comparison-ground context
  (QUA-05).
- **A dropped date's reason on the client page** printed the failing tool's
  own error text ("burst assembly failed: exit 1: ERROR conda.cli..."). The
  page now gives the plain reason; the record keeps the detail. Rebuild a
  report (Products only) to pick it up.

### Numbers that move
None. The report gains a "Left out of this build" list only when a part
fails.

## [2026.10.16] - 2026-10-16

The first version under git. The fixes come from the technical
readiness review of 16 Oct 2026.

### Fixed
- **Release tarball:** `install/make_release.sh` left `results_viewer.html`
  out, so the results map (`m`, `insar viewer`) failed on an installed copy.
  The release test now checks that every page the code serves ships.
- **Release gate fails closed:** a check that raised an error used to count
  as "nothing to report". For example, a failure in the reissue check or the
  GNSS-tie staleness check made that hard blocker vanish. Now the check adds
  a line that says it could not run, as hard as the check it stands for, and
  the report page gets plain wording for it.
- **"Hold for specialist" is enforced:** it was a reading item, and any
  reviewer name made a held report FINAL. It is now a hard blocker until the
  sign-off names the specialist who read the hold reasons
  (`--specialist "Name"`, or the dashboard's Specialist field).
  `deliverables/review.json` records the name with those reasons, and new
  reasons hold the report again.
- **Spreadsheet formula injection:** a polygon name typed with a line break
  began a CSV line that a spreadsheet ran as a formula. Names now reach the
  results-viewer CSV and KMZ on one line, and report-area names with a line
  break or a leading `=` `+` `-` `@` are refused.
- **Provenance:** `provenance.json` goes to the client in the package. It
  named the operator by their OS login and computer name; it now names the
  preparer from the settings, and paths show the home folder as `~`.
- **Tests:** no longer write into the real notifications log or raise
  desktop toasts (`tests/conftest.py`; the log path follows
  `INSAR_NOTIFY_LOG`). A guard test also ignores the version number now,
  not only the code hash, when it compares products with an older build's.

### Added
- The git repository files: README section "This repository", LICENSE
  (placeholder until Legal's wording), this changelog, CONTRIBUTING.md,
  `.gitignore`, the client-data guard (`scripts/check_repo_clean.sh`, run
  by `.githooks/pre-commit` and the GitHub checks), and issue and pull-request
  templates.
- `configs/config_example_pl.yaml`: a worked example on a public site
  (Katy / Houston SW, Texas), written by the map builder's own code.

### Changed
- Client project configs, client report areas, the golden client numbers and
  the engineering log (HANDOFF.md) are not in the repository. They stay on
  the operator's machine, and tests that need them skip on a fresh clone.
- The first client projects are named Site A to Site D in code, tests and
  docs, and their report areas Area A1 to A3. Exact client coordinates in
  test fixtures were moved by 0.15 to 3 km. The tests that read the real
  client projects and records are in `tests/test_local_*.py`, which is
  ignored by git and runs only on the machine that has them.
- `configs/` and `context/` take only the examples and the public layers:
  any other project config or outline put there is ignored.
- The internal release tarball no longer ships HANDOFF.md, and now ships
  LICENSE, CHANGELOG.md, CONTRIBUTING.md and VALIDATION.md.

### Numbers that move
None. The gate and sign-off changes decide which reports can go FINAL, not
any figure.

## 2026.10.5 and the unversioned builds to 15 Oct 2026

Before version control the version number was not bumped, so these builds
are told apart only by the code hash in their outputs. The detail is in the
engineering log, which is kept on the developer's machine.

- **15 Oct:** wooded cells (ESA WorldCover) are left out of the ordinary
  ground that figures are compared with (`landcover_label_wooded`, on by
  default). Comparison figures move on rebuilt projects.
- **13 Oct:**
  - The client KMZ shows every measurement point.
  - `insar storage` gives a disk overview, and "keep results only" runs
    after a FINAL package.
  - Dates that fail on a known ASF catalogue defect are accepted
    automatically.
- **8-12 Oct:** validation against independent ground truth
  (`VALIDATION.md`):
  - the ground-truth and validation harness, and experiment mode;
  - the NISAR GUNW route;
  - the pre-unwrap fast-motion flag;
  - ascending + descending decomposition.
  
  Slow and moderate subsidence passes. Fast longwall subsidence from C-band
  fails.
- **8 Oct:** measurement points over the whole field in the KMZ,
  line-of-sight rasters, and the results map (`insar viewer`).
- **5-7 Oct:** fixes from the full review:
  - every client report is a DRAFT until a named reviewer signs off its
    numbers, and the client package is built only when it is FINAL;
  - detection limit at 80% power;
  - totals are measured change;
  - version and code hash in the outputs.
  
  Rebuilt reports move.

## 2026.9.28 - 2026-09-28

- Packaged for coworkers: `install/install.sh` with pinned conda
  environments, `settings.py` for machine paths, the `insar` command and
  dashboard, and the patched MiaplPy build.
- The phase-linked (MiaplPy) route is the official one. OPERA DISP-S1 is
  for screening and cross-checks.
