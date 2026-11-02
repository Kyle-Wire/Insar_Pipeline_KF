# Technical reference

How the pipeline works inside, for whoever maintains it or needs to check a
figure in depth: the commands behind the screens, the processing routes and
which of them gives client numbers, every file a run produces, the settings,
how measurement and uncertainty are worked out, and the design decisions.

New users start with [README.md](README.md), install with
[install/README.md](install/README.md) and work from
[USER_GUIDE.md](USER_GUIDE.md). What has been validated is in
[VALIDATION.md](VALIDATION.md); what the program cannot yet do is in
[LIMITATIONS.md](LIMITATIONS.md).

## Commands behind the screens

On a machine installed with `install/install.sh`:

    insar doctor        # must end in "Ready."
    insar               # the dashboard

In the dashboard: `b` opens the map builder (draw or upload the area and the
mine outlines, check the archive, create the project); Enter opens a
project; `s` starts the recommended run after the preflight checks; `o` opens
the client report; `r` signs it off; `k` builds the client package; `?` opens
the user guide. Runs are detached: closing the dashboard never stops one.

Other commands (`insar help` lists them):

    insar builder                         # the map builder on its own
    insar cleanup <config.yaml>           # free disk on a finished project (dry run unless --delete)
    insar results <config.yaml> --area "Area A2" --from 2020-01-01 --to 2024-12-31 --export
    insar injection-test <config.yaml>    # plant known movements, measure how well they come back
    insar cross-check <config.yaml>       # compare the record with OPERA DISP-S1
    insar --version                       # version and code hash
    insar --menu                          # the old numbered menu

The same run from a terminal (same project lock, run record and exit codes):

    conda activate insar && cd ~
    python -m insar_pipeline.pipeline <config.yaml> [--resume | --start products|mintpy|miaplpy|unwrap] [--plan-only]

The release gate and the client package:

    python -m insar_pipeline.release <config.yaml>             # what blocks release
    python -m insar_pipeline.release <config.yaml> --sign "Name" --features 1,2
    python -m insar_pipeline.package <config.yaml>             # only when FINAL

---

## Routes: which numbers are official

| Route | `backend` | Status (`config.ROUTE_TAG`) |
|---|---|---|
| Phase-linked: topsStack coregistration, patched MiaplPy, snaphu 2, MintPy | `miaplpy` | **official**: the only route whose numbers go to clients |
| OPERA DISP-S1 ready-made time series | `disp` | **screening**: where to look. No client report or KMZ unless `disp_client_report: true`, and the release gate blocks a screening report as a hard blocker. The builder gives it its own project, `<name>_disp_screening`. |
| Classic topsStack short-baseline stack | `stack` | comparison only |
| topsApp per pair, ASF HyP3, NISAR GUNW | `local`, `hyp3`, `nisar` | comparison or testing only |

Any route other than `miaplpy` logs a warning at the start of the run that
its numbers are for comparison or screening.

**Why phase linking.** Farmland has no dominant reflector per cell. A
network of short pairs over it carries a phase bias that appears as a
fictitious rate (measured at Site A: −12 mm/yr on coherence 0.4 to 0.6
against high-coherence ground). Phase linking estimates one consistent phase
history per cell from the whole coherence matrix instead.

**How the phase-linked route runs** (`miaplpy_backend.py`):

1. topsStack run_01 to run_12: coregistration and a real merge of the
   full-resolution images (MiaplPy reads them).
2. MiaplPy `load_data`, `phase_linking` (sequential EMI, mini-stacks of 10
   images, 15 x 15 pixel windows) and `concatenate_patches`, on a crop of the
   stack aligned to the 11 x 3 look grid. MiaplPy runs in its own conda
   environment and must be **the patched build** (`0.2.1+overlapdatum`, see
   `patches/miaplpy/README.md`): upstream MiaplPy leaves a random datum step
   at every mini-stack boundary on farmland. The patch is checked before every
   run with the project's own mini-stack size and time lag.
3. The linked phases are exported as interferograms on the stack's grid, a
   network with `miaplpy_network_connections` (default 3) connections per
   date, into `stack/merged/interferograms/`.
4. The stack route's own unwrapper (cropped snaphu 2.0.7, DEFO cost, MST
   initialisation, outputs verified) and MintPy chain (reference check,
   connectivity, bridging, ERA5, the deramp fitted outside the mines and a
   2 km buffer, DEM error, geocoding). MintPy writes to `<project>/mintpy/`,
   the same place and grid as the classic route.

Every step leaves a completion marker, and a resume skips only marked work.
Measured on Site A (278 dates, 23-24 Sep 2026): downloads 4 h 23 min,
coregistration 1 h 57 min, merge 16 min, load 17 min, phase linking 21 h 09 min,
export 6 min.

**What the classic stack route changes.** ISCE's topsStack hardcodes
`noMCF='False'` (`topsStack/Stack.py:11-12`), which routes `unwrap.py` to
`costMode='SMOOTH', initMethod='MCF', initOnly=True`: snaphu initialises
and then stops. `stack_backend.patch_configs()` rewrites every generated
`stack/configs/config_igram_unw_*` to `nomcf : True` (full DEFO cost mode) and
`defomax : 4.0`, and verifies every config on disk before anything runs.
The phase-linked route uses the same runner and unwrapper.

---

## What a run produces

In `<project>/deliverables/` unless noted. Units follow `display_units`
(inches by default), except the point tables, which are always mm.

**For the client** (copied into `releases/<stamp>/client/` by the package):

- `client_report.html` / `.json`: the areas in `report_areas`, as VERTICAL
  movement relative to ordinary ground plus the regional GNSS rate, each
  with a 95% range, a detection limit (80% power: "would be detected 4 times
  in 5"), signal ÷ threshold, the measured total over the record, the recent
  rate and one plain reading. A DRAFT watermark and the list of blockers
  until the report is signed off. `USER_GUIDE.md` section 6 explains every
  number.
- `client_area_summaries.pdf`: one page per area.
- `<Title>_ground_movement[_DRAFT].kmz`: every number read from
  `client_report.json`; `_SCREENING_` in the name on the DISP route.
- Results workbooks from `insar results ... --export`.

**Data** (`releases/<stamp>/data/`): GeoTIFFs in the report's terms
(`vertical_movement_in_per_yr.tif`, its 1-sigma uncertainty,
`vertical_change_at_average_rate_in_<d0>_<d1>.tif` = each cell's average rate ×
years - not the measured change, which the report gives for areas - and
`rate_signal_to_noise.tif`, rewritten by the client report from its own
farmland so a cell stands out by one rule in every product), each
tagged with quantity, units, sign, reference, dates, route, posting and
resolution, the Copernicus notice and the code version (check with
`gdalinfo`), DEFLATE-compressed with overviews. New quality layers, each
written only when its input exists and none of them masking a figure:
`quality_unwrapping_consistency.tif` (MintPy's network coherence, the second
quality check), `quality_position_uncertainty_m.tif`,
`quality_too_fast_to_measure.tif` (neighbouring cells differing by more than
a quarter wavelength per revisit, about 422 mm/yr at 12 days) and
`disturbed_ground_runs.tif` (written only when the layer is marked usable; it
is not on phase-linked records today, where it flags most reliable ground, and
its effect is recorded in the QA as `coverage.disturbed`).
`los_velocity_cell_noise_1sigma_<unit>.tif` is the per-cell noise to quote
for one cell. `quality_temporal_coherence.tif` is deliberately unmasked.

**Points (compact by default).** `points_attributes.gpkg` (EPSG:4326
cell-centre points, about 20 columns, EGMS-style names in mm and mm/yr,
`in_aoi`, `radar_pixel_id`), `points_timeseries.parquet` (GeoParquet, one row
per point, one column per date) and `points_README.txt`. Geocoding copies each
radar pixel to about 7.5 map cells: count over distinct `radar_pixel_id`, not
rows. A products run first geocodes any of MintPy's extra layers that are
missing (unwrapping consistency, DEM error, disturbed ground;
`pipeline.geocode_missing_layers`), so the point table carries the second-gate
columns on every route that makes them. The old per-point CSV (`points_full_csv: true`) and MintPy's
time-series KMZ (`full_timeseries_kmz: true`) are made only when asked;
together they were about 50 GB per project. `measurement_points_egms.gpkg`
files made before 5 Oct 2026 put UTM metres in a latitude/longitude file and
land off the globe: do not hand them out.

**Records** (`technical/` in a package; never for the client):

- `<project>/quality_assessment.json`: the six checks, each with
  `on_farmland_expected` and a `plain` meaning, and the old verdict string.
  New: `send_rule` (send / send with notes / hold for specialist, with
  reasons), `coverage.noise_level`, `coverage.reliable_percent`,
  `coverage.second_gate`, `coverage.disturbed`, `ramp_sensitivity`,
  `matching_sensitivity`, `geolocation` (against mapped water),
  `dates` (available, processed, dropped with reasons), `far_field`,
  `plausibility`, `dem_error`, `area_testability`, `resolution_check`,
  `reference_distance`, `phase_linking`.
- `summary.json`: `deforming_area` is now relative to farmland of the same
  coherence; the old absolute count is `raw_los_rate_below_threshold`; rate
  keys are `raw_los_velocity_vs_reference_*`; `data_quality` is the QA
  verdict; `independent_radar_measurements` beside the cell count.
- `gnss_tie.json` (written on every run, with a plain reason when no tie is
  possible), `uncertainty_model.json` (per coherence class, now with the
  farmland cell spread and the cell null), `provenance.json` with
  `config_snapshot.yaml` (the resolved config and its sha256, the code version
  and hash, the MiaplPy build and patch check, the real network and the
  stages that followed), `technical/cross_check.json`,
  `technical/injection_recovery.json` (the QA, the report and the release
  gate read these two only while their option is on and the record is no
  older than the velocity product).
- `<project>/mintpy/run_products.json`: which series, velocity and
  corrections a successful MintPy run produced. A re-run moves the previous
  run's products into `mintpy/superseded/run_<stamp>/` (never deletes; a
  stack reload moves the previous radar velocity and series there too,
  several GB). `mintpy/run_incomplete.json` is written when MintPy starts and
  removed only when it finishes: while it is there (a run under way, failed
  or stopped, even in the plot and KMZ tail), the deliverables refuse, the
  release is blocked and Resume offers MintPy again (or an earlier start
  when pairs are still to be unwrapped).
- `<project>/run_result.json`: how the last run ended, in plain words, with
  what to do next (`USER_GUIDE.md` section 8), the build that ran it, and a
  count of the log's warnings. Earlier ones are kept in
  `<project>/run_result_history.jsonl`, one per line.
- `provenance.json` `builds`: every build that has worked on the project
  (processing, products, report), appended, never replaced.
- `<projects root>/run_history.jsonl`: one line per run of any project (folder
  name, kind, start and end, status, version and code hash), so the outputs
  of a given build can be found. `INSAR_RUN_HISTORY` moves it (the tests do).
- Every run log starts with the build that wrote it.

**Release** (`USER_GUIDE.md` section 7): `review.json` (the sign-off, bound
to a hash of the report's numbers) and `releases/<YYYYmmdd_HHMM>/` with
`client/`, `data/`, `technical/`, `README.txt` and `MANIFEST.json` (a sha256
per file). Release folders are read-only; cleanup never touches them.

---

## New settings (5 Oct 2026)

All off or neutral by default. `load_config` rejects unknown keys and
checks values, files and polygons when the config loads, so mistakes surface
in the builder or the preflight, not mid-run. Since 16 Oct 2026 it also checks
each value's type (true/false settings take true/false, yes/no, on/off or
1/0; counts whole numbers; numbers unquoted), names the line of a YAML
syntax error, and holds `stack_filter_strength` to 0-1 and
`gnss_tie_radius_km` to 0-100.

| Setting | Default | Does |
|---|---|---|
| `disp_client_report` | false | lets the DISP-S1 screening route write a client report and KMZ |
| `allow_missing_dates` | [] | acquisitions (YYYYMMDD) accepted as missing; `['all']` accepts any |
| `min_unwrap_consistency` | 0.0 | above 0, drops cells below this MintPy unwrapping consistency from every report figure (off: Kyle's "measure, then decide") |
| `deramp_buffer_m` | 2000 | true distance from the mines that the ramp fit keeps out |
| `points_full_csv` | false | also write the big per-point CSV |
| `full_timeseries_kmz` | false | also write MintPy's time-series KMZ |
| `cross_check_disp` | false | run the OPERA DISP-S1 cross-check in a products run, before the quality assessment (which copies its record) |
| `injection_test` | false | run the injection-recovery test in a products run, before the quality assessment (which copies its record) |
| `mined_year_geojson` | "" | dated mined-area polygons for the panel-by-panel history |
| `mined_year_field` | "" | the property holding the year mined |
| `mined_name_field` | "" | the property holding the mine or panel name |
| `landcover_raster` | "" | an NLCD-like GeoTIFF as a second farmland-matching axis in the QA |
| `control_ground_label` | farmland | the word the report uses for the comparison ground |
| `mine_layer_sources` | [] | "Label: source, vintage" for each mine layer; the report credits them |
| `other_movement_geojson` | "" | the builder's "Other known movement" layer |
| `report_basemap` | usgs | the client map's basemap; "none" turns it off |
| `era5_allow_partial` | false | deliver even when ERA5 is missing for some dates, recording them |

### Validation experiments (8 Oct 2026)

For the research runs of `review_2026-10-08/PLAN.md` (phase 1; their configs
left the repository on 28 Oct 2026, `configs/experiments/README.md` keeps the
record). Every default is today's behaviour, so client projects and the
Indiana numbers do not move.

| Setting | Default | Does |
|---|---|---|
| `purpose` | client | `experiment`: the products stage builds the deliverables, the QA and the figures, then runs the validation harness; never the GNSS tie, client report, client KMZ, release status or package |
| `site_profile` | farmland | the kind of ground (`farmland`, `urban`, `mixed`, `mine_longwall`, `groundwater_basin`): which QA flags the ground explains. Only farmland's are checked, so on the others the flags still hold the report |
| `validation_protocol` | "" | the fixed protocol YAML the harness reads (absolute path; experiment runs only) |
| `reference_mode` | farmland | what relative figures are relative to: `farmland` (as always), `stable_polygon` (the polygon below) or `gnss_absolute` (a datum fitted to GNSS stations, leave-one-out) |
| `reference_stable_geojson` | "" | `stable_polygon`: the ground drawn as stable, e.g. a ring 1-3 km outside the angle of draw |
| `reference_gnss_stations` | [] | `gnss_absolute`: the NGL stations that set the datum (at least 3; 5 with the plane). Never a station an experiment scores |
| `reference_gnss_plane` | false | `gnss_absolute`: fit a plane at each date as well as the offset |
| `reference_gnss_radius_m` | 100 | `gnss_absolute`: InSAR at a station = median of the reliable cells within this distance |
| `reference_gnss_match_days` | 3 | `gnss_absolute`: GNSS days within this many days of a radar date are averaged |
| `reference_gnss_cache_dir` | "" | `gnss_absolute`: where the GNSS series are downloaded (empty: beside the project) |

`insar validate <config> --protocol <yaml> [--out DIR] [--cache DIR]
[--ga-xlsx FILE] [--points FILE] [--up FILE]` compares a finished run with
ground truth under a fixed protocol. It only reads the project and writes
`validation.json`, a plain-English summary (validation.md), `validation_epochs.csv` and plots
(default `<project>/validation/protocol_<id>_<run>/`). `python -m
insar_pipeline.decompose CFG_ASC CFG_DESC --out DIR` makes the Up/East
decomposition whose `vertical_timeseries_decomposed.h5` `--up` reads.

### Validation experiments, phases 2 and 3 (8 Oct 2026)

Fast motion (F1-F3), event products (F5), the plate-fixed GNSS datum, the
digitised truth (T1) and the NISAR GUNW route (M1, M2). All off or neutral
for client projects: `aliasing_diagnostic: auto` is on only for experiment,
`mine_longwall` or fast-zone configs; the others do nothing until set.

| Setting | Default | Does |
|---|---|---|
| `aliasing_diagnostic` | auto | F2: before unwrapping, each pair is checked for aliasing (wrapped gradient, coherence holes, residues) into `<project>/aliasing/`, then `deliverables/quality_aliasing_risk.tif` + `aliasing_risk.json` (a lower bound). `on`, `off`, or `auto` (experiment, `mine_longwall` or fast zones) |
| `aliasing_zone_geojson` | "" | the deforming polygons F2's statistics and the F1 preflight use (empty: `deramp_exclude_geojson`) |
| `expected_max_rate_mm_yr` | 0 | F1: the fastest subsidence expected while a face passes (mm/yr, vertical); the run plan then says, per pair length, whether C-band can follow it. 0 = not set |
| `expected_trough_halfwidth_m` | 0 | F1: the trough half-width that rate builds over (0: half the narrow side of each zone polygon) |
| `fast_zone_geojson` | "" | F3: inside these polygons only each date's nearest pairs reach MintPy (longer ones alias first); originals kept and put back |
| `fast_zone_connections` | 1 | F3: how many nearest pairs per date inside the fast zones |
| `miaplpy_relink_from` | "" | F3 / A1b: re-link another project's coregistered and SLC stacks with this config's MiaplPy windows (start with "Phase linking onward") |
| `event_products` | [] | F5: `true` (the whole record) or windows `YYYY-MM-DD:YYYY-MM-DD`: change maps, per-panel deepest subsidence and its date, profiles, in `deliverables/events/` |
| `event_polygons` | "" | F5: panel outlines (GeoJSON); a polygon's own dates (`start`/`end`, `mined_start`/`mined_end`) win. The harness's protocol A stable ring and Tahmoor West peak |
| `event_lines` | "" | F5: survey lines (GeoJSON) profiled for each window; the harness's Appin peak ratio |
| `validation_points` | "" | digitised or survey truth (CSV) the validation reads, as `insar validate --points` |
| `reference_gnss_frame` | plate | `gnss_absolute`: the GNSS datum relative to the stations' plate (ITRF2020-PMM), or `itrf` for NGL's IGS20 as is. The harness always compares IGS20 with IGS20 |
| `reference_gnss_plate` | auto | `gnss_absolute` with `plate`: the plate code (NOAM, AUST, PCFC, ...), or `auto` from the stations' place |
| `nisar_allow_beta` | false | NISAR: also search BETA GUNW (never mixed with PROVISIONAL) |
| `nisar_polarization` | "" | NISAR: HH or VV; empty = the product's own co-pol |
| `nisar_ionosphere` | split_spectrum | NISAR: subtract the product's ionosphere screen, or `no` |
| `nisar_chain_gap` | longest_segment | NISAR: a broken chain keeps its longest piece, or `fail` stops |
| `nisar_fetch` | window | NISAR: read only the box from each granule over HTTPS, or `download` the whole file (cropped, then deleted) |
| `nisar_posting_m` | 80 | NISAR: the common lattice posting (m) |
| `nisar_crop_margin_deg` | 0.02 | NISAR: margin read round the box (degrees) |
| `nisar_min_valid_ratio` | 0.5 | NISAR: a pair below this share of the group's median valid area is dropped |

Also: `insar validate ... [--outlines FILE] [--lines FILE]` (the digitised
panels and survey lines; default the config's `event_polygons` /
`event_lines`; `--points` defaults to `validation_points`); `python -m
insar_pipeline.events CFG [--between D0 D1] [--polygons F] [--lines F]
[--points CSV] [--out DIR]`; `python -m insar_pipeline.aliasing CFG
[--detectability] [--measure] [--export]`; `python -m
insar_pipeline.miaplpy_backend relink CFG [--check]` (A1b); `python -m
insar_pipeline.lband CFG_C CFG_L --out DIR` (C-band against L-band, F4);
`python -m insar_pipeline.nisar_backend cleanup CFG --level 1|2 [--delete]`
(also offered by `insar cleanup` on a NISAR project). The runs, their order
and how to launch them: `configs/experiments/README.md`.

Changed defaults: `gnss_tie_radius_km` 50 → 40 km (`config.GNSS_TIE_RADIUS_KM`).
`stack_num_proc`, `stack_num_proc_topo`, `stack_num_proc_unwrap` and
`mintpy_max_memory_gb` are now sized for the machine from its memory and
cores (unchanged on a 23 GB, 24-core machine); a value in the config wins.
`max_plausible_velocity_mm_yr` is no longer a physical limit, only a count;
the QA's `plausibility` block measures the real limit.

---

## Measurement, uncertainty and reporting

**Relative to ordinary ground, by coherence class.** Every site figure is the
area's rate minus the far-field farmland rate of the same coherence class
(mean class centres), so the distributed-scatterer bias and the reference
point's drift cancel. The farmland outside the mines and their 2 km buffer is
the comparison; features that stand out are taken out of it and the
comparison is measured again (`client_report.json` `farmland_second_pass`).

**The farmland null.** The uncertainty quoted is the spread of the same
measurement on the area's own footprint placed across ordinary farmland,
not a fitted line's standard error (the residuals are correlated in time and
fields differ from each other). The decision is |excess over the farmland
median| ≥ the empirical 95% threshold. An area that fits fewer than 5
independent times in the farmland is "Not testable"; 5 to 9 placements widen
the ranges and limits.

**Totals** are the measured change over the record
(`significance.record_change`: a continuous curve that may change pace once a
year, season removed); rate × years is kept as `total_at_average_rate_mm`.
The per-cell raster of rate × years is named for what it is,
`vertical_change_at_average_rate_<unit>_<d0>_<d1>.tif` (until 6 Oct 2026
`vertical_movement_total_*`, which read as the measured total).

**The GNSS anchor** (`gnss_tie.py`) uses Nevada Geodetic Laboratory stations
within 40 km and at least 3 years over the record's own dates. Its
uncertainty is the largest of the formal error, the scatter between the
stations used, and a regional floor (the scatter of station rates within
100 km). With fewer than 3 such stations within 100 km (3 years or more of
data during the record), or no cached station catalogue, the regional floor
cannot be measured and the record is not anchored: its figures stay relative
to the surrounding farmland. A `gnss_tie.json` without the floor (an older
file, or one whose floor is null) counts as stale and is not used. A station
inside the processed area is the check, not an input.

**Local features** are tested against 199 simulated copies of the site's own
farmland under the same search, so a selected extreme is not mistaken for a
finding (59 until the 7 Oct 2026 follow-ups). The first search's three
strongest patches are also re-tested with themselves left out of the farmland
the copies are made from (`client_report._leave_candidate_out`), so a large
patch cannot hide in its own comparison ground; each re-test is corrected for
the number made (`report_insights.passes_selection`: at most 4 of 199 copies
as strong for one test, none for three).

**Posting is not resolution.** Points are 15 m apart. Each value combines
similar ground within about 90 m east-west and 240 m north-south on the
phase-linked route (15 x 15 window plus 11 x 3 looks), about 41 x 42 m on the
classic route, about 200 m on DISP-S1 (`resolution.effective_resolution`
writes the sentence the products use). Say both.

**Acceptance is coherence plus unwrapping validity, deliberately not
uncertainty.** An unwrapping error produces a LOW formal sigma: a smooth
series, a good fit and a rate wrong by whole cycles. Gating on sigma would
admit the cells that look cleanest and are most wrong. Precision travels as
an attribute, and points whose rate is indistinguishable from zero are
delivered: "monitored, no motion detected" is a real finding.

**Unwrapping engine and loop closure.** `stack_unw_engine: snaphu2` (what the
builder writes) runs the standalone snaphu 2.0.7 with every parameter
explicit; the `isce` engine keeps topsStack's unwrap.py, which caps connected
components at 20. snaphu is built without its noncommercial CS2 module
(`THIRD_PARTY.md`). `mintpy_closure_correction: true` corrects whole-cycle
errors per pixel from loop closure; it was tested and left off on the
Site A record (HANDOFF, "Unwrapping").

---

## Run control

One stage chain (`pipeline.run_from`) serves every launcher: the dashboard,
`python -m insar_pipeline.pipeline`, the `scripts/` runners and the old menu.
It owns the project lock (`<project>/.run.lock`: one run per project; a second
start says who holds it), the run record (`run.json`), a progress event per
stage, `run_result.json`, a Windows notification at the end, the disk gate
before a run and a disk watchdog during it (warns below 60 GB of real budget,
stops cleanly below 20 GB). A run that vanishes without reporting
(WSL restart, out of memory) shows as "interrupted". `runs.suggest_kind`
works out the right restart from the files on disk; the dashboard offers it
as "Recommended". Processes are stopped only by their recorded process group
and start time, never by name.

---

## Design notes worth knowing

**Each stage shells out to a CLI rather than importing a library.** These
tools pin conflicting dependency versions; keeping them as subprocesses
means one can be upgraded or moved to another environment without
touching the others. MiaplPy and burst2safe have their own conda
environments for that reason.

**Verify outcomes, not actions.** The most frequent bug class here has been
code reporting what it attempted rather than what is true. The runner marks a
command done only when its outputs exist with the right size;
`patch_configs` verifies every config after writing; MintPy's success is
recorded in `run_products.json` only after the science steps return 0 with
fresh outputs; `provenance.py` reads the products, not the log.

**The DEM must cover the frame, not the area.** topsStack selects whole
bursts; where the DEM is absent topo fills height with −500 and no
topographic phase is removed. The DEM is stitched over the burst footprint,
and the run checks after run_01 and after the merge that at least 99% of the
area lies inside the processed sub-swaths.

**Disk is the production constraint.** The runner interleaves per-date and
per-pair chains and deletes each intermediate the moment its consumer's
output exists, leaving a `.cleaned` marker the checks accept on resume. On WSL
the real budget is the WSL disk's headroom plus the Windows drive's free
space (`settings.disk_budget_gb`), not what `df` shows; the planner, doctor,
preflight and watchdog all use it.

**MintPy runs in phases with the state checked between them** (load and
network pruning; reference check; connectivity; unwrap-error correction;
inversion; then corrections, deramp, DEM error, geocode). MintPy warns and
continues where it should stop (a disconnected network, a missing closure
dataset); the pipeline stops instead.

**Network redundancy is not optional.** `stack_num_connections: 3` gives
closed triplets. A sequential chain has none: temporal coherence reads 1.000
by construction and phase-closure correction has nothing to work on
([Yunjun et al. 2019](https://doi.org/10.1016/j.cageo.2019.104331)).

---

## Tests

    conda activate insar && cd ~
    python -m pytest insar_pipeline/tests -q -p no:cacheprovider

About 8-9 minutes (about 1,960 tests). `insar doctor` runs the same suite.
`tests/test_fix_docs.py` checks that every command and file these documents
name exists. On a fresh clone the tests that need a client project or a
local record (the golden client numbers, the validation truth data) skip;
`tests/conftest.py` keeps every test away from the real notifications log.

## Install, requirements and licences

`install/README.md`. The environments are pinned exactly in
`install/envs/` (`insar.yml`, `miaplpy.yml`, `burst2safe.yml`,
`raider-env.yml`, `dolphin.yml`; conda-forge). `requirements.txt` is a
reference list of the packages the code imports, at those versions, not the
install source; every release carries `SBOM.cdx.json`, the full list. Two free accounts are needed:
NASA Earthdata (Sentinel-1, OPERA, ionosphere maps) and the Copernicus CDS
(ERA5, with its CC-BY licence accepted). Licences and required credits:
`THIRD_PARTY.md`. The pipeline is proprietary; see [LICENSE](LICENSE).

## Validation

Before trusting any result, check the magnitude. Subsidence is centimetres
per year; hundreds of metres per year means something upstream is wrong,
typically unwrapping errors or a misread band.

**Every serious bug found while building this produced a clean run,
complete outputs, and no error at all.** The worst of them, a masked array
reused as an accumulator, quietly removed 99% of a scene and inverted the
apparent sign of the result. Three checks catch this class of failure, and
the pipeline runs the first two itself:

1. **Is coverage stable across the stack?** The runner verifies every
   command's outputs, and the QA reports coverage at the working threshold,
   against the measured noise floor, and on both quality checks.
2. **Do two quantities that should agree, agree?** Velocity against
   cumulative displacement; the official route against DISP-S1
   (`insar cross-check`); known planted movements against what comes back
   (`insar injection-test`).
3. **Where are the surviving pixels, not just how many?** No summary
   statistic distinguishes good coverage from a single surviving corner.

The reference pixel and the distributed-scatterer bias are stated limits of
any C-band record over farmland, not oversights: the QA measures both, and
the site figures are relative to the surrounding ground because that cancels
them.
