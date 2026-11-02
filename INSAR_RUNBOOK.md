# InSAR pipeline runbook

From an installed machine to a signed-off client package on the official
route (phase-linked, `backend: miaplpy`), using the dashboard. What the
numbers mean, the quality checks and the failure table are in
`USER_GUIDE.md` (press `?` in the dashboard). The retired topsApp route and
the old hand-built setup are in `configs/archive/INSAR_RUNBOOK_topsApp_2026.md`
for the record only.

The order of work:

1. install, then `insar doctor`
2. build the project in the map builder
3. preflight
4. run
5. read the verdict
6. the report is a DRAFT until it is signed off
7. sign off
8. package and send
9. clean up with `insar cleanup`

---

## 1. Install and check the machine (once)

Follow `install/README.md`: WSL2 Ubuntu, two free accounts (NASA Earthdata,
Copernicus CDS with the ERA5 licence accepted), then
`bash insar_pipeline/install/install.sh`. It ends by running the doctor.
The release file is internal: do not distribute outside the operating
organisation. It holds no client data.

Run the doctor again any time:

    insar doctor            # everything, including the test suite (10-30 minutes)
    insar doctor --quick    # skip the test suite

It must end in **Ready.** Each red line says how to fix it. It checks the
logins, the ERA5 licence, the three environments, snaphu (by unwrapping a
known answer), the patched MiaplPy, the real disk budget and the memory.

Tell it who prepares the reports (the report footer prints this):

    insar doctor --firm "Your Firm" --preparer "Your Name" --contact "you@yourfirm.com"

Memory: if the doctor says WSL has too little, edit `C:\Users\<you>\.wslconfig`
as it shows (`USER_GUIDE.md` section 9), run `wsl --shutdown` in Windows
PowerShell and reopen Ubuntu.

Keep the projects folder under `/home`, never under `/mnt/c`: processing on
the Windows file system is slow and throws I/O errors under load.

---

## 2. Build the project in the map builder

Type `insar`, then press `b` (or run `insar builder`). A map opens in your
browser.

1. **Project area.** Draw it or upload its outline. The dashed box around it
   is the processing box: the margin gives ordinary ground to compare with.
2. **Mines and known movement.** Upload or draw underground workings, surface
   mines, other known movement, and dated mined areas (one polygon per panel
   or year, with the year in the attribute table). Fill in each layer's
   "Source, date". See `USER_GUIDE.md` section 4.2 for where to get outlines.
3. **Extra report areas** (optional). Each gets its own row in the report.
   Give each the plain name the client uses.
4. **Reference point** (optional). Leave it empty unless you know a stable
   spot; the pipeline then picks one more than 2 km from every mapped mine
   edge.
5. **Dates and route.** The end date defaults to a week ago, because ERA5
   weather data are published about 5 days late. Choose **Phase-linked**
   for official results. OPERA DISP-S1 is screening only and gets its own
   project folder, `<name>_disp_screening`.
6. **Check the satellite archive.** It lists the tracks with their
   sub-swaths and how much of the box each covers, the DISP-S1 frames, and
   the GNSS stations the tie will use (within 40 km, at least 3 years over
   your dates). It warns below 3 stations.
7. **Create the project.** The config is written to
   `$INSAR_PROJECTS/configs/<name>.yaml` with the production settings, and
   the project appears in the dashboard within seconds.

The builder refuses names the client should not see: underscores, colons,
file names, and placeholders such as "Area 2". It refuses a sub-swath choice
that covers less than 99% of the box or any area, and it will not overwrite
an existing project without asking (the old config is kept as `.bak_<time>`).

Searching for a place sends your text to OpenStreetMap: type a town or road,
never a client's or a site's name.

Mine-layer quality matters more than any setting. Mine movement left in the
"ordinary ground" makes every comparison wrong. When in doubt, include the
layer.

---

## 3. Preflight

Open the project (Enter), press `s`. The list shows **Recommended** first
(for a new project: Whole pipeline). Choosing it opens the preflight screen.
Start stays disabled while any line is red.

| Line | Red means |
|---|---|
| Disk budget | the planner's projected peak, minus what the project already holds, does not fit the real budget (on WSL: the WSL disk's headroom plus the Windows drive's free space) |
| Products memory | the products stage needs more than 90% of the machine's memory |
| Earthdata login | ASF refused the login (checked only when the run downloads) |
| ERA5 | the CDS token is wrong or the licence is not accepted (yellow when the CDS could not be reached) |
| ERA5 publication lag | (yellow) the end date is within about 5 days of today |
| Patched MiaplPy | the installed MiaplPy fails the patch check with this project's mini-stack and time lag |
| Report-area names | (yellow) a name reads like a file; the release gate will block it, so fix it now with `e` |
| Other runs | (yellow) other projects are running on this machine and share its disk and memory |

`e` edits the config, `r` checks again. To size a run without starting it,
choose **Size the run only**, or from a terminal:

    python -m insar_pipeline.plan_run <config.yaml>

---

## 4. Run

Press Start. The run is detached: closing the dashboard, the window or the
terminal does not stop it. Reopen `insar` to follow it. Only `x` (Stop) stops
it, together with every program it started. When it ends you get a Windows
notification.

The stages, in order (labels as the dashboard shows them):

| Stage | What happens | Measured on Site A (278 dates) |
|---|---|---|
| Download Sentinel-1 data | one SAFE per date, assembled from the needed bursts, 4 at a time; each date gets 3 attempts | 4 h 23 min (about 0.45 GB per date) |
| Orbit files | precise orbits | minutes |
| ERA5 weather download (in the background) | the weather files MintPy needs, fetched during coregistration | runs alongside |
| Stack 01 to 12 | coregistration with ISCE2 topsStack, then the merge of full-resolution images | 1 h 57 min, merge 16 min |
| Phase linking: load data, phase linking, concatenate patches | MiaplPy, the patched build, on a crop around the area | load 17 min, phase linking 21 h 09 min, concatenation about 9 min |
| Stack 16: unwrap | the linked phases exported as pairs (3 connections per date) and unwrapped with snaphu | export 6 min |
| MintPy steps | network inversion, ERA5 troposphere, the deramp fitted outside the mines and their 2 km buffer, DEM error, geocoding to 15 m | |
| Deliverables | GeoTIFFs, compact point products, summary | |
| DISP-S1 cross-check (optional), Injection-recovery test (optional) | only when the config asks (section 10) | |
| Quality assessment | the six checks and the send rule | about 40 s (5 Oct 2026 code) |
| Figures, panel + site histories | technical figures and per-panel mining history | |
| GNSS tie + client report | regional GNSS rate, client report, area summaries PDF, KMZ | report about 2.5 min (5 Oct 2026 code) |
| Provenance + release status | the record of what produced the result; the log ends with FINAL or DRAFT | |

The download, coregistration and phase-linking figures are from Site A's
first production attempt (23-24 Sep 2026). A whole run takes 1 to 2 days and
peaks at about 300 GB (Site B and Site A, about 270 dates each). The live view shows each stage with
progress, elapsed and remaining time, and the log.

Optional stages, off by default, run after the deliverables and before the
quality assessment (which copies their records) when the config asks:
`cross_check_disp: true` (compare with OPERA DISP-S1) and
`injection_test: true` (plant known movements and measure how well they come
back). See section 10.

---

## 5. Read the verdict

When the run ends, the projects screen's Verdict column shows the send rule
and DRAFT or FINAL. Then:

1. Open `<project>/deliverables/client_report.json` → `send_rule` (the
   QA's rule re-evaluated with the GNSS tie; `quality_assessment.json` holds
   the QA's own), or read the "send rule" lines at the end of the run's log.
   If the QA was re-run after the report, the report's rule is out of date:
   the dashboard marks it "(report out of date)" and the release is blocked
   (`report_older_than_qa`) until the report is rebuilt (Products only).
   "hold for specialist" means Kyle looks before anyone signs; the release
   stays DRAFT until the sign-off names that specialist (`--specialist`,
   or the dashboard's Specialist field; `USER_GUIDE.md` section 7.2).
2. Press `o` to open the client report. Read it the way the client will.
   `USER_GUIDE.md` section 6 explains every number.
3. For every feature the report lists, check imagery and dates. A sharp patch
   in a working mine area can be spoil, fill or a new pit rather than
   subsidence.
4. Check that every report area reads as a real reading. A "Not testable"
   area does not block release: it is on the reviewer's list (`[read]`).
   Read its reason in the report and accept it before signing off; do not
   shrink the client's area to make it testable.

To list what still blocks release:

    python -m insar_pipeline.release <config.yaml>

`[must]` lines block; `[read]` lines go on the reviewer's list.

---

## 6. DRAFT until signed off

Every report starts as a DRAFT: a watermark, a list of what blocks it, a KMZ
named `..._DRAFT.kmz`. It becomes FINAL when there are no hard blockers and a
reviewer has signed off these exact numbers. The full list of blockers is in
`USER_GUIDE.md` section 7.3.

To preview a report or KMZ without touching `deliverables/`:

    python -m insar_pipeline.client_report <config.yaml> --out-dir <folder>
    python -m insar_pipeline.client_kmz <config.yaml> --out-dir <folder>

(Site B: about 1 min and 0.9 GB; Site A about 2.5 min and 1.4 GB.)

---

## 7. Sign off

Press `r` on the project's live view. Enter your name, tick each feature you
checked, add notes, confirm. The dashboard records the sign-off in
`deliverables/review.json`, rebuilds the report, and builds the package when
the gate says FINAL. If the rebuild changed a number, sign again.

From a terminal:

    python -m insar_pipeline.release <config.yaml> --sign "Your Name" --features 1,2,3 --notes "..."

then rebuild the report (dashboard: Products only) so it renders without the
DRAFT mark.

---

## 8. Package and send

Press `k` (or let the sign-off build it), or:

    python -m insar_pipeline.package <config.yaml>

It refuses unless the report is FINAL. It writes
`deliverables/releases/<YYYYmmdd_HHMM>/`. Send `client/` (and `data/` if the
client wants GIS files). Never send `technical/`, and never send files
straight from `deliverables/`. In `data/`,
`vertical_change_at_average_rate_in_<d0>_<d1>.tif` is each cell's average
rate × years, not the measured change the report gives for areas. A Results
workbook exported before the report was last rebuilt, or without its Notes
sheet, is left out of `client/` (README.txt lists it): export the area again.

---

## 9. Clean up

After the package is sent, free disk space:

    insar cleanup <config.yaml>                      # level 1: history and scratch, dry run
    insar cleanup <config.yaml> --level 2            # also the processing inputs, dry run
    insar cleanup <config.yaml> --level 2 --delete   # delete; asks you to type 'yes'

Level 3 adds the large point files of the old layout. Levels 2 and 3 run
only on a finished project (last run done; QA and client report newer than the
velocity). After level 2 only "Products only" is sure to work: "Phase linking
onward", "Re-unwrap" and "MintPy onward" need the inputs it deletes, and
"Whole pipeline" downloads and coregisters everything again (about 2 days).

Cleanup never touches `logs/`, `deliverables/releases/`, the sign-off, the
point products, `mintpy/geo/` or the delivered rasters and reports, and
refuses while a run holds the project. Before deleting a level-2 folder it
copies its small records to `logs/records/`.

---

## 10. Optional checks on a finished project

Both hold the project lock (exit code 3 means another run holds it) and never
change the client numbers or the verdict. The QA, the report and the release
gate use their records only while the option is on in the config
(`injection_test: true`, `cross_check_disp: true`) and the record is no older
than the velocity product: after a new MintPy run, run the check again.

    insar injection-test <config.yaml>      # about 4 to 8 min, 1.1 GB; deliverables/technical/injection_recovery.json
    insar cross-check <config.yaml>         # about an hour of DISP-S1 download the first time

The injection test plants troughs and spoil settlement of known size into the
official rates and applies the report's own rule. It covers the deramp, DEM
error, rate fit, matching and decision, not phase linking or unwrapping, so
its detection rates are an upper bound. Until 7 Oct 2026 it planted uplift
while calling it sinking, so the Site A figures quoted earlier (6, 4 and
0 of 6) are withdrawn. A review rerun with subsidence on site_a_pl gave:
farmland, 0.6 km patches, 40 mm/yr 6 of 6, 20 mm/yr 2 of 6, 10 mm/yr 0 of 6;
farmland, 1.0 km, 6, 5 and 0 of 6; mine sites (both sizes) 6, 5 and 1 of 6.
Re-run it on Site A into a preview folder
(`insar injection-test <config.yaml> --out-dir <folder>`) before quoting any
of these.

The cross-check runs OPERA DISP-S1 through the record's chain into a sibling
project `<project>_disp_check` and compares over the dates both cover. Its
pass thresholds are calibrated only on Site A.

**Extra quality layers on a project finished before 5 Oct 2026.** MintPy's
unwrapping-consistency coherence, the DEM-error layer and the disturbed-ground
layer can be added without re-running MintPy (10 to 20 s, adds files only):

    python -c "from insar_pipeline.config import load_config; from insar_pipeline.time_series import geocode_extra_layers; geocode_extra_layers(load_config('<config.yaml>'))"

then run "Products only".

---

## 11. Restarting the phase-linked route

Use the dashboard's **Recommended** restart (or Resume). It reads the disk and
picks the right start; wrong ones are refused with a plain message. The table
of restarts and the failure table by stage are in `USER_GUIDE.md` section 8.
The phase-linked specifics:

**Steps are skipped only on a completion marker.** Each finished MiaplPy step
leaves `miaplpy/.status/<step>.done.json`; the concatenation also leaves
`miaplpy/inverted/phase_series.complete`, written after a full scan. A Stop or
crash during the 9-minute concatenation re-concatenates from the patches.

**An interrupted load.** MiaplPy will not refill an existing
`miaplpy/inputs/slcStack.h5`, so the resume stops and asks for it to be
deleted. Kyle runs the delete; then "Phase linking onward".

**Changing MiaplPy settings after linking.** The method, mini-stack size,
time lag, windows, subset and dates are recorded in
`miaplpy/inverted/linking_fingerprint.json`. A resume with different settings
stops and names them. Put them back, or delete `miaplpy/inverted` to re-link
(most of a day: 21 h on Site A).

**The datum gate tripped.** The export checks every mini-stack boundary on
cells coherent on both sides (Site B: ratio 0.96, weakest boundary 0.82,
pass line 0.6; details in `stack/merged/linked_datum_check.json`). If it
fails:

1. Check the patch with this project's settings, in the miaplpy environment:

       conda run -n miaplpy python ~/insar_pipeline/patches/miaplpy/check_miaplpy_patch.py --ministack 10 --time-lag 3

   It must say patched. If not: `bash patches/miaplpy/build_miaplpy.sh miaplpy`.
2. Raise `miaplpy_time_lag` (for example 3 to 5; with mini-stack 10 the
   patch check fails from about 7).
3. Kyle deletes `miaplpy/inverted` (the SLC stack is kept unless
   `miaplpy_delete_slc_stack_after_concatenate` is set).
4. "Phase linking onward".

Do not switch to non-sequential EMI: it solves the full stack per pixel and
did not finish (HANDOFF, 25 Sep 2026).

**Failed unwraps.** Any failed pair's partial output is moved to
`stack/quarantine_unwrap_timeout/<pair>/` and the run stops naming the pairs.
Fix the cause, then "Re-unwrap"; pairs already unwrapped are kept. A
Re-unwrap that was interrupted, failed or stopped is carried on the same way:
Resume and Recommended choose "Re-unwrap", so no pair keeps its old unwrap
beside the new ones.

**The reference point.** The automatic reference lies more than
`acceptance_far_field_buffer_m` (2 km) from the edge of every mapped mine;
`stack/reference_choice.json` records the distance. If no pixel qualifies,
set `reference_lalo` in the config to a stable point and restart "Phase
linking onward" (linking, export and unwrapping are kept).

**Adding dates.** On the phase-linked route a later end date, or any change
to the stack settings, means a new project. Once the merge has replaced the
coregistered images, the plan cannot be remade in place, and the run refuses
with the changed settings listed.

**Dates the archive cannot supply.** A date that fails 3 download attempts
stops the run, naming it. Wait and resume, or accept the gap with
`allow_missing_dates: ['YYYYMMDD']` in the config (`['all']` accepts any).

**ERA5 not yet published.** If any date lacks a weather delay, the run stops
after the troposphere step and lists the dates. Wait about a week and resume
(weather files already fetched are kept), set an earlier `end_date`, or set
`era5_allow_partial: true` to deliver anyway with the uncorrected dates
recorded.

**The deramp refused.** The run stops when the mine polygons plus
`deramp_buffer_m` (2000 m) leave no ground to fit, or ground on fewer than 3
sides of the area. Reduce `deramp_buffer_m`, widen the box, or set
`mintpy_deramp: no`.

---

## 12. From a terminal

Everything the dashboard does can be started from a terminal, with the same
project lock, run record and `run_result.json`. Activate the environment
first and run from your home folder:

    conda activate insar
    cd ~
    python -m insar_pipeline.pipeline <config.yaml>                    # whole pipeline
    python -m insar_pipeline.pipeline <config.yaml> --resume           # the restart that fits the disk
    python -m insar_pipeline.pipeline <config.yaml> --start products   # or miaplpy, unwrap, mintpy
    python -m insar_pipeline.pipeline <config.yaml> --plan-only        # size it, download nothing

`--force` overrides a refusal (it logs a warning); `--no-notify` skips the
notification. Exit codes: 0 finished, 1 a step failed, 3 another run holds the
project, 130 or 143 stopped. The `scripts/run_pipeline.py`,
`scripts/run_mintpy_only.py`, `scripts/run_miaplpy_only.py`,
`scripts/rebuild_products.py` and `scripts/reunwrap_linked.py` runners are thin
wrappers around the same chain and require the config argument.
Runs started this way also appear in the dashboard, and a terminal run in the
foreground stops if its terminal closes.

The quality assessment alone (it writes `quality_assessment.json`):

    python -m insar_pipeline.quality_assessment <config.yaml> [--no-fetch]

`insar --menu` opens the old numbered menu; it starts and stops runs exactly
as the dashboard does. `insar --version` prints the version and the code
hash. `insar help` lists every command.

---

## 13. Troubleshooting outside a run

**`insar: command not found`.** Open a new terminal (the installer added
`~/.local/bin` to the path for new terminals), or re-run
`bash insar_pipeline/install/install.sh`.

**`ASFAuthenticationError: HTTP 401`.** Log in at urs.earthdata.nasa.gov in a
browser with the same username and password, and check that "Alaska
Satellite Facility Data Access" is authorised under Applications → Authorized
Apps. `insar doctor` checks the login the way the downloads use it.

**ERA5 licence.** Each user accepts the "CC-BY licence" on the CDS ERA5
pressure-levels dataset page (Download data → Terms of use). The doctor shows
a red line only when a required licence is really missing.

**Stopping something.** Use the dashboard's `x`. It stops the run and every
program it started, identified by process ID and start time, so nothing else
is touched. Never kill processes by name or pattern (`pkill -f`, `killall`):
the pattern also matches the dashboard and other projects' runs.

**A run's leftovers.** If a runner died hard (killed for memory), its child
programs are recorded in `<project>/.run_children.json`; the next Stop or
start stops them.

**A code change seems to have no effect.** `insar --version` prints the code
hash; it changes with any edit to the package. Compare it with the hash in the
project's `deliverables/provenance.json`.

**The WSL disk.** It is a file on C:. Space freed inside Linux does not return
to Windows until the file is compacted. `insar doctor` reports the real
budget; `df -h` inside Linux does not.

**Looking at files in Windows.** In Explorer or QGIS, open
`\\wsl.localhost\Ubuntu\home\<you>\insar_projects\...` (WSL must be running).

---

## Where InSAR will not work

Coherence is the limit, and it collapses over glaciers, snow and ice, dense
vegetation (especially at C-band), steep terrain (layover and shadow), and
anywhere the surface itself was destroyed (a landslide runout, an active pit
being dug). In the last case the coherence loss is itself the measurement:
comparing coherence before and after maps the disturbed ground. Phase-based
movement figures are the wrong tool there; the report flags ground reshaped
by metres and gives no rate for it.

Bare ground, rock and built-up areas work well. Farmland works, with the
care this pipeline takes: comparison with farmland of the same coherence, a
measured farmland null, and a GNSS anchor.
