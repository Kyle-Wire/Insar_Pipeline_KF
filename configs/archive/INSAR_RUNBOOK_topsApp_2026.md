# ARCHIVED: InSAR runbook for the topsApp ("local") route and hand-built environments

**Archived 6 Oct 2026. Do not follow this for project work.** It describes the
retired `local` backend (topsApp per pair, RAiDER, a sequential pair network)
and environments built by hand, as the pipeline was in early September 2026.
The current runbook is `INSAR_RUNBOOK.md` in the package folder (the
dashboard and the official phase-linked route); installation is
`install/README.md`. Several commands below no longer exist
(`config_ridgecrest_ts.yaml`, `insar_pipeline.screening_report`,
`archive_project --level all`): they are kept only as a record. Advice that
was wrong or dangerous has been removed and marked "[removed 6 Oct 2026]".

---


## 0. One-time setup

Only needed once per machine. Skip to section 1 if the environments exist.

### Environments

```bash
# ISCE2 has no Windows build — everything runs inside WSL2.
conda create -n insar python=3.11 -c conda-forge --yes
conda activate insar
conda install -c conda-forge isce2 mintpy --yes
pip install asf-search rasterio pyyaml dem_stitcher h5py scipy

# RAiDER gets its own environment: its numpy/xarray requirements
# conflict with ISCE2's, and installing both together silently
# resolves to a years-old RAiDER that crashes on np.float.
conda create -n raider-env -c conda-forge raider --yes
```

### Put ISCE2 on PATH permanently

The conda-forge isce2 package installs `topsApp.py` into site-packages
without adding it to PATH. This is the single most common "it worked
yesterday" failure.

```bash
conda activate insar
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export ISCE_HOME=$CONDA_PREFIX/lib/python3.11/site-packages/isce' >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo 'export PATH=$ISCE_HOME/applications:$ISCE_HOME/bin:$PATH' >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
conda deactivate && conda activate insar
which topsApp.py      # must print a path, not nothing
```

### Credentials

Two accounts, both free. Use `nano` rather than a heredoc — pasting
multi-line blocks into the shell mangles them, and a `!` in a password
gets eaten by bash history expansion.

**NASA Earthdata** (for Sentinel-1 downloads) — register at
urs.earthdata.nasa.gov, then authorise "Alaska Satellite Facility Data
Access" under Applications → Authorized Apps. The pipeline prompts for
these at runtime, so no file is needed.

**Copernicus CDS** (for ERA5 weather data) — register at
cds.climate.copernicus.eu, accept the ERA5 licence on the dataset page
(separate from registering, easy to miss), then:

```bash
nano ~/.cdsapirc
```
```
url: https://cds.climate.copernicus.eu/api
key: YOUR-TOKEN-HERE
```
```bash
chmod 600 ~/.cdsapirc
```

### WSL memory headroom

Four concurrent ISCE2 processes once took down the whole VM. In Windows,
create `C:\Users\<you>\.wslconfig`:

```
[wsl2]
memory=20GB
swap=16GB
```

Then `wsl --shutdown` in PowerShell and reopen Ubuntu.

---

## 1. Start a session

Every time. The PATH export is only needed if you skipped the permanent
fix above.

```bash
conda activate insar
which topsApp.py          # if empty: export PATH=$ISCE_HOME/applications:$PATH
cd ~
```

---

## 2. Build a config

Run `insar builder` (or press `b` in the dashboard). A map opens in your browser:
draw or upload the project area, upload the mine shapefiles (a statewide layer is
fine; only mines in the processing box are kept), add any extra report areas and
an optional stable reference point, check the archive, and press **Create
project**. Uploads can be a shapefile (.shp with .dbf .shx .prj, or zipped, any
projection), KML/KMZ, GeoJSON or GeoPackage. The project is written to
`~/insar_projects/configs/<name>.yaml` from the production defaults and appears
in the dashboard within seconds. "Open an existing project" reopens its layers
for changes (the old config is kept as a `.bak`). To change a processing setting
for one project, press `e` on the dashboard's preflight screen.

Or copy an existing one and edit:

```bash
cp ~/insar_pipeline/config_ridgecrest_ts.yaml ~/insar_pipeline/config_myproject.yaml
nano ~/insar_pipeline/config_myproject.yaml
```

**Keep `project_dir` under `/home/`, never `/mnt/c/`.** Processing on the
Windows filesystem is slow and throws intermittent I/O errors under
sustained load.

---

## 3. Check before you commit hours

```bash
df -h /home        # stack backend, MEASURED at Site A: ~0.45 GB of burst SAFEs per date;
                   # a 251-date run peaked at 242 GB with every cleanup on; --plan-only prints the estimate
python -m insar_pipeline.pipeline insar_pipeline/config_myproject.yaml --plan-only
```

Watch the first two lines of output and stop it (Ctrl+C) if they look wrong:

- `Found N candidate scenes` — if N is much larger than you expected,
  tighten the date range. Each pair is ~30 minutes.
- `Built N-1 pairs from N scenes`

Downloads are 4–5 GB per scene. After the first successful run, add
`--skip-download` to every subsequent invocation.

---

## 4. Run it

```bash
python -m insar_pipeline.pipeline insar_pipeline/config_myproject.yaml --skip-download
```

A run started this way in the foreground stops when its terminal closes. (Runs started from the dashboard are detached and survive the dashboard, the terminal and an SSH drop.) Stages in order:

1. **Data acquisition** — prompts for Earthdata credentials, downloads SLCs, stages the DEM
2. **Pair selection** — sequential network
3. **Interferogram formation** — ISCE2, ~30 min per pair, the long part
4. **Tropospheric correction** — RAiDER + ERA5, a few minutes per date
5. **Time series inversion** — MintPy, minutes

Check progress from another terminal:

```bash
ls ~/insar_projects/myproject/interferograms/*/merged/wrapped_phase.tif | wc -l
```

---

## 5. Check the results are real

A pipeline that completes is not a pipeline that's correct. Both of the
worst bugs here produced clean runs and plausible-looking output.

```bash
cd ~/insar_projects/myproject/mintpy
python -c "
import h5py, numpy as np
v = h5py.File('velocity.h5','r')['velocity'][:]
m = h5py.File('maskTempCoh.h5','r')['mask'][:].astype(bool)
print('velocity m/yr: min %.4f  max %.4f' % (np.nanmin(v[m]), np.nanmax(v[m])))
tc = h5py.File('temporalCoherence.h5','r')['temporalCoherence'][:]
print('temporal coherence: mean %.3f' % np.nanmean(tc))
"
```

**Magnitude check.** Tectonic and subsidence velocities are centimetres
per year. Hundreds of metres per year means something is wrong upstream —
most likely unwrapping errors, or a band being misread.

**A site figure is only a finding if it beats ordinary farmland.** Read
`significance` beside every regional rate in `site_history.json` /
`panel_history.json`: `farmland_spread_1sigma_mm_yr` is how much patches of
farmland with the same footprint differ over the record (several mm/yr at
C-band), and `distinguishable_from_farmland` says whether the figure stands
outside it. A fitted line's standard error is not an uncertainty here — the
residuals are correlated in time and fields differ from each other. And a
feature found by searching for the deepest point is a selected extreme: compare
it with `extreme.expected_deepest_by_chance_mm_yr`, not with zero.

**Temporal coherence of exactly 1.000 is not a good result.** A purely
sequential network has no closed loops, so the inversion is exactly
determined and the residual is zero by construction. The metric is
measuring nothing. Redundant connections are what make it meaningful.

---

## 6. Look at it

```bash
cd ~/insar_projects/myproject/mintpy
save_gdal.py velocity.h5 -o velocity.tif
```

In QGIS, open via `\\wsl.localhost\Ubuntu\home\<you>\insar_projects\...`
(WSL must be running for that path to resolve).

Per-pair products on the stack backend, in `stack/merged/interferograms/<pair>/`:

| File | What it is |
|---|---|
| `filt_fine.int` | Filtered wrapped interferogram (complex). Immune to unwrapping error — the most trustworthy view near the active face. |
| `filt_fine.cor` | Spatial coherence, 0–1. Where the measurement is reliable. |
| `filt_fine.unw` | Unwrapped phase (2-band: amplitude, phase in radians). Inherits any unwrapping errors. Zero outside the unwrap crop window (`stack/crop_window.json`). |
| `filt_fine.unw.conncomp` | snaphu connected components; 0 = not reliably unwrapped. MintPy's bridging correction needs this. |
| `unwrap_crop.json` | Window, seconds, and the fraction of the crop in a nonzero component for this pair. |

The delivered products are the MintPy outputs in `mintpy/geo/` and the
GeoTIFFs in `deliverables/`, all masked by temporal coherence
(`quality_temporal_coherence.tif` is the mask's source). Always mask by
coherence before trusting displacement values.

The `local` (topsApp) backend writes `interferograms/<pair>/merged/`
`wrapped_phase.tif`, `los_displacement_m.tif` and `phsig.cor.geo` instead.

InSAR returns a number everywhere, including where the measurement is
meaningless. The coherence layer is what separates signal from artifact.

---

## 7. Screen pairs on atmosphere (optional)

```bash
python -m insar_pipeline.screening_report insar_pipeline/config_myproject.yaml
```

Shows each pair's differential tropospheric delay variation. Set
`max_tropo_std_cm` in the config to drop the worst. Two cautions: ERA5's
grid is tens of km so it misses small-scale turbulent vapour, and cutting
too many pairs fragments the SBAS network — check `mintpy/pic/` for the
network plot.

---

## 8. Reclaim disk space

Dry run first; it deletes nothing without `--apply`.

```bash
python -m insar_pipeline.archive_project insar_pipeline/config_myproject.yaml --level all
python -m insar_pipeline.archive_project insar_pipeline/config_myproject.yaml --level scratch --apply
```

Levels: `scratch` (intermediates, safe), `weather` (cached ERA5,
re-downloadable), `slc` (the big zips, slow to replace), `all`.

Never removed: `merged/` outputs, GIS products, MintPy results, DEM,
logs, configs.

---

## Troubleshooting

### `topsApp.py: command not found`
PATH lost on a new session. `export PATH=$ISCE_HOME/applications:$PATH`,
or redo the permanent fix in section 0.

### `ASFAuthenticationError: HTTP 401`
Earthdata credentials rejected. Log in at urs.earthdata.nasa.gov in a
browser with the same username and password to confirm they work, and
check the ASF app is authorised under Applications → Authorized Apps.

### `No suitable orbit file found`
`orbit directory` is set to an empty folder. Unset it and ISCE2 fetches
orbits itself at process time.

### ESD: `Coherence threshold too strict. No points left`
Burst-overlap coherence is too low for the default 0.85 — normal over
ice, snow, or dense vegetation. Lower `esd_coherence_threshold` toward
0.6, or set `do_esd: false` and accept geometric coregistration only.

### `Number of products to geocode: 0`
The geocode list must name *input* files (`merged/filt_topophase.unw`);
ISCE2 appends `.geo` itself.

### Run dies with no traceback, or WSL becomes unreachable from Explorer
Out of memory, or orphaned processes competing. Check first:

[removed 6 Oct 2026: two commands stood here that killed processes by
matching text in their command lines. They killed the dashboard and every
project's run on the machine. Stop a run with the dashboard's `x`, which
signals only that run's own process group.]

Then clear any half-finished pair before restarting — cleanup removes the
pickles, so a partial pair cannot resume:

```bash
cd ~/insar_projects/myproject/interferograms
for d in */; do [ -f "$d/merged/wrapped_phase.tif" ] || rm -rf "$d"; done
```

### MintPy `KeyError` on HEIGHT, DATE12, P_BASELINE_TOP_HDR, or ALOOKS
MintPy's ISCE reader was built around topsStack, not topsApp, so several
metadata fields have to be staged manually. The pipeline handles all of
these — if one resurfaces, the likely cause is an edited file that didn't
get copied into `~/insar_pipeline/`.

### A code change seems to have no effect
Check it actually landed before debugging anything else:

```bash
grep -c "<something unique to the new version>" ~/insar_pipeline/<file>.py
```

This wasted more time than any real bug during development. If an error
names a function that should no longer exist, that's the tell.

### MintPy exits non-zero but `velocity.h5` exists
The final plotting step fails on a one-pixel DEM/interferogram size
mismatch after every science product is already written. The pipeline
treats this as success.

---

## Where InSAR won't work

Worth knowing before spending a night processing. Coherence is the
limiting factor, and it collapses over:

- Glaciers, snow, and ice
- Dense vegetation, especially at C-band
- Steep terrain, where layover and shadow make one look direction unusable
- Anywhere the surface itself was destroyed — a landslide runout, a
  rupture zone

That last case is not a failure to route around. Coherence loss *is* the
measurement: comparing pre- and post-event coherence maps the disturbed
area directly, which is often the right product for a rapid mass
movement. Phase-based deformation retrieval is the wrong tool there.

Bare desert, bare rock, and built-up areas are where this works well.
