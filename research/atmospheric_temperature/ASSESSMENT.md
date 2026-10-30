# Hybrid InSAR–atmospheric temperature reconstruction: feasibility assessment and plan

Branch `research/atmospheric-temperature`, cut from `main` at 90a6abc on 2026-10-28.
**Update, 2026-10-28:** Phase 1 is done. Results are in [PHASE1_RESULTS.md](PHASE1_RESULTS.md). U1 is resolved there, and gate G1 has been measured.
Status: **assessment only.** No code was changed, nothing was installed or downloaded, and no job was run.
The only reads of data were small local files: cached RAiDER cubes and MintPy HDF5 headers in existing projects.

Labels used throughout:

- **[CODE]**: verified in this repository.
- **[D]**: demonstrated in peer-reviewed literature.
- **[P]**: plausible or partly shown.
- **[H]**: a research hypothesis that has not been shown.
- **[UNVERIFIED]**: could not be confirmed this session.

---

## 0. Short answer

1. **Radar delay is a moisture and pressure measurement, not a temperature measurement.**
   - Under hydrostatic balance, the integrated hydrostatic delay depends only on surface pressure. A temperature anomaly at fixed surface pressure leaves it unchanged.
   - A 1 K anomaly in a 100 m surface layer changes the zenith wet delay by about 0.05 mm if humidity is held fixed. It changes it by about 0.4 mm if relative humidity is held fixed, and in that case the signal is really the humidity change.
   - For comparison, single-date InSAR noise is several mm and turbulent wet delay is about 10 mm.
   - In the delay, temperature and humidity are degenerate. (Section 4.1 has the arithmetic.)
2. **No published work retrieves near-surface air temperature from InSAR delay.** Two independent literature sweeps found none.
   - What *is* demonstrated is InSAR differential precipitable water vapour (PWV) to about 1–2 mm PWV [D] and its assimilation into NWP models [D].
3. **80 m is not reachable through the atmospheric-delay path.**
   - Turbulent delay power falls steeply with wavenumber, so at 80 m separation the atmospheric signal is far below phase noise.
   - Any 80 m temperature structure must come from terrain and surface predictors. Radar can supply those only as *surface* descriptors (coherence, backscatter), which is not atmospheric information.
4. **The defensible research question is narrower than the brief.** Does the residual delay after HRRR/ERA5, at the ~06:00 and ~18:00 local overpasses, carry km-scale information about the boundary-layer state (stratification, cold-air pools, moisture)? And does that information improve station-validated 2 m temperature beyond a non-radar baseline once humidity is controlled? This is testable and falsifiable [H]. The physics predicts a weak or null result except possibly in strong, deep cold-air pools.
5. **Recommended first step:** a cheap observability test and a non-radar baseline, before any radar-informed model. Section 7 gives the details.

---

## 1. Existing pipeline architecture

`insar_pipeline` is a single package: the repository root *is* the package (`pyproject.toml`, `package-dir = "."`). It is about 86 k lines of Python. Its purpose is ground-motion time series for subsidence.

**Entry points**

- `pipeline.py` (`launch()` / `run_from()` / `_run_from()`) owns the stage chains: `full`, `miaplpy`, `unwrap`, `mintpy` and `products`. It also owns teardown, locks, `run_result.json` and the disk gate.
- `cli.py`, `dashboard.py` (`insar`), `app_server.py` (`insar app`) and `scripts/run_*.py` all route through `run_from`.

**Backends** (`config.backend`; `pipeline.py:1-30`)

| Backend | Role |
|---|---|
| `miaplpy` | Phase-linked on a topsStack coregistration. Official route. |
| `stack` | topsStack and snaphu. Comparison only. |
| `disp` | OPERA DISP-S1. Screening only. |
| `local` | topsApp per pair. Testing only. |
| `hyp3` | ASF HyP3. Comparison only. |
| `nisar` | NISAR L2 GUNW, plus the `nisar_pl` GSLC + dolphin route. Comparison only. |

**Time series**

- Every route except DISP ends in MintPy 1.6.4 (`time_series.py`).
- Reference point: `mintpy.reference.*`.
- Troposphere: `corrections.mintpy_tropo_lines`.
- Deramp: default `linear`, with a mask.
- `topographicResidual = yes` with `polyOrder = 2`, written at `time_series.py:220`.
- Ionosphere: IGS GIM (`corrections.apply_gim_ionosphere`) for large areas, or the NISAR GUNW split-spectrum screen.

**Conventions**

- Configuration is a dataclass, `config.PipelineConfig`, with enum checks in `config.py:1220+`, read from project YAML files in `~/insar_projects/configs/`.
- `settings.py` holds per-machine settings.
- Logging uses the standard `logging` module.
- `provenance.py` writes `provenance.json` with versions, granules, thresholds and config sha256.
- Tests: `pytest`, with 145 files in `tests/`. Client-reading tests are `tests/test_local_*`, which git ignores.
- Dependencies come from conda environments in `install/envs/`: `insar`, `raider-env`, `miaplpy`, `dolphin` and `burst2safe`. There are no pip dependencies.
- Validation runs use `purpose: experiment` with a pre-registered `validation_protocol` YAML (`validation.py`). The protocols are fixed before results exist.

---

## 2. Map of relevant code and integration points

### 2.1 What exists and can be reused [CODE]

| Need | Existing implementation | Reusable as | Caveat for this research |
|---|---|---|---|
| Modelled delay, 3-D | `tropo_disp.delay_cube_for_date` runs RAiDER 0.6.0 (in `raider-env`) once per acquisition in zenith mode, with `interpolate_time: center_time`, and caches `<project>/tropo_corrections/<MODEL>/<date>/*_ztd.nc` | **Modelled hydrostatic and wet delay cubes, kept as separate variables.** Measured on a cached cube: `wet` and `hydro` on 57 height levels from −500 m to 26 km, on a ~0.02° lattice. | Delay only. The HRRR T, q, P and u fields are **not** kept; the raw model files are deleted after the run. A small AOI gives a small cube. |
| Model selection and time handling | `weather_raider.model_name`, `.acquisition_time`, `.check_dates`, `.archive_gaps`; `tropo_disp.check_model_range` | Same, for overpass-time background fields | HRRR covers CONUS only, from 2016-07-15, and its archive has gaps (the gap handling is already in the code). |
| Modelled LOS delay per date | `weather_raider.write_delay_file` → `mintpy/inputs/HRRR.h5` | Model delay on the radar grid | **Wet and hydrostatic are summed** (`tropo_disp._sample_cube`). Projection is 1/cos(inc), not ray-traced. The sign is negated to the pyaps3 convention. |
| Observed time series | `mintpy/timeseries.h5` (uncorrected), `timeseries_<MODEL>*.h5` (corrected) | **Observed differential phase per date** | Referenced to one pixel and one date. Deramp and the DEM-error fit come after the troposphere step. |
| Residual atmosphere estimate | `mintpy/timeseriesResidual.h5` (output of `topographicResidual`) | The closest existing "observed minus modelled" atmosphere: obs − HRRR − ramp − (DEM error + quadratic-in-time deformation) | It mixes residual APS, decorrelation noise, unwrapping errors, the phase-linking (DS) bias and unmodelled deformation. Present in 3 local projects. |
| NISAR modelled screens | `nisar_backend._read_cube` reads the GUNW radarGrid `wetTroposphericPhaseScreen`, `hydrostaticTroposphericPhaseScreen`, `incidenceAngle`, LOS vectors and baselines into each crop (`CUBE_FIELDS`, ~500 m, height levels) | A second, independent modelled delay, with wet and hydrostatic separate | It is a model, not an observation. The NISAR record is short (see 4.5). |
| L-band ionosphere | `nisar_backend` reads the GUNW split-spectrum `ionospherePhaseScreen`; `ionosphere_sign_stats`; the D3 chained per-date screens on `nisar_pl` | A per-date ionosphere estimate, to be removed before any tropospheric analysis | Split-spectrum noise at km scales is mm-level [P]. |
| C-band ionosphere | `corrections.apply_gim_ionosphere` (IGS GIM, `iono_tec.py`), auto at ≥100 km | Large-area option | GIM is ~500 km resolution, so it is no help at km scales; C-band iono is ~18× smaller than L-band. |
| Coherence | `temporalCoherence.h5`, MiaplPy/dolphin temporal coherence, `nisar_backend.coherence_gate`, `stack_qc.py` | The surface radar feature group (Section 9 of the brief) | Coherence measures surface change (vegetation, moisture, snow), not atmosphere. |
| Geometry | MintPy `geometryRadar.h5` / `geometryGeo.h5` (height, incidence, azimuth, lat/lon); `static_layers.py` (DISP-S1-STATIC); `nisar_backend.los_*` | Per-pixel LOS projection for the forward operator | — |
| DEM | `data_acquisition.get_dem` (Copernicus GLO-30 via dem_stitcher; ellipsoidal heights for NISAR, `nisar_backend.stage_dem`) | Terrain predictors: elevation, slope, aspect, TPI | GLO-30 dates from 2011–15 TanDEM-X (`provenance._DEM_EPOCH`). No 3DEP. |
| Land cover | `landcover.py` (ESA WorldCover 10 m, `ensure`, `sample_on_grid`, `classes_on_grid`) | Land-cover predictor | No NLCD canopy/impervious, NDVI, LST or albedo. |
| GNSS | `gnss_tie.py`, `ground_truth.py`, `gnss_validation.py` (NGL tenv3 *positions*, steps, MIDAS) | Station discovery, caching and NGL download patterns | **No GNSS troposphere (ZTD/ZWD/IWV) reader.** |
| Pre-registered validation | `validation.py` and `review_2026-10-08/protocol_*.yaml` (hash-fixed) | The *pattern* (protocol first, sha256, `purpose: experiment`) | The harness itself is GNSS-displacement specific. |
| Grid utilities | `lband.Grid` / `resample_to`, `nisar_backend.Lattice` / `make_lattice` (UTM/lat-lon lattices) | Common analysis grid at ~80 m | — |
| Provenance | `provenance.py`, `version.describe()` | Processing record for research products | — |
| Disk | `settings.disk_budget_gb` | Gate before any large job | — |

### 2.2 What is missing (must be built)

1. **A background atmospheric state reader** for T, q (or e), P, u, v and geopotential:
   - HRRR via Herbie, which is in `raider-env` only;
   - ERA5 or ERA5-Land, via CDS (whose queue is saturated) or ARCO-ERA5 Zarr.
   Today the pipeline only ever holds *delays*.
2. **A refractivity forward operator**: the state (P, T, e on levels), integrated along zenith or LOS, giving delay. RAiDER does this internally, but its cubes do not keep the inputs. A small, tested in-house operator is needed so the Jacobian ∂delay/∂(T, q) can be examined.
3. **Station observations**: ASOS/AWOS (IEM), NOAA USCRN, NRCS SNOTEL, IGRA radiosondes, and optionally state mesonets (licensed). The reader must keep height, QC flags, exposure and units.
4. **A GNSS troposphere reader**: NGL 5-min ZTD, gradients, IWV and Tm. (The NGL README warns that the N/E gradient columns are swapped in its SINEX files.)
5. **Terrain derivatives**: slope, aspect, TPI at several radii, cold-air-pool indices (Lundquist 2008), sky-view factor.
6. **Surface layers**: NLCD canopy and impervious, NDVI, and optionally Landsat or ECOSTRESS LST, labelled as *surface* temperature.
7. **A temperature validation harness**: leave-station-out, spatial-block and temporal holdouts; distance-binned error structure functions; uncertainty calibration.
8. **A radar-observation exporter** that writes observed phase, model delay (wet and hydrostatic kept apart) and residual as *separate* labelled products. It must record which steps (reference, deramp, DEM-error fit, masks) touched each one.

### 2.3 Proposed integration

- New sub-package `research/atmospheric_temperature/`, importable as `insar_pipeline.research.atmospheric_temperature`. It **reads** pipeline outputs and calls existing helpers. It does **not** change any stage, default or product of the existing routes.
- Hooks are read-only. It imports `tropo_disp`, `weather_raider`, `landcover`, `data_acquisition.get_dem`, `nisar_backend` readers, `lband.Grid` and `provenance` helpers.
- One possible later change to shared code: an *opt-in* flag for RAiDER to keep wet and hydrostatic separate when writing per-date LOS delay (`write_delay_file`). Default behaviour would be unchanged, and it would only be proposed if Phase 4 needs it.
- Configuration: a research YAML (e.g. `~/insar_projects/configs/atmos_<region>.yaml`) parsed by the research module's own dataclass. `PipelineConfig` is not extended, so the pipeline's validation stays untouched.
- Tests: `tests/test_atmos_*.py`, using synthetic fixtures that are labelled synthetic.
- Outputs go under `~/insar_projects/<research_project>/atmos/`, never inside the repository.

---

## 3. Conflicts between pipeline assumptions and atmospheric research

The pipeline was built to *remove* atmosphere. This research needs to *keep* and *characterise* it.

| # | Assumption in the pipeline | Effect on atmospheric research | Mitigation |
|---|---|---|---|
| C1 | `mintpy_deramp: linear` (default) | Removes long-wavelength atmosphere and any stratified part correlated with position | Use the pre-deramp series, or the DISP-S1 raw series |
| C2 | One reference pixel and one reference date | Every observation is a double difference. The absolute per-date delay is never observed, and the reference date's atmosphere leaks into every date. | Work with spatial differences only; estimate the reference-date field as minus the temporal mean of residuals; or anchor with GNSS ZTD |
| C3 | Troposphere corrected *before* the DEM-error and polynomial deformation fit | `timeseriesResidual` = residual after HRRR, minus whatever the quadratic-in-time and DEM-error fit absorbed | Recompute residuals from `timeseries.h5` − model, with a deformation model fitted separately and recorded |
| C4 | Wet and hydrostatic summed in `inputs/<MODEL>.h5`; 1/cos(inc) projection | Separation is lost on the radar grid; small slant error in relief | Read the cached cubes directly; ray-tracing (needs `.EOF` orbits) only if Phase 1 shows the error matters |
| C5 | AOIs sized for mines (~10–30 km) | Too small for the turbulent spectrum beyond ~10 km, and contain 0–2 stations | A new research AOI of ≥100 km (Section 6) |
| C6 | Coherence masks and the farmland drop | Summer farmland decorrelates (already documented in this project), so the atmospheric sampling is uneven | Accept the gaps; record the sample masks |
| C7 | Phase-linking DS bias and the skipped closure correction (both documented in this project) | Adds a slowly varying, land-cover-dependent phase that is *not* atmospheric, and correlates with land cover (a confound!) | Use short-temporal-baseline or 12-day differences for atmospheric analysis; treat land cover as a nuisance covariate |
| C8 | Deformation is the signal | For this research, deformation is a confound (mine subsidence, seasonal soil swell, thermal dilation of structures [D: Monserrat 2011]) | Choose a region without known deformation; model and mask it |
| C9 | Project data are client data | Reusing client AOIs for research raises a confidentiality question | Kyle's decision. The default is a separate, public research AOI. |

---

## 4. Scientific feasibility

### 4.1 Observation physics (arithmetic, P = 1000 hPa, T = 293 K, e = 15 hPa, Thayer constants)

Refractivity: N = k1·P/T + k2′·e/T + k3·e/T², and the one-way delay is 10⁻⁶ ∫N ds.

**Baseline refractivity:** N_hyd ≈ 261–265 and N_wet ≈ 66.

**Hydrostatic term**

- k1·P/T = k1·R_d·ρ, so ∫N_hyd dz = k1·R_d·P_s/g.
- The integrated hydrostatic delay depends **only on surface pressure**: ZHD ≈ 2.2768 mm/hPa × P_s [D: Saastamoinen 1972; Davis 1985].
- At a fixed pixel and fixed P_s, a warmer layer changes ZHD by **zero** to first order.
- Temperature enters only through the *difference* between pixels at different heights. That difference follows the hypsometric equation, through the mean virtual temperature of the layer between them:

| Relief between pixels | ΔZHD | Change per 1 K of layer-mean T_v |
|---|---|---|
| 50 m | 13 mm | 0.045 mm |
| 100 m | 26 mm | 0.09 mm |
| 500 m | 129 mm | 0.44 mm |
| 1000 m | 250 mm | 0.85 mm |

**Wet term**

- dN_wet/dT is about −0.45 N/K at fixed e, and about +3.8 N/K at fixed relative humidity.
- So a 1 K anomaly in a 100 m layer changes the zenith delay by **−0.045 mm** at fixed e and **+0.38 mm** at fixed RH.
- A **1 hPa change in e** in the same layer gives **0.44 mm**.
- Humidity therefore dominates: 1 hPa of e (≈0.6 g/kg, ≈6 % RH) is worth ~10 K at fixed e.

**Cold-air pool, best case**

- A pool 5 K colder and 100–200 m deep at dawn gives ~0.3–1 mm, mostly hydrostatic-hypsometric, between floor and rim.
- Two independent calculations, mine and one literature agent's, agree.

**Noise**

| Source | Magnitude |
|---|---|
| Single-date multilooked phase noise | a few mm |
| Turbulent wet delay | ~10 mm at 10 km, mm-level at sub-km [D: Emardson 2003; Hanssen 2001] |
| Residual L-band ionosphere | mm to cm |

- A single date cannot detect a T signal. Only statistical or regression estimates across many pixels (a phase–elevation slope) or many dates could reach sub-mm precision, and they carry the humidity degeneracy with them.

### 4.2 The eight required questions

1. **What radar can infer.**
   - Spatial (and date-to-date) differences of the *total* slant delay. After model removal, this is the residual wet delay, i.e. differential PWV at ~1–2 mm PWV, tens of m to km resolution [D: Hanssen 1999; Mateus 2017/2020; Alshawaf 2015].
   - At high-relief sites, the stratified part (the delay–elevation relation) is also observable [D].
   - Hydrometeor/convective delays up to tens of cm at L-band are observable [D: Oliver-Cabrera 2025, abstract].
2. **What needs independent constraints.**
   - Absolute delay per date (needs GNSS ZTD or a model).
   - The split into hydrostatic and wet (needs surface pressure or a model).
   - Any temperature (needs a humidity constraint *and* a pressure constraint).
   - Vertical distribution (needs a profile: radiosonde, microwave radiometer, or a model).
3. **Hydrostatic and wet separation.**
   - Not separable from radar alone: both are non-dispersive and identical in frequency dependence [D].
   - The differential hydrostatic signal is essentially elevation-correlated surface pressure, so the split is *partially* constrained only through the model plus terrain.
4. **Assumptions needed to relate delay to 2 m T.** All of these at once:
   - a known humidity field (or a fixed-q or fixed-RH closure);
   - known surface pressure;
   - a vertical structure model linking layer-mean T to 2 m T (the surface-layer and inversion shape);
   - no deformation, ionosphere or DS-bias leakage;
   - representativeness between a slant-path column average and a 2 m point.
   None of these is supplied by radar.
5. **What is lost in the integrated path delay.**
   - All vertical structure: a 1-D profile collapses to one number per pixel, so inversions and elevated moist layers are indistinguishable.
   - The T/q partition.
   - Any absolute level (in the differential case).
   - Time resolution: 2 instants per 6–12 days per track.
6. **When radar might add information independent of ERA5/HRRR [H].**
   - Sub-model-grid (<3–9 km) moisture structure: convective cells, sea breezes, valley moisture pooling. Shown for HRES-like models [D: Murray 2019; Oveisgharan 2026, where model screens fail to reproduce dominant variability].
   - Model errors in nocturnal or dawn stratification at high-relief sites (cold-air pools that models mis-place). This is the only route that touches temperature directly, and it is sub-mm per K.
   - Mis-timed fronts or outflow boundaries at overpass time.
   - In flat, humid terrain the T signal is effectively nil.
7. **Supportable scales.**
   - Atmospheric delay information: ~1–50 km horizontally, at 2 instants per revisit. Overpass times measured on the local projects:
     - S1 ascending ~00:53 UTC, ≈18:53 local;
     - S1 descending ~13:01 UTC, ≈07:01 local.
     These are close to the 00/12 UTC radiosonde times, which is a useful coincidence.
   - Surface radar features: ~40 m resolution (the pipeline's 11×3 looks).
   - **80 m atmospheric information is not supportable.**
8. **Identifiability.**
   - Identifiable: the residual differential total delay field (km scale); the stratified delay–elevation coefficient per date, given relief.
   - Partially identifiable: wet versus hydrostatic, with a model; differential PWV, given Tm.
   - Unidentifiable from radar: T versus q in the column; vertical profile; absolute state; 2 m T at any scale without strong priors.
   - The proposed 𝒥(x) with x = full 3-D state is badly underdetermined. Any SAR term would update mostly q, unless the background-error covariance couples T and q, in which case the T increment is a *prior assumption*, not an observation.

### 4.3 Oliver-Cabrera et al. (2025)

Only the abstract and bibliographic record could be read. Wiley returns 403 to automated fetches, and no open preprint was found. Everything below is from the abstract; the methods detail is unverified.

- UAVSAR L-band interferograms matched against NOAA NEXRAD reflectivity.
- High-reflectivity masks coincide with InSAR phase-outlier masks; delays reach up to 25 rad, about 48 cm one-way (L-band).
- NEXRAD's 5-min scans, against UAVSAR's ~30-min revisit, catch fast-developing clouds.
- GACOS (HRES) lacks the resolution to reproduce these delays.

What it is:

- A **detection/flagging** method for hydrometeor-associated wet delay.
- It is **not** a quantitative state retrieval, and it says nothing about temperature.

How it adapts here [P]:

- As a **QC screen**: drop or flag dates where NEXRAD shows reflectivity in the scene at overpass time. These dates are dominated by liquid water, ice and convective vapour, and would swamp any sub-mm temperature signal.
- NEXRAD Level-II/III archives are public on AWS. A screen is cheap (a few reflectivity scans per date).

To read before relying on it: the full text (Kyle's institutional access, or a PDF placed locally).

### 4.4 OPERA TROPO (OPERA_L4_TROPO-ZENITH_V1)

**What it is**

- Global zenith hydrostatic and wet delay cubes from **ECMWF HRES** via RAiDER: 0.07° (~8 km), 00/06/12/18 UTC, about 2–3 days latency.
- Record from 2016 (CMR says 2016-01-01; the JPL page says July 2016, which conflicts).
- NetCDF-4, released 19 Sep 2025, DOI 10.5067/SNWG/OPL4TROZEN-V1, distributed by ASF DAAC.
- Vertical levels: 145, from −0.5 to ~81 km, from a Fringe 2026 talk [UNVERIFIED against the spec, which is an image-only PDF].

**It contains no InSAR information. It is a model product.**

**Use here**

- An independent second background delay (HRES), to sit alongside HRRR (RAiDER, already in the pipeline) and the NISAR GUNW screens.
- A baseline the SAR residuals must beat.
- At 6-hourly cadence it must be interpolated to overpass time (≤3 h), and that interpolation must be recorded.
- It carries no T, q or P layers, as far as every description says [UNVERIFIED in the spec table].

**Validation status:** a GNSS comparison at ~25 km scale is not peer-reviewed. Independent work finds model screens miss the dominant InSAR variability [D: Oveisgharan et al. 2026, JGR-A].

### 4.5 Other evidence that bears on the design

- **InSAR PWV assimilation** [D]: Pichelli 2015 (MM5), Mateus 2018 and 2021 (WRF 3D-Var). Forecast PWV improved against GNSS. Precipitation gains were not significant on average. Temperature impacts were not reported as a target.
- **GNSS meteorology** [D]: Bevis 1992 Tm–Ts gives PWV/ZWD ≈ 0.15–0.165. A 1 K error in Ts changes PWV by ~0.06 mm. The link runs from Ts into the conversion, not out of the delay.
- **Fine-scale T2m downscaling**: typical hourly RMSE 1–2.5 °C. It is worst on calm, clear nights in complex terrain [P, synthesis].
  - Dense crowdsourced urban networks (Venter 2020, Oslo, 1,310 stations) reach Tmin RMSE ~1.5 °C at 10–30 m pixels. Their predictors average over 100–500 m, and skill collapsed below ~1 station/km² [D].
  - No peer-reviewed ERA5-Land → ~100 m hourly product validated on independent stations was found.
  - Mechanistic microclimate downscaling (microclima, 100 m) is the strongest physics-based non-radar comparator [D for capability].
- **LST is not T2m** [D: Mildrexler 2011; Good 2017]. Daytime LST over bare ground exceeds Tmax by tens of °C. The coupling breaks with wind, canopy and cloud.
- **Independence**: HRRR, RAP and RTMA/URMA assimilate METAR and mesonet data via MADIS [D: Benjamin 2016; De Pondeca 2011], so they are **not independent** of station validation. ERA5 assimilates SYNOP T2m, so ASOS stations overlap with it [P]. Validation stations must be chosen with this in mind (Section 7).
- **Missions (as of Oct 2026)**:
  - Sentinel-1 is now S1C + S1D, with S1A operations ended June 2026 [D, ESA/ASF notices; 6-day revisit status UNVERIFIED].
  - NISAR is dawn-dusk, 12-day. Provisional GUNW and GSLC are on ASF. This project's own search found T019A GSLC dates from 2025-10-30, while ASF's release notes describe provisional coverage from June 2026; the two conflict and need checking.

### 4.6 Verdict

| Claim | Status |
|---|---|
| InSAR maps residual differential wet delay / PWV at km scale beyond models | **[D]** |
| Residual delays carry boundary-layer *moisture* structure that HRRR/ERA5 miss at overpass time | **[D/P]**, site and season dependent |
| Residual delays carry a detectable *temperature* signal (via hypsometric cold-pool effects) | **[H]**, sub-mm per K; plausible only for deep, strong pools in high relief, statistically over many dates |
| Adding InSAR delay features improves station-validated T2m beyond a non-radar baseline | **[H]**, untested; physics predicts small or null |
| Radar surface features (coherence, backscatter) improve T2m downscaling | **[H/P]**; if it works, it is a land-surface proxy, not atmosphere |
| A reliable ~80 m T2m field with validated 80 m effective resolution | **Not supportable** with any available station network; the validated scale is set by station spacing (≥ several km for ASOS/mesonets) |

---

## 5. Unverified assumptions (to check before or while implementing)

| # | Assumption | Status / where it is checked |
|---|---|---|
| U1 | MintPy `timeseriesResidual.h5` here is computed from the HRRR-corrected, deramped series | MintPy's step order; verify from `smallbaselineApp.cfg` and file attributes in Phase 1 |
| U2 | Cached RAiDER `hydro` and `wet` follow RAiDER's documented definitions (zenith, one-way, metres) | Check against Saastamoinen ZHD from station pressure in Phase 1 |
| U3 | RAiDER `center_time` interpolation between the two HRRR hours is adequate at dawn transitions | Compare with the hourly fields in Phase 2 |
| U4 | Station networks chosen are independent of the background used | ASOS feeds HRRR and ERA5. USCRN, SNOTEL and the state mesonets are partly independent of ERA5 (per network, UNVERIFIED). |
| U5 | The deformation in the research AOI is negligible or modelled | AOI choice; check with DISP-S1 or GNSS |
| U6 | NEXRAD screening removes hydrometeor-dominated dates | Only the abstract was read, so this is unverified |
| U7 | OPERA TROPO carries no T, q or P fields; 145 levels | Spec table unreadable |
| U8 | NISAR overpass ~06/18 local; S1 6-day revisit active | Check from granule metadata, as the pipeline already does |
| U9 | Licensing permits commercial-firm use of the station data | Oklahoma and NYS Mesonet charge non-state or commercial users and bar redistribution; Synoptic's open tier is .edu only |
| U10 | Disk and compute for the radar side of a ≥100 km AOI fit the budget | Measure DISP-S1 granule sizes before asking for approval |

---

## 6. Data and candidate study areas (nothing downloaded)

### 6.1 Free, license-clean sources (preferred for a firm)

- **Stations:**
  - ASOS/AWOS via the IEM (1-min, 5-min, hourly);
  - NOAA USCRN (high-quality, sparse);
  - NRCS SNOTEL (mountain T, hourly);
  - RAWS (via WRCC; terms UNVERIFIED);
  - IGRA radiosondes (00/12 UTC, which coincides with S1 passes).
- **Background:**
  - ERA5/ERA5-Land: ARCO-ERA5 Zarr on GCS and the AWS Earthmover copy avoid the saturated CDS queue. That needs `gcsfs` or `s3fs` and `zarr`, which is a dependency decision.
  - HRRR on AWS via Herbie (already in `raider-env`, byte-range 2 m fields).
  - OPERA TROPO (ASF).
- **GNSS:** NGL 5-min ZTD, IWV, Tm (free).
- **Surface:**
  - GLO-30 (in the pipeline);
  - 3DEP 10 m;
  - NLCD (land cover, impervious, canopy);
  - WorldCover (in the pipeline);
  - Sentinel-2 NDVI;
  - Landsat C2 ST, ECOSTRESS.
- **Radar:**
  - OPERA DISP-S1. It is supported by `disp_backend.py`, has **no troposphere applied**, and so the atmosphere is intact, on frames of ~200 km.
  - NISAR GUNW (80 m, carries model screens and the ionosphere screen).
  - The pipeline's own MiaplPy route is expensive (hundreds of GB) and is not proposed for the first experiment.

### 6.2 Candidate AOIs

| AOI | Why | Against |
|---|---|---|
| **Salt Lake Valley / Wasatch (recommended for the physics test)** | Deep, frequent, documented cold-air pools (PCAPS); ~2 km relief; arid, so coherent; SLC radiosonde; ASOS + SNOTEL + USCRN-class free stations spanning elevations; dense GNSS. The best case: if no signal appears here, it is unlikely anywhere. | The dense MesoWest network is behind Synoptic licensing; urban and lake confounds; winter snow decorrelates C-band (L-band helps) |
| New York State | Mesonet 2 m + 9 m T everywhere; 17 profiler sites (microwave radiometer + lidar = column T/q truth) | Forest and snow decorrelation; licensing and no redistribution; weaker pools |
| Central Oklahoma | Best-QC mesonet (~120 sites); ARM SGP profiles | Weak relief, so a near-null T signal; rural grass siting under-samples land cover; fees |
| Existing local projects (two US sites with HRRR cubes cached, one with asc + desc S1 + NISAR) | Zero download; good for Phase 1 code and the noise budget | 10–30 km boxes, 0–2 stations, client AOIs (C9). **Not suitable for temperature validation.** |

---

## 7. Proposed initial experiment (smallest falsifiable test)

**E0, observability.** Does the residual delay after the model carry information about the station-observed near-surface refractivity anomaly, and about its *temperature* part, at overpass time?

1. **Region and period.** One AOI (recommended: Salt Lake Valley). S1 DISP-S1 frames for one ascending and one descending track over two winters (~60–120 dates per track). Exact frames and sizes are to be measured and approved first.
2. **Observed.** Per-date residual delay r(x, t) = DISP-S1 displacement-equivalent phase − HRRR modelled LOS delay. The existing `tropo_disp.differential_los_delay` is used, with wet and hydrostatic kept separate.
   - Remove the deformation estimate (a smooth temporal model per pixel).
   - Remove the ionosphere screen (C-band: none needed below ~100 km; record the decision).
   - Screen dates with NEXRAD reflectivity.
3. **Truth.**
   - At stations: observed surface refractivity N_obs = f(P, T, Td), and its anomaly against HRRR's N at the same point and time, split into T-only and e-only parts.
   - In the column: the SLC radiosonde profile (inversion strength, pool depth) and NGL ZTD and IWV.
4. **Tests**, pre-registered before any result:
   - **E0a.** Does the spatial difference of r between station pairs (valley floor against bench or slope) correlate with the station-pair difference of the HRRR refractivity anomaly? Yes means radar sees what HRRR misses.
   - **E0b.** Does r add predictive power for the *temperature* component, after the humidity component (from station Td and GNSS IWV) is in the model? This is the identifiability test.
   - **E0c.** Does the per-date stratified coefficient (residual delay against elevation) track the radiosonde or HRRR inversion error?
   - Placebo: shuffle dates, and use the opposite track's time.
5. **Pass/fail (proposed).** E0b passes only if the partial correlation's 95 % bootstrap CI (blocked by date) excludes zero *and* the placebo does not. A fail of E0b is a valid, reportable result. The plan then continues with the moisture and surface pathways only.

**E1, non-radar baseline.** Run in parallel and needed anyway. In the same AOI and period, at overpass instants and at all hours for context:

- Native ERA5, native ERA5-Land, and HRRR (labelled non-independent) bilinear to stations.
- A transparent downscaler: ERA5-Land plus lapse rate plus terrain and land-cover terms in a linear mixed or regression-kriging model.
- Gradient boosting only if a residual diagnostic justifies it.
- Validation: leave-station-out, spatial-block (NNDM-matched) and temporal holdout. Metrics are MAE, RMSE, bias, by class, by condition (calm and clear against windy), and an error structure function by station separation.

**E2 runs only if E0 passes, or to test the surface pathway.** Nested models:

- B = baseline;
- B + radar surface features (coherence, backscatter);
- B + residual-delay features;
- B + residual-delay + humidity controls.

They are compared on the same held-out stations and times.

---

## 8. Phased implementation plan (each phase gated; nothing runs without approval)

| Phase | Work | New code (under `research/atmospheric_temperature/`) | Data / compute | Gate |
|---|---|---|---|---|
| **1. Observation-operator spike** (no downloads) | Forward operator; reproduce RAiDER `hydro`/`wet` from first principles on a synthetic column; Jacobians ∂d/∂T, ∂d/∂q by layer; measured residual delay spectra and phase–elevation slopes from **existing** local cubes and `timeseriesResidual.h5`; minimum detectable layer-T anomaly given measured noise | `refractivity.py`, `radar_obs.py` (read-only exporters, labelled products, step-history record), `tests/test_atmos_refractivity.py` (synthetic, labelled) | 0 GB download; ~200 MB of existing cubes read; minutes of CPU | G1: an operator matching RAiDER to <1 % on a column, a measured noise budget, and a written detectability number. Kyle approves the AOI and the E0 protocol. |
| **2. Data interfaces** | Station readers (IEM ASOS, USCRN, SNOTEL, IGRA) with height, QC flags and exposure; background readers (HRRR via Herbie in `raider-env`, ERA5-Land via ARCO or CDS); NGL ZTD reader; terrain derivatives; NLCD | `stations.py`, `background.py`, `gnss_trop.py`, `predictors.py`, `provenance` records | Stations: <100 MB. ERA5-Land T2m for one AOI over 2 years: tens of MB. HRRR 2 m fields at overpass hours: ~1–2 MB each, ~0.5 GB in total. **Approval before any download.** | G2: every product carries source, units, CRS, times and interpolation notes; tests pass |
| **3. Baseline (E1)** | Native ERA5 / ERA5-Land / HRRR at stations; regression or mixed-model downscaler; CV harness; structure function | `baseline.py`, `validate.py`, protocol YAML (hash-fixed) | CPU minutes | G3: baseline metrics written down *before* any radar features are added |
| **4. Radar side for the AOI** | DISP-S1 frames for the AOI (disp_backend, read-only use); RAiDER HRRR cubes at overpass (existing `tropo_disp`); optionally OPERA TROPO; NEXRAD screen | Extends `radar_obs.py`; `nexrad_screen.py` | **Size to be measured and approved** (DISP-S1 frames, RAiDER cubes ~1–50 MB/date for a 100 km box [estimate]) | G4: observed, modelled and residual stored separately, with the step history |
| **5. E0 and E2 experiments** | Pre-registered E0a/b/c and the nested E2 comparisons, with placebos | `experiment.py` | CPU minutes to an hour | G5: the pass/fail rule applied as written. Negative results are written up. |
| **6. Physics state estimation** (conditional on G5) | Column 1D-Var: background (HRRR/ERA5 with error covariances) + SAR residual through the operator + station and GNSS terms; inspect T increments versus q increments; posterior uncertainty | `state_estimation.py` | Modest | Only if E0b passes |
| **7. Uncertainty and scale** | Uncertainty rasters (measurement, delay, parameter, structural, interpolation); scale-dependent skill; area of applicability | `uncertainty.py` | Modest | — |
| **8. Outputs** | Rasters labelled *observation / model constraint / reconstruction / validation*; run record | Uses `provenance.py` | — | No final-looking map before G5 and G7 |

**Dependencies.** Phases 1–3 need **none new**:

- the `insar` env has numpy, scipy, xarray, pandas, h5py, rasterio and pyproj;
- `raider-env` has Herbie, metpy, scikit-learn and s3fs.

Possible later additions, each to be argued when needed:

- `zarr` + `gcsfs` for ARCO-ERA5;
- `scikit-learn` in `insar` (or run the models in `raider-env`) for gradient boosting;
- a NEXRAD reader (`pyart` or raw Level-III parsing).

---

## 9. Main risks and failure modes

1. **A null temperature signal** (most likely outcome, by the physics). This is mitigated by the design: E0 is cheap, and a null result is a deliverable.
2. **Humidity masquerading as temperature.** Any T "gain" without humidity controls is uninterpretable. E0b and E2 require q controls.
3. **Leakage.**
   - HRRR, RTMA and ERA5 assimilate ASOS, which inflates baseline skill.
   - Random CV splits inflate it too.
   - Spatial coordinates as predictors overfit.
4. **Confounds that correlate with land cover and elevation**: DS bias, seasonal soil and vegetation phase, deformation, thermal dilation, snow. These can create spurious "temperature skill".
5. **Representativeness.** Point 2 m stations against slant-column delays against 80 m pixels; deliberately rural mesonet siting.
6. **Sampling.** 2 instants per revisit per track. Results cannot be extrapolated to other hours.
7. **Ionosphere at L-band** (NISAR), with residual split-spectrum noise comparable to the target.
8. **Data access.** The CDS queue; licensing for mesonets; NISAR record length and calibration state.
9. **Disk and compute** if the radar side drifts toward a full MiaplPy run. Avoided by using DISP-S1 for E0.
10. **The 80 m claim.** No available network validates 80 m. Reports must give the validated scale (station spacing), not the posting.

---

## 10. What to implement first, and why

**Phase 1: the refractivity forward operator, plus a read-only radar-observation exporter, run on data already on disk.**

- It needs no download, no new dependency and no change to the pipeline.
- It turns the question "can radar see temperature?" into a number: the minimum detectable layer-temperature anomaly given the measured residual-delay noise on our own data.
- It builds the two pieces every later phase needs: the operator, and labelled observation/model/residual products with their step history.
- If that number is far above physically plausible anomalies (as Section 4.1 predicts for anything but deep cold pools), the radar-temperature path can be closed early. Effort then goes to the baseline and the moisture and surface pathways, rather than into a model whose radar term could only fit noise.

---

## Sources

Section numbers in the literature sweeps refer to the session record. The key references with DOIs:

- **Refractivity and delay physics:**
  - Smith & Weintraub 1953, 10.1109/JRPROC.1953.274297
  - Thayer 1974, 10.1029/RS009i010p00803
  - Saastamoinen 1972, 10.1029/GM015p0247
  - Davis et al. 1985, 10.1029/RS020i006p01593
  - Hanssen 2001, 10.1007/0-306-47633-9
- **Tropospheric correction of InSAR:**
  - Doin et al. 2009, 10.1016/j.jappgeo.2009.03.010
  - Jolivet et al. 2011, 10.1029/2011GL048757
  - Jolivet et al. 2014, 10.1002/2013JB010588
  - Bekaert et al. 2015, 10.1016/j.rse.2015.08.035
  - Yu et al. 2018 (GACOS), 10.1029/2017JB015305
  - Murray et al. 2019, 10.1016/j.rse.2019.111326
- **InSAR water vapour and assimilation:**
  - Hanssen et al. 1999, 10.1126/science.283.5406.1297
  - Mateus et al. 2017, 10.1109/TGRS.2017.2658342
  - Mateus et al. 2018, 10.1002/2017JD027472
  - Mateus et al. 2021, 10.1029/2020JD034171
  - Pichelli et al. 2015, 10.1109/JSTARS.2014.2357685
  - Alshawaf et al. 2015, 10.5194/hess-19-4747-2015
  - Oveisgharan et al. 2026, 10.1029/2026JD047910
  - Oliver-Cabrera et al. 2025, 10.1029/2025EA004382 (abstract only)
- **Ionosphere:**
  - Gomba et al. 2016, 10.1109/TGRS.2015.2481079
  - Fattahi et al. 2017, 10.1109/TGRS.2017.2718566
  - Liang et al. 2019, 10.1109/TGRS.2019.2908494
- **GNSS meteorology:**
  - Bevis et al. 1992, 10.1029/92JD01517
  - NGL troposphere README: https://geodesy.unr.edu/gps_timeseries/readmes/README_trop2.txt
- **OPERA TROPO:**
  - 10.5067/SNWG/OPL4TROZEN-V1
  - https://www.jpl.nasa.gov/go/opera/products/tropo-product/
  - https://www.earthdata.nasa.gov/data/catalog/asf-opera-l4-tropo-zenith-v1-1
- **Temperature downscaling and validation:**
  - Lundquist et al. 2008, 10.1029/2008JD009879
  - Dobrowski et al. 2009, 10.1016/j.agrformet.2009.06.006
  - Holden et al. 2011, 10.1016/j.agrformet.2011.03.011
  - Venter et al. 2020, 10.1016/j.rse.2020.111791
  - Maclean et al. 2019, 10.1111/2041-210X.13093
  - Mildrexler et al. 2011, 10.1029/2010JG001486
  - Good et al. 2017, 10.1002/2017JD026880
  - Benjamin et al. 2016, 10.1175/MWR-D-15-0242.1
  - De Pondeca et al. 2011, 10.1175/WAF-D-10-05037.1
  - Roberts et al. 2017, 10.1111/ecog.02881
  - Meyer & Pebesma 2021, 10.1111/2041-210X.13650
  - Milà et al. 2022, 10.1111/2041-210X.13851
  - Lareau et al. 2013 (PCAPS), 10.1175/BAMS-D-11-00255.1
  - Brotzge et al. 2020 (NYSM), 10.1175/JTECH-D-19-0220.1
  - Shrestha et al. 2022 (NYSM profilers), 10.5194/amt-15-6011-2022
