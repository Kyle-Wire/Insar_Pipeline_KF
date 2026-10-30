# Phase 1 results: forward operator, radar-observation exporter, measured detectability

Run on 2026-10-28, branch `research/atmospheric-temperature`. Plan: [ASSESSMENT.md](ASSESSMENT.md) §8, Phase 1.
Nothing was downloaded. The two projects were read only.

Outputs are local, outside the repository: `~/insar_projects/_research/atmos_phase1/<label>/`, holding `radar_obs.h5/.json`, `observability.json/.png`, and `phase1_summary.json`.

To reproduce:

```
python -m insar_pipeline.research.atmospheric_temperature.run_phase1 <project> [<project> ...] --out <dir> --label <name> [...]
```

## 1. What was built

| Module | What it does | How it was checked |
|---|---|---|
| `refractivity.py` | State (P, T, e on levels) → zenith hydrostatic and wet delay. Hydrostatic pressure is rebuilt from T, so a temperature change keeps the column balanced. Includes Jacobians, layer/humidity/relief responses, Saastamoinen, and a SYNTHETIC test column. | Matches **RAiDER 0.6.0's own code** (`_get_*_refractivity`, `_getZTD`, run in `raider-env` via `raider_crosscheck.py`) to about 1e-15 relative. The integral matches Saastamoinen within 0.5 %. |
| `radar_obs.py` | Read-only exporter. Writes `model_hydro`, `model_wet` and `residual` as separate, labelled products in the observation's own convention. Also writes the step history, the observation's sha256, and a check against the correction MintPy actually applied. Refuses to write inside a project. | Reproduces MintPy's applied correction (see below). On synthetic data it catches a sign error and returns the injected motion plus noise. |
| `observability.py` | On short-interval date pairs: structure function; per-date residual slope against height with a spatial-block bootstrap; site sensitivities from the operator. Writes `observability.json`. | Synthetic tests: a ramp, a known slope, the gap rule. |
| `tests/test_atmos_phase1.py` | 14 tests, all on SYNTHETIC data. | 14 pass. |

No pipeline module, default or product was changed.

**U1 resolved.** On the S1 project, `timeseriesResidual.h5` comes from `timeseries_HRRR.h5` with `mintpy.deramp = no`, followed by DEM error and a quadratic-in-time deformation fit. The exporter does not use it: its residual is the observation minus the model, before any of that.

## 2. Data used (existing projects, labels only)

| Label | Sensor | Dates (pairs ≤24 d) | Grid | Relief (p2–p98) | Site pressure (HRRR cube) |
|---|---|---|---|---|---|
| `site_e_s1_desc` | Sentinel-1 C-band, descending, ~07:00 local, MiaplPy | 292 (290) | ~42 × 39 m radar grid, 92,880 coherent px | 571 m | 799 hPa |
| `site_e_nisar_pl` | NISAR L-band, `nisar_pl` (dolphin), split-spectrum ionosphere already removed | 25 (24) | 15 m lat/lon, 460,784 px | 371 m | 797 hPa |

Both cover the same ~10–14 km area. They are client AOIs: the numbers stay in this note, and no names or coordinates appear in tracked files.

## 3. Results

### 3.1 The model is what MintPy applied (outcome check)

The HRRR cubes, sampled and projected by the exporter, reproduce `observation − timeseries_*_HRRR.h5` at:

- **S1:** RMS **0.0008 mm** over 27.0 M pixel-dates.
- **NISAR:** RMS **0.009 mm** over 11.0 M pixel-dates.

A few pixel-dates differ by more than 0.1 mm:

- **S1:** 1 pixel-date (20211015), with a 4 mm difference.
- **NISAR:** 85 pixel-dates on 4 dates (20251111, 20251205, 20260215, 20260510), up to 6.4 mm.

At each of these, MintPy's *applied* value is the odd one out against its neighbours; the exporter's model is smooth there. The cause is **not identified**. The share is below 1e-5, so the check counts them and lists them rather than failing. It is worth a look on the pipeline side (`weather_raider.write_delay_file` → MintPy `diff.py`), but it does not affect this result.

### 3.2 Structure of the residual (observation − HRRR), per pair of consecutive dates

Values are the median RMS difference in mm, LOS:

| Separation | S1 residual | S1 HRRR model | NISAR residual | NISAR HRRR model |
|---|---|---|---|---|
| ~40 m (S1) / 16 m (NISAR) | 1.6 | 0.10 | 7.0 | 0.02 |
| ~80 m (87 / 69 m) | 2.0 | 0.19 | 9.6 | 0.10 |
| ~500 m | 2.7 | 0.72 | 10.1 | 0.50 |
| ~1 km | 3.4 | 1.4 | 10.3 | 0.9 |
| ~4 km | 5.3 | 3.6 | 10.8 | 2.1 |
| ~10 km | 6.7 | 5.1 | 11.5 | 3.4 |

- **S1:** the residual grows with separation, which is unmodelled atmosphere on top of a ~1.5 mm floor.
- **NISAR:** the residual is flat at 7–11 mm at every scale, so the 15 m phase-linked L-band series is dominated by pixel noise here.
- **HRRR:** carries almost no structure below 1 km. That is expected for a 3 km model sampled at pixel heights.

### 3.3 Detectability per date (gate G1)

These assume dates are independent, so a per-date figure is the pair figure divided by √2. Sensitivities come from the operator on SYNTHETIC columns started at the site's lowest ground and pressure (268 K and 293 K surface), projected to LOS.

**Pixel route.** Two pixels about 80 m apart differ only if the surface layer differs between them.

| | S1 (87 m) | NISAR (69 m) |
|---|---|---|
| Per-date noise | 1.38 mm | 6.8 mm |
| 1 K in a 100 m layer, fixed e | 0.015–0.056 mm | 0.014–0.052 mm |
| 1 K in a 100 m layer, fixed RH | 0.12–0.45 mm | 0.11–0.41 mm |
| **Smallest detectable ΔT (2σ), fixed e** | **49 K** | **263 K** |
| Smallest detectable ΔT (2σ), fixed RH (really a humidity change) | 6 K | 33 K |
| 1σ expressed as vapour pressure in the layer | 2.0 hPa | 10.5 hPa |

**Stratified route.** The residual slope against height, per date, across the relief.

| | S1 (571 m) | NISAR (371 m) |
|---|---|---|
| Slope precision (block bootstrap, 2 km blocks) | 2.0 mm/km | 3.4 mm/km |
| Response to 1 K layer-mean anomaly (best case) | 1.2–2.9 mm/km | 1.1–2.9 mm/km |
| **Smallest detectable layer-mean ΔT (2σ), if humidity were known exactly** | **1.4 K** | **2.4 K** |
| Observed residual slope spread (robust) | 10.7 mm/km | 12.2 mm/km |
| … if it were all temperature | 3.8 K | 4.2 K |
| … if it were all moisture | 1.4 hPa e | 1.8 hPa e |
| Residual slope vs HRRR's own slope change: gain / r | −0.28 / −0.39 | −0.33 / −0.24 |

**Seasonality (S1).** The median |residual slope| peaks in Jul–Sep at 14–16 mm/km and is lowest in Jan–Mar at 5–8 mm/km.

### 3.4 What this says

1. **The 80 m atmospheric route is closed.** At this site, at ~80 m, one date's residual noise (1.4 mm C-band, 6.8 mm L-band) is 25–100× the response to a 1 K surface-layer anomaly at fixed humidity. The same noise equals about 2 hPa of vapour pressure. This is measured on our own data, and it agrees with ASSESSMENT.md §4.1.
2. **The stratified route is detectable but not attributable.** The C-band slope is precise enough (1.4 K at 2σ) that a layer-mean temperature error of a few K *would* show, if humidity were known. The observed residual stratification is about 4 K if read as temperature, or about 1.4 hPa if read as moisture. Both are plausible HRRR errors at a ~07:00 overpass. Two things lean the reading toward moisture:
   - The summer peak. Moisture-season stratification errors would be expected to peak in summer; dawn-inversion temperature errors would be expected to peak in winter.
   - The negative gain. Residual slopes partly cancel HRRR's own slope changes, so HRRR's day-to-day stratified delay is about 30 % too large in amplitude. That is consistent with a humidity-profile bias.

   Neither is proof. Separating the two needs an independent humidity constraint at two elevations (GNSS IWV, or station dewpoint on the valley floor and at height). That is E0b in ASSESSMENT.md §7.
3. **L-band at 15 m is not the instrument for this.** It is noise-dominated at every scale. Its 24 pairs are also too few to resolve seasonality. Any NISAR use would need coarser looks and a longer record.
4. **Relief helps the stratified route.** Precision in mm/km improves with relief, while the per-K response per km stays roughly constant. An AOI with ~2 km relief (the Salt Lake option in ASSESSMENT.md §6.2) should push the 2σ floor below 1 K. This is an estimate, not a measurement.

## 4. Caveats

- Per-date figures assume consecutive dates are independent (÷√2). If they are correlated, per-date noise is understated.
- The residual still holds anything that changes within 12–24 days and correlates with height or position: fast deformation, phase-linking bias, unwrapping errors, snow. None was modelled.
- The sensitivities use SYNTHETIC columns (lapse rate 6.5 K/km, RH 0.6 decaying with height), not HRRR's actual profiles, because the cubes keep no T, q or P. The 268/293 K pair brackets the temperature dependence.
- One site, two sensors. Both AOIs are small (10–14 km) and hold no weather stations, so nothing here tests temperature *skill*. It only tests whether a signal could be seen.

## 5. Gate G1

| Criterion (ASSESSMENT.md §8) | Result |
|---|---|
| Operator matches RAiDER to <1 % on a column | **Met** (1e-15) |
| Measured noise budget | **Met** (§3.2) |
| A written detectability number | **Met.** Pixel route at 80 m: ≥49 K (C-band, fixed e), closed. Stratified route: 1.4 K (C-band, 571 m relief) if humidity is known, otherwise not attributable. |

**Recommendation for Phase 2.**

- Drop the pixel-scale atmospheric route.
- Keep the stratified route only as E0b: residual delay–height slope against *independently measured* humidity and temperature at two elevations.
- Choose the E0 AOI for relief (≥1.5 km) and for stations or GNSS at both the valley floor and at height.
- The non-radar baseline (E1) and the radar *surface* features stay as planned.
