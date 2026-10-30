# Geohazards research — plan (28 Oct 2026)

Hobby branch `research/geohazards`, for Kyle only. No client deliverables and no changes to the
pipeline routes. Each gate has a pass test, and nothing past a gate gets built until that gate passes.

## The idea

Use public seismograph data (EarthScope / USGS networks, not our own instruments) and machine
learning to find small, noisy earthquakes the official catalog misses. Then track where they
happen and how the rates change over time, first for earthquake zones and then for volcanoes.
Later, connect that to InSAR (seismic activity points the radar at an area) and add slope
instability.

## Smoke test, 28 Oct (done, before any plan was written)

`smoke_phasenet.py`: pretrained PhaseNet (SeisBench, "instance" weights) on whole days of
New Madrid network (NM) broadband data from 17 stations, compared with the USGS catalog.

| Day | Catalog events | Found | New real events |
|---|---|---|---|
| 2026-10-02 | 2 (M1.77, M2.19) | 2 | 0 |
| 2026-10-03 | 1 (M2.29) | 1 | 0 |
| 2026-10-07 | 2 (M1.52, M1.97) | 2 | 0 |

- **The machine works:** PyTorch 2.11 + CUDA 12.8 on the RTX 5070 Ti. That card needs a cu128 or
  newer build; older builds don't support it. The throwaway env was 7.3 GB.
- **Speed:** download about 100 s and picking 30–50 s per 17-station day, so a month takes about
  75 minutes.
- **What it proved:** we can reproduce the catalog down to M1.5.
- **What it didn't prove:** that we can find more than the catalog. The "extra" candidates were the
  S wave of the same quake 6–8 s later, split off by the crude 6-second grouping.
- **Noise:** about 450 P "picks" a day, but only 2–3 real quakes. Almost all single-station picks
  are noise, so the association step is where the real work is.

## Pipeline being built (research code, standalone)

1. **Data:** continuous waveforms from EarthScope FDSN, one station-day at a time. Keep the picks
   and throw the waveforms away, so disk use stays small.
2. **Picker:** pretrained PhaseNet / EQTransformer first. Later, our own model fine-tuned on this
   region's analyst picks (Gate 3).
3. **Associator:** PyOcto or GaMMA in place of the 6-second window. It groups picks across stations
   into events and gives a first location.
4. **Location:** relative relocation (GrowClust or HypoDD) to sharpen clusters and faults.
5. **Magnitude:** local magnitude from amplitudes, calibrated against catalog events.
6. **Not-earthquakes:** quarry and mine blasts (daytime, near known mines, which our Indiana
   mine layers can screen), trains and storms.
7. **Trends:** event rates over time, the mix of small to large quakes (b-value), the smallest size
   reliably caught, and activity moving over time. Aftershocks removed before reading trends.

## Gates

**G1 — Reproduce (New Madrid, 1 month).** Proper associator and location.
- Pass: at least 90% of catalog events inside the network found, median location within about
  2 km of the catalog's, and every unmatched candidate looked at on a waveform plot.

**G2 — Go deeper.** Lower the thresholds and hunt for events the catalog doesn't have.
- Pass: at least twice the catalog's count over the month, with at least 90% of a random 50
  extras judged real by eye on waveform plots.
- Also required: the extras extend the magnitude–frequency line smoothly below the catalog's
  limit, and they are not bunched in daytime (bunching would mean blasts).
- Template matching (EQcorrscan) as an independent cross-check.

**G3 — Our own model.** Fine-tune the picker on regional analyst picks from the USGS phase data.
- Pass: it beats the pretrained picker on a held-out month (more true events, fewer false ones).

**G4 — Volcanoes.** Start with Kīlauea (HV network), then Mount St. Helens or Yellowstone.
- Volcanic signals differ from tectonic quakes: sharp volcano-tectonic events, slow long-period
  events, and continuous tremor.
- Pickers trained on tectonic quakes are expected to miss long-period events, so fine-tune on
  volcano data.
- Tremor has no picks at all, so it gets its own track: a measure of how hard the ground is
  shaking (RSAM).
- Pass: compared against the observatory catalogs (HVO / PNSN) the same way as G1–G2.

**G5 — Keep it up to date.** A daily job that adds yesterday's data and updates the trend charts.
- When rates jump in a watched area, it flags the area for an InSAR look. The pipeline already has
  ascending/descending decomposition, and DISP-S1 covers North America.
- Pass: two weeks running unattended with no gaps or duplicate events.

**G6 — Slope instability (later).** A slope activity watchlist, not an early warning.
- InSAR finds slowly creeping slopes and spots acceleration over months.
- Movement direction comes from ascending + descending passes, assuming the ground moves straight
  downhill on the elevation model, with a map of where the satellites can't see (slopes facing
  north or south).
- Context layers: slope angle, soil survey, rainfall over the previous days, soil moisture.
- Test sites: Slumgullion, CO (slow, well studied), then Portuguese Bend, CA (accelerating, to show
  where we lose the signal), then Kyle's mine highwalls, spoil piles and impoundments.
- Landslide risk levels come after this. A percentage only where past landslides can check it.

## Limits to state on every output

- This does not predict earthquakes or eruptions. A rate change means "this changed", not
  "something is coming".
- Location quality depends on station spacing. New Madrid is dense; most of Indiana isn't, so
  depths there will be rough.
- Machine-learning picks include false events. Every catalog we produce is "candidate events"
  until checked.
- InSAR passes are 6–12 days apart, and only quakes of roughly magnitude 5 and up move the ground
  visibly.
- Slopes: InSAR cannot warn of sudden failure (see G6).

## Not decided yet

- Whether research code stays under `research/geohazards/` or becomes a package with its own pinned
  env (`install/envs/seismic.yml`). Decide after G1.
