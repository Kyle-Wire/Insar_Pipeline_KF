# External services register

Every outside service the pipeline contacts: who runs it, when it is used,
what leaves the machine, how it authenticates, and whether its use is
approved. Built from the code on 16 Oct 2026 (readiness review item NET-02);
change it whenever the code adds, removes or changes a call.

**Approval status:** none of these has been reviewed by Information Security
yet. "What the provider keeps" has not been reviewed either: each provider's
own terms and privacy notice apply.

**Rule for users:** type a town or a road in the builder's search, never a
client's or a site's name. Every service below receives the site's location
in some form, but none receives a client's name, a report or report numbers.

No telemetry, analytics, crash reporting, licence check or update check
exists in the pipeline.

## During a run (needed for every phase-linked project)

| Service | Provider | Used for | What is sent | Authentication |
|---|---|---|---|---|
| ASF search and NASA CMR | NASA / Alaska Satellite Facility | finding Sentinel-1 scenes and bursts | the area outline or processing box, dates, track | none for search |
| ASF data download | NASA / Alaska Satellite Facility | downloading radar scenes and bursts | file names (which reveal track and dates) | NASA Earthdata login (`~/.netrc`) |
| Sentinel-1 orbits | ASF (via `sentineleof`; ISCE2 can also fetch them while it runs) | precise orbit files | the scene names (mission and dates) | Earthdata login |
| Copernicus DEM (via `dem_stitcher`) | ESA / AWS open data | elevation for the processing box | the tiles covering the box | none |
| ERA5 weather (Copernicus CDS) | ECMWF | weather delay correction | the area, padded by 2 degrees and rounded out to 10 degrees, and the dates | CDS API token (`~/.cdsapirc`) |
| GNSS station series | Nevada Geodetic Laboratory (UNR) | the regional GNSS tie | station IDs | none |
| JPL station velocity table | NASA JPL | GNSS validation (when on) | nothing site-specific | none |
| ESA WorldCover | ESA / AWS open data | which measured cells are wooded | the tiles covering the area | none |
| Overpass (OpenStreetMap) | overpass-api.de | the water map that checks map positions | the processing box | none |
| Overpass mirrors | overpass.kumi.systems, overpass.private.coffee (third parties) | the same, when the main server is busy | the processing box | none |
| Nominatim reverse search | OpenStreetMap Foundation | the place name in the report | the area's centre (5 decimals); the firm and contact in the User-Agent | none |
| USGS topographic basemap | USGS National Map | the report's map background | the map tiles of the area | none |

## Only on some routes or settings

| Service | Provider | When | What is sent | Authentication |
|---|---|---|---|---|
| OPERA DISP-S1 products | NASA / ASF | the screening route and the cross-check | the area, dates; then file reads | Earthdata login |
| NISAR products | NASA / ASF | the NISAR route (not validated) | the area, dates; then file reads | Earthdata login |
| IGS ionosphere maps (MintPy `iono_tec`) | NASA CDDIS | areas of about 100 km or more, or `ionosphere_correction: gim` | dates | Earthdata login |
| HyP3 | NASA / ASF | the HyP3 route (example config only) | pairs of scene names | Earthdata login |
| RAiDER weather models | ECMWF / NOAA, by model | the DISP-S1 weather correction | the area and dates | as the model requires |
| Geoscience Australia spreadsheet | Geoscience Australia | the Appin validation experiment only | nothing site-specific | none |

## In the browser pages (map builder, results map)

| Service | Provider | What is sent |
|---|---|---|
| Map tiles | USGS (default), Esri World Imagery, OpenStreetMap or OpenTopoMap, as chosen | the tiles of whatever area is on screen |
| Nominatim search | OpenStreetMap Foundation | the text typed in the builder's search box |
| Leaflet map library | cdnjs (Cloudflare) | the request only (IP address); the map builder only - the results map serves its own copy (`viewer_assets/vendor/`, 19 Oct 2026) |
| Fonts | Google Fonts | the request only (IP address); the map builder only - the results map uses the system fonts |

## At install

| Service | What is fetched | Integrity check |
|---|---|---|
| GitHub (conda-forge Miniforge installer) | the Miniforge installer, version "latest" | none (open item NET-04) |
| conda-forge | the pinned packages of `install/envs/*.yml` | conda's repodata hashes |
| PyPI | the pinned pip packages | none beyond pinning (open item DEP-02) |
| GitHub (MiaplPy) | MiaplPy at a pinned commit | commit pin, patch check, functional test |
| Stanford (snaphu source) | snaphu 2.0.7 source | pinned sha256 |
| GitHub (this repository) | the pipeline itself | git |

## On the client's computer

The client KMZ refers to two icons on `maps.google.com` (over https; the
opt-in technical KMZ to a third). Google Earth on the client's computer
fetches them when the file is opened. Nothing from the pipeline is involved.

The Earthdata session cookies GDAL keeps for the remote reads live in
`~/.cache/insar_pipeline/earthdata/` (readable by the user only).

## Open decisions

- The two third-party Overpass mirrors receive the processing box without
  any agreement with their operators. Proposed: remove them, and let the
  geolocation check report "not run" when the main server is busy.
- The firm name and contact go to OpenStreetMap in the User-Agent. Nominatim's
  usage policy asks for an identifying contact, so the choice is that contact
  or not using Nominatim.

Code locations: `data_acquisition.py` (ASF, CMR, orbits, DEM), `stack_backend.py`
(ERA5), `gnss_tie.py` and `gnss_validation.py` (NGL, JPL), `landcover.py`
(WorldCover), `stack_qc.py` (Overpass), `areas.py` and `report_insights.py`
(Nominatim, USGS basemap), `disp_backend.py`, `nisar_backend.py`,
`hyp3_backend.py`, `corrections.py` (optional routes), `ground_truth.py`
(validation), `project_builder.html` and `results_viewer.html` with
`viewer_assets/viewer_app.js` (browser; `results_viewer.TILE_HOSTS` is the
list its Content-Security-Policy allows),
`install/` (install).
