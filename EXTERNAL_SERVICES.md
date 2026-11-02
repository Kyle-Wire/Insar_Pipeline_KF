# External services register

Every outside service the pipeline contacts: who runs it, when it is used,
what leaves the machine, how it authenticates, and whether its use is
approved. Built from the code on 16 Oct 2026 (readiness review item NET-02)
and last checked against the code on 2 Nov 2026; change it whenever the code
adds, removes or changes a call.

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
| NOAA HRRR weather (RAiDER with Herbie, in the `raider-env` environment) | NOAA's archive on AWS open data; when a file is not there, Herbie tries Google Cloud, Microsoft Azure, NOAA NOMADS and the University of Utah's Pando archive | weather delay correction in the continental US: the default for new phase-linked and stack-route projects since 24 Oct 2026 (`areas.weather_default`); also the DISP-S1 weather correction when it is on | the model file names (the date and the hours either side of each pass) and byte ranges of the fields RAiDER reads; not the area: Herbie fetches the fields for the whole continental US and RAiDER cuts the box out on this computer | none (no account) |
| ERA5 weather (Copernicus CDS) | ECMWF | weather delay correction outside the continental US, or when `raider-env` is not installed | the area, padded by 2 degrees and rounded out to 10 degrees, and the dates | CDS API token (`~/.cdsapirc`) |
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
| NISAR products | NASA / ASF | the NISAR routes (comparison only, not validated): L2 GUNW interferograms; L2 GSLC images on the phase-linked NISAR route (`nisar_phase_linked: true`, since 28 Oct 2026) | the area outline and dates (search); then reads over HTTPS of only the parts of each file that cover the box (the file names reveal track, frame and dates) | Earthdata user name and password from `~/.netrc`, sent as HTTP basic authentication |
| IGS ionosphere maps (MintPy `iono_tec`) | NASA CDDIS | areas of about 100 km or more, or `ionosphere_correction: gim` | dates | Earthdata login |
| HyP3 | NASA / ASF | the HyP3 route (example config only) | pairs of scene names | Earthdata login |
| RAiDER, other weather models | ECMWF, NASA and others, by model | a `weather_model` other than HRRR (with `mintpy_tropo_method: raider`, or on the DISP-S1 route with `disp_tropo_correction`); HRRR itself is in the table above | the box (padded by 0.15 degrees), the date and the time, as far as the model's server takes them | as the model requires |
| Geoscience Australia spreadsheet | Geoscience Australia | the Appin validation experiment only | nothing site-specific | none |
| Map export base map | USGS (topographic), Esri World Imagery, Esri Light Gray Canvas, OpenStreetMap or OpenTopoMap, as chosen | Export map in the app window (`map_export.py`) | the tiles covering the map (at most 100 a map), with the User-Agent "insar-pipeline/<version> (map export)"; tiles are kept for 30 days in `~/.cache/insar_app/tiles` and not fetched again | none |

## In the browser pages (map builder, results map, app window)

| Service | Provider | What is sent |
|---|---|---|
| Map tiles | USGS (default: aerial imagery in the builder, topographic in the app window), Esri World Imagery, Esri Light Gray Canvas, OpenStreetMap or OpenTopoMap, as chosen | the tiles of whatever area is on screen |
| Nominatim search | OpenStreetMap Foundation | the text typed in the builder's search box |
| Leaflet map library | none: served by the page itself | nothing. The results map, the app window and the map builder all serve their own copy of Leaflet (and the builder Leaflet.draw) from `viewer_assets/vendor/` (results map since 19 Oct 2026, map builder since 2 Nov 2026). No page loads anything from cdnjs. |
| Fonts | none: system fonts | nothing. The pages use the system fonts; the app window also uses two pixel fonts shipped in `app_assets/fonts/` and served by the app. No page loads anything from Google Fonts (2 Nov 2026). |

The tile hosts each page may load from are the ones its
Content-Security-Policy allows: `results_viewer.TILE_HOSTS` and
`app_server.TILE_HOSTS`.

## At install

| Service | What is fetched | Integrity check |
|---|---|---|
| Microsoft (Windows setup only) | WSL2 and Ubuntu, through `wsl --install -d Ubuntu` (`install/windows/setup.ps1`, only when Ubuntu is not installed yet) | Microsoft's own; not checked by the pipeline |
| GitHub (conda-forge Miniforge installer) | the Miniforge installer, version "latest" | none (open item NET-04) |
| conda-forge | the pinned packages of `install/envs/*.yml`: five environments (`insar`, `miaplpy`, `burst2safe`, `raider-env`, `dolphin`) | conda's repodata hashes |
| PyPI | the pinned pip packages (the `insar` environment's pip list; Herbie 2026.9.2 in `raider-env`) | none beyond pinning (open item DEP-02) |
| GitHub (MiaplPy) | MiaplPy at a pinned commit | commit pin, patch check, functional test |
| Stanford (snaphu source) | snaphu 2.0.7 source | pinned sha256 |
| none (copied in) | the pipeline itself: a release file (`insar_pipeline_<version>.tar.gz`) put beside the setup, or a clone of the repository | the release's `.sha256` (checked by `setup.ps1` when it is beside the file); git for a clone |

`insar doctor` (run at the end of the install, and whenever asked) tries the
Earthdata login and the CDS key and licence. Nothing site-specific is sent.

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
(ERA5), `weather_raider.py` and `tropo_disp.py` (RAiDER, HRRR through Herbie),
`areas.py` (`weather_default`, the HRRR default), `gnss_tie.py` and
`gnss_validation.py` (NGL, JPL), `landcover.py` (WorldCover), `stack_qc.py`
(Overpass), `areas.py` and `report_insights.py` (Nominatim, USGS basemap),
`disp_backend.py` and `static_layers.py` (DISP-S1), `nisar_backend.py`,
`nisar_gslc.py` and `nisar_mining.py` (NISAR; the HTTPS reader is
`nisar_backend.open_gunw`), `hyp3_backend.py`, `corrections.py` (optional
routes), `ground_truth.py` (validation), `map_export.py` (map export tiles),
`doctor.py` (login checks), `project_builder.html` and `map_builder.py`,
`results_viewer.html` with `viewer_assets/viewer_app.js`, and `app_server.py`
with `app_assets/app.js` (browser; `results_viewer.TILE_HOSTS` and
`app_server.TILE_HOSTS` are the lists their Content-Security-Policy allows),
`install/` and `patches/miaplpy/build_miaplpy.sh` (install).
