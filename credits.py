"""The attribution every client output owes its data sources, built from what
the run actually used.

Until 5 Oct 2026 no client output carried any: the Site B and Site A
reports, KMZs, workbooks and README contained no "Copernicus", no licence and
no ECMWF notice, although README.md itself told the operator to add "Contains
modified Copernicus Sentinel data [year]" to any published product. The
Sentinel notice is a condition of the Copernicus data licence when data is
communicated; ERA5 is CC BY 4.0 (credit, licence link, change notice and the
ECMWF disclaimer); the Copernicus DEM licence (Art. 6) asks for the "produced
using Copernicus WorldDEM-30 ..." notice on adapted data and the liability
sentence on any distribution (both since 7 Oct 2026; the footer line carries
the notice in full); NGL asks for its citation and the station rates are
MIDAS's (Blewitt et al. 2016); ASF asks for its retrieval line; the software
credits are the published papers (MiaplPy: volume 171, not 173 as printed
until 7 Oct 2026).

Wording lives here and nowhere else, so a legal edit is a one-line change.
Each entry is included only when the record shows the source was used:
ERA5 from the corrections actually applied (mintpy/run_products.json, then
provenance, then the report), not from the config alone, because a
correction can be requested and declined. When nothing is recorded the
config decides - an attribution too many costs nothing, one missing is a
licence breach.

    credits(cfg, rep=None, prov=None) -> [{source, text, url, kind}]
    text_block(cfg, rep=None, prov=None, html=False) -> str
"""
import html as _html
import json
from datetime import datetime
from pathlib import Path

SENTINEL_URL = "https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice"
ERA5_URL = "https://doi.org/10.24381/cds.bd0915c6"        # ERA5 hourly data on pressure levels (what PyAPS fetches)
CC_BY_URL = "https://creativecommons.org/licenses/by/4.0/"
GLO30_URL = "https://doi.org/10.5270/ESA-c5d3d65"
NGL_URL = "https://doi.org/10.1029/2018EO104623"
OPERA_URL = "https://doi.org/10.5067/SNWG/OPL3DISPS1-V1"
HRRR_URL = "https://registry.opendata.aws/noaa-hrrr-pds/"
ASF_URL = "https://asf.alaska.edu"
NISAR_URL = "https://nisar.jpl.nasa.gov"
ASF_SLC_ROUTES = ("miaplpy", "stack", "local", "hyp3")     # routes whose Sentinel-1 images came from ASF
# The Copernicus DEM licence (License-COPDEM-30, Art. 6): (b) the notice for
# adapted data - the DEM here is resampled to ellipsoid heights
# (dem_stitcher) and its heights travel per point in the products - and (c)
# the liability sentence for any distribution, modified or not. Until
# 7 Oct 2026 the credit was the plain copyright line without "produced
# using" or (c), and figure footers shortened it further.
DEM_NOTICE = ("produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
              "2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved")
DEM_LIABILITY = ("The organisations in charge of the Copernicus programme by law or by delegation do not incur "
                 "any liability for any use of the Copernicus WorldDEM-30.")

# Software, by the route that used it. The citations are the tools' own
# requested references.
SOFTWARE = {
    "isce2": ("ISCE2", "Coregistration and interferograms: ISCE2 with topsStack (Rosen et al. 2012, "
                       "EUSAR, 730-733; Fattahi, Agram & Simons 2017, IEEE TGRS 55(2), 777-786).",
              "https://github.com/isce-framework/isce2"),
    "miaplpy": ("MiaplPy", "Phase linking: MiaplPy (Mirzaee, Amelung & Fattahi 2023, Non-linear phase linking "
                           "using joined distributed and persistent scatterers, Computers & Geosciences 171, "
                           "105291, doi:10.1016/j.cageo.2022.105291){patch}.",
                "https://github.com/insarlab/MiaplPy"),
    "snaphu": ("snaphu", "Phase unwrapping: snaphu (Chen & Zebker 2000, J. Opt. Soc. Am. A 17(3), 401-414; "
                         "Chen & Zebker 2001, J. Opt. Soc. Am. A 18(2), 338-351; Chen & Zebker 2002, IEEE TGRS "
                         "40(8), 1709-1719){ver}.",
               "https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/"),
    "mintpy": ("MintPy", "Time-series analysis: MintPy (Yunjun, Fattahi & Amelung 2019, Computers & "
                         "Geosciences 133, 104331, doi:10.1016/j.cageo.2019.104331).",
               "https://github.com/insarlab/MintPy"),
    "hyp3": ("ASF HyP3", "Interferograms: ASF HyP3 on-demand processing (Alaska Satellite Facility).",
             "https://hyp3-docs.asf.alaska.edu"),
}
ROUTE_SOFTWARE = {
    "miaplpy": ("isce2", "miaplpy", "snaphu", "mintpy"),
    "stack": ("isce2", "snaphu", "mintpy"),
    "local": ("isce2", "snaphu", "mintpy"),
    "hyp3": ("hyp3", "mintpy"),
    "disp": ("mintpy",),
    "nisar": ("mintpy",),
}


def _load(p: Path) -> dict | None:
    try:
        return json.loads(Path(p).read_text()) if Path(p).exists() else None
    except (OSError, ValueError):
        return None


def _years(cfg, rep, prov) -> tuple[str, str]:
    d = []
    if rep:
        d = list(rep.get("dates") or []) or list(rep.get("dates_list") or [])
    if not d and prov:
        d = [x for x in (prov.get("date_range") or []) if x]
    if not d and cfg is not None:
        d = [getattr(cfg, "start_date", "") or "", getattr(cfg, "end_date", "") or ""]
    ys = sorted(str(x).replace("-", "")[:4] for x in d if str(x)[:4].isdigit())
    if not ys:
        y = str(datetime.now().year)
        return y, y
    return ys[0], ys[-1]


def era5_used(cfg, rep=None, prov=None) -> bool | None:
    """True/False from the record of what was applied; None when nothing says."""
    P = Path(getattr(cfg, "project_dir", "") or ".")
    mintpy_dir = Path(getattr(cfg, "mintpy_dir", P / "mintpy"))
    rp = _load(mintpy_dir / "run_products.json")
    if rp and isinstance(rp.get("corrections_applied"), list):
        return any("era5" in str(t).lower() for t in rp["corrections_applied"])
    outc = (prov or {}).get("outcome") or {}
    toks = outc.get("corrections_in_product_name")
    if isinstance(toks, list):
        return any("era5" in str(t).lower() for t in toks)
    if rep and isinstance(rep.get("corrections"), list):
        return any("era5" in str(t).lower() for t in rep["corrections"])
    return None


def gnss_used(cfg, rep=None) -> bool:
    if rep:
        g = rep.get("gnss") or {}
        if isinstance(g, dict) and (g.get("anchored") or g.get("stations_used")):
            return True
        if rep.get("anchored") or rep.get("gnss_stations_used"):
            return True
    tie = _load(Path(getattr(cfg, "project_dir", ".")) / "deliverables" / "gnss_tie.json") or {}
    return bool(tie.get("background"))


def opera_used(cfg, rep=None) -> bool:
    if str(getattr(cfg, "backend", "")) == "disp":
        return True
    # client_report.json always carries an independent_check block
    # ({"available": False, "status": "not run"} when no cross-check ran), so
    # only one that ran and compared counts; until 5 Oct 2026 any block did,
    # and Site B's and Site A's reports credited OPERA DISP-S1 unused
    ic = (rep or {}).get("independent_check")
    if isinstance(ic, dict):
        return bool(ic.get("available")) and ic.get("status") in (None, "complete")
    if ic:                                                    # an older truthy record: it ran
        return True
    # Only a cross-check that actually compared against OPERA DISP-S1 used it
    # ('no independent product' / 'failed' records did not).
    D = Path(getattr(cfg, "project_dir", ".")) / "deliverables"
    # and only this run's record (an older one, or one left after the option
    # was turned off, did not feed this report)
    from .crosscheck import record_is_current
    for f in (D / "technical" / "cross_check.json", D / "crosscheck.json"):
        rec = _load(f)
        if isinstance(rec, dict):
            return (rec.get("status", "complete") == "complete" and rec.get("available", True) is not False
                    and record_is_current(cfg, rec, "cross_check_disp"))
    return False


OSM_URL = "https://www.openstreetmap.org/copyright"


def osm_used(cfg) -> bool:
    """The QA's geolocation check compared the product with OpenStreetMap water
    (stack_qc.fetch_osm_water -> context/osm_water.geojson); ODbL asks for the
    attribution wherever the result is shown."""
    qa = _load(Path(getattr(cfg, "project_dir", ".")) / "quality_assessment.json") or {}
    geo = qa.get("geolocation") if isinstance(qa, dict) else None
    return isinstance(geo, dict) and str(geo.get("water_map") or "").endswith("osm_water.geojson")


def credits(cfg, rep=None, prov=None) -> list[dict]:
    """[{source, text, url, kind}] for every source this run used."""
    P = Path(getattr(cfg, "project_dir", "") or ".")
    if prov is None:
        prov = _load(P / "deliverables" / "provenance.json")
    backend = str(getattr(cfg, "backend", "") or (rep or {}).get("backend") or "")
    y0, y1 = _years(cfg, rep, prov)
    yrs = y0 if y0 == y1 else f"{y0}-{y1}"
    if backend == "nisar":
        # NISAR's interferograms, not Sentinel-1 images: the Copernicus notice
        # does not apply (26 Oct 2026: the route's GeoTIFFs and README carried it)
        out = [{"source": "NISAR", "kind": "data", "url": NISAR_URL,
                "text": f"NISAR data {yrs}: Level-2 geocoded unwrapped interferograms (GUNW, PROVISIONAL release) "
                        "from the NASA-ISRO Synthetic Aperture Radar mission (NASA/JPL-Caltech and ISRO), "
                        "retrieved from the Alaska Satellite Facility DAAC (ASF DAAC), processed here into "
                        "ground-movement measurements."}]
    else:
        out = [{"source": "Copernicus Sentinel-1", "kind": "data", "url": SENTINEL_URL,
                "text": f"Contains modified Copernicus Sentinel data {yrs} (Sentinel-1 radar images from the "
                        "European Union's Copernicus programme, processed here into ground-movement measurements)."}]

    if backend in ASF_SLC_ROUTES:
        # the radar images came from the ASF archive (data_acquisition: asf_search); ASF asks for this line
        out.append({"source": "ASF DAAC", "kind": "data", "url": ASF_URL,
                    "text": f"Copernicus Sentinel data {yrs}, retrieved from the Alaska Satellite Facility "
                            "DAAC (ASF DAAC), processed by ESA."})

    if opera_used(cfg, rep):
        out.append({"source": "OPERA DISP-S1", "kind": "data", "url": OPERA_URL,
                    "text": "OPERA Surface Displacement from Sentinel-1 (DISP-S1), NASA/JPL OPERA project, "
                            "distributed by the Alaska Satellite Facility DAAC (doi:10.5067/SNWG/OPL3DISPS1-V1)."})

    era5 = era5_used(cfg, rep, prov)
    if era5 is None:            # nothing recorded: the config decides (see the module note)
        era5 = (str(getattr(cfg, "mintpy_tropo_method", "")) == "pyaps"
                or (backend == "disp" and bool(getattr(cfg, "disp_tropo_correction", False))
                    and "ERA5" in str(getattr(cfg, "weather_model", "")).upper()))
    if era5:
        out.append({"source": "ERA5 (Copernicus Climate Change Service)", "kind": "data", "url": ERA5_URL,
                    "text": f"Generated using Copernicus Climate Change Service information {datetime.now().year}: "
                            "ERA5 hourly data on pressure levels (Hersbach et al. 2020), licensed under CC BY 4.0 "
                            f"({CC_BY_URL}). Atmospheric delays were computed from it and removed from the "
                            "measurements. Neither the European Commission nor ECMWF is responsible for any use "
                            "that may be made of the Copernicus information or data it contains."})
    if ((backend == "disp" and bool(getattr(cfg, "disp_tropo_correction", False))
            or str(getattr(cfg, "mintpy_tropo_method", "")) == "raider")
            and "HRRR" in str(getattr(cfg, "weather_model", "")).upper()):
        out.append({"source": "NOAA HRRR", "kind": "data", "url": HRRR_URL,
                    "text": "Atmospheric delays from NOAA High-Resolution Rapid Refresh (HRRR) analyses, "
                            "via the NOAA Open Data Dissemination programme on AWS."})

    if backend != "disp":
        dem = getattr(cfg, "dem_path", None)
        ext = _load(Path(getattr(cfg, "dem_dir", P / "dem")) / "dem_extent.json") or {}
        if not dem or str(ext.get("source", "")).lower().startswith("glo"):
            out.append({"source": "Copernicus DEM GLO-30", "kind": "data", "url": GLO30_URL,
                        "text": f"Terrain {DEM_NOTICE}. {DEM_LIABILITY}"})
        else:
            out.append({"source": "Elevation model", "kind": "data", "url": "",
                        "text": f"Terrain from the elevation model supplied for this project ({Path(str(dem)).name})."})

    if gnss_used(cfg, rep):
        out.append({"source": "Nevada Geodetic Laboratory GNSS", "kind": "data", "url": NGL_URL,
                    "text": "GNSS station positions from the Nevada Geodetic Laboratory: Blewitt, G., W. C. "
                            "Hammond and C. Kreemer (2018), Harnessing the GPS data explosion for "
                            "interdisciplinary science, Eos, 99, doi:10.1029/2018EO104623; station rates by the "
                            "MIDAS method (Blewitt, G., C. Kreemer, W. C. Hammond and J. Gazeaux (2016), MIDAS "
                            "robust trend estimator for accurate GPS station velocities without step detection, "
                            "J. Geophys. Res. Solid Earth, 121, 2054-2068, doi:10.1002/2015JB012552)."})

    if osm_used(cfg):
        out.append({"source": "OpenStreetMap", "kind": "reference layer", "url": OSM_URL,
                    "text": "Water bodies used to check map positions: © OpenStreetMap contributors, "
                            "available under the Open Database License (ODbL)."})

    # the fetched WorldCover map (landcover, 15 Oct 2026; CC BY 4.0) - not a user's own raster
    try:
        from . import landcover
        lp = landcover.local_path(cfg)
        wc = (landcover.enabled(cfg) and not str(getattr(cfg, "landcover_raster", "") or "")
              and lp is not None and lp.exists())
    except Exception:                                            # noqa: BLE001
        wc = False
    if wc:
        out.append({"source": "ESA WorldCover", "kind": "reference layer",
                    "url": "https://doi.org/10.5281/zenodo.7254221",
                    "text": "Tree cover: © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data "
                            "(2021) processed by ESA WorldCover consortium, CC BY 4.0."})

    for s in getattr(cfg, "mine_layer_sources", ()) or ():
        s = str(s).strip()
        if s:
            label, _, rest = s.partition(":")
            out.append({"source": label.strip() or "Mine outlines", "kind": "reference layer", "url": "",
                        "text": f"{label.strip()}: {rest.strip()}" if rest else s})

    sw = (prov or {}).get("software") or {}
    mpv = str(((sw.get("miaplpy") or {}) if isinstance(sw.get("miaplpy"), dict) else {"version": sw.get("miaplpy")})
              .get("version") or "")
    snv = str(sw.get("snaphu") or "")
    for key in ROUTE_SOFTWARE.get(backend, ("mintpy",)):
        name, text, url = SOFTWARE[key]
        text = text.format(patch=(", modified by us to fix the datum between overlapping processing windows"
                                  if "overlapdatum" in mpv else ""),
                           ver=(f", {snv}" if snv.startswith("snaphu v") else ""))
        out.append({"source": name, "kind": "software", "url": url, "text": text})
    return out


def text_block(cfg, rep=None, prov=None, html: bool = False) -> str:
    """The credits as one block: plain lines for README.txt and KML
    descriptions, or an HTML list for the report."""
    items = credits(cfg, rep, prov)
    if not html:
        lines = ["Data sources and credits"]
        for c in items:
            lines.append(f"- {c['text']}" + (f" <{c['url']}>" if c.get("url") and c["url"] not in c["text"] else ""))
        return "\n".join(lines)
    li = []
    for c in items:
        t = _html.escape(c["text"])
        if c.get("url"):
            u = _html.escape(c["url"], quote=True)
            t += f' <a href="{u}">{_html.escape(c["url"])}</a>' if c["url"] not in c["text"] else ""
        li.append(f"<li>{t}</li>")
    return '<ul class="credits">' + "".join(li) + "</ul>"


def short_line(cfg, rep=None, prov=None) -> str:
    """One line for a figure footnote: the notices that must travel with an image."""
    parts = []
    for c in credits(cfg, rep, prov):
        if c["source"] == "Copernicus Sentinel-1":
            parts.append(c["text"].split(" (")[0])
        elif c["source"] == "NISAR":
            parts.append(c["text"].split(":")[0] + " (NASA-ISRO, via ASF DAAC)")
        elif c["source"].startswith("ERA5"):
            parts.append("ERA5: Copernicus Climate Change Service (CC BY 4.0)")
        elif c["source"] == "Copernicus DEM GLO-30":
            parts.append(f"Terrain {DEM_NOTICE}")
        elif c["source"].startswith("Nevada"):
            parts.append("GNSS: Nevada Geodetic Laboratory")
        elif c["source"] == "OPERA DISP-S1":
            parts.append("OPERA DISP-S1 (NASA/JPL)")
    return " · ".join(parts)
