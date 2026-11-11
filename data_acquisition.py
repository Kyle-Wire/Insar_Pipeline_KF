"""Stage 1: search, download SLCs, orbits, and DEM."""
import getpass
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import asf_search as asf

from .config import PipelineConfig

log = logging.getLogger(__name__)


def get_credentials() -> tuple[str, str]:
    """
    Earthdata credentials: EARTHDATA_USERNAME / EARTHDATA_PASSWORD, else the
    urs.earthdata.nasa.gov entry in ~/.netrc (what install.sh writes), else
    an interactive prompt - but only when there is a terminal to answer it.

    A run started from the dashboard is detached with no stdin, so a prompt
    there died with EOFError before downloading anything (28 Sep 2026, the
    first Site B run); only the old run_pipeline.py launcher had been
    exporting the ~/.netrc login into the environment.
    """
    import netrc
    import sys
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not (username and password):
        try:
            auth = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("urs.earthdata.nasa.gov")
        except (FileNotFoundError, netrc.NetrcParseError):
            auth = None
        if auth:
            username, _, password = auth
    if not (username and password):
        if not sys.stdin or not sys.stdin.isatty():
            raise RuntimeError("No NASA Earthdata login: add it to ~/.netrc (install/install.sh asks for it, or "
                               "'machine urs.earthdata.nasa.gov login USER password PASS'), or set "
                               "EARTHDATA_USERNAME / EARTHDATA_PASSWORD.")
        username = username or input("Earthdata username: ")
        password = password or getpass.getpass("Earthdata password: ")
    return username, password


def _stored_credentials():
    """(username, password) from the environment or ~/.netrc, never a prompt;
    (None, None) when there are none."""
    import netrc
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not (username and password):
        try:
            auth = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("urs.earthdata.nasa.gov")
        except (FileNotFoundError, netrc.NetrcParseError):
            auth = None
        if auth:
            username, _, password = auth
    return (username, password) if (username and password) else (None, None)


def _earthdata_login_status() -> tuple[str, str]:
    """
    Try the Earthdata login the way the downloads will use it.
    Returns (code, plain message), code one of
      ok | missing | rejected | not_authorised | unreachable.

    ASFSession.auth_with_creds does two things that can fail separately
    (asf_search 13.1): an EDL token request, which fails on a wrong
    username or password, and the ASF authorisation redirect that sets the
    asf-urs cookie, which fails when the account has not authorised ASF
    data access (the burst extractor needs that cookie). Preflight used to
    check only that ~/.netrc mentioned urs.earthdata.nasa.gov, so a mistyped
    password passed and every one of ~270 burst2safe calls then failed with
    a 1500-character tail that never said "login".
    """
    username, password = _stored_credentials()
    if not username:
        return "missing", ("No NASA Earthdata login is stored: add "
                           "'machine urs.earthdata.nasa.gov login USER password PASS' to "
                           "~/.netrc (install/install.sh asks for it), or set "
                           "EARTHDATA_USERNAME / EARTHDATA_PASSWORD.")
    try:
        asf.ASFSession().auth_with_creds(username, password)
    except Exception as exc:                                    # noqa: BLE001
        text = str(exc)
        if type(exc).__name__ == "ASFAuthenticationError":
            # asf_search wraps ANY failed answer from either step in this
            # error, a server outage included ('Failed to log in with provided
            # credentials. Original Exception: 503 Server Error: ...'). A 5xx
            # says nothing about the login, so it must not send anyone to
            # change a correct ~/.netrc.
            if re.search(r"\b5\d\d Server Error", text):
                return "unreachable", (
                    f"NASA Earthdata or ASF answered with a server error ({text[:200]}) - an "
                    "outage, not a login problem; try again later.")
            if "cookie" in text.lower():
                return "not_authorised", (
                    f"The Earthdata login for '{username}' works, but the account has not "
                    "authorised ASF data access, so downloads will be refused. Sign in at "
                    "https://search.asf.alaska.edu once (accept the terms when asked), "
                    "then run the check again.")
            return "rejected", (
                f"NASA Earthdata rejected the login for '{username}' (wrong username or "
                "password in ~/.netrc, or the account is not activated). Correct it at "
                "https://urs.earthdata.nasa.gov and in ~/.netrc.")
        return "unreachable", (f"Could not reach NASA Earthdata to check the login "
                               f"({type(exc).__name__}: {text[:200]}).")
    return "ok", f"NASA Earthdata login for '{username}' works, including ASF data access."


def check_earthdata_login() -> tuple[bool, str]:
    """(ok, message) for preflight and doctor: True only when the stored login
    was accepted by Earthdata AND by ASF. Never prompts."""
    code, msg = _earthdata_login_status()
    return code == "ok", msg


# Catalogue defects (13 Oct 2026): dates ASF serves in a form burst2safe
# cannot assemble, the same way on every attempt - not a connection, login or
# disk problem. Seen at Corcoran, Houston, Site B and Site C: bursts
# recovered for a date the catalogue returns short ("max() arg is an empty
# sequence" building the annotation), and a date catalogued twice under two
# processing versions ("must have consecutive burst IDs"). Retrying never
# helps, so such a date is accepted as missing automatically (see
# auto_accept_catalogue_defects), up to AUTO_ACCEPT_MAX_SHARE of the dates.
CATALOGUE_DEFECTS = (
    ("max() arg is an empty sequence", "burst annotations empty (catalogue returned the date short)"),
    ("must have consecutive burst IDs", "date catalogued twice (two processing versions of each burst)"),
)
DEFECT_PREFIX = "catalogue defect: "
# the second line of record_auto_accepted's note, which auto_accepted_in_config reads back
AUTO_ACCEPT_NOTE = "# (catalogue defects, not connection or login problems): "
AUTO_ACCEPT_MAX_SHARE = 0.05
AUTO_ACCEPT_MIN = 3


def catalogue_defect(text: str) -> str | None:
    """The plain name of a known catalogue defect in a failed attempt's
    output, or None (any other failure: network, login, disk, timeout)."""
    for sig, name in CATALOGUE_DEFECTS:
        if sig in (text or ""):
            return name
    return None


def record_auto_accepted(cfg: PipelineConfig, dates: dict) -> Path | None:
    """Append dates ({date: reason}) to allow_missing_dates in the config
    file the run was started from, with a dated comment saying why, so the
    record of what was left out lives in the config. Returns the file, or None
    when the config has no file (then the dates are accepted for this run
    only, and the assembly status says so)."""
    src = getattr(cfg, "_source_path", None)
    if not src or not Path(src).exists() or not dates:
        return None
    import yaml
    path = Path(src)
    text = path.read_text()
    doc = yaml.safe_load(text) or {}
    have = doc.get("allow_missing_dates") or []
    if isinstance(have, str):
        have = [x for x in have.replace(",", " ").split() if x]
    new = sorted(set(str(x) for x in have) | set(dates))
    stamp = time.strftime("%d %b %Y")
    note = (f"# {stamp}: added automatically - ASF serves these dates in a form that cannot be "
            "assembled on any attempt\n" + AUTO_ACCEPT_NOTE
            + "; ".join(f"{d} {r}" for d, r in sorted(dates.items())) + "\n")
    line = "allow_missing_dates: [" + ", ".join(f"'{d}'" for d in new) + "]\n"
    lines = text.splitlines(keepends=True)
    kept = [ln for ln in lines if not ln.startswith("allow_missing_dates:")]
    out = "".join(kept)
    if out and not out.endswith("\n"):
        out += "\n"
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(out + note + line)
    tmp.replace(path)
    cfg.allow_missing_dates = tuple(new)
    return path


def auto_accepted_in_config(cfg: PipelineConfig) -> dict:
    """{date: defect name} from the notes record_auto_accepted left in the
    config: why a date was accepted, read back on a later run, whose
    download record no longer says (the first city site, 29 Oct 2026: two
    defect dates were relabelled "found only after processing had started")."""
    src = getattr(cfg, "_source_path", None)
    try:
        text = Path(src).read_text() if src else ""
    except OSError:
        return {}
    out = {}
    for ln in text.splitlines():
        if ln.startswith(AUTO_ACCEPT_NOTE):
            for part in ln[len(AUTO_ACCEPT_NOTE):].split("; "):
                d, _, why = part.strip().partition(" ")
                if d.isdigit() and len(d) == 8 and why:
                    out[d] = why
    return out


def earlier_failures(cfg: PipelineConfig) -> dict:
    """{date: reason} for the dates an earlier download could not assemble
    and that were accepted as missing: the last assembly_status.json's
    accepted failures, then the config's auto-accept notes. A later run
    never fetches such a date again, so this is the only record of why."""
    try:
        prev = json.loads((cfg.slc_dir / ASSEMBLY_STATUS_FILE).read_text())
    except (OSError, ValueError):
        prev = {}
    failed = prev.get("failed") if isinstance(prev.get("failed"), dict) else {}
    out = {str(d): str(failed[d]) for d in prev.get("accepted_failures") or [] if failed.get(d)}
    for d, why in auto_accepted_in_config(cfg).items():
        out.setdefault(d, DEFECT_PREFIX + why)
    return out


def accepted_missing_dates(cfg: PipelineConfig) -> tuple[set, bool]:
    """
    (dates, accept_all) from cfg.allow_missing_dates: acquisitions the user
    has accepted will be missing from the record (a date ASF cannot serve,
    or one left out of a stack that cannot be re-planned). ("all",) or True
    accepts any. Missing dates are otherwise a hard stop, never a log line.
    """
    v = cfg.allow_missing_dates or ()
    if v is True:
        return set(), True
    if isinstance(v, str):
        v = [x for x in v.replace(",", " ").split() if x]
    vals = {str(x).strip().replace("-", "") for x in v}
    vals |= excluded_dates(cfg)                     # left out on purpose, so accepted missing too
    return {x for x in vals if x.isdigit() and len(x) == 8}, "all" in {x.lower() for x in vals}


def excluded_dates(cfg: PipelineConfig) -> set:
    """cfg.stack_exclude_dates as a set of YYYYMMDD strings."""
    v = getattr(cfg, "stack_exclude_dates", None) or ()
    if isinstance(v, str):
        v = [x for x in v.replace(",", " ").split() if x]
    return {str(x).strip().replace("-", "") for x in v} - {""}


def _burst_key(fid: str):
    """'038886_IW2' from a burst granule name (S1_038886_IW2_<time>_VV_<hex>-BURST)."""
    m = re.match(r"S1_(\d+)_(IW\d)", str(fid))
    return f"{m.group(1)}_{m.group(2)}" if m else None


def _expected_per_swath(granules) -> dict:
    """{'IW2': 3} - bursts requested per sub-swath."""
    out: dict = {}
    for g in granules:
        k = _burst_key(g)
        if k:
            sw = k.split("_")[1]
            out[sw] = out.get(sw, 0) + 1
    return out


def verify_safe(safe: Path, expected: dict, pol: str) -> str:
    """
    Why an assembled SAFE cannot be used, or "".

    The outcome check a fresh assembly always had (manifest + tiffs), made
    exact and applied to SAFEs found on disk too: on a resume an existing
    SAFE used to be accepted on its name alone, so a directory left by an
    interrupted assembly went to stackSentinel and failed run_02/run_10
    hours later. Per requested sub-swath: an annotation whose burstList
    count is the number requested, and a measurement tiff at least
    numberOfLines x numberOfSamples x 4 bytes (complex int16). Measured on
    Site B's SAFEs: 4515 x 25001 x 4 = 451,518,060 bytes of data in a
    451,594,876-byte tiff (76.8 kB of header).
    """
    safe = Path(safe)
    man = safe / "manifest.safe"
    if not man.exists() or man.stat().st_size == 0:
        return f"{safe.name} has no manifest.safe"
    p = pol.lower()
    for sw, n in (expected or {}).items():
        s = sw.lower()
        anns = sorted(safe.glob(f"annotation/s1?-{s}-slc-{p}-*.xml"))
        tifs = sorted(safe.glob(f"measurement/s1?-{s}-slc-{p}-*.tiff"))
        if not anns or not tifs:
            return f"{safe.name} has no {sw} {pol} annotation or measurement tiff"
        txt = anns[0].read_text(errors="ignore")
        m = re.search(r'<burstList count="(\d+)"', txt)
        if m and n and int(m.group(1)) != n:
            return f"{safe.name}: {sw} annotation lists {m.group(1)} bursts, {n} were requested"
        lines = re.search(r"<numberOfLines>(\d+)</numberOfLines>", txt)
        samples = re.search(r"<numberOfSamples>(\d+)</numberOfSamples>", txt)
        if lines and samples:
            need = int(lines.group(1)) * int(samples.group(1)) * 4
            have = tifs[0].stat().st_size
            if have < need:
                return f"{safe.name}: {tifs[0].name} is {have} bytes, needs at least {need}"
    if not expected and not list(safe.glob("measurement/*.tiff")):
        return f"{safe.name} has no measurement tiffs"
    return ""


def search_scenes(cfg: PipelineConfig):
    """Query ASF for Sentinel-1 SLC scenes matching the AOI/date range/track."""
    opts = dict(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.SLC,
        start=cfg.start_date,
        end=cfg.end_date,
        intersectsWith=cfg.aoi_wkt,
        flightDirection=cfg.orbit_direction.upper(),
    )
    if cfg.track:
        opts["relativeOrbit"] = cfg.track

    results = cmr_search(**opts)
    log.info("Found %d candidate scenes", len(results))
    return results


CMR_TIMEOUT_S = 120            # asf_search's default is 30 s
CMR_RETRY_PAUSES_S = (30.0, 60.0, 120.0)


def cmr_search(**opts):
    """asf.search with a longer CMR timeout, retried on a transient failure.

    22 Oct 2026: a 10-year burst search over Site D died three times in an
    hour on "CMR took too long to respond" (asf_search's 30 s timeout), each
    time before anything was downloaded, while small queries answered in
    6 s. asf_search reads INTERNAL.CMR_TIMEOUT at call time, so raising it
    here takes effect. Anything that is not a timeout or a 502-504 is
    raised at once."""
    import asf_search as asf_mod             # at call time, as search_bursts_complete does (tests swap the module)
    internal = getattr(getattr(asf_mod, "constants", None), "INTERNAL", None)
    if internal is not None and hasattr(internal, "CMR_TIMEOUT"):
        internal.CMR_TIMEOUT = max(int(internal.CMR_TIMEOUT), CMR_TIMEOUT_S)
    for attempt, pause in enumerate((*CMR_RETRY_PAUSES_S, None), start=1):
        try:
            return asf_mod.search(**opts)
        except Exception as exc:                              # noqa: BLE001 - re-raised unless transient
            if pause is None or not _TRANSIENT_RE.search(str(exc)):
                raise
            log.warning("CMR search failed (%s); retrying in %.0f s (attempt %d of %d)",
                        str(exc)[:120], pause, attempt, len(CMR_RETRY_PAUSES_S) + 1)
            time.sleep(pause)


def download_scenes(cfg: PipelineConfig, results, username: str, password: str) -> list[Path]:
    """Download each scene using an explicitly authenticated session."""
    cfg.slc_dir.mkdir(parents=True, exist_ok=True)
    session = asf.ASFSession().auth_with_creds(username, password)
    downloaded = []
    for product in results:
        target = cfg.slc_dir / product.properties["fileName"]
        if target.exists() and cfg.skip_existing:
            log.info("Already have %s, skipping", target.name)
        else:
            product.download(path=str(cfg.slc_dir), session=session)
        downloaded.append(target)
    return downloaded


def download_orbits(cfg: PipelineConfig, scene_paths: list[Path], username: str, password: str) -> None:
    """
    Precise orbit files (POEORB) are typically published ~3 weeks after
    acquisition; for very recent scenes you'll only get restituted orbits
    (RESORB), which are less accurate. ISCE2 can also fetch orbits itself
    at runtime — this is here for staging them in advance (e.g. before an
    offline/HPC run).
    """
    from eof.download import download_eofs  # from the `sentineleof` package

    def _one(scene):
        try:
            # sentineleof 0.13 takes the SAFE/zip path as `sentinel_file` and
            # derives the mission and time window from it. Passing the name as
            # the first positional argument instead binds it to `orbit_dts`,
            # which expects datetimes - that silently produced an empty-message
            # failure for every scene. Credentials are `asf_user`/`asf_password`;
            # there is no `username`/`password`, and passing them raises
            # TypeError before any download is attempted.
            out = download_eofs(sentinel_file=str(scene),
                                save_dir=str(cfg.orbit_dir),
                                orbit_type="precise",
                                asf_user=username or "",
                                asf_password=password or "")
            return list(out or []), None
        except TypeError:
            # Older/newer signatures: fall back to whatever auth it finds on
            # its own (~/.netrc). Wrapped, because an exception raised inside
            # an except block is not caught by the sibling handlers below.
            try:
                out = download_eofs(sentinel_file=str(scene),
                                    save_dir=str(cfg.orbit_dir))
                return list(out or []), None
            except Exception as exc:                        # noqa: BLE001
                return [], (scene.name, exc)
        except Exception as exc:                            # noqa: BLE001
            return [], (scene.name, exc)

    from . import progress
    got, failed = [], []
    workers = max(1, int(getattr(cfg, "download_workers", 4) or 1))
    scenes = list(scene_paths)
    n = len(scenes)
    progress.emit(cfg.project_dir, "orbits", "start", done=0, total=n)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (files, fail) in enumerate(pool.map(_one, scenes), 1):
            got.extend(files)
            if fail:
                failed.append(fail)
            progress.emit(cfg.project_dir, "orbits", "running", done=i, total=n,
                          msg=f"{len(failed)} without a precise orbit" if failed else "")

    # A download can come back EMPTY without raising (a one-off empty reply
    # from the orbit service: study_areav2, 10 Nov 2026, scene 20261031). So
    # the files on disk are checked, not the download results: every scene
    # needs an orbit file whose validity window covers it, by the same rule
    # stackSentinel applies. Scenes still uncovered are retried after a pause.
    missing = orbitless_scenes(scenes, cfg.orbit_dir)
    for pause in ORBIT_RETRY_PAUSES_S:
        if not missing:
            break
        log.warning("%d scene(s) have no orbit file after staging (%s); "
                    "retrying in %.0f s", len(missing),
                    ", ".join(_scene_date(s) for s in missing[:5]), pause)
        time.sleep(pause)
        for files, fail in map(_one, missing):
            got.extend(files)
            if fail:
                failed.append(fail)
        missing = orbitless_scenes(missing, cfg.orbit_dir)

    progress.emit(cfg.project_dir, "orbits", "failed" if missing else "done",
                  done=n, total=n,
                  msg=f"{len(missing)} without an orbit" if missing else "")
    if got:
        log.info("Staged %d orbit file(s) into %s", len(got), cfg.orbit_dir)
    if not missing:
        return

    errors = {name: exc for name, exc in failed}
    for scene in missing:
        exc = errors.get(scene.name)
        log.warning("No orbit for %s: %s", scene.name,
                    f"{type(exc).__name__}: {exc or '(no detail)'}" if exc
                    else "the orbit service returned no file")
    dates = ", ".join(_scene_date(s) for s in missing)
    if _has_copernicus_netrc():
        # stackSentinel's own fallback (fetchOrbit.py) can log in, so let it try.
        log.warning("Leaving the orbit for %s to stackSentinel, which will "
                    "fetch it from Copernicus Data Space during generation.", dates)
        return
    raise OrbitsMissing(
        f"no orbit file could be downloaded for {len(missing)} of {n} scene(s) "
        f"({dates}) after {len(ORBIT_RETRY_PAUSES_S) + 1} attempts. This is "
        "usually the orbit service being briefly unavailable; press Resume to "
        "try again. If it keeps failing for the newest dates, their orbits may "
        "not be published yet; set stop_date before them.")


# Pauses before each re-download of scenes still without an orbit.
ORBIT_RETRY_PAUSES_S = (30, 120)

_ORBIT_TIME_FMT = "%Y%m%dT%H%M%S"


class OrbitsMissing(RuntimeError):
    """Scenes left without an orbit file that stackSentinel could not fetch
    itself. Raised past stage_inputs' catch-all: without it, stackSentinel
    prompts for a Copernicus login at generation and crashes on EOF."""


def _scene_date(scene) -> str:
    parts = Path(scene).name.split("_")
    return parts[5][:8] if len(parts) > 5 else Path(scene).name


def orbitless_scenes(scenes, orbit_dir) -> list:
    """Scenes with no orbit file in `orbit_dir` that covers them.

    The rule is stackSentinel's (topsStack Stack.get_orbit): same platform
    prefix (S1A, S1D, ...), and the scene's start/stop must lie inside the
    file's validity window shrunk by 60 s at each end. Names it cannot parse
    are treated as covered, leaving the decision to stackSentinel."""
    from datetime import datetime, timedelta

    margin = timedelta(seconds=60)
    windows: dict[str, list] = {}
    for f in Path(orbit_dir).glob("*.EOF"):
        p = f.name.split("_")
        try:
            start = datetime.strptime(p[6].replace("V", ""), _ORBIT_TIME_FMT) + margin
            stop = datetime.strptime(p[7].replace(".EOF", ""), _ORBIT_TIME_FMT) - margin
        except (IndexError, ValueError):
            continue
        windows.setdefault(p[0], []).append((start, stop))
    out = []
    for scene in scenes:
        p = Path(scene).name.split("_")
        try:
            s0 = datetime.strptime(p[5], _ORBIT_TIME_FMT)
            s1 = datetime.strptime(p[6], _ORBIT_TIME_FMT)
        except (IndexError, ValueError):
            continue
        if not any(a <= s0 and s1 < b for a, b in windows.get(p[0], ())):
            out.append(scene)
    return out


def _has_copernicus_netrc() -> bool:
    """True when ~/.netrc has the login fetchOrbit.py looks up."""
    import netrc
    try:
        return "dataspace.copernicus.eu" in netrc.netrc().hosts
    except (OSError, netrc.NetrcParseError):
        return False


FRAME_FOOTPRINT_FILE = "frame_footprint.json"
DEM_MARGIN_DEG = 0.1


def _safe_footprints(slc_dir: Path) -> list:
    """(W, S, E, N) of every SAFE in slc_dir, read from manifest.safe."""
    import re
    boxes = []
    for manifest in sorted(slc_dir.glob("S1*.SAFE/manifest.safe")):
        txt = manifest.read_text(errors="ignore")
        m = re.search(r"<gml:coordinates>([^<]+)</gml:coordinates>", txt)
        if not m:
            continue
        pts = [tuple(float(v) for v in p.split(",")) for p in m.group(1).split()]
        lats = [p[0] for p in pts]
        lons = [p[1] for p in pts]
        boxes.append((min(lons), min(lats), max(lons), max(lats)))
    return boxes


def frame_bbox(cfg: PipelineConfig) -> tuple:
    """
    (W, S, E, N) of the ground the processor will actually image.

    The AOI bbox is NOT this. stackSentinel -b selects whole bursts, and
    the merged frame is the full extent of those bursts - here, a
    90 km x 60 km frame around a 24 km x 30 km AOI. The DEM must cover the
    frame: where it does not, topo fills height with -500, topographic phase
    is never removed, and (measured) 86% of every interferogram unwraps as
    noise while costing snaphu the same as real ground.

    Sources, in order: the footprint recorded by the burst download; the
    SAFE manifests on disk; and finally the AOI bbox padded by 0.6 degrees,
    which comfortably contains one IW sub-swath and a few bursts.
    """
    rec = cfg.slc_dir / FRAME_FOOTPRINT_FILE
    if rec.exists():
        try:
            d = json.loads(rec.read_text())
            return tuple(d["bbox_wsen"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    boxes = _safe_footprints(cfg.slc_dir)
    if boxes:
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))
    w, s, e, n = cfg.bbox
    log.warning("No SAFE footprints available - padding the AOI by 0.6 deg for "
                "the DEM extent.")
    return (w - 0.6, s - 0.6, e + 0.6, n + 0.6)


def dem_bbox(cfg: PipelineConfig) -> tuple:
    """Frame footprint plus DEM_MARGIN_DEG on every side, (W, S, E, N)."""
    w, s, e, n = frame_bbox(cfg)
    m = DEM_MARGIN_DEG
    return (w - m, s - m, e + m, n + m)


def get_dem(cfg: PipelineConfig) -> Path:
    """
    Stage a DEM for ISCE2 covering the whole frame, not just the AOI.

    Writes an uncompressed, untiled GeoTIFF (ISCE2's bundled GDAL is picky
    about compressed/tiled input), then converts it to ISCE's native format
    via gdal_translate, which writes the .xml sidecar topsApp expects.

    The extent actually stitched is recorded next to the DEM in
    dem_extent.json; a staged DEM that does not cover the current frame
    bbox is re-stitched rather than reused.
    """
    if cfg.dem_path and Path(cfg.dem_path).exists():
        return Path(cfg.dem_path)

    cfg.dem_dir.mkdir(parents=True, exist_ok=True)
    dem_tif = cfg.dem_dir / "dem_raw.tif"
    dem_isce = cfg.dem_dir / "dem.wgs84"
    extent_rec = cfg.dem_dir / "dem_extent.json"
    want = dem_bbox(cfg)
    if dem_isce.exists() and Path(str(dem_isce) + ".xml").exists() and cfg.skip_existing:
        have = None
        if extent_rec.exists():
            try:
                have = tuple(json.loads(extent_rec.read_text())["bbox_wsen"])
            except (json.JSONDecodeError, KeyError, TypeError):
                have = None
        covers = have is not None and (have[0] <= want[0] and have[1] <= want[1]
                                       and have[2] >= want[2] and have[3] >= want[3])
        if covers:
            log.info("DEM already staged at %s (covers %s), skipping", dem_isce,
                     ["%.3f" % v for v in want])
            return dem_isce
        log.warning("Staged DEM %s does not cover the frame %s (recorded extent: "
                    "%s) - re-stitching. A DEM that stops at the AOI leaves the "
                    "rest of the frame with height -500 and no topographic "
                    "correction.", dem_isce, ["%.3f" % v for v in want],
                    have and ["%.3f" % v for v in have])

    from dem_stitcher import stitch_dem
    import rasterio

    min_lon, min_lat, max_lon, max_lat = want
    log.info("Stitching Copernicus GLO-30 over W %.3f S %.3f E %.3f N %.3f "
             "(frame footprint + %.2f deg)", *want, DEM_MARGIN_DEG)
    X, profile = stitch_dem(
        [min_lon, min_lat, max_lon, max_lat],
        dem_name="glo_30",           # Global Copernicus 30m DEM, public, no login needed
        dst_ellipsoidal_height=True,  # ISCE2 expects heights above the WGS84 ellipsoid,
                                       # not the geoid — this converts it
        dst_area_or_point="Point",
    )
    profile.update(driver="GTiff", count=1, dtype=X.dtype,
                    height=X.shape[0], width=X.shape[1],
                    compress=None, tiled=False)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    with rasterio.open(dem_tif, "w", **profile) as dst:
        dst.write(X, 1)
        dst.update_tags(AREA_OR_POINT="Point")

    # Convert to ISCE native format — the ISCE GDAL driver writes the
    # accompanying .xml metadata sidecar that topsApp.py requires.
    subprocess.run(
        ["gdal_translate", "-of", "ISCE", str(dem_tif), str(dem_isce)],
        check=True,
    )

    # Two fixes to the generated sidecar:
    #   1. gdal_translate writes the DEM's *basename* into the xml. topsApp
    #      runs with cwd set to each pair directory, so it won't find the
    #      DEM there — rewrite with the absolute path.
    #   2. Tag the DEM as WGS84 (ellipsoidal). We already requested
    #      ellipsoidal heights from dem_stitcher; without this tag ISCE2
    #      assumes geoid heights and applies a geoid correction on top,
    #      corrupting the elevations.
    from .isceenv import import_isce
    _, isceobj = import_isce()

    img = isceobj.createDemImage()
    img.load(str(dem_isce) + ".xml")
    img.filename = str(dem_isce)
    img.setAccessMode("READ")
    img.reference = "WGS84"
    img.renderVRT()
    img.renderHdr()

    # Record what was stitched, and check it: the file on disk is the fact.
    with rasterio.open(dem_tif) as src:
        got = tuple(src.bounds)          # (W, S, E, N)
    extent_rec.write_text(json.dumps(
        {"bbox_wsen": list(got), "requested_wsen": list(want),
         "source": "glo_30", "shape": list(X.shape)}, indent=2))
    tol = 0.01
    if (got[0] > want[0] + tol or got[1] > want[1] + tol
            or got[2] < want[2] - tol or got[3] < want[3] - tol):
        raise RuntimeError(
            f"DEM stitched to {got} does not cover the requested frame {want}")
    log.info("DEM staged: %d x %d px covering W %.3f S %.3f E %.3f N %.3f",
             X.shape[1], X.shape[0], *got)
    return dem_isce


def search_bursts_complete(cfg: PipelineConfig, swaths, pol, want, attempts: int = 4,
                           pause_s: float = 20.0, footprints: dict | None = None):
    """
    The burst search, repeated until CMR stops dropping results.

    asf_search pages through CMR and, when a page comes back short, logs
    "CMR returned page of incomplete results. Expected N, got M" and moves
    on - the caller sees a shorter list, not an error. On the P2 launch two
    dates came back with 2 bursts instead of 3 that way. Granule IDs are
    deterministic, so the union over several attempts is safe; the search
    is repeated until one attempt adds nothing new (or `attempts` is
    exhausted), and the per-date burst list is the union.

    Retrying repairs only a TRANSIENT gap. Some gaps are deterministic: CMR
    renders certain burst records in UMM-JSON (what asf_search reads) short
    on every query, while the same records are complete in plain JSON
    (Site B, 28 Sep 2026: 20250517, 20251113, 20260605, 20260624 - two
    identical attempts, then dropped). The short dates that remain after
    this are therefore checked against the plain-JSON catalogue by
    catalogue_burst_entries(), in download_bursts_as_safes.

    Returns (by_date: {yyyymmdd: [fileID...]}, lons, lats) for the footprint.
    `footprints`, when given, is filled with {burst ID: polygon ring} - one
    per distinct burst (fullBurstID), for burst_coverage_check.
    """
    import asf_search as asf
    from collections import defaultdict

    by_date: dict = defaultdict(set)
    lons, lats = [], []
    seen_ids: set = set()
    for attempt in range(1, attempts + 1):
        results = cmr_search(dataset=asf.DATASET.SLC_BURST, intersectsWith=cfg.aoi_wkt,
                             start=cfg.start_date, end=cfg.end_date,
                             relativeOrbit=cfg.track, polarization=[pol])
        new_ids = 0
        for r in results:
            pr = r.properties
            # flightDirection cannot be passed to a burst query - it returns
            # zero results - so it is filtered here instead.
            if pr.get("flightDirection", "").upper()[:3] != want:
                continue
            if pr["burst"]["subswath"] not in swaths:
                continue
            fid = pr["fileID"]
            if fid in seen_ids:
                continue
            seen_ids.add(fid)
            new_ids += 1
            by_date[pr["startTime"][:10].replace("-", "")].add(fid)
            geom = getattr(r, "geometry", None) or {}
            rings = geom.get("coordinates", []) if geom.get("type") == "Polygon" else []
            if footprints is not None and rings:
                footprints.setdefault(pr["burst"].get("fullBurstID") or fid, rings[0])
            for ring in rings:
                for x, y in ring:
                    lons.append(x)
                    lats.append(y)
        log.info("burst search attempt %d: %d results, %d new burst IDs, %d dates",
                 attempt, len(results), new_ids, len(by_date))
        if attempt > 1 and new_ids == 0:
            break
        if attempt < attempts:
            time.sleep(pause_s)
    else:
        log.warning("Burst search still adding results after %d attempts - CMR is "
                    "unstable right now; the per-date counts below may be short "
                    "(short dates are checked against the plain-JSON catalogue next).",
                    attempts)
    return {d: sorted(v) for d, v in by_date.items()}, lons, lats


CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
_BURST_TITLE_RE = re.compile(r"^S1_(\d{6})_(IW\d)_(\d{8})T\d{6}_(VV|VH|HH|HV)_[0-9A-F]+-BURST$")
_EXTRACTOR_RE = re.compile(r"sentinel1-burst\.asf\.alaska\.edu/([^/]+)/(IW\d)/(VV|VH|HH|HV)/(\d+)\.(tiff|xml)$")


def catalogue_burst_entries(cfg: PipelineConfig, date: str, pol: str, swaths) -> dict:
    """
    The plain-JSON CMR catalogue's bursts for one acquisition date:
    {"038886_IW2": {"title", "slc", "swath", "pol", "index", "tiff", "xml"}}.

    The same records asf_search reads as UMM-JSON, rendered by CMR's plain
    JSON endpoint - which, measured on Site B's four short dates, lists
    every burst that UMM-JSON drops (verified twice per format, review 5 Oct
    2026). Each entry carries the burst extractor URLs of its tiff and
    metadata, which is how a date the burst search reports short can still
    be assembled. One small query per short date; no download.
    """
    import datetime as _dt
    import requests

    d0 = _dt.datetime.strptime(date, "%Y%m%d")
    d1 = d0 + _dt.timedelta(days=1)
    w, s, e, n = cfg.bbox
    params = {"short_name": "SENTINEL-1_BURSTS", "page_size": 2000,
              "temporal": f"{d0:%Y-%m-%dT00:00:00Z},{d1:%Y-%m-%dT00:00:00Z}",
              "bounding_box": f"{w},{s},{e},{n}"}
    r = requests.get(CMR_GRANULES, params=params, timeout=60)
    r.raise_for_status()
    out: dict = {}
    for ent in r.json().get("feed", {}).get("entry", []):
        m = _BURST_TITLE_RE.match(ent.get("title", ""))
        if not m or m.group(3) != date or m.group(4) != pol.upper() or m.group(2) not in swaths:
            continue
        rec = {"title": ent["title"], "swath": m.group(2), "pol": m.group(4)}
        for link in ent.get("links", []):
            lm = _EXTRACTOR_RE.search(link.get("href", ""))
            if lm:
                rec.update(slc=lm.group(1), index=int(lm.group(4)))
                rec[lm.group(5)] = link["href"]
        if rec.get("tiff") and rec.get("xml"):
            out[f"{m.group(1)}_{m.group(2)}"] = rec
    return out


# A CMR timeout or 5xx is a passing hiccup, not a verdict on the date: the
# one query per short date is tried three times before it is given up on.
CATALOGUE_ATTEMPTS = 3
CATALOGUE_BACKOFF_S = (10.0, 30.0)


def classify_short_dates(cfg: PipelineConfig, by_date: dict, short: list, modal_keys: set,
                         pol: str, swaths) -> dict:
    """
    {date: {"reason": "render_gap" | "catalogue_short" | "catalogue_unreachable",
    "bursts_found": n, "bursts_needed": m, "entries": {...}}} for dates the burst
    search returned short.

    render_gap: the plain-JSON catalogue holds every burst of the stack's
    modal set - the data exist, only asf_search's view of them is short;
    the date can be assembled from the burst extractor (recover_from_extractor).
    catalogue_short: the archive really covers only part of the area that
    day (Site B 20160516: one burst of three in both renderings) - the
    date is dropped, and recorded with the reason.
    catalogue_unreachable: the catalogue did not answer after
    CATALOGUE_ATTEMPTS tries. Nobody has chosen to leave the date out, so
    download_bursts_as_safes counts it as failed (a hard stop unless
    allow_missing_dates lists it) and the next run checks it again.
    """
    out = {}
    for d in short:
        have = {_burst_key(g) for g in by_date.get(d, [])}
        ents, err = None, None
        for attempt in range(1, CATALOGUE_ATTEMPTS + 1):
            try:
                ents = catalogue_burst_entries(cfg, d, pol, swaths)
                break
            except Exception as exc:                            # noqa: BLE001
                err = exc
                log.warning("Plain-JSON catalogue check for %s failed on attempt %d of %d (%s).",
                            d, attempt, CATALOGUE_ATTEMPTS, exc)
                if attempt < CATALOGUE_ATTEMPTS:
                    time.sleep(CATALOGUE_BACKOFF_S[min(attempt, len(CATALOGUE_BACKOFF_S)) - 1])
        if ents is None:
            out[d] = {"reason": "catalogue_unreachable", "bursts_found": len(have),
                      "bursts_needed": len(modal_keys), "detail": str(err)[:200]}
            continue
        found = modal_keys & set(ents)
        if modal_keys and found == modal_keys:
            out[d] = {"reason": "render_gap", "bursts_found": len(have),
                      "bursts_needed": len(modal_keys),
                      "entries": {k: ents[k] for k in sorted(modal_keys)}}
        else:
            out[d] = {"reason": "catalogue_short", "bursts_found": len(found | have),
                      "bursts_needed": len(modal_keys)}
    return out


# Assembly: three attempts with backoff (an ASF hiccup or a Wi-Fi drop
# during a ~3 h, 270-date download is the expected case, not the exception),
# and a timeout so a stalled transfer cannot hang a worker for ever. A
# burst2safe call takes ~93 s per date when healthy (measured serially).
ASSEMBLE_ATTEMPTS = 3
ASSEMBLE_BACKOFF_S = (30.0, 120.0)
ASSEMBLE_TIMEOUT_S = 1800.0
_AUTH_FAIL_RE = re.compile(r"\b(401|403)\b|unauthori[sz]ed|forbidden|"
                           r"authentication (failed|error)|login (failed|rejected)", re.IGNORECASE)
# A timeout is not a refused login (22 Oct 2026: a CMR timeout's traceback
# carried "line 403", the 401/403 search matched it, and the whole download
# stopped as "login refused" while the login itself was fine).
_TRANSIENT_RE = re.compile(r"took too long to respond|\(Timeout\)|timed out|Read timeout|ConnectTimeout|"
                           r"\b50[234]\b Server Error|Service Unavailable", re.IGNORECASE)


def auth_refused(text: str) -> bool:
    """True when a download's output says the login was refused: an HTTP 401/403
    or an authentication message, not a traceback line number, and not a timeout."""
    text = text or ""
    if _TRANSIENT_RE.search(text):
        return False
    return bool(_AUTH_FAIL_RE.search(re.sub(r"\bline \d+", "line N", text)))


ASSEMBLY_STATUS_FILE = "assembly_status.json"


def _partial_dir(cfg: PipelineConfig, date: str) -> Path:
    """Where a date is assembled before it is verified and renamed into slc/.
    Hidden, so stackSentinel's '*SAFE' glob and cleanup never see a partial."""
    return cfg.slc_dir / ".partial" / date


def _quarantine(cfg: PipelineConfig, safe: Path, why: str) -> Path:
    q = cfg.slc_dir / "quarantine"
    q.mkdir(parents=True, exist_ok=True)
    dest = q / safe.name
    if dest.exists():
        dest = q / f"{safe.name}.{time.strftime('%Y%m%d_%H%M%S')}"
    safe.rename(dest)
    (q / f"{safe.name}.REASON.txt").write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n{why}\n")
    log.warning("Quarantined %s -> %s: %s", safe.name, dest, why)
    return dest


def recover_from_extractor(cfg: PipelineConfig, date: str, entries: dict, out_dir: Path,
                           session=None) -> list:
    """
    Assemble a render-gap date from the burst extractor: download each
    burst's tiff and metadata xml (the URLs the plain-JSON catalogue lists)
    and build the SAFE with burst2safe's local2safe - the same library and
    SAFE layout as every other date, at burst size (~150 MB per burst), not
    a 4-5 GB full SLC. burst2safe itself cannot do it: it looks granules up
    through the same UMM-JSON search that drops them (burst2safe/search.py:29;
    asked for all three 20260605 VV bursts it returns only 038888).
    Writes the SAFE into out_dir; returns (returncode, output text).
    """
    import requests
    from . import procs

    if session is None:
        user, pw = _stored_credentials()
        session = asf.ASFSession().auth_with_creds(user, pw) if user else requests.Session()
    bursts = out_dir / "bursts"
    bursts.mkdir(parents=True, exist_ok=True)
    tree: dict = {}
    for key, ent in sorted(entries.items()):
        files = {}
        for kind in ("tiff", "xml"):
            dest = bursts / f"{ent['slc']}_{ent['swath']}_{ent['pol']}_{ent['index']}.{kind}"
            # The extractor answers 202 while it cuts the burst out of the
            # SLC; burst2safe retries until it does not (download.get_async,
            # up to 120 s), and so does this.
            t0 = time.time()
            while True:
                r = session.get(ent[kind], stream=True, timeout=600)
                if r.status_code != 202 or time.time() - t0 > 120:
                    break
                r.close()
                time.sleep(2.0)
            with r:
                if r.status_code in (401, 403):
                    return 401, f"HTTP {r.status_code} from the burst extractor for {key}"
                if r.status_code != 200:
                    return 1, f"HTTP {r.status_code} from the burst extractor for {key} {kind}"
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if dest.stat().st_size == 0:
                return 1, f"empty {kind} for {key}"
            files["DATA" if kind == "tiff" else "METADATA"] = str(dest)
        tree.setdefault(ent["slc"], {}).setdefault(ent["swath"], {}) \
            .setdefault(ent["pol"], {})[str(ent["index"])] = files
    tree_path = out_dir / "slc_tree.json"
    tree_path.write_text(json.dumps(tree, indent=1))
    # local2safe's own CLI passes --work_dir as keep_files (burst2safe 1.x
    # local2safe.main), so call the function.
    code = ("import json, sys; from burst2safe.local2safe import local2safe; "
            "print(local2safe(json.load(open(sys.argv[1])), work_dir=sys.argv[2]))")
    cmd = ["conda", "run", "-n", cfg.burst2safe_env, "--no-capture-output",
           "python", "-c", code, str(tree_path), str(out_dir)]
    proc = procs.run_group(cmd, timeout=ASSEMBLE_TIMEOUT_S, cwd=str(out_dir))
    return proc.returncode, (proc.stderr or "") + (proc.stdout or "")


def _processed(status: dict) -> list:
    """The dates a download pass leaves ready for the stack: a verified SAFE
    on disk (found or made now), or already coregistered."""
    return sorted(set(status["existing"]) | set(status["assembled"])
                  | set(status["recovered"]) | set(status["skipped_coregistered"]))


def _write_assembly_status(cfg: PipelineConfig, status: dict) -> None:
    """slc/assembly_status.json: every catalogue date and what became of it.
    QA, provenance and the report read it (contract, 5 Oct 2026)."""
    status = dict(status, written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        (cfg.slc_dir / ASSEMBLY_STATUS_FILE).write_text(json.dumps(status, indent=1, sort_keys=True))
    except OSError as exc:
        log.warning("could not write %s: %s", ASSEMBLY_STATUS_FILE, exc)


MIN_BURST_COVERAGE = 0.99          # = areas.MIN_BURST_COVERAGE (the builder's rule)


def _report_shapes(cfg: PipelineConfig) -> dict:
    """{name: shapely geometry} of the report areas that are boxes or
    polygons ('full' is the processing box itself)."""
    from shapely.geometry import box, shape
    from shapely.ops import unary_union
    from .config import split_area_spec
    out = {}
    for spec in getattr(cfg, "report_areas", ()) or ():
        try:
            name, kind, val = split_area_spec(str(spec))
        except ValueError:
            continue
        if kind == "box":
            s_, w, n, e = val
            out[name] = box(min(w, e), min(s_, n), max(w, e), max(s_, n))
        elif kind == "geojson":
            try:
                gj = json.loads(Path(os.path.expandvars(os.path.expanduser(str(val)))).read_text())
            except (OSError, ValueError):
                continue
            feats = gj.get("features") if gj.get("type") == "FeatureCollection" else [gj]
            geoms = [shape(f.get("geometry") or f) for f in feats if (f.get("geometry") or f.get("coordinates"))]
            if geoms:
                out[name] = unary_union([g.buffer(0) for g in geoms])
    return out


def burst_coverage_check(cfg: PipelineConfig, footprints: dict) -> dict:
    """
    The share of the processing box and of every report area inside the
    union of the selected bursts' footprints (one per distinct burst). The
    builder checks this when it writes the config; a config written by hand
    or before 5 Oct 2026 (or a stack_swaths edit) was never checked, and the
    part outside the bursts is simply missing from every product. Raises
    below MIN_BURST_COVERAGE, before anything is downloaded.
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    if not footprints:
        return {"checked": False, "reason": "no burst footprints in the catalogue results"}
    union = unary_union([Polygon(r).buffer(0) for r in footprints.values() if len(r) >= 4])
    w, s_, e, n = (float(v) for v in cfg.bbox)
    targets = {"processing box": box(w, s_, e, n), **_report_shapes(cfg)}
    shares = {k: round(float(g.intersection(union).area / g.area), 4) if g.area > 0 else 1.0
              for k, g in targets.items()}
    out = {"checked": True, "distinct_bursts": len(footprints), "shares": shares,
           "minimum": MIN_BURST_COVERAGE}
    short = {k: v for k, v in shares.items() if v < MIN_BURST_COVERAGE}
    if short:
        sw = " ".join(f"IW{x}" for x in str(getattr(cfg, "stack_swaths", "2")).split())
        raise RuntimeError(
            f"The selected bursts ({sw}, track {cfg.track} {cfg.orbit_direction}) cover only "
            + ", ".join(f"{100 * v:.1f}% of the {k}" for k, v in short.items())
            + f" (needed: {100 * MIN_BURST_COVERAGE:.0f}%). That part would be missing from every product. Add the "
              "neighbouring sub-swath to stack_swaths, move or shrink the area, or choose another track. "
              "Nothing was downloaded.")
    log.info("Burst coverage: %s", ", ".join(f"{k} {100 * v:.1f}%" for k, v in shares.items()))
    return out


def download_bursts_as_safes(cfg: PipelineConfig) -> list[Path]:
    """
    Build one SAFE per acquisition from individual bursts.

    A Sentinel-1 IW SLC archive carries three sub-swaths and both
    polarisations. A single AOI usually needs one sub-swath and one
    polarisation, so most of a 5.3 GB download is discarded on arrival.
    ASF publishes the bursts separately, and burst2safe reassembles the
    ones we ask for into a SAFE that ISCE reads natively.

    Measured on Site A track 19: 3 bursts of IW2/VV, 450 MB per date
    against 5.3 GB - a factor of about twelve, and the difference between a
    production run that fits on disk and one that does not.

    burst2safe lives in its own conda environment because its dependency
    pins conflict with ISCE's, the same reason raider_env exists.
    """
    swaths = {f"IW{s}" for s in str(getattr(cfg, "stack_swaths", "2")).split()}
    pol = str(getattr(cfg, "stack_polarization", "vv")).upper()

    log.info("Searching bursts: %s, %s, track %s, %s to %s",
             ",".join(sorted(swaths)), pol, cfg.track, cfg.start_date, cfg.end_date)
    want = cfg.orbit_direction.upper()[:3]
    footprints: dict = {}
    by_date, lons, lats = search_bursts_complete(cfg, swaths, pol, want, footprints=footprints)

    if not by_date:
        raise RuntimeError(
            "No bursts matched. Check track, orbit_direction, stack_swaths and "
            "the AOI - note a burst query returns nothing if flightDirection is "
            "passed to the search itself.")
    coverage = burst_coverage_check(cfg, footprints)

    cfg.slc_dir.mkdir(parents=True, exist_ok=True)
    if lons:
        # The union of the burst footprints is the frame the processor will
        # image. get_dem() sizes the DEM from this, so it is written before
        # any download and survives SAFE cleanup.
        fp = (min(lons), min(lats), max(lons), max(lats))
        (cfg.slc_dir / FRAME_FOOTPRINT_FILE).write_text(json.dumps(
            {"bbox_wsen": list(fp), "n_bursts": sum(len(v) for v in by_date.values()),
             "distinct_bursts": len(footprints), "coverage": coverage,
             "source": "ASF burst catalogue footprints"}, indent=2))
        log.info("Frame footprint from %d burst geometries: W %.3f S %.3f E %.3f "
                 "N %.3f", len(lons) // 5 or len(lons), *fp)

    from collections import Counter

    from . import progress
    from . import stack_backend as sb

    catalogue_dates = sorted(by_date)
    counts = {len(v) for v in by_date.values()}
    log.info("%d acquisitions, %s bursts each (~%d MB per date)",
             len(by_date), "/".join(str(c) for c in sorted(counts)),
             150 * max(counts))
    accepted, accept_all = accepted_missing_dates(cfg)
    short_info: dict = {}
    short: list = []
    # the most common burst count; a tie goes to the larger count (more ground)
    modal = max(counts, key=lambda c: (sum(1 for v in by_date.values() if len(v) == c), c))
    if len(counts) > 1:
        # A date with fewer bursts than the rest shrinks the common valid
        # region of the WHOLE stack (extractCommonValidRegion intersects
        # every date). One short date is worth far less than the ground it
        # would remove from every other one, so it is not processed short.
        # But "short" in asf_search's view is not always short in the
        # archive - see search_bursts_complete - so each one is checked
        # against the plain-JSON catalogue first: a render gap is assembled
        # from the burst extractor, a genuinely short date is dropped, and
        # both are recorded with the reason.
        short = sorted(d for d, v in by_date.items() if len(v) < modal)
        modal_keys = set(Counter(frozenset(_burst_key(g) for g in v)
                                 for v in by_date.values() if len(v) == modal).most_common(1)[0][0])
        short_info = classify_short_dates(cfg, by_date, short, modal_keys, pol, swaths)
        for d in short:
            by_date.pop(d)
        gaps = [d for d in short if short_info[d]["reason"] == "render_gap"]
        unchecked = [d for d in short if short_info[d]["reason"] == "catalogue_unreachable"]
        dropped = [d for d in short if d not in gaps and d not in unchecked]
        log.warning("Burst count varies between dates %s. %d date(s) came back with fewer "
                    "than %d bursts: %s. The plain-JSON catalogue holds every burst for %d "
                    "of them (%s) - they are assembled from the burst extractor. %d are "
                    "genuinely short in the archive and are dropped rather than shrink the "
                    "common overlap region for the whole stack: %s.%s", sorted(counts),
                    len(short), modal, ", ".join(short), len(gaps), ", ".join(gaps) or "none",
                    len(dropped), ", ".join(dropped) or "none",
                    f" {len(unchecked)} could not be checked (the catalogue did not answer) and "
                    f"count as failed: {', '.join(unchecked)}." if unchecked else "")
    recoverable = {d: i["entries"] for d, i in short_info.items() if i["reason"] == "render_gap"}
    left_out = {d: i for d, i in short_info.items()
                if i["reason"] not in ("render_gap", "catalogue_unreachable")}
    status = {"catalogue_dates": catalogue_dates, "modal_burst_count": modal,
              "dropped": {d: i["reason"] for d, i in left_out.items()},
              "dropped_detail": {d: {k: v for k, v in i.items() if k != "entries"}
                                 for d, i in left_out.items()},
              "render_gap": sorted(recoverable), "existing": [], "assembled": [],
              "recovered": [], "skipped_coregistered": [],
              # Nobody chose to leave out a date the catalogue could not be
              # asked about: it is a failure (a hard stop unless
              # allow_missing_dates lists it), and the next run checks again.
              "failed": {d: f"burst catalogue unreachable after {CATALOGUE_ATTEMPTS} tries: "
                            f"{i.get('detail', '')}"
                         for d, i in short_info.items() if i["reason"] == "catalogue_unreachable"},
              "accepted_missing": ["all"] if accept_all else sorted(accepted)}
    carried: dict = {}                    # failures of an earlier run kept (the plan check below)

    def write_short_record():
        """slc/dropped_short_dates.json, once it is known which short dates
        this run leaves out (the legacy record: QA and the report read its
        'dropped' when there is no assembly_status reason)."""
        if short:
            (cfg.slc_dir / "dropped_short_dates.json").write_text(json.dumps(
                {"modal_burst_count": modal,
                 "dropped": [d for d in short if d in status["dropped"]
                             or (d in status["failed"] and d not in carried)],
                 "recoverable_render_gap": [d for d in short if d in recoverable
                                            and d not in status["dropped"]]}, indent=2))

    stray = list(cfg.slc_dir.glob("*.zip"))
    if stray:
        raise RuntimeError(
            f"{len(stray)} .zip archive(s) are present in {cfg.slc_dir}. "
            "stackSentinel globs '*zip' before '*SAFE', so it would silently "
            "use those and ignore every assembled SAFE. Remove them first.")

    # Dates already coregistered do not need their archive back. Without this,
    # SAFE cleanup and resumability fight each other: cleanup deletes the
    # archives it no longer needs, and the next resume dutifully re-downloads
    # every one of them. The predicate is the one SAFE cleanup uses
    # (stack_backend.coregistered_dates): it counts the <date>.cleaned
    # markers the phase-linked merge leaves - the directory-only test here
    # re-queued ~263 dates (~110 GB) on Site B - and never a date whose
    # resample is incomplete. The reference date is never skipped: its
    # burst SLCs are VRTs into its SAFE.
    already = sb.coregistered_dates(cfg, include_reference=False)
    if already:
        log.info("%d date(s) are already coregistered - their archives are not "
                 "needed and will not be re-fetched.", len(already))
    # A coregistered date needs no catalogue answer, since nothing is fetched
    # for it: a catalogue hiccup on a resume must not stop the run over it,
    # nor record a date that is in the stack as failed.
    for d in sorted(already & set(status["failed"])) if cfg.skip_existing else ():
        status["failed"].pop(d)
        status["skipped_coregistered"].append(d)
        log.info("%s could not be checked against the catalogue, but it is already "
                 "coregistered, so it needs no download.", d)

    expected = {d: _expected_per_swath(g) for d, g in by_date.items()}
    for d, ents in recoverable.items():
        expected[d] = _expected_per_swath([e["title"] for e in ents.values()])

    # A stack whose coregistration has started cannot take dates its plan
    # does not have, unless they append to an intact stack
    # (stack_backend._extension_problem says why not); find that out BEFORE
    # downloading them, not after. Dates accepted as missing are not fetched.
    plan = sb.planned_dates(cfg) if sb.coreg_started(cfg) else None
    if plan is not None:
        # dates left out on purpose are recorded as that below ('excluded'),
        # not as dates the plan cannot take
        beyond = sorted((set(by_date) | set(recoverable)) - plan - excluded_dates(cfg))
        acc = [d for d in beyond if accept_all or d in accepted]
        unacc = [d for d in beyond if d not in acc]
        why = sb._extension_problem(cfg, unacc, sb._generation_record(cfg)) if unacc else ""
        # An accepted date that is not in the plan because an earlier run
        # could not assemble it keeps that reason (a catalogue defect, say):
        # it was never late to the archive.
        before = earlier_failures(cfg)
        carried.update({d: before[d] for d in acc if d in before})
        status["failed"].update(carried)
        # Every date the plan cannot take is left out of this project's
        # record, whether or not anyone accepted it: say so where QA and
        # provenance read the reasons, or the report counts it nowhere.
        for d in (beyond if why else acc):
            if d in carried:
                continue
            status["dropped"][d] = "not_in_plan"
            status["dropped_detail"][d] = dict(
                {k: v for k, v in short_info.get(d, {}).items() if k != "entries"},
                reason="not_in_plan",
                detail=("listed in allow_missing_dates" if d in acc
                        else f"coregistration had started and it cannot be added: {why}")[:300])
        if why:
            status["not_in_plan"] = unacc
            # what the project already holds, so the record a refused run
            # leaves does not read as nothing processed
            for d in sorted((set(by_date) | set(recoverable)) - set(beyond)):
                safes = sorted(cfg.slc_dir.glob(f"S1*_{d}T*.SAFE"))
                if safes and not verify_safe(safes[0], expected[d], pol):
                    status["existing"].append(d)
                elif d in already:
                    status["skipped_coregistered"].append(d)
            status["processed"] = _processed(status)
            write_short_record()
            _write_assembly_status(cfg, status)
            raise RuntimeError(
                f"{len(unacc)} acquisition(s) in the catalogue are not in this project's "
                f"processing plan: {', '.join(unacc[:12])}{'...' if len(unacc) > 12 else ''}. "
                f"Coregistration has already started and they cannot be added in place: "
                f"{why}. Nothing was downloaded. List them under allow_missing_dates in "
                "the config to carry on without them (they are then recorded as "
                "deliberately excluded), or start a new project for the full record.")
        for d in acc:
            by_date.pop(d, None)
            recoverable.pop(d, None)
        status["accepted_not_in_plan"] = [d for d in acc if d not in carried]
        if unacc:
            log.info("%d new acquisition(s) after the end of the plan will be added to the "
                     "stack: %s", len(unacc), ", ".join(unacc))
    write_short_record()

    skip = sorted(excluded_dates(cfg) & (set(by_date) | set(recoverable)))
    if skip:
        log.info("Not downloading %d date(s) left out of the stack (stack_exclude_dates): %s",
                 len(skip), ", ".join(skip))
        status["excluded"] = skip
        for d in skip:
            by_date.pop(d, None)
            recoverable.pop(d, None)
    out = []
    todo = []
    for date in sorted(set(by_date) | set(recoverable)):
        existing = sorted(cfg.slc_dir.glob(f"S1*_{date}T*.SAFE"))
        if cfg.skip_existing and existing:
            why = verify_safe(existing[0], expected[date], pol)
            if not why:
                out.append(existing[0])
                status["existing"].append(date)
                continue
            # an interrupted or short assembly: never hand it to stackSentinel
            for e in existing:
                _quarantine(cfg, e, why)
        if cfg.skip_existing and date in already:
            status["skipped_coregistered"].append(date)
            continue
        todo.append(date)

    # Each date is an independent burst2safe call writing to its own SAFE
    # directory, and the 48 minutes a 31-date stack spent here were almost entirely
    # ASF latency on three small fetches per date - so run them in a pool.
    # Measured serially: ~93 s per date; the network, not the disk, is idle.
    workers = max(1, int(getattr(cfg, "download_workers", 4) or 1))
    n = len(todo)
    auth_failed = threading.Event()

    def _assemble(date):
        """(date, SAFE path or None, reason). Assembled in a hidden partial
        directory, verified, then renamed into slc/ - a partial SAFE never
        sits where stackSentinel globs."""
        from . import procs
        procs.install_teardown()
        tmp = _partial_dir(cfg, date)
        why = ""
        defects = []                      # one entry per failed attempt: defect name or None
        for attempt in range(1, ASSEMBLE_ATTEMPTS + 1):
            if auth_failed.is_set():
                return date, None, "not attempted: the Earthdata login was refused"
            if procs.stopping():
                return date, None, "not attempted: the run was stopped"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            if date in recoverable:
                log.info("assembling %s from %d bursts via the burst extractor (attempt %d)",
                         date, len(recoverable[date]), attempt)
                try:
                    rc, text = recover_from_extractor(cfg, date, recoverable[date], tmp)
                except Exception as exc:                        # noqa: BLE001
                    rc, text = 1, f"{type(exc).__name__}: {exc}"
            else:
                granules = by_date[date]
                cmd = ["conda", "run", "-n", cfg.burst2safe_env, "--no-capture-output",
                       "burst2safe", "--output-dir", str(tmp), *sorted(granules)]
                log.info("assembling %s from %d bursts (attempt %d)", date, len(granules), attempt)
                # In its own process group and registered for teardown:
                # stopping the run must stop the downloads too (four were
                # left running after the first P2 launch was killed). Run in
                # the partial directory so burst2safe's intermediate files
                # go where a failed attempt is cleared.
                proc = procs.run_group(cmd, timeout=ASSEMBLE_TIMEOUT_S, cwd=str(tmp))
                rc, text = proc.returncode, (proc.stderr or "") + (proc.stdout or "")
            if rc != 0 and auth_refused(text):
                auth_failed.set()
                log.error("ASF refused the login while assembling %s:\n%s", date, (text or "")[-600:])
                return date, None, "login refused (HTTP 401/403)"
            made = sorted(tmp.glob(f"S1*_{date}T*.SAFE"))
            if rc != 0 or not made:
                tail = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
                why = (f"exit {rc}" + (" (timed out)" if rc == -9 else "")
                       + (f": {tail[-1][:200]}" if tail else "") if rc != 0
                       else "no SAFE was produced")
            else:
                # Verify the outcome, not the exit code: a SAFE must carry
                # every requested burst or ISCE fails on it hours from now.
                why = verify_safe(made[0], expected[date], pol)
                if not why:
                    dest = cfg.slc_dir / made[0].name
                    if dest.exists():
                        shutil.rmtree(dest, ignore_errors=True)
                    made[0].rename(dest)
                    shutil.rmtree(tmp, ignore_errors=True)
                    return date, dest, ""
            defects.append(catalogue_defect(text) if rc != 0 else None)
            log.warning("assembling %s failed on attempt %d of %d: %s\n%s", date, attempt,
                        ASSEMBLE_ATTEMPTS, why, (text or "")[-1500:])
            if attempt < ASSEMBLE_ATTEMPTS:
                time.sleep(ASSEMBLE_BACKOFF_S[min(attempt, len(ASSEMBLE_BACKOFF_S)) - 1])
        shutil.rmtree(tmp, ignore_errors=True)
        if defects and all(defects) and len(set(defects)) == 1:
            return date, None, DEFECT_PREFIX + defects[0]
        return date, None, why

    if todo:
        # Fail on a refused login now, not after ~270 identical errors.
        code, msg = _earthdata_login_status()
        if code in ("missing", "rejected", "not_authorised"):
            status["processed"] = _processed(status)
            _write_assembly_status(cfg, status)
            raise RuntimeError(msg)
        if code != "ok":
            log.warning("%s Trying the downloads anyway.", msg)
        log.info("Assembling %d acquisition(s) with %d concurrent downloads", n, workers)
        progress.emit(cfg.project_dir, "download", "start", done=0, total=n)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (date, safe, why) in enumerate(pool.map(_assemble, todo), 1):
                if safe is not None:
                    out.append(safe)
                    status["recovered" if date in recoverable else "assembled"].append(date)
                    log.info("[%d/%d] %s ready", i, n, date)
                else:
                    status["failed"][date] = why
                    log.error("[%d/%d] %s FAILED after %d attempts: %s", i, n, date,
                              ASSEMBLE_ATTEMPTS, why)
                progress.emit(cfg.project_dir, "download", "running", done=i, total=n,
                              msg=(f"{len(status['failed'])} failed" if status["failed"]
                                   else f"{date} ready"))
        shutil.rmtree(cfg.slc_dir / ".partial", ignore_errors=True)

    failed = status["failed"]
    unaccepted = sorted(d for d in failed if not (accept_all or d in accepted))
    defect_dates = {d: failed[d][len(DEFECT_PREFIX):] for d in unaccepted
                    if str(failed[d]).startswith(DEFECT_PREFIX)}
    cap = max(AUTO_ACCEPT_MIN, int(AUTO_ACCEPT_MAX_SHARE * max(len(catalogue_dates), 1)))
    if defect_dates and getattr(cfg, "auto_accept_catalogue_defects", True):
        if len(defect_dates) <= cap:
            written = record_auto_accepted(cfg, defect_dates)
            status["auto_accepted"] = {"dates": defect_dates, "config": str(written) if written else None,
                                       "cap": cap}
            log.warning("%d date(s) cannot be assembled because of known ASF catalogue defects and are "
                        "accepted as missing automatically%s: %s", len(defect_dates),
                        f" (added to allow_missing_dates in {written})" if written else
                        " for this run only (the config has no file to record them in)",
                        "; ".join(f"{d} - {r}" for d, r in sorted(defect_dates.items())))
            accepted |= set(defect_dates)
            unaccepted = [d for d in unaccepted if d not in defect_dates]
        else:
            log.error("%d dates failed with catalogue defects, more than the %d accepted automatically "
                      "(%d%% of %d dates): something systematic is wrong, so they are not accepted.",
                      len(defect_dates), cap, round(100 * AUTO_ACCEPT_MAX_SHARE), len(catalogue_dates))
    status["processed"] = _processed(status)
    status["accepted_failures"] = sorted(set(failed) - set(unaccepted))
    _write_assembly_status(cfg, status)
    if todo:
        progress.emit(cfg.project_dir, "download", "failed" if unaccepted or auth_failed.is_set()
                      else "done", done=n, total=n,
                      msg=(f"{len(unaccepted)} date(s) could not be downloaded: "
                           + ", ".join(unaccepted[:8])) if unaccepted else "")
    if auth_failed.is_set():
        raise RuntimeError(
            "ASF refused the NASA Earthdata login part-way through the downloads (HTTP "
            "401/403) - the password in ~/.netrc is wrong or has expired, or the account "
            "has not authorised ASF data access. Fix the login (the dashboard's preflight "
            "checks it), then start again: everything already downloaded is kept.")
    if unaccepted:
        raise RuntimeError(
            f"{len(unaccepted)} acquisition(s) could not be downloaded after "
            f"{ASSEMBLE_ATTEMPTS} attempts each: "
            + "; ".join(f"{d} ({failed[d]})" for d in unaccepted[:8])
            + ("..." if len(unaccepted) > 8 else "")
            + ". Carrying on would leave them out of the record without anyone choosing "
            "that. Start the run again to retry them (everything already downloaded is "
            "kept), or, if ASF cannot serve them, list them under allow_missing_dates in "
            f"the config. Details: {cfg.slc_dir / ASSEMBLY_STATUS_FILE}.")
    if failed:
        log.warning("%d acquisition(s) could not be downloaded and are accepted as missing "
                    "(allow_missing_dates): %s", len(failed), ", ".join(sorted(failed)))

    if not out and not already:
        raise RuntimeError("burst2safe produced no SAFEs - see the errors above.")
    if not out:
        log.info("Nothing to assemble - every date is already coregistered.")
        return []
    total = sum(f.stat().st_size for s in out for f in s.rglob("*") if f.is_file())
    log.info("Assembled %d of %d acquisitions, %.1f GB total (%d already coregistered, "
             "%d left out as short in the archive or not in the plan, %d accepted as missing)",
             len(out), len(catalogue_dates), total / 1024 ** 3,
             len(status["skipped_coregistered"]), len(status["dropped"]),
             len(status["accepted_failures"]))
    return out
