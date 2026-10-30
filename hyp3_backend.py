"""Sentinel-1 client backend: order interferograms from ASF HyP3.

Replaces stages 2-3 (pair selection and interferogram formation). HyP3
runs essentially the same ISCE2 code on ASF's hardware and returns
geocoded, unwrapped interferograms ready for MintPy - which supports
HyP3 natively, so none of the topsApp metadata staging applies here.

Use this for client work: faster, reproducible, and far less to maintain.
Use the local topsApp backend when you need processing parameters HyP3
does not expose.

Requires: pip install hyp3_sdk, and a NASA Earthdata login.
Note HyP3 allocates processing credits per user per month - a large stack
can exhaust them, so check your remaining quota before submitting.
"""
import logging
from datetime import datetime
from pathlib import Path

import asf_search as asf

from .config import PipelineConfig

log = logging.getLogger(__name__)

# 20x4 (range x azimuth) is HyP3's default and gives ~80 m pixels - a
# reasonable balance for site-scale monitoring. 10x2 and 5x1 give finer
# output at the cost of more noise and larger files.
DEFAULT_LOOKS = "20x4"


def find_bursts(cfg: PipelineConfig):
    """Find Sentinel-1 burst SLCs over the AOI."""
    # flightDirection is NOT passed to the query: burst products do not
    # support it as a filter and passing it silently returns zero results.
    # Filter on the returned metadata instead.
    # polarization: InSAR needs CO-polarized data. Sentinel-1 IW returns
    # both VV and VH over most land, and a VH burst will not interfere
    # usefully - without this filter roughly half the results are unusable
    # and the pairing quietly mixes the two.
    opts = dict(
        dataset=asf.DATASET.SLC_BURST,
        start=cfg.start_date,
        end=cfg.end_date,
        intersectsWith=cfg.aoi_wkt,
        polarization=["VV"],
    )
    if cfg.track:
        opts["relativeOrbit"] = cfg.track

    results = list(asf.search(**opts))
    log.info("Found %d burst SLCs before direction filter", len(results))

    # Burst metadata does not carry flightDirection, so direction can only
    # be honoured through the track number. Ascending and descending
    # tracks over one AOI are different geometries measuring different
    # projections of the same motion - mixing them in a stack is wrong in
    # the same way mixing sensors is.
    want = cfg.orbit_direction.upper()
    filtered = [r for r in results
                if (r.properties.get("flightDirection") or "").upper() == want]
    if filtered:
        results = filtered
        log.info("%d bursts on %s passes", len(results), want.lower())
    elif results and not cfg.track:
        tracks = sorted({r.properties.get("pathNumber") for r in results
                         if r.properties.get("pathNumber")})
        log.warning(
            "Burst products do not carry flightDirection, and no track is "
            "set - these %d bursts span tracks %s, which mixes ascending "
            "and descending geometries. Set `track` in the config to one "
            "of them before processing.", len(results), tracks)

    # A burst ID identifies one fixed patch of ground imaged repeatedly, so
    # grouping by it gives a time series per patch. An AOI spanning several
    # bursts needs the multi-burst job type to merge them.
    by_burst = {}
    for r in results:
        bid = r.properties.get("burst", {}).get("fullBurstID") or r.properties["fileID"]
        by_burst.setdefault(bid, []).append(r)
    for bid in by_burst:
        by_burst[bid].sort(key=lambda r: r.properties["startTime"])

    log.info("Covering %d distinct burst footprints", len(by_burst))
    return by_burst


def build_multiburst_pairs(by_burst: dict, max_temporal_days: int,
                           n_neighbors: int):
    """
    Pair by DATE across all burst footprints, not per footprint.

    Each job then carries every burst for that date, and HyP3 merges them
    into one interferogram covering the whole AOI. Two advantages over
    pairing each footprint separately: one continuous product instead of
    several disjoint patches, and one job per date pair rather than one
    per footprint per date pair - which is where the credit saving comes
    from.

    Bursts must be contiguous along-track, and HyP3 accepts at most 15 per
    job.
    """
    by_date: dict[str, list] = {}
    for bid, scenes in by_burst.items():
        for s in scenes:
            day = s.properties["startTime"][:10]
            by_date.setdefault(day, []).append(s)

    # A date missing some footprints would produce a product with a
    # different extent to the others, which MintPy cannot stack cleanly.
    expected = max(len(v) for v in by_date.values())
    complete = {d: v for d, v in by_date.items() if len(v) == expected}
    dropped = sorted(set(by_date) - set(complete))
    if dropped:
        log.warning("Dropping %d dates with partial burst coverage (%d of %d "
                    "footprints): %s", len(dropped), 
                    min(len(by_date[d]) for d in dropped), expected,
                    ", ".join(dropped))

    if expected > 15:
        log.warning("%d bursts per date exceeds HyP3's limit of 15 per job. "
                    "Shrink the AOI or split it.", expected)

    dates = sorted(complete)
    log.info("%d complete dates, %d bursts each", len(dates), expected)

    from datetime import datetime as _dt
    pairs = []
    for i, ref_day in enumerate(dates):
        added = 0
        for sec_day in dates[i + 1:]:
            gap = (_dt.strptime(sec_day, "%Y-%m-%d")
                   - _dt.strptime(ref_day, "%Y-%m-%d")).days
            if gap > max_temporal_days:
                break
            pairs.append((
                ref_day, sec_day,
                [s.properties["fileID"] for s in complete[ref_day]],
                [s.properties["fileID"] for s in complete[sec_day]],
            ))
            added += 1
            if added >= n_neighbors:
                break

    log.info("Built %d multi-burst pairs (= %d HyP3 jobs)", len(pairs), len(pairs))
    return pairs


def check_credits(hyp3, n_jobs: int) -> None:
    """Report remaining quota before committing to a submission."""
    try:
        info = hyp3.my_info()
        remaining = info.get("remaining_credits")
        if remaining is None:
            log.info("HyP3 reports no credit limit on this account")
            return
        log.info("HyP3 credits remaining: %s; this batch is %d jobs",
                 remaining, n_jobs)
        if remaining < n_jobs:
            raise RuntimeError(
                f"Not enough HyP3 credits: {remaining} remaining, {n_jobs} "
                f"needed. Shorten the date range or lower n_neighbors.")
    except RuntimeError:
        raise
    except Exception as exc:
        log.warning("Could not read HyP3 credit balance (%s) - continuing", exc)


def submit_jobs(cfg: PipelineConfig, pairs, looks: str = DEFAULT_LOOKS,
                apply_water_mask: bool = True, job_name: str | None = None,
                dry_run: bool = False):
    """
    Submit multi-burst InSAR jobs to HyP3 and return the batch.

    apply_water_mask defaults on: water decorrelates completely, and
    leaving it in degrades phase unwrapping across the whole scene rather
    than only over the water.

    Set dry_run to see the job count and credit balance without spending
    anything.
    """
    import hyp3_sdk as sdk
    from .data_acquisition import get_credentials

    # hyp3_sdk reads ~/.netrc by default. The rest of this pipeline
    # deliberately prompts instead (a .netrc is easy to mangle, and a
    # password containing '!' gets eaten by bash history expansion), so
    # pass the same credentials through explicitly.
    username, password = get_credentials()
    hyp3 = sdk.HyP3(username=username, password=password)
    name = job_name or f"{cfg.project_dir.name}-{datetime.now():%Y%m%d}"

    check_credits(hyp3, len(pairs))
    if dry_run:
        log.info("Dry run - would submit %d jobs as '%s'", len(pairs), name)
        for ref_day, sec_day, refs, secs in pairs[:3]:
            log.info("   %s -> %s (%d bursts)", ref_day, sec_day, len(refs))
        return hyp3, None

    log.info("Submitting %d multi-burst jobs as '%s' (looks=%s, water_mask=%s)",
             len(pairs), name, looks, apply_water_mask)

    batch = sdk.Batch()
    for ref_day, sec_day, refs, secs in pairs:
        batch += hyp3.submit_insar_isce_multi_burst_job(
            reference=refs, secondary=secs, name=name,
            apply_water_mask=apply_water_mask, looks=looks,
        )
    return hyp3, batch


def watch_and_download(cfg: PipelineConfig, hyp3, batch) -> Path:
    """Block until jobs finish, then download and unzip the products."""
    out_dir = cfg.project_dir / "hyp3_products"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Waiting on %d HyP3 jobs - this typically takes tens of minutes",
             len(batch))
    batch = hyp3.watch(batch)

    succeeded = [j for j in batch if j.succeeded()]
    failed = [j for j in batch if not j.succeeded()]
    if failed:
        log.warning("%d of %d jobs failed and will be excluded",
                    len(failed), len(batch))
        for j in failed:
            log.warning("  failed: %s", j.job_id)

    import zipfile
    for job in succeeded:
        for f in job.download_files(out_dir):
            if f.suffix == ".zip":
                with zipfile.ZipFile(f) as z:
                    z.extractall(out_dir)
                f.unlink()

    log.info("Downloaded %d products to %s", len(succeeded), out_dir)
    return out_dir


def existing_products(cfg: PipelineConfig) -> set[tuple[str, str]]:
    """Date pairs already downloaded, read from the product directory names.

    HyP3 names products ..._IW_<ref>_<sec>_VV_INT80_<hash>, so the dates
    are recoverable without any bookkeeping of our own.
    """
    out_dir = cfg.project_dir / "hyp3_products"
    found = set()
    if not out_dir.is_dir():
        return found
    for d in out_dir.iterdir():
        if not d.is_dir():
            continue
        parts = d.name.split("_")
        for i, tok in enumerate(parts):
            if tok == "IW" and i + 2 < len(parts):
                ref, sec = parts[i + 1], parts[i + 2]
                if len(ref) == 8 and len(sec) == 8 and ref.isdigit():
                    found.add((ref, sec))
                break
    return found


def run(cfg: PipelineConfig, looks: str = DEFAULT_LOOKS) -> Path:
    """Full HyP3 backend: search, pair, submit, wait, download.

    Pairs whose product is already on disk are skipped. Without that check
    a rerun silently submits the whole batch again - HyP3 has no notion of
    "I already made this", so every resubmission costs credits.
    """
    by_burst = find_bursts(cfg)
    if not by_burst:
        raise RuntimeError("No burst SLCs found for this AOI and date range")

    pairs = build_multiburst_pairs(by_burst, cfg.max_temp_baseline, cfg.n_neighbors)
    if not pairs:
        raise RuntimeError("No pairs could be built - widen max_temp_baseline")

    out_dir = cfg.project_dir / "hyp3_products"
    if cfg.skip_existing:
        have = existing_products(cfg)
        todo = [p for p in pairs
                if (p[0].replace("-", ""), p[1].replace("-", "")) not in have]
        if len(todo) < len(pairs):
            log.info("%d of %d pairs already downloaded - submitting %d",
                     len(pairs) - len(todo), len(pairs), len(todo))
        pairs = todo

    if not pairs:
        log.info("Every pair is already on disk; nothing to submit")
        return out_dir

    hyp3, batch = submit_jobs(cfg, pairs, looks=looks)
    return watch_and_download(cfg, hyp3, batch)


def mintpy_config_lines(cfg: PipelineConfig, products_dir: Path) -> str:
    """
    MintPy loader settings for HyP3 products.

    MintPy reads HyP3 output natively via prep_hyp3, so none of the
    metadata staging the topsApp path needs applies here.
    """
    # The water mask is deliberately NOT loaded. HyP3 ships it one row
    # taller than the interferograms, and MintPy indexes one array with
    # the other during network plotting - which fails outright on the
    # shape mismatch. It is also redundant: jobs are submitted with
    # apply_water_mask=True, so water is already masked during unwrapping,
    # which is where it actually affects the result.
    #
    # connComp is loaded instead. It flags regions the unwrapper could not
    # tie together, which is the layer that matters most over fast
    # subsidence where 2*pi errors appear.
    return f"""\
mintpy.load.processor              = hyp3
mintpy.load.unwFile                = {products_dir}/*/*unw_phase.tif
mintpy.load.corFile                = {products_dir}/*/*corr.tif
mintpy.load.connCompFile           = {products_dir}/*/*conncomp.tif
mintpy.load.demFile                = {products_dir}/*/*dem.tif
mintpy.load.incAngleFile           = {products_dir}/*/*lv_theta.tif
mintpy.load.azAngleFile            = {products_dir}/*/*lv_phi.tif
"""
