"""Deliverable figures, drawn from the products on disk into deliverables/figures/.

    python -m insar_pipeline.figures <config.yaml>

Every figure reads the delivered rasters, the interferograms in the stack, the
MintPy stack and quality_assessment.json - never a log. figures.json beside
the PNGs carries each figure's title and interpretation note, which the
summary page embeds unchanged.
"""
import datetime as _dt
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
MM_IN = 25.4
KM2_MI2 = 1 / 2.58999


def _rings(path):
    """Outer rings of every polygon in a GeoJSON file, or [] when no file is
    configured. A screening project made without mine outlines has
    deramp_exclude_geojson = "" (write_project sets it only when a mine or
    known-movement layer exists), and Path("") is the current directory:
    reading it raised IsADirectoryError and took every technical figure with
    it. A configured file that is missing is still an error."""
    if not str(path or "").strip():
        return []
    g = json.loads(Path(path).read_text()); out = []
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            out.append(np.array(poly[0]))
    return out


def _xml_dims(xml):
    import re
    t = Path(xml).read_text()
    return (int(re.search(r'<property name="length">\s*<value>(\d+)', t).group(1)),
            int(re.search(r'<property name="width">\s*<value>(\d+)', t).group(1)))


def _fmt_pair(pair):
    a, b = pair.split("_")
    return f"{a[:4]}-{a[4:6]}-{a[6:]} → {b[:4]}-{b[4:6]}-{b[6:]}"


def _pair_days(pair) -> int:
    a, b = pair.split("_")
    return (_dt.datetime.strptime(b, "%Y%m%d") - _dt.datetime.strptime(a, "%Y%m%d")).days


def _season(pair) -> str:
    """Northern-hemisphere season of the pair's middle date (the sites are all
    in the Midwest; the caption says which months, so the word is a label)."""
    a, b = (_dt.datetime.strptime(d, "%Y%m%d") for d in pair.split("_"))
    m = (a + (b - a) / 2).month
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer"}.get(m, "autumn")


def _pick_pairs(cfg):
    """The first pair on the stack reference date (normally the best-behaved
    kind) and the June-August pair that took longest to unwrap (the worst;
    the slowest pair of any month when the record has no summer pair), both
    read from the record. Returns (first, worst, worst_is_summer)."""
    hf = cfg.stack_dir / "unwrap_history.json"
    if not hf.exists():             # no interferograms of our own (DISP-S1)
        return None, None, False
    hist = json.loads(hf.read_text())
    if not hist:
        return None, None, False
    ref = getattr(cfg, "stack_reference_date", "") or ""
    first = sorted(p for p in hist if ref and p.startswith(ref + "_")) or sorted(hist)
    summer = [p for p, _ in sorted(hist.items(), key=lambda kv: -kv[1]) if p[4:6] in ("06", "07", "08")]
    return first[0], (summer[0] if summer else max(hist, key=hist.get)), bool(summer)


def _unit_files(cfg):
    """(display unit tag, factor to mm) of the delivered rasters, from the
    same rule deliverables uses to name them. The figures used to read
    los_velocity_in_per_year.tif whatever display_units said, so a project
    set to mm lost every map, and one switched from in to mm drew the stale
    inch raster."""
    try:
        from .deliverables import _unit_label
        _, unit, div = _unit_label(cfg)
    except (ImportError, AttributeError):
        unit = "in" if str(getattr(cfg, "display_units", "in") or "in").lower().startswith("in") else "mm"
        div = MM_IN if unit == "in" else 1.0
    return unit, float(div)


def _route(cfg) -> str:
    return str(getattr(cfg, "backend", "") or "").lower()


def _coherence_source(cfg, Q) -> str:
    """What the temporal-coherence layer is, in a phrase: the QA records it
    (coverage.coherence_source); the route is the fallback for older QA files.
    The phase-linked layer is MiaplPy's mini-stack coherence, not MintPy's
    network temporal coherence, and the two are not comparable."""
    src = ((Q.get("checks") or {}).get("coverage") or {}).get("coherence_source")
    if src:
        return str(src)
    return {"miaplpy": "MiaplPy phase-linking (average mini-stack) temporal coherence - not MintPy's",
            "disp": "the OPERA DISP-S1 product's temporal coherence"}.get(
        _route(cfg), "MintPy's temporal coherence of the interferogram network")


def _noise_sentence(cov: dict, thr) -> str:
    """How close the working threshold sits to what pure noise scores on this
    coherence layer: the QA's measured floor, else the noise level the
    time-series step recorded (linked_noise_level: 0.384 at Site B, 0.387 at
    Site A, so 0.4 is ~0.015 above noise). '' when neither is recorded."""
    nl = cov.get("noise_level") or {}
    # the QA's floor (typical noise + 3 spreads) is what its
    # threshold_above_noise judges against; the typical level is its mode
    # (E21, 7 Oct 2026; a QA record from before then has its median), else
    # the run's own level
    typ = nl.get("mode")
    typ = nl.get("median") if typ is None else typ
    typ = nl.get("linked_noise_level_from_run") if typ is None else typ
    top = nl.get("floor")
    try:
        thr = float(thr)
        typ = None if typ is None else float(typ)
        top = None if top is None else float(top)
    except (TypeError, ValueError):
        return ""
    if typ is None and top is None:
        return ""
    # the noise population is decorrelated cells; call it open water only
    # when most of it lies on mapped water (Site A 88%, Site B 29%)
    try:
        share = nl.get("share_on_mapped_water")
        share = None if share is None else float(share)
    except (TypeError, ValueError):
        share = None
    what = ("open water" if share is not None and share >= 0.5 else
            "decorrelated ground" if share is not None else "decorrelated ground and water")
    if top is None:
        return (f". Pure noise ({what}) scores about {typ:.3f} on this layer, so the threshold {thr:g} is "
                f"{thr - typ:+.3f} from it" + (": cells just above it are not reliable on this layer alone"
                                               if thr - typ < 0.05 else ""))
    lead = f". Pure noise ({what}) scores about {typ:.3f} on this layer and" if typ is not None else \
        f". Pure noise ({what}) on this layer scores"
    return (f"{lead} up to about {top:.3f}, so the threshold {thr:g} is {thr - top:+.3f} from the top of the noise"
            + (": cells just above it are not reliable on this layer alone" if thr - top < 0.05 else ""))


def _unwrapper(cfg) -> str:
    if str(getattr(cfg, "stack_unw_method", "snaphu")).lower() == "icu":
        return "ICU"
    return "snaphu 2" if str(getattr(cfg, "stack_unw_engine", "isce")).lower() == "snaphu2" else "snaphu"


def _dates_bperp_file(cfg):
    """A MintPy time series holding `date` and `bperp` (only those small
    datasets are read): the radar timeseries.h5, else the delivered chain
    file (cleanup can delete the radar one), else any timeseries*.h5."""
    import h5py
    cands = [cfg.mintpy_dir / "timeseries.h5"]
    try:
        from .deliverables import timeseries_file
        cands.append(Path(timeseries_file(cfg)))
    except Exception:                                         # noqa: BLE001
        pass
    cands += sorted(cfg.mintpy_dir.glob("timeseries*.h5")) + sorted((cfg.mintpy_dir / "geo").glob("geo_timeseries*.h5"))
    for p in cands:
        try:
            if p.exists():
                with h5py.File(p, "r") as f:
                    if "date" in f and "bperp" in f:
                        return p
        except OSError:
            continue
    return None


def _product_wavelength_m(cfg) -> float:
    """The radar wavelength the product states (MintPy's WAVELENGTH, read from
    the velocity, else a time series), via stack_qc.product_wavelength_m;
    Sentinel-1's when no product states one (M1, 8 Oct 2026)."""
    import h5py
    from . import stack_qc
    M = Path(cfg.mintpy_dir)
    for p in [M / "geo" / "geo_velocity.h5", M / "velocity.h5"] + sorted(M.glob("timeseries*.h5")):
        try:
            if p.exists():
                with h5py.File(p, "r") as f:
                    attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
                if "WAVELENGTH" in attrs or "PLATFORM" in attrs:
                    return stack_qc.product_wavelength_m(attrs)
        except OSError:
            continue
    return stack_qc.SENTINEL1_WAVELENGTH_M


def _posting_m(path, bbox) -> float | None:
    """Grid spacing of a delivered raster in metres (mean of the two axes)."""
    import rasterio
    try:
        with rasterio.open(path) as src:
            rx, ry = abs(src.res[0]), abs(src.res[1])
            geographic = src.crs is None or src.crs.is_geographic
    except Exception:                                         # noqa: BLE001
        return None
    if geographic:
        lat = (bbox[1] + bbox[3]) / 2
        rx, ry = rx * 111_320.0 * np.cos(np.radians(lat)), ry * 111_320.0
    return float((rx + ry) / 2)


def _resolution(cfg) -> dict:
    """Posting and the ground each value draws on, from resolution.py (route
    aware: phase-linked support is the SHP window plus looks, about 55 m x
    230 m on Sentinel-1, not the classic route's ~41 m look cell). Falls back
    to the posting from the product attributes alone."""
    attrs = None
    try:
        import h5py
        from .deliverables import _product
        with h5py.File(_product(cfg, "velocity.h5"), "r") as f:
            attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
    except Exception:                                         # noqa: BLE001
        attrs = None
    try:
        from .resolution import effective_resolution
        return dict(effective_resolution(cfg, attrs) or {})
    except (ImportError, AttributeError):
        pass
    if attrs and {"X_STEP", "Y_STEP"} <= set(attrs):
        from .significance import pixel_metres
        dy, dx = pixel_metres(attrs)
        return {"posting_m": round(float((dx + dy) / 2), 1), "support_m": None, "route": _route(cfg)}
    return {}


# the figures make_figures draws itself (winter/summer pairs are carried over
# on purpose when the stack's interferograms have been deleted)
_OWN_FIGURES = ("velocity_relative.png", "cumulative_relative.png", "temporal_coherence.png",
                "closure_phase.png", "network.png", "summary_table.png")


# the two interferogram figures; carried over when the stack's
# interferograms have been deleted, if they show this route's kind of pair
PAIR_FIGURES = ("winter_pair.png", "summer_pair.png")
PAIR_KIND_LINKED = "phase-linked"
PAIR_KIND_RAW = "raw two-image"
def ground_words(cfg, Q: dict | None) -> str:
    """The comparison ground's words: the quality assessment's (its
    comparison_ground, from the measured land cover or a control_ground_label
    someone set; 28 Oct 2026), else the config's label. The figures are drawn
    before the client report, which measures its own ground the same way."""
    cg = (Q or {}).get("comparison_ground") or {}
    return str(cg.get("words") or cg.get("label") or getattr(cfg, "control_ground_label", "") or "farmland")


def _noun(words: str) -> str:
    from .report_insights import ground_noun
    return ground_noun(words)


# the rate map's class means are the quality assessment's far field: drawn
# before the client report exists, so the first farmland pass (E14: first-pass
# values stay in technical files only if labelled)
FIRST_PASS_MEANS = ("these are the FIRST {ctrl_n} pass, measured by the quality assessment before the client report "
                    "takes the movement it finds out of the {ctrl}, so the report's own figures can differ from "
                    "this map by a constant per class, as client_report.json farmland_second_pass records")


def _pair_kind(entry: dict) -> str:
    """The kind of interferogram a figures.json pair entry shows: recorded
    since 7 Oct 2026 ('kind'); before then read from its caption (the
    phase-linked captions name the phase-linked interferogram, the raw
    ones do not)."""
    k = entry.get("kind")
    if k:
        return str(k)
    return PAIR_KIND_LINKED if "phase-linked" in str(entry.get("caption") or "").lower() else PAIR_KIND_RAW


def _carry_pair_figures(out_dir: Path, D: Path, linked: bool) -> list:
    """
    The pair figures (and figures.json entries) kept when the stack's
    interferograms are gone. Until 7 Oct 2026 every earlier figure was kept
    as it was: Site A's phase-linked record carried two raw ISCE
    interferograms drawn on 28 Sep under captions saying 'Pairs like this
    are what the record is built from' (final verification, 7 Oct 2026).
    Now a figure of this route's kind of pair is kept with its caption
    saying when it was drawn and that the pairs have since been deleted; one
    of another kind (or with no entry saying what it is) is moved to
    deliverables/superseded/ - moved, never deleted - and left out.
    """
    man = out_dir / "figures.json"
    try:
        old = json.loads(man.read_text()) if man.exists() else []
    except (OSError, ValueError):
        old = []
    want = PAIR_KIND_LINKED if linked else PAIR_KIND_RAW
    entries = {m.get("file"): m for m in old if isinstance(m, dict) and m.get("file") in PAIR_FIGURES}
    keep, dest = [], None
    for name in PAIR_FIGURES:
        png = out_dir / name
        if not png.exists():
            continue
        m = entries.get(name)
        if m is not None and _pair_kind(m) == want:
            m = dict(m, kind=want)
            when = m.get("drawn") or _dt.date.fromtimestamp(png.stat().st_mtime).isoformat()
            m["drawn"] = when
            note = (f"Drawn on {when} from the stack's interferograms, which have since been deleted from disk; "
                    "kept as an illustration of the record's pairs. ")
            if not str(m.get("caption") or "").startswith("Drawn on "):
                m["caption"] = note + str(m.get("caption") or "")
            keep.append(m)
            continue
        if dest is None:
            dest = D / "superseded" / _dt.datetime.now().strftime("figures_%Y%m%d_%H%M%S")
            dest.mkdir(parents=True, exist_ok=True)
        try:
            png.replace(dest / name)
            log.warning("%s shows %s interferograms, not this record's %s pairs: moved to %s",
                        name, _pair_kind(m) if m is not None else "unrecorded", want, dest)
        except OSError as exc:
            log.warning("Could not move the superseded %s aside (%s)", name, exc)
    return keep


def make_figures(cfg) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    import h5py
    import rasterio
    from rasterio.windows import from_bounds

    P = cfg.project_dir
    out_dir = P / "deliverables" / "figures"; out_dir.mkdir(parents=True, exist_ok=True)
    Q = json.loads((P / "quality_assessment.json").read_text())
    S_f = P / "deliverables" / "summary.json"
    S = json.loads(S_f.read_text()) if S_f.exists() else {}
    bbox = tuple(cfg.bbox)
    route = _route(cfg)
    linked = route == "miaplpy"
    ctrl = ground_words(cfg, Q)
    ctrl_n = _noun(ctrl)                             # titles and labels: no land-cover note
    unit, to_mm = _unit_files(cfg)
    # the pixel MintPy referenced (REF_Y/REF_X), not the configured lat/lon
    rp = Q.get("reference_pixel") or {}
    if rp.get("lat") is not None:
        ref = (rp["lat"], rp["lon"])
    elif str(getattr(cfg, "reference_lalo", "") or "").strip():
        ref = tuple(float(v) for v in cfg.reference_lalo.split(","))
    else:
        ref = None
    mp = Q.get("findings", {}).get("monitoring_point", {}) or {}
    mon = (mp["lat"], mp["lon"]) if mp.get("lat") is not None and mp.get("lon") is not None else None
    FF = Q["far_field"]
    # far-field mean rate of each temporal-coherence class: the rate map is
    # drawn class-matched, like every figure quoted from it
    class_means = [(float(k.split("-")[0]), float(k.split("-")[1]), v) for k, v in FF["class_mean_rate_mm_yr"].items()]
    # the far field's own change over the record. Its series is MintPy's,
    # referenced to REF_DATE, so this is last MINUS FIRST - the last value
    # alone is the change since REF_DATE (-7 mm instead of -96 on the record)
    far_cum = FF["endpoint_total_mm"]
    # the summary's wavelength, else the product's own (M1, 8 Oct 2026: an
    # L-band record drawn with Sentinel-1's wavelength would scale every
    # wrapped-phase figure by 4.4); Sentinel-1's only when neither states one
    wl = float(S.get("wavelength_m") or _product_wavelength_m(cfg))
    phase2range = -wl / (4 * np.pi)       # MintPy's convention: + = toward the satellite
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 110})
    panels = _rings(getattr(cfg, "deramp_exclude_geojson", ""))
    south = _rings(getattr(cfg, "acceptance_panel_geojson", ""))
    manifest = []

    def overlay(ax, legend=True):
        for r in panels: ax.plot(r[:, 0], r[:, 1], color="k", lw=0.8)
        for r in south: ax.plot(r[:, 0], r[:, 1], color="#eb6834", lw=1.6)
        if ref is not None:
            ax.plot(ref[1], ref[0], marker="*", ms=13, mfc="#ffd400", mec="k", mew=0.8, ls="none")
        if mon is not None:
            ax.plot(mon[1], mon[0], marker="o", ms=9, mfc="none", mec="#1baf7a", mew=2, ls="none")
        ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
        ax.set_aspect(1 / np.cos(np.radians((bbox[1] + bbox[3]) / 2)))
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.tick_params(labelsize=8)
        if legend:
            # only what is drawn: a project without mine outlines has no
            # outline entry, one without a QA monitoring point no circle
            h = ([Line2D([], [], color="k", lw=0.8, label="mapped mines / known movement")] if panels else []) \
                + ([Line2D([], [], color="#eb6834", lw=1.6, label="named panel")] if south else []) \
                + ([Line2D([], [], marker="*", ms=11, mfc="#ffd400", mec="k", ls="none", label="reference pixel")] if ref is not None else []) \
                + ([Line2D([], [], marker="o", ms=8, mfc="none", mec="#1baf7a", mew=2, ls="none", label="deepest 1 km patch (a selected extreme)")] if mon is not None else [])
            if h:
                ax.legend(handles=h, loc="lower left", fontsize=7.5, framealpha=0.9)

    def dual_cbar(im, ax, ticks, unit_mm, unit_in, label):
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=ticks)
        cb.ax.set_yticklabels([f"{t:+.0f} {unit_mm}  |  {t / MM_IN:+.2f} {unit_in}" for t in ticks], fontsize=8)
        cb.set_label(label, fontsize=8)

    def save(fig, name, title, caption, **extra):
        fig.tight_layout(); fig.savefig(out_dir / name, dpi=110); plt.close(fig)
        manifest.append({"file": name, "title": title, "caption": caption, **extra})
        log.info("Wrote %s", out_dir / name)

    def read_tif(path):
        with rasterio.open(path) as src:
            win = from_bounds(*bbox, transform=src.transform)
            a = src.read(1, window=win, masked=True).astype(float)
            b = src.window_bounds(win)
        return np.ma.filled(a, np.nan), (b[0], b[2], b[1], b[3])

    D = P / "deliverables"
    # ---------------------------------------------------------------- rate, relative to the far field
    vel_tif = D / f"los_velocity_{unit}_per_year.tif"
    if not vel_tif.exists():
        raise FileNotFoundError(f"{vel_tif.name} is not in {D}: display_units is {getattr(cfg, 'display_units', 'in')!r} "
                                "and the deliverables were exported in other units or not at all - re-export the deliverables")
    v, ext = read_tif(vel_tif)
    tcm, _ = read_tif(D / "quality_temporal_coherence.tif")
    posting = _posting_m(vel_tif, bbox)
    cell = f"{posting:.0f} m" if posting else "product"
    v_mm = np.full(v.shape, np.nan)
    for lo_, hi_, mean_ in class_means:
        sel = (tcm >= lo_) & ((tcm < hi_) if hi_ < 1.0 else (tcm <= 1.0))
        v_mm[sel] = v[sel] * to_mm - mean_
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    im = ax.imshow(v_mm, extent=ext, cmap="RdBu", norm=TwoSlopeNorm(0, -30, 30), interpolation="nearest")
    overlay(ax)
    dual_cbar(im, ax, [-30, -20, -10, 0, 10, 20, 30], "mm/yr", "in/yr", f"LOS rate relative to the surrounding {ctrl_n} (+ toward satellite)")
    ax.set_title(f"Line-of-sight rate over the record, relative to {ctrl_n} of the same coherence")
    lows = ", ".join(f"{lo_:.1f}–{min(hi_, 1.0):.1f}: {mean_:+.1f}" for lo_, hi_, mean_ in class_means)
    sd1 = (mp.get("extreme") or {}).get("farmland_1km_spread_1sigma_mm_yr")
    save(fig, "velocity_relative.png", f"Line-of-sight rate, relative to {ctrl_n} of the same coherence",
         f"Each {cell} cell's fitted rate over the record minus the mean rate of far-field {ctrl} in the same temporal-coherence class (far-field means, mm/yr: {lows}; {FIRST_PASS_MEANS.format(ctrl=ctrl, ctrl_n=ctrl_n)}), so that zero means 'moving like {ctrl} of the same kind' and the coherence-dependent bias is removed rather than painted red. Red is motion away from the satellite (subsidence), blue toward it; white is below the coherence threshold. A single cell is noisy - its rate is repeatable only to several mm/yr on low-coherence ground - and whole fields differ from each other"
         + (f": 1 km patches of ordinary {ctrl} spread by about {sd1} mm/yr" if sd1 is not None else "")
         + ". So only a patch that is both large and strongly red stands out."
         + (" The green circle is the deepest 1 km patch in the AOI, which is a selected extreme (see the summary table for how it compares with chance)." if mon is not None else "")
         + (" The star is the pixel the record is referenced to; absolute rates on the delivered rasters are relative to it." if ref is not None else ""))

    # ---------------------------------------------------------------- cumulative displacement
    cums = sorted((p for p in D.iterdir() if p.name.startswith(f"los_cumulative_displacement_{unit}_") and p.suffix == ".tif"),
                  key=lambda p: p.stat().st_mtime)
    if not cums:
        log.warning("cumulative figure skipped: no los_cumulative_displacement_%s_*.tif in %s", unit, D)
    for cum in cums[-1:]:                       # the newest raster in the configured unit
        c, ext = read_tif(cum)
        c_mm = c * to_mm - far_cum
        lim = float(np.nanpercentile(np.abs(c_mm), 98)) if np.isfinite(c_mm).any() else 50.0
        lim = max(50.0, 50 * np.ceil(lim / 50))
        fig, ax = plt.subplots(figsize=(9.2, 7.6))
        im = ax.imshow(c_mm, extent=ext, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
        overlay(ax)
        dual_cbar(im, ax, list(np.linspace(-lim, lim, 7)), "mm", "in", "cumulative LOS displacement over the record, relative to the far field")
        d0, d1 = cum.stem.split("_")[-2:]
        ax.set_title(f"Cumulative line-of-sight displacement, {d0[:4]}-{d0[4:6]} → {d1[:4]}-{d1[4:6]}")
        # how much the unmatched map still carries by coherence, measured on
        # this map: the mean of the least and the most coherent class
        by_cls = []
        for lo_, hi_, _m in class_means:
            sel = (tcm >= lo_) & ((tcm < hi_) if hi_ < 1.0 else (tcm <= 1.0)) & np.isfinite(c_mm)
            if sel.sum() >= 200:
                by_cls.append((f"{lo_:.1f}–{min(hi_, 1.0):.1f}", float(np.mean(c_mm[sel]))))
        bias_txt = (f" Over the record the least coherent class ({by_cls[0][0]}) averages {by_cls[0][1]:+.0f} mm ({by_cls[0][1] / MM_IN:+.1f} in) "
                    f"and the most coherent ({by_cls[-1][0]}) {by_cls[-1][1]:+.0f} mm ({by_cls[-1][1] / MM_IN:+.1f} in) on this map."
                    if len(by_cls) >= 2 else "")
        noise = ((Q.get("checks") or {}).get("single_epoch_noise") or {}).get("robust_mm")
        noise_txt = f"about {noise:.0f} mm ({noise / MM_IN:.1f} in)" if isinstance(noise, (int, float)) else "several millimetres"
        save(fig, "cumulative_relative.png", "Cumulative displacement over the record, relative to the far field",
             f"Line-of-sight change between the first and last acquisition, minus the far field's own change over the same dates ({far_cum:+.0f} mm, {far_cum / MM_IN:+.1f} in). This map is NOT matched by coherence class, so any coherence-dependent (distributed-scatterer) bias still shows across the less coherent ground.{bias_txt} Use the rate map for comparisons. It rests on two single dates, each with {noise_txt} of noise, so it is a check for motion that started and stopped, not a measurement: the totals quoted elsewhere are fitted to every date, not read off two.")

    # ---------------------------------------------------------------- temporal coherence
    tc, ext = read_tif(D / "quality_temporal_coherence.tif")
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    im = ax.imshow(tc, extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    overlay(ax)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("temporal coherence (0 = noise, 1 = perfectly consistent)", fontsize=8)
    ax.set_title("Temporal coherence: how consistent each cell's phase history is")
    src_txt = _coherence_source(cfg, Q)
    gate = float(getattr(cfg, "min_unwrap_consistency", 0.0) or 0.0)
    if linked:
        what = ("It measures how well each cell's phases agree with one history within MiaplPy's mini-stacks; it is not "
                "MintPy's network temporal coherence and does not show whether the interferograms were unwrapped consistently. "
                + (f"The report figures also drop cells whose MintPy unwrapping-consistency coherence is below {gate:g} "
                   "(min_unwrap_consistency). " if gate > 0 else
                   "Unwrapping consistency is not screened here (min_unwrap_consistency is off). "))
    else:
        what = "1.0 means every observation agrees perfectly with one displacement history; values near 0 mean the phase is noise. "
    cov = (Q.get("checks") or {}).get("coverage") or {}
    ref_tc = None
    if ref is not None:
        with rasterio.open(D / "quality_temporal_coherence.tif") as src:
            try:
                rr, rc_ = src.index(ref[1], ref[0])
                if 0 <= rr < src.height and 0 <= rc_ < src.width:
                    ref_tc = float(src.read(1, window=((rr, rr + 1), (rc_, rc_ + 1)))[0, 0])
            except Exception:                                 # noqa: BLE001
                ref_tc = None
    save(fig, "temporal_coherence.png", "Where the measurement is trusted",
         f"Temporal coherence for every cell ({src_txt}). {what}Cells below {cfg.min_temporal_coherence} are removed from every product (white here)"
         + (f"; {cov['aoi_percent_at_threshold']}% of the imaged AOI passes" if cov.get("aoi_percent_at_threshold") is not None else "")
         + _noise_sentence(cov, cfg.min_temporal_coherence)
         + ". Bright ground is typically bare or sparsely vegetated ground and built surfaces; dark ground is typically woodland, dense crops and water"
         + (f". The reference pixel (star) has coherence {ref_tc:.2f}" if ref_tc is not None and np.isfinite(ref_tc) else "")
         + ". Anything read off a dark area should be discounted even if it survives the threshold.")

    # ---------------------------------------------------------------- closure-phase susceptibility
    cp = cfg.mintpy_dir / "geo" / "geo_maskClosurePhase.h5"
    acp = cfg.mintpy_dir / "geo" / "geo_avgCpxClosurePhase.h5"
    if cp.exists() and acp.exists():
        with h5py.File(acp) as f:
            ph = f["phase"][:]; at = dict(f.attrs)
        with h5py.File(cp) as f:
            m = f["mask"][:].astype(bool)
        x0, y0, dx, dy = (float(at[k]) for k in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP"))
        ny, nx = ph.shape
        extg = (x0, x0 + nx * dx, y0 + ny * dy, y0)
        fig, axs = plt.subplots(1, 2, figsize=(15, 6.4))
        im0 = axs[0].imshow(np.degrees(ph), extent=extg, cmap="RdBu", vmin=-30, vmax=30, interpolation="nearest"); overlay(axs[0], legend=False)
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02).set_label("average closure phase, degrees (0 = no bias)", fontsize=8)
        axs[0].set_title("Average closure phase of the shortest three-interferogram loops")
        im1 = axs[1].imshow(m.astype(float), extent=extg, cmap="Greens", vmin=0, vmax=1, interpolation="nearest"); overlay(axs[1], legend=False)
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02, ticks=[0, 1]).set_label("1 = closure phase consistent with zero (bias-free)", fontsize=8)
        axs[1].set_title("Bias-susceptibility mask (MintPy closure_phase_bias)")
        cm = ((Q.get("checks") or {}).get("distributed_scatterer_bias") or {}).get("closure_phase_mask") or {}
        fig.suptitle(f"Closure phase: {cm.get('bias_free_percent_of_reported_aoi', '—')}% of reported AOI cells pass the 3-connection loop test (a weak test - see the note); far-field median rate {cm.get('far_field_median_rate_bias_free_mm_yr', '—')} mm/yr on them vs {cm.get('far_field_median_rate_susceptible_mm_yr', '—')} on the rest", fontsize=10)
        save(fig, "closure_phase.png", "Closure phase: where the distributed-scatterer bias lives",
             ("Around any closed loop of three interferograms the phases should sum to zero; over distributed scatterers they do not, and the average remainder (left, in degrees) is the fingerprint of the bias. MintPy's mask (right) marks cells whose closure phase is consistent with zero. "
              + ("On the phase-linked route the interferograms are formed from one set of linked phases, so their loops close by construction and this test cannot see the bias at all. "
                 if linked else
                 "With only the shortest loops (three nearest connections), almost every cell usually passes: the short loops close well, and the bias shows up between short and long pairs rather than within short loops. So the mask is a weak test, not evidence that the bias is absent. ")
              + "The rate-by-coherence-class table is the stronger evidence. The full correction of Zheng et al. (2022) needs interferograms at a connection level well above three, which this network does not contain and which would not stay coherent over vegetated ground. The assessment instead recomputes every relative figure matched by coherence class, which is how the bias is cancelled where the region and the comparison ground differ in coherence."))

    # ---------------------------------------------------------------- two interferograms
    first, worst, worst_is_summer = _pick_pairs(cfg)
    have_stack = first is not None and all((cfg.stack_dir / "merged" / "interferograms" / pr / "filt_fine.int").exists()
                                           for pr in (first, worst))
    if not have_stack:
        # the stack's interferograms were deleted to make room: keep the
        # figures already drawn of this route's pairs, dated, and move aside
        # those of another kind (_carry_pair_figures)
        log.warning("interferogram figures skipped: the stack's filt_fine products are no longer on disk")
        manifest += _carry_pair_figures(out_dir, P / "deliverables", linked)
    if have_stack:
        G = cfg.stack_dir / "merged" / "geom_reference"
        cw = json.loads((cfg.stack_dir / "crop_window.json").read_text())
        r0, r1, c0, c1, L, W = cw["r0"], cw["r1"], cw["c0"], cw["c1"], cw["length"], cw["width"]
        lat = np.asarray(np.memmap(G / "lat.rdr", dtype=np.float64, mode="r", shape=(L, W))[r0:r1, c0:c1])
        lon = np.asarray(np.memmap(G / "lon.rdr", dtype=np.float64, mode="r", shape=(L, W))[r0:r1, c0:c1])
        dlat, dlon = 0.0004, 0.0005
        nrow = int(np.ceil((bbox[3] - bbox[1]) / dlat)); ncol = int(np.ceil((bbox[2] - bbox[0]) / dlon))
        iy = np.floor((bbox[3] - lat) / dlat).astype(int); ix = np.floor((lon - bbox[0]) / dlon).astype(int)
        ok = (iy >= 0) & (iy < nrow) & (ix >= 0) & (ix < ncol); flat = (iy * ncol + ix)[ok]
        cnt = np.bincount(flat, minlength=nrow * ncol).reshape(nrow, ncol)
        EXT = (bbox[0], bbox[0] + ncol * dlon, bbox[3] - nrow * dlat, bbox[3])
        rref = ref if ref is not None else ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
        ry, rx = np.unravel_index(np.argmin((lat - rref[0]) ** 2 + (lon - rref[1]) ** 2), lat.shape)

        def geocode(arr, cyclic=False):
            a = np.asarray(arr, dtype=float)[ok]
            if cyclic:
                z = (np.bincount(flat, weights=np.cos(a), minlength=nrow * ncol)
                     + 1j * np.bincount(flat, weights=np.sin(a), minlength=nrow * ncol))
                o = np.angle(z)
            else:
                g = np.isfinite(a)
                s = np.bincount(flat[g], weights=a[g], minlength=nrow * ncol)
                n = np.bincount(flat[g], minlength=nrow * ncol).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    o = s / n
            o = o.reshape(nrow, ncol); o[cnt == 0] = np.nan
            return o

        def pair_arrays(pair):
            d = cfg.stack_dir / "merged" / "interferograms" / pair
            igram = np.memmap(d / "filt_fine.int", dtype=np.complex64, mode="r", shape=(L, W))[r0:r1, c0:c1]
            cor = np.memmap(d / "filt_fine.cor", dtype=np.float32, mode="r", shape=(L, W))[r0:r1, c0:c1]
            unw = np.memmap(d / "filt_fine.unw", dtype=np.float32, mode="r", shape=(L, 2, W))[r0:r1, 1, c0:c1]
            cc = np.memmap(d / "filt_fine.unw.conncomp", dtype=np.uint8, mode="r", shape=(L, W))[r0:r1, c0:c1]
            return igram, cor, unw, cc

    unw_name = _unwrapper(cfg)
    # what the three panels are depends on the route: the phase-linked route
    # exports interferograms formed from MiaplPy's linked phases (every image
    # of the mini-stack contributes), the stack route the filtered two-image
    # interferograms. The captions said "12-day pair" and "snaphu can only
    # unwrap the scattered patches" whatever the record was.
    if linked:
        wrapped_txt = ("the phase-linked interferogram: MiaplPy estimated each date's phase from every image of its mini-stack, "
                       "and this pair is the difference of two such phases, so it is smoother than a raw two-image interferogram")
        coh_txt = f"the coherence the linked pair carries ({getattr(cfg, 'miaplpy_pair_coherence', 'phasor')}), which {unw_name}'s cost model and MintPy's weights read"
    else:
        wrapped_txt = "the filtered interferogram of the two images, the raw measurement"
        coh_txt = "spatial coherence, bright where the two acquisitions look alike"
    # what the overlay draws on the pairs, said in their captions (the green
    # circle was explained only on the rate map)
    marks_txt = ((" The star is the reference pixel." if ref is not None else "")
                 + (" The green circle is the deepest 1 km patch in the AOI, a selected extreme (the quality "
                    "assessment's monitoring point), not a finding." if mon is not None else ""))
    drawn = {"kind": PAIR_KIND_LINKED if linked else PAIR_KIND_RAW, "drawn": _dt.date.today().isoformat()}
    for pair, name, role in ((first, "winter_pair.png", "first"), (worst, "summer_pair.png", "worst")) if have_stack else ():
        igram, cor, unw, cc = pair_arrays(pair)
        wrapped = geocode(np.angle(igram), cyclic=True); coh = geocode(cor)
        valid = np.abs(np.asarray(igram)) > 0                 # the zero padding outside the data is not ground
        ccv = np.asarray(cc)[valid] if valid.any() else np.asarray(cc).ravel()
        frac = float((ccv > 0).mean()); medc = float(np.nanmedian(np.asarray(cor)[valid] if valid.any() else cor))
        days, season = _pair_days(pair), _season(pair)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5.8))
        im0 = axs[0].imshow(wrapped, extent=EXT, cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest"); overlay(axs[0], legend=False)
        cb = plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.02, ticks=[-np.pi, 0, np.pi]); cb.ax.set_yticklabels(["−π", "0", "+π"])
        cb.set_label(f"wrapped phase (one cycle = {wl / 2 * 1000:.1f} mm = {wl / 2 * 1000 / MM_IN:.2f} in of range)", fontsize=8)
        axs[0].set_title("Phase-linked interferogram" if linked else "Wrapped interferogram")
        im1 = axs[1].imshow(coh, extent=EXT, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); overlay(axs[1], legend=False)
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.02).set_label("pair coherence" if linked else "spatial coherence", fontsize=8)
        axs[1].set_title("Coherence")
        if role == "first":
            disp = np.asarray(unw, float) * phase2range * 1000.0
            disp = disp - disp[ry, rx]; disp[np.asarray(cc) == 0] = np.nan
            im2 = axs[2].imshow(geocode(disp), extent=EXT, cmap="RdBu", norm=TwoSlopeNorm(0, -30, 30), interpolation="nearest"); overlay(axs[2], legend=False)
            dual_cbar(im2, axs[2], [-30, -15, 0, 15, 30], "mm", "in", "unwrapped, as LOS displacement relative to the reference")
            axs[2].set_title("Unwrapped (reliable components only)")
        else:
            im2 = axs[2].imshow(geocode((np.asarray(cc) > 0).astype(float)), extent=EXT, cmap="Greens", vmin=0, vmax=1, interpolation="nearest"); overlay(axs[2], legend=False)
            plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02, ticks=[0, 1]).set_label("share of cell reliably unwrapped", fontsize=8)
            axs[2].set_title("Reliably unwrapped (connected components ≠ 0)")
        fig.suptitle(f"A {season} {days}-day pair, {_fmt_pair(pair)}: median coherence {medc:.2f}, {frac:.0%} of the imaged crop reliably unwrapped", fontsize=11)
        measured = (f"In this pair the median coherence is {medc:.2f} and {unw_name} placed {frac:.0%} of the imaged crop in "
                    "reliably unwrapped (non-zero) connected components.")
        if role == "first":
            save(fig, name, f"A {season} interferogram from the reference date, {_fmt_pair(pair)} ({days} days)",
                 f"Left: {wrapped_txt}. One full colour cycle is half a radar wavelength of line-of-sight change "
                 f"({wl / 2 * 1000:.1f} mm, {wl / 2 * 1000 / MM_IN:.2f} in). Smooth, slowly varying colour is a good interferogram; speckled colour is decorrelated ground. "
                 f"Middle: {coh_txt}. Right: the unwrapped phase, resolved by {unw_name} into a continuous surface and shown as line-of-sight displacement "
                 f"relative to the reference pixel over these {days} days; cells outside the reliable components are blank. A residual pattern of a few millimetres "
                 f"over a pair this short is mostly atmosphere, which the network inversion and the weather-model correction average down. {measured}{marks_txt}",
                 **drawn)
        else:
            which = ("the June-August pair that took longest to unwrap" if worst_is_summer else
                     "the pair that took longest to unwrap (the record has no June-August pair)")
            save(fig, name, f"The hardest {'growing-season ' if worst_is_summer else ''}interferogram, {_fmt_pair(pair)} ({days} days)",
                 f"The same views for {which}, {_fmt_pair(pair)} ({days} days). Left: {wrapped_txt}. Middle: {coh_txt}. "
                 f"Right: where {unw_name} unwrapped the phase into reliable connected components (green). {measured} "
                 "Where coherence is low (standing crops, woodland, water) the phase is close to speckle and only scattered patches unwrap; "
                 "dates like these carry less weight in the record, and low-coherence ground is where the distributed-scatterer bias arises."
                 + marks_txt, **drawn)

    # ---------------------------------------------------------------- network
    tsf = _dates_bperp_file(cfg)
    if (cfg.mintpy_dir / "inputs" / "ifgramStack.h5").exists() and tsf is None:
        # the radar time series was cleaned up (site_a_pl): keep the
        # figure drawn when it existed, as for the interferogram pairs
        log.warning("network figure skipped: no MintPy time series with dates and baselines is left on disk")
        man = out_dir / "figures.json"
        if man.exists():
            manifest += [m for m in json.loads(man.read_text()) if m["file"] == "network.png" and (out_dir / m["file"]).exists()]
    if (cfg.mintpy_dir / "inputs" / "ifgramStack.h5").exists() and tsf is not None:    # DISP-S1 has no network of ours
        with h5py.File(cfg.mintpy_dir / "inputs" / "ifgramStack.h5") as f:
            d12 = f["date"][:].astype(str); keep = f["dropIfgram"][:].astype(bool)
        with h5py.File(tsf) as f:
            dates = f["date"][:].astype(str); pb = f["bperp"][:]
        t = {d: _dt.datetime.strptime(d, "%Y%m%d") for d in dates}; pbd = dict(zip(dates, pb))
        fig, ax = plt.subplots(figsize=(15, 5.2))
        for (a, b), k in zip(d12, keep):
            if a in t and b in t:
                ax.plot([t[a], t[b]], [pbd[a], pbd[b]], color="#2a78d6" if k else "#c9ced4", lw=0.7 if k else 0.6, ls="-" if k else "--", zorder=2 if k else 1)
        ax.plot([t[d] for d in dates], [pbd[d] for d in dates], "o", ms=3.2, color="#17202a", zorder=3)
        ds = sorted(dates)
        n_gaps = 0
        for a, b in zip(ds, ds[1:]):
            if (t[b] - t[a]).days > 36:
                ax.axvspan(t[a], t[b], color="#eb6834", alpha=0.12, lw=0); n_gaps += 1
        ax.set_ylabel("perpendicular baseline, m"); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
        ax.set_title(f"The interferogram network: {int(keep.sum())} kept pairs (blue) of {len(d12)} formed over {len(dates)} dates; dropped pairs dashed grey; acquisition gaps shaded")
        # how the network is connected, measured on it: mean pairs per date
        # and the longest reach in acquisitions (was "three nearest
        # neighbours" whatever the network was)
        idx = {d: i for i, d in enumerate(ds)}
        kept = [(a, b) for (a, b), k in zip(d12, keep) if k and a in idx and b in idx]
        per_date = 2 * len(kept) / max(1, len(ds))
        reach = max((abs(idx[b] - idx[a]) for a, b in kept), default=0)
        nw = (Q.get("checks") or {}).get("network") or {}
        conn = ("The kept pairs form one connected piece. " if nw.get("connected") else
                "The kept pairs do NOT form one connected piece. " if nw.get("connected") is False else "")
        save(fig, "network.png", "The interferogram network",
             "Every dot is an acquisition, placed by date and by how far the satellite's orbit was from the reference orbit (perpendicular baseline). "
             + ("Each line is an interferogram exported from MiaplPy's phase linking and unwrapped; " if linked else "Each line is an interferogram; ")
             + "blue ones were inverted, dashed grey ones were dropped (the reference pixel unreliable in them, or hand-excluded). "
             f"On average each date is in {per_date:.1f} kept pairs, and the longest kept pair spans {reach} acquisitions; "
             "closed loops of pairs are what let unwrapping errors be detected. " + conn
             + (f"The {n_gaps} shaded gap{'s are' if n_gaps != 1 else ' is'} longer than 36 days without an acquisition; pairs spanning them keep the record together. " if n_gaps else "")
             + "Small baselines mean topography contributes almost nothing to the phase.")

    # ---------------------------------------------------------------- summary table
    # panel_history writes its own figures and manifest entries; keep them.
    # A figure this function draws but skipped this time (no cumulative
    # raster in the configured unit, no closure-phase layer) is NOT carried
    # over from the previous manifest: its PNG is from another run.
    man = out_dir / "figures.json"
    if man.exists():
        mine = {m["file"] for m in manifest} | set(_OWN_FIGURES) | set(PAIR_FIGURES)
        manifest += [m for m in json.loads(man.read_text()) if m["file"] not in mine and (out_dir / m["file"]).exists()]
    manifest.append(summary_table(cfg, Q, out_dir))
    man.write_text(json.dumps(manifest, indent=1))
    log.info("Wrote %d figures to %s", len(manifest), out_dir)
    return out_dir


def _num(x, spec, suffix="", missing="—"):
    """x formatted with `spec`, or `missing` when it was not measured. The
    table used f"{x:+.2f}" directly, and one None (the reference-stability
    fit is None when the reference series has fewer than 6 finite dates)
    raised TypeError and lost every technical figure."""
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)) or not np.isfinite(x):
        return missing
    return f"{x:{spec}}{suffix}"


def summary_table(cfg, Q, out_dir) -> dict:
    """One graphic with every critical figure in both units, its uncertainty
    (the spread of ordinary farmland of the same footprint) and whether it
    stands outside that spread. Every cell is None-safe: a quantity the QA
    could not measure reads '—', never a crash."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .quality_assessment import check_counts
    c, f = Q.get("checks") or {}, Q.get("findings") or {}
    S_f = cfg.project_dir / "deliverables" / "summary.json"
    S = json.loads(S_f.read_text()) if S_f.exists() else {}
    ctrl = ground_words(cfg, Q)
    ctrl_n = _noun(ctrl)                             # table cells: no land-cover note
    w = f.get("workings_relative_to_far_field") or {}; pnl = f.get("named_panel") or {}
    pt = f.get("monitoring_point") or {}; da = f.get("deforming_area") or {}
    rs, rp, sn, cv, nw, ds = (c.get(k) or {} for k in ("reference_stability", "rate_precision", "single_epoch_noise", "coverage", "network", "distributed_scatterer_bias"))
    methods = []                                  # how each quoted total was obtained

    def mmin(x, nd_mm=1, nd_in=2, unit=""):
        x = x if isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) else None
        return ("—", "—") if x is None else (f"{x:+.{nd_mm}f} mm{unit}", f"{x / MM_IN:+.{nd_in}f} in{unit}")

    def rate_cells(m):
        """metric, imperial, verdict for a regional relative rate"""
        sig = m.get("significance") or {}
        sd = sig.get("farmland_spread_1sigma_mm_yr")
        rate, tot = m.get("rate_mm_yr"), m.get("total_mm")
        if tot is not None:
            methods.append(m.get("total_method") or "rate_x_span")
        pm = f" ± {_num(sd, '.1f')}" if sd is not None else ""
        pmi = f" ± {_num(sd / MM_IN, '.2f')}" if isinstance(sd, (int, float)) else ""
        metric = f"{_num(rate, '+.2f')}{pm} mm/yr" + (f" · {_num(tot, '+.0f')} mm" if tot is not None else "")
        imp = (f"{_num(rate / MM_IN, '+.3f') if isinstance(rate, (int, float)) else '—'}{pmi} in/yr"
               + (f" · {_num(tot / MM_IN, '+.2f')} in" if isinstance(tot, (int, float)) else ""))
        dist = sig.get("distinguishable_from_farmland")
        if sig.get("testable") is False:
            return metric, imp, f"Not testable against {ctrl_n}"
        if not sig or dist is None:
            return metric, imp, f"No {ctrl_n} null"
        grade = f"Stands out from {ctrl_n}" if dist else f"Within {ctrl_n} spread"
        # the statistic the decision uses: S/N = |excess| / 95% threshold when
        # significance gives it (>= 1 <=> stands out), else the z of old files
        if sig.get("signal_to_threshold") is not None:
            stat = f" (S/N {_num(sig['signal_to_threshold'], '.1f')})"
        elif sig.get("z") is not None:
            stat = f" (z {_num(sig['z'], '+.1f')})"
        else:
            stat = ""
        return metric, imp, grade + stat

    res = _resolution(cfg)
    sup = res.get("support_m") or res.get("look_cell_m")
    sup_txt = (f"each value draws on ~{float(sup[0]):.0f} m E–W × {float(sup[1]):.0f} m N–S"
               if isinstance(sup, (list, tuple)) and len(sup) == 2 else
               f"each value draws on ~{float(sup):.0f} m" if isinstance(sup, (int, float)) else
               "support not computed")
    span_days = S.get("span_days")
    rows = [
        ("RECORD", "", "", ""),
        ("Acquisitions / interferograms inverted", f"{S.get('n_acquisitions', '—')} / {nw.get('pairs_inverted', '—')} of {nw.get('pairs_formed', '—')}", "", ""),
        ("Span", str(S.get("date_range", "—")).replace(" to ", " → "), _num(span_days / 365.25 if isinstance(span_days, (int, float)) else None, ".1f", " years"), ""),
        ("Measurement points (coherence ≥ %s)" % _num(cv.get("threshold", getattr(cfg, "min_temporal_coherence", None)), ".1f"),
         _num(S.get("total_measured_points"), ","),
         f"{_num(cv.get('aoi_percent_at_threshold'), 'g')}% of AOI ({_num(cv.get('aoi_percent_at_0_7'), 'g')}% at 0.7)", ""),
        ("Posting / ground each value draws on", _num(res.get("posting_m"), ".0f", " m"), sup_txt,
         {"miaplpy": "phase-linked route", "stack": "classic stack route", "disp": "DISP-S1 route"}.get(str(res.get("route") or _route(cfg)), str(res.get("route") or _route(cfg) or ""))),
        ("SITE FIGURES  (relative; ± = farmland spread; total = see note)", "", "", ""),
    ]
    head = len(rows) - 1
    if w:
        rows.append(("All mapped panels together", *rate_cells(w)))
    ph_f = cfg.project_dir / "deliverables" / "panel_history.json"
    sh_f = cfg.project_dir / "deliverables" / "site_history.json"
    if ph_f.exists():
        ph = json.loads(ph_f.read_text())
        sh = json.loads(sh_f.read_text()) if sh_f.exists() else {"mines": {}, "sites": {}}
        for mine, m in (ph.get("mines") or {}).items():
            rows.append((f"{mine}, all its panels ({_num(m.get('n_pixels'), ',')} cells)", *rate_cells(m)))
        for name, d in (sh.get("sites") or {}).items():
            r = d.get("relative_all_cells") or {}
            rows.append((f"{name} ({_num(d.get('coherent_cells'), ',')} cells)", *rate_cells(r)))
            cl, ca = d.get("cluster_relative") or {}, d.get("cluster_absolute") or {}
            if cl:
                rows.append((f"   {d.get('cluster_cells', '—')}-point cluster at its centre, relative", *rate_cells(cl)))
                rows.append(("   same cluster, absolute (vs reference pixel)",
                             f"{_num(ca.get('rate_mm_yr'), '+.2f')} mm/yr · {_num(ca.get('total_mm'), '+.0f')} mm",
                             f"{_num(ca.get('rate_in_yr'), '+.3f')} in/yr · {_num(ca.get('total_in'), '+.2f')} in",
                             "Carries the bias - not the site's motion"))
    if pnl:
        rows.append(("Named panel", *rate_cells({"rate_mm_yr": pnl.get("rate_relative_to_far_field_mm_yr"), "significance": pnl.get("significance")})))
    ex = pt.get("extreme") or {}
    if pt:
        rows.append(("%s: %s, %s" % ("Deepest 1 km patch in the AOI" if "deepest" in str(pt.get("how_chosen", "")) else "Pinned point",
                                     _num(pt.get("lat"), ".4f"), _num(pt.get("lon"), ".4f")),
                     *mmin(pt.get("rate_relative_1km_mm_yr"), 1, 2, "/yr"),
                     ("Stands out from chance" if ex.get("stands_out") else "A selected extreme, within chance")
                     + (f" (p {ex['p_selection_corrected']:.2f})" if isinstance(ex.get("p_selection_corrected"), (int, float)) else "")))
        rows.append(("   deepest expected by chance over %s patches" % ex.get("independent_1km_patches_in_aoi", "—"),
                     *mmin(ex.get("expected_deepest_by_chance_mm_yr"), 1, 2, "/yr"), ""))
    if da:
        rows.append(("Ground faster than %s" % str(da.get("threshold") or "").split(" relative")[0].replace("faster than ", ""),
                     f"{_num(da.get('area_km2'), 'g')} km²",
                     f"{_num((da.get('area_km2') or 0) * KM2_MI2, '.1f')} mi²  ({_num(da.get('inside_mapped_panels_percent'), 'g')}% inside panels)",
                     f"{_num(da.get('far_field_share_beyond_threshold_percent'), 'g')}% of {ctrl_n} crosses it by chance"))
    v_rs = rs.get("value_mm_yr")
    send = (Q.get("send_rule") or {}).get("value")
    cnt = check_counts(Q)
    rows += [
        ("QUALITY  (against field practice)", "", "", ""),
        ("Rate repeatability of one cell, empirical (median / coherence ≥ 0.7)",
         f"{_num(rp.get('empirical_median_mm_yr'), 'g')} / {_num(rp.get('empirical_median_high_coherence_mm_yr'), 'g')} mm/yr",
         f"formal {_num(rp.get('median_sigma_mm_yr'), 'g')} mm/yr assumes independent dates",
         "Not measured" if rp.get("pass") is None else "Meets practice" if rp.get("pass") else "Flagged"),
        ("Single-epoch noise, far from the reference (robust / RMS)",
         f"{_num(sn.get('robust_mm'), 'g')} / {_num(sn.get('rms_mm'), 'g')} mm",
         f"{_num((sn.get('robust_mm') / MM_IN) if isinstance(sn.get('robust_mm'), (int, float)) else None, '.2f')} / "
         f"{_num((sn.get('rms_mm') / MM_IN) if isinstance(sn.get('rms_mm'), (int, float)) else None, '.2f')} in",
         "Not measured" if sn.get("pass") is None else "Meets practice" if sn.get("pass") else "Flagged"),
        # None when the reference or the stable-ground series has too few
        # finite dates to fit (quality_assessment sets value_mm_yr None)
        ("Reference pixel drift vs stable ground", _num(v_rs, "+.2f", " mm/yr"),
         _num(v_rs / MM_IN if isinstance(v_rs, (int, float)) else None, "+.3f", " in/yr", missing="not measured"),
         "Not measured" if v_rs is None else ("Meets practice" if rs.get("pass") else "Flagged") + "; absolute rates only"),
        ("Distributed-scatterer bias (low − high coherence)", *mmin(ds.get("low_minus_high_coherence_mm_yr"), 1, 2, "/yr"),
         "Flagged; absolute rates biased" if ds.get("pass") is False else "Meets practice" if ds.get("pass")
         else "Not applicable" if ds.get("not_applicable") else "Not measured"),
        ("Network", f"{_num(nw.get('mean_connections_per_date'), 'g')} connections/date",
         "one connected piece" if nw.get("connected") else "NOT connected" if nw.get("connected") is False else "—",
         "Meets practice" if nw.get("pass") else "Not measured" if nw.get("pass") is None else "Flagged"),
        # counted as the verdict counts them: applicable checks only, and
        # the ones that do not apply named on their own line (D2, 7 Oct 2026)
        ("Verdict", str(Q.get("verdict", "")), cnt["head"] if c else "—",
         f"Send rule: {send}" if send else ""),
    ]
    if cnt["not_applicable"]:
        n_na = len(cnt["not_applicable"])
        rows.append(("   not applicable here, so not counted", f"{n_na} check{'' if n_na == 1 else 's'}",
                     ", ".join(n.replace("_", " ") for n in cnt["not_applicable"]), "Not a pass, not a flag"))
    # how the totals were obtained, from the records themselves (Kyle 5 Oct:
    # totals are the measured change; older panel histories carry rate x span)
    measured = bool(methods) and any(m != "rate_x_span" for m in methods)
    tot_txt = ("measured change over the record" if measured and all(m != "rate_x_span" for m in methods) else
               "measured change, or rate × span where the record is too short to fit" if measured else "rate × span")
    tot_sentence = (f" Totals are the {tot_txt}." if measured else " Totals are rate × span." if methods else "")
    tot_short = ("measured change" if measured and all(m != "rate_x_span" for m in methods) else
                 "measured / rate × span" if measured else "rate × span")
    rows[head] = (f"SITE FIGURES  (relative; ± = {ctrl_n} spread; total = {tot_short})", "", "", "")
    fig, ax = plt.subplots(figsize=(18, 0.38 * len(rows) + 1.0)); ax.axis("off")
    tbl = ax.table(cellText=[[r[0], r[1], r[2], r[3]] for r in rows],
                   colLabels=["Quantity", "Metric", "Imperial / basis", "Reading"],
                   colWidths=[0.36, 0.20, 0.22, 0.22], bbox=[0, 0, 1, 1], cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (r, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d9dfe5")
        if r == 0:
            cell.set_facecolor("#17202a"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        elif rows[r - 1][1] == "" and rows[r - 1][3] == "":
            cell.set_facecolor("#e8edf2"); cell.get_text().set_weight("bold")
        elif rows[r - 1][0].startswith("   "):
            cell.get_text().set_color("#55616e")
        if col == 3 and r > 0:
            txt = rows[r - 1][3]
            # a check that does not apply is neither a pass nor a limitation
            # (D2): grey, not the amber of "Not measured"
            cell.get_text().set_color("#0ca30c" if txt.startswith(("Stands out", "Meets")) else
                                      "#55616e" if txt.startswith(("Not applicable", "Not a pass, not a flag")) else
                                      "#b07a1e" if txt.startswith(("Within", "A selected", "Flagged", "Carries", "Not ")) else "#17202a")
    from .report_insights import identity
    sensor = "NISAR" if str(getattr(cfg, "backend", "")) == "nisar" else "Sentinel-1"
    ax.set_title(f"{identity(cfg)['title']} ground-motion record {str(S.get('date_range', '')).replace(' to ', ' – ')}  ·  {sensor} track {S.get('track')} "
                 f"{str(S.get('orbit_direction') or '').lower()}  ·  generated {_dt.date.today()}", fontsize=12, pad=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01); fig.savefig(out_dir / "summary_table.png", dpi=120); plt.close(fig)
    log.info("Wrote %s", out_dir / "summary_table.png")
    return {"file": "summary_table.png", "title": "Summary of the critical figures",
            "caption": (f"Every headline figure in both units. Site rates are relative to far-field {ctrl} of the same "
                        "temporal-coherence class, so the reference pixel and the coherence-dependent bias cancel. The ± "
                        f"is the spread of the same measurement over ordinary {ctrl} of the same footprint - the "
                        "uncertainty of a site figure on this ground - and 'Reading' says whether the figure stands "
                        f"outside the range that 95 of 100 same-size placements on {ctrl} show.{tot_sentence} The deepest 1 km patch is found by "
                        "searching the whole AOI, so it is compared with the deepest patch chance alone would produce. "
                        "Posting and support: " + (res.get("text") or (_num(res.get("posting_m"), ".0f", " m posting; the ground each value draws on was not computed (no resolution module)"))) + ".")}


def main():
    import sys
    from insar_pipeline.config import load_config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(sys.argv[1])
    print(make_figures(cfg))


if __name__ == "__main__":
    main()
